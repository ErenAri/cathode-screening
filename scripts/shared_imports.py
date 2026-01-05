"""
Shared imports and utilities for decision-grade scripts.

This module re-exports classes from the training script for use in other scripts.
"""
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import from 04_train.py - we need to extract these classes
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Normalizer:
    """Normalize targets to zero mean and unit variance."""
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = torch.mean(tensor).item()
            self.std = torch.std(tensor).item()
            if self.std < 1e-8:
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class CGCNNBlock(nn.Module):
    """CGCNN convolution block with skip connection."""
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.lin_upd = nn.Linear(node_dim * 2, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, src, dst, e_attr):
        identity = x
        m_in = torch.cat([x[src], x[dst], e_attr], dim=-1)
        m = torch.tanh(self.lin_msg(m_in))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        u_in = torch.cat([x, agg], dim=-1)
        x = self.bn(self.lin_upd(u_in))
        x = F.relu(x + identity)
        return x


class CGCNN(nn.Module):
    """Crystal Graph CNN with multi-head output for regression + classification."""
    def __init__(self, node_in: int, node_dim: int, edge_dim: int, layers: int, dropout: float, pooling: str):
        super().__init__()
        self.embed = nn.Linear(node_in, node_dim)
        self.blocks = nn.ModuleList([CGCNNBlock(node_dim, edge_dim) for _ in range(layers)])
        self.dropout = dropout
        self.pooling = pooling
        self.pool_attn = nn.Linear(node_dim, 1) if pooling == "attn" else None
        
        self.fc_hidden = nn.Linear(node_dim, node_dim)
        
        self.head_q50 = nn.Linear(node_dim, 1)
        self.head_d_lo = nn.Linear(node_dim, 1)
        self.head_d_hi = nn.Linear(node_dim, 1)
        
        nn.init.constant_(self.head_d_lo.bias, -1.0)
        nn.init.constant_(self.head_d_hi.bias, -1.0)
        
        self.head_p_stable = nn.Linear(node_dim, 1)
        self.head_p_metastable = nn.Linear(node_dim, 1)

    def pool(self, x, batch_index):
        batch_index = batch_index.long()
        n_graphs = int(batch_index.max().item()) + 1
        
        if self.pooling == "attn":
            w = self.pool_attn(x).squeeze(-1)
            w_max = torch.full((n_graphs,), -torch.inf, device=x.device, dtype=w.dtype)
            w_max.scatter_reduce_(0, batch_index, w, reduce="amax", include_self=True)
            w_exp = torch.exp(w - w_max[batch_index])
            denom = torch.zeros(n_graphs, device=x.device, dtype=w.dtype)
            denom.index_add_(0, batch_index, w_exp)
            alpha = w_exp / denom[batch_index].clamp_min(1e-12)
            out = torch.zeros((n_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
            out.index_add_(0, batch_index, x * alpha.unsqueeze(-1))
            return out
        else:
            out = torch.zeros((n_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
            out.index_add_(0, batch_index, x)
            counts = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
            counts.index_add_(0, batch_index, torch.ones(x.size(0), device=x.device, dtype=x.dtype))
            return out / counts.unsqueeze(-1).clamp_min(1e-8)

    def forward(self, x, src, dst, e_attr, batch_index, debug=False):
        batch_index = batch_index.long()
        x = self.embed(x)
        
        for blk in self.blocks:
            x = blk(x, src, dst, e_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        g = self.pool(x, batch_index)
        h = F.relu(self.fc_hidden(g))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        q50 = self.head_q50(h).squeeze(-1)
        d_lo = F.softplus(self.head_d_lo(h).squeeze(-1))
        d_hi = F.softplus(self.head_d_hi(h).squeeze(-1))
        
        q10 = q50 - d_lo
        q90 = q50 + d_hi
        
        logit_stable = self.head_p_stable(h).squeeze(-1)
        logit_metastable = self.head_p_metastable(h).squeeze(-1)
        
        return q10, q50, q90, logit_stable, logit_metastable
    
    def get_embedding(self, x, src, dst, e_attr, batch_index):
        """Extract graph-level embedding (for OOD detection)."""
        batch_index = batch_index.long()
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, src, dst, e_attr)
        g = self.pool(x, batch_index)
        return g


class GraphNPZDataset(Dataset):
    """Dataset for graph NPZ files."""
    def __init__(self, df, target_col: str):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        z = np.load(r["graph_npz"])
        x = torch.from_numpy(z["x"])
        src = torch.from_numpy(z["edge_src"]).long()
        dst = torch.from_numpy(z["edge_dst"]).long()
        e = torch.from_numpy(z["e_attr"])
        y = torch.tensor(float(r[self.target_col]), dtype=torch.float32)
        return x, src, dst, e, y


def collate(batch):
    """Collate function for batching graphs."""
    xs, srcs, dsts, es, ys = zip(*batch)
    x = torch.cat(xs, dim=0)
    y = torch.stack(ys, dim=0)

    node_offsets = []
    off = 0
    for xi in xs:
        node_offsets.append(off)
        off += xi.shape[0]

    src = []
    dst = []
    for i, (s, d) in enumerate(zip(srcs, dsts)):
        o = node_offsets[i]
        src.append(s + o)
        dst.append(d + o)
    src = torch.cat(src, dim=0)
    dst = torch.cat(dst, dim=0)
    e = torch.cat(es, dim=0)

    batch_index = []
    for i, xi in enumerate(xs):
        batch_index.append(torch.full((xi.shape[0],), i, dtype=torch.long))
    batch_index = torch.cat(batch_index, dim=0)

    return x, src, dst, e, batch_index, y


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device (CUDA if available)."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
