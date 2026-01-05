#!/usr/bin/env python3
"""
Build OOD (Out-of-Distribution) detection index from trained model.

This script extracts penultimate layer embeddings from the training set
and builds a kNN index for OOD detection at inference time.

Usage:
    python scripts/06_build_ood_index.py \
        --checkpoint artifacts/models/run_seed42/best.pt \
        --data-config configs/train_cgcnn_ehull.yaml \
        --output-dir artifacts/ood/run_seed42

Outputs:
    artifacts/ood/<run_id>/
    ├── composition_stats.npz     # Composition centroids and covariance
    ├── train_stats.json          # Distribution statistics for z-score normalization
    ├── embedding_index.faiss     # kNN index for embedding distance (if faiss available)
    ├── embedding_matrix.npy      # Raw embeddings [N_train, dim]
    ├── material_ids.json         # Mapping from index to material_id
    └── ood_thresholds.json       # Calibrated thresholds (if --calibrate)
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from pymatgen.core import Composition

# Check for faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Using sklearn NearestNeighbors instead.")
    from sklearn.neighbors import NearestNeighbors


# Composition vocabulary (must match training)
COMPOSITION_VOCAB = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def composition_fingerprint(formula: str) -> np.ndarray:
    """Convert formula to normalized composition vector."""
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(COMPOSITION_VOCAB), dtype=np.float32)
    for i, el in enumerate(COMPOSITION_VOCAB):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


# ============================================================================
# Model Definition (must match training)
# ============================================================================

class Normalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def denorm(self, x):
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        return x * self.std + self.mean
    
    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class CGCNNBlock(nn.Module):
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
        x = torch.nn.functional.relu(x + identity)
        return x


class CGCNNWithEmbedding(nn.Module):
    """CGCNN that also returns penultimate layer embedding."""
    
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

    def forward(self, x, src, dst, e_attr, batch_index, return_embedding: bool = False):
        batch_index = batch_index.long()
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, src, dst, e_attr)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        g = self.pool(x, batch_index)  # Graph-level embedding after pooling
        h = torch.nn.functional.relu(self.fc_hidden(g))  # Penultimate layer
        
        # Predictions
        q50 = self.head_q50(h).squeeze(-1)
        d_lo = torch.nn.functional.softplus(self.head_d_lo(h).squeeze(-1))
        d_hi = torch.nn.functional.softplus(self.head_d_hi(h).squeeze(-1))
        q10 = q50 - d_lo
        q90 = q50 + d_hi
        
        logit_stable = self.head_p_stable(h).squeeze(-1)
        logit_metastable = self.head_p_metastable(h).squeeze(-1)
        
        if return_embedding:
            return q10, q50, q90, logit_stable, logit_metastable, h
        return q10, q50, q90, logit_stable, logit_metastable


# ============================================================================
# Data Loading
# ============================================================================

class GraphNPZDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str):
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
        return x, src, dst, e, y, r["material_id"], r.get("formula_pretty", "")


def collate(batch):
    xs, srcs, dsts, es, ys, mids, formulas = zip(*batch)
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

    return x, src, dst, e, batch_index, y, list(mids), list(formulas)


# ============================================================================
# OOD Index Building
# ============================================================================

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    normalizer: Normalizer
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract penultimate layer embeddings and predictions from model.
    
    Returns:
        embeddings: [N, embedding_dim] array
        q50_preds: [N] array of median predictions (denormalized)
        material_ids: List of material IDs
        formulas: List of formulas
    """
    model.eval()
    
    all_embeddings = []
    all_q50 = []
    all_mids = []
    all_formulas = []
    
    for x, src, dst, e, b, y, mids, formulas in loader:
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        q10, q50, q90, logit_s, logit_m, emb = model(
            x, src, dst, e, b, return_embedding=True
        )
        
        # Denormalize predictions
        q50_denorm = normalizer.denorm(q50).cpu().numpy()
        
        all_embeddings.append(emb.cpu().numpy())
        all_q50.append(q50_denorm)
        all_mids.extend(mids)
        all_formulas.extend(formulas)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    q50_preds = np.concatenate(all_q50, axis=0)
    
    return embeddings, q50_preds, all_mids, all_formulas


def compute_composition_stats(formulas: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute composition statistics for OOD detection.
    
    Returns:
        comp_mu: Mean composition vector
        comp_cov_inv: Inverse covariance matrix
        comp_dists: Mahalanobis distances for all samples
    """
    X = np.stack([composition_fingerprint(f) for f in formulas], axis=0)
    
    comp_mu = X.mean(axis=0)
    comp_cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])  # Regularization
    comp_cov_inv = np.linalg.inv(comp_cov)
    
    # Compute Mahalanobis distances
    comp_dists = []
    for x in X:
        delta = x - comp_mu
        d_sq = delta @ comp_cov_inv @ delta
        comp_dists.append(np.sqrt(max(0, d_sq)))
    
    return comp_mu, comp_cov_inv, np.array(comp_dists)


def compute_embedding_knn_distances(
    embeddings: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, object]:
    """
    Compute leave-one-out kNN distances for embeddings.
    
    Returns:
        knn_dists: [N] array of mean k-NN distances
        index: FAISS index or sklearn NearestNeighbors
    """
    n_samples, dim = embeddings.shape
    embeddings = embeddings.astype(np.float32)
    
    if FAISS_AVAILABLE:
        # Build FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Query k+1 neighbors (first is self)
        distances, _ = index.search(embeddings, k + 1)
        knn_dists = distances[:, 1:].mean(axis=1)  # Skip self
        
        return knn_dists, index
    else:
        # Fallback to sklearn
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(embeddings)
        
        distances, _ = nn.kneighbors(embeddings)
        knn_dists = distances[:, 1:].mean(axis=1)  # Skip self
        
        return knn_dists, nn


def build_ood_index(
    checkpoint_path: str,
    data_config_path: str,
    output_dir: str,
    device: str = "auto",
    k_neighbors: int = 10,
) -> Dict:
    """
    Build OOD detection index from checkpoint and training data.
    
    Returns:
        Dictionary with statistics and paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    
    # Create model with embedding output
    model = CGCNNWithEmbedding(
        node_in=cfg["model"].get("node_in_dim", 6),
        node_dim=cfg["model"]["node_embed_dim"],
        edge_dim=cfg["model"]["edge_rbf_bins"],
        layers=cfg["model"]["message_passing_layers"],
        dropout=cfg["model"]["dropout"],
        pooling=cfg["model"]["pooling"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Load normalizer
    normalizer = Normalizer()
    if "normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer"])
    
    # Load data config
    data_cfg = load_cfg(data_config_path)
    
    # Load processed data
    processed_dir = Path(data_cfg["data"]["processed_dir"])
    ds_name = data_cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)
    print(f"Loaded {len(df)} total samples")
    
    # Load splits
    splits_path = Path(data_cfg["data"]["splits_manifest"])
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Get training split
    train_ids = set(splits["train"])
    df_train = df[df["material_id"].isin(train_ids)].reset_index(drop=True)
    print(f"Training set: {len(df_train)} samples")
    
    # Create dataset and loader
    target_col = data_cfg["data"]["target"]["name"]
    train_ds = GraphNPZDataset(df_train, target_col)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate
    )
    
    # Extract embeddings
    print("Extracting embeddings from training set...")
    embeddings, q50_preds, material_ids, formulas = extract_embeddings(
        model, train_loader, device, normalizer
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Compute composition stats
    print("Computing composition statistics...")
    comp_mu, comp_cov_inv, comp_dists = compute_composition_stats(formulas)
    print(f"  Composition distance: mean={comp_dists.mean():.4f}, std={comp_dists.std():.4f}")
    
    # Compute embedding kNN distances
    print(f"Computing {k_neighbors}-NN distances...")
    emb_dists, emb_index = compute_embedding_knn_distances(embeddings, k=k_neighbors)
    print(f"  Embedding kNN distance: mean={emb_dists.mean():.4f}, std={emb_dists.std():.4f}")
    
    # For training set, disagreement is 0 (single model)
    # This will be populated from ensemble during calibration
    disagree_dists = np.zeros(len(embeddings))
    
    # Compile statistics
    stats = {
        "n_train": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "k_neighbors": k_neighbors,
        
        "comp_mean": float(comp_dists.mean()),
        "comp_std": float(comp_dists.std()),
        "tau_comp_p95": float(np.percentile(comp_dists, 95)),
        "tau_comp_p99": float(np.percentile(comp_dists, 99)),
        
        "emb_mean": float(emb_dists.mean()),
        "emb_std": float(emb_dists.std()),
        "tau_emb_p95": float(np.percentile(emb_dists, 95)),
        "tau_emb_p99": float(np.percentile(emb_dists, 99)),
        
        # Placeholder for disagreement (will be updated by ensemble calibration)
        "disagree_mean": 0.0,
        "disagree_std": 1.0,
        "tau_disagree_p95": 0.0,
        "tau_disagree_p99": 0.0,
    }
    
    # Save artifacts
    print(f"Saving artifacts to {output_dir}...")
    
    # Stats JSON
    with open(output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Composition stats
    np.savez(
        output_dir / "composition_stats.npz",
        mu=comp_mu,
        cov_inv=comp_cov_inv
    )
    
    # Embedding matrix
    np.save(output_dir / "embedding_matrix.npy", embeddings)
    
    # Material ID mapping
    with open(output_dir / "material_ids.json", "w", encoding="utf-8") as f:
        json.dump(material_ids, f)
    
    # FAISS index
    if FAISS_AVAILABLE:
        faiss.write_index(emb_index, str(output_dir / "embedding_index.faiss"))
        print("  Saved FAISS index")
    else:
        # Save sklearn index via pickle
        import pickle
        with open(output_dir / "embedding_index.pkl", "wb") as f:
            pickle.dump(emb_index, f)
        print("  Saved sklearn NearestNeighbors index")
    
    print("\nOOD index built successfully!")
    print(f"  Composition: mean={stats['comp_mean']:.4f}, p95={stats['tau_comp_p95']:.4f}")
    print(f"  Embedding kNN: mean={stats['emb_mean']:.4f}, p95={stats['tau_emb_p95']:.4f}")
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build OOD detection index")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--data-config", required=True, help="Data config YAML")
    ap.add_argument("--output-dir", required=True, help="Output directory for OOD artifacts")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--k-neighbors", type=int, default=10, help="Number of neighbors for kNN")
    args = ap.parse_args()
    
    build_ood_index(
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        output_dir=args.output_dir,
        device=args.device,
        k_neighbors=args.k_neighbors,
    )


if __name__ == "__main__":
    main()
