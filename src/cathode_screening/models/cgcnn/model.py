from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNBlock(nn.Module):
    """CGCNN convolution block with residual connection."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.lin_msg = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.lin_upd = nn.Linear(node_dim * 2, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, e_attr: torch.Tensor) -> torch.Tensor:
        identity = x
        m_in = torch.cat([x[src], x[dst], e_attr], dim=-1)
        msg = torch.tanh(self.lin_msg(m_in))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        u_in = torch.cat([x, agg], dim=-1)
        x = self.bn(self.lin_upd(u_in))
        x = F.relu(x + identity)
        return x


class CGCNN(nn.Module):
    """Crystal Graph CNN with ordered quantile heads and auxiliary classifiers."""

    def __init__(
        self,
        node_in: int,
        node_dim: int,
        edge_dim: int,
        layers: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(node_in, node_dim)
        self.blocks = nn.ModuleList([CGCNNBlock(node_dim, edge_dim) for _ in range(layers)])
        self.dropout = float(dropout)
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

    def pool(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
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

        out = torch.zeros((n_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch_index, x)
        counts = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
        counts.index_add_(0, batch_index, torch.ones(x.size(0), device=x.device, dtype=x.dtype))
        return out / counts.unsqueeze(-1).clamp_min(1e-8)

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        e_attr: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def get_embedding(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        e_attr: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return pooled graph embedding before the head layers."""
        batch_index = batch_index.long()
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, src, dst, e_attr)
        return self.pool(x, batch_index)
