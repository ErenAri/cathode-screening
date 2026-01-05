#!/usr/bin/env python3
"""
Calibrate OOD gate thresholds using validation set.

Usage:
    python scripts/06_calibrate_ood.py \
        --ensemble-dir artifacts/models/ensemble_20260105 \
        --data-config configs/train_cgcnn_ehull.yaml \
        --output-dir artifacts/ood/ensemble_20260105 \
        --target-fkr 0.05

This script:
1. Loads ensemble members and runs inference on validation set
2. Computes training set statistics (composition, embeddings, disagreement)
3. Builds FAISS index for embedding kNN
4. Selects OOD thresholds to achieve target false KEEP rate
5. Saves all artifacts for inference-time OOD gating
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Embedding index will not be built.")

from pymatgen.core import Composition

# Import OOD module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cathode_screening.inference.ood import (
    composition_fingerprint,
    COMPOSITION_VOCAB,
    compute_train_stats,
    save_ood_artifacts,
    select_ood_thresholds,
    save_ood_thresholds,
    OODGate,
)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Normalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def denorm(self, x):
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        return x * self.std + self.mean
    
    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std = d["std"]


# Model definition (must match training)
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


class CGCNN(nn.Module):
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

    def forward(self, x, src, dst, e_attr, batch_index):
        batch_index = batch_index.long()
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, src, dst, e_attr)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        g = self.pool(x, batch_index)
        h = torch.nn.functional.relu(self.fc_hidden(g))
        h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        
        q50 = self.head_q50(h).squeeze(-1)
        d_lo = torch.nn.functional.softplus(self.head_d_lo(h).squeeze(-1))
        d_hi = torch.nn.functional.softplus(self.head_d_hi(h).squeeze(-1))
        q10 = q50 - d_lo
        q90 = q50 + d_hi
        
        logit_stable = self.head_p_stable(h).squeeze(-1)
        logit_metastable = self.head_p_metastable(h).squeeze(-1)
        
        return q10, q50, q90, logit_stable, logit_metastable, h  # Also return embedding
    
    def forward_with_embedding(self, x, src, dst, e_attr, batch_index):
        """Same as forward but explicitly returns penultimate embedding."""
        return self.forward(x, src, dst, e_attr, batch_index)


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


def load_ensemble(ensemble_dir: Path, device: torch.device):
    """Load all ensemble members."""
    meta_path = ensemble_dir / "ensemble_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    members = []
    for m in meta["members"]:
        ckpt_path = Path(m["checkpoint"])
        if not ckpt_path.is_absolute():
            ckpt_path = ensemble_dir / ckpt_path
        
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"]
        
        model = CGCNN(
            node_in=cfg["model"].get("node_in_dim", 6),
            node_dim=cfg["model"]["node_embed_dim"],
            edge_dim=cfg["model"]["edge_rbf_bins"],
            layers=cfg["model"]["message_passing_layers"],
            dropout=cfg["model"]["dropout"],
            pooling=cfg["model"]["pooling"]
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        normalizer = Normalizer()
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
        
        members.append((model, normalizer))
    
    return members, meta


@torch.no_grad()
def extract_predictions_and_embeddings(
    members: List[Tuple[nn.Module, Normalizer]],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict, np.ndarray]:
    """
    Run inference and extract embeddings for all samples.
    
    Returns:
        predictions: Dict with arrays for q50_mean, q50_ensemble, decisions, etc.
        embeddings: [N, embed_dim] array (from first member, for kNN)
    """
    K = len(members)
    
    all_mids = []
    all_formulas = []
    all_y = []
    all_q50_ensemble = []  # [N, K]
    all_embeddings = []    # [N, embed_dim]
    all_q90_mean = []
    all_p_stable_mean = []
    
    for x, src, dst, e, b, y, mids, formulas in loader:
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        batch_size = len(mids)
        
        q50_batch = []
        q90_batch = []
        p_stable_batch = []
        emb_batch = None
        
        for i, (model, normalizer) in enumerate(members):
            q10, q50, q90, logit_stable, logit_meta, emb = model(x, src, dst, e, b)
            
            q50_denorm = normalizer.denorm(q50).cpu().numpy()
            q90_denorm = normalizer.denorm(q90).cpu().numpy()
            p_stable = torch.sigmoid(logit_stable).cpu().numpy()
            
            q50_batch.append(q50_denorm)
            q90_batch.append(q90_denorm)
            p_stable_batch.append(p_stable)
            
            # Use first member's embedding for kNN index
            if i == 0:
                emb_batch = emb.cpu().numpy()
        
        # Stack: [K, batch_size]
        q50_batch = np.stack(q50_batch, axis=0)
        q90_batch = np.stack(q90_batch, axis=0)
        p_stable_batch = np.stack(p_stable_batch, axis=0)
        
        # Per-sample aggregates
        q50_mean = q50_batch.mean(axis=0)
        q90_mean = q90_batch.mean(axis=0)
        p_stable_mean = p_stable_batch.mean(axis=0)
        
        all_mids.extend(mids)
        all_formulas.extend(formulas)
        all_y.append(y.numpy())
        all_q50_ensemble.append(q50_batch.T)  # [batch_size, K]
        all_embeddings.append(emb_batch)
        all_q90_mean.append(q90_mean)
        all_p_stable_mean.append(p_stable_mean)
    
    predictions = {
        "material_id": all_mids,
        "formula": all_formulas,
        "y_true": np.concatenate(all_y),
        "q50_ensemble": np.concatenate(all_q50_ensemble, axis=0),  # [N, K]
        "q90_mean": np.concatenate(all_q90_mean),
        "p_stable_mean": np.concatenate(all_p_stable_mean),
    }
    
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    
    return predictions, embeddings


def compute_decisions(q90_mean: np.ndarray, p_stable: np.ndarray) -> np.ndarray:
    """Compute raw decisions (before OOD gating)."""
    decisions = []
    for q90, ps in zip(q90_mean, p_stable):
        if q90 < 0.05 and ps > 0.70:
            decisions.append("KEEP")
        elif q90 < 0.10 and ps > 0.60:
            decisions.append("KEEP")
        elif q90 > 0.10:  # Even mean q90 is unstable
            decisions.append("KILL")
        else:
            decisions.append("MAYBE")
    return np.array(decisions)


def main():
    ap = argparse.ArgumentParser(description="Calibrate OOD gate thresholds")
    ap.add_argument("--ensemble-dir", required=True, help="Path to ensemble directory")
    ap.add_argument("--data-config", required=True, help="Data config YAML")
    ap.add_argument("--output-dir", required=True, help="Output directory for OOD artifacts")
    ap.add_argument("--target-fkr", type=float, default=0.05, 
                    help="Target false KEEP rate (default: 0.05)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load ensemble
    ensemble_dir = Path(args.ensemble_dir)
    members, meta = load_ensemble(ensemble_dir, device)
    print(f"Loaded {len(members)} ensemble members")
    
    # Load data
    data_cfg = load_cfg(args.data_config)
    processed_dir = Path(data_cfg["data"]["processed_dir"])
    ds_name = data_cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)
    
    splits_path = Path(data_cfg["data"]["splits_manifest"])
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    target_col = data_cfg["data"]["target"]["name"]
    
    # Split data
    df_train = df[df["material_id"].isin(set(splits["train"]))].reset_index(drop=True)
    df_val = df[df["material_id"].isin(set(splits["val"]))].reset_index(drop=True)
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    
    # Create loaders
    train_ds = GraphNPZDataset(df_train, target_col)
    val_ds = GraphNPZDataset(df_val, target_col)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    
    # Extract train predictions and embeddings
    print("\nExtracting training set predictions and embeddings...")
    train_preds, train_embeddings = extract_predictions_and_embeddings(members, train_loader, device)
    
    # Compute training disagreements
    train_disagreements = np.std(train_preds["q50_ensemble"], axis=1)
    
    # Compute OOD statistics from training set
    print("\nComputing OOD statistics from training set...")
    stats, comp_mu, comp_cov_inv = compute_train_stats(
        train_formulas=train_preds["formula"],
        train_embeddings=train_embeddings,
        train_disagreements=train_disagreements,
    )
    
    # Save OOD artifacts (stats, composition, FAISS index)
    output_dir = Path(args.output_dir)
    save_ood_artifacts(output_dir, stats, comp_mu, comp_cov_inv, train_embeddings)
    
    # Load the gate to score validation samples
    print("\nScoring validation samples...")
    gate = OODGate.from_artifacts(output_dir)
    
    # Extract validation predictions
    val_preds, val_embeddings = extract_predictions_and_embeddings(members, val_loader, device)
    
    # Compute OOD scores for validation
    val_ood_scores = []
    for i in range(len(val_preds["material_id"])):
        result = gate.score(
            formula=val_preds["formula"][i],
            embedding=val_embeddings[i],
            q50_ensemble=val_preds["q50_ensemble"][i].tolist(),
        )
        val_ood_scores.append(result.ood_score)
    
    val_ood_scores = np.array(val_ood_scores)
    
    # Compute raw decisions
    val_decisions = compute_decisions(val_preds["q90_mean"], val_preds["p_stable_mean"])
    
    # Select thresholds
    print(f"\nSelecting OOD thresholds (target FKR={args.target_fkr:.2%})...")
    thresholds = select_ood_thresholds(
        val_ood_scores=val_ood_scores,
        val_decisions=val_decisions,
        val_y_true=val_preds["y_true"],
        target_fkr=args.target_fkr,
    )
    
    # Save thresholds
    save_ood_thresholds(thresholds, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("OOD CALIBRATION COMPLETE")
    print("="*60)
    print(f"Artifacts saved to: {output_dir}")
    print(f"\nTraining statistics:")
    print(f"  Composition: mean={stats['comp_mean']:.3f}, std={stats['comp_std']:.3f}")
    print(f"  Embedding kNN: mean={stats['emb_mean']:.3f}, std={stats['emb_std']:.3f}")
    print(f"  Disagreement: mean={stats['disagree_mean']:.4f}, std={stats['disagree_std']:.4f}")
    print(f"\nCalibrated thresholds:")
    print(f"  t_hi (DEFER): {thresholds.t_hi:.3f}")
    print(f"  t_mid (STRICT): {thresholds.t_mid:.3f}")
    print(f"  Achieved FKR: {thresholds.achieved_fkr:.2%}")
    
    # Validation statistics
    n_keep = (val_decisions == "KEEP").sum()
    n_defer = (val_ood_scores > thresholds.t_hi).sum()
    n_strict = ((val_ood_scores > thresholds.t_mid) & (val_ood_scores <= thresholds.t_hi)).sum()
    
    print(f"\nValidation set breakdown:")
    print(f"  Total: {len(val_ood_scores)}")
    print(f"  Raw KEEPs: {n_keep}")
    print(f"  Would DEFER (ood > t_hi): {n_defer}")
    print(f"  Would apply STRICT (t_mid < ood < t_hi): {n_strict}")


if __name__ == "__main__":
    main()
