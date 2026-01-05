#!/usr/bin/env python3
"""
Ensemble inference for cathode screening with OOD gating.

Loads K ensemble members, runs inference on a dataset split, and aggregates
predictions with uncertainty quantification and OOD detection.

Usage:
    python scripts/07_predict_ensemble.py \
        --ensemble-dir artifacts/models/ensemble_20260105 \
        --data-config configs/dataset_mp.yaml \
        --split test \
        --output predictions/ensemble_test.parquet \
        --ood-dir artifacts/ood/run_seed42

Outputs:
    Parquet/CSV with columns:
    - material_id, formula
    - y_true (if available)
    - q10_raw, q50, q90_raw (raw ensemble averages)
    - q10_cal, q90_cal (calibrated bounds)
    - epistemic_var, epistemic_std
    - aleatoric_unc (interval width / 2)
    - total_unc
    - p_stable, p_metastable
    - ood_score, gate_level (IN/BORDERLINE/OOD)
    - decision, gated_decision
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

# Stability thresholds
THRESH_STABLE = 0.05
THRESH_METASTABLE = 0.10

# Composition vocabulary for OOD detection
COMPOSITION_VOCAB = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]


def composition_fingerprint(formula: str) -> np.ndarray:
    """Convert formula to normalized composition vector."""
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(COMPOSITION_VOCAB), dtype=np.float32)
    for i, el in enumerate(COMPOSITION_VOCAB):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Normalizer:
    """Target normalizer matching training."""
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


# ============================================================================
# Model Definition (must match training)
# ============================================================================

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
        y = torch.tensor(float(r[self.target_col]), dtype=torch.float32) if self.target_col in r else torch.tensor(float('nan'))
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
# OOD Gating
# ============================================================================

class OODGate:
    """
    Out-of-Distribution gating with three signals:
    1. Composition distance (Mahalanobis to training distribution)
    2. Embedding kNN distance (requires FAISS or sklearn)
    3. Ensemble disagreement (epistemic variance)
    
    Gate levels:
    - IN: ood_score < t_mid (trust predictions)
    - BORDERLINE: t_mid <= ood_score < t_hi (stricter KEEP criteria)
    - OOD: ood_score >= t_hi (force MAYBE)
    """
    
    def __init__(
        self,
        comp_mu: np.ndarray,
        comp_cov_inv: np.ndarray,
        comp_stats: Dict[str, float],
        emb_index: Optional[object],
        emb_stats: Dict[str, float],
        disagree_stats: Dict[str, float],
        t_hi: float = 0.70,
        t_mid: float = 0.50,
    ):
        self.comp_mu = comp_mu
        self.comp_cov_inv = comp_cov_inv
        self.comp_stats = comp_stats
        self.emb_index = emb_index
        self.emb_stats = emb_stats
        self.disagree_stats = disagree_stats
        self.t_hi = t_hi
        self.t_mid = t_mid
    
    @classmethod
    def from_artifacts(cls, ood_dir: Path) -> "OODGate":
        """Load OOD gate from saved artifacts."""
        ood_dir = Path(ood_dir)
        
        # Load composition stats
        comp_data = np.load(ood_dir / "composition_stats.npz")
        comp_mu = comp_data["mu"]
        comp_cov_inv = comp_data["cov_inv"]
        
        # Load train stats
        with open(ood_dir / "train_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        comp_stats = {
            "mean": stats["comp_mean"],
            "std": stats["comp_std"],
        }
        emb_stats = {
            "mean": stats["emb_mean"],
            "std": stats["emb_std"],
        }
        disagree_stats = {
            "mean": stats.get("disagree_mean", 0.0),
            "std": max(stats.get("disagree_std", 1.0), 1e-6),
        }
        
        # Load FAISS index or sklearn index
        emb_index = None
        faiss_path = ood_dir / "embedding_index.faiss"
        sklearn_path = ood_dir / "embedding_index.pkl"
        
        if FAISS_AVAILABLE and faiss_path.exists():
            emb_index = faiss.read_index(str(faiss_path))
        elif sklearn_path.exists():
            import pickle
            with open(sklearn_path, "rb") as f:
                emb_index = pickle.load(f)
        
        # Load thresholds if available
        thresholds_path = ood_dir / "ood_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            t_hi = thresholds.get("t_hi", 0.70)
            t_mid = thresholds.get("t_mid", 0.50)
        else:
            t_hi, t_mid = 0.70, 0.50
        
        return cls(
            comp_mu=comp_mu,
            comp_cov_inv=comp_cov_inv,
            comp_stats=comp_stats,
            emb_index=emb_index,
            emb_stats=emb_stats,
            disagree_stats=disagree_stats,
            t_hi=t_hi,
            t_mid=t_mid,
        )
    
    def compute_comp_distance(self, formula: str) -> float:
        """Mahalanobis distance for composition."""
        x = composition_fingerprint(formula)
        delta = x - self.comp_mu
        d_sq = delta @ self.comp_cov_inv @ delta
        return float(np.sqrt(max(0, d_sq)))
    
    def compute_emb_distance(self, embedding: np.ndarray, k: int = 10) -> float:
        """Mean distance to k nearest training embeddings."""
        if self.emb_index is None:
            return 0.0
        
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        
        if FAISS_AVAILABLE and hasattr(self.emb_index, 'search'):
            distances, _ = self.emb_index.search(embedding, k)
            return float(distances[0].mean())
        else:
            # sklearn NearestNeighbors
            distances, _ = self.emb_index.kneighbors(embedding, n_neighbors=k)
            return float(distances[0].mean())
    
    def score(
        self,
        formula: str,
        epistemic_var: float,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[float, str, Dict]:
        """
        Compute OOD score and gate level.
        
        Returns:
            ood_score: Combined score [0, 1]
            gate_level: "IN", "BORDERLINE", or "OOD"
            details: Dictionary with individual scores
        """
        # Composition distance
        d_comp = self.compute_comp_distance(formula)
        z_comp = (d_comp - self.comp_stats["mean"]) / max(self.comp_stats["std"], 1e-6)
        
        # Embedding kNN distance
        if embedding is not None and self.emb_index is not None:
            d_emb = self.compute_emb_distance(embedding)
            z_emb = (d_emb - self.emb_stats["mean"]) / max(self.emb_stats["std"], 1e-6)
        else:
            d_emb = 0.0
            z_emb = 0.0
        
        # Ensemble disagreement
        d_disagree = np.sqrt(epistemic_var) if epistemic_var > 0 else 0.0
        z_disagree = (d_disagree - self.disagree_stats["mean"]) / max(self.disagree_stats["std"], 1e-6)
        
        # Combined: max of z-scores
        z_max = max(z_comp, z_emb, z_disagree)
        
        # Sigmoid mapping: z=0 → 0.12, z=2 → 0.5, z=4 → 0.88
        ood_score = 1.0 / (1.0 + np.exp(-z_max + 2.0))
        
        # Gate level
        if ood_score >= self.t_hi:
            gate_level = "OOD"
        elif ood_score >= self.t_mid:
            gate_level = "BORDERLINE"
        else:
            gate_level = "IN"
        
        details = {
            "d_comp": d_comp,
            "d_emb": d_emb,
            "d_disagree": d_disagree,
            "z_comp": z_comp,
            "z_emb": z_emb,
            "z_disagree": z_disagree,
            "z_max": z_max,
        }
        
        return float(ood_score), gate_level, details


def apply_ood_gating(
    raw_decision: str,
    gate_level: str,
    q90_cal: float,
    p_stable: float,
    strict_q90_thresh: float = 0.03,
    strict_p_stable_thresh: float = 0.85,
) -> str:
    """
    Apply OOD gating policy to modify decision.
    
    Policy:
    - OOD → force KEEP to MAYBE (DEFER)
    - BORDERLINE → stricter KEEP criteria
    - IN → trust raw decision
    """
    if gate_level == "OOD":
        if raw_decision == "KEEP":
            return "MAYBE"
        return raw_decision
    
    if gate_level == "BORDERLINE":
        if raw_decision == "KEEP":
            # Apply stricter criteria
            if q90_cal < strict_q90_thresh and p_stable > strict_p_stable_thresh:
                return "KEEP"
            else:
                return "MAYBE"
        return raw_decision
    
    # IN: trust raw decision
    return raw_decision


# ============================================================================
# Ensemble Loading
# ============================================================================

def load_ensemble_members(ensemble_dir: Path, device: torch.device) -> List[Tuple[nn.Module, Normalizer]]:
    """Load all ensemble members from directory."""
    
    meta_path = ensemble_dir / "ensemble_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Ensemble metadata not found: {meta_path}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    models = []
    normalizers = []
    
    for member_info in meta["members"]:
        checkpoint_path = member_info.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError(f"Missing checkpoint for member {member_info['member_idx']}")
        
        checkpoint_path = Path(checkpoint_path)
        # Only prepend ensemble_dir if checkpoint path doesn't exist as-is
        # (handles both absolute paths and paths relative to project root)
        if not checkpoint_path.exists():
            checkpoint_path = ensemble_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["cfg"]
        
        # Create model
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
        
        # Load normalizer
        normalizer = Normalizer()
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
        
        models.append(model)
        normalizers.append(normalizer)
        
        member_idx = member_info.get('member_idx', member_info.get('seed', 'unknown'))
        seed = member_info.get('seed', 'unknown')
        print(f"  Loaded member {member_idx} (seed={seed})")
    
    return list(zip(models, normalizers)), meta


def load_calibration_params(ensemble_dir: Path, calibration_dir: Optional[Path] = None) -> Optional[Dict]:
    """Load conformal calibration params if available."""
    
    # Check multiple possible locations
    candidates = []
    
    # If explicit calibration dir provided, check there first
    if calibration_dir is not None:
        candidates.append(calibration_dir / "conformal_params.json")
        candidates.append(calibration_dir / "ensemble" / "conformal_params.json")
    
    # Standard locations relative to ensemble dir
    candidates.extend([
        ensemble_dir / "calibration" / "conformal_params.json",
        ensemble_dir / "conformal_params.json",
        ensemble_dir.parent / "calibration" / "conformal_params.json",
        ensemble_dir.parent / "calibration" / "ensemble" / "conformal_params.json",
    ])
    
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"Loaded calibration params from {path}")
            return params
    
    print("No calibration params found - using raw quantiles")
    return None


# ============================================================================
# Ensemble Inference
# ============================================================================

@torch.no_grad()
def run_ensemble_inference(
    members: List[Tuple[nn.Module, Normalizer]],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    calibration_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Run inference with ensemble and aggregate predictions.
    
    Aggregation rules:
    - q50 = mean(q50_k) across K members
    - q10_raw = mean(q10_k) across K members  
    - q90_raw = mean(q90_k) across K members
    - epistemic_var = var(q50_k) across K members
    - p_stable = mean(sigmoid(logit_stable_k))
    
    Calibration (if params available):
    - q10_cal = q10_raw - delta_lower
    - q90_cal = q90_raw + delta_upper
    """
    
    K = len(members)
    print(f"Running ensemble inference with K={K} members...")
    
    all_results = []
    
    for batch_idx, (x, src, dst, e, b, y, mids, formulas) in enumerate(loader):
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        batch_size = len(mids)
        
        # Collect predictions from all members
        q10_all = []  # [K, batch_size]
        q50_all = []
        q90_all = []
        p_stable_all = []
        p_meta_all = []
        
        for model, normalizer in members:
            q10, q50, q90, logit_stable, logit_meta = model(x, src, dst, e, b)
            
            # Denormalize
            q10 = normalizer.denorm(q10).cpu().numpy()
            q50 = normalizer.denorm(q50).cpu().numpy()
            q90 = normalizer.denorm(q90).cpu().numpy()
            
            p_stable = torch.sigmoid(logit_stable).cpu().numpy()
            p_meta = torch.sigmoid(logit_meta).cpu().numpy()
            
            q10_all.append(q10)
            q50_all.append(q50)
            q90_all.append(q90)
            p_stable_all.append(p_stable)
            p_meta_all.append(p_meta)
        
        # Stack: [K, batch_size]
        q10_all = np.stack(q10_all, axis=0)
        q50_all = np.stack(q50_all, axis=0)
        q90_all = np.stack(q90_all, axis=0)
        p_stable_all = np.stack(p_stable_all, axis=0)
        p_meta_all = np.stack(p_meta_all, axis=0)
        
        # Aggregate across ensemble (axis=0)
        q10_mean = np.mean(q10_all, axis=0)
        q50_mean = np.mean(q50_all, axis=0)
        q90_mean = np.mean(q90_all, axis=0)
        
        # Epistemic uncertainty = variance of q50 across members
        epistemic_var = np.var(q50_all, axis=0, ddof=1)
        epistemic_std = np.sqrt(epistemic_var)
        
        # Average probabilities (not logits!)
        p_stable_mean = np.mean(p_stable_all, axis=0)
        p_meta_mean = np.mean(p_meta_all, axis=0)
        
        # Apply calibration if available
        if calibration_params is not None:
            delta_upper = calibration_params.get("delta_upper", 0.0)
            delta_lower = calibration_params.get("delta_lower", 0.0)
            q10_cal = q10_mean - delta_lower
            q90_cal = q90_mean + delta_upper
        else:
            q10_cal = q10_mean
            q90_cal = q90_mean
        
        # Aleatoric uncertainty = half interval width
        aleatoric_unc = (q90_cal - q10_cal) / 2.0
        
        # Total uncertainty (combine in quadrature)
        total_unc = np.sqrt(aleatoric_unc**2 + epistemic_var)
        
        # Ground truth
        y_np = y.numpy()
        
        # Build results for this batch
        for i in range(batch_size):
            result = {
                "material_id": mids[i],
                "formula": formulas[i],
                "y_true": float(y_np[i]) if not np.isnan(y_np[i]) else None,
                "q10_raw": float(q10_mean[i]),
                "q50": float(q50_mean[i]),
                "q90_raw": float(q90_mean[i]),
                "q10_cal": float(q10_cal[i]),
                "q90_cal": float(q90_cal[i]),
                "epistemic_var": float(epistemic_var[i]),
                "epistemic_std": float(epistemic_std[i]),
                "aleatoric_unc": float(aleatoric_unc[i]),
                "total_unc": float(total_unc[i]),
                "p_stable": float(p_stable_mean[i]),
                "p_metastable": float(p_meta_mean[i]),
                # Per-member q50 for diagnostics
                "q50_members": [float(q50_all[k, i]) for k in range(K)],
            }
            all_results.append(result)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size} samples...")
    
    return pd.DataFrame(all_results)


def add_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add decision-related columns to predictions DataFrame."""
    
    # Decision based on calibrated q90
    def make_decision(row):
        q90 = row["q90_cal"]
        q10 = row["q10_cal"]
        p_stable = row["p_stable"]
        
        # KEEP: pessimistic bound is stable AND classifier confident
        if q90 < THRESH_STABLE and p_stable > 0.7:
            return "KEEP"
        elif q90 < THRESH_METASTABLE and p_stable > 0.6:
            return "KEEP"
        # KILL: optimistic bound is unstable
        elif q10 > THRESH_METASTABLE:
            return "KILL"
        else:
            return "MAYBE"
    
    df["decision"] = df.apply(make_decision, axis=1)
    
    # Decision confidence based on margin to threshold
    def compute_confidence(row):
        q50 = row["q50"]
        q10 = row["q10_cal"]
        q90 = row["q90_cal"]
        
        if row["decision"] == "KEEP":
            # Confidence = how far q90 is below threshold
            margin = THRESH_STABLE - q90
            return min(1.0, margin / 0.05)
        elif row["decision"] == "KILL":
            # Confidence = how far q10 is above threshold
            margin = q10 - THRESH_METASTABLE
            return min(1.0, margin / 0.10)
        else:
            # MAYBE: confidence is low
            return 0.3
    
    df["decision_confidence"] = df.apply(compute_confidence, axis=1)
    
    return df


def main():
    ap = argparse.ArgumentParser(description="Run ensemble inference on a dataset")
    ap.add_argument("--ensemble-dir", required=True, help="Path to ensemble directory")
    ap.add_argument("--data-config", required=True, help="Data config YAML")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                    help="Dataset split to predict on (default: test)")
    ap.add_argument("--output", required=True, help="Output path (.parquet or .csv)")
    ap.add_argument("--ood-dir", default=None, help="Path to OOD artifacts directory (optional)")
    ap.add_argument("--calibration-dir", default=None, help="Path to calibration directory (optional)")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load ensemble
    ensemble_dir = Path(args.ensemble_dir)
    members, meta = load_ensemble_members(ensemble_dir, device)
    print(f"Loaded {len(members)} ensemble members")
    
    # Load calibration params
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None
    calibration_params = load_calibration_params(ensemble_dir, calibration_dir)
    
    # Load OOD gate if provided
    ood_gate = None
    if args.ood_dir:
        ood_dir = Path(args.ood_dir)
        if ood_dir.exists():
            ood_gate = OODGate.from_artifacts(ood_dir)
            print(f"Loaded OOD gate from {ood_dir}")
            print(f"  Thresholds: t_hi={ood_gate.t_hi:.3f}, t_mid={ood_gate.t_mid:.3f}")
        else:
            print(f"Warning: OOD directory not found: {ood_dir}")
    
    # Load data config
    data_cfg = load_cfg(args.data_config)
    
    # Load processed data
    processed_dir = Path(data_cfg["data"]["processed_dir"])
    ds_name = data_cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)
    print(f"Loaded {len(df)} total samples from {processed_path}")
    
    # Load splits
    splits_path = Path(data_cfg["data"]["splits_manifest"])
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
    else:
        raise FileNotFoundError(f"Splits manifest not found: {splits_path}")
    
    # Filter to requested split
    if args.split == "all":
        df_split = df
    else:
        split_ids = set(splits[args.split])
        df_split = df[df["material_id"].isin(split_ids)].reset_index(drop=True)
    print(f"Predicting on {args.split} split: {len(df_split)} samples")
    
    # Create dataset and loader
    target_col = data_cfg["data"]["target"]["name"]
    dataset = GraphNPZDataset(df_split, target_col)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate
    )
    
    # Run inference
    results_df = run_ensemble_inference(members, loader, device, calibration_params)
    
    # Add decision columns
    results_df = add_decision_columns(results_df)
    
    # Add OOD gating if available
    if ood_gate is not None:
        print("Computing OOD scores...")
        ood_scores = []
        gate_levels = []
        gated_decisions = []
        
        for _, row in results_df.iterrows():
            ood_score, gate_level, _ = ood_gate.score(
                formula=row["formula"],
                epistemic_var=row["epistemic_var"],
                embedding=None,  # Would need embeddings from inference
            )
            ood_scores.append(ood_score)
            gate_levels.append(gate_level)
            
            gated_decision = apply_ood_gating(
                raw_decision=row["decision"],
                gate_level=gate_level,
                q90_cal=row["q90_cal"],
                p_stable=row["p_stable"],
            )
            gated_decisions.append(gated_decision)
        
        results_df["ood_score"] = ood_scores
        results_df["gate_level"] = gate_levels
        results_df["gated_decision"] = gated_decisions
    else:
        # No OOD gating - default values
        results_df["ood_score"] = 0.0
        results_df["gate_level"] = "IN"
        results_df["gated_decision"] = results_df["decision"]
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(results_df)}")
    
    if results_df["y_true"].notna().any():
        y_true = results_df["y_true"].dropna().values
        q50 = results_df.loc[results_df["y_true"].notna(), "q50"].values
        q10_cal = results_df.loc[results_df["y_true"].notna(), "q10_cal"].values
        q90_cal = results_df.loc[results_df["y_true"].notna(), "q90_cal"].values
        
        mae = np.mean(np.abs(y_true - q50))
        rmse = np.sqrt(np.mean((y_true - q50)**2))
        coverage = np.mean((y_true >= q10_cal) & (y_true <= q90_cal))
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Coverage (calibrated 80% interval): {coverage:.2%}")
    
    print(f"\nDecision distribution (raw):")
    print(results_df["decision"].value_counts())
    
    if ood_gate is not None:
        print(f"\nOOD gate levels:")
        print(results_df["gate_level"].value_counts())
        
        print(f"\nDecision distribution (after OOD gating):")
        print(results_df["gated_decision"].value_counts())
        
        # Count gating changes
        n_changed = (results_df["decision"] != results_df["gated_decision"]).sum()
        print(f"\nDecisions changed by OOD gating: {n_changed}")
    
    print(f"\nEpistemic uncertainty:")
    print(f"  Mean: {results_df['epistemic_std'].mean():.4f}")
    print(f"  Median: {results_df['epistemic_std'].median():.4f}")
    print(f"  Max: {results_df['epistemic_std'].max():.4f}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove q50_members list column for parquet (convert to string)
    results_df["q50_members_str"] = results_df["q50_members"].apply(json.dumps)
    results_df_out = results_df.drop(columns=["q50_members"])
    
    if output_path.suffix == ".parquet":
        results_df_out.to_parquet(output_path, index=False)
    else:
        results_df_out.to_csv(output_path, index=False)
    
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
