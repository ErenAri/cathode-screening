"""
OOD (Out-of-Distribution) Gating Module for Decision-Grade Cathode Screening.

Three gates for detecting out-of-distribution inputs:
1. Composition distance (Mahalanobis to training centroids)
2. Embedding kNN distance (k=10, aggregated over ensemble)
3. Ensemble disagreement (std of q50 predictions)

OOD Score = max(z_comp, z_emb, z_disagree) mapped through sigmoid.

Gating Policy:
- ood_score > t_hi  → DEFER (force KEEP → MAYBE)
- ood_score > t_mid → Stricter KEEP criteria (tighter q90, higher p_stable)
- ood_score < t_mid → Trust base decision

Usage:
    from cathode_screening.inference.ood import OODGate
    
    gate = OODGate.from_artifacts("data/artifacts/ood")
    result = gate.score(formula, embedding, q50_ensemble)
    decision = gate.apply_gating(raw_decision, result, q90_cal, p_stable)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Embedding kNN gate will be disabled.")

from pymatgen.core import Composition


# Composition vocabulary (must match training)
COMPOSITION_VOCAB = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]

# Default thresholds (should be calibrated on validation set)
DEFAULT_T_HI = 0.70    # High OOD → DEFER
DEFAULT_T_MID = 0.50   # Medium OOD → stricter KEEP


def composition_fingerprint(formula: str) -> np.ndarray:
    """Convert formula to normalized composition vector."""
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(COMPOSITION_VOCAB), dtype=np.float32)
    for i, el in enumerate(COMPOSITION_VOCAB):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


@dataclass
class OODGateResult:
    """Result from OOD gating module."""
    ood_score: float              # Combined score [0, 1], higher = more OOD
    flag: bool                    # True if any gate triggered
    
    # Individual gate values (raw distances)
    d_comp: float                 # Mahalanobis distance
    d_emb: float                  # kNN distance
    d_disagree: float             # Ensemble disagreement (std)
    
    # Normalized z-scores
    z_comp: float
    z_emb: float
    z_disagree: float
    z_max: float                  # max(z_comp, z_emb, z_disagree)
    
    # Individual gate flags (raw thresholds)
    flag_comp: bool
    flag_emb: bool
    flag_disagree: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OODThresholds:
    """Calibrated OOD thresholds."""
    t_hi: float                   # Above this → DEFER (force MAYBE)
    t_mid: float                  # Above this → stricter KEEP
    
    # Stricter KEEP criteria for medium OOD
    strict_q90_thresh: float = 0.03      # q90 < this for KEEP
    strict_p_stable_thresh: float = 0.85  # p_stable > this for KEEP
    
    # Metadata
    target_fkr: float = 0.05             # Target false KEEP rate
    achieved_fkr: float = 0.0            # Actual FKR at t_hi
    n_val_samples: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "OODThresholds":
        return cls(**d)
    
    @classmethod
    def default(cls) -> "OODThresholds":
        return cls(t_hi=DEFAULT_T_HI, t_mid=DEFAULT_T_MID)


class OODGate:
    """
    Out-of-Distribution gating module.
    
    Combines three gates:
    1. Composition Mahalanobis distance
    2. Embedding kNN distance (requires FAISS)
    3. Ensemble disagreement
    
    OOD score = sigmoid(max(z_scores) - 2)
    """
    
    def __init__(
        self,
        comp_mu: np.ndarray,
        comp_cov_inv: np.ndarray,
        comp_stats: Dict[str, float],
        emb_index: Optional[object],  # FAISS index
        emb_stats: Dict[str, float],
        disagree_stats: Dict[str, float],
        thresholds: Optional[OODThresholds] = None,
    ):
        self.comp_mu = comp_mu
        self.comp_cov_inv = comp_cov_inv
        self.comp_stats = comp_stats
        
        self.emb_index = emb_index
        self.emb_stats = emb_stats
        
        self.disagree_stats = disagree_stats
        
        # Calibrated thresholds
        self.thresholds = thresholds or OODThresholds.default()
    
    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> "OODGate":
        """Load OOD gate from saved artifacts."""
        artifact_dir = Path(artifact_dir)
        
        # Load composition stats
        comp_path = artifact_dir / "composition_stats.npz"
        if comp_path.exists():
            comp_data = np.load(comp_path)
            comp_mu = comp_data["mu"]
            comp_cov_inv = comp_data["cov_inv"]
        else:
            raise FileNotFoundError(f"Composition stats not found: {comp_path}")
        
        # Load train stats
        stats_path = artifact_dir / "train_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            raise FileNotFoundError(f"Train stats not found: {stats_path}")
        
        comp_stats = {
            "mean": stats["comp_mean"],
            "std": stats["comp_std"],
            "tau_p95": stats["tau_comp_p95"],
            "tau_p99": stats["tau_comp_p99"],
        }
        
        emb_stats = {
            "mean": stats["emb_mean"],
            "std": stats["emb_std"],
            "tau_p95": stats["tau_emb_p95"],
            "tau_p99": stats["tau_emb_p99"],
        }
        
        disagree_stats = {
            "mean": stats["disagree_mean"],
            "std": stats["disagree_std"],
            "tau_p95": stats["tau_disagree_p95"],
            "tau_p99": stats["tau_disagree_p99"],
        }
        
        # Load FAISS index if available
        emb_index = None
        emb_index_path = artifact_dir / "embedding_index.faiss"
        if FAISS_AVAILABLE and emb_index_path.exists():
            emb_index = faiss.read_index(str(emb_index_path))
        
        # Load calibrated thresholds if available
        thresholds_path = artifact_dir / "ood_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds = OODThresholds.from_dict(json.load(f))
        else:
            thresholds = OODThresholds.default()
        
        return cls(
            comp_mu=comp_mu,
            comp_cov_inv=comp_cov_inv,
            comp_stats=comp_stats,
            emb_index=emb_index,
            emb_stats=emb_stats,
            disagree_stats=disagree_stats,
            thresholds=thresholds,
        )
    
    def mahalanobis_distance(self, formula: str) -> float:
        """Compute Mahalanobis distance for composition."""
        x = composition_fingerprint(formula)
        delta = x - self.comp_mu
        d_sq = delta @ self.comp_cov_inv @ delta
        return float(np.sqrt(max(0, d_sq)))
    
    def embedding_knn_distance(
        self, 
        embedding: np.ndarray, 
        k: int = 10
    ) -> float:
        """Compute mean distance to k nearest training embeddings."""
        if self.emb_index is None:
            return 0.0  # Disabled
        
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        distances, _ = self.emb_index.search(embedding, k)
        return float(distances[0].mean())
    
    def ensemble_disagreement(self, q50_ensemble: List[float]) -> float:
        """Compute standard deviation of ensemble predictions."""
        return float(np.std(q50_ensemble))
    
    def score(
        self,
        formula: str,
        embedding: Optional[np.ndarray] = None,
        q50_ensemble: Optional[List[float]] = None,
        threshold_level: str = "p95",  # "p95" or "p99"
    ) -> OODGateResult:
        """
        Compute OOD score and individual gate flags.
        
        OOD score = sigmoid(max(z_scores) - 2)
        
        Args:
            formula: Material formula (e.g., "LiCoO2")
            embedding: Concatenated graph embeddings from ensemble [dim * K]
            q50_ensemble: List of q50 predictions from each ensemble member
            threshold_level: Which threshold to use for flagging ("p95" or "p99")
        
        Returns:
            OODGateResult with combined score and individual gate details
        """
        tau_key = f"tau_{threshold_level}"
        
        # Gate 1: Composition
        d_comp = self.mahalanobis_distance(formula)
        z_comp = (d_comp - self.comp_stats["mean"]) / max(self.comp_stats["std"], 1e-6)
        flag_comp = d_comp > self.comp_stats[tau_key]
        
        # Gate 2: Embedding kNN
        if embedding is not None and self.emb_index is not None:
            d_emb = self.embedding_knn_distance(embedding)
            z_emb = (d_emb - self.emb_stats["mean"]) / max(self.emb_stats["std"], 1e-6)
            flag_emb = d_emb > self.emb_stats[tau_key]
        else:
            d_emb = 0.0
            z_emb = 0.0
            flag_emb = False
        
        # Gate 3: Disagreement
        if q50_ensemble is not None and len(q50_ensemble) > 1:
            d_disagree = self.ensemble_disagreement(q50_ensemble)
            z_disagree = (d_disagree - self.disagree_stats["mean"]) / max(self.disagree_stats["std"], 1e-6)
            flag_disagree = d_disagree > self.disagree_stats[tau_key]
        else:
            d_disagree = 0.0
            z_disagree = 0.0
            flag_disagree = False
        
        # Combined score: max of z-scores (conservative)
        z_max = max(z_comp, z_emb, z_disagree)
        
        # Sigmoid with shift: z=0 → 0.12, z=2 → 0.5, z=4 → 0.88
        ood_score = 1.0 / (1.0 + np.exp(-z_max + 2.0))
        
        return OODGateResult(
            ood_score=float(ood_score),
            flag=flag_comp or flag_emb or flag_disagree,
            d_comp=d_comp,
            d_emb=d_emb,
            d_disagree=d_disagree,
            z_comp=z_comp,
            z_emb=z_emb,
            z_disagree=z_disagree,
            z_max=z_max,
            flag_comp=flag_comp,
            flag_emb=flag_emb,
            flag_disagree=flag_disagree,
        )
    
    def apply_gating(
        self,
        raw_decision: str,
        ood_result: OODGateResult,
        q90_cal: float,
        p_stable: float,
    ) -> Tuple[str, str]:
        """
        Apply OOD gating policy to modify decision.
        
        Policy:
        - ood_score > t_hi  → DEFER (force KEEP → MAYBE)
        - ood_score > t_mid → Stricter KEEP (tighter q90, higher p_stable)
        - ood_score < t_mid → Trust base decision
        
        Args:
            raw_decision: Original decision from predictor (KEEP/MAYBE/KILL)
            ood_result: OOD scoring result
            q90_cal: Calibrated upper bound
            p_stable: Stability probability
        
        Returns:
            Tuple of (gated_decision, gate_action) where gate_action is
            "none", "strict", or "defer"
        """
        t = self.thresholds
        score = ood_result.ood_score
        
        # High OOD → DEFER
        if score > t.t_hi:
            if raw_decision == "KEEP":
                return "MAYBE", "defer"
            return raw_decision, "none"
        
        # Medium OOD → Stricter KEEP criteria
        if score > t.t_mid:
            if raw_decision == "KEEP":
                # Apply stricter criteria
                if q90_cal < t.strict_q90_thresh and p_stable > t.strict_p_stable_thresh:
                    return "KEEP", "strict"  # Passed strict check
                else:
                    return "MAYBE", "strict"  # Failed strict check
            return raw_decision, "none"
        
        # Low OOD → Trust base decision
        return raw_decision, "none"


def compute_train_stats(
    train_formulas: List[str],
    train_embeddings: np.ndarray,
    train_disagreements: np.ndarray,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Compute training statistics for OOD detection.
    
    Args:
        train_formulas: List of training set formulas
        train_embeddings: [N_train, embedding_dim] array
        train_disagreements: [N_train] array of ensemble disagreement per sample
    
    Returns:
        stats_dict: Dictionary with means, stds, and thresholds
        comp_mu: Mean composition vector
        comp_cov_inv: Inverse covariance matrix for composition
    """
    # Composition statistics
    X_comp = np.stack([composition_fingerprint(f) for f in train_formulas])
    comp_mu = X_comp.mean(axis=0)
    comp_cov = np.cov(X_comp.T) + 1e-6 * np.eye(X_comp.shape[1])
    comp_cov_inv = np.linalg.inv(comp_cov)
    
    # Compute Mahalanobis distances on training set
    comp_dists = []
    for x in X_comp:
        delta = x - comp_mu
        d_sq = delta @ comp_cov_inv @ delta
        comp_dists.append(np.sqrt(max(0, d_sq)))
    comp_dists = np.array(comp_dists)
    
    # Embedding kNN distances (LOO approximation)
    emb_dists = []
    if FAISS_AVAILABLE and train_embeddings is not None:
        index = faiss.IndexFlatL2(train_embeddings.shape[1])
        index.add(train_embeddings.astype(np.float32))
        
        for i in range(len(train_embeddings)):
            dists, _ = index.search(train_embeddings[i:i+1].astype(np.float32), k=11)
            emb_dists.append(dists[0, 1:].mean())  # Skip self (index 0)
    else:
        emb_dists = np.zeros(len(train_formulas))
    emb_dists = np.array(emb_dists)
    
    stats = {
        "comp_mean": float(comp_dists.mean()),
        "comp_std": float(comp_dists.std()),
        "tau_comp_p95": float(np.percentile(comp_dists, 95)),
        "tau_comp_p99": float(np.percentile(comp_dists, 99)),
        
        "emb_mean": float(emb_dists.mean()),
        "emb_std": float(emb_dists.std()),
        "tau_emb_p95": float(np.percentile(emb_dists, 95)),
        "tau_emb_p99": float(np.percentile(emb_dists, 99)),
        
        "disagree_mean": float(train_disagreements.mean()),
        "disagree_std": float(train_disagreements.std()),
        "tau_disagree_p95": float(np.percentile(train_disagreements, 95)),
        "tau_disagree_p99": float(np.percentile(train_disagreements, 99)),
    }
    
    return stats, comp_mu, comp_cov_inv


def save_ood_artifacts(
    output_dir: str | Path,
    stats: Dict,
    comp_mu: np.ndarray,
    comp_cov_inv: np.ndarray,
    train_embeddings: Optional[np.ndarray] = None,
):
    """Save OOD gate artifacts to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stats
    with open(output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Save composition stats
    np.savez(
        output_dir / "composition_stats.npz",
        mu=comp_mu,
        cov_inv=comp_cov_inv,
    )
    
    # Save FAISS index
    if FAISS_AVAILABLE and train_embeddings is not None:
        index = faiss.IndexFlatL2(train_embeddings.shape[1])
        index.add(train_embeddings.astype(np.float32))
        faiss.write_index(index, str(output_dir / "embedding_index.faiss"))
    
    print(f"Saved OOD artifacts to {output_dir}")


def select_ood_thresholds(
    val_ood_scores: np.ndarray,
    val_decisions: np.ndarray,
    val_y_true: np.ndarray,
    target_fkr: float = 0.05,
    thresh_unstable: float = 0.10,
) -> OODThresholds:
    """
    Select OOD thresholds to control false KEEP rate.
    
    Procedure:
    1. Grid search over candidate thresholds
    2. Find threshold where false_keep_rate <= target among filtered samples
    3. Maximize number of valid KEEPs at that FKR
    
    Args:
        val_ood_scores: [N] OOD scores for validation samples
        val_decisions: [N] decisions ("KEEP"/"MAYBE"/"KILL")
        val_y_true: [N] ground truth E_hull
        target_fkr: Target false KEEP rate (default 0.05)
        thresh_unstable: E_hull threshold for "unstable" (default 0.10)
    
    Returns:
        OODThresholds with t_hi, t_mid, and metadata
    """
    candidates = np.linspace(0.1, 0.95, 86)  # 0.1 to 0.95 in 0.01 steps
    
    best_t_hi = 0.95
    best_fkr = 1.0
    best_n_keeps = 0
    
    for t in candidates:
        # Filter to samples with ood_score < t
        mask = val_ood_scores < t
        decisions_filtered = val_decisions[mask]
        y_filtered = val_y_true[mask]
        
        # Find KEEPs in this filtered set
        keep_mask = decisions_filtered == "KEEP"
        n_keeps = keep_mask.sum()
        
        if n_keeps == 0:
            continue
        
        # False KEEP rate: fraction of KEEPs that are actually unstable
        y_keeps = y_filtered[keep_mask]
        fkr = (y_keeps > thresh_unstable).mean()
        
        # Accept if FKR within target and more KEEPs than previous best
        if fkr <= target_fkr and n_keeps > best_n_keeps:
            best_t_hi = t
            best_fkr = fkr
            best_n_keeps = n_keeps
    
    # Set t_mid = t_hi - 0.15 (medium zone before hard cutoff)
    t_mid = max(0.2, best_t_hi - 0.15)
    
    return OODThresholds(
        t_hi=float(best_t_hi),
        t_mid=float(t_mid),
        target_fkr=target_fkr,
        achieved_fkr=float(best_fkr),
        n_val_samples=len(val_ood_scores),
    )


def save_ood_thresholds(thresholds: OODThresholds, output_dir: Path) -> None:
    """Save OOD thresholds to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ood_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds.to_dict(), f, indent=2)
    
    print(f"Saved OOD thresholds: t_hi={thresholds.t_hi:.3f}, t_mid={thresholds.t_mid:.3f}")
    print(f"  Target FKR={thresholds.target_fkr:.2%}, Achieved FKR={thresholds.achieved_fkr:.2%}")
