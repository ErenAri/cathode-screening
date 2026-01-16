"""
Split conformal calibration for quantile regression.

Applies finite-sample valid corrections to q10/q90 bounds to achieve
guaranteed coverage at level (1-alpha).

Nonconformity scores:
    Upper: s_i = y_i - q90_i  (positive when q90 underestimates)
    Lower: s_i = q10_i - y_i  (positive when q10 overestimates)

Calibration:
    k = ceil((n+1)*(1-alpha))
    Δ_upper = k-th order statistic of upper scores
    Δ_lower = k-th order statistic of lower scores

Calibrated bounds:
    q90_cal = q90 + Δ_upper
    q10_cal = q10 - Δ_lower

Group-Conditional Mode:
    Per-cluster Δ_upper/Δ_lower when n_cluster >= N_min (default 50).
    Falls back to global deltas for small clusters.
    Storage: {cluster_id: {delta_upper, delta_lower}, "_global": {...}}
"""
import json
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List

import numpy as np


@dataclass
class ConformalCalibrationParams:
    """Parameters for conformal calibration of quantile bounds."""
    alpha: float
    n_calibration: int
    delta_upper: float
    delta_lower: float
    timestamp: str
    split_name: str
    checkpoint_path: Optional[str] = None
    
    # Diagnostics
    raw_coverage: Optional[float] = None
    calibrated_coverage: Optional[float] = None


@dataclass
class GroupConformalParams:
    """
    Group-conditional conformal calibration parameters.
    
    Stores per-cluster deltas with fallback to global.
    Format: {cluster_id: {delta_upper, delta_lower, n}, "_global": {...}}
    """
    alpha: float
    n_min: int  # Minimum samples for per-cluster calibration
    timestamp: str
    split_name: str
    checkpoint_path: Optional[str] = None
    
    # Per-cluster deltas: cluster_id -> {"delta_upper", "delta_lower", "n", "coverage"}
    cluster_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Global fallback
    global_delta_upper: float = 0.0
    global_delta_lower: float = 0.0
    global_n: int = 0
    global_raw_coverage: Optional[float] = None
    global_calibrated_coverage: Optional[float] = None
    
    def get_deltas(self, cluster_id: Optional[str] = None) -> Tuple[float, float]:
        """Get (delta_upper, delta_lower) for a cluster, falling back to global."""
        if cluster_id is not None and cluster_id in self.cluster_deltas:
            d = self.cluster_deltas[cluster_id]
            return d["delta_upper"], d["delta_lower"]
        return self.global_delta_upper, self.global_delta_lower
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alpha": self.alpha,
            "n_min": self.n_min,
            "timestamp": self.timestamp,
            "split_name": self.split_name,
            "checkpoint_path": self.checkpoint_path,
            "_global": {
                "delta_upper": self.global_delta_upper,
                "delta_lower": self.global_delta_lower,
                "n": self.global_n,
                "raw_coverage": self.global_raw_coverage,
                "calibrated_coverage": self.global_calibrated_coverage,
            },
            **{k: v for k, v in self.cluster_deltas.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupConformalParams":
        """Load from dictionary."""
        global_data = data.pop("_global", {})
        cluster_deltas = {
            k: v for k, v in data.items() 
            if k not in ("alpha", "n_min", "timestamp", "split_name", "checkpoint_path")
        }
        return cls(
            alpha=data["alpha"],
            n_min=data["n_min"],
            timestamp=data["timestamp"],
            split_name=data["split_name"],
            checkpoint_path=data.get("checkpoint_path"),
            cluster_deltas=cluster_deltas,
            global_delta_upper=global_data.get("delta_upper", 0.0),
            global_delta_lower=global_data.get("delta_lower", 0.0),
            global_n=global_data.get("n", 0),
            global_raw_coverage=global_data.get("raw_coverage"),
            global_calibrated_coverage=global_data.get("calibrated_coverage"),
        )


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute conformal quantile with finite-sample correction.
    
    k = ceil((n+1)*(1-alpha))
    Returns the k-th order statistic (1-indexed), or equivalently
    the ceil((n+1)*(1-alpha))/n quantile.
    
    Args:
        scores: [N] array of nonconformity scores
        alpha: Miscoverage level (e.g., 0.10 for 90% coverage)
    
    Returns:
        Conformal quantile value
    """
    n = len(scores)
    # k = ceil((n+1)*(1-alpha)), but we need index for 0-based array
    # quantile level = k/n = ceil((n+1)*(1-alpha))/n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # Cap at 1.0
    return float(np.quantile(scores, q_level))


def fit_conformal_calibration(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    alpha: float = 0.10,
    split_name: str = "val",
    checkpoint_path: Optional[str] = None
) -> ConformalCalibrationParams:
    """
    Fit conformal calibration parameters from validation predictions.
    
    For a target coverage of (1-alpha), we use alpha/2 for each tail since
    we're calibrating both bounds of the interval independently.
    
    Args:
        y_true: [N] ground truth values
        q10: [N] lower quantile predictions (10th percentile)
        q90: [N] upper quantile predictions (90th percentile)
        alpha: Target miscoverage rate (default 0.10 for 90% coverage)
        split_name: Name of the calibration split
        checkpoint_path: Path to model checkpoint used for predictions
    
    Returns:
        ConformalCalibrationParams with delta_upper and delta_lower
    """
    n = len(y_true)
    assert len(q10) == n and len(q90) == n, "Arrays must have same length"
    
    # Compute nonconformity scores
    s_upper = y_true - q90  # Positive when y > q90 (undercoverage)
    s_lower = q10 - y_true  # Positive when q10 > y (undercoverage)
    
    # Use alpha/2 for each tail to achieve (1-alpha) total coverage
    # (Bonferroni-style correction for two-sided intervals)
    alpha_per_tail = alpha / 2
    delta_upper = compute_conformal_quantile(s_upper, alpha_per_tail)
    delta_lower = compute_conformal_quantile(s_lower, alpha_per_tail)
    
    # Compute coverage diagnostics
    raw_coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    q10_cal = q10 - delta_lower
    q90_cal = q90 + delta_upper
    calibrated_coverage = float(np.mean((y_true >= q10_cal) & (y_true <= q90_cal)))
    
    return ConformalCalibrationParams(
        alpha=alpha,
        n_calibration=n,
        delta_upper=delta_upper,
        delta_lower=delta_lower,
        timestamp=datetime.now().isoformat(),
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        raw_coverage=raw_coverage,
        calibrated_coverage=calibrated_coverage
    )


def fit_group_conformal_calibration(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    cluster_ids: np.ndarray,
    alpha: float = 0.10,
    n_min: int = 50,
    split_name: str = "val",
    checkpoint_path: Optional[str] = None
) -> GroupConformalParams:
    """
    Fit group-conditional conformal calibration with per-cluster deltas.
    
    For each cluster with n >= n_min samples, computes cluster-specific
    Δ_upper and Δ_lower. Smaller clusters fall back to global deltas.
    
    Args:
        y_true: [N] ground truth values
        q10: [N] lower quantile predictions
        q90: [N] upper quantile predictions
        cluster_ids: [N] cluster labels for each sample
        alpha: Target miscoverage rate (default 0.10 for 90% coverage)
        n_min: Minimum samples for per-cluster calibration (default 50)
        split_name: Name of the calibration split
        checkpoint_path: Path to model checkpoint
    
    Returns:
        GroupConformalParams with per-cluster and global deltas
    """
    n = len(y_true)
    assert len(q10) == n and len(q90) == n and len(cluster_ids) == n
    
    # Compute nonconformity scores
    s_upper = y_true - q90
    s_lower = q10 - y_true
    alpha_per_tail = alpha / 2
    
    # Global calibration (always computed as fallback)
    global_delta_upper = compute_conformal_quantile(s_upper, alpha_per_tail)
    global_delta_lower = compute_conformal_quantile(s_lower, alpha_per_tail)
    
    global_raw_coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    q10_cal_global = q10 - global_delta_lower
    q90_cal_global = q90 + global_delta_upper
    global_calibrated_coverage = float(np.mean(
        (y_true >= q10_cal_global) & (y_true <= q90_cal_global)
    ))
    
    # Per-cluster calibration
    cluster_deltas = {}
    unique_clusters = np.unique(cluster_ids)
    
    for cid in unique_clusters:
        mask = cluster_ids == cid
        n_cluster = mask.sum()
        
        if n_cluster >= n_min:
            # Compute cluster-specific deltas
            d_upper = compute_conformal_quantile(s_upper[mask], alpha_per_tail)
            d_lower = compute_conformal_quantile(s_lower[mask], alpha_per_tail)
            
            # Cluster coverage diagnostics
            y_c, q10_c, q90_c = y_true[mask], q10[mask], q90[mask]
            raw_cov = float(np.mean((y_c >= q10_c) & (y_c <= q90_c)))
            cal_cov = float(np.mean(
                (y_c >= q10_c - d_lower) & (y_c <= q90_c + d_upper)
            ))
            
            cluster_deltas[str(cid)] = {
                "delta_upper": float(d_upper),
                "delta_lower": float(d_lower),
                "n": int(n_cluster),
                "raw_coverage": raw_cov,
                "calibrated_coverage": cal_cov,
            }
    
    return GroupConformalParams(
        alpha=alpha,
        n_min=n_min,
        timestamp=datetime.now().isoformat(),
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        cluster_deltas=cluster_deltas,
        global_delta_upper=float(global_delta_upper),
        global_delta_lower=float(global_delta_lower),
        global_n=n,
        global_raw_coverage=global_raw_coverage,
        global_calibrated_coverage=global_calibrated_coverage,
    )


def apply_conformal_calibration(
    q10: np.ndarray,
    q90: np.ndarray,
    params: ConformalCalibrationParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conformal calibration to quantile predictions.
    
    Args:
        q10: [N] lower quantile predictions
        q90: [N] upper quantile predictions
        params: Calibration parameters
    
    Returns:
        q10_cal: [N] calibrated lower bounds
        q90_cal: [N] calibrated upper bounds
    """
    q10_cal = q10 - params.delta_lower
    q90_cal = q90 + params.delta_upper
    return q10_cal, q90_cal


def apply_group_conformal_calibration(
    q10: np.ndarray,
    q90: np.ndarray,
    cluster_ids: np.ndarray,
    params: GroupConformalParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Apply group-conditional conformal calibration.
    
    Uses per-cluster deltas when available, falling back to global.
    
    Args:
        q10: [N] lower quantile predictions
        q90: [N] upper quantile predictions
        cluster_ids: [N] cluster labels for each sample
        params: Group calibration parameters
    
    Returns:
        q10_cal: [N] calibrated lower bounds
        q90_cal: [N] calibrated upper bounds
        is_cluster_calibrated: [N] bool array, True if per-cluster delta used
        calibrated_source: [N] list of "cluster" or "global"
    """
    n = len(q10)
    q10_cal = np.zeros(n)
    q90_cal = np.zeros(n)
    is_cluster_calibrated = np.zeros(n, dtype=bool)
    calibrated_source: List[str] = []
    
    for i in range(n):
        cid = str(cluster_ids[i])
        if cid in params.cluster_deltas:
            d = params.cluster_deltas[cid]
            d_upper, d_lower = d["delta_upper"], d["delta_lower"]
            is_cluster_calibrated[i] = True
            calibrated_source.append("cluster")
        else:
            d_upper, d_lower = params.global_delta_upper, params.global_delta_lower
            is_cluster_calibrated[i] = False
            calibrated_source.append("global")
        q10_cal[i] = q10[i] - d_lower
        q90_cal[i] = q90[i] + d_upper
    
    return q10_cal, q90_cal, is_cluster_calibrated, calibrated_source


def save_calibration_params(params: ConformalCalibrationParams, path: Union[str, Path]) -> None:
    """Save calibration parameters to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)


def save_group_calibration_params(params: GroupConformalParams, path: Union[str, Path]) -> None:
    """Save group calibration parameters to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.to_dict(), f, indent=2)


def load_calibration_params(path: Union[str, Path]) -> ConformalCalibrationParams:
    """Load calibration parameters from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    allowed = {f.name for f in fields(ConformalCalibrationParams)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return ConformalCalibrationParams(**filtered)


def load_group_calibration_params(path: Union[str, Path]) -> GroupConformalParams:
    """Load group calibration parameters from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GroupConformalParams.from_dict(data)


class ConformalCalibrator:
    """
    Wrapper class for applying conformal calibration to predictions.
    
    Usage:
        calibrator = ConformalCalibrator.from_file("calibration_params.json")
        q10_cal, q90_cal = calibrator.calibrate(q10, q90)
    """
    
    def __init__(self, params: ConformalCalibrationParams):
        self.params = params
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ConformalCalibrator":
        """Load calibrator from JSON file."""
        params = load_calibration_params(path)
        return cls(params)
    
    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        q10: np.ndarray,
        q90: np.ndarray,
        alpha: float = 0.10,
        **kwargs
    ) -> "ConformalCalibrator":
        """Fit calibrator from validation data."""
        params = fit_conformal_calibration(y_true, q10, q90, alpha, **kwargs)
        return cls(params)
    
    def calibrate(
        self,
        q10: np.ndarray,
        q90: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibration to quantile predictions."""
        return apply_conformal_calibration(q10, q90, self.params)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save calibration parameters to file."""
        save_calibration_params(self.params, path)
    
    @property
    def delta_upper(self) -> float:
        return self.params.delta_upper
    
    @property
    def delta_lower(self) -> float:
        return self.params.delta_lower
    
    @property
    def alpha(self) -> float:
        return self.params.alpha
