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
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union

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


def save_calibration_params(params: ConformalCalibrationParams, path: Union[str, Path]) -> None:
    """Save calibration parameters to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)


def load_calibration_params(path: Union[str, Path]) -> ConformalCalibrationParams:
    """Load calibration parameters from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ConformalCalibrationParams(**data)


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
