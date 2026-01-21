import numpy as np
import json
from typing import List, Optional, Union
import os
from sklearn.linear_model import LogisticRegression

class CalibratedPredictor:
    """
    Calibrates ensemble predictions to probabilities using Platt scaling.
    
    P(stable | predictions) = sigmoid(a * z + b)
    where z = (threshold - mean) / std

    feature_mode:
    - "z": z-score (threshold - mean) / std
    - "mu": mean energy only
    - "mu_sigma": mean and std as two features
    
    Features:
    - Numerical guards (sigma_floor, z_clip)
    - Persistence (save/load)
    - Class-balanced fitting
    """
    
    SIGMA_FLOOR = 1e-3    # Minimum sigma to avoid z explosion
    Z_CLIP = 10.0         # Maximum |z| value
    
    def __init__(self, threshold: float = 0.0, feature_mode: str = "z"):
        """
        Args:
            threshold: Stability threshold (e.g., 0.0 eV/atom).
                       Predictions < threshold are considered stable.
        """
        self.threshold = threshold
        self.feature_mode = feature_mode.strip().lower()
        if self.feature_mode not in {"z", "mu", "mu_sigma"}:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")
        self.platt_a: Optional[Union[float, List[float]]] = None
        self.platt_b: Optional[float] = None
        self.is_fitted = False

    def _featurize(self, preds: List[float]) -> np.ndarray:
        mu = float(np.mean(preds))
        if self.feature_mode == "mu":
            return np.array([mu], dtype=float)
        sigma = max(float(np.std(preds)), self.SIGMA_FLOOR)
        if self.feature_mode == "mu_sigma":
            return np.array([mu, sigma], dtype=float)
        z = (self.threshold - mu) / sigma
        z = float(np.clip(z, -self.Z_CLIP, self.Z_CLIP))
        return np.array([z], dtype=float)
    
    def fit(self, preds: List[List[float]], labels: List[bool]):
        """
        Fit calibrator on ensemble predictions and ground truth labels.
        
        Args:
            preds: List of ensemble predictions for each sample [[p1, p2..], ...]
            labels: True stability labels (True=Stable)
        """
        X = np.vstack([self._featurize(p) for p in preds])
        y = np.array(labels).astype(int)
        
        # Class-balanced for rare positives
        lr = LogisticRegression(class_weight="balanced", solver="lbfgs")
        lr.fit(X, y)
        
        coef = lr.coef_[0]
        if coef.shape[0] == 1:
            self.platt_a = float(coef[0])
        else:
            self.platt_a = coef.astype(float).tolist()
        self.platt_b = float(lr.intercept_[0])
        self.is_fitted = True
        
    def predict_p_stable(self, preds: List[float]) -> float:
        """
        Return calibrated probability P(true_value <= threshold | preds).
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        x = self._featurize(preds)
        if isinstance(self.platt_a, list):
            logit = float(np.dot(self.platt_a, x) + self.platt_b)
        else:
            logit = float(self.platt_a) * float(x[0]) + float(self.platt_b)
        p_stable = 1 / (1 + np.exp(-logit))
        
        return float(p_stable)
    
    def save(self, path: str):
        """Save calibrator state to JSON."""
        with open(path, "w") as f:
            json.dump({
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
                "threshold": self.threshold,
                "feature_mode": self.feature_mode,
                "sigma_floor": self.SIGMA_FLOOR,
                "z_clip": self.Z_CLIP,
                "is_fitted": self.is_fitted
            }, f, indent=2)
            
    @classmethod
    def load(cls, path: str):
        """Load calibrator from JSON."""
        with open(path) as f:
            d = json.load(f)
        
        c = cls(threshold=d.get("threshold", 0.0), feature_mode=d.get("feature_mode", "z"))
        c.platt_a = d.get("platt_a")
        c.platt_b = d.get("platt_b")
        if "sigma_floor" in d:
            c.SIGMA_FLOOR = float(d["sigma_floor"])
        if "z_clip" in d:
            c.Z_CLIP = float(d["z_clip"])
        c.is_fitted = bool(
            d.get("is_fitted", c.platt_a is not None and c.platt_b is not None)
        )
        # Allow loading older versions where constants might differ, 
        # but warn/override if critical? For now trust saved params if we were to save them,
        # but here we just use class constants for guards unless we want to make them instance vars.
        return c
