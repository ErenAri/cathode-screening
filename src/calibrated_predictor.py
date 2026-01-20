"""
Platt-calibrated predictor for stability probability.

Fits a logistic regression on z-scores derived from ensemble mean and std.
"""
from __future__ import annotations

from typing import List

import numpy as np


class CalibratedPredictor:
    SIGMA_FLOOR = 1e-3
    Z_CLIP = 10.0

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
        self.platt_a = None
        self.platt_b = None
        self.is_fitted = False

    def fit(self, preds: List[List[float]], labels: List[bool]) -> None:
        z_scores = []
        for p in preds:
            mu = float(np.mean(p))
            sigma = max(float(np.std(p)), self.SIGMA_FLOOR)
            z = np.clip((self.threshold - mu) / sigma, -self.Z_CLIP, self.Z_CLIP)
            z_scores.append(z)

        from sklearn.linear_model import LogisticRegression

        X = np.array(z_scores, dtype=np.float32).reshape(-1, 1)
        y = np.array(labels, dtype=np.int64)

        lr = LogisticRegression(class_weight="balanced", solver="lbfgs")
        lr.fit(X, y)

        self.platt_a = float(lr.coef_[0][0])
        self.platt_b = float(lr.intercept_[0])
        self.is_fitted = True

    def predict_p_stable(self, preds: List[float]) -> float:
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        mu = float(np.mean(preds))
        sigma = max(float(np.std(preds)), self.SIGMA_FLOOR)
        z = np.clip((self.threshold - mu) / sigma, -self.Z_CLIP, self.Z_CLIP)
        return float(1 / (1 + np.exp(-(self.platt_a * z + self.platt_b))))

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "platt_a": self.platt_a,
                    "platt_b": self.platt_b,
                    "threshold": self.threshold,
                    "sigma_floor": self.SIGMA_FLOOR,
                    "z_clip": self.Z_CLIP,
                    "is_fitted": self.is_fitted,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CalibratedPredictor":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(threshold=data["threshold"])
        inst.platt_a = data["platt_a"]
        inst.platt_b = data["platt_b"]
        inst.is_fitted = True
        return inst
