"""
Calibration Layer for CHGNet Predictions.

Ensures that predicted uncertainties match actual error rates.
Uses temperature scaling for post-hoc calibration.

Example:
    If model says "90% confident", it should be correct 90% of the time.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
from scipy.optimize import minimize_scalar


@dataclass
class CalibrationResult:
    """Result of calibrated prediction."""
    prediction: float
    uncertainty: float  # Calibrated uncertainty (std)
    confidence: float   # 0-1, probability of being within uncertainty
    confidence_interval: Tuple[float, float]  # (lower, upper)


class CalibratedPredictor:
    """
    Provides calibrated uncertainty estimates for ensemble predictions.
    
    Uses temperature scaling learned on a held-out validation set.
    
    Usage:
        calibrator = CalibratedPredictor()
        calibrator.fit(val_predictions, val_targets)
        result = calibrator.predict(ensemble_predictions)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        confidence_level: float = 0.90,
    ):
        """
        Initialize calibrator.
        
        Args:
            temperature: Scaling factor for uncertainties
            confidence_level: Target confidence level (default 90%)
        """
        self.temperature = temperature
        self.confidence_level = confidence_level
        self.is_fitted = False
        self.ece = None  # Expected Calibration Error
    
    def fit(
        self,
        ensemble_predictions: List[List[float]],
        targets: List[float],
    ):
        """
        Learn optimal temperature from validation data.
        
        Args:
            ensemble_predictions: List of [pred1, pred2, ...] for each sample
            targets: Ground truth values
        """
        means = np.array([np.mean(p) for p in ensemble_predictions])
        stds = np.array([np.std(p) for p in ensemble_predictions])
        targets = np.array(targets)
        
        # Optimize temperature to minimize calibration error
        def calibration_loss(temp):
            calibrated_stds = stds * temp
            
            # For each sample, check if target is within confidence interval
            z_score = 1.645  # 90% confidence
            lower = means - z_score * calibrated_stds
            upper = means + z_score * calibrated_stds
            
            coverage = np.mean((targets >= lower) & (targets <= upper))
            return abs(coverage - self.confidence_level)
        
        # Find optimal temperature
        result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        # Compute ECE
        self.ece = self._compute_ece(ensemble_predictions, targets)
        self.is_fitted = True
        
        print(f"Calibration complete: temperature={self.temperature:.3f}, ECE={self.ece:.4f}")
    
    def _compute_ece(
        self,
        ensemble_predictions: List[List[float]],
        targets: List[float],
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures how well predicted confidence matches actual accuracy.
        Lower is better (0 = perfectly calibrated).
        """
        means = np.array([np.mean(p) for p in ensemble_predictions])
        stds = np.array([np.std(p) for p in ensemble_predictions]) * self.temperature
        targets = np.array(targets)
        
        errors = np.abs(means - targets)
        
        # Bin by predicted uncertainty
        bin_edges = np.percentile(stds, np.linspace(0, 100, n_bins + 1))
        
        ece = 0.0
        for i in range(n_bins):
            mask = (stds >= bin_edges[i]) & (stds < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_errors = errors[mask]
            bin_stds = stds[mask]
            
            # Expected error should be proportional to predicted std
            expected_error = bin_stds.mean()
            actual_error = bin_errors.mean()
            
            bin_weight = mask.sum() / len(stds)
            ece += bin_weight * abs(expected_error - actual_error)
        
        return ece
    
    def predict(
        self,
        ensemble_predictions: List[float],
    ) -> CalibrationResult:
        """
        Get calibrated prediction with uncertainty.
        
        Args:
            ensemble_predictions: List of predictions from ensemble models
            
        Returns:
            CalibrationResult with calibrated uncertainty
        """
        mean = np.mean(ensemble_predictions)
        std = np.std(ensemble_predictions)
        
        # Apply temperature scaling
        calibrated_std = std * self.temperature
        
        # Compute confidence interval
        z_score = 1.645  # 90% confidence
        lower = mean - z_score * calibrated_std
        upper = mean + z_score * calibrated_std
        
        # Confidence based on std (inverse relationship)
        # Low std = high confidence
        confidence = 1.0 / (1.0 + calibrated_std * 10)  # Sigmoid-like
        
        return CalibrationResult(
            prediction=float(mean),
            uncertainty=float(calibrated_std),
            confidence=float(confidence),
            confidence_interval=(float(lower), float(upper)),
        )
    
    def reliability_diagram_data(
        self,
        ensemble_predictions: List[List[float]],
        targets: List[float],
        n_bins: int = 10,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get data for reliability diagram.
        
        Returns:
            (predicted_confidence, actual_accuracy, bin_counts)
        """
        means = np.array([np.mean(p) for p in ensemble_predictions])
        stds = np.array([np.std(p) for p in ensemble_predictions]) * self.temperature
        targets = np.array(targets)
        
        # Convert std to confidence
        confidences = 1.0 / (1.0 + stds * 10)
        errors = np.abs(means - targets)
        
        # Bin by confidence
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        predicted_conf = []
        actual_acc = []
        counts = []
        
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_conf = confidences[mask].mean()
            # Accuracy = fraction with error below threshold
            threshold = 0.05  # 50 meV/atom
            bin_acc = (errors[mask] < threshold).mean()
            
            predicted_conf.append(bin_conf)
            actual_acc.append(bin_acc)
            counts.append(mask.sum())
        
        return predicted_conf, actual_acc, counts
    
    def save(self, path: str):
        """Save calibrator to file."""
        save_dict = {
            "temperature": self.temperature,
            "confidence_level": self.confidence_level,
            "ece": self.ece,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "CalibratedPredictor":
        """Load calibrator from file."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        calibrator = cls(
            temperature=save_dict["temperature"],
            confidence_level=save_dict["confidence_level"],
        )
        calibrator.ece = save_dict["ece"]
        calibrator.is_fitted = save_dict["is_fitted"]
        return calibrator


def fit_calibration_from_validation(
    ensemble_predictions: List[List[float]],
    targets: List[float],
    save_path: str = "checkpoints/calibrator.pkl",
) -> CalibratedPredictor:
    """
    Fit calibrator on validation data and save.
    
    Args:
        ensemble_predictions: List of ensemble predictions per sample
        targets: Ground truth values
        save_path: Where to save calibrator
        
    Returns:
        Fitted CalibratedPredictor
    """
    calibrator = CalibratedPredictor()
    calibrator.fit(ensemble_predictions, targets)
    calibrator.save(save_path)
    print(f"Saved calibrator to {save_path}")
    return calibrator
