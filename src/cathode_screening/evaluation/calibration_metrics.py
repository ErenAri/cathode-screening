"""
Calibration metrics for probability predictions.

Computes Expected Calibration Error (ECE) and Mean Absolute Calibration Error (MACE)
for stability probability predictions (p_stable).
"""

from typing import Tuple, Optional
import numpy as np


def compute_calibration_bins(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration statistics per bin.
    
    Args:
        predicted_probs: Predicted probabilities [N]
        true_labels: Binary ground truth labels [N]
        n_bins: Number of calibration bins
    
    Returns:
        bin_edges: [n_bins+1] bin edge values
        bin_accuracies: [n_bins] actual accuracy per bin
        bin_confidences: [n_bins] mean predicted prob per bin
        bin_counts: [n_bins] number of samples per bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        
        # Handle edge case for final bin (include upper bound)
        if i == n_bins - 1:
            mask = (predicted_probs >= lower) & (predicted_probs <= upper)
        else:
            mask = (predicted_probs >= lower) & (predicted_probs < upper)
        
        bin_counts[i] = np.sum(mask)
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(true_labels[mask])
            bin_confidences[i] = np.mean(predicted_probs[mask])
    
    return bin_edges, bin_accuracies, bin_confidences, bin_counts


def expected_calibration_error(
    p_stable: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.05,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for stability probability.
    
    ECE = Σ (n_b / N) * |accuracy_b - confidence_b|
    
    This is the weighted average of absolute calibration error per bin,
    where bins are weighted by sample count.
    
    Args:
        p_stable: Predicted probability of stability [N]
        y_true: True E_hull values [N]
        threshold: Stability threshold (materials with E_hull <= threshold are "stable")
        n_bins: Number of calibration bins
    
    Returns:
        Expected Calibration Error (lower is better, 0 = perfectly calibrated)
    """
    # Convert E_hull to binary labels
    true_labels = (y_true <= threshold).astype(float)
    
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_bins(
        p_stable, true_labels, n_bins
    )
    
    n_total = len(p_stable)
    
    # Weighted average of absolute differences
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            weight = bin_counts[i] / n_total
            ece += weight * np.abs(bin_accuracies[i] - bin_confidences[i])
    
    return ece


def mean_absolute_calibration_error(
    p_stable: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.05,
    n_bins: int = 10,
) -> float:
    """
    Compute Mean Absolute Calibration Error (MACE).
    
    MACE = (1/B) * Σ |accuracy_b - confidence_b| for bins with samples
    
    Unlike ECE, MACE gives equal weight to each non-empty bin,
    making it more sensitive to miscalibration in low-density regions.
    
    Args:
        p_stable: Predicted probability of stability [N]
        y_true: True E_hull values [N]
        threshold: Stability threshold (materials with E_hull <= threshold are "stable")
        n_bins: Number of calibration bins
    
    Returns:
        Mean Absolute Calibration Error (lower is better, 0 = perfectly calibrated)
    """
    # Convert E_hull to binary labels
    true_labels = (y_true <= threshold).astype(float)
    
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_bins(
        p_stable, true_labels, n_bins
    )
    
    # Simple average over non-empty bins
    non_empty_mask = bin_counts > 0
    if not np.any(non_empty_mask):
        return np.nan
    
    mace = np.mean(np.abs(bin_accuracies[non_empty_mask] - bin_confidences[non_empty_mask]))
    
    return mace


def maximum_calibration_error(
    p_stable: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.05,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max |accuracy_b - confidence_b|
    
    This captures the worst-case miscalibration across all bins.
    
    Args:
        p_stable: Predicted probability of stability [N]
        y_true: True E_hull values [N]
        threshold: Stability threshold (materials with E_hull <= threshold are "stable")
        n_bins: Number of calibration bins
    
    Returns:
        Maximum Calibration Error (lower is better)
    """
    # Convert E_hull to binary labels
    true_labels = (y_true <= threshold).astype(float)
    
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_bins(
        p_stable, true_labels, n_bins
    )
    
    # Max over non-empty bins
    non_empty_mask = bin_counts > 0
    if not np.any(non_empty_mask):
        return np.nan
    
    mce = np.max(np.abs(bin_accuracies[non_empty_mask] - bin_confidences[non_empty_mask]))
    
    return mce


def compute_calibration_metrics(
    p_stable: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.05,
    n_bins: int = 10,
) -> dict:
    """
    Compute all calibration metrics.
    
    Args:
        p_stable: Predicted probability of stability [N]
        y_true: True E_hull values [N]  
        threshold: Stability threshold
        n_bins: Number of calibration bins
    
    Returns:
        Dictionary with ECE, MACE, MCE values
    """
    return {
        "ece": expected_calibration_error(p_stable, y_true, threshold, n_bins),
        "mace": mean_absolute_calibration_error(p_stable, y_true, threshold, n_bins),
        "mce": maximum_calibration_error(p_stable, y_true, threshold, n_bins),
    }


def format_calibration_report(
    metrics: dict,
    threshold: float = 0.05,
) -> str:
    """
    Format calibration metrics as readable string.
    
    Args:
        metrics: Output from compute_calibration_metrics()
        threshold: Stability threshold used
    
    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        f"CALIBRATION METRICS (stability threshold: {threshold:.2f} eV)",
        "=" * 60,
        f"Expected Calibration Error (ECE):     {metrics['ece']:.4f}",
        f"Mean Absolute Calibration Error:      {metrics['mace']:.4f}",
        f"Maximum Calibration Error (MCE):      {metrics['mce']:.4f}",
        "=" * 60,
        "",
        "Interpretation:",
        "  - ECE < 0.05: Well-calibrated",
        "  - ECE 0.05-0.10: Acceptable",
        "  - ECE > 0.10: Needs recalibration",
    ]
    return "\n".join(lines)
