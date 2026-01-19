"""
Benchmark Suite for CHGNet Li-Cathode Model.

Computes comprehensive metrics for model evaluation:
- Energy Accuracy: MAE, RMSE, R²
- Screening Metrics: EF@1%, Recall@100, False-kill rate
- Stability Prediction: AUC, Precision@k, NDCG
- Uncertainty Quality: ECE, OOD Detection AUC

Usage:
    python scripts/39_benchmark_model.py --checkpoint-dir checkpoints/gcp_l4
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_score, ndcg_score
)
import pandas as pd

from pymatgen.core import Structure


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    # Energy Accuracy
    mae: float
    rmse: float
    r2: float
    
    # Screening Metrics
    ef_1: float    # Enrichment Factor @ 1%
    ef_5: float    # Enrichment Factor @ 5%
    ef_10: float   # Enrichment Factor @ 10%
    recall_100: float
    recall_500: float
    false_kill_rate: float
    false_positive_rate: float
    
    # Stability Prediction
    stability_auc: float
    precision_at_10: float
    precision_at_50: float
    precision_at_100: float
    ndcg: float
    
    # Uncertainty Quality
    calibration_ece: float
    ood_detection_auc: Optional[float]
    
    # Comparison
    baseline_mae: Optional[float]
    improvement_vs_baseline: Optional[float]


def enrichment_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float,
    stability_threshold: float = 0.0,
) -> float:
    """
    Compute Enrichment Factor at given top fraction.
    
    EF = (# stable in top X%) / (expected # stable if random)
    
    Args:
        y_true: True stability values (e.g., formation energy)
        y_pred: Predicted values
        top_fraction: Fraction to consider (e.g., 0.01 for 1%)
        stability_threshold: Threshold for "stable" (e.g., 0 eV/atom)
        
    Returns:
        Enrichment factor (>1 means better than random)
    """
    n_samples = len(y_true)
    n_top = max(1, int(n_samples * top_fraction))
    
    # Get indices of top predictions (most stable = lowest predicted energy)
    top_indices = np.argsort(y_pred)[:n_top]
    
    # Count true stables in our top predictions
    true_stables_in_top = np.sum(y_true[top_indices] <= stability_threshold)
    
    # Expected if random
    total_stables = np.sum(y_true <= stability_threshold)
    expected_stables = total_stables * top_fraction
    
    if expected_stables == 0:
        return 0.0
    
    return true_stables_in_top / expected_stables


def recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    stability_threshold: float = 0.0,
) -> float:
    """
    Compute Recall@k - fraction of top-k true stables found in predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        k: Number of top predictions to consider
        stability_threshold: Threshold for stability
        
    Returns:
        Recall (0-1)
    """
    # Get true top-k (ground truth best materials)
    true_top_k = set(np.argsort(y_true)[:k])
    
    # Get predicted top-k
    pred_top_k = set(np.argsort(y_pred)[:k])
    
    # Overlap
    overlap = len(true_top_k & pred_top_k)
    return overlap / k


def false_kill_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float = 0.1,
    stability_threshold: float = 0.0,
) -> float:
    """
    Compute False-Kill Rate - fraction of truly stable materials we filtered out.
    
    This is critical for screening: we don't want to miss good candidates.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        top_fraction: Fraction we keep (filter rest)
        stability_threshold: Threshold for stability
        
    Returns:
        False-kill rate (lower is better)
    """
    n_keep = max(1, int(len(y_true) * top_fraction))
    
    # What we would keep based on predictions
    keep_indices = set(np.argsort(y_pred)[:n_keep])
    
    # Truly stable materials
    stable_indices = set(np.where(y_true <= stability_threshold)[0])
    
    if len(stable_indices) == 0:
        return 0.0
    
    # Stable materials we killed (didn't keep)
    killed_stables = stable_indices - keep_indices
    
    return len(killed_stables) / len(stable_indices)


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float = 0.1,
    stability_threshold: float = 0.0,
) -> float:
    """
    Compute False-Positive Rate - fraction of our top picks that are unstable.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        top_fraction: Fraction we keep
        stability_threshold: Threshold for stability
        
    Returns:
        False-positive rate (lower is better)
    """
    n_keep = max(1, int(len(y_true) * top_fraction))
    
    # What we would keep
    keep_indices = np.argsort(y_pred)[:n_keep]
    
    # Count unstable in our picks
    unstable_in_picks = np.sum(y_true[keep_indices] > stability_threshold)
    
    return unstable_in_picks / n_keep


def precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    stability_threshold: float = 0.0,
) -> float:
    """
    Precision@k - fraction of top-k predictions that are truly stable.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        k: Number of predictions
        stability_threshold: Stability threshold
        
    Returns:
        Precision (0-1)
    """
    top_k_indices = np.argsort(y_pred)[:k]
    stable_in_top_k = np.sum(y_true[top_k_indices] <= stability_threshold)
    return stable_in_top_k / k


def compute_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 100,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    Measures ranking quality - how well we rank the best materials at the top.
    """
    # Convert to relevance scores (lower energy = higher relevance)
    relevance = -y_true  # Negate so lower energy = higher score
    
    # Normalize to 0-1
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)
    
    # NDCG
    return ndcg_score([relevance], [y_pred], k=k)


def expected_calibration_error(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.
    
    Measures how well predicted uncertainty matches actual error.
    """
    errors = np.abs(predictions - targets)
    
    # Bin by uncertainty
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    
    ece = 0.0
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        
        expected_error = uncertainties[mask].mean()
        actual_error = errors[mask].mean()
        
        bin_weight = mask.sum() / len(uncertainties)
        ece += bin_weight * abs(expected_error - actual_error)
    
    return ece


def run_benchmark(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    ood_scores: Optional[np.ndarray] = None,
    ood_labels: Optional[np.ndarray] = None,
    baseline_predictions: Optional[np.ndarray] = None,
    stability_threshold: float = 0.0,
) -> BenchmarkResults:
    """
    Run complete benchmark suite.
    
    Args:
        predictions: Model predictions (energy per atom)
        targets: Ground truth values
        uncertainties: Predicted uncertainties (optional)
        ood_scores: OOD detection scores (optional)
        ood_labels: True OOD labels (optional)
        baseline_predictions: Baseline model predictions (optional)
        stability_threshold: Threshold for stability classification
        
    Returns:
        BenchmarkResults with all metrics
    """
    n_samples = len(predictions)
    
    # Energy Accuracy
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    # Screening Metrics
    ef_1 = enrichment_factor(targets, predictions, 0.01, stability_threshold)
    ef_5 = enrichment_factor(targets, predictions, 0.05, stability_threshold)
    ef_10 = enrichment_factor(targets, predictions, 0.10, stability_threshold)
    
    recall_100_val = recall_at_k(targets, predictions, min(100, n_samples))
    recall_500_val = recall_at_k(targets, predictions, min(500, n_samples))
    
    fkr = false_kill_rate(targets, predictions, 0.10, stability_threshold)
    fpr = false_positive_rate(targets, predictions, 0.10, stability_threshold)
    
    # Stability Prediction
    # Convert to binary classification for AUC
    true_stable = (targets <= stability_threshold).astype(int)
    pred_stable_score = -predictions  # Lower prediction = more stable
    
    if len(np.unique(true_stable)) > 1:
        stability_auc = roc_auc_score(true_stable, pred_stable_score)
    else:
        stability_auc = 0.5
    
    p_at_10 = precision_at_k(targets, predictions, min(10, n_samples), stability_threshold)
    p_at_50 = precision_at_k(targets, predictions, min(50, n_samples), stability_threshold)
    p_at_100 = precision_at_k(targets, predictions, min(100, n_samples), stability_threshold)
    
    ndcg = compute_ndcg(targets, predictions, min(100, n_samples))
    
    # Uncertainty Quality
    if uncertainties is not None:
        calibration_ece = expected_calibration_error(predictions, uncertainties, targets)
    else:
        calibration_ece = 0.0
    
    # OOD Detection
    if ood_scores is not None and ood_labels is not None:
        if len(np.unique(ood_labels)) > 1:
            ood_detection_auc = roc_auc_score(ood_labels, ood_scores)
        else:
            ood_detection_auc = 0.5
    else:
        ood_detection_auc = None
    
    # Baseline comparison
    if baseline_predictions is not None:
        baseline_mae = mean_absolute_error(targets, baseline_predictions)
        improvement = (baseline_mae - mae) / baseline_mae * 100
    else:
        baseline_mae = None
        improvement = None
    
    return BenchmarkResults(
        mae=mae,
        rmse=rmse,
        r2=r2,
        ef_1=ef_1,
        ef_5=ef_5,
        ef_10=ef_10,
        recall_100=recall_100_val,
        recall_500=recall_500_val,
        false_kill_rate=fkr,
        false_positive_rate=fpr,
        stability_auc=stability_auc,
        precision_at_10=p_at_10,
        precision_at_50=p_at_50,
        precision_at_100=p_at_100,
        ndcg=ndcg,
        calibration_ece=calibration_ece,
        ood_detection_auc=ood_detection_auc,
        baseline_mae=baseline_mae,
        improvement_vs_baseline=improvement,
    )


def print_benchmark_report(results: BenchmarkResults):
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print("CHGNet Li-Cathode Model Benchmark Report")
    print("=" * 60)
    
    print("\n📊 ENERGY ACCURACY")
    print(f"  MAE:  {results.mae:.4f} eV/atom")
    print(f"  RMSE: {results.rmse:.4f} eV/atom")
    print(f"  R²:   {results.r2:.4f}")
    
    print("\n🎯 SCREENING METRICS")
    print(f"  EF@1%:  {results.ef_1:.2f}x (higher is better)")
    print(f"  EF@5%:  {results.ef_5:.2f}x")
    print(f"  EF@10%: {results.ef_10:.2f}x")
    print(f"  Recall@100: {results.recall_100:.1%}")
    print(f"  Recall@500: {results.recall_500:.1%}")
    print(f"  False-kill Rate: {results.false_kill_rate:.1%} (lower is better)")
    print(f"  False-positive Rate: {results.false_positive_rate:.1%}")
    
    print("\n⚡ STABILITY PREDICTION")
    print(f"  Stability AUC: {results.stability_auc:.4f}")
    print(f"  Precision@10:  {results.precision_at_10:.1%}")
    print(f"  Precision@50:  {results.precision_at_50:.1%}")
    print(f"  Precision@100: {results.precision_at_100:.1%}")
    print(f"  NDCG: {results.ndcg:.4f}")
    
    print("\n📈 UNCERTAINTY QUALITY")
    print(f"  Calibration ECE: {results.calibration_ece:.4f} (lower is better)")
    if results.ood_detection_auc is not None:
        print(f"  OOD Detection AUC: {results.ood_detection_auc:.4f}")
    
    if results.baseline_mae is not None:
        print("\n🔄 VS BASELINE")
        print(f"  Baseline MAE: {results.baseline_mae:.4f} eV/atom")
        print(f"  Improvement: {results.improvement_vs_baseline:+.1f}%")
    
    print("\n" + "=" * 60)


def save_benchmark_report(results: BenchmarkResults, path: str):
    """Save benchmark results to JSON file."""
    with open(path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"Saved benchmark report to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CHGNet Li-Cathode Model")
    parser.add_argument("--checkpoint-dir", default="checkpoints/gcp_l4")
    parser.add_argument("--test-data", default="data/training/li_cathode_structures.json")
    parser.add_argument("--output", default="checkpoints/benchmark_results.json")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    
    # This would be implemented fully after training completes
    # For now, just skeleton
    print(f"Benchmark script ready!")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Test data: {args.test_data}")
    print(f"  Output: {args.output}")
    print("\nRun after training completes to generate benchmark report.")


if __name__ == "__main__":
    main()
