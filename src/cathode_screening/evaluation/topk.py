"""
Enrichment Factor and Recall metrics for cathode screening evaluation.

Computes EF@k% and Recall@k for multiple stability thresholds and ranking scores.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Stability thresholds (eV above hull)
STABILITY_THRESHOLDS = [0.01, 0.02, 0.05, 0.10]

# Top-k percentiles for EF
EF_PERCENTILES = [1, 5]

# Top-k counts for Recall
RECALL_K_VALUES = [50, 100, 500]

# Target counts for DAF (Discovery Acceleration Factor)
DAF_TARGET_COUNTS = [10, 25, 50]


@dataclass
class RankingScore:
    """Definition of a ranking score."""
    
    name: str
    description: str
    
    def compute(
        self,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        epistemic_std: Optional[np.ndarray] = None,
        lambda_rank: float = 0.5,
    ) -> np.ndarray:
        """Compute ranking score (lower is better for stability)."""
        raise NotImplementedError


class Q50Score(RankingScore):
    """Rank by point estimate (q50)."""
    
    def __init__(self):
        super().__init__(name="q50", description="Point estimate (median)")
    
    def compute(
        self,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        epistemic_std: Optional[np.ndarray] = None,
        lambda_rank: float = 0.5,
    ) -> np.ndarray:
        return q50


class Q90CalScore(RankingScore):
    """Rank by calibrated upper bound (q90_cal) - conservative."""
    
    def __init__(self):
        super().__init__(name="q90_cal", description="Calibrated upper bound (90th percentile)")
    
    def compute(
        self,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        epistemic_std: Optional[np.ndarray] = None,
        lambda_rank: float = 0.5,
    ) -> np.ndarray:
        return q90_cal


class UncertaintyPenalizedScore(RankingScore):
    """Rank by q50 + lambda * epistemic_std - penalizes uncertainty."""
    
    def __init__(self, lambda_rank: float = 0.5):
        super().__init__(
            name=f"q50+{lambda_rank}σ",
            description=f"q50 + {lambda_rank} * epistemic_std"
        )
        self.lambda_rank = lambda_rank
    
    def compute(
        self,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        epistemic_std: Optional[np.ndarray] = None,
        lambda_rank: float = 0.5,
    ) -> np.ndarray:
        if epistemic_std is None:
            # Fall back to q50 if no epistemic uncertainty
            return q50
        return q50 + self.lambda_rank * epistemic_std


def get_default_scores(lambda_rank: float = 0.5) -> List[RankingScore]:
    """Get default ranking scores for evaluation."""
    return [
        Q50Score(),
        Q90CalScore(),
        UncertaintyPenalizedScore(lambda_rank=lambda_rank),
    ]


def enrichment_factor(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_percent: float,
) -> float:
    """
    Compute Enrichment Factor at top k%.
    
    EF@k% = (hits in top k%) / (expected hits in random k%)
          = (hits_topk / n_topk) / (total_hits / n_total)
    
    Args:
        y_true: True E_hull values
        scores: Ranking scores (lower is better)
        threshold: Stability threshold (e.g., 0.05 eV)
        top_percent: Top percentage to consider (e.g., 1 for top 1%)
    
    Returns:
        Enrichment factor (>1 means better than random)
    """
    n_total = len(y_true)
    n_topk = max(1, int(n_total * top_percent / 100))
    
    # Count total hits (stable materials)
    is_stable = y_true <= threshold
    total_hits = np.sum(is_stable)
    
    if total_hits == 0:
        return np.nan  # No positives to find
    
    # Get top-k by score (lower is better)
    top_indices = np.argsort(scores)[:n_topk]
    hits_topk = np.sum(is_stable[top_indices])
    
    # Enrichment factor
    expected_hits = total_hits * (n_topk / n_total)
    ef = hits_topk / expected_hits if expected_hits > 0 else np.nan
    
    return ef


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    k: int,
) -> float:
    """
    Compute Recall at top k samples.
    
    Recall@k = (hits in top k) / (total hits)
    
    Args:
        y_true: True E_hull values
        scores: Ranking scores (lower is better)
        threshold: Stability threshold (e.g., 0.05 eV)
        k: Number of top samples to consider
    
    Returns:
        Recall (fraction of stable materials found in top k)
    """
    # Count total hits (stable materials)
    is_stable = y_true <= threshold
    total_hits = np.sum(is_stable)
    
    if total_hits == 0:
        return np.nan  # No positives to find
    
    # Clamp k to dataset size
    k = min(k, len(y_true))
    
    # Get top-k by score (lower is better)
    top_indices = np.argsort(scores)[:k]
    hits_topk = np.sum(is_stable[top_indices])
    
    return hits_topk / total_hits


def discovery_acceleration_factor(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    target_count: int,
) -> float:
    """
    Compute Discovery Acceleration Factor.
    
    DAF = n_random / n_model
    
    Where:
        n_model = number of samples to screen (using model ranking) to find `target_count` stable materials
        n_random = expected number of samples to screen randomly to find `target_count` stable materials
    
    DAF > 1 means model is faster than random.
    DAF = 2 means model finds the same materials in half the screening effort.
    
    Args:
        y_true: True E_hull values
        scores: Ranking scores (lower is better)
        threshold: Stability threshold (e.g., 0.05 eV)
        target_count: Number of stable materials to find
    
    Returns:
        Discovery Acceleration Factor (higher is better)
    """
    n_total = len(y_true)
    is_stable = y_true <= threshold
    total_hits = np.sum(is_stable)
    
    if total_hits == 0:
        return np.nan  # No positives to find
    
    if target_count > total_hits:
        return np.nan  # Can't find more than exist
    
    # Sort by model score (lower is better = more stable prediction)
    sorted_indices = np.argsort(scores)
    sorted_stable = is_stable[sorted_indices]
    
    # Find how many samples model needs to screen to get target_count hits
    cumulative_hits = np.cumsum(sorted_stable)
    
    # Find first index where we reach target_count hits
    hit_indices = np.where(cumulative_hits >= target_count)[0]
    if len(hit_indices) == 0:
        return np.nan
    
    n_model = hit_indices[0] + 1  # +1 because index is 0-based
    
    # Expected samples for random screening (geometric distribution expectation)
    # E[samples to find k hits] = k * n_total / total_hits
    n_random = target_count * n_total / total_hits
    
    return n_random / n_model


def compute_ranking_metrics(
    y_true: np.ndarray,
    q50: np.ndarray,
    q90_cal: np.ndarray,
    epistemic_std: Optional[np.ndarray] = None,
    lambda_rank: float = 0.5,
    thresholds: Optional[List[float]] = None,
    ef_percentiles: Optional[List[float]] = None,
    recall_k_values: Optional[List[int]] = None,
    daf_target_counts: Optional[List[int]] = None,
    scores: Optional[List[RankingScore]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute EF and Recall metrics for multiple thresholds and ranking scores.
    
    Args:
        y_true: True E_hull values
        q50: Point estimates (median predictions)
        q90_cal: Calibrated upper bounds
        epistemic_std: Ensemble disagreement (optional)
        lambda_rank: Weight for uncertainty penalty
        thresholds: Stability thresholds (default: [0.01, 0.02, 0.05, 0.10])
        ef_percentiles: Top-k% for EF (default: [1, 5])
        recall_k_values: Top-k for Recall (default: [50, 100, 500])
        daf_target_counts: Target counts for DAF (default: [10, 25, 50])
        scores: Ranking score definitions (default: q50, q90_cal, q50+λσ)
    
    Returns:
        Nested dict: {score_name: {metric_name: value}}
    """
    thresholds = thresholds or STABILITY_THRESHOLDS
    ef_percentiles = ef_percentiles or EF_PERCENTILES
    recall_k_values = recall_k_values or RECALL_K_VALUES
    daf_target_counts = daf_target_counts or DAF_TARGET_COUNTS
    scores = scores or get_default_scores(lambda_rank)
    
    results = {}
    
    for score_def in scores:
        score_values = score_def.compute(
            q50=q50,
            q90_cal=q90_cal,
            epistemic_std=epistemic_std,
            lambda_rank=lambda_rank,
        )
        
        score_results = {}
        
        for thresh in thresholds:
            thresh_label = f"≤{thresh:.2f}"
            
            # EF metrics
            for pct in ef_percentiles:
                ef = enrichment_factor(y_true, score_values, thresh, pct)
                score_results[f"EF@{pct}%_{thresh_label}"] = ef
            
            # Recall metrics
            for k in recall_k_values:
                rec = recall_at_k(y_true, score_values, thresh, k)
                score_results[f"Recall@{k}_{thresh_label}"] = rec
            
            # DAF metrics
            for target in daf_target_counts:
                daf = discovery_acceleration_factor(y_true, score_values, thresh, target)
                score_results[f"DAF@{target}_{thresh_label}"] = daf
        
        results[score_def.name] = score_results
    
    return results


def format_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    thresholds: Optional[List[float]] = None,
    ef_percentiles: Optional[List[float]] = None,
    recall_k_values: Optional[List[int]] = None,
    daf_target_counts: Optional[List[int]] = None,
) -> str:
    """
    Format metrics as a readable table.
    
    Args:
        metrics: Output from compute_ranking_metrics()
        thresholds: Stability thresholds
        ef_percentiles: Top-k% for EF
        recall_k_values: Top-k for Recall
        daf_target_counts: Target counts for DAF
    
    Returns:
        Formatted table string
    """
    thresholds = thresholds or STABILITY_THRESHOLDS
    ef_percentiles = ef_percentiles or EF_PERCENTILES
    recall_k_values = recall_k_values or RECALL_K_VALUES
    daf_target_counts = daf_target_counts or DAF_TARGET_COUNTS
    
    lines = []
    score_names = list(metrics.keys())
    
    # Header
    lines.append("=" * 80)
    lines.append("ENRICHMENT FACTOR AND RECALL METRICS (Post Group-Conformal Calibration)")
    lines.append("=" * 80)
    lines.append("")
    
    # EF Table
    lines.append("ENRICHMENT FACTOR (higher is better, 1.0 = random)")
    lines.append("-" * 80)
    
    # Build EF header
    ef_header = f"{'Threshold':<12}"
    for score_name in score_names:
        for pct in ef_percentiles:
            ef_header += f" {score_name}@{pct}%".rjust(14)
    lines.append(ef_header)
    lines.append("-" * 80)
    
    for thresh in thresholds:
        thresh_label = f"≤{thresh:.2f}"
        row = f"{thresh_label:<12}"
        for score_name in score_names:
            for pct in ef_percentiles:
                key = f"EF@{pct}%_{thresh_label}"
                val = metrics[score_name].get(key, np.nan)
                row += f"{val:>14.2f}" if not np.isnan(val) else f"{'N/A':>14}"
        lines.append(row)
    
    lines.append("")
    
    # Recall Table
    lines.append("RECALL (fraction of stable materials found in top-k)")
    lines.append("-" * 80)
    
    # Build Recall header
    rec_header = f"{'Threshold':<12}"
    for score_name in score_names:
        for k in recall_k_values:
            rec_header += f" {score_name}@{k}".rjust(14)
    lines.append(rec_header)
    lines.append("-" * 80)
    
    for thresh in thresholds:
        thresh_label = f"≤{thresh:.2f}"
        row = f"{thresh_label:<12}"
        for score_name in score_names:
            for k in recall_k_values:
                key = f"Recall@{k}_{thresh_label}"
                val = metrics[score_name].get(key, np.nan)
                row += f"{val:>14.1%}" if not np.isnan(val) else f"{'N/A':>14}"
        lines.append(row)
    
    lines.append("")
    
    # DAF Table
    lines.append("DISCOVERY ACCELERATION FACTOR (higher is better, 1.0 = random)")
    lines.append("-" * 80)
    
    # Build DAF header
    daf_header = f"{'Threshold':<12}"
    for score_name in score_names:
        for target in daf_target_counts:
            daf_header += f" {score_name}@{target}".rjust(14)
    lines.append(daf_header)
    lines.append("-" * 80)
    
    for thresh in thresholds:
        thresh_label = f"≤{thresh:.2f}"
        row = f"{thresh_label:<12}"
        for score_name in score_names:
            for target in daf_target_counts:
                key = f"DAF@{target}_{thresh_label}"
                val = metrics[score_name].get(key, np.nan)
                row += f"{val:>14.2f}" if not np.isnan(val) else f"{'N/A':>14}"
        lines.append(row)
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def metrics_to_dataframe(
    metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convert metrics dict to pandas DataFrame for further analysis.
    
    Args:
        metrics: Output from compute_ranking_metrics()
    
    Returns:
        DataFrame with multi-index (score, metric)
    """
    rows = []
    for score_name, score_metrics in metrics.items():
        for metric_name, value in score_metrics.items():
            # Parse metric name
            parts = metric_name.split("_")
            metric_type = parts[0]  # e.g., "EF@1%" or "Recall@50"
            threshold = parts[1]    # e.g., "≤0.05"
            
            rows.append({
                "score": score_name,
                "metric": metric_type,
                "threshold": threshold,
                "value": value,
            })
    
    df = pd.DataFrame(rows)
    return df.pivot_table(
        index=["threshold", "metric"],
        columns="score",
        values="value",
    )


def evaluate_ranking_pipeline(
    y_true: np.ndarray,
    q50: np.ndarray,
    q90_cal: np.ndarray,
    epistemic_std: Optional[np.ndarray] = None,
    lambda_rank: float = 0.5,
    print_table: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Full evaluation pipeline: compute metrics and format table.
    
    Args:
        y_true: True E_hull values
        q50: Point estimates
        q90_cal: Calibrated upper bounds
        epistemic_std: Ensemble disagreement
        lambda_rank: Weight for uncertainty penalty
        print_table: Whether to print the table
    
    Returns:
        Tuple of (metrics_dict, formatted_table_string)
    """
    metrics = compute_ranking_metrics(
        y_true=y_true,
        q50=q50,
        q90_cal=q90_cal,
        epistemic_std=epistemic_std,
        lambda_rank=lambda_rank,
    )
    
    table = format_metrics_table(metrics)
    
    if print_table:
        print(table)
    
    return metrics, table
