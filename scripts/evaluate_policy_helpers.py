#!/usr/bin/env python3
"""
Helper functions for policy evaluation scripts.

These are extracted here so they can be imported by tests.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

# Default stability threshold (for backward compatibility)
THRESH_STABLE = 0.05

# Multiple thresholds for comprehensive evaluation
STABILITY_THRESHOLDS = {
    "ehull_001": 0.01,  # Ultra-stable (on convex hull or very close)
    "ehull_002": 0.02,  # Highly stable  
    "ehull_005": 0.05,  # Stable (standard threshold)
    "ehull_010": 0.10,  # Metastable (may be synthesizable)
}


@dataclass
class ThresholdMetrics:
    """Metrics for a single stability threshold."""
    threshold: float
    threshold_name: str
    base_rate: float           # fraction of materials below threshold
    n_positive: int            # count below threshold
    
    # Enrichment factors
    ef_at_1pct: float
    ef_at_5pct: float
    
    # Recall at budget
    recall_at_50: float
    recall_at_100: float
    recall_at_500: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class MultiThresholdReport:
    """Evaluation report across multiple stability thresholds."""
    n_total: int
    metrics_by_threshold: Dict[str, ThresholdMetrics]
    
    def to_dict(self) -> Dict:
        return {
            "n_total": self.n_total,
            "metrics_by_threshold": {
                k: v.to_dict() for k, v in self.metrics_by_threshold.items()
            }
        }
    
    def print_table(self) -> None:
        """Print a clean summary table."""
        print("\n" + "="*80)
        print("SCREENING METRICS BY STABILITY THRESHOLD")
        print("="*80)
        print(f"{'Threshold':<12} {'Base Rate':>10} {'N_pos':>7} "
              f"{'EF@1%':>8} {'EF@5%':>8} "
              f"{'R@50':>8} {'R@100':>8} {'R@500':>8}")
        print("-"*80)
        
        for name in ["ehull_001", "ehull_002", "ehull_005", "ehull_010"]:
            if name not in self.metrics_by_threshold:
                continue
            m = self.metrics_by_threshold[name]
            print(f"E<={m.threshold:.2f} eV   {m.base_rate:>9.1%} {m.n_positive:>7d} "
                  f"{m.ef_at_1pct:>8.2f} {m.ef_at_5pct:>8.2f} "
                  f"{m.recall_at_50:>7.1%} {m.recall_at_100:>7.1%} {m.recall_at_500:>7.1%}")
        
        print("-"*80)
        print("EF = Enrichment Factor (higher = better ranking)")
        print("R@B = Recall at budget B (fraction of positives found in top B)")
        print("="*80)


def get_ehull_bin(y_true: float) -> str:
    """Bin E_hull into interpretable categories."""
    if y_true < 0.02:
        return "highly_stable"
    elif y_true < 0.05:
        return "stable"
    elif y_true < 0.10:
        return "metastable"
    elif y_true < 0.20:
        return "marginal"
    else:
        return "unstable"


def derive_gate_level_from_score(
    ood_score: np.ndarray,
    thresh_borderline: float = 0.3,
    thresh_ood: float = 0.7,
) -> np.ndarray:
    """Convert OOD score to gate level."""
    levels = np.where(
        ood_score >= thresh_ood, "OOD",
        np.where(ood_score >= thresh_borderline, "BORDERLINE", "IN")
    )
    return levels


def compute_enrichment_factor(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    thresh_stable: float = THRESH_STABLE,
) -> float:
    """
    Compute enrichment factor at given fraction.
    
    EF = (fraction stable in top K) / (fraction stable overall)
    where K = fraction * N
    
    IMPORTANT: Use ranking_score = q50 + λ*epistemic_std for ranking.
    - q50 captures predicted E_hull (lower = more stable)
    - epistemic_std penalizes uncertain predictions
    - DO NOT use q90_cal (measures interval width, not stability)
    
    Args:
        y_true: Ground truth E_hull
        scores: Ranking scores (lower = better). Use policy.compute_ranking_scores().
        fraction: Fraction of samples to consider (e.g., 0.01 for 1%)
        thresh_stable: Threshold for stability
    
    Returns:
        Enrichment factor
    """
    n = len(y_true)
    k = max(1, int(np.ceil(fraction * n)))
    
    # Sort by score ascending (best first)
    sorted_idx = np.argsort(scores)
    top_k_idx = sorted_idx[:k]
    
    is_stable = y_true < thresh_stable
    
    # Fraction stable in top K
    frac_top_k = is_stable[top_k_idx].mean()
    
    # Fraction stable overall
    frac_overall = is_stable.mean()
    
    if frac_overall == 0:
        return 0.0
    
    return frac_top_k / frac_overall


def compute_recall_at_budget(
    y_true: np.ndarray,
    decisions: np.ndarray,
    ranking_scores: np.ndarray,
    budgets: List[int],
    thresh_stable: float = THRESH_STABLE,
) -> Dict[int, float]:
    """
    Compute recall when limited to B followups.
    
    Strategy: take all KEEPs, then top MAYBEs by ranking_score until budget reached.
    
    IMPORTANT: Use ranking_score = q50 + λ*epistemic_std for ranking.
    - This prioritizes confident low E_hull predictions
    - DO NOT use q90_cal (measures interval width, not stability)
    
    Args:
        y_true: Ground truth E_hull
        decisions: "KEEP"/"MAYBE"/"KILL" decisions
        ranking_scores: Scores for ranking MAYBEs (lower = better).
                        Use policy.compute_ranking_scores(q50, epistemic_std).
        budgets: List of budget values
        thresh_stable: Threshold for stability
    
    Returns:
        Dict mapping budget -> recall
    """
    is_stable = y_true < thresh_stable
    n_stable = is_stable.sum()
    
    if n_stable == 0:
        return {b: 0.0 for b in budgets}
    
    # Get indices by decision
    keep_idx = np.where(decisions == "KEEP")[0]
    maybe_idx = np.where(decisions == "MAYBE")[0]
    
    # Sort MAYBEs by ranking score ascending (best first)
    maybe_sorted = maybe_idx[np.argsort(ranking_scores[maybe_idx])]
    
    results = {}
    for budget in budgets:
        if budget <= 0:
            results[budget] = 0.0
            continue
        
        # Take all KEEPs up to budget
        selected_keeps = keep_idx[:budget]
        remaining = budget - len(selected_keeps)
        
        # Fill remaining with top MAYBEs
        selected_maybes = maybe_sorted[:remaining] if remaining > 0 else np.array([], dtype=int)
        
        selected = np.concatenate([selected_keeps, selected_maybes])
        
        # Compute recall
        n_found = is_stable[selected].sum() if len(selected) > 0 else 0
        recall = n_found / n_stable
        results[budget] = float(recall)
    
    return results


def compute_multi_threshold_metrics(
    y_true: np.ndarray,
    ranking_scores: np.ndarray,
    decisions: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> MultiThresholdReport:
    """
    Compute EF and Recall metrics across multiple stability thresholds.
    
    This addresses the issue where EF@5% ≈ 1.0 when base rate is high (~47%).
    By evaluating at multiple thresholds, we can see screening value at
    different stringency levels.
    
    Args:
        y_true: Ground truth E_hull values
        ranking_scores: Ranking scores (lower = better). 
                        Use policy.compute_ranking_scores(q50, epistemic_std).
        decisions: Optional decision array ("KEEP"/"MAYBE"/"KILL").
                   If None, all samples are treated as MAYBE for recall.
        thresholds: Dict mapping threshold name -> E_hull value.
                    Defaults to STABILITY_THRESHOLDS.
    
    Returns:
        MultiThresholdReport with metrics for each threshold
    """
    if thresholds is None:
        thresholds = STABILITY_THRESHOLDS
    
    if decisions is None:
        # Treat all as MAYBE (pure ranking evaluation)
        decisions = np.array(["MAYBE"] * len(y_true))
    
    n_total = len(y_true)
    metrics = {}
    
    # Sort indices by ranking score once (lower = better)
    sorted_idx = np.argsort(ranking_scores)
    
    for name, thresh in thresholds.items():
        is_positive = y_true < thresh
        n_positive = is_positive.sum()
        base_rate = n_positive / n_total if n_total > 0 else 0.0
        
        # Skip if no positives (EF undefined)
        if n_positive == 0:
            metrics[name] = ThresholdMetrics(
                threshold=thresh,
                threshold_name=name,
                base_rate=0.0,
                n_positive=0,
                ef_at_1pct=0.0,
                ef_at_5pct=0.0,
                recall_at_50=0.0,
                recall_at_100=0.0,
                recall_at_500=0.0,
            )
            continue
        
        # Compute EF at 1% and 5%
        # EF = (frac positive in top K) / (frac positive overall)
        k_1pct = max(1, int(np.ceil(0.01 * n_total)))
        k_5pct = max(1, int(np.ceil(0.05 * n_total)))
        
        top_1pct_idx = sorted_idx[:k_1pct]
        top_5pct_idx = sorted_idx[:k_5pct]
        
        frac_1pct = is_positive[top_1pct_idx].mean()
        frac_5pct = is_positive[top_5pct_idx].mean()
        
        ef_at_1pct = frac_1pct / base_rate if base_rate > 0 else 0.0
        ef_at_5pct = frac_5pct / base_rate if base_rate > 0 else 0.0
        
        # Compute Recall at budget (using decision-aware ranking)
        recall_dict = compute_recall_at_budget(
            y_true, decisions, ranking_scores, 
            budgets=[50, 100, 500],
            thresh_stable=thresh
        )
        
        metrics[name] = ThresholdMetrics(
            threshold=thresh,
            threshold_name=name,
            base_rate=base_rate,
            n_positive=int(n_positive),
            ef_at_1pct=float(ef_at_1pct),
            ef_at_5pct=float(ef_at_5pct),
            recall_at_50=recall_dict.get(50, 0.0),
            recall_at_100=recall_dict.get(100, 0.0),
            recall_at_500=recall_dict.get(500, 0.0),
        )
    
    return MultiThresholdReport(n_total=n_total, metrics_by_threshold=metrics)
