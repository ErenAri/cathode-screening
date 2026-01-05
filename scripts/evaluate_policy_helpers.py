#!/usr/bin/env python3
"""
Helper functions for policy evaluation scripts.

These are extracted here so they can be imported by tests.
"""
from typing import Dict, List, Optional

import numpy as np

# Import stability threshold
THRESH_STABLE = 0.05


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
    
    Args:
        y_true: Ground truth E_hull
        scores: Ranking scores (lower = better, e.g., q90_cal)
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
    q90_cal: np.ndarray,
    budgets: List[int],
    thresh_stable: float = THRESH_STABLE,
) -> Dict[int, float]:
    """
    Compute recall when limited to B followups.
    
    Strategy: take all KEEPs, then top MAYBEs by q90_cal until budget reached.
    
    Args:
        y_true: Ground truth E_hull
        decisions: "KEEP"/"MAYBE"/"KILL" decisions
        q90_cal: Scores for ranking MAYBEs (lower = better)
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
    
    # Sort MAYBEs by q90_cal ascending (best first)
    maybe_sorted = maybe_idx[np.argsort(q90_cal[maybe_idx])]
    
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
