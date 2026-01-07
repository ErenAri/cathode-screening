"""
Acquisition functions for active learning.

These functions rank unlabeled candidates by their informativeness
to guide efficient exploration of the materials space.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm


def uncertainty_sampling(
    epistemic_std: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Select top-k candidates with highest epistemic uncertainty.
    
    Pure exploration strategy - samples where model is most uncertain.
    
    Args:
        epistemic_std: [N] array of epistemic uncertainties (ensemble std)
        k: Number of candidates to select
    
    Returns:
        [k] indices of selected candidates
    """
    # Higher uncertainty = more informative
    scores = epistemic_std
    
    # Get top-k indices
    top_k_idx = np.argsort(scores)[-k:][::-1]
    
    return top_k_idx


def expected_improvement(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    threshold: float = 0.05,
    k: int = 10,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.
    
    Balances exploitation (low predicted E_hull) with exploration (high uncertainty).
    Specifically designed for minimization (finding low E_hull materials).
    
    Args:
        mean_pred: [N] predicted E_hull values
        std_pred: [N] prediction uncertainties (aleatoric + epistemic)
        threshold: Stability threshold (τ = 0.05 eV/atom)
        k: Number of candidates to select
        xi: Exploration-exploitation trade-off (higher = more exploration)
    
    Returns:
        [k] indices of selected candidates
    """
    # For minimization: EI = E[max(τ - y, 0)]
    # Using closed-form solution for Gaussian predictions
    
    # Normalize to get improvement over threshold
    z = (threshold - mean_pred - xi) / (std_pred + 1e-8)
    
    # Expected improvement
    # EI = (τ - μ - ξ) * Φ(z) + σ * φ(z)
    ei = (threshold - mean_pred - xi) * norm.cdf(z) + std_pred * norm.pdf(z)
    
    # Handle edge cases
    ei = np.where(std_pred > 0, ei, 0)
    
    # Get top-k indices
    top_k_idx = np.argsort(ei)[-k:][::-1]
    
    return top_k_idx


def upper_confidence_bound(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    kappa: float = 2.0,
    k: int = 10,
) -> np.ndarray:
    """
    Lower Confidence Bound (LCB) for minimization problems.
    
    Selects candidates with lowest (mean - kappa * std), i.e.,
    candidates that could be very stable given uncertainty.
    
    Args:
        mean_pred: [N] predicted E_hull values
        std_pred: [N] prediction uncertainties
        kappa: Exploration parameter (higher = more exploration)
        k: Number of candidates to select
    
    Returns:
        [k] indices of selected candidates
    """
    # For minimization, use LCB = μ - κσ
    # Lower is better (more likely to be stable)
    lcb = mean_pred - kappa * std_pred
    
    # Get top-k indices (lowest LCB scores)
    top_k_idx = np.argsort(lcb)[:k]
    
    return top_k_idx


def greedy_batch_selection(
    epistemic_std: np.ndarray,
    k: int,
    diversity_weight: float = 0.5,
    features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Greedy batch selection with diversity.
    
    Selects informative candidates while maintaining diversity
    in the batch to avoid redundant queries.
    
    Args:
        epistemic_std: [N] epistemic uncertainties
        k: Batch size
        diversity_weight: Weight for diversity term (0-1)
        features: [N, D] feature matrix for diversity calculation
    
    Returns:
        [k] indices of selected candidates
    """
    n = len(epistemic_std)
    
    if features is None or diversity_weight == 0:
        # Fall back to simple uncertainty sampling
        return uncertainty_sampling(epistemic_std, k)
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Greedy selection
    selected = []
    available = set(range(n))
    
    for _ in range(k):
        if not available:
            break
        
        scores = np.full(n, -np.inf)
        
        for idx in available:
            # Informativeness score
            info_score = epistemic_std[idx]
            
            # Diversity score (min distance to already selected)
            if selected:
                selected_features = features_norm[selected]
                similarities = features_norm[idx] @ selected_features.T
                div_score = 1 - np.max(similarities)  # Lower similarity = more diverse
            else:
                div_score = 1.0
            
            # Combined score
            scores[idx] = (1 - diversity_weight) * info_score + diversity_weight * div_score
        
        # Select best
        best_idx = np.argmax(scores)
        selected.append(best_idx)
        available.remove(best_idx)
    
    return np.array(selected)


def probability_of_improvement(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    threshold: float = 0.05,
    k: int = 10,
) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition function.
    
    Selects candidates most likely to be below the stability threshold.
    
    Args:
        mean_pred: [N] predicted E_hull values
        std_pred: [N] prediction uncertainties
        threshold: Stability threshold
        k: Number of candidates to select
    
    Returns:
        [k] indices of selected candidates
    """
    # P(y < threshold) = Φ((threshold - μ) / σ)
    z = (threshold - mean_pred) / (std_pred + 1e-8)
    pi = norm.cdf(z)
    
    # Get top-k indices
    top_k_idx = np.argsort(pi)[-k:][::-1]
    
    return top_k_idx


def random_sampling(
    n_candidates: int,
    k: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Random baseline for comparison.
    
    Args:
        n_candidates: Total number of candidates
        k: Number to select
        seed: Random seed
    
    Returns:
        [k] random indices
    """
    rng = np.random.default_rng(seed)
    return rng.choice(n_candidates, size=min(k, n_candidates), replace=False)
