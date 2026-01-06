"""
Tests for data split leakage detection.

These tests verify that train/val/test splits maintain proper separation
and don't allow trivial memorization across similar structures.
"""

import numpy as np
import pytest
from typing import Dict, List


class TestSplitLeakage:
    """Tests for detecting and preventing data leakage in splits."""
    
    def test_no_material_id_overlap(
        self,
        splits: Dict[str, List[str]],
    ):
        """
        MUST PASS: No material_id appears in multiple splits.
        
        This is the basic sanity check for any split.
        """
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        
        assert len(train_set & val_set) == 0, "Train-val overlap detected"
        assert len(train_set & test_set) == 0, "Train-test overlap detected"
        assert len(val_set & test_set) == 0, "Val-test overlap detected"
    
    def test_composition_novelty(
        self,
        train_compositions: List[str],
        test_compositions: List[str],
        min_novel_fraction: float = 0.10,
    ):
        """
        SHOULD PASS: Test set contains some novel compositions.
        
        A good split should have at least some test compositions
        not seen in training.
        """
        train_set = set(train_compositions)
        novel_count = sum(1 for c in test_compositions if c not in train_set)
        novel_fraction = novel_count / len(test_compositions)
        
        assert novel_fraction >= min_novel_fraction, (
            f"Test set has only {novel_fraction:.1%} novel compositions. "
            f"Expected at least {min_novel_fraction:.1%}. "
            "This suggests leakage or trivial splits."
        )
    
    def test_cluster_purity(
        self,
        splits: Dict[str, List[str]],
        cluster_assignments: Dict[str, int],
    ):
        """
        SHOULD PASS: Clusters are not split across train/test.
        
        If using cluster-based splitting (composition or SOAP-LOCO),
        entire clusters should be assigned to a single split.
        """
        train_clusters = set(cluster_assignments[mid] for mid in splits["train"] if mid in cluster_assignments)
        test_clusters = set(cluster_assignments[mid] for mid in splits["test"] if mid in cluster_assignments)
        
        overlap = train_clusters & test_clusters
        
        # Allow small leakage for edge cases (< 5% of test clusters)
        max_overlap_fraction = 0.05
        overlap_fraction = len(overlap) / len(test_clusters) if test_clusters else 0
        
        assert overlap_fraction <= max_overlap_fraction, (
            f"Found {len(overlap)} clusters split across train/test "
            f"({overlap_fraction:.1%} of test clusters). "
            "This violates leave-one-cluster-out principle."
        )
    
    def test_soap_novelty_vs_composition(
        self,
        soap_novelty_metrics: Dict[str, float],
        composition_novelty_metrics: Dict[str, float],
    ):
        """
        OPTIONAL: SOAP-LOCO splits should have higher novelty than composition splits.
        
        This validates that SOAP-LOCO is working as intended—it should
        create test sets that are more structurally novel.
        """
        soap_mean_dist = soap_novelty_metrics["mean_min_distance"]
        comp_mean_dist = composition_novelty_metrics["mean_min_distance"]
        
        # SOAP should have equal or higher novelty
        assert soap_mean_dist >= comp_mean_dist * 0.9, (
            f"SOAP-LOCO novelty ({soap_mean_dist:.3f}) is significantly lower "
            f"than composition novelty ({comp_mean_dist:.3f}). "
            "This suggests SOAP-LOCO is not improving split quality."
        )


class TestSplitStatistics:
    """Tests for split size and balance statistics."""
    
    def test_split_sizes_reasonable(
        self,
        splits: Dict[str, List[str]],
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        tolerance: float = 0.10,
    ):
        """
        SHOULD PASS: Split sizes are within tolerance of target fractions.
        
        Due to cluster-based splitting, exact fractions aren't possible,
        but they should be reasonably close.
        """
        total = sum(len(v) for v in splits.values())
        
        actual_train_frac = len(splits["train"]) / total
        actual_val_frac = len(splits["val"]) / total
        
        assert abs(actual_train_frac - train_frac) <= tolerance, (
            f"Train fraction {actual_train_frac:.2f} differs from target {train_frac:.2f} "
            f"by more than tolerance {tolerance:.2f}"
        )
        
        assert abs(actual_val_frac - val_frac) <= tolerance, (
            f"Val fraction {actual_val_frac:.2f} differs from target {val_frac:.2f} "
            f"by more than tolerance {tolerance:.2f}"
        )
    
    def test_class_balance_preserved(
        self,
        splits: Dict[str, List[str]],
        labels: Dict[str, float],
        threshold: float = 0.05,
        max_imbalance: float = 0.20,
    ):
        """
        SHOULD PASS: Stable/unstable ratio is similar across splits.
        
        Cluster-based splits can skew class balance, but it shouldn't
        be too extreme.
        """
        def get_stable_fraction(material_ids: List[str]) -> float:
            values = [labels[mid] for mid in material_ids if mid in labels]
            return sum(1 for v in values if v < threshold) / len(values) if values else 0
        
        train_stable = get_stable_fraction(splits["train"])
        test_stable = get_stable_fraction(splits["test"])
        
        imbalance = abs(train_stable - test_stable)
        
        assert imbalance <= max_imbalance, (
            f"Class imbalance between train ({train_stable:.1%} stable) "
            f"and test ({test_stable:.1%} stable) exceeds {max_imbalance:.1%}. "
            "Consider stratified cluster assignment."
        )


# Helper functions for running tests programmatically

def validate_splits_basic(
    splits: Dict[str, List[str]],
) -> Dict[str, bool]:
    """
    Run basic split validation checks.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' lists
    
    Returns:
        Dictionary with test names and pass/fail status
    """
    results = {}
    
    # Check no overlap
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    results["no_train_val_overlap"] = len(train_set & val_set) == 0
    results["no_train_test_overlap"] = len(train_set & test_set) == 0
    results["no_val_test_overlap"] = len(val_set & test_set) == 0
    
    # Check non-empty
    results["train_not_empty"] = len(splits["train"]) > 0
    results["val_not_empty"] = len(splits["val"]) > 0
    results["test_not_empty"] = len(splits["test"]) > 0
    
    return results


def compute_split_novelty_score(
    train_fingerprints: np.ndarray,
    test_fingerprints: np.ndarray,
) -> float:
    """
    Compute novelty score for test set relative to train set.
    
    Higher = more novel = better for evaluating generalization.
    
    Args:
        train_fingerprints: [N_train, D] descriptors
        test_fingerprints: [N_test, D] descriptors
    
    Returns:
        Mean minimum distance from test to train
    """
    from sklearn.preprocessing import normalize
    
    train_normed = normalize(train_fingerprints)
    test_normed = normalize(test_fingerprints)
    
    # Cosine similarity
    similarity = test_normed @ train_normed.T
    max_sim = np.max(similarity, axis=1)
    
    # Distance = 1 - similarity
    min_dist = 1 - max_sim
    
    return float(np.mean(min_dist))
