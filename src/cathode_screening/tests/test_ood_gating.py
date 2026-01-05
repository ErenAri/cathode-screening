"""
Tests for OOD (Out-of-Distribution) gating module.

Tests:
1. kNN distances reproduce expected values on synthetic embeddings
2. OOD gating suppresses KEEP when gate_level=OOD
3. Composition distance calculation
4. Gate level assignment based on thresholds
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# Import OOD module functions
from cathode_screening.inference.ood import (
    OODGate,
    OODGateResult,
    OODThresholds,
    composition_fingerprint,
    compute_train_stats,
    save_ood_artifacts,
    select_ood_thresholds,
    COMPOSITION_VOCAB,
)


class TestCompositionFingerprint:
    """Tests for composition fingerprint calculation."""
    
    def test_simple_formula(self):
        """LiCoO2 should have Li, Co, O components."""
        fp = composition_fingerprint("LiCoO2")
        
        # Should be length of vocab
        assert len(fp) == len(COMPOSITION_VOCAB)
        
        # Check Li (index 0)
        assert fp[0] > 0  # Li present
        
        # Check O (index 1)
        assert fp[1] > 0  # O present
        
        # Check Co (index 4)
        assert fp[4] > 0  # Co present
        
        # Should sum to 1 (normalized)
        assert np.isclose(fp.sum(), 1.0, atol=0.01)
    
    def test_unknown_element(self):
        """Elements not in vocab should not appear."""
        fp = composition_fingerprint("NaCl")
        
        # Na and Cl not in vocab, so should be all zeros
        assert np.allclose(fp, 0)


class TestKNNDistances:
    """Tests for kNN distance calculations on synthetic data."""
    
    def test_exact_match_distance_zero(self):
        """Point in training set should have small kNN distance."""
        # Create synthetic training embeddings
        np.random.seed(42)
        n_train = 100
        dim = 32
        train_embeddings = np.random.randn(n_train, dim).astype(np.float32)
        
        # Build index
        try:
            import faiss
            index = faiss.IndexFlatL2(dim)
            index.add(train_embeddings)
            
            # Query with a training point
            query = train_embeddings[0:1]
            distances, indices = index.search(query, k=2)
            
            # First neighbor is self with distance 0
            assert distances[0, 0] < 1e-6
            # Second neighbor has non-zero distance
            assert distances[0, 1] > 0
            
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
            nn.fit(train_embeddings)
            
            distances, _ = nn.kneighbors(train_embeddings[0:1])
            assert distances[0, 0] < 1e-6
    
    def test_outlier_has_large_distance(self):
        """Point far from training distribution should have large kNN distance."""
        np.random.seed(42)
        n_train = 100
        dim = 32
        
        # Training points near origin
        train_embeddings = np.random.randn(n_train, dim).astype(np.float32)
        
        # Outlier far from origin
        outlier = (np.ones((1, dim)) * 100).astype(np.float32)
        
        try:
            import faiss
            index = faiss.IndexFlatL2(dim)
            index.add(train_embeddings)
            
            distances, _ = index.search(outlier, k=10)
            mean_dist = distances[0].mean()
            
            # Should be much larger than typical training distance
            train_dists, _ = index.search(train_embeddings[:10], k=10)
            train_mean = train_dists[:, 1:].mean()  # Skip self
            
            assert mean_dist > 10 * train_mean
            
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
            nn.fit(train_embeddings)
            
            distances, _ = nn.kneighbors(outlier)
            mean_dist = distances[0].mean()
            
            train_dists, _ = nn.kneighbors(train_embeddings[:10])
            train_mean = train_dists[:, 1:].mean()
            
            assert mean_dist > 10 * train_mean
    
    def test_reproducible_distances(self):
        """Same query should give same distances."""
        np.random.seed(42)
        train_embeddings = np.random.randn(50, 16).astype(np.float32)
        query = np.random.randn(1, 16).astype(np.float32)
        
        try:
            import faiss
            index = faiss.IndexFlatL2(16)
            index.add(train_embeddings)
            
            d1, _ = index.search(query, k=5)
            d2, _ = index.search(query, k=5)
            
            np.testing.assert_array_equal(d1, d2)
            
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(train_embeddings)
            
            d1, _ = nn.kneighbors(query)
            d2, _ = nn.kneighbors(query)
            
            np.testing.assert_array_equal(d1, d2)


class TestOODGating:
    """Tests for OOD gating policy."""
    
    @pytest.fixture
    def mock_ood_gate(self):
        """Create a mock OOD gate for testing."""
        comp_mu = np.zeros(len(COMPOSITION_VOCAB), dtype=np.float32)
        comp_cov_inv = np.eye(len(COMPOSITION_VOCAB), dtype=np.float32)
        
        comp_stats = {"mean": 1.0, "std": 0.5, "tau_p95": 2.0, "tau_p99": 3.0}
        emb_stats = {"mean": 1.0, "std": 0.5, "tau_p95": 2.0, "tau_p99": 3.0}
        disagree_stats = {"mean": 0.01, "std": 0.005, "tau_p95": 0.02, "tau_p99": 0.03}
        
        return OODGate(
            comp_mu=comp_mu,
            comp_cov_inv=comp_cov_inv,
            comp_stats=comp_stats,
            emb_index=None,
            emb_stats=emb_stats,
            disagree_stats=disagree_stats,
            thresholds=OODThresholds(t_hi=0.70, t_mid=0.50),
        )
    
    def test_ood_forces_keep_to_maybe(self, mock_ood_gate):
        """When gate_level=OOD, KEEP should become MAYBE."""
        # Create OOD result with high score
        result = OODGateResult(
            ood_score=0.85,  # Above t_hi=0.70
            flag=True,
            d_comp=5.0,
            d_emb=0.0,
            d_disagree=0.0,
            z_comp=8.0,
            z_emb=0.0,
            z_disagree=0.0,
            z_max=8.0,
            flag_comp=True,
            flag_emb=False,
            flag_disagree=False,
        )
        
        # Apply gating to KEEP decision
        gated_decision, action = mock_ood_gate.apply_gating(
            raw_decision="KEEP",
            ood_result=result,
            q90_cal=0.03,  # Would normally be KEEP
            p_stable=0.9,
        )
        
        assert gated_decision == "MAYBE"
        assert action == "defer"
    
    def test_ood_does_not_change_kill(self, mock_ood_gate):
        """When gate_level=OOD, KILL should remain KILL."""
        result = OODGateResult(
            ood_score=0.85,
            flag=True,
            d_comp=5.0, d_emb=0.0, d_disagree=0.0,
            z_comp=8.0, z_emb=0.0, z_disagree=0.0, z_max=8.0,
            flag_comp=True, flag_emb=False, flag_disagree=False,
        )
        
        gated_decision, action = mock_ood_gate.apply_gating(
            raw_decision="KILL",
            ood_result=result,
            q90_cal=0.15,
            p_stable=0.2,
        )
        
        assert gated_decision == "KILL"
        assert action == "none"
    
    def test_borderline_applies_strict_criteria(self, mock_ood_gate):
        """When gate_level=BORDERLINE, stricter KEEP criteria apply."""
        # Score in borderline zone
        result = OODGateResult(
            ood_score=0.60,  # Between t_mid=0.50 and t_hi=0.70
            flag=False,
            d_comp=2.0, d_emb=0.0, d_disagree=0.0,
            z_comp=2.0, z_emb=0.0, z_disagree=0.0, z_max=2.0,
            flag_comp=False, flag_emb=False, flag_disagree=False,
        )
        
        # KEEP that doesn't meet strict criteria
        gated_decision, action = mock_ood_gate.apply_gating(
            raw_decision="KEEP",
            ood_result=result,
            q90_cal=0.04,  # Above strict threshold of 0.03
            p_stable=0.80,  # Below strict threshold of 0.85
        )
        
        assert gated_decision == "MAYBE"
        assert action == "strict"
        
        # KEEP that meets strict criteria
        gated_decision2, action2 = mock_ood_gate.apply_gating(
            raw_decision="KEEP",
            ood_result=result,
            q90_cal=0.02,  # Below strict threshold
            p_stable=0.90,  # Above strict threshold
        )
        
        assert gated_decision2 == "KEEP"
        assert action2 == "strict"
    
    def test_in_distribution_trusts_decision(self, mock_ood_gate):
        """When gate_level=IN, raw decision is trusted."""
        result = OODGateResult(
            ood_score=0.30,  # Below t_mid=0.50
            flag=False,
            d_comp=0.5, d_emb=0.0, d_disagree=0.0,
            z_comp=-1.0, z_emb=0.0, z_disagree=0.0, z_max=-1.0,
            flag_comp=False, flag_emb=False, flag_disagree=False,
        )
        
        gated_decision, action = mock_ood_gate.apply_gating(
            raw_decision="KEEP",
            ood_result=result,
            q90_cal=0.04,
            p_stable=0.80,
        )
        
        assert gated_decision == "KEEP"
        assert action == "none"


class TestOODThresholdSelection:
    """Tests for OOD threshold calibration."""
    
    def test_selects_threshold_for_target_fkr(self):
        """Threshold selection should achieve target false KEEP rate."""
        np.random.seed(42)
        
        # Create synthetic data with correlation between OOD scores and stability
        # Low OOD score samples tend to be truly stable (correct KEEPs)
        # High OOD score samples tend to be false KEEPs (unstable)
        
        # 30 true stable KEEPs with LOW OOD scores (in-distribution)
        n_true_keeps = 30
        ood_true_keeps = np.random.uniform(0.1, 0.4, n_true_keeps)
        y_true_keeps = np.random.uniform(0, 0.05, n_true_keeps)  # Stable
        
        # 20 false KEEPs with HIGH OOD scores (out-of-distribution)
        n_false_keeps = 20
        ood_false_keeps = np.random.uniform(0.6, 0.95, n_false_keeps)
        y_false_keeps = np.random.uniform(0.15, 0.3, n_false_keeps)  # Unstable
        
        # 100 MAYBEs with mixed OOD scores
        n_maybes = 100
        ood_maybes = np.random.uniform(0, 1, n_maybes)
        y_maybes = np.random.uniform(0, 0.3, n_maybes)
        
        # 50 KILLs with mixed OOD scores
        n_kills = 50
        ood_kills = np.random.uniform(0, 1, n_kills)
        y_kills = np.random.uniform(0.12, 0.5, n_kills)
        
        ood_scores = np.concatenate([ood_true_keeps, ood_false_keeps, ood_maybes, ood_kills])
        decisions = np.array(
            ["KEEP"] * (n_true_keeps + n_false_keeps) + 
            ["MAYBE"] * n_maybes + 
            ["KILL"] * n_kills
        )
        y_true = np.concatenate([y_true_keeps, y_false_keeps, y_maybes, y_kills])
        
        thresholds = select_ood_thresholds(
            val_ood_scores=ood_scores,
            val_decisions=decisions,
            val_y_true=y_true,
            target_fkr=0.10,
            thresh_unstable=0.10,
        )
        
        assert thresholds.t_hi > 0
        assert thresholds.t_mid < thresholds.t_hi
        assert thresholds.achieved_fkr <= 0.10 + 0.05  # Allow some slack


class TestSaveLoadOODGate:
    """Tests for saving and loading OOD gate artifacts."""
    
    def test_round_trip(self):
        """Save and load OOD artifacts should preserve values."""
        np.random.seed(42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create synthetic stats
            formulas = ["LiCoO2", "LiMnO2", "LiFePO4"] * 10
            embeddings = np.random.randn(30, 32).astype(np.float32)
            disagreements = np.random.uniform(0, 0.02, 30)
            
            stats, comp_mu, comp_cov_inv = compute_train_stats(
                train_formulas=formulas,
                train_embeddings=embeddings,
                train_disagreements=disagreements,
            )
            
            save_ood_artifacts(
                output_dir=output_dir,
                stats=stats,
                comp_mu=comp_mu,
                comp_cov_inv=comp_cov_inv,
                train_embeddings=embeddings,
            )
            
            # Load gate
            gate = OODGate.from_artifacts(output_dir)
            
            # Check composition stats preserved
            np.testing.assert_array_almost_equal(gate.comp_mu, comp_mu)
            np.testing.assert_array_almost_equal(gate.comp_cov_inv, comp_cov_inv)
            
            # Check can compute score
            score_result = gate.score(
                formula="LiCoO2",
                embedding=embeddings[0],
                q50_ensemble=[0.05, 0.06, 0.04],
            )
            
            assert 0 <= score_result.ood_score <= 1


class TestOODGateIntegration:
    """Integration tests for OOD gating in prediction pipeline."""
    
    def test_gating_reduces_false_keeps(self):
        """OOD gating should reduce false KEEPs on OOD samples."""
        # Simulate scenario where OOD samples have higher error
        n_in = 50
        n_ood = 20
        
        # In-distribution: low error, accurate predictions
        in_dist_q90 = np.random.uniform(0.02, 0.06, n_in)
        in_dist_ytrue = np.random.uniform(0.01, 0.05, n_in)
        in_dist_ood_score = np.random.uniform(0, 0.4, n_in)
        
        # OOD: high error, predictions are unreliable
        ood_q90 = np.random.uniform(0.02, 0.06, n_ood)  # Look similar
        ood_ytrue = np.random.uniform(0.10, 0.25, n_ood)  # But actually unstable!
        ood_ood_score = np.random.uniform(0.6, 0.9, n_ood)
        
        # All samples would be KEEP based on q90 alone
        all_q90 = np.concatenate([in_dist_q90, ood_q90])
        all_ytrue = np.concatenate([in_dist_ytrue, ood_ytrue])
        all_ood_score = np.concatenate([in_dist_ood_score, ood_ood_score])
        
        # Without gating: count false KEEPs
        false_keeps_no_gating = np.sum(all_ytrue > 0.10)
        
        # With gating: filter out high OOD score
        t_hi = 0.5
        kept_mask = all_ood_score < t_hi
        false_keeps_with_gating = np.sum(all_ytrue[kept_mask] > 0.10)
        
        # Gating should reduce false KEEPs
        assert false_keeps_with_gating < false_keeps_no_gating
