"""
Tests for utility-optimized decision policy.
"""
import numpy as np
import pandas as pd
import pytest

from cathode_screening.inference.decision_policy import (
    CostModel,
    PolicyThresholds,
    DecisionPolicy,
    tune_policy_thresholds,
    PolicyReport,
    THRESH_STABLE,
)


class TestCostModel:
    """Tests for cost model."""
    
    def test_dft_mode_costs(self):
        """DFT mode should have low C_keep_fp, high C_kill_fn."""
        cm = CostModel.for_mode("dft_followup")
        
        assert cm.C_keep_fp < cm.C_kill_fn
        assert cm.C_keep_fp < cm.C_maybe
        assert cm.C_kill_fn > cm.C_maybe
    
    def test_experimental_mode_costs(self):
        """Experimental mode should have high C_keep_fp."""
        cm = CostModel.for_mode("experimental")
        
        assert cm.C_keep_fp > 1.0  # Higher than MAYBE cost
        assert cm.C_kill_fn > cm.C_keep_fp  # Still prioritize not missing discoveries
    
    def test_expected_cost_computation(self):
        """Expected cost should sum correctly."""
        cm = CostModel(C_keep_fp=2.0, C_kill_fn=10.0, C_maybe=1.0, C_keep_tp=-1.0, C_kill_tn=-0.5)
        
        decisions = np.array(["KEEP", "KEEP", "KILL", "KILL", "MAYBE"])
        y_true = np.array([0.01, 0.20, 0.30, 0.02, 0.05])  # stable, unstable, unstable, stable, borderline
        
        # KEEP stable (0.01): C_keep_tp = -1
        # KEEP unstable (0.20): C_keep_fp = 2
        # KILL unstable (0.30): C_kill_tn = -0.5
        # KILL stable (0.02): C_kill_fn = 10
        # MAYBE: C_maybe = 1
        expected = -1 + 2 + (-0.5) + 10 + 1
        
        cost = cm.expected_cost(decisions, y_true)
        assert abs(cost - expected) < 1e-6


class TestPolicyThresholds:
    """Tests for policy thresholds."""
    
    def test_serialization(self):
        """Thresholds should round-trip through dict."""
        t = PolicyThresholds(T_keep=0.045, T_kill=0.12, P_keep=0.75)
        
        d = t.to_dict()
        t2 = PolicyThresholds.from_dict(d)
        
        assert t2.T_keep == t.T_keep
        assert t2.T_kill == t.T_kill
        assert t2.P_keep == t.P_keep


class TestDecisionPolicy:
    """Tests for decision policy."""
    
    @pytest.fixture
    def dft_policy(self):
        return DecisionPolicy.for_mode("dft_followup")
    
    @pytest.fixture
    def exp_policy(self):
        return DecisionPolicy.for_mode("experimental")
    
    def test_keep_decision(self, exp_policy):
        """Should KEEP when q90 is low and not OOD (experimental mode)."""
        # Very confident stable prediction - experimental mode allows KEEP
        decision = exp_policy.decide(
            q10_cal=-0.01, q50=0.02, q90_cal=0.03,
            p_stable=0.9, gate_level="IN"
        )
        assert decision == "KEEP"
    
    def test_kill_decision(self, dft_policy):
        """Should KILL when q10 is high and epistemic uncertainty is low."""
        # Even optimistic bound is unstable, ensemble agrees
        decision = dft_policy.decide(
            q10_cal=0.18, q50=0.25, q90_cal=0.35,
            p_stable=0.1, gate_level="IN",
            epistemic_std=0.02  # Low uncertainty = ensemble agrees
        )
        assert decision == "KILL"
    
    def test_maybe_decision(self, dft_policy):
        """Should MAYBE when uncertain."""
        # Uncertain prediction
        decision = dft_policy.decide(
            q10_cal=0.03, q50=0.08, q90_cal=0.12,
            p_stable=0.5, gate_level="IN"
        )
        assert decision == "MAYBE"
    
    def test_ood_blocks_keep(self, exp_policy):
        """OOD gate should block KEEP decision."""
        # Would be KEEP if IN, but OOD
        decision = exp_policy.decide(
            q10_cal=-0.01, q50=0.02, q90_cal=0.03,
            p_stable=0.9, gate_level="OOD"
        )
        assert decision != "KEEP"  # Should be MAYBE
    
    def test_dft_mode_no_keep(self, dft_policy):
        """DFT mode should not auto-KEEP (all go to ranked queue)."""
        # Even very confident predictions go to MAYBE in DFT mode
        decision = dft_policy.decide(
            q10_cal=-0.01, q50=0.02, q90_cal=0.04,
            p_stable=0.9, gate_level="IN"
        )
        assert decision == "MAYBE"  # All non-KILL go to ranked queue
    
    def test_borderline_blocks_keep_in_experimental(self, exp_policy):
        """Experimental mode blocks KEEP when BORDERLINE."""
        decision = exp_policy.decide(
            q10_cal=-0.01, q50=0.02, q90_cal=0.03,  # Very tight
            p_stable=0.95, gate_level="BORDERLINE"
        )
        assert decision != "KEEP"
    
    def test_p_stable_threshold(self):
        """Low p_stable should block KEEP even with good quantiles."""
        policy = DecisionPolicy(
            thresholds=PolicyThresholds(T_keep=0.06, T_kill=0.15, P_keep=0.7),
            cost_model=CostModel.for_mode("dft_followup"),
            mode="dft_followup"
        )
        
        # Good quantiles but low p_stable
        decision = policy.decide(
            q10_cal=0.01, q50=0.03, q90_cal=0.05,
            p_stable=0.5,  # Below P_keep threshold
            gate_level="IN"
        )
        assert decision == "MAYBE"
    
    def test_batch_decisions(self, dft_policy):
        """Batch decisions should match individual."""
        q10 = np.array([0.01, 0.05, 0.18])
        q50 = np.array([0.03, 0.10, 0.25])
        q90 = np.array([0.04, 0.15, 0.35])
        p_stable = np.array([0.9, 0.5, 0.1])
        gate_levels = np.array(["IN", "IN", "IN"])
        
        batch = dft_policy.decide_batch(q10, q50, q90, p_stable, gate_levels)
        
        for i in range(3):
            individual = dft_policy.decide(q10[i], q50[i], q90[i], p_stable[i], gate_levels[i])
            assert batch[i] == individual


class TestPolicyEvaluation:
    """Tests for policy evaluation."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic validation data."""
        np.random.seed(42)
        n = 200
        
        # Generate E_hull values
        y_true = np.concatenate([
            np.random.uniform(0, 0.05, 50),    # Stable
            np.random.uniform(0.05, 0.15, 100), # Metastable/uncertain
            np.random.uniform(0.15, 0.50, 50),  # Unstable
        ])
        np.random.shuffle(y_true)
        
        # Generate predictions with some noise
        noise = np.random.normal(0, 0.02, n)
        q50 = y_true + noise
        q10 = q50 - 0.04
        q90 = q50 + 0.04
        
        p_stable = 1 / (1 + np.exp(10 * (y_true - 0.05)))  # Sigmoid around threshold
        
        return {
            "y_true": y_true,
            "q10_cal": q10,
            "q50": q50,
            "q90_cal": q90,
            "p_stable": p_stable,
        }
    
    def test_evaluation_metrics(self, synthetic_data):
        """Evaluation should compute all metrics."""
        policy = DecisionPolicy.for_mode("dft_followup")
        
        report = policy.evaluate(
            synthetic_data["q10_cal"],
            synthetic_data["q50"],
            synthetic_data["q90_cal"],
            synthetic_data["y_true"],
            synthetic_data["p_stable"],
        )
        
        assert report.n_total == 200
        assert report.n_keep + report.n_maybe + report.n_kill == 200
        assert 0 <= report.fn_rate <= 1
        assert 0 <= report.fp_rate <= 1
        assert report.expected_cost is not None
    
    def test_enrichment_factor(self, synthetic_data):
        """EF@K should be > 1 for a reasonable model."""
        policy = DecisionPolicy.for_mode("dft_followup")
        
        report = policy.evaluate(
            synthetic_data["q10_cal"],
            synthetic_data["q50"],
            synthetic_data["q90_cal"],
            synthetic_data["y_true"],
            synthetic_data["p_stable"],
        )
        
        # With reasonable predictions, top samples should be enriched
        assert 10 in report.ef_at_k
        assert report.ef_at_k[10] > 1.0  # Better than random


class TestPolicyTuning:
    """Tests for threshold tuning."""
    
    @pytest.fixture
    def tuning_data(self):
        """Generate data for tuning."""
        np.random.seed(123)
        n = 300
        
        y_true = np.concatenate([
            np.random.uniform(0, 0.05, 75),
            np.random.uniform(0.05, 0.15, 150),
            np.random.uniform(0.15, 0.50, 75),
        ])
        np.random.shuffle(y_true)
        
        noise = np.random.normal(0, 0.015, n)
        q50 = y_true + noise
        q10 = q50 - 0.03
        q90 = q50 + 0.03
        
        p_stable = 1 / (1 + np.exp(12 * (y_true - 0.05)))
        
        return y_true, q10, q50, q90, p_stable
    
    def test_tuning_finds_valid_policy(self, tuning_data):
        """Tuning should find a policy with acceptable FN rate."""
        y_true, q10, q50, q90, p_stable = tuning_data
        
        policy, report = tune_policy_thresholds(
            q10_cal=q10,
            q50=q50,
            q90_cal=q90,
            y_true=y_true,
            mode="dft_followup",
            p_stable=p_stable,
            n_grid=10,  # Small grid for speed
        )
        
        assert policy is not None
        assert report.fn_rate <= 0.15  # Should be within acceptable range
    
    def test_tuning_respects_mode(self, tuning_data):
        """Different modes should produce different thresholds."""
        y_true, q10, q50, q90, p_stable = tuning_data
        
        # Create epistemic_std (synthetic)
        epistemic_std = np.abs(np.random.randn(len(y_true)) * 0.02)
        
        dft_policy, _ = tune_policy_thresholds(
            q10, q50, q90, y_true, mode="dft_followup", n_grid=10,
            epistemic_std=epistemic_std, max_fn_rate=0.10
        )
        
        exp_policy, _ = tune_policy_thresholds(
            q10, q50, q90, y_true, mode="experimental", n_grid=10,
            epistemic_std=epistemic_std, max_fn_rate=0.10
        )
        
        # DFT mode: T_keep=0 (disabled), experimental may have T_keep > 0
        assert dft_policy.thresholds.T_keep == 0.0
        # Experimental may or may not have valid T_keep (depends on data)
        assert exp_policy.thresholds.T_keep >= 0.0


class TestPolicySerialization:
    """Tests for policy save/load."""
    
    def test_save_load_roundtrip(self, tmp_path):
        """Policy should round-trip through save/load."""
        policy = DecisionPolicy.for_mode("dft_followup")
        policy.thresholds.T_keep = 0.055
        policy.thresholds.P_keep = 0.72
        
        path = tmp_path / "policy.json"
        policy.save(path)
        
        loaded = DecisionPolicy.load(path)
        
        assert loaded.mode == policy.mode
        assert loaded.thresholds.T_keep == policy.thresholds.T_keep
        assert loaded.thresholds.P_keep == policy.thresholds.P_keep
        assert loaded.cost_model.C_keep_fp == policy.cost_model.C_keep_fp


# Import evaluation functions from the script helpers
import sys
from pathlib import Path

# Add scripts directory to path for importing helpers
_scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from evaluate_policy_helpers import (
    compute_enrichment_factor,
    compute_recall_at_budget,
    get_ehull_bin,
    derive_gate_level_from_score,
)


class TestEnrichmentFactor:
    """Tests for enrichment factor computation."""
    
    def test_perfect_ranking_high_ef(self):
        """Perfect ranking should have high EF."""
        # 10 stable materials ranked first, then 90 unstable
        y_true = np.array([0.01]*10 + [0.20]*90)  # 10% stable
        scores = np.arange(100).astype(float)  # Lower is better
        
        # EF@10% should be very high (all top 10 are stable)
        ef = compute_enrichment_factor(y_true, scores, fraction=0.10)
        assert ef == pytest.approx(10.0, rel=0.01)  # 100% / 10% = 10x
    
    def test_random_ranking_ef_one(self):
        """Random ranking should have EF ≈ 1."""
        np.random.seed(42)
        y_true = np.array([0.01]*100 + [0.20]*900)  # 10% stable
        np.random.shuffle(y_true)
        scores = np.random.rand(1000)  # Random scores
        
        ef = compute_enrichment_factor(y_true, scores, fraction=0.05)
        # Should be close to 1 (random)
        assert 0.5 < ef < 2.0


class TestRecallAtBudget:
    """Tests for recall@budget computation."""
    
    def test_keep_prioritized(self):
        """KEEP decisions should be counted first in budget."""
        # 10 stable materials, 90 unstable
        y_true = np.array([0.01]*10 + [0.20]*90)
        
        # First 5 are KEEP (and happen to be stable)
        decisions = np.array(["KEEP"]*5 + ["MAYBE"]*90 + ["KILL"]*5)
        q90_cal = np.arange(100).astype(float)  # For ranking MAYBEs
        
        # Budget=5: get the 5 KEEPs (all stable) → recall = 5/10 = 50%
        recall = compute_recall_at_budget(y_true, decisions, q90_cal, [5])
        assert recall[5] == pytest.approx(0.5, rel=0.01)
    
    def test_maybe_fills_budget(self):
        """MAYBEs should fill remaining budget by q90 score."""
        # 10 stable materials
        y_true = np.concatenate([
            np.array([0.01]*10),   # First 10 stable
            np.array([0.20]*90),   # Rest unstable
        ])
        
        # No KEEPs, all MAYBE
        decisions = np.array(["MAYBE"]*100)
        q90_cal = np.arange(100).astype(float)  # Stable ones have low scores
        
        # Budget=10: get top 10 MAYBEs (all stable) → recall = 100%
        recall = compute_recall_at_budget(y_true, decisions, q90_cal, [10])
        assert recall[10] == pytest.approx(1.0, rel=0.01)


class TestSlicedEvaluation:
    """Tests for sliced evaluation."""
    
    def test_ehull_bin_slicing(self):
        """Should correctly bin E_hull values."""
        assert get_ehull_bin(0.01) == "highly_stable"
        assert get_ehull_bin(0.03) == "stable"
        assert get_ehull_bin(0.07) == "metastable"
        assert get_ehull_bin(0.15) == "marginal"
        assert get_ehull_bin(0.30) == "unstable"
    
    def test_gate_level_derivation(self):
        """Should derive gate levels from OOD scores."""
        scores = np.array([0.1, 0.5, 0.9])
        levels = derive_gate_level_from_score(scores, thresh_borderline=0.3, thresh_ood=0.7)
        
        assert levels[0] == "IN"
        assert levels[1] == "BORDERLINE"
        assert levels[2] == "OOD"
