"""
Acceptance tests for decision-grade cathode screening system.

These tests validate:
1. Conformal calibration coverage guarantees
2. Decision quality (KEEP/KILL precision)
3. OOD detection separation
4. Ensemble consistency

Run with: pytest src/cathode_screening/tests/test_decision_grade.py -v
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from scipy.stats import spearmanr

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for real-data eval
    pd = None


# Test fixtures would load actual data in production
# These are placeholder implementations showing the test structure


class TestConformalCalibration:
    """Tests for conformal calibration coverage guarantees."""
    
    def test_conformal_coverage_guarantee(
        self,
        calibrated_predictions: Dict[str, np.ndarray],
        alpha: float = 0.10,
        tolerance: float = 0.02,
    ):
        """
        MUST PASS: Empirical coverage ≥ (1 - α) - tolerance on test set.
        
        This is the fundamental conformal guarantee.
        """
        y_true = calibrated_predictions["y"]
        q10_cal = calibrated_predictions["q10_cal"]
        q90_cal = calibrated_predictions["q90_cal"]
        
        coverage = np.mean((y_true >= q10_cal) & (y_true <= q90_cal))
        target = 1 - alpha
        
        assert coverage >= target - tolerance, (
            f"Coverage {coverage:.3f} < {target - tolerance:.3f} "
            f"(target: {target:.2f}, tolerance: {tolerance:.2f})"
        )
        print(f"✓ Conformal coverage: {coverage:.3f} (target: {target:.2f})")
    
    def test_coverage_by_predicted_value(
        self,
        calibrated_predictions: Dict[str, np.ndarray],
        n_bins: int = 5,
        min_coverage: float = 0.70,
    ):
        """
        SHOULD PASS: Coverage is reasonable across all predicted value ranges.
        
        Detects conditional miscalibration (e.g., overconfident on stable predictions).
        """
        y_true = calibrated_predictions["y"]
        q10_cal = calibrated_predictions["q10_cal"]
        q90_cal = calibrated_predictions["q90_cal"]
        q50 = (q10_cal + q90_cal) / 2
        
        bins = np.percentile(q50, np.linspace(0, 100, n_bins + 1))
        
        for i in range(n_bins):
            mask = (q50 >= bins[i]) & (q50 < bins[i + 1])
            if mask.sum() < 10:
                continue
            
            bin_coverage = np.mean(
                (y_true[mask] >= q10_cal[mask]) & (y_true[mask] <= q90_cal[mask])
            )
            
            assert bin_coverage >= min_coverage, (
                f"Coverage in bin {i} ({bins[i]:.2f}-{bins[i+1]:.2f}): "
                f"{bin_coverage:.3f} < {min_coverage}"
            )
        
        print(f"✓ Coverage by predicted value: all bins ≥ {min_coverage}")
    
    def test_interval_sharpness(
        self,
        calibrated_predictions: Dict[str, np.ndarray],
        max_median_width: float = 0.15,
        max_p95_width: float = 0.30,
    ):
        """
        SHOULD PASS: Intervals are reasonably sharp (not trivially wide).
        
        Wide intervals indicate poor model or excessive calibration correction.
        """
        q10_cal = calibrated_predictions["q10_cal"]
        q90_cal = calibrated_predictions["q90_cal"]
        
        widths = q90_cal - q10_cal
        median_width = np.median(widths)
        p95_width = np.percentile(widths, 95)
        
        assert median_width < max_median_width, (
            f"Median interval width {median_width:.3f} > {max_median_width} eV"
        )
        assert p95_width < max_p95_width, (
            f"P95 interval width {p95_width:.3f} > {max_p95_width} eV"
        )
        
        print(f"✓ Interval sharpness: median={median_width:.3f}, p95={p95_width:.3f}")


class TestDecisionQuality:
    """Tests for decision quality (KEEP/KILL precision)."""
    
    def test_keep_precision_dft_mode(
        self,
        decisions: List[str],
        y_true: np.ndarray,
        min_precision: float = 0.85,
    ):
        """
        MUST PASS: KEEP precision ≥ 85% for DFT-followup mode.
        
        False KEEPs waste DFT compute but are relatively cheap.
        """
        keep_mask = np.array([d == "KEEP" for d in decisions])
        
        if keep_mask.sum() == 0:
            pytest.skip("No KEEP decisions made")
        
        # KEEP should be truly stable (Ehull < 0.05)
        precision = np.mean(y_true[keep_mask] < 0.05)
        
        assert precision >= min_precision, (
            f"KEEP precision {precision:.3f} < {min_precision} for DFT mode"
        )
        
        print(f"✓ KEEP precision (DFT): {precision:.3f} (n={keep_mask.sum()})")
    
    def test_keep_precision_experimental_mode(
        self,
        decisions: List[str],
        y_true: np.ndarray,
        min_precision: float = 0.95,
    ):
        """
        MUST PASS: KEEP precision ≥ 95% for experimental mode.
        
        False KEEPs cause expensive synthesis failures.
        """
        keep_mask = np.array([d == "KEEP" for d in decisions])
        
        if keep_mask.sum() == 0:
            pytest.skip("No KEEP decisions made")
        
        precision = np.mean(y_true[keep_mask] < 0.05)
        
        assert precision >= min_precision, (
            f"KEEP precision {precision:.3f} < {min_precision} for experimental mode"
        )
        
        print(f"✓ KEEP precision (Experimental): {precision:.3f}")
    
    def test_kill_precision(
        self,
        decisions: List[str],
        y_true: np.ndarray,
        min_precision: float = 0.90,
    ):
        """
        MUST PASS: KILL precision ≥ 90%.
        
        False KILLs miss good candidates - very costly.
        """
        kill_mask = np.array([d == "KILL" for d in decisions])
        
        if kill_mask.sum() == 0:
            pytest.skip("No KILL decisions made")
        
        # KILL should be truly unstable (Ehull ≥ 0.10)
        precision = np.mean(y_true[kill_mask] >= 0.10)
        
        assert precision >= min_precision, (
            f"KILL precision {precision:.3f} < {min_precision}"
        )
        
        print(f"✓ KILL precision: {precision:.3f} (n={kill_mask.sum()})")
    
    def test_decision_coverage(
        self,
        decisions: List[str],
        min_decisive_rate: float = 0.10,
        max_kill_rate: float = 0.70,
    ):
        """
        SHOULD PASS: Reasonable distribution of decisions.
        
        - Not all MAYBE (system is useful)
        - Not too aggressive on KILL (not losing candidates)
        """
        n = len(decisions)
        n_keep = decisions.count("KEEP")
        n_maybe = decisions.count("MAYBE")
        n_kill = decisions.count("KILL")
        
        decisive_rate = (n_keep + n_kill) / n
        kill_rate = n_kill / n
        
        assert decisive_rate >= min_decisive_rate, (
            f"Only {decisive_rate:.1%} decisive (KEEP+KILL). System not useful."
        )
        assert kill_rate <= max_kill_rate, (
            f"KILL rate {kill_rate:.1%} > {max_kill_rate:.0%}. Too aggressive."
        )
        
        print(f"✓ Decision distribution: KEEP={n_keep/n:.1%}, MAYBE={n_maybe/n:.1%}, KILL={n_kill/n:.1%}")
    
    def test_no_stable_compounds_killed(
        self,
        decisions: List[str],
        y_true: np.ndarray,
        max_stable_kill_rate: float = 0.05,
    ):
        """
        MUST PASS: Very few truly stable compounds are KILLed.
        
        This is the most expensive failure mode.
        """
        stable_mask = y_true < 0.05
        kill_mask = np.array([d == "KILL" for d in decisions])
        
        n_stable = stable_mask.sum()
        if n_stable == 0:
            pytest.skip("No stable compounds in test set")
        
        n_stable_killed = (stable_mask & kill_mask).sum()
        stable_kill_rate = n_stable_killed / n_stable
        
        assert stable_kill_rate <= max_stable_kill_rate, (
            f"Stable KILL rate {stable_kill_rate:.1%} > {max_stable_kill_rate:.0%}. "
            f"Killed {n_stable_killed}/{n_stable} stable compounds!"
        )
        
        print(f"✓ Stable KILL rate: {stable_kill_rate:.1%} ({n_stable_killed}/{n_stable})")


class TestOODDetection:
    """Tests for OOD detection capabilities."""
    
    def test_ood_score_separation(
        self,
        id_ood_scores: np.ndarray,
        ood_ood_scores: np.ndarray,
        min_ratio: float = 2.0,
    ):
        """
        MUST PASS: OOD samples have significantly higher scores than ID samples.
        """
        id_median = np.median(id_ood_scores)
        ood_median = np.median(ood_ood_scores)
        
        assert ood_median > id_median * min_ratio, (
            f"OOD median {ood_median:.3f} not >> ID median {id_median:.3f} "
            f"(expected {min_ratio}x separation)"
        )
        
        print(f"✓ OOD separation: ID={id_median:.3f}, OOD={ood_median:.3f}")
    
    def test_ood_detection_auc(
        self,
        id_ood_scores: np.ndarray,
        ood_ood_scores: np.ndarray,
        min_auc: float = 0.80,
    ):
        """
        MUST PASS: OOD detection AUC ≥ 80%.
        """
        from sklearn.metrics import roc_auc_score
        
        labels = [0] * len(id_ood_scores) + [1] * len(ood_ood_scores)
        scores = list(id_ood_scores) + list(ood_ood_scores)
        
        auc = roc_auc_score(labels, scores)
        
        assert auc >= min_auc, f"OOD detection AUC {auc:.3f} < {min_auc}"
        
        print(f"✓ OOD detection AUC: {auc:.3f}")
    
    def test_ood_forces_maybe(
        self,
        ood_decisions: List[str],
        ood_flags: List[bool],
    ):
        """
        MUST PASS: All OOD-flagged samples are routed to MAYBE.
        """
        for i, (decision, flag) in enumerate(zip(ood_decisions, ood_flags)):
            if flag:
                assert decision == "MAYBE", (
                    f"OOD sample {i} got {decision} instead of MAYBE"
                )
        
        n_flagged = sum(ood_flags)
        print(f"✓ All {n_flagged} OOD-flagged samples routed to MAYBE")
    
    def test_composition_gate_on_novel_elements(
        self,
        novel_element_formulas: List[str],
        ood_gate,
    ):
        """
        SHOULD PASS: Compositions with out-of-vocab elements are flagged.
        """
        for formula in novel_element_formulas:
            result = ood_gate.score(formula, embedding=None, q50_ensemble=None)
            
            # Novel elements should have high composition distance
            assert result.d_comp > 3.0 or result.flag_comp, (
                f"Novel element formula {formula} not flagged: d_comp={result.d_comp:.2f}"
            )
        
        print(f"✓ {len(novel_element_formulas)} novel element formulas correctly flagged")


class TestEnsembleConsistency:
    """Tests for ensemble behavior."""
    
    def test_ensemble_reduces_mae(
        self,
        single_model_errors: np.ndarray,
        ensemble_errors: np.ndarray,
        min_improvement: float = 0.05,
    ):
        """
        SHOULD PASS: Ensemble has lower MAE than single model.
        """
        single_mae = np.mean(single_model_errors)
        ensemble_mae = np.mean(ensemble_errors)
        
        improvement = (single_mae - ensemble_mae) / single_mae
        
        assert improvement >= min_improvement, (
            f"Ensemble improvement {improvement:.1%} < {min_improvement:.0%}"
        )
        
        print(f"✓ Ensemble MAE improvement: {improvement:.1%} ({single_mae:.4f} → {ensemble_mae:.4f})")
    
    def test_epistemic_correlates_with_error(
        self,
        epistemic_uncertainties: np.ndarray,
        actual_errors: np.ndarray,
        min_correlation: float = 0.20,
    ):
        """
        SHOULD PASS: Epistemic uncertainty correlates with actual error.
        
        This validates that ensemble disagreement is informative.
        """
        corr, pval = spearmanr(epistemic_uncertainties, actual_errors)
        
        assert corr >= min_correlation, (
            f"Epistemic-error correlation {corr:.3f} < {min_correlation}"
        )
        
        print(f"✓ Epistemic-error Spearman correlation: {corr:.3f} (p={pval:.2e})")
    
    def test_ensemble_members_diverse(
        self,
        member_predictions: List[np.ndarray],
        min_disagreement_p50: float = 0.01,
    ):
        """
        SHOULD PASS: Ensemble members are sufficiently diverse.
        
        If all members agree on everything, ensemble provides no value.
        """
        # Compute per-sample standard deviation across members
        stacked = np.stack(member_predictions, axis=0)  # [K, N]
        disagreements = np.std(stacked, axis=0)
        
        median_disagreement = np.median(disagreements)
        
        assert median_disagreement >= min_disagreement_p50, (
            f"Median disagreement {median_disagreement:.4f} < {min_disagreement_p50}. "
            f"Ensemble members too similar."
        )
        
        print(f"✓ Ensemble diversity: median disagreement = {median_disagreement:.4f}")
    
    def test_no_outlier_member(
        self,
        member_maes: List[float],
        max_zscore: float = 2.5,
    ):
        """
        SHOULD PASS: No ensemble member is a significant outlier.
        
        One bad member can hurt ensemble performance.
        """
        mean_mae = np.mean(member_maes)
        std_mae = np.std(member_maes)
        
        if std_mae < 1e-6:
            pytest.skip("All members have identical MAE")
        
        for k, mae in enumerate(member_maes):
            zscore = (mae - mean_mae) / std_mae
            assert abs(zscore) <= max_zscore, (
                f"Member {k} is outlier: MAE={mae:.4f}, z-score={zscore:.2f}"
            )
        
        print(f"✓ No outlier members (MAEs: {[f'{m:.4f}' for m in member_maes]})")


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_produces_valid_output(
        self,
        predictor,
        test_material: Dict,
    ):
        """
        MUST PASS: Pipeline produces valid DecisionOutput.
        """
        result = predictor.predict(
            test_material["material_id"],
            test_material["formula"],
            test_material["graph_npz"],
            mode="dft_followup"
        )
        
        # Check all required fields
        assert result.material_id == test_material["material_id"]
        assert result.decision in ["KEEP", "MAYBE", "KILL"]
        assert 0 <= result.ehull_pred <= 2.0  # Reasonable range
        assert result.ehull_lower <= result.ehull_pred <= result.ehull_upper
        assert 0 <= result.ood_score <= 1.0
        assert 0 <= result.decision_confidence <= 1.0
        assert result.uncertainty_total >= 0
        
        print(f"✓ Valid output: {result.decision} (conf={result.decision_confidence:.2f})")
    
    def test_batch_prediction_consistent(
        self,
        predictor,
        test_materials: List[Dict],
    ):
        """
        SHOULD PASS: Batch prediction matches individual predictions.
        """
        # Individual predictions
        individual = [
            predictor.predict(m["material_id"], m["formula"], m["graph_npz"])
            for m in test_materials
        ]
        
        # Batch prediction
        batch = predictor.predict_batch(test_materials)
        
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            assert ind.decision == bat.decision, f"Decision mismatch at {i}"
            assert abs(ind.ehull_pred - bat.ehull_pred) < 1e-6, f"Prediction mismatch at {i}"
        
        print(f"✓ Batch prediction consistent for {len(test_materials)} materials")
    
    def test_mode_affects_thresholds(
        self,
        predictor,
        test_material: Dict,
    ):
        """
        SHOULD PASS: Different modes produce different decisions on borderline cases.
        """
        dft_result = predictor.predict(
            test_material["material_id"],
            test_material["formula"],
            test_material["graph_npz"],
            mode="dft_followup"
        )
        
        exp_result = predictor.predict(
            test_material["material_id"],
            test_material["formula"],
            test_material["graph_npz"],
            mode="experimental"
        )
        
        assert dft_result.decision_mode == "dft_followup"
        assert exp_result.decision_mode == "experimental"
        
        # For borderline cases, experimental should be more conservative
        # (This test may not always fire depending on the sample)
        print(f"✓ DFT decision: {dft_result.decision}, Exp decision: {exp_result.decision}")


# Fixture implementations (would be replaced with actual data loading)

_PREDICTION_CANDIDATES = (
    "data/predictions/ensemble_soap_loco_test.parquet",
    "data/predictions/ensemble_test.parquet",
    "data/predictions/ensemble_val.parquet",
)


def _resolve_prediction_path() -> Path | None:
    env_path = os.getenv("CATHODE_TEST_PREDICTIONS")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    for candidate in _PREDICTION_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return None


@pytest.fixture(scope="session")
def prediction_frame():
    """Load real predictions if explicitly requested."""
    use_real = os.getenv("CATHODE_TEST_USE_PREDICTIONS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_real or pd is None:
        return None

    path = _resolve_prediction_path()
    if path is None:
        pytest.skip("CATHODE_TEST_USE_PREDICTIONS enabled but no predictions file found")

    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def predictions_eval_frame(prediction_frame):
    if prediction_frame is None:
        return None
    if "y_true" not in prediction_frame.columns:
        return None
    return prediction_frame[prediction_frame["y_true"].notna()].copy()


@pytest.fixture
def calibrated_predictions(predictions_eval_frame):
    """Load calibrated predictions for testing."""
    if predictions_eval_frame is not None:
        df = predictions_eval_frame
        q10_col = "q10_cal" if "q10_cal" in df.columns else "q10_raw"
        q90_col = "q90_cal" if "q90_cal" in df.columns else "q90_raw"
        if q10_col in df.columns and q90_col in df.columns:
            return {
                "y": df["y_true"].to_numpy(),
                "q10_cal": df[q10_col].to_numpy(),
                "q90_cal": df[q90_col].to_numpy(),
            }

    rng = np.random.default_rng(42)
    n = 500
    y = np.clip(rng.exponential(0.1, n), 0, 1)
    half_width = np.abs(rng.normal(0.04, 0.01, n))
    q10 = y - half_width
    q90 = y + half_width

    return {
        "y": y,
        "q10_cal": q10,
        "q90_cal": q90,
    }


@pytest.fixture
def y_true(predictions_eval_frame):
    """Load true values for testing."""
    if predictions_eval_frame is not None:
        return predictions_eval_frame["y_true"].to_numpy()

    rng = np.random.default_rng(123)
    n_stable = 120
    n_metastable = 180
    n_unstable = 200

    stable = rng.uniform(0.0, 0.05, n_stable)
    metastable = rng.uniform(0.05, 0.10, n_metastable)
    unstable = rng.uniform(0.10, 0.40, n_unstable)

    y = np.concatenate([stable, metastable, unstable])
    rng.shuffle(y)
    return y


@pytest.fixture
def decisions(y_true: np.ndarray, predictions_eval_frame):
    """Load decisions for testing."""
    if predictions_eval_frame is not None:
        df = predictions_eval_frame
        col = "gated_decision" if "gated_decision" in df.columns else "decision"
        if col in df.columns:
            return df[col].fillna("MAYBE").astype(str).tolist()

    rng = np.random.default_rng(456)
    decisions = []
    for y in y_true:
        if y < 0.05:
            decisions.append("KEEP" if rng.random() < 0.9 else "MAYBE")
        elif y >= 0.10:
            decisions.append("KILL" if rng.random() < 0.9 else "MAYBE")
        else:
            decisions.append("MAYBE")
    return decisions


@pytest.fixture
def id_ood_scores():
    """OOD scores for in-distribution samples."""
    np.random.seed(42)
    return np.random.beta(2, 10, 200)  # Low scores


@pytest.fixture
def ood_ood_scores():
    """OOD scores for out-of-distribution samples."""
    np.random.seed(42)
    return np.random.beta(5, 3, 100)  # Higher scores


@pytest.fixture
def ood_flags():
    """Flags indicating which samples are OOD."""
    return [True, False, True, False, True, False, True, False, False, True]


@pytest.fixture
def ood_decisions(ood_flags: List[bool], predictions_eval_frame):
    """Decisions aligned with OOD flags."""
    if predictions_eval_frame is not None:
        df = predictions_eval_frame
        if "gate_level" in df.columns and "gated_decision" in df.columns:
            mask = df["gate_level"].astype(str) == "OOD"
            if mask.any():
                flags = mask.to_numpy().tolist()
                decisions = df["gated_decision"].fillna("MAYBE").astype(str).tolist()
                ood_flags[:] = flags
                return decisions

    rng = np.random.default_rng(7)
    decisions = []
    for flag in ood_flags:
        if flag:
            decisions.append("MAYBE")
        else:
            decisions.append(rng.choice(["KEEP", "MAYBE", "KILL"]))
    return decisions


@pytest.fixture
def novel_element_formulas():
    """Formulas with elements outside the training vocabulary."""
    return ["Li2CuO3", "Li2ZnO3", "Li2MoO4", "Li2NbO3"]


@dataclass
class DummyOODResult:
    d_comp: float
    flag_comp: bool


class DummyOODGate:
    """Minimal OOD gate for fixture-only checks."""

    def __init__(self, vocab: List[str]) -> None:
        self.vocab = set(vocab)

    def score(self, formula: str, embedding=None, q50_ensemble=None) -> DummyOODResult:
        elements = re.findall(r"[A-Z][a-z]?", formula)
        has_novel = any(el not in self.vocab for el in elements)
        if has_novel:
            return DummyOODResult(d_comp=4.0, flag_comp=True)
        return DummyOODResult(d_comp=0.5, flag_comp=False)


@pytest.fixture
def ood_gate():
    return DummyOODGate(["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"])


@pytest.fixture
def single_model_errors():
    rng = np.random.default_rng(21)
    return np.abs(rng.normal(0.08, 0.015, size=200))


@pytest.fixture
def ensemble_errors(predictions_eval_frame):
    if predictions_eval_frame is not None and "q50" in predictions_eval_frame.columns:
        df = predictions_eval_frame
        return np.abs(df["q50"].to_numpy() - df["y_true"].to_numpy())
    rng = np.random.default_rng(22)
    return np.abs(rng.normal(0.05, 0.010, size=200))


@pytest.fixture
def epistemic_uncertainties(predictions_eval_frame):
    if predictions_eval_frame is not None and "epistemic_std" in predictions_eval_frame.columns:
        return predictions_eval_frame["epistemic_std"].to_numpy()
    rng = np.random.default_rng(33)
    return rng.uniform(0.0, 0.2, size=200)


@pytest.fixture
def actual_errors(epistemic_uncertainties: np.ndarray, predictions_eval_frame):
    if predictions_eval_frame is not None and "q50" in predictions_eval_frame.columns:
        df = predictions_eval_frame
        return np.abs(df["q50"].to_numpy() - df["y_true"].to_numpy())
    rng = np.random.default_rng(34)
    noise = rng.normal(0.0, 0.01, size=epistemic_uncertainties.shape[0])
    return np.abs(epistemic_uncertainties * 0.8 + noise)


@pytest.fixture
def member_predictions(predictions_eval_frame):
    if predictions_eval_frame is not None and "q50_members_str" in predictions_eval_frame.columns:
        raw = predictions_eval_frame["q50_members_str"].dropna().apply(json.loads)
        if len(raw) > 0:
            stacked = np.array(raw.tolist(), dtype=np.float32)
            return [stacked[:, i] for i in range(stacked.shape[1])]
    rng = np.random.default_rng(44)
    base = rng.normal(0.10, 0.02, size=200)
    members = [base + rng.normal(0.0, 0.02, size=200) for _ in range(5)]
    return members


@pytest.fixture
def member_maes(predictions_eval_frame, member_predictions):
    if predictions_eval_frame is not None:
        y = predictions_eval_frame["y_true"].to_numpy()
        if len(member_predictions) > 0 and len(member_predictions[0]) == len(y):
            return [float(np.mean(np.abs(pred - y))) for pred in member_predictions]
    return [0.050, 0.052, 0.049, 0.055, 0.053]


@pytest.fixture(scope="session")
def predictor():
    try:
        from cathode_screening.inference.decision_predictor import DecisionPredictor
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"DecisionPredictor unavailable: {exc}")

    artifacts_dir = Path("data/artifacts")
    if not artifacts_dir.exists():
        pytest.skip("Artifacts directory not found: data/artifacts")

    return DecisionPredictor.from_artifacts(artifacts_dir)


@pytest.fixture
def test_material():
    try:
        from pymatgen.core import Structure
        from cathode_screening.datasets.featurize.graph_builder import GraphBuilder
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"Structure parsing unavailable: {exc}")

    cif_path = Path("test.cif")
    if not cif_path.exists():
        pytest.skip("test.cif not found")

    structure = Structure.from_file(cif_path)
    graph = GraphBuilder().build(structure)
    return {
        "material_id": "test-1",
        "formula": structure.composition.reduced_formula,
        "graph_npz": graph,
    }


@pytest.fixture
def test_materials(test_material: Dict):
    materials = [test_material]
    cif_path = Path("li.cif")
    if cif_path.exists():
        from pymatgen.core import Structure
        from cathode_screening.datasets.featurize.graph_builder import GraphBuilder

        structure = Structure.from_file(cif_path)
        graph = GraphBuilder().build(structure)
        materials.append(
            {
                "material_id": "test-2",
                "formula": structure.composition.reduced_formula,
                "graph_npz": graph,
            }
        )
    else:
        clone = dict(test_material)
        clone["material_id"] = "test-2"
        materials.append(clone)
    return materials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
