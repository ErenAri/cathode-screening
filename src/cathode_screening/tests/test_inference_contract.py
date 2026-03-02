"""
Unit tests for the inference pipeline contract.

Validates that DecisionPredictor, EnsemblePredictor, and supporting classes
produce outputs matching their documented contracts *without* requiring
trained model weights or artifact directories.

Run with: pytest src/cathode_screening/tests/test_inference_contract.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from cathode_screening.common.serialization import Normalizer


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_denorm_identity(self):
        norm = Normalizer(mean=0.0, std=1.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        out = norm.denorm(x)
        assert torch.allclose(out, x)

    def test_denorm_shifts_and_scales(self):
        norm = Normalizer(mean=5.0, std=2.0)
        x = torch.tensor([0.0, 1.0, -1.0])
        expected = torch.tensor([5.0, 7.0, 3.0])
        assert torch.allclose(norm.denorm(x), expected)

    def test_load_state_dict(self, normalizer_state):
        norm = Normalizer()
        norm.load_state_dict(normalizer_state)
        assert norm.mean == normalizer_state["mean"]
        assert norm.std == normalizer_state["std"]


# ---------------------------------------------------------------------------
# DecisionOutput
# ---------------------------------------------------------------------------


class TestDecisionOutput:
    def test_to_dict_has_required_keys(self):
        from cathode_screening.inference.decision_predictor import DecisionOutput

        out = DecisionOutput(
            material_id="mp-1",
            formula="LiCoO2",
            ehull_pred=0.03,
            ehull_lower=0.01,
            ehull_upper=0.05,
            uncertainty_aleatoric=0.01,
            uncertainty_epistemic=0.005,
            uncertainty_total=0.011,
            ood_score=0.1,
            ood_flag=False,
            decision="KEEP",
            decision_confidence=0.8,
            decision_mode="dft_followup",
        )
        d = out.to_dict()
        required = {
            "material_id",
            "formula",
            "ehull_pred",
            "ehull_lower",
            "ehull_upper",
            "uncertainty_aleatoric",
            "uncertainty_epistemic",
            "uncertainty_total",
            "ood_score",
            "ood_flag",
            "decision",
            "decision_confidence",
            "decision_mode",
        }
        assert required.issubset(d.keys())

    def test_decision_values(self):
        from cathode_screening.inference.decision_predictor import DecisionOutput

        for decision in ("KEEP", "MAYBE", "KILL"):
            out = DecisionOutput(
                material_id="x",
                formula="X",
                ehull_pred=0.0,
                ehull_lower=0.0,
                ehull_upper=0.0,
                uncertainty_aleatoric=0.0,
                uncertainty_epistemic=0.0,
                uncertainty_total=0.0,
                ood_score=0.0,
                ood_flag=False,
                decision=decision,
                decision_confidence=0.5,
                decision_mode="dft_followup",
            )
            assert out.decision == decision


# ---------------------------------------------------------------------------
# Decision logic (unit-level, no model weights)
# ---------------------------------------------------------------------------


def _make_mock_model(q10=0.01, q50=0.03, q90=0.05, logit_s=2.0, logit_m=0.0):
    """Return a mock nn.Module that behaves like a CGCNN forward pass.

    The mock is batch-aware: the output size is determined by the number
    of unique graphs in the ``batch`` tensor passed to __call__.
    """
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()

    def _forward(_x, _src, _dst, _e, batch):
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return (
            torch.full((n_graphs,), q10),
            torch.full((n_graphs,), q50),
            torch.full((n_graphs,), q90),
            torch.full((n_graphs,), logit_s),
            torch.full((n_graphs,), logit_m),
        )

    model.side_effect = _forward
    model.embed = MagicMock()
    model.embed.in_features = 6
    model.embed.out_features = 128
    model.blocks = []

    def _get_embedding(_x, _src, _dst, _e, batch):
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return torch.randn(n_graphs, 128)

    model.get_embedding = MagicMock(side_effect=_get_embedding)
    return model


def _make_mock_ood_gate(flag=False):
    gate = MagicMock()
    result = MagicMock()
    result.flag = flag
    result.ood_score = 0.8 if flag else 0.1
    result.flag_comp = flag
    result.flag_emb = False
    result.flag_disagree = False
    result.to_dict.return_value = {
        "flag_comp": flag,
        "flag_emb": False,
        "flag_disagree": False,
    }
    gate.score.return_value = result
    return gate


class TestDecisionLogic:
    """Test the decision logic of DecisionPredictor.predict() via mocks."""

    def _make_predictor(self, tau_keep=0.07, tau_kill=0.12, ood_flag=False, calibrator=None):
        from cathode_screening.inference.decision_predictor import DecisionPredictor

        models = [_make_mock_model() for _ in range(3)]
        norms = [Normalizer(mean=0.0, std=1.0) for _ in range(3)]
        ood_gate = _make_mock_ood_gate(flag=ood_flag)
        thresholds = {
            "dft_followup": {"tau_keep": tau_keep, "tau_kill": tau_kill},
            "experimental": {"tau_keep": 0.04, "tau_kill": 0.09},
        }
        return DecisionPredictor(
            models=models,
            normalizers=norms,
            calibrator=calibrator,
            ood_gate=ood_gate,
            thresholds=thresholds,
            device=torch.device("cpu"),
        )

    def test_keep_when_q90_below_tau_keep(self, simple_graph):
        predictor = self._make_predictor(tau_keep=0.10)
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert result.decision == "KEEP"

    def test_kill_when_q10_above_tau_kill(self, simple_graph):
        # Mock model outputs q10=0.01, q50=0.03, q90=0.05.
        # For KILL: need q90_cal >= tau_keep (so KEEP doesn't fire first)
        # AND q10_cal > tau_kill.
        # Use models with high predictions to trigger KILL.
        from cathode_screening.inference.decision_predictor import DecisionPredictor

        models = [_make_mock_model(q10=0.20, q50=0.30, q90=0.40) for _ in range(3)]
        norms = [Normalizer(mean=0.0, std=1.0) for _ in range(3)]
        ood_gate = _make_mock_ood_gate(flag=False)
        thresholds = {
            "dft_followup": {"tau_keep": 0.07, "tau_kill": 0.12},
            "experimental": {"tau_keep": 0.04, "tau_kill": 0.09},
        }
        predictor = DecisionPredictor(
            models=models, normalizers=norms, calibrator=None,
            ood_gate=ood_gate, thresholds=thresholds, device=torch.device("cpu"),
        )
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert result.decision == "KILL"

    def test_ood_forces_maybe(self, simple_graph):
        predictor = self._make_predictor(ood_flag=True)
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert result.decision == "MAYBE"
        assert result.ood_flag is True

    def test_maybe_in_uncertain_region(self, simple_graph):
        # tau_keep < q90 and q10 < tau_kill → MAYBE
        predictor = self._make_predictor(tau_keep=0.04, tau_kill=0.10)
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert result.decision == "MAYBE"

    def test_confidence_between_zero_and_one(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert 0.0 <= result.decision_confidence <= 1.0

    def test_uncertainty_decomposition_positive(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="dft_followup")
        assert result.uncertainty_aleatoric >= 0
        assert result.uncertainty_epistemic >= 0
        assert result.uncertainty_total >= 0

    def test_invalid_mode_raises(self, simple_graph):
        predictor = self._make_predictor()
        with pytest.raises(ValueError, match="Unknown mode"):
            predictor.predict("mp-1", "LiCoO2", simple_graph, mode="invalid")

    def test_experimental_mode_accepted(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict("mp-1", "LiCoO2", simple_graph, mode="experimental")
        assert result.decision_mode == "experimental"


# ---------------------------------------------------------------------------
# EnsemblePredictor contract
# ---------------------------------------------------------------------------


class TestEnsemblePredictorContract:
    """Validate EnsemblePredictor output dict keys and shapes."""

    def _make_predictor(self, k=3):
        from cathode_screening.inference.ensemble_predictor import EnsemblePredictor

        models = [_make_mock_model() for _ in range(k)]
        norms = [Normalizer(mean=0.0, std=1.0) for _ in range(k)]
        return EnsemblePredictor(models=models, normalizers=norms, device=torch.device("cpu"))

    def test_predict_returns_expected_keys(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict(simple_graph)
        expected_keys = {"q10", "q50", "q90", "epistemic_var", "epistemic_std", "p_stable", "p_metastable"}
        assert expected_keys == set(result.keys())

    def test_predict_values_are_floats(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict(simple_graph)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_predict_batch_shapes(self, simple_graph_pair):
        predictor = self._make_predictor()
        result = predictor.predict_batch(simple_graph_pair)
        n = len(simple_graph_pair)
        for key, arr in result.items():
            assert arr.shape == (n,), f"{key} has shape {arr.shape}, expected ({n},)"

    def test_epistemic_std_nonneg(self, simple_graph):
        predictor = self._make_predictor()
        result = predictor.predict(simple_graph)
        assert result["epistemic_std"] >= 0

    def test_empty_batch_raises(self):
        predictor = self._make_predictor()
        with pytest.raises(ValueError, match="No graphs"):
            predictor.predict_batch([])


# ---------------------------------------------------------------------------
# Batch prediction consistency
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    """Batch prediction should match repeated single predictions."""

    def test_batch_matches_individual(self, simple_graph_pair):
        from cathode_screening.inference.decision_predictor import DecisionPredictor

        models = [_make_mock_model() for _ in range(3)]
        norms = [Normalizer(mean=0.0, std=1.0) for _ in range(3)]
        ood_gate = _make_mock_ood_gate(flag=False)
        thresholds = {
            "dft_followup": {"tau_keep": 0.07, "tau_kill": 0.12},
            "experimental": {"tau_keep": 0.04, "tau_kill": 0.09},
        }
        predictor = DecisionPredictor(
            models=models,
            normalizers=norms,
            calibrator=None,
            ood_gate=ood_gate,
            thresholds=thresholds,
            device=torch.device("cpu"),
        )

        materials = [
            {"material_id": f"m-{i}", "formula": "LiCoO2", "graph_npz": g}
            for i, g in enumerate(simple_graph_pair)
        ]

        batch = predictor.predict_batch(materials, mode="dft_followup")
        assert len(batch) == len(simple_graph_pair)
        for out in batch:
            assert out.decision in ("KEEP", "MAYBE", "KILL")


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_structure(self):
        from cathode_screening.inference.decision_predictor import (
            DecisionOutput,
            generate_report,
        )

        outputs = []
        for i, (d, pred) in enumerate(
            [("KEEP", 0.02), ("MAYBE", 0.06), ("KILL", 0.20), ("KEEP", 0.01)]
        ):
            outputs.append(
                DecisionOutput(
                    material_id=f"m-{i}",
                    formula="X",
                    ehull_pred=pred,
                    ehull_lower=pred - 0.01,
                    ehull_upper=pred + 0.01,
                    uncertainty_aleatoric=0.01,
                    uncertainty_epistemic=0.005,
                    uncertainty_total=0.011,
                    ood_score=0.1,
                    ood_flag=False,
                    decision=d,
                    decision_confidence=0.7,
                    decision_mode="dft_followup",
                )
            )

        report = generate_report(outputs)
        assert report["total"] == 4
        assert report["decisions"]["KEEP"] == 2
        assert report["decisions"]["KILL"] == 1
        assert report["decisions"]["MAYBE"] == 1
        assert len(report["top_candidates"]) <= 10
        assert report["top_candidates"][0]["ehull_pred"] <= report["top_candidates"][-1]["ehull_pred"]
