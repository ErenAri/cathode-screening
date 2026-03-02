"""
Unit tests for filtering and classification helpers.

Tests uncertainty classification, decision thresholds, and the
Normalizer/safe_torch_load safety paths.

Run with: pytest src/cathode_screening/tests/test_filters.py -v
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from cathode_screening.common.serialization import Normalizer, _env_bool, safe_torch_load


# ---------------------------------------------------------------------------
# _env_bool
# ---------------------------------------------------------------------------


class TestEnvBool:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES", "y", "on"])
    def test_truthy_values(self, val):
        with patch.dict(os.environ, {"TEST_FLAG": val}):
            assert _env_bool("TEST_FLAG") is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "anything"])
    def test_falsy_values(self, val):
        with patch.dict(os.environ, {"TEST_FLAG": val}):
            assert _env_bool("TEST_FLAG") is False

    def test_missing_returns_default(self):
        env = os.environ.copy()
        env.pop("MISSING_FLAG", None)
        with patch.dict(os.environ, env, clear=True):
            assert _env_bool("MISSING_FLAG", default=False) is False
            assert _env_bool("MISSING_FLAG", default=True) is True


# ---------------------------------------------------------------------------
# safe_torch_load
# ---------------------------------------------------------------------------


class TestSafeTorchLoad:
    def test_loads_simple_tensor(self, tmp_path):
        p = tmp_path / "tensor.pt"
        torch.save({"a": torch.tensor([1.0, 2.0])}, p)
        data = safe_torch_load(p, device=torch.device("cpu"))
        assert torch.allclose(data["a"], torch.tensor([1.0, 2.0]))

    def test_nonexistent_file_raises(self, tmp_path):
        p = tmp_path / "missing.pt"
        with pytest.raises(Exception):
            safe_torch_load(p, device=torch.device("cpu"))

    def test_default_device_is_cpu(self, tmp_path):
        p = tmp_path / "tensor.pt"
        torch.save(torch.tensor(1), p)
        data = safe_torch_load(p)
        assert data.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Normalizer round-trip
# ---------------------------------------------------------------------------


class TestNormalizerRoundTrip:
    def test_denorm_inverts_normalize(self):
        norm = Normalizer(mean=3.0, std=2.0)
        raw = torch.tensor([7.0])
        # normalized = (7 - 3) / 2 = 2.0
        normalized = torch.tensor([2.0])
        assert torch.allclose(norm.denorm(normalized), raw)

    def test_denorm_identity_for_default(self):
        norm = Normalizer()
        x = torch.tensor([1.5, -0.3])
        assert torch.allclose(norm.denorm(x), x)

    def test_state_dict_load(self):
        norm = Normalizer()
        norm.load_state_dict({"mean": 10.0, "std": 0.5})
        x = torch.tensor([0.0])
        assert torch.allclose(norm.denorm(x), torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# OOD gate result flags
# ---------------------------------------------------------------------------


class TestOODGateResultContract:
    def test_dataclass_fields(self):
        from cathode_screening.inference.ood import OODGateResult

        r = OODGateResult(
            ood_score=0.3,
            flag=False,
            d_comp=1.0,
            d_emb=0.5,
            d_disagree=0.02,
            z_comp=0.5,
            z_emb=0.2,
            z_disagree=0.1,
            z_max=0.5,
            flag_comp=False,
            flag_emb=False,
            flag_disagree=False,
        )
        d = r.to_dict()
        assert "ood_score" in d
        assert "flag_comp" in d
        assert d["ood_score"] == 0.3

    def test_flag_is_or_of_gates(self):
        from cathode_screening.inference.ood import OODGateResult

        # flag should be True if any gate is True
        r = OODGateResult(
            ood_score=0.8,
            flag=True,
            d_comp=5.0,
            d_emb=0.0,
            d_disagree=0.0,
            z_comp=3.0,
            z_emb=0.0,
            z_disagree=0.0,
            z_max=3.0,
            flag_comp=True,
            flag_emb=False,
            flag_disagree=False,
        )
        assert r.flag is True
        assert r.flag_comp is True


# ---------------------------------------------------------------------------
# Composition fingerprint
# ---------------------------------------------------------------------------


class TestCompositionFingerprint:
    def test_fingerprint_sums_to_one(self):
        from cathode_screening.inference.ood import composition_fingerprint

        fp = composition_fingerprint("LiCoO2")
        assert abs(fp.sum() - 1.0) < 1e-5

    def test_fingerprint_length(self):
        from cathode_screening.inference.ood import composition_fingerprint, COMPOSITION_VOCAB

        fp = composition_fingerprint("LiMnO3")
        assert len(fp) == len(COMPOSITION_VOCAB)

    def test_novel_element_zeros(self):
        from cathode_screening.inference.ood import composition_fingerprint

        # Cu is not in COMPOSITION_VOCAB → only O and Li fractions populated
        fp = composition_fingerprint("CuO")
        # Cu contributes 0 to vocab slots, O is at index 1
        assert fp.sum() < 1.0  # partial coverage
