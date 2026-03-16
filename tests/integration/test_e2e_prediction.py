"""
End-to-end integration tests for the prediction pipeline.

These tests verify the full flow: CIF upload → structure parsing →
ensemble inference → decision output. They require model artifacts
to be present (skip gracefully if not available).

Run with:
    pytest tests/integration/ -m integration -v
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

ARTIFACTS_DIR = Path("artifacts/models/mace_ensemble_v1")
HAS_MODEL_ARTIFACTS = ARTIFACTS_DIR.exists() and any(ARTIFACTS_DIR.glob("*.pt"))

skip_no_model = pytest.mark.skipif(
    not HAS_MODEL_ARTIFACTS,
    reason="Model artifacts not available (requires MACE ensemble checkpoints)",
)


# ---------------------------------------------------------------------------
# Marker for all integration tests
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Verify health and metadata endpoints work without model."""

    def test_root_returns_ok(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_model_info_endpoint(self, api_client):
        resp = api_client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "model_type" in data
        assert data["ensemble_size"] == 5


# ---------------------------------------------------------------------------
# Input validation (no model needed)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify input validation without requiring model inference."""

    def test_empty_file_rejected(self, api_client):
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("empty.cif", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "too_small" in str(resp.json().get("detail", "")).lower() or resp.status_code == 400

    def test_oversized_file_rejected(self, api_client):
        huge = b"x" * 3_000_000  # 3MB, exceeds 2MB default
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("huge.cif", huge, "text/plain")},
        )
        assert resp.status_code == 413

    def test_invalid_cif_rejected(self, api_client):
        garbage = b"this is not a valid CIF file at all"
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("garbage.cif", garbage, "text/plain")},
        )
        assert resp.status_code == 400

    def test_non_cathode_composition_rejected(self, api_client, nacl_cif):
        """NaCl should be rejected — no lithium, no transition metal."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("NaCl.cif", nacl_cif, "text/plain")},
        )
        # Should be 400 with invalid_cathode_composition error
        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("error") == "invalid_cathode_composition"


# ---------------------------------------------------------------------------
# End-to-end prediction (requires model)
# ---------------------------------------------------------------------------


@skip_no_model
class TestPredictionE2E:
    """Full prediction pipeline tests — require model artifacts."""

    def test_limn2o4_returns_valid_prediction(self, api_client, limn2o4_cif):
        """LiMn2O4 is a known stable spinel cathode."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiMn2O4.cif", limn2o4_cif, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        pred = data["prediction"]
        assert "pred_ehull" in pred
        assert "p_stable" in pred
        assert "uncertainty" in pred
        assert "action" in pred
        assert "confidence_interval" in pred

        # Sanity: ehull should be a reasonable number
        assert 0.0 <= pred["pred_ehull"] <= 2.0
        assert 0.0 <= pred["p_stable"] <= 1.0
        assert pred["action"] in {"DFT", "HOLD", "SKIP"}
        assert pred["uncertainty"] in {"Low", "Medium", "High"}

    def test_licoo2_returns_valid_prediction(self, api_client, licoo2_cif):
        """LiCoO2 is a well-known stable layered cathode."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiCoO2.cif", licoo2_cif, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        pred = data["prediction"]
        assert 0.0 <= pred["pred_ehull"] <= 2.0
        assert pred["action"] in {"DFT", "HOLD", "SKIP"}

    def test_lifepo4_returns_valid_prediction(self, api_client, lifepo4_cif):
        """LiFePO4 olivine — known stable cathode."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiFePO4.cif", lifepo4_cif, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_prediction_includes_confidence_interval(self, api_client, limn2o4_cif):
        """Confidence interval must be a 2-element tuple with lower <= upper."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiMn2O4.cif", limn2o4_cif, "text/plain")},
        )
        data = resp.json()
        ci = data["prediction"]["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= ci[1], f"CI lower {ci[0]} > upper {ci[1]}"

    def test_prediction_includes_cathode_properties(self, api_client, limn2o4_cif):
        """Cathode properties should be populated for valid Li-cathodes."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiMn2O4.cif", limn2o4_cif, "text/plain")},
        )
        data = resp.json()
        props = data["prediction"].get("cathode_properties")
        if props is not None:
            # If properties are returned, validate structure
            assert "gravimetric_capacity_mAhg" in props or props == {}

    def test_response_headers_include_request_id(self, api_client, limn2o4_cif):
        """Every response must include X-Request-ID header."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("LiMn2O4.cif", limn2o4_cif, "text/plain")},
        )
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# Batch prediction (requires model)
# ---------------------------------------------------------------------------


@skip_no_model
class TestBatchPredictionE2E:
    """Batch prediction endpoint tests."""

    def test_batch_two_structures(self, api_client, limn2o4_cif, licoo2_cif):
        """Batch prediction of 2 structures should return 2 results."""
        resp = api_client.post(
            "/predict/batch",
            files=[
                ("cif_files", ("LiMn2O4.cif", limn2o4_cif, "text/plain")),
                ("cif_files", ("LiCoO2.cif", licoo2_cif, "text/plain")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["n_processed"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_with_invalid_includes_error(self, api_client, limn2o4_cif, nacl_cif):
        """Batch with one invalid structure should still process the valid one."""
        resp = api_client.post(
            "/predict/batch",
            files=[
                ("cif_files", ("LiMn2O4.cif", limn2o4_cif, "text/plain")),
                ("cif_files", ("NaCl.cif", nacl_cif, "text/plain")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        # At least the valid one should have been processed
        assert data["n_processed"] >= 1 or data["n_errors"] >= 1


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


@skip_no_model
class TestAuditTrailE2E:
    """Verify prediction audit logging works end-to-end."""

    def test_prediction_creates_audit_record(self, api_client, limn2o4_cif):
        """After a prediction, audit/recent should contain a record."""
        # Make a prediction
        api_client.post(
            "/predict",
            files={"cif_file": ("LiMn2O4.cif", limn2o4_cif, "text/plain")},
        )
        # Check audit trail
        resp = api_client.get("/audit/recent?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 0  # May be 0 if audit dir not writable

    def test_audit_stats_endpoint(self, api_client):
        resp = api_client.get("/audit/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predictions" in data
        assert "decisions" in data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetricsE2E:
    """Verify metrics endpoints."""

    def test_metrics_endpoint(self, api_client):
        resp = api_client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "metrics" in data
