"""
API contract tests to verify response schemas and error formats.

These tests validate that the API conforms to its documented contract,
regardless of model availability. They focus on schema compliance,
error format consistency, and endpoint behavior.

Run with:
    pytest tests/integration/test_api_contract.py -v
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


class TestErrorResponseFormat:
    """All error responses must follow the standard error format."""

    def test_400_error_has_structured_detail(self, api_client):
        """400 errors should return structured error detail."""
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("empty.cif", b"", "text/plain")},
        )
        assert resp.status_code == 400
        detail = resp.json().get("detail")
        assert detail is not None
        if isinstance(detail, dict):
            assert "error" in detail
            assert "message" in detail

    def test_413_for_oversized_payload(self, api_client):
        resp = api_client.post(
            "/predict",
            files={"cif_file": ("big.cif", b"x" * 3_000_000, "text/plain")},
        )
        assert resp.status_code == 413

    def test_404_for_unknown_campaign(self, api_client):
        resp = api_client.get("/discovery/campaigns/nonexistent_campaign_xyz")
        assert resp.status_code == 404


class TestResponseSchemaCompliance:
    """Verify response schemas match documented API contract."""

    def test_root_schema(self, api_client):
        resp = api_client.get("/")
        data = resp.json()
        assert set(data.keys()) >= {"status", "service", "version"}

    def test_model_info_schema(self, api_client):
        resp = api_client.get("/model/info")
        data = resp.json()
        required_fields = {
            "model_name",
            "model_type",
            "ensemble_size",
            "training_data",
            "daf_at_10",
            "version",
        }
        assert required_fields.issubset(set(data.keys()))
        assert isinstance(data["ensemble_size"], int)
        assert isinstance(data["daf_at_10"], (int, float))

    def test_metrics_schema(self, api_client):
        resp = api_client.get("/metrics")
        data = resp.json()
        assert data["success"] is True
        metrics = data["metrics"]
        assert "request_count" in metrics or "uptime_s" in metrics

    def test_discovery_strategies_schema(self, api_client):
        resp = api_client.get("/discovery/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "strategies" in data
        assert "default" in data
        for strategy in data["strategies"]:
            assert "id" in strategy
            assert "label" in strategy
            assert "description" in strategy


class TestAPIVersioning:
    """Verify API version is exposed consistently."""

    def test_root_includes_version(self, api_client):
        resp = api_client.get("/")
        assert "version" in resp.json()

    def test_model_info_includes_version(self, api_client):
        resp = api_client.get("/model/info")
        assert "version" in resp.json()


class TestRateLimiting:
    """Verify rate limiting behavior."""

    def test_rate_limit_header_not_error_on_single_request(self, api_client):
        """A single request should not be rate-limited."""
        resp = api_client.get("/")
        assert resp.status_code != 429
