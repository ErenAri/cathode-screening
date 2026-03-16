"""
Locust load testing for CathodeScreen API.

Simulates realistic traffic patterns for capacity planning:
  - Mixed single and batch predictions
  - Health check polling
  - Database queries
  - Discovery campaign interactions

Usage:
    # Basic run (web UI)
    locust -f tests/load/locustfile.py --host http://localhost:8080

    # Headless run (CI/CD)
    locust -f tests/load/locustfile.py --host http://localhost:8080 \
        --headless -u 50 -r 5 -t 5m \
        --csv results/load_test

    # Target: 1000 predictions/minute with <10s p95 latency

Environment:
    CATHODE_LOADTEST_API_KEY   - API key for authenticated endpoints
    CATHODE_LOADTEST_CIF_DIR   - Directory with CIF fixtures (default: tests/integration/fixtures)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from locust import HttpUser, between, task, tag


# Load CIF fixtures for realistic predictions
_FIXTURES_DIR = Path(
    os.getenv("CATHODE_LOADTEST_CIF_DIR", "tests/integration/fixtures")
)
_CIF_FILES = list(_FIXTURES_DIR.glob("*.cif")) if _FIXTURES_DIR.exists() else []
_API_KEY = os.getenv("CATHODE_LOADTEST_API_KEY", "")


def _auth_headers() -> dict:
    if _API_KEY:
        return {"X-API-Key": _API_KEY}
    return {}


def _random_cif() -> tuple[str, bytes]:
    """Pick a random CIF file from fixtures."""
    if not _CIF_FILES:
        # Fallback: minimal LiCoO2 CIF
        content = b"""data_LiCoO2
_cell_length_a 2.8166
_cell_length_b 2.8166
_cell_length_c 14.0536
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'R -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0.00000 0.00000 0.50000
Co1 Co 0.00000 0.00000 0.00000
O1  O  0.00000 0.00000 0.23890
"""
        return "LiCoO2.cif", content

    f = random.choice(_CIF_FILES)
    return f.name, f.read_bytes()


class PredictionUser(HttpUser):
    """Simulates a materials scientist using the prediction API."""

    wait_time = between(1, 5)

    @tag("predict")
    @task(10)
    def predict_single(self):
        """Upload a single CIF and get prediction."""
        name, content = _random_cif()
        self.client.post(
            "/predict",
            files={"cif_file": (name, content, "text/plain")},
            headers=_auth_headers(),
            name="/predict",
        )

    @tag("predict")
    @task(3)
    def predict_single_v1(self):
        """Upload via v1 versioned endpoint."""
        name, content = _random_cif()
        self.client.post(
            "/v1/predict",
            files={"cif_file": (name, content, "text/plain")},
            headers=_auth_headers(),
            name="/v1/predict",
        )

    @tag("predict", "batch")
    @task(2)
    def predict_batch(self):
        """Batch predict 3-5 structures."""
        n = random.randint(3, 5)
        files = []
        for i in range(n):
            name, content = _random_cif()
            files.append(("cif_files", (f"{i}_{name}", content, "text/plain")))

        self.client.post(
            "/predict/batch",
            files=files,
            headers=_auth_headers(),
            name="/predict/batch",
        )

    @tag("health")
    @task(5)
    def health_check(self):
        """Poll health endpoint."""
        self.client.get("/", name="/")

    @tag("health")
    @task(3)
    def ready_check(self):
        """Poll readiness endpoint."""
        self.client.get("/ready", name="/ready")

    @tag("info")
    @task(2)
    def model_info(self):
        """Get model information."""
        self.client.get(
            "/model/info",
            headers=_auth_headers(),
            name="/model/info",
        )

    @tag("metrics")
    @task(1)
    def get_metrics(self):
        """Fetch metrics."""
        self.client.get(
            "/metrics",
            headers=_auth_headers(),
            name="/metrics",
        )


class MonitoringUser(HttpUser):
    """Simulates a monitoring dashboard polling metrics and audit."""

    wait_time = between(5, 15)

    @tag("metrics")
    @task(5)
    def poll_metrics(self):
        self.client.get(
            "/metrics",
            headers=_auth_headers(),
            name="/metrics",
        )

    @tag("audit")
    @task(3)
    def poll_audit_recent(self):
        self.client.get(
            "/audit/recent?n=10",
            headers=_auth_headers(),
            name="/audit/recent",
        )

    @tag("audit")
    @task(2)
    def poll_audit_stats(self):
        self.client.get(
            "/audit/stats",
            headers=_auth_headers(),
            name="/audit/stats",
        )

    @tag("health")
    @task(10)
    def ready_poll(self):
        self.client.get("/ready", name="/ready [monitor]")
