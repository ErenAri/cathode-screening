"""Integration test fixtures for CathodeScreen API."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure source is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to integration test fixtures."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def limn2o4_cif():
    """LiMn2O4 spinel CIF content (known stable cathode)."""
    return (FIXTURES_DIR / "LiMn2O4_spinel.cif").read_bytes()


@pytest.fixture(scope="session")
def licoo2_cif():
    """LiCoO2 layered CIF content (known stable cathode)."""
    return (FIXTURES_DIR / "LiCoO2_layered.cif").read_bytes()


@pytest.fixture(scope="session")
def lifepo4_cif():
    """LiFePO4 olivine CIF content (known stable cathode)."""
    return (FIXTURES_DIR / "LiFePO4_olivine.cif").read_bytes()


@pytest.fixture(scope="session")
def nacl_cif():
    """NaCl CIF content (invalid cathode - no Li, no TM)."""
    return (FIXTURES_DIR / "NaCl_invalid_cathode.cif").read_bytes()


@pytest.fixture(scope="session")
def api_client():
    """FastAPI test client with auth disabled for integration tests."""
    os.environ["CATHODE_ENV"] = "development"
    os.environ["CATHODE_AUTH_ENABLED"] = "false"
    os.environ["CATHODE_REQUIRE_CATHODE_COMPOSITION"] = "true"
    os.environ["CATHODE_FORCE_HTTPS"] = "false"
    os.environ["CATHODE_STRICT_STARTUP"] = "false"
    os.environ["CATHODE_REQUIRE_MANIFEST_SIGNATURE"] = "false"

    from fastapi.testclient import TestClient
    from web.api.main import app

    return TestClient(app)
