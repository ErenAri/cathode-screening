"""Shared pytest fixtures for cathode-screening tests."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic graph helpers
# ---------------------------------------------------------------------------

def _make_graph(n_atoms: int = 8, node_dim: int = 6, edge_rbf: int = 41, seed: int = 0):
    """Create a synthetic crystal graph dict matching CGCNN expectations."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_atoms, node_dim)).astype(np.float32)
    # Build a simple ring + cross edges
    src, dst = [], []
    for i in range(n_atoms):
        for j in range(i + 1, min(i + 4, n_atoms)):
            src.extend([i, j])
            dst.extend([j, i])
    n_edges = len(src)
    e_attr = rng.standard_normal((n_edges, edge_rbf)).astype(np.float32)
    return {
        "x": x,
        "edge_src": np.array(src, dtype=np.int64),
        "edge_dst": np.array(dst, dtype=np.int64),
        "e_attr": e_attr,
    }


@pytest.fixture()
def simple_graph():
    """Single synthetic graph dict (8 atoms, 6-dim features, 41 RBF bins)."""
    return _make_graph()


@pytest.fixture()
def simple_graph_pair():
    """Two distinct synthetic graphs for batch tests."""
    return [_make_graph(n_atoms=6, seed=1), _make_graph(n_atoms=10, seed=2)]


# ---------------------------------------------------------------------------
# Normalizer fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def normalizer_state():
    """Return a normalizer state dict."""
    return {"mean": 0.12, "std": 0.08}


# ---------------------------------------------------------------------------
# Split leakage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def splits():
    """Synthetic split ids with no overlap and realistic fractions."""
    ids = [f"mat-{i:03d}" for i in range(100)]
    return {
        "train": ids[:70],
        "val": ids[70:85],
        "test": ids[85:100],
    }


@pytest.fixture()
def train_compositions():
    """Training composition labels for novelty checks."""
    return [
        "LiCoO2",
        "LiFePO4",
        "LiMn2O4",
        "LiNiO2",
        "LiVPO4",
        "LiCrO2",
        "Li2MnO3",
        "LiNi0.5Mn0.5O2",
        "LiCo0.5Mn0.5O2",
        "LiNi0.8Co0.1Mn0.1O2",
    ] * 7


@pytest.fixture()
def test_compositions():
    """Test compositions with >=10% novelty vs train."""
    return [
        "LiCoO2",
        "LiFePO4",
        "LiMn2O4",
        "LiNiO2",
        "LiVPO4",
        "LiCrO2",
        "Li2MnO3",
        "LiNi0.5Mn0.5O2",
        "LiCo0.5Mn0.5O2",
        "LiNi0.8Co0.1Mn0.1O2",
        "LiTi2(PO4)3",
        "Li3V2(PO4)3",
        "Li2FeSiO4",
        "LiMnPO4",
        "Li2CoSiO4",
    ]


@pytest.fixture()
def cluster_assignments(splits):
    """Cluster labels where test clusters are disjoint from train clusters."""
    assignments = {}
    for idx, material_id in enumerate(splits["train"]):
        assignments[material_id] = idx % 8
    for idx, material_id in enumerate(splits["val"]):
        assignments[material_id] = 20 + (idx % 2)
    for idx, material_id in enumerate(splits["test"]):
        assignments[material_id] = 30 + (idx % 3)
    return assignments


@pytest.fixture()
def soap_novelty_metrics():
    return {"mean_min_distance": 0.34}


@pytest.fixture()
def composition_novelty_metrics():
    return {"mean_min_distance": 0.30}


@pytest.fixture()
def labels(splits):
    """Synthetic ehull labels with similar train/test stable fractions."""
    out = {}
    train_ids = splits["train"]
    test_ids = splits["test"]
    val_ids = splits["val"]

    # train: 30 stable of 70 (~43%)
    for i, material_id in enumerate(train_ids):
        out[material_id] = 0.03 if i < 30 else 0.18
    # val: mixed but unused by tested assertions
    for i, material_id in enumerate(val_ids):
        out[material_id] = 0.04 if i < 6 else 0.16
    # test: 6 stable of 15 (40%)
    for i, material_id in enumerate(test_ids):
        out[material_id] = 0.02 if i < 6 else 0.17

    return out
