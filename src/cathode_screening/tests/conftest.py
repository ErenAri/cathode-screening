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
