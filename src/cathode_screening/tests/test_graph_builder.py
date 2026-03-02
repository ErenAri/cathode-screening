"""
Unit tests for graph_builder module.

Tests element_features, gaussian_rbf, GraphConfig defaults, and GraphBuilder
output shapes. Uses a minimal pymatgen Structure (LiCoO2 rocksalt cell).

Run with: pytest src/cathode_screening/tests/test_graph_builder.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from cathode_screening.datasets.featurize.graph_builder import (
    GraphConfig,
    element_features,
    gaussian_rbf,
)

# Defer pymatgen-heavy imports; skip if unavailable
try:
    from pymatgen.core import Lattice, Structure

    from cathode_screening.datasets.featurize.graph_builder import (
        GraphBuilder,
        structure_to_graph,
    )

    HAS_PYMATGEN = True
except Exception:  # pragma: no cover
    HAS_PYMATGEN = False


# ---------------------------------------------------------------------------
# element_features
# ---------------------------------------------------------------------------


class TestElementFeatures:
    def test_returns_6_floats(self):
        feat = element_features("Li")
        assert feat.shape == (6,)
        assert feat.dtype == np.float32

    def test_z_is_first(self):
        assert element_features("Li")[0] == 3.0
        assert element_features("O")[0] == 8.0
        assert element_features("Co")[0] == 27.0

    def test_different_elements_differ(self):
        li = element_features("Li")
        co = element_features("Co")
        assert not np.allclose(li, co)


# ---------------------------------------------------------------------------
# gaussian_rbf
# ---------------------------------------------------------------------------


class TestGaussianRBF:
    def test_output_shape(self):
        d = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = gaussian_rbf(d, bins=41, gamma=10.0)
        assert out.shape == (3, 41)
        assert out.dtype == np.float32

    def test_values_between_0_and_1(self):
        d = np.linspace(0, 1, 50, dtype=np.float32)
        out = gaussian_rbf(d, bins=20, gamma=5.0)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_peak_near_center(self):
        # Distance 0.5 should peak near center bin
        d = np.array([0.5], dtype=np.float32)
        out = gaussian_rbf(d, bins=11, gamma=100.0)
        peak_bin = np.argmax(out[0])
        assert peak_bin == 5  # center of 11 bins (0-indexed)

    def test_clipping_beyond_1(self):
        d = np.array([1.5, 2.0], dtype=np.float32)
        out = gaussian_rbf(d, bins=5, gamma=10.0)
        # Clipped to 1.0, so peak should be at last bin
        assert np.argmax(out[0]) == 4


# ---------------------------------------------------------------------------
# GraphConfig
# ---------------------------------------------------------------------------


class TestGraphConfig:
    def test_defaults(self):
        cfg = GraphConfig()
        assert cfg.cutoff == 8.0
        assert cfg.max_neighbors == 12
        assert cfg.rbf_bins == 41
        assert cfg.use_voronoi_features is False
        assert cfg.use_angle_features is False

    def test_frozen(self):
        cfg = GraphConfig()
        with pytest.raises(AttributeError):
            cfg.cutoff = 10.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GraphBuilder (requires pymatgen)
# ---------------------------------------------------------------------------


def _make_licoo2() -> "Structure":
    """Minimal LiCoO2-like rocksalt structure for testing."""
    lattice = Lattice.cubic(4.0)
    return Structure(
        lattice,
        ["Li", "Co", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5]],
    )


@pytest.mark.skipif(not HAS_PYMATGEN, reason="pymatgen not available")
class TestGraphBuilder:
    def test_output_keys(self):
        g = GraphBuilder().build(_make_licoo2())
        assert set(g.keys()) == {"x", "edge_src", "edge_dst", "e_attr"}

    def test_node_feature_shape(self):
        g = GraphBuilder().build(_make_licoo2())
        assert g["x"].shape == (4, 6)  # 4 atoms, 6 features
        assert g["x"].dtype == np.float32

    def test_edge_index_consistent(self):
        g = GraphBuilder().build(_make_licoo2())
        n_edges = len(g["edge_src"])
        assert len(g["edge_dst"]) == n_edges
        assert g["e_attr"].shape[0] == n_edges

    def test_edge_rbf_dim(self):
        cfg = GraphConfig(rbf_bins=41)
        g = GraphBuilder(cfg).build(_make_licoo2())
        assert g["e_attr"].shape[1] == 41

    def test_custom_rbf_bins(self):
        cfg = GraphConfig(rbf_bins=20)
        g = GraphBuilder(cfg).build(_make_licoo2())
        assert g["e_attr"].shape[1] == 20

    def test_voronoi_features_adds_columns(self):
        cfg = GraphConfig(use_voronoi_features=True)
        g = GraphBuilder(cfg).build(_make_licoo2())
        # node features: 6 (element) + 1 (coord_num)
        assert g["x"].shape[1] == 7
        # edge features: 41 (rbf) + 1 (voronoi weight)
        assert g["e_attr"].shape[1] == 42

    def test_angle_features_adds_column(self):
        cfg = GraphConfig(use_angle_features=True)
        g = GraphBuilder(cfg).build(_make_licoo2())
        # edge features: 41 (rbf) + 1 (angle)
        assert g["e_attr"].shape[1] == 42

    def test_edge_indices_in_range(self):
        g = GraphBuilder().build(_make_licoo2())
        n_atoms = g["x"].shape[0]
        assert np.all(g["edge_src"] >= 0)
        assert np.all(g["edge_src"] < n_atoms)
        assert np.all(g["edge_dst"] >= 0)
        assert np.all(g["edge_dst"] < n_atoms)

    def test_max_neighbors_limit(self):
        cfg = GraphConfig(max_neighbors=2)
        g = GraphBuilder(cfg).build(_make_licoo2())
        n_atoms = g["x"].shape[0]
        # Each atom should have at most 2 outgoing edges
        for i in range(n_atoms):
            assert np.sum(g["edge_src"] == i) <= 2

    def test_convenience_wrapper_matches(self):
        s = _make_licoo2()
        g1 = GraphBuilder().build(s)
        g2 = structure_to_graph(s)
        for key in g1:
            np.testing.assert_array_equal(g1[key], g2[key])
