from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Element, Structure


@dataclass(frozen=True)
class GraphConfig:
    cutoff: float = 8.0
    max_neighbors: int = 12
    rbf_bins: int = 41
    rbf_gamma: float = 10.0
    use_voronoi_features: bool = False
    use_angle_features: bool = False


def element_features(symbol: str) -> np.ndarray:
    """Return raw element features matching training graphs."""
    e = Element(symbol)
    z = float(e.Z)
    x = float(e.X) if e.X is not None else 0.0
    r = float(e.atomic_radius) if e.atomic_radius is not None else 0.0
    m = float(e.atomic_mass) if e.atomic_mass is not None else 0.0
    row = float(e.row) if e.row is not None else 0.0
    group = float(e.group) if e.group is not None else 0.0
    return np.array([z, x, r, m, row, group], dtype=np.float32)


def gaussian_rbf(d: np.ndarray, bins: int, gamma: float) -> np.ndarray:
    centers = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    d_norm = np.clip(d.astype(np.float32), 0.0, 1.0)
    return np.exp(-gamma * (d_norm[..., None] - centers[None, :]) ** 2).astype(np.float32)


def compute_voronoi_features(
    structure: Structure,
    max_retries: int = 2,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """Compute per-atom coordination and per-edge Voronoi weights."""
    vnn = VoronoiNN(tol=0.5, cutoff=10.0)
    n_atoms = len(structure)
    coord_nums = np.zeros(n_atoms, dtype=np.float32)
    voronoi_edges: Dict[Tuple[int, int], float] = {}

    for i in range(n_atoms):
        for attempt in range(max_retries):
            try:
                nn_info = vnn.get_nn_info(structure, i)
                coord_nums[i] = len(nn_info)
                for info in nn_info:
                    j = info["site_index"]
                    weight = info.get("weight", 1.0)
                    voronoi_edges[(i, j)] = weight
                break
            except Exception:
                if attempt == max_retries - 1:
                    coord_nums[i] = 0.0

    return coord_nums, voronoi_edges


def compute_bond_angles(
    structure: Structure,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> np.ndarray:
    """Compute mean cosine bond angle per edge."""
    n_edges = len(edge_src)
    angle_features = np.zeros(n_edges, dtype=np.float32)

    adj: Dict[int, list[tuple[int, int]]] = {}
    for idx, (i, j) in enumerate(zip(edge_src, edge_dst)):
        adj.setdefault(int(j), []).append((int(i), idx))

    cart_coords = structure.cart_coords
    for edge_idx, (i, j) in enumerate(zip(edge_src, edge_dst)):
        vec_ji = cart_coords[int(i)] - cart_coords[int(j)]
        norm_ji = np.linalg.norm(vec_ji)
        if norm_ji < 1e-6:
            continue
        vec_ji = vec_ji / norm_ji

        angles = []
        for k, _ in adj.get(int(j), []):
            if k == int(i):
                continue
            vec_jk = cart_coords[int(k)] - cart_coords[int(j)]
            norm_jk = np.linalg.norm(vec_jk)
            if norm_jk < 1e-6:
                continue
            vec_jk = vec_jk / norm_jk
            cos_angle = np.clip(np.dot(vec_ji, vec_jk), -1.0, 1.0)
            angles.append(cos_angle)

        if angles:
            angle_features[edge_idx] = float(np.mean(angles))

    return angle_features


class GraphBuilder:
    """Build CGCNN graph tensors from a pymatgen Structure."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()

    def build(self, structure: Structure) -> Dict[str, np.ndarray]:
        cfg = self.config
        n = len(structure)

        node_feats = np.stack([element_features(str(site.specie)) for site in structure], axis=0).astype(
            np.float32
        )

        if cfg.use_voronoi_features:
            coord_nums, voronoi_weights = compute_voronoi_features(structure)
            node_feats = np.concatenate([node_feats, coord_nums.reshape(-1, 1)], axis=1)
        else:
            voronoi_weights = {}

        edge_src = []
        edge_dst = []
        edge_dist = []

        neighbors = structure.get_all_neighbors(cfg.cutoff, include_index=True)
        for i, lst in enumerate(neighbors):
            lst_sorted = sorted(lst, key=lambda t: float(t[1]))[: cfg.max_neighbors]
            for site, dist, j, *_ in lst_sorted:
                edge_src.append(int(i))
                edge_dst.append(int(j))
                edge_dist.append(float(dist))

        if not edge_src:
            raise ValueError("No edges found for structure; check cutoff/max_neighbors.")

        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_dist = np.array(edge_dist, dtype=np.float32)

        d_norm = edge_dist / max(cfg.cutoff, 1e-6)
        e_attr = gaussian_rbf(d_norm, bins=cfg.rbf_bins, gamma=cfg.rbf_gamma)

        if cfg.use_voronoi_features:
            v_weights = np.array(
                [voronoi_weights.get((int(i), int(j)), 0.0) for i, j in zip(edge_src, edge_dst)],
                dtype=np.float32,
            ).reshape(-1, 1)
            e_attr = np.concatenate([e_attr, v_weights], axis=1)

        if cfg.use_angle_features:
            angle_features = compute_bond_angles(structure, edge_src, edge_dst).reshape(-1, 1)
            e_attr = np.concatenate([e_attr, angle_features], axis=1)

        return {
            "x": node_feats,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "e_attr": e_attr.astype(np.float32),
        }


def structure_to_graph(structure: Structure, config: GraphConfig | None = None) -> Dict[str, np.ndarray]:
    """Convenience wrapper for GraphBuilder."""
    return GraphBuilder(config=config).build(structure)
