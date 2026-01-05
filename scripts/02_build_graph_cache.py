import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import VoronoiNN


TM_SET_DEFAULT = {"Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"}


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def element_features(symbol: str) -> np.ndarray:
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


def compute_voronoi_features(structure: Structure, max_retries: int = 2):
    """
    Compute Voronoi-based features: coordination number and solid angles.
    Returns per-atom coordination numbers and per-edge solid angle weights.
    """
    vnn = VoronoiNN(tol=0.5, cutoff=10.0)
    n_atoms = len(structure)
    coord_nums = np.zeros(n_atoms, dtype=np.float32)
    voronoi_edges = {}  # (i, j) -> solid_angle_weight
    
    for i in range(n_atoms):
        for attempt in range(max_retries):
            try:
                nn_info = vnn.get_nn_info(structure, i)
                coord_nums[i] = len(nn_info)
                for info in nn_info:
                    j = info['site_index']
                    weight = info.get('weight', 1.0)  # Solid angle / max solid angle
                    voronoi_edges[(i, j)] = weight
                break
            except Exception:
                if attempt == max_retries - 1:
                    coord_nums[i] = 0
    
    return coord_nums, voronoi_edges


def compute_bond_angles(structure: Structure, edge_src: np.ndarray, edge_dst: np.ndarray):
    """
    Compute bond angles for 3-body interactions.
    For each edge i->j, compute angles with other edges from j.
    Returns mean angle feature per edge.
    """
    n_edges = len(edge_src)
    angle_features = np.zeros(n_edges, dtype=np.float32)
    
    # Build adjacency for quick lookup
    adj = {}
    for idx, (i, j) in enumerate(zip(edge_src, edge_dst)):
        if j not in adj:
            adj[j] = []
        adj[j].append((i, idx))
    
    cart_coords = structure.cart_coords
    
    for edge_idx, (i, j) in enumerate(zip(edge_src, edge_dst)):
        vec_ji = cart_coords[i] - cart_coords[j]
        norm_ji = np.linalg.norm(vec_ji)
        if norm_ji < 1e-6:
            continue
        vec_ji = vec_ji / norm_ji
        
        # Get other edges from j
        angles = []
        for k, _ in adj.get(j, []):
            if k == i:
                continue
            vec_jk = cart_coords[k] - cart_coords[j]
            norm_jk = np.linalg.norm(vec_jk)
            if norm_jk < 1e-6:
                continue
            vec_jk = vec_jk / norm_jk
            cos_angle = np.clip(np.dot(vec_ji, vec_jk), -1.0, 1.0)
            angles.append(cos_angle)
        
        if angles:
            angle_features[edge_idx] = np.mean(angles)
    
    return angle_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    raw_path = Path(cfg["dataset"]["raw_dir"]) / f"raw_{cfg['dataset']['name']}.parquet"
    interim_dir = Path(cfg["dataset"]["interim_dir"])
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    interim_graph_dir = interim_dir / "graphs"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_graph_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_path)

    req = set(cfg["filters"]["require_elements"])
    tm_any = set(cfg["filters"]["require_any_transition_metals"])
    excl = set(cfg["filters"].get("exclude_elements", []))
    min_atoms = int(cfg["filters"]["min_atoms"])
    max_atoms = int(cfg["filters"]["max_atoms"])
    allow_partial = bool(cfg["filters"]["allow_partial_occupancy"])

    def keep_row(row) -> bool:
        elems_raw = row["elements"]
        elems = set(elems_raw) if elems_raw is not None and len(elems_raw) > 0 else set()
        if not req.issubset(elems):
            return False
        if elems.intersection(excl):
            return False
        if not elems.intersection(tm_any):
            return False
        if row.get("cif") is None:
            return False
        if row.get("energy_above_hull") is None or row.get("formation_energy_per_atom") is None:
            return False
        try:
            s = Structure.from_str(row["cif"], fmt="cif")
        except Exception:
            return False
        n = int(len(s))
        if n < min_atoms or n > max_atoms:
            return False
        if not allow_partial:
            for site in s.sites:
                if len(site.species) != 1 or list(site.species.values())[0] != 1:
                    return False
        return True

    kept = []
    for _, r in df.iterrows():
        if keep_row(r):
            kept.append(r)

    if not kept:
        raise RuntimeError("No rows passed filters. Check filters or fetch step.")

    dfk = pd.DataFrame(kept).reset_index(drop=True)

    cutoff = float(cfg["graph"]["neighbor_cutoff_angstrom"])
    max_neighbors = int(cfg["graph"]["max_neighbors"])
    bins = int(cfg["graph"]["rbf_bins"])
    gamma = float(cfg["graph"]["rbf_gamma"])
    use_voronoi = cfg["graph"].get("use_voronoi_features", True)
    use_angles = cfg["graph"].get("use_angle_features", True)

    out_rows = []
    failures = 0
    total = len(dfk)

    for idx, r in dfk.iterrows():
        mid = r["material_id"]
        if idx % 500 == 0:
            print(f"Processing {idx}/{total} ({100*idx/total:.1f}%)...")
        try:
            s = Structure.from_str(r["cif"], fmt="cif")
        except Exception:
            failures += 1
            continue

        n = len(s)
        frac = s.frac_coords.astype(np.float32)
        species = [str(sp) for sp in s.species]
        x = np.stack([element_features(sym) for sym in species], axis=0).astype(np.float32)

        # Compute Voronoi features (coordination number per atom)
        if use_voronoi:
            try:
                coord_nums, voronoi_weights = compute_voronoi_features(s)
                # Add coordination number to node features
                x = np.concatenate([x, coord_nums.reshape(-1, 1)], axis=1)
            except Exception:
                coord_nums = np.zeros(n, dtype=np.float32)
                voronoi_weights = {}
                x = np.concatenate([x, coord_nums.reshape(-1, 1)], axis=1)

        edge_src = []
        edge_dst = []
        edge_dist = []

        neigh = s.get_all_neighbors(cutoff, include_index=True)
        for i, lst in enumerate(neigh):
            lst_sorted = sorted(lst, key=lambda t: float(t[1]))[:max_neighbors]
            for site, dist, j, *_ in lst_sorted:
                edge_src.append(i)
                edge_dst.append(int(j))
                edge_dist.append(float(dist))

        if not edge_src:
            failures += 1
            continue

        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_dist = np.array(edge_dist, dtype=np.float32)

        d_norm = edge_dist / max(cutoff, 1e-6)
        e_attr = gaussian_rbf(d_norm, bins=bins, gamma=gamma)
        
        # Add Voronoi solid angle weights to edge features
        if use_voronoi:
            voronoi_weights_arr = np.array(
                [voronoi_weights.get((int(i), int(j)), 0.0) for i, j in zip(edge_src, edge_dst)],
                dtype=np.float32
            ).reshape(-1, 1)
            e_attr = np.concatenate([e_attr, voronoi_weights_arr], axis=1)
        
        # Add bond angle features (3-body)
        if use_angles:
            try:
                angle_features = compute_bond_angles(s, edge_src, edge_dst).reshape(-1, 1)
                e_attr = np.concatenate([e_attr, angle_features], axis=1)
            except Exception:
                # Fallback: add zeros
                e_attr = np.concatenate([e_attr, np.zeros((len(edge_src), 1), dtype=np.float32)], axis=1)

        gpath = interim_graph_dir / f"{mid}.npz"
        np.savez_compressed(
            gpath,
            x=x,
            frac=frac,
            edge_src=edge_src,
            edge_dst=edge_dst,
            e_attr=e_attr,
        )

        out_rows.append(
            {
                "material_id": mid,
                "formula_pretty": r["formula_pretty"],
                "elements": r["elements"],
                "energy_above_hull": float(r["energy_above_hull"]),
                "formation_energy_per_atom": float(r["formation_energy_per_atom"]),
                "band_gap": None if pd.isna(r.get("band_gap")) else float(r.get("band_gap")),
                "volume": None if pd.isna(r.get("volume")) else float(r.get("volume")),
                "graph_npz": str(gpath),
                "n_atoms": int(n),
            }
        )

    processed_path = processed_dir / f"processed_{cfg['dataset']['name']}.parquet"
    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(processed_path, index=False)

    meta = {
        "dataset": cfg["dataset"]["name"],
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "graph_dir": str(interim_graph_dir),
        "rows": int(len(out_df)),
        "graph_failures": int(failures),
        "filters": cfg["filters"],
        "graph": cfg["graph"],
    }
    (processed_dir / f"processed_{cfg['dataset']['name']}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_df)} rows -> {processed_path} (graph failures: {failures})")


if __name__ == "__main__":
    main()
