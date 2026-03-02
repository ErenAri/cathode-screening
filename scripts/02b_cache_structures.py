"""Build a structure cache for MACE fine-tuning.

Reads CIF strings from the *raw* parquet, parses them into pymatgen
Structures, and saves (atomic_numbers, cart_positions, lattice_matrix) as
compressed ``.npz`` files — one per material.

Usage::

    python scripts/02b_cache_structures.py --config configs/dataset_mp.yaml

The output directory is ``<processed_dir>/structures/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache structures for MACE fine-tuning")
    ap.add_argument("--config", required=True, help="Dataset config YAML")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-cache even if NPZ already exists",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    ds_name = cfg["dataset"]["name"]

    raw_path = raw_dir / f"raw_{ds_name}.parquet"
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    out_dir = processed_dir / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw parquet not found: {raw_path}\n"
            "Run the data fetch step first."
        )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {processed_path}\n"
            "Run 01_filter_cathodes.py first."
        )

    df_raw = pd.read_parquet(raw_path)
    df_proc = pd.read_parquet(processed_path)

    needed_ids = set(df_proc["material_id"])
    df = df_raw[df_raw["material_id"].isin(needed_ids)][["material_id", "cif"]].copy()

    cached = 0
    skipped = 0
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching structures"):
        mid = row["material_id"]
        out_path = out_dir / f"{mid}.npz"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            s = Structure.from_str(row["cif"], fmt="cif")
            np.savez_compressed(
                out_path,
                atomic_numbers=np.array(
                    [site.specie.Z for site in s], dtype=np.int32
                ),
                positions=s.cart_coords.astype(np.float64),
                cell=np.array(s.lattice.matrix, dtype=np.float64),
            )
            cached += 1
        except Exception as exc:
            print(f"  FAIL {mid}: {exc}")
            failures += 1

    meta = {
        "dataset": ds_name,
        "structure_dir": str(out_dir),
        "cached": cached,
        "skipped": skipped,
        "failures": failures,
        "total_expected": len(needed_ids),
    }
    meta_path = out_dir / "cache_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"Done: {cached} cached, {skipped} skipped, {failures} failed "
        f"→ {out_dir}  (meta: {meta_path})"
    )


if __name__ == "__main__":
    main()
