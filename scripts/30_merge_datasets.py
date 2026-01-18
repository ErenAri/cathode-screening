"""
Phase 7C: Merge and deduplicate OQMD + MP datasets.

Combines:
- Existing MP Li-cathode data (11K)
- New OQMD data (~15K)
- MP 2024 updates (~5K)

Deduplicates using pymatgen StructureMatcher to identify identical structures.
Harmonizes E_hull values across databases.

Output:
    data/processed/merged_dataset/
        structures.pkl
        energies.npy
        metadata.json
        train_idx.npy, val_idx.npy, test_idx.npy

Usage:
    python scripts/30_merge_datasets.py
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("Warning: pymatgen not installed")

# Input paths
EXISTING_DATA_PATH = Path("data/processed/chgnet_soap_loco")
OQMD_DATA_PATH = Path("data/external/oqmd")
MP_2024_DATA_PATH = Path("data/external/mp_2024")

# Output path
OUTPUT_DIR = Path("data/processed/merged_dataset")


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_data() -> Tuple[List[Structure], np.ndarray, List[Dict]]:
    """Load existing MP Li-cathode training data."""
    print("Loading existing MP data...")
    
    structures_path = EXISTING_DATA_PATH / "structures.pkl"
    energies_path = EXISTING_DATA_PATH / "energies.npy"
    metadata_path = EXISTING_DATA_PATH / "metadata.json"
    
    if not all(p.exists() for p in [structures_path, energies_path, metadata_path]):
        print("Existing data not found. Skipping.")
        return [], np.array([]), []
    
    with open(structures_path, "rb") as f:
        structures = pickle.load(f)
    
    energies = np.load(energies_path)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"  Loaded {len(structures)} structures from existing MP data")
    return structures, energies, metadata


def load_oqmd_data() -> Tuple[List[Structure], List[float], List[Dict]]:
    """Load OQMD data."""
    print("Loading OQMD data...")
    
    parquet_path = OQMD_DATA_PATH / "oqmd_li_cathodes.parquet"
    structures_path = OQMD_DATA_PATH / "structures.pkl"
    
    if not parquet_path.exists():
        print("  OQMD data not found. Run 28_download_oqmd.py first.")
        return [], [], []
    
    df = pd.read_parquet(parquet_path)
    
    structures = []
    energies = []
    metadata = []
    
    if structures_path.exists():
        with open(structures_path, "rb") as f:
            struct_data = pickle.load(f)
        
        # Map oqmd_id -> structure
        struct_map = {s["oqmd_id"]: s["structure"] for s in struct_data}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing OQMD"):
            oqmd_id = row["oqmd_id"]
            if oqmd_id in struct_map:
                try:
                    # Convert to pymatgen Structure if needed
                    struct = struct_map[oqmd_id]
                    if isinstance(struct, dict):
                        struct = Structure.from_dict(struct)
                    
                    structures.append(struct)
                    energies.append(row["stability_oqmd"])  # E_hull equivalent
                    metadata.append({
                        "material_id": f"oqmd-{oqmd_id}",
                        "source": "oqmd",
                        "formula": row["composition"],
                    })
                except Exception:
                    continue
    
    print(f"  Loaded {len(structures)} structures from OQMD")
    return structures, energies, metadata


def load_mp_2024_data() -> Tuple[List[Structure], List[float], List[Dict]]:
    """Load MP 2024 update data."""
    print("Loading MP 2024 data...")
    
    parquet_path = MP_2024_DATA_PATH / "mp_li_cathodes_2024.parquet"
    structures_path = MP_2024_DATA_PATH / "structures.pkl"
    
    if not parquet_path.exists():
        print("  MP 2024 data not found. Run 29_update_mp_data.py first.")
        return [], [], []
    
    df = pd.read_parquet(parquet_path)
    
    structures = []
    energies = []
    metadata = []
    
    if structures_path.exists():
        with open(structures_path, "rb") as f:
            struct_data = pickle.load(f)
        
        # Map material_id -> structure
        struct_map = {s["material_id"]: s["structure"] for s in struct_data}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MP 2024"):
            material_id = row["material_id"]
            if material_id in struct_map:
                try:
                    struct = struct_map[material_id]
                    if isinstance(struct, dict):
                        struct = Structure.from_dict(struct)
                    
                    structures.append(struct)
                    energies.append(row["e_hull"])
                    metadata.append({
                        "material_id": material_id,
                        "source": "mp_2024",
                        "formula": row["formula"],
                    })
                except Exception:
                    continue
    
    print(f"  Loaded {len(structures)} structures from MP 2024")
    return structures, energies, metadata


def deduplicate_structures(
    structures: List[Structure],
    energies: List[float],
    metadata: List[Dict],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> Tuple[List[Structure], np.ndarray, List[Dict]]:
    """
    Remove duplicate structures using pymatgen StructureMatcher.
    
    For duplicates, keep the one with lower E_hull.
    """
    if not HAS_PYMATGEN:
        print("Warning: pymatgen not available. Skipping deduplication.")
        return structures, np.array(energies), metadata
    
    print(f"\nDeduplicating {len(structures)} structures...")
    
    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,
        scale=True,
    )
    
    unique_structures = []
    unique_energies = []
    unique_metadata = []
    
    for i, (struct, energy, meta) in tqdm(
        enumerate(zip(structures, energies, metadata)),
        total=len(structures),
        desc="Deduplicating",
    ):
        is_duplicate = False
        
        # Check against existing unique structures
        for j, unique_struct in enumerate(unique_structures):
            try:
                if matcher.fit(struct, unique_struct):
                    is_duplicate = True
                    # Keep the one with lower energy
                    if energy < unique_energies[j]:
                        unique_structures[j] = struct
                        unique_energies[j] = energy
                        unique_metadata[j] = meta
                    break
            except Exception:
                continue
        
        if not is_duplicate:
            unique_structures.append(struct)
            unique_energies.append(energy)
            unique_metadata.append(meta)
    
    print(f"  Deduplicated: {len(structures)} -> {len(unique_structures)}")
    return unique_structures, np.array(unique_energies), unique_metadata


def create_train_val_test_split(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random train/val/test split."""
    np.random.seed(seed)
    
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx


def save_merged_dataset(
    structures: List[Structure],
    energies: np.ndarray,
    metadata: List[Dict],
) -> None:
    """Save merged dataset."""
    # Save structures
    structures_path = OUTPUT_DIR / "structures.pkl"
    with open(structures_path, "wb") as f:
        pickle.dump(structures, f)
    print(f"Saved {len(structures)} structures to {structures_path}")
    
    # Save energies
    energies_path = OUTPUT_DIR / "energies.npy"
    np.save(energies_path, energies)
    print(f"Saved energies to {energies_path}")
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")
    
    # Create and save splits
    train_idx, val_idx, test_idx = create_train_val_test_split(len(structures))
    
    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_idx.npy", test_idx)
    
    print(f"Saved splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Summary
    summary = {
        "total_samples": len(structures),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "e_hull_range": [float(energies.min()), float(energies.max())],
        "sources": {
            "mp": sum(1 for m in metadata if m["source"] == "mp" or "mp-" in m["material_id"]),
            "oqmd": sum(1 for m in metadata if m["source"] == "oqmd"),
            "mp_2024": sum(1 for m in metadata if m["source"] == "mp_2024"),
        },
    }
    
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("Phase 7C: Merge and Deduplicate Datasets")
    print("=" * 60)
    
    setup_directories()
    
    # Load all data sources
    existing_structs, existing_energies, existing_meta = load_existing_data()
    oqmd_structs, oqmd_energies, oqmd_meta = load_oqmd_data()
    mp2024_structs, mp2024_energies, mp2024_meta = load_mp_2024_data()
    
    # Combine all data
    all_structures = list(existing_structs) + list(oqmd_structs) + list(mp2024_structs)
    all_energies = list(existing_energies) + list(oqmd_energies) + list(mp2024_energies)
    
    # Update existing metadata with source
    for m in existing_meta:
        m["source"] = "mp"
    all_metadata = existing_meta + oqmd_meta + mp2024_meta
    
    print(f"\nCombined dataset: {len(all_structures)} structures")
    
    if len(all_structures) == 0:
        print("No data to merge. Run download scripts first.")
        return
    
    # Deduplicate
    unique_structs, unique_energies, unique_meta = deduplicate_structures(
        all_structures, all_energies, all_metadata
    )
    
    # Save
    save_merged_dataset(unique_structs, unique_energies, unique_meta)
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  Original: {len(all_structures)}")
    print(f"  After dedup: {len(unique_structs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
