#!/usr/bin/env python
"""
Generate SOAP-LOCO splits from raw Materials Project data.

Reads CIF strings from parquet, computes SOAP descriptors, and creates
Leave-One-Cluster-Out splits based on structural similarity.
"""

import argparse
import json
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure

from cathode_screening.datasets.splits.soap_loco import (
    check_dscribe_available,
    compute_soap_descriptors,
    soap_cluster,
    compute_split_novelty,
)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_structures_from_parquet(parquet_path: Path, max_structures: int = None):
    """Load structures from parquet with CIF column."""
    print(f"Loading raw data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} materials")
    
    if "cif" not in df.columns:
        raise ValueError("Parquet must have 'cif' column")
    
    if max_structures and len(df) > max_structures:
        print(f"  Sampling {max_structures} structures for SOAP computation...")
        df = df.sample(n=max_structures, random_state=42).reset_index(drop=True)
    
    structures = []
    valid_indices = []
    failed = 0
    
    print("  Parsing CIF strings to pymatgen Structures...")
    for i, row in df.iterrows():
        try:
            cif_str = row["cif"]
            struct = Structure.from_str(cif_str, fmt="cif")
            structures.append(struct)
            valid_indices.append(i)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"    Warning: Failed to parse {row['material_id']}: {e}")
    
    if failed > 5:
        print(f"    ... and {failed - 5} more failures")
    
    print(f"  Successfully parsed {len(structures)} / {len(df)} structures")
    
    return df.iloc[valid_indices].reset_index(drop=True), structures


def soap_loco_split_from_descriptors(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create LOCO splits from precomputed cluster IDs."""
    df = df.copy()
    df["soap_cluster"] = cluster_ids
    
    # Shuffle cluster order and assign to splits
    clusters = df["soap_cluster"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_ids, val_ids, test_ids = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}
    
    for cid in clusters:
        mids = df.loc[df["soap_cluster"] == cid, "material_id"].tolist()
        if counts["train"] < n_train:
            train_ids.extend(mids)
            counts["train"] += len(mids)
        elif counts["val"] < n_val:
            val_ids.extend(mids)
            counts["val"] += len(mids)
        else:
            test_ids.extend(mids)
            counts["test"] += len(mids)
    
    # Shuffle within splits
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def main():
    parser = argparse.ArgumentParser(description="Generate SOAP-LOCO splits")
    parser.add_argument("--config", type=str, default="configs/train_cgcnn_ehull.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=30, help="Number of SOAP clusters")
    parser.add_argument("--max-structures", type=int, default=None, 
                        help="Max structures for SOAP (for memory/speed)")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    
    if not check_dscribe_available():
        print("ERROR: dscribe is required. Install with: pip install dscribe")
        return
    
    cfg = load_cfg(args.config)
    
    # Load raw data with CIF
    raw_dir = Path("data/raw/mp")
    raw_parquet = raw_dir / "raw_mp_cathodes_v1.parquet"
    
    df, structures = load_structures_from_parquet(raw_parquet, args.max_structures)
    
    print(f"\nComputing SOAP descriptors for {len(structures)} structures...")
    print("  (This may take a few minutes for large datasets)")
    
    # Use smaller SOAP parameters to keep dimensionality manageable
    # Auto-detect species from the structures
    all_species = set()
    for struct in structures:
        all_species.update([str(s) for s in struct.species])
    cathode_species = sorted(list(all_species))
    print(f"  Detected {len(cathode_species)} unique species: {cathode_species[:10]}...")
    
    descriptors = compute_soap_descriptors(
        structures,
        r_cut=4.0,  # Even smaller cutoff to reduce dimensionality
        n_max=3,    # Fewer radial basis functions  
        l_max=2,    # Fewer angular channels
        species=cathode_species,
    )
    print(f"  SOAP descriptors shape: {descriptors.shape}")
    
    print(f"\nClustering into {args.n_clusters} groups by structural similarity...")
    cluster_ids = soap_cluster(descriptors, n_clusters=args.n_clusters, seed=args.seed)
    
    unique_clusters = len(np.unique(cluster_ids))
    print(f"  Created {unique_clusters} clusters")
    
    print("\nCreating SOAP-LOCO splits...")
    splits = soap_loco_split_from_descriptors(
        df,
        cluster_ids,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    
    n = len(df)
    print(f"  Train: {len(splits['train'])} ({100*len(splits['train'])/n:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({100*len(splits['val'])/n:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({100*len(splits['test'])/n:.1f}%)")
    
    # Compute novelty metrics
    train_mask = df["material_id"].isin(splits["train"])
    test_mask = df["material_id"].isin(splits["test"])
    
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        # Map back to descriptor indices
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        novelty = compute_split_novelty(descriptors[train_idx], descriptors[test_idx])
        print(f"\nSplit Novelty Metrics:")
        print(f"  Mean min distance: {novelty['mean_min_distance']:.4f}")
        print(f"  Fraction OOD (>0.1): {novelty['fraction_ood']:.1%}")
    
    # Save splits
    splits_dir = Path("data/splits/mp")
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / "splits_mp_cathodes_v1_soap-loco.json"
    
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSaved SOAP-LOCO splits to {splits_path}")
    
    # Print class balance
    target_col = cfg["data"]["target"]["name"]
    processed_parquet = Path(cfg["data"]["processed_dir"]) / f"processed_{cfg['data']['dataset_name']}.parquet"
    
    if processed_parquet.exists():
        df_processed = pd.read_parquet(processed_parquet)
        print("\nClass balance in SOAP-LOCO splits:")
        for split_name, material_ids in splits.items():
            split_df = df_processed[df_processed["material_id"].isin(material_ids)]
            if len(split_df) > 0:
                stable = (split_df[target_col] < 0.05).sum()
                metastable = ((split_df[target_col] >= 0.05) & (split_df[target_col] < 0.10)).sum()
                unstable = (split_df[target_col] >= 0.10).sum()
                print(f"  {split_name}: stable={stable} ({100*stable/len(split_df):.1f}%), "
                      f"metastable={metastable} ({100*metastable/len(split_df):.1f}%), "
                      f"unstable={unstable} ({100*unstable/len(split_df):.1f}%)")
    
    print("\nDone! To train with SOAP-LOCO splits, update your config to use:")
    print(f"  splits_manifest: {splits_path}")


if __name__ == "__main__":
    main()
