#!/usr/bin/env python
"""
03_make_splits.py - Create train/val/test splits for cathode screening.

Splits are based on composition clustering to ensure test set contains
novel chemistries not seen in training (prevents trivial lookup).

Usage:
    python scripts/03_make_splits.py --config configs/train_cgcnn_ehull.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from pymatgen.core import Composition


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def composition_fingerprint(formula: str) -> np.ndarray:
    """Convert formula to element fraction vector."""
    vocab = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(vocab), dtype=np.float32)
    for i, el in enumerate(vocab):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


def make_cluster_splits(
    df: pd.DataFrame, 
    seed: int = 42, 
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    n_clusters: int = 50
) -> dict:
    """
    Create cluster-based splits to prevent composition leakage.
    
    Clusters similar compositions together, then assigns entire clusters
    to train/val/test splits. This ensures test contains truly novel
    compositions not seen during training.
    """
    print(f"Creating cluster-based splits (seed={seed})...")
    
    # Build composition fingerprints
    X = np.stack([composition_fingerprint(f) for f in df["formula_pretty"]], axis=0)
    X = normalize(X)
    
    # Determine number of clusters (can't exceed unique compositions)
    unique = np.unique(X, axis=0).shape[0]
    k = min(n_clusters, unique)
    print(f"  Using {k} clusters (unique compositions: {unique})")
    
    # K-Means clustering
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_ids = km.fit_predict(X)
    
    df = df.copy()
    df["cluster_id"] = cluster_ids
    
    # Shuffle cluster order and assign to splits
    clusters = df["cluster_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_ids, val_ids, test_ids = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}
    
    for cid in clusters:
        mids = df.loc[df["cluster_id"] == cid, "material_id"].tolist()
        if counts["train"] < n_train:
            train_ids.extend(mids)
            counts["train"] += len(mids)
        elif counts["val"] < n_val:
            val_ids.extend(mids)
            counts["val"] += len(mids)
        else:
            test_ids.extend(mids)
            counts["test"] += len(mids)
    
    # Shuffle within splits for good measure
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    
    print(f"  Train: {len(train_ids)} ({100*len(train_ids)/n:.1f}%)")
    print(f"  Val:   {len(val_ids)} ({100*len(val_ids)/n:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({100*len(test_ids)/n:.1f}%)")
    
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", type=str, default="configs/train_cgcnn_ehull.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--n-clusters", type=int, default=50)
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    
    # Load processed data
    processed_dir = Path(cfg["data"]["processed_dir"])
    dataset_name = cfg["data"]["dataset_name"]
    parquet_path = processed_dir / f"processed_{dataset_name}.parquet"
    
    print(f"Loading processed data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} materials")
    
    # Create splits
    splits = make_cluster_splits(
        df, 
        seed=seed, 
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_clusters=args.n_clusters
    )
    
    # Save splits manifest
    splits_path = Path(cfg["data"]["splits_manifest"])
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    
    print(f"Saved splits to {splits_path}")
    
    # Print class balance stats
    target_col = cfg["data"]["target"]["name"]
    for split_name, material_ids in splits.items():
        split_df = df[df["material_id"].isin(material_ids)]
        stable = (split_df[target_col] < 0.05).sum()
        metastable = ((split_df[target_col] >= 0.05) & (split_df[target_col] < 0.10)).sum()
        unstable = (split_df[target_col] >= 0.10).sum()
        print(f"  {split_name}: stable={stable} ({100*stable/len(split_df):.1f}%), "
              f"metastable={metastable} ({100*metastable/len(split_df):.1f}%), "
              f"unstable={unstable} ({100*unstable/len(split_df):.1f}%)")


if __name__ == "__main__":
    main()
