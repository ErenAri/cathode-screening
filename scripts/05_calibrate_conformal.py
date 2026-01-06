#!/usr/bin/env python3
"""
Compute split conformal calibration parameters for quantile predictions.

Loads a trained checkpoint, runs inference on the validation set, and computes
conformal calibration parameters (Δ_upper, Δ_lower) for calibrated prediction
intervals with guaranteed coverage.

Supports two modes:
    - Global: Single Δ_upper/Δ_lower for all samples (default)
    - Group-conditional: Per-cluster Δ when n_cluster >= N_min, else fallback

Usage:
    # Global calibration
    python scripts/05_calibrate_conformal.py \
        --checkpoint data/artifacts/ensemble/member_seed42/best.pt \
        --data-config configs/train_cgcnn_ehull.yaml \
        --alpha 0.10 \
        --output-dir artifacts/calibration/run_seed42

    # Group-conditional calibration
    python scripts/05_calibrate_conformal.py \
        --checkpoint data/artifacts/ensemble/member_seed42/best.pt \
        --data-config configs/train_cgcnn_ehull.yaml \
        --alpha 0.10 \
        --group-conditional --n-min 50 \
        --output-dir artifacts/calibration/run_seed42

Outputs:
    artifacts/calibration/<run_id>/conformal_params.json
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from pymatgen.core import Composition

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared_imports import (
    CGCNN, GraphNPZDataset, collate, Normalizer, load_cfg, get_device
)

from cathode_screening.evaluation.conformal import (
    fit_conformal_calibration,
    fit_group_conformal_calibration,
    save_calibration_params,
    save_group_calibration_params,
)


def composition_fingerprint(formula: str) -> np.ndarray:
    """Convert formula to element fraction vector."""
    vocab = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(vocab), dtype=np.float32)
    for i, el in enumerate(vocab):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


def compute_cluster_ids(df: pd.DataFrame, n_clusters: int = 50, seed: int = 42) -> np.ndarray:
    """
    Compute chemistry cluster IDs for a dataframe.
    
    Uses K-means on composition fingerprints to group similar chemistries.
    """
    X = np.stack([composition_fingerprint(f) for f in df["formula_pretty"]], axis=0)
    X = normalize(X)
    
    unique = np.unique(X, axis=0).shape[0]
    k = min(n_clusters, unique)
    
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(X)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model and normalizer from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    
    model = CGCNN(
        node_in=int(cfg["model"].get("node_in_dim", 6)),
        node_dim=int(cfg["model"]["node_embed_dim"]),
        edge_dim=int(cfg["model"]["edge_rbf_bins"]),
        layers=int(cfg["model"]["message_passing_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        pooling=str(cfg["model"]["pooling"])
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    normalizer = Normalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    
    return model, normalizer, cfg


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    normalizer: Normalizer,
    loader: DataLoader,
    device: torch.device
) -> dict:
    """Run inference and collect predictions."""
    y_all = []
    q10_all = []
    q50_all = []
    q90_all = []
    
    for x, src, dst, e, b, y in loader:
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        q10, q50, q90, _, _ = model(x, src, dst, e, b)
        
        # Denormalize
        q10 = normalizer.denorm(q10).cpu().numpy()
        q50 = normalizer.denorm(q50).cpu().numpy()
        q90 = normalizer.denorm(q90).cpu().numpy()
        
        y_all.append(y.numpy())
        q10_all.append(q10)
        q50_all.append(q50)
        q90_all.append(q90)
    
    return {
        "y_true": np.concatenate(y_all),
        "q10": np.concatenate(q10_all),
        "q50": np.concatenate(q50_all),
        "q90": np.concatenate(q90_all),
    }


def main():
    ap = argparse.ArgumentParser(description="Compute conformal calibration parameters")
    ap.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    ap.add_argument("--data-config", required=True, help="Data config YAML")
    ap.add_argument("--alpha", type=float, default=0.10, 
                    help="Miscoverage level (default: 0.10 for 90%% coverage)")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="Which split to use for calibration (default: val)")
    ap.add_argument("--output-dir", required=True, help="Output directory for calibration params")
    ap.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu")
    # Group-conditional options
    ap.add_argument("--group-conditional", action="store_true",
                    help="Enable per-cluster conformal calibration")
    ap.add_argument("--n-min", type=int, default=50,
                    help="Minimum cluster size for per-cluster calibration (default: 50)")
    ap.add_argument("--n-clusters", type=int, default=50,
                    help="Number of chemistry clusters (default: 50)")
    ap.add_argument("--cluster-seed", type=int, default=42,
                    help="Random seed for clustering (default: 42)")
    args = ap.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, normalizer, model_cfg = load_checkpoint(checkpoint_path, device)
    
    # Load data config and splits
    cfg = load_cfg(args.data_config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    ds_name = cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    
    print(f"Loading data from: {processed_path}")
    df = pd.read_parquet(processed_path)
    
    splits_path = Path(cfg["data"]["splits_manifest"])
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Get calibration split
    split_ids = set(splits[args.split])
    df_split = df[df["material_id"].isin(split_ids)].reset_index(drop=True)
    
    target = cfg["data"]["target"]["name"]
    print(f"\nCalibration split: {args.split}")
    print(f"  Samples: {len(df_split)}")
    print(f"  Target: {target}")
    
    # Create data loader
    dataset = GraphNPZDataset(df_split, target_col=target)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )
    
    # Run inference
    print("\nRunning inference on calibration split...")
    preds = collect_predictions(model, normalizer, loader, device)
    
    y_true = preds["y_true"]
    q10 = preds["q10"]
    q90 = preds["q90"]
    
    # Compute raw coverage
    raw_coverage = np.mean((y_true >= q10) & (y_true <= q90))
    print(f"\nRaw interval coverage: {raw_coverage:.3f}")
    
    # Fit conformal calibration
    if args.group_conditional:
        # Compute cluster IDs for the calibration split
        print(f"\nComputing chemistry clusters (n={args.n_clusters}, seed={args.cluster_seed})...")
        cluster_ids = compute_cluster_ids(df_split, n_clusters=args.n_clusters, seed=args.cluster_seed)
        
        # Count cluster sizes
        unique, counts = np.unique(cluster_ids, return_counts=True)
        n_large = sum(c >= args.n_min for c in counts)
        print(f"  Found {len(unique)} clusters, {n_large} with n >= {args.n_min}")
        
        print(f"\nFitting group-conditional conformal calibration (α={args.alpha}, n_min={args.n_min})...")
        params = fit_group_conformal_calibration(
            y_true=y_true,
            q10=q10,
            q90=q90,
            cluster_ids=cluster_ids,
            alpha=args.alpha,
            n_min=args.n_min,
            split_name=args.split,
            checkpoint_path=str(checkpoint_path)
        )
        
        print(f"\nGroup calibration results:")
        print(f"  Global Δ_upper: {params.global_delta_upper:.6f}")
        print(f"  Global Δ_lower: {params.global_delta_lower:.6f}")
        print(f"  Global raw coverage: {params.global_raw_coverage:.3f}")
        print(f"  Global calibrated coverage: {params.global_calibrated_coverage:.3f}")
        print(f"  Clusters with per-cluster Δ: {len(params.cluster_deltas)}")
        print(f"  Target coverage: {1 - args.alpha:.2f}")
        
        # Per-cluster coverage summary
        if params.cluster_deltas:
            coverages = [d["calibrated_coverage"] for d in params.cluster_deltas.values()]
            print(f"\nPer-cluster calibrated coverage:")
            print(f"  Min: {min(coverages):.3f}, Max: {max(coverages):.3f}, Mean: {np.mean(coverages):.3f}")
        
        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "conformal_params_group.json"
        save_group_calibration_params(params, output_path)
        print(f"\nSaved group calibration params to: {output_path}")
        
        # Save predictions with cluster IDs
        preds_path = output_dir / "calibration_predictions.npz"
        np.savez(
            preds_path,
            y_true=y_true,
            q10=q10,
            q50=preds["q50"],
            q90=q90,
            cluster_ids=cluster_ids,
        )
        print(f"Saved predictions to: {preds_path}")
        
        # Coverage check
        if params.global_calibrated_coverage < (1 - args.alpha - 0.02):
            print(f"\n⚠ WARNING: Global calibrated coverage {params.global_calibrated_coverage:.3f} "
                  f"below target {1-args.alpha:.2f}")
        else:
            print(f"\n✓ Group-conditional calibration successful")
    
    else:
        # Global calibration (original behavior)
        print(f"\nFitting conformal calibration (α={args.alpha})...")
        params = fit_conformal_calibration(
            y_true=y_true,
            q10=q10,
            q90=q90,
            alpha=args.alpha,
            split_name=args.split,
            checkpoint_path=str(checkpoint_path)
        )
        
        print(f"\nCalibration results:")
        print(f"  n_calibration: {params.n_calibration}")
        print(f"  Δ_upper: {params.delta_upper:.6f}")
        print(f"  Δ_lower: {params.delta_lower:.6f}")
        print(f"  Raw coverage: {params.raw_coverage:.3f}")
        print(f"  Calibrated coverage: {params.calibrated_coverage:.3f}")
        print(f"  Target coverage: {1 - args.alpha:.2f}")
        
        # Save calibration parameters
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "conformal_params.json"
        save_calibration_params(params, output_path)
        print(f"\nSaved calibration params to: {output_path}")
        
        # Also save predictions for diagnostics
        preds_path = output_dir / "calibration_predictions.npz"
        np.savez(
            preds_path,
            y_true=y_true,
            q10=q10,
            q50=preds["q50"],
            q90=q90,
            q10_cal=q10 - params.delta_lower,
            q90_cal=q90 + params.delta_upper
        )
        print(f"Saved predictions to: {preds_path}")
        
        # Coverage check
        if params.calibrated_coverage < (1 - args.alpha - 0.02):
            print(f"\n⚠ WARNING: Calibrated coverage {params.calibrated_coverage:.3f} "
                  f"below target {1-args.alpha:.2f}")
        else:
            print(f"\n✓ Calibration successful")
    
    # GPU cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
