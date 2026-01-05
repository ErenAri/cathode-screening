#!/usr/bin/env python3
"""
Compute split conformal calibration parameters for quantile predictions.

Loads a trained checkpoint, runs inference on the validation set, and computes
conformal calibration parameters (Δ_upper, Δ_lower) for calibrated prediction
intervals with guaranteed coverage.

Usage:
    python scripts/05_calibrate_conformal.py \
        --checkpoint data/artifacts/ensemble/member_seed42/best.pt \
        --data-config configs/train_cgcnn_ehull.yaml \
        --alpha 0.10 \
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared_imports import (
    CGCNN, GraphNPZDataset, collate, Normalizer, load_cfg, get_device
)

from cathode_screening.evaluation.conformal import (
    fit_conformal_calibration,
    save_calibration_params,
)


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
