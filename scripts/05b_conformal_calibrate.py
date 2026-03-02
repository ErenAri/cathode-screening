#!/usr/bin/env python3
"""
Compute split conformal calibration deltas from ensemble validation predictions.

Reads q10_raw, q90_raw, y_true from the val parquet and computes symmetric
delta to achieve target coverage (default 90%).

Usage:
    python scripts/05b_conformal_calibrate.py \
        --val-predictions data/predictions/ensemble_val.parquet \
        --output-dir artifacts/models/mace_ensemble_v1/calibration \
        --target-coverage 0.90
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-predictions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-coverage", type=float, default=0.90)
    args = ap.parse_args()

    df = pd.read_parquet(args.val_predictions)
    df = df.dropna(subset=["y_true"])
    n = len(df)

    y = df["y_true"].values
    q10 = df["q10_raw"].values
    q90 = df["q90_raw"].values

    # Nonconformity scores: how far outside the interval is y_true?
    scores = np.maximum(q10 - y, y - q90)  # positive = outside interval

    # Conformal quantile with finite-sample correction
    alpha = 1.0 - args.target_coverage
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = float(np.quantile(scores, level))

    # delta must be non-negative (only widen, never shrink)
    delta = max(q_hat, 0.0)

    # Current coverage
    raw_coverage = float(np.mean((y >= q10) & (y <= q90)))
    cal_coverage = float(np.mean((y >= (q10 - delta)) & (y <= (q90 + delta))))

    print(f"Validation samples: {n}")
    print(f"Target coverage: {args.target_coverage:.0%}")
    print(f"Raw coverage: {raw_coverage:.1%}")
    print(f"Conformal delta: {delta:.6f}")
    print(f"Calibrated coverage: {cal_coverage:.1%}")
    print(f"Raw interval width (median): {np.median(q90 - q10):.4f}")
    print(f"Calibrated interval width (median): {np.median(q90 - q10) + 2*delta:.4f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "delta_upper": delta,
        "delta_lower": delta,
        "target_coverage": args.target_coverage,
        "raw_coverage": raw_coverage,
        "calibrated_coverage_val": cal_coverage,
        "n_val": n,
        "method": "split_conformal_symmetric",
    }

    out_path = out_dir / "conformal_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
