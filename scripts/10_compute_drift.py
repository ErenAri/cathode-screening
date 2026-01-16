#!/usr/bin/env python3
"""
Compute drift metrics between baseline and current prediction sets.

Example:
    python scripts/10_compute_drift.py --baseline data/predictions/baseline.parquet \
        --current data/predictions/current.parquet --output data/monitoring/drift.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> Optional[float]:
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if baseline.size < 2 or current.size < 2:
        return None

    quantiles = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 3:
        min_val = float(np.min(baseline))
        max_val = float(np.max(baseline))
        if min_val == max_val:
            return 0.0
        edges = np.linspace(min_val, max_val, bins + 1)

    base_hist, _ = np.histogram(baseline, bins=edges)
    curr_hist, _ = np.histogram(current, bins=edges)
    base_dist = base_hist / max(base_hist.sum(), 1)
    curr_dist = curr_hist / max(curr_hist.sum(), 1)

    eps = 1e-6
    psi_vals = (base_dist - curr_dist) * np.log((base_dist + eps) / (curr_dist + eps))
    return float(np.sum(psi_vals))


def _column_stats(baseline: np.ndarray, current: np.ndarray) -> Dict[str, Optional[float]]:
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    return {
        "baseline_mean": float(np.mean(baseline)) if baseline.size else None,
        "baseline_std": float(np.std(baseline)) if baseline.size else None,
        "current_mean": float(np.mean(current)) if current.size else None,
        "current_std": float(np.std(current)) if current.size else None,
    }


def _resolve_columns(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    requested: Optional[List[str]],
) -> List[str]:
    if requested:
        return [col for col in requested if col in baseline.columns and col in current.columns]

    default_cols = [
        "q50",
        "ood_score",
        "epistemic_std",
        "p_stable",
        "ehull_pred",
    ]
    cols = [col for col in default_cols if col in baseline.columns and col in current.columns]
    if cols:
        return cols

    numeric_cols = set(baseline.select_dtypes(include=[np.number]).columns)
    numeric_cols &= set(current.select_dtypes(include=[np.number]).columns)
    return sorted(numeric_cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute drift metrics between prediction sets")
    ap.add_argument("--baseline", required=True, help="Baseline predictions parquet")
    ap.add_argument("--current", required=True, help="Current predictions parquet")
    ap.add_argument("--output", help="Optional JSON output path")
    ap.add_argument("--columns", nargs="*", help="Columns to evaluate (default: common numeric)")
    ap.add_argument("--bins", type=int, default=10, help="Number of PSI bins")
    ap.add_argument("--psi-threshold", type=float, default=0.2, help="PSI threshold for drift flag")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current file not found: {current_path}")

    baseline_df = pd.read_parquet(baseline_path)
    current_df = pd.read_parquet(current_path)

    columns = _resolve_columns(baseline_df, current_df, args.columns)
    if not columns:
        raise ValueError("No common numeric columns found for drift evaluation")

    results: Dict[str, Dict[str, Optional[float]]] = {}
    drifted = []

    for col in columns:
        base_vals = baseline_df[col].to_numpy()
        curr_vals = current_df[col].to_numpy()
        psi_val = _psi(base_vals, curr_vals, bins=args.bins)
        stats = _column_stats(base_vals, curr_vals)
        stats["psi"] = psi_val
        results[col] = stats
        if psi_val is not None and psi_val >= args.psi_threshold:
            drifted.append(col)

    payload = {
        "baseline": str(baseline_path),
        "current": str(current_path),
        "n_baseline": int(len(baseline_df)),
        "n_current": int(len(current_df)),
        "psi_threshold": float(args.psi_threshold),
        "columns": results,
        "drifted_columns": drifted,
        "retrain_recommended": bool(drifted),
    }

    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
