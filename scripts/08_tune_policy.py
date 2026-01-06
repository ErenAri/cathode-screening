#!/usr/bin/env python3
"""
Tune decision policy thresholds on validation set.

Usage:
    python scripts/08_tune_policy.py \
        --predictions predictions/ensemble_val.parquet \
        --mode dft_followup \
        --run-id run_seed42 \
        --output-dir artifacts/policy

Inputs:
    Predictions parquet/csv with columns:
    - y_true, q10_cal, q50, q90_cal
    - p_stable (optional): probability of stability
    - ood_score (optional): OOD score [0,1]
    - gate_level (optional): "IN"/"BORDERLINE"/"OOD"
    - chemistry_cluster (optional): cluster label for slicing
    - family (optional): material family label for slicing

Grid search over:
    T_keep in [0.05, 0.30] - upper bound must be <= T_keep for KEEP
                            (denser in 0.10-0.25 range)
    T_kill in [0.08, 0.20] - lower bound must be >= T_kill for KILL

Outputs:
    - artifacts/policy/<run_id>/policy_<mode>.json
    - artifacts/policy/<run_id>/report_<mode>.json
    - artifacts/policy/<run_id>/report_<mode>_sliced.json (if slicing columns present)
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cathode_screening.inference.decision_policy import (
    DecisionPolicy,
    CostModel,
    PolicyThresholds,
    PolicyReport,
    tune_policy_thresholds,
    save_policy_report,
    THRESH_STABLE,
)


# --------------------------------------------------------------------------- #
#  Slicing utilities
# --------------------------------------------------------------------------- #
def get_ehull_bin(y_true: float) -> str:
    """Bin E_hull into interpretable categories."""
    if y_true < 0.02:
        return "highly_stable"
    elif y_true < 0.05:
        return "stable"
    elif y_true < 0.10:
        return "metastable"
    elif y_true < 0.20:
        return "marginal"
    else:
        return "unstable"


def compute_sliced_metrics(
    df: pd.DataFrame,
    policy: DecisionPolicy,
    slice_col: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute policy metrics for each slice.
    
    Args:
        df: DataFrame with predictions and slice column
        policy: Tuned policy
        slice_col: Column name to slice by
    
    Returns:
        Dict mapping slice_value -> metrics dict
    """
    results = {}
    
    for slice_val in df[slice_col].unique():
        if pd.isna(slice_val):
            slice_name = "unknown"
            mask = df[slice_col].isna()
        else:
            slice_name = str(slice_val)
            mask = df[slice_col] == slice_val
        
        subset = df[mask]
        if len(subset) < 10:
            continue
        
        # Extract arrays
        q10_cal = subset["q10_cal"].values
        q50 = subset["q50"].values
        q90_cal = subset["q90_cal"].values
        y_true = subset["y_true"].values
        p_stable = subset["p_stable"].values if "p_stable" in subset.columns else None
        gate_levels = subset["gate_level"].values if "gate_level" in subset.columns else None
        
        # Evaluate
        report = policy.evaluate(q10_cal, q50, q90_cal, y_true, p_stable, gate_levels)
        
        results[slice_name] = {
            "n_samples": len(subset),
            "n_stable": int((y_true < THRESH_STABLE).sum()),
            "stable_frac": float((y_true < THRESH_STABLE).mean()),
            "n_keep": report.n_keep,
            "n_maybe": report.n_maybe,
            "n_kill": report.n_kill,
            "fn_rate": report.fn_rate,
            "fp_rate": report.fp_rate,
            "expected_cost": report.expected_cost,
            "cost_per_sample": report.cost_per_sample,
        }
    
    return results


def derive_gate_level_from_score(
    ood_score: np.ndarray,
    thresh_borderline: float = 0.3,
    thresh_ood: float = 0.7,
) -> np.ndarray:
    """Convert OOD score to gate level."""
    levels = np.where(
        ood_score >= thresh_ood, "OOD",
        np.where(ood_score >= thresh_borderline, "BORDERLINE", "IN")
    )
    return levels


def main():
    ap = argparse.ArgumentParser(description="Tune decision policy thresholds")
    ap.add_argument("--predictions", required=True, help="Predictions parquet/csv")
    ap.add_argument("--mode", required=True, choices=["dft_followup", "experimental"])
    ap.add_argument("--run-id", default="default", help="Run identifier")
    ap.add_argument("--output-dir", default="artifacts/policy", help="Output directory")
    
    # Grid parameters
    ap.add_argument("--t-keep-min", type=float, default=0.05, help="T_keep grid min")
    ap.add_argument("--t-keep-max", type=float, default=0.30, help="T_keep grid max")
    ap.add_argument("--t-kill-min", type=float, default=0.08, help="T_kill grid min")
    ap.add_argument("--t-kill-max", type=float, default=0.20, help="T_kill grid max")
    ap.add_argument("--n-grid", type=int, default=25, help="Grid search resolution")
    
    # Constraints
    ap.add_argument("--max-fn-rate", type=float, default=None, 
                    help="Max allowed FN rate (default: 0.10 for dft, 0.05 for experimental)")
    
    # OOD thresholds for score -> gate conversion
    ap.add_argument("--ood-thresh-borderline", type=float, default=0.3,
                    help="OOD score threshold for BORDERLINE")
    ap.add_argument("--ood-thresh-ood", type=float, default=0.7,
                    help="OOD score threshold for OOD")
    
    args = ap.parse_args()
    
    # Load predictions
    pred_path = Path(args.predictions)
    if pred_path.suffix == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)
    
    print(f"Loaded {len(df)} predictions from {pred_path}")
    
    # Extract arrays
    required_cols = ["y_true", "q10_cal", "q50", "q90_cal"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    y_true = df["y_true"].values
    q10_cal = df["q10_cal"].values
    q50 = df["q50"].values
    q90_cal = df["q90_cal"].values
    
    # Optional columns
    p_stable = df["p_stable"].values if "p_stable" in df.columns else None
    
    # Handle OOD gating: prefer gate_level, derive from ood_score if needed
    if "gate_level" in df.columns:
        gate_levels = df["gate_level"].values
    elif "ood_score" in df.columns:
        print(f"Deriving gate_level from ood_score (thresholds: {args.ood_thresh_borderline}, {args.ood_thresh_ood})")
        gate_levels = derive_gate_level_from_score(
            df["ood_score"].values,
            args.ood_thresh_borderline,
            args.ood_thresh_ood,
        )
        df["gate_level"] = gate_levels
    else:
        gate_levels = None
    
    # Print data summary
    n_stable = (y_true < THRESH_STABLE).sum()
    print(f"Ground truth: {n_stable} stable ({n_stable/len(y_true):.1%}), "
          f"{len(y_true) - n_stable} unstable")
    
    if gate_levels is not None:
        n_in = (gate_levels == "IN").sum()
        n_border = (gate_levels == "BORDERLINE").sum()
        n_ood = (gate_levels == "OOD").sum()
        print(f"Gate levels: IN={n_in}, BORDERLINE={n_border}, OOD={n_ood}")
    
    # Tune policy
    print(f"\nTuning policy for mode: {args.mode}")
    print(f"Grid: T_keep in [{args.t_keep_min}, {args.t_keep_max}], "
          f"T_kill in [{args.t_kill_min}, {args.t_kill_max}]")
    print(f"Grid resolution: {args.n_grid}")
    
    # Sanity check: can any samples satisfy KEEP at max threshold?
    q90_min = q90_cal.min()
    q90_median = np.median(q90_cal)
    print(f"\nData stats: q90_cal min={q90_min:.4f}, median={q90_median:.4f}")
    
    if args.mode != "dft_followup":  # DFT mode has T_keep=0
        n_keep_possible_at_max = (q90_cal <= args.t_keep_max).sum()
        if n_keep_possible_at_max == 0:
            print(f"\n*** WARNING: No samples have q90_cal <= {args.t_keep_max:.3f} ***")
            print(f"    KEEP decisions will be impossible at all grid points!")
            print(f"    Consider increasing --t-keep-max above {q90_min:.4f}")
        else:
            print(f"KEEP candidates at max threshold ({args.t_keep_max:.3f}): {n_keep_possible_at_max}")
    
    # Extract epistemic_std if present
    epistemic_std = df["epistemic_std"].values if "epistemic_std" in df.columns else None
    
    best_policy, best_report = tune_policy_thresholds(
        q10_cal=q10_cal,
        q50=q50,
        q90_cal=q90_cal,
        y_true=y_true,
        mode=args.mode,
        p_stable=p_stable,
        gate_levels=gate_levels,
        epistemic_std=epistemic_std,
        T_keep_range=(args.t_keep_min, args.t_keep_max),
        T_kill_range=(args.t_kill_min, args.t_kill_max),
        n_grid=args.n_grid,
        verbose=True,
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save policy
    policy_path = output_dir / f"policy_{args.mode}.json"
    best_policy.save(policy_path)
    print(f"\nSaved policy to {policy_path}")
    
    # Save report
    report_path = output_dir / f"report_{args.mode}.json"
    save_policy_report(best_report, report_path)
    print(f"Saved report to {report_path}")
    
    # Sliced analysis
    slice_cols = []
    if "chemistry_cluster" in df.columns:
        slice_cols.append("chemistry_cluster")
    if "family" in df.columns:
        slice_cols.append("family")
    
    # Add Ehull bins as a slice dimension
    df["ehull_bin"] = [get_ehull_bin(y) for y in y_true]
    slice_cols.append("ehull_bin")
    
    sliced_results = {}
    for slice_col in slice_cols:
        print(f"\nComputing sliced metrics by {slice_col}...")
        sliced_results[slice_col] = compute_sliced_metrics(df, best_policy, slice_col)
    
    # Save sliced report
    sliced_path = output_dir / f"report_{args.mode}_sliced.json"
    with open(sliced_path, "w", encoding="utf-8") as f:
        json.dump(sliced_results, f, indent=2)
    print(f"Saved sliced report to {sliced_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(best_report)
    print("="*60)
    
    # Detailed breakdown
    print("\nCost Model:")
    cm = best_report.cost_model
    print(f"  C_keep_fp (false KEEP): {cm.C_keep_fp}")
    print(f"  C_kill_fn (false KILL): {cm.C_kill_fn}")
    print(f"  C_maybe: {cm.C_maybe}")
    
    print("\nTuned Thresholds:")
    t = best_report.thresholds
    print(f"  T_keep: {t.T_keep:.4f} (KEEP if q90_cal <= T_keep)")
    print(f"  T_kill: {t.T_kill:.4f} (KILL if q10_cal >= T_kill)")
    if t.P_keep is not None:
        print(f"  P_keep: {t.P_keep:.3f} (AND p_stable >= P_keep)")
    print(f"  Allow KEEP when BORDERLINE: {t.allow_keep_borderline}")
    print(f"  Allow KEEP when OOD: {t.allow_keep_ood}")
    
    # Print sliced summary for Ehull bins
    print("\nSliced by Ehull bin:")
    for bin_name, metrics in sorted(sliced_results.get("ehull_bin", {}).items()):
        print(f"  {bin_name}: n={metrics['n_samples']}, "
              f"KEEP={metrics['n_keep']}, MAYBE={metrics['n_maybe']}, KILL={metrics['n_kill']}, "
              f"FN={metrics['fn_rate']:.1%}")
    
    # Compare to baseline
    print("\nComparison to default policy:")
    default_policy = DecisionPolicy.for_mode(args.mode)
    default_report = default_policy.evaluate(q10_cal, q50, q90_cal, y_true, p_stable, gate_levels)
    
    print(f"  Default cost: {default_report.expected_cost:.2f}")
    print(f"  Tuned cost:   {best_report.expected_cost:.2f}")
    improvement = (default_report.expected_cost - best_report.expected_cost)
    if abs(default_report.expected_cost) > 0:
        improvement_pct = improvement / abs(default_report.expected_cost) * 100
        print(f"  Improvement:  {improvement_pct:.1f}%")


if __name__ == "__main__":
    main()
