#!/usr/bin/env python
"""
Evaluate ensemble predictions with new decision-grade metrics.

Runs DAF, calibration metrics, and decision-level calibration on existing predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import new evaluation modules
from cathode_screening.evaluation.topk import (
    compute_ranking_metrics,
    format_metrics_table,
    discovery_acceleration_factor,
    DAF_TARGET_COUNTS,
)
from cathode_screening.evaluation.calibration_metrics import (
    compute_calibration_metrics,
    format_calibration_report,
)
from cathode_screening.evaluation.decision_calibration import (
    compute_decision_calibration,
    compute_decision_level_ece,
    format_decision_calibration_report,
)


def main():
    # Load predictions
    predictions_dir = Path("data/predictions")
    test_path = predictions_dir / "ensemble_test.parquet"
    
    if not test_path.exists():
        print(f"ERROR: Predictions not found at {test_path}")
        return
    
    print(f"Loading predictions from {test_path}...")
    df = pd.read_parquet(test_path)
    print(f"Loaded {len(df)} predictions")
    
    # Check required columns
    required = ["y_true", "q50", "q90_cal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available: {list(df.columns)}")
        return
    
    # Extract arrays
    y_true = df["y_true"].values
    q50 = df["q50"].values
    q90_cal = df["q90_cal"].values
    
    epistemic_std = df["epistemic_std"].values if "epistemic_std" in df.columns else None
    p_stable = df["p_stable"].values if "p_stable" in df.columns else None
    decisions = df["decision"].values if "decision" in df.columns else None
    
    print("\n" + "=" * 80)
    print("EVALUATION WITH NEW DECISION-GRADE METRICS")
    print("=" * 80 + "\n")
    
    # 1. Ranking metrics (EF, Recall, DAF)
    print("Computing ranking metrics...")
    metrics = compute_ranking_metrics(
        y_true=y_true,
        q50=q50,
        q90_cal=q90_cal,
        epistemic_std=epistemic_std,
    )
    print(format_metrics_table(metrics))
    print()
    
    # 2. Calibration metrics (if p_stable available)
    if p_stable is not None:
        print("\nComputing calibration metrics...")
        cal_metrics = compute_calibration_metrics(p_stable, y_true, threshold=0.05)
        print(format_calibration_report(cal_metrics))
    else:
        print("\nSkipping calibration metrics (p_stable not available)")
    
    # 3. Decision-level calibration (if decisions available)
    if decisions is not None:
        print("\nComputing decision-level calibration...")
        dec_stats = compute_decision_calibration(decisions, y_true, p_stable)
        
        ece_by_decision = None
        if p_stable is not None:
            ece_by_decision = compute_decision_level_ece(decisions, y_true, p_stable)
        
        print(format_decision_calibration_report(dec_stats, ece_by_decision))
    else:
        print("\nSkipping decision-level calibration (decision column not available)")
    
    # 4. Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    
    n_stable = (y_true < 0.05).sum()
    print(f"Stable (E_hull < 0.05): {n_stable} ({100*n_stable/len(df):.1f}%)")
    
    if decisions is not None:
        for dec in ["KEEP", "MAYBE", "KILL"]:
            n = (decisions == dec).sum()
            print(f"  {dec}: {n} ({100*n/len(df):.1f}%)")
    
    # Top DAF value
    daf_10 = discovery_acceleration_factor(y_true, q50, 0.05, 10)
    print(f"\nDAF@10 (threshold=0.05): {daf_10:.2f}x speedup vs random")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
