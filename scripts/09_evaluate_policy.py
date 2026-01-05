#!/usr/bin/env python3
"""
Evaluate tuned decision policy on test set.

Usage:
    python scripts/09_evaluate_policy.py \
        --predictions predictions/ensemble_test.parquet \
        --policy artifacts/policy/run_seed42/policy_dft_followup.json \
        --output artifacts/policy/run_seed42/eval_test_dft_followup.json

Inputs:
    Predictions parquet/csv with columns:
    - y_true, q10_cal, q50, q90_cal
    - p_stable (optional): probability of stability
    - ood_score (optional): OOD score [0,1]  
    - gate_level (optional): "IN"/"BORDERLINE"/"OOD"
    - chemistry_cluster (optional): cluster label for slicing
    - family (optional): material family label for slicing

Reports:
    - false_keep_rate: P(KEEP | unstable)
    - false_kill_rate: P(KILL | stable)
    - recall_stable: P(KEEP | stable)
    - precision_keep: P(stable | KEEP)
    - EF@1%, EF@5%: Enrichment factors
    - recall@budget: recall when limited to B followups
    
    All metrics sliced by:
    - chemistry cluster (if present)
    - family (if present)  
    - Ehull bins
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cathode_screening.inference.decision_policy import (
    DecisionPolicy,
    PolicyReport,
    THRESH_STABLE,
)

# Import shared helpers
from evaluate_policy_helpers import (
    compute_enrichment_factor,
    compute_recall_at_budget,
    get_ehull_bin,
    derive_gate_level_from_score,
)


# --------------------------------------------------------------------------- #
#  Metrics computation
# --------------------------------------------------------------------------- #
@dataclass
class PolicyEvaluationReport:
    """Comprehensive policy evaluation metrics."""
    # Sample counts
    n_total: int
    n_stable: int
    n_unstable: int
    
    # Decision counts
    n_keep: int
    n_maybe: int  
    n_kill: int
    
    # Core rates
    false_keep_rate: float     # P(KEEP | unstable) - false positives
    false_kill_rate: float     # P(KILL | stable) - false negatives (missed)
    recall_stable: float       # P(KEEP | stable) = 1 - false_kill_rate
    precision_keep: float      # P(stable | KEEP)
    
    # Coverage
    keep_coverage: float       # fraction of samples in KEEP
    kill_coverage: float       # fraction of samples in KILL
    maybe_coverage: float      # fraction in MAYBE (need review)
    
    # Enrichment factors
    ef_at_1pct: float         # EF@1% of dataset
    ef_at_5pct: float         # EF@5% of dataset
    
    # Recall at budget
    recall_at_budget: Dict[int, float]  # budget -> recall
    
    # Cost metrics
    expected_cost: float
    cost_per_sample: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_policy_detailed(
    policy: DecisionPolicy,
    df: pd.DataFrame,
    budgets: Optional[List[int]] = None,
) -> PolicyEvaluationReport:
    """
    Compute comprehensive policy evaluation metrics.
    
    Args:
        policy: Decision policy to evaluate
        df: DataFrame with predictions and y_true
        budgets: Budget values for recall@budget (default: [10, 25, 50, 100, 200])
    
    Returns:
        PolicyEvaluationReport
    """
    if budgets is None:
        n = len(df)
        budgets = [10, 25, 50, 100, 200, 500]
        budgets = [b for b in budgets if b <= n]
    
    # Extract arrays
    q10_cal = df["q10_cal"].values
    q50 = df["q50"].values
    q90_cal = df["q90_cal"].values
    y_true = df["y_true"].values
    
    p_stable = df["p_stable"].values if "p_stable" in df.columns else None
    gate_levels = df["gate_level"].values if "gate_level" in df.columns else None
    
    # Get decisions
    decisions = policy.decide_batch(q10_cal, q50, q90_cal, p_stable, gate_levels)
    
    # Ground truth
    is_stable = y_true < THRESH_STABLE
    n_stable = is_stable.sum()
    n_unstable = (~is_stable).sum()
    
    # Decision masks
    keep_mask = decisions == "KEEP"
    maybe_mask = decisions == "MAYBE"
    kill_mask = decisions == "KILL"
    
    n_keep = keep_mask.sum()
    n_maybe = maybe_mask.sum()
    n_kill = kill_mask.sum()
    
    # Core rates
    false_keep_rate = (keep_mask & ~is_stable).sum() / max(n_unstable, 1)
    false_kill_rate = (kill_mask & is_stable).sum() / max(n_stable, 1)
    recall_stable = 1.0 - false_kill_rate
    precision_keep = is_stable[keep_mask].mean() if n_keep > 0 else 0.0
    
    # Coverage
    n_total = len(y_true)
    keep_coverage = n_keep / n_total
    kill_coverage = n_kill / n_total
    maybe_coverage = n_maybe / n_total
    
    # Enrichment factors
    ef_at_1pct = compute_enrichment_factor(y_true, q90_cal, 0.01)
    ef_at_5pct = compute_enrichment_factor(y_true, q90_cal, 0.05)
    
    # Recall at budget
    recall_at_budget = compute_recall_at_budget(
        y_true, decisions, q90_cal, budgets
    )
    
    # Cost
    expected_cost = policy.cost_model.expected_cost(decisions, y_true)
    
    return PolicyEvaluationReport(
        n_total=n_total,
        n_stable=int(n_stable),
        n_unstable=int(n_unstable),
        n_keep=int(n_keep),
        n_maybe=int(n_maybe),
        n_kill=int(n_kill),
        false_keep_rate=float(false_keep_rate),
        false_kill_rate=float(false_kill_rate),
        recall_stable=float(recall_stable),
        precision_keep=float(precision_keep),
        keep_coverage=float(keep_coverage),
        kill_coverage=float(kill_coverage),
        maybe_coverage=float(maybe_coverage),
        ef_at_1pct=float(ef_at_1pct),
        ef_at_5pct=float(ef_at_5pct),
        recall_at_budget=recall_at_budget,
        expected_cost=float(expected_cost),
        cost_per_sample=float(expected_cost / n_total),
    )


# --------------------------------------------------------------------------- #
#  Slicing utilities  
# --------------------------------------------------------------------------- #
def compute_sliced_evaluation(
    policy: DecisionPolicy,
    df: pd.DataFrame,
    slice_col: str,
    budgets: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute policy evaluation for each slice.
    
    Args:
        policy: Decision policy
        df: DataFrame with predictions
        slice_col: Column to slice by
        budgets: Budget values for recall@budget
    
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
        
        subset = df[mask].copy()
        if len(subset) < 5:
            continue
        
        # Compute evaluation
        report = evaluate_policy_detailed(policy, subset, budgets)
        results[slice_name] = report.to_dict()
    
    return results


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Evaluate decision policy on test set")
    ap.add_argument("--predictions", required=True, help="Predictions parquet/csv")
    ap.add_argument("--policy", required=True, help="Policy JSON from tuning")
    ap.add_argument("--output", required=True, help="Output evaluation JSON")
    
    # OOD thresholds
    ap.add_argument("--ood-thresh-borderline", type=float, default=0.3)
    ap.add_argument("--ood-thresh-ood", type=float, default=0.7)
    
    # Budget values for recall@budget
    ap.add_argument("--budgets", type=int, nargs="+", 
                    default=[10, 25, 50, 100, 200, 500],
                    help="Budget values for recall@budget")
    
    args = ap.parse_args()
    
    # Load policy
    policy_path = Path(args.policy)
    policy = DecisionPolicy.load(policy_path)
    print(f"Loaded policy from {policy_path}")
    print(f"  Mode: {policy.mode}")
    print(f"  T_keep: {policy.thresholds.T_keep:.4f}")
    print(f"  T_kill: {policy.thresholds.T_kill:.4f}")
    
    # Load predictions
    pred_path = Path(args.predictions)
    if pred_path.suffix == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)
    
    print(f"\nLoaded {len(df)} predictions from {pred_path}")
    
    # Validate columns
    required_cols = ["y_true", "q10_cal", "q50", "q90_cal"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Handle OOD gating
    if "gate_level" in df.columns:
        pass  # Already present
    elif "ood_score" in df.columns:
        print(f"Deriving gate_level from ood_score")
        df["gate_level"] = derive_gate_level_from_score(
            df["ood_score"].values,
            args.ood_thresh_borderline,
            args.ood_thresh_ood,
        )
    
    # Data summary
    y_true = df["y_true"].values
    n_stable = (y_true < THRESH_STABLE).sum()
    print(f"Ground truth: {n_stable} stable ({n_stable/len(df):.1%})")
    
    if "gate_level" in df.columns:
        gate_levels = df["gate_level"].values
        n_in = (gate_levels == "IN").sum()
        n_border = (gate_levels == "BORDERLINE").sum()
        n_ood = (gate_levels == "OOD").sum()
        print(f"Gate levels: IN={n_in}, BORDERLINE={n_border}, OOD={n_ood}")
    
    # Compute main evaluation
    print("\n" + "="*60)
    print("OVERALL EVALUATION")
    print("="*60)
    
    budgets = [b for b in args.budgets if b <= len(df)]
    main_report = evaluate_policy_detailed(policy, df, budgets)
    
    print(f"\nDecisions:")
    print(f"  KEEP:  {main_report.n_keep:5d} ({main_report.keep_coverage:.1%})")
    print(f"  MAYBE: {main_report.n_maybe:5d} ({main_report.maybe_coverage:.1%})")
    print(f"  KILL:  {main_report.n_kill:5d} ({main_report.kill_coverage:.1%})")
    
    print(f"\nError Rates:")
    print(f"  False KEEP rate (KEEP|unstable): {main_report.false_keep_rate:.2%}")
    print(f"  False KILL rate (KILL|stable):   {main_report.false_kill_rate:.2%}")
    
    print(f"\nPerformance:")
    print(f"  Recall (stable materials):  {main_report.recall_stable:.2%}")
    print(f"  Precision (KEEP decisions): {main_report.precision_keep:.2%}")
    
    print(f"\nEnrichment Factors (ranking by q90_cal):")
    print(f"  EF@1%: {main_report.ef_at_1pct:.2f}")
    print(f"  EF@5%: {main_report.ef_at_5pct:.2f}")
    
    print(f"\nRecall at Budget:")
    for budget, recall in sorted(main_report.recall_at_budget.items()):
        print(f"  Budget {budget:4d}: recall = {recall:.2%}")
    
    print(f"\nCost:")
    print(f"  Expected cost: {main_report.expected_cost:.2f}")
    print(f"  Cost per sample: {main_report.cost_per_sample:.4f}")
    
    # Sliced evaluation
    slice_cols = []
    if "chemistry_cluster" in df.columns:
        slice_cols.append("chemistry_cluster")
    if "family" in df.columns:
        slice_cols.append("family")
    
    # Add Ehull bins
    df["ehull_bin"] = [get_ehull_bin(y) for y in y_true]
    slice_cols.append("ehull_bin")
    
    sliced_results = {}
    for slice_col in slice_cols:
        print(f"\n{'='*60}")
        print(f"SLICED BY: {slice_col}")
        print("="*60)
        
        sliced = compute_sliced_evaluation(policy, df, slice_col, budgets)
        sliced_results[slice_col] = sliced
        
        # Print summary table
        print(f"\n{'Slice':<20} {'N':>6} {'Stable':>6} {'KEEP':>6} {'KILL':>6} "
              f"{'FN%':>7} {'FP%':>7} {'Prec':>7}")
        print("-" * 80)
        
        for slice_name, metrics in sorted(sliced.items()):
            print(f"{slice_name:<20} "
                  f"{metrics['n_total']:>6} "
                  f"{metrics['n_stable']:>6} "
                  f"{metrics['n_keep']:>6} "
                  f"{metrics['n_kill']:>6} "
                  f"{metrics['false_kill_rate']*100:>6.1f}% "
                  f"{metrics['false_keep_rate']*100:>6.1f}% "
                  f"{metrics['precision_keep']*100:>6.1f}%")
    
    # Prepare output
    output_data = {
        "policy": {
            "mode": policy.mode,
            "thresholds": policy.thresholds.to_dict(),
            "cost_model": policy.cost_model.to_dict(),
        },
        "data": {
            "source": str(pred_path),
            "n_samples": len(df),
            "n_stable": int(n_stable),
        },
        "overall": main_report.to_dict(),
        "sliced": sliced_results,
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
