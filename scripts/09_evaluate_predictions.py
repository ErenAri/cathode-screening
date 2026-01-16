#!/usr/bin/env python3
"""
Evaluate prediction parquet files against decision-grade metrics.

Example:
    python scripts/09_evaluate_predictions.py --input data/predictions/ensemble_soap_loco_test.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _as_array(df: pd.DataFrame, column: str) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None
    return df[column].to_numpy()


def _parse_member_predictions(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "q50_members_str" not in df.columns:
        return None
    raw = df["q50_members_str"].dropna().apply(json.loads)
    if len(raw) == 0:
        return None
    return np.array(raw.tolist(), dtype=np.float32)


def compute_metrics(df: pd.DataFrame, use_gated: bool = True) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "n_samples": len(df),
    }

    y_true = _as_array(df, "y_true")
    q50 = _as_array(df, "q50")
    q10 = _as_array(df, "q10_cal") or _as_array(df, "q10_raw")
    q90 = _as_array(df, "q90_cal") or _as_array(df, "q90_raw")

    if y_true is not None and q10 is not None and q90 is not None:
        coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
        widths = q90 - q10
        metrics["coverage"] = coverage
        metrics["interval_width_median"] = float(np.median(widths))
        metrics["interval_width_p95"] = float(np.percentile(widths, 95))

    decision_col = None
    if use_gated and "gated_decision" in df.columns:
        decision_col = "gated_decision"
    elif "decision" in df.columns:
        decision_col = "decision"

    if decision_col is not None and y_true is not None:
        decisions = df[decision_col].fillna("MAYBE").astype(str).to_numpy()
        keep_mask = decisions == "KEEP"
        kill_mask = decisions == "KILL"

        metrics["decision_counts"] = {
            "KEEP": int(keep_mask.sum()),
            "MAYBE": int((decisions == "MAYBE").sum()),
            "KILL": int(kill_mask.sum()),
        }
        metrics["decisive_rate"] = float((keep_mask.sum() + kill_mask.sum()) / len(decisions))

        if keep_mask.any():
            metrics["keep_precision"] = float(np.mean(y_true[keep_mask] < 0.05))
        if kill_mask.any():
            metrics["kill_precision"] = float(np.mean(y_true[kill_mask] >= 0.10))

        stable_mask = y_true < 0.05
        if stable_mask.any():
            metrics["stable_kill_rate"] = float(((stable_mask) & kill_mask).sum() / stable_mask.sum())

    if "ood_score" in df.columns:
        metrics["ood_score_mean"] = float(df["ood_score"].mean())
    if "gate_level" in df.columns:
        metrics["ood_gate_counts"] = df["gate_level"].value_counts().to_dict()

    if y_true is not None and q50 is not None:
        metrics["mae"] = float(np.mean(np.abs(q50 - y_true)))
        metrics["rmse"] = float(np.sqrt(np.mean((q50 - y_true) ** 2)))

    members = _parse_member_predictions(df)
    if members is not None and y_true is not None:
        member_maes = [float(np.mean(np.abs(members[:, i] - y_true))) for i in range(members.shape[1])]
        metrics["member_maes"] = member_maes
        metrics["member_mae_mean"] = float(np.mean(member_maes))
        if q50 is not None:
            ensemble_mae = float(np.mean(np.abs(q50 - y_true)))
            metrics["ensemble_mae"] = ensemble_mae
            improvement = (np.mean(member_maes) - ensemble_mae) / max(np.mean(member_maes), 1e-8)
            metrics["ensemble_mae_improvement"] = float(improvement)
        disagreements = np.std(members, axis=1)
        metrics["ensemble_disagreement_median"] = float(np.median(disagreements))

    if "epistemic_std" in df.columns and y_true is not None and q50 is not None:
        errors = np.abs(q50 - y_true)
        corr = np.corrcoef(df["epistemic_std"].to_numpy(), errors)[0, 1]
        metrics["epistemic_error_corr"] = float(corr)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate prediction parquet files")
    ap.add_argument("--input", required=True, help="Path to predictions parquet")
    ap.add_argument("--output", help="Optional JSON output path")
    ap.add_argument("--use-gated", action="store_true", help="Use gated decisions if available")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_parquet(path)
    metrics = compute_metrics(df, use_gated=args.use_gated)

    output = json.dumps(metrics, indent=2, sort_keys=True)
    print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
