#!/usr/bin/env python3
"""
Validate prediction metrics against governance thresholds.

Example:
    python scripts/12_validate_release.py --input data/predictions/ensemble_soap_loco_test.parquet \
        --thresholds configs/governance_thresholds.json --output data/monitoring/eval_report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_thresholds(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        try:
            result[k] = float(v)
        except (TypeError, ValueError):
            continue
    return result


def _load_eval_module() -> object:
    root = Path(__file__).resolve().parents[1]
    eval_path = root / "scripts" / "09_evaluate_predictions.py"
    spec = importlib.util.spec_from_file_location("eval_predictions", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluation module: {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_predictions"] = module
    spec.loader.exec_module(module)
    return module


def _check_min(value: float | None, threshold: float, key: str, failures: List[str]) -> None:
    if value is None:
        return
    if value < threshold:
        failures.append(f"{key}={value:.4f} < {threshold:.4f}")


def _check_max(value: float | None, threshold: float, key: str, failures: List[str]) -> None:
    if value is None:
        return
    if value > threshold:
        failures.append(f"{key}={value:.4f} > {threshold:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate prediction metrics")
    ap.add_argument("--input", required=True, help="Predictions parquet file")
    ap.add_argument(
        "--thresholds",
        default="configs/governance_thresholds.json",
        help="Thresholds JSON",
    )
    ap.add_argument("--output", help="Optional JSON output path")
    ap.add_argument("--use-gated", action="store_true", help="Use gated decisions if available")
    ap.add_argument("--conformal-params", help="Path to conformal_params.json for calibration")
    args = ap.parse_args()

    input_path = Path(args.input)
    thresholds_path = Path(args.thresholds)
    if not input_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {input_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    df = pd.read_parquet(input_path)

    if args.conformal_params:
        cp = Path(args.conformal_params)
        if cp.exists():
            params = json.loads(cp.read_text(encoding="utf-8"))
            q10_col = "q10_cal" if "q10_cal" in df.columns else "q10_raw"
            q90_col = "q90_cal" if "q90_cal" in df.columns else "q90_raw"
            df = df.copy()
            df["q10_cal"] = df[q10_col] - params["delta_lower"]
            df["q90_cal"] = df[q90_col] + params["delta_upper"]
    thresholds = _load_thresholds(thresholds_path)
    eval_module = _load_eval_module()
    metrics = eval_module.compute_metrics(df, use_gated=args.use_gated)

    failures: List[str] = []
    _check_min(metrics.get("coverage"), thresholds.get("coverage_min", 0.0), "coverage", failures)
    _check_min(metrics.get("keep_precision"), thresholds.get("keep_precision_min", 0.0), "keep_precision", failures)
    _check_min(metrics.get("kill_precision"), thresholds.get("kill_precision_min", 0.0), "kill_precision", failures)
    _check_min(metrics.get("decisive_rate"), thresholds.get("decisive_rate_min", 0.0), "decisive_rate", failures)
    _check_max(metrics.get("stable_kill_rate"), thresholds.get("stable_kill_rate_max", 1.0), "stable_kill_rate", failures)
    _check_max(metrics.get("mae"), thresholds.get("mae_max", 1.0), "mae", failures)

    report = {
        "input": str(input_path),
        "thresholds": thresholds,
        "metrics": metrics,
        "failures": failures,
        "passed": len(failures) == 0,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
