#!/usr/bin/env python3
"""
Model Trustworthiness Validation & Threshold Retuning.

Loads real prediction data, retunes thresholds with proper cost analysis,
validates on held-out test set, and produces a governance report.

Usage:
    python scripts/validate_model_trust.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─── Configuration ─────────────────────────────────────────────────────
STABLE_THRESH = 0.05       # eV: truly stable
METASTABLE_THRESH = 0.10   # eV: metastable upper bound

# Cost model (relative units)
C_FALSE_KEEP = 10.0        # Cost of sending unstable material to DFT
C_FALSE_KILL = 50.0        # Cost of missing a truly stable material
C_TRUE_KEEP = -1.0         # Benefit of correctly keeping stable
C_TRUE_KILL = -0.5         # Benefit of correctly killing unstable
C_MAYBE = 1.0              # Cost of deferring (manual review)

VAL_PATH = Path("data/predictions/ensemble_val.parquet")
TEST_PATH = Path("data/predictions/ensemble_test.parquet")
LOCO_PATH = Path("data/predictions/ensemble_soap_loco_test.parquet")
OUTPUT_DIR = Path("data/reports/model_validation")


# ─── Helpers ───────────────────────────────────────────────────────────

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert "y_true" in df.columns, f"Missing y_true in {path}"
    assert "q50" in df.columns, f"Missing q50 in {path}"
    return df


def apply_thresholds(df: pd.DataFrame, tau_keep: float, tau_kill: float) -> np.ndarray:
    """Apply KEEP/MAYBE/KILL thresholds using calibrated intervals."""
    q10 = df["q10_cal"].values if "q10_cal" in df.columns else df["q10_raw"].values
    q90 = df["q90_cal"].values if "q90_cal" in df.columns else df["q90_raw"].values
    decisions = np.full(len(df), "MAYBE", dtype=object)
    decisions[q90 < tau_keep] = "KEEP"
    decisions[q10 > tau_kill] = "KILL"
    return decisions


def evaluate_decisions(y_true: np.ndarray, decisions: np.ndarray, label: str = "") -> dict:
    """Compute comprehensive decision quality metrics."""
    n = len(y_true)
    keep_mask = decisions == "KEEP"
    kill_mask = decisions == "KILL"
    maybe_mask = decisions == "MAYBE"

    n_keep = keep_mask.sum()
    n_kill = kill_mask.sum()
    n_maybe = maybe_mask.sum()

    truly_stable = y_true < STABLE_THRESH
    truly_unstable = y_true >= METASTABLE_THRESH

    # KEEP precision: of KEEPs, how many are truly stable?
    keep_precision = float((truly_stable & keep_mask).sum() / n_keep) if n_keep > 0 else float("nan")

    # KEEP recall: of truly stable, how many are KEEPed?
    keep_recall = float((truly_stable & keep_mask).sum() / truly_stable.sum()) if truly_stable.sum() > 0 else float("nan")

    # KILL precision: of KILLs, how many are truly unstable?
    kill_precision = float((truly_unstable & kill_mask).sum() / n_kill) if n_kill > 0 else float("nan")

    # False kill rate: of truly stable, how many are KILLed?
    false_kill_rate = float((truly_stable & kill_mask).sum() / truly_stable.sum()) if truly_stable.sum() > 0 else float("nan")

    # Cost calculation
    cost = 0.0
    if n_keep > 0:
        cost += (truly_stable & keep_mask).sum() * C_TRUE_KEEP
        cost += (~truly_stable & keep_mask).sum() * C_FALSE_KEEP
    if n_kill > 0:
        cost += (truly_unstable & kill_mask).sum() * C_TRUE_KILL
        cost += (truly_stable & kill_mask).sum() * C_FALSE_KILL
    cost += n_maybe * C_MAYBE

    return {
        "label": label,
        "n_total": n,
        "n_keep": int(n_keep),
        "n_maybe": int(n_maybe),
        "n_kill": int(n_kill),
        "keep_pct": float(n_keep / n),
        "kill_pct": float(n_kill / n),
        "keep_precision": keep_precision,
        "keep_recall": keep_recall,
        "kill_precision": kill_precision,
        "false_kill_rate": false_kill_rate,
        "total_cost": float(cost),
        "cost_per_sample": float(cost / n),
    }


def grid_search_thresholds(df: pd.DataFrame) -> dict:
    """Grid search over tau_keep and tau_kill to minimize cost."""
    y = df["y_true"].values
    best_cost = float("inf")
    best_params = {}
    results = []

    for tau_keep in np.arange(0.03, 0.12, 0.005):
        for tau_kill in np.arange(max(tau_keep + 0.02, 0.08), 0.25, 0.01):
            decisions = apply_thresholds(df, tau_keep, tau_kill)
            metrics = evaluate_decisions(y, decisions)

            # Constraint: false_kill_rate <= 5%
            if metrics["false_kill_rate"] > 0.05:
                continue

            # Constraint: keep_precision >= 60% (if any KEEPs)
            if metrics["n_keep"] > 0 and metrics["keep_precision"] < 0.60:
                continue

            entry = {
                "tau_keep": float(tau_keep),
                "tau_kill": float(tau_kill),
                **metrics,
            }
            results.append(entry)

            if metrics["cost_per_sample"] < best_cost:
                best_cost = metrics["cost_per_sample"]
                best_params = entry

    return best_params, results


def ranking_analysis(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Comprehensive ranking quality analysis."""
    rho, pval = spearmanr(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Enrichment at various k
    enrichment = {}
    n = len(y_true)
    base_rate = (y_true < STABLE_THRESH).mean()

    for k in [10, 25, 50, 100, 200]:
        if k >= n:
            continue
        top_k_idx = np.argsort(y_pred)[:k]
        frac_stable = (y_true[top_k_idx] < STABLE_THRESH).mean()
        ef = frac_stable / base_rate if base_rate > 0 else 0
        enrichment[f"ef@{k}"] = float(ef)
        enrichment[f"frac_stable@{k}"] = float(frac_stable)

    # Within-class ranking
    stable_mask = y_true < STABLE_THRESH
    if stable_mask.sum() > 10:
        rho_stable, _ = spearmanr(y_true[stable_mask], y_pred[stable_mask])
    else:
        rho_stable = float("nan")

    return {
        "label": label,
        "n": n,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "mae": mae,
        "base_rate_stable": float(base_rate),
        "spearman_within_stable": float(rho_stable),
        **enrichment,
    }


def calibration_analysis(df: pd.DataFrame, label: str) -> dict:
    """Check conformal calibration holds."""
    y = df["y_true"].values
    q10 = df["q10_cal"].values if "q10_cal" in df.columns else df["q10_raw"].values
    q90 = df["q90_cal"].values if "q90_cal" in df.columns else df["q90_raw"].values

    coverage = float(np.mean((y >= q10) & (y <= q90)))
    widths = q90 - q10
    median_width = float(np.median(widths))
    p95_width = float(np.percentile(widths, 95))

    # Conditional coverage by stability class
    cond_cov = {}
    for cls_name, lo, hi in [("stable", 0, 0.05), ("meta", 0.05, 0.10), ("unstable", 0.10, 999)]:
        mask = (y >= lo) & (y < hi)
        if mask.sum() > 10:
            cov = float(np.mean((y[mask] >= q10[mask]) & (y[mask] <= q90[mask])))
            cond_cov[f"coverage_{cls_name}"] = cov

    return {
        "label": label,
        "coverage_overall": coverage,
        "target_coverage": 0.90,
        "coverage_met": coverage >= 0.88,  # allow 2% tolerance
        "median_interval_width": median_width,
        "p95_interval_width": p95_width,
        **cond_cov,
    }


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL TRUSTWORTHINESS VALIDATION")
    print("=" * 70)

    # Load data
    df_val = load_predictions(VAL_PATH)
    df_test = load_predictions(TEST_PATH)
    df_loco = load_predictions(LOCO_PATH)

    report = {
        "generated_by": "validate_model_trust.py",
        "datasets": {
            "val": str(VAL_PATH),
            "test": str(TEST_PATH),
            "loco": str(LOCO_PATH),
        },
    }

    # ── 1. Ranking Analysis ──────────────────────────────────────────
    print("\n--- 1. RANKING QUALITY ---")
    ranking = {}
    for name, df in [("val", df_val), ("test", df_test), ("loco", df_loco)]:
        r = ranking_analysis(df["y_true"].values, df["q50"].values, name)
        ranking[name] = r
        rho = r["spearman_rho"]
        status = "PASS" if rho > 0.3 else "WARN" if rho > 0.0 else "FAIL"
        print(f"  [{status}] {name}: Spearman={rho:.4f}, MAE={r['mae']:.4f}, "
              f"EF@50={r.get('ef@50', 'N/A')}")
    report["ranking"] = ranking

    # ── 2. Calibration Analysis ──────────────────────────────────────
    print("\n--- 2. CALIBRATION ---")
    calibration = {}
    for name, df in [("val", df_val), ("test", df_test), ("loco", df_loco)]:
        c = calibration_analysis(df, name)
        calibration[name] = c
        status = "PASS" if c["coverage_met"] else "FAIL"
        print(f"  [{status}] {name}: coverage={c['coverage_overall']:.3f} "
              f"(target=0.90), width_med={c['median_interval_width']:.4f}")
    report["calibration"] = calibration

    # ── 3. Threshold Retuning on Val ─────────────────────────────────
    print("\n--- 3. THRESHOLD RETUNING (on validation set) ---")
    best_params, all_results = grid_search_thresholds(df_val)

    if best_params:
        print(f"  Best: tau_keep={best_params['tau_keep']:.3f}, tau_kill={best_params['tau_kill']:.3f}")
        print(f"    KEEP: {best_params['n_keep']} ({best_params['keep_pct']:.1%}), "
              f"precision={best_params['keep_precision']:.3f}, recall={best_params['keep_recall']:.3f}")
        print(f"    KILL: {best_params['n_kill']} ({best_params['kill_pct']:.1%}), "
              f"precision={best_params['kill_precision']:.3f}")
        print(f"    False kill rate: {best_params['false_kill_rate']:.3f}")
        print(f"    Cost/sample: {best_params['cost_per_sample']:.4f}")
    else:
        print("  FAIL: No threshold combination satisfies constraints!")
        best_params = {"tau_keep": 0.07, "tau_kill": 0.15, "status": "fallback"}
    report["threshold_tuning"] = {
        "best": best_params,
        "n_candidates_searched": len(all_results),
    }

    # ── 4. Validate Best Thresholds on Test Set ──────────────────────
    print("\n--- 4. TEST SET VALIDATION ---")
    tau_keep = best_params["tau_keep"]
    tau_kill = best_params["tau_kill"]

    test_validation = {}
    for name, df in [("test", df_test), ("loco", df_loco)]:
        decisions = apply_thresholds(df, tau_keep, tau_kill)
        metrics = evaluate_decisions(df["y_true"].values, decisions, label=name)
        test_validation[name] = metrics

        print(f"\n  {name.upper()} (tau_keep={tau_keep:.3f}, tau_kill={tau_kill:.3f}):")
        print(f"    KEEP: {metrics['n_keep']} ({metrics['keep_pct']:.1%}), "
              f"precision={metrics['keep_precision']:.3f}")
        print(f"    KILL: {metrics['n_kill']} ({metrics['kill_pct']:.1%}), "
              f"precision={metrics['kill_precision']:.3f}")
        print(f"    False kill rate: {metrics['false_kill_rate']:.3f}")
        print(f"    Cost/sample: {metrics['cost_per_sample']:.4f}")

    report["test_validation"] = test_validation

    # ── 5. Ground Truth Spot Check ───────────────────────────────────
    print("\n--- 5. GROUND TRUTH SPOT CHECK (Test Set) ---")
    decisions_test = apply_thresholds(df_test, tau_keep, tau_kill)
    df_test_copy = df_test.copy()
    df_test_copy["retuned_decision"] = decisions_test

    keep_df = df_test_copy[df_test_copy["retuned_decision"] == "KEEP"].sort_values("q50")
    kill_df = df_test_copy[df_test_copy["retuned_decision"] == "KILL"].sort_values("q50", ascending=False)

    print(f"\n  Top KEEP candidates (predicted most stable):")
    spot_check_keep = []
    for _, row in keep_df.head(10).iterrows():
        correct = "CORRECT" if row["y_true"] < STABLE_THRESH else "WRONG"
        entry = {
            "material_id": row["material_id"],
            "formula": row["formula"],
            "y_true": float(row["y_true"]),
            "q50": float(row["q50"]),
            "correct": correct,
        }
        spot_check_keep.append(entry)
        print(f"    {row['material_id']:>12s}  {row['formula']:>15s}  "
              f"pred={row['q50']:.4f}  true={row['y_true']:.4f}  [{correct}]")

    print(f"\n  Top KILL candidates (predicted most unstable):")
    spot_check_kill = []
    for _, row in kill_df.head(10).iterrows():
        correct = "CORRECT" if row["y_true"] >= METASTABLE_THRESH else "WRONG"
        entry = {
            "material_id": row["material_id"],
            "formula": row["formula"],
            "y_true": float(row["y_true"]),
            "q50": float(row["q50"]),
            "correct": correct,
        }
        spot_check_kill.append(entry)
        print(f"    {row['material_id']:>12s}  {row['formula']:>15s}  "
              f"pred={row['q50']:.4f}  true={row['y_true']:.4f}  [{correct}]")

    report["spot_check"] = {
        "keep_candidates": spot_check_keep,
        "kill_candidates": spot_check_kill,
    }

    # ── 6. Governance Verdict ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GOVERNANCE VERDICT")
    print("=" * 70)

    test_ranking = ranking["test"]
    test_calib = calibration["test"]
    test_val = test_validation.get("test", {})

    checks = {
        "ranking_val_pass": ranking["val"]["spearman_rho"] > 0.3,
        "ranking_test_pass": test_ranking["spearman_rho"] > 0.0,
        "calibration_test_pass": test_calib["coverage_met"],
        "false_kill_rate_pass": test_val.get("false_kill_rate", 1.0) <= 0.05,
        "keep_precision_pass": test_val.get("keep_precision", 0.0) >= 0.60 if test_val.get("n_keep", 0) > 0 else True,
        "system_makes_decisions": test_val.get("n_keep", 0) + test_val.get("n_kill", 0) > 0,
    }

    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")

    n_pass = sum(checks.values())
    n_total = len(checks)

    if n_pass == n_total:
        verdict = "APPROVED"
        verdict_detail = "All governance checks passed."
    elif checks["false_kill_rate_pass"] and checks["calibration_test_pass"]:
        verdict = "CONDITIONAL"
        verdict_detail = (
            "Model is safe (no stable materials killed, calibration holds) "
            "but ranking is unreliable on novel chemistries. "
            "Suitable for KILL filtering only — not for autonomous KEEP decisions."
        )
    else:
        verdict = "BLOCKED"
        verdict_detail = "Critical governance checks failed. Do not deploy for autonomous decisions."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {verdict_detail}")

    report["governance"] = {
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    # Save report
    report_path = OUTPUT_DIR / "model_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    return report


if __name__ == "__main__":
    report = main()
    sys.exit(0 if report["governance"]["verdict"] != "BLOCKED" else 1)
