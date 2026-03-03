"""
Generate a public benchmark report from CathodeScreen screening results.

This script aggregates all available evaluation artifacts and produces
a reproducible benchmark report summarizing the full screening pipeline.

Output: reports/benchmark_report.json + reports/benchmark_report.md
"""
from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_csv_decisions(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("decision", "unknown")
            counts[d] = counts.get(d, 0) + 1
    return counts


def build_benchmark() -> Dict[str, Any]:
    """Aggregate all evaluation artifacts into a benchmark report."""
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "CathodeScreen",
        "version": "1.0.0",
    }

    # 1. Ensemble Model Metrics
    meta_path = ARTIFACTS_DIR / "models" / "mace_ensemble_v1" / "ensemble_meta.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
        report["model"] = {
            "type": "MACE-MP-0 fine-tuned ensemble",
            "backbone": "medium",
            "ensemble_size": meta.get("k", 5),
            "created_at": meta.get("created_at"),
            "target": "energy_above_hull (eV/atom)",
            "training_data": {
                "dataset": "mp_cathodes_v1",
                "train": meta.get("members", [{}])[0].get("metrics", {}).get("counts", {}).get("train", 0),
                "val": meta.get("members", [{}])[0].get("metrics", {}).get("counts", {}).get("val", 0),
                "test": meta.get("members", [{}])[0].get("metrics", {}).get("counts", {}).get("test", 0),
            },
            "performance": {
                "val_mae_mean": meta.get("ensemble_stats", {}).get("val_mae_mean"),
                "val_mae_std": meta.get("ensemble_stats", {}).get("val_mae_std"),
                "test_mae_mean": meta.get("ensemble_stats", {}).get("test_mae_mean"),
                "test_mae_std": meta.get("ensemble_stats", {}).get("test_mae_std"),
            },
            "per_member_val_mae": [],
            "per_member_test_mae": [],
        }

        for m in meta.get("members", []):
            metrics = m.get("metrics", {})
            val_met = metrics.get("val", {})
            test_met = metrics.get("test", {})
            report["model"]["per_member_val_mae"].append(round(val_met.get("mae", 0), 4))
            report["model"]["per_member_test_mae"].append(round(test_met.get("mae", 0), 4))

    # 2. Conformal Calibration
    cal_path = ARTIFACTS_DIR / "models" / "mace_ensemble_v1" / "calibration" / "conformal_params.json"
    if cal_path.exists():
        cal = _load_json(cal_path)
        report["calibration"] = {
            "method": cal.get("method", "split_conformal_symmetric"),
            "target_coverage": cal.get("target_coverage", 0.9),
            "raw_coverage": round(cal.get("raw_coverage", 0), 4),
            "calibrated_coverage": round(cal.get("calibrated_coverage_val", 0), 4),
            "conformal_delta": round(cal.get("delta_upper", 0), 4),
            "n_calibration": cal.get("n_val", 0),
        }

    # 3. Cross-Database Screening Performance
    screening_results = {}
    for name, filename in [
        ("h100_ehull", "grounded_win_h100_ehull_ens_v1.json"),
        ("oqmd", "grounded_win_oqmd_ens_v1.json"),
        ("jarvis", "grounded_win_jarvis_ens_v1.json"),
    ]:
        path = REPORTS_DIR / filename
        if path.exists():
            screening_results[name] = _load_json(path)

    if screening_results:
        report["cross_database_screening"] = screening_results

    # 4. DFT Validation Campaign
    decision_path = REPORTS_DIR / "screening_decision_provisional.csv"
    if decision_path.exists():
        decisions = _count_csv_decisions(decision_path)
        total = sum(decisions.values())
        report["dft_validation"] = {
            "total_candidates_screened": total,
            "decisions": decisions,
            "accept_rate": round(decisions.get("accept", 0) / total * 100, 1) if total > 0 else 0,
            "source": "JARVIS cathode subset (50 candidates)",
            "dft_method": "Quantum ESPRESSO (PBE, PAW)",
        }

    # 5. Summary
    summary_path = REPORTS_DIR / "screening_decision_summary.txt"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            report["screening_summary_raw"] = f.read()

    # 6. Pipeline Description
    report["pipeline"] = {
        "steps": [
            "1. Structure ingestion (CIF/pymatgen)",
            "2. MACE-MP-0 backbone feature extraction (frozen)",
            "3. 5-member ensemble quantile regression (q10, q50, q90)",
            "4. Conformal calibration (90% coverage guarantee)",
            "5. OOD detection (composition, embedding, disagreement gates)",
            "6. Decision policy (KEEP/MAYBE/KILL)",
            "7. Multi-property analysis (voltage, capacity, energy density)",
            "8. Composite screening score (stability + capacity + voltage + energy)",
        ],
        "infrastructure": {
            "training": "RTX 2060 (local, ~45 min per member)",
            "inference": "Render (CPU, ~2s per structure)",
            "frontend": "Vercel (Next.js)",
            "api": "FastAPI + MACE-MP-0 ensemble",
        },
    }

    # 7. Key Claims (for sales deck / publications)
    perf = report.get("model", {}).get("performance", {})
    val_mae_mean = perf.get("val_mae_mean") or 0
    val_mae_std = perf.get("val_mae_std") or 0
    test_mae_mean = perf.get("test_mae_mean") or 0
    test_mae_std = perf.get("test_mae_std") or 0
    cal_cov = report.get("calibration", {}).get("calibrated_coverage") or 0
    train_n = report.get("model", {}).get("training_data", {}).get("train") or 0
    dft_rate = report.get("dft_validation", {}).get("accept_rate") or 0

    report["key_claims"] = {
        "ensemble_val_mae": f"{val_mae_mean:.4f} +/- {val_mae_std:.4f} eV/atom",
        "ensemble_test_mae": f"{test_mae_mean:.4f} +/- {test_mae_std:.4f} eV/atom",
        "calibration_coverage": f"{cal_cov * 100:.1f}%",
        "training_data_size": f"{train_n:,}",
        "dft_accept_rate": f"{dft_rate:.0f}%",
        "jarvis_precision_at_100": screening_results.get("jarvis", {}).get("precision_100", "N/A"),
        "jarvis_enrichment_1pct": screening_results.get("jarvis", {}).get("ef_1pct", "N/A"),
    }

    return report


def write_markdown(report: Dict[str, Any], output: Path) -> None:
    """Write the benchmark report as Markdown."""
    model = report.get("model", {})
    perf = model.get("performance", {})
    cal = report.get("calibration", {})
    dft = report.get("dft_validation", {})
    claims = report.get("key_claims", {})
    screening = report.get("cross_database_screening", {})

    # Safe numeric access
    val_mae_mean = perf.get("val_mae_mean") or 0
    val_mae_std = perf.get("val_mae_std") or 0
    test_mae_mean = perf.get("test_mae_mean") or 0
    test_mae_std = perf.get("test_mae_std") or 0
    cal_cov = cal.get("calibrated_coverage") or 0
    cal_target = cal.get("target_coverage") or 0.9
    cal_delta = cal.get("conformal_delta") or 0

    lines = [
        "# CathodeScreen Benchmark Report",
        "",
        f"Generated: {report.get('generated_at', 'N/A')}",
        "",
        "## Model",
        "",
        f"- **Architecture**: {model.get('type', 'N/A')}",
        f"- **Ensemble size**: {model.get('ensemble_size', 'N/A')} members",
        f"- **Target**: {model.get('target', 'N/A')}",
        f"- **Training set**: {model.get('training_data', {}).get('train', 'N/A'):,} structures",
        f"- **Validation set**: {model.get('training_data', {}).get('val', 'N/A'):,} structures",
        f"- **Test set**: {model.get('training_data', {}).get('test', 'N/A'):,} structures",
        "",
        "## Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Val MAE | {val_mae_mean:.4f} +/- {val_mae_std:.4f} eV/atom |",
        f"| Test MAE | {test_mae_mean:.4f} +/- {test_mae_std:.4f} eV/atom |",
        f"| Conformal coverage | {cal_cov * 100:.1f}% (target: {cal_target * 100:.0f}%) |",
        f"| Conformal delta | {cal_delta:.4f} eV |",
        "",
        "### Per-Member MAE",
        "",
        "| Member | Val MAE | Test MAE |",
        "|--------|---------|----------|",
    ]

    val_maes = model.get("per_member_val_mae", [])
    test_maes = model.get("per_member_test_mae", [])
    for i, (v, t) in enumerate(zip(val_maes, test_maes)):
        lines.append(f"| {i} | {v:.4f} | {t:.4f} |")

    lines.extend([
        "",
        "## Cross-Database Screening",
        "",
        "| Dataset | EF@1% | Precision@100 | AUPRC |",
        "|---------|-------|---------------|-------|",
    ])

    for name, data in screening.items():
        lines.append(
            f"| {name} | {data.get('ef_1pct', 'N/A')} | {data.get('precision_100', 'N/A')} | {data.get('auprc', 'N/A')} |"
        )

    if dft:
        lines.extend([
            "",
            "## DFT Validation Campaign",
            "",
            f"- **Total candidates**: {dft.get('total_candidates_screened', 0)}",
            f"- **Source**: {dft.get('source', 'N/A')}",
            f"- **DFT method**: {dft.get('dft_method', 'N/A')}",
            f"- **Accept rate**: {dft.get('accept_rate', 0):.0f}%",
            "",
            "| Decision | Count |",
            "|----------|-------|",
        ])
        for d, c in dft.get("decisions", {}).items():
            lines.append(f"| {d} | {c} |")

    lines.extend([
        "",
        "## Multi-Property Screening",
        "",
        "In addition to ML-predicted E_hull, CathodeScreen computes:",
        "",
        "- **Theoretical gravimetric capacity** (mAh/g) from Li stoichiometry",
        "- **Average voltage proxy** (V vs Li/Li+) from TM-anion empirical correlations",
        "- **Gravimetric energy density** (Wh/kg) = capacity x voltage",
        "- **Volumetric capacity** (mAh/cm^3) from crystal density",
        "- **Composite screening score** (weighted: 35% stability, 25% capacity, 15% voltage, 25% energy density)",
        "",
        "## Pipeline",
        "",
    ])

    for step in report.get("pipeline", {}).get("steps", []):
        lines.append(f"- {step}")

    lines.extend([
        "",
        "## Infrastructure",
        "",
    ])

    for k, v in report.get("pipeline", {}).get("infrastructure", {}).items():
        lines.append(f"- **{k}**: {v}")

    lines.extend([
        "",
        "---",
        "",
        "## Key Claims",
        "",
    ])

    for k, v in claims.items():
        lines.append(f"- **{k}**: {v}")

    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    report = build_benchmark()

    # Write JSON
    json_out = REPORTS_DIR / "benchmark_report.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report: {json_out}")

    # Write Markdown
    md_out = REPORTS_DIR / "benchmark_report.md"
    write_markdown(report, md_out)
    print(f"Markdown report: {md_out}")


if __name__ == "__main__":
    main()
