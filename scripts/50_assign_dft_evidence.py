#!/usr/bin/env python3
"""Assign DFT evidence tiers to a QE campaign and emit summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cathode_screening.evaluation.dft_evidence import assess_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        default="reports/dft_qe_jarvis_50_mix",
        help="QE campaign directory containing manifest.csv and candidate folders.",
    )
    parser.add_argument(
        "--status-csv",
        default="reports/qe_run3_run4_run5_final_status.csv",
        help="CSV with committed QE status snapshots.",
    )
    parser.add_argument(
        "--estimated-status-csv",
        default="reports/qe_run3_run4_run5_estimated_status.csv",
        help="Optional CSV with estimated QE status snapshots.",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/dft_qe_jarvis_50_mix_evidence.csv",
        help="Per-candidate evidence table output.",
    )
    parser.add_argument(
        "--out-summary",
        default="reports/dft_qe_jarvis_50_mix_evidence_summary.json",
        help="Summary JSON output.",
    )
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    status_csv = Path(args.status_csv)
    estimated_status_csv = Path(args.estimated_status_csv)

    records, summary = assess_campaign(
        campaign_dir,
        status_csv=status_csv if status_csv.exists() else None,
        estimated_status_csv=estimated_status_csv if estimated_status_csv.exists() else None,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)

    out_summary = Path(args.out_summary)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote={out_csv}")
    print(f"wrote={out_summary}")
    for tier, count in summary["tier_counts"].items():
        print(f"{tier}={count}")
    for blocker, count in summary["blocker_counts"].items():
        print(f"blocker:{blocker}={count}")


if __name__ == "__main__":
    main()
