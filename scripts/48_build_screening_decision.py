#!/usr/bin/env python3
"""Build a provisional screening decision table from ranked QE results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def to_int(value: str, default: int = 999999) -> int:
    try:
        return int(value)
    except Exception:
        return default


def decide(rank: int, qe_state: str) -> tuple[str, str, str]:
    # Primary production gate:
    # - top 20 resolved -> accept
    # - unresolved in top 20 -> hold
    # - lower ranks become backup/unknown
    if rank <= 20:
        if qe_state == "done":
            return "accept", "primary", "top20_resolved"
        return "hold", "primary", "top20_unresolved"

    if rank <= 30:
        if qe_state == "done":
            return "hold", "backup", "rank21_30_resolved"
        return "unknown", "backup", "rank21_30_unresolved"

    if qe_state == "done":
        return "hold", "longtail", "rank31_50_resolved"
    return "unknown", "longtail", "rank31_50_unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="reports/dft_batch_jarvis_50_mix_final_ranked_estimated.csv",
        help="Ranked CSV with qe_final_state_est column.",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/screening_decision_provisional.csv",
    )
    parser.add_argument(
        "--out-summary",
        default="reports/screening_decision_summary.txt",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.out_csv)
    summary_path = Path(args.out_summary)

    with in_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: to_int(r.get("rank", "")))

    out_rows = []
    decision_counter: Counter[str] = Counter()
    unresolved_top20 = 0

    for row in rows:
        rank = to_int(row.get("rank", ""))
        qe_state = row.get("qe_final_state_est", "missing")
        decision, priority, reason = decide(rank, qe_state)

        if rank <= 20 and qe_state != "done":
            unresolved_top20 += 1

        decision_counter[decision] += 1

        out_row = {
            "rank": row.get("rank", ""),
            "jarvis_id": row.get("jarvis_id", ""),
            "formula": row.get("formula", ""),
            "p_stable": row.get("p_stable", ""),
            "qe_final_state_est": qe_state,
            "qe_status_source_est": row.get("qe_status_source_est", ""),
            "decision": decision,
            "priority_bucket": priority,
            "decision_reason": reason,
        }
        out_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    summary_lines = [
        "Screening Decision Summary (Provisional)",
        f"input={in_path}",
        f"output={out_path}",
        f"total_candidates={len(out_rows)}",
        f"accept={decision_counter.get('accept', 0)}",
        f"hold={decision_counter.get('hold', 0)}",
        f"unknown={decision_counter.get('unknown', 0)}",
        f"top20_unresolved={unresolved_top20}",
        "note=Based on estimated run5 status while billing-disabled blocks exact content reads.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"wrote={out_path}")
    print(f"wrote={summary_path}")
    for key in ("accept", "hold", "unknown"):
        print(f"{key}={decision_counter.get(key, 0)}")
    print(f"top20_unresolved={unresolved_top20}")


if __name__ == "__main__":
    main()
