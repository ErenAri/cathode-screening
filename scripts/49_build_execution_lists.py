#!/usr/bin/env python3
"""Create compact execution lists from screening_decision_provisional.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def to_int(value: str, default: int = 999999) -> int:
    try:
        return int(value)
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="reports/screening_decision_provisional.csv",
    )
    parser.add_argument(
        "--out-accept",
        default="reports/screening_execution_accept.csv",
    )
    parser.add_argument(
        "--out-must-resolve",
        default="reports/screening_execution_must_resolve_top20.csv",
    )
    parser.add_argument(
        "--out-compact",
        default="reports/screening_execution_compact.csv",
    )
    args = parser.parse_args()

    with Path(args.input_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: to_int(r.get("rank", "")))

    keep_cols = [
        "rank",
        "jarvis_id",
        "formula",
        "p_stable",
        "qe_final_state_est",
        "evidence_tier",
        "evidence_label",
        "evidence_blockers",
        "evidence_next_step",
        "decision",
        "decision_reason",
    ]

    accept_rows = [
        {k: r.get(k, "") for k in keep_cols}
        for r in rows
        if r.get("decision") == "accept"
    ]

    must_resolve_rows = [
        {k: r.get(k, "") for k in keep_cols}
        for r in rows
        if r.get("decision") == "hold" and to_int(r.get("rank", "")) <= 20
    ]

    compact_rows: list[dict] = []
    for r in accept_rows:
        row = dict(r)
        row["action"] = "screen_now"
        compact_rows.append(row)
    for r in must_resolve_rows:
        row = dict(r)
        row["action"] = "resolve_qe_first"
        compact_rows.append(row)

    compact_cols = keep_cols + ["action"]

    write_csv(Path(args.out_accept), accept_rows, keep_cols)
    write_csv(Path(args.out_must_resolve), must_resolve_rows, keep_cols)
    write_csv(Path(args.out_compact), compact_rows, compact_cols)

    print(f"accept={len(accept_rows)}")
    print(f"must_resolve_top20={len(must_resolve_rows)}")
    print(f"wrote={args.out_accept}")
    print(f"wrote={args.out_must_resolve}")
    print(f"wrote={args.out_compact}")


if __name__ == "__main__":
    main()
