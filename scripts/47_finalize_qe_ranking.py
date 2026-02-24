#!/usr/bin/env python3
"""Merge QE run statuses and produce final ranked screening tables."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Dict, Iterable, List, Tuple

GSUTIL_BIN = which("gsutil") or which("gsutil.cmd") or "gsutil"


@dataclass
class Status:
    candidate_rel: str
    state: str
    return_code: str
    job_done: str
    source: str


def run_cmd(cmd: List[str]) -> str:
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr.strip()}"
        )
    return cp.stdout


def parse_status_text(text: str, source: str) -> Dict[str, Status]:
    status_map: Dict[str, Status] = {}
    curr: Dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("candidate_rel="):
            if "candidate_rel" in curr:
                s = Status(
                    candidate_rel=curr.get("candidate_rel", ""),
                    state=curr.get("state", ""),
                    return_code=curr.get("return_code", ""),
                    job_done=curr.get("job_done", ""),
                    source=source,
                )
                if s.candidate_rel:
                    status_map[s.candidate_rel] = s
            curr = {"candidate_rel": line.split("=", 1)[1]}
        elif "=" in line:
            k, v = line.split("=", 1)
            curr[k] = v

    if "candidate_rel" in curr:
        s = Status(
            candidate_rel=curr.get("candidate_rel", ""),
            state=curr.get("state", ""),
            return_code=curr.get("return_code", ""),
            job_done=curr.get("job_done", ""),
            source=source,
        )
        if s.candidate_rel:
            status_map[s.candidate_rel] = s

    return status_map


def fetch_status_map(status_prefix: str, source: str) -> Dict[str, Status]:
    # Pull all status files in one stream (fast enough for small batches).
    cmd = [GSUTIL_BIN, "cat", f"{status_prefix.rstrip('/') }/*.status"]
    try:
        text = run_cmd(cmd)
    except RuntimeError as exc:
        msg = str(exc)
        if "matched no objects" in msg or "No URLs matched" in msg:
            return {}
        raise
    return parse_status_text(text, source=source)


def normalize_state(s: Status) -> str:
    if s.job_done == "1":
        return "done"
    return s.state or "unknown"


def candidate_jarvis_id(candidate_rel: str) -> str:
    m = re.match(r"^\d+_(.+)$", candidate_rel)
    return m.group(1) if m else candidate_rel


def load_ranked_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def summarize(states: Iterable[str]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ranked-csv",
        default="reports/dft_batch_jarvis_50_mix.csv",
        help="Base ranked candidate CSV (must include rank,jarvis_id).",
    )
    p.add_argument(
        "--run3-status",
        default="gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run3/progress/status",
    )
    p.add_argument(
        "--run4-status",
        default="gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run4_timeout_retry1/progress/status",
    )
    p.add_argument(
        "--run5-status",
        default="gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run5_timeout_retry2/progress/status",
    )
    p.add_argument(
        "--out-status-csv",
        default="reports/qe_run3_run4_run5_final_status.csv",
    )
    p.add_argument(
        "--out-ranked-csv",
        default="reports/dft_batch_jarvis_50_mix_final_ranked.csv",
    )
    args = p.parse_args()

    ranked_rows = load_ranked_csv(Path(args.ranked_csv))
    rank_map = {row["jarvis_id"]: row for row in ranked_rows}

    run3 = fetch_status_map(args.run3_status, "run3")
    run4 = fetch_status_map(args.run4_status, "run4_retry1")
    run5 = fetch_status_map(args.run5_status, "run5_retry2")

    final_rows: List[dict] = []

    for cand_rel, s3 in sorted(run3.items(), key=lambda kv: kv[0]):
        jarvis_id = candidate_jarvis_id(cand_rel)
        chosen = s3
        final_state = normalize_state(s3)

        if final_state != "done" and cand_rel in run4:
            chosen = run4[cand_rel]
            final_state = normalize_state(chosen)

        if final_state != "done" and cand_rel in run5:
            chosen = run5[cand_rel]
            final_state = normalize_state(chosen)

        base = rank_map.get(jarvis_id, {})
        final_rows.append(
            {
                "candidate_rel": cand_rel,
                "jarvis_id": jarvis_id,
                "rank": base.get("rank", ""),
                "p_stable": base.get("p_stable", ""),
                "formula": base.get("formula", ""),
                "final_state": final_state,
                "source": chosen.source,
                "return_code": chosen.return_code,
                "job_done": chosen.job_done,
            }
        )

    # Write status table
    out_status = Path(args.out_status_csv)
    out_status.parent.mkdir(parents=True, exist_ok=True)
    with out_status.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)

    # Write ranked table enriched with QE status.
    qe_by_jarvis = {r["jarvis_id"]: r for r in final_rows}
    ranked_out_rows = []
    for row in ranked_rows:
        q = qe_by_jarvis.get(row["jarvis_id"], {})
        out = dict(row)
        out["qe_final_state"] = q.get("final_state", "missing")
        out["qe_status_source"] = q.get("source", "")
        out["qe_job_done"] = q.get("job_done", "")
        out["qe_return_code"] = q.get("return_code", "")
        ranked_out_rows.append(out)

    out_ranked = Path(args.out_ranked_csv)
    with out_ranked.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked_out_rows[0].keys()))
        w.writeheader()
        w.writerows(ranked_out_rows)

    # Console summary
    state_pairs = summarize(r["final_state"] for r in final_rows)
    for state, count in state_pairs:
        print(f"{state}={count}")

    unresolved = [r for r in final_rows if r["final_state"] != "done"]
    unresolved_sorted = sorted(unresolved, key=lambda r: to_int(r["rank"], 999999))
    top10_unresolved = sum(1 for r in unresolved if to_int(r["rank"], 999999) <= 10)
    top20_unresolved = sum(1 for r in unresolved if to_int(r["rank"], 999999) <= 20)

    print(f"total={len(final_rows)}")
    print(f"unresolved={len(unresolved)}")
    print(f"top10_unresolved={top10_unresolved}")
    print(f"top20_unresolved={top20_unresolved}")
    print(f"wrote={out_status}")
    print(f"wrote={out_ranked}")

    if unresolved_sorted:
        print("unresolved_candidates:")
        for r in unresolved_sorted:
            print(f"{r['rank']}\t{r['candidate_rel']}\t{r['final_state']}")


if __name__ == "__main__":
    main()
