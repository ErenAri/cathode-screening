#!/usr/bin/env python3
"""Monitor a Vertex AI QE custom job plus GCS progress snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from shutil import which
from typing import Dict, Optional

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}

GCLOUD_BIN = which("gcloud") or which("gcloud.cmd") or "gcloud"
GSUTIL_BIN = which("gsutil") or which("gsutil.cmd") or "gsutil"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def read_job(project: str, region: str, job_ref: str) -> Dict[str, object]:
    job_name = job_ref
    if job_ref.isdigit():
        job_name = f"projects/{project}/locations/{region}/customJobs/{job_ref}"
    cp = run_cmd(
        [
            GCLOUD_BIN,
            "ai",
            "custom-jobs",
            "describe",
            job_name,
            "--region",
            region,
            "--project",
            project,
            "--format=json",
        ]
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or cp.stdout.strip())
    return json.loads(cp.stdout)


def parse_key_value_text(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def read_progress(progress_gcs: str) -> Optional[Dict[str, str]]:
    uri = progress_gcs.rstrip("/") + "/progress/progress.txt"
    cp = run_cmd([GSUTIL_BIN, "cat", uri])
    if cp.returncode != 0:
        return None
    return parse_key_value_text(cp.stdout)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="cathode-screening")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--job-id",
        required=True,
        help="Numeric job id or full customJobs resource name.",
    )
    parser.add_argument(
        "--progress-gcs",
        required=True,
        help="Base output path, e.g. gs://bucket/qe_results/run3",
    )
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        try:
            job = read_job(args.project, args.region, args.job_id)
        except Exception as exc:  # pragma: no cover
            print(f"[{utc_now()}] ERROR reading job: {exc}")
            if args.once:
                return
            time.sleep(args.interval_sec)
            continue

        state = str(job.get("state", "UNKNOWN"))
        start_time = str(job.get("startTime", ""))
        end_time = str(job.get("endTime", ""))
        progress = read_progress(args.progress_gcs)

        if progress is None:
            print(
                f"[{utc_now()}] state={state} start={start_time or '-'} "
                f"end={end_time or '-'} progress=not_uploaded_yet"
            )
        else:
            print(
                f"[{utc_now()}] state={state} "
                f"done={progress.get('done', '?')} "
                f"timeout={progress.get('timeout', '?')} "
                f"failed={progress.get('failed', '?')} "
                f"running={progress.get('running', '?')} "
                f"pending={progress.get('pending', '?')} "
                f"pct={progress.get('percent', '?')}%"
            )

        if args.once or state in TERMINAL_STATES:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
