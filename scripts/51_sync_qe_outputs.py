#!/usr/bin/env python3
"""Sync QE output files into a committed campaign directory and rerun the evidence audit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from shutil import which
from typing import Iterable

from cathode_screening.evaluation.qe_sync import (
    CANDIDATE_REL_RE,
    SyncResult,
    load_campaign_candidates,
    summarize_sync,
    sync_local_outputs,
)


GSUTIL_BIN = which("gsutil") or which("gsutil.cmd") or "gsutil"
DEFAULT_GCS_PREFIXES = (
    "gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run3",
    "gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run4_timeout_retry1",
    "gs://cathode-screening-training/qe_results/dft_qe_jarvis_50_mix_precision_run5_timeout_retry2",
)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def list_gcs_files(prefix: str, filenames: set[str]) -> list[str]:
    cp = run_cmd([GSUTIL_BIN, "ls", "-r", f"{prefix.rstrip('/')}/**"])
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or cp.stdout.strip())
    uris: list[str] = []
    for raw in cp.stdout.splitlines():
        uri = raw.strip()
        if not uri or uri.endswith("/"):
            continue
        if Path(uri).name in filenames:
            uris.append(uri)
    return uris


def detect_candidate_from_uri(uri: str, known_candidates: set[str]) -> str | None:
    for part in Path(uri).parts:
        if part in known_candidates:
            return part
        if CANDIDATE_REL_RE.match(part) and part in known_candidates:
            return part
    return None


def sync_gcs_outputs(
    campaign_dir: Path,
    prefixes: Iterable[str],
    *,
    filenames: tuple[str, ...],
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[SyncResult]:
    known_candidates = load_campaign_candidates(campaign_dir)
    wanted = set(filenames)
    results: list[SyncResult] = []
    seen: set[tuple[str, str]] = set()

    for prefix in prefixes:
        for uri in list_gcs_files(prefix, wanted):
            filename = Path(uri).name
            candidate_rel = detect_candidate_from_uri(uri, known_candidates)
            if candidate_rel is None:
                continue
            key = (candidate_rel, filename)
            if key in seen:
                continue
            seen.add(key)

            target_path = campaign_dir / candidate_rel / filename
            if target_path.exists() and not overwrite:
                results.append(
                    SyncResult(
                        candidate_rel=candidate_rel,
                        filename=filename,
                        source=uri,
                        target=str(target_path),
                        action="skipped_existing",
                    )
                )
                continue

            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                cp = run_cmd([GSUTIL_BIN, "cp", uri, str(target_path)])
                if cp.returncode != 0:
                    results.append(
                        SyncResult(
                            candidate_rel=candidate_rel,
                            filename=filename,
                            source=uri,
                            target=str(target_path),
                            action="copy_failed",
                        )
                    )
                    continue

            results.append(
                SyncResult(
                    candidate_rel=candidate_rel,
                    filename=filename,
                    source=uri,
                    target=str(target_path),
                    action="copied",
                )
            )
    return results


def rerun_evidence_audit() -> subprocess.CompletedProcess[str]:
    return run_cmd(["python", "scripts/50_assign_dft_evidence.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        default="reports/dft_qe_jarvis_50_mix",
        help="QE campaign directory containing manifest.csv and candidate folders.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Local directory to scan recursively for candidate_rel/pw.out and pw.err.",
    )
    parser.add_argument(
        "--gcs-prefix",
        action="append",
        default=[],
        help="GCS prefix to scan recursively with gsutil, e.g. gs://bucket/qe_results/run3.",
    )
    parser.add_argument(
        "--use-default-gcs-prefixes",
        action="store_true",
        help="Use the three default QE run prefixes already referenced by scripts/47_finalize_qe_ranking.py.",
    )
    parser.add_argument(
        "--filename",
        action="append",
        default=[],
        help="Output filename to sync. Defaults to pw.out and pw.err.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-evidence-audit",
        action="store_true",
        help="Do not rerun scripts/50_assign_dft_evidence.py after syncing.",
    )
    args = parser.parse_args()

    gcs_prefixes = list(args.gcs_prefix)
    if args.use_default_gcs_prefixes:
        gcs_prefixes.extend(prefix for prefix in DEFAULT_GCS_PREFIXES if prefix not in gcs_prefixes)

    if not args.source_dir and not gcs_prefixes:
        raise SystemExit("Provide at least one --source-dir or --gcs-prefix.")

    filenames = tuple(args.filename) if args.filename else ("pw.out", "pw.err")
    campaign_dir = Path(args.campaign_dir)

    results: list[SyncResult] = []
    if args.source_dir:
        results.extend(
            sync_local_outputs(
                campaign_dir,
                [Path(source) for source in args.source_dir],
                filenames=filenames,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    if gcs_prefixes:
        results.extend(
            sync_gcs_outputs(
                campaign_dir,
                gcs_prefixes,
                filenames=filenames,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    summary = summarize_sync(results)
    if summary:
        for action, count in summary.items():
            print(f"{action}={count}")
    else:
        print("no_matches_found=1")

    if results:
        print("synced_candidates:")
        for result in results[:20]:
            print(f"{result.action}\t{result.candidate_rel}\t{result.filename}\t{result.source}")

    if not args.dry_run and not args.skip_evidence_audit:
        audit = rerun_evidence_audit()
        print(audit.stdout.strip())
        if audit.returncode != 0:
            raise SystemExit(audit.stderr.strip() or "Evidence audit failed after sync.")


if __name__ == "__main__":
    main()
