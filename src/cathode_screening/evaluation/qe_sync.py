"""Utilities for syncing QE output files into a committed campaign directory."""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


CANDIDATE_REL_RE = re.compile(r"^\d+_.+$")


@dataclass(frozen=True)
class SyncResult:
    candidate_rel: str
    filename: str
    source: str
    target: str
    action: str


def load_campaign_candidates(campaign_dir: Path) -> set[str]:
    """Return candidate folder names listed in the campaign manifest."""

    manifest_path = campaign_dir / "manifest.csv"
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: set[str] = set()
    for row in rows:
        raw_path = (row.get("path") or "").replace("\\", "/").strip()
        if raw_path:
            candidates.add(Path(raw_path).name)
    return candidates


def detect_candidate_rel(path: Path, known_candidates: set[str]) -> str | None:
    """Find the candidate_rel segment in a path if it belongs to this campaign."""

    for part in path.parts:
        if part in known_candidates:
            return part
        if CANDIDATE_REL_RE.match(part) and (not known_candidates or part in known_candidates):
            return part
    return None


def collect_local_outputs(
    source_dirs: Sequence[Path],
    *,
    known_candidates: set[str],
    filenames: Sequence[str],
) -> Dict[tuple[str, str], Path]:
    """Index candidate output files from one or more local source directories."""

    found: Dict[tuple[str, str], Path] = {}
    wanted = {name for name in filenames}

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name not in wanted:
                continue
            candidate_rel = detect_candidate_rel(path, known_candidates)
            if candidate_rel is None:
                continue
            key = (candidate_rel, path.name)
            if key not in found:
                found[key] = path
    return found


def sync_local_outputs(
    campaign_dir: Path,
    source_dirs: Sequence[Path],
    *,
    filenames: Sequence[str] = ("pw.out", "pw.err"),
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[SyncResult]:
    """Copy candidate output files from local directories into the campaign."""

    known_candidates = load_campaign_candidates(campaign_dir)
    indexed = collect_local_outputs(source_dirs, known_candidates=known_candidates, filenames=filenames)

    results: list[SyncResult] = []

    for (candidate_rel, filename), source_path in sorted(indexed.items()):
        target_path = campaign_dir / candidate_rel / filename
        if target_path.exists() and not overwrite:
            results.append(
                SyncResult(
                    candidate_rel=candidate_rel,
                    filename=filename,
                    source=str(source_path),
                    target=str(target_path),
                    action="skipped_existing",
                )
            )
            continue

        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        results.append(
            SyncResult(
                candidate_rel=candidate_rel,
                filename=filename,
                source=str(source_path),
                target=str(target_path),
                action="copied",
            )
        )

    return results


def summarize_sync(results: Iterable[SyncResult]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for result in results:
        summary[result.action] = summary.get(result.action, 0) + 1
    return dict(sorted(summary.items()))
