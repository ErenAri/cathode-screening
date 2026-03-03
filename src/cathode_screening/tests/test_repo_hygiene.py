"""Repository hygiene checks used by CI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWLIST_PATH = REPO_ROOT / ".ci" / "empty-file-allowlist.txt"


def _load_allowlist() -> set[str]:
    raw = ALLOWLIST_PATH.read_text(encoding="utf-8")
    entries = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _git_tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _tracked_empty_files() -> set[str]:
    empties = set()
    for rel in _git_tracked_files():
        path = REPO_ROOT / rel
        if path.is_file() and path.stat().st_size == 0:
            empties.add(rel)
    return empties


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_no_unallowlisted_tracked_empty_files():
    allowlist = _load_allowlist()
    tracked_empties = _tracked_empty_files()
    unexpected = sorted(tracked_empties - allowlist)
    assert not unexpected, (
        "Found new tracked empty files not in .ci/empty-file-allowlist.txt: "
        + ", ".join(unexpected)
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_allowlist_has_no_stale_entries():
    allowlist = _load_allowlist()
    tracked_empties = _tracked_empty_files()
    stale = sorted(allowlist - tracked_empties)
    assert not stale, (
        "Allowlist contains files that are no longer tracked empty files: "
        + ", ".join(stale)
    )
