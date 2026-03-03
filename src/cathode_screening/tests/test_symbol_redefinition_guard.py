"""Guardrail to prevent adding new script-level redefinitions of core symbols."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWLIST_PATH = REPO_ROOT / ".ci" / "core-symbol-redefinition-allowlist.json"
SCAN_DIRS = ("src", "scripts")
PATTERN = re.compile(
    r"^(class (Normalizer|CGCNN|CGCNNBlock|OODGate)\b|def composition_fingerprint\b)",
    re.MULTILINE,
)


def _collect_matches() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            raw = path.read_text(encoding="utf-8")
            for match in PATTERN.finditer(raw):
                symbol = match.group(1)
                found.setdefault(symbol, set()).add(rel)
    return found


def test_core_symbol_redefinitions_are_allowlisted():
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    found = _collect_matches()

    unexpected: list[str] = []
    for symbol, paths in found.items():
        allowed = set(allowlist.get(symbol, []))
        new_paths = sorted(paths - allowed)
        for rel in new_paths:
            unexpected.append(f"{symbol} -> {rel}")

    assert not unexpected, (
        "Found new core symbol redefinitions not in allowlist: "
        + ", ".join(unexpected)
    )


def test_redefinition_allowlist_has_no_stale_entries():
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    found = _collect_matches()

    stale: list[str] = []
    for symbol, allowed_paths in allowlist.items():
        active = found.get(symbol, set())
        for rel in sorted(set(allowed_paths) - active):
            stale.append(f"{symbol} -> {rel}")

    assert not stale, (
        "Redefinition allowlist has stale entries: " + ", ".join(stale)
    )
