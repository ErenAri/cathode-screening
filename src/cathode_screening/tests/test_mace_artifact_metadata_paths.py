"""Checks for portable path formatting in bundled MACE artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
META_PATH = REPO_ROOT / "artifacts" / "models" / "mace_ensemble_v1" / "ensemble_meta.json"


def test_mace_ensemble_meta_has_portable_member_paths():
    if not META_PATH.exists():
        return

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    members = meta.get("members", [])
    assert members, "Expected members in mace ensemble metadata"

    for member in members:
        checkpoint = str(member.get("checkpoint", ""))
        config = str(member.get("config", ""))

        assert "\\" not in checkpoint, f"checkpoint contains backslashes: {checkpoint}"
        assert "\\" not in config, f"config contains backslashes: {config}"
        assert not checkpoint.startswith(
            "artifacts/models/mace_ensemble_v1/"
        ), f"checkpoint should be relative to ensemble dir: {checkpoint}"
        assert not config.startswith(
            "artifacts/models/mace_ensemble_v1/"
        ), f"config should be relative to ensemble dir: {config}"
