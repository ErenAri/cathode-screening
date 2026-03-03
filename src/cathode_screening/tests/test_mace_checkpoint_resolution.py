"""Tests for robust MACE checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

mace_adapter = pytest.importorskip("cathode_screening.inference.mace_adapter")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")
    return path


def test_resolve_checkpoint_relative_to_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    ensemble_dir = artifacts_dir / "models" / "mace_ensemble_v1"
    ckpt = _touch(ensemble_dir / "member_0" / "member_0" / "best.pt")

    resolved = mace_adapter._resolve_checkpoint_path(
        checkpoint_raw="member_0/member_0/best.pt",
        ensemble_dir=ensemble_dir,
        artifacts_dir=artifacts_dir,
    )

    assert resolved.resolve() == ckpt.resolve()


def test_resolve_checkpoint_with_artifacts_models_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    ensemble_dir = artifacts_dir / "models" / "mace_ensemble_v1"
    ckpt = _touch(ensemble_dir / "member_1" / "member_1" / "best.pt")

    resolved = mace_adapter._resolve_checkpoint_path(
        checkpoint_raw=r"artifacts\models\mace_ensemble_v1\member_1\member_1\best.pt",
        ensemble_dir=ensemble_dir,
        artifacts_dir=artifacts_dir,
    )

    assert resolved.resolve() == ckpt.resolve()


def test_resolve_checkpoint_raises_with_attempted_paths(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    ensemble_dir = artifacts_dir / "models" / "mace_ensemble_v1"

    with pytest.raises(FileNotFoundError, match="Tried:"):
        mace_adapter._resolve_checkpoint_path(
            checkpoint_raw="artifacts/models/mace_ensemble_v1/member_x/best.pt",
            ensemble_dir=ensemble_dir,
            artifacts_dir=artifacts_dir,
        )
