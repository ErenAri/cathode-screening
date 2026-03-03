"""Tests for retrain orchestration command/path contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from cathode_screening.discovery.retrain_contract import (
    build_calibration_command,
    build_ood_command,
    build_policy_command,
    build_predictions_command,
    resolve_retrain_layout,
    validate_required_flags,
)


def test_validate_required_flags_accepts_complete_command():
    cmd = ["python", "script.py", "--a", "1", "--b", "2"]
    validate_required_flags(cmd, required_flags=("--a", "--b"), step_name="test step")


def test_validate_required_flags_rejects_missing_flags():
    cmd = ["python", "script.py", "--a", "1"]
    with pytest.raises(ValueError, match="missing required flags"):
        validate_required_flags(cmd, required_flags=("--a", "--b"), step_name="test step")


def test_command_builders_include_required_flags(tmp_path: Path):
    script = tmp_path / "step.py"
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    output = tmp_path / "out"
    ensemble_dir = tmp_path / "ensemble"
    preds = tmp_path / "predictions.parquet"

    calib_cmd = build_calibration_command(
        python_executable="python",
        script_path=script,
        checkpoint_path=checkpoint,
        data_config=config,
        output_dir=output,
    )
    validate_required_flags(
        calib_cmd,
        required_flags=("--checkpoint", "--data-config", "--output-dir"),
        step_name="calibration",
    )

    ood_cmd = build_ood_command(
        python_executable="python",
        script_path=script,
        ensemble_dir=ensemble_dir,
        data_config=config,
        output_dir=output,
    )
    validate_required_flags(
        ood_cmd,
        required_flags=("--ensemble-dir", "--data-config", "--output-dir"),
        step_name="ood",
    )

    pred_cmd = build_predictions_command(
        python_executable="python",
        script_path=script,
        ensemble_dir=ensemble_dir,
        data_config=config,
        output_path=preds,
        ood_dir=output / "ood",
        calibration_dir=output / "calibration",
        split="val",
    )
    validate_required_flags(
        pred_cmd,
        required_flags=(
            "--ensemble-dir",
            "--data-config",
            "--split",
            "--output",
            "--ood-dir",
            "--calibration-dir",
        ),
        step_name="predict",
    )

    policy_cmd = build_policy_command(
        python_executable="python",
        script_path=script,
        predictions_path=preds,
        run_id="run_123",
        output_dir=output / "policy",
        mode="dft_followup",
    )
    validate_required_flags(
        policy_cmd,
        required_flags=("--predictions", "--mode", "--run-id", "--output-dir"),
        step_name="policy",
    )


def test_resolve_retrain_layout_reads_training_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: workspace_out",
                "train:",
                "  seed: 42",
            ]
        ),
        encoding="utf-8",
    )

    run_id = "discovery_test"
    ensemble_dir = tmp_path / "workspace_out" / "artifacts" / "models" / run_id
    member_dir = ensemble_dir / "member_0" / "member_0"
    member_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = member_dir / "best.pt"
    checkpoint.write_bytes(b"pt")
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta_path.write_text(
        '{"members":[{"member_idx":0,"checkpoint":"member_0/member_0/best.pt"}]}',
        encoding="utf-8",
    )

    layout = resolve_retrain_layout(config_path, run_id)

    assert layout.ensemble_dir == ensemble_dir.resolve()
    assert layout.first_checkpoint == checkpoint.resolve()
    assert layout.calibration_dir == ensemble_dir.resolve() / "calibration"
    assert layout.ood_dir == ensemble_dir.resolve() / "ood"
    assert layout.predictions_val_path == ensemble_dir.resolve() / "predictions" / "ensemble_val.parquet"
