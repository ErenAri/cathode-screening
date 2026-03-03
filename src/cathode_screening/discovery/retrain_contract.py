"""Shared command and path contracts for discovery retraining orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from dataclasses import dataclass
from pathlib import Path


def _load_project_output_dir(base_config: Path) -> Path:
    """Read project.output_dir from YAML config with a no-deps fallback parser."""
    raw = base_config.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw) or {}
        if isinstance(parsed, dict):
            project = parsed.get("project", {})
            if isinstance(project, dict):
                output_dir = project.get("output_dir")
                if output_dir:
                    return Path(str(output_dir))
    except Exception:
        # Fall back to a minimal parser below when yaml is unavailable or invalid.
        pass

    in_project = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_project = stripped.startswith("project:")
            continue
        if in_project and stripped.startswith("output_dir:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            if value and value.lower() != "null":
                return Path(value)
            return Path(".")
    return Path(".")


def validate_required_flags(
    command: Sequence[str],
    required_flags: Iterable[str],
    step_name: str,
) -> None:
    """Ensure generated command includes every required CLI flag."""
    required = list(required_flags)
    missing = [flag for flag in required if flag not in command]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{step_name} command is missing required flags: {missing_str}")


def resolve_ensemble_dir(base_config: Path, run_id: str) -> Path:
    """Resolve expected ensemble directory from training config + run id."""
    output_dir = _load_project_output_dir(base_config)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    return output_dir / "artifacts" / "models" / run_id


def load_ensemble_meta(ensemble_dir: Path) -> dict:
    """Load ensemble metadata after training."""
    meta_path = ensemble_dir / "ensemble_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected ensemble metadata at {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    members = data.get("members")
    if not isinstance(members, list) or not members:
        raise RuntimeError(f"Invalid ensemble metadata, no members at {meta_path}")
    return data


def resolve_member_checkpoint(ensemble_dir: Path, member_entry: dict) -> Path:
    """Resolve a member checkpoint path from ensemble metadata."""
    checkpoint_raw = member_entry.get("checkpoint")
    if not checkpoint_raw:
        raise RuntimeError("Ensemble member metadata missing 'checkpoint'")
    checkpoint = Path(str(checkpoint_raw))
    if checkpoint.is_absolute():
        resolved = checkpoint
    else:
        candidate_in_ensemble = (ensemble_dir / checkpoint).resolve()
        resolved = candidate_in_ensemble if candidate_in_ensemble.exists() else checkpoint.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found for member: {checkpoint_raw}")
    return resolved


@dataclass(frozen=True)
class RetrainArtifactLayout:
    """Canonical retrain artifact paths derived from run id + base config."""

    ensemble_dir: Path
    calibration_dir: Path
    ood_dir: Path
    predictions_dir: Path
    predictions_val_path: Path
    policy_dir: Path
    first_checkpoint: Path


def resolve_retrain_layout(base_config: Path, run_id: str) -> RetrainArtifactLayout:
    """Build canonical layout and validate critical post-train artifacts."""
    ensemble_dir = resolve_ensemble_dir(base_config, run_id)
    meta = load_ensemble_meta(ensemble_dir)
    first_checkpoint = resolve_member_checkpoint(ensemble_dir, meta["members"][0])

    calibration_dir = ensemble_dir / "calibration"
    ood_dir = ensemble_dir / "ood"
    predictions_dir = ensemble_dir / "predictions"
    policy_dir = ensemble_dir / "policy"

    return RetrainArtifactLayout(
        ensemble_dir=ensemble_dir,
        calibration_dir=calibration_dir,
        ood_dir=ood_dir,
        predictions_dir=predictions_dir,
        predictions_val_path=predictions_dir / "ensemble_val.parquet",
        policy_dir=policy_dir,
        first_checkpoint=first_checkpoint,
    )


def build_calibration_command(
    python_executable: str,
    script_path: Path,
    checkpoint_path: Path,
    data_config: Path,
    output_dir: Path,
) -> list[str]:
    return [
        python_executable,
        str(script_path),
        "--checkpoint",
        str(checkpoint_path),
        "--data-config",
        str(data_config),
        "--output-dir",
        str(output_dir),
    ]


def build_ood_command(
    python_executable: str,
    script_path: Path,
    ensemble_dir: Path,
    data_config: Path,
    output_dir: Path,
) -> list[str]:
    return [
        python_executable,
        str(script_path),
        "--ensemble-dir",
        str(ensemble_dir),
        "--data-config",
        str(data_config),
        "--output-dir",
        str(output_dir),
    ]


def build_predictions_command(
    python_executable: str,
    script_path: Path,
    ensemble_dir: Path,
    data_config: Path,
    output_path: Path,
    ood_dir: Path,
    calibration_dir: Path,
    split: str = "val",
) -> list[str]:
    return [
        python_executable,
        str(script_path),
        "--ensemble-dir",
        str(ensemble_dir),
        "--data-config",
        str(data_config),
        "--split",
        split,
        "--output",
        str(output_path),
        "--ood-dir",
        str(ood_dir),
        "--calibration-dir",
        str(calibration_dir),
    ]


def build_policy_command(
    python_executable: str,
    script_path: Path,
    predictions_path: Path,
    run_id: str,
    output_dir: Path,
    mode: str = "dft_followup",
) -> list[str]:
    return [
        python_executable,
        str(script_path),
        "--predictions",
        str(predictions_path),
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
    ]
