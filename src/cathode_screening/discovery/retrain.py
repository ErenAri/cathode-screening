"""
Retrain trigger for discovery engine.

Orchestrates model retraining after new DFT data arrives:
- Merges new records into the training pool
- Calls existing ensemble training pipeline as subprocess
- Runs post-training calibration and OOD artifact generation
- Validates retrained model against governance thresholds
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cathode_screening.discovery.retrain_contract import (
    build_calibration_command,
    build_ood_command,
    build_policy_command,
    build_predictions_command,
    resolve_retrain_layout,
    validate_required_flags,
)

logger = logging.getLogger(__name__)


def merge_new_records(
    existing_pool_path: Path,
    new_records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Merge new DFT-validated records into the existing training pool.

    Args:
        existing_pool_path: Path to the existing pool (directory with structures.pkl, etc).
        new_records: List of new training record dicts (from dft_ingest.to_training_records).
        output_path: Directory for the merged dataset.

    Returns:
        Path to the merged dataset directory.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing data
    existing_structures = []
    existing_metadata = []
    structs_file = existing_pool_path / "structures.pkl"
    meta_file = existing_pool_path / "metadata.json"

    if structs_file.exists():
        with open(structs_file, "rb") as f:
            existing_structures = pickle.load(f)
        logger.info("Loaded %d existing structures", len(existing_structures))

    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            existing_metadata = json.load(f)

    # Deduplicate by material_id
    existing_ids = set()
    if existing_metadata:
        existing_ids = {m.get("material_id") for m in existing_metadata if "material_id" in m}
    elif existing_structures:
        existing_ids = {s.get("material_id") for s in existing_structures if isinstance(s, dict)}

    added = 0
    for rec in new_records:
        mid = rec["material_id"]
        if mid in existing_ids:
            logger.debug("Skipping duplicate %s", mid)
            continue

        existing_structures.append(rec)
        existing_ids.add(mid)
        added += 1

    # Save merged dataset
    with open(output_path / "structures.pkl", "wb") as f:
        pickle.dump(existing_structures, f)

    # Rebuild energies array
    energies = []
    metadata_out = []
    for item in existing_structures:
        if isinstance(item, dict):
            energies.append(item.get("energy_above_hull", 0.0))
            metadata_out.append({
                "material_id": item.get("material_id", ""),
                "is_stable": item.get("is_stable", False),
                "source": item.get("source", "unknown"),
            })
        else:
            energies.append(0.0)
            metadata_out.append({})

    np.save(output_path / "energies.npy", np.array(energies, dtype=np.float32))

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)

    merge_info = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "source_pool": str(existing_pool_path),
        "new_records_added": added,
        "new_records_skipped_dup": len(new_records) - added,
        "total_pool_size": len(existing_structures),
    }
    with open(output_path / "merge_info.json", "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2)

    logger.info(
        "Merged pool: %d existing + %d new = %d total",
        len(existing_structures) - added, added, len(existing_structures),
    )
    return output_path


def run_retrain(
    dataset_path: Path,
    base_config: str,
    output_dir: Path,
    ensemble_k: int = 5,
    epochs: int | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Retrain ensemble on expanded dataset.

    Calls existing scripts/04_train_ensemble.py as subprocess, then runs:
    - conformal calibration
    - OOD artifact generation
    - val-split ensemble prediction export
    - decision policy tuning

    Args:
        dataset_path: Path to the merged training dataset.
        base_config: Path to base training config YAML.
        output_dir: Directory for model artifacts.
        ensemble_k: Number of ensemble members.
        epochs: Override training epochs (if None, uses config default).
        run_id: Run identifier (auto-generated if None).

    Returns:
        Path to the new ensemble artifacts directory.
    """
    if run_id is None:
        run_id = f"discovery_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    base_config_path = Path(base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if output_dir:
        logger.info(
            "run_retrain output_dir argument set to %s; canonical artifact paths are "
            "resolved from the training config output_dir.",
            output_dir,
        )

    if epochs is not None:
        logger.warning(
            "epochs override (%d) requested but scripts/04_train_ensemble.py does not accept "
            "an epoch override; using config epochs.",
            epochs,
        )

    def _run_checked(cmd: list[str], step_name: str) -> None:
        logger.info("Running %s: %s", step_name, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"{step_name} failed with return code {result.returncode}: {' '.join(cmd)}"
            )

    # Step 1: Train ensemble
    logger.info("Retraining ensemble (k=%d) on %s ...", ensemble_k, dataset_path)
    train_cmd = [
        sys.executable,
        "scripts/04_train_ensemble.py",
        "--config",
        str(base_config_path),
        "--k", str(ensemble_k),
        "--run-id", run_id,
    ]
    _run_checked(train_cmd, "ensemble training")

    layout = resolve_retrain_layout(base_config_path, run_id)
    logger.info("Resolved retrain artifact layout at %s", layout.ensemble_dir)

    # Step 2: Calibration
    calib_script = Path("scripts/05_calibrate_conformal.py")
    if not calib_script.exists():
        raise FileNotFoundError(f"Required script not found: {calib_script}")
    calib_cmd = build_calibration_command(
        python_executable=sys.executable,
        script_path=calib_script,
        checkpoint_path=layout.first_checkpoint,
        data_config=base_config_path,
        output_dir=layout.calibration_dir,
    )
    validate_required_flags(
        calib_cmd,
        required_flags=("--checkpoint", "--data-config", "--output-dir"),
        step_name="conformal calibration",
    )
    _run_checked(calib_cmd, "conformal calibration")

    # Step 3: OOD artifacts
    ood_script = Path("scripts/07_build_ood_artifacts.py")
    if not ood_script.exists():
        raise FileNotFoundError(f"Required script not found: {ood_script}")
    ood_cmd = build_ood_command(
        python_executable=sys.executable,
        script_path=ood_script,
        ensemble_dir=layout.ensemble_dir,
        data_config=base_config_path,
        output_dir=layout.ood_dir,
    )
    validate_required_flags(
        ood_cmd,
        required_flags=("--ensemble-dir", "--data-config", "--output-dir"),
        step_name="ood artifact build",
    )
    _run_checked(ood_cmd, "ood artifact build")

    # Step 4: Build validation predictions used for policy tuning
    predict_script = Path("scripts/07_predict_ensemble.py")
    if not predict_script.exists():
        raise FileNotFoundError(f"Required script not found: {predict_script}")
    layout.predictions_dir.mkdir(parents=True, exist_ok=True)
    predict_cmd = build_predictions_command(
        python_executable=sys.executable,
        script_path=predict_script,
        ensemble_dir=layout.ensemble_dir,
        data_config=base_config_path,
        output_path=layout.predictions_val_path,
        ood_dir=layout.ood_dir,
        calibration_dir=layout.calibration_dir,
        split="val",
    )
    validate_required_flags(
        predict_cmd,
        required_flags=(
            "--ensemble-dir",
            "--data-config",
            "--split",
            "--output",
            "--ood-dir",
            "--calibration-dir",
        ),
        step_name="ensemble val prediction export",
    )
    _run_checked(predict_cmd, "ensemble val prediction export")

    # Step 5: Threshold tuning
    thresh_script = Path("scripts/08_tune_policy.py")
    if not thresh_script.exists():
        raise FileNotFoundError(f"Required script not found: {thresh_script}")
    layout.policy_dir.mkdir(parents=True, exist_ok=True)
    policy_cmd = build_policy_command(
        python_executable=sys.executable,
        script_path=thresh_script,
        predictions_path=layout.predictions_val_path,
        run_id=run_id,
        output_dir=layout.policy_dir,
        mode="dft_followup",
    )
    validate_required_flags(
        policy_cmd,
        required_flags=("--predictions", "--mode", "--run-id", "--output-dir"),
        step_name="policy tuning",
    )
    _run_checked(policy_cmd, "policy tuning")

    logger.info("Retrain complete. Artifacts at %s", layout.ensemble_dir)
    return layout.ensemble_dir


def validate_model(
    artifact_path: Path,
    governance_path: str = "configs/governance_thresholds.json",
) -> bool:
    """
    Validate a retrained model against governance quality gates.

    Checks metrics against thresholds defined in governance_thresholds.json:
    - coverage_min, keep_precision_min, kill_precision_min
    - stable_kill_rate_max, mae_max, decisive_rate_min

    Args:
        artifact_path: Path to model artifacts directory.
        governance_path: Path to governance thresholds JSON.

    Returns:
        True if model passes all gates, False otherwise.
    """
    gov_path = Path(governance_path)
    if not gov_path.exists():
        logger.warning("Governance thresholds not found at %s, skipping validation", gov_path)
        return True

    with open(gov_path, encoding="utf-8") as f:
        thresholds = json.load(f)

    # Look for evaluation metrics in the artifact directory
    metrics_candidates = [
        artifact_path / "evaluation_metrics.json",
        artifact_path / "metrics.json",
        artifact_path / "ensemble_meta.json",
    ]

    metrics: dict[str, Any] = {}
    for mp in metrics_candidates:
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                metrics = json.load(f)
            break

    if not metrics:
        logger.warning("No evaluation metrics found in %s, skipping validation", artifact_path)
        return True

    # Flatten ensemble_stats if present
    if "ensemble_stats" in metrics:
        metrics.update(metrics["ensemble_stats"])

    passed = True
    checks = [
        ("mae_max", "val_mae_mean", lambda m, t: m <= t, "MAE"),
        ("mae_max", "test_mae_mean", lambda m, t: m <= t, "test MAE"),
    ]

    for threshold_key, metric_key, check_fn, label in checks:
        if threshold_key in thresholds and metric_key in metrics:
            threshold_val = thresholds[threshold_key]
            metric_val = metrics[metric_key]
            if metric_val is not None and not check_fn(metric_val, threshold_val):
                logger.error(
                    "Governance FAIL: %s = %.4f (threshold: %.4f)",
                    label, metric_val, threshold_val,
                )
                passed = False
            else:
                logger.info(
                    "Governance PASS: %s = %s (threshold: %.4f)",
                    label, metric_val, threshold_val,
                )

    # Run validate_release script if available
    validate_script = Path("scripts/12_validate_release.py")
    if validate_script.exists():
        result = subprocess.run(
            [sys.executable, str(validate_script), "--artifacts", str(artifact_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("Release validation script failed: %s", result.stderr.strip())
            passed = False

    return passed
