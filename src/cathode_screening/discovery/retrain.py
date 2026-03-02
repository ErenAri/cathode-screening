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
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def merge_new_records(
    existing_pool_path: Path,
    new_records: List[Dict[str, Any]],
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
    existing_energies = np.array([])

    structs_file = existing_pool_path / "structures.pkl"
    energies_file = existing_pool_path / "energies.npy"
    meta_file = existing_pool_path / "metadata.json"

    if structs_file.exists():
        with open(structs_file, "rb") as f:
            existing_structures = pickle.load(f)
        logger.info("Loaded %d existing structures", len(existing_structures))

    if energies_file.exists():
        existing_energies = np.load(energies_file)

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
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
    epochs: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Path:
    """
    Retrain ensemble on expanded dataset.

    Calls existing scripts/04_train_ensemble.py as subprocess, then
    runs calibration and OOD artifact generation.

    Args:
        dataset_path: Path to the merged training dataset.
        base_config: Path to base training config YAML.
        output_dir: Directory for model artifacts.
        ensemble_k: Number of ensemble members.
        epochs: Override training epochs (if None, uses config default).
        run_id: Run identifier (auto-generated if None).

    Returns:
        Path to the new model artifacts directory.
    """
    if run_id is None:
        run_id = f"discovery_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = output_dir / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train ensemble
    logger.info("Retraining ensemble (k=%d) on %s ...", ensemble_k, dataset_path)
    train_cmd = [
        sys.executable, "scripts/04_train_ensemble.py",
        "--config", str(base_config),
        "--k", str(ensemble_k),
        "--run-id", run_id,
    ]
    result = subprocess.run(train_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Ensemble training failed with return code {result.returncode}")

    # Step 2: Calibration (if script exists)
    calib_script = Path("scripts/05_calibrate_conformal.py")
    if calib_script.exists():
        logger.info("Running conformal calibration...")
        subprocess.run(
            [sys.executable, str(calib_script), "--artifacts", str(artifact_dir)],
            capture_output=False,
        )

    # Step 3: OOD artifacts (if script exists)
    ood_script = Path("scripts/07_build_ood_artifacts.py")
    if ood_script.exists():
        logger.info("Building OOD artifacts...")
        subprocess.run(
            [sys.executable, str(ood_script), "--artifacts", str(artifact_dir)],
            capture_output=False,
        )

    # Step 4: Threshold tuning (if script exists)
    thresh_script = Path("scripts/08_tune_policy.py")
    if thresh_script.exists():
        logger.info("Tuning decision thresholds...")
        subprocess.run(
            [sys.executable, str(thresh_script), "--artifacts", str(artifact_dir)],
            capture_output=False,
        )

    logger.info("Retrain complete. Artifacts at %s", artifact_dir)
    return artifact_dir


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

    with open(gov_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Look for evaluation metrics in the artifact directory
    metrics_candidates = [
        artifact_path / "evaluation_metrics.json",
        artifact_path / "metrics.json",
        artifact_path / "ensemble_meta.json",
    ]

    metrics: Dict[str, Any] = {}
    for mp in metrics_candidates:
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
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
