"""
Campaign manifest creation for discovery runs.

Includes model/dataset hashes and calibrator parameters for reproducibility.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable
import hashlib

from pymatgen.io.cif import CifWriter


def _hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def create_manifest(
    campaign_id: str,
    candidates: Iterable[dict],
    model_path: str | Path,
    dataset_path: str | Path,
    calibrator,
    thresholds: Dict,
    seed: int,
) -> Dict:
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)

    manifest = {
        "campaign_id": campaign_id,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_hash": _hash_file(model_path),
        "dataset_hash": _hash_file(dataset_path),
        "calibrator_params": {
            "platt_a": calibrator.platt_a,
            "platt_b": calibrator.platt_b,
            "stability_threshold_meV": calibrator.threshold * 1000,
            "sigma_floor": calibrator.SIGMA_FLOOR,
            "z_clip": calibrator.Z_CLIP,
        },
        "thresholds": thresholds,
        "candidates": [
            {
                "rank": c["rank"],
                "cif": CifWriter(c["structure"]).write_string(),
                "p_stable": c["p_stable"],
                "sigma": c["sigma"],
            }
            for c in candidates
        ],
    }

    return manifest
