import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

def create_manifest(
    campaign_id: str,
    candidates: List[Dict[str, Any]],
    model_path: str,
    dataset_path: str,
    calibrator: Any,
    thresholds: Dict[str, float],
    seed: int,
) -> Dict[str, Any]:
    """
    Create auditable campaign manifest dict.
    
    Args:
        candidates: List of dicts with 'structure' (Structure), 'rank', 'p_stable', etc.
    """
    # Hash critical artifacts
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    with open(dataset_path, "rb") as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
    # Serialize candidates (Top-K only assumed passed in)
    serialized_candidates = []
    for c in candidates:
        struct = c.get("structure")
        cif_str = ""
        if struct and isinstance(struct, Structure):
            try:
                cif_str = CifWriter(struct).write_string()
            except:
                cif_str = "CIF_GENERATION_FAILED"
        
        serialized_candidates.append({
            "rank": c.get("rank"),
            "cif": cif_str,
            "p_stable": c.get("p_stable"),
            "sigma": c.get("sigma"),
        })

    manifest = {
        "campaign_id": campaign_id,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_hash": model_hash,
        "dataset_hash": dataset_hash,
        "calibrator_params": {
            "platt_a": getattr(calibrator, "platt_a", None),
            "platt_b": getattr(calibrator, "platt_b", None),
            "stability_threshold_meV": getattr(calibrator, "threshold", 0) * 1000,
            "sigma_floor": getattr(calibrator, "SIGMA_FLOOR", None),
            "z_clip": getattr(calibrator, "Z_CLIP", None),
        },
        "thresholds": thresholds,
        "candidates": serialized_candidates
    }
    
    return manifest

def save_manifest(manifest: Dict, path: str):
    """Save manifest to JSON file."""
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
