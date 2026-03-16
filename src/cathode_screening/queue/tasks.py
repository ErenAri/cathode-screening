"""
Celery tasks for async prediction and monitoring.

Tasks:
    predict_structure_task: Single CIF prediction via worker
    predict_batch_task:     Batch CIF predictions via worker
    compute_drift_task:     Periodic drift detection
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import shared_task

logger = logging.getLogger(__name__)

# Lazy-loaded predictor singleton per worker process
_worker_predictor = None


def _get_predictor():
    """Get or create the model predictor in the worker process."""
    global _worker_predictor
    if _worker_predictor is None:
        import sys
        src_path = Path(__file__).resolve().parent.parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        model_type = os.getenv("CATHODE_MODEL_TYPE", "mace").strip().lower()
        if model_type == "mace":
            from cathode_screening.inference.mace_adapter import MACEDecisionService
            _worker_predictor = MACEDecisionService.from_env()
        elif model_type == "chgnet":
            from cathode_screening.inference.chgnet_adapter import CHGNetDecisionService
            _worker_predictor = CHGNetDecisionService.from_env()
        else:
            from cathode_screening.inference.predictor import DecisionService
            _worker_predictor = DecisionService.from_env()

        logger.info("Worker predictor loaded: %s", type(_worker_predictor).__name__)
    return _worker_predictor


@shared_task(
    name="cathode_screening.queue.tasks.predict_structure_task",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
)
def predict_structure_task(self, cif_content: str, material_id: str = "uploaded") -> Dict[str, Any]:
    """Predict stability for a single CIF structure asynchronously.

    Args:
        cif_content: CIF file content as string
        material_id: Identifier for the material

    Returns:
        Prediction result dict with ehull, decision, uncertainty, etc.
    """
    from pymatgen.core import Structure

    start = time.perf_counter()
    try:
        structure = Structure.from_str(cif_content, fmt="cif")
        predictor = _get_predictor()
        result = predictor.predict_structure(structure, material_id=material_id)

        latency_s = time.perf_counter() - start
        output = result.to_dict()
        output["latency_s"] = round(latency_s, 3)
        output["worker_id"] = self.request.hostname
        return output

    except Exception as exc:
        logger.error("Prediction task failed for %s: %s", material_id, exc)
        raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc


@shared_task(
    name="cathode_screening.queue.tasks.predict_batch_task",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
    acks_late=True,
)
def predict_batch_task(
    self,
    cif_contents: List[str],
    material_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Predict stability for a batch of CIF structures asynchronously.

    Args:
        cif_contents: List of CIF file contents as strings
        material_ids: Optional list of material identifiers

    Returns:
        Dict with predictions list, errors list, and timing info.
    """
    from pymatgen.core import Structure

    if material_ids is None:
        material_ids = [f"batch-{i}" for i in range(len(cif_contents))]

    start = time.perf_counter()
    predictions = []
    errors = []
    structures = []
    valid_ids = []

    # Parse all structures first
    for i, (cif, mid) in enumerate(zip(cif_contents, material_ids)):
        try:
            structure = Structure.from_str(cif, fmt="cif")
            structures.append(structure)
            valid_ids.append(mid)
        except Exception as exc:
            errors.append({
                "material_id": mid,
                "index": i,
                "error": str(exc),
            })

    # Batch predict
    if structures:
        try:
            predictor = _get_predictor()
            results = predictor.predict_structures(
                structures, material_ids=valid_ids
            )
            predictions = [r.to_dict() for r in results]
        except Exception as exc:
            logger.error("Batch prediction failed: %s", exc)
            errors.append({"error": f"Batch inference failed: {exc}"})

    latency_s = time.perf_counter() - start
    return {
        "predictions": predictions,
        "errors": errors,
        "n_processed": len(predictions),
        "n_errors": len(errors),
        "latency_s": round(latency_s, 3),
        "worker_id": self.request.hostname,
    }


@shared_task(name="cathode_screening.queue.tasks.compute_drift_task")
def compute_drift_task(
    baseline_path: Optional[str] = None,
    current_path: Optional[str] = None,
    psi_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Compute drift metrics and trigger alerts if threshold exceeded.

    Called periodically by Celery beat or on-demand after model updates.
    """
    import numpy as np
    import pandas as pd

    baseline_path = baseline_path or os.getenv(
        "CATHODE_DRIFT_BASELINE", "data/predictions/ensemble_soap_loco_test.parquet"
    )
    current_path = current_path or os.getenv(
        "CATHODE_DRIFT_CURRENT", "data/predictions/latest.parquet"
    )

    bp = Path(baseline_path)
    cp = Path(current_path)

    if not bp.exists() or not cp.exists():
        return {
            "status": "skipped",
            "reason": f"Missing files: baseline={bp.exists()}, current={cp.exists()}",
        }

    baseline_df = pd.read_parquet(bp)
    current_df = pd.read_parquet(cp)

    # Compute PSI for key columns
    columns = ["q50", "epistemic_std", "p_stable"]
    columns = [c for c in columns if c in baseline_df.columns and c in current_df.columns]

    drifted = []
    results = {}

    for col in columns:
        base_vals = baseline_df[col].dropna().to_numpy()
        curr_vals = current_df[col].dropna().to_numpy()

        if base_vals.size < 2 or curr_vals.size < 2:
            continue

        # PSI computation
        edges = np.quantile(base_vals, np.linspace(0, 1, 11))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue

        base_hist, _ = np.histogram(base_vals, bins=edges)
        curr_hist, _ = np.histogram(curr_vals, bins=edges)
        eps = 1e-6
        base_dist = base_hist / max(base_hist.sum(), 1)
        curr_dist = curr_hist / max(curr_hist.sum(), 1)
        psi = float(np.sum((base_dist - curr_dist) * np.log((base_dist + eps) / (curr_dist + eps))))

        results[col] = {"psi": round(psi, 4)}
        if psi >= psi_threshold:
            drifted.append(col)

    payload = {
        "status": "completed",
        "columns": results,
        "drifted_columns": drifted,
        "retrain_recommended": bool(drifted),
    }

    # Trigger alert if drift detected
    if drifted:
        try:
            from cathode_screening.monitoring.drift_alerter import send_drift_alert
            send_drift_alert(payload)
        except Exception as exc:
            logger.warning("Failed to send drift alert: %s", exc)
            payload["alert_error"] = str(exc)

    return payload
