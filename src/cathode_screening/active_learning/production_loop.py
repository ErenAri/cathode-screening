"""
Production active learning loop for CathodeScreen.

Bridges the existing AL framework with the production API to create a
closed-loop system where:

1. MAYBE predictions are automatically queued for DFT validation
2. DFT results are ingested as ground truth labels
3. The model is periodically retrained on the growing labeled pool
4. Acquisition functions prioritize the most informative candidates

This module provides:
- ``FeedbackIngester``: Accepts DFT ground-truth labels from external
  sources (LIMS, manual upload, API) and writes them to the labeled pool.
- ``ProductionALOrchestrator``: Coordinates the full cycle from prediction
  feedback through retraining triggers.
- Celery tasks for async orchestration.

Environment variables:
    CATHODE_AL_ENABLED          Enable production AL loop (default: false)
    CATHODE_AL_RETRAIN_THRESHOLD  Min new labels before triggering retrain (default: 50)
    CATHODE_AL_ACQUISITION_FN   Acquisition function (default: expected_improvement)
    CATHODE_AL_BATCH_SIZE       Candidates per AL batch (default: 20)
    CATHODE_AL_FEEDBACK_DIR     Directory for feedback files (default: data/feedback)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cathode_screening.active_learning")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ALConfig:
    """Production active learning configuration."""

    enabled: bool = False
    retrain_threshold: int = 50
    acquisition_fn: str = "expected_improvement"
    batch_size: int = 20
    feedback_dir: str = "data/feedback"
    stability_threshold: float = 0.05  # eV/atom

    @classmethod
    def from_env(cls) -> "ALConfig":
        return cls(
            enabled=os.getenv(
                "CATHODE_AL_ENABLED", "false"
            ).strip().lower() in {"1", "true", "yes"},
            retrain_threshold=int(os.getenv("CATHODE_AL_RETRAIN_THRESHOLD", "50")),
            acquisition_fn=os.getenv("CATHODE_AL_ACQUISITION_FN", "expected_improvement"),
            batch_size=int(os.getenv("CATHODE_AL_BATCH_SIZE", "20")),
            feedback_dir=os.getenv("CATHODE_AL_FEEDBACK_DIR", "data/feedback"),
            stability_threshold=float(os.getenv("CATHODE_AL_STABILITY_THRESHOLD", "0.05")),
        )


# ---------------------------------------------------------------------------
# Feedback data model
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """A single ground-truth label from DFT or experiment."""

    material_id: str
    ehull_true: float  # eV/atom, ground truth
    source: str  # "dft", "experiment", "literature"
    source_id: Optional[str] = None  # Job ID, DOI, LIMS sample ID
    submitted_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc).isoformat()


@dataclass
class FeedbackPool:
    """Collection of ground-truth feedback labels."""

    records: List[FeedbackRecord] = field(default_factory=list)
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def add(self, record: FeedbackRecord) -> bool:
        """Add a feedback record. Returns True if new, False if duplicate."""
        if record.material_id in self._index:
            # Update existing
            idx = self._index[record.material_id]
            self.records[idx] = record
            return False
        self._index[record.material_id] = len(self.records)
        self.records.append(record)
        return True

    def since(self, last_count: int) -> List[FeedbackRecord]:
        """Get records added since index `last_count`."""
        return self.records[last_count:]

    @property
    def n_new_since_last_train(self) -> int:
        """Convenience: total records (caller tracks last train count)."""
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["material_id", "ehull_true", "source"])
        return pd.DataFrame([asdict(r) for r in self.records])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.records]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeedbackPool":
        pool = cls()
        if not path.exists():
            return pool
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            meta = entry.pop("metadata", {})
            rec = FeedbackRecord(**entry, metadata=meta) if "metadata" not in entry else FeedbackRecord(**entry)
            pool.add(rec)
        return pool


# ---------------------------------------------------------------------------
# Feedback ingester
# ---------------------------------------------------------------------------

class FeedbackIngester:
    """Accepts ground-truth labels from multiple sources and writes to pool.

    Sources:
      - Direct API call (``ingest_single``)
      - Batch CSV/JSON upload (``ingest_batch``)
      - LIMS webhook (``ingest_from_lims``)
    """

    def __init__(self, config: Optional[ALConfig] = None):
        self.config = config or ALConfig.from_env()
        self._feedback_path = Path(self.config.feedback_dir) / "feedback_pool.json"
        self._pool = FeedbackPool.load(self._feedback_path)

    @property
    def pool(self) -> FeedbackPool:
        return self._pool

    def ingest_single(
        self,
        material_id: str,
        ehull_true: float,
        source: str = "api",
        source_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Ingest a single ground-truth label. Returns True if new."""
        record = FeedbackRecord(
            material_id=material_id,
            ehull_true=ehull_true,
            source=source,
            source_id=source_id,
            metadata=metadata or {},
        )
        is_new = self._pool.add(record)
        self._pool.save(self._feedback_path)
        logger.info(
            "Ingested feedback for %s: ehull=%.4f source=%s (new=%s)",
            material_id, ehull_true, source, is_new,
        )
        return is_new

    def ingest_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest a batch of ground-truth labels.

        Each record must have ``material_id`` and ``ehull_true``.

        Returns:
            Dict with ``new``, ``updated``, and ``total`` counts.
        """
        n_new = 0
        n_updated = 0
        for entry in records:
            record = FeedbackRecord(
                material_id=entry["material_id"],
                ehull_true=float(entry["ehull_true"]),
                source=entry.get("source", "batch"),
                source_id=entry.get("source_id"),
                metadata=entry.get("metadata", {}),
            )
            if self._pool.add(record):
                n_new += 1
            else:
                n_updated += 1
        self._pool.save(self._feedback_path)
        logger.info("Batch ingestion: %d new, %d updated", n_new, n_updated)
        return {"new": n_new, "updated": n_updated, "total": len(self._pool.records)}

    def ingest_from_lims(self, lims_payload: Dict[str, Any]) -> Dict[str, int]:
        """Ingest feedback from a LIMS webhook payload.

        Expected format::

            {
                "samples": [
                    {"sample_id": "...", "material_id": "...", "ehull_measured": 0.023},
                    ...
                ],
                "lims_system": "LabWare",
                "batch_id": "B-2026-0042"
            }
        """
        records = []
        for sample in lims_payload.get("samples", []):
            records.append({
                "material_id": sample["material_id"],
                "ehull_true": sample.get("ehull_measured", sample.get("ehull_true", 0.0)),
                "source": "lims",
                "source_id": sample.get("sample_id"),
                "metadata": {
                    "lims_system": lims_payload.get("lims_system", ""),
                    "batch_id": lims_payload.get("batch_id", ""),
                },
            })
        return self.ingest_batch(records)


# ---------------------------------------------------------------------------
# Production AL orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ALCycleResult:
    """Result of one production AL cycle."""

    cycle_number: int
    n_new_labels: int
    retrain_triggered: bool
    candidates_selected: List[str]
    acquisition_scores: Dict[str, float]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ProductionALOrchestrator:
    """Coordinates the production active learning loop.

    Workflow:
      1. Monitor feedback pool for new labels
      2. When threshold is reached, select next batch of candidates
         using acquisition function on current model predictions
      3. Optionally trigger model retraining
      4. Log cycle results to audit trail
    """

    def __init__(
        self,
        config: Optional[ALConfig] = None,
        ingester: Optional[FeedbackIngester] = None,
    ):
        self.config = config or ALConfig.from_env()
        self.ingester = ingester or FeedbackIngester(self.config)
        self._last_train_count = 0
        self._cycle_history: List[ALCycleResult] = []
        self._state_path = Path(self.config.feedback_dir) / "al_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._last_train_count = data.get("last_train_count", 0)
            except Exception:
                pass

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_train_count": self._last_train_count,
            "total_cycles": len(self._cycle_history),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def new_labels_since_train(self) -> int:
        return len(self.ingester.pool.records) - self._last_train_count

    @property
    def should_retrain(self) -> bool:
        return self.new_labels_since_train >= self.config.retrain_threshold

    def select_next_batch(
        self,
        pool_predictions: pd.DataFrame,
        exclude_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Select the next batch of candidates for DFT/experimental validation.

        Args:
            pool_predictions: DataFrame with columns:
                material_id, pred_mean (or q50), pred_std (or epistemic_std)
            exclude_ids: Material IDs already labeled (auto-excluded).

        Returns:
            DataFrame with selected candidates and acquisition scores.
        """
        from cathode_screening.active_learning import acquisition

        df = pool_predictions.copy()

        # Exclude already-labeled materials
        labeled_ids = set(r.material_id for r in self.ingester.pool.records)
        if exclude_ids:
            labeled_ids.update(exclude_ids)
        df = df[~df["material_id"].isin(labeled_ids)].copy()

        if df.empty:
            logger.warning("No unlabeled candidates remaining in pool")
            return pd.DataFrame()

        mean_col = "pred_mean" if "pred_mean" in df.columns else "q50"
        std_col = "pred_std" if "pred_std" in df.columns else "epistemic_std"

        mean_pred = df[mean_col].values
        std_pred = df[std_col].values
        k = min(self.config.batch_size, len(df))

        if self.config.acquisition_fn == "expected_improvement":
            selected_idx = acquisition.expected_improvement(
                mean_pred, std_pred, self.config.stability_threshold, k
            )
        elif self.config.acquisition_fn == "uncertainty":
            selected_idx = acquisition.uncertainty_sampling(std_pred, k)
        elif self.config.acquisition_fn == "ucb":
            selected_idx = acquisition.upper_confidence_bound(
                mean_pred, std_pred, kappa=2.0, k=k
            )
        elif self.config.acquisition_fn == "pi":
            selected_idx = acquisition.probability_of_improvement(
                mean_pred, std_pred, self.config.stability_threshold, k
            )
        else:
            selected_idx = acquisition.random_sampling(len(df), k)

        selected = df.iloc[selected_idx].copy()
        # Compute acquisition scores for logging
        if self.config.acquisition_fn == "expected_improvement":
            from scipy.stats import norm
            z = (self.config.stability_threshold - mean_pred[selected_idx]) / (std_pred[selected_idx] + 1e-8)
            selected["acquisition_score"] = (
                (self.config.stability_threshold - mean_pred[selected_idx]) * norm.cdf(z)
                + std_pred[selected_idx] * norm.pdf(z)
            )
        else:
            selected["acquisition_score"] = std_pred[selected_idx]

        return selected.sort_values("acquisition_score", ascending=False)

    def run_cycle(
        self,
        pool_predictions: Optional[pd.DataFrame] = None,
    ) -> ALCycleResult:
        """Run one AL cycle: check feedback, select candidates, maybe retrain.

        Args:
            pool_predictions: If provided, used to select next batch.
                Otherwise only checks retrain trigger.

        Returns:
            ALCycleResult with cycle details.
        """
        n_new = self.new_labels_since_train
        retrain_triggered = self.should_retrain

        candidates = []
        scores = {}

        if pool_predictions is not None and not pool_predictions.empty:
            selected = self.select_next_batch(pool_predictions)
            if not selected.empty:
                candidates = selected["material_id"].tolist()
                scores = dict(zip(
                    selected["material_id"],
                    selected["acquisition_score"].round(6),
                ))

        if retrain_triggered:
            self._last_train_count = len(self.ingester.pool.records)
            self._save_state()
            logger.info(
                "Retrain triggered: %d new labels since last train", n_new
            )

        result = ALCycleResult(
            cycle_number=len(self._cycle_history) + 1,
            n_new_labels=n_new,
            retrain_triggered=retrain_triggered,
            candidates_selected=candidates,
            acquisition_scores=scores,
        )
        self._cycle_history.append(result)
        return result

    def get_status(self) -> Dict[str, Any]:
        """Return current AL loop status."""
        return {
            "enabled": self.config.enabled,
            "total_feedback": len(self.ingester.pool.records),
            "new_since_last_train": self.new_labels_since_train,
            "retrain_threshold": self.config.retrain_threshold,
            "should_retrain": self.should_retrain,
            "acquisition_fn": self.config.acquisition_fn,
            "batch_size": self.config.batch_size,
            "total_cycles": len(self._cycle_history),
        }
