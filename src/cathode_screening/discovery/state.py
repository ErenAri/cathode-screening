"""
Campaign state manager for discovery engine.

Tracks the full lifecycle of a discovery campaign: cycles, stages,
DFT queries, labeled pool, and model artifacts. JSON-serialized for
checkpoint/resume across failures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STAGES = ("screen", "select", "submit_dft", "wait_dft", "ingest", "retrain", "report")


@dataclass
class CycleRecord:
    """Record for a single discovery cycle."""

    cycle: int
    started_at: str = ""
    completed_at: Optional[str] = None
    stage: str = "screen"
    candidates_selected: List[str] = field(default_factory=list)
    dft_job_id: Optional[str] = None
    dft_gcs_output: Optional[str] = None
    dft_results: Dict[str, float] = field(default_factory=dict)
    n_stable_found: int = 0
    model_artifact_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Intermediate data paths (for resume)
    predictions_path: Optional[str] = None


@dataclass
class CampaignState:
    """Persistent state for a discovery campaign."""

    campaign_name: str
    created_at: str
    pool_source: str = "ensemble_soap_loco_test"
    ranking_strategy: str = "balanced"
    total_dft_queries: int = 0
    cycles: List[CycleRecord] = field(default_factory=list)
    pool_labeled_ids: List[str] = field(default_factory=list)
    pool_remaining_ids: List[str] = field(default_factory=list)
    current_cycle: int = 0
    current_stage: str = "screen"
    current_model_artifacts: Optional[str] = None

    # --- construction ---

    @classmethod
    def new(cls, campaign_name: str, pool_ids: List[str]) -> CampaignState:
        now = datetime.now(timezone.utc).isoformat()
        state = cls(
            campaign_name=campaign_name,
            created_at=now,
            pool_remaining_ids=list(pool_ids),
        )
        state._ensure_cycle()
        return state

    # --- stage machine ---

    def _ensure_cycle(self) -> None:
        """Ensure a CycleRecord exists for the current cycle."""
        while len(self.cycles) <= self.current_cycle:
            self.cycles.append(
                CycleRecord(
                    cycle=len(self.cycles),
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
            )

    @property
    def current_cycle_record(self) -> CycleRecord:
        self._ensure_cycle()
        return self.cycles[self.current_cycle]

    def advance(self, next_stage: str) -> None:
        """Move to *next_stage* within the current cycle."""
        if next_stage not in STAGES:
            raise ValueError(f"Unknown stage {next_stage!r}; must be one of {STAGES}")
        self.current_stage = next_stage
        self.current_cycle_record.stage = next_stage
        logger.info(
            "Campaign %s: cycle %d -> stage %s",
            self.campaign_name,
            self.current_cycle,
            next_stage,
        )

    def advance_cycle(self) -> None:
        """Complete the current cycle and move to the next."""
        self.current_cycle_record.completed_at = datetime.now(timezone.utc).isoformat()
        self.current_cycle += 1
        self.current_stage = "screen"
        self._ensure_cycle()
        logger.info(
            "Campaign %s: advanced to cycle %d",
            self.campaign_name,
            self.current_cycle,
        )

    def record_dft_results(self, results: Dict[str, float]) -> None:
        """Record DFT results and update pool bookkeeping."""
        rec = self.current_cycle_record
        rec.dft_results = results
        rec.n_stable_found = sum(1 for v in results.values() if v < 0.05)

        labeled_set = set(self.pool_labeled_ids)
        remaining_set = set(self.pool_remaining_ids)
        for mid in results:
            labeled_set.add(mid)
            remaining_set.discard(mid)
        self.pool_labeled_ids = sorted(labeled_set)
        self.pool_remaining_ids = sorted(remaining_set)
        self.total_dft_queries += len(results)

    # --- persistence ---

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.debug("Saved campaign state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CampaignState:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        cycles = [CycleRecord(**c) for c in data.pop("cycles", [])]
        # Reconcile: rebuild from raw dict to handle both cases cleanly
        state_obj = cls(
            campaign_name=data["campaign_name"],
            created_at=data["created_at"],
            pool_source=data.get("pool_source", "ensemble_soap_loco_test"),
            ranking_strategy=data.get("ranking_strategy", "balanced"),
            total_dft_queries=data.get("total_dft_queries", 0),
            cycles=cycles,
            pool_labeled_ids=data.get("pool_labeled_ids", []),
            pool_remaining_ids=data.get("pool_remaining_ids", []),
            current_cycle=data.get("current_cycle", 0),
            current_stage=data.get("current_stage", "screen"),
            current_model_artifacts=data.get("current_model_artifacts"),
        )
        logger.info(
            "Loaded campaign %s: cycle %d, stage %s, %d DFT queries so far",
            state_obj.campaign_name,
            state_obj.current_cycle,
            state_obj.current_stage,
            state_obj.total_dft_queries,
        )
        return state_obj

    # --- summary ---

    def summary(self) -> Dict[str, Any]:
        completed = [c for c in self.cycles if c.completed_at is not None]
        total_stable = sum(c.n_stable_found for c in self.cycles)
        return {
            "campaign_name": self.campaign_name,
            "pool_source": self.pool_source,
            "ranking_strategy": self.ranking_strategy,
            "current_cycle": self.current_cycle,
            "current_stage": self.current_stage,
            "completed_cycles": len(completed),
            "total_dft_queries": self.total_dft_queries,
            "total_stable_found": total_stable,
            "pool_labeled": len(self.pool_labeled_ids),
            "pool_remaining": len(self.pool_remaining_ids),
        }
