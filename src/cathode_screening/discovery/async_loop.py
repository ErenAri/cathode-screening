"""Async discovery loop primitives (job tracking + mock DFT outcomes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional
import uuid

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DiscoveryJobState:
    job_id: str
    campaign_name: str
    cycle: int
    mode: str
    status: str = "queued"  # queued | running | completed | failed
    stage: str = "wait_dft"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    selected_count: int = 0
    processed_count: int = 0
    stable_found: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "campaign_name": self.campaign_name,
            "cycle": self.cycle,
            "mode": self.mode,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "selected_count": self.selected_count,
            "processed_count": self.processed_count,
            "stable_found": self.stable_found,
            "metrics": self.metrics,
        }


class DiscoveryJobStore:
    """In-memory job store for discovery cycle execution."""

    def __init__(self) -> None:
        self._jobs: Dict[str, DiscoveryJobState] = {}
        self._campaign_index: Dict[str, List[str]] = {}
        self._lock = Lock()

    def create(self, campaign_name: str, cycle: int, mode: str, selected_count: int) -> DiscoveryJobState:
        with self._lock:
            job_id = f"disc_{campaign_name}_{cycle}_{uuid.uuid4().hex[:10]}"
            job = DiscoveryJobState(
                job_id=job_id,
                campaign_name=campaign_name,
                cycle=cycle,
                mode=mode,
                selected_count=selected_count,
            )
            self._jobs[job_id] = job
            self._campaign_index.setdefault(campaign_name, []).append(job_id)
            return job

    def update(self, job_id: str, **changes: Any) -> DiscoveryJobState:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown discovery job: {job_id}")
            job = self._jobs[job_id]
            for key, value in changes.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return job

    def get(self, job_id: str) -> Optional[DiscoveryJobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_campaign(self, campaign_name: str) -> List[DiscoveryJobState]:
        with self._lock:
            ids = list(self._campaign_index.get(campaign_name, []))
            return [self._jobs[i] for i in ids if i in self._jobs]

    def latest_for_campaign(self, campaign_name: str) -> Optional[DiscoveryJobState]:
        jobs = self.list_for_campaign(campaign_name)
        if not jobs:
            return None
        return jobs[-1]


def simulate_dft_outcomes(
    candidate_rows: Iterable[Dict[str, Any]] | pd.DataFrame,
    id_col: str = "material_id",
    seed: int = 17,
    stability_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Simulate DFT ehull outcomes from model predictions.

    The simulation adds controlled stochastic drift around q50/pred_ehull and
    uses epistemic uncertainty as variance proxy.
    """
    if isinstance(candidate_rows, pd.DataFrame):
        rows = candidate_rows.to_dict(orient="records")
    else:
        rows = list(candidate_rows)

    rng = np.random.default_rng(seed)
    outcomes: Dict[str, float] = {}

    for row in rows:
        material_id = str(row.get(id_col, ""))
        if not material_id:
            continue
        q50 = float(row.get("q50", row.get("pred_ehull", 0.20)))
        epistemic = float(row.get("epistemic_std", 0.10))
        p_stable = float(row.get("p_stable", 0.50))

        # Exploration-aware drift:
        # - higher epistemic => wider uncertainty
        # - high p_stable gently biases towards lower ehull
        drift = rng.normal(0.0, max(0.01, epistemic * 0.35))
        stable_bias = (0.5 - p_stable) * 0.03
        dft_ehull = max(0.0, q50 + drift + stable_bias)
        outcomes[material_id] = round(float(dft_ehull), 6)

    return outcomes


def summarize_outcomes(outcomes: Dict[str, float], stability_threshold: float = 0.05) -> Dict[str, Any]:
    if not outcomes:
        return {"count": 0, "stable_count": 0, "stable_rate": 0.0, "mean_ehull": None}
    values = np.array(list(outcomes.values()), dtype=float)
    stable_count = int(np.sum(values < stability_threshold))
    return {
        "count": int(values.size),
        "stable_count": stable_count,
        "stable_rate": round(stable_count / float(values.size), 4),
        "mean_ehull": round(float(values.mean()), 6),
    }
