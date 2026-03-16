"""
Shadow deployment — run a candidate model alongside production.

Routes a configurable percentage of traffic to a shadow model, logs
both predictions for comparison, and never returns shadow results
to the caller. This enables safe model validation before promotion.

Configuration:
    CATHODE_SHADOW_ENABLED=true
    CATHODE_SHADOW_MODEL_DIR=artifacts/models/mace_ensemble_v2
    CATHODE_SHADOW_TRAFFIC_PERCENT=10  (0-100)
    CATHODE_SHADOW_LOG_DIR=data/shadow_logs

Architecture:
    Request → Production Model → Response (returned to caller)
               ↓ (async, N% traffic)
             Shadow Model → Log comparison (not returned)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class ShadowRunner:
    """Runs a shadow model on a fraction of production traffic.

    Production predictions are unaffected — shadow predictions are
    logged asynchronously for offline comparison.
    """

    def __init__(
        self,
        shadow_predictor,
        traffic_percent: int = 10,
        log_dir: Optional[str] = None,
        max_workers: int = 2,
    ):
        self.shadow_predictor = shadow_predictor
        self.traffic_percent = max(0, min(100, traffic_percent))
        self.log_dir = Path(log_dir or os.getenv("CATHODE_SHADOW_LOG_DIR", "data/shadow_logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="shadow",
        )
        self._lock = threading.Lock()
        self._log_file = None
        self._current_date = None

        # Stats
        self.total_sampled = 0
        self.total_completed = 0
        self.total_errors = 0
        self.agreement_count = 0
        self.disagreement_count = 0

    def should_shadow(self) -> bool:
        """Decide if this request should be shadowed."""
        return random.randint(1, 100) <= self.traffic_percent

    def submit_shadow(
        self,
        structure: Structure,
        material_id: str,
        production_result: Dict[str, Any],
    ) -> None:
        """Submit a shadow prediction asynchronously.

        This never blocks the production path. Errors are swallowed.
        """
        if not self.should_shadow():
            return

        self.total_sampled += 1
        self._executor.submit(
            self._run_shadow,
            structure,
            material_id,
            production_result,
        )

    def _run_shadow(
        self,
        structure: Structure,
        material_id: str,
        production_result: Dict[str, Any],
    ) -> None:
        """Execute shadow prediction and log comparison."""
        start = time.perf_counter()
        try:
            shadow_result = self.shadow_predictor.predict_structure(
                structure, material_id=material_id
            )
            shadow_dict = shadow_result.to_dict()
            latency_s = time.perf_counter() - start

            # Compare decisions
            prod_decision = production_result.get("decision", "MAYBE")
            shadow_decision = shadow_dict.get("decision", "MAYBE")
            agree = prod_decision == shadow_decision

            if agree:
                self.agreement_count += 1
            else:
                self.disagreement_count += 1

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "material_id": material_id,
                "production": {
                    "decision": prod_decision,
                    "ehull_pred": production_result.get("ehull_pred"),
                    "p_stable": production_result.get("p_stable"),
                },
                "shadow": {
                    "decision": shadow_decision,
                    "ehull_pred": shadow_dict.get("ehull_pred"),
                    "p_stable": shadow_dict.get("p_stable"),
                },
                "agree": agree,
                "shadow_latency_s": round(latency_s, 3),
            }

            self._log_comparison(record)
            self.total_completed += 1

        except Exception as exc:
            self.total_errors += 1
            logger.debug("Shadow prediction failed for %s: %s", material_id, exc)

    def _log_comparison(self, record: Dict[str, Any]) -> None:
        """Append comparison record to daily JSONL log."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            if self._current_date != today or self._log_file is None:
                if self._log_file is not None:
                    self._log_file.close()
                path = self.log_dir / f"shadow_{today}.jsonl"
                self._log_file = open(path, "a", encoding="utf-8")
                self._current_date = today

            self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._log_file.flush()

    def stats(self) -> Dict[str, Any]:
        """Return shadow deployment statistics."""
        total = self.agreement_count + self.disagreement_count
        agreement_rate = (
            round(self.agreement_count / total, 3) if total > 0 else None
        )
        return {
            "enabled": self.traffic_percent > 0,
            "traffic_percent": self.traffic_percent,
            "total_sampled": self.total_sampled,
            "total_completed": self.total_completed,
            "total_errors": self.total_errors,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "agreement_rate": agreement_rate,
        }

    def analyze_logs(self, days: int = 7) -> Dict[str, Any]:
        """Analyze recent shadow comparison logs.

        Returns aggregate statistics for promotion decision-making.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        records = []

        for log_file in sorted(self.log_dir.glob("shadow_*.jsonl")):
            # Parse date from filename
            date_str = log_file.stem.replace("shadow_", "")
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    continue
            except ValueError:
                continue

            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        if not records:
            return {"status": "no_data", "days_analyzed": days}

        total = len(records)
        agree = sum(1 for r in records if r.get("agree"))
        disagree = total - agree

        # Decision-level analysis
        decision_matrix = {}
        for r in records:
            prod = r.get("production", {}).get("decision", "MAYBE")
            shadow = r.get("shadow", {}).get("decision", "MAYBE")
            key = f"{prod}→{shadow}"
            decision_matrix[key] = decision_matrix.get(key, 0) + 1

        # Check for dangerous disagreements (prod KEEP but shadow KILL)
        dangerous = sum(1 for r in records
                       if r.get("production", {}).get("decision") == "KEEP"
                       and r.get("shadow", {}).get("decision") == "KILL")

        return {
            "status": "analyzed",
            "days_analyzed": days,
            "total_comparisons": total,
            "agreement_count": agree,
            "disagreement_count": disagree,
            "agreement_rate": round(agree / total, 4) if total else None,
            "decision_matrix": decision_matrix,
            "dangerous_disagreements": dangerous,
            "safe_to_promote": dangerous == 0 and (agree / total >= 0.95 if total else False),
        }

    def shutdown(self) -> None:
        """Shut down the shadow executor."""
        self._executor.shutdown(wait=False)
        with self._lock:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_shadow_runner: Optional[ShadowRunner] = None


def get_shadow_runner() -> Optional[ShadowRunner]:
    """Get or create the shadow runner if enabled.

    Reads:
        CATHODE_SHADOW_ENABLED=true/false
        CATHODE_SHADOW_MODEL_DIR=path/to/shadow/ensemble
        CATHODE_SHADOW_TRAFFIC_PERCENT=10
    """
    global _shadow_runner

    enabled = os.getenv("CATHODE_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None

    if _shadow_runner is not None:
        return _shadow_runner

    shadow_dir = os.getenv("CATHODE_SHADOW_MODEL_DIR", "")
    if not shadow_dir:
        logger.warning("CATHODE_SHADOW_ENABLED=true but CATHODE_SHADOW_MODEL_DIR not set")
        return None

    traffic_pct = int(os.getenv("CATHODE_SHADOW_TRAFFIC_PERCENT", "10"))

    try:
        # Load shadow model using the same adapter
        from cathode_screening.inference.mace_adapter import MACEDecisionService

        # Temporarily override artifacts dir to load shadow model
        original_dir = os.environ.get("CATHODE_ARTIFACTS_DIR")
        os.environ["CATHODE_ARTIFACTS_DIR"] = shadow_dir
        try:
            shadow_predictor = MACEDecisionService.from_env()
        finally:
            if original_dir:
                os.environ["CATHODE_ARTIFACTS_DIR"] = original_dir
            else:
                os.environ.pop("CATHODE_ARTIFACTS_DIR", None)

        _shadow_runner = ShadowRunner(
            shadow_predictor=shadow_predictor,
            traffic_percent=traffic_pct,
        )
        logger.info(
            "Shadow deployment enabled: %d%% traffic to %s",
            traffic_pct, shadow_dir,
        )
        return _shadow_runner

    except Exception as exc:
        logger.warning("Failed to initialize shadow model: %s", exc)
        return None
