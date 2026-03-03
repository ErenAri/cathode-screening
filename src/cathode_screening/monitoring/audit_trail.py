"""
Persistent audit trail for prediction logging.

Every prediction is logged as a JSONL record with:
- Timestamp (ISO 8601)
- Request ID
- Input hash (SHA-256 of CIF content)
- Model version / ensemble metadata
- Full prediction output
- Cathode properties
- Latency

This enables reproducibility tracking, regulatory compliance,
and post-hoc analysis of screening campaigns.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AUDIT_DIR = Path(os.getenv("CATHODE_AUDIT_DIR", "data/audit"))
_MAX_FILE_SIZE_MB = int(os.getenv("CATHODE_AUDIT_MAX_FILE_MB", "50"))


class AuditTrail:
    """Thread-safe JSONL prediction logger."""

    def __init__(
        self,
        audit_dir: Optional[Path] = None,
        model_version: Optional[str] = None,
        ensemble_id: Optional[str] = None,
    ):
        self.audit_dir = audit_dir or _AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version or os.getenv("CATHODE_MODEL_VERSION", "unknown")
        self.ensemble_id = ensemble_id or os.getenv("CATHODE_ENSEMBLE_ID", "unknown")
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._record_count = 0

    def _get_log_file(self) -> Path:
        """Get current log file, rotating if needed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = self.audit_dir / f"predictions_{today}.jsonl"

        if self._current_file != target:
            if self._file_handle is not None:
                self._file_handle.close()
            self._current_file = target
            self._file_handle = open(target, "a", encoding="utf-8")
            self._record_count = 0

        # Rotate if file too large
        if target.exists() and target.stat().st_size > _MAX_FILE_SIZE_MB * 1024 * 1024:
            if self._file_handle is not None:
                self._file_handle.close()
            rotated = self.audit_dir / f"predictions_{today}_{int(time.time())}.jsonl"
            target.rename(rotated)
            self._file_handle = open(target, "a", encoding="utf-8")
            self._record_count = 0

        return target

    def log_prediction(
        self,
        request_id: str,
        input_hash: str,
        material_id: str,
        formula: str,
        prediction: Dict[str, Any],
        cathode_properties: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        client_id: Optional[str] = None,
        mode: str = "dft_followup",
    ) -> None:
        """Log a single prediction to the audit trail."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "input_hash": input_hash,
            "model_version": self.model_version,
            "ensemble_id": self.ensemble_id,
            "client_id": client_id,
            "material_id": material_id,
            "formula": formula,
            "mode": mode,
            "prediction": {
                "ehull_pred": prediction.get("ehull_pred"),
                "ehull_lower": prediction.get("ehull_lower"),
                "ehull_upper": prediction.get("ehull_upper"),
                "p_stable": prediction.get("p_stable"),
                "decision": prediction.get("decision"),
                "decision_confidence": prediction.get("decision_confidence"),
                "uncertainty_epistemic": prediction.get("uncertainty_epistemic"),
                "uncertainty_total": prediction.get("uncertainty_total"),
                "ood_flag": prediction.get("ood_flag"),
                "ood_score": prediction.get("ood_score"),
            },
            "cathode_properties": cathode_properties,
            "latency_ms": round(latency_ms, 2),
        }

        with self._lock:
            try:
                self._get_log_file()
                if self._file_handle is not None:
                    self._file_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    self._file_handle.flush()
                    self._record_count += 1
            except Exception as exc:
                logger.warning("Failed to write audit record: %s", exc)

    def get_recent(self, n: int = 50) -> List[Dict]:
        """Read the most recent N predictions from the audit trail."""
        records: List[Dict] = []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.audit_dir / f"predictions_{today}.jsonl"

        if not log_file.exists():
            # Try yesterday
            from datetime import timedelta
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            log_file = self.audit_dir / f"predictions_{yesterday}.jsonl"

        if not log_file.exists():
            return []

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed to read audit trail: %s", exc)

        return records

    def get_stats(self) -> Dict:
        """Get summary statistics from the audit trail."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.audit_dir / f"predictions_{today}.jsonl"

        stats: Dict[str, Any] = {
            "date": today,
            "total_predictions": 0,
            "decisions": {"KEEP": 0, "MAYBE": 0, "KILL": 0},
            "ood_flagged": 0,
            "avg_latency_ms": 0.0,
        }

        if not log_file.exists():
            return stats

        latencies = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    stats["total_predictions"] += 1
                    decision = rec.get("prediction", {}).get("decision", "MAYBE")
                    stats["decisions"][decision] = stats["decisions"].get(decision, 0) + 1
                    if rec.get("prediction", {}).get("ood_flag"):
                        stats["ood_flagged"] += 1
                    latencies.append(rec.get("latency_ms", 0))
        except Exception as exc:
            logger.warning("Failed to compute audit stats: %s", exc)

        if latencies:
            stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)

        return stats

    def close(self):
        """Close the file handle."""
        with self._lock:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None


def compute_input_hash(content: bytes) -> str:
    """Compute SHA-256 hash of input content for reproducibility tracking."""
    return hashlib.sha256(content).hexdigest()[:16]


# Module-level singleton
_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    """Get or create the global audit trail instance."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail
