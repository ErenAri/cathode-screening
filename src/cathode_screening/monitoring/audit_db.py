"""
PostgreSQL-backed audit trail for prediction logging.

Provides the same interface as the JSONL AuditTrail but persists records
to a PostgreSQL database for queryability, durability, and compliance.

Schema is auto-created on first connection. Falls back to JSONL if
PostgreSQL is unavailable.

Configuration:
    CATHODE_AUDIT_BACKEND=postgres   (default: "jsonl")
    CATHODE_DATABASE_URL=postgresql://user:pass@host:5432/cathode

Usage:
    from cathode_screening.monitoring.audit_db import get_audit_backend
    audit = get_audit_backend()
    audit.log_prediction(...)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SQL schema for the audit table
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_audit (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id      VARCHAR(64) NOT NULL,
    input_hash      VARCHAR(64) NOT NULL,
    model_version   VARCHAR(64),
    ensemble_id     VARCHAR(64),
    client_id       VARCHAR(128),
    material_id     VARCHAR(128),
    formula         VARCHAR(256),
    mode            VARCHAR(32) DEFAULT 'dft_followup',

    -- Prediction results
    ehull_pred              DOUBLE PRECISION,
    ehull_lower             DOUBLE PRECISION,
    ehull_upper             DOUBLE PRECISION,
    p_stable                DOUBLE PRECISION,
    decision                VARCHAR(16),
    decision_confidence     DOUBLE PRECISION,
    uncertainty_epistemic   DOUBLE PRECISION,
    uncertainty_total       DOUBLE PRECISION,
    ood_flag                BOOLEAN,
    ood_score               DOUBLE PRECISION,

    -- Cathode properties (JSONB for flexibility)
    cathode_properties      JSONB,

    -- Operational metadata
    latency_ms              DOUBLE PRECISION,

    -- Indexes for common queries
    CONSTRAINT chk_decision CHECK (decision IN ('KEEP', 'MAYBE', 'KILL'))
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON prediction_audit (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_request_id ON prediction_audit (request_id);
CREATE INDEX IF NOT EXISTS idx_audit_material_id ON prediction_audit (material_id);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON prediction_audit (decision);
CREATE INDEX IF NOT EXISTS idx_audit_input_hash ON prediction_audit (input_hash);
CREATE INDEX IF NOT EXISTS idx_audit_client_id ON prediction_audit (client_id);
"""

_INSERT_SQL = """
INSERT INTO prediction_audit (
    timestamp, request_id, input_hash, model_version, ensemble_id,
    client_id, material_id, formula, mode,
    ehull_pred, ehull_lower, ehull_upper, p_stable,
    decision, decision_confidence, uncertainty_epistemic, uncertainty_total,
    ood_flag, ood_score, cathode_properties, latency_ms
) VALUES (
    %(timestamp)s, %(request_id)s, %(input_hash)s, %(model_version)s, %(ensemble_id)s,
    %(client_id)s, %(material_id)s, %(formula)s, %(mode)s,
    %(ehull_pred)s, %(ehull_lower)s, %(ehull_upper)s, %(p_stable)s,
    %(decision)s, %(decision_confidence)s, %(uncertainty_epistemic)s, %(uncertainty_total)s,
    %(ood_flag)s, %(ood_score)s, %(cathode_properties)s, %(latency_ms)s
)
"""

_SELECT_RECENT_SQL = """
SELECT * FROM prediction_audit
ORDER BY timestamp DESC
LIMIT %(limit)s
"""

_SELECT_STATS_SQL = """
SELECT
    COUNT(*) AS total_predictions,
    COUNT(*) FILTER (WHERE decision = 'KEEP') AS keep_count,
    COUNT(*) FILTER (WHERE decision = 'MAYBE') AS maybe_count,
    COUNT(*) FILTER (WHERE decision = 'KILL') AS kill_count,
    COUNT(*) FILTER (WHERE ood_flag = TRUE) AS ood_flagged,
    AVG(latency_ms) AS avg_latency_ms,
    MIN(timestamp) AS first_prediction,
    MAX(timestamp) AS last_prediction
FROM prediction_audit
WHERE timestamp >= %(since)s
"""


class PostgresAuditTrail:
    """PostgreSQL-backed audit trail with connection pooling."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        model_version: Optional[str] = None,
        ensemble_id: Optional[str] = None,
        pool_size: int = 5,
    ):
        self.database_url = database_url or os.getenv("CATHODE_DATABASE_URL", "")
        self.model_version = model_version or os.getenv("CATHODE_MODEL_VERSION", "unknown")
        self.ensemble_id = ensemble_id or os.getenv("CATHODE_ENSEMBLE_ID", "unknown")
        self._pool = None
        self._lock = threading.Lock()
        self._initialized = False

    def _get_pool(self):
        """Lazy-initialize connection pool."""
        if self._pool is not None:
            return self._pool

        with self._lock:
            if self._pool is not None:
                return self._pool

            try:
                import psycopg2
                from psycopg2 import pool as pg_pool

                self._pool = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=int(os.getenv("CATHODE_DB_POOL_SIZE", "5")),
                    dsn=self.database_url,
                )
                logger.info("PostgreSQL audit trail connected")
            except ImportError:
                raise RuntimeError(
                    "psycopg2 is required for PostgreSQL audit trail. "
                    "Install with: pip install psycopg2-binary"
                )
            except Exception as exc:
                logger.error("Failed to create PostgreSQL connection pool: %s", exc)
                raise

        return self._pool

    def _ensure_schema(self) -> None:
        """Create audit table if it doesn't exist."""
        if self._initialized:
            return

        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
            self._initialized = True
            logger.info("PostgreSQL audit schema initialized")
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to initialize audit schema: %s", exc)
            raise
        finally:
            pool.putconn(conn)

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
        """Log a prediction to PostgreSQL."""
        try:
            self._ensure_schema()
            pool = self._get_pool()
            conn = pool.getconn()
            try:
                params = {
                    "timestamp": datetime.now(timezone.utc),
                    "request_id": request_id,
                    "input_hash": input_hash,
                    "model_version": self.model_version,
                    "ensemble_id": self.ensemble_id,
                    "client_id": client_id,
                    "material_id": material_id,
                    "formula": formula,
                    "mode": mode,
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
                    "cathode_properties": json.dumps(cathode_properties) if cathode_properties else None,
                    "latency_ms": round(latency_ms, 2),
                }
                with conn.cursor() as cur:
                    cur.execute(_INSERT_SQL, params)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.warning("Failed to write audit record to PostgreSQL: %s", exc)
            finally:
                pool.putconn(conn)
        except Exception as exc:
            logger.warning("PostgreSQL audit trail unavailable: %s", exc)

    def get_recent(self, n: int = 50) -> List[Dict]:
        """Read the most recent N predictions."""
        try:
            self._ensure_schema()
            pool = self._get_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(_SELECT_RECENT_SQL, {"limit": min(n, 500)})
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                records = []
                for row in rows:
                    record = dict(zip(columns, row))
                    # Convert datetime to ISO string
                    if record.get("timestamp"):
                        record["timestamp"] = record["timestamp"].isoformat()
                    records.append(record)
                return records
            finally:
                pool.putconn(conn)
        except Exception as exc:
            logger.warning("Failed to read from PostgreSQL audit: %s", exc)
            return []

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            self._ensure_schema()
            pool = self._get_pool()
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(_SELECT_STATS_SQL, {"since": today})
                    row = cur.fetchone()
                    if row is None:
                        return {
                            "date": today.strftime("%Y-%m-%d"),
                            "total_predictions": 0,
                            "decisions": {"KEEP": 0, "MAYBE": 0, "KILL": 0},
                            "ood_flagged": 0,
                            "avg_latency_ms": 0.0,
                        }
                    return {
                        "date": today.strftime("%Y-%m-%d"),
                        "total_predictions": row[0],
                        "decisions": {
                            "KEEP": row[1],
                            "MAYBE": row[2],
                            "KILL": row[3],
                        },
                        "ood_flagged": row[4],
                        "avg_latency_ms": round(row[5] or 0, 2),
                    }
            finally:
                pool.putconn(conn)
        except Exception as exc:
            logger.warning("Failed to get stats from PostgreSQL audit: %s", exc)
            return {
                "date": today.strftime("%Y-%m-%d"),
                "total_predictions": 0,
                "decisions": {"KEEP": 0, "MAYBE": 0, "KILL": 0},
                "ood_flagged": 0,
                "avg_latency_ms": 0.0,
            }

    def close(self):
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


# ---------------------------------------------------------------------------
# Backend selector — choose between JSONL and PostgreSQL
# ---------------------------------------------------------------------------


def get_audit_backend():
    """Get the configured audit backend.

    Reads CATHODE_AUDIT_BACKEND env var:
      - "jsonl" (default): File-based JSONL audit trail
      - "postgres": PostgreSQL-backed audit trail

    Falls back to JSONL if PostgreSQL is unavailable.
    """
    backend = os.getenv("CATHODE_AUDIT_BACKEND", "jsonl").strip().lower()

    if backend == "postgres":
        database_url = os.getenv("CATHODE_DATABASE_URL", "")
        if not database_url:
            logger.warning(
                "CATHODE_AUDIT_BACKEND=postgres but CATHODE_DATABASE_URL not set. "
                "Falling back to JSONL."
            )
            from cathode_screening.monitoring.audit_trail import get_audit_trail
            return get_audit_trail()

        try:
            trail = PostgresAuditTrail(database_url=database_url)
            trail._ensure_schema()
            return trail
        except Exception as exc:
            logger.warning(
                "PostgreSQL audit trail initialization failed: %s. Falling back to JSONL.",
                exc,
            )
            from cathode_screening.monitoring.audit_trail import get_audit_trail
            return get_audit_trail()

    # Default: JSONL
    from cathode_screening.monitoring.audit_trail import get_audit_trail
    return get_audit_trail()
