"""
LIMS (Laboratory Information Management System) integration for CathodeScreen.

Provides bidirectional integration between CathodeScreen and common LIMS
platforms used in battery manufacturing and R&D labs:

  - **Inbound**: Receive experimental results (DFT, electrochemical testing)
    from LIMS webhooks and feed them into the active learning loop.
  - **Outbound**: Push screening results back to LIMS for sample tracking
    and chain-of-custody documentation.

Supported LIMS platforms (via adapters):
  - Generic webhook (any LIMS with HTTP callback support)
  - LabWare LIMS
  - STARLIMS
  - Benchling (ELN/LIMS hybrid)
  - Custom CSV/JSON batch import

Environment variables:
    CATHODE_LIMS_ENABLED        Enable LIMS integration (default: false)
    CATHODE_LIMS_PROVIDER       LIMS adapter name (default: generic)
    CATHODE_LIMS_WEBHOOK_URL    URL for outbound pushes to LIMS
    CATHODE_LIMS_API_KEY        API key for LIMS authentication
    CATHODE_LIMS_CALLBACK_SECRET  Shared secret for validating inbound webhooks
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cathode_screening.integrations.lims")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LIMSSample:
    """A sample record exchanged with LIMS."""

    sample_id: str
    material_id: str
    formula: Optional[str] = None
    status: str = "pending"  # pending, screening, tested, completed

    # CathodeScreen prediction results
    screening_decision: Optional[str] = None  # KEEP, MAYBE, KILL
    ehull_pred: Optional[float] = None
    p_stable: Optional[float] = None
    confidence_interval: Optional[List[float]] = None

    # Experimental results (from LIMS)
    ehull_measured: Optional[float] = None
    capacity_measured: Optional[float] = None
    voltage_measured: Optional[float] = None
    test_date: Optional[str] = None
    test_method: Optional[str] = None

    # Metadata
    batch_id: Optional[str] = None
    lab_id: Optional[str] = None
    operator: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LIMSExportPayload:
    """Payload sent to LIMS after screening."""

    samples: List[LIMSSample]
    campaign_name: Optional[str] = None
    screening_timestamp: str = ""
    model_version: str = ""
    total_screened: int = 0
    n_keep: int = 0
    n_maybe: int = 0
    n_kill: int = 0

    def __post_init__(self):
        if not self.screening_timestamp:
            self.screening_timestamp = datetime.now(timezone.utc).isoformat()
        self.total_screened = len(self.samples)
        self.n_keep = sum(1 for s in self.samples if s.screening_decision == "KEEP")
        self.n_maybe = sum(1 for s in self.samples if s.screening_decision == "MAYBE")
        self.n_kill = sum(1 for s in self.samples if s.screening_decision == "KILL")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "campaign_name": self.campaign_name,
            "screening_timestamp": self.screening_timestamp,
            "model_version": self.model_version,
            "summary": {
                "total_screened": self.total_screened,
                "n_keep": self.n_keep,
                "n_maybe": self.n_maybe,
                "n_kill": self.n_kill,
            },
        }


# ---------------------------------------------------------------------------
# LIMS adapter interface
# ---------------------------------------------------------------------------

class LIMSAdapter(ABC):
    """Abstract base class for LIMS platform adapters."""

    @abstractmethod
    def push_screening_results(self, payload: LIMSExportPayload) -> Dict[str, Any]:
        """Push screening results to LIMS."""

    @abstractmethod
    def parse_webhook(self, body: bytes, headers: Dict[str, str]) -> List[LIMSSample]:
        """Parse an inbound webhook payload from LIMS."""

    @abstractmethod
    def validate_webhook(self, body: bytes, headers: Dict[str, str]) -> bool:
        """Validate the authenticity of an inbound webhook."""


class GenericLIMSAdapter(LIMSAdapter):
    """Generic LIMS adapter using HTTP webhooks.

    Works with any LIMS that can:
      - Receive JSON POST webhooks
      - Send JSON POST callbacks with sample results
    """

    def __init__(self):
        self.webhook_url = os.getenv("CATHODE_LIMS_WEBHOOK_URL", "").strip()
        self.api_key = os.getenv("CATHODE_LIMS_API_KEY", "").strip()
        self.callback_secret = os.getenv("CATHODE_LIMS_CALLBACK_SECRET", "").strip()

    def push_screening_results(self, payload: LIMSExportPayload) -> Dict[str, Any]:
        """POST screening results to the configured LIMS webhook URL."""
        if not self.webhook_url:
            logger.warning("LIMS webhook URL not configured, skipping push")
            return {"status": "skipped", "reason": "no_webhook_url"}

        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = httpx.post(
                self.webhook_url,
                json=payload.to_dict(),
                headers=headers,
                timeout=30.0,
            )
            logger.info(
                "Pushed %d screening results to LIMS (status=%d)",
                len(payload.samples), response.status_code,
            )
            return {
                "status": "success",
                "http_status": response.status_code,
                "samples_pushed": len(payload.samples),
            }
        except ImportError:
            logger.warning("httpx not available, cannot push to LIMS")
            return {"status": "error", "reason": "httpx_not_installed"}
        except Exception as exc:
            logger.error("Failed to push to LIMS: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def parse_webhook(self, body: bytes, headers: Dict[str, str]) -> List[LIMSSample]:
        """Parse a generic LIMS webhook payload.

        Expected format::

            {
                "samples": [
                    {
                        "sample_id": "S-2026-001",
                        "material_id": "mp-12345",
                        "ehull_measured": 0.023,
                        "capacity_measured": 274.0,
                        "voltage_measured": 3.90,
                        "test_method": "half-cell_cycling",
                        "batch_id": "B-2026-042"
                    },
                    ...
                ]
            }
        """
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.error("Invalid LIMS webhook JSON: %s", exc)
            return []

        samples = []
        for entry in data.get("samples", []):
            sample = LIMSSample(
                sample_id=entry.get("sample_id", ""),
                material_id=entry.get("material_id", ""),
                formula=entry.get("formula"),
                status="tested",
                ehull_measured=entry.get("ehull_measured"),
                capacity_measured=entry.get("capacity_measured"),
                voltage_measured=entry.get("voltage_measured"),
                test_date=entry.get("test_date"),
                test_method=entry.get("test_method"),
                batch_id=entry.get("batch_id"),
                lab_id=entry.get("lab_id"),
                operator=entry.get("operator"),
            )
            samples.append(sample)

        return samples

    def validate_webhook(self, body: bytes, headers: Dict[str, str]) -> bool:
        """Validate webhook using HMAC-SHA256 signature."""
        if not self.callback_secret:
            return True  # No secret configured = accept all

        sig_header = headers.get("X-LIMS-Signature", headers.get("x-lims-signature", ""))
        if not sig_header:
            logger.warning("LIMS webhook missing signature header")
            return False

        expected = hmac.new(
            self.callback_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", sig_header)


class LabWareLIMSAdapter(GenericLIMSAdapter):
    """LabWare LIMS adapter with LabWare-specific payload mapping."""

    def parse_webhook(self, body: bytes, headers: Dict[str, str]) -> List[LIMSSample]:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []

        samples = []
        # LabWare uses different field names
        for entry in data.get("results", data.get("samples", [])):
            sample = LIMSSample(
                sample_id=entry.get("SampleId", entry.get("sample_id", "")),
                material_id=entry.get("MaterialRef", entry.get("material_id", "")),
                formula=entry.get("ChemFormula", entry.get("formula")),
                status="tested",
                ehull_measured=entry.get("EhullResult", entry.get("ehull_measured")),
                capacity_measured=entry.get("CapacityResult", entry.get("capacity_measured")),
                voltage_measured=entry.get("VoltageResult", entry.get("voltage_measured")),
                test_date=entry.get("TestDate", entry.get("test_date")),
                test_method=entry.get("TestMethod", entry.get("test_method")),
                batch_id=entry.get("BatchId", entry.get("batch_id")),
                lab_id=entry.get("LabId", entry.get("lab_id")),
            )
            samples.append(sample)
        return samples


class BenchlingLIMSAdapter(GenericLIMSAdapter):
    """Benchling ELN/LIMS adapter."""

    def parse_webhook(self, body: bytes, headers: Dict[str, str]) -> List[LIMSSample]:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []

        samples = []
        # Benchling uses entity-based structure
        for entity in data.get("entities", data.get("samples", [])):
            fields = entity.get("fields", entity)
            sample = LIMSSample(
                sample_id=entity.get("id", entity.get("sample_id", "")),
                material_id=fields.get("material_id", entity.get("material_id", "")),
                formula=fields.get("formula", entity.get("formula")),
                status="tested",
                ehull_measured=fields.get("ehull_measured"),
                capacity_measured=fields.get("capacity_measured"),
                voltage_measured=fields.get("voltage_measured"),
                test_method=fields.get("test_method"),
                batch_id=fields.get("batch_id"),
                metadata={"benchling_id": entity.get("id", "")},
            )
            samples.append(sample)
        return samples


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "generic": GenericLIMSAdapter,
    "labware": LabWareLIMSAdapter,
    "starlims": GenericLIMSAdapter,  # Uses generic format
    "benchling": BenchlingLIMSAdapter,
}


def get_lims_adapter(provider: Optional[str] = None) -> LIMSAdapter:
    """Get the configured LIMS adapter."""
    if provider is None:
        provider = os.getenv("CATHODE_LIMS_PROVIDER", "generic").strip().lower()
    cls = _ADAPTERS.get(provider, GenericLIMSAdapter)
    return cls()


def is_lims_enabled() -> bool:
    return os.getenv(
        "CATHODE_LIMS_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes"}
