"""
CathodeScreen API client.

Provides a typed Python interface to the CathodeScreen REST API
for single and batch predictions of cathode material stability.

Supports:
  - Synchronous and asynchronous clients
  - API versioning (v1)
  - RBAC-aware identity inspection
  - Multi-tenant X-Tenant-ID header
  - Async prediction submission and polling
  - Model registry queries
  - Audit trail access
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

DEFAULT_BASE_URL = "https://cathode-screening-api.onrender.com"


@dataclass
class CathodeProperties:
    """Analytical cathode material properties."""
    gravimetric_capacity: Optional[float] = None   # mAh/g
    volumetric_capacity: Optional[float] = None     # mAh/cm^3
    voltage: Optional[float] = None                 # V vs Li/Li+
    voltage_confidence: Optional[str] = None
    gravimetric_energy: Optional[float] = None      # Wh/kg
    volumetric_energy: Optional[float] = None       # Wh/L
    li_count: Optional[float] = None
    li_fraction: Optional[float] = None
    extractable_li: Optional[float] = None
    density: Optional[float] = None                 # g/cm^3
    tm_elements: Optional[List[str]] = None
    anion_framework: Optional[str] = None
    composite_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None

    @classmethod
    def from_api(cls, data: Optional[dict]) -> Optional["CathodeProperties"]:
        if data is None:
            return None
        return cls(
            gravimetric_capacity=data.get("gravimetric_capacity_mAhg"),
            volumetric_capacity=data.get("volumetric_capacity_mAhcm3"),
            voltage=data.get("avg_voltage_V"),
            voltage_confidence=data.get("voltage_confidence"),
            gravimetric_energy=data.get("gravimetric_energy_Whkg"),
            volumetric_energy=data.get("volumetric_energy_WhL"),
            li_count=data.get("li_count"),
            li_fraction=data.get("li_fraction"),
            extractable_li=data.get("n_extractable_li"),
            density=data.get("density_gcm3"),
            tm_elements=data.get("tm_elements"),
            anion_framework=data.get("anion_framework"),
            composite_score=data.get("composite_score"),
            score_breakdown=data.get("score_breakdown"),
        )


@dataclass
class PredictionResult:
    """Result from a single CathodeScreen prediction."""
    material_id: str
    ehull: float                                    # eV/atom
    p_stable: float                                 # [0, 1]
    uncertainty: str                                # "Low" / "Medium" / "High"
    decision: str                                   # "DFT" / "HOLD" / "SKIP"
    confidence_interval: Tuple[float, float]        # (lower, upper) eV/atom
    properties: Optional[CathodeProperties] = None

    @classmethod
    def from_api(cls, data: dict) -> "PredictionResult":
        ci = data.get("confidence_interval", [0, 0])
        return cls(
            material_id=data["material_id"],
            ehull=data["pred_ehull"],
            p_stable=data["p_stable"],
            uncertainty=data["uncertainty"],
            decision=data["action"],
            confidence_interval=(ci[0], ci[1]),
            properties=CathodeProperties.from_api(data.get("cathode_properties")),
        )

    @property
    def capacity(self) -> Optional[float]:
        """Gravimetric capacity in mAh/g."""
        return self.properties.gravimetric_capacity if self.properties else None

    @property
    def voltage(self) -> Optional[float]:
        """Average voltage in V vs Li/Li+."""
        return self.properties.voltage if self.properties else None

    @property
    def energy_density(self) -> Optional[float]:
        """Gravimetric energy density in Wh/kg."""
        return self.properties.gravimetric_energy if self.properties else None

    @property
    def composite_score(self) -> Optional[float]:
        """Multi-property composite screening score [0, 1]."""
        return self.properties.composite_score if self.properties else None

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [
            f"{self.material_id}: {self.decision}",
            f"E_hull={self.ehull:.3f} eV",
            f"P(stable)={self.p_stable:.0%}",
            f"UQ={self.uncertainty}",
        ]
        if self.capacity is not None:
            parts.append(f"Cap={self.capacity:.0f} mAh/g")
        if self.voltage is not None:
            parts.append(f"V={self.voltage:.2f} V")
        if self.composite_score is not None:
            parts.append(f"Score={self.composite_score:.0%}")
        return " | ".join(parts)


class CathodeAPIError(Exception):
    """Raised when the CathodeScreen API returns an error response."""

    def __init__(self, status_code: int, detail: str, response: Optional[dict] = None):
        self.status_code = status_code
        self.detail = detail
        self.response = response or {}
        super().__init__(f"HTTP {status_code}: {detail}")


@dataclass
class AsyncJob:
    """Status of an async prediction job."""

    job_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: Optional[PredictionResult] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "AsyncJob":
        result = None
        if data.get("result"):
            result = PredictionResult.from_api(data["result"])
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", "UNKNOWN"),
            result=result,
            error=data.get("error"),
            raw=data,
        )

    @property
    def is_complete(self) -> bool:
        return self.status in ("SUCCESS", "FAILURE")

    @property
    def is_success(self) -> bool:
        return self.status == "SUCCESS"


class CathodeClient:
    """Python client for the CathodeScreen API.

    Args:
        api_key: API key for authentication. Falls back to
            CATHODE_API_KEY environment variable.
        base_url: API base URL. Falls back to CATHODE_API_URL
            environment variable or the public endpoint.
        timeout: Request timeout in seconds.
        api_version: API version prefix (default: ``v1``). Set to
            ``None`` for unversioned endpoints.
        org_id: Organization ID for multi-tenant deployments.
            Sent as ``X-Tenant-ID`` header.

    Example:
        >>> client = CathodeClient(api_key="sk-...")
        >>> result = client.predict("LiCoO2.cif")
        >>> print(result.decision, result.ehull, result.capacity)
        DFT 0.023 274.0
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        api_version: Optional[str] = "v1",
        org_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("CATHODE_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("CATHODE_API_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self.api_version = api_version
        self.org_id = org_id or os.getenv("CATHODE_ORG_ID")

        versioned = f"{self.base_url}/{api_version}" if api_version else self.base_url
        self._client = httpx.Client(
            base_url=versioned,
            timeout=timeout,
            headers=self._headers(),
            transport=httpx.HTTPTransport(retries=2),
        )

    def _headers(self) -> dict:
        h: dict = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        if self.org_id:
            h["X-Tenant-ID"] = self.org_id
        return h

    def predict(
        self,
        cif_path: Union[str, Path],
    ) -> PredictionResult:
        """Predict stability and cathode properties for a single CIF file.

        Args:
            cif_path: Path to a CIF file.

        Returns:
            PredictionResult with stability, voltage, capacity, and more.

        Raises:
            httpx.HTTPStatusError: On API errors.
            FileNotFoundError: If the CIF file doesn't exist.
        """
        cif_path = Path(cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        with open(cif_path, "rb") as f:
            response = self._client.post(
                "/predict",
                files={"cif_file": (cif_path.name, f, "chemical/x-cif")},
            )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(data.get("error", "Prediction failed"))

        return PredictionResult.from_api(data["prediction"])

    def predict_batch(
        self,
        cif_paths: List[Union[str, Path]],
    ) -> List[PredictionResult]:
        """Predict stability for multiple CIF files.

        Args:
            cif_paths: List of paths to CIF files.

        Returns:
            List of PredictionResult objects.
        """
        files = []
        for p in cif_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"CIF file not found: {p}")
            files.append(("cif_files", (p.name, open(p, "rb"), "chemical/x-cif")))

        try:
            response = self._client.post("/predict/batch", files=files)
        finally:
            for _, (_, fh, _) in files:
                fh.close()

        response.raise_for_status()
        data = response.json()

        results = []
        for pred in data.get("predictions", []):
            results.append(PredictionResult.from_api(pred))

        if data.get("errors"):
            import warnings
            for err in data["errors"]:
                warnings.warn(f"Batch error: {err}")

        return results

    def model_info(self) -> dict:
        """Get model information from the API."""
        response = self._client.get("/model/info")
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self._client.get("/ready")
            return response.status_code == 200
        except Exception:
            return False

    # ----- Async prediction -----

    def predict_async(
        self,
        cif_path: Union[str, Path],
    ) -> AsyncJob:
        """Submit a CIF for async (queued) prediction.

        Returns an AsyncJob with a job_id that can be polled.
        """
        cif_path = Path(cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        with open(cif_path, "rb") as f:
            response = self._client.post(
                "/predict/async",
                files={"cif_file": (cif_path.name, f, "chemical/x-cif")},
            )
        response.raise_for_status()
        return AsyncJob.from_api(response.json())

    def get_job(self, job_id: str) -> AsyncJob:
        """Poll the status of an async prediction job."""
        response = self._client.get(f"/predict/async/{job_id}")
        response.raise_for_status()
        return AsyncJob.from_api(response.json())

    def predict_and_wait(
        self,
        cif_path: Union[str, Path],
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> PredictionResult:
        """Submit async prediction and block until the result is ready.

        Args:
            cif_path: Path to a CIF file.
            poll_interval: Seconds between status polls.
            timeout: Maximum wait time in seconds.

        Returns:
            PredictionResult once the job completes.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            CathodeAPIError: If the job fails.
        """
        job = self.predict_async(cif_path)
        deadline = time.monotonic() + timeout
        while not job.is_complete:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Job {job.job_id} timed out after {timeout}s")
            time.sleep(poll_interval)
            job = self.get_job(job.job_id)

        if job.is_success and job.result:
            return job.result
        raise CathodeAPIError(500, job.error or "Async prediction failed")

    # ----- Auth & identity -----

    def auth_info(self) -> dict:
        """Get the caller's identity, role, and tenant context (RBAC)."""
        response = self._client.get("/auth/info")
        response.raise_for_status()
        return response.json()

    # ----- Audit -----

    def audit_recent(self, n: int = 50) -> list:
        """Get recent audit trail entries."""
        response = self._client.get("/audit/recent", params={"n": n})
        response.raise_for_status()
        return response.json()

    def audit_stats(self) -> dict:
        """Get audit trail statistics."""
        response = self._client.get("/audit/stats")
        response.raise_for_status()
        return response.json()

    # ----- Metrics -----

    def metrics(self) -> dict:
        """Get application metrics."""
        response = self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    # ----- Model registry -----

    def registry_models(self, stage: Optional[str] = None) -> dict:
        """List model versions in the registry."""
        params = {}
        if stage:
            params["stage"] = stage
        response = self._client.get("/registry/models", params=params)
        response.raise_for_status()
        return response.json()

    def registry_production(self) -> dict:
        """Get the current production model from the registry."""
        response = self._client.get("/registry/production")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncCathodeClient:
    """Asynchronous client for the CathodeScreen API.

    Usage::

        async with AsyncCathodeClient(api_key="sk-...") as client:
            result = await client.predict("structure.cif")

    Args:
        api_key: API key for authentication.
        base_url: API base URL.
        timeout: Request timeout in seconds.
        api_version: API version prefix (default: ``v1``).
        org_id: Organization ID for multi-tenant deployments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        api_version: Optional[str] = "v1",
        org_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("CATHODE_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("CATHODE_API_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.org_id = org_id or os.getenv("CATHODE_ORG_ID")

        versioned = f"{self.base_url}/{api_version}" if api_version else self.base_url
        headers: dict = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.org_id:
            headers["X-Tenant-ID"] = self.org_id
        self._client = httpx.AsyncClient(
            base_url=versioned,
            timeout=timeout,
            headers=headers,
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def predict(self, cif_path: Union[str, Path]) -> PredictionResult:
        """Predict stability for a single CIF file (async)."""
        cif_path = Path(cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")
        content = cif_path.read_bytes()
        response = await self._client.post(
            "/predict",
            files={"cif_file": (cif_path.name, content, "chemical/x-cif")},
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(data.get("error", "Prediction failed"))
        return PredictionResult.from_api(data["prediction"])

    async def predict_batch(
        self, cif_paths: List[Union[str, Path]]
    ) -> List[PredictionResult]:
        """Predict stability for multiple CIF files (async)."""
        files = []
        for p in cif_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"CIF file not found: {p}")
            files.append(("cif_files", (p.name, p.read_bytes(), "chemical/x-cif")))
        response = await self._client.post("/predict/batch", files=files)
        response.raise_for_status()
        data = response.json()
        return [PredictionResult.from_api(pred) for pred in data.get("predictions", [])]

    async def predict_async(self, cif_path: Union[str, Path]) -> AsyncJob:
        """Submit a CIF for async (queued) prediction."""
        cif_path = Path(cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")
        content = cif_path.read_bytes()
        response = await self._client.post(
            "/predict/async",
            files={"cif_file": (cif_path.name, content, "chemical/x-cif")},
        )
        response.raise_for_status()
        return AsyncJob.from_api(response.json())

    async def get_job(self, job_id: str) -> AsyncJob:
        """Poll async job status."""
        response = await self._client.get(f"/predict/async/{job_id}")
        response.raise_for_status()
        return AsyncJob.from_api(response.json())

    async def predict_and_wait(
        self,
        cif_path: Union[str, Path],
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> PredictionResult:
        """Submit async prediction and await the result."""
        import asyncio

        job = await self.predict_async(cif_path)
        deadline = time.monotonic() + timeout
        while not job.is_complete:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Job {job.job_id} timed out after {timeout}s")
            await asyncio.sleep(poll_interval)
            job = await self.get_job(job.job_id)

        if job.is_success and job.result:
            return job.result
        raise CathodeAPIError(500, job.error or "Async prediction failed")

    async def model_info(self) -> dict:
        response = await self._client.get("/model/info")
        response.raise_for_status()
        return response.json()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/ready")
            return response.status_code == 200
        except Exception:
            return False

    async def auth_info(self) -> dict:
        response = await self._client.get("/auth/info")
        response.raise_for_status()
        return response.json()

    async def metrics(self) -> dict:
        response = await self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    async def registry_models(self, stage: Optional[str] = None) -> dict:
        params = {"stage": stage} if stage else {}
        response = await self._client.get("/registry/models", params=params)
        response.raise_for_status()
        return response.json()

    async def registry_production(self) -> dict:
        response = await self._client.get("/registry/production")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
