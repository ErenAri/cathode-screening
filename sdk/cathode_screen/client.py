"""
CathodeScreen API client.

Provides a typed Python interface to the CathodeScreen REST API
for single and batch predictions of cathode material stability.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


class CathodeClient:
    """Python client for the CathodeScreen API.

    Args:
        api_key: API key for authentication. Falls back to
            CATHODE_API_KEY environment variable.
        base_url: API base URL. Falls back to CATHODE_API_URL
            environment variable or the public endpoint.
        timeout: Request timeout in seconds.

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
    ):
        self.api_key = api_key or os.getenv("CATHODE_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("CATHODE_API_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._headers(),
        )

    def _headers(self) -> dict:
        h: dict = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
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

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
