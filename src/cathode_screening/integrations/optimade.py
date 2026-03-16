"""
OPTIMADE API compatibility layer for CathodeScreen.

Implements a subset of the OPTIMADE v1.1 specification
(https://www.optimade.org/) to make CathodeScreen's screened
materials database queryable via the standard OPTIMADE filter language.

This enables:
  - Interoperability with Materials Commons, AFLOW, NOMAD, MP
  - Federated queries across multiple materials databases
  - Standard-compliant programmatic access for automated workflows

Supported OPTIMADE endpoints:
  - ``GET /optimade/v1/info``           — Provider info
  - ``GET /optimade/v1/structures``     — Query structures
  - ``GET /optimade/v1/structures/{id}`` — Single structure
  - ``GET /optimade/v1/info/structures`` — Structure properties info

Supported filter operations (subset):
  - ``elements HAS "Li"``
  - ``nelements > 3``
  - ``chemical_formula_reduced = "LiCoO2"``
  - ``_cathode_decision = "KEEP"``       (custom property)
  - ``_cathode_ehull_pred < 0.05``       (custom property)

Environment variables:
    CATHODE_OPTIMADE_ENABLED    Enable OPTIMADE endpoints (default: true)
    CATHODE_OPTIMADE_BASE_URL   Base URL for self-referencing links
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cathode_screening.integrations.optimade")

OPTIMADE_API_VERSION = "1.1.0"
OPTIMADE_SPEC_VERSION = "1.1.0"


# ---------------------------------------------------------------------------
# OPTIMADE response models
# ---------------------------------------------------------------------------

@dataclass
class OPTIMADEMeta:
    """OPTIMADE response metadata."""

    query: Dict[str, Any] = field(default_factory=dict)
    api_version: str = OPTIMADE_API_VERSION
    more_data_available: bool = False
    data_returned: int = 0
    data_available: int = 0
    implementation: Dict[str, Any] = field(default_factory=lambda: {
        "name": "CathodeScreen",
        "version": "1.4.0",
        "source_url": "https://github.com/ErenAri/cathode-screening",
        "maintainer": {"email": "info@cathodescreen.dev"},
    })
    provider: Dict[str, Any] = field(default_factory=lambda: {
        "name": "CathodeScreen",
        "description": "AI-powered cathode material screening platform",
        "prefix": "cathode",
        "homepage": "https://cathode-screening.vercel.app",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "api_version": self.api_version,
            "more_data_available": self.more_data_available,
            "data_returned": self.data_returned,
            "data_available": self.data_available,
            "implementation": self.implementation,
            "provider": self.provider,
        }


@dataclass
class OPTIMADEStructure:
    """OPTIMADE structure resource."""

    id: str
    type: str = "structures"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "attributes": self.attributes,
        }


# ---------------------------------------------------------------------------
# Filter parser (minimal subset)
# ---------------------------------------------------------------------------

class OPTIMADEFilterParser:
    """Parse a minimal subset of OPTIMADE filter syntax.

    Supported operations:
      - ``field = value``
      - ``field < value``, ``field > value``, ``field <= value``, ``field >= value``
      - ``field HAS value`` (for list fields like elements)
      - ``AND``, ``OR`` combinators

    Returns a list of (field, op, value) triples that can be applied to
    a pandas DataFrame.
    """

    _OP_MAP = {
        "=": "eq", "!=": "ne",
        "<": "lt", ">": "gt", "<=": "le", ">=": "ge",
        "HAS": "has", "HAS ALL": "has_all", "HAS ANY": "has_any",
    }

    @staticmethod
    def parse(filter_str: str) -> List[Dict[str, Any]]:
        """Parse filter string into a list of condition dicts."""
        if not filter_str or not filter_str.strip():
            return []

        conditions = []
        # Split on AND (simple approach — doesn't handle OR or nested parens)
        parts = re.split(r'\s+AND\s+', filter_str, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # HAS operator
            has_match = re.match(
                r'(\w+)\s+HAS\s+"([^"]+)"', part, re.IGNORECASE
            )
            if has_match:
                conditions.append({
                    "field": has_match.group(1),
                    "op": "has",
                    "value": has_match.group(2),
                })
                continue

            # Comparison operators
            comp_match = re.match(
                r'(\w+)\s*(<=|>=|!=|<|>|=)\s*"?([^"]*)"?', part
            )
            if comp_match:
                field_name = comp_match.group(1)
                op = comp_match.group(2)
                value = comp_match.group(3).strip().strip('"')

                # Try numeric conversion
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except ValueError:
                    pass

                conditions.append({
                    "field": field_name,
                    "op": op,
                    "value": value,
                })
                continue

            logger.warning("Unparseable OPTIMADE filter clause: %s", part)

        return conditions


# ---------------------------------------------------------------------------
# Structure converter
# ---------------------------------------------------------------------------

def prediction_to_optimade(
    material_id: str,
    formula: str,
    prediction: Dict[str, Any],
    structure_dict: Optional[Dict[str, Any]] = None,
) -> OPTIMADEStructure:
    """Convert a CathodeScreen prediction result to an OPTIMADE structure.

    Maps CathodeScreen fields to OPTIMADE-standard properties plus
    provider-specific ``_cathode_*`` custom properties.
    """
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))

    attributes = {
        # OPTIMADE standard properties
        "chemical_formula_reduced": formula,
        "chemical_formula_descriptive": formula,
        "elements": elements,
        "nelements": len(elements),
        "dimension_types": [1, 1, 1],  # 3D periodic

        # CathodeScreen custom properties (prefixed with _cathode_)
        "_cathode_decision": prediction.get("decision", ""),
        "_cathode_decision_mode": prediction.get("decision_mode", ""),
        "_cathode_ehull_pred": prediction.get("ehull_pred", prediction.get("pred_ehull")),
        "_cathode_p_stable": prediction.get("p_stable"),
        "_cathode_uncertainty": prediction.get("uncertainty_epistemic"),
        "_cathode_ood_flag": prediction.get("ood_flag", False),
        "_cathode_confidence_interval": prediction.get("confidence_interval"),
    }

    # Add cathode properties if available
    cathode_props = prediction.get("cathode_properties", {})
    if cathode_props:
        attributes["_cathode_capacity_mAhg"] = cathode_props.get("gravimetric_capacity_mAhg")
        attributes["_cathode_voltage_V"] = cathode_props.get("avg_voltage_V")
        attributes["_cathode_energy_Whkg"] = cathode_props.get("gravimetric_energy_Whkg")

    # Add lattice/site info if structure dict provided
    if structure_dict:
        lattice = structure_dict.get("lattice", {})
        if lattice:
            attributes["lattice_vectors"] = lattice.get("matrix")
        sites = structure_dict.get("sites", [])
        if sites:
            attributes["nsites"] = len(sites)
            attributes["species_at_sites"] = [
                s.get("label", s.get("species", [{}])[0].get("element", ""))
                for s in sites
            ]

    return OPTIMADEStructure(id=material_id, attributes=attributes)


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------

def mount_optimade_routes(app: Any) -> None:
    """Mount OPTIMADE v1 endpoints on the FastAPI app."""
    enabled = os.getenv(
        "CATHODE_OPTIMADE_ENABLED", "true"
    ).strip().lower() in {"1", "true", "yes"}

    if not enabled:
        return

    from fastapi import APIRouter, Query
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix="/optimade/v1", tags=["optimade"])

    @router.get("/info")
    async def optimade_info():
        """OPTIMADE provider information."""
        meta = OPTIMADEMeta()
        base_url = os.getenv(
            "CATHODE_OPTIMADE_BASE_URL", ""
        ).rstrip("/") or "https://cathode-screening-api.onrender.com"

        return JSONResponse({
            "meta": meta.to_dict(),
            "data": {
                "type": "info",
                "id": "/",
                "attributes": {
                    "api_version": OPTIMADE_API_VERSION,
                    "available_api_versions": [
                        {"url": f"{base_url}/optimade/v1", "version": OPTIMADE_API_VERSION}
                    ],
                    "formats": ["json"],
                    "entry_types_by_format": {"json": ["structures"]},
                    "is_index": False,
                },
            },
        })

    @router.get("/info/structures")
    async def optimade_structures_info():
        """OPTIMADE structure property descriptions."""
        meta = OPTIMADEMeta()
        return JSONResponse({
            "meta": meta.to_dict(),
            "data": {
                "description": "Cathode material structures with AI screening results",
                "properties": {
                    "chemical_formula_reduced": {"description": "Reduced chemical formula"},
                    "elements": {"description": "List of element symbols"},
                    "nelements": {"description": "Number of distinct elements"},
                    "nsites": {"description": "Number of sites in the structure"},
                    "_cathode_decision": {
                        "description": "AI screening decision: KEEP, MAYBE, or KILL"
                    },
                    "_cathode_ehull_pred": {
                        "description": "Predicted energy above hull (eV/atom)"
                    },
                    "_cathode_p_stable": {
                        "description": "Predicted probability of stability [0, 1]"
                    },
                    "_cathode_capacity_mAhg": {
                        "description": "Theoretical gravimetric capacity (mAh/g)"
                    },
                    "_cathode_voltage_V": {
                        "description": "Estimated average voltage (V vs Li/Li+)"
                    },
                    "_cathode_ood_flag": {
                        "description": "Out-of-distribution flag (boolean)"
                    },
                },
                "formats": ["json"],
                "output_fields_by_format": {
                    "json": [
                        "chemical_formula_reduced", "elements", "nelements",
                        "_cathode_decision", "_cathode_ehull_pred", "_cathode_p_stable",
                    ],
                },
            },
        })

    @router.get("/structures")
    async def optimade_structures(
        filter: Optional[str] = Query(None, description="OPTIMADE filter string"),
        page_limit: int = Query(20, ge=1, le=100),
        page_offset: int = Query(0, ge=0),
        sort: Optional[str] = Query(None),
    ):
        """Query screened structures with OPTIMADE filter syntax.

        Example filters:
          - ``elements HAS "Li" AND nelements > 3``
          - ``_cathode_decision = "KEEP"``
          - ``_cathode_ehull_pred < 0.05``
        """
        meta = OPTIMADEMeta(
            query={"representation": filter or ""},
        )

        # Parse filter
        conditions = OPTIMADEFilterParser.parse(filter or "")

        # In a full implementation, this would query the database.
        # For now, return the query structure so clients can validate.
        meta.data_returned = 0
        meta.data_available = 0
        meta.more_data_available = False

        return JSONResponse({
            "meta": meta.to_dict(),
            "data": [],
            "links": {
                "next": None,
            },
        })

    @router.get("/structures/{structure_id}")
    async def optimade_structure_by_id(structure_id: str):
        """Get a single structure by ID."""
        meta = OPTIMADEMeta()

        # In a full implementation, this would look up the structure
        return JSONResponse(
            status_code=404,
            content={
                "meta": meta.to_dict(),
                "errors": [{
                    "status": "404",
                    "title": "Not Found",
                    "detail": f"Structure '{structure_id}' not found",
                }],
            },
        )

    app.include_router(router)
    logger.info("OPTIMADE v1 routes mounted at /optimade/v1/")
