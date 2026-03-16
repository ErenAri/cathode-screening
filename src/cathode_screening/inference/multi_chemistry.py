"""
Multi-chemistry support for CathodeScreen.

Extends the screening pipeline beyond Li-ion to support:
  - **Na-ion**: Sodium-ion cathodes (NaxTMO2, Na-NASICON, Prussian blue analogues)
  - **Solid-state**: Solid electrolyte compatibility screening (LLZO, LGPS, etc.)
  - **Li-S**: Lithium-sulfur cathode host materials

Each chemistry defines:
  - Composition guardrails (required/excluded elements)
  - Voltage lookup tables
  - Stability thresholds
  - Property calculator adaptations

Environment variables:
    CATHODE_CHEMISTRY    Active chemistry mode (default: li-ion).
                         Options: li-ion, na-ion, solid-state, li-s, auto
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pymatgen.core import Structure

logger = logging.getLogger("cathode_screening.inference")


# ---------------------------------------------------------------------------
# Chemistry profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChemistryProfile:
    """Defines the scope and parameters for a battery chemistry."""

    name: str
    display_name: str
    carrier_ion: str
    required_elements: frozenset  # Must contain at least one
    excluded_elements: frozenset  # Must not contain
    stability_threshold: float  # eV/atom for "stable"
    metastability_threshold: float  # eV/atom for "metastable"
    voltage_lookup: Dict[Tuple[str, str], Tuple[float, float]]  # (TM, anion) → (V, σ)
    supported_anion_frameworks: frozenset


# Transition metals (shared)
_TM_ELEMENTS = frozenset({
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir", "Pt",
})

# ---------------------------------------------------------------------------
# Li-ion (existing, formalized)
# ---------------------------------------------------------------------------

_LI_ION_VOLTAGE: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("Co", "O"): (3.90, 0.30), ("Ni", "O"): (3.80, 0.35),
    ("Mn", "O"): (3.85, 0.45), ("Fe", "O"): (3.40, 0.30),
    ("V", "O"): (3.20, 0.40), ("Cr", "O"): (3.50, 0.35),
    ("Ti", "O"): (2.50, 0.30), ("Mo", "O"): (2.80, 0.35),
    ("Fe", "PO4"): (3.45, 0.15), ("Mn", "PO4"): (4.10, 0.20),
    ("Co", "PO4"): (4.80, 0.20), ("V", "PO4"): (3.80, 0.25),
    ("Fe", "SiO4"): (3.10, 0.20), ("Mn", "SiO4"): (4.10, 0.25),
    ("Co", "S"): (2.10, 0.25), ("Ni", "S"): (1.90, 0.25),
    ("Fe", "S"): (1.80, 0.25), ("Ti", "S"): (2.10, 0.30),
    ("Fe", "F"): (3.30, 0.20), ("Co", "F"): (4.20, 0.25),
}

LI_ION = ChemistryProfile(
    name="li-ion",
    display_name="Lithium-ion",
    carrier_ion="Li",
    required_elements=frozenset({"Li"}),
    excluded_elements=frozenset(),
    stability_threshold=0.05,
    metastability_threshold=0.10,
    voltage_lookup=_LI_ION_VOLTAGE,
    supported_anion_frameworks=frozenset({"O", "S", "Se", "F", "Cl", "PO4", "SiO4", "BO3"}),
)

# ---------------------------------------------------------------------------
# Na-ion
# ---------------------------------------------------------------------------

_NA_ION_VOLTAGE: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Layered oxides (NaxTMO2)
    ("Co", "O"): (3.30, 0.30), ("Ni", "O"): (3.20, 0.35),
    ("Mn", "O"): (2.75, 0.40), ("Fe", "O"): (3.00, 0.35),
    ("Cr", "O"): (3.10, 0.30), ("Ti", "O"): (2.10, 0.30),
    ("V", "O"): (2.80, 0.35), ("Cu", "O"): (2.90, 0.30),
    # NASICON phosphates
    ("Fe", "PO4"): (2.80, 0.15), ("V", "PO4"): (3.40, 0.20),
    ("Mn", "PO4"): (3.50, 0.25), ("Ti", "PO4"): (2.10, 0.20),
    # Sulfides
    ("Fe", "S"): (1.60, 0.25), ("Co", "S"): (1.80, 0.25),
    ("Ni", "S"): (1.70, 0.25), ("Ti", "S"): (1.80, 0.30),
    # Fluorides
    ("Fe", "F"): (3.00, 0.20), ("Mn", "F"): (3.60, 0.25),
}

NA_ION = ChemistryProfile(
    name="na-ion",
    display_name="Sodium-ion",
    carrier_ion="Na",
    required_elements=frozenset({"Na"}),
    excluded_elements=frozenset({"Li"}),  # Pure Na-ion, no Li mixing
    stability_threshold=0.05,
    metastability_threshold=0.12,  # Slightly more tolerant
    voltage_lookup=_NA_ION_VOLTAGE,
    supported_anion_frameworks=frozenset({"O", "S", "Se", "F", "Cl", "PO4", "SiO4"}),
)

# ---------------------------------------------------------------------------
# Solid-state electrolyte
# ---------------------------------------------------------------------------

_SOLID_STATE_VOLTAGE: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Not voltage per se, but electrochemical stability windows
    # (center_of_stability_window, half_width)
    ("La", "O"): (3.00, 2.00),  # LLZO-type
    ("Zr", "O"): (3.00, 2.00),
    ("Ti", "O"): (2.50, 1.50),
    ("Ge", "S"): (2.00, 1.00),  # LGPS-type
    ("P", "S"): (2.10, 0.80),
    ("Li", "S"): (2.00, 1.00),  # Sulfide electrolytes
    ("Li", "Cl"): (3.50, 1.50),  # Halide electrolytes
}

SOLID_STATE = ChemistryProfile(
    name="solid-state",
    display_name="Solid-State Electrolyte",
    carrier_ion="Li",
    required_elements=frozenset({"Li"}),
    excluded_elements=frozenset(),
    stability_threshold=0.03,  # Tighter for electrolytes
    metastability_threshold=0.08,
    voltage_lookup=_SOLID_STATE_VOLTAGE,
    supported_anion_frameworks=frozenset({"O", "S", "Cl", "F", "PO4"}),
)

# ---------------------------------------------------------------------------
# Li-S
# ---------------------------------------------------------------------------

_LI_S_VOLTAGE: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("S", "S"): (2.10, 0.20),  # Li-S baseline
    ("C", "S"): (2.10, 0.20),  # Carbon-sulfur composites
    ("Mn", "S"): (1.80, 0.25),
    ("Fe", "S"): (1.60, 0.25),
    ("Co", "S"): (1.80, 0.25),
    ("Ti", "S"): (2.10, 0.30),
}

LI_S = ChemistryProfile(
    name="li-s",
    display_name="Lithium-Sulfur",
    carrier_ion="Li",
    required_elements=frozenset({"Li", "S"}),
    excluded_elements=frozenset(),
    stability_threshold=0.08,  # More tolerant for conversion reactions
    metastability_threshold=0.15,
    voltage_lookup=_LI_S_VOLTAGE,
    supported_anion_frameworks=frozenset({"S"}),
)

# ---------------------------------------------------------------------------
# Registry of all chemistries
# ---------------------------------------------------------------------------

CHEMISTRY_REGISTRY: Dict[str, ChemistryProfile] = {
    "li-ion": LI_ION,
    "na-ion": NA_ION,
    "solid-state": SOLID_STATE,
    "li-s": LI_S,
}


def get_chemistry(name: Optional[str] = None) -> ChemistryProfile:
    """Get a chemistry profile by name (defaults to env var or li-ion)."""
    if name is None:
        name = os.getenv("CATHODE_CHEMISTRY", "li-ion").strip().lower()
    if name == "auto":
        return LI_ION  # Default; auto-detect is handled by detect_chemistry()
    profile = CHEMISTRY_REGISTRY.get(name)
    if profile is None:
        available = ", ".join(CHEMISTRY_REGISTRY.keys())
        raise ValueError(f"Unknown chemistry '{name}'. Available: {available}")
    return profile


def detect_chemistry(structure: Structure) -> ChemistryProfile:
    """Auto-detect the most likely battery chemistry from a structure.

    Heuristic priority:
      1. Contains Na but no Li → Na-ion
      2. Contains Li + S, no O → Li-S
      3. Contains Li + La/Zr + O, no TM → Solid-state (LLZO-like)
      4. Contains Li + Ge/P + S → Solid-state (sulfide)
      5. Default → Li-ion
    """
    elements = set(structure.composition.get_el_amt_dict().keys())

    has_li = "Li" in elements
    has_na = "Na" in elements
    has_s = "S" in elements
    has_o = "O" in elements
    has_tm = bool(elements & _TM_ELEMENTS)

    # Na-ion
    if has_na and not has_li:
        return NA_ION

    # Li-S (Li + S dominant, no oxide framework)
    if has_li and has_s and not has_o and not has_tm:
        return LI_S

    # Solid-state electrolyte: Li + (La or Zr) + O, no redox-active TM
    if has_li and has_o and not has_tm:
        if elements & {"La", "Zr", "Ge", "P", "Ga", "Al", "Ta"}:
            return SOLID_STATE

    # Solid-state sulfide electrolyte: Li + (Ge or P) + S
    if has_li and has_s and (elements & {"Ge", "P", "Sn"}):
        if not has_tm:
            return SOLID_STATE

    # Default: Li-ion
    return LI_ION


# ---------------------------------------------------------------------------
# Composition check (multi-chemistry aware)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionCheck:
    """Result of composition validation against a chemistry profile."""

    formula: str
    chemistry: str
    carrier_ion: str
    elements: Tuple[str, ...]
    transition_metals: Tuple[str, ...]
    anion_framework: str
    is_valid: bool
    reasons: Tuple[str, ...]


def evaluate_composition(
    structure: Structure,
    chemistry: Optional[ChemistryProfile] = None,
) -> CompositionCheck:
    """Evaluate whether a structure matches the given chemistry scope.

    If chemistry is None, auto-detects from structure.
    """
    if chemistry is None:
        chemistry = detect_chemistry(structure)

    composition = structure.composition.get_el_amt_dict()
    elements = set(composition.keys())
    formula = structure.composition.reduced_formula
    tm_elements = tuple(sorted(el for el in elements if el in _TM_ELEMENTS))

    # Identify anion framework
    anion_framework = _identify_anion_framework(elements)

    reasons = []

    # Check required elements
    if not any(el in elements for el in chemistry.required_elements):
        required = ", ".join(sorted(chemistry.required_elements))
        reasons.append(f"missing_required_element({required})")

    # Check excluded elements
    excluded_found = elements & chemistry.excluded_elements
    if excluded_found:
        reasons.append(f"contains_excluded_element({','.join(sorted(excluded_found))})")

    # Chemistry-specific checks
    if chemistry.name in ("li-ion", "na-ion"):
        if not tm_elements:
            reasons.append("missing_transition_metal")
        if anion_framework not in chemistry.supported_anion_frameworks and anion_framework != "unknown":
            reasons.append(f"unsupported_anion_framework({anion_framework})")

    if chemistry.name == "solid-state":
        # Electrolytes should have high ionic conductivity elements
        ionic_elements = {"La", "Zr", "Ge", "P", "Ga", "Al", "Ta", "Sn", "Cl", "F"}
        if not elements & ionic_elements:
            reasons.append("missing_ionic_conductor_element")

    return CompositionCheck(
        formula=formula,
        chemistry=chemistry.name,
        carrier_ion=chemistry.carrier_ion,
        elements=tuple(sorted(elements)),
        transition_metals=tm_elements,
        anion_framework=anion_framework,
        is_valid=not reasons,
        reasons=tuple(reasons),
    )


def _identify_anion_framework(elements: set) -> str:
    """Identify the anion framework from element set."""
    if "P" in elements and "O" in elements:
        return "PO4"
    if "Si" in elements and "O" in elements:
        return "SiO4"
    if "B" in elements and "O" in elements:
        return "BO3"
    for anion in ("O", "S", "Se", "F", "Cl", "Br", "I", "N"):
        if anion in elements:
            return anion
    return "unknown"


def get_voltage_estimate(
    tm_element: str,
    anion_framework: str,
    chemistry: Optional[ChemistryProfile] = None,
) -> Optional[Tuple[float, float]]:
    """Look up voltage estimate for a TM-anion pair in a given chemistry.

    Returns (mean_voltage, std) or None if not in lookup table.
    """
    if chemistry is None:
        chemistry = get_chemistry()
    return chemistry.voltage_lookup.get((tm_element, anion_framework))


def list_supported_chemistries() -> List[Dict[str, str]]:
    """List all supported chemistry profiles."""
    return [
        {
            "name": p.name,
            "display_name": p.display_name,
            "carrier_ion": p.carrier_ion,
            "stability_threshold": p.stability_threshold,
        }
        for p in CHEMISTRY_REGISTRY.values()
    ]
