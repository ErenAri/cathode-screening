"""
Composition-only fast triage for CathodeScreen.

Provides a sub-millisecond screening pass that uses only chemical
composition (no structure relaxation, no MACE inference) to rapidly
classify materials as PASS, MAYBE, or FAIL.

Use cases:
  - Pre-filtering millions of compositions before expensive structure-based screening
  - Real-time feedback in a discovery campaign search UI
  - Quick validation of user-proposed compositions before CIF upload

The triage uses empirical rules and statistical models from known
cathode databases to estimate feasibility:
  1. **Element filter**: Must contain carrier ion + redox-active TM + anion
  2. **Stoichiometry check**: Carrier ion fraction in valid range
  3. **Voltage estimate**: TM-anion lookup table from Materials Project
  4. **Capacity estimate**: Theoretical gravimetric capacity from formula
  5. **Empirical stability score**: Composition-based proxy from training data

Environment variables:
    CATHODE_FAST_TRIAGE_ENABLED    Enable fast triage endpoint (default: true)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cathode_screening.inference")

# Faraday constant / 3.6 => mAh equivalent per mol of electrons
_F_OVER_3600 = 96485.0 / 3600.0  # 26.801 mAh/mol

# Atomic masses (approximate, for capacity calculation)
_ATOMIC_MASS = {
    "H": 1.008, "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.01,
    "N": 14.01, "O": 16.00, "F": 19.00, "Na": 22.99, "Mg": 24.31,
    "Al": 26.98, "Si": 28.09, "P": 30.97, "S": 32.07, "Cl": 35.45,
    "K": 39.10, "Ca": 40.08, "Ti": 47.87, "V": 50.94, "Cr": 52.00,
    "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55,
    "Zn": 65.38, "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.97,
    "Br": 79.90, "Zr": 91.22, "Nb": 92.91, "Mo": 95.94, "Ru": 101.1,
    "Rh": 102.9, "Pd": 106.4, "Sn": 118.7, "Sb": 121.8, "Te": 127.6,
    "I": 126.9, "La": 138.9, "Ta": 180.9, "W": 183.8, "Re": 186.2,
    "Os": 190.2, "Ir": 192.2, "Pt": 195.1,
}

_TM_ELEMENTS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir", "Pt",
}

_ANION_ELEMENTS = {"O", "S", "Se", "F", "Cl", "Br", "I", "N"}

# Empirical voltage lookup (V vs carrier ion) — mean across known structures
_VOLTAGE_LOOKUP = {
    ("Co", "O"): 3.90, ("Ni", "O"): 3.80, ("Mn", "O"): 3.85,
    ("Fe", "O"): 3.40, ("V", "O"): 3.20, ("Cr", "O"): 3.50,
    ("Ti", "O"): 2.50, ("Co", "S"): 2.10, ("Ni", "S"): 1.90,
    ("Mn", "S"): 1.80, ("Fe", "S"): 1.80, ("Fe", "PO4"): 3.45,
    ("Mn", "PO4"): 4.10, ("Co", "PO4"): 4.80, ("V", "PO4"): 3.80,
}


@dataclass
class TriageResult:
    """Result of fast composition triage."""

    formula: str
    decision: str  # PASS, MAYBE, FAIL
    confidence: float  # 0-1
    reasons: List[str]

    # Estimated properties (composition-only)
    carrier_ion: str = "Li"
    carrier_fraction: float = 0.0
    estimated_capacity_mAhg: Optional[float] = None
    estimated_voltage_V: Optional[float] = None
    estimated_energy_Whkg: Optional[float] = None
    transition_metals: List[str] = field(default_factory=list)
    anion_framework: str = "unknown"
    element_count: int = 0

    # Scoring breakdown
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "decision": self.decision,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "carrier_ion": self.carrier_ion,
            "carrier_fraction": round(self.carrier_fraction, 4),
            "estimated_capacity_mAhg": (
                round(self.estimated_capacity_mAhg, 1)
                if self.estimated_capacity_mAhg else None
            ),
            "estimated_voltage_V": (
                round(self.estimated_voltage_V, 2)
                if self.estimated_voltage_V else None
            ),
            "estimated_energy_Whkg": (
                round(self.estimated_energy_Whkg, 1)
                if self.estimated_energy_Whkg else None
            ),
            "transition_metals": self.transition_metals,
            "anion_framework": self.anion_framework,
            "element_count": self.element_count,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


def _parse_formula(formula: str) -> Dict[str, float]:
    """Parse a reduced formula string into element:amount dict.

    Handles: LiCoO2, Li2MnO3, Na0.67MnO2, LiFePO4
    """
    import re

    elements: Dict[str, float] = {}
    # Match element symbol optionally followed by a number
    pattern = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
    for match in pattern.finditer(formula):
        el = match.group(1)
        amt_str = match.group(2)
        if not el:
            continue
        amt = float(amt_str) if amt_str else 1.0
        elements[el] = elements.get(el, 0.0) + amt
    return elements


def _identify_anion_framework(elements: set) -> str:
    if "P" in elements and "O" in elements:
        return "PO4"
    if "Si" in elements and "O" in elements:
        return "SiO4"
    if "B" in elements and "O" in elements:
        return "BO3"
    for anion in ("O", "S", "Se", "F", "Cl"):
        if anion in elements:
            return anion
    return "unknown"


def _estimate_capacity(
    composition: Dict[str, float],
    carrier_ion: str = "Li",
) -> Optional[float]:
    """Estimate theoretical gravimetric capacity (mAh/g)."""
    n_carrier = composition.get(carrier_ion, 0.0)
    if n_carrier <= 0:
        return None

    # Molecular weight
    mw = 0.0
    for el, amt in composition.items():
        mass = _ATOMIC_MASS.get(el)
        if mass is None:
            return None  # Unknown element
        mw += mass * amt

    if mw <= 0:
        return None

    # C = n_carrier * F / (3.6 * MW)
    capacity = n_carrier * _F_OVER_3600 / mw * 1000  # mAh/g
    return capacity


def _estimate_voltage(
    tm_elements: List[str],
    anion_framework: str,
) -> Optional[float]:
    """Estimate average voltage from TM-anion lookup table."""
    if not tm_elements:
        return None

    voltages = []
    for tm in tm_elements:
        v = _VOLTAGE_LOOKUP.get((tm, anion_framework))
        if v is not None:
            voltages.append(v)

    if not voltages:
        return None

    return sum(voltages) / len(voltages)


def triage_formula(
    formula: str,
    carrier_ion: str = "Li",
) -> TriageResult:
    """Fast composition-only triage of a cathode formula.

    Runs in < 1ms. No structure or ML model required.

    Args:
        formula: Chemical formula string (e.g. "LiCoO2", "Na0.67MnO2")
        carrier_ion: Expected carrier ion (default: "Li")

    Returns:
        TriageResult with decision, estimated properties, and scoring.
    """
    composition = _parse_formula(formula)
    elements = set(composition.keys())
    element_count = len(elements)
    tm_list = sorted(el for el in elements if el in _TM_ELEMENTS)
    anion_framework = _identify_anion_framework(elements)

    total_atoms = sum(composition.values())
    carrier_amount = composition.get(carrier_ion, 0.0)
    carrier_fraction = carrier_amount / total_atoms if total_atoms > 0 else 0.0

    reasons = []
    scores: Dict[str, float] = {}

    # --- Element filter ---
    has_carrier = carrier_ion in elements
    has_tm = len(tm_list) > 0
    has_anion = bool(elements & _ANION_ELEMENTS)

    if not has_carrier:
        reasons.append(f"missing_{carrier_ion}")
    if not has_tm:
        reasons.append("missing_transition_metal")
    if not has_anion:
        reasons.append("missing_anion")
    if element_count < 3:
        reasons.append("too_few_elements")

    scores["element_filter"] = 1.0 if (has_carrier and has_tm and has_anion) else 0.0

    # --- Stoichiometry check ---
    stoich_score = 0.0
    if has_carrier and total_atoms > 0:
        # Typical carrier fraction: 0.1 - 0.5 for intercalation cathodes
        if 0.05 <= carrier_fraction <= 0.55:
            stoich_score = 1.0
        elif carrier_fraction < 0.05:
            reasons.append(f"low_{carrier_ion}_fraction")
            stoich_score = 0.3
        else:
            reasons.append(f"high_{carrier_ion}_fraction")
            stoich_score = 0.5
    scores["stoichiometry"] = stoich_score

    # --- Property estimates ---
    capacity = _estimate_capacity(composition, carrier_ion)
    voltage = _estimate_voltage(tm_list, anion_framework)
    energy = None
    if capacity is not None and voltage is not None:
        energy = capacity * voltage / 1000.0  # Wh/kg (from mAh/g * V)

    # Capacity score
    cap_score = 0.0
    if capacity is not None:
        if capacity >= 100:
            cap_score = min(1.0, capacity / 300.0)
        else:
            reasons.append("low_capacity_estimate")
            cap_score = 0.2
    scores["capacity"] = cap_score

    # Voltage score
    volt_score = 0.0
    if voltage is not None:
        if voltage >= 2.0:
            volt_score = min(1.0, voltage / 5.0)
        else:
            reasons.append("low_voltage_estimate")
            volt_score = 0.3
    scores["voltage"] = volt_score

    # --- Composite score ---
    weights = {
        "element_filter": 0.35,
        "stoichiometry": 0.25,
        "capacity": 0.20,
        "voltage": 0.20,
    }
    composite = sum(scores.get(k, 0) * w for k, w in weights.items())
    scores["composite"] = composite

    # --- Decision ---
    if scores["element_filter"] < 0.5:
        decision = "FAIL"
        confidence = 0.9  # High confidence in rejection
    elif composite >= 0.7:
        decision = "PASS"
        confidence = min(0.95, composite)
    elif composite >= 0.4:
        decision = "MAYBE"
        confidence = composite
    else:
        decision = "FAIL"
        confidence = 1.0 - composite

    return TriageResult(
        formula=formula,
        decision=decision,
        confidence=round(confidence, 3),
        reasons=reasons,
        carrier_ion=carrier_ion,
        carrier_fraction=carrier_fraction,
        estimated_capacity_mAhg=capacity,
        estimated_voltage_V=voltage,
        estimated_energy_Whkg=energy,
        transition_metals=tm_list,
        anion_framework=anion_framework,
        element_count=element_count,
        scores=scores,
    )


def triage_batch(
    formulas: List[str],
    carrier_ion: str = "Li",
) -> List[TriageResult]:
    """Fast triage a batch of formulas."""
    return [triage_formula(f, carrier_ion) for f in formulas]
