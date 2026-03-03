"""
Analytical property calculators for Li-ion cathode materials.

Computes physically-grounded estimates of key cathode properties directly
from crystal structure and composition — no ML required. These serve as
complementary screening axes alongside the ML-predicted E_hull.

Properties:
    - Theoretical gravimetric capacity (mAh/g)
    - Theoretical volumetric capacity (mAh/cm^3)
    - Average voltage proxy (V vs Li/Li+)
    - Li fraction and extractable Li count
    - Structural descriptors (density, volume per atom)
    - Multi-property screening score

References:
    - Capacity: C = nF / (3.6 * M), standard electrochemistry
    - Voltage: Empirical TM-anion correlations from ICSD/MP data
      (Hautier et al., Chem. Mater. 2011; Ceder group lookup tables)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Faraday constant / 3.6 => mAh equivalent per mol of electrons
_F_OVER_3600 = 96485.0 / 3600.0  # 26.801 mAh/mol

# Empirical average intercalation voltages (V vs Li/Li+) for TM-anion pairs.
# Sources: Materials Project statistics, Hautier et al. 2011, experimental data.
# Format: (transition_metal_symbol, anion_framework) -> (mean_voltage, std)
_VOLTAGE_LOOKUP: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Oxides
    ("Co", "O"): (3.90, 0.30),
    ("Ni", "O"): (3.80, 0.35),
    ("Mn", "O"): (3.85, 0.45),  # Broad: spinel ~4.0V, layered ~3.3V
    ("Fe", "O"): (3.40, 0.30),
    ("V", "O"): (3.20, 0.40),
    ("Cr", "O"): (3.50, 0.35),
    ("Ti", "O"): (2.50, 0.30),
    ("Mo", "O"): (2.80, 0.35),
    ("W", "O"): (2.60, 0.30),
    ("Cu", "O"): (3.20, 0.30),
    ("Nb", "O"): (2.40, 0.30),
    # Sulfides
    ("Co", "S"): (2.10, 0.25),
    ("Ni", "S"): (1.90, 0.25),
    ("Mn", "S"): (1.80, 0.25),
    ("Fe", "S"): (1.80, 0.25),
    ("Ti", "S"): (2.10, 0.30),
    ("V", "S"): (2.20, 0.30),
    # Phosphates (polyanionic)
    ("Fe", "PO4"): (3.45, 0.15),  # LiFePO4: very well-known
    ("Mn", "PO4"): (4.10, 0.20),
    ("Co", "PO4"): (4.80, 0.20),
    ("Ni", "PO4"): (5.10, 0.25),
    ("V", "PO4"): (3.80, 0.30),
    # Silicates
    ("Fe", "SiO4"): (3.10, 0.20),
    ("Mn", "SiO4"): (4.10, 0.25),
    # Fluorides / oxyfluorides (high voltage)
    ("Fe", "F"): (3.30, 0.25),
    ("Co", "F"): (3.90, 0.30),
    ("Mn", "F"): (3.60, 0.30),
    ("Ni", "F"): (4.00, 0.30),
}

# Fallback: average voltage by anion type when TM is unknown
_VOLTAGE_BY_ANION: Dict[str, float] = {
    "O": 3.50,
    "S": 2.00,
    "Se": 1.60,
    "F": 3.50,
    "Cl": 2.80,
    "PO4": 3.80,
    "SiO4": 3.50,
}

# Common transition metals in cathodes
_TM_ELEMENTS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir", "Pt",
}

# Common oxidation states for TMs in cathode context
_TM_OXIDATION_STATES: Dict[str, List[int]] = {
    "Ti": [3, 4], "V": [3, 4, 5], "Cr": [3, 6], "Mn": [2, 3, 4],
    "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3, 4], "Cu": [1, 2],
    "Mo": [4, 6], "W": [4, 6], "Nb": [3, 5], "Zr": [4],
}


@dataclass
class CathodeProperties:
    """Computed cathode material properties."""

    # Composition
    formula: str
    li_count: float                  # Li atoms per formula unit
    li_fraction: float               # Li / total atoms
    n_extractable_li: float          # Extractable Li per formula unit

    # Capacity
    gravimetric_capacity: float      # mAh/g (theoretical)
    volumetric_capacity: float       # mAh/cm^3 (theoretical)
    molecular_weight: float          # g/mol per formula unit

    # Voltage
    avg_voltage_proxy: float         # V vs Li/Li+
    voltage_confidence: str          # "high" / "medium" / "low"
    voltage_source: str              # Which lookup matched

    # Structural
    density: float                   # g/cm^3
    volume_per_atom: float           # A^3/atom

    # Transition metal info
    tm_elements: List[str]           # Active TMs found
    anion_framework: str             # "O", "S", "PO4", etc.

    # Energy density (capacity * voltage)
    gravimetric_energy_density: float  # Wh/kg
    volumetric_energy_density: float   # Wh/L

    # Multi-property screening score [0-1]
    composite_score: Optional[float] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "formula": self.formula,
            "li_count": round(self.li_count, 2),
            "li_fraction": round(self.li_fraction, 3),
            "n_extractable_li": round(self.n_extractable_li, 2),
            "gravimetric_capacity_mAhg": round(self.gravimetric_capacity, 1),
            "volumetric_capacity_mAhcm3": round(self.volumetric_capacity, 1),
            "molecular_weight_gmol": round(self.molecular_weight, 2),
            "avg_voltage_V": round(self.avg_voltage_proxy, 2),
            "voltage_confidence": self.voltage_confidence,
            "voltage_source": self.voltage_source,
            "density_gcm3": round(self.density, 3),
            "volume_per_atom_A3": round(self.volume_per_atom, 2),
            "tm_elements": self.tm_elements,
            "anion_framework": self.anion_framework,
            "gravimetric_energy_Whkg": round(self.gravimetric_energy_density, 1),
            "volumetric_energy_WhL": round(self.volumetric_energy_density, 1),
            "composite_score": round(self.composite_score, 3) if self.composite_score is not None else None,
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
        }


def _identify_anion_framework(composition: dict) -> str:
    """Identify the anion framework from composition.

    Returns one of: "O", "S", "Se", "F", "Cl", "PO4", "SiO4", or "unknown".
    """
    elements = set(composition.keys())

    # Polyanionic detection
    if "P" in elements and "O" in elements:
        return "PO4"
    if "Si" in elements and "O" in elements:
        return "SiO4"
    if "B" in elements and "O" in elements:
        return "BO3"

    # Simple anions (by priority)
    for anion in ["O", "S", "Se", "F", "Cl", "Br", "I", "N"]:
        if anion in elements:
            return anion

    return "unknown"


def _identify_tm_elements(composition: dict) -> List[str]:
    """Find transition metals present in the composition."""
    return sorted(el for el in composition if el in _TM_ELEMENTS)


def _estimate_extractable_li(li_per_fu: float, tm_count: float) -> float:
    """Estimate extractable Li based on TM redox capacity.

    Heuristic: each TM can typically accommodate 1 electron of redox,
    so extractable Li <= min(li_per_fu, tm_count).
    """
    if tm_count <= 0:
        return min(li_per_fu, 1.0)
    return min(li_per_fu, tm_count)


def _estimate_voltage(
    tm_elements: List[str],
    anion_framework: str,
) -> Tuple[float, str, str]:
    """Estimate average intercalation voltage from TM-anion chemistry.

    Returns (voltage, confidence, source_description).
    """
    if not tm_elements:
        # No TM — use anion-only fallback
        v = _VOLTAGE_BY_ANION.get(anion_framework, 3.0)
        return v, "low", f"anion-only fallback ({anion_framework})"

    # Weighted average over TMs present (equal weight)
    voltages = []
    sources = []
    for tm in tm_elements:
        key = (tm, anion_framework)
        if key in _VOLTAGE_LOOKUP:
            v, _ = _VOLTAGE_LOOKUP[key]
            voltages.append(v)
            sources.append(f"{tm}-{anion_framework}")

    if voltages:
        avg_v = float(np.mean(voltages))
        confidence = "high" if len(voltages) == len(tm_elements) else "medium"
        return avg_v, confidence, "+".join(sources)

    # Fallback: anion-based estimate
    v = _VOLTAGE_BY_ANION.get(anion_framework, 3.0)
    return v, "low", f"anion fallback ({anion_framework}), TMs={tm_elements}"


def compute_cathode_properties(
    structure,  # pymatgen Structure
    formula: Optional[str] = None,
) -> CathodeProperties:
    """Compute all cathode properties from a pymatgen Structure.

    Args:
        structure: pymatgen Structure object
        formula: Optional override formula string

    Returns:
        CathodeProperties with all computed fields
    """
    comp = structure.composition
    formula = formula or comp.reduced_formula

    # Element amounts per formula unit (use string keys for consistency)
    reduced_comp = comp.get_el_amt_dict()  # {str: float}
    fu_factor = comp.get_reduced_formula_and_factor()[1]
    # Per-formula-unit amounts (string keys)
    comp_per_fu = {str(el): amt / fu_factor for el, amt in reduced_comp.items()}

    # Li content
    li_per_fu = comp_per_fu.get("Li", 0.0)
    total_atoms_per_fu = sum(comp_per_fu.values())
    li_fraction = li_per_fu / total_atoms_per_fu if total_atoms_per_fu > 0 else 0.0

    # Identify chemistry
    tm_elements = _identify_tm_elements(reduced_comp)
    anion_framework = _identify_anion_framework(reduced_comp)

    # TM count per formula unit
    tm_count_per_fu = sum(comp_per_fu.get(tm, 0.0) for tm in tm_elements)

    # Extractable Li
    n_extractable = _estimate_extractable_li(li_per_fu, tm_count_per_fu)

    # Molecular weight per formula unit
    mw = float(comp.weight) / fu_factor  # g/mol per FU

    # Theoretical gravimetric capacity: C = n * F / (3.6 * M)
    if mw > 0 and n_extractable > 0:
        grav_cap = n_extractable * _F_OVER_3600 / mw * 1000  # mAh/g
    else:
        grav_cap = 0.0

    # Density and volume
    density = float(structure.density)  # g/cm^3
    vol_per_atom = structure.volume / len(structure)  # A^3/atom

    # Volumetric capacity
    vol_cap = grav_cap * density  # mAh/cm^3

    # Voltage estimate
    avg_voltage, v_confidence, v_source = _estimate_voltage(tm_elements, anion_framework)

    # Energy density: mAh/g * V = Wh/kg (no conversion needed)
    grav_energy = grav_cap * avg_voltage  # Wh/kg
    vol_energy = vol_cap * avg_voltage  # Wh/L

    return CathodeProperties(
        formula=formula,
        li_count=li_per_fu,
        li_fraction=li_fraction,
        n_extractable_li=n_extractable,
        gravimetric_capacity=grav_cap,
        volumetric_capacity=vol_cap,
        molecular_weight=mw,
        avg_voltage_proxy=avg_voltage,
        voltage_confidence=v_confidence,
        voltage_source=v_source,
        density=density,
        volume_per_atom=vol_per_atom,
        tm_elements=tm_elements,
        anion_framework=anion_framework,
        gravimetric_energy_density=grav_energy,
        volumetric_energy_density=vol_energy,
    )


def compute_composite_score(
    props: CathodeProperties,
    ehull_pred: float,
    ehull_confidence: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> CathodeProperties:
    """Add a multi-property composite screening score to CathodeProperties.

    Score components (all normalized to [0, 1], higher = better):
        - stability: from ML-predicted E_hull
        - capacity: from theoretical gravimetric capacity
        - voltage: from voltage proxy
        - energy_density: from gravimetric energy density

    Args:
        props: Pre-computed CathodeProperties
        ehull_pred: ML-predicted E_hull (eV/atom)
        ehull_confidence: Decision confidence from ML model [0, 1]
        weights: Optional dict of component weights (sum to 1)

    Returns:
        Same CathodeProperties with composite_score and score_breakdown filled
    """
    w = weights or {
        "stability": 0.35,
        "capacity": 0.25,
        "voltage": 0.15,
        "energy_density": 0.25,
    }

    # Normalize weights
    w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    breakdown: Dict[str, float] = {}

    # Stability score: sigmoid-like mapping of E_hull
    # E_hull = 0 → score = 1.0; E_hull = 0.1 → score ~0.27; E_hull = 0.2 → score ~0.07
    breakdown["stability"] = math.exp(-20.0 * max(0.0, ehull_pred))

    # Capacity score: normalized against practical range [50, 300] mAh/g
    cap = props.gravimetric_capacity
    breakdown["capacity"] = max(0.0, min(1.0, (cap - 50) / 250))

    # Voltage score: normalized against practical range [2.0, 5.0] V
    v = props.avg_voltage_proxy
    breakdown["voltage"] = max(0.0, min(1.0, (v - 2.0) / 3.0))

    # Energy density: normalized against range [100, 800] Wh/kg
    ed = props.gravimetric_energy_density
    breakdown["energy_density"] = max(0.0, min(1.0, (ed - 100) / 700))

    # Composite
    composite = sum(w.get(k, 0) * v for k, v in breakdown.items())

    props.composite_score = composite
    props.score_breakdown = breakdown
    return props
