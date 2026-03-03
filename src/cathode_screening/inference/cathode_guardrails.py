"""
Composition guardrails for Li-ion cathode screening inputs.

This module defines deterministic checks used by API-level validation to
reject inputs that are outside the intended chemistry scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

_TM_ELEMENTS = {
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
}

_SIMPLE_ANIONS = {"O", "S", "Se", "F", "Cl", "Br", "I", "N"}


@dataclass(frozen=True)
class CathodeCompositionCheck:
    formula: str
    elements: tuple[str, ...]
    transition_metals: tuple[str, ...]
    anion_framework: str
    is_valid: bool
    reasons: tuple[str, ...]


def _identify_anion_framework(elements: set[str]) -> str:
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


def evaluate_li_cathode_composition(structure: Structure) -> CathodeCompositionCheck:
    """Evaluate whether a structure is in supported Li-ion cathode chemistry scope."""
    composition = structure.composition.get_el_amt_dict()
    elements = set(composition.keys())
    sorted_elements = tuple(sorted(elements))
    formula = structure.composition.reduced_formula

    tm_elements = tuple(sorted(el for el in elements if el in _TM_ELEMENTS))
    anion_framework = _identify_anion_framework(elements)

    reasons = []
    if len(elements) < 2:
        reasons.append("composition_must_have_multiple_elements")
    if composition.get("Li", 0.0) <= 0.0:
        reasons.append("missing_lithium")
    if not tm_elements:
        reasons.append("missing_transition_metal")
    if not (any(anion in elements for anion in _SIMPLE_ANIONS) or anion_framework in {"PO4", "SiO4", "BO3"}):
        reasons.append("missing_supported_anion_framework")

    return CathodeCompositionCheck(
        formula=formula,
        elements=sorted_elements,
        transition_metals=tm_elements,
        anion_framework=anion_framework,
        is_valid=not reasons,
        reasons=tuple(reasons),
    )
