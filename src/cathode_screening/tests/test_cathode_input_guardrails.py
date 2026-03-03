from pymatgen.core import Lattice, Structure

from cathode_screening.inference.cathode_guardrails import (
    evaluate_li_cathode_composition,
)


def _simple_structure(species: list[str]) -> Structure:
    lattice = Lattice.cubic(4.0)
    base_coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
    ]
    coords = base_coords[: len(species)]
    return Structure(lattice, species, coords)


def test_valid_li_tm_oxide_passes_guardrail():
    structure = _simple_structure(["Li", "Co", "O", "O"])

    result = evaluate_li_cathode_composition(structure)

    assert result.is_valid is True
    assert result.reasons == ()
    assert result.anion_framework == "O"
    assert "Co" in result.transition_metals


def test_pure_li_fails_with_expected_reasons():
    structure = _simple_structure(["Li"])

    result = evaluate_li_cathode_composition(structure)

    assert result.is_valid is False
    assert "composition_must_have_multiple_elements" in result.reasons
    assert "missing_transition_metal" in result.reasons
    assert "missing_supported_anion_framework" in result.reasons


def test_li_oxide_without_transition_metal_fails():
    structure = _simple_structure(["Li", "Li", "O"])

    result = evaluate_li_cathode_composition(structure)

    assert result.is_valid is False
    assert "missing_transition_metal" in result.reasons
    assert "missing_lithium" not in result.reasons
