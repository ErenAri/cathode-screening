"""
Candidate generation utilities with plausibility checks and de-duplication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional
import hashlib

import numpy as np
from pymatgen.core import Structure


@dataclass(frozen=True)
class CandidateFilters:
    require_elements: set[str] = field(default_factory=lambda: {"Li", "O"})
    require_any_transition_metals: set[str] = field(
        default_factory=lambda: {"Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"}
    )
    exclude_elements: set[str] = field(default_factory=lambda: {"F", "Cl", "Br", "I", "S", "Se"})
    min_atoms: int = 5
    max_atoms: int = 80
    allow_partial_occupancy: bool = False


def structure_hash(structure: Structure) -> str:
    cif = structure.to(fmt="cif")
    return hashlib.sha256(cif.encode("utf-8")).hexdigest()


def has_partial_occupancy(structure: Structure) -> bool:
    for site in structure.sites:
        if len(site.species) != 1 or list(site.species.values())[0] != 1:
            return True
    return False


def is_plausible(structure: Structure, filters: CandidateFilters) -> bool:
    elements = {str(specie) for specie in structure.composition.elements}

    if not filters.require_elements.issubset(elements):
        return False
    if elements.intersection(filters.exclude_elements):
        return False
    if not elements.intersection(filters.require_any_transition_metals):
        return False

    n_atoms = len(structure)
    if n_atoms < filters.min_atoms or n_atoms > filters.max_atoms:
        return False
    if not filters.allow_partial_occupancy and has_partial_occupancy(structure):
        return False

    return True


def deduplicate_candidates(candidates: Iterable[dict], key: str = "structure_hash") -> List[dict]:
    seen = set()
    unique: List[dict] = []
    for cand in candidates:
        value = cand.get(key)
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        unique.append(cand)
    return unique


def generate_candidates(
    structures: Iterable[Structure],
    metadata: Optional[Iterable[dict]] = None,
    filters: Optional[CandidateFilters] = None,
    dedup_key: str = "structure_hash",
) -> List[dict]:
    filters = filters or CandidateFilters()
    meta_list = list(metadata) if metadata is not None else None
    structures_list = list(structures)

    if meta_list is not None and len(meta_list) != len(structures_list):
        raise ValueError("metadata length must match structures length")

    candidates: List[dict] = []
    for idx, structure in enumerate(structures_list):
        if not is_plausible(structure, filters):
            continue
        entry = {
            "structure": structure,
            "formula": structure.composition.reduced_formula,
            "structure_hash": structure_hash(structure),
        }
        if meta_list is not None:
            entry["metadata"] = meta_list[idx]
        candidates.append(entry)

    return deduplicate_candidates(candidates, key=dedup_key)
