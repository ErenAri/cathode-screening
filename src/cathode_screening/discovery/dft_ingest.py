"""
DFT result ingestion from completed QE jobs.

Parses QE output files from GCS, extracts energies, and converts
them into the training-ready format used by the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

GSUTIL_BIN = which("gsutil") or which("gsutil.cmd") or "gsutil"


@dataclass
class QEResult:
    """Parsed result from a single QE calculation."""

    material_id: str
    state: str  # "done", "failed", "timeout"
    total_energy_ry: Optional[float] = None
    energy_per_atom_ev: Optional[float] = None
    n_atoms: Optional[int] = None
    converged: bool = False
    duration_sec: int = 0
    source_run: str = ""

    @property
    def is_done(self) -> bool:
        return self.state == "done" and self.converged


# ---------------------------------------------------------------------------
# Status file parsing (reuses scripts/47_finalize_qe_ranking.py logic)
# ---------------------------------------------------------------------------

def _parse_status_entries(text: str, source: str) -> Dict[str, Dict[str, str]]:
    """Parse a concatenated block of .status files into per-candidate dicts."""
    entries: Dict[str, Dict[str, str]] = {}
    curr: Dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("candidate_rel="):
            if "candidate_rel" in curr and curr["candidate_rel"]:
                entries[curr["candidate_rel"]] = {**curr, "_source": source}
            curr = {"candidate_rel": line.split("=", 1)[1]}
        elif "=" in line:
            k, v = line.split("=", 1)
            curr[k] = v

    if "candidate_rel" in curr and curr["candidate_rel"]:
        entries[curr["candidate_rel"]] = {**curr, "_source": source}

    return entries


def _normalize_state(entry: Dict[str, str]) -> str:
    if entry.get("job_done") == "1":
        return "done"
    return entry.get("state", "unknown")


def _candidate_material_id(candidate_rel: str) -> str:
    """Extract material_id from candidate_rel like '003_mp-12345'."""
    m = re.match(r"^\d+_(.+)$", candidate_rel)
    return m.group(1) if m else candidate_rel


# ---------------------------------------------------------------------------
# QE output parsing
# ---------------------------------------------------------------------------

RY_TO_EV = 13.605693122994  # 1 Ry = 13.6057 eV

_RE_TOTAL_ENERGY = re.compile(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry")
_RE_N_ATOMS = re.compile(r"number of atoms/cell\s+=\s+(\d+)")


def parse_pw_output(pw_out_text: str) -> Dict[str, Any]:
    """
    Parse a QE pw.out file and extract final energy.

    Returns dict with 'total_energy_ry', 'energy_per_atom_ev', 'n_atoms', 'converged'.
    """
    result: Dict[str, Any] = {
        "total_energy_ry": None,
        "energy_per_atom_ev": None,
        "n_atoms": None,
        "converged": False,
    }

    n_atoms_match = _RE_N_ATOMS.search(pw_out_text)
    if n_atoms_match:
        result["n_atoms"] = int(n_atoms_match.group(1))

    # Find last total energy (final relaxation step)
    energies = _RE_TOTAL_ENERGY.findall(pw_out_text)
    if energies:
        total_ry = float(energies[-1])
        result["total_energy_ry"] = total_ry
        total_ev = total_ry * RY_TO_EV
        if result["n_atoms"] and result["n_atoms"] > 0:
            result["energy_per_atom_ev"] = total_ev / result["n_atoms"]

    if "JOB DONE" in pw_out_text:
        result["converged"] = True

    return result


# ---------------------------------------------------------------------------
# GCS fetching
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def fetch_status_map(status_gcs_prefix: str, source: str) -> Dict[str, Dict[str, str]]:
    """Fetch and parse all .status files from a GCS prefix."""
    cmd = [GSUTIL_BIN, "cat", f"{status_gcs_prefix.rstrip('/')}/*.status"]
    result = _run_cmd(cmd)
    if result.returncode != 0:
        msg = result.stderr
        if "matched no objects" in msg or "No URLs matched" in msg:
            return {}
        raise RuntimeError(f"gsutil failed: {msg}")
    return _parse_status_entries(result.stdout, source=source)


def fetch_pw_output(gcs_base: str, candidate_rel: str) -> Optional[str]:
    """Fetch pw.out for a specific candidate from GCS."""
    uri = f"{gcs_base.rstrip('/')}/{candidate_rel}/pw.out"
    result = _run_cmd([GSUTIL_BIN, "cat", uri])
    if result.returncode != 0:
        return None
    return result.stdout


# ---------------------------------------------------------------------------
# High-level ingestion
# ---------------------------------------------------------------------------

def ingest_dft_results(
    gcs_output_path: str,
    candidate_material_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, QEResult]:
    """
    Ingest all DFT results from a completed Vertex AI QE job.

    Args:
        gcs_output_path: GCS base path for the job output.
        candidate_material_ids: Optional mapping of candidate_rel -> material_id.
            If not provided, material_id is extracted from the candidate directory name.

    Returns:
        Dict mapping material_id -> QEResult.
    """
    status_prefix = f"{gcs_output_path.rstrip('/')}/progress/status"
    entries = fetch_status_map(status_prefix, source="ingest")

    results: Dict[str, QEResult] = {}

    for cand_rel, entry in entries.items():
        state = _normalize_state(entry)
        mid = (candidate_material_ids or {}).get(cand_rel, _candidate_material_id(cand_rel))

        qe_result = QEResult(
            material_id=mid,
            state=state,
            duration_sec=int(entry.get("duration_sec", 0)),
            source_run=entry.get("_source", ""),
        )

        # For completed jobs, parse the actual QE output
        if state == "done":
            pw_out = fetch_pw_output(gcs_output_path, cand_rel)
            if pw_out:
                parsed = parse_pw_output(pw_out)
                qe_result.total_energy_ry = parsed["total_energy_ry"]
                qe_result.energy_per_atom_ev = parsed["energy_per_atom_ev"]
                qe_result.n_atoms = parsed["n_atoms"]
                qe_result.converged = parsed["converged"]

        results[mid] = qe_result

    done = sum(1 for r in results.values() if r.is_done)
    failed = sum(1 for r in results.values() if not r.is_done)
    logger.info(
        "Ingested %d DFT results: %d converged, %d failed/timeout",
        len(results), done, failed,
    )

    return results


def to_training_records(
    qe_results: Dict[str, QEResult],
    structures: Dict[str, Any],
    reference_energies: Optional[Dict[str, float]] = None,
    stability_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Convert QE results to training-ready records.

    Args:
        qe_results: material_id -> QEResult from ingest_dft_results.
        structures: material_id -> pymatgen Structure (or dict).
        reference_energies: Optional reference energies for E_hull calculation.
            If not provided, uses raw energy_per_atom_ev as approximate E_hull.
        stability_threshold: E_hull threshold for is_stable label.

    Returns:
        List of training record dicts with keys matching existing dataset format.
    """
    records: List[Dict[str, Any]] = []

    for mid, qe in qe_results.items():
        if not qe.is_done or qe.energy_per_atom_ev is None:
            continue

        struct = structures.get(mid)
        if struct is None:
            logger.warning("No structure for %s, skipping", mid)
            continue

        # E_hull approximation: if reference energies are available, compute
        # difference; otherwise use the raw energy as a proxy.
        if reference_energies and mid in reference_energies:
            e_hull = qe.energy_per_atom_ev - reference_energies[mid]
        else:
            # Without a proper convex hull, energy_per_atom is a proxy.
            # In practice, the discovery loop should provide reference energies
            # from the phase diagram or a pre-computed hull.
            e_hull = qe.energy_per_atom_ev

        struct_dict = struct.as_dict() if hasattr(struct, "as_dict") else struct

        records.append({
            "material_id": mid,
            "structure": struct_dict,
            "energy_above_hull": max(0.0, e_hull),
            "is_stable": e_hull < stability_threshold,
            "dft_energy_per_atom_ev": qe.energy_per_atom_ev,
            "dft_total_energy_ry": qe.total_energy_ry,
            "source": "discovery_qe",
        })

    logger.info("Produced %d training records from %d QE results", len(records), len(qe_results))
    return records
