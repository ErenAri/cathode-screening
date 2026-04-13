"""Assign evidence tiers to QE follow-up campaigns."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


RY_PER_BOHR_TO_EV_PER_ANG = 25.71104309541616

TIER_LABELS = {
    "T0": "ML-screened",
    "T1": "QE-relaxed",
    "T2": "DFT-hull-checked",
    "T3": "DFT-verified",
    "T4": "Phonon-screened",
    "T5": "Experimentally supported",
}

PW_TOTAL_ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry")
PW_TOTAL_FORCE_RE = re.compile(r"Total force\s*=\s*([-+0-9.EeDd]+)")


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip().replace("D", "E").replace("d", "e")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: Iterable[Mapping[str, str]], key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            out[value] = dict(row)
    return out


def _collect_reference_hull_artifacts(campaign_dir: Path) -> list[str]:
    patterns = [
        "reference_hull*.json",
        "reference_hull*.csv",
        "competing_phase*.json",
        "competing_phase*.csv",
        "reference_phase*.json",
        "reference_phase*.csv",
        "hull_summary*.json",
        "hull_summary*.md",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for path in campaign_dir.glob(pattern):
            if path.is_file():
                found.add(path.name)
    return sorted(found)


def _collect_convergence_artifacts(candidate_dir: Path) -> list[str]:
    patterns = [
        "*cutoff*convergence*.json",
        "*cutoff*convergence*.csv",
        "*kpoint*convergence*.json",
        "*kpoint*convergence*.csv",
        "*convergence_summary*.json",
        "*convergence_summary*.md",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for path in candidate_dir.glob(pattern):
            if path.is_file():
                found.add(path.name)
    return sorted(found)


def _collect_phonon_artifacts(candidate_dir: Path) -> list[str]:
    patterns = [
        "phonopy.yaml",
        "band.yaml",
        "mesh.yaml",
        "thermal_properties.yaml",
        "phonon_summary.json",
        "phonon_summary.md",
        "ph.out",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for path in candidate_dir.glob(pattern):
            if path.is_file():
                found.add(path.name)
    return sorted(found)


@dataclass(frozen=True)
class ParsedPwOutput:
    has_job_done_marker: bool
    ionic_converged: bool
    scf_converged: bool
    total_energy_ry: float | None
    total_force_ry_per_bohr: float | None
    total_force_ev_per_ang: float | None
    errors_detected: tuple[str, ...]


def parse_pw_output(text: str) -> ParsedPwOutput:
    """Extract conservative QE completion signals from pw.x output text."""

    lower = text.lower()
    energy_match = PW_TOTAL_ENERGY_RE.search(text)
    force_match = PW_TOTAL_FORCE_RE.search(text)

    total_energy_ry = _to_float(energy_match.group(1)) if energy_match else None
    total_force_ry_per_bohr = _to_float(force_match.group(1)) if force_match else None
    total_force_ev_per_ang = (
        total_force_ry_per_bohr * RY_PER_BOHR_TO_EV_PER_ANG
        if total_force_ry_per_bohr is not None
        else None
    )

    errors: list[str] = []
    if "convergence not achieved" in lower:
        errors.append("scf_not_converged")
    if "maximum cpu time exceeded" in lower:
        errors.append("max_cpu_time_exceeded")
    if "error in routine" in lower:
        errors.append("qe_error")

    ionic_converged = (
        "bfgs converged" in lower
        or "end of bfgs geometry optimization" in lower
        or "final enthalpy" in lower
    )
    scf_converged = "convergence has been achieved" in lower or ionic_converged
    has_job_done_marker = "job done." in lower

    return ParsedPwOutput(
        has_job_done_marker=has_job_done_marker,
        ionic_converged=ionic_converged,
        scf_converged=scf_converged,
        total_energy_ry=total_energy_ry,
        total_force_ry_per_bohr=total_force_ry_per_bohr,
        total_force_ev_per_ang=total_force_ev_per_ang,
        errors_detected=tuple(errors),
    )


@dataclass(frozen=True)
class EvidenceRecord:
    candidate_rel: str
    jarvis_id: str
    rank: int | None
    formula: str
    selection: str
    p_stable: float | None
    e_hull: float | None
    has_metadata_json: bool
    has_pw_input: bool
    has_structure_cif: bool
    has_pw_output: bool
    has_pw_error: bool
    has_out_dir: bool
    out_file_count: int
    workflow_declared: bool
    qe_status_reported: str
    qe_status_source: str
    qe_status_estimated: str
    runtime_state: str
    qe_job_done_marker: bool
    qe_ionic_converged: bool
    qe_scf_converged: bool
    qe_total_energy_ry: float | None
    qe_total_force_ev_per_ang: float | None
    reference_hull_artifacts: tuple[str, ...]
    convergence_artifacts: tuple[str, ...]
    phonon_artifacts: tuple[str, ...]
    evidence_tier: str
    evidence_label: str
    blockers: tuple[str, ...]
    recommended_next_step: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reference_hull_artifacts"] = ";".join(self.reference_hull_artifacts)
        data["convergence_artifacts"] = ";".join(self.convergence_artifacts)
        data["phonon_artifacts"] = ";".join(self.phonon_artifacts)
        data["blockers"] = ";".join(self.blockers)
        return data


def _recommended_next_step(
    tier: str,
    has_pw_output: bool,
    qe_status_reported: str,
    blockers: Iterable[str],
) -> str:
    blocker_set = set(blockers)
    if "missing_qe_output" in blocker_set and qe_status_reported == "done":
        return "Sync completed QE outputs into the campaign directory."
    if "missing_qe_output" in blocker_set:
        return "Run QE relaxations and store pw.out/pw.err in the candidate directory."
    if "qe_relaxation_incomplete" in blocker_set or "qe_output_has_errors" in blocker_set:
        return "Inspect QE failure mode and rerun the relaxation."
    if tier == "T1":
        return "Build or document a consistent competing-phase reference hull."
    if tier == "T2":
        return "Run cutoff and k-point convergence checks for this candidate."
    if tier == "T3":
        return "Run phonons on finalists that survive hull and robustness screens."
    if tier == "T4":
        return "Attach external or experimental validation if available."
    if tier == "T5":
        return "No action required."
    if has_pw_output:
        return "Review candidate evidence manually."
    return "Keep candidate as ML-screened until DFT outputs are available."


def _tier_and_blockers(
    *,
    workflow_declared: bool,
    has_inputs: bool,
    has_pw_output: bool,
    parsed_pw: ParsedPwOutput | None,
    qe_status_reported: str,
    reference_hull_artifacts: list[str],
    convergence_artifacts: list[str],
    phonon_artifacts: list[str],
) -> tuple[str, list[str]]:
    blockers: list[str] = []

    if not workflow_declared:
        blockers.append("missing_workflow_declaration")
    if not has_inputs:
        blockers.append("missing_campaign_inputs")

    tier = "T0"

    if not has_pw_output:
        blockers.append("missing_qe_output")
        if qe_status_reported == "done":
            blockers.append("reported_done_missing_pw_out")
        return tier, blockers

    assert parsed_pw is not None

    if parsed_pw.errors_detected:
        blockers.append("qe_output_has_errors")
    if not parsed_pw.has_job_done_marker or not parsed_pw.ionic_converged:
        blockers.append("qe_relaxation_incomplete")

    if not blockers:
        tier = "T1"

    if tier == "T1":
        if reference_hull_artifacts:
            tier = "T2"
        else:
            blockers.append("missing_reference_hull")

    if tier == "T2":
        if convergence_artifacts:
            tier = "T3"
        else:
            blockers.append("missing_convergence_checks")

    if tier == "T3":
        if phonon_artifacts:
            tier = "T4"
        else:
            blockers.append("missing_phonon_screen")

    return tier, blockers


def _runtime_state(
    *,
    has_pw_output: bool,
    has_out_dir: bool,
    out_file_count: int,
    qe_status_reported: str,
    qe_status_estimated: str,
) -> str:
    status = (qe_status_reported or qe_status_estimated or "").strip().lower()
    if has_pw_output:
        return "output_synced"
    if status == "done":
        return "status_done_output_missing"
    if status == "timeout":
        return "status_timeout"
    if status:
        return f"status_{status}"
    if out_file_count > 0:
        return "scratch_present_output_missing"
    if has_out_dir:
        return "prepared_only"
    return "inputs_only"


def assess_campaign(
    campaign_dir: Path,
    *,
    status_csv: Path | None = None,
    estimated_status_csv: Path | None = None,
) -> tuple[list[EvidenceRecord], dict[str, object]]:
    """Assess a QE campaign and assign evidence tiers per candidate."""

    manifest_path = campaign_dir / "manifest.csv"
    settings_path = campaign_dir / "settings.json"

    manifest_rows = _load_csv_rows(manifest_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}

    status_by_candidate: dict[str, dict[str, str]] = {}
    if status_csv and status_csv.exists():
        status_by_candidate = _index_rows(_load_csv_rows(status_csv), "candidate_rel")

    estimated_status_by_candidate: dict[str, dict[str, str]] = {}
    if estimated_status_csv and estimated_status_csv.exists():
        estimated_status_by_candidate = _index_rows(_load_csv_rows(estimated_status_csv), "candidate_rel")

    workflow_declared = all(
        settings.get(key) is not None
        for key in ("calculation", "kspacing", "ecutwfc", "ecutrho", "degauss", "pseudo_map")
    )
    reference_hull_artifacts = _collect_reference_hull_artifacts(campaign_dir)

    records: list[EvidenceRecord] = []

    for row in manifest_rows:
        candidate_rel = Path(row["path"].replace("\\", "/")).name
        candidate_dir = campaign_dir / candidate_rel

        metadata_path = candidate_dir / "metadata.json"
        pw_input_path = candidate_dir / "pw.in"
        structure_path = candidate_dir / "structure.cif"
        pw_output_path = candidate_dir / "pw.out"
        pw_error_path = candidate_dir / "pw.err"
        out_dir_path = candidate_dir / "out"

        status_row = status_by_candidate.get(candidate_rel, {})
        estimated_status_row = estimated_status_by_candidate.get(candidate_rel, {})

        parsed_pw: ParsedPwOutput | None = None
        if pw_output_path.exists():
            parsed_pw = parse_pw_output(pw_output_path.read_text(encoding="utf-8", errors="ignore"))

        has_inputs = metadata_path.exists() and pw_input_path.exists() and structure_path.exists()
        has_out_dir = out_dir_path.exists() and out_dir_path.is_dir()
        out_file_count = (
            sum(1 for _ in out_dir_path.rglob("*") if _.is_file()) if has_out_dir else 0
        )
        convergence_artifacts = _collect_convergence_artifacts(candidate_dir)
        phonon_artifacts = _collect_phonon_artifacts(candidate_dir)
        runtime_state = _runtime_state(
            has_pw_output=pw_output_path.exists(),
            has_out_dir=has_out_dir,
            out_file_count=out_file_count,
            qe_status_reported=status_row.get("final_state", ""),
            qe_status_estimated=estimated_status_row.get("final_state", ""),
        )

        tier, blockers = _tier_and_blockers(
            workflow_declared=workflow_declared,
            has_inputs=has_inputs,
            has_pw_output=pw_output_path.exists(),
            parsed_pw=parsed_pw,
            qe_status_reported=status_row.get("final_state", "").strip(),
            reference_hull_artifacts=reference_hull_artifacts,
            convergence_artifacts=convergence_artifacts,
            phonon_artifacts=phonon_artifacts,
        )

        records.append(
            EvidenceRecord(
                candidate_rel=candidate_rel,
                jarvis_id=row.get("jarvis_id", ""),
                rank=int(row["rank"]) if row.get("rank", "").strip().isdigit() else None,
                formula=row.get("formula", ""),
                selection=row.get("selection", ""),
                p_stable=_to_float(row.get("p_stable")),
                e_hull=_to_float(row.get("e_hull")),
                has_metadata_json=metadata_path.exists(),
                has_pw_input=pw_input_path.exists(),
                has_structure_cif=structure_path.exists(),
                has_pw_output=pw_output_path.exists(),
                has_pw_error=pw_error_path.exists(),
                has_out_dir=has_out_dir,
                out_file_count=out_file_count,
                workflow_declared=workflow_declared,
                qe_status_reported=status_row.get("final_state", ""),
                qe_status_source=status_row.get("source", ""),
                qe_status_estimated=estimated_status_row.get("final_state", ""),
                runtime_state=runtime_state,
                qe_job_done_marker=parsed_pw.has_job_done_marker if parsed_pw else False,
                qe_ionic_converged=parsed_pw.ionic_converged if parsed_pw else False,
                qe_scf_converged=parsed_pw.scf_converged if parsed_pw else False,
                qe_total_energy_ry=parsed_pw.total_energy_ry if parsed_pw else None,
                qe_total_force_ev_per_ang=parsed_pw.total_force_ev_per_ang if parsed_pw else None,
                reference_hull_artifacts=tuple(reference_hull_artifacts),
                convergence_artifacts=tuple(convergence_artifacts),
                phonon_artifacts=tuple(phonon_artifacts),
                evidence_tier=tier,
                evidence_label=TIER_LABELS[tier],
                blockers=tuple(blockers),
                recommended_next_step=_recommended_next_step(
                    tier=tier,
                    has_pw_output=pw_output_path.exists(),
                    qe_status_reported=status_row.get("final_state", "").strip(),
                    blockers=blockers,
                ),
            )
        )

    tier_counts = Counter(record.evidence_tier for record in records)
    blocker_counts = Counter(blocker for record in records for blocker in record.blockers)
    runtime_state_counts = Counter(record.runtime_state for record in records)
    summary = {
        "campaign_dir": str(campaign_dir),
        "candidate_count": len(records),
        "tier_counts": dict(sorted(tier_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "runtime_state_counts": dict(sorted(runtime_state_counts.items())),
        "workflow_declared": workflow_declared,
        "reference_hull_artifacts": reference_hull_artifacts,
    }

    return records, summary
