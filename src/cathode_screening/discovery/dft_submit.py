"""
DFT job submission via Quantum Espresso on Vertex AI.

Wraps the existing GCP QE infrastructure:
- Generates QE input decks (reusing scripts/45 logic)
- Uploads to GCS
- Submits Vertex AI custom job
- Polls for completion
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter

logger = logging.getLogger(__name__)

GCLOUD_BIN = which("gcloud") or which("gcloud.cmd") or "gcloud"
GSUTIL_BIN = which("gsutil") or which("gsutil.cmd") or "gsutil"

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}

# Transition-metal spin defaults (from scripts/45)
TM_SPIN_DEFAULTS = {
    "Ti": 0.2, "V": 0.3, "Cr": 0.5, "Mn": 0.6, "Fe": 0.6,
    "Co": 0.5, "Ni": 0.5, "Cu": 0.1, "Mo": 0.3, "W": 0.3,
}


@dataclass
class DFTJobConfig:
    """Configuration for a DFT batch job."""

    project: str = "cathode-screening"
    region: str = "us-central1"
    gcs_bucket: str = "cathode-screening-training"
    machine_type: str = "n1-standard-32"
    image_uri: str = "gcr.io/cathode-screening/qe-runner:latest"
    jobs_parallel: int = 8
    mpi_procs: int = 4
    timeout_per_struct_sec: int = 14400
    progress_sync_sec: int = 300
    kspacing: float = 0.25
    ecutwfc: float = 60.0
    ecutrho: float = 480.0
    degauss: float = 0.02
    calculation: str = "relax"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DFTJobConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# QE input generation (extracted from scripts/45_generate_qe_inputs.py)
# ---------------------------------------------------------------------------

def _get_species_order(structure: Structure) -> List[str]:
    elements: List[str] = []
    for site in structure.sites:
        el = site.specie.symbol
        if el not in elements:
            elements.append(el)
    return elements


def _kpoints_from_kspacing(structure: Structure, kspacing: float) -> tuple[int, int, int]:
    rec_lengths = structure.lattice.reciprocal_lattice.lengths
    return tuple(max(1, int(round(float(b) / float(kspacing)))) for b in rec_lengths)


def render_pw_input(
    structure: Structure,
    pseudo_map: Dict[str, str],
    kspacing: float = 0.25,
    calculation: str = "relax",
    ecutwfc: float = 60.0,
    ecutrho: float = 480.0,
    degauss: float = 0.02,
    pseudo_dir: str = "../pseudos",
    outdir: str = "./out",
) -> str:
    """Render a Quantum Espresso pw.x input file for a structure."""
    species_order = _get_species_order(structure)
    ntyp = len(species_order)
    nat = len(structure)
    kx, ky, kz = _kpoints_from_kspacing(structure, kspacing)
    has_tm = any(el in TM_SPIN_DEFAULTS for el in species_order)
    nspin = 2 if has_tm else 1

    lines: List[str] = []

    # &CONTROL
    lines += [
        "&CONTROL",
        f"  calculation = '{calculation}',",
        "  restart_mode = 'from_scratch',",
        f"  outdir = '{outdir}',",
        f"  pseudo_dir = '{pseudo_dir}',",
        "  verbosity = 'high',",
        "/",
    ]

    # &SYSTEM
    sys_lines = [
        "&SYSTEM",
        f"  ibrav = 0,",
        f"  nat = {nat},",
        f"  ntyp = {ntyp},",
        f"  ecutwfc = {ecutwfc:.1f},",
        f"  ecutrho = {ecutrho:.1f},",
        "  occupations = 'smearing',",
        "  smearing = 'mp',",
        f"  degauss = {degauss:.4f},",
    ]
    if nspin == 2:
        sys_lines.append("  nspin = 2,")
        for idx, el in enumerate(species_order, start=1):
            mag = TM_SPIN_DEFAULTS.get(el, 0.0)
            sys_lines.append(f"  starting_magnetization({idx}) = {mag:.2f},")
    sys_lines.append("/")
    lines += sys_lines

    # &ELECTRONS
    lines += ["&ELECTRONS", "  conv_thr = 1.0d-6,", "  mixing_beta = 0.7,", "/"]

    # &IONS
    if calculation in {"relax", "vc-relax"}:
        lines += ["&IONS", "  ion_dynamics = 'bfgs',", "/"]
    # &CELL
    if calculation == "vc-relax":
        lines += ["&CELL", "  cell_dynamics = 'bfgs',", "/"]

    # ATOMIC_SPECIES
    lines.append("")
    lines.append("ATOMIC_SPECIES")
    for el in species_order:
        mass = float(Element(el).atomic_mass)
        pseudo = pseudo_map.get(el, f"{el}.UPF")
        lines.append(f"  {el} {mass:.4f} {pseudo}")

    # ATOMIC_POSITIONS
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure.sites:
        el = site.specie.symbol
        fx, fy, fz = site.frac_coords
        lines.append(f"  {el} {fx:.8f} {fy:.8f} {fz:.8f}")

    # CELL_PARAMETERS
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for row in structure.lattice.matrix:
        lines.append(f"  {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")

    # K_POINTS
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"  {kx} {ky} {kz} 0 0 0")

    return "\n".join(lines) + "\n"


def generate_qe_inputs(
    candidates: List[Dict],
    output_dir: Path,
    config: DFTJobConfig,
    pseudo_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Generate QE input decks for a list of candidate materials.

    Args:
        candidates: List of dicts with keys 'material_id', 'structure' (pymatgen Structure).
        output_dir: Local directory to write QE inputs.
        config: DFT job configuration.
        pseudo_map: Element -> pseudo filename mapping.

    Returns:
        Path to the output directory containing all inputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudos").mkdir(parents=True, exist_ok=True)

    if pseudo_map is None:
        pseudo_map = {}

    all_elements: set[str] = set()
    manifest_rows: List[Dict] = []

    for rank, cand in enumerate(candidates, start=1):
        mid = cand["material_id"]
        structure: Structure = cand["structure"]

        elements = _get_species_order(structure)
        all_elements.update(elements)

        out_name = f"{rank:03d}_{mid}"
        out_path = output_dir / out_name
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "out").mkdir(parents=True, exist_ok=True)

        pw_in = render_pw_input(
            structure=structure,
            pseudo_map=pseudo_map,
            kspacing=config.kspacing,
            calculation=config.calculation,
            ecutwfc=config.ecutwfc,
            ecutrho=config.ecutrho,
            degauss=config.degauss,
            pseudo_dir="../pseudos",
            outdir="./out",
        )
        (out_path / "pw.in").write_text(pw_in)

        CifWriter(structure).write_file(str(out_path / "structure.cif"))

        meta = {"material_id": mid, "rank": rank, "path": out_name}
        (out_path / "metadata.json").write_text(json.dumps(meta, indent=2))
        manifest_rows.append(meta)

    # Fill missing pseudo entries
    for el in sorted(all_elements):
        pseudo_map.setdefault(el, f"{el}.UPF")

    (output_dir / "pseudos.json").write_text(json.dumps(pseudo_map, indent=2, sort_keys=True))

    # Write candidates.txt for the entrypoint script
    with (output_dir / "candidates.txt").open("w") as f:
        for row in manifest_rows:
            f.write(f"{row['path']}\n")

    logger.info("Generated %d QE inputs in %s", len(manifest_rows), output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# GCS upload / Vertex AI job submission / polling
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def upload_to_gcs(local_dir: Path, gcs_path: str) -> None:
    """Upload a local directory to GCS."""
    _run_cmd([GSUTIL_BIN, "-m", "cp", "-r", str(local_dir), gcs_path])
    logger.info("Uploaded %s -> %s", local_dir, gcs_path)


def submit_vertex_job(
    input_gcs_path: str,
    output_gcs_path: str,
    config: DFTJobConfig,
    job_display_name: Optional[str] = None,
) -> str:
    """
    Submit a Vertex AI custom job for QE DFT calculations.

    Returns the job resource name (e.g., projects/.../customJobs/12345).
    """
    if job_display_name is None:
        job_display_name = f"discovery-qe-{int(time.time())}"

    # Build the job spec inline (matching gcp/custom_job_qe_*.yaml structure)
    job_spec = {
        "displayName": job_display_name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "replicaCount": "1",
                    "machineSpec": {"machineType": config.machine_type},
                    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 200},
                    "containerSpec": {
                        "imageUri": config.image_uri,
                        "env": [
                            {"name": "QE_INPUT_GCS", "value": input_gcs_path},
                            {"name": "QE_OUTPUT_GCS", "value": output_gcs_path},
                            {"name": "JOBS", "value": str(config.jobs_parallel)},
                            {"name": "MPI_PROCS", "value": str(config.mpi_procs)},
                            {"name": "STRUCT_TIMEOUT_SEC", "value": str(config.timeout_per_struct_sec)},
                            {"name": "PROGRESS_SYNC_SEC", "value": str(config.progress_sync_sec)},
                        ],
                    },
                }
            ]
        },
    }

    spec_file = Path(tempfile.mktemp(suffix=".json"))
    spec_file.write_text(json.dumps(job_spec, indent=2))

    try:
        result = _run_cmd([
            GCLOUD_BIN, "ai", "custom-jobs", "create",
            "--region", config.region,
            "--project", config.project,
            f"--config={spec_file}",
            "--format=json",
        ])
    finally:
        spec_file.unlink(missing_ok=True)

    response = json.loads(result.stdout)
    job_name = response.get("name", "")
    logger.info("Submitted Vertex AI job: %s", job_name)
    return job_name


def read_job_state(project: str, region: str, job_ref: str) -> Dict[str, Any]:
    """Read a Vertex AI custom job's state (reuses scripts/46 logic)."""
    job_name = job_ref
    if job_ref.isdigit():
        job_name = f"projects/{project}/locations/{region}/customJobs/{job_ref}"

    result = _run_cmd(
        [
            GCLOUD_BIN, "ai", "custom-jobs", "describe", job_name,
            "--region", region, "--project", project, "--format=json",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def read_progress(progress_gcs: str) -> Optional[Dict[str, str]]:
    """Read progress snapshot from GCS (reuses scripts/46 logic)."""
    uri = progress_gcs.rstrip("/") + "/progress/progress.txt"
    result = subprocess.run(
        [GSUTIL_BIN, "cat", uri], check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    parsed: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def poll_until_complete(
    job_name: str,
    gcs_output: str,
    project: str,
    region: str,
    interval_sec: int = 300,
) -> str:
    """
    Poll a Vertex AI job until it reaches a terminal state.

    Returns the final state string (e.g., JOB_STATE_SUCCEEDED).
    """
    logger.info("Polling job %s every %ds...", job_name, interval_sec)
    while True:
        try:
            job = read_job_state(project, region, job_name)
        except RuntimeError as exc:
            logger.warning("Error reading job: %s", exc)
            time.sleep(interval_sec)
            continue

        state = str(job.get("state", "UNKNOWN"))
        progress = read_progress(gcs_output)

        progress_str = ""
        if progress:
            progress_str = (
                f" done={progress.get('done', '?')}"
                f" failed={progress.get('failed', '?')}"
                f" pct={progress.get('percent', '?')}%"
            )
        logger.info("Job state=%s%s", state, progress_str)

        if state in TERMINAL_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                logger.error("Job ended with state %s", state)
            return state

        time.sleep(interval_sec)
