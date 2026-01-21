#!/usr/bin/env python3
"""
Generate Quantum Espresso input decks from a candidate CSV.

Defaults are conservative and meant as a starting point. You must provide
appropriate pseudopotential filenames before running QE.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter


TM_SPIN_DEFAULTS = {
    "Ti": 0.2,
    "V": 0.3,
    "Cr": 0.5,
    "Mn": 0.6,
    "Fe": 0.6,
    "Co": 0.5,
    "Ni": 0.5,
    "Cu": 0.1,
    "Mo": 0.3,
    "W": 0.3,
}


def _jarvis_to_structure(entry: dict) -> Structure:
    atoms = entry.get("atoms")
    if isinstance(atoms, dict):
        lattice = atoms.get("lattice_mat")
        coords = atoms.get("coords")
        elements = atoms.get("elements")
        if lattice is not None and coords is not None and elements is not None:
            cartesian = bool(atoms.get("cartesian", False))
            return Structure(lattice, elements, coords, coords_are_cartesian=cartesian)
    final_str = entry.get("final_str")
    if isinstance(final_str, dict):
        return Structure.from_dict(final_str)
    raise ValueError("No structure data in JARVIS entry")


def load_jarvis_by_id(jarvis_json: Path) -> Dict[str, dict]:
    with open(jarvis_json, "r") as f:
        data = json.load(f)
    out = {}
    for entry in data:
        jid = entry.get("jid")
        if jid:
            out[jid] = entry
    return out


def get_species_order(structure: Structure) -> List[str]:
    elements = []
    for site in structure.sites:
        el = site.specie.symbol
        if el not in elements:
            elements.append(el)
    return elements


def kpoints_from_kspacing(structure: Structure, kspacing: float) -> Tuple[int, int, int]:
    rec_lengths = structure.lattice.reciprocal_lattice.lengths
    kpts = []
    for b in rec_lengths:
        k = max(1, int(round(float(b) / float(kspacing))))
        kpts.append(k)
    return tuple(kpts)


def render_pw_input(
    structure: Structure,
    pseudo_map: Dict[str, str],
    kspacing: float,
    calculation: str,
    ecutwfc: float,
    ecutrho: float,
    degauss: float,
    pseudo_dir: str,
    outdir: str,
) -> str:
    species_order = get_species_order(structure)
    ntyp = len(species_order)
    nat = len(structure)

    kx, ky, kz = kpoints_from_kspacing(structure, kspacing)

    has_tm = any(el in TM_SPIN_DEFAULTS for el in species_order)
    nspin = 2 if has_tm else 1

    control_lines = [
        "&CONTROL",
        f"  calculation = '{calculation}',",
        "  restart_mode = 'from_scratch',",
        f"  outdir = '{outdir}',",
        f"  pseudo_dir = '{pseudo_dir}',",
        "  verbosity = 'high',",
        "/",
    ]

    system_lines = [
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
        system_lines.append("  nspin = 2,")
        for idx, el in enumerate(species_order, start=1):
            mag = TM_SPIN_DEFAULTS.get(el, 0.0)
            system_lines.append(f"  starting_magnetization({idx}) = {mag:.2f},")
    system_lines.append("/")

    electrons_lines = [
        "&ELECTRONS",
        "  conv_thr = 1.0d-6,",
        "  mixing_beta = 0.7,",
        "/",
    ]

    ions_lines = []
    if calculation in {"relax", "vc-relax"}:
        ions_lines = [
            "&IONS",
            "  ion_dynamics = 'bfgs',",
            "/",
        ]

    cell_lines = []
    if calculation == "vc-relax":
        cell_lines = [
            "&CELL",
            "  cell_dynamics = 'bfgs',",
            "/",
        ]

    lines = []
    lines.extend(control_lines)
    lines.extend(system_lines)
    lines.extend(electrons_lines)
    lines.extend(ions_lines)
    lines.extend(cell_lines)

    lines.append("")
    lines.append("ATOMIC_SPECIES")
    for el in species_order:
        mass = float(Element(el).atomic_mass)
        pseudo = pseudo_map.get(el, f"{el}.UPF")
        lines.append(f"  {el} {mass:.4f} {pseudo}")

    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure.sites:
        el = site.specie.symbol
        fx, fy, fz = site.frac_coords
        lines.append(f"  {el} {fx:.8f} {fy:.8f} {fz:.8f}")

    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for row in structure.lattice.matrix:
        lines.append(f"  {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")

    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"  {kx} {ky} {kz} 0 0 0")

    return "\n".join(lines) + "\n"


def load_pseudo_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def write_pseudo_map(path: Path, pseudo_map: Dict[str, str]) -> None:
    with open(path, "w") as f:
        json.dump(pseudo_map, f, indent=2, sort_keys=True)


def write_helpers(output_dir: Path, manifest_rows: List[dict]) -> None:
    count = len(manifest_rows)
    candidates_path = output_dir / "candidates.txt"
    with open(candidates_path, "w") as f:
        for row in manifest_rows:
            f.write(f"{row['path']}\n")

    run_sh = output_dir / "run_all_qe.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PW_CMD=${PW_CMD:-pw.x}\n"
        "ROOT_DIR=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "while IFS= read -r jobdir; do\n"
        "  [ -z \"$jobdir\" ] && continue\n"
        "  echo \"Running $jobdir\"\n"
        "  cd \"$ROOT_DIR/$jobdir\"\n"
        "  $PW_CMD -in pw.in > pw.out 2> pw.err\n"
        "done < \"$ROOT_DIR/candidates.txt\"\n"
    )

    slurm_sh = output_dir / "submit_slurm_array.sh"
    slurm_sh.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=qe_jarvis\n"
        "#SBATCH --time=08:00:00\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --mem=16G\n"
        f"#SBATCH --array=0-{max(0, count - 1)}\n"
        "\n"
        "set -euo pipefail\n"
        "PW_CMD=${PW_CMD:-pw.x}\n"
        "ROOT_DIR=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "jobdir=$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" \"$ROOT_DIR/candidates.txt\")\n"
        "if [ -z \"$jobdir\" ]; then\n"
        "  echo \"No jobdir found for index $SLURM_ARRAY_TASK_ID\"\n"
        "  exit 1\n"
        "fi\n"
        "cd \"$ROOT_DIR/$jobdir\"\n"
        "$PW_CMD -in pw.in > pw.out 2> pw.err\n"
    )

    check_py = output_dir / "check_pseudos.py"
    check_py.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path(__file__).resolve().parent\n"
        "pseudos = root / \"pseudos\"\n"
        "with open(root / \"pseudos.json\") as f:\n"
        "    data = json.load(f)\n"
        "missing = []\n"
        "for fname in sorted(set(data.values())):\n"
        "    if not (pseudos / fname).exists():\n"
        "        missing.append(fname)\n"
        "if missing:\n"
        "    print(\"Missing pseudopotentials:\")\n"
        "    for name in missing:\n"
        "        print(f\"  {name}\")\n"
        "    raise SystemExit(1)\n"
        "print(\"All pseudopotentials found.\")\n"
    )

    readme = output_dir / "README.txt"
    readme.write_text(
        "QE batch instructions\\n"
        "1) Copy UPF files into ./pseudos and update pseudos.json if needed.\\n"
        "2) Run: python check_pseudos.py\\n"
        "3) Local run: PW_CMD='pw.x' bash run_all_qe.sh\\n"
        "4) SLURM: sbatch submit_slurm_array.sh\\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QE input decks from candidates CSV.")
    parser.add_argument("--candidates", required=True, help="CSV with jarvis_id candidates")
    parser.add_argument("--jarvis-json", required=True, help="JARVIS JSON path")
    parser.add_argument("--output-dir", default="reports/dft_qe_inputs", help="Output directory")
    parser.add_argument("--pseudo-map", default="", help="JSON mapping element -> pseudo filename")
    parser.add_argument("--kspacing", type=float, default=0.25, help="K-point spacing (1/angstrom)")
    parser.add_argument("--ecutwfc", type=float, default=60.0, help="Wavefunction cutoff (Ry)")
    parser.add_argument("--ecutrho", type=float, default=480.0, help="Charge density cutoff (Ry)")
    parser.add_argument("--degauss", type=float, default=0.02, help="Smearing width (Ry)")
    parser.add_argument("--calculation", default="relax", choices=["scf", "relax", "vc-relax"])
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jarvis_json = Path(args.jarvis_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudos").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(candidates_path)
    if "jarvis_id" not in df.columns:
        raise ValueError("Candidates CSV must contain jarvis_id column")

    jarvis_map = load_jarvis_by_id(jarvis_json)
    pseudo_map = load_pseudo_map(Path(args.pseudo_map)) if args.pseudo_map else {}

    manifest_rows = []
    missing = 0
    all_elements = set()

    for idx, row in df.iterrows():
        jid = str(row["jarvis_id"])
        entry = jarvis_map.get(jid)
        if entry is None:
            missing += 1
            continue
        struct = _jarvis_to_structure(entry)
        elements = get_species_order(struct)
        all_elements.update(elements)

        rank = int(row["rank"]) if "rank" in row else idx + 1
        out_name = f"{rank:03d}_{jid}"
        out_path = output_dir / out_name
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "out").mkdir(parents=True, exist_ok=True)

        pw_in = render_pw_input(
            structure=struct,
            pseudo_map=pseudo_map,
            kspacing=args.kspacing,
            calculation=args.calculation,
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            degauss=args.degauss,
            pseudo_dir="../pseudos",
            outdir="./out",
        )
        with open(out_path / "pw.in", "w") as f:
            f.write(pw_in)

        cif_writer = CifWriter(struct)
        cif_writer.write_file(str(out_path / "structure.cif"))

        meta = row.to_dict()
        meta["jarvis_id"] = jid
        meta["path"] = str(out_path)
        with open(out_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        manifest_rows.append(meta)

    if missing:
        print(f"Warning: {missing} jarvis_id entries missing in JSON")

    for el in sorted(all_elements):
        pseudo_map.setdefault(el, f"{el}.UPF")

    pseudo_path = output_dir / "pseudos.json"
    write_pseudo_map(pseudo_path, pseudo_map)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    settings = {
        "calculation": args.calculation,
        "kspacing": args.kspacing,
        "ecutwfc": args.ecutwfc,
        "ecutrho": args.ecutrho,
        "degauss": args.degauss,
        "pseudo_map": str(pseudo_path),
    }
    with open(output_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print(f"Generated {len(manifest_rows)} QE inputs in {output_dir}")
    print(f"Pseudo mapping written to {pseudo_path}")
    print("Place UPF files in the pseudos/ directory and update pseudos.json if needed.")
    write_helpers(output_dir, manifest_rows)
    print(f"Wrote helper scripts to {output_dir}")


if __name__ == "__main__":
    main()
