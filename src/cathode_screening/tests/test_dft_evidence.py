from __future__ import annotations

import csv
import json
from pathlib import Path

from cathode_screening.evaluation.dft_evidence import assess_campaign, parse_pw_output


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "settings.json").write_text(
        json.dumps(
            {
                "calculation": "relax",
                "kspacing": 0.25,
                "ecutwfc": 100.0,
                "ecutrho": 1080.0,
                "degauss": 0.02,
                "pseudo_map": "campaign/pseudos.json",
            }
        ),
        encoding="utf-8",
    )
    (campaign_dir / "pseudos.json").write_text(json.dumps({"Li": "Li.UPF"}), encoding="utf-8")
    candidate_dir = campaign_dir / "001_JVASP-1"
    candidate_dir.mkdir()
    (candidate_dir / "metadata.json").write_text(
        json.dumps(
            {
                "rank": 1,
                "jarvis_id": "JVASP-1",
                "formula": "LiCoO2",
                "p_stable": 0.91,
                "e_hull": 0.01,
                "selection": "top",
                "path": str(candidate_dir),
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "pw.in").write_text("&CONTROL\n/\n", encoding="utf-8")
    (candidate_dir / "structure.cif").write_text("data_test\n", encoding="utf-8")

    manifest_path = campaign_dir / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {
                "rank": "1",
                "jarvis_id": "JVASP-1",
                "formula": "LiCoO2",
                "p_stable": "0.91",
                "mu": "-0.1",
                "sigma": "0.01",
                "e_hull": "0.01",
                "is_stable": "True",
                "selection": "top",
                "path": str(candidate_dir),
            }
        ],
    )

    status_path = tmp_path / "status.csv"
    _write_csv(
        status_path,
        [
            {
                "candidate_rel": "001_JVASP-1",
                "jarvis_id": "JVASP-1",
                "rank": "1",
                "p_stable": "0.91",
                "formula": "LiCoO2",
                "final_state": "done",
                "source": "run3",
                "return_code": "0",
                "job_done": "1",
            }
        ],
    )
    estimated_status_path = tmp_path / "estimated_status.csv"
    _write_csv(
        estimated_status_path,
        [{"candidate_rel": "001_JVASP-1", "final_state": "done", "source": "run3"}],
    )
    return campaign_dir, status_path, estimated_status_path


def test_parse_pw_output_extracts_completion_and_force():
    parsed = parse_pw_output(
        """
        convergence has been achieved in 8 iterations
        !    total energy              =   -114.23456789 Ry
             Total force =     0.000100     Total SCF correction =   0.000000
        End of BFGS Geometry Optimization
        JOB DONE.
        """
    )
    assert parsed.has_job_done_marker is True
    assert parsed.ionic_converged is True
    assert parsed.scf_converged is True
    assert parsed.total_energy_ry == -114.23456789
    assert parsed.total_force_ev_per_ang is not None
    assert parsed.total_force_ev_per_ang > 0
    assert parsed.errors_detected == ()


def test_assess_campaign_without_pw_out_stays_ml_screened(tmp_path: Path):
    campaign_dir, status_path, estimated_status_path = _make_campaign(tmp_path)
    records, summary = assess_campaign(
        campaign_dir,
        status_csv=status_path,
        estimated_status_csv=estimated_status_path,
    )

    assert len(records) == 1
    record = records[0]
    assert record.evidence_tier == "T0"
    assert record.evidence_label == "ML-screened"
    assert "missing_qe_output" in record.blockers
    assert "reported_done_missing_pw_out" in record.blockers
    assert record.runtime_state == "status_done_output_missing"
    assert summary["tier_counts"] == {"T0": 1}
    assert summary["runtime_state_counts"] == {"status_done_output_missing": 1}


def test_assess_campaign_with_pw_out_promotes_to_qe_relaxed(tmp_path: Path):
    campaign_dir, status_path, estimated_status_path = _make_campaign(tmp_path)
    candidate_dir = campaign_dir / "001_JVASP-1"
    (candidate_dir / "pw.out").write_text(
        """
        convergence has been achieved in 8 iterations
        !    total energy              =   -114.23456789 Ry
             Total force =     0.000100     Total SCF correction =   0.000000
        End of BFGS Geometry Optimization
        JOB DONE.
        """,
        encoding="utf-8",
    )

    records, summary = assess_campaign(
        campaign_dir,
        status_csv=status_path,
        estimated_status_csv=estimated_status_path,
    )

    record = records[0]
    assert record.evidence_tier == "T1"
    assert record.evidence_label == "QE-relaxed"
    assert "missing_reference_hull" in record.blockers
    assert record.runtime_state == "output_synced"
    assert summary["tier_counts"] == {"T1": 1}
