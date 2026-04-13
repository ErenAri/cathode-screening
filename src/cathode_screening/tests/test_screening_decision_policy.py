from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_provisional_screening_requires_evidence_threshold(tmp_path: Path):
    ranked_csv = tmp_path / "ranked.csv"
    evidence_csv = tmp_path / "evidence.csv"
    out_csv = tmp_path / "screening.csv"
    out_summary = tmp_path / "summary.txt"

    _write_csv(
        ranked_csv,
        [
            {
                "rank": "1",
                "jarvis_id": "JVASP-1",
                "formula": "LiCoO2",
                "p_stable": "0.95",
                "qe_final_state_est": "done",
                "qe_status_source_est": "run3",
            },
            {
                "rank": "2",
                "jarvis_id": "JVASP-2",
                "formula": "LiMn2O4",
                "p_stable": "0.90",
                "qe_final_state_est": "done",
                "qe_status_source_est": "run3",
            },
            {
                "rank": "3",
                "jarvis_id": "JVASP-3",
                "formula": "LiFePO4",
                "p_stable": "0.88",
                "qe_final_state_est": "timeout",
                "qe_status_source_est": "run5",
            },
        ],
    )

    _write_csv(
        evidence_csv,
        [
            {
                "jarvis_id": "JVASP-1",
                "evidence_tier": "T0",
                "evidence_label": "ML-screened",
                "blockers": "missing_qe_output",
                "recommended_next_step": "Sync outputs",
            },
            {
                "jarvis_id": "JVASP-2",
                "evidence_tier": "T1",
                "evidence_label": "QE-relaxed",
                "blockers": "",
                "recommended_next_step": "Build hull",
            },
        ],
    )

    subprocess.run(
        [
            "python",
            "scripts/48_build_screening_decision.py",
            "--input-csv",
            str(ranked_csv),
            "--evidence-csv",
            str(evidence_csv),
            "--out-csv",
            str(out_csv),
            "--out-summary",
            str(out_summary),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["decision"] == "hold"
    assert rows[0]["decision_reason"] == "top20_missing_evidence"
    assert rows[1]["decision"] == "accept"
    assert rows[1]["decision_reason"] == "top20_evidence_ready"
    assert rows[2]["decision"] == "hold"
    assert rows[2]["decision_reason"] == "top20_unresolved"

    summary = out_summary.read_text(encoding="utf-8")
    assert "min_accept_tier=T1" in summary
    assert "top20_missing_evidence=1" in summary
