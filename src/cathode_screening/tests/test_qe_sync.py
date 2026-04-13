from __future__ import annotations

import csv
from pathlib import Path

from cathode_screening.evaluation.qe_sync import (
    detect_candidate_rel,
    load_campaign_candidates,
    summarize_sync,
    sync_local_outputs,
)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_campaign(tmp_path: Path) -> Path:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    candidate_dir = campaign_dir / "001_JVASP-1"
    candidate_dir.mkdir()
    (candidate_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (candidate_dir / "pw.in").write_text("&CONTROL\n/\n", encoding="utf-8")
    (candidate_dir / "structure.cif").write_text("data_test\n", encoding="utf-8")
    _write_manifest(
        campaign_dir / "manifest.csv",
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
    return campaign_dir


def test_load_campaign_candidates_from_manifest(tmp_path: Path):
    campaign_dir = _make_campaign(tmp_path)
    assert load_campaign_candidates(campaign_dir) == {"001_JVASP-1"}


def test_detect_candidate_rel_filters_to_known_candidates(tmp_path: Path):
    campaign_dir = _make_campaign(tmp_path)
    known = load_campaign_candidates(campaign_dir)
    path = tmp_path / "scratch" / "001_JVASP-1" / "pw.out"
    assert detect_candidate_rel(path, known) == "001_JVASP-1"
    other = tmp_path / "scratch" / "999_JVASP-X" / "pw.out"
    assert detect_candidate_rel(other, known) is None


def test_sync_local_outputs_copies_qe_files(tmp_path: Path):
    campaign_dir = _make_campaign(tmp_path)
    source_root = tmp_path / "source"
    source_candidate = source_root / "nested" / "001_JVASP-1"
    source_candidate.mkdir(parents=True)
    (source_candidate / "pw.out").write_text("JOB DONE.\n", encoding="utf-8")
    (source_candidate / "pw.err").write_text("", encoding="utf-8")

    results = sync_local_outputs(campaign_dir, [source_root])
    summary = summarize_sync(results)

    assert summary == {"copied": 2}
    assert (campaign_dir / "001_JVASP-1" / "pw.out").read_text(encoding="utf-8") == "JOB DONE.\n"
    assert (campaign_dir / "001_JVASP-1" / "pw.err").exists()


def test_sync_local_outputs_skips_existing_without_overwrite(tmp_path: Path):
    campaign_dir = _make_campaign(tmp_path)
    target = campaign_dir / "001_JVASP-1" / "pw.out"
    target.write_text("old\n", encoding="utf-8")

    source_root = tmp_path / "source"
    source_candidate = source_root / "001_JVASP-1"
    source_candidate.mkdir(parents=True)
    (source_candidate / "pw.out").write_text("new\n", encoding="utf-8")

    results = sync_local_outputs(campaign_dir, [source_root], filenames=("pw.out",))
    summary = summarize_sync(results)

    assert summary == {"skipped_existing": 1}
    assert target.read_text(encoding="utf-8") == "old\n"
