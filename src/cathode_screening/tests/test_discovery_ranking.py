import pandas as pd

from cathode_screening.discovery.ranking import (
    rank_candidates,
    validate_ranking_strategy,
)


def _sample_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "material_id": "A",
                "formula": "LiCoO2",
                "q50": 0.02,
                "p_stable": 0.91,
                "epistemic_std": 0.03,
                "decision_confidence": 0.9,
                "ood_score": 0.0,
            },
            {
                "material_id": "B",
                "formula": "LiNiO2",
                "q50": 0.06,
                "p_stable": 0.75,
                "epistemic_std": 0.08,
                "decision_confidence": 0.7,
                "ood_score": 0.05,
            },
            {
                "material_id": "C",
                "formula": "LiMnTiO4",
                "q50": 0.09,
                "p_stable": 0.60,
                "epistemic_std": 0.19,
                "decision_confidence": 0.6,
                "ood_score": 0.02,
            },
        ]
    )


def test_validate_ranking_strategy_rejects_unknown():
    try:
        validate_ranking_strategy("mystery_mode")
    except ValueError as exc:
        assert "Unknown ranking strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid strategy")


def test_exploit_prefers_low_ehull_high_confidence_candidates():
    ranked = rank_candidates(_sample_pool(), id_col="material_id", strategy="exploit")
    ordered = ranked["material_id"].tolist()
    assert ordered[0] == "A"


def test_explore_boosts_high_uncertainty_candidates():
    ranked = rank_candidates(_sample_pool(), id_col="material_id", strategy="explore")
    ordered = ranked["material_id"].tolist()
    assert ordered.index("C") < ordered.index("A")


def test_rank_candidates_adds_score_columns():
    ranked = rank_candidates(_sample_pool(), id_col="material_id", strategy="balanced")
    for column in (
        "acquisition_score",
        "acq_stability",
        "acq_p_stable",
        "acq_uncertainty",
        "acq_novelty",
        "acq_confidence",
        "acq_ood_penalty",
        "ranking_strategy",
    ):
        assert column in ranked.columns
