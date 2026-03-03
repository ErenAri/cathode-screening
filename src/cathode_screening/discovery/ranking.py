"""Acquisition ranking utilities for discovery campaigns."""

from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

RANKING_STRATEGIES = ("balanced", "exploit", "explore", "risk_averse")
DEFAULT_RANKING_STRATEGY = "balanced"


def validate_ranking_strategy(strategy: Optional[str]) -> str:
    value = (strategy or DEFAULT_RANKING_STRATEGY).strip().lower()
    if value not in RANKING_STRATEGIES:
        allowed = ", ".join(RANKING_STRATEGIES)
        raise ValueError(f"Unknown ranking strategy '{value}'. Allowed: {allowed}")
    return value


def _safe_float_series(df: pd.DataFrame, column: str, fallback: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([fallback] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(fallback).astype(float)


def _minmax(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    min_v = float(values.min())
    max_v = float(values.max())
    if math.isclose(min_v, max_v, rel_tol=1e-12, abs_tol=1e-12):
        return pd.Series([0.5] * len(values), index=values.index, dtype=float)
    return (values - min_v) / (max_v - min_v)


def _formula_novelty_score(formula: str) -> float:
    # Simple proxy: richer multi-element formulas receive higher novelty credit.
    if not formula:
        return 0.0
    elements = set(re.findall(r"[A-Z][a-z]?", formula))
    return max(0.0, min(1.0, len(elements) / 6.0))


def rank_candidates(
    pool_df: pd.DataFrame,
    id_col: str,
    strategy: Optional[str] = None,
) -> pd.DataFrame:
    """Return candidates ranked by multi-objective acquisition score."""
    strategy = validate_ranking_strategy(strategy)
    ranked = pool_df.copy()

    q50 = _safe_float_series(ranked, "q50")
    if "q50" not in ranked.columns:
        q50 = _safe_float_series(ranked, "pred_ehull", fallback=0.25)

    p_stable = _safe_float_series(ranked, "p_stable", fallback=0.5).clip(0.0, 1.0)
    epistemic = _safe_float_series(ranked, "epistemic_std", fallback=0.1).clip(lower=0.0)
    decision_conf = _safe_float_series(ranked, "decision_confidence", fallback=0.5).clip(0.0, 1.0)
    ood_score = _safe_float_series(ranked, "ood_score", fallback=0.0).clip(0.0, 1.0)

    q50_positive = q50.clip(lower=0.0)
    stability_term = q50_positive.apply(lambda v: math.exp(-16.0 * v)).clip(0.0, 1.0)

    epistemic_norm = _minmax(epistemic).clip(0.0, 1.0)
    low_uncertainty_term = 1.0 - epistemic_norm
    high_uncertainty_term = epistemic_norm

    if "formula" in ranked.columns:
        novelty_term = ranked["formula"].astype(str).apply(_formula_novelty_score).astype(float)
    else:
        novelty_term = pd.Series([0.0] * len(ranked), index=ranked.index, dtype=float)

    # Strategy weights (sum of positive terms ~=1.0, penalty applied separately).
    if strategy == "exploit":
        w_stability, w_pstable, w_unc, w_novelty, w_conf, w_penalty = (0.48, 0.32, 0.10, 0.04, 0.06, 0.08)
        uncertainty_term = low_uncertainty_term
    elif strategy == "explore":
        w_stability, w_pstable, w_unc, w_novelty, w_conf, w_penalty = (0.20, 0.20, 0.38, 0.14, 0.08, 0.05)
        uncertainty_term = high_uncertainty_term
    elif strategy == "risk_averse":
        w_stability, w_pstable, w_unc, w_novelty, w_conf, w_penalty = (0.45, 0.30, 0.15, 0.04, 0.06, 0.16)
        uncertainty_term = low_uncertainty_term
    else:  # balanced
        w_stability, w_pstable, w_unc, w_novelty, w_conf, w_penalty = (0.34, 0.30, 0.16, 0.10, 0.10, 0.10)
        uncertainty_term = low_uncertainty_term

    score = (
        w_stability * stability_term
        + w_pstable * p_stable
        + w_unc * uncertainty_term
        + w_novelty * novelty_term
        + w_conf * decision_conf
        - w_penalty * ood_score
    ).clip(0.0, 1.0)

    ranked["ranking_strategy"] = strategy
    ranked["acquisition_score"] = score
    ranked["acq_stability"] = stability_term
    ranked["acq_p_stable"] = p_stable
    ranked["acq_uncertainty"] = uncertainty_term
    ranked["acq_novelty"] = novelty_term
    ranked["acq_confidence"] = decision_conf
    ranked["acq_ood_penalty"] = ood_score

    ranked = ranked.sort_values(
        by=["acquisition_score", "q50" if "q50" in ranked.columns else "pred_ehull", "p_stable"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    if id_col in ranked.columns:
        ranked[id_col] = ranked[id_col].astype(str)

    return ranked
