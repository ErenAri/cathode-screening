"""
Grounded Win report utilities for discovery evaluation.

Computes bootstrap confidence intervals for EF@1%, Recall@100, Precision@100,
and reports AUPRC plus prevalence diagnostics.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np


def grounded_win_report(
    test_set: Iterable[dict],
    ensemble,
    calibrator,
    n_random: int = 200,
    n_boot: int = 1000,
) -> Dict[str, str]:
    """
    Generate a grounded win report for a test set.

    Expects each test_set entry to have:
      - "structure": structure object for ensemble prediction
      - "is_stable": boolean ground truth

    The ensemble must support:
      - predict(structure) -> List[float] (per-model predictions)

    The calibrator must support:
      - predict_p_stable(preds: List[float]) -> float
    """
    test_list = list(test_set)
    if not test_list:
        raise ValueError("test_set must contain at least one sample")

    p_stable = np.array(
        [calibrator.predict_p_stable(ensemble.predict(s["structure"])) for s in test_list],
        dtype=np.float32,
    )
    y_true = np.array([s["is_stable"] for s in test_list], dtype=np.int64)
    n = len(y_true)

    top_n_1pct = max(1, int(0.01 * n))
    top_n_100 = min(100, n)
    prevalence = float(y_true.mean()) if n > 0 else 0.0

    order = np.argsort(p_stable)[::-1]

    def bootstrap_metric(metric_fn: Callable[[np.ndarray, np.ndarray], float]) -> tuple[float, float, float]:
        scores: List[float] = []
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            scores.append(metric_fn(p_stable[idx], y_true[idx]))
        return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

    def ef_1pct(p: np.ndarray, y: np.ndarray) -> float:
        top = np.argsort(p)[::-1][:top_n_1pct]
        hit_rate = float(y[top].mean()) if top.size else 0.0
        prev = float(y.mean()) if y.size else 0.0
        return hit_rate / prev if prev > 0 else 0.0

    ef_mean, ef_lo, ef_hi = bootstrap_metric(ef_1pct)

    def recall_100(p: np.ndarray, y: np.ndarray) -> float:
        top = np.argsort(p)[::-1][:top_n_100]
        denom = float(y.sum())
        return float(y[top].sum()) / denom if denom > 0 else 0.0

    rec_mean, rec_lo, rec_hi = bootstrap_metric(recall_100)

    def prec_100(p: np.ndarray, y: np.ndarray) -> float:
        top = np.argsort(p)[::-1][:top_n_100]
        return float(y[top].mean()) if top.size else 0.0

    prec_mean, prec_lo, prec_hi = bootstrap_metric(prec_100)

    from sklearn.metrics import precision_recall_curve, auc

    prec_curve, rec_curve, _ = precision_recall_curve(y_true.astype(int), p_stable)
    auprc = float(auc(rec_curve, prec_curve))

    k = min(100, n)
    random_scores = [
        float(y_true[np.random.choice(n, k, replace=False)].mean()) for _ in range(n_random)
    ]

    top_1pct_hits = int(y_true[order[:top_n_1pct]].sum())
    effective_prev = top_1pct_hits / top_n_1pct if top_n_1pct > 0 else 0.0

    return {
        "ef_1pct": f"{ef_mean:.2f}x [{ef_lo:.2f}, {ef_hi:.2f}]",
        "auprc": f"{auprc:.3f}",
        "recall_100": f"{rec_mean:.1%} [{rec_lo:.1%}, {rec_hi:.1%}]",
        "precision_100": f"{prec_mean:.1%} [{prec_lo:.1%}, {prec_hi:.1%}]",
        "random_baseline": f"{np.mean(random_scores):.1%} +/- {np.std(random_scores):.1%}",
        "dataset_prevalence": f"{prevalence:.1%}",
        "top_1pct_hit_rate": f"{effective_prev:.1%}",
    }
