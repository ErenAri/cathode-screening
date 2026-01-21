import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_curve, auc


def grounded_win_report(
    test_set: List[Dict],
    ensemble,
    calibrator,
    n_random: int = 200,
) -> Dict[str, Any]:
    """
    Generate a production-grade grounded win report.

    Metrics:
    - EF@1% +/- 95% CI (Bootstrap)
    - Recall@100 +/- 95% CI
    - Precision@100 +/- 95% CI
    - AUPRC (Threshold-free headline)
    - Random baseline (averaged over trials)
    - Effective prevalence (top-1% hit rate)

    Args:
        test_set: List of dicts with 'structure' and 'is_stable' keys (and optionally 'e_hull')
        ensemble: Model ensemble with .predict(structure) method
        calibrator: Fitted CalibratedPredictor
        n_random: Number of random stratifications for baseline
    """
    # 1. Generate predictions and align with ground truth
    #    (Convert to arrays for fast bootstrapping)
    p_stable = np.array(
        [
            calibrator.predict_p_stable(ensemble.predict(item["structure"]))
            for item in test_set
        ]
    )
    y_true = np.array([item["is_stable"] for item in test_set]).astype(int)
    n = len(y_true)

    # 2. Define fixed window sizes
    #    EF@1%: top 1% of corpus size
    top_n_1pct = max(1, int(0.01 * n))
    #    Fixed 100 candidates
    top_n_100 = min(100, n)

    prevalence = y_true.mean()

    # Sort once
    order = np.argsort(p_stable)[::-1]

    # === Bootstrap (fast, array-based) ===
    def bootstrap_metric(metric_fn, n_boot=1000):
        scores = []
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            scores.append(metric_fn(p_stable[idx], y_true[idx]))
        return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    # EF@1% (fixed top_n)
    def ef_1pct(p, y):
        top = np.argsort(p)[::-1][:top_n_1pct]
        hit_rate = y[top].mean()
        prev = y.mean()
        return hit_rate / prev if prev > 0 else 0

    ef_mean, ef_lo, ef_hi = bootstrap_metric(ef_1pct)

    # Recall@100
    def recall_100(p, y):
        top = np.argsort(p)[::-1][:top_n_100]
        return y[top].sum() / y.sum() if y.sum() > 0 else 0

    rec_mean, rec_lo, rec_hi = bootstrap_metric(recall_100)

    # Precision@100
    def prec_100(p, y):
        top = np.argsort(p)[::-1][:top_n_100]
        return y[top].mean()

    prec_mean, prec_lo, prec_hi = bootstrap_metric(prec_100)

    # AUPRC
    prec_curve, rec_curve, _ = precision_recall_curve(
        y_true.astype(int), p_stable
    )
    auprc = auc(rec_curve, prec_curve)

    # Random baseline (200 trials, handle n < 100)
    k = min(100, n)
    random_scores = [
        y_true[np.random.choice(n, k, replace=False)].mean()
        for _ in range(n_random)
    ]

    # Effective prevalence reporting
    top_1pct_hits = y_true[order[:top_n_1pct]].sum()
    effective_prev = top_1pct_hits / top_n_1pct

    return {
        "ef_1pct": f"{ef_mean:.2f}x [{ef_lo:.2f}, {ef_hi:.2f}]",
        "auprc": f"{auprc:.3f}",
        "recall_100": f"{rec_mean:.1%} [{rec_lo:.1%}, {rec_hi:.1%}]",
        "precision_100": f"{prec_mean:.1%} [{prec_lo:.1%}, {prec_hi:.1%}]",
        "random_baseline": f"{np.mean(random_scores):.1%} +/- {np.std(random_scores):.1%}",
        "dataset_prevalence": f"{prevalence:.1%}",
        "top_1pct_hit_rate": f"{effective_prev:.1%}",
    }
