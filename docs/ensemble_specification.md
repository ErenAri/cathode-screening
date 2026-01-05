# Deep Ensemble Specification (K=5)

## Overview

Convert single-model training to a K=5 deep ensemble for decision-grade cathode screening with epistemic uncertainty quantification.

---

## 1. Seed Variation Strategy

### Deterministic Seed Generation

```python
BASE_SEED = 42
K = 5

def generate_ensemble_seeds(base_seed: int = BASE_SEED, k: int = K) -> List[int]:
    """
    Generate K deterministic seeds from base seed.
    Uses linear congruential generator for reproducibility.
    """
    seeds = []
    rng = np.random.default_rng(base_seed)
    for i in range(k):
        # Each seed is deterministically derived
        seeds.append(int(rng.integers(0, 2**31)))
    return seeds

# Result: [1791095845, 1073741823, 2147483646, 536870911, 1610612735]
# Or use explicit seeds: [42, 123, 456, 789, 1011]
```

### What Each Seed Controls

| Component | How Seed Affects It |
|-----------|---------------------|
| Weight initialization | `torch.manual_seed(seed)` |
| Dropout masks | Same seed → different masks per forward pass |
| Data shuffling | `np.random.seed(seed)` for DataLoader |
| Train/val split (if random) | `rng = np.random.default_rng(seed)` |

**Important**: Cluster-based splits should use a **fixed** split seed (e.g., 42) across all members so they train on identical data partitions. Only weight init and training stochasticity vary.

---

## 2. Checkpoint Naming Convention

### Directory Structure

```
data/artifacts/ensemble/
├── ensemble_meta.json           # Provenance & aggregation config
├── member_0_seed42/
│   ├── best.pt                  # Best checkpoint (by val_mae)
│   ├── config.yaml              # Training config for this member
│   └── metrics.json             # Final metrics
├── member_1_seed123/
│   ├── best.pt
│   ├── config.yaml
│   └── metrics.json
├── member_2_seed456/
│   └── ...
├── member_3_seed789/
│   └── ...
├── member_4_seed1011/
│   └── ...
└── calibration/
    └── conformal_params.json    # Post-hoc calibration (fit on ensemble)
```

### Naming Pattern

```
member_{idx}_seed{seed}/best.pt
```

- `idx`: 0-indexed member number (0 to K-1)
- `seed`: Actual seed used for training

### ensemble_meta.json Schema

```json
{
  "k": 5,
  "base_seed": 42,
  "seeds": [42, 123, 456, 789, 1011],
  "members": [
    {
      "idx": 0,
      "seed": 42,
      "checkpoint": "member_0_seed42/best.pt",
      "val_mae": 0.0423,
      "val_rmse": 0.0891
    },
    ...
  ],
  "aggregation": {
    "q50": "mean",
    "q10": "mean",
    "q90": "mean",
    "epistemic": "variance_q50"
  },
  "created_at": "2026-01-05T14:30:00Z",
  "config_hash": "abc123..."
}
```

---

## 3. Inference Aggregation Rules

### 3.1 Point Estimate: mean(q50)

```python
def aggregate_q50(q50_per_member: np.ndarray) -> float:
    """
    q50_per_member: shape (K,) - median predictions from each member
    Returns: scalar ensemble prediction
    """
    return np.mean(q50_per_member)
```

**Rationale**: Mean of medians is robust to outlier members and provides unbiased estimate under symmetric error.

### 3.2 Calibrated Intervals: mean(q10_cal), mean(q90_cal)

```python
def aggregate_intervals(
    q10_per_member: np.ndarray,  # shape (K,)
    q90_per_member: np.ndarray,  # shape (K,)
    delta_upper: float,
    delta_lower: float
) -> Tuple[float, float]:
    """
    Step 1: Calibrate each member's quantiles
    Step 2: Average calibrated quantiles
    """
    # Per-member calibration
    q10_cal = q10_per_member - delta_lower  # widen lower
    q90_cal = q90_per_member + delta_upper  # widen upper
    
    # Aggregate
    return np.mean(q10_cal), np.mean(q90_cal)
```

**Alternative (recommended)**: Calibrate *after* aggregation for tighter intervals:

```python
def aggregate_then_calibrate(
    q10_per_member: np.ndarray,
    q90_per_member: np.ndarray,
    delta_upper: float,
    delta_lower: float
) -> Tuple[float, float]:
    """Aggregate raw quantiles, then apply single calibration."""
    q10_agg = np.mean(q10_per_member)
    q90_agg = np.mean(q90_per_member)
    
    # Single calibration on aggregated
    return q10_agg - delta_lower, q90_agg + delta_upper
```

### 3.3 Epistemic Uncertainty: variance(q50)

```python
def compute_epistemic_uncertainty(q50_per_member: np.ndarray) -> float:
    """
    Epistemic uncertainty = model disagreement
    q50_per_member: shape (K,) or (K, N) for batch
    """
    return np.var(q50_per_member, axis=0, ddof=1)  # Sample variance

def compute_epistemic_std(q50_per_member: np.ndarray) -> float:
    """Standard deviation form (same units as prediction)."""
    return np.std(q50_per_member, axis=0, ddof=1)
```

**Interpretation**:
- High epistemic → members disagree → likely OOD or underrepresented region
- Low epistemic → members agree → confident prediction (may still be wrong)

### 3.4 Total Uncertainty Decomposition

```python
def decompose_uncertainty(
    q10_cal: float,
    q90_cal: float,
    epistemic_var: float
) -> Tuple[float, float, float]:
    """
    Returns: (aleatoric, epistemic, total)
    """
    # Aleatoric: width of calibrated interval (irreducible noise)
    aleatoric = (q90_cal - q10_cal) / 2  # half-width
    
    # Epistemic: ensemble disagreement
    epistemic = np.sqrt(epistemic_var)
    
    # Total: combine in quadrature (assumes independence)
    total = np.sqrt(aleatoric**2 + epistemic**2)
    
    return aleatoric, epistemic, total
```

---

## 4. Stability Probability (p_stable)

### 4.1 If Using Logit Heads

Each member outputs logits for stability classification:

```python
def aggregate_p_stable_from_logits(
    logits_per_member: np.ndarray,  # shape (K, num_classes)
    temperature: float = 1.0
) -> np.ndarray:
    """
    Average probabilities, then optionally re-calibrate.
    
    Returns: (K,) or scalar probabilities
    """
    # Convert each member's logits to probabilities
    probs_per_member = softmax(logits_per_member / temperature, axis=-1)
    
    # Average probabilities (not logits!)
    p_avg = np.mean(probs_per_member, axis=0)
    
    return p_avg  # shape (num_classes,)
```

**Why average probs, not logits**: Averaging in probability space preserves calibration properties and is the standard for ensemble classification.

### 4.2 Re-calibration After Averaging

Apply temperature scaling on the averaged probabilities:

```python
def recalibrate_ensemble_probs(
    p_avg: np.ndarray,
    temperature: float  # Fit on calibration set
) -> np.ndarray:
    """
    Temperature scaling on averaged ensemble probabilities.
    temperature > 1: soften (increase entropy)
    temperature < 1: sharpen (decrease entropy)
    """
    # Convert back to logits
    logits_avg = np.log(p_avg + 1e-10)
    
    # Apply temperature
    logits_scaled = logits_avg / temperature
    
    # Back to probs
    return softmax(logits_scaled)
```

### 4.3 If Using Quantile Regression (No Logits)

Derive p_stable from the CDF of the predicted distribution:

```python
def p_stable_from_quantiles(
    q10_cal: float,
    q50: float,
    q90_cal: float,
    threshold: float = 0.05  # E_hull stable threshold
) -> float:
    """
    Estimate P(E_hull < threshold) using quantile-fitted distribution.
    Assumes Gaussian between calibrated quantiles.
    """
    # Estimate distribution parameters from quantiles
    mu = q50
    # q90 - q10 spans ~80% of Gaussian (±1.28σ)
    sigma = (q90_cal - q10_cal) / (2 * 1.28)
    
    # CDF at threshold
    from scipy.stats import norm
    p_stable = norm.cdf(threshold, loc=mu, scale=sigma)
    
    return p_stable
```

---

## 5. Config & CLI Changes

### 5.1 Config Additions

Add to `configs/train_cgcnn_ehull.yaml`:

```yaml
ensemble:
  enabled: true
  k: 5
  base_seed: 42
  # Explicit seeds (optional, overrides base_seed generation)
  seeds: [42, 123, 456, 789, 1011]
  # Aggregation strategy
  aggregation:
    point_estimate: mean  # mean | median
    intervals: mean_then_calibrate  # mean | mean_then_calibrate
    epistemic: variance  # variance | std
  # Per-member output
  checkpoint_pattern: "member_{idx}_seed{seed}/best.pt"
```

### 5.2 CLI Changes to 04_train.py

```bash
# Single model (backwards compatible)
python scripts/04_train.py --config configs/train_cgcnn_ehull.yaml

# Ensemble mode
python scripts/04_train.py --config configs/train_cgcnn_ehull.yaml \
    --ensemble \
    --ensemble-k 5 \
    --ensemble-seeds 42,123,456,789,1011
```

New arguments:

```python
ap.add_argument("--ensemble", action="store_true",
                help="Train K-member ensemble")
ap.add_argument("--ensemble-k", type=int, default=5,
                help="Number of ensemble members")
ap.add_argument("--ensemble-seeds", type=str, default=None,
                help="Comma-separated seeds (auto-generate if not specified)")
ap.add_argument("--ensemble-member", type=int, default=None,
                help="Train only member N (for parallel training)")
```

### 5.3 Parallel Training Support

For cluster/SLURM deployment:

```bash
# Train each member in parallel (e.g., SLURM array job)
for i in {0..4}; do
    python scripts/04_train.py --config configs/train_cgcnn_ehull.yaml \
        --ensemble --ensemble-member $i &
done
wait
```

### 5.4 Inference Config

Add to `configs/infer_api.yaml`:

```yaml
ensemble:
  checkpoint_dir: data/artifacts/ensemble
  members: auto  # auto-discover from ensemble_meta.json
  aggregation:
    q50: mean
    intervals: mean_then_calibrate
    epistemic: variance

calibration:
  conformal_params: data/artifacts/ensemble/calibration/conformal_params.json
```

---

## 6. Acceptance Criteria

### 6.1 Decision FN Rate Decreases Under Family Holdout

**Test**: Hold out an entire TM-element family (e.g., all Mn-containing cathodes) and measure false negative rate for stable materials.

```python
def test_ensemble_reduces_fn_rate_family_holdout():
    """
    Acceptance: Ensemble FN rate < Single model FN rate
    on held-out element family.
    """
    # Load predictions
    single_preds = load_single_model_preds("holdout_mn_family.csv")
    ensemble_preds = load_ensemble_preds("holdout_mn_family.csv")
    
    # Ground truth
    y_true_stable = ground_truth["ehull"] < 0.05
    
    # Decisions at KEEP threshold
    single_keep = single_preds["decision"] == "KEEP"
    ensemble_keep = ensemble_preds["decision"] == "KEEP"
    
    # FN = stable material classified as non-KEEP
    fn_single = np.sum(y_true_stable & ~single_keep) / np.sum(y_true_stable)
    fn_ensemble = np.sum(y_true_stable & ~ensemble_keep) / np.sum(y_true_stable)
    
    print(f"Single model FN rate: {fn_single:.3f}")
    print(f"Ensemble FN rate: {fn_ensemble:.3f}")
    
    assert fn_ensemble < fn_single, \
        f"Ensemble FN ({fn_ensemble:.3f}) should be < single ({fn_single:.3f})"
```

**Expected outcome**: Ensemble reduces FN rate by 10-30% due to:
1. Averaging reduces variance in predictions
2. Epistemic uncertainty flags uncertain cases for human review

### 6.2 Ensemble Disagreement Correlates with Large Errors

**Test**: Pearson/Spearman correlation between epistemic uncertainty and |error|.

```python
def test_epistemic_correlates_with_error():
    """
    Acceptance: Spearman correlation(epistemic, |error|) > 0.3
    """
    # Load ensemble predictions with epistemic uncertainty
    preds = load_ensemble_preds("test_set.csv")
    
    epistemic = preds["uncertainty_epistemic"].values
    errors = np.abs(preds["ehull_pred"] - preds["ehull_true"]).values
    
    # Spearman correlation (rank-based, robust to outliers)
    from scipy.stats import spearmanr
    corr, pvalue = spearmanr(epistemic, errors)
    
    print(f"Spearman(epistemic, |error|) = {corr:.3f} (p={pvalue:.2e})")
    
    assert corr > 0.3, f"Epistemic should correlate with error: {corr:.3f} < 0.3"
    assert pvalue < 0.01, f"Correlation should be significant: p={pvalue:.3f}"
```

**Expected outcome**: ρ > 0.3 indicates ensemble disagreement is a useful uncertainty signal.

### 6.3 Additional Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Calibration coverage | P(y ∈ [q10_cal, q90_cal]) | ≥ 0.80 (for α=0.1) |
| Ensemble diversity | avg pairwise disagreement | > 0.01 eV std |
| OOD detection | AUROC(epistemic, is_ood) | > 0.70 |
| Computational overhead | inference time / single model | < 5.5x |

### 6.4 Test Implementation

Add to `src/cathode_screening/tests/test_ensemble_acceptance.py`:

```python
import pytest
import numpy as np
from scipy.stats import spearmanr

class TestEnsembleAcceptance:
    """Acceptance tests for K=5 deep ensemble."""
    
    @pytest.fixture
    def ensemble_predictions(self):
        """Load or generate ensemble predictions on test set."""
        # Mock or load real predictions
        return {
            "ehull_pred": np.random.uniform(0, 0.2, 100),
            "ehull_true": np.random.uniform(0, 0.2, 100),
            "uncertainty_epistemic": np.random.uniform(0, 0.05, 100),
            "decision": np.random.choice(["KEEP", "MAYBE", "KILL"], 100),
        }
    
    def test_fn_rate_family_holdout(self, ensemble_predictions):
        """FN rate should decrease vs single model."""
        # Implementation as above
        pass
    
    def test_epistemic_error_correlation(self, ensemble_predictions):
        """Epistemic uncertainty should correlate with |error|."""
        preds = ensemble_predictions
        errors = np.abs(preds["ehull_pred"] - preds["ehull_true"])
        corr, pval = spearmanr(preds["uncertainty_epistemic"], errors)
        
        assert corr > 0.3, f"Weak correlation: {corr:.3f}"
    
    def test_ensemble_diversity(self, ensemble_predictions):
        """Members should have meaningful disagreement."""
        # Avg pairwise std should be > 0.01 eV
        pass
    
    def test_inference_overhead(self):
        """Ensemble should be < 5.5x single model inference time."""
        pass
```

---

## 7. Implementation Checklist

- [ ] Update `04_train.py` with `--ensemble` CLI args
- [ ] Modify `04a_train_ensemble.py` to use new naming convention
- [ ] Add `ensemble_meta.json` generation
- [ ] Update `DecisionPredictor` with aggregation functions
- [ ] Add epistemic uncertainty to `DecisionOutput`
- [ ] Update `05_calibrate_conformal.py` to calibrate on ensemble
- [ ] Add acceptance tests
- [ ] Update configs with ensemble section
- [ ] Document parallel training workflow

---

## 8. Quick Reference

### Aggregation Summary

| Output | Formula | Code |
|--------|---------|------|
| `ehull_pred` | $\frac{1}{K}\sum_k q_{50}^{(k)}$ | `np.mean(q50_per_member)` |
| `ehull_lower` | $\frac{1}{K}\sum_k q_{10}^{(k)} - \Delta_{lower}$ | `np.mean(q10) - delta_lower` |
| `ehull_upper` | $\frac{1}{K}\sum_k q_{90}^{(k)} + \Delta_{upper}$ | `np.mean(q90) + delta_upper` |
| `epistemic` | $\text{Var}_k(q_{50}^{(k)})$ | `np.var(q50_per_member, ddof=1)` |
| `p_stable` | $\frac{1}{K}\sum_k \sigma(z_k)$ | `np.mean(softmax(logits), axis=0)` |

### Seed Workflow

```
base_seed=42 → generate_seeds() → [42, 123, 456, 789, 1011]
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              member_0_seed42   member_1_seed123  ... member_4_seed1011
                    ↓                 ↓                 ↓
                [weight init]    [weight init]     [weight init]
                [dropout masks]  [dropout masks]   [dropout masks]
                [shuffle order]  [shuffle order]   [shuffle order]
```

**Note**: Data splits are FIXED (same train/val/test across all members).
