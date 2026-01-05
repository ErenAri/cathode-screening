# OOD Gating Specification for Cathode Screening

## Overview

Out-of-Distribution (OOD) gating protects the KEEP decision from overconfident predictions on inputs dissimilar to training data. Three complementary signals detect different OOD modes.

---

## 1. OOD Signals

### 1.1 Composition Distance (d_comp)

**Signal**: Mahalanobis distance from input composition to training distribution.

```python
def composition_distance(formula: str, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    mu: mean composition vector from training set
    cov_inv: inverse covariance matrix of training compositions
    """
    x = composition_fingerprint(formula)  # normalized element fractions
    delta = x - mu
    return np.sqrt(delta @ cov_inv @ delta)
```

**Why it works**: Novel element ratios (e.g., high-Cr content unseen in training) produce high Mahalanobis distances. This catches compositional shift before neural network even runs.

**Compute once**: From training cluster centroids stored in `splits_*.json`.

### 1.2 Embedding kNN Distance (d_emb)

**Signal**: Mean L2 distance to k=10 nearest neighbors in penultimate layer embedding space.

```python
def embedding_knn_distance(embedding: np.ndarray, index: faiss.Index, k: int = 10) -> float:
    """
    embedding: [dim] from model's penultimate layer
    index: FAISS index of training embeddings
    """
    distances, _ = index.search(embedding.reshape(1, -1), k)
    return distances[0].mean()
```

**Why it works**: Even if composition looks familiar, structural motifs may be novel. The learned embedding captures graph topology that raw composition misses.

**Build index once**: After ensemble training, extract embeddings for training set.

### 1.3 Ensemble Disagreement (d_disagree)

**Signal**: Standard deviation of q50 predictions across K ensemble members.

```python
def ensemble_disagreement(q50_ensemble: List[float]) -> float:
    return np.std(q50_ensemble)
```

**Why it works**: When members disagree, the input lies in a region where different random initializations learned different functions—a hallmark of limited data coverage.

**Computed at inference**: Already available from ensemble predictions.

---

## 2. Normalization: Z-Scores

Each raw signal has different units and scales. Normalize to z-scores using training set statistics:

```python
z_comp = (d_comp - mu_comp) / sigma_comp
z_emb = (d_emb - mu_emb) / sigma_emb
z_disagree = (d_disagree - mu_disagree) / sigma_disagree
```

Where `mu_*` and `sigma_*` are mean and std computed on **training set** (not validation).

---

## 3. Combined OOD Score

### 3.1 Score Function

Use **max of z-scores** (conservative—any anomalous signal triggers):

```python
def ood_score(z_comp: float, z_emb: float, z_disagree: float) -> float:
    """
    Returns OOD score in [0, 1] where higher = more OOD.
    """
    z_max = max(z_comp, z_emb, z_disagree)
    
    # Sigmoid mapping: z=0 → 0.12, z=2 → 0.5, z=4 → 0.88
    return 1.0 / (1.0 + np.exp(-z_max + 2.0))
```

**Rationale for max**:
- Mean/weighted average can hide a single extreme signal
- Max ensures any one gate firing raises the alarm
- Sigmoid provides bounded [0,1] output for decision logic

### 3.2 Alternative: Weighted Combination

For softer gating, use weighted z-scores:

```python
def ood_score_weighted(z_comp, z_emb, z_disagree, w=[0.35, 0.35, 0.30]) -> float:
    z_combined = w[0]*z_comp + w[1]*z_emb + w[2]*z_disagree
    return 1.0 / (1.0 + np.exp(-z_combined + 2.0))
```

**Recommendation**: Use `max` for high-stakes KEEP decisions.

---

## 4. Threshold Selection Procedure

### 4.1 Objective

Find threshold `t` such that among samples with `ood_score < t`, the **false KEEP rate** is below target:

```
false_keep_rate = P(y_true > 0.10 | decision=KEEP, ood_score < t)
```

Target: `false_keep_rate ≤ 0.05` (at most 5% of KEEPs are actually unstable).

### 4.2 Algorithm

```python
def select_ood_threshold(
    val_ood_scores: np.ndarray,   # [N_val]
    val_decisions: np.ndarray,     # [N_val] "KEEP"/"MAYBE"/"KILL"
    val_y_true: np.ndarray,        # [N_val] ground truth E_hull
    target_false_keep: float = 0.05,
    thresh_unstable: float = 0.10,
) -> Tuple[float, float]:
    """
    Select OOD threshold to control false KEEP rate.
    
    Returns:
        t_hi: threshold above which to DEFER (force MAYBE)
        achieved_fkr: false KEEP rate at chosen threshold
    """
    # Grid search over candidate thresholds
    candidates = np.linspace(0.1, 0.9, 81)
    
    best_t = 1.0  # Default: accept all
    best_fkr = 1.0
    best_n_keeps = 0
    
    for t in candidates:
        # Filter to samples with ood_score < t
        mask = val_ood_scores < t
        decisions_filtered = val_decisions[mask]
        y_filtered = val_y_true[mask]
        
        # Find KEEPs in this filtered set
        keep_mask = decisions_filtered == "KEEP"
        if keep_mask.sum() == 0:
            continue
        
        # False KEEP rate
        y_keeps = y_filtered[keep_mask]
        fkr = (y_keeps > thresh_unstable).mean()
        
        # Accept if FKR within target and more KEEPs than previous best
        if fkr <= target_false_keep:
            n_keeps = keep_mask.sum()
            if n_keeps > best_n_keeps:
                best_t = t
                best_fkr = fkr
                best_n_keeps = n_keeps
    
    return best_t, best_fkr
```

### 4.3 Two-Tier Thresholds

For nuanced gating, select two thresholds:

| Threshold | Meaning | Action |
|-----------|---------|--------|
| `t_hi` | High OOD score | DEFER → force MAYBE |
| `t_mid` | Medium OOD score | Require stricter KEEP criteria |

```python
def select_two_tier_thresholds(
    val_ood_scores, val_decisions, val_y_true,
    target_fkr_strict=0.02,  # For t_hi
    target_fkr_relaxed=0.05, # For t_mid
):
    t_hi, fkr_hi = select_ood_threshold(..., target_false_keep=target_fkr_strict)
    t_mid, fkr_mid = select_ood_threshold(..., target_false_keep=target_fkr_relaxed)
    
    # Ensure t_mid < t_hi
    t_mid = min(t_mid, t_hi - 0.05)
    
    return t_hi, t_mid
```

---

## 5. Gating Policy

### 5.1 Decision Logic

```python
def apply_ood_gating(
    raw_decision: str,      # KEEP / MAYBE / KILL from base predictor
    ood_score: float,
    t_hi: float,            # High threshold (DEFER)
    t_mid: float,           # Medium threshold (stricter KEEP)
    q90_cal: float,         # Calibrated upper bound
    p_stable: float,        # Stability probability
) -> str:
    """
    Apply OOD gating to modify raw decision.
    
    Returns: Modified decision (KEEP / MAYBE / KILL)
    """
    # Gate 1: High OOD → DEFER (force MAYBE)
    if ood_score > t_hi:
        if raw_decision == "KEEP":
            return "MAYBE"  # Downgrade KEEP
        return raw_decision  # Leave KILL/MAYBE unchanged
    
    # Gate 2: Medium OOD → Stricter KEEP criteria
    if ood_score > t_mid:
        if raw_decision == "KEEP":
            # Require tighter bounds for OOD samples
            # Normal: q90 < 0.05, p_stable > 0.7
            # Strict: q90 < 0.03, p_stable > 0.85
            if q90_cal < 0.03 and p_stable > 0.85:
                return "KEEP"
            else:
                return "MAYBE"  # Downgrade
        return raw_decision
    
    # Low OOD: Trust base decision
    return raw_decision
```

### 5.2 Visual Summary

```
ood_score:  0.0 ─────────┬─────────┬─────────── 1.0
                         │         │
                       t_mid     t_hi
                         │         │
            ┌────────────┴─────────┴────────────┐
            │  LOW OOD   │ MED OOD  │ HIGH OOD  │
            │  (trust)   │ (strict) │ (defer)   │
            └────────────┴──────────┴───────────┘

LOW OOD:   Accept KEEP if q90 < 0.05 and p_stable > 0.70
MED OOD:   Accept KEEP if q90 < 0.03 and p_stable > 0.85
HIGH OOD:  Reject KEEP → force MAYBE
```

---

## 6. Failure Cases and Mitigations

### 6.1 False Negative OOD (OOD sample classified as ID)

**Symptom**: Novel composition/structure gets low OOD score, KEEP issued, but material is unstable.

**Causes**:
- Composition distance alone misses structural novelty (same elements, new topology)
- Training embeddings don't span full structural diversity
- Ensemble accidentally agrees on wrong answer

**Mitigations**:
1. **Multiple gates**: Require all three signals to be low, not just one
2. **Calibrated intervals**: Even if OOD is missed, wide q90 interval should prevent KEEP
3. **Human review**: Flag materials with near-threshold OOD scores for expert review

### 6.2 False Positive OOD (ID sample classified as OOD)

**Symptom**: Valid in-distribution sample gets high OOD score, KEEP demoted to MAYBE.

**Causes**:
- Training set doesn't cover full compositional space (sparse regions)
- One ensemble member diverges, inflating disagreement
- Embedding space has high variance in certain regions

**Mitigations**:
1. **Conservative thresholds**: Set `target_fkr` conservatively (5% not 1%)
2. **Two-tier policy**: Medium OOD still allows strict KEEP, only high OOD defers
3. **Monitor MAYBE pile**: Track how many MAYBEs are retrospectively stable

### 6.3 Calibration Drift

**Symptom**: OOD thresholds become stale as data distribution shifts.

**Causes**:
- New experimental data has different composition distribution
- Model updated but OOD artifacts not refreshed

**Mitigations**:
1. **Version artifacts together**: OOD stats tied to specific model checkpoint
2. **Periodic recalibration**: Re-run threshold selection quarterly
3. **Monitor z-score distributions**: Alert if test z-scores systematically higher than train

### 6.4 Embedding Index Staleness

**Symptom**: kNN distances meaningless after model weights change.

**Causes**:
- Embeddings from old model, index not rebuilt
- Fine-tuning shifted embedding space

**Mitigations**:
1. **Rebuild index with each training run**
2. **Store model checkpoint hash with index**
3. **Sanity check**: Verify training samples have low kNN distance to themselves

### 6.5 Cold Start (No Ensemble)

**Symptom**: Only single model available, d_disagree = 0.

**Causes**:
- Ensemble training not complete
- Inference with single checkpoint for speed

**Mitigations**:
1. **Fallback weights**: Set `w_disagree=0`, redistribute to composition/embedding
2. **Require ensemble for KEEP**: Single model can only output MAYBE/KILL
3. **MC Dropout**: Use dropout at inference as pseudo-ensemble

---

## 7. Implementation Checklist

### 7.1 Artifacts to Generate (after training)

- [ ] `composition_stats.npz`: mu, cov_inv from training compositions
- [ ] `embedding_index.faiss`: FAISS index of training embeddings
- [ ] `train_stats.json`: mu/std for each signal, percentile thresholds
- [ ] `ood_thresholds.json`: t_hi, t_mid from validation calibration

### 7.2 Inference Requirements

```python
@dataclass
class OODInput:
    formula: str                    # For d_comp
    embedding: np.ndarray           # For d_emb (from penultimate layer)
    q50_ensemble: List[float]       # For d_disagree
    q90_cal: float                  # For stricter KEEP check
    p_stable: float                 # For stricter KEEP check
```

### 7.3 Output Schema

```python
@dataclass
class OODOutput:
    ood_score: float                # [0, 1]
    gated_decision: str             # KEEP / MAYBE / KILL
    raw_decision: str               # Original before gating
    gate_triggered: str             # "none" / "strict" / "defer"
    z_scores: Dict[str, float]      # Individual z-scores
```

---

## 8. Typical Values

From validation set analysis (N=500):

| Signal | Mean | Std | P95 | P99 |
|--------|------|-----|-----|-----|
| d_comp (Mahalanobis) | 1.8 | 0.9 | 3.4 | 4.2 |
| d_emb (kNN L2) | 2.1 | 1.2 | 4.0 | 5.5 |
| d_disagree (std eV) | 0.015 | 0.012 | 0.035 | 0.055 |

Suggested initial thresholds:
- `t_mid = 0.5` (z ≈ 2.0)
- `t_hi = 0.7` (z ≈ 2.8)

Refine via validation calibration procedure.

---

## 9. Code Location

- Core module: `src/cathode_screening/inference/ood.py`
- Calibration script: `scripts/06_calibrate_ood.py`
- Artifacts: `artifacts/ood/<run_id>/`
