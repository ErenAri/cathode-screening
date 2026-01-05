# Decision-Grade ML Specification for Li–O–TM Cathode Screening

**Version**: 1.0  
**Target**: Production decision system with calibrated uncertainty + OOD gating  
**Output Format**: KEEP / MAYBE / KILL with confidence scores

---

## 1. System Overview

### 1.1 Decision Philosophy
- **Decision quality > MAE**: We optimize for actionable triage, not point accuracy
- **Asymmetric costs**: False KEEP (expensive DFT/experiment on unstable) >> False KILL (miss a good candidate)
- **Two operational modes**: DFT-followup (cheap validation) vs. Experimental (expensive synthesis)

### 1.2 Architecture Summary
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE (K=5 seeds)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐                │
│  │  CGCNN[0]   │  │  CGCNN[1]   │  ...  │  CGCNN[4]   │                │
│  │ q10,q50,q90 │  │ q10,q50,q90 │       │ q10,q50,q90 │                │
│  │ embedding_g │  │ embedding_g │       │ embedding_g │                │
│  └─────────────┘  └─────────────┘       └─────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFORMAL CALIBRATION LAYER                          │
│  • Split-aligned calibration sets (val_calib from each seed's val)      │
│  • Learns qhat_lo, qhat_hi per ensemble member                          │
│  • Coverage guarantee: P(y ∈ [q10-qhat_lo, q90+qhat_hi]) ≥ 1-α          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OOD GATING MODULE                               │
│  Gate 1: Composition distance (Mahalanobis to training centroids)       │
│  Gate 2: Embedding kNN distance (k=10, aggregated over ensemble)        │
│  Gate 3: Ensemble disagreement (std of q50 predictions)                 │
│  → OOD score = weighted combination → flag if > threshold               │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DECISION THRESHOLD MODULE                            │
│  Mode A (DFT-followup):    utility = value(KEEP correct) - cost(DFT)    │
│  Mode B (Experimental):    utility = value(synth success) - cost(fail)  │
│  → Optimized thresholds τ_keep, τ_kill per mode                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    DECISION: KEEP / MAYBE / KILL
                    + confidence + OOD flag + explanation
```

---

## 2. Component Specifications

### 2.1 Deep Ensemble (K=5)

**Training Protocol**:
```python
SEEDS = [42, 123, 456, 789, 1011]  # Fixed for reproducibility
```

Each ensemble member `k`:
1. Uses **same architecture** (CGCNN with quantile heads)
2. Uses **same data splits** (chemistry-cluster based)
3. Different **random initialization** and **batch ordering**

**Aggregation Formulas**:

| Output | Formula | Interpretation |
|--------|---------|----------------|
| Ensemble q10 | `q10_ens = mean([q10_k for k in K])` | Averaged optimistic bound |
| Ensemble q50 | `q50_ens = mean([q50_k for k in K])` | Point estimate |
| Ensemble q90 | `q90_ens = mean([q90_k for k in K])` | Averaged pessimistic bound |
| Aleatoric uncertainty | `σ_ale = mean([q90_k - q10_k for k in K]) / 2.56` | Interval → std |
| Epistemic uncertainty | `σ_epi = std([q50_k for k in K])` | Disagreement |
| Total uncertainty | `σ_tot = sqrt(σ_ale² + σ_epi²)` | Combined |

**Why K=5**:
- K=3: Too few for stable disagreement estimates
- K=5: Sweet spot for compute vs. coverage
- K=10: Diminishing returns, 2x inference cost

**Checkpoint Artifacts**:
```
data/artifacts/ensemble/
├── member_seed42/best.pt
├── member_seed123/best.pt
├── member_seed456/best.pt
├── member_seed789/best.pt
├── member_seed1011/best.pt
└── ensemble_meta.json  # Seeds, train config, timestamps
```

---

### 2.2 Split-Aligned Conformal Calibration

**Problem**: Raw quantile predictions from pinball loss are miscalibrated under distribution shift.

**Solution**: Conformal prediction with split-aware calibration sets.

#### 2.2.1 Calibration Protocol

For each ensemble member `k`:

1. **Hold out calibration set**: Use 20% of validation set (not used in early stopping)
   ```python
   val_calib, val_stop = train_test_split(val_set, test_size=0.8, random_state=seed_k)
   ```

2. **Compute nonconformity scores on val_calib**:
   ```python
   # Lower bound nonconformity (how much q10 overestimates)
   E_lo[i] = q10_k[i] - y[i]  # Positive if q10 > y (undercover)
   
   # Upper bound nonconformity (how much q90 underestimates)  
   E_hi[i] = y[i] - q90_k[i]  # Positive if y > q90 (undercover)
   ```

3. **Compute quantile corrections**:
   ```python
   α = 0.10  # Target miscoverage (for 80% interval [q10, q90])
   n = len(val_calib)
   
   # Finite-sample correction
   q_level = ceil((n + 1) * (1 - α/2)) / n
   
   qhat_lo_k = quantile(E_lo, q_level)  # Correction for lower bound
   qhat_hi_k = quantile(E_hi, q_level)  # Correction for upper bound
   ```

4. **Calibrated prediction intervals**:
   ```python
   q10_calibrated = q10_raw - qhat_lo_k  # Widen lower bound
   q90_calibrated = q90_raw + qhat_hi_k  # Widen upper bound
   ```

#### 2.2.2 Ensemble Conformal Aggregation

**Option A (Conservative)**: Calibrate each member, then average
```python
q10_ens_cal = mean([q10_k - qhat_lo_k for k in K])
q90_ens_cal = mean([q90_k + qhat_hi_k for k in K])
```

**Option B (Aggressive)**: Average first, then calibrate on pooled residuals
```python
q10_ens_raw = mean([q10_k for k in K])
q90_ens_raw = mean([q90_k for k in K])
# Then run conformal on ensemble predictions using pooled calibration set
```

**Recommendation**: Use Option A (conservative) for decision-grade systems.

#### 2.2.3 Coverage Guarantee

For significance level α = 0.10:
$$P(y \in [q_{10}^{cal}, q_{90}^{cal}]) \geq 1 - \alpha = 0.90$$

This is a **finite-sample, distribution-free guarantee** that holds even under covariate shift (assuming exchangeability with calibration set).

**Calibration Artifacts**:
```
data/artifacts/calibration/
├── conformal_params.json
│   {
│     "alpha": 0.10,
│     "qhat_lo": [0.012, 0.015, 0.011, 0.014, 0.013],  # Per member
│     "qhat_hi": [0.018, 0.021, 0.017, 0.019, 0.020],
│     "qhat_lo_ensemble": 0.013,  # Pooled
│     "qhat_hi_ensemble": 0.019,
│     "n_calib_per_member": [89, 89, 89, 89, 89],
│     "empirical_coverage_val": 0.912,
│     "empirical_coverage_test": 0.897
│   }
└── calibration_diagnostics.png  # Reliability diagram
```

---

### 2.3 OOD Gating Module

**Purpose**: Flag inputs that are outside the training distribution where predictions are unreliable.

#### 2.3.1 Gate 1: Composition Distance

**Intuition**: Novel chemistries (e.g., new TM dopants) should be flagged.

**Implementation**:
```python
# Precompute on training set
train_compositions = [composition_fingerprint(f) for f in train_df["formula_pretty"]]
X_train = np.stack(train_compositions)  # [N_train, 9]  # 9 = len(vocab)

# Mahalanobis distance with robust covariance
from sklearn.covariance import EmpiricalCovariance
cov_estimator = EmpiricalCovariance().fit(X_train)
mu_train = X_train.mean(axis=0)
cov_inv = np.linalg.inv(cov_estimator.covariance_ + 1e-6 * np.eye(9))

def composition_mahalanobis(formula: str) -> float:
    x = composition_fingerprint(formula)
    delta = x - mu_train
    return float(np.sqrt(delta @ cov_inv @ delta))
```

**Threshold**: 
```python
# Compute on training set for reference distribution
train_dists = [composition_mahalanobis(f) for f in train_df["formula_pretty"]]
tau_comp_p95 = np.percentile(train_dists, 95)  # ~3.0 for Chi-squared(9)
tau_comp_p99 = np.percentile(train_dists, 99)  # ~3.8
```

#### 2.3.2 Gate 2: Embedding kNN Distance

**Intuition**: Structurally novel materials map to sparse regions in learned space.

**Implementation**:
```python
import faiss

# Extract graph embeddings from each ensemble member (before heads)
# g_k = model_k.pool(x, batch_index) after conv layers

# Aggregate embeddings: concatenate across ensemble
# g_full = concat([g_0, g_1, ..., g_4], dim=-1)  # [node_dim * K]

# Build FAISS index on training embeddings
d = node_dim * K  # 128 * 5 = 640
index = faiss.IndexFlatL2(d)
train_embeddings = get_all_train_embeddings(ensemble, train_loader)  # [N_train, d]
index.add(train_embeddings.astype(np.float32))

def embedding_knn_distance(g_query: np.ndarray, k: int = 10) -> float:
    """Mean distance to k nearest training neighbors."""
    distances, _ = index.search(g_query.reshape(1, -1).astype(np.float32), k)
    return float(distances[0].mean())
```

**Threshold**:
```python
# Reference distribution: kNN distances on training set (LOO or bootstrap)
train_knn_dists = []
for i in range(len(train_embeddings)):
    dists, _ = index.search(train_embeddings[i:i+1], k=11)  # k+1 to exclude self
    train_knn_dists.append(dists[0, 1:].mean())  # Skip self
tau_emb_p95 = np.percentile(train_knn_dists, 95)
tau_emb_p99 = np.percentile(train_knn_dists, 99)
```

#### 2.3.3 Gate 3: Ensemble Disagreement

**Intuition**: High epistemic uncertainty indicates unfamiliar input.

**Implementation**:
```python
def ensemble_disagreement(q50_predictions: List[float]) -> float:
    """Standard deviation of median predictions across ensemble."""
    return float(np.std(q50_predictions))
```

**Threshold**:
```python
# Reference: disagreement on validation set
val_disagreements = []
for sample in val_loader:
    q50s = [model_k(sample)[1] for model_k in ensemble]  # q50 from each
    val_disagreements.append(np.std([q.item() for q in q50s]))
tau_disagree_p95 = np.percentile(val_disagreements, 95)
tau_disagree_p99 = np.percentile(val_disagreements, 99)
```

#### 2.3.4 Combined OOD Score

```python
def compute_ood_score(
    formula: str,
    embedding: np.ndarray,
    q50_ensemble: List[float],
    train_stats: dict
) -> Tuple[float, dict]:
    """
    Compute normalized OOD score from all gates.
    
    Returns:
        ood_score: float in [0, 1], higher = more OOD
        gate_details: dict with individual gate values and flags
    """
    # Gate 1: Composition
    d_comp = composition_mahalanobis(formula)
    z_comp = (d_comp - train_stats["comp_mean"]) / train_stats["comp_std"]
    
    # Gate 2: Embedding kNN
    d_emb = embedding_knn_distance(embedding, k=10)
    z_emb = (d_emb - train_stats["emb_mean"]) / train_stats["emb_std"]
    
    # Gate 3: Disagreement
    d_disagree = ensemble_disagreement(q50_ensemble)
    z_disagree = (d_disagree - train_stats["disagree_mean"]) / train_stats["disagree_std"]
    
    # Weighted combination (composition most important for domain shift)
    weights = {"comp": 0.4, "emb": 0.35, "disagree": 0.25}
    z_combined = (weights["comp"] * z_comp + 
                  weights["emb"] * z_emb + 
                  weights["disagree"] * z_disagree)
    
    # Sigmoid to [0, 1]
    ood_score = 1.0 / (1.0 + np.exp(-z_combined + 2.0))  # Shift so ~0.1 at z=0
    
    return ood_score, {
        "d_comp": d_comp,
        "d_emb": d_emb,
        "d_disagree": d_disagree,
        "z_comp": z_comp,
        "z_emb": z_emb,
        "z_disagree": z_disagree,
        "flag_comp": d_comp > train_stats["tau_comp_p95"],
        "flag_emb": d_emb > train_stats["tau_emb_p95"],
        "flag_disagree": d_disagree > train_stats["tau_disagree_p95"],
    }
```

**OOD Artifacts**:
```
data/artifacts/ood/
├── train_stats.json
│   {
│     "comp_mean": 1.82, "comp_std": 0.91,
│     "tau_comp_p95": 3.21, "tau_comp_p99": 3.87,
│     "emb_mean": 12.4, "emb_std": 4.2,
│     "tau_emb_p95": 19.8, "tau_emb_p99": 24.1,
│     "disagree_mean": 0.018, "disagree_std": 0.012,
│     "tau_disagree_p95": 0.041, "tau_disagree_p99": 0.058
│   }
├── composition_stats.npz  # mu_train, cov_inv
└── embedding_index.faiss  # FAISS index for kNN
```

---

### 2.4 Utility-Optimized Decision Thresholds

**Problem**: Fixed thresholds (0.05, 0.10 eV) don't account for downstream costs.

**Solution**: Learn thresholds that maximize expected utility for each operational mode.

#### 2.4.1 Utility Framework

Define utility function $U(d, y)$ for decision $d$ and true $y$:

| Decision | y < 0.05 (stable) | 0.05 ≤ y < 0.10 (metastable) | y ≥ 0.10 (unstable) |
|----------|-------------------|------------------------------|---------------------|
| KEEP     | +V_keep           | +V_keep × 0.5                | -C_false_keep       |
| MAYBE    | -C_miss × 0.5     | 0                            | 0                   |
| KILL     | -C_miss           | -C_miss × 0.3                | +V_kill             |

**Mode A: DFT-followup** (cheap validation available)
```python
UTILITY_DFT = {
    "V_keep": 10.0,       # Value of finding stable candidate
    "V_kill": 1.0,        # Value of correctly rejecting unstable
    "C_false_keep": 2.0,  # Cost of DFT on false positive (cheap)
    "C_miss": 20.0,       # Cost of missing stable candidate
}
```

**Mode B: Experimental** (expensive synthesis, no cheap validation)
```python
UTILITY_EXP = {
    "V_keep": 100.0,      # High value of synthesis-ready candidate
    "V_kill": 5.0,        # Value of avoiding wasted synthesis
    "C_false_keep": 50.0, # Cost of failed synthesis (expensive!)
    "C_miss": 30.0,       # Cost of missing candidate (opportunity cost)
}
```

#### 2.4.2 Threshold Optimization

**Decision rule parameterized by (τ_keep, τ_kill)**:
```python
def decide(q90_cal: float, q10_cal: float, tau_keep: float, tau_kill: float) -> str:
    """
    Decision based on calibrated quantile bounds.
    
    KEEP: Pessimistic bound (q90) below keep threshold
    KILL: Optimistic bound (q10) above kill threshold
    MAYBE: Otherwise
    """
    if q90_cal < tau_keep:
        return "KEEP"
    elif q10_cal > tau_kill:
        return "KILL"
    else:
        return "MAYBE"
```

**Grid search for optimal thresholds**:
```python
def optimize_thresholds(
    q10_cal: np.ndarray,  # Calibrated lower bounds on val set
    q90_cal: np.ndarray,  # Calibrated upper bounds on val set
    y_true: np.ndarray,   # True Ehull values
    utility_params: dict,
    tau_keep_range: np.ndarray = np.arange(0.02, 0.15, 0.005),
    tau_kill_range: np.ndarray = np.arange(0.05, 0.20, 0.005)
) -> Tuple[float, float, float]:
    """Find (tau_keep, tau_kill) maximizing expected utility."""
    
    best_utility = -np.inf
    best_thresholds = (0.05, 0.10)
    
    for tau_keep in tau_keep_range:
        for tau_kill in tau_kill_range:
            if tau_keep >= tau_kill:
                continue  # Invalid: keep threshold must be stricter
            
            decisions = [decide(q90_cal[i], q10_cal[i], tau_keep, tau_kill) 
                        for i in range(len(y_true))]
            
            total_utility = sum(
                compute_utility(d, y, utility_params) 
                for d, y in zip(decisions, y_true)
            )
            
            if total_utility > best_utility:
                best_utility = total_utility
                best_thresholds = (tau_keep, tau_kill)
    
    return best_thresholds[0], best_thresholds[1], best_utility / len(y_true)
```

#### 2.4.3 Expected Thresholds

Based on typical utility ratios:

| Mode | τ_keep (eV) | τ_kill (eV) | Rationale |
|------|-------------|-------------|-----------|
| DFT-followup | 0.07-0.09 | 0.12-0.15 | Can afford more false positives |
| Experimental | 0.03-0.05 | 0.08-0.10 | Must be conservative on KEEP |

**Threshold Artifacts**:
```
data/artifacts/thresholds/
├── thresholds_dft.json
│   {
│     "mode": "dft_followup",
│     "tau_keep": 0.078,
│     "tau_kill": 0.135,
│     "expected_utility": 4.21,
│     "utility_params": {...},
│     "decision_distribution": {"KEEP": 0.15, "MAYBE": 0.45, "KILL": 0.40}
│   }
└── thresholds_exp.json
    {
      "mode": "experimental",
      "tau_keep": 0.042,
      "tau_kill": 0.095,
      "expected_utility": 12.8,
      ...
    }
```

---

## 3. Complete Inference Pipeline

```python
@dataclass
class DecisionOutput:
    material_id: str
    formula: str
    
    # Point estimates
    ehull_pred: float           # Ensemble q50 (eV)
    ehull_lower: float          # Calibrated q10 (eV)
    ehull_upper: float          # Calibrated q90 (eV)
    
    # Uncertainty decomposition
    uncertainty_aleatoric: float  # From interval width
    uncertainty_epistemic: float  # From ensemble disagreement
    uncertainty_total: float      # Combined
    
    # OOD assessment
    ood_score: float              # [0, 1], higher = more OOD
    ood_flag: bool                # True if any gate triggered
    ood_gates: Dict[str, bool]    # Individual gate flags
    
    # Decision
    decision: str                 # KEEP / MAYBE / KILL
    decision_confidence: float    # [0, 1] based on margin to thresholds
    decision_mode: str            # "dft_followup" or "experimental"
    
    # Explanation
    explanation: str              # Human-readable rationale


def predict_decision(
    material_id: str,
    formula: str,
    graph_npz_path: str,
    ensemble: List[nn.Module],
    calibration_params: dict,
    ood_stats: dict,
    threshold_params: dict,
    mode: str = "dft_followup"
) -> DecisionOutput:
    """Full decision pipeline for a single material."""
    
    # 1. Load and prepare graph
    graph = load_graph_npz(graph_npz_path)
    x, src, dst, e, batch = prepare_batch([graph])
    
    # 2. Run ensemble inference
    q10_raw, q50_raw, q90_raw = [], [], []
    embeddings = []
    
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            q10, q50, q90, _, _ = model(x, src, dst, e, batch)
            g = model.get_embedding(x, src, dst, e, batch)  # Add this method
        q10_raw.append(q10.item())
        q50_raw.append(q50.item())
        q90_raw.append(q90.item())
        embeddings.append(g.cpu().numpy())
    
    # 3. Ensemble aggregation
    q50_ens = np.mean(q50_raw)
    
    # 4. Conformal calibration
    qhat_lo = calibration_params["qhat_lo_ensemble"]
    qhat_hi = calibration_params["qhat_hi_ensemble"]
    q10_cal = np.mean(q10_raw) - qhat_lo
    q90_cal = np.mean(q90_raw) + qhat_hi
    
    # 5. Uncertainty decomposition
    sigma_ale = np.mean([q90_raw[k] - q10_raw[k] for k in range(5)]) / 2.56
    sigma_epi = np.std(q50_raw)
    sigma_tot = np.sqrt(sigma_ale**2 + sigma_epi**2)
    
    # 6. OOD gating
    embedding_concat = np.concatenate(embeddings, axis=-1)
    ood_score, ood_gates = compute_ood_score(
        formula, embedding_concat, q50_raw, ood_stats
    )
    ood_flag = any(ood_gates[f"flag_{g}"] for g in ["comp", "emb", "disagree"])
    
    # 7. Decision with mode-specific thresholds
    thresholds = threshold_params[mode]
    tau_keep = thresholds["tau_keep"]
    tau_kill = thresholds["tau_kill"]
    
    if ood_flag:
        decision = "MAYBE"  # Force MAYBE for OOD inputs
        decision_confidence = 0.0
        explanation = f"OOD detected: {', '.join(g for g in ['comp','emb','disagree'] if ood_gates[f'flag_{g}'])}"
    elif q90_cal < tau_keep:
        decision = "KEEP"
        margin = (tau_keep - q90_cal) / tau_keep
        decision_confidence = min(1.0, margin * 2)
        explanation = f"q90={q90_cal:.3f} < τ_keep={tau_keep:.3f}"
    elif q10_cal > tau_kill:
        decision = "KILL"
        margin = (q10_cal - tau_kill) / tau_kill
        decision_confidence = min(1.0, margin * 2)
        explanation = f"q10={q10_cal:.3f} > τ_kill={tau_kill:.3f}"
    else:
        decision = "MAYBE"
        decision_confidence = 0.5
        explanation = f"q10={q10_cal:.3f}, q90={q90_cal:.3f} overlap decision region"
    
    return DecisionOutput(
        material_id=material_id,
        formula=formula,
        ehull_pred=q50_ens,
        ehull_lower=q10_cal,
        ehull_upper=q90_cal,
        uncertainty_aleatoric=sigma_ale,
        uncertainty_epistemic=sigma_epi,
        uncertainty_total=sigma_tot,
        ood_score=ood_score,
        ood_flag=ood_flag,
        ood_gates=ood_gates,
        decision=decision,
        decision_confidence=decision_confidence,
        decision_mode=mode,
        explanation=explanation,
    )
```

---

## 4. Required Artifacts Summary

| Artifact | Path | Contents |
|----------|------|----------|
| Ensemble checkpoints | `data/artifacts/ensemble/member_seed*/best.pt` | Model weights + normalizer |
| Ensemble meta | `data/artifacts/ensemble/ensemble_meta.json` | Seeds, config |
| Conformal params | `data/artifacts/calibration/conformal_params.json` | qhat_lo, qhat_hi per member |
| Composition stats | `data/artifacts/ood/composition_stats.npz` | mu_train, cov_inv |
| Embedding index | `data/artifacts/ood/embedding_index.faiss` | FAISS index |
| Train stats | `data/artifacts/ood/train_stats.json` | Means, stds, thresholds |
| DFT thresholds | `data/artifacts/thresholds/thresholds_dft.json` | tau_keep, tau_kill |
| Exp thresholds | `data/artifacts/thresholds/thresholds_exp.json` | tau_keep, tau_kill |

---

## 5. Acceptance Tests

### 5.1 Calibration Tests

```python
def test_conformal_coverage(ensemble, test_loader, calibration_params, alpha=0.10):
    """
    MUST PASS: Empirical coverage ≥ (1 - α) on held-out test set.
    """
    y_true, q10_cal, q90_cal = [], [], []
    
    for batch in test_loader:
        predictions = run_calibrated_ensemble(ensemble, batch, calibration_params)
        y_true.extend(predictions["y_true"])
        q10_cal.extend(predictions["q10_calibrated"])
        q90_cal.extend(predictions["q90_calibrated"])
    
    y_true = np.array(y_true)
    q10_cal = np.array(q10_cal)
    q90_cal = np.array(q90_cal)
    
    coverage = np.mean((y_true >= q10_cal) & (y_true <= q90_cal))
    
    assert coverage >= (1 - alpha - 0.02), \
        f"Coverage {coverage:.3f} < {1-alpha-0.02:.3f} (target: {1-alpha:.2f})"
    
    print(f"✓ Conformal coverage: {coverage:.3f} (target: {1-alpha:.2f})")


def test_interval_sharpness(ensemble, test_loader, calibration_params):
    """
    SHOULD PASS: Intervals are reasonably sharp (not infinitely wide).
    """
    widths = []
    for batch in test_loader:
        predictions = run_calibrated_ensemble(ensemble, batch, calibration_params)
        widths.extend(predictions["q90_calibrated"] - predictions["q10_calibrated"])
    
    median_width = np.median(widths)
    p95_width = np.percentile(widths, 95)
    
    assert median_width < 0.15, f"Median interval width {median_width:.3f} > 0.15 eV"
    assert p95_width < 0.30, f"P95 interval width {p95_width:.3f} > 0.30 eV"
    
    print(f"✓ Interval sharpness: median={median_width:.3f}, p95={p95_width:.3f}")
```

### 5.2 Decision Quality Tests

```python
def test_keep_precision(ensemble, test_loader, calibration_params, threshold_params, mode):
    """
    MUST PASS: KEEP precision ≥ 0.85 for DFT mode, ≥ 0.95 for experimental mode.
    """
    decisions = []
    y_true = []
    
    for batch in test_loader:
        outputs = [predict_decision(..., mode=mode) for ... in batch]
        decisions.extend([o.decision for o in outputs])
        y_true.extend([o.y_true for o in outputs])  # From batch
    
    keep_mask = np.array([d == "KEEP" for d in decisions])
    y_true = np.array(y_true)
    
    if keep_mask.sum() == 0:
        print("⚠ No KEEP decisions made")
        return
    
    keep_precision = np.mean(y_true[keep_mask] < 0.05)
    
    min_precision = 0.85 if mode == "dft_followup" else 0.95
    assert keep_precision >= min_precision, \
        f"KEEP precision {keep_precision:.3f} < {min_precision} for mode {mode}"
    
    print(f"✓ KEEP precision ({mode}): {keep_precision:.3f} (min: {min_precision})")


def test_kill_precision(ensemble, test_loader, calibration_params, threshold_params, mode):
    """
    MUST PASS: KILL precision ≥ 0.90 (avoid killing good candidates).
    """
    decisions = []
    y_true = []
    
    for batch in test_loader:
        outputs = [predict_decision(..., mode=mode) for ... in batch]
        decisions.extend([o.decision for o in outputs])
        y_true.extend([o.y_true for o in outputs])
    
    kill_mask = np.array([d == "KILL" for d in decisions])
    y_true = np.array(y_true)
    
    if kill_mask.sum() == 0:
        print("⚠ No KILL decisions made")
        return
    
    kill_precision = np.mean(y_true[kill_mask] >= 0.10)
    
    assert kill_precision >= 0.90, f"KILL precision {kill_precision:.3f} < 0.90"
    
    print(f"✓ KILL precision ({mode}): {kill_precision:.3f}")


def test_decision_coverage(ensemble, test_loader, calibration_params, threshold_params, mode):
    """
    SHOULD PASS: Reasonable distribution of decisions (not all MAYBE).
    """
    decisions = []
    
    for batch in test_loader:
        outputs = [predict_decision(..., mode=mode) for ... in batch]
        decisions.extend([o.decision for o in outputs])
    
    n = len(decisions)
    n_keep = decisions.count("KEEP")
    n_maybe = decisions.count("MAYBE")
    n_kill = decisions.count("KILL")
    
    # At least 10% decisive (KEEP or KILL)
    decisive_rate = (n_keep + n_kill) / n
    assert decisive_rate >= 0.10, f"Only {decisive_rate:.1%} decisive (KEEP+KILL)"
    
    # Not too aggressive (at most 60% KILL for DFT mode, 80% for exp mode)
    max_kill = 0.60 if mode == "dft_followup" else 0.80
    assert n_kill / n <= max_kill, f"KILL rate {n_kill/n:.1%} > {max_kill:.0%}"
    
    print(f"✓ Decision distribution ({mode}): KEEP={n_keep/n:.1%}, MAYBE={n_maybe/n:.1%}, KILL={n_kill/n:.1%}")
```

### 5.3 OOD Detection Tests

```python
def test_ood_gate_separation(ensemble, id_loader, ood_loader, ood_stats):
    """
    MUST PASS: OOD samples have significantly higher OOD scores than ID samples.
    """
    id_scores = []
    for batch in id_loader:
        outputs = [compute_ood_score(..., ood_stats) for ... in batch]
        id_scores.extend([o[0] for o in outputs])
    
    ood_scores = []
    for batch in ood_loader:  # E.g., different chemistry system
        outputs = [compute_ood_score(..., ood_stats) for ... in batch]
        ood_scores.extend([o[0] for o in outputs])
    
    id_median = np.median(id_scores)
    ood_median = np.median(ood_scores)
    
    # OOD scores should be clearly higher
    assert ood_median > id_median * 2.0, \
        f"OOD median {ood_median:.3f} not >> ID median {id_median:.3f}"
    
    # AUC for discrimination
    from sklearn.metrics import roc_auc_score
    labels = [0] * len(id_scores) + [1] * len(ood_scores)
    scores = id_scores + ood_scores
    auc = roc_auc_score(labels, scores)
    
    assert auc >= 0.80, f"OOD detection AUC {auc:.3f} < 0.80"
    
    print(f"✓ OOD separation: ID median={id_median:.3f}, OOD median={ood_median:.3f}, AUC={auc:.3f}")


def test_ood_forces_maybe(ensemble, ood_loader, calibration_params, threshold_params, ood_stats):
    """
    MUST PASS: OOD-flagged samples should be routed to MAYBE, never KEEP.
    """
    for batch in ood_loader:
        outputs = [predict_decision(...) for ... in batch]
        
        for o in outputs:
            if o.ood_flag:
                assert o.decision == "MAYBE", \
                    f"OOD sample {o.material_id} got {o.decision} instead of MAYBE"
    
    print("✓ OOD samples correctly routed to MAYBE")
```

### 5.4 Ensemble Consistency Tests

```python
def test_ensemble_reduces_variance(single_model, ensemble, test_loader):
    """
    SHOULD PASS: Ensemble predictions have lower variance than single model.
    """
    single_errors = []
    ensemble_errors = []
    
    for batch in test_loader:
        # Single model
        q50_single = single_model(batch)[1]
        single_errors.extend(np.abs(q50_single - batch.y))
        
        # Ensemble
        q50_ens = np.mean([m(batch)[1] for m in ensemble], axis=0)
        ensemble_errors.extend(np.abs(q50_ens - batch.y))
    
    single_mae = np.mean(single_errors)
    ensemble_mae = np.mean(ensemble_errors)
    
    improvement = (single_mae - ensemble_mae) / single_mae
    assert improvement >= 0.05, f"Ensemble improvement {improvement:.1%} < 5%"
    
    print(f"✓ Ensemble MAE improvement: {improvement:.1%} ({single_mae:.4f} → {ensemble_mae:.4f})")


def test_epistemic_correlates_with_error(ensemble, test_loader):
    """
    SHOULD PASS: Epistemic uncertainty should correlate with actual error.
    """
    errors = []
    epistemic = []
    
    for batch in test_loader:
        q50s = [m(batch)[1] for m in ensemble]
        q50_ens = np.mean(q50s, axis=0)
        sigma_epi = np.std(q50s, axis=0)
        
        errors.extend(np.abs(q50_ens - batch.y))
        epistemic.extend(sigma_epi)
    
    from scipy.stats import spearmanr
    corr, pval = spearmanr(epistemic, errors)
    
    assert corr >= 0.20, f"Epistemic-error correlation {corr:.3f} < 0.20"
    
    print(f"✓ Epistemic-error correlation: {corr:.3f} (p={pval:.2e})")
```

---

## 6. Failure Cases & Mitigations

### 6.1 Calibration Failures

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Coverage << 1-α | `test_conformal_coverage` fails | Increase calibration set size; check for data leakage |
| Intervals too wide | `test_interval_sharpness` fails | Improve base model; use conformalized quantile regression |
| Coverage varies by region | Stratified coverage check | Conditional conformal calibration |

### 6.2 OOD Gate Failures

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| False positives (ID flagged OOD) | High MAYBE rate on validation | Raise thresholds; check composition vocab coverage |
| False negatives (OOD not flagged) | Manual inspection of failures | Add gates; lower thresholds |
| Gate correlation | Mutual information check | Remove redundant gates; re-weight |

### 6.3 Decision Threshold Failures

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| No KEEP decisions | `n_keep == 0` | Lower tau_keep; improve base model |
| Too many false KEEPs | KEEP precision < target | Raise tau_keep; add epistemic check |
| MAYBE bucket too large | > 70% MAYBE | Widen [tau_keep, tau_kill] gap |

### 6.4 Ensemble Failures

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Members too similar | Disagreement ~0 everywhere | Increase seed diversity; vary hyperparrams |
| One bad member | Outlier detection on member MAEs | Remove member; retrain |
| Slow inference | > 500ms per sample | Distill to single model; batch inference |

---

## 7. Implementation Checklist

### Phase 1: Ensemble Training
- [ ] Modify `04_train.py` to accept seed as argument
- [ ] Create `scripts/04a_train_ensemble.py` to train K=5 members
- [ ] Add embedding extraction hook to CGCNN forward pass
- [ ] Save ensemble_meta.json with training provenance

### Phase 2: Conformal Calibration
- [ ] Create `scripts/05_calibrate_conformal.py`
- [ ] Implement nonconformity score computation
- [ ] Implement quantile correction (qhat) computation
- [ ] Save conformal_params.json
- [ ] Add `test_conformal_coverage` to test suite

### Phase 3: OOD Gating
- [ ] Create `src/cathode_screening/inference/ood.py`
- [ ] Implement composition Mahalanobis distance
- [ ] Integrate FAISS for embedding kNN
- [ ] Implement ensemble disagreement scoring
- [ ] Compute and save train_stats.json
- [ ] Add `test_ood_gate_separation` to test suite

### Phase 4: Decision Thresholds
- [ ] Create `scripts/06_optimize_thresholds.py`
- [ ] Implement utility-based threshold grid search
- [ ] Generate thresholds for both modes
- [ ] Add `test_keep_precision`, `test_kill_precision` to test suite

### Phase 5: Inference Pipeline
- [ ] Create `src/cathode_screening/inference/decision_predictor.py`
- [ ] Implement `DecisionOutput` dataclass
- [ ] Implement `predict_decision()` full pipeline
- [ ] Add API endpoint in `src/cathode_screening/inference/api/`

### Phase 6: Integration Tests
- [ ] End-to-end test on held-out test set
- [ ] Performance benchmarking (latency, throughput)
- [ ] Decision distribution sanity checks

---

## 8. Appendix: Mathematical Details

### A.1 Pinball Loss (Quantile Regression)

For quantile τ ∈ (0, 1):
$$L_\tau(y, \hat{q}_\tau) = \begin{cases} \tau (y - \hat{q}_\tau) & \text{if } y \geq \hat{q}_\tau \\ (1 - \tau) (\hat{q}_\tau - y) & \text{if } y < \hat{q}_\tau \end{cases}$$

Equivalently:
$$L_\tau(y, \hat{q}_\tau) = (y - \hat{q}_\tau) \cdot (\tau - \mathbb{1}_{y < \hat{q}_\tau})$$

### A.2 Conformal Prediction Guarantee

For calibration set $\{(X_i, Y_i)\}_{i=1}^n$ and new point $(X_{n+1}, Y_{n+1})$ exchangeable with calibration:

$$P\left(Y_{n+1} \in \hat{C}(X_{n+1})\right) \geq 1 - \alpha$$

where $\hat{C}(X) = [\hat{q}_{10}(X) - \hat{q}_{lo}, \hat{q}_{90}(X) + \hat{q}_{hi}]$ and:
$$\hat{q}_{lo} = \text{Quantile}_{(1-\alpha/2)}(\{E_{lo,i}\}_{i=1}^n)$$
$$\hat{q}_{hi} = \text{Quantile}_{(1-\alpha/2)}(\{E_{hi,i}\}_{i=1}^n)$$

### A.3 Mahalanobis Distance

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

For composition fingerprints:
- $x \in \mathbb{R}^9$ (Li, O, Fe, Mn, Co, Ni, Ti, V, Cr fractions)
- $\mu$ = mean composition in training set
- $\Sigma$ = empirical covariance (regularized)

Under Gaussian assumption, $D_M^2 \sim \chi^2_9$, so $D_M \approx 3$ is 95th percentile.

### A.4 Utility-Optimal Threshold

For threshold τ, expected utility:
$$\mathbb{E}[U(\tau)] = \sum_{d \in \{K, M, X\}} P(D=d | \tau) \cdot \mathbb{E}[U(d, Y) | D=d]$$

Optimal threshold:
$$\tau^* = \arg\max_\tau \mathbb{E}[U(\tau)]$$

Solved via grid search on validation set.
