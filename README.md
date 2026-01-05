# Cathode Screening

A machine learning system for high-throughput screening of lithium-ion battery cathode materials. The system predicts thermodynamic stability from crystal structure using graph neural networks with calibrated uncertainty quantification.

## Problem

Discovering new cathode materials requires evaluating thermodynamic stability through Density Functional Theory (DFT) calculations. These calculations are computationally expensive, often taking hours per structure. With thousands of candidate materials, exhaustive DFT screening becomes prohibitively costly.

This project reduces that computational burden by providing fast, reliable stability predictions that identify promising candidates before committing to full DFT evaluation.

## Approach

The pipeline ingests crystal structures from the Materials Project database and applies domain-specific filters for cathode-relevant chemistries (lithium-oxygen systems with transition metals). Each structure is converted to a periodic graph representation where atoms are nodes and interatomic distances define edges.

A Crystal Graph Convolutional Neural Network (CGCNN) processes these graphs to predict energy above hull (E_hull), the primary metric for thermodynamic stability. The model outputs three quantities:

1. Predicted E_hull value
2. Learned uncertainty estimate (heteroscedastic)
3. Probability of stability (E_hull < 0.05 eV)

Training uses chemistry-aware data splits to prevent information leakage between structurally similar compounds. Post-hoc calibration ensures uncertainty estimates are reliable for downstream decision-making.

## Decision Framework

The outputs support a risk-controlled screening policy:

- Candidates with high stability probability and low uncertainty proceed to DFT validation
- Candidates with mixed signals are flagged for further analysis
- Candidates with low stability probability are deprioritized

This framework maximizes discovery rate while controlling false positives, reducing wasted DFT cycles on unstable materials.

## Technical Stack

- Crystal graph representation with radial basis function edge encoding
- 4-layer message passing network with attention pooling
- Huber loss for robust regression under heavy-tailed error distributions
- Conformal prediction for coverage-guaranteed prediction intervals
- Deep ensemble (K=5) for epistemic uncertainty quantification

## Quick Start

### Single Model Training

```bash
python scripts/04_train.py --config configs/train_cgcnn_ehull.yaml
```

### Ensemble Training

Train a K=5 deep ensemble with different random seeds:

```bash
# Train all 5 members sequentially
python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull.yaml --k 5

# Or train members in parallel (e.g., on a cluster)
python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull.yaml --k 5 --member 0
python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull.yaml --k 5 --member 1
# ... etc
```

Seeds are generated deterministically: `seed_i = base_seed + i` where `base_seed` comes from the config.

Outputs:
```
artifacts/models/<run_id>/
├── member_0/best.pt
├── member_1/best.pt
├── ...
└── ensemble_meta.json
```

### Conformal Calibration

After training, calibrate the prediction intervals:

```bash
python scripts/05_calibrate_conformal.py \
    --checkpoint artifacts/models/<run_id>/member_0/member_0/best.pt \
    --data-config configs/train_cgcnn_ehull.yaml \
    --alpha 0.10 \
    --output-dir artifacts/models/<run_id>/calibration
```

### Ensemble Inference

Run inference with the trained ensemble:

```bash
python scripts/07_predict_ensemble.py \
    --ensemble-dir artifacts/models/<run_id> \
    --data-config configs/train_cgcnn_ehull.yaml \
    --split test \
    --output predictions/ensemble_test.parquet
```

Output columns:
| Column | Description |
|--------|-------------|
| `material_id` | Material Project ID |
| `y_true` | Ground truth E_hull (if available) |
| `q50` | Ensemble mean of median predictions |
| `q10_cal`, `q90_cal` | Calibrated 80% prediction interval |
| `epistemic_var` | Variance of q50 across ensemble members |
| `epistemic_std` | Std dev (same units as E_hull) |
| `aleatoric_unc` | Half-width of calibrated interval |
| `total_unc` | Combined uncertainty |
| `p_stable` | Averaged probability E_hull < 0.05 |
| `decision` | KEEP / MAYBE / KILL |

### Ensemble Aggregation Rules

| Output | Formula |
|--------|---------|
| `q50` | mean(q50_k) across K members |
| `q10_raw` | mean(q10_k) across K members |
| `q90_raw` | mean(q90_k) across K members |
| `q10_cal` | q10_raw - Δ_lower (conformal) |
| `q90_cal` | q90_raw + Δ_upper (conformal) |
| `epistemic_var` | var(q50_k) across K members |
| `p_stable` | mean(sigmoid(logit_k)) - average probs, not logits |

## License

Proprietary. All rights reserved.

