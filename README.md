# Cathode Screening: Machine Learning for Accelerated Battery Materials Discovery

A decision-grade machine learning framework for high-throughput virtual screening of Li-ion cathode materials. The system combines graph neural networks with calibrated uncertainty quantification to achieve a **2.3× improvement in discovery efficiency** over standard approaches.

## Key Results

| Metric | Baseline | This Work | Improvement |
|--------|----------|-----------|-------------|
| **DAF@10** | 0.71× | **1.64×** | +131% |
| **DAF@25** | 1.00× | **1.76×** | +76% |
| **AL Efficiency** | Random | **PI Acquisition** | +86% |

- **DAF@10 = 1.64×**: The model identifies 10 stable materials 1.64× faster than random screening
- **Active Learning**: Probability of Improvement (PI) acquisition finds 311/388 stable materials (80%) vs 167/388 (43%) with random sampling using identical query budgets

## Methodology

### Problem Formulation

The discovery of novel cathode materials for lithium-ion batteries requires evaluation of thermodynamic stability via Density Functional Theory (DFT). Each DFT calculation requires O(10²-10³) CPU-hours, rendering exhaustive screening of candidate spaces computationally intractable.

We formulate this as a sequential decision problem: given a pool of candidate structures, efficiently identify thermodynamically stable phases (energy above convex hull Eₕᵤₗₗ < 0.05 eV/atom) while minimizing the number of expensive DFT queries.

### Model Architecture

The screening pipeline employs a Crystal Graph Convolutional Neural Network (CGCNN) with the following specifications:

| Component | Specification |
|-----------|--------------|
| Node features | 92-dimensional CGCNN atom embeddings |
| Edge features | 41 RBF bins (cutoff = 8Å) |
| Message passing | 4 layers, 128 hidden units |
| Pooling | Multi-head attention (4 heads) |
| Output heads | μ(x), σ(x), P(stable\|x) |
| Ensemble size | K = 5 members |

The model outputs calibrated prediction intervals via conformal quantile regression and epistemic uncertainty estimates via ensemble disagreement.

### Data Splitting: SOAP-LOCO

We introduce SOAP-LOCO (Smooth Overlap of Atomic Positions - Leave One Cluster Out) splitting to prevent information leakage between structurally similar polymorphs:

1. Compute SOAP descriptors for all structures (r_cut=4Å, n_max=3, l_max=2)
2. Cluster structures via K-Means (K=50) in SOAP feature space
3. Assign entire clusters to train/val/test splits

This produces test sets with **6.9% OOD fraction** (structures dissimilar to training data), better simulating real discovery scenarios than composition-based splitting.

```bash
python scripts/03a_make_soap_loco_splits.py --config configs/train_cgcnn_ehull.yaml --n-clusters 50
```

### Active Learning

The framework includes a pool-based active learning loop with multiple acquisition functions:

| Acquisition | Formula | Description |
|-------------|---------|-------------|
| **PI** | P(y < τ) = Φ((τ - μ)/σ) | Probability of improvement over threshold |
| **EI** | E[max(τ - y, 0)] | Expected improvement |
| **UCB** | μ - κσ | Upper confidence bound (minimization) |
| **Uncertainty** | σ_epistemic | Pure exploration |

Empirical evaluation on a 764-sample pool demonstrates **PI achieves 1.86× higher discovery rate** than random sampling.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd cathode-screening

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: SOAP-LOCO splitting
pip install dscribe

# Optional: ALIGNN tiered architecture
pip install alignn dgl
```

## Usage

### Training Pipeline

```bash
# 1. Generate SOAP-LOCO splits
python scripts/03a_make_soap_loco_splits.py --config configs/train_cgcnn_ehull.yaml

# 2. Train 5-member ensemble
python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull_soap_loco.yaml --k 5

# 3. Generate predictions
python scripts/07_predict_ensemble.py \
    --ensemble-dir artifacts/models/<run_id> \
    --data-config configs/train_cgcnn_ehull_soap_loco.yaml \
    --split test \
    --output data/predictions/ensemble_test.parquet

# 4. Evaluate decision-grade metrics
python scripts/10_evaluate_new_metrics.py --predictions data/predictions/ensemble_test.parquet
```

### Active Learning Simulation

```bash
python scripts/13_run_active_learning.py \
    --predictions data/predictions/ensemble_soap_loco_test.parquet \
    --n-iterations 50 \
    --batch-size 10 \
    --output-dir data/reports/active_learning
```

## Evaluation Metrics

The framework prioritizes **decision quality** over point prediction accuracy:

| Metric | Definition | Target |
|--------|------------|--------|
| **DAF@N** | Queries to find N stable / (N / base_rate) | > 1.0 |
| **EF@k%** | Precision@k% / base_rate | > 1.0 |
| **ECE** | |P(stable) - actual_rate| averaged over bins | < 0.05 |
| **MACE** | Mean absolute calibration error | < 0.05 |

## Project Structure

```
cathode-screening/
├── configs/                    # Training configurations
├── data/
│   ├── raw/                    # Source data (Materials Project)
│   ├── processed/              # Graph-formatted data
│   ├── splits/                 # Train/val/test manifests
│   └── predictions/            # Model outputs
├── src/cathode_screening/
│   ├── datasets/               # Data loading and splitting
│   │   └── splits/
│   │       └── soap_loco.py    # SOAP-LOCO implementation
│   ├── models/
│   │   ├── cgcnn/              # CGCNN architecture
│   │   └── alignn/             # ALIGNN wrapper (tiered screening)
│   ├── evaluation/
│   │   ├── topk.py             # DAF, EF, Recall metrics
│   │   ├── calibration_metrics.py  # ECE, MACE
│   │   └── decision_calibration.py
│   ├── inference/
│   │   ├── decision_policy.py  # KEEP/MAYBE/KILL decisions
│   │   └── tiered_pipeline.py  # CGCNN→ALIGNN pipeline
│   └── active_learning/
│       ├── acquisition.py      # Acquisition functions
│       └── loop.py             # AL iteration loop
└── scripts/                    # CLI entry points
```

## References

1. Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, 120(14).

2. Bartók, A. P., et al. (2013). On representing chemical environments. *Physical Review B*, 87(18).

3. Choudhary, K., & DeCost, B. (2021). Atomistic Line Graph Neural Network for Improved Materials Property Predictions. *npj Computational Materials*, 7(1).

## License

Proprietary. All rights reserved.
