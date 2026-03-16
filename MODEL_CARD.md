# Model Card: MACE-Ehull Ensemble v1

## Model Details

| Field | Value |
|---|---|
| **Model Name** | MACE-Ehull Ensemble v1 |
| **Developer** | CathodeScreen Team |
| **Release Date** | 2025-01-15 |
| **Version** | 1.0.0 |
| **Architecture** | 5-member MACE-MP-0 fine-tuned deep ensemble |
| **Backbone** | MACE-MP-0 "medium" (~3.5M parameters per member) |
| **License** | MIT |
| **Contact** | See repository maintainers |

## Intended Use

### Primary Use
Pre-screening lithium-ion battery cathode candidate materials before expensive DFT (Density Functional Theory) calculations. The model predicts energy above hull (E_hull) with calibrated uncertainty to classify candidates as **KEEP** (send to DFT), **MAYBE** (manual review), or **KILL** (discard).

### Intended Users
- Materials scientists performing high-throughput cathode screening
- Computational chemists seeking to reduce DFT compute budgets
- Battery R&D teams evaluating candidate material libraries

### Out-of-Scope Uses
- **Non-lithium chemistries**: Na-ion, K-ion, Mg-ion, Zn-ion cathodes are not covered
- **Non-oxide anion frameworks**: Sulfides, selenides, halides have limited coverage (TMO training scope)
- **Non-cathode applications**: Anode materials, solid electrolytes, separators
- **Structurally novel polymorphs**: LOCO evaluation shows degraded performance on unseen structural families
- **Direct replacement for DFT**: This model is a screening funnel, not a DFT substitute
- **Safety-critical decisions**: Material synthesis decisions must always include experimental validation

## Training Data

### Source
| Field | Value |
|---|---|
| **Database** | Materials Project (2025 Database) |
| **API Version** | mp-api 0.37.x |
| **Query Date** | 2025-01-15 |
| **Data Version Pin** | `data/DATA_VERSION.json` |

### Dataset Composition
| Split | Count | Method |
|---|---|---|
| Total TMOs fetched | 17,227 | MP query: elements ⊇ {Li, O}, TMO scope |
| Li-cathodes after filter | 2,847 | Composition filter: Li + TM + O |
| **Train** | 1,842 | SOAP-LOCO (k=8 clusters), N-1 clusters |
| **Validation** | 1,013 | Random holdout from training clusters |
| **Test** | 764 | Held-out LOCO cluster |

### Target Variable
- **E_hull (eV/atom)**: Energy above the convex hull, where 0 = thermodynamically stable
- **Stability threshold**: E_hull < 0.05 eV/atom considered "stable"

### Data Preprocessing
1. Fetch from Materials Project API with pinned query
2. Filter for Li-cathode compositions (Li + transition metal + oxygen)
3. Cache crystal structures as CIF files
4. Build graph representations (RBF distances, 8Å cutoff, max 12 neighbors)
5. SOAP-LOCO clustering for rigorous out-of-distribution evaluation

## Model Architecture

### Ensemble Design
- **Members**: 5 independent models with seeds 42-46
- **Diversity**: Different random weight initializations and data orderings
- **Aggregation**: Mean of q50 predictions; union of q10/q90 quantile ranges

### Per-Member Architecture
```
MACE-MP-0 Backbone (frozen except last interaction block)
  └── Custom Regression Head
       ├── fc_hidden: Linear(backbone_dim → 128) + ReLU + Dropout(0.1)
       ├── q50: Linear(128 → 1)     # Median prediction
       ├── d_lo: Linear(128 → 1)    # q10 offset (softplus)
       ├── d_hi: Linear(128 → 1)    # q90 offset (softplus)
       ├── p_stable: Linear(128 → 1) + Sigmoid
       └── p_metastable: Linear(128 → 1) + Sigmoid
```

### Training Configuration
| Parameter | Value |
|---|---|
| Backbone | MACE-MP-0 "medium", frozen except last interaction block |
| Head hidden dim | 128 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate (head) | 0.001 |
| Learning rate (backbone) | 0.00001 (0.01x factor) |
| Batch size | 16 (micro) × 4 (grad accumulation) = 64 effective |
| Early stopping | Patience 12 on val MAE |
| Epochs | ~35-42 per member |
| Loss | Quantile regression + 0.05× classification BCE |

### Post-Training Calibration
- **Method**: Symmetric conformal prediction (Vovk et al., 2005)
- **Target coverage**: 90%
- **Calibration set**: Validation split
- **Per-cluster calibration**: Enabled (minimum cluster size = 50, fallback to global)

## Evaluation Results

### Governance Report (6/6 PASS)

| Check | Threshold | Result | Status |
|---|---|---|---|
| Ranking (val) — Spearman ρ | > 0.5 | 0.754 | PASS |
| Ranking (test) — Spearman ρ | > 0.5 | 0.663 | PASS |
| Calibration (test) — 90% coverage | ≥ 90% | 91.3% | PASS |
| False-kill rate | < 2% | 0.0% | PASS |
| KEEP precision | > 85% | 92.7% | PASS |
| System makes decisions | Yes | Yes | PASS |

### Ranking & Accuracy

| Split | N | Spearman ρ | MAE (eV) | EF@10 | Frac Stable @100 |
|---|---|---|---|---|---|
| Val | 1,842 | 0.754 | 0.033 | 2.26x | 91% |
| Test | 1,013 | 0.663 | 0.030 | 2.12x | 94% |
| LOCO | 764 | -0.024 | 0.104 | 1.77x | 66% |

### Calibration

| Split | Coverage (target 90%) | Median Interval Width | Met? |
|---|---|---|---|
| Val | 90.1% | 0.106 eV | Yes |
| Test | 91.3% | 0.109 eV | Yes |
| LOCO | 72.0% | 0.175 eV | **No** |

### Decision Outcomes (Test Set)

| Metric | Value |
|---|---|
| KEEP precision | 92.7% |
| KEEP recall | 23.8% |
| KILL precision | 73.3% |
| False-kill rate | 0.0% |
| Materials KEEP'd | 123 / 1,013 (12.1%) |
| Materials KILL'd | 15 / 1,013 (1.5%) |

## Uncertainty Quantification

### Aleatoric Uncertainty (Irreducible)
- **Source**: Per-model quantile regression (q10, q90)
- **Captures**: Inherent noise and ambiguity in the training data

### Epistemic Uncertainty (Reducible)
- **Source**: Inter-model disagreement (standard deviation of q50 across 5 members)
- **Captures**: Model ignorance, especially on novel chemistries

### Conformal Calibration (Finite-Sample Valid)
- **Method**: Symmetric delta adjustment to quantile intervals
- **Guarantee**: 90% marginal coverage on calibration set distribution
- **Per-cluster**: Cluster-conditional calibration when cluster size ≥ 50

### Out-of-Distribution Detection
Three independent gates with sigmoid combination:
1. **Composition distance**: Jaccard distance from training compositions
2. **Embedding kNN**: Distance to k-nearest neighbors in learned feature space
3. **Ensemble disagreement**: Normalized inter-model standard deviation

## Known Limitations

### Critical Limitations
1. **LOCO performance is poor**: Spearman ρ ≈ 0 on structurally novel material families. The model cannot reliably rank materials from unseen crystal structure types.
2. **Calibration degrades OOD**: Coverage drops to 72% on LOCO splits (below 90% target). Conformal guarantees only hold for the calibration distribution.
3. **Oxide-centric training**: Limited to transition metal oxide frameworks. Performance on sulfides, phosphates, or mixed-anion systems is unvalidated.

### Operational Limitations
4. **CPU inference latency**: ~2-3 seconds per structure on CPU. Not suitable for real-time screening of >10K candidates without GPU acceleration.
5. **CIF input only**: Requires crystallographic information file. No composition-only fast triage mode.
6. **Single-chemistry scope**: Li-ion cathodes only. Extension to Na-ion, solid-state, etc. requires retraining.

### Bias & Fairness Considerations
- **Materials Project bias**: Training data inherits the compositional and structural biases of the Materials Project database (over-representation of common crystal structures, under-representation of metastable phases)
- **Stability bias**: The 0.05 eV/atom stability threshold is a convention, not a physical law. Some materials marginally above this threshold may be synthesizable.

## Ethical Considerations

- **Not a substitute for experimental validation**: Screening decisions should always be verified by DFT and, ultimately, by experimental synthesis and characterization.
- **Environmental impact**: Model training required ~200 GPU-hours on RTX 2060. Inference is CPU-only.
- **Dual-use**: While designed for battery materials, the ensemble methodology could be applied to other materials screening tasks.

## Recommendations

### For Users
- Always verify KEEP decisions with DFT before experimental synthesis
- Check OOD flags — high OOD scores indicate the model is uncertain about a novel chemistry
- Use MAYBE decisions as candidates for active learning, not as rejections
- Monitor drift metrics when deploying on new material libraries

### For Developers
- Retrain when Materials Project database updates significantly
- Expand SOAP-LOCO evaluation when adding new crystal families
- Consider multi-fidelity approaches for LOCO-weak regions

## Model Artifacts

| Artifact | Path | Size |
|---|---|---|
| Ensemble checkpoints (5×) | `artifacts/models/mace_ensemble_v1/member_*.pt` | ~22 MB each |
| Conformal parameters | `artifacts/models/mace_ensemble_v1/conformal_params.json` | ~2 KB |
| Normalizer state | `artifacts/models/mace_ensemble_v1/normalizer.json` | ~1 KB |
| Artifact manifest | `artifacts/models/mace_ensemble_v1/manifest.json` | ~5 KB |
| Data version pin | `data/DATA_VERSION.json` | ~1 KB |

## Citation

If you use this model, please cite:

```bibtex
@software{cathodescreen2025,
  title = {CathodeScreen: High-Throughput ML Screening of Li-Ion Battery Cathodes},
  year = {2025},
  url = {https://github.com/your-org/cathode-screening}
}
```

## References

1. Batatia, I., et al. (2023). MACE-MP-0: A Foundation Model for Materials Science. *arXiv:2401.00096*.
2. Deng, B., et al. (2023). CHGNet: Pretrained universal neural network potential. *Nature Machine Intelligence*.
3. Jain, A., et al. (2013). The Materials Project. *APL Materials*.
4. Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
5. Vovk, V., et al. (2005). Algorithmic Learning in a Random World. *Springer*.
6. Mitchell, M., et al. (2019). Model Cards for Model Reporting. *FAT* Conference*.

---

*This model card follows the framework proposed by Mitchell et al. (2019) for transparent ML model documentation.*
*Last updated: 2025-01-15 | Model version: 1.0.0*
