# CathodeScreen Benchmark Report

Generated: 2026-03-03T07:54:49.445382+00:00

## Model

- **Architecture**: MACE-MP-0 fine-tuned ensemble
- **Ensemble size**: 5 members
- **Target**: energy_above_hull (eV/atom)
- **Training set**: 8,522 structures
- **Validation set**: 1,842 structures
- **Test set**: 1,013 structures

## Performance

| Metric | Value |
|--------|-------|
| Val MAE | 0.0337 +/- 0.0025 eV/atom |
| Test MAE | 0.0307 +/- 0.0018 eV/atom |
| Conformal coverage | 90.1% (target: 90%) |
| Conformal delta | 0.0213 eV |

### Per-Member MAE

| Member | Val MAE | Test MAE |
|--------|---------|----------|
| 0 | 0.0328 | 0.0301 |
| 1 | 0.0345 | 0.0326 |
| 2 | 0.0346 | 0.0307 |
| 3 | 0.0297 | 0.0276 |
| 4 | 0.0372 | 0.0323 |

## Cross-Database Screening

| Dataset | EF@1% | Precision@100 | AUPRC |
|---------|-------|---------------|-------|
| h100_ehull | 7.60x [0.00, 22.17] | 3.9% [1.0%, 8.0%] | 0.082 |
| oqmd | 1.40x [0.57, 2.34] | 68.1% [57.0%, 78.0%] | 0.702 |
| jarvis | 1.77x [1.44, 2.08] | 79.0% [69.0%, 87.0%] | 0.685 |

## DFT Validation Campaign

- **Total candidates**: 50
- **Source**: JARVIS cathode subset (50 candidates)
- **DFT method**: Quantum ESPRESSO (PBE, PAW)
- **Accept rate**: 36%

| Decision | Count |
|----------|-------|
| accept | 18 |
| hold | 31 |
| unknown | 1 |

## Multi-Property Screening

In addition to ML-predicted E_hull, CathodeScreen computes:

- **Theoretical gravimetric capacity** (mAh/g) from Li stoichiometry
- **Average voltage proxy** (V vs Li/Li+) from TM-anion empirical correlations
- **Gravimetric energy density** (Wh/kg) = capacity x voltage
- **Volumetric capacity** (mAh/cm^3) from crystal density
- **Composite screening score** (weighted: 35% stability, 25% capacity, 15% voltage, 25% energy density)

## Pipeline

- 1. Structure ingestion (CIF/pymatgen)
- 2. MACE-MP-0 backbone feature extraction (frozen)
- 3. 5-member ensemble quantile regression (q10, q50, q90)
- 4. Conformal calibration (90% coverage guarantee)
- 5. OOD detection (composition, embedding, disagreement gates)
- 6. Decision policy (KEEP/MAYBE/KILL)
- 7. Multi-property analysis (voltage, capacity, energy density)
- 8. Composite screening score (stability + capacity + voltage + energy)

## Infrastructure

- **training**: RTX 2060 (local, ~45 min per member)
- **inference**: Render (CPU, ~2s per structure)
- **frontend**: Vercel (Next.js)
- **api**: FastAPI + MACE-MP-0 ensemble

---

## Key Claims

- **ensemble_val_mae**: 0.0337 +/- 0.0025 eV/atom
- **ensemble_test_mae**: 0.0307 +/- 0.0018 eV/atom
- **calibration_coverage**: 90.1%
- **training_data_size**: 8,522
- **dft_accept_rate**: 36%
- **jarvis_precision_at_100**: 79.0% [69.0%, 87.0%]
- **jarvis_enrichment_1pct**: 1.77x [1.44, 2.08]