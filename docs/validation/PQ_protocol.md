# Performance Qualification (PQ) Protocol

**Document ID:** VV-PQ-001
**Version:** 1.0
**Effective Date:** 2026-03-16
**System:** CathodeScreen v1.1.0
**Classification:** Quality Assurance

---

## 1. Purpose

This Performance Qualification protocol verifies that the CathodeScreen system meets its specified performance requirements under realistic operating conditions, including ML model accuracy, latency, throughput, and reliability.

## 2. Scope

Covers performance requirements for:
- ML model accuracy and calibration (governance gates)
- Inference latency and throughput
- System reliability under load
- Data integrity and reproducibility

## 3. Performance Requirements

### 3.1 ML Model Performance (Governance Gates)

| ID | Requirement | Threshold | Measurement Method |
|---|---|---|---|
| PQ-ML-001 | Ranking correlation (val) | Spearman ρ > 0.5 | `scripts/09_evaluate_predictions.py` on val split |
| PQ-ML-002 | Ranking correlation (test) | Spearman ρ > 0.5 | `scripts/09_evaluate_predictions.py` on test split |
| PQ-ML-003 | Prediction interval coverage | ≥ 90% at α=0.10 | Conformal coverage on test set |
| PQ-ML-004 | False-kill rate | < 2% | Fraction of stable materials classified KILL |
| PQ-ML-005 | KEEP precision | > 85% | Precision of KEEP decisions on test set |
| PQ-ML-006 | Decision-making capability | System produces decisions | KEEP + KILL > 0 on test set |
| PQ-ML-007 | Test MAE | < 0.050 eV/atom | Mean absolute error on test split |

### 3.2 Inference Performance

| ID | Requirement | Threshold | Measurement Method |
|---|---|---|---|
| PQ-INF-001 | Single prediction latency (CPU) | < 10 seconds (p95) | `scripts/11_load_test_api.py` |
| PQ-INF-002 | Batch prediction throughput | ≥ 5 structures/minute (CPU) | Batch of 25 structures |
| PQ-INF-003 | API startup time | < 60 seconds | Time from `uvicorn` start to `/ready` returning 200 |
| PQ-INF-004 | Memory usage (steady state) | < 4 GB | Monitor after 100 predictions |

### 3.3 Reliability

| ID | Requirement | Threshold | Measurement Method |
|---|---|---|---|
| PQ-REL-001 | Error rate | < 1% on valid inputs | 100 predictions with known-valid CIFs |
| PQ-REL-002 | Graceful degradation | Informative error on invalid input | Test with invalid CIFs, check error codes |
| PQ-REL-003 | Concurrent handling | No crash under 5 concurrent requests | `scripts/11_load_test_api.py` with concurrency=5 |
| PQ-REL-004 | Audit trail integrity | 100% of predictions logged | Compare prediction count vs audit record count |

### 3.4 Data Integrity

| ID | Requirement | Threshold | Measurement Method |
|---|---|---|---|
| PQ-DAT-001 | Reproducibility | Same CIF → same prediction | Submit identical CIF 3 times, verify identical output |
| PQ-DAT-002 | Input hash uniqueness | Different CIFs → different hashes | Verify SHA-256 hashes |
| PQ-DAT-003 | Split isolation | No train/test leakage | `test_split_leakage.py` |
| PQ-DAT-004 | Artifact integrity | Manifest signature valid | `scripts/12_validate_release.py` |

## 4. PQ Test Execution

### 4.1 ML Governance Gate (Automated)

```bash
# Run the full governance evaluation
python scripts/09_evaluate_predictions.py
python scripts/12_validate_release.py

# Expected: All 6 governance checks PASS
```

**Record results:**

| Gate | Threshold | Measured Value | Pass/Fail | Date |
|---|---|---|---|---|
| PQ-ML-001: Ranking (val) | ρ > 0.5 | | | |
| PQ-ML-002: Ranking (test) | ρ > 0.5 | | | |
| PQ-ML-003: Coverage | ≥ 90% | | | |
| PQ-ML-004: False-kill | < 2% | | | |
| PQ-ML-005: KEEP precision | > 85% | | | |
| PQ-ML-006: Decisions | > 0 | | | |
| PQ-ML-007: Test MAE | < 0.050 eV | | | |

### 4.2 Inference Load Test

```bash
# Run load test (requires running API server)
python scripts/11_load_test_api.py --url http://localhost:8000 --n 100 --concurrency 5
```

**Record results:**

| Metric | Threshold | Measured Value | Pass/Fail | Date |
|---|---|---|---|---|
| PQ-INF-001: p95 latency | < 10s | | | |
| PQ-INF-002: Throughput | ≥ 5/min | | | |
| PQ-INF-003: Startup time | < 60s | | | |
| PQ-INF-004: Memory | < 4 GB | | | |

### 4.3 Reliability Test

```bash
# Run integration tests
pytest tests/integration/ -v --tb=short

# Run unit tests
pytest src/cathode_screening/tests/ -v --tb=short
```

**Record results:**

| Metric | Threshold | Measured Value | Pass/Fail | Date |
|---|---|---|---|---|
| PQ-REL-001: Error rate | < 1% | | | |
| PQ-REL-002: Graceful errors | All coded | | | |
| PQ-REL-003: Concurrency | No crash | | | |
| PQ-REL-004: Audit completeness | 100% | | | |

### 4.4 Reproducibility Test

```bash
# Submit same CIF 3 times, compare results
for i in 1 2 3; do
  curl -s -X POST http://localhost:8000/predict \
    -F "cif_file=@tests/integration/fixtures/LiMn2O4_spinel.cif" \
    | jq .prediction.pred_ehull
done
# All three values must be identical
```

## 5. Acceptance Criteria

- **All governance gates** (PQ-ML-001 through PQ-ML-006): Must PASS
- **Inference performance**: p95 latency and throughput within thresholds
- **Reliability**: Error rate < 1%, no crashes under load
- **Data integrity**: Reproducible predictions, valid artifact signatures

## 6. Deviation Handling

Any PQ failure requires:
1. **Deviation Report**: Document the failure with measured values
2. **Impact Assessment**: Evaluate risk to screening accuracy
3. **Corrective Action**: Fix, retrain, or adjust thresholds
4. **Retest**: Re-execute failed PQ checks after corrective action
5. **Approval**: QA and ML leads must sign off on deviations

## 7. Requalification Triggers

PQ must be re-executed when:
- Model is retrained or fine-tuned
- Training data source is updated (Materials Project version change)
- Decision policy thresholds are modified
- Conformal calibration parameters are recomputed
- Infrastructure changes (new hardware, different cloud provider)
- Major dependency updates (PyTorch, pymatgen, MACE)

## 8. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| QA Lead | | | |
| ML Engineer | | | |
| Materials Scientist | | | |
| Project Manager | | | |
