# Operational Qualification (OQ) Protocol

**Document ID:** VV-OQ-001
**Version:** 1.0
**Effective Date:** 2026-03-16
**System:** CathodeScreen v1.1.0
**Classification:** Quality Assurance

---

## 1. Purpose

This Operational Qualification protocol verifies that the CathodeScreen system operates correctly within specified operational parameters across all functional modules.

## 2. Scope

Covers all functional capabilities:
- Prediction pipeline (single and batch)
- Uncertainty quantification
- Decision policy
- Input validation and guardrails
- Discovery campaign engine
- Audit trail and observability
- Security controls

## 3. OQ Test Cases

### OQ-001: Single Prediction Pipeline
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Upload LiCoO2.cif to `/predict` | Status 200, valid prediction | |
| 2 | Verify `pred_ehull` is numeric ≥ 0 | True | |
| 3 | Verify `p_stable` is in [0, 1] | True | |
| 4 | Verify `action` is one of {DFT, HOLD, SKIP} | True | |
| 5 | Verify `confidence_interval` has lower ≤ upper | True | |
| 6 | Verify `uncertainty` is one of {Low, Medium, High} | True | |
| 7 | Verify `X-Request-ID` header in response | Present | |

### OQ-002: Batch Prediction
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Upload 3 valid CIF files to `/predict/batch` | Status 200 | |
| 2 | Verify `n_processed == 3` | True | |
| 3 | Upload mix of 2 valid + 1 invalid | `n_processed >= 1`, `n_errors >= 1` | |
| 4 | Upload >25 files (exceed `MAX_BATCH_SIZE`) | Status 413 | |

### OQ-003: Uncertainty Quantification
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Predict LiMn2O4 (in-distribution) | Uncertainty should be Low or Medium | |
| 2 | Verify confidence interval width is reasonable | CI width < 0.5 eV | |
| 3 | Verify epistemic uncertainty is non-negative | ≥ 0 | |

### OQ-004: Decision Policy
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Verify KEEP criteria: q90_cal < threshold AND p_stable > threshold | Decision = KEEP → action = DFT | |
| 2 | Verify KILL criteria: q10_cal > threshold | Decision = KILL → action = SKIP | |
| 3 | Verify MAYBE: intermediate cases | Decision = MAYBE → action = HOLD | |

### OQ-005: Input Validation
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Upload empty file | Status 400, error = "file_too_small" | |
| 2 | Upload 3MB file | Status 413, error = "file_too_large" | |
| 3 | Upload non-CIF text | Status 400, error = "parse_error" | |
| 4 | Upload NaCl.cif (non-cathode) | Status 400, error = "invalid_cathode_composition" | |
| 5 | Upload structure with >512 atoms | Status 413, error = "structure_too_large" | |

### OQ-006: Cathode Composition Guardrails
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Submit Li-cathode (LiCoO2) | Accepted | |
| 2 | Submit non-Li (NaCl) | Rejected: "no lithium" | |
| 3 | Submit Li-only (Li metal) | Rejected: "no transition metal" | |

### OQ-007: Rate Limiting
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Send single request | Status 200 | |
| 2 | Send >60 requests in <60s | Status 429 after limit exceeded | |
| 3 | Wait 60s, retry | Status 200 | |

### OQ-008: Authentication
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Request with valid API key | Status 200 | |
| 2 | Request with invalid API key | Status 403 | |
| 3 | Request with no API key (auth enabled) | Status 403 | |
| 4 | Request with Bearer token format | Status 200 (if valid) | |

### OQ-009: Audit Trail
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Make prediction | Audit record created | |
| 2 | `GET /audit/recent` | Returns ≥1 record | |
| 3 | Verify record has: timestamp, request_id, input_hash | All present | |
| 4 | `GET /audit/stats` | Returns decision counts | |

### OQ-010: Discovery Campaign
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | Create campaign | Status 200, campaign created | |
| 2 | Screen candidates | Returns ranked candidates | |
| 3 | Select batch | Returns selected IDs | |
| 4 | Submit DFT (mock) | Job created, runs async | |
| 5 | Check job status | Status = completed (after wait) | |

### OQ-011: Metrics & Observability
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | `GET /metrics` | JSON with request_count, decision_counts | |
| 2 | `GET /metrics/prometheus` (if enabled) | Prometheus text format | |
| 3 | Verify `X-Request-ID` header on all responses | Present | |

### OQ-012: API Versioning
| Step | Action | Expected Result | Pass/Fail |
|---|---|---|---|
| 1 | `GET /v1/` | Same response as `GET /` | |
| 2 | `GET /v1/model/info` | Same response as `GET /model/info` | |
| 3 | `POST /v1/predict` | Same response as `POST /predict` | |

## 4. Acceptance Criteria

- All OQ test cases must PASS
- Any FAIL must be documented with:
  - Root cause analysis
  - Corrective action
  - Retest results
- OQ must be re-executed after any software update

## 5. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| QA Lead | | | |
| ML Engineer | | | |
| Project Manager | | | |
