# Requirements Traceability Matrix

**Document ID:** VV-TM-001
**Version:** 1.0
**Effective Date:** 2026-03-16
**System:** CathodeScreen v1.1.0

---

## 1. Purpose

This traceability matrix links system requirements to their verification tests and qualification evidence, ensuring complete test coverage for regulatory compliance.

## 2. Legend

| Abbreviation | Meaning |
|---|---|
| REQ | Requirement |
| IQ | Installation Qualification |
| OQ | Operational Qualification |
| PQ | Performance Qualification |
| UT | Unit Test |
| IT | Integration Test |

---

## 3. Functional Requirements

| REQ ID | Requirement | Test ID(s) | Test Type | Evidence | Status |
|---|---|---|---|---|---|
| REQ-F-001 | System shall predict E_hull from CIF input | OQ-001, IT-test_e2e_prediction | OQ, IT | API response with pred_ehull | |
| REQ-F-002 | System shall classify materials as KEEP/MAYBE/KILL | OQ-004, UT-test_decision_policy | OQ, UT | Decision field in response | |
| REQ-F-003 | System shall provide calibrated confidence intervals | OQ-003, UT-test_conformal_calibration | OQ, UT | CI in response, coverage ≥ 90% | |
| REQ-F-004 | System shall detect out-of-distribution inputs | UT-test_ood_gating | UT | OOD flag and score in response | |
| REQ-F-005 | System shall process batch predictions | OQ-002, IT-test_batch_prediction | OQ, IT | Batch response with n_processed | |
| REQ-F-006 | System shall validate cathode composition | OQ-005, OQ-006, UT-test_cathode_input_guardrails | OQ, UT | Rejection of non-cathode inputs | |
| REQ-F-007 | System shall compute cathode properties | IT-test_prediction_includes_cathode_properties | IT | Properties in prediction response | |
| REQ-F-008 | System shall manage discovery campaigns | OQ-010 | OQ | Campaign create/screen/select/submit flow | |
| REQ-F-009 | System shall rank candidates by acquisition score | UT-test_discovery_ranking | UT | Ranked candidates with scores | |
| REQ-F-010 | System shall provide model metadata | OQ-001 (step 7), IT-test_api_contract | OQ, IT | /model/info response | |

## 4. ML Performance Requirements

| REQ ID | Requirement | Test ID(s) | Test Type | Evidence | Status |
|---|---|---|---|---|---|
| REQ-ML-001 | Spearman ρ > 0.5 on val set | PQ-ML-001 | PQ | model_validation_report.json | |
| REQ-ML-002 | Spearman ρ > 0.5 on test set | PQ-ML-002 | PQ | model_validation_report.json | |
| REQ-ML-003 | 90% conformal coverage on test set | PQ-ML-003, UT-test_conformal_calibration | PQ, UT | Coverage metric | |
| REQ-ML-004 | False-kill rate < 2% | PQ-ML-004 | PQ | Decision analysis | |
| REQ-ML-005 | KEEP precision > 85% | PQ-ML-005 | PQ | Decision analysis | |
| REQ-ML-006 | Test MAE < 0.050 eV/atom | PQ-ML-007 | PQ | model_validation_report.json | |
| REQ-ML-007 | Ensemble of ≥ 5 members | IQ-003, UT-test_mace_checkpoint_resolution | IQ, UT | 5 checkpoint files | |

## 5. Security Requirements

| REQ ID | Requirement | Test ID(s) | Test Type | Evidence | Status |
|---|---|---|---|---|---|
| REQ-S-001 | API key authentication in production | OQ-008, UT-test_deployment_security_defaults | OQ, UT | 403 on invalid key | |
| REQ-S-002 | Rate limiting per client | OQ-007 | OQ | 429 after limit exceeded | |
| REQ-S-003 | HTTPS enforcement in production | UT-test_deployment_security_defaults | UT | Redirect middleware active | |
| REQ-S-004 | Safe torch checkpoint loading | UT-test_deployment_security_defaults | UT | ALLOW_UNSAFE_TORCH_LOAD=false | |
| REQ-S-005 | Artifact manifest signature verification | IQ-003, UT-test_mace_artifact_metadata_paths | IQ, UT | HMAC verification | |
| REQ-S-006 | Input size limits | OQ-005, IT-test_input_validation | OQ, IT | 413 on oversize | |
| REQ-S-007 | Security headers in production | UT-test_deployment_security_defaults | UT | X-Content-Type-Options, etc. | |

## 6. Data Integrity Requirements

| REQ ID | Requirement | Test ID(s) | Test Type | Evidence | Status |
|---|---|---|---|---|---|
| REQ-D-001 | No train/test data leakage | UT-test_split_leakage | UT | Split intersection = ∅ | |
| REQ-D-002 | Prediction audit trail | OQ-009, IT-test_audit_trail | OQ, IT | JSONL/PostgreSQL records | |
| REQ-D-003 | Reproducible predictions | PQ-DAT-001 | PQ | Same CIF → same output | |
| REQ-D-004 | Input hashing for provenance | PQ-DAT-002 | PQ | SHA-256 in audit record | |
| REQ-D-005 | Data version pinning | IQ (data/DATA_VERSION.json) | IQ | Pinned MP query params | |
| REQ-D-006 | DVC pipeline reproducibility | IQ (dvc.yaml) | IQ | `dvc repro` succeeds | |

## 7. Operational Requirements

| REQ ID | Requirement | Test ID(s) | Test Type | Evidence | Status |
|---|---|---|---|---|---|
| REQ-O-001 | Health check endpoint | IQ-004, IT-test_health_endpoints | IQ, IT | / returns ok | |
| REQ-O-002 | Readiness probe | IQ-004 | IQ | /ready returns ready | |
| REQ-O-003 | Metrics collection | OQ-011, IT-test_metrics | OQ, IT | /metrics returns data | |
| REQ-O-004 | API versioning | OQ-012, IT-test_api_versioning | OQ, IT | /v1/ prefix works | |
| REQ-O-005 | p95 inference latency < 10s | PQ-INF-001 | PQ | Load test results | |
| REQ-O-006 | Startup time < 60s | PQ-INF-003 | PQ | Startup timing | |
| REQ-O-007 | Graceful error handling | OQ-005, IT-test_error_response_format | OQ, IT | Structured error JSON | |

## 8. Test Coverage Summary

| Category | Total Requirements | Tests Mapped | Coverage |
|---|---|---|---|
| Functional (REQ-F) | 10 | 10 | 100% |
| ML Performance (REQ-ML) | 7 | 7 | 100% |
| Security (REQ-S) | 7 | 7 | 100% |
| Data Integrity (REQ-D) | 6 | 6 | 100% |
| Operational (REQ-O) | 7 | 7 | 100% |
| **Total** | **37** | **37** | **100%** |

## 9. Unit Test Mapping

| Unit Test File | Requirements Covered |
|---|---|
| `test_decision_policy.py` | REQ-F-002 |
| `test_conformal_calibration.py` | REQ-F-003, REQ-ML-003 |
| `test_ood_gating.py` | REQ-F-004 |
| `test_cathode_input_guardrails.py` | REQ-F-006 |
| `test_discovery_ranking.py` | REQ-F-009 |
| `test_discovery_state.py` | REQ-F-008 |
| `test_split_leakage.py` | REQ-D-001 |
| `test_deployment_security_defaults.py` | REQ-S-001, REQ-S-003, REQ-S-004, REQ-S-007 |
| `test_mace_checkpoint_resolution.py` | REQ-ML-007 |
| `test_mace_artifact_metadata_paths.py` | REQ-S-005 |
| `test_inference_contract.py` | REQ-F-001 |
| `test_repo_hygiene.py` | Code quality |
| `test_symbol_redefinition_guard.py` | Code quality |
| `test_graph_builder.py` | REQ-F-001 (data pipeline) |
| `test_filters.py` | REQ-F-006 |
| `test_retrain_contract.py` | REQ-F-008 |
| `test_decision_grade.py` | REQ-F-002, REQ-F-003 |

## 10. Integration Test Mapping

| Integration Test File | Requirements Covered |
|---|---|
| `test_e2e_prediction.py::TestHealthEndpoints` | REQ-O-001, REQ-O-002 |
| `test_e2e_prediction.py::TestInputValidation` | REQ-F-006, REQ-S-006 |
| `test_e2e_prediction.py::TestPredictionE2E` | REQ-F-001, REQ-F-003, REQ-F-007 |
| `test_e2e_prediction.py::TestBatchPredictionE2E` | REQ-F-005 |
| `test_e2e_prediction.py::TestAuditTrailE2E` | REQ-D-002 |
| `test_e2e_prediction.py::TestMetricsE2E` | REQ-O-003 |
| `test_api_contract.py::TestErrorResponseFormat` | REQ-O-007 |
| `test_api_contract.py::TestResponseSchemaCompliance` | REQ-F-010, REQ-O-003 |
| `test_api_contract.py::TestAPIVersioning` | REQ-O-004 |
| `test_api_contract.py::TestRateLimiting` | REQ-S-002 |

## 11. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| QA Lead | | | |
| ML Engineer | | | |
| Project Manager | | | |
