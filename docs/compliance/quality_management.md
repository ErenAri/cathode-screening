# CathodeScreen Quality Management System

**Applicable Standards**: ISO 9001:2015, IATF 16949:2016
**Document ID**: CS-QMS-001
**Revision**: 1.0
**Effective Date**: 2026-03-16
**Scope**: AI-powered screening of lithium-ion battery cathode materials

---

## 1. Quality Policy

CathodeScreen is committed to delivering reliable, reproducible, and
scientifically validated AI predictions for cathode material screening.
Our quality management system ensures:

- **Accuracy**: Model predictions meet published governance thresholds
  (MAE ≤ 0.05 eV/atom, KEEP precision ≥ 90%, false-kill rate < 1%).
- **Traceability**: Every prediction is auditable from input CIF through
  model inference to final decision output.
- **Reproducibility**: DVC-managed data pipelines produce bit-identical
  results across environments.
- **Continuous improvement**: Drift monitoring, shadow deployments, and
  active learning loops systematically improve model performance.

## 2. Context of the Organization (ISO 9001 §4)

### 2.1 Scope

CathodeScreen provides AI-based stability screening for lithium-ion
battery cathode materials, serving:

- **Battery manufacturers** screening candidate cathode compositions
- **Research laboratories** accelerating materials discovery
- **Automotive OEMs** validating cathode supply chain quality

### 2.2 Interested Parties

| Party | Needs & Expectations |
|-------|---------------------|
| Battery manufacturers | Reliable KEEP/KILL decisions, low false-kill rate |
| Research labs | Accurate E_hull predictions, uncertainty quantification |
| Automotive OEMs | IATF 16949 compliance evidence, audit trails |
| Regulatory bodies | Data integrity, model validation evidence |
| End users (EV drivers) | Safe, high-performance battery cells |

### 2.3 Quality Management System Processes

```
Customer Need → CIF Upload → Input Validation → Model Inference
→ OOD Detection → Decision Policy → Audit Log → Customer Response
```

## 3. Leadership (ISO 9001 §5)

### 3.1 Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| Model Owner | Training, validation, governance gate approvals |
| API Administrator | Deployment, monitoring, RBAC configuration |
| Data Steward | DVC pipeline, data version control, drift detection |
| Quality Manager | QMS audits, CAPA management, compliance reviews |

### 3.2 RBAC Alignment

CathodeScreen's Role-Based Access Control maps to quality roles:

- **viewer**: Quality auditors, read-only access to metrics and audit trails
- **operator**: Lab technicians, can submit predictions and run campaigns
- **admin**: Model owners, can promote models and manage registry

## 4. Planning (ISO 9001 §6)

### 4.1 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| False KEEP (unsafe material accepted) | Low | High | Conformal calibration at 90% coverage, conservative thresholds |
| False KILL (good material rejected) | Very Low | Medium | Zero false-kill rate on validation set, mandatory MAYBE review |
| Model drift (distribution shift) | Medium | Medium | Hourly PSI drift monitoring, automated alerts |
| OOD input (unknown chemistry) | Medium | Low | 3-gate OOD detection, automatic MAYBE classification |
| Data integrity breach | Low | High | HMAC-signed manifest, PostgreSQL audit trail |

### 4.2 Quality Objectives

| Objective | Target | Measurement | Frequency |
|-----------|--------|-------------|-----------|
| Prediction accuracy | MAE ≤ 0.05 eV/atom | SOAP-LOCO cross-validation | Per model release |
| KEEP precision | ≥ 90% | Validation set evaluation | Per model release |
| False-kill rate | < 1% | Validation set evaluation | Per model release |
| Calibration coverage | 90% ± 2% | Conformal coverage test | Per model release |
| API availability | ≥ 99.5% | Kubernetes health probes | Continuous |
| Prediction latency (p95) | < 10s | Prometheus metrics | Continuous |
| Drift detection | PSI < 0.2 | Automated hourly check | Hourly |
| Audit trail completeness | 100% | Audit log vs request count | Weekly |

## 5. Support (ISO 9001 §7)

### 5.1 Infrastructure

- **Compute**: Kubernetes cluster with HPA auto-scaling (2-10 API pods)
- **Storage**: PostgreSQL for audit, Redis for task queue, PVC for artifacts
- **Monitoring**: Prometheus + OpenTelemetry, Grafana dashboards
- **Security**: TLS termination, RBAC, rate limiting, IP allowlisting

### 5.2 Competence

Personnel operating CathodeScreen must be trained in:
- Materials science fundamentals (crystal structure, stability concepts)
- API operation and RBAC role assignments
- Interpretation of prediction results and uncertainty quantification
- Incident response procedures for model drift alerts

### 5.3 Documented Information

| Document | ID | Purpose |
|----------|----|---------|
| Model Card | CS-MC-001 | Model architecture, training data, limitations |
| V&V Protocols | CS-VV-{IQ,OQ,PQ} | Installation, operational, performance qualification |
| Traceability Matrix | CS-TM-001 | Requirements → tests → evidence mapping |
| Data Version Record | CS-DV-001 | Materials Project query parameters, dataset stats |
| This document | CS-QMS-001 | Quality management system overview |
| IATF Supplement | CS-IATF-001 | Automotive-specific requirements |

## 6. Operation (ISO 9001 §8)

### 6.1 Operational Planning — Model Lifecycle

```
Data Collection (MP API) → Feature Engineering → Training (MACE ensemble)
→ Calibration (conformal) → Evaluation (SOAP-LOCO) → Governance Gates
→ Registry (staging → production) → Deployment → Monitoring
```

### 6.2 Control of Production (Inference)

| Control Point | Mechanism |
|---------------|-----------|
| Input validation | CIF parsing, atom count limits, composition checks |
| Cathode guardrails | Lithium presence check, transition metal validation |
| OOD detection | Composition gate, embedding gate, disagreement gate |
| Decision policy | Configurable thresholds, conservative defaults |
| Audit logging | Every prediction logged with full input/output trace |

### 6.3 Release of Products (Model Promotion)

Models must pass all 6 governance gates before production promotion:

1. **Test MAE** ≤ threshold
2. **Spearman ρ** ≥ threshold
3. **Calibration coverage** within tolerance
4. **KEEP precision** ≥ threshold
5. **False-kill rate** < threshold
6. **Manifest signature** valid (HMAC)

### 6.4 Control of Nonconforming Outputs

When a prediction triggers OOD detection:
1. Decision is downgraded to MAYBE (never auto-KEEP for OOD inputs)
2. OOD score and gate details are logged in audit trail
3. Material is flagged for manual expert review

## 7. Performance Evaluation (ISO 9001 §9)

### 7.1 Monitoring and Measurement

| Metric | Tool | Alert Threshold |
|--------|------|----------------|
| API error rate | Prometheus | > 1% over 5min window |
| Prediction latency p95 | Prometheus | > 10 seconds |
| Feature drift (PSI) | Celery beat task | PSI > 0.2 |
| Decision distribution shift | Drift alerter | > 20% change in KEEP rate |
| Model disagreement rate | Shadow deployment | > 15% decision disagreements |

### 7.2 Internal Audit

- **Quarterly**: QMS process audit against ISO 9001 clauses
- **Per release**: Model governance gate review
- **Weekly**: Audit trail completeness check
- **On alert**: Root cause analysis for drift/disagreement alerts

### 7.3 Management Review

Inputs:
- Model performance metrics (MAE, precision, coverage)
- Drift alert history and resolution
- Shadow deployment comparison results
- Customer feedback and prediction accuracy reports
- CAPA status

## 8. Improvement (ISO 9001 §10)

### 8.1 CAPA Process (Corrective and Preventive Action)

```
Nonconformity Detected → Root Cause Analysis → Corrective Action Plan
→ Implementation → Effectiveness Verification → Closure
```

Triggers for CAPA:
- Governance gate failure during model release
- Sustained drift alert (PSI > 0.2 for > 24 hours)
- Customer-reported incorrect prediction
- Shadow deployment unsafe disagreement

### 8.2 Continual Improvement

- **Active learning**: Priority DFT validation of MAYBE predictions
- **Model retraining**: Triggered by drift alerts or new training data
- **Shadow deployment**: Candidate models validated against production
  before promotion
- **Load testing**: Regular capacity verification (target: 1000 pred/min)
