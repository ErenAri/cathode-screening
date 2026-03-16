# IATF 16949:2016 Supplement for CathodeScreen

**Document ID**: CS-IATF-001
**Revision**: 1.0
**Effective Date**: 2026-03-16
**Scope**: Automotive-specific quality requirements for AI-based cathode screening

---

## 1. Purpose

This supplement extends the CathodeScreen Quality Management System
(CS-QMS-001) to address automotive-specific requirements from
IATF 16949:2016, relevant to battery cathode material suppliers serving
the automotive industry.

## 2. Applicability

CathodeScreen serves as a **screening tool** in the cathode material
qualification process. It does not replace physical testing or DFT
calculations but provides a rapid, AI-powered pre-filter to prioritize
candidates for expensive validation.

**Position in the automotive V-model:**
```
OEM cathode specification
  └→ CathodeScreen AI screening (this system)
       └→ DFT validation of KEEP candidates
            └→ Lab synthesis & electrochemical testing
                 └→ Cell-level validation
                      └→ Pack-level qualification (IATF scope)
```

## 3. IATF 16949 Clause Mapping

### 4.4.1.2 — Product Safety

| Requirement | CathodeScreen Implementation |
|-------------|------------------------------|
| Identify safety-related products | Cathode materials in automotive cells are safety-critical |
| Process controls for safety | Conservative decision policy: false-kill rate < 1%, OOD → MAYBE |
| Escalation procedures | Drift alerts via PagerDuty, MAYBE materials require expert review |
| Traceability | Full audit trail: CIF input → hash → prediction → decision → timestamp |

### 5.1.1.1 — Corporate Responsibility

CathodeScreen's Model Card (CS-MC-001) documents:
- Known limitations and failure modes
- Ethical considerations for AI in materials screening
- Bias assessment across crystal structure types

### 7.1.5.1 — Measurement System Analysis (MSA)

The CathodeScreen prediction system is treated as a measurement system:

| MSA Component | Implementation |
|---------------|---------------|
| **Repeatability** | Same CIF input produces identical prediction (deterministic ensemble) |
| **Reproducibility** | DVC pipeline ensures identical model across environments |
| **Bias** | MAE measured against DFT ground truth: 0.030 eV/atom |
| **Linearity** | Spearman ρ = 0.663 across full E_hull range |
| **Stability** | Hourly drift monitoring via PSI metric |
| **Resolution** | Conformal prediction interval width ~ 0.1 eV/atom |

**Gauge R&R equivalent**: The 5-member ensemble disagreement quantifies
measurement uncertainty. Epistemic uncertainty < 0.02 eV/atom indicates
high measurement confidence.

### 7.2.1 — Competence — Supplemental

Training requirements for CathodeScreen operators:

| Training Module | Required For | Evidence |
|----------------|--------------|----------|
| CathodeScreen API operation | All operators | Completion of OQ protocol |
| CIF format and crystal structure basics | Operators submitting structures | Internal certification |
| Prediction interpretation & UQ | All operators | Documented training record |
| Model governance & release | Model owners (admin role) | Governance checklist sign-off |
| Incident response (drift alerts) | On-call engineers | Runbook review and drill |

### 7.5.1.1 — Control of Documented Information

| Document Type | Retention Period | Storage | Access Control |
|---------------|-----------------|---------|---------------|
| Audit trail (predictions) | 15 years | PostgreSQL + backup | RBAC (viewer+) |
| Model artifacts | Life of product + 1 year | DVC + artifact manifest | RBAC (admin) |
| Training data version | Life of product + 1 year | DVC + DATA_VERSION.json | RBAC (admin) |
| Governance gate results | 15 years | Model registry | RBAC (viewer+) |
| CAPA records | 15 years | External QMS | Quality manager |

### 8.3.3.1 — Special Characteristics

CathodeScreen predictions include the following special characteristics:

| Characteristic | Classification | Control Method |
|---------------|---------------|---------------|
| E_hull prediction | Critical | Conformal calibration, 90% coverage |
| KEEP decision | Critical | Governance-gated precision threshold ≥ 90% |
| OOD flag | Significant | 3-gate detection system |
| KILL decision | Critical | Zero false-kill validation |

### 8.5.1.1 — Control Plan

| Process Step | Control | Specification | Reaction Plan |
|-------------|---------|--------------|---------------|
| CIF upload | Input validation | Valid CIF, ≤ 512 atoms, cathode composition | Reject with error code |
| Model inference | Ensemble prediction | 5-member MACE ensemble | Fall back to single model + flag |
| OOD check | 3-gate detection | Composition + embedding + disagreement | Auto-classify as MAYBE |
| Decision | Policy engine | Configurable thresholds | Default to conservative policy |
| Calibration | Conformal intervals | 90% coverage target | Alert if coverage < 85% |
| Output | Audit logging | 100% predictions logged | Alert on logging failure |

### 8.5.6.1 — Change Management

Model changes follow a controlled promotion pipeline:

```
Development → Staging (governance gates) → Shadow deployment (N% traffic)
→ Production promotion (admin approval) → Monitoring
```

**Change triggers requiring re-qualification:**
- Training data update (new Materials Project release)
- Model architecture change
- Decision policy threshold change
- Calibration methodology change

**Evidence required for promotion:**
- All 6 governance gates pass
- Shadow deployment shows < 15% decision disagreement
- No unsafe disagreements (shadow KILL ≠ production KEEP)

### 8.7.1.1 — Customer Authorization for Concession

If CathodeScreen is operating with known limitations:
- Drift alert active but below critical threshold
- Reduced ensemble (< 5 models) due to infrastructure issues
- Calibration coverage slightly outside target range

The system must:
1. Log the degraded state in the audit trail
2. Append a warning flag to all predictions during this period
3. Notify the quality manager for customer communication

### 9.1.1.1 — Process Performance Monitoring

| KPI | Target | Monitoring | Escalation |
|-----|--------|-----------|------------|
| KEEP precision (rolling 30d) | ≥ 90% | Automated DFT feedback loop | Retrain if < 85% |
| False-kill rate (rolling 30d) | < 1% | Automated DFT feedback loop | Immediate investigation |
| API availability | ≥ 99.5% | Kubernetes health probes | PagerDuty alert |
| Prediction p95 latency | < 10s | Prometheus | Scale up workers |
| Drift PSI | < 0.2 | Hourly Celery beat | Alert → investigate → retrain |

### 10.2.3 — Problem Solving

CathodeScreen uses the following 8D-aligned process for quality issues:

| 8D Step | Implementation |
|---------|---------------|
| D1: Team | Model owner + data steward + API admin |
| D2: Problem description | Audit trail evidence, drift metrics |
| D3: Containment | Shadow deployment freeze, conservative policy override |
| D4: Root cause | SOAP-LOCO analysis on failing cluster |
| D5: Corrective action | Retrain with expanded data, adjust thresholds |
| D6: Verify | Governance gates + shadow deployment validation |
| D7: Prevent recurrence | Add to drift monitoring features, update OOD gates |
| D8: Closure | Registry promotion, CAPA closure in QMS |

## 4. PPAP Evidence Package

For customer Part Production Approval Process (PPAP) submissions,
CathodeScreen can provide:

| PPAP Element | CathodeScreen Evidence |
|-------------|----------------------|
| Design records | Model Card (CS-MC-001) |
| Process flow diagram | DVC pipeline (dvc.yaml) |
| PFMEA | Risk assessment (QMS §4.1) |
| Control plan | Control plan (§8.5.1.1 above) |
| MSA | Gauge R&R equivalent (§7.1.5.1 above) |
| Process capability | Governance gate results |
| Qualified lab | SOAP-LOCO validation methodology |
| Dimensional results | Model performance metrics per release |
| Material/performance test results | IQ/OQ/PQ protocol results |
| Appearance approval | N/A (software) |
| Sample product | API demo with known-answer fixtures |

## 5. Regulatory Compliance Matrix

| Regulation | Relevance | CathodeScreen Coverage |
|-----------|-----------|----------------------|
| ISO 9001:2015 | Base QMS | CS-QMS-001 |
| IATF 16949:2016 | Automotive supplement | This document |
| EU AI Act (2024) | AI system classification | Limited risk (decision-support tool) |
| IEC 62660 | Li-ion cell testing | Screening input to physical testing |
| UN 38.3 | Transport safety | Material pre-screening upstream of testing |
| REACH | Chemical safety | Composition validation in cathode guardrails |
