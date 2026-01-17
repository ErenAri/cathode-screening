# CathodeScreen: High-Throughput Screening of Li-Ion Battery Cathodes

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=flat&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.js)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)

**CathodeScreen** is an enterprise-grade machine learning framework designed to accelerate the discovery of thermodynamically stable lithium-ion battery cathode materials. It implements a scalable inference pipeline utilizing a deep ensemble of **CHGNet** (Crystal Hamiltonian Graph Neural Network) models to robustly predict energy above hull ($E_{hull}$) with quantified epistemic uncertainty.


## Table of Contents
1.  [Abstract](#-abstract)
2.  [Problem Statement](#-problem-statement)
3.  [System Architecture](#-system-architecture)
4.  [Machine Learning Pipeline](#-machine-learning-pipeline)
    *   [Dataset & Splitting](#dataset--splitting)
    *   [Model Architecture](#model-architecture-cgcnn)
    *   [Uncertainty Quantification](#uncertainty-quantification)
5.  [Performance Metrics](#-performance-metrics)
6.  [Installation & Deployment](#-installation--deployment)
7.  [Citation](#-citation)

---

## Abstract

The discovery of novel cathode materials is constrained by the computationally expensive nature of Density Functional Theory (DFT) calculations, which scale as $O(N^3)$. **CathodeScreen** implements a data-driven screening funnel that serves as a pre-filter for DFT. By leveraging a CHGNet ensemble trained on Li-cathode materials, the system identifies thermodynamically stable candidates ($E_{hull} < 0.05$ eV/atom) with a **6.6× enrichment** over random sampling — while maintaining <0.3% false-discard rate.

## Problem Statement

Traditional high-throughput screening relies on massive DFT compute resources. However:
1.  **Cost**: A single relaxation can take hundreds of CPU hours.
2.  **Efficiency**: Most candidate materials are unstable ($E_{hull} > 0.1$ eV/atom) and are discarded after expensive computation.
3.  **Trust**: Single-point ML predictions fail on Out-of-Distribution (OOD) data (e.g., novel crystal polymorphs).

**Solution**: A deep ensemble that not only predicts stability but estimates its own *competence* (uncertainty) to flag OOD materials for "Active Learning" or manual review.

---

## System Architecture

The application uses a decoupled microservices architecture optimized for cloud deployment. In production, the API can be fronted by an HTTPS load balancer with Cloud Armor (optional, requires a custom domain) and connected to centralized logging/metrics/tracing for SRE.

```mermaid
graph LR
    User[User / Chemist] -->|HTTPS| FE(Next.js Frontend)
    User -->|API Clients| Edge[HTTPS LB + Cloud Armor]
    FE -->|JSON Request| Edge
    Edge -->|API| API(FastAPI Inference)
    subgraph "Inference Engine"
        API -->|Parse| Pymatgen(Structure Parser)
        Pymatgen -->|Graph| GNN1(CHGNet Model 1)
        Pymatgen -->|Graph| GNN2(CHGNet Model 2)
        Pymatgen -->|Graph| GNN3(CHGNet Model 3)
        Pymatgen -->|Graph| GNN4(CHGNet Model 4)
        Pymatgen -->|Graph| GNN5(CHGNet Model 5)
    end
    GNN1 & GNN2 & GNN3 & GNN4 & GNN5 -->|Aggregator| Stats(Mean & Variance)     
    Stats -->|Policy| Result[Action Recommendation]
    subgraph "Observability"
        Logs[Cloud Logging]
        Metrics[Prometheus / Cloud Monitoring]
        Traces[OpenTelemetry]
    end
    API --> Logs
    API --> Metrics
    API --> Traces
```

### Components
*   **Inference Engine (Backend)**: Built with **FastAPI** and **PyTorch**. Validates crystal structures via `pymatgen`, generates graph embeddings, and executes vectorized inference on CPU (< 50ms latency).
*   **User Interface (Frontend)**: built with **Next.js 14** (App Router). Features server-side rendering (SSR) for SEO and a responsive UI tailored for researchers.
*   **Orchestration**: Fully containerized with highly optimized Docker images (multi-stage builds to minimize size).

---

## Machine Learning Pipeline

### Dataset & Splitting
*   **Source**: The Materials Project (2025 Database).
*   **Scope**: 17,227 Transition Metal Oxides (TMOs).
*   **Validation Strategy**: **SOAP-LOCO** (Smooth Overlap of Atomic Positions - Leave One Cluster Out).
    *   Instead of random splitting, we cluster materials by structural similarity (using SOAP descriptors).
    *   We train on $N-1$ clusters and test on the unseen cluster. This mimics the real-world scenario of discovering *new* families of materials, ensuring our metrics are rigorous.

### Model Architecture: CHGNet
We utilize **CHGNet** (Crystal Hamiltonian Graph Neural Network, Deng et al., 2023) — a universal neural network potential pre-trained on the Materials Project.
*   **Base Model**: CHGNet v0.3.0 with 412,525 parameters
*   **Fine-tuning**: Trained on Li-O-TM cathodes with SOAP-LOCO splits
*   **Ensemble**: 5 models with different random seeds for uncertainty
*   **Output**: Direct E_hull prediction (not total energy)

### Uncertainty Quantification
We implement **Deep Ensembles** (Lakshminarayanan et al., 2017) to quantify **Epistemic Uncertainty** (model ignorance).
*   We train **M=5** models with different random initializations and data shuffles.
*   **Prediction**: μ = (1/M) Σ μ<sub>m</sub>(x)
*   **Uncertainty**: σ² = (1/M) Σ (μ<sub>m</sub>(x)² − μ²)

### Decision Policy
Materials are classified based on a hybrid policy:

| Action | Criterion | Meaning |
| :--- | :--- | :--- |
| **KEEP** | μ < 0.05 eV & σ < 0.02 | High confidence stable. Send to DFT. |
| **MAYBE** | 0.05 ≤ μ ≤ 0.15 or σ > 0.02 | Uncertain. Manual review recommended. |
| **KILL** | μ > 0.15 eV | Confident unstable. Do not compute. |

---

## Performance Metrics (v1-Li-Cathode)

> **Industrial Claim**: On unseen Li-cathode chemistries, our system recovers **55% of all ultra-stable (E_hull ≤ 10 meV) materials** within the top-100 candidates — a **6.6× enrichment** over random — while maintaining **< 0.3% false-discard rate**.

### Core Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MAE** | **0.032 eV/atom** | Mean Absolute Error (SOAP-LOCO test) |
| **RMSE** | 0.063 eV/atom | Root Mean Square Error |
| **Mean σ** | 0.008 eV/atom | Ensemble epistemic uncertainty |
| **False Kill Rate** | **< 0.3%** | Stable materials incorrectly discarded |

### Enrichment Factors

| Threshold | EF@1% | EF@5% | Recall@100 |
| :--- | :--- | :--- | :--- |
| **0.01 eV** | **6.66×** | 4.91× | 55% |
| 0.02 eV | 3.30× | 3.19× | 45% |
| 0.05 eV | 1.96× | 1.81× | 23% |

**Model Scope**: Li-containing oxide cathode materials (Li–O–TM)  
**Validation**: SOAP-LOCO chemistry-aware holdout split (764 test samples)

---

## Installation & Deployment

### Option 1: Docker (Recommended)
The system is ready for cloud deployment (e.g., Google Cloud Run, AWS Fargate).

```bash
# 1. Start the stack
docker-compose up --build -d

# 2. Access
# Frontend: http://localhost:3000
# Backend Docs: http://localhost:8080/docs
```

Optional edge proxy (rate limiting, headers):

```bash
docker-compose -f docker-compose.yml -f docker-compose.edge.yml up --build -d
```

GCP deployment notes are in `docs/gcp_deployment.md` (uses `deploy_gcp.ps1`).
GCP edge, observability, and scaling guides: `docs/gcp_edge_cloud_armor.md`, `docs/gcp_observability.md`, `docs/gcp_scaling.md`.

### Option 2: Local Development

**Prerequisites**: Python 3.10+, Node.js 20+

```bash
# Backend (Terminal 1)
# Must run from project root
pip install -r web/api/requirements.txt
python -m uvicorn web.api.main:app --port 8001

# Frontend (Terminal 2)
# Must navigate to frontend directory first!
cd web/frontend
npm install
npm run dev
```

### API Security & Limits

For production deployments:
- Set `CATHODE_ENV=production`, `CATHODE_AUTH_ENABLED=true`, and provide `CATHODE_API_KEY` (or `CATHODE_API_KEYS` / `CATHODE_API_KEY_HASHES`). Requests can use `X-API-Key` or `Authorization: Bearer`.
- Configure request limits and validation, e.g. `CATHODE_RATE_LIMIT_PER_MINUTE`, `CATHODE_MAX_FILE_BYTES`, `CATHODE_MAX_BATCH_SIZE`, and `CATHODE_MAX_ATOMS`.
- Apply backpressure with `CATHODE_MAX_CONCURRENT_REQUESTS` and `CATHODE_CONCURRENCY_TIMEOUT_SECONDS`.
- Consider `CATHODE_IP_ALLOWLIST`, `CATHODE_TRUST_PROXY`, `CATHODE_FORCE_HTTPS`, and `CATHODE_SECURITY_HEADERS` behind a trusted reverse proxy.
- Use `CATHODE_SECRET_FILE` or `CATHODE_SECRET_COMMAND` to load secrets at startup.
- Enforce startup checks with `CATHODE_STRICT_STARTUP=true` and `CATHODE_REQUIRE_CALIBRATION=true`.
- Sign and verify artifacts with `CATHODE_MANIFEST_HMAC_KEY` + `CATHODE_REQUIRE_MANIFEST_SIGNATURE=true`.
- Keep safe checkpoint loading enabled; only set `CATHODE_ALLOW_UNSAFE_TORCH_LOAD=true` when artifacts are fully trusted.

### Observability

- Each response includes `X-Request-ID` (client-supplied or generated).
- Enable request logging with `CATHODE_LOG_REQUESTS=true`.
- Enable Prometheus text metrics at `/metrics/prometheus` with `CATHODE_PROMETHEUS_ENABLED=true`.
- Enable OpenTelemetry tracing with `CATHODE_OTEL_ENABLED=true` and set `CATHODE_OTEL_EXPORTER_OTLP_ENDPOINT`.
- See `docs/observability.md` for example alert rules.
- GCP-specific alert setup is covered in `docs/gcp_observability.md`.

### ML Governance

- Generate and sign artifact manifests with `scripts/08_generate_artifact_manifest.py --sign` and verify via `CATHODE_REQUIRE_MANIFEST_SIGNATURE=true`.
- Evaluate prediction quality with `scripts/09_evaluate_predictions.py`.
- Track drift with `scripts/10_compute_drift.py` (outputs `retrain_recommended` when PSI exceeds threshold).
- Gate releases with `scripts/12_validate_release.py` and publish to a registry using `scripts/13_publish_registry.py`.

See `docs/production_checklist.md` for a full production readiness checklist.

### Load Testing

- Use `scripts/11_load_test_api.py` to generate baseline latency/error stats for `/predict`.
- For Cloud Run scaling guidance, see `docs/gcp_scaling.md`.

---

## References
1.  Deng, B., et al. (2023). CHGNet: Pretrained universal neural network potential for charge-informed atomistic modelling. *Nature Machine Intelligence*.
2.  Jain, A., et al. (2013). The Materials Project: A materials genome approach. *APL Mater.*
3.  Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
4.  Bartók, A. P., et al. (2013). On representing chemical environments. *Phys. Rev. B*.
