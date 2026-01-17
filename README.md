# CathodeScreen: High-Throughput Screening of Li-Ion Battery Cathodes

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=flat&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.js)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)

**CathodeScreen** is an enterprise-grade machine learning framework designed to accelerate the discovery of thermodynamically stable lithium-ion battery cathode materials. It implements a scalable inference pipeline utilizing a deep ensemble of Crystal Graph Convolutional Neural Networks (CGCNN) to robustly predict energy above hull ($E_{hull}$) with quantified epistemic uncertainty.


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

The discovery of novel cathode materials is constrained by the computationally expensive nature of Density Functional Theory (DFT) calculations, which scale as $O(N^3)$. **CathodeScreen** implements a data-driven screening funnel that serves as a pre-filter for DFT. By leveraging a specialized graph neural network ensemble trained on 17,227 transition metal oxides, the system identifies thermodynamically stable candidates ($E_{hull} < 0.05$ eV/atom) with a Discovery Acceleration Factor (DAF) of **1.64×** compared to random sampling.

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
        Pymatgen -->|Graph| GNN1(CGCNN Model 1)
        Pymatgen -->|Graph| GNN2(CGCNN Model 2)
        Pymatgen -->|Graph| GNN3(CGCNN Model 3)
        Pymatgen -->|Graph| GNN4(CGCNN Model 4)
        Pymatgen -->|Graph| GNN5(CGCNN Model 5)
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

### Model Architecture: CGCNN
We utilize the **Crystal Graph Convolutional Neural Network** (Xie et al., 2018).
*   **Input**: Crystal structure converted to a multigraph $G = (V, E)$.
    *   **Nodes ($v_i$)**: Atom embeddings (atomic number, electronegativity, etc.).
    *   **Edges ($e_{ij}$)**: Bond distances encoded via Gaussian Basis expansion.
*   **Update Rule**:
    $$ v_i^{(t+1)} = v_i^{(t)} + \sum_{j \in N(i)} \sigma(z_{i,j} W_f + b_f) \odot g(z_{i,j} W_s + b_s) $$

### Uncertainty Quantification
We implement **Deep Ensembles** (Lakshminarayanan et al., 2017) to quantify **Epistemic Uncertainty** (model ignorance).
*   We train $M=5$ models with different random initializations and data shuffles.
*   **Prediction**: $\mu_* = \frac{1}{M} \sum \mu_m(x)$
*   **Uncertainty**: $\sigma_*^2 = \frac{1}{M} \sum (\mu_m(x)^2 - \mu_*^2)$

### Decision Policy
Materials are classified based on a hybrid policy:

| Action | Criterion | Meaning |
| :--- | :--- | :--- |
| **RECOMMEND** | $\mu < 0.08$ eV & $\sigma < 0.05$ | High confidence stable. Send to DFT. |
| **HOLD** | $\sigma > 0.05$ | Model is confused (OOD). Candidate for Active Learning. |
| **SKIP** | $\mu > 0.15$ eV | Confident unstable. Do not compute. |

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
1.  Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks. *Phys. Rev. Lett.*
2.  Jain, A., et al. (2013). The Materials Project: A materials genome approach. *APL Mater.*
3.  Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
