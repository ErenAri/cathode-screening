# CathodeScreen: High-Throughput Screening of Li-Ion Battery Cathodes

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)
![Next.js](https://img.shields.io/badge/Next.js-13-black?style=flat&logo=next.js)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**CathodeScreen** is an open-source framework for the accelerated discovery of thermodynamic stability in lithium-ion battery cathode materials. It implements a scalable inference pipeline utilizing a deep ensemble of Crystal Graph Convolutional Neural Networks (CGCNN) to robustly predict energy above hull ($E_{hull}$) with quantified epistemic uncertainty.

![System Architecture](assets/home_page.png)

## Abstract

The discovery of novel cathode materials is constrained by the computationally expensive nature of Density Functional Theory (DFT) calculations. This repository implements a data-driven screening funnel that serves as a pre-filter for DFT. By leveraging a specialized graph neural network ensemble trained on 17,227 transition metal oxides, the system identifies thermodynamically stable candidates ($E_{hull} < 0.05$ eV/atom) with a Discovery Acceleration Factor (DAF) of 1.64× compared to random sampling.

## System Architecture

The application is architected as a decoupled microservices system designed for containerized deployment:

*   **Inference Engine (Backend)**:
    *   **Framework**: FastAPI (Python 3.10).
    *   **Model**: 5-member CGCNN ensemble (`torch_geometric`).
    *   **Optimization**: CPU-optimized inference (< 50ms per structure) via vectorized message passing.
    *   **Quantification**: Epistemic uncertainty estimation via variance of ensemble predictions.
*   **User Interface (Frontend)**:
    *   **Framework**: Next.js 14 (React / TypeScript).
    *   **Rendering**: Server-Side Rendering (SSR) for SEO and initial load performance.
    *   **Visualization**: Interactive confidence intervals and decision boundary visualization.

## Validation Metrics

The model was evaluated using **SOAP-LOCO** (Smooth Overlap of Atomic Positions - Leave One Cluster Out) cross-validation to ensure generalization to out-of-distribution (OOD) chemistries.

| Metric | Value | Definition |
| :--- | :--- | :--- |
| **MAE** | **0.038 eV/atom** | Mean Absolute Error on Test Set |
| **DAF@10** | **1.64×** | Top-10% Discovery Acceleration Factor |
| **Efficiency** | **+86%** | Improvement over random sampling in Active Learning Sim |
| **Inference Latency** | **42ms** | Average per-structure processing time (Intel Xeon CPU) |

## Installation & Deployment

### Docker Orchestration (Recommended)

The system is fully containerized. To initialize the local development environment:

```bash
# 1. Initialize the container stack
docker-compose up --build -d

# 2. Tail logs
docker-compose logs -f
```

The application exposes the following endpoints:
*   **Frontend**: `http://localhost:3000`
*   **API Documentation (Swagger)**: `http://localhost:8080/docs`

### Manual Building

**Requirements**: Python 3.10+, Node.js 18+, PyTorch 2.0+

```bash
# Backend Initialization
pip install -r web/api/requirements.txt
python -m uvicorn web.api.main:app --host 0.0.0.0 --port 8001

# Frontend Initialization
cd web/frontend
npm ci
npm run build && npm start
```

## Methodology

### Crystal Graph Convolution (CGCNN)
We represent crystal structures as a multigraph $G = (V, E)$, where nodes $v_i$ represent atoms and edges $e_{ij}$ represent bonds. The node feature vectors $h_i$ are updated via:

$$
v_i^{(t+1)} = v_i^{(t)} + \sum_{j \in N(i)} \sigma(z_{i,j} W_f + b_f) \odot g(z_{i,j} W_s + b_s)
$$

Where $z_{i,j}$ is the concatenation of neighbor vectors and bond features.

### Decision Policy
Materials are classified based on a hybrid policy of predicted stability and uncertainty $\sigma$:

1.  **RECOMMEND (DFT)**: $\mu_{pred} < 0.08$ eV/atom AND $\sigma < 0.05$ (High Confidence Stable)
2.  **HOLD**: $\sigma > 0.05$ (High Uncertainty / OOD)
3.  **SKIP**: $\mu_{pred} > 0.15$ eV/atom (Confident Unstable)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

```bibtex
@software{cathodescreen2026,
  title = {CathodeScreen: ML-Powered Cathode Materials Screening},
  author = {Ari, Eren},
  year = {2026},
  url = {https://github.com/ErenAri/cathode-screening}
}
```
