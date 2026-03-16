# Installation Qualification (IQ) Protocol

**Document ID:** VV-IQ-001
**Version:** 1.0
**Effective Date:** 2026-03-16
**System:** CathodeScreen v1.1.0
**Classification:** Quality Assurance

---

## 1. Purpose

This Installation Qualification protocol verifies that all hardware, software, and infrastructure components of the CathodeScreen system are installed correctly and meet specified requirements.

## 2. Scope

Applies to all deployment environments:
- Development (local Docker Compose)
- Staging (Render/Vercel preview)
- Production (Render backend + Vercel frontend)

## 3. Prerequisites

| Item | Requirement |
|---|---|
| Python | 3.10 - 3.12 |
| Node.js | 20.x LTS |
| Docker | 24.0+ |
| Git | 2.40+ |
| GPU (training) | CUDA 11.8+ / RTX 2060+ (optional for inference) |
| Memory | ≥ 4 GB (API), ≥ 8 GB (training) |

## 4. IQ Checks

### IQ-001: Python Environment
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| Python version | `python --version` | 3.10.x - 3.12.x | |
| pip version | `pip --version` | ≥ 23.0 | |
| venv creation | `python -m venv .venv` | No error | |

### IQ-002: Core Dependencies
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| Install core | `pip install -e .` | No error | |
| PyTorch import | `python -c "import torch; print(torch.__version__)"` | ≥ 2.0 | |
| pymatgen import | `python -c "import pymatgen; print(pymatgen.__version__)"` | ≥ 2024.1.1 | |
| MACE import | `python -c "import mace"` | No error (if [mace] installed) | |

### IQ-003: Model Artifacts
| Check | Verification | Expected | Pass/Fail |
|---|---|---|---|
| Ensemble checkpoints | `ls artifacts/models/mace_ensemble_v1/*.pt` | 5 files | |
| Normalizer | `ls artifacts/models/mace_ensemble_v1/normalizer.json` | 1 file | |
| Conformal params | `ls artifacts/models/mace_ensemble_v1/conformal_params.json` | 1 file | |
| Manifest | `ls artifacts/models/mace_ensemble_v1/manifest.json` | 1 file | |
| Manifest signature | `python -c "from cathode_screening.inference.artifact_manifest import load_manifest; m = load_manifest(...)"` | Signature valid | |

### IQ-004: API Server
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| FastAPI starts | `uvicorn web.api.main:app --port 8000` | No startup error | |
| Health check | `curl http://localhost:8000/` | `{"status":"ok"}` | |
| Readiness check | `curl http://localhost:8000/ready` | `{"status":"ready"}` | |
| OpenAPI docs | `curl http://localhost:8000/docs` | HTML page loads | |
| v1 routes | `curl http://localhost:8000/v1/` | `{"status":"ok"}` | |

### IQ-005: Frontend
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| npm install | `cd web/frontend && npm install` | No error | |
| Build | `npm run build` | No error | |
| Dev server | `npm run dev` | Starts on :3000 | |

### IQ-006: Docker
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| Backend build | `docker build -f render.Dockerfile .` | Success | |
| Frontend build | `docker build -f frontend.Dockerfile .` | Success | |
| Compose up | `docker-compose up -d` | Both services running | |
| Backend health | `curl http://localhost:8080/` | `{"status":"ok"}` | |

### IQ-007: Database (PostgreSQL, if enabled)
| Check | Command | Expected | Pass/Fail |
|---|---|---|---|
| Connection | `psql $CATHODE_DATABASE_URL -c "SELECT 1"` | Returns 1 | |
| Schema creation | Start API with `CATHODE_AUDIT_BACKEND=postgres` | `prediction_audit` table created | |
| Write test | Make prediction, check `SELECT COUNT(*) FROM prediction_audit` | Count incremented | |

## 5. Acceptance Criteria

- All IQ checks must PASS
- Any FAIL must be documented with root cause and corrective action
- IQ must be re-executed after any infrastructure change

## 6. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| QA Lead | | | |
| DevOps Lead | | | |
| Project Manager | | | |
