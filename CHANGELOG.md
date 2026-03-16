# Changelog

All notable changes to CathodeScreen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-03-16

### Added
- **Async prediction queue**: Celery + Redis based task queue for offloading batch predictions to GPU workers. New endpoints: `POST /predict/async`, `GET /predict/async/{job_id}`.
- **GPU dynamic batcher**: `DynamicBatcher` accumulates structures and dispatches optimally-sized batches to minimize GPU idle time. Configurable via `CATHODE_ENABLE_BATCHING`, `CATHODE_BATCH_SIZE`, `CATHODE_BATCH_TIMEOUT_MS`.
- **Shadow deployment**: Run a candidate model alongside production on N% of traffic. Compares decisions and logs disagreements for safe promotion. Endpoints: `GET /shadow/stats`, `GET /shadow/analysis`.
- **Drift alerting**: Automated PSI-based drift detection with multi-channel alerting (Slack webhooks, PagerDuty Events API v2, generic webhooks). Runs hourly via Celery beat.
- **Model registry**: Dual-backend registry (local JSON + MLflow) for versioning, governance gating, and stage promotion (staging → production → archived). Endpoints: `GET /registry/models`, `GET /registry/production`.
- **Locust load testing**: Full load test suite at `tests/load/locustfile.py` with realistic traffic patterns (single predictions, batches, monitoring, health). Target: 1000 predictions/minute.
- **Celery worker and beat**: Docker services for async inference workers and periodic drift monitoring.
- **Redis service**: Added to docker-compose for task queue and result backend.

### Changed
- `docker-compose.yml`: Added `redis`, `celery-worker`, `celery-beat` services with health checks.
- `pyproject.toml` version bumped to 1.2.0.
- Production predict endpoint now fires shadow predictions asynchronously when enabled.

### Dependencies
- Added optional `[queue]` dependency group: `celery[redis]>=5.3`, `redis>=5.0`
- Added optional `[registry]` dependency group: `mlflow>=2.10`
- Added optional `[loadtest]` dependency group: `locust>=2.20`

## [1.1.0] - 2026-03-16

### Added
- **API Versioning**: All endpoints now available under `/v1/` prefix. Unversioned routes remain for backward compatibility.
- **PostgreSQL audit trail**: New `CATHODE_AUDIT_BACKEND=postgres` option with full schema, connection pooling, and indexed queries. Falls back to JSONL if unavailable.
- **DVC pipeline**: Data version control with `dvc.yaml` defining reproducible stages from fetch → train → calibrate → evaluate → release.
- **Model Card**: Formal `MODEL_CARD.md` following Mitchell et al. (2019) framework with full training data, metrics, limitations, and ethical considerations.
- **Data version pinning**: `data/DATA_VERSION.json` tracks exact Materials Project query parameters, dataset statistics, and data hashes.
- **Integration test suite**: End-to-end tests covering prediction pipeline (CIF upload → decision), input validation, API contracts, and audit trail verification.
- **CIF test fixtures**: Known-answer materials (LiCoO2, LiMn2O4, LiFePO4) and invalid compositions (NaCl) for regression testing.
- **V&V documentation**: Installation, Operational, and Performance Qualification protocols under `docs/validation/`.
- **Traceability matrix**: Requirements → Tests → Evidence mapping for regulatory compliance.
- **CHANGELOG**: This file, tracking all versioned changes.

### Changed
- `pyproject.toml` version bumped to 1.1.0
- pytest now discovers tests from both `src/cathode_screening/tests` and `tests/` directories
- Audit trail in prediction endpoints now uses configurable backend selector (`get_audit_backend()`)

### Dependencies
- Added optional `[postgres]` dependency group: `psycopg2-binary>=2.9`
- Added optional `[dvc]` dependency group: `dvc>=3.0`, `dvc-gs`, `dvc-s3`
- Added `httpx>=0.25` to `[dev]` dependencies for integration testing

## [1.0.0] - 2025-01-15

### Added
- Initial release of CathodeScreen
- 5-member MACE-MP-0 fine-tuned ensemble with quantile regression
- Conformal calibration for 90% prediction interval coverage
- Out-of-distribution detection (3-gate: composition, embedding, disagreement)
- Decision policy with KEEP/MAYBE/KILL classification
- FastAPI backend with authentication, rate limiting, and CORS
- Next.js 14 frontend with prediction UI, database viewer, and discovery dashboard
- Discovery campaign engine with active learning loop framework
- JSONL audit trail with daily rotation
- Prometheus and OpenTelemetry observability
- Artifact manifest with HMAC signing
- Docker deployment (Render backend + Vercel frontend)
- GCP Cloud Run deployment support
- 17 unit test files covering core inference, policy, calibration, OOD, and security
- SOAP-LOCO validation methodology
- Governance checks (6/6 automated gates)
- Cathode property calculators (capacity, voltage, energy density)

### Governance Results
- Test MAE: 0.030 eV/atom
- Spearman ρ: 0.663
- Calibration coverage: 91.3% (target 90%)
- KEEP precision: 92.7%
- False-kill rate: 0.0%

[Unreleased]: https://github.com/your-org/cathode-screening/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/your-org/cathode-screening/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/your-org/cathode-screening/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/cathode-screening/releases/tag/v1.0.0
