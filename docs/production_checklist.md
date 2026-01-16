# Production Readiness Checklist

Security and edge
- Place the API behind a reverse proxy/WAF (see `docker-compose.edge.yml`).
- For GCP, configure HTTPS load balancer + Cloud Armor (`docs/gcp_edge_cloud_armor.md`).
- Store secrets in a secrets manager and inject via `CATHODE_SECRET_FILE` or `CATHODE_SECRET_COMMAND`.
- Enable `CATHODE_FORCE_HTTPS=true` and `CATHODE_SECURITY_HEADERS=true` at the edge.

Observability and SRE
- Enable Prometheus metrics (`CATHODE_PROMETHEUS_ENABLED=true`) and configure alert rules.
- For GCP, create alert policies with `scripts/gcp_create_alerts.ps1` (see `docs/gcp_observability.md`).
- Enable OpenTelemetry tracing with `CATHODE_OTEL_ENABLED=true`.
- Log requests with `CATHODE_LOG_REQUESTS=true` and ship logs to a central sink.

Reliability and scale
- Set `CATHODE_MAX_CONCURRENT_REQUESTS` to apply backpressure.
- Run load tests with `scripts/11_load_test_api.py` and set autoscaling policies.
- For Cloud Run tuning, use `scripts/gcp_scale_tune.ps1` (see `docs/gcp_scaling.md`).
- Use `/ready` for readiness checks and `/metrics/prometheus` for signals.

ML governance
- Sign artifacts and enforce signature verification.
- Run `scripts/12_validate_release.py` before deploys.
- Track drift with `scripts/10_compute_drift.py` and retrain on sustained PSI alerts.
