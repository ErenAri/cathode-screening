# GCP Observability

Cloud Run provides native request metrics and Cloud Logging out of the box. This project also exposes in-memory Prometheus metrics at `/metrics/prometheus` when `CATHODE_PROMETHEUS_ENABLED=true`.

Cloud Run metrics to watch
- `run.googleapis.com/request_count` (error rate by response class)
- `run.googleapis.com/request_latencies` (latency distribution)

Request logging
- Enable `CATHODE_LOG_REQUESTS=true` to emit request/response metadata.
- Use Cloud Logging queries like:
```
resource.type="cloud_run_revision"
resource.labels.service_name="cathode-backend"
severity>=ERROR
```

Create alert policies (Cloud Monitoring)
The repository includes alert policy templates under `configs/gcp` and a helper script.
```
.\scripts\gcp_create_alerts.ps1 -ProjectId YOUR_PROJECT_ID -ServiceName cathode-backend
```

To update thresholds, edit:
- `configs/gcp/alert_policy_error_rate.json`
- `configs/gcp/alert_policy_latency.json`

If you need to update existing policies, pass `-UpdateExisting`.

Optional: export logs/metrics
- Configure a log sink to BigQuery or Pub/Sub if you need long-term analytics.
- Export metrics to Prometheus or a managed OTEL collector using `CATHODE_OTEL_ENABLED=true`.
