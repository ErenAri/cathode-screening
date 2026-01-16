# Observability Guide

This service exposes in-memory metrics via `/metrics/prometheus` (enable with `CATHODE_PROMETHEUS_ENABLED=true`).

Sample Prometheus alert rules:

```
groups:
- name: cathode-alerts
  rules:
  - alert: CathodeHighErrorRate
    expr: increase(cathode_errors_total[5m]) / increase(cathode_requests_total[5m]) > 0.05
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: High error rate in cathode screening API
  - alert: CathodeHighLatency
    expr: cathode_latency_seconds_mean > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: High average latency in cathode screening API
```

OpenTelemetry tracing:

- Enable with `CATHODE_OTEL_ENABLED=true`.
- Set `CATHODE_OTEL_EXPORTER_OTLP_ENDPOINT` to your collector.
