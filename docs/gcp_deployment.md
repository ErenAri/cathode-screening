# GCP Deployment Guide

This project deploys cleanly to Cloud Run using Cloud Build. The `deploy_gcp.ps1` script now supports Secret Manager and common production flags.

## Prerequisites

- `gcloud` installed and authenticated
- APIs enabled: `run.googleapis.com`, `cloudbuild.googleapis.com`, `secretmanager.googleapis.com`

Example:

```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

## Secrets (recommended)

Create secrets for API keys and the manifest signing key:

```
printf "YOUR_API_KEY" | gcloud secrets create cathode-api-keys --data-file=-
printf "YOUR_MANIFEST_HMAC_KEY" | gcloud secrets create cathode-manifest-hmac-key --data-file=-
```

If the secrets already exist, add new versions:

```
printf "YOUR_API_KEY" | gcloud secrets versions add cathode-api-keys --data-file=-
printf "YOUR_MANIFEST_HMAC_KEY" | gcloud secrets versions add cathode-manifest-hmac-key --data-file=-
```

## Deploy

```
.\deploy_gcp.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1 -UseSecretManager
```

By default the script sets production env vars, turns on request logging, and enforces manifest signature checks.

## Optional flags

- `-AllowUnauthenticatedBackend` (use API key auth only)
- `-BackendConcurrency` / `-BackendMaxInstances` / `-BackendMinInstances`
- `-ServiceAccount` for Cloud Run service account

## Edge protection (Cloud Armor)

Cloud Armor requires an HTTPS load balancer in front of Cloud Run. If you need WAF/rate limiting at the edge, place a load balancer in front of Cloud Run and attach Cloud Armor rules there.

Additional GCP guides:
- Edge setup: `docs/gcp_edge_cloud_armor.md`
- Observability and alerts: `docs/gcp_observability.md`
- Scaling and load testing: `docs/gcp_scaling.md`
