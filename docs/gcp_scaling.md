# GCP Scaling and Load Testing

Cloud Run autoscaling is driven by concurrency and request latency. ML inference tends to be CPU-bound, so start with low concurrency and scale out by instances.

Suggested baseline
- Concurrency: 1-4 (start at 2).
- Min instances: 1 (avoid cold starts if latency is critical).
- Max instances: 20+ depending on traffic.

Tune with load tests
```
python scripts/11_load_test_api.py --url https://YOUR_BACKEND_URL/predict --cif test.cif --requests 50 --concurrency 5 --api-key YOUR_API_KEY
```

Update Cloud Run scaling
```
.\scripts\gcp_scale_tune.ps1 -ProjectId YOUR_PROJECT_ID -ServiceName cathode-backend -Region us-central1 -Concurrency 2 -MinInstances 1 -MaxInstances 20
```

Manual gcloud update (optional)
```
gcloud run services update cathode-backend --region us-central1 --concurrency 2 --min-instances 1 --max-instances 20 --memory 2Gi --cpu 2
```

Recommendations
- Increase concurrency only after confirming acceptable p95 latency.
- Raise `CATHODE_MAX_CONCURRENT_REQUESTS` to match the Cloud Run concurrency if you keep internal backpressure.
- Monitor `run.googleapis.com/request_latencies` while adjusting.
