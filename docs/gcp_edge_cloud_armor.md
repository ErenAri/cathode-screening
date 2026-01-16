# GCP Edge + Cloud Armor

Cloud Armor requires an HTTPS Load Balancer in front of Cloud Run. This guide shows how to put a global HTTPS load balancer in front of the backend service and attach a Cloud Armor policy.

Prerequisites
- A custom domain you control (DNS access).
- Cloud Run backend deployed.
- APIs enabled: `compute.googleapis.com`, `run.googleapis.com`.

Quick start (scripted)
```
.\scripts\gcp_setup_edge.ps1 -ProjectId YOUR_PROJECT_ID -Domain api.example.com -ServiceName cathode-backend -Region us-central1
```

Manual steps (gcloud)
```
gcloud config set project YOUR_PROJECT_ID

# Lock Cloud Run ingress to the load balancer.
gcloud run services update cathode-backend --region us-central1 --ingress internal-and-cloud-load-balancing

# Reserve a global IP.
gcloud compute addresses create cathode-lb-ip --global

# Create managed certificate.
gcloud compute ssl-certificates create cathode-managed-cert --domains api.example.com

# Create serverless NEG for Cloud Run.
gcloud compute network-endpoint-groups create cathode-backend-neg --region us-central1 --network-endpoint-type=serverless --cloud-run-service cathode-backend

# Create backend service and attach NEG.
gcloud compute backend-services create cathode-backend-service --global --load-balancing-scheme=EXTERNAL --protocol=HTTP
gcloud compute backend-services add-backend cathode-backend-service --global --network-endpoint-group cathode-backend-neg --network-endpoint-group-region us-central1

# Create URL map, HTTPS proxy, and forwarding rule.
gcloud compute url-maps create cathode-url-map --default-service cathode-backend-service
gcloud compute target-https-proxies create cathode-https-proxy --url-map cathode-url-map --ssl-certificates cathode-managed-cert
gcloud compute forwarding-rules create cathode-https-forwarding-rule --global --target-https-proxy cathode-https-proxy --ports 443 --address cathode-lb-ip

# Create and attach Cloud Armor policy.
gcloud compute security-policies create cathode-edge-policy --description "Cathode API edge policy"
gcloud compute backend-services update cathode-backend-service --global --security-policy cathode-edge-policy
```

DNS and certificate provisioning
- Point your DNS A record (e.g. `api.example.com`) to the reserved IP from `gcloud compute addresses describe cathode-lb-ip --global --format "value(address)"`.
- Managed SSL certificates may take 15-60 minutes to become active.

Optional Cloud Armor rules
```
gcloud compute security-policies rules create 1000 --security-policy cathode-edge-policy --expression "true" --action "throttle" --rate-limit-threshold-count 60 --rate-limit-threshold-interval-sec 60 --conform-action allow --exceed-action deny-429
```

Validation
- Hit the load balancer URL and confirm `/ready` returns 200.
- Confirm Cloud Armor policy is attached via `gcloud compute backend-services describe cathode-backend-service --global`.
