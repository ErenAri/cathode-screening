# GCP Training Commands Reference

## 1. Upload Data to Cloud Storage

```bash
# Upload full MPTrj (11.35 GB)
gsutil -m cp data/external/mptrj_full/MPtrj_2022.9_full.json gs://cathode-screening-training/data/mptrj_full/

# Upload Li-cathode data
gsutil -m cp -r data/external/oqmd gs://cathode-screening-training/data/external/
gsutil -m cp -r data/external/mp_2024 gs://cathode-screening-training/data/external/
gsutil -m cp -r data/external/mptrj gs://cathode-screening-training/data/external/
gsutil -m cp -r data/external/nomad gs://cathode-screening-training/data/external/
gsutil -m cp -r data/external/wbm gs://cathode-screening-training/data/external/

# Upload training scripts
gsutil -m cp -r scripts gs://cathode-screening-training/
gsutil -m cp -r src gs://cathode-screening-training/
```

## 2. Create A100 VM

```bash
# Create VM with A100 GPU (40GB)
gcloud compute instances create chgnet-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata-from-file=startup-script=gcp/startup_script.sh

# SSH into VM
gcloud compute ssh chgnet-training --zone=us-central1-a
```

## 3. Alternative: Use Vertex AI Training

```bash
# Submit training job to Vertex AI
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=chgnet-training \
    --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=gcr.io/cathode-screening/chgnet-training:latest
```

## 4. Monitor Training

```bash
# Check VM status
gcloud compute instances list

# View training logs
gcloud compute ssh chgnet-training --zone=us-central1-a --command="tail -f /var/log/syslog"

# Download checkpoints
gsutil -m cp -r gs://cathode-screening-training/checkpoints ./
```

## 5. Cleanup (Save Money!)

```bash
# Stop VM (keeps disk, stops billing for GPU)
gcloud compute instances stop chgnet-training --zone=us-central1-a

# Delete VM completely
gcloud compute instances delete chgnet-training --zone=us-central1-a

# Delete bucket (if done)
# gsutil rm -r gs://cathode-screening-training
```

## Cost Estimate

| Phase | Time | Cost/hr | Total |
|-------|------|---------|-------|
| Pretrain (1.58M) | 20-30h | $3.50 | $70-105 |
| Fine-tune (35K×5) | 5-10h | $3.50 | $17-35 |
| **Total** | | | **~$90-140** |
