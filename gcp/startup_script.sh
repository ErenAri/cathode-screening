#!/bin/bash
# GCP A100 VM Startup Script
# This script sets up the training environment on a fresh A100 VM

set -e

echo "=============================================="
echo "CHGNet Training VM Setup"
echo "=============================================="

# Install NVIDIA drivers (if not already installed)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535
fi

# Install Docker (if not already installed)
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Install gcloud (if not already installed)
if ! command -v gcloud &> /dev/null; then
    echo "Installing gcloud CLI..."
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh --quiet
    source ~/.bashrc
fi

# Create working directory
mkdir -p /home/$USER/cathode-training
cd /home/$USER/cathode-training

# Download training data from Cloud Storage
echo "Downloading training data from gs://cathode-screening-training..."
gsutil -m cp -r gs://cathode-screening-training/data .

# Clone repository (optional - if using git)
# git clone https://github.com/ErenAri/cathode-screening.git
# cd cathode-screening

# Pull Docker image
echo "Pulling training Docker image..."
docker pull gcr.io/cathode-screening/chgnet-training:latest

# Run training
echo "Starting training..."
docker run --gpus all \
    -v /home/$USER/cathode-training/data:/app/data \
    -v /home/$USER/cathode-training/checkpoints:/app/checkpoints \
    gcr.io/cathode-screening/chgnet-training:latest \
    python scripts/35_train_gcp_a100.py --phase both

echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to /home/$USER/cathode-training/checkpoints"
echo "=============================================="

# Upload results back to Cloud Storage
echo "Uploading results to Cloud Storage..."
gsutil -m cp -r /home/$USER/cathode-training/checkpoints gs://cathode-screening-training/

echo "Done!"
