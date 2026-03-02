# Render deployment for MACE-based cathode screening backend
FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling ~2GB CUDA torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies (torch already satisfied)
COPY web/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web /app/web
COPY src /app/src
COPY configs /app/configs

# Copy MACE ensemble artifacts + calibration
COPY artifacts/models/mace_ensemble_v1 /app/artifacts/models/mace_ensemble_v1

# Copy predictions (force-added past .gitignore)
COPY data/predictions /app/data/predictions

# Copy reports for screening endpoints (may be empty)
COPY reports /app/reports

# Environment
ENV PYTHONPATH=/app/src
ENV CATHODE_MODEL_TYPE=mace
ENV CATHODE_ARTIFACTS_DIR=/app/artifacts
ENV CATHODE_DEVICE=cpu
ENV CATHODE_ENV=production
ENV CATHODE_AUTH_ENABLED=false
ENV CATHODE_STRICT_STARTUP=false
ENV CATHODE_REQUIRE_CALIBRATION=false
ENV CATHODE_FORCE_HTTPS=false

# Unprivileged user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render sets PORT automatically
ENV PORT=8080
EXPOSE 8080

CMD uvicorn web.api.main:app --host 0.0.0.0 --port ${PORT}
