# Render deployment for MACE-based cathode screening backend
FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY web/api/requirements.txt /app/requirements.txt
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web /app/web
COPY src /app/src
COPY configs /app/configs

# Copy MACE ensemble artifacts + calibration + predictions
COPY artifacts/models/mace_ensemble_v1 /app/artifacts/models/mace_ensemble_v1
COPY data/predictions /app/data/predictions

# Copy reports for screening endpoints
COPY reports /app/reports

# Environment
ENV PYTHONPATH=/app/src
ENV CATHODE_MODEL_TYPE=mace
ENV CATHODE_ARTIFACTS_DIR=/app/artifacts
ENV CATHODE_DEVICE=cpu
ENV CATHODE_AUTH_ENABLED=false

# Unprivileged user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render sets PORT automatically
ENV PORT=8080
EXPOSE 8080

CMD uvicorn web.api.main:app --host 0.0.0.0 --port ${PORT}
