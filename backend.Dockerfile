# Use official Python 3.10 slim image (smaller size)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
# gcc and python3-dev are often needed for building wheels
# curl is good for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version FIRST (to avoid downloading huge GPU version)
# This saves ~2GB of image size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements
COPY web/api/requirements.txt /app/requirements.txt

# Install other dependencies
# (excluding torch since we installed it manually)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web /app/web

# Copy all artifacts (models, meta, etc)
COPY artifacts /app/artifacts

# Copy database predictions (Parquet)
COPY data/predictions /app/data/predictions

# Create unprivileged user for security (optional but AWS/GCP best practice)
# But often simpler to run as root in simple containers. Sticking to root for MVP simplicity.

# Expose port (Cloud Run sets PORT env var, defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Run the application
# We use shell form to expand $PORT
CMD uvicorn web.api.main:app --host 0.0.0.0 --port ${PORT}
