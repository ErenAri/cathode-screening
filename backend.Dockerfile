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

# Copy requirements
COPY web/api/requirements.txt /app/requirements.txt

# Install other dependencies
# Ensure CPU-only torch wheels are used if torch is in requirements.txt
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and core package
COPY web /app/web
COPY src /app/src
COPY configs /app/configs

# Copy artifacts and database predictions
COPY data/artifacts /app/data/artifacts
COPY data/predictions /app/data/predictions

# Create unprivileged user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run sets PORT env var, defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Run the application
# We use shell form to expand $PORT
CMD uvicorn web.api.main:app --host 0.0.0.0 --port ${PORT}
