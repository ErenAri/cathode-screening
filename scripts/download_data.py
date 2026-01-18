"""
Script to download training data from Cloud Storage to local disk.
Used inside Docker container for Vertex AI training.
"""
import os
import shutil
from pathlib import Path
from google.cloud import storage

BUCKET_NAME = "cathode-screening-training"
GCS_DATA_PREFIX = "data"
LOCAL_DATA_DIR = "/app/data"

def download_blob(bucket, blob, local_path):
    """Download a single blob to local path."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    # print(f"Downloaded {blob.name} to {local_path}")

def download_data():
    print(f"Connecting to GCS bucket: {BUCKET_NAME}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    print(f"Downloading data from gs://{BUCKET_NAME}/{GCS_DATA_PREFIX} to {LOCAL_DATA_DIR}...")
    
    # List all blobs
    blobs = list(bucket.list_blobs(prefix=GCS_DATA_PREFIX))
    print(f"Found {len(blobs)} files.")
    
    for blob in blobs:
        # Construct local path
        rel_path = str(blob.name) # e.g. data/external/foo.parquet
        # We want to map gs://.../data/... to /app/data/...
        # rel_path contains "data/" prefix already if GCS_DATA_PREFIX is "data"
        
        local_path = os.path.join("/app", rel_path) # /app/data/...
        
        # Skip directories
        if blob.name.endswith('/'):
            continue
            
        # Download if not exists or size differs (simple check)
        if not os.path.exists(local_path):
            print(f"Downloading {blob.name} ({blob.size / 1e6:.1f} MB)...")
            download_blob(bucket, blob, local_path)
        else:
            print(f"Skipping {blob.name} (already exists)")
            
    print("Download complete!")

if __name__ == "__main__":
    download_data()
