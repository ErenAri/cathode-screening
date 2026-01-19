"""
Script to upload H100 training results from local disk to Cloud Storage.
Used inside Docker container for Vertex AI training.
"""
import os
import glob
from google.cloud import storage

BUCKET_NAME = "cathode-screening-training"
LOCAL_CHECKPOINT_DIR = "checkpoints/gcp_h100"
GCS_CHECKPOINT_PREFIX = "checkpoints/gcp_h100"

def upload_directory():
    print(f"Connecting to GCS bucket: {BUCKET_NAME}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    print(f"Uploading results from {LOCAL_CHECKPOINT_DIR} to gs://{BUCKET_NAME}/{GCS_CHECKPOINT_PREFIX}...")
    
    files_uploaded = 0
    if not os.path.exists(LOCAL_CHECKPOINT_DIR):
        print(f"Warning: Directory {LOCAL_CHECKPOINT_DIR} does not exist. Nothing to upload.")
        return

    for root, dirs, files in os.walk(LOCAL_CHECKPOINT_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, start="checkpoints")
            blob_path = os.path.join("checkpoints", rel_path).replace("\\", "/")
            
            print(f"Uploading {local_path} -> {blob_path}...")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            files_uploaded += 1
            
    print(f"Upload complete! {files_uploaded} files uploaded.")

if __name__ == "__main__":
    upload_directory()
