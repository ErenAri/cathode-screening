"""
Download checkpoints from GCS to local disk.
Used inside Docker container for Vertex AI evaluation jobs.
"""
import argparse
from pathlib import Path
from google.cloud import storage


def download_prefix(bucket_name: str, prefix: str, dest: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    normalized_prefix = prefix.rstrip("/") + "/"
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=normalized_prefix))
    print(f"Found {len(blobs)} files under gs://{bucket_name}/{normalized_prefix}")

    downloaded = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(normalized_prefix):]
        local_path = dest_path / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {blob.name} -> {local_path}")
        blob.download_to_filename(str(local_path))
        downloaded += 1

    print(f"Download complete: {downloaded} files.")


def main():
    parser = argparse.ArgumentParser(description="Download GCS checkpoints")
    parser.add_argument("--bucket", default="cathode-screening-training")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    download_prefix(args.bucket, args.prefix, args.dest)


if __name__ == "__main__":
    main()
