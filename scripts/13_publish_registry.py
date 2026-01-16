#!/usr/bin/env python3
"""
Publish artifact manifest + signature to a registry endpoint.

Example:
    python scripts/13_publish_registry.py --manifest data/artifacts/manifest.json \
        --signature data/artifacts/manifest.sig --registry-url https://registry.example.com/v1/artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest


def _load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish artifact manifest to registry")
    ap.add_argument("--manifest", required=True, help="Manifest JSON path")
    ap.add_argument("--signature", required=True, help="Manifest signature path")
    ap.add_argument("--registry-url", required=True, help="Registry endpoint URL")
    ap.add_argument("--token", default=None, help="Bearer token for registry")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    signature_path = Path(args.signature)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not signature_path.exists():
        raise FileNotFoundError(f"Signature not found: {signature_path}")

    manifest = json.loads(_load_file(manifest_path))
    signature = _load_file(signature_path).strip()

    payload = json.dumps(
        {
            "manifest": manifest,
            "signature": signature,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    req = urlrequest.Request(args.registry_url, data=payload, headers=headers, method="POST")
    with urlrequest.urlopen(req, timeout=args.timeout) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main()
