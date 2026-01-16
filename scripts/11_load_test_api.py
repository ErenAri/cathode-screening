#!/usr/bin/env python3
"""
Simple load test for the /predict endpoint.

Example:
    python scripts/11_load_test_api.py --url http://127.0.0.1:8001/predict --cif test.cif --requests 50 --concurrency 5
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def _build_multipart(field_name: str, filename: str, content: bytes) -> Tuple[bytes, str]:
    boundary = f"----cathode-{uuid.uuid4().hex}"
    lines = []
    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    lines.append(b"Content-Type: text/plain\r\n\r\n")
    lines.append(content)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _post_predict(url: str, cif_path: Path, api_key: str | None) -> Dict[str, object]:
    content = cif_path.read_bytes()
    body, content_type = _build_multipart("cif_file", cif_path.name, content)
    headers = {"Content-Type": content_type}
    if api_key:
        headers["X-API-Key"] = api_key

    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": resp.status,
                "latency_ms": latency_ms,
                "response": payload,
            }
    except HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "status": exc.code,
            "latency_ms": latency_ms,
            "error": exc.read().decode("utf-8"),
        }
    except URLError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "status": 0,
            "latency_ms": latency_ms,
            "error": str(exc),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Load test /predict endpoint")
    ap.add_argument("--url", required=True, help="Predict endpoint URL")
    ap.add_argument("--cif", required=True, help="Path to CIF file")
    ap.add_argument("--requests", type=int, default=20, help="Total requests")
    ap.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    ap.add_argument("--api-key", default=None, help="API key for auth")
    args = ap.parse_args()

    cif_path = Path(args.cif)
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    latencies = []
    statuses = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(_post_predict, args.url, cif_path, args.api_key)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            latency = result.get("latency_ms")
            if latency is not None:
                latencies.append(float(latency))
            status = int(result.get("status", 0))
            statuses[status] = statuses.get(status, 0) + 1
            if status >= 400 or status == 0:
                errors += 1

    latencies.sort()
    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "status_counts": statuses,
        "error_count": errors,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
