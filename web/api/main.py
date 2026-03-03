"""
CathodeScreen API - FastAPI backend for cathode materials screening.

Provides endpoints for:
- CIF structure upload and prediction
- Batch predictions
- Model information
"""

from pathlib import Path
from typing import List, Optional
import asyncio
import csv
import mimetypes
import subprocess
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
import secrets
import sys
import tempfile
import time
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Security, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from cathode_screening.discovery.state import CampaignState, STAGES

# Ensure core package is importable when running from web/api
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = REPO_ROOT / "src"
REPORTS_DIR = REPO_ROOT / "reports"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

SCREENING_PROOF_FILES: dict[str, Path] = {
    "screening_decision_summary": REPORTS_DIR / "screening_decision_summary.txt",
    "screening_decision_provisional": REPORTS_DIR / "screening_decision_provisional.csv",
    "screening_execution_compact": REPORTS_DIR / "screening_execution_compact.csv",
    "screening_execution_accept": REPORTS_DIR / "screening_execution_accept.csv",
    "screening_execution_must_resolve_top20": REPORTS_DIR / "screening_execution_must_resolve_top20.csv",
    "qe_final_status_estimated": REPORTS_DIR / "qe_run3_run4_run5_estimated_status.csv",
    "qe_ranked_final_estimated": REPORTS_DIR / "dft_batch_jarvis_50_mix_final_ranked_estimated.csv",
    "grounded_win_h100_ehull_ens_v1": REPORTS_DIR / "grounded_win_h100_ehull_ens_v1.json",
    "grounded_win_oqmd_ens_v1": REPORTS_DIR / "grounded_win_oqmd_ens_v1.json",
    "grounded_win_jarvis_ens_v1": REPORTS_DIR / "grounded_win_jarvis_ens_v1.json",
}

from cathode_screening.inference.artifact_manifest import load_manifest
from cathode_screening.inference.cathode_guardrails import (
    evaluate_li_cathode_composition,
)
from cathode_screening.monitoring.audit_trail import (
    AuditTrail,
    compute_input_hash,
    get_audit_trail,
)
from cathode_screening.monitoring.metrics import MetricsCollector
from cathode_screening.monitoring.rate_limit import RateLimiter

# Optional secrets bootstrap before reading env config.
def _parse_kv_lines(raw: str) -> dict:
    values = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _bootstrap_secrets() -> None:
    secret_file = os.getenv("CATHODE_SECRET_FILE")
    secret_command = os.getenv("CATHODE_SECRET_COMMAND")
    if not secret_file and not secret_command:
        return

    payload = {}
    if secret_file:
        path = Path(secret_file)
        if not path.exists():
            raise RuntimeError(f"CATHODE_SECRET_FILE not found: {path}")
        payload.update(_parse_kv_lines(path.read_text(encoding="utf-8")))
    if secret_command:
        result = subprocess.run(
            secret_command.split(),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CATHODE_SECRET_COMMAND failed: {result.stderr.strip()}"
            )
        payload.update(_parse_kv_lines(result.stdout))

    for key, value in payload.items():
        if key and key not in os.environ:
            os.environ[key] = value


_bootstrap_secrets()

# Security Configuration
API_KEY_NAME = "X-API-Key"
REQUEST_ID_HEADER = "X-Request-ID"
ENVIRONMENT = os.getenv("CATHODE_ENV", "production").strip().lower()
AUTH_DEFAULT = "true" if ENVIRONMENT == "production" else "false"
AUTH_ENABLED = os.getenv("CATHODE_AUTH_ENABLED", AUTH_DEFAULT).strip().lower() in {
    "1",
    "true",
    "yes",
}

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _split_values(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    values = []
    for part in raw.replace(";", ",").split(","):
        value = part.strip()
        if value:
            values.append(value)
    return values

def _load_values_from_file(path_value: Optional[str]) -> List[str]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError(f"Secret file not found: {path}")
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(line)
    return values

API_KEYS = set(_split_values(os.getenv("CATHODE_API_KEYS")))
API_KEYS.update(_load_values_from_file(os.getenv("CATHODE_API_KEYS_FILE")))
API_KEYS.update(_load_values_from_file(os.getenv("CATHODE_API_KEY_FILE")))
single_key = os.getenv("CATHODE_API_KEY")
if single_key:
    API_KEYS.add(single_key)

API_KEY_HASHES = set(_split_values(os.getenv("CATHODE_API_KEY_HASHES")))
API_KEY_HASHES.update(_load_values_from_file(os.getenv("CATHODE_API_KEY_HASHES_FILE")))

if AUTH_ENABLED and not API_KEYS and not API_KEY_HASHES:
    raise RuntimeError("CATHODE_API_KEY(S) or CATHODE_API_KEY_HASHES must be set when auth is enabled")

LOG_FORMAT = os.getenv("CATHODE_LOG_FORMAT", "plain").strip().lower()
LOG_PREDICTIONS = os.getenv("CATHODE_LOG_PREDICTIONS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
LOG_REQUESTS = _env_bool("CATHODE_LOG_REQUESTS", ENVIRONMENT == "production")
METRICS_ENABLED = os.getenv("CATHODE_METRICS_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
RATE_LIMIT_PER_MINUTE = _env_int("CATHODE_RATE_LIMIT_PER_MINUTE", 60)
RATE_LIMIT_WINDOW_SECONDS = _env_int("CATHODE_RATE_LIMIT_WINDOW_SECONDS", 60)
MAX_FILE_BYTES = _env_int("CATHODE_MAX_FILE_BYTES", 2_000_000)
MIN_FILE_BYTES = _env_int("CATHODE_MIN_FILE_BYTES", 10)
MAX_BATCH_SIZE = _env_int("CATHODE_MAX_BATCH_SIZE", 25)
MAX_ATOMS = _env_int("CATHODE_MAX_ATOMS", 512)
STRICT_STARTUP = _env_bool("CATHODE_STRICT_STARTUP", ENVIRONMENT == "production")
REQUIRE_CALIBRATION = _env_bool(
    "CATHODE_REQUIRE_CALIBRATION", ENVIRONMENT == "production"
)
REQUIRE_MANIFEST_SIGNATURE = _env_bool(
    "CATHODE_REQUIRE_MANIFEST_SIGNATURE", ENVIRONMENT == "production"
)
TRUST_PROXY = _env_bool("CATHODE_TRUST_PROXY", False)
TRUST_PROXY_HOPS = max(_env_int("CATHODE_TRUST_PROXY_HOPS", 1), 0)
IP_ALLOWLIST = set(_split_values(os.getenv("CATHODE_IP_ALLOWLIST")))
FORCE_HTTPS = _env_bool("CATHODE_FORCE_HTTPS", ENVIRONMENT == "production")
SECURITY_HEADERS = _env_bool(
    "CATHODE_SECURITY_HEADERS", ENVIRONMENT == "production"
)
PROMETHEUS_ENABLED = _env_bool("CATHODE_PROMETHEUS_ENABLED", False)
OTEL_ENABLED = _env_bool("CATHODE_OTEL_ENABLED", False)
OTEL_SERVICE_NAME = os.getenv("CATHODE_OTEL_SERVICE_NAME", "cathode-screening")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("CATHODE_OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
OTEL_EXPORTER_OTLP_INSECURE = _env_bool("CATHODE_OTEL_EXPORTER_OTLP_INSECURE", True)
MAX_CONCURRENT_REQUESTS = _env_int("CATHODE_MAX_CONCURRENT_REQUESTS", 0)
CONCURRENCY_TIMEOUT_SECONDS = _env_int("CATHODE_CONCURRENCY_TIMEOUT_SECONDS", 5)
REQUIRE_CIF_EXT = os.getenv("CATHODE_REQUIRE_CIF_EXT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIRE_CATHODE_COMPOSITION = _env_bool(
    "CATHODE_REQUIRE_CATHODE_COMPOSITION", ENVIRONMENT == "production"
)
RAW_CONTENT_TYPES = os.getenv(
    "CATHODE_ALLOWED_CONTENT_TYPES",
    "chemical/x-cif,text/plain,application/octet-stream,application/x-cif,text/cif",
)
ALLOWED_CONTENT_TYPES = {t.strip() for t in RAW_CONTENT_TYPES.split(",") if t.strip()}
ALLOW_UNSAFE_TORCH_LOAD = _env_bool("CATHODE_ALLOW_UNSAFE_TORCH_LOAD", False)
ALLOW_INSECURE_PROD = _env_bool("CATHODE_ALLOW_INSECURE_PROD", False)


def _validate_security_configuration() -> None:
    """Fail fast on insecure production defaults unless explicitly overridden."""
    if ENVIRONMENT != "production" or ALLOW_INSECURE_PROD:
        return

    insecure_reasons = []
    if not AUTH_ENABLED:
        insecure_reasons.append("CATHODE_AUTH_ENABLED must be true in production")
    if not FORCE_HTTPS:
        insecure_reasons.append("CATHODE_FORCE_HTTPS must be true in production")
    if ALLOW_UNSAFE_TORCH_LOAD:
        insecure_reasons.append(
            "CATHODE_ALLOW_UNSAFE_TORCH_LOAD must be false in production"
        )
    if not REQUIRE_MANIFEST_SIGNATURE:
        insecure_reasons.append(
            "CATHODE_REQUIRE_MANIFEST_SIGNATURE must be true in production"
        )
    if REQUIRE_MANIFEST_SIGNATURE and not os.getenv("CATHODE_MANIFEST_HMAC_KEY"):
        insecure_reasons.append(
            "CATHODE_MANIFEST_HMAC_KEY must be set when signature verification is required"
        )

    if insecure_reasons:
        detail = "; ".join(insecure_reasons)
        raise RuntimeError(
            "Insecure production configuration detected: "
            f"{detail}. Set CATHODE_ALLOW_INSECURE_PROD=true only for emergency overrides."
        )


_validate_security_configuration()

logger = logging.getLogger("cathode_screening")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

metrics = MetricsCollector() if METRICS_ENABLED else None
rate_limiter = (
    RateLimiter(RATE_LIMIT_PER_MINUTE, window_seconds=RATE_LIMIT_WINDOW_SECONDS)
    if RATE_LIMIT_PER_MINUTE > 0
    else None
)

_startup_error: Optional[str] = None
_startup_complete = False
_manifest_cache: Optional[dict] = None
_inference_semaphore = (
    asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) if MAX_CONCURRENT_REQUESTS > 0 else None
)


def _log_event(payload: dict) -> None:
    if LOG_FORMAT == "json":
        logger.info(json.dumps(payload, separators=(",", ":")))
        return
    event = payload.get("event", "event")
    logger.info("%s %s", event, payload)


def _client_id(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
            if ips:
                if TRUST_PROXY_HOPS > 0 and len(ips) > TRUST_PROXY_HOPS:
                    return ips[-(TRUST_PROXY_HOPS + 1)]
                return ips[0]
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request, cost: int = 1) -> None:
    if rate_limiter is None:
        return
    key = _client_id(request)
    if not rate_limiter.allow(key, cost=cost):
        _log_event({"event": "rate_limit", "client": key, "cost": cost})
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

def _error_detail(code: str, message: str, **extra: object) -> dict:
    detail = {"error": code, "message": message}
    for key, value in extra.items():
        if value is not None:
            detail[key] = value
    return detail


@asynccontextmanager
async def _inference_slot(request_id: Optional[str] = None):
    if _inference_semaphore is None:
        yield
        return
    try:
        await asyncio.wait_for(
            _inference_semaphore.acquire(),
            timeout=CONCURRENCY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail=_error_detail(
                "concurrency_limit",
                "Server is busy, try again later",
                request_id=request_id,
            ),
        ) from exc
    try:
        yield
    finally:
        _inference_semaphore.release()


def _load_manifest_summary() -> Optional[dict]:
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache

    artifact_dir = Path(os.getenv("CATHODE_ARTIFACTS_DIR", "data/artifacts"))
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        logger.warning("Failed to load artifact manifest: %s", exc)
        return None

    total_bytes = sum(f.size_bytes for f in manifest.files)
    _manifest_cache = {
        "generated_at": manifest.generated_at,
        "git_commit": manifest.git_commit,
        "file_count": len(manifest.files),
        "total_bytes": total_bytes,
        "summary": manifest.summary,
    }
    return _manifest_cache


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_valid_api_key(candidate: str) -> bool:
    for key in API_KEYS:
        if secrets.compare_digest(candidate, key):
            return True
    if API_KEY_HASHES:
        digest = _hash_key(candidate)
        for key_hash in API_KEY_HASHES:
            if secrets.compare_digest(digest, key_hash):
                return True
    return False


def _extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _format_prometheus(snapshot: dict) -> str:
    lines: List[str] = []
    seen: set[str] = set()

    def add_sample(
        name: str,
        value: Optional[float],
        labels: Optional[dict] = None,
        mtype: str = "gauge",
        help_text: Optional[str] = None,
    ) -> None:
        if value is None:
            return
        if name not in seen:
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            seen.add(name)
        label_str = ""
        if labels:
            label_parts = [f'{k}="{v}"' for k, v in labels.items()]
            label_str = "{" + ",".join(label_parts) + "}"
        lines.append(f"{name}{label_str} {value}")

    def add_stats(prefix: str, stats: dict) -> None:
        count = stats.get("count")
        mean = stats.get("mean")
        if count is not None:
            add_sample(
                f"{prefix}_count",
                float(count),
                mtype="counter",
                help_text=f"{prefix} sample count",
            )
            if mean is not None:
                add_sample(
                    f"{prefix}_sum",
                    float(mean) * float(count),
                    mtype="counter",
                    help_text=f"{prefix} sample sum",
                )
        add_sample(f"{prefix}_min", stats.get("min"), help_text=f"{prefix} minimum")
        add_sample(f"{prefix}_max", stats.get("max"), help_text=f"{prefix} maximum")
        add_sample(f"{prefix}_mean", stats.get("mean"), help_text=f"{prefix} mean")
        add_sample(f"{prefix}_std", stats.get("std"), help_text=f"{prefix} stddev")

    add_sample(
        "cathode_uptime_seconds",
        snapshot.get("uptime_s"),
        help_text="Service uptime in seconds",
    )
    add_sample(
        "cathode_requests_total",
        snapshot.get("request_count"),
        mtype="counter",
        help_text="Total requests recorded by inference endpoints",
    )
    add_sample(
        "cathode_errors_total",
        snapshot.get("error_count"),
        mtype="counter",
        help_text="Total errors recorded by inference endpoints",
    )

    for decision, count in snapshot.get("decision_counts", {}).items():
        add_sample(
            "cathode_decisions_total",
            count,
            labels={"decision": decision},
            mtype="counter",
            help_text="Decision counts",
        )

    for mode, count in snapshot.get("mode_counts", {}).items():
        add_sample(
            "cathode_modes_total",
            count,
            labels={"mode": mode},
            mtype="counter",
            help_text="Decision mode counts",
        )

    add_sample(
        "cathode_ood_flag_total",
        snapshot.get("ood_flag_count"),
        mtype="counter",
        help_text="Total OOD-flagged predictions",
    )

    add_stats("cathode_latency_seconds", snapshot.get("latency_s", {}))
    add_stats("cathode_ood_score", snapshot.get("ood_score", {}))
    add_stats("cathode_uncertainty_epistemic", snapshot.get("uncertainty_epistemic", {}))
    add_stats("cathode_ehull_pred", snapshot.get("ehull_pred", {}))

    return "\n".join(lines) + "\n"


def _validate_upload(cif_file: UploadFile, content: bytes) -> None:
    filename = cif_file.filename or "uploaded"
    if len(content) < MIN_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "file_too_small",
                "CIF file is empty or too small",
                filename=filename,
                min_bytes=MIN_FILE_BYTES,
            ),
        )
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=_error_detail(
                "file_too_large",
                "CIF file too large",
                filename=filename,
                max_bytes=MAX_FILE_BYTES,
            ),
        )
    if REQUIRE_CIF_EXT and cif_file.filename and not cif_file.filename.lower().endswith(".cif"):
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "invalid_extension",
                "File extension must be .cif",
                filename=filename,
            ),
        )
    if cif_file.content_type and cif_file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=_error_detail(
                "unsupported_content_type",
                "Unsupported CIF content type",
                filename=filename,
                content_type=cif_file.content_type,
            ),
        )


def _validate_cathode_scope(structure, filename: str) -> None:
    if not REQUIRE_CATHODE_COMPOSITION:
        return
    check = evaluate_li_cathode_composition(structure)
    if check.is_valid:
        return
    raise HTTPException(
        status_code=400,
        detail=_error_detail(
            "invalid_cathode_composition",
            "Input composition is not a supported Li-ion cathode host",
            filename=filename,
            formula=check.formula,
            elements=list(check.elements),
            transition_metals=list(check.transition_metals),
            anion_framework=check.anion_framework,
            reasons=list(check.reasons),
            requirements={
                "lithium_present": True,
                "transition_metal_present": True,
                "supported_anion_framework": ["O", "S", "Se", "F", "Cl", "Br", "I", "N", "PO4", "SiO4", "BO3"],
                "multi_element_composition": True,
            },
        ),
    )


async def _parse_structure(cif_file: UploadFile):
    content = await cif_file.read()
    _validate_upload(cif_file, content)
    filename = cif_file.filename or "uploaded"
    try:
        cif_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "decode_error",
                "CIF must be utf-8 encoded",
                filename=filename,
            ),
        ) from exc

    from pymatgen.core import Structure

    try:
        structure = Structure.from_str(cif_text, fmt="cif")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "parse_error",
                "Failed to parse CIF",
                filename=filename,
                reason=str(exc),
            ),
        ) from exc
    if len(structure) > MAX_ATOMS:
        raise HTTPException(
            status_code=413,
            detail=_error_detail(
                "structure_too_large",
                "Structure exceeds max atom limit",
                filename=filename,
                max_atoms=MAX_ATOMS,
                atom_count=len(structure),
            ),
        )
    if len(structure) == 0:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "empty_structure",
                "Structure contains no sites",
                filename=filename,
            ),
        )
    _validate_cathode_scope(structure, filename)
    return structure


async def _predict_structure_core(cif_file: UploadFile):
    from .inference import get_predictor

    # Read content for hashing before parsing resets the stream
    await cif_file.seek(0)
    raw_content = await cif_file.read()
    await cif_file.seek(0)
    input_hash = compute_input_hash(raw_content)

    structure = await _parse_structure(cif_file)

    predictor = get_predictor()
    result = predictor.predict_structure(
        structure,
        material_id=cif_file.filename or "uploaded",
    )
    return result, input_hash

async def get_api_key(
    request: Request,
    api_key_header: str = Security(api_key_header),
):
    if not AUTH_ENABLED:
        return None
    client_id = _client_id(request)
    if IP_ALLOWLIST and client_id not in IP_ALLOWLIST:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_error_detail(
                "ip_not_allowed",
                "Client IP not allowed",
                client_id=client_id,
            ),
        )
    candidate = api_key_header
    if not candidate:
        candidate = _extract_bearer_token(request.headers.get("authorization"))
    if candidate and _is_valid_api_key(candidate):
        return candidate
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=_error_detail(
            "invalid_api_key",
            "Could not validate credentials",
        ),
    )

app = FastAPI(
    title="CathodeScreen API",
    description="AI-powered screening of battery cathode materials",
    version="1.0.0",
)

def _setup_otel() -> None:
    if not OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry enabled but dependencies are missing")
        return

    resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter = (
        OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)
        if OTEL_EXPORTER_OTLP_ENDPOINT
        else OTLPSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    LoggingInstrumentor().instrument(set_logging_format=True)

if FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

if SECURITY_HEADERS:
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'"
        )
        if FORCE_HTTPS or request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains",
            )
        return response

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if LOG_REQUESTS:
            _log_event(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }
            )
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    if LOG_REQUESTS:
        _log_event(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            }
        )
    return response

@app.on_event("startup")
async def startup_event() -> None:
    global _startup_error, _startup_complete
    try:
        from .inference import get_predictor, get_model_type

        predictor = get_predictor()
        
        # Check models based on model type
        model_type = get_model_type()
        if model_type == "mace":
            # MACEDecisionService has .models directly
            if not predictor.models:
                raise RuntimeError("No MACE ensemble models loaded")
        elif model_type == "chgnet":
            # CHGNet adapter uses screener.models
            if not predictor.screener.models:
                raise RuntimeError("No CHGNet ensemble models loaded")
        else:
            # CGCNN uses predictor.models
            if not predictor.predictor.models:
                raise RuntimeError("No ensemble models loaded")
            if REQUIRE_CALIBRATION and predictor.predictor.calibrator is None:
                raise RuntimeError("Calibration parameters not loaded")
        
        _startup_complete = True
        _startup_error = None
    except Exception as exc:
        _startup_error = str(exc)
        _startup_complete = False
        logger.error("Startup check failed: %s", exc)
        if STRICT_STARTUP:
            raise

# CORS for frontend
raw_origins = os.getenv("CATHODE_CORS_ORIGINS", "http://localhost:3000")
cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
allow_credentials = os.getenv("CATHODE_CORS_ALLOW_CREDENTIALS", "false").strip().lower() in {"1", "true", "yes"}
if "*" in cors_origins:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

_setup_otel()


class CathodePropertiesResult(BaseModel):
    """Analytical cathode material properties."""
    gravimetric_capacity_mAhg: Optional[float] = None
    volumetric_capacity_mAhcm3: Optional[float] = None
    avg_voltage_V: Optional[float] = None
    voltage_confidence: Optional[str] = None
    gravimetric_energy_Whkg: Optional[float] = None
    volumetric_energy_WhL: Optional[float] = None
    li_count: Optional[float] = None
    li_fraction: Optional[float] = None
    n_extractable_li: Optional[float] = None
    density_gcm3: Optional[float] = None
    tm_elements: Optional[List[str]] = None
    anion_framework: Optional[str] = None
    composite_score: Optional[float] = None
    score_breakdown: Optional[dict] = None


class PredictionResult(BaseModel):
    """Single prediction result."""
    material_id: str
    pred_ehull: float
    p_stable: float
    uncertainty: str  # "Low", "Medium", "High"
    action: str  # "DFT", "HOLD", "SKIP"
    confidence_interval: tuple[float, float]
    cathode_properties: Optional[CathodePropertiesResult] = None


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    success: bool
    prediction: Optional[PredictionResult] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    success: bool
    predictions: List[PredictionResult] = Field(default_factory=list)
    errors: List[dict] = Field(default_factory=list)
    n_processed: int = 0
    n_errors: int = 0


class ModelInfo(BaseModel):
    """Model information."""
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_type: str
    ensemble_size: int
    training_data: str
    daf_at_10: float
    version: str
    artifact_manifest: Optional[dict] = None


# In-memory model (will be loaded on startup)
_model = None


# Uncertainty classification thresholds (eV/atom)
UNCERTAINTY_LOW_THRESHOLD = 0.05
UNCERTAINTY_MED_THRESHOLD = 0.15

# Action policy thresholds (eV/atom)
ACTION_EHULL_DFT = 0.08
ACTION_EHULL_DFT_CLASSIFIER = 0.10
ACTION_EHULL_HOLD = 0.15
ACTION_P_STABLE_STRONG = 0.70
ACTION_P_STABLE_MODERATE = 0.50


def classify_uncertainty(std: float) -> str:
    """Classify uncertainty level."""
    if std < UNCERTAINTY_LOW_THRESHOLD:
        return "Low"
    elif std < UNCERTAINTY_MED_THRESHOLD:
        return "Medium"
    return "High"


def get_action(p_stable: float, unc: str, pred: float) -> str:
    if (unc == "Low" and pred < ACTION_EHULL_DFT) or (
        p_stable > ACTION_P_STABLE_STRONG and unc == "Low" and pred < ACTION_EHULL_DFT_CLASSIFIER
    ):
        return "DFT"
    elif p_stable > ACTION_P_STABLE_MODERATE or pred < ACTION_EHULL_HOLD:
        return "HOLD"
    return "SKIP"


def decision_to_action(decision: str) -> str:
    mapping = {
        "KEEP": "DFT",
        "MAYBE": "HOLD",
        "KILL": "SKIP",
    }
    return mapping.get(decision, "HOLD")


def _to_int(value: Optional[str], default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


CAMPAIGNS_DIR = REPO_ROOT / "data" / "campaigns"


def _campaign_path(name: str) -> Path:
    """Return the JSON file path for a campaign."""
    return CAMPAIGNS_DIR / f"{name}.json"


def _list_campaigns() -> list[CampaignState]:
    """Load all campaign states from data/campaigns/."""
    if not CAMPAIGNS_DIR.exists():
        return []
    campaigns = []
    for p in sorted(CAMPAIGNS_DIR.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            campaigns.append(CampaignState.load(p))
        except Exception as exc:
            logger.warning("Failed to load campaign %s: %s", p.name, exc)
    return campaigns


def _parse_key_value_summary(path: Path) -> tuple[str, dict[str, str]]:
    if not path.exists():
        return "", {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "", {}
    title = lines[0]
    values: dict[str, str] = {}
    for line in lines[1:]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return title, values


def _relative_repo_path(path: Path) -> str:
    try:
        rel = path.relative_to(REPO_ROOT)
        return rel.as_posix()
    except ValueError:
        return str(path)


def _proof_descriptor(proof_id: str, path: Path) -> dict:
    exists = path.exists() and path.is_file()
    size_bytes = path.stat().st_size if exists else None
    mtime_utc = (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        if exists
        else None
    )
    return {
        "id": proof_id,
        "relative_path": _relative_repo_path(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "mtime_utc": mtime_utc,
        "download_url": f"/screening/proof/{proof_id}" if exists else None,
    }


@app.get("/metrics")
async def get_metrics(api_key: str = Security(get_api_key)):
    """Return in-memory metrics for monitoring and drift checks."""
    if not METRICS_ENABLED or metrics is None:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return {"success": True, "metrics": metrics.snapshot()}


@app.get("/metrics/prometheus")
async def get_metrics_prometheus(api_key: str = Security(get_api_key)):
    """Return Prometheus-formatted metrics."""
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=404, detail="Prometheus metrics disabled")
    if not METRICS_ENABLED or metrics is None:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    content = _format_prometheus(metrics.snapshot())
    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "CathodeScreen API", "version": "1.0.0"}

@app.get("/ready")
async def ready():
    if _startup_error:
        raise HTTPException(
            status_code=503,
            detail={"error": "startup_failed", "message": _startup_error},
        )
    if not _startup_complete:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "startup_incomplete",
                "message": "Startup checks not completed",
            },
        )
    return {"status": "ready"}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    from .inference import get_model_type
    model_type = get_model_type()
    model_names = {
        "mace": ("MACE-MP-0 Ensemble", "MACE-MP-0 Fine-tuned (5 members)"),
        "chgnet": ("CHGNet-Ensemble", "Crystal Hamiltonian Graph Neural Network"),
        "cgcnn": ("CGCNN-Ensemble", "Crystal Graph Convolutional Neural Network"),
    }
    name, desc = model_names.get(model_type, ("Unknown", "Unknown"))
    return ModelInfo(
        model_name=name,
        model_type=desc,
        ensemble_size=5,
        training_data="Materials Project cathodes (SOAP-LOCO split)",
        daf_at_10=2.12,
        version="2.0.0",
        artifact_manifest=_load_manifest_summary(),
    )


@app.get("/screening/provisional")
async def get_screening_provisional(api_key: str = Security(get_api_key)):
    """Return provisional execution lists and proof artifacts for the JARVIS-50 QE campaign."""
    summary_title, summary_fields = _parse_key_value_summary(
        SCREENING_PROOF_FILES["screening_decision_summary"]
    )
    decision_rows = _read_csv_rows(SCREENING_PROOF_FILES["screening_decision_provisional"])
    compact_rows = _read_csv_rows(SCREENING_PROOF_FILES["screening_execution_compact"])
    accept_rows = _read_csv_rows(SCREENING_PROOF_FILES["screening_execution_accept"])
    must_resolve_rows = _read_csv_rows(
        SCREENING_PROOF_FILES["screening_execution_must_resolve_top20"]
    )

    if not decision_rows and not compact_rows:
        raise HTTPException(
            status_code=500,
            detail="Provisional screening artifacts not found on server",
        )

    decision_counts = {
        "accept": sum(1 for row in decision_rows if row.get("decision") == "accept"),
        "hold": sum(1 for row in decision_rows if row.get("decision") == "hold"),
        "unknown": sum(1 for row in decision_rows if row.get("decision") == "unknown"),
    }
    top20_unresolved = sum(
        1
        for row in decision_rows
        if _to_int(row.get("rank"), default=999999) <= 20
        and row.get("qe_final_state_est", "").strip().lower() != "done"
    )

    if not accept_rows:
        accept_rows = [
            row
            for row in compact_rows
            if row.get("action", "").strip().lower() == "screen_now"
        ]
    if not must_resolve_rows:
        must_resolve_rows = [
            row
            for row in compact_rows
            if row.get("action", "").strip().lower() == "resolve_qe_first"
        ]

    proofs = [
        _proof_descriptor(proof_id, path)
        for proof_id, path in SCREENING_PROOF_FILES.items()
    ]

    grounded_win = {
        "h100_ehull_ens_v1": _read_json_dict(
            SCREENING_PROOF_FILES["grounded_win_h100_ehull_ens_v1"]
        ),
        "oqmd_ens_v1": _read_json_dict(SCREENING_PROOF_FILES["grounded_win_oqmd_ens_v1"]),
        "jarvis_ens_v1": _read_json_dict(SCREENING_PROOF_FILES["grounded_win_jarvis_ens_v1"]),
    }

    counts = {
        "total_candidates": _to_int(
            summary_fields.get("total_candidates"), default=len(decision_rows)
        ),
        "accept": _to_int(summary_fields.get("accept"), default=decision_counts["accept"]),
        "hold": _to_int(summary_fields.get("hold"), default=decision_counts["hold"]),
        "unknown": _to_int(summary_fields.get("unknown"), default=decision_counts["unknown"]),
        "top20_unresolved": _to_int(
            summary_fields.get("top20_unresolved"), default=top20_unresolved
        ),
    }

    return {
        "success": True,
        "mode": "provisional",
        "summary_title": summary_title,
        "summary_fields": summary_fields,
        "counts": counts,
        "screening": {
            "screen_now": accept_rows,
            "resolve_qe_first": must_resolve_rows,
            "compact": compact_rows,
        },
        "grounded_win": grounded_win,
        "proofs": proofs,
    }


@app.get("/screening/proof/{proof_id}")
async def download_screening_proof(proof_id: str, api_key: str = Security(get_api_key)):
    """Download a proof artifact that backs the provisional screening output."""
    path = SCREENING_PROOF_FILES.get(proof_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Proof artifact not found")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Proof artifact missing on server")
    if not path.resolve().is_relative_to(REPORTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/database")
async def get_database(api_key: str = Security(get_api_key)):
    """Get pre-computed predictions from the database."""
    predictions_path = Path("data/predictions/ensemble_soap_loco_test.parquet")
    if not predictions_path.exists():
        raise HTTPException(status_code=500, detail="Predictions file not found on server")

    def _build_database_payload() -> dict:
        import pandas as pd

        df = pd.read_parquet(predictions_path)
        sort_col = "q50" if "q50" in df.columns else "pred_ehull"
        df = df.sort_values(sort_col, ascending=True)

        data = []
        for idx, row in df.iterrows():
            pred_ehull = row.get("q50", row.get("pred_ehull", 0))
            std = row.get("epistemic_std", 0.1)
            p_stable = row.get("p_stable", 0.5)
            unc = classify_uncertainty(float(std))
            action = get_action(float(p_stable), unc, float(pred_ehull))

            data.append(
                {
                    "material_id": row.get("material_id", f"mp-{idx}"),
                    "formula": row.get("formula", ""),
                    "pred_ehull": round(float(pred_ehull), 4),
                    "p_stable": round(float(p_stable), 3),
                    "uncertainty": unc,
                    "action": action,
                    "confidence_interval": (0.0, 0.0),
                }
            )
        return {"success": True, "data": data, "total": len(df)}

    try:
        return await asyncio.to_thread(_build_database_payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Discovery Engine endpoints
# ---------------------------------------------------------------------------

import re

_CAMPAIGN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PREDICTIONS_PARQUET = Path("data/predictions/ensemble_soap_loco_test.parquet")


def _read_predictions_pool_df(parquet_path: Path):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    id_col = "material_id" if "material_id" in df.columns else df.columns[0]
    df[id_col] = df[id_col].astype(str)
    return df, id_col


async def _load_predictions_pool_df(parquet_path: Path = _PREDICTIONS_PARQUET):
    if not parquet_path.exists():
        raise HTTPException(status_code=500, detail="Predictions file not found")
    try:
        return await asyncio.to_thread(_read_predictions_pool_df, parquet_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


from cathode_screening.discovery.ranking import (
    DEFAULT_RANKING_STRATEGY,
    RANKING_STRATEGIES,
    rank_candidates,
    validate_ranking_strategy,
)
from cathode_screening.discovery.async_loop import (
    DiscoveryJobStore,
    simulate_dft_outcomes,
    summarize_outcomes,
    utc_now_iso,
)


def _resolve_ranking_strategy(state: CampaignState, requested: Optional[str]) -> str:
    try:
        return validate_ranking_strategy(requested or state.ranking_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _format_ranked_candidate(row: dict, id_col: str, rank: int, strategy: str) -> dict:
    pred_ehull = float(row.get("q50", row.get("pred_ehull", 0.0)))
    std = float(row.get("epistemic_std", 0.1))
    p_stable = float(row.get("p_stable", 0.5))
    unc = classify_uncertainty(std)
    action = get_action(p_stable, unc, pred_ehull)
    score = float(row.get("acquisition_score", 0.0))

    if score >= 0.75 and action == "DFT":
        priority = "priority"
    elif score >= 0.55 and action != "SKIP":
        priority = "high"
    elif action != "SKIP":
        priority = "medium"
    else:
        priority = "low"

    return {
        "material_id": str(row.get(id_col, "")),
        "formula": str(row.get("formula", "")),
        "pred_ehull": round(pred_ehull, 4),
        "p_stable": round(p_stable, 3),
        "uncertainty": unc,
        "action": action,
        "confidence_interval": (
            round(float(row.get("q10", pred_ehull - 2 * std)), 4),
            round(float(row.get("q90", pred_ehull + 2 * std)), 4),
        ),
        "rank": rank,
        "ranking_strategy": strategy,
        "acquisition_score": round(score, 4),
        "screen_priority": priority,
        "score_breakdown": {
            "stability": round(float(row.get("acq_stability", 0.0)), 4),
            "p_stable": round(float(row.get("acq_p_stable", 0.0)), 4),
            "uncertainty": round(float(row.get("acq_uncertainty", 0.0)), 4),
            "novelty": round(float(row.get("acq_novelty", 0.0)), 4),
            "confidence": round(float(row.get("acq_confidence", 0.0)), 4),
            "ood_penalty": round(float(row.get("acq_ood_penalty", 0.0)), 4),
        },
    }


def _rank_and_format_candidates(pool_df, id_col: str, strategy: str, offset: int = 0):
    ranked_df = rank_candidates(pool_df, id_col=id_col, strategy=strategy)
    candidates = [
        _format_ranked_candidate(row.to_dict(), id_col=id_col, rank=offset + idx + 1, strategy=strategy)
        for idx, (_, row) in enumerate(ranked_df.iterrows())
    ]
    return ranked_df, candidates


_DISCOVERY_JOB_STORE = DiscoveryJobStore()
_DISCOVERY_JOB_TASKS: dict[str, asyncio.Task] = {}


def _active_discovery_job(campaign_name: str):
    for job in reversed(_DISCOVERY_JOB_STORE.list_for_campaign(campaign_name)):
        if job.status in {"queued", "running"}:
            return job
    return None


def _select_ranked_candidates(
    ranked_candidates: list[dict],
    batch_size: int,
    include_hold: bool,
    min_score: Optional[float],
) -> tuple[list[dict], list[str]]:
    allowed_actions = {"DFT"}
    if include_hold:
        allowed_actions.add("HOLD")
    filtered = [c for c in ranked_candidates if c["action"] in allowed_actions]
    if min_score is not None:
        filtered = [c for c in filtered if float(c["acquisition_score"]) >= float(min_score)]
    selected = filtered[: batch_size]
    selected_ids = [c["material_id"] for c in selected]
    return selected, selected_ids


def _mock_retrain_artifact(state: CampaignState, cycle: int, dft_results: dict[str, float]) -> str:
    artifact_dir = REPO_ROOT / "data" / "campaign_artifacts" / state.campaign_name / f"cycle_{cycle:03d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_name": state.campaign_name,
        "cycle": cycle,
        "generated_at": utc_now_iso(),
        "retrain_mode": "mock",
        "dft_results_count": len(dft_results),
        "stable_count": sum(1 for v in dft_results.values() if v < 0.05),
        "mean_ehull": (sum(dft_results.values()) / len(dft_results)) if dft_results else None,
    }
    out = artifact_dir / "mock_model_artifact.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out.as_posix()


async def _run_discovery_job(job_id: str, campaign_name: str, cycle: int, mode: str) -> None:
    try:
        _DISCOVERY_JOB_STORE.update(
            job_id,
            status="running",
            stage="wait_dft",
            started_at=utc_now_iso(),
        )

        path = _campaign_path(campaign_name)
        if not path.exists():
            raise RuntimeError(f"Campaign '{campaign_name}' no longer exists")
        state = CampaignState.load(path)
        if cycle != state.current_cycle:
            raise RuntimeError(
                f"Campaign cycle advanced before job start (job cycle={cycle}, current={state.current_cycle})"
            )

        selected_ids = list(state.current_cycle_record.candidates_selected)
        if not selected_ids:
            raise RuntimeError("No selected candidates for DFT submission")

        # Simulated asynchronous DFT runtime (can be replaced with external runner mode).
        await asyncio.sleep(min(15.0, max(2.0, 0.15 * len(selected_ids))))

        parquet_path = Path("data/predictions") / f"{state.pool_source}.parquet"
        df, id_col = await _load_predictions_pool_df(parquet_path)
        selected_df = df[df[id_col].astype(str).isin(set(selected_ids))].copy()

        dft_results = await asyncio.to_thread(simulate_dft_outcomes, selected_df, id_col=id_col)
        dft_summary = summarize_outcomes(dft_results)

        state = CampaignState.load(path)
        if cycle != state.current_cycle:
            raise RuntimeError(
                f"Campaign cycle changed during job execution (job cycle={cycle}, current={state.current_cycle})"
            )

        state.advance("ingest")
        rec = state.current_cycle_record
        rec.dft_results = dft_results
        rec.n_stable_found = int(dft_summary["stable_count"])
        rec.metrics["dft_mode"] = mode
        rec.metrics["dft_job_started_at"] = _DISCOVERY_JOB_STORE.get(job_id).started_at
        rec.metrics["dft_job_completed_at"] = utc_now_iso()
        rec.metrics["dft_summary"] = dft_summary
        state.record_dft_results(dft_results)
        state.save(path)

        _DISCOVERY_JOB_STORE.update(
            job_id,
            stage="ingest",
            processed_count=len(dft_results),
            stable_found=int(dft_summary["stable_count"]),
            metrics={"dft_summary": dft_summary},
        )

        state = CampaignState.load(path)
        if cycle != state.current_cycle:
            raise RuntimeError(
                f"Campaign cycle changed before retrain stage (job cycle={cycle}, current={state.current_cycle})"
            )
        state.advance("retrain")
        state.save(path)

        # Lightweight retrain artifact for async loop continuity.
        artifact_path = await asyncio.to_thread(_mock_retrain_artifact, state, cycle, dft_results)

        state = CampaignState.load(path)
        if cycle != state.current_cycle:
            raise RuntimeError(
                f"Campaign cycle changed before report stage (job cycle={cycle}, current={state.current_cycle})"
            )
        rec = state.current_cycle_record
        rec.model_artifact_path = artifact_path
        rec.metrics["model_validation_passed"] = True
        rec.metrics["job_id"] = job_id
        rec.metrics["job_status"] = "completed"
        rec.metrics["report_generated_at"] = utc_now_iso()
        state.current_model_artifacts = artifact_path
        state.advance("report")
        state.save(path)

        state = CampaignState.load(path)
        if cycle != state.current_cycle:
            raise RuntimeError(
                f"Campaign cycle changed before cycle advance (job cycle={cycle}, current={state.current_cycle})"
            )
        state.advance_cycle()
        state.save(path)

        _DISCOVERY_JOB_STORE.update(
            job_id,
            status="completed",
            stage="completed",
            completed_at=utc_now_iso(),
            processed_count=len(dft_results),
            stable_found=int(dft_summary["stable_count"]),
            metrics={
                "dft_summary": dft_summary,
                "artifact_path": artifact_path,
            },
        )
    except Exception as exc:
        path = _campaign_path(campaign_name)
        if path.exists():
            try:
                state = CampaignState.load(path)
                if cycle == state.current_cycle:
                    rec = state.current_cycle_record
                    rec.metrics["job_id"] = job_id
                    rec.metrics["job_status"] = "failed"
                    rec.metrics["job_error"] = str(exc)
                    state.save(path)
            except Exception:
                pass
        _DISCOVERY_JOB_STORE.update(
            job_id,
            status="failed",
            stage="failed",
            completed_at=utc_now_iso(),
            error=str(exc),
        )
    finally:
        _DISCOVERY_JOB_TASKS.pop(job_id, None)


@app.get("/discovery/campaigns")
async def list_discovery_campaigns(api_key: str = Security(get_api_key)):
    """List all discovery campaigns."""
    campaigns = _list_campaigns()
    return {
        "success": True,
        "campaigns": [c.summary() for c in campaigns],
    }


class CreateCampaignBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    pool_source: str = Field(default="ensemble_soap_loco_test")
    ranking_strategy: str = Field(default=DEFAULT_RANKING_STRATEGY)


@app.post("/discovery/campaigns")
async def create_discovery_campaign(
    body: CreateCampaignBody,
    api_key: str = Security(get_api_key),
):
    """Create a new discovery campaign from a predictions parquet pool."""
    if not _CAMPAIGN_NAME_RE.match(body.name):
        raise HTTPException(
            status_code=400,
            detail="Campaign name must be alphanumeric, hyphens, or underscores (1-64 chars)",
        )
    try:
        ranking_strategy = validate_ranking_strategy(body.ranking_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _campaign_path(body.name).exists():
        raise HTTPException(status_code=409, detail=f"Campaign '{body.name}' already exists")

    parquet_path = Path("data/predictions") / f"{body.pool_source}.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Pool source '{body.pool_source}' not found")

    df, id_col = await _load_predictions_pool_df(parquet_path)
    pool_ids = df[id_col].tolist()

    if not pool_ids:
        raise HTTPException(status_code=400, detail="Pool is empty")

    state = CampaignState.new(body.name, pool_ids)
    state.pool_source = body.pool_source
    state.ranking_strategy = ranking_strategy
    state.save(_campaign_path(body.name))

    return {"success": True, "campaign": state.summary()}


@app.get("/discovery/campaigns/{name}")
async def get_discovery_campaign(name: str, api_key: str = Security(get_api_key)):
    """Get full details for a specific campaign."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)
    active_job = _active_discovery_job(name)

    return {
        "success": True,
        "campaign": {
            **state.summary(),
            "created_at": state.created_at,
            "cycles": [
                {
                    "cycle": c.cycle,
                    "started_at": c.started_at,
                    "completed_at": c.completed_at,
                    "stage": c.stage,
                    "candidates_selected": c.candidates_selected,
                    "n_stable_found": c.n_stable_found,
                    "metrics": c.metrics,
                }
                for c in state.cycles
            ],
            "stages": list(STAGES),
            "active_job": active_job.to_dict() if active_job is not None else None,
        },
    }


@app.get("/discovery/strategies")
async def get_discovery_strategies(api_key: str = Security(get_api_key)):
    return {
        "success": True,
        "default": DEFAULT_RANKING_STRATEGY,
        "strategies": [
            {
                "id": "balanced",
                "label": "Balanced",
                "description": "Mixes stability, confidence, and moderate exploration.",
            },
            {
                "id": "exploit",
                "label": "Exploit",
                "description": "Prioritize highest-likelihood stable candidates for immediate DFT.",
            },
            {
                "id": "explore",
                "label": "Explore",
                "description": "Prioritize uncertainty and novelty to discover new regions.",
            },
            {
                "id": "risk_averse",
                "label": "Risk Averse",
                "description": "Prioritize conservative candidates and penalize OOD risk.",
            },
        ],
    }


@app.post("/discovery/campaigns/{name}/screen")
async def screen_discovery_candidates(
    name: str,
    strategy: Optional[str] = None,
    api_key: str = Security(get_api_key),
):
    """Run ML screening on the remaining candidate pool for the current cycle."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)

    if state.current_stage != "screen":
        raise HTTPException(
            status_code=400,
            detail=f"Campaign is at stage '{state.current_stage}', not 'screen'",
        )

    if not state.pool_remaining_ids:
        raise HTTPException(status_code=400, detail="No remaining candidates in pool")

    parquet_path = Path("data/predictions") / f"{state.pool_source}.parquet"
    df, id_col = await _load_predictions_pool_df(parquet_path)

    remaining_set = set(state.pool_remaining_ids)
    pool_df = df[df[id_col].isin(remaining_set)].copy()

    if pool_df.empty:
        raise HTTPException(status_code=400, detail="No matching candidates found in predictions")

    active_strategy = _resolve_ranking_strategy(state, strategy)
    _, candidates = _rank_and_format_candidates(pool_df, id_col=id_col, strategy=active_strategy)

    # Advance campaign state
    rec = state.current_cycle_record
    rec.metrics["candidates_scored"] = len(candidates)
    rec.metrics["pool_remaining_at_screen"] = len(remaining_set)
    rec.metrics["ranking_strategy"] = active_strategy
    state.ranking_strategy = active_strategy
    state.advance("select")
    state.save(path)

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "candidates_screened": len(candidates),
        "ranking_strategy": active_strategy,
        "candidates": candidates,
    }


class SelectCandidatesBody(BaseModel):
    batch_size: int = Field(default=20, ge=1, le=200)
    strategy: Optional[str] = Field(default=None)
    include_hold: bool = Field(default=True)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    commit: bool = Field(default=True)


@app.post("/discovery/campaigns/{name}/select")
async def select_discovery_batch(
    name: str,
    body: SelectCandidatesBody,
    api_key: str = Security(get_api_key),
):
    """Select a top-ranked candidate batch for DFT follow-up."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)
    if state.current_stage not in {"screen", "select"}:
        raise HTTPException(
            status_code=400,
            detail=f"Campaign is at stage '{state.current_stage}', not 'screen' or 'select'",
        )

    parquet_path = Path("data/predictions") / f"{state.pool_source}.parquet"
    df, id_col = await _load_predictions_pool_df(parquet_path)
    remaining_set = set(state.pool_remaining_ids)
    pool_df = df[df[id_col].isin(remaining_set)].copy()
    if pool_df.empty:
        raise HTTPException(status_code=400, detail="No remaining candidates in pool")

    active_strategy = _resolve_ranking_strategy(state, body.strategy)
    _, ranked_candidates = _rank_and_format_candidates(pool_df, id_col=id_col, strategy=active_strategy)

    allowed_actions = {"DFT"}
    if body.include_hold:
        allowed_actions.add("HOLD")

    filtered = [c for c in ranked_candidates if c["action"] in allowed_actions]
    if body.min_score is not None:
        filtered = [c for c in filtered if float(c["acquisition_score"]) >= float(body.min_score)]

    selected = filtered[: body.batch_size]
    selected_ids = [c["material_id"] for c in selected]

    if body.commit:
        rec = state.current_cycle_record
        rec.candidates_selected = selected_ids
        rec.metrics["selection_strategy"] = active_strategy
        rec.metrics["selected_batch_size"] = len(selected_ids)
        rec.metrics["selection_include_hold"] = body.include_hold
        rec.metrics["selection_min_score"] = body.min_score
        state.ranking_strategy = active_strategy
        state.advance("submit_dft")
        state.save(path)

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "stage": state.current_stage,
        "committed": body.commit,
        "ranking_strategy": active_strategy,
        "requested_batch_size": body.batch_size,
        "selected_count": len(selected),
        "selected_ids": selected_ids,
        "candidates": selected,
    }


class SubmitDFTBody(BaseModel):
    mode: str = Field(default="mock")


@app.post("/discovery/campaigns/{name}/submit-dft")
async def submit_discovery_dft(
    name: str,
    body: SubmitDFTBody,
    api_key: str = Security(get_api_key),
):
    """Submit selected candidates for async DFT execution."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)
    if state.current_stage != "submit_dft":
        raise HTTPException(
            status_code=400,
            detail=f"Campaign is at stage '{state.current_stage}', not 'submit_dft'",
        )

    active = _active_discovery_job(name)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Campaign already has an active job: {active.job_id}",
        )

    selected_ids = list(state.current_cycle_record.candidates_selected)
    if not selected_ids:
        raise HTTPException(status_code=400, detail="No selected candidates to submit for DFT")

    mode = (body.mode or "mock").strip().lower()
    if mode != "mock":
        raise HTTPException(
            status_code=400,
            detail="Only mode='mock' is currently available in API runtime",
        )

    job = _DISCOVERY_JOB_STORE.create(
        campaign_name=name,
        cycle=state.current_cycle,
        mode=mode,
        selected_count=len(selected_ids),
    )

    rec = state.current_cycle_record
    rec.dft_job_id = job.job_id
    rec.metrics["dft_submission_mode"] = mode
    rec.metrics["dft_submission_selected"] = len(selected_ids)
    rec.metrics["dft_submission_at"] = utc_now_iso()
    state.advance("wait_dft")
    state.save(path)

    task = asyncio.create_task(_run_discovery_job(job.job_id, name, state.current_cycle, mode))
    _DISCOVERY_JOB_TASKS[job.job_id] = task

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "stage": state.current_stage,
        "job": job.to_dict(),
    }


class RunCycleBody(BaseModel):
    batch_size: int = Field(default=20, ge=1, le=200)
    strategy: Optional[str] = Field(default=None)
    include_hold: bool = Field(default=True)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mode: str = Field(default="mock")


@app.post("/discovery/campaigns/{name}/run-cycle")
async def run_discovery_cycle(
    name: str,
    body: RunCycleBody,
    api_key: str = Security(get_api_key),
):
    """
    Run an end-to-end async cycle:
    screen/select -> submit_dft -> wait_dft -> ingest -> retrain -> report -> next cycle.
    """
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)
    if state.current_stage not in {"screen", "select", "submit_dft", "wait_dft"}:
        raise HTTPException(
            status_code=400,
            detail=f"Campaign stage '{state.current_stage}' cannot run auto-cycle",
        )

    active = _active_discovery_job(name)
    if active is not None:
        return {
            "success": True,
            "campaign_name": name,
            "cycle": state.current_cycle,
            "stage": state.current_stage,
            "message": "Active job already running",
            "job": active.to_dict(),
        }

    if state.current_stage in {"screen", "select"}:
        parquet_path = Path("data/predictions") / f"{state.pool_source}.parquet"
        df, id_col = await _load_predictions_pool_df(parquet_path)
        remaining_set = set(state.pool_remaining_ids)
        pool_df = df[df[id_col].isin(remaining_set)].copy()
        if pool_df.empty:
            raise HTTPException(status_code=400, detail="No remaining candidates in pool")

        active_strategy = _resolve_ranking_strategy(state, body.strategy)
        _, ranked_candidates = _rank_and_format_candidates(pool_df, id_col=id_col, strategy=active_strategy)
        selected, selected_ids = _select_ranked_candidates(
            ranked_candidates=ranked_candidates,
            batch_size=body.batch_size,
            include_hold=body.include_hold,
            min_score=body.min_score,
        )
        if not selected_ids:
            raise HTTPException(status_code=400, detail="No candidates matched auto-cycle selection criteria")

        rec = state.current_cycle_record
        rec.metrics["candidates_scored"] = len(ranked_candidates)
        rec.metrics["selection_strategy"] = active_strategy
        rec.metrics["selected_batch_size"] = len(selected_ids)
        rec.metrics["selection_include_hold"] = body.include_hold
        rec.metrics["selection_min_score"] = body.min_score
        rec.candidates_selected = selected_ids
        state.ranking_strategy = active_strategy
        state.advance("submit_dft")
        state.save(path)
    else:
        selected_ids = list(state.current_cycle_record.candidates_selected)
        if not selected_ids:
            raise HTTPException(status_code=400, detail="Campaign is in submit/wait stage but no selected candidates exist")

    submit_body = SubmitDFTBody(mode=body.mode)
    submit_resp = await submit_discovery_dft(name=name, body=submit_body, api_key=api_key)
    submit_resp["auto_cycle"] = True
    return submit_resp


@app.get("/discovery/campaigns/{name}/jobs")
async def get_discovery_campaign_jobs(
    name: str,
    api_key: str = Security(get_api_key),
):
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")
    jobs = [job.to_dict() for job in _DISCOVERY_JOB_STORE.list_for_campaign(name)]
    return {
        "success": True,
        "campaign_name": name,
        "jobs": jobs,
        "active_job": next((j for j in reversed(jobs) if j["status"] in {"queued", "running"}), None),
    }


@app.get("/discovery/jobs/{job_id}")
async def get_discovery_job(job_id: str, api_key: str = Security(get_api_key)):
    job = _DISCOVERY_JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Discovery job '{job_id}' not found")
    return {"success": True, "job": job.to_dict()}


@app.get("/discovery/campaigns/{name}/candidates")
async def get_discovery_candidates(
    name: str,
    limit: int = 50,
    offset: int = 0,
    strategy: Optional[str] = None,
    api_key: str = Security(get_api_key),
):
    """Get paginated ranked candidates for a campaign."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)

    parquet_path = Path("data/predictions") / f"{state.pool_source}.parquet"
    df, id_col = await _load_predictions_pool_df(parquet_path)

    remaining_set = set(state.pool_remaining_ids)
    pool_df = df[df[id_col].isin(remaining_set)].copy()

    active_strategy = _resolve_ranking_strategy(state, strategy)
    ranked_df, _ = _rank_and_format_candidates(pool_df, id_col=id_col, strategy=active_strategy)

    total = len(ranked_df)
    page_df = ranked_df.iloc[offset: offset + limit]

    candidates = [
        _format_ranked_candidate(row.to_dict(), id_col=id_col, rank=offset + idx + 1, strategy=active_strategy)
        for idx, (_, row) in enumerate(page_df.iterrows())
    ]

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "stage": state.current_stage,
        "ranking_strategy": active_strategy,
        "total": total,
        "offset": offset,
        "limit": limit,
        "candidates": candidates,
    }


# ===========================================================================
# Audit Trail Endpoints
# ===========================================================================

@app.get("/audit/recent")
async def audit_recent(
    request: Request,
    n: int = 50,
    api_key: str = Security(get_api_key),
):
    """Get the most recent N prediction audit records."""
    if n > 500:
        n = 500
    audit = get_audit_trail()
    records = audit.get_recent(n)
    return {"count": len(records), "records": records}


@app.get("/audit/stats")
async def audit_stats(
    request: Request,
    api_key: str = Security(get_api_key),
):
    """Get today's prediction audit statistics."""
    audit = get_audit_trail()
    return audit.get_stats()


@app.post("/predict", response_model=PredictionResponse)
async def predict_structure(
    request: Request,
    cif_file: UploadFile = File(...),
    api_key: str = Security(get_api_key),
):
    """
    Predict stability for a single CIF structure.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    start = time.perf_counter()
    if metrics is not None:
        metrics.record_request()
    _enforce_rate_limit(request, cost=1)

    try:
        async with _inference_slot(request_id):
            result, input_hash = await _predict_structure_core(cif_file)

        p_stable = result.p_stable if result.p_stable is not None else 0.0
        uncertainty = classify_uncertainty(result.uncertainty_epistemic)
        action = decision_to_action(result.decision)
        latency_s = time.perf_counter() - start

        if metrics is not None:
            metrics.record_prediction(
                decision=result.decision,
                decision_mode=result.decision_mode,
                ood_flag=result.ood_flag,
                ood_score=result.ood_score,
                uncertainty_epistemic=result.uncertainty_epistemic,
                ehull_pred=result.ehull_pred,
                latency_s=latency_s,
            )

        # Audit trail
        try:
            audit = get_audit_trail()
            audit.log_prediction(
                request_id=request_id,
                input_hash=input_hash,
                material_id=result.material_id,
                formula=result.formula,
                prediction=result.to_dict(),
                cathode_properties=result.cathode_properties,
                latency_ms=latency_s * 1000,
                client_id=_client_id(request),
                mode=result.decision_mode,
            )
        except Exception:
            pass  # Audit logging must never break predictions

        if LOG_PREDICTIONS:
            _log_event(
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "material_id": result.material_id,
                    "formula": result.formula,
                    "decision": result.decision,
                    "decision_mode": result.decision_mode,
                    "ood_score": result.ood_score,
                    "ood_flag": result.ood_flag,
                    "uncertainty_epistemic": result.uncertainty_epistemic,
                    "ehull_pred": result.ehull_pred,
                    "latency_ms": round(latency_s * 1000, 2),
                }
            )

        # Build cathode properties if available
        cathode_props = None
        if result.cathode_properties:
            cathode_props = CathodePropertiesResult(**{
                k: result.cathode_properties[k]
                for k in CathodePropertiesResult.model_fields
                if k in result.cathode_properties
            })

        return PredictionResponse(
            success=True,
            prediction=PredictionResult(
                material_id=cif_file.filename or "uploaded",
                pred_ehull=round(result.ehull_pred, 4),
                p_stable=round(p_stable, 3),
                uncertainty=uncertainty,
                action=action,
                confidence_interval=(round(result.ehull_lower, 4), round(result.ehull_upper, 4)),
                cathode_properties=cathode_props,
            ),
        )
        
    except HTTPException as exc:
        if metrics is not None:
            metrics.record_error()
        _log_event(
            {
                "event": "prediction_error",
                "request_id": request_id,
                "status": exc.status_code,
                "error": exc.detail,
            }
        )
        raise
    except Exception as e:
        if metrics is not None:
            metrics.record_error()
        _log_event(
            {
                "event": "prediction_error",
                "request_id": request_id,
                "error": str(e),
            }
        )
        # CRITICAL: No mock fallback here. Error out.
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    cif_files: List[UploadFile] = File(...),
    api_key: str = Security(get_api_key),
):
    """
    Predict stability for multiple CIF structures.

    Upload multiple CIF files for batch processing.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    if len(cif_files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail="Batch size exceeds limit")

    _enforce_rate_limit(request, cost=len(cif_files))
    if metrics is not None:
        metrics.record_request(count=len(cif_files))

    structures = []
    material_ids = []
    formulas = []
    predictions = []
    errors = []
    n_errors = 0

    for cif_file in cif_files:
        try:
            structure = await _parse_structure(cif_file)
            structures.append(structure)
            material_id = cif_file.filename or "uploaded"
            material_ids.append(material_id)
            formulas.append(structure.composition.reduced_formula)
        except HTTPException as exc:
            n_errors += 1
            if metrics is not None:
                metrics.record_error()
            detail = (
                dict(exc.detail)
                if isinstance(exc.detail, dict)
                else {"error": "parse_error", "message": str(exc.detail)}
            )
            detail.setdefault("filename", cif_file.filename or "uploaded")
            errors.append(detail)
            _log_event(
                {
                    "event": "batch_parse_error",
                    "request_id": request_id,
                    "filename": cif_file.filename,
                    "status": exc.status_code,
                    "error": detail,
                }
            )
        except Exception as e:
            n_errors += 1
            if metrics is not None:
                metrics.record_error()
            detail = {
                "error": "parse_error",
                "message": str(e),
                "filename": cif_file.filename or "uploaded",
            }
            errors.append(detail)
            _log_event(
                {
                    "event": "batch_parse_error",
                    "request_id": request_id,
                    "filename": cif_file.filename,
                    "error": str(e),
                }
            )

    if not structures:
        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            errors=errors,
            n_processed=0,
            n_errors=n_errors,
        )

    try:
        from .inference import get_predictor

        start = time.perf_counter()
        predictor = get_predictor()
        async with _inference_slot(request_id):
            results = predictor.predict_structures(
                structures,
                material_ids=material_ids,
                formulas=formulas,
            )
        latency_s = time.perf_counter() - start
    except Exception as e:
        if metrics is not None:
            metrics.record_error(count=len(structures))
        _log_event(
            {
                "event": "batch_inference_error",
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    per_item_latency = latency_s / len(results) if results else None

    for result in results:
        p_stable = result.p_stable if result.p_stable is not None else 0.0
        uncertainty = classify_uncertainty(result.uncertainty_epistemic)
        action = decision_to_action(result.decision)

        batch_cathode_props = None
        if result.cathode_properties:
            batch_cathode_props = CathodePropertiesResult(**{
                k: result.cathode_properties[k]
                for k in CathodePropertiesResult.model_fields
                if k in result.cathode_properties
            })

        predictions.append(
            PredictionResult(
                material_id=result.material_id,
                pred_ehull=round(result.ehull_pred, 4),
                p_stable=round(p_stable, 3),
                uncertainty=uncertainty,
                action=action,
                confidence_interval=(round(result.ehull_lower, 4), round(result.ehull_upper, 4)),
                cathode_properties=batch_cathode_props,
            )
        )

        if metrics is not None:
            metrics.record_prediction(
                decision=result.decision,
                decision_mode=result.decision_mode,
                ood_flag=result.ood_flag,
                ood_score=result.ood_score,
                uncertainty_epistemic=result.uncertainty_epistemic,
                ehull_pred=result.ehull_pred,
                latency_s=per_item_latency,
            )

        if LOG_PREDICTIONS:
            _log_event(
                {
                    "event": "batch_prediction",
                    "request_id": request_id,
                    "material_id": result.material_id,
                    "formula": result.formula,
                    "decision": result.decision,
                    "decision_mode": result.decision_mode,
                    "ood_score": result.ood_score,
                    "ood_flag": result.ood_flag,
                    "uncertainty_epistemic": result.uncertainty_epistemic,
                    "ehull_pred": result.ehull_pred,
                    "latency_ms": round((per_item_latency or 0.0) * 1000, 2),
                }
            )

    return BatchPredictionResponse(
        success=True,
        predictions=predictions,
        errors=errors,
        n_processed=len(predictions),
        n_errors=n_errors,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
