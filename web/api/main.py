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
RAW_CONTENT_TYPES = os.getenv(
    "CATHODE_ALLOWED_CONTENT_TYPES",
    "chemical/x-cif,text/plain,application/octet-stream,application/x-cif,text/cif",
)
ALLOWED_CONTENT_TYPES = {t.strip() for t in RAW_CONTENT_TYPES.split(",") if t.strip()}

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
    return structure


async def _predict_structure_core(cif_file: UploadFile):
    from .inference import get_predictor

    structure = await _parse_structure(cif_file)

    predictor = get_predictor()
    return predictor.predict_structure(
        structure,
        material_id=cif_file.filename or "uploaded",
    )

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


class PredictionResult(BaseModel):
    """Single prediction result."""
    material_id: str
    pred_ehull: float
    p_stable: float
    uncertainty: str  # "Low", "Medium", "High"
    action: str  # "DFT", "HOLD", "SKIP"
    confidence_interval: tuple[float, float]


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
    import pandas as pd
    from pathlib import Path
    
    # Try to load predictions file
    predictions_path = Path("data/predictions/ensemble_soap_loco_test.parquet")
    
    if not predictions_path.exists():
        # Raise 500 error instead of failing silently
        raise HTTPException(status_code=500, detail="Predictions file not found on server")
    
    try:
        df = pd.read_parquet(predictions_path)
        
        # Sort by predicted E_hull (best first)
        sort_col = "q50" if "q50" in df.columns else "pred_ehull"
        df = df.sort_values(sort_col, ascending=True)  # Load all materials
        
        # Build response data
        data = []
        for _, row in df.iterrows():
            pred_ehull = row.get("q50", row.get("pred_ehull", 0))
            std = row.get("epistemic_std", 0.1)
            p_stable = row.get("p_stable", 0.5)
            
            unc = classify_uncertainty(std)
            
            # Get action
            action = get_action(p_stable, unc, pred_ehull)
            
            data.append({
                "material_id": row.get("material_id", f"mp-{_}"),
                "formula": row.get("formula", ""),
                "pred_ehull": round(float(pred_ehull), 4),
                "p_stable": round(float(p_stable), 3),
                "uncertainty": unc,
                "action": action,
                "confidence_interval": (0.0, 0.0) # Placeholder if needed
            })
        
        return {"success": True, "data": data, "total": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Discovery Engine endpoints
# ---------------------------------------------------------------------------

import re

_CAMPAIGN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


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


@app.post("/discovery/campaigns")
async def create_discovery_campaign(
    body: CreateCampaignBody,
    api_key: str = Security(get_api_key),
):
    """Create a new discovery campaign from a predictions parquet pool."""
    import pandas as pd

    if not _CAMPAIGN_NAME_RE.match(body.name):
        raise HTTPException(
            status_code=400,
            detail="Campaign name must be alphanumeric, hyphens, or underscores (1-64 chars)",
        )

    if _campaign_path(body.name).exists():
        raise HTTPException(status_code=409, detail=f"Campaign '{body.name}' already exists")

    parquet_path = Path("data/predictions") / f"{body.pool_source}.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Pool source '{body.pool_source}' not found")

    df = pd.read_parquet(parquet_path)
    id_col = "material_id" if "material_id" in df.columns else df.columns[0]
    pool_ids = df[id_col].astype(str).tolist()

    if not pool_ids:
        raise HTTPException(status_code=400, detail="Pool is empty")

    state = CampaignState.new(body.name, pool_ids)
    state.save(_campaign_path(body.name))

    return {"success": True, "campaign": state.summary()}


@app.get("/discovery/campaigns/{name}")
async def get_discovery_campaign(name: str, api_key: str = Security(get_api_key)):
    """Get full details for a specific campaign."""
    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)

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
        },
    }


@app.post("/discovery/campaigns/{name}/screen")
async def screen_discovery_candidates(
    name: str,
    api_key: str = Security(get_api_key),
):
    """Run ML screening on the remaining candidate pool for the current cycle."""
    import pandas as pd

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

    parquet_path = Path("data/predictions/ensemble_soap_loco_test.parquet")
    if not parquet_path.exists():
        raise HTTPException(status_code=500, detail="Predictions file not found")

    df = pd.read_parquet(parquet_path)
    id_col = "material_id" if "material_id" in df.columns else df.columns[0]
    df[id_col] = df[id_col].astype(str)

    remaining_set = set(state.pool_remaining_ids)
    pool_df = df[df[id_col].isin(remaining_set)].copy()

    if pool_df.empty:
        raise HTTPException(status_code=400, detail="No matching candidates found in predictions")

    sort_col = "q50" if "q50" in pool_df.columns else "pred_ehull"
    pool_df = pool_df.sort_values(sort_col, ascending=True)

    candidates = []
    for rank, (_, row) in enumerate(pool_df.iterrows(), start=1):
        pred_ehull = float(row.get("q50", row.get("pred_ehull", 0)))
        std = float(row.get("epistemic_std", 0.1))
        p_stable = float(row.get("p_stable", 0.5))

        unc = classify_uncertainty(std)
        action = get_action(p_stable, unc, pred_ehull)

        candidates.append({
            "material_id": str(row[id_col]),
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
        })

    # Advance campaign state
    rec = state.current_cycle_record
    rec.metrics["candidates_scored"] = len(candidates)
    rec.metrics["pool_remaining_at_screen"] = len(remaining_set)
    state.advance("select")
    state.save(path)

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "candidates_screened": len(candidates),
        "candidates": candidates,
    }


@app.get("/discovery/campaigns/{name}/candidates")
async def get_discovery_candidates(
    name: str,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Security(get_api_key),
):
    """Get paginated ranked candidates for a campaign."""
    import pandas as pd

    path = _campaign_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Campaign '{name}' not found")

    state = CampaignState.load(path)

    parquet_path = Path("data/predictions/ensemble_soap_loco_test.parquet")
    if not parquet_path.exists():
        raise HTTPException(status_code=500, detail="Predictions file not found")

    df = pd.read_parquet(parquet_path)
    id_col = "material_id" if "material_id" in df.columns else df.columns[0]
    df[id_col] = df[id_col].astype(str)

    remaining_set = set(state.pool_remaining_ids)
    pool_df = df[df[id_col].isin(remaining_set)].copy()

    sort_col = "q50" if "q50" in pool_df.columns else "pred_ehull"
    pool_df = pool_df.sort_values(sort_col, ascending=True)

    total = len(pool_df)
    page_df = pool_df.iloc[offset: offset + limit]

    candidates = []
    for rank, (_, row) in enumerate(page_df.iterrows(), start=offset + 1):
        pred_ehull = float(row.get("q50", row.get("pred_ehull", 0)))
        std = float(row.get("epistemic_std", 0.1))
        p_stable = float(row.get("p_stable", 0.5))
        unc = classify_uncertainty(std)
        action = get_action(p_stable, unc, pred_ehull)

        candidates.append({
            "material_id": str(row[id_col]),
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
        })

    return {
        "success": True,
        "campaign_name": name,
        "cycle": state.current_cycle,
        "stage": state.current_stage,
        "total": total,
        "offset": offset,
        "limit": limit,
        "candidates": candidates,
    }


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
            result = await _predict_structure_core(cif_file)

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

        return PredictionResponse(
            success=True,
            prediction=PredictionResult(
                material_id=cif_file.filename or "uploaded",
                pred_ehull=round(result.ehull_pred, 4),
                p_stable=round(p_stable, 3),
                uncertainty=uncertainty,
                action=action,
                confidence_interval=(round(result.ehull_lower, 4), round(result.ehull_upper, 4)),
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

        predictions.append(
            PredictionResult(
                material_id=result.material_id,
                pred_ehull=round(result.ehull_pred, 4),
                p_stable=round(p_stable, 3),
                uncertainty=uncertainty,
                action=action,
                confidence_interval=(round(result.ehull_lower, 4), round(result.ehull_upper, 4)),
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
