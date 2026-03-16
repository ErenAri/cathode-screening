"""
API v1 Router — versioned endpoint prefix.

All business endpoints are mounted under /v1/ for API versioning.
The unversioned root (/) endpoints remain for backward compatibility
and will be deprecated in a future release.

Usage:
    from web.api.v1.router import router_v1
    app.include_router(router_v1, prefix="/v1")
"""
from __future__ import annotations

from fastapi import APIRouter

router_v1 = APIRouter(
    prefix="/v1",
    tags=["v1"],
    responses={
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)


def mount_v1_routes(app) -> None:
    """Mount versioned routes that mirror unversioned endpoints.

    This function creates /v1/ prefixed routes that forward to the
    same handler functions used by the unversioned endpoints. This
    ensures behavioral parity while providing a stable versioned API.

    Called during app startup in main.py.
    """
    # Import here to avoid circular imports — main.py defines the handlers
    from web.api.main import (
        # Health & info
        root,
        ready,
        get_model_info,
        get_metrics,
        get_metrics_prometheus,
        # Prediction
        predict_structure,
        predict_batch,
        # Database & screening
        get_database,
        get_screening_provisional,
        download_screening_proof,
        # Discovery
        list_discovery_campaigns,
        create_discovery_campaign,
        get_discovery_campaign,
        get_discovery_strategies,
        screen_discovery_candidates,
        select_discovery_batch,
        submit_discovery_dft,
        run_discovery_cycle,
        get_discovery_campaign_jobs,
        get_discovery_job,
        get_discovery_candidates,
        # Audit
        audit_recent,
        audit_stats,
        # Models & schemas needed for response_model
        PredictionResponse,
        BatchPredictionResponse,
        ModelInfo,
    )

    # --- Health & Info ---
    router_v1.add_api_route("/", root, methods=["GET"])
    router_v1.add_api_route("/ready", ready, methods=["GET"])
    router_v1.add_api_route(
        "/model/info", get_model_info, methods=["GET"], response_model=ModelInfo
    )
    router_v1.add_api_route("/metrics", get_metrics, methods=["GET"])
    router_v1.add_api_route("/metrics/prometheus", get_metrics_prometheus, methods=["GET"])

    # --- Prediction ---
    router_v1.add_api_route(
        "/predict",
        predict_structure,
        methods=["POST"],
        response_model=PredictionResponse,
    )
    router_v1.add_api_route(
        "/predict/batch",
        predict_batch,
        methods=["POST"],
        response_model=BatchPredictionResponse,
    )

    # --- Database & Screening ---
    router_v1.add_api_route("/database", get_database, methods=["GET"])
    router_v1.add_api_route(
        "/screening/provisional", get_screening_provisional, methods=["GET"]
    )
    router_v1.add_api_route(
        "/screening/proof/{proof_id}", download_screening_proof, methods=["GET"]
    )

    # --- Discovery ---
    router_v1.add_api_route(
        "/discovery/campaigns", list_discovery_campaigns, methods=["GET"]
    )
    router_v1.add_api_route(
        "/discovery/campaigns", create_discovery_campaign, methods=["POST"]
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}", get_discovery_campaign, methods=["GET"]
    )
    router_v1.add_api_route(
        "/discovery/strategies", get_discovery_strategies, methods=["GET"]
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/screen",
        screen_discovery_candidates,
        methods=["POST"],
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/select",
        select_discovery_batch,
        methods=["POST"],
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/submit-dft",
        submit_discovery_dft,
        methods=["POST"],
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/run-cycle",
        run_discovery_cycle,
        methods=["POST"],
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/jobs",
        get_discovery_campaign_jobs,
        methods=["GET"],
    )
    router_v1.add_api_route(
        "/discovery/jobs/{job_id}", get_discovery_job, methods=["GET"]
    )
    router_v1.add_api_route(
        "/discovery/campaigns/{name}/candidates",
        get_discovery_candidates,
        methods=["GET"],
    )

    # --- Audit ---
    router_v1.add_api_route("/audit/recent", audit_recent, methods=["GET"])
    router_v1.add_api_route("/audit/stats", audit_stats, methods=["GET"])

    # Mount the router on the app
    app.include_router(router_v1)
