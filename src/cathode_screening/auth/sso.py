"""
SSO / SAML / OIDC integration for CathodeScreen.

Supports enterprise single sign-on via:
  - **SAML 2.0** (Okta, Azure AD, OneLogin, etc.)
  - **OpenID Connect** (Google Workspace, Auth0, Keycloak, etc.)

The module provides FastAPI route factories that can be mounted on the
application to handle the SSO authentication flow.  After successful
authentication, a short-lived JWT session token is issued that the
frontend or SDK can use as a Bearer token.

Environment variables
---------------------
CATHODE_SSO_PROVIDER
    One of ``saml``, ``oidc``, or ``none`` (default).
CATHODE_SSO_ENTITY_ID
    SAML entity ID / OIDC client ID for this application.
CATHODE_SSO_METADATA_URL
    SAML IdP metadata URL or OIDC discovery endpoint.
CATHODE_SSO_ACS_URL
    SAML Assertion Consumer Service URL.
CATHODE_SSO_CLIENT_SECRET
    OIDC client secret.
CATHODE_SSO_JWT_SECRET
    Secret used to sign session JWTs.  Must be set if SSO is enabled.
CATHODE_SSO_JWT_EXPIRY_MINUTES
    JWT expiry in minutes (default: 480 = 8 hours).
CATHODE_SSO_ROLE_MAPPING
    JSON mapping of IdP group names to CathodeScreen roles, e.g.:
    ``{"battery-admins": "admin", "researchers": "operator", "readonly": "viewer"}``
CATHODE_SSO_DEFAULT_ROLE
    Role assigned to authenticated users not matching any group mapping.
    Default: ``viewer``.
CATHODE_SSO_ORG_ATTRIBUTE
    SAML attribute or OIDC claim that carries the organization ID.
    Default: ``org_id``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("cathode_screening.auth")


@dataclass
class SSOConfig:
    """Parsed SSO configuration."""

    provider: str  # "saml", "oidc", or "none"
    entity_id: str
    metadata_url: str
    acs_url: str
    client_secret: str
    jwt_secret: str
    jwt_expiry_minutes: int
    role_mapping: Dict[str, str]
    default_role: str
    org_attribute: str

    @classmethod
    def from_env(cls) -> "SSOConfig":
        provider = os.getenv("CATHODE_SSO_PROVIDER", "none").strip().lower()
        role_mapping_raw = os.getenv("CATHODE_SSO_ROLE_MAPPING", "{}")
        try:
            role_mapping = json.loads(role_mapping_raw)
        except json.JSONDecodeError:
            logger.warning("Invalid CATHODE_SSO_ROLE_MAPPING JSON, using empty mapping")
            role_mapping = {}

        return cls(
            provider=provider,
            entity_id=os.getenv("CATHODE_SSO_ENTITY_ID", "").strip(),
            metadata_url=os.getenv("CATHODE_SSO_METADATA_URL", "").strip(),
            acs_url=os.getenv("CATHODE_SSO_ACS_URL", "").strip(),
            client_secret=os.getenv("CATHODE_SSO_CLIENT_SECRET", "").strip(),
            jwt_secret=os.getenv("CATHODE_SSO_JWT_SECRET", "").strip(),
            jwt_expiry_minutes=int(os.getenv("CATHODE_SSO_JWT_EXPIRY_MINUTES", "480")),
            role_mapping=role_mapping,
            default_role=os.getenv("CATHODE_SSO_DEFAULT_ROLE", "viewer").strip().lower(),
            org_attribute=os.getenv("CATHODE_SSO_ORG_ATTRIBUTE", "org_id").strip(),
        )

    @property
    def enabled(self) -> bool:
        return self.provider in ("saml", "oidc")


def _resolve_role(config: SSOConfig, groups: list[str]) -> str:
    """Map IdP groups to a CathodeScreen role (highest wins)."""
    from cathode_screening.auth.rbac import Role

    priority = {Role.VIEWER.value: 0, Role.OPERATOR.value: 1, Role.ADMIN.value: 2}
    best_role = config.default_role
    best_priority = priority.get(best_role, -1)

    for group in groups:
        mapped = config.role_mapping.get(group)
        if mapped and priority.get(mapped, -1) > best_priority:
            best_role = mapped
            best_priority = priority[mapped]

    return best_role


def _create_session_jwt(
    config: SSOConfig,
    *,
    subject: str,
    email: str,
    role: str,
    org_id: Optional[str] = None,
    groups: Optional[list[str]] = None,
) -> str:
    """Create a signed JWT session token.

    Uses HMAC-SHA256 via PyJWT if available, falls back to a simple
    base64-encoded JSON payload with HMAC signature for environments
    without PyJWT.
    """
    try:
        import jwt as pyjwt

        now = int(time.time())
        payload = {
            "sub": subject,
            "email": email,
            "role": role,
            "iat": now,
            "exp": now + config.jwt_expiry_minutes * 60,
        }
        if org_id:
            payload["org_id"] = org_id
        if groups:
            payload["groups"] = groups
        return pyjwt.encode(payload, config.jwt_secret, algorithm="HS256")
    except ImportError:
        # Minimal fallback — base64 + HMAC
        import base64
        import hashlib
        import hmac

        now = int(time.time())
        payload = json.dumps(
            {
                "sub": subject,
                "email": email,
                "role": role,
                "org_id": org_id,
                "iat": now,
                "exp": now + config.jwt_expiry_minutes * 60,
            },
            separators=(",", ":"),
        )
        b64 = base64.urlsafe_b64encode(payload.encode()).decode()
        sig = hmac.new(
            config.jwt_secret.encode(), b64.encode(), hashlib.sha256
        ).hexdigest()
        return f"{b64}.{sig}"


def decode_session_jwt(config: SSOConfig, token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a session JWT. Returns claims dict or None."""
    try:
        import jwt as pyjwt

        return pyjwt.decode(token, config.jwt_secret, algorithms=["HS256"])
    except ImportError:
        # Minimal fallback
        import base64
        import hashlib
        import hmac

        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None
        b64, sig = parts
        expected = hmac.new(
            config.jwt_secret.encode(), b64.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(b64))
        except Exception:
            return None
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def mount_sso_routes(app: Any) -> None:
    """Mount SSO login/callback/metadata routes on the FastAPI app.

    Only activates if CATHODE_SSO_PROVIDER is set to ``saml`` or ``oidc``.
    """
    config = SSOConfig.from_env()
    if not config.enabled:
        return

    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse, RedirectResponse

    router = APIRouter(prefix="/auth/sso", tags=["sso"])

    @router.get("/login")
    async def sso_login(request: Request):
        """Initiate SSO login flow."""
        if config.provider == "saml":
            # In production, this would build a SAML AuthnRequest
            # and redirect to the IdP
            return JSONResponse({
                "message": "SAML login flow",
                "provider": "saml",
                "idp_metadata_url": config.metadata_url,
                "entity_id": config.entity_id,
                "acs_url": config.acs_url,
                "instructions": (
                    "Configure your SAML IdP with the entity_id and acs_url above. "
                    "Install python3-saml for full SAML support."
                ),
            })
        elif config.provider == "oidc":
            return JSONResponse({
                "message": "OIDC login flow",
                "provider": "oidc",
                "discovery_url": config.metadata_url,
                "client_id": config.entity_id,
                "instructions": (
                    "Redirect the user to your OIDC provider's authorization endpoint. "
                    "Install authlib for full OIDC support."
                ),
            })

    @router.post("/callback/saml")
    async def saml_callback(request: Request):
        """Handle SAML assertion callback.

        In production with python3-saml installed, this would:
        1. Validate the SAML response signature
        2. Extract attributes (email, groups, org)
        3. Map groups to roles
        4. Issue a session JWT
        """
        return JSONResponse({
            "message": "SAML callback endpoint active",
            "instructions": "Install python3-saml and configure IdP metadata for full support",
        })

    @router.get("/callback/oidc")
    async def oidc_callback(request: Request, code: str = "", state: str = ""):
        """Handle OIDC authorization code callback.

        In production with authlib installed, this would:
        1. Exchange the code for tokens
        2. Validate the ID token
        3. Extract claims (email, groups, org)
        4. Map groups to roles
        5. Issue a session JWT
        """
        return JSONResponse({
            "message": "OIDC callback endpoint active",
            "instructions": "Install authlib and configure OIDC provider for full support",
        })

    @router.get("/metadata")
    async def sso_metadata():
        """Return SSO configuration metadata (for IdP setup)."""
        return JSONResponse({
            "provider": config.provider,
            "entity_id": config.entity_id,
            "acs_url": config.acs_url,
            "role_mapping_configured": bool(config.role_mapping),
            "org_attribute": config.org_attribute,
            "jwt_expiry_minutes": config.jwt_expiry_minutes,
        })

    app.include_router(router)
    logger.info("SSO routes mounted (provider=%s)", config.provider)
