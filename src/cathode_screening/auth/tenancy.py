"""
Multi-tenancy middleware for CathodeScreen.

Provides organization-level data isolation so that enterprise customers
can share a single deployment without seeing each other's data.

Isolation model
---------------
- Each API key can be bound to an ``org_id`` via the RBAC keys file.
- When multi-tenancy is enabled, the ``org_id`` is injected into every
  audit record, discovery campaign, and async job so queries are scoped.
- Admin keys with no ``org_id`` can access all tenants (super-admin).

Environment variables
---------------------
CATHODE_MULTI_TENANT
    Set to ``true`` to enforce tenant isolation.
CATHODE_TENANT_HEADER
    Optional header name that overrides the org_id from the key
    (only honored for admin keys).  Default: ``X-Tenant-ID``.
"""
from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("cathode_screening.auth")

# Context variable carrying the current request's tenant
_current_tenant: ContextVar[Optional["TenantContext"]] = ContextVar(
    "current_tenant", default=None
)


@dataclass(frozen=True)
class TenantContext:
    """Tenant context for the current request."""

    org_id: str
    org_name: Optional[str] = None
    is_super_admin: bool = False


def set_tenant(ctx: Optional[TenantContext]) -> None:
    """Set the tenant context for the current async task / request."""
    _current_tenant.set(ctx)


def get_tenant() -> Optional[TenantContext]:
    """Retrieve the current tenant context (None if not in a request)."""
    return _current_tenant.get()


def require_tenant() -> TenantContext:
    """Retrieve the current tenant context, raising if absent."""
    ctx = _current_tenant.get()
    if ctx is None:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "tenant_required",
                "message": "Multi-tenancy is enabled but no tenant context was resolved",
            },
        )
    return ctx


def is_multi_tenant() -> bool:
    """Check whether multi-tenancy mode is active."""
    return os.getenv(
        "CATHODE_MULTI_TENANT", "false"
    ).strip().lower() in {"1", "true", "yes"}


TENANT_HEADER = os.getenv("CATHODE_TENANT_HEADER", "X-Tenant-ID").strip()


def resolve_tenant(
    *,
    org_id_from_key: Optional[str],
    header_value: Optional[str] = None,
    is_admin: bool = False,
) -> Optional[TenantContext]:
    """Resolve the effective tenant for a request.

    Rules:
      1. If multi-tenancy is disabled, return None (no isolation).
      2. Admin keys with no org_id are super-admins — they can optionally
         set ``X-Tenant-ID`` to act on behalf of a tenant.
      3. Non-admin keys must have an org_id; the header is ignored.
    """
    if not is_multi_tenant():
        return None

    # Admin without org binding → super-admin
    if is_admin and not org_id_from_key:
        if header_value:
            return TenantContext(org_id=header_value, is_super_admin=True)
        return TenantContext(org_id="__global__", is_super_admin=True)

    # Admin with org binding — header can override
    if is_admin and org_id_from_key:
        effective = header_value or org_id_from_key
        return TenantContext(
            org_id=effective,
            is_super_admin=(effective != org_id_from_key),
        )

    # Non-admin: always scoped to their key's org_id
    if org_id_from_key:
        return TenantContext(org_id=org_id_from_key)

    # Non-admin with no org_id — reject under multi-tenancy
    from fastapi import HTTPException, status

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "tenant_required",
            "message": "API key must be bound to an organization when multi-tenancy is enabled",
        },
    )
