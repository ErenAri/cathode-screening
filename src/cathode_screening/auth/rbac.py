"""
Role-Based Access Control (RBAC) for CathodeScreen.

Replaces the flat API-key authentication with role-aware authorization.
Three built-in roles with hierarchical permissions:

  - **viewer**: Read-only access to predictions, database, and metrics.
  - **operator**: All viewer permissions plus write operations
    (predictions, batch, discovery campaigns, async jobs).
  - **admin**: Full access including model registry, shadow deployment,
    audit management, and user/key administration.

Keys are configured via environment variables or a JSON key-store file.
Backward compatible: existing flat ``CATHODE_API_KEY`` keys default to
the ``operator`` role so nothing breaks on upgrade.

Environment variables
---------------------
CATHODE_RBAC_ENABLED
    Set to ``true`` to activate RBAC enforcement.  When ``false``
    (default), the system falls back to the legacy flat-key check.
CATHODE_RBAC_KEYS_FILE
    Path to a JSON file mapping API keys to metadata::

        [
          {
            "key_hash": "sha256-hex-digest",
            "role": "admin",
            "org_id": "acme-corp",
            "description": "CI pipeline key"
          }
        ]

    Plain-text keys can also be supplied for dev via ``"key": "..."``
    instead of ``"key_hash"``.
CATHODE_RBAC_DEFAULT_ROLE
    Role assigned to legacy (non-RBAC) keys.  Default: ``operator``.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("cathode_screening.auth")


# ---------------------------------------------------------------------------
# Roles & permissions
# ---------------------------------------------------------------------------

class Role(str, enum.Enum):
    """Built-in authorization roles (hierarchical)."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


class Permission(str, enum.Enum):
    """Fine-grained permissions mapped to API operations."""

    # Read
    READ_HEALTH = "read:health"
    READ_METRICS = "read:metrics"
    READ_MODEL_INFO = "read:model_info"
    READ_DATABASE = "read:database"
    READ_SCREENING = "read:screening"
    READ_AUDIT = "read:audit"
    READ_DISCOVERY = "read:discovery"
    READ_REGISTRY = "read:registry"
    READ_SHADOW = "read:shadow"
    READ_BATCHER = "read:batcher"

    # Write
    WRITE_PREDICT = "write:predict"
    WRITE_BATCH_PREDICT = "write:batch_predict"
    WRITE_ASYNC_PREDICT = "write:async_predict"
    WRITE_DISCOVERY = "write:discovery"

    # Admin
    ADMIN_REGISTRY = "admin:registry"
    ADMIN_SHADOW = "admin:shadow"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_KEYS = "admin:keys"


# Role → permission mapping
_ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_HEALTH,
        Permission.READ_METRICS,
        Permission.READ_MODEL_INFO,
        Permission.READ_DATABASE,
        Permission.READ_SCREENING,
        Permission.READ_AUDIT,
        Permission.READ_DISCOVERY,
        Permission.READ_REGISTRY,
        Permission.READ_SHADOW,
        Permission.READ_BATCHER,
    },
    Role.OPERATOR: {
        # Inherits all viewer permissions
        Permission.READ_HEALTH,
        Permission.READ_METRICS,
        Permission.READ_MODEL_INFO,
        Permission.READ_DATABASE,
        Permission.READ_SCREENING,
        Permission.READ_AUDIT,
        Permission.READ_DISCOVERY,
        Permission.READ_REGISTRY,
        Permission.READ_SHADOW,
        Permission.READ_BATCHER,
        # Plus write
        Permission.WRITE_PREDICT,
        Permission.WRITE_BATCH_PREDICT,
        Permission.WRITE_ASYNC_PREDICT,
        Permission.WRITE_DISCOVERY,
    },
    Role.ADMIN: set(Permission),  # All permissions
}


def role_has_permission(role: Role, permission: Permission) -> bool:
    """Check whether a role grants a specific permission."""
    return permission in _ROLE_PERMISSIONS.get(role, set())


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Identity:
    """Authenticated caller identity."""

    key_id: str
    role: Role
    org_id: Optional[str] = None
    description: str = ""
    permissions: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.permissions:
            perms = _ROLE_PERMISSIONS.get(self.role, set())
            object.__setattr__(self, "permissions", frozenset(perms))

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def require_permission(self, permission: Permission) -> None:
        """Raise if this identity lacks the required permission."""
        if not self.has_permission(permission):
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_permissions",
                    "message": f"Role '{self.role.value}' lacks permission '{permission.value}'",
                    "required_permission": permission.value,
                    "your_role": self.role.value,
                },
            )


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------

@dataclass
class _KeyEntry:
    """Internal representation of a registered API key."""

    key_hash: str  # SHA-256 hex digest
    role: Role
    org_id: Optional[str]
    description: str
    key_id: str  # Truncated hash for logging (first 8 chars)


class KeyStore:
    """In-memory key store loaded from env vars and/or JSON file."""

    def __init__(self) -> None:
        self._entries: List[_KeyEntry] = []

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def add_key(
        self,
        *,
        key: Optional[str] = None,
        key_hash: Optional[str] = None,
        role: Role = Role.OPERATOR,
        org_id: Optional[str] = None,
        description: str = "",
    ) -> None:
        if key:
            h = self._hash(key)
        elif key_hash:
            h = key_hash
        else:
            raise ValueError("Either 'key' or 'key_hash' must be provided")

        self._entries.append(
            _KeyEntry(
                key_hash=h,
                role=role,
                org_id=org_id,
                description=description,
                key_id=h[:8],
            )
        )

    def authenticate(self, candidate: str) -> Optional[Identity]:
        """Validate an API key and return the caller Identity, or None."""
        digest = self._hash(candidate)
        for entry in self._entries:
            if secrets.compare_digest(digest, entry.key_hash):
                return Identity(
                    key_id=entry.key_id,
                    role=entry.role,
                    org_id=entry.org_id,
                    description=entry.description,
                )
        return None

    def __len__(self) -> int:
        return len(self._entries)


def _load_keys_file(path: str) -> List[dict]:
    """Load key definitions from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"RBAC keys file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("RBAC keys file must contain a JSON array")
    return data


def build_key_store() -> KeyStore:
    """Build a KeyStore from environment configuration.

    1. Load keys from CATHODE_RBAC_KEYS_FILE (if set).
    2. Import legacy CATHODE_API_KEY(S) with default role.
    """
    store = KeyStore()
    default_role_str = os.getenv("CATHODE_RBAC_DEFAULT_ROLE", "operator").strip().lower()
    try:
        default_role = Role(default_role_str)
    except ValueError:
        logger.warning("Invalid CATHODE_RBAC_DEFAULT_ROLE '%s', using 'operator'", default_role_str)
        default_role = Role.OPERATOR

    # 1. Load from RBAC keys file
    keys_file = os.getenv("CATHODE_RBAC_KEYS_FILE", "").strip()
    if keys_file:
        for entry in _load_keys_file(keys_file):
            role_str = entry.get("role", default_role.value).strip().lower()
            try:
                role = Role(role_str)
            except ValueError:
                logger.warning("Unknown role '%s' in keys file, skipping entry", role_str)
                continue
            store.add_key(
                key=entry.get("key"),
                key_hash=entry.get("key_hash"),
                role=role,
                org_id=entry.get("org_id"),
                description=entry.get("description", ""),
            )
        logger.info("Loaded %d RBAC keys from %s", len(store), keys_file)

    # 2. Import legacy keys with default role
    def _import_legacy(env_var: str) -> None:
        raw = os.getenv(env_var)
        if not raw:
            return
        for part in raw.replace(";", ",").split(","):
            value = part.strip()
            if value:
                store.add_key(key=value, role=default_role, description=f"legacy:{env_var}")

    _import_legacy("CATHODE_API_KEY")
    _import_legacy("CATHODE_API_KEYS")

    # Import legacy key hashes
    raw_hashes = os.getenv("CATHODE_API_KEY_HASHES")
    if raw_hashes:
        for part in raw_hashes.replace(";", ",").split(","):
            h = part.strip()
            if h:
                store.add_key(key_hash=h, role=default_role, description="legacy:hash")

    return store


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_key_store: Optional[KeyStore] = None
_rbac_enabled: Optional[bool] = None


def get_key_store() -> KeyStore:
    """Return the global key store (created on first call)."""
    global _key_store
    if _key_store is None:
        _key_store = build_key_store()
    return _key_store


def is_rbac_enabled() -> bool:
    """Check whether RBAC mode is active."""
    global _rbac_enabled
    if _rbac_enabled is None:
        _rbac_enabled = os.getenv(
            "CATHODE_RBAC_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes"}
    return _rbac_enabled
