"""Policy checks for secure deployment defaults across runtime configs."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _render_env_value(text: str, key: str) -> str | None:
    pattern = re.compile(rf"- key: {re.escape(key)}\s+value: \"?([^\n\"]+)\"?")
    match = pattern.search(text)
    return match.group(1) if match else None


def test_render_defaults_enable_auth_and_https():
    text = _read("render.yaml")
    assert _render_env_value(text, "CATHODE_AUTH_ENABLED") == "true"
    assert _render_env_value(text, "CATHODE_REQUIRE_MANIFEST_SIGNATURE") == "true"
    assert _render_env_value(text, "CATHODE_FORCE_HTTPS") == "true"
    assert _render_env_value(text, "CATHODE_ALLOW_UNSAFE_TORCH_LOAD") == "false"
    assert "CATHODE_MANIFEST_HMAC_KEY" in text


def test_render_dockerfile_defaults_are_secure():
    text = _read("render.Dockerfile")
    assert "ENV CATHODE_AUTH_ENABLED=true" in text
    assert "ENV CATHODE_REQUIRE_MANIFEST_SIGNATURE=true" in text
    assert "ENV CATHODE_FORCE_HTTPS=true" in text
    assert "ENV CATHODE_ALLOW_UNSAFE_TORCH_LOAD=false" in text


def test_cloudbuild_uses_signature_substitution_flag():
    text = _read("cloudbuild.yaml")
    expected = "CATHODE_REQUIRE_MANIFEST_SIGNATURE=${_REQUIRE_MANIFEST_SIGNATURE}"
    assert expected in text


def test_deploy_script_disables_unsafe_torch_load():
    text = _read("deploy_gcp.ps1")
    assert '"CATHODE_ALLOW_UNSAFE_TORCH_LOAD=false"' in text
