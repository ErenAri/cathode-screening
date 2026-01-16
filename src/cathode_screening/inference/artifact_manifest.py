from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass
class ArtifactManifest:
    schema_version: str
    generated_at: str
    artifact_dir: str
    git_commit: Optional[str]
    files: List[ArtifactFile]
    summary: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ArtifactManifest":
        files = [
            ArtifactFile(**f) for f in data.get("files", [])  # type: ignore[arg-type]
        ]
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            generated_at=str(data.get("generated_at", "")),
            artifact_dir=str(data.get("artifact_dir", "")),
            git_commit=data.get("git_commit") if data.get("git_commit") else None,
            files=files,
            summary=data.get("summary", {}) if isinstance(data.get("summary"), dict) else {},
        )


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _manifest_payload(manifest: ArtifactManifest) -> bytes:
    data = json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return data.encode("utf-8")


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _iter_files(
    artifact_dir: Path,
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Iterable[Tuple[Path, str]]:
    include = include_globs or ["**/*"]
    exclude = exclude_globs or []
    for path in artifact_dir.rglob("*"):
        if path.is_dir():
            continue
        if output_path is not None and path.resolve() == output_path.resolve():
            continue
        rel = path.relative_to(artifact_dir).as_posix()
        if rel == "manifest.json":
            continue
        if not _matches_any(rel, include):
            continue
        if exclude and _matches_any(rel, exclude):
            continue
        yield path, rel


def _load_summary(artifact_dir: Path) -> Dict[str, object]:
    summary: Dict[str, object] = {}

    meta_path = artifact_dir / "ensemble_meta.json"
    if not meta_path.exists():
        meta_path = artifact_dir / "ensemble" / "ensemble_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary["ensemble"] = {
                "meta_path": meta_path.relative_to(artifact_dir).as_posix(),
                "n_members": meta.get("n_members") or meta.get("k"),
                "seeds": meta.get("seeds", []),
                "base_config": meta.get("base_config"),
                "created_at": meta.get("created_at"),
            }
        except json.JSONDecodeError:
            summary["ensemble"] = {"meta_path": meta_path.relative_to(artifact_dir).as_posix()}

    calib_candidates = [
        artifact_dir / "calibration" / "ensemble" / "conformal_params.json",
        artifact_dir / "calibration" / "conformal_params.json",
        artifact_dir / "calibration_global" / "conformal_params.json",
    ]
    for calib_path in calib_candidates:
        if calib_path.exists():
            try:
                calib = json.loads(calib_path.read_text(encoding="utf-8"))
                summary["calibration"] = {
                    "path": calib_path.relative_to(artifact_dir).as_posix(),
                    "alpha": calib.get("alpha"),
                    "n_calibration": calib.get("n_calibration"),
                    "delta_upper": calib.get("delta_upper"),
                    "delta_lower": calib.get("delta_lower"),
                }
            except json.JSONDecodeError:
                summary["calibration"] = {"path": calib_path.relative_to(artifact_dir).as_posix()}
            break

    thresholds_dir = artifact_dir / "thresholds"
    if thresholds_dir.exists():
        thresholds: Dict[str, object] = {}
        for name in ("thresholds_dft.json", "thresholds_exp.json"):
            path = thresholds_dir / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                thresholds[name] = {
                    "path": path.relative_to(artifact_dir).as_posix(),
                    "tau_keep": data.get("tau_keep"),
                    "tau_kill": data.get("tau_kill"),
                }
            except json.JSONDecodeError:
                thresholds[name] = {"path": path.relative_to(artifact_dir).as_posix()}
        if thresholds:
            summary["thresholds"] = thresholds

    ood_dir = artifact_dir / "ood"
    if (ood_dir / "ensemble").exists():
        ood_dir = ood_dir / "ensemble"
    thresholds_path = ood_dir / "ood_thresholds.json"
    if thresholds_path.exists():
        try:
            data = json.loads(thresholds_path.read_text(encoding="utf-8"))
            summary["ood"] = {
                "path": thresholds_path.relative_to(artifact_dir).as_posix(),
                "t_hi": data.get("t_hi"),
                "t_mid": data.get("t_mid"),
            }
        except json.JSONDecodeError:
            summary["ood"] = {"path": thresholds_path.relative_to(artifact_dir).as_posix()}

    return summary


def compute_manifest(
    artifact_dir: str | Path,
    git_commit: Optional[str] = None,
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> ArtifactManifest:
    artifact_dir = Path(artifact_dir)
    output_path = Path(output_path) if output_path is not None else None
    files: List[ArtifactFile] = []

    for path, rel in _iter_files(
        artifact_dir, include_globs, exclude_globs, output_path=output_path
    ):
        stat = path.stat()
        files.append(
            ArtifactFile(
                path=rel,
                size_bytes=stat.st_size,
                sha256=_hash_file(path),
            )
        )

    files.sort(key=lambda f: f.path)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = _load_summary(artifact_dir)

    return ArtifactManifest(
        schema_version="1.0",
        generated_at=generated_at,
        artifact_dir=artifact_dir.resolve().as_posix(),
        git_commit=git_commit,
        files=files,
        summary=summary,
    )


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def load_manifest(path: str | Path) -> ArtifactManifest:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ArtifactManifest.from_dict(data)


def sign_manifest(manifest: ArtifactManifest, key: str) -> str:
    payload = _manifest_payload(manifest)
    digest = hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return digest


def sign_manifest_file(manifest_path: str | Path, key: str) -> str:
    manifest = load_manifest(manifest_path)
    return sign_manifest(manifest, key)


def write_signature(path: str | Path, signature: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(signature.strip(), encoding="utf-8")


def load_signature(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def verify_manifest_signature(
    manifest_path: str | Path,
    signature_path: str | Path,
    key: str,
) -> bool:
    manifest = load_manifest(manifest_path)
    expected = sign_manifest(manifest, key)
    actual = load_signature(signature_path)
    return hmac.compare_digest(expected, actual)


def validate_manifest(
    manifest_path: str | Path,
    artifact_dir: Optional[str | Path] = None,
) -> List[str]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    root = Path(artifact_dir) if artifact_dir is not None else Path(manifest.artifact_dir)
    issues: List[str] = []

    for entry in manifest.files:
        path = root / entry.path
        if not path.exists():
            issues.append(f"missing:{entry.path}")
            continue
        if path.stat().st_size != entry.size_bytes:
            issues.append(f"size_mismatch:{entry.path}")
            continue
        digest = _hash_file(path)
        if digest != entry.sha256:
            issues.append(f"hash_mismatch:{entry.path}")

    return issues
