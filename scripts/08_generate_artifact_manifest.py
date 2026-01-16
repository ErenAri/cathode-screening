#!/usr/bin/env python3
"""
Generate a manifest for artifact governance and reproducibility.

Example:
    python scripts/08_generate_artifact_manifest.py --artifact-dir data/artifacts
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cathode_screening.inference.artifact_manifest import (
    compute_manifest,
    sign_manifest,
    write_manifest,
    write_signature,
)


def resolve_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate artifact manifest")
    ap.add_argument("--artifact-dir", default="data/artifacts", help="Artifacts root directory")
    ap.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: <artifact-dir>/manifest.json)",
    )
    ap.add_argument(
        "--include",
        action="append",
        default=None,
        help="Glob(s) to include (repeatable, default: all files)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob(s) to exclude (repeatable)",
    )
    ap.add_argument("--git-commit", default=None, help="Override git commit")
    ap.add_argument("--no-git", action="store_true", help="Skip git commit lookup")
    ap.add_argument(
        "--sign",
        action="store_true",
        help="Generate HMAC signature file for manifest",
    )
    ap.add_argument(
        "--signature-output",
        default=None,
        help="Signature output path (default: <manifest>.sig)",
    )
    ap.add_argument(
        "--sign-key-env",
        default="CATHODE_MANIFEST_HMAC_KEY",
        help="Env var name for signing key",
    )
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.output) if args.output else artifact_dir / "manifest.json"

    git_commit = args.git_commit
    if git_commit is None and not args.no_git:
        git_commit = resolve_git_commit(ROOT)

    manifest = compute_manifest(
        artifact_dir=artifact_dir,
        git_commit=git_commit,
        include_globs=args.include,
        exclude_globs=args.exclude,
        output_path=output_path,
    )
    write_manifest(output_path, manifest)

    print(f"Wrote manifest with {len(manifest.files)} files to {output_path}")
    if git_commit:
        print(f"Git commit: {git_commit}")

    if args.sign:
        key = os.getenv(args.sign_key_env)
        if not key:
            raise RuntimeError(
                f"Signing requested but {args.sign_key_env} is not set"
            )
        signature = sign_manifest(manifest, key)
        sig_path = Path(args.signature_output) if args.signature_output else output_path.with_suffix(".sig")
        write_signature(sig_path, signature)
        print(f"Wrote manifest signature to {sig_path}")


if __name__ == "__main__":
    main()
