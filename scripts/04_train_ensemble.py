#!/usr/bin/env python3
"""
Train a K-member deep ensemble for cathode screening.

Usage:
    python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull.yaml --k 5
    python scripts/04_train_ensemble.py --config configs/train_cgcnn_ehull.yaml --k 5 --member 2

Outputs:
    artifacts/models/<run_id>/member_0/best.pt
    artifacts/models/<run_id>/member_1/best.pt
    ...
    artifacts/models/<run_id>/ensemble_meta.json
"""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_seeds(base_seed: int, k: int) -> List[int]:
    """
    Generate K deterministic seeds from base seed.
    
    Uses simple linear formula: seed_i = base_seed + i
    This ensures reproducibility and easy identification.
    """
    return [base_seed + i for i in range(k)]


def train_single_member(
    config_path: str,
    seed: int,
    member_idx: int,
    ensemble_dir: Path,
    k: int
) -> dict:
    """Train a single ensemble member with the given seed."""
    
    member_dir = ensemble_dir / f"member_{member_idx}"
    member_dir.mkdir(parents=True, exist_ok=True)
    
    # Modify config for this member
    cfg = load_cfg(config_path)
    cfg["train"]["seed"] = seed
    cfg["project"]["run_id"] = f"member_{member_idx}"
    
    # Set artifacts/reports to be inside member directory.
    # Build the relative path directly to avoid relative_to(Path(".")) issues on Windows/Py3.10.
    base_output = Path(cfg["project"]["output_dir"])
    try:
        rel = str(member_dir.relative_to(base_output))
    except ValueError:
        rel = str(member_dir)
    cfg["project"]["artifacts_dir"] = rel
    cfg["project"]["reports_dir"] = rel
    
    # Write member-specific config
    temp_config = member_dir / "config.yaml"
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Training ensemble member {member_idx + 1}/{k} (seed={seed})")
    print(f"Output: {member_dir}")
    print(f"{'='*60}\n")
    
    # Run training via subprocess
    result = subprocess.run(
        [sys.executable, "scripts/04_train.py", "--config", str(temp_config)],
        capture_output=False
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for member {member_idx} (seed={seed})")
    
    # Read metrics
    metrics_path = member_dir / f"member_{member_idx}" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"error": "metrics not found"}
    
    # Checkpoint path
    checkpoint_path = member_dir / f"member_{member_idx}" / "best.pt"
    
    return {
        "member_idx": member_idx,
        "seed": seed,
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        "config": str(temp_config),
        "metrics": metrics
    }


def main():
    ap = argparse.ArgumentParser(description="Train K-member deep ensemble")
    ap.add_argument("--config", required=True, help="Base training config YAML")
    ap.add_argument("--k", type=int, default=5, help="Number of ensemble members (default: 5)")
    ap.add_argument("--base-seed", type=int, default=None, 
                    help="Base seed for generating member seeds (default: from config)")
    ap.add_argument("--member", type=int, default=None,
                    help="Train only this member index (0 to K-1). For parallel training.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds (overrides --k and --base-seed)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip members that already have checkpoints")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Ensemble run ID (default: ensemble_<timestamp>)")
    args = ap.parse_args()

    # Load base config
    cfg = load_cfg(args.config)

    # Determine seeds: explicit list takes priority, then base_seed + k
    base_seed = args.base_seed or cfg["train"].get("seed", 42)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        args.k = len(seeds)
    else:
        seeds = generate_seeds(base_seed, args.k)

    # Set up ensemble directory
    base_output = Path(cfg["project"]["output_dir"])
    run_id = args.run_id or f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensemble_dir = base_output / "artifacts" / "models" / run_id
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ensemble training configuration:")
    print(f"  K = {args.k}")
    print(f"  Base seed = {base_seed}")
    print(f"  Seeds = {seeds}")
    print(f"  Output = {ensemble_dir}")
    if args.member is not None:
        print(f"  Training only member {args.member}")
    print()
    
    # Determine which members to train
    if args.member is not None:
        if args.member < 0 or args.member >= args.k:
            raise ValueError(f"--member must be in range [0, {args.k-1}]")
        member_indices = [args.member]
    else:
        member_indices = list(range(args.k))
    
    # Train members
    members = []
    for idx in member_indices:
        seed = seeds[idx]
        member_dir = ensemble_dir / f"member_{idx}"
        checkpoint_path = member_dir / f"member_{idx}" / "best.pt"
        
        # Check if we should skip
        if args.skip_existing and checkpoint_path.exists():
            print(f"Skipping member {idx} (seed={seed}) - checkpoint exists")
            members.append({
                "member_idx": idx,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "config": str(member_dir / "config.yaml"),
                "metrics": "skipped"
            })
            continue
        
        # Train this member
        member_info = train_single_member(
            args.config, seed, idx, ensemble_dir, args.k
        )
        members.append(member_info)
    
    # Compute ensemble statistics (only if we trained all members)
    val_maes = []
    test_maes = []
    for m in members:
        if isinstance(m["metrics"], dict):
            if "val" in m["metrics"] and "mae" in m["metrics"]["val"]:
                val_maes.append(m["metrics"]["val"]["mae"])
            if "test" in m["metrics"] and "mae" in m["metrics"]["test"]:
                test_maes.append(m["metrics"]["test"]["mae"])
    
    # Save/update ensemble metadata
    meta_path = ensemble_dir / "ensemble_meta.json"
    
    # Load existing metadata if it exists (for incremental training)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Update members list
        existing_members = {m["member_idx"]: m for m in meta.get("members", [])}
        for m in members:
            existing_members[m["member_idx"]] = m
        meta["members"] = list(existing_members.values())
        meta["updated_at"] = datetime.now().isoformat()
    else:
        meta = {
            "created_at": datetime.now().isoformat(),
            "base_config": str(Path(args.config).resolve()),
            "k": args.k,
            "base_seed": base_seed,
            "seeds": seeds,
            "members": members,
        }
    
    # Update statistics
    if val_maes:
        import numpy as np
        meta["ensemble_stats"] = {
            "val_mae_mean": float(np.mean(val_maes)),
            "val_mae_std": float(np.std(val_maes)) if len(val_maes) > 1 else 0.0,
            "test_mae_mean": float(np.mean(test_maes)) if test_maes else None,
            "test_mae_std": float(np.std(test_maes)) if len(test_maes) > 1 else None,
        }
    
    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Ensemble directory: {ensemble_dir}")
    print(f"Members trained: {len(members)}/{args.k}")
    if val_maes:
        print(f"Val MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    if test_maes:
        print(f"Test MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
