#!/usr/bin/env python3
"""
Train a K=5 deep ensemble for decision-grade cathode screening.

Usage:
    python scripts/04a_train_ensemble.py --config configs/train_cgcnn_ehull.yaml

Outputs:
    data/artifacts/ensemble/member_seed{S}/best.pt  (K checkpoints)
    data/artifacts/ensemble/ensemble_meta.json      (provenance)
"""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List

import yaml


# Ensemble configuration
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]
K = len(ENSEMBLE_SEEDS)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_single_member(
    config_path: str,
    seed: int,
    output_dir: Path,
    member_idx: int
) -> dict:
    """Train a single ensemble member with the given seed."""
    
    member_dir = output_dir / f"member_seed{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    
    # Modify config for this member
    cfg = load_cfg(config_path)
    cfg["train"]["seed"] = seed
    cfg["project"]["run_id"] = f"member_seed{seed}"
    cfg["project"]["artifacts_dir"] = str(member_dir.relative_to(Path(cfg["project"]["output_dir"])))
    cfg["project"]["reports_dir"] = str(member_dir.relative_to(Path(cfg["project"]["output_dir"])))
    
    # Write temporary config
    temp_config = member_dir / "config.yaml"
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Training ensemble member {member_idx + 1}/{K} (seed={seed})")
    print(f"{'='*60}\n")
    
    # Run training
    result = subprocess.run(
        [sys.executable, "scripts/04_train.py", "--config", str(temp_config)],
        capture_output=False
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed}")
    
    # Read metrics
    metrics_path = member_dir / f"member_seed{seed}" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"error": "metrics not found"}
    
    return {
        "seed": seed,
        "checkpoint": str(member_dir / f"member_seed{seed}" / "best.pt"),
        "config": str(temp_config),
        "metrics": metrics
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base training config")
    ap.add_argument("--seeds", type=str, default=None, 
                    help="Comma-separated seeds (default: 42,123,456,789,1011)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip members that already have checkpoints")
    args = ap.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else ENSEMBLE_SEEDS
    
    cfg = load_cfg(args.config)
    base_output = Path(cfg["project"]["output_dir"])
    ensemble_dir = base_output / "data" / "artifacts" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training ensemble with K={len(seeds)} members")
    print(f"Seeds: {seeds}")
    print(f"Output: {ensemble_dir}")
    
    members = []
    for i, seed in enumerate(seeds):
        member_dir = ensemble_dir / f"member_seed{seed}"
        checkpoint_path = member_dir / f"member_seed{seed}" / "best.pt"
        
        if args.skip_existing and checkpoint_path.exists():
            print(f"Skipping member {i+1}/{len(seeds)} (seed={seed}) - checkpoint exists")
            members.append({
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "config": str(member_dir / "config.yaml"),
                "metrics": "skipped"
            })
            continue
        
        member_info = train_single_member(args.config, seed, ensemble_dir, i)
        members.append(member_info)
    
    # Compute ensemble statistics
    val_maes = [m["metrics"].get("val", {}).get("mae", float("inf")) 
                for m in members if isinstance(m["metrics"], dict)]
    test_maes = [m["metrics"].get("test", {}).get("mae", float("inf")) 
                 for m in members if isinstance(m["metrics"], dict)]
    
    # Save ensemble metadata
    meta = {
        "created_at": datetime.now().isoformat(),
        "base_config": args.config,
        "n_members": len(seeds),
        "seeds": seeds,
        "members": members,
        "ensemble_stats": {
            "val_mae_mean": sum(val_maes) / len(val_maes) if val_maes else None,
            "val_mae_std": (sum((m - sum(val_maes)/len(val_maes))**2 for m in val_maes) / len(val_maes))**0.5 if len(val_maes) > 1 else None,
            "test_mae_mean": sum(test_maes) / len(test_maes) if test_maes else None,
            "test_mae_std": (sum((m - sum(test_maes)/len(test_maes))**2 for m in test_maes) / len(test_maes))**0.5 if len(test_maes) > 1 else None,
        }
    }
    
    meta_path = ensemble_dir / "ensemble_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Members: {len(members)}")
    if val_maes:
        print(f"Val MAE: {sum(val_maes)/len(val_maes):.4f} ± {meta['ensemble_stats']['val_mae_std']:.4f}")
    if test_maes:
        print(f"Test MAE: {sum(test_maes)/len(test_maes):.4f} ± {meta['ensemble_stats']['test_mae_std']:.4f}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
