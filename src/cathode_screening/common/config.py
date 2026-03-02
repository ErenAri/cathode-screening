"""Training config validation.

Provides lightweight schema checking for YAML training configs so that
typos in key names are caught early rather than silently ignored.
"""
from __future__ import annotations

from typing import Dict, List, Set


# Required top-level sections
REQUIRED_SECTIONS: Set[str] = {"project", "data", "model", "train"}

# Known keys per section (not exhaustive — extra keys are warnings, not errors)
KNOWN_KEYS: Dict[str, Set[str]] = {
    "project": {"name", "run_id", "output_dir", "artifacts_dir", "reports_dir"},
    "data": {"dataset_name", "processed_dir", "splits_manifest", "target", "aux_target", "structure_dir"},
    "model": {
        "type", "node_in_dim", "node_embed_dim", "edge_rbf_bins",
        "message_passing_layers", "dropout", "pooling", "heads",
        # MACE fine-tuning keys
        "backbone", "head_dim", "freeze_backbone", "unfreeze_last_n", "r_max",
    },
    "train": {
        "device", "precision", "batch_size", "balanced_sampling", "num_workers",
        "persistent_workers", "epochs", "warmup_epochs", "balanced_sampling_start_epoch",
        "early_stopping", "optimizer", "scheduler", "grad_clip_norm", "seed",
        "grad_accum_steps",
        "lr_backbone_factor",
    },
    "loss": {"primary", "cls_weight", "sigma_weight", "fallback"},
    "metrics": {"report", "topk"},
    "logging": {"console", "save_every_epoch", "save_best_only", "wandb"},
}


def validate_training_config(cfg: dict) -> List[str]:
    """Validate a training config dict. Returns list of issues (empty = valid)."""
    issues: List[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            issues.append(f"missing required section: '{section}'")

    for section, keys in cfg.items():
        if not isinstance(keys, dict):
            continue
        known = KNOWN_KEYS.get(section)
        if known is None:
            continue
        for key in keys:
            if key not in known:
                issues.append(f"unknown key '{section}.{key}' (typo?)")

    # Type checks for critical numeric fields
    if "train" in cfg and isinstance(cfg["train"], dict):
        train = cfg["train"]
        for field in ("batch_size", "epochs", "seed"):
            if field in train and not isinstance(train[field], int):
                issues.append(f"train.{field} should be int, got {type(train[field]).__name__}")
        if "num_workers" in train and not isinstance(train["num_workers"], int):
            issues.append(f"train.num_workers should be int, got {type(train['num_workers']).__name__}")

    if "model" in cfg and isinstance(cfg["model"], dict):
        model = cfg["model"]
        if "type" in model and model["type"] not in ("cgcnn", "chgnet", "mace"):
            issues.append(f"model.type '{model['type']}' not recognized (expected cgcnn, chgnet, or mace)")

    return issues
