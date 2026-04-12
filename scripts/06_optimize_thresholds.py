#!/usr/bin/env python3
"""
Optimize utility-based decision thresholds for KEEP/MAYBE/KILL triage.

Two modes:
- DFT-followup: Cheap validation available (can afford more false positives)
- Experimental: Expensive synthesis (must be conservative)

Usage:
    python scripts/06_optimize_thresholds.py \
        --ensemble-dir data/artifacts/ensemble \
        --calibration-params data/artifacts/calibration/conformal_params.json \
        --data-config configs/train_cgcnn_ehull.yaml

Outputs:
    data/artifacts/thresholds/thresholds_dft.json
    data/artifacts/thresholds/thresholds_exp.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import shared utilities
from shared_imports import (
    CGCNN, Normalizer, GraphNPZDataset, collate, load_cfg, get_device
)


# Utility parameters for each mode
UTILITY_DFT = {
    "V_keep": 10.0,       # Value of finding stable candidate
    "V_kill": 1.0,        # Value of correctly rejecting unstable
    "C_false_keep": 2.0,  # Cost of DFT on false positive (cheap)
    "C_miss": 20.0,       # Cost of missing stable candidate
}

UTILITY_EXP = {
    "V_keep": 100.0,      # High value of synthesis-ready candidate
    "V_kill": 5.0,        # Value of avoiding wasted synthesis
    "C_false_keep": 50.0, # Cost of failed synthesis (expensive!)
    "C_miss": 30.0,       # Cost of missing candidate (opportunity cost)
}

# Stability thresholds (eV/atom)
THRESH_STABLE = 0.05
THRESH_METASTABLE = 0.10


def load_ensemble(ensemble_dir: Path, device: torch.device):
    """Load ensemble models."""
    meta_path = ensemble_dir / "ensemble_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    models = []
    normalizers = []
    
    for member in meta["members"]:
        ckpt_path = Path(member["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["cfg"]
        
        model = CGCNN(
            node_in=int(cfg["model"].get("node_in_dim", 6)),
            node_dim=int(cfg["model"]["node_embed_dim"]),
            edge_dim=int(cfg["model"]["edge_rbf_bins"]),
            layers=int(cfg["model"]["message_passing_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            pooling=str(cfg["model"]["pooling"])
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        
        norm = Normalizer()
        norm.load_state_dict(ckpt["normalizer"])
        normalizers.append(norm)
    
    return models, normalizers, meta


@torch.no_grad()
def run_ensemble_inference(
    models: List,
    normalizers: List,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference with ensemble and return raw quantile predictions."""
    
    q10_all = [[] for _ in models]
    q50_all = [[] for _ in models]
    q90_all = [[] for _ in models]
    
    for batch_data in loader:
        x, src, dst, e, b, y = batch_data
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        for k, (model, norm) in enumerate(zip(models, normalizers)):
            q10, q50, q90, _, _ = model(x, src, dst, e, b)
            q10_all[k].append(norm.denorm(q10).cpu().numpy())
            q50_all[k].append(norm.denorm(q50).cpu().numpy())
            q90_all[k].append(norm.denorm(q90).cpu().numpy())
    
    # Stack and average across ensemble
    q10 = np.mean([np.concatenate(q10_all[k]) for k in range(len(models))], axis=0)
    q50 = np.mean([np.concatenate(q50_all[k]) for k in range(len(models))], axis=0)
    q90 = np.mean([np.concatenate(q90_all[k]) for k in range(len(models))], axis=0)
    
    return q10, q50, q90


def compute_utility(decision: str, y_true: float, params: Dict[str, float]) -> float:
    """
    Compute utility for a single decision given true Ehull.
    
    Utility matrix:
    | Decision | y < 0.05 (stable) | 0.05 ≤ y < 0.10 (meta) | y ≥ 0.10 (unstable) |
    |----------|-------------------|------------------------|---------------------|
    | KEEP     | +V_keep           | +V_keep × 0.5          | -C_false_keep       |
    | MAYBE    | -C_miss × 0.5     | 0                      | 0                   |
    | KILL     | -C_miss           | -C_miss × 0.3          | +V_kill             |
    """
    is_stable = y_true < THRESH_STABLE
    is_metastable = THRESH_STABLE <= y_true < THRESH_METASTABLE
    is_unstable = y_true >= THRESH_METASTABLE
    
    if decision == "KEEP":
        if is_stable:
            return params["V_keep"]
        elif is_metastable:
            return params["V_keep"] * 0.5  # Partial value
        else:
            return -params["C_false_keep"]
    
    elif decision == "KILL":
        if is_stable:
            return -params["C_miss"]  # Missed opportunity
        elif is_metastable:
            return -params["C_miss"] * 0.3  # Minor miss
        else:
            return params["V_kill"]  # Correct rejection
    
    else:  # MAYBE
        if is_stable:
            return -params["C_miss"] * 0.5  # Delayed but not killed
        else:
            return 0.0  # Neutral


def decide(q10: float, q90: float, tau_keep: float, tau_kill: float) -> str:
    """
    Make decision based on calibrated quantile bounds and thresholds.
    
    KEEP: Pessimistic bound (q90) below keep threshold -> confidently stable
    KILL: Optimistic bound (q10) above kill threshold -> confidently unstable
    MAYBE: Otherwise
    """
    if q90 < tau_keep:
        return "KEEP"
    elif q10 > tau_kill:
        return "KILL"
    else:
        return "MAYBE"


def evaluate_thresholds(
    q10_cal: np.ndarray,
    q90_cal: np.ndarray,
    y_true: np.ndarray,
    tau_keep: float,
    tau_kill: float,
    utility_params: Dict[str, float],
) -> Dict:
    """Evaluate a threshold pair on validation data."""
    
    decisions = [decide(q10_cal[i], q90_cal[i], tau_keep, tau_kill) 
                 for i in range(len(y_true))]
    
    n = len(y_true)
    n_keep = decisions.count("KEEP")
    n_maybe = decisions.count("MAYBE")
    n_kill = decisions.count("KILL")
    
    # Compute utilities
    utilities = [compute_utility(d, y, utility_params) 
                 for d, y in zip(decisions, y_true)]
    total_utility = sum(utilities)
    mean_utility = total_utility / n
    
    # Compute precision for KEEP and KILL
    keep_mask = np.array([d == "KEEP" for d in decisions])
    kill_mask = np.array([d == "KILL" for d in decisions])
    
    keep_precision = float(np.mean(y_true[keep_mask] < THRESH_STABLE)) if keep_mask.sum() > 0 else 0.0
    kill_precision = float(np.mean(y_true[kill_mask] >= THRESH_METASTABLE)) if kill_mask.sum() > 0 else 0.0
    
    # Recall for stable compounds
    n_true_stable = (y_true < THRESH_STABLE).sum()
    keep_recall = float(keep_mask[y_true < THRESH_STABLE].sum() / n_true_stable) if n_true_stable > 0 else 0.0
    
    return {
        "tau_keep": tau_keep,
        "tau_kill": tau_kill,
        "total_utility": total_utility,
        "mean_utility": mean_utility,
        "n_keep": n_keep,
        "n_maybe": n_maybe,
        "n_kill": n_kill,
        "keep_frac": n_keep / n,
        "maybe_frac": n_maybe / n,
        "kill_frac": n_kill / n,
        "keep_precision": keep_precision,
        "kill_precision": kill_precision,
        "keep_recall": keep_recall,
    }


def grid_search_thresholds(
    q10_cal: np.ndarray,
    q90_cal: np.ndarray,
    y_true: np.ndarray,
    utility_params: Dict[str, float],
    tau_keep_range: np.ndarray = None,
    tau_kill_range: np.ndarray = None,
    min_keep_precision: float = 0.80,
    min_kill_precision: float = 0.85,
) -> Tuple[Dict, List[Dict]]:
    """
    Grid search for optimal thresholds.
    
    Constraints:
    - tau_keep < tau_kill (must have gap for MAYBE region)
    - keep_precision >= min_keep_precision
    - kill_precision >= min_kill_precision
    """
    if tau_keep_range is None:
        tau_keep_range = np.arange(0.02, 0.12, 0.005)
    if tau_kill_range is None:
        tau_kill_range = np.arange(0.06, 0.20, 0.005)
    
    results = []
    best_result = None
    best_utility = float("-inf")
    
    for tau_keep in tau_keep_range:
        for tau_kill in tau_kill_range:
            if tau_keep >= tau_kill:
                continue  # Invalid: no MAYBE region
            
            result = evaluate_thresholds(
                q10_cal, q90_cal, y_true, tau_keep, tau_kill, utility_params
            )
            results.append(result)
            
            # Check constraints
            if result["keep_precision"] < min_keep_precision:
                continue
            if result["kill_precision"] < min_kill_precision:
                continue
            
            if result["mean_utility"] > best_utility:
                best_utility = result["mean_utility"]
                best_result = result
    
    # Fallback if no valid result found
    if best_result is None:
        print("Warning: No threshold pair satisfies precision constraints. Using best unconstrained.")
        best_result = max(results, key=lambda r: r["mean_utility"])
    
    return best_result, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", required=True, help="Path to ensemble directory")
    ap.add_argument("--calibration-params", required=True, help="Path to conformal_params.json")
    ap.add_argument("--data-config", required=True, help="Training config for data loading")
    ap.add_argument("--output-dir", default="data/artifacts/thresholds", help="Output directory")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, or cpu")
    args = ap.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    # Load calibration params
    with open(args.calibration_params, "r", encoding="utf-8") as f:
        calib_params = json.load(f)
    
    qhat_lo = calib_params["qhat_lo"]  # List per member
    qhat_hi = calib_params["qhat_hi"]
    
    # Load data config
    cfg = load_cfg(args.data_config)
    
    # Load validation data
    processed_dir = Path(cfg["data"]["processed_dir"])
    ds_name = cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)
    
    splits_path = Path(cfg["data"]["splits_manifest"])
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Use validation set for threshold optimization
    val_ids = splits["val"]
    df_val = df[df["material_id"].isin(set(val_ids))].reset_index(drop=True)
    
    target = cfg["data"]["target"]["name"]
    y_val = df_val[target].values
    
    # Load ensemble and run inference
    print("Loading ensemble and running inference...")
    models, normalizers, meta = load_ensemble(Path(args.ensemble_dir), device)
    print(f"Loaded {len(models)} ensemble members")
    
    # Create validation data loader
    val_ds = GraphNPZDataset(df_val, target_col=target)
    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin_memory,
        num_workers=0
    )
    
    # Run ensemble inference
    q10_raw, q50_raw, q90_raw = run_ensemble_inference(models, normalizers, val_loader, device)
    
    # Apply calibration (average across members)
    qhat_lo_mean = np.mean(qhat_lo)
    qhat_hi_mean = np.mean(qhat_hi)
    
    q10_cal = q10_raw - qhat_lo_mean
    q90_cal = q90_raw + qhat_hi_mean
    
    n_val = len(y_val)
    print(f"Validation set size: {n_val}")
    print(f"True stable (Ehull < 0.05): {(y_val < THRESH_STABLE).sum()}")
    print(f"Calibration corrections: qhat_lo={qhat_lo_mean:.4f}, qhat_hi={qhat_hi_mean:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimize for DFT mode
    print("\n" + "="*60)
    print("Optimizing thresholds for DFT-followup mode")
    print("="*60)
    
    best_dft, all_dft = grid_search_thresholds(
        q10_cal, q90_cal, y_val, UTILITY_DFT,
        min_keep_precision=0.85,
        min_kill_precision=0.85,
    )
    
    print(f"Best DFT thresholds:")
    print(f"  τ_keep = {best_dft['tau_keep']:.3f}")
    print(f"  τ_kill = {best_dft['tau_kill']:.3f}")
    print(f"  Mean utility = {best_dft['mean_utility']:.2f}")
    print(f"  Decision distribution: KEEP={best_dft['keep_frac']:.1%}, MAYBE={best_dft['maybe_frac']:.1%}, KILL={best_dft['kill_frac']:.1%}")
    print(f"  KEEP precision = {best_dft['keep_precision']:.2f}, recall = {best_dft['keep_recall']:.2f}")
    print(f"  KILL precision = {best_dft['kill_precision']:.2f}")
    
    dft_output = {
        "mode": "dft_followup",
        "tau_keep": best_dft["tau_keep"],
        "tau_kill": best_dft["tau_kill"],
        "expected_utility": best_dft["mean_utility"],
        "utility_params": UTILITY_DFT,
        "decision_distribution": {
            "KEEP": best_dft["keep_frac"],
            "MAYBE": best_dft["maybe_frac"],
            "KILL": best_dft["kill_frac"],
        },
        "performance": {
            "keep_precision": best_dft["keep_precision"],
            "keep_recall": best_dft["keep_recall"],
            "kill_precision": best_dft["kill_precision"],
        },
        "n_validation": n_val,
        "calibration_qhat_lo": qhat_lo_mean,
        "calibration_qhat_hi": qhat_hi_mean,
    }
    
    with open(output_dir / "thresholds_dft.json", "w", encoding="utf-8") as f:
        json.dump(dft_output, f, indent=2)
    
    # Optimize for Experimental mode
    print("\n" + "="*60)
    print("Optimizing thresholds for Experimental mode")
    print("="*60)
    
    best_exp, all_exp = grid_search_thresholds(
        q10_cal, q90_cal, y_val, UTILITY_EXP,
        min_keep_precision=0.95,  # Much stricter for expensive synthesis
        min_kill_precision=0.90,
    )
    
    print(f"Best Experimental thresholds:")
    print(f"  τ_keep = {best_exp['tau_keep']:.3f}")
    print(f"  τ_kill = {best_exp['tau_kill']:.3f}")
    print(f"  Mean utility = {best_exp['mean_utility']:.2f}")
    print(f"  Decision distribution: KEEP={best_exp['keep_frac']:.1%}, MAYBE={best_exp['maybe_frac']:.1%}, KILL={best_exp['kill_frac']:.1%}")
    print(f"  KEEP precision = {best_exp['keep_precision']:.2f}, recall = {best_exp['keep_recall']:.2f}")
    print(f"  KILL precision = {best_exp['kill_precision']:.2f}")
    
    exp_output = {
        "mode": "experimental",
        "tau_keep": best_exp["tau_keep"],
        "tau_kill": best_exp["tau_kill"],
        "expected_utility": best_exp["mean_utility"],
        "utility_params": UTILITY_EXP,
        "decision_distribution": {
            "KEEP": best_exp["keep_frac"],
            "MAYBE": best_exp["maybe_frac"],
            "KILL": best_exp["kill_frac"],
        },
        "performance": {
            "keep_precision": best_exp["keep_precision"],
            "keep_recall": best_exp["keep_recall"],
            "kill_precision": best_exp["kill_precision"],
        },
        "n_validation": n_val,
        "calibration_qhat_lo": qhat_lo_mean,
        "calibration_qhat_hi": qhat_hi_mean,
    }
    
    with open(output_dir / "thresholds_exp.json", "w", encoding="utf-8") as f:
        json.dump(exp_output, f, indent=2)
    
    print(f"\nSaved thresholds to {output_dir}")
    
    # GPU memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
