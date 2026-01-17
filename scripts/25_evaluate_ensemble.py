"""
Standalone evaluation script for CHGNet ensemble.
Computes MAE, RMSE, EF@k, Recall@k, and saves per-sample predictions.

Usage:
    python scripts/25_evaluate_ensemble.py
"""

import json
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm

from chgnet.model import CHGNet


def get_best_checkpoint(model_dir):
    """Get best checkpoint deterministically by MAE."""
    best_ckpts = list(Path(model_dir).glob("bestE_*.pth.tar"))
    if not best_ckpts:
        return None
    
    def parse_mae(p):
        m = re.search(r"_e(\d+)_", p.name)
        return int(m.group(1)) if m else float("inf")
    
    return sorted(best_ckpts, key=parse_mae)[0]


def compute_ef(sorted_true, n_true_stable, n_total, n_top, thresh):
    """Compute enrichment factor.
    
    EF = (precision in top) / (base rate)
       = (hits_in_top / n_top) / (n_true_stable / n_total)
    """
    hits_in_top = np.sum(sorted_true[:n_top] <= thresh)
    precision_in_top = hits_in_top / n_top if n_top > 0 else 0
    base_rate = n_true_stable / n_total if n_total > 0 else 0
    
    return precision_in_top / base_rate if base_rate > 0 else 0


def main():
    # Load data
    data_path = Path("data/processed/chgnet_soap_loco")
    structures = pickle.load(open(data_path / "structures.pkl", "rb"))
    energies = np.load(data_path / "energies.npy")
    test_idx = np.load(data_path / "test_idx.npy")
    metadata = json.load(open(data_path / "metadata.json"))
    
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    test_ids = [metadata[i]["material_id"] for i in test_idx]
    
    print(f"Test set: {len(test_structures)} samples")
    
    # Load all 5 models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42, 123, 456, 789, 1024]
    model_dirs = [f"data/artifacts/chgnet_ensemble/seed_{s}" for s in seeds]
    models = []
    
    for md in model_dirs:
        model = CHGNet.load()
        ckpt_path = get_best_checkpoint(md)
        if ckpt_path:
            print(f"Loading {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"]["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models")
    
    # Predict on full test set
    all_preds = []
    for struct in tqdm(test_structures, desc="Predicting"):
        preds = []
        for model in models:
            try:
                result = model.predict_structure(struct)
                preds.append(float(result["e"]))
            except Exception:
                preds.append(float("nan"))
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)
    print(f"Predictions shape: {all_preds.shape}")
    
    # Compute ensemble statistics
    ensemble_mean = np.nanmean(all_preds, axis=1)
    ensemble_std = np.nanstd(all_preds, axis=1)
    valid = ~np.isnan(ensemble_mean)
    N = np.sum(valid)
    
    # Basic metrics
    mae = np.mean(np.abs(ensemble_mean[valid] - test_energies[valid]))
    rmse = np.sqrt(np.mean((ensemble_mean[valid] - test_energies[valid])**2))
    
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS (Li-Cathode Subset, SOAP-LOCO)")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.4f} eV/atom")
    print(f"RMSE: {rmse:.4f} eV/atom")
    print(f"Mean Uncertainty (σ): {np.nanmean(ensemble_std):.4f}")
    print(f"Valid samples: {N}/{len(test_structures)}")
    
    # Decision metrics
    thresholds = [0.01, 0.02, 0.05, 0.10]
    decision_metrics = {}
    
    # Sort predictions for ranking metrics
    sorted_idx = np.argsort(ensemble_mean[valid])
    sorted_true = test_energies[valid][sorted_idx]
    
    print(f"\nDecision Metrics:")
    for thresh in thresholds:
        n_true_stable = np.sum(test_energies[valid] <= thresh)
        base_rate = n_true_stable / N
        
        for pct in [0.01, 0.05]:
            n_top = max(1, int(N * pct))
            ef = compute_ef(sorted_true, n_true_stable, N, n_top, thresh)
            key = f"EF@{int(pct*100)}%_thresh{thresh}"
            decision_metrics[key] = float(ef)
            print(f"  {key}: {ef:.2f}")
        
        for k in [100, 500]:
            if k <= N:
                hits_in_k = np.sum(sorted_true[:k] <= thresh)
                recall = hits_in_k / n_true_stable if n_true_stable > 0 else 0
                key = f"Recall@{k}_thresh{thresh}"
                decision_metrics[key] = float(recall)
                print(f"  {key}: {recall:.3f}")
    
    # False kill rate
    stable_thresh = 0.05
    kill_thresh = 0.15
    true_stable = test_energies[valid] <= stable_thresh
    pred_kill = ensemble_mean[valid] > kill_thresh
    false_kills = np.sum(true_stable & pred_kill)
    false_kill_rate = false_kills / np.sum(true_stable) if np.sum(true_stable) > 0 else 0
    
    print(f"\nFalse Kill Rate (stable@{stable_thresh} killed@{kill_thresh}): {false_kill_rate:.4f}")
    
    # Save per-sample predictions
    output_dir = Path("data/artifacts/chgnet_ensemble")
    predictions_df = pd.DataFrame({
        "material_id": [test_ids[i] for i in range(len(test_idx)) if valid[i]],
        "y_true": test_energies[valid],
        "pred_mean": ensemble_mean[valid],
        "pred_std": ensemble_std[valid],
        "n_valid_models": np.sum(~np.isnan(all_preds[valid]), axis=1),
    })
    
    for i in range(all_preds.shape[1]):
        predictions_df[f"pred_model_{i}"] = all_preds[valid, i]
    
    predictions_df.to_parquet(output_dir / "test_predictions.parquet", index=False)
    print(f"\nSaved predictions to {output_dir / 'test_predictions.parquet'}")
    
    # Save results JSON
    results = {
        "mae": float(mae),
        "rmse": float(rmse),
        "n_test": int(N),
        "n_models": len(models),
        "mean_uncertainty": float(np.nanmean(ensemble_std)),
        "false_kill_rate": float(false_kill_rate),
        **decision_metrics,
    }
    
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_dir / 'ensemble_results.json'}")


if __name__ == "__main__":
    main()
