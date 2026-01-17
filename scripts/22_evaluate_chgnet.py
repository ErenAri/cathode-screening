"""
Evaluate fine-tuned CHGNet on SOAP-LOCO test set.

Usage:
    python scripts/22_evaluate_chgnet.py --checkpoint data/artifacts/chgnet/chgnet_best.pt
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from chgnet.model import CHGNet
from pymatgen.core import Structure


def load_model(checkpoint_path: str, device: str = "cuda") -> CHGNet:
    """Load fine-tuned CHGNet model."""
    model = CHGNet.load()
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint val MAE: {ckpt.get('val_mae', 'unknown'):.4f}")
    
    return model


def load_test_data(
    data_dir: str = "data/processed/chgnet",
) -> Tuple[List[Structure], np.ndarray, List[str]]:
    """Load test structures and targets."""
    data_path = Path(data_dir)
    
    # Load structures
    with open(data_path / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    
    # Load energies
    energies = np.load(data_path / "energies.npy")
    
    # Load test indices
    test_idx = np.load(data_path / "test_idx.npy")
    
    # Load metadata for material IDs
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)
    
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    test_ids = [metadata[i]["material_id"] for i in test_idx]
    
    print(f"Loaded {len(test_structures)} test structures")
    
    return test_structures, test_energies, test_ids


@torch.no_grad()
def predict_batch(
    model: CHGNet,
    structures: List[Structure],
    device: str = "cuda",
) -> np.ndarray:
    """Predict E_hull for batch of structures."""
    model.eval()
    predictions = []
    
    for struct in tqdm(structures, desc="Predicting"):
        try:
            result = model.predict_structure(struct, return_site_energies=False)
            predictions.append(result.get("e", 0.0))
        except Exception as e:
            print(f"Warning: Failed to predict: {e}")
            predictions.append(np.nan)
    
    return np.array(predictions)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute regression metrics."""
    # Remove NaN predictions
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Compute DAF@10
    threshold = 0.05  # Stability threshold
    target_count = 10
    
    n_stable = np.sum(y_true <= threshold)
    sorted_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_idx]
    
    hits = 0
    n_screened = 0
    for val in y_true_sorted:
        n_screened += 1
        if val <= threshold:
            hits += 1
        if hits >= target_count:
            break
    
    hit_rate = n_stable / len(y_true) if len(y_true) > 0 else 0
    n_random = target_count / hit_rate if hit_rate > 0 else len(y_true)
    daf = n_random / n_screened if n_screened > 0 else 0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "daf_at_10": daf,
        "n_samples": len(y_true),
        "n_stable": int(n_stable),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CHGNet on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", default="data/processed/chgnet")
    parser.add_argument("--output", default="data/predictions/chgnet_test.parquet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load test data
    test_structures, test_energies, test_ids = load_test_data(args.data_dir)
    
    # Predict
    predictions = predict_batch(model, test_structures, args.device)
    
    # Compute metrics
    metrics = compute_metrics(test_energies, predictions)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (SOAP-LOCO Test Set)")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']}")
    print(f"Stable materials (E_hull <= 0.05): {metrics['n_stable']}")
    print()
    print(f"MAE:      {metrics['mae']:.4f} eV/atom")
    print(f"RMSE:     {metrics['rmse']:.4f} eV/atom")
    print(f"R²:       {metrics['r2']:.4f}")
    print(f"DAF@10:   {metrics['daf_at_10']:.2f}x")
    print("=" * 60)
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        "material_id": test_ids,
        "y_true": test_energies,
        "y_pred": predictions,
        "error": predictions - test_energies,
    })
    df.to_parquet(output_path)
    print(f"\nPredictions saved to {output_path}")
    
    # Save metrics
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
