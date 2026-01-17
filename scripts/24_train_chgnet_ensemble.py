"""
Train CHGNet ensemble for decision-grade predictions.

Trains K=5 models with different seeds for uncertainty quantification.
Includes proper EF calculation and per-sample predictions output.

Usage:
    python scripts/24_train_chgnet_ensemble.py --epochs 50
"""

import json
import pickle
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pymatgen.core import Structure


SEEDS = [42, 123, 456, 789, 1024]


def train_single_model(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> str:
    """Train a single CHGNet model."""
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData, collate_graphs
    from torch.utils.data import DataLoader
    
    print(f"\n{'='*60}")
    print(f"Training CHGNet with seed {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    data_path = Path(data_dir)
    with open(data_path / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    energies = np.load(data_path / "energies.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")
    
    # Split data
    train_structures = [structures[i] for i in train_idx]
    train_energies = [float(energies[i]) for i in train_idx]
    train_forces = [[[0.0, 0.0, 0.0] for _ in range(len(s))] for s in train_structures]
    
    val_structures = [structures[i] for i in val_idx]
    val_energies = [float(energies[i]) for i in val_idx]
    val_forces = [[[0.0, 0.0, 0.0] for _ in range(len(s))] for s in val_structures]
    
    # Create StructureData
    train_dataset = StructureData(
        structures=train_structures,
        energies=train_energies,
        forces=train_forces,
    )
    
    val_dataset = StructureData(
        structures=val_structures,
        energies=val_energies,
        forces=val_forces,
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    
    # Load model
    model = CHGNet.load()
    model.to(device)
    
    # Create output directory
    output_path = Path(output_dir) / f"seed_{seed}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = Trainer(
        model=model,
        targets="e",
        optimizer="AdamW",
        scheduler="CosLR",
        criterion="MAE",
        epochs=epochs,
        learning_rate=lr,
        use_device=device,
        print_freq=50,
    )
    
    trainer.train(train_loader, val_loader, save_dir=str(output_path))
    
    return str(output_path)


def get_best_checkpoint(model_dir: Path) -> Path:
    """Get best checkpoint deterministically by parsing the MAE from filename.
    
    Filename format: bestE_epoch{N}_e{MAE}_fNA_sNA_mNA.pth.tar
    where MAE is an integer (e.g., e85 means MAE=0.085)
    """
    best_ckpts = list(model_dir.glob("bestE_*.pth.tar"))
    
    if not best_ckpts:
        raise FileNotFoundError(f"No bestE checkpoint found in {model_dir}")
    
    def parse_mae(ckpt_path: Path) -> float:
        """Parse MAE from checkpoint filename."""
        name = ckpt_path.name
        # Extract e{number} part
        match = re.search(r"_e(\d+)_", name)
        if match:
            return int(match.group(1))
        return float("inf")
    
    # Sort by MAE (ascending) and return best
    best_ckpts_sorted = sorted(best_ckpts, key=parse_mae)
    return best_ckpts_sorted[0]


@torch.no_grad()
def evaluate_ensemble(
    model_dirs: List[str],
    data_dir: str,
    output_dir: str,
    device: str = "cuda",
) -> dict:
    """Evaluate ensemble on test set with proper metrics."""
    from chgnet.model import CHGNet
    
    # Load data
    data_path = Path(data_dir)
    with open(data_path / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    energies = np.load(data_path / "energies.npy")
    test_idx = np.load(data_path / "test_idx.npy")
    
    # Load metadata for material IDs
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)
    
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    test_ids = [metadata[i]["material_id"] for i in test_idx]
    
    print(f"\nEvaluating ensemble on {len(test_structures)} test structures...")
    
    # Load models deterministically
    models = []
    for model_dir in model_dirs:
        model = CHGNet.load()
        
        try:
            ckpt_path = get_best_checkpoint(Path(model_dir))
            print(f"  Loading: {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"]["state_dict"])
        except Exception as e:
            print(f"  Warning: Could not load checkpoint from {model_dir}: {e}")
        
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models")
    
    # Predict
    all_preds = []
    
    for struct in tqdm(test_structures, desc="Predicting"):
        preds = []
        for model in models:
            try:
                result = model.predict_structure(struct)
                preds.append(result["e"])
            except:
                preds.append(np.nan)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)  # (n_samples, n_models)
    
    # Ensemble statistics
    ensemble_mean = np.nanmean(all_preds, axis=1)
    ensemble_std = np.nanstd(all_preds, axis=1)
    
    # Filter valid predictions
    valid = ~np.isnan(ensemble_mean)
    N = np.sum(valid)
    
    # Basic metrics
    mae = np.mean(np.abs(ensemble_mean[valid] - test_energies[valid]))
    rmse = np.sqrt(np.mean((ensemble_mean[valid] - test_energies[valid])**2))
    
    # Decision metrics at various thresholds
    thresholds = [0.01, 0.02, 0.05, 0.10]
    decision_metrics = {}
    
    for thresh in thresholds:
        # True stable = E_hull <= thresh
        true_stable = test_energies[valid] <= thresh
        n_true_stable = np.sum(true_stable)
        base_rate = n_true_stable / N if N > 0 else 0
        
        # Sort by predicted E_hull
        sorted_idx = np.argsort(ensemble_mean[valid])
        sorted_true = test_energies[valid][sorted_idx]
        
        # EF@1%, EF@5% - CORRECTED FORMULA
        # EF = (precision in top) / (base rate)
        # EF = (hits_in_top / n_top) / (n_true_stable / N)
        for pct in [0.01, 0.05]:
            n_top = max(1, int(N * pct))
            hits_in_top = np.sum(sorted_true[:n_top] <= thresh)
            precision_in_top = hits_in_top / n_top if n_top > 0 else 0
            ef = precision_in_top / base_rate if base_rate > 0 else 0
            decision_metrics[f"EF@{int(pct*100)}%_thresh{thresh}"] = float(ef)
        
        # Recall@100, Recall@500
        for k in [100, 500]:
            if k <= N:
                hits_in_k = np.sum(sorted_true[:k] <= thresh)
                recall = hits_in_k / n_true_stable if n_true_stable > 0 else 0
                decision_metrics[f"Recall@{k}_thresh{thresh}"] = float(recall)
    
    # False kill rate (predicting unstable when actually stable)
    stable_thresh = 0.05
    kill_thresh = 0.15
    true_stable_mask = test_energies[valid] <= stable_thresh
    pred_kill_mask = ensemble_mean[valid] > kill_thresh
    false_kills = np.sum(true_stable_mask & pred_kill_mask)
    false_kill_rate = false_kills / np.sum(true_stable_mask) if np.sum(true_stable_mask) > 0 else 0
    
    # Save per-sample predictions
    predictions_df = pd.DataFrame({
        "material_id": [test_ids[i] for i in range(len(test_idx)) if valid[i]],
        "y_true": test_energies[valid],
        "pred_mean": ensemble_mean[valid],
        "pred_std": ensemble_std[valid],
        "n_valid_models": np.sum(~np.isnan(all_preds[valid]), axis=1),
    })
    
    # Add individual model predictions
    for i in range(all_preds.shape[1]):
        predictions_df[f"pred_model_{i}"] = all_preds[valid, i]
    
    output_path = Path(output_dir)
    predictions_df.to_parquet(output_path / "test_predictions.parquet", index=False)
    print(f"\nSaved per-sample predictions to {output_path / 'test_predictions.parquet'}")
    
    results = {
        "mae": float(mae),
        "rmse": float(rmse),
        "n_test": int(N),
        "n_models": len(models),
        "mean_uncertainty": float(np.nanmean(ensemble_std)),
        "false_kill_rate": float(false_kill_rate),
        **decision_metrics,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train CHGNet ensemble")
    parser.add_argument("--data-dir", default="data/processed/chgnet_soap_loco")
    parser.add_argument("--output-dir", default="data/artifacts/chgnet_ensemble")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train ensemble
    model_dirs = []
    for seed in SEEDS:
        model_dir = train_single_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
        )
        model_dirs.append(model_dir)
    
    # Evaluate ensemble
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_ensemble(model_dirs, args.data_dir, str(output_path), device)
    
    print("\nResults:")
    print(f"  MAE: {results['mae']:.4f} eV/atom")
    print(f"  RMSE: {results['rmse']:.4f} eV/atom")
    print(f"  Mean Uncertainty (σ): {results['mean_uncertainty']:.4f}")
    print(f"  False Kill Rate: {results['false_kill_rate']:.4f}")
    print("\nDecision Metrics:")
    for k, v in results.items():
        if k.startswith("EF") or k.startswith("Recall"):
            print(f"  {k}: {v:.2f}")
    
    # Save results
    with open(output_path / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'ensemble_results.json'}")


if __name__ == "__main__":
    main()
