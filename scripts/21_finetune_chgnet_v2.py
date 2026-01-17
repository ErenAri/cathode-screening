"""
Fine-tune CHGNet on cathode E_hull prediction using CHGNet's native Trainer.

This version uses the official CHGNet trainer which handles graph conversion
and batching properly for optimal performance.

Usage:
    python scripts/21_finetune_chgnet_v2.py --epochs 50
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from pymatgen.core import Structure


def load_dataset(
    data_dir: str = "data/processed/chgnet",
) -> Tuple[List[Structure], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load prepared CHGNet dataset."""
    data_path = Path(data_dir)
    
    # Load structures
    with open(data_path / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    
    # Load energies
    energies = np.load(data_path / "energies.npy")
    
    # Load splits
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")
    test_idx = np.load(data_path / "test_idx.npy")
    
    print(f"Loaded {len(structures)} structures")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return structures, energies, train_idx, val_idx, test_idx


def prepare_training_data(
    structures: List[Structure],
    energies: np.ndarray,
    indices: np.ndarray,
) -> List[dict]:
    """Prepare data in CHGNet's expected format."""
    data_list = []
    
    for idx in tqdm(indices, desc="Preparing data"):
        struct = structures[idx]
        energy = energies[idx]
        
        # CHGNet expects energy per atom for training
        n_atoms = len(struct)
        
        data_list.append({
            "structure": struct,
            "energy": float(energy),  # E_hull as target
            "energy_per_atom": float(energy),  # Already per atom
        })
    
    return data_list


@torch.no_grad()
def evaluate_chgnet(
    model: CHGNet,
    structures: List[Structure],
    targets: np.ndarray,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Evaluate CHGNet on a set of structures."""
    predictions = []
    
    for struct in tqdm(structures, desc="Evaluating", leave=False):
        try:
            pred = model.predict_structure(struct)
            predictions.append(pred.get("e", 0.0))
        except Exception:
            predictions.append(np.nan)
    
    predictions = np.array(predictions)
    
    # Filter NaN
    valid_mask = ~np.isnan(predictions)
    predictions = predictions[valid_mask]
    targets_valid = targets[valid_mask]
    
    if len(predictions) == 0:
        return float('inf'), float('inf')
    
    mae = np.mean(np.abs(predictions - targets_valid))
    rmse = np.sqrt(np.mean((predictions - targets_valid) ** 2))
    
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CHGNet using native Trainer")
    parser.add_argument("--data-dir", default="data/processed/chgnet")
    parser.add_argument("--output-dir", default="data/artifacts/chgnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Using device: {args.device}")
    
    # Load data
    structures, energies, train_idx, val_idx, test_idx = load_dataset(args.data_dir)
    
    # Prepare training data
    print("\nPreparing training data...")
    train_structures = [structures[i] for i in train_idx]
    train_energies = energies[train_idx]
    
    val_structures = [structures[i] for i in val_idx]
    val_energies = energies[val_idx]
    
    # For CHGNet trainer, we need StructureData objects
    print("Creating StructureData objects...")
    train_data = []
    for i, (struct, energy) in enumerate(tqdm(zip(train_structures, train_energies), 
                                               total=len(train_structures), desc="Train data")):
        try:
            sd = StructureData(
                structure=struct,
                energy=energy * len(struct),  # Total energy (CHGNet expects total, not per-atom)
                forces=None,
                stress=None,
            )
            train_data.append(sd)
        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"Warning: Failed to create StructureData for idx {i}: {e}")
    
    val_data = []
    for struct, energy in tqdm(zip(val_structures, val_energies), 
                               total=len(val_structures), desc="Val data"):
        try:
            sd = StructureData(
                structure=struct,
                energy=energy * len(struct),
                forces=None,
                stress=None,
            )
            val_data.append(sd)
        except Exception:
            pass
    
    print(f"\nSuccessfully created: {len(train_data)} train, {len(val_data)} val samples")
    
    # Load pretrained model
    print("\nLoading pretrained CHGNet...")
    model = CHGNet.load()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use CHGNet's native Trainer
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    trainer = Trainer(
        model=model,
        targets="e",  # Energy only (E_hull)
        optimizer="AdamW",
        scheduler="CosLR",
        criterion="MAE",
        epochs=args.epochs,
        learning_rate=args.lr,
        print_freq=10,
        wandb_path=None,
        save_dir=str(output_path),
    )
    
    # Create dataloaders
    train_loader, val_loader, _ = get_train_val_test_loader(
        train_data,
        val_data,
        test_data=None,
        batch_size=args.batch_size,
        train_ratio=0.9,  # Already split
        val_ratio=0.1,
        shuffle=True,
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    
    model.to(args.device)
    model.eval()
    
    mae, rmse = evaluate_chgnet(model, test_structures, test_energies, args.device)
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test MAE: {mae:.4f} eV/atom")
    print(f"Test RMSE: {rmse:.4f} eV/atom")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")
    
    # Save test results
    with open(output_path / "test_results.json", "w") as f:
        json.dump({
            "test_mae": mae,
            "test_rmse": rmse,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_idx),
        }, f, indent=2)


if __name__ == "__main__":
    main()
