"""
Fine-tune CHGNet on cathode E_hull prediction (CHGNet v0.4.2 API).

Usage:
    python scripts/21_finetune_chgnet_v3.py --epochs 50
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, collate_graphs
from pymatgen.core import Structure


def load_dataset(
    data_dir: str = "data/processed/chgnet",
) -> Tuple[List[Structure], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load prepared CHGNet dataset."""
    data_path = Path(data_dir)
    
    with open(data_path / "structures.pkl", "rb") as f:
        structures = pickle.load(f)
    
    energies = np.load(data_path / "energies.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")
    test_idx = np.load(data_path / "test_idx.npy")
    
    print(f"Loaded {len(structures)} structures")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return structures, energies, train_idx, val_idx, test_idx


def create_dummy_forces(structure: Structure) -> List[List[float]]:
    """Create zero forces for structure (we only have E_hull, no forces)."""
    n_atoms = len(structure)
    return [[0.0, 0.0, 0.0] for _ in range(n_atoms)]


@torch.no_grad()
def evaluate_model(
    model: CHGNet,
    structures: List[Structure],
    targets: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate CHGNet on structures."""
    predictions = []
    
    for struct in tqdm(structures, desc="Evaluating", leave=False):
        try:
            pred = model.predict_structure(struct)
            predictions.append(pred.get("e", 0.0))
        except Exception:
            predictions.append(np.nan)
    
    predictions = np.array(predictions)
    valid_mask = ~np.isnan(predictions)
    
    if np.sum(valid_mask) == 0:
        return float('inf'), float('inf')
    
    mae = np.mean(np.abs(predictions[valid_mask] - targets[valid_mask]))
    rmse = np.sqrt(np.mean((predictions[valid_mask] - targets[valid_mask]) ** 2))
    
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CHGNet v0.4.2")
    parser.add_argument("--data-dir", default="data/processed/chgnet")
    parser.add_argument("--output-dir", default="data/artifacts/chgnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    structures, energies, train_idx, val_idx, test_idx = load_dataset(args.data_dir)
    
    # Split data
    train_structures = [structures[i] for i in train_idx]
    train_energies = [float(energies[i]) for i in train_idx]
    train_forces = [create_dummy_forces(s) for s in train_structures]
    
    val_structures = [structures[i] for i in val_idx]
    val_energies = [float(energies[i]) for i in val_idx]
    val_forces = [create_dummy_forces(s) for s in val_structures]
    
    print(f"\nCreating StructureData datasets...")
    
    # Create StructureData - CHGNet v0.4.2 API takes lists
    try:
        train_dataset = StructureData(
            structures=train_structures,
            energies=train_energies,
            forces=train_forces,
        )
        print(f"Train dataset created: {len(train_dataset)} samples")
    except Exception as e:
        print(f"Error creating train dataset: {e}")
        raise
    
    try:
        val_dataset = StructureData(
            structures=val_structures,
            energies=val_energies,
            forces=val_forces,
        )
        print(f"Val dataset created: {len(val_dataset)} samples")
    except Exception as e:
        print(f"Error creating val dataset: {e}")
        raise
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    
    # Load pretrained CHGNet
    print("\nLoading pretrained CHGNet...")
    model = CHGNet.load()
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    print(f"\nStarting training for {args.epochs} epochs...")
    
    trainer = Trainer(
        model=model,
        targets="e",  # Energy only
        optimizer="AdamW",
        scheduler="CosLR",
        criterion="MAE",
        epochs=args.epochs,
        learning_rate=args.lr,
        use_device=device,
        print_freq=10,
    )
    
    # Train
    trainer.train(train_loader, val_loader, save_dir=str(output_path))
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    
    model.eval()
    mae, rmse = evaluate_model(model, test_structures, test_energies)
    
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test MAE: {mae:.4f} eV/atom")
    print(f"Test RMSE: {rmse:.4f} eV/atom")
    print(f"Model saved to: {output_path}")
    print("=" * 60)
    
    # Save results
    with open(output_path / "test_results.json", "w") as f:
        json.dump({
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "epochs": args.epochs,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_test": len(test_idx),
        }, f, indent=2)


if __name__ == "__main__":
    main()
