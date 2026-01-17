"""
Fine-tune ALIGNN on cathode E_hull prediction.

ALIGNN (Atomistic Line Graph Neural Network) is a state-of-the-art GNN
for materials property prediction.

Usage:
    python scripts/23_finetune_alignn.py --epochs 100 --seed 42
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pymatgen.core import Structure


class CathodeDataset(Dataset):
    """Dataset for cathode E_hull prediction."""
    
    def __init__(
        self,
        structures: List[Structure],
        energies: np.ndarray,
    ):
        self.structures = structures
        self.energies = energies
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        return self.structures[idx], self.energies[idx]


def load_dataset(
    data_dir: str = "data/processed/chgnet",
) -> Tuple[List[Structure], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load prepared dataset."""
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


def structure_to_jarvis(structure: Structure):
    """Convert pymatgen Structure to JARVIS Atoms."""
    from jarvis.core.atoms import Atoms
    
    return Atoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[str(s.specie) for s in structure.sites],
        cartesian=False,
    )


def prepare_alignn_data(
    structures: List[Structure],
    energies: np.ndarray,
    output_dir: str,
):
    """Prepare data in ALIGNN format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = []
    for i, (struct, energy) in enumerate(tqdm(zip(structures, energies), 
                                               total=len(structures), desc="Converting")):
        try:
            atoms = structure_to_jarvis(struct)
            
            # Save structure
            struct_file = f"structure_{i}.json"
            atoms.write_json(output_path / struct_file)
            
            data.append({
                "id": f"structure_{i}",
                "file": struct_file,
                "target": float(energy),
            })
        except Exception as e:
            print(f"Warning: Failed to convert {i}: {e}")
    
    # Save manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Prepared {len(data)} structures in {output_path}")
    return data


@torch.no_grad()
def evaluate_model(model, val_structures, val_energies, device):
    """Evaluate ALIGNN model."""
    from alignn.graphs import Graph
    
    model.eval()
    predictions = []
    
    for struct in tqdm(val_structures, desc="Evaluating", leave=False):
        try:
            atoms = structure_to_jarvis(struct)
            g, lg = Graph.atom_dgl_multigraph(atoms)
            g = g.to(device)
            lg = lg.to(device)
            
            pred = model([g, lg])
            predictions.append(pred.item())
        except Exception:
            predictions.append(np.nan)
    
    predictions = np.array(predictions)
    valid = ~np.isnan(predictions)
    
    mae = np.mean(np.abs(predictions[valid] - val_energies[valid]))
    rmse = np.sqrt(np.mean((predictions[valid] - val_energies[valid])**2))
    
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ALIGNN")
    parser.add_argument("--data-dir", default="data/processed/chgnet")
    parser.add_argument("--output-dir", default="data/artifacts/alignn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    structures, energies, train_idx, val_idx, test_idx = load_dataset(args.data_dir)
    
    # Get splits
    train_structures = [structures[i] for i in train_idx]
    train_energies = energies[train_idx]
    val_structures = [structures[i] for i in val_idx]
    val_energies = energies[val_idx]
    
    # Load ALIGNN model
    print("\nLoading pretrained ALIGNN...")
    from alignn.pretrained import get_figshare_model
    from alignn.models.alignn import ALIGNN
    
    # Load default config and modify for our task
    model = get_figshare_model(name="jv_formation_energy_peratom_alignn")
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.L1Loss()
    
    # Training loop
    from alignn.graphs import Graph
    
    best_val_mae = float("inf")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        # Shuffle training data
        perm = np.random.permutation(len(train_structures))
        
        pbar = tqdm(perm[:min(1000, len(perm))], desc=f"Epoch {epoch}")  # Sample subset
        for idx in pbar:
            struct = train_structures[idx]
            target = train_energies[idx]
            
            try:
                atoms = structure_to_jarvis(struct)
                g, lg = Graph.atom_dgl_multigraph(atoms)
                g = g.to(device)
                lg = lg.to(device)
                
                optimizer.zero_grad()
                pred = model([g, lg])
                loss = criterion(pred, torch.tensor([[target]], device=device))
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-10:]):.4f}"})
            except Exception:
                continue
        
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            val_mae, val_rmse = evaluate_model(
                model, val_structures[:200], val_energies[:200], device
            )
            
            print(f"\nEpoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f}, "
                  f"Val MAE = {val_mae:.4f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "seed": args.seed,
                }, output_path / f"alignn_best_seed{args.seed}.pt")
                print(f"✓ Saved best model")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    
    test_structures = [structures[i] for i in test_idx]
    test_energies = energies[test_idx]
    
    mae, rmse = evaluate_model(model, test_structures, test_energies, device)
    
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test MAE: {mae:.4f} eV/atom")
    print(f"Test RMSE: {rmse:.4f} eV/atom")
    print(f"Best val MAE: {best_val_mae:.4f} eV/atom")
    print("=" * 60)
    
    # Save results
    with open(output_path / f"results_seed{args.seed}.json", "w") as f:
        json.dump({
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "best_val_mae": float(best_val_mae),
            "seed": args.seed,
            "epochs": args.epochs,
        }, f, indent=2)


if __name__ == "__main__":
    main()
