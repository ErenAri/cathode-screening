"""
Fine-tune CHGNet on cathode E_hull prediction.

CHGNet is a state-of-the-art universal potential that can be fine-tuned
for specific property prediction tasks like energy above hull.

Usage:
    python scripts/21_finetune_chgnet.py --epochs 100 --batch-size 8
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# CHGNet imports
from chgnet.model import CHGNet
from chgnet.graph import CrystalGraphConverter

# For structures
from pymatgen.core import Structure


class CathodeDataset(Dataset):
    """Dataset for cathode E_hull prediction with CHGNet."""
    
    def __init__(
        self,
        structures: List[Structure],
        energies: np.ndarray,
        converter: CrystalGraphConverter,
    ):
        self.structures = structures
        self.energies = energies
        self.converter = converter
        
        # Pre-convert structures to graphs for faster training
        self.graphs = []
        print("Converting structures to graphs...")
        for struct in tqdm(structures, desc="Building graphs"):
            try:
                graph = self.converter(struct)
                self.graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to convert structure: {e}")
                self.graphs.append(None)
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.energies[idx]


def collate_fn(batch):
    """Custom collate function for CHGNet graphs."""
    graphs, energies = zip(*batch)
    
    # Filter out None graphs
    valid_data = [(g, e) for g, e in zip(graphs, energies) if g is not None]
    if not valid_data:
        return None, None
    
    graphs, energies = zip(*valid_data)
    
    # CHGNet handles batching internally
    return list(graphs), torch.tensor(energies, dtype=torch.float32)


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


class CHGNetEhullPredictor(nn.Module):
    """CHGNet with custom head for E_hull prediction."""
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pretrained CHGNet
        self.chgnet = CHGNet.load() if pretrained else CHGNet()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.chgnet.parameters():
                param.requires_grad = False
        
        # Get the output dimension from CHGNet's final layer
        # CHGNet predicts energy, so we can use it directly or add a custom head
        self.use_custom_head = False
        
        if self.use_custom_head:
            # Custom head for E_hull prediction
            hidden_dim = 64
            self.head = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, graphs):
        """Forward pass for batch of graphs."""
        predictions = []
        
        for graph in graphs:
            # CHGNet prediction
            result = self.chgnet.predict_structure(
                graph.structure if hasattr(graph, 'structure') else graph,
                return_site_energies=False,
            )
            
            # Extract energy prediction (per atom)
            energy = result.get("e", 0.0)
            predictions.append(energy)
        
        predictions = torch.tensor(predictions, dtype=torch.float32)
        
        if self.use_custom_head:
            predictions = self.head(predictions.unsqueeze(-1)).squeeze(-1)
        
        return predictions


def train_epoch(
    model: CHGNet,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    clip_grad: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for graphs, targets in pbar:
        if graphs is None:
            continue
        
        targets = targets.to(device)
        
        # Forward pass - CHGNet expects structures
        predictions = []
        for graph in graphs:
            try:
                pred = model.predict_structure(
                    graph.structure if hasattr(graph, 'structure') else graph,
                    return_site_energies=False,
                )
                predictions.append(pred.get("e", 0.0))
            except Exception as e:
                predictions.append(0.0)
        
        predictions = torch.tensor(predictions, dtype=torch.float32, device=device)
        
        # Loss (MAE for E_hull)
        loss = torch.abs(predictions - targets).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: CHGNet,
    val_loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for graphs, targets in tqdm(val_loader, desc="Evaluating"):
        if graphs is None:
            continue
        
        targets = targets.to(device)
        
        predictions = []
        for graph in graphs:
            try:
                pred = model.predict_structure(
                    graph.structure if hasattr(graph, 'structure') else graph,
                    return_site_energies=False,
                )
                predictions.append(pred.get("e", 0.0))
            except Exception:
                predictions.append(0.0)
        
        predictions = torch.tensor(predictions, dtype=torch.float32, device=device)
        
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CHGNet on E_hull prediction")
    parser.add_argument("--data-dir", default="data/processed/chgnet")
    parser.add_argument("--output-dir", default="data/artifacts/chgnet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Using device: {args.device}")
    
    # Load data
    structures, energies, train_idx, val_idx, test_idx = load_dataset(args.data_dir)
    
    # Create datasets
    converter = CrystalGraphConverter()
    
    train_structures = [structures[i] for i in train_idx]
    train_energies = energies[train_idx]
    val_structures = [structures[i] for i in val_idx]
    val_energies = energies[val_idx]
    
    train_dataset = CathodeDataset(train_structures, train_energies, converter)
    val_dataset = CathodeDataset(val_structures, val_energies, converter)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Load model
    print("Loading pretrained CHGNet...")
    model = CHGNet.load()
    model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "val_rmse": []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        history["train_loss"].append(train_loss)
        
        # Evaluate
        val_mae, val_rmse = evaluate(model, val_loader, args.device)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        
        # Update scheduler
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f} eV/atom, Val RMSE: {val_rmse:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }, output_path / "chgnet_best.pt")
            print(f"✓ Saved best model (MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Save final model and history
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, output_path / "chgnet_final.pt")
    
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation MAE: {best_val_mae:.4f} eV/atom")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
