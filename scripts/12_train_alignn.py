#!/usr/bin/env python
"""
Train ALIGNN model on cathode screening data.

Uses the prepared POSCAR files and id_prop.csv from ALIGNN format.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from jarvis.core.atoms import Atoms as JarvisAtoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig


class CathodeDataset(Dataset):
    """Custom dataset for cathode materials."""
    
    def __init__(self, folder_path, cutoff=8.0, max_neighbors=12):
        self.folder = Path(folder_path)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        # Load id_prop.csv
        id_prop_path = self.folder / "id_prop.csv"
        if not id_prop_path.exists():
            raise FileNotFoundError(f"id_prop.csv not found in {self.folder}")
        
        self.df = pd.read_csv(id_prop_path, header=None, names=["id", "target"])
        print(f"  Loaded {len(self.df)} samples from {folder_path}")
        
        # Pre-load all structures
        self.samples = []
        for _, row in self.df.iterrows():
            poscar_path = self.folder / f"{row['id']}.vasp"
            if poscar_path.exists():
                self.samples.append({
                    "id": row["id"],
                    "poscar_path": str(poscar_path),
                    "target": row["target"],
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load atoms
        atoms = JarvisAtoms.from_poscar(sample["poscar_path"])
        
        # Create graph
        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        
        # Get lattice matrix as tensor
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        
        return g, lg, lattice, torch.tensor(sample["target"], dtype=torch.float32)


def collate_fn(batch):
    """Collate function for DataLoader."""
    import dgl
    
    graphs = [item[0] for item in batch]
    line_graphs = [item[1] for item in batch]
    lattices = torch.stack([item[2] for item in batch])
    targets = torch.stack([item[3] for item in batch])
    
    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)
    
    return batched_graph, batched_line_graph, lattices, targets


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for g, lg, lat, targets in dataloader:
        g = g.to(device)
        lg = lg.to(device)
        lat = lat.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        out = model([g, lg, lat])
        
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out
        
        loss = criterion(pred.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for g, lg, lat, targets in dataloader:
            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            targets = targets.to(device)
            
            out = model([g, lg, lat])
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            
            loss = criterion(pred.squeeze(), targets)
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.extend(pred.squeeze().cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss / max(n_batches, 1), mae


def main():
    parser = argparse.ArgumentParser(description="Train ALIGNN on cathode data")
    parser.add_argument("--train-folder", type=str, default="data/alignn_format/train")
    parser.add_argument("--val-folder", type=str, default="data/alignn_format/val")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/alignn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("ALIGNN Training")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CathodeDataset(args.train_folder)
    val_dataset = CathodeDataset(args.val_folder)
    
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
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # Initialize model with correct feature dimensions
    # ALIGNN default graph uses 92-dim atom features
    config = ALIGNNConfig(
        name="alignn",
        atom_input_features=92,  # Default CGCNN atom features
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=64,
        output_features=1,
        gcn_layers=2,
        alignn_layers=2,
        classification=False,
    )
    model = ALIGNN(config)
    model = model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = torch.nn.L1Loss()  # MAE loss for E_hull
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_mae = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - start
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_mae": val_mae,
            }, output_dir / "best.pt")
            print(f"  → Saved best model (MAE: {val_mae:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best Val MAE: {best_val_mae:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
