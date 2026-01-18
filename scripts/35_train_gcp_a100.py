"""
GCP A100 Training Script for CHGNet Ensemble on Full MPTrj + Li-Cathode Data.

This script is optimized for A100 (40GB) GPUs on GCP:
- Large batch size (48-64)
- Mixed precision (bf16)
- Efficient data loading

Training Strategy:
1. Pretrain on full MPTrj (1.58M structures)
2. Fine-tune on Li-cathode subset (35K)
3. Train 5-model ensemble for uncertainty quantification

Usage:
    # On GCP VM with A100:
    python scripts/35_train_gcp_a100.py --phase pretrain
    python scripts/35_train_gcp_a100.py --phase finetune

Cost Estimate:
    - Pretrain (1.58M): ~20-30 hours @ $3.50/hr = ~$70-105
    - Fine-tune (35K × 5): ~5-10 hours @ $3.50/hr = ~$17-35
    - Total: ~$90-140 (~3000-5000 TL)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Check for CUDA/bf16 support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

print(f"Device: {DEVICE}")
print(f"Mixed Precision (bf16): {USE_BF16}")


# A100 Optimized Config
A100_CONFIG = {
    # Data loading
    "batch_size": 48,  # A100 can handle larger batches
    "num_workers": 12,  # More workers for faster data loading
    "pin_memory": True,
    
    # Training
    "learning_rate": 2e-3,  # Scale with batch size
    "weight_decay": 1e-5,
    "epochs_pretrain": 30,  # Full MPTrj
    "epochs_finetune": 50,  # Li-cathode
    
    # Ensemble
    "n_models": 5,
    "seeds": [42, 123, 456, 789, 1024],
    
    # Mixed precision
    "use_bf16": USE_BF16,
    
    # Checkpointing
    "save_every": 5,  # Save every 5 epochs
    "checkpoint_dir": "checkpoints/gcp_a100",
}


def load_mptrj_data(mptrj_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load full MPTrj dataset (1.58M structures).
    
    The file is large (11GB), so we stream it efficiently.
    """
    print(f"Loading MPTrj from {mptrj_path}...")
    start = time.time()
    
    with open(mptrj_path, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} structures in {time.time() - start:.1f}s")
    
    if max_samples:
        data = data[:max_samples]
        print(f"  Using {len(data)} samples")
    
    return data


def load_li_cathode_data(data_dir: str) -> Dict[str, List]:
    """
    Load Li-cathode subset data from multiple sources.
    """
    import pandas as pd
    
    sources = [
        "oqmd/oqmd_li_cathodes.parquet",
        "mp_2024/mp_li_cathodes_2024.parquet",
        "mptrj/mptrj_li_cathodes.parquet",
        "nomad/nomad_li_cathodes.parquet",
        "wbm/wbm_li_cathodes.parquet",
    ]
    
    all_data = []
    for source in sources:
        path = Path(data_dir) / source
        if path.exists():
            df = pd.read_parquet(path)
            all_data.extend(df.to_dict('records'))
            print(f"  {source}: {len(df)} materials")
    
    print(f"Total Li-cathode: {len(all_data)}")
    return all_data


class MPTrjDataset(Dataset):
    """Dataset for MPTrj training data."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Return structure dict and energy label
        return item.get("structure", {}), item.get("energy_per_atom", 0.0)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    config: Dict,
) -> float:
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if config["use_bf16"]:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    return total_loss / n_batches


def train_model(
    model_idx: int,
    seed: int,
    data: List[Dict],
    config: Dict,
    phase: str = "pretrain",
) -> Dict:
    """
    Train a single CHGNet model.
    """
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    
    print(f"\n{'='*60}")
    print(f"Training Model {model_idx + 1}/5 (seed={seed}, phase={phase})")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load pretrained CHGNet
    model = CHGNet.load()
    model = model.to(DEVICE)
    
    # Create data splits
    n_val = int(len(data) * 0.1)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create trainer
    epochs = config["epochs_pretrain"] if phase == "pretrain" else config["epochs_finetune"]
    
    trainer = Trainer(
        model=model,
        targets="ef",  # energy and forces
        optimizer="AdamW",
        scheduler="CosLR",
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        epochs=epochs,
        batch_size=config["batch_size"],
        use_device=DEVICE,
    )
    
    # Train
    checkpoint_dir = Path(config["checkpoint_dir"]) / f"{phase}_model_{model_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.train(
        train_data,
        val_data,
        save_dir=str(checkpoint_dir),
    )
    
    # Get best model metrics
    results = {
        "model_idx": model_idx,
        "seed": seed,
        "phase": phase,
        "best_mae": trainer.best_model_val_mae,
        "checkpoint": str(checkpoint_dir),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GCP A100 CHGNet Training")
    parser.add_argument(
        "--phase", 
        choices=["pretrain", "finetune", "both"],
        default="both",
        help="Training phase"
    )
    parser.add_argument(
        "--mptrj-path",
        default="data/external/mptrj_full/MPtrj_2022.9_full.json",
        help="Path to full MPTrj JSON"
    )
    parser.add_argument(
        "--li-cathode-dir",
        default="data/external",
        help="Directory containing Li-cathode parquet files"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for testing (default: use all)"
    )
    args = parser.parse_args()
    
    config = A100_CONFIG.copy()
    
    print("=" * 60)
    print("GCP A100 CHGNet Training")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    all_results = []
    
    # Phase 1: Pretrain on full MPTrj
    if args.phase in ["pretrain", "both"]:
        print("\n" + "=" * 60)
        print("PHASE 1: PRETRAIN ON FULL MPTRJ (1.58M structures)")
        print("=" * 60)
        
        mptrj_data = load_mptrj_data(args.mptrj_path, args.max_samples)
        
        for i, seed in enumerate(config["seeds"]):
            results = train_model(i, seed, mptrj_data, config, phase="pretrain")
            all_results.append(results)
    
    # Phase 2: Fine-tune on Li-cathode
    if args.phase in ["finetune", "both"]:
        print("\n" + "=" * 60)
        print("PHASE 2: FINE-TUNE ON LI-CATHODE (35K materials)")
        print("=" * 60)
        
        li_data = load_li_cathode_data(args.li_cathode_dir)
        
        for i, seed in enumerate(config["seeds"]):
            results = train_model(i, seed, li_data, config, phase="finetune")
            all_results.append(results)
    
    # Save results
    results_path = Path(config["checkpoint_dir"]) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
