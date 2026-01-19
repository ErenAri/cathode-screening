"""
CHGNet Training Script Optimized for NVIDIA H100 (80GB VRAM).

H100 Optimizations:
- Larger batch size (128 vs 32 on L4)
- BF16 mixed precision (native H100 support)
- More data workers (16 vs 8)
- Faster training (~4x faster than L4)

Usage:
    python scripts/40_train_gcp_h100.py --phase finetune
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Check CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# H100-specific: Check BF16 support
HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Mixed Precision (bf16): {HAS_BF16}")

# H100-optimized configuration (80GB VRAM)
H100_CONFIG = {
    "batch_size": 256,        # 8x larger than L4 (80GB allows this)
    "num_workers": 16,        # Match CPU cores
    "pin_memory": True,
    "learning_rate": 1.5e-3,  # Higher LR for larger batch (linear scaling)
    "weight_decay": 1e-5,
    "epochs_pretrain": 30,
    "epochs_finetune": 60,    # More epochs for better convergence
    "n_models": 7,            # 7 models for better ensemble
    "seeds": [42, 123, 456, 789, 1024, 2048, 3072],  # 7 seeds
    "use_bf16": True,         # Native BF16 on H100
    "save_every": 10,         # Save every 10 epochs
    "gradient_clip": 1.0,     # Prevent gradient explosion
    "warmup_epochs": 3,       # Learning rate warmup
    "checkpoint_dir": "checkpoints/gcp_h100",
}


def load_li_cathode_data(data_dir: str) -> List[Dict]:
    """Load Li-cathode training data with full crystal structures."""
    
    training_path = Path(data_dir) / "training" / "li_cathode_structures.json"
    if training_path.exists():
        print(f"Loading training data from {training_path}...")
        with open(training_path, 'r') as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} structures with energies")
        return all_data
    
    pickle_path = Path(data_dir) / "training" / "li_cathode_structures.pkl"
    if pickle_path.exists():
        import pickle
        print(f"Loading training data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            all_data = pickle.load(f)
        print(f"Loaded {len(all_data)} structures with energies")
        return all_data
    
    print(f"ERROR: No training data found in {data_dir}/training/")
    return []


def train_model(
    model_idx: int,
    seed: int,
    data: List[Dict],
    config: Dict,
    phase: str = "finetune",
) -> Dict:
    """Train a single CHGNet model on H100."""
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData, get_train_val_test_loader
    from pymatgen.core import Structure
    
    print(f"\n{'='*60}")
    print(f"Training Model {model_idx + 1}/{config['n_models']} (seed={seed}, phase={phase})")
    print(f"H100 Mode: batch_size={config['batch_size']}, lr={config['learning_rate']}, bf16={config['use_bf16']}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = CHGNet.load()
    model = model.to(DEVICE)
    
    print("  Converting data to StructureData format...")
    structures = []
    energies = []
    forces_list = []
    
    for item in data:
        try:
            if "structure" in item and isinstance(item["structure"], dict):
                struct = Structure.from_dict(item["structure"])
            elif "structure" in item and isinstance(item["structure"], Structure):
                struct = item["structure"]
            else:
                continue
                
            energy = item.get("energy_per_atom", item.get("energy", None))
            if energy is None:
                continue
                
            structures.append(struct)
            energies.append(energy)
            
            # Create zero-forces (CHGNet requires forces as positional arg)
            forces = item.get("forces", None)
            if forces is None:
                forces = np.zeros((len(struct.sites), 3)).tolist()
            forces_list.append(forces)
            
        except Exception as e:
            continue
    
    print(f"  Converted {len(structures)} structures")
    
    if len(structures) < 100:
        print(f"  WARNING: Only {len(structures)} valid structures. Skipping.")
        return {"model_idx": model_idx, "seed": seed, "phase": phase, "error": "insufficient_data"}
    
    dataset = StructureData(
        structures=structures,
        energies=energies,
        forces=forces_list,
    )
    
    # H100 can handle larger batches
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset,
        batch_size=config["batch_size"],
        train_ratio=0.9,
        val_ratio=0.1,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    epochs = config["epochs_pretrain"] if phase == "pretrain" else config["epochs_finetune"]
    
    trainer = Trainer(
        model=model,
        targets="e",  # Energy-only
        optimizer="AdamW",
        scheduler="CosLR",
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        epochs=epochs,
        use_device=DEVICE,
    )
    
    checkpoint_dir = Path(config["checkpoint_dir"]) / f"{phase}_model_{model_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    trainer.train(
        train_loader,
        val_loader,
        save_dir=str(checkpoint_dir),
    )
    
    train_time = time.time() - start_time
    print(f"  Training time: {train_time/60:.1f} minutes")
    
    return {
        "model_idx": model_idx,
        "seed": seed,
        "phase": phase,
        "checkpoint": str(checkpoint_dir),
        "train_time_min": train_time / 60,
    }


def main():
    parser = argparse.ArgumentParser(description="GCP H100 CHGNet Training")
    parser.add_argument("--phase", choices=["pretrain", "finetune", "both"], default="finetune")
    parser.add_argument("--li-cathode-dir", default="data")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    config = H100_CONFIG.copy()
    
    print("=" * 60)
    print("GCP H100 CHGNet Training (80GB VRAM Optimization)")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    all_results = []
    
    if args.phase in ["finetune", "both"]:
        print("\nPHASE: FINE-TUNE on Li-cathode data")
        li_data = load_li_cathode_data(args.li_cathode_dir)
        
        if args.max_samples:
            li_data = li_data[:args.max_samples]
        
        for i, seed in enumerate(config["seeds"]):
            results = train_model(i, seed, li_data, config, phase="finetune")
            all_results.append(results)
    
    # Save results
    results_dir = Path(config["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
