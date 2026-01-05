#!/usr/bin/env python3
"""
Build OOD detection artifacts from trained ensemble.

Usage:
    python scripts/07_build_ood_artifacts.py \
        --ensemble-dir data/artifacts/ensemble \
        --data-config configs/train_cgcnn_ehull.yaml \
        --output-dir data/artifacts/ood
"""
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import shared utilities
from shared_imports import (
    CGCNN, Normalizer, GraphNPZDataset, collate, load_cfg, get_device
)

from cathode_screening.inference.ood import (
    compute_train_stats,
    save_ood_artifacts,
    COMPOSITION_VOCAB,
)


def load_ensemble(ensemble_dir: Path, device: torch.device):
    """Load ensemble models."""
    meta_path = ensemble_dir / "ensemble_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    models = []
    normalizers = []
    
    for member in meta["members"]:
        ckpt_path = Path(member["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["cfg"]
        
        model = CGCNN(
            node_in=int(cfg["model"].get("node_in_dim", 6)),
            node_dim=int(cfg["model"]["node_embed_dim"]),
            edge_dim=int(cfg["model"]["edge_rbf_bins"]),
            layers=int(cfg["model"]["message_passing_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            pooling=str(cfg["model"]["pooling"])
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        
        norm = Normalizer()
        norm.load_state_dict(ckpt["normalizer"])
        normalizers.append(norm)
    
    return models, normalizers, meta


def extract_embedding(model, x, src, dst, e, batch):
    """Extract graph-level embedding from model."""
    x = model.embed(x)
    for blk in model.blocks:
        x = blk(x, src, dst, e)
    g = model.pool(x, batch)
    return g


@torch.no_grad()
def compute_train_embeddings_and_disagreements(
    models: List,
    normalizers: List,
    train_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute embeddings and ensemble disagreements for training set.
    
    Returns:
        embeddings: [N_train, embedding_dim * K] concatenated embeddings
        disagreements: [N_train] std of q50 predictions per sample
    """
    all_embeddings = []
    all_disagreements = []
    
    for batch_data in train_loader:
        x, src, dst, e, b, y = batch_data
        x = x.to(device)
        src = src.to(device)
        dst = dst.to(device)
        e = e.to(device)
        b = b.to(device)
        
        # Collect predictions and embeddings from each member
        q50_per_member = []
        emb_per_member = []
        
        for model, norm in zip(models, normalizers):
            model.eval()
            q10, q50, q90, _, _ = model(x, src, dst, e, b)
            q50_denorm = norm.denorm(q50).cpu().numpy()
            q50_per_member.append(q50_denorm)
            
            emb = extract_embedding(model, x, src, dst, e, b)
            emb_per_member.append(emb.cpu().numpy())
        
        # Stack and compute disagreement
        q50_stack = np.stack(q50_per_member, axis=0)  # [K, batch_size]
        disagreements = np.std(q50_stack, axis=0)  # [batch_size]
        all_disagreements.extend(disagreements)
        
        # Concatenate embeddings across members
        emb_concat = np.concatenate(emb_per_member, axis=-1)  # [batch_size, dim*K]
        all_embeddings.append(emb_concat)
    
    embeddings = np.vstack(all_embeddings)
    disagreements = np.array(all_disagreements)
    
    return embeddings, disagreements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", required=True, help="Path to ensemble directory")
    ap.add_argument("--data-config", required=True, help="Training config for data loading")
    ap.add_argument("--output-dir", default="data/artifacts/ood", help="Output directory")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, or cpu")
    args = ap.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    # Load ensemble
    ensemble_dir = Path(args.ensemble_dir)
    models, normalizers, meta = load_ensemble(ensemble_dir, device)
    print(f"Loaded {len(models)} ensemble members")
    
    # Load training data
    cfg = load_cfg(args.data_config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    ds_name = cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)
    
    splits_path = Path(cfg["data"]["splits_manifest"])
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    train_ids = set(splits["train"])
    df_train = df[df["material_id"].isin(train_ids)].reset_index(drop=True)
    
    target = cfg["data"]["target"]["name"]
    train_formulas = df_train["formula_pretty"].tolist()
    
    print(f"Training set: {len(df_train)} samples")
    print(f"Composition vocabulary: {COMPOSITION_VOCAB}")
    
    # Create data loader with GPU optimizations
    train_ds = GraphNPZDataset(df_train, target_col=target)
    pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False, 
        collate_fn=collate,
        pin_memory=pin_memory,
        num_workers=0
    )
    
    # Compute embeddings and disagreements
    print("\nComputing embeddings and disagreements...")
    embeddings, disagreements = compute_train_embeddings_and_disagreements(
        models, normalizers, train_loader, device
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Disagreements: mean={disagreements.mean():.4f}, std={disagreements.std():.4f}")
    
    # Compute OOD statistics
    print("\nComputing OOD statistics...")
    stats, comp_mu, comp_cov_inv = compute_train_stats(
        train_formulas, embeddings, disagreements
    )
    
    print("\nOOD Statistics:")
    print(f"  Composition: mean={stats['comp_mean']:.3f}, τ_p95={stats['tau_comp_p95']:.3f}")
    print(f"  Embedding: mean={stats['emb_mean']:.3f}, τ_p95={stats['tau_emb_p95']:.3f}")
    print(f"  Disagreement: mean={stats['disagree_mean']:.4f}, τ_p95={stats['tau_disagree_p95']:.4f}")
    
    # Save artifacts
    output_dir = Path(args.output_dir)
    save_ood_artifacts(output_dir, stats, comp_mu, comp_cov_inv, embeddings)
    
    print(f"\n✓ Saved OOD artifacts to {output_dir}")
    
    # GPU memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
