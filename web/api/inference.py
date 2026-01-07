"""
Inference service for CathodeScreen API.

Loads the trained CGCNN ensemble and provides prediction methods
for single structure and batch inference.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from pymatgen.core import Structure


# Add src to path
SRC_PATH = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Stability thresholds
THRESH_STABLE = 0.05


# ============================================================================
# Model Definition (matching training)
# ============================================================================

class CGCNNBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.lin_upd = nn.Linear(node_dim * 2, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, src, dst, e_attr):
        msg_input = torch.cat([x[src], x[dst], e_attr], dim=-1)
        msg = torch.sigmoid(self.lin_msg(msg_input)) * torch.relu(msg_input[:, :x.size(1)])
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        upd = torch.cat([x, agg], dim=-1)
        return x + self.bn(torch.relu(self.lin_upd(upd)))


class CGCNN(nn.Module):
    def __init__(self, node_in: int, node_dim: int, edge_dim: int, layers: int, dropout: float, pooling: str):
        super().__init__()
        self.embed = nn.Linear(node_in, node_dim)
        self.blocks = nn.ModuleList([CGCNNBlock(node_dim, edge_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        # Heads matching training checkpoint
        self.head_q50 = nn.Linear(node_dim, 1)  # Median prediction
        self.head_q10 = nn.Linear(node_dim, 1)  # Lower quantile
        self.head_q90 = nn.Linear(node_dim, 1)  # Upper quantile
        self.head_p_stable = nn.Linear(node_dim, 1)
        self.head_p_metastable = nn.Linear(node_dim, 1)

    def pool(self, x, batch_index):
        if batch_index is None:
            return x.mean(dim=0, keepdim=True)
        n_graphs = batch_index.max().item() + 1
        out = torch.zeros(n_graphs, x.size(1), device=x.device)
        counts = torch.zeros(n_graphs, device=x.device)
        out.index_add_(0, batch_index, x)
        counts.index_add_(0, batch_index, torch.ones(x.size(0), device=x.device))
        return out / counts.clamp(min=1).unsqueeze(1)

    def forward(self, x, src, dst, e_attr, batch_index):
        h = torch.relu(self.embed(x))
        for blk in self.blocks:
            h = blk(h, src, dst, e_attr)
            h = self.dropout(h)
        h = self.pool(h, batch_index)
        q50 = self.head_q50(h)
        q10 = self.head_q10(h)
        q90 = self.head_q90(h)
        p_stable = self.head_p_stable(h)
        p_metastable = self.head_p_metastable(h)
        return q50, q10, q90, p_stable, p_metastable


class Normalizer:
    """Target normalizer matching training."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def denorm(self, x):
        return x * self.std + self.mean

    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict.get("mean", 0.0)
        self.std = state_dict.get("std", 1.0)


# ============================================================================
# Graph Construction from Structure
# ============================================================================

# ============================================================================
# Graph Construction from Structure
# ============================================================================

def element_features(symbol: str) -> np.ndarray:
    """
    Get 6-dimensional physical embedding for an element.
    [AtomicNumber, Electronegativity, Radius, Mass, Row, Group]
    """
    from pymatgen.core import Element
    e = Element(symbol)
    
    # Handle missing data with valid defaults (0.0)
    # Heuristic scaling to 0-1 range based on training data distribution
    z = float(e.Z) / 100.0
    x = (float(e.X) if e.X is not None else 0.0) / 4.0
    r = (float(e.atomic_radius) if e.atomic_radius is not None else 0.0) / 3.0
    m = (float(e.atomic_mass) if e.atomic_mass is not None else 0.0) / 250.0
    row = (float(e.row) if e.row is not None else 0.0) / 9.0
    group = (float(e.group) if e.group is not None else 0.0) / 18.0
    
    return np.array([z, x, r, m, row, group], dtype=np.float32)


def structure_to_graph(structure: Structure, r_cutoff: float = 8.0, n_rbf: int = 41, max_neighbors: int = 12) -> Dict:
    """Convert pymatgen Structure to graph tensors."""
    # Node features: physical properties (N, 6)
    node_feats = np.stack([element_features(str(site.specie)) for site in structure], axis=0)
    
    # Get neighbors within cutoff
    all_neighbors = structure.get_all_neighbors(r_cutoff, include_index=True)
    
    src_list = []
    dst_list = []
    dist_list = []
    
    for i, neighbors in enumerate(all_neighbors):
        # Sort by distance and take top k
        neighbors = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]
        
        for neighbor in neighbors:
            dist = neighbor[1]
            j = neighbor[2]
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(dist)
    
    if len(src_list) == 0:
        # No edges found, add self-loops
        for i in range(len(structure)):
            src_list.append(i)
            dst_list.append(i)
            dist_list.append(0.1)
    
    # Edge features: RBF encoding of distances
    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    distances = np.array(dist_list, dtype=np.float32)
    
    # RBF encoding
    centers = np.linspace(0, r_cutoff, n_rbf)
    gamma = 1.0 / (0.5 ** 2)  # Gaussian width
    edge_feats = np.exp(-gamma * (distances[:, None] - centers[None, :]) ** 2)
    
    return {
        "node_feats": node_feats.astype(np.float32),
        "edge_index_src": src,
        "edge_index_dst": dst,
        "edge_attr": edge_feats.astype(np.float32),
    }


# ============================================================================
# Ensemble Predictor
# ============================================================================

class EnsemblePredictor:
    """Loads and runs ensemble of CGCNN models."""
    
    def __init__(self, ensemble_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.models = []
        self.normalizer = Normalizer()
        
        ensemble_dir = Path(ensemble_dir)
        
        # Load ensemble metadata
        meta_path = ensemble_dir / "ensemble_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
        
        # Find and load all member models
        member_dirs = sorted(ensemble_dir.glob("member_*"))
        for member_dir in member_dirs:
            # Find checkpoint - structure is member_X/member_X/best.pt
            member_name = member_dir.name  # e.g., "member_0"
            ckpt_path = member_dir / member_name / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = member_dir / "best.pt"  # Fallback
            if not ckpt_path.exists():
                print(f"Warning: No checkpoint found in {member_dir}")
                continue
            
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt["state_dict"]
            
            # Infer dimensions from state_dict
            embed_shape = state_dict["embed.weight"].shape  # [node_dim, node_in]
            node_dim = embed_shape[0]
            node_in = embed_shape[1]
            
            # Count layers
            n_layers = sum(1 for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".lin_msg.weight"))
            
            # Edge dim from first block
            edge_dim = state_dict["blocks.0.lin_msg.weight"].shape[1] - 2 * node_dim
            
            model_cfg = {
                "node_in": node_in,
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "layers": n_layers,
                "dropout": 0.1,  # Not stored in state_dict
                "pooling": "mean",
            }
            
            # Create and load model
            model = CGCNN(**model_cfg)
            model.load_state_dict(state_dict, strict=False)  # Ignore extra keys like head_d_lo/hi
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
            # Load normalizer from first member
            if len(self.models) == 1 and "normalizer" in ckpt:
                norm_data = ckpt["normalizer"]
                if isinstance(norm_data, dict):
                    self.normalizer.load_state_dict(norm_data)
                else:
                    self.normalizer.mean = float(norm_data.mean)
                    self.normalizer.std = float(norm_data.std)
        
        if len(self.models) == 0:
            raise RuntimeError(f"No models found in {ensemble_dir}")
        
        print(f"Loaded {len(self.models)} ensemble members")
    
    @torch.no_grad()
    def predict_structure(self, structure: Structure) -> Dict:
        """Predict stability for a single structure."""
        # Convert to graph
        graph = structure_to_graph(structure)
        
        # Prepare tensors
        x = torch.tensor(graph["node_feats"], device=self.device)
        src = torch.tensor(graph["edge_index_src"], device=self.device)
        dst = torch.tensor(graph["edge_index_dst"], device=self.device)
        e_attr = torch.tensor(graph["edge_attr"], device=self.device)
        batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        # Run all ensemble members
        q50s = []
        p_stables = []
        
        for model in self.models:
            q50, q10, q90, p_stable, p_meta = model(x, src, dst, e_attr, batch_idx)
            q50s.append(self.normalizer.denorm(q50.item()))
            p_stables.append(torch.sigmoid(p_stable).item())
        
        # Aggregate predictions
        pred_ehull = np.mean(q50s)
        epistemic_std = np.std(q50s)
        p_stable = np.mean(p_stables)
        
        # Classify uncertainty
        if epistemic_std < 0.05:
            uncertainty = "Low"
        elif epistemic_std < 0.15:
            uncertainty = "Medium"
        else:
            uncertainty = "High"
        
        # Determine action
        if p_stable > 0.7 and uncertainty == "Low" and pred_ehull < 0.1:
            action = "DFT"
        elif p_stable > 0.5 or pred_ehull < 0.15:
            action = "HOLD"
        else:
            action = "SKIP"
        
        return {
            "pred_ehull": float(pred_ehull),
            "p_stable": float(p_stable),
            "epistemic_std": float(epistemic_std),
            "uncertainty": uncertainty,
            "action": action,
            "confidence_interval": (
                float(pred_ehull - 1.96 * epistemic_std),
                float(pred_ehull + 1.96 * epistemic_std),
            ),
        }


# Global predictor instance (lazy loaded)
_predictor: Optional[EnsemblePredictor] = None


def get_predictor() -> EnsemblePredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        # Find the ensemble directory
        ensemble_dir = Path(__file__).parent.parent.parent / "artifacts" / "models" / "ensemble_20260106_014934"
        if not ensemble_dir.exists():
            # Try to find any ensemble
            models_dir = Path(__file__).parent.parent.parent / "artifacts" / "models"
            if models_dir.exists():
                ensembles = list(models_dir.glob("ensemble_*"))
                if ensembles:
                    ensemble_dir = sorted(ensembles)[-1]  # Use latest
        
        if not ensemble_dir.exists():
            raise RuntimeError("No trained ensemble found. Run training first.")
        
        _predictor = EnsemblePredictor(str(ensemble_dir))
    return _predictor
