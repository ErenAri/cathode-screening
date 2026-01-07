"""
ALIGNN model wrapper for cathode screening.

Provides a unified interface to load pretrained ALIGNN models
and run inference on pymatgen Structures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch

# Check ALIGNN availability
ALIGNN_AVAILABLE = False
ALIGNN = None

try:
    from alignn.models.alignn import ALIGNN
    from alignn.config import TrainingConfig
    from jarvis.core.atoms import Atoms as JarvisAtoms
    ALIGNN_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"ALIGNN not available: {e}")


def check_alignn_available() -> bool:
    """Check if ALIGNN is available."""
    return ALIGNN_AVAILABLE


def pymatgen_to_jarvis(structure) -> "JarvisAtoms":
    """
    Convert pymatgen Structure to JARVIS Atoms.
    
    Args:
        structure: pymatgen Structure object
    
    Returns:
        JarvisAtoms object
    """
    if not ALIGNN_AVAILABLE:
        raise ImportError("alignn/jarvis-tools not installed")
    
    from jarvis.core.atoms import Atoms as JarvisAtoms
    
    return JarvisAtoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[str(s) for s in structure.species],
        cartesian=False,
    )


class ALIGNNWrapper:
    """
    Wrapper for ALIGNN model inference.
    
    Loads a pretrained or fine-tuned ALIGNN checkpoint and provides
    predictions for pymatgen structures.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
    ):
        """
        Initialize ALIGNN wrapper.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            config: Model configuration dict (or uses default)
            device: Device to run on ("cuda" or "cpu")
        """
        if not ALIGNN_AVAILABLE:
            raise ImportError(
                "ALIGNN not available. Install with: pip install alignn"
            )
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = config
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize model from config
        if self.config is None:
            if "config" in ckpt:
                self.config = ckpt["config"]
            else:
                # Use default ALIGNN config
                self.config = TrainingConfig().dict()
        
        # Create model
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        
        model_config = ALIGNNConfig(
            name="alignn",
            **self.config.get("model", {})
        )
        self.model = ALIGNN(model_config)
        
        # Load weights
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded ALIGNN from {checkpoint_path}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = "jv_formation_energy_peratom_alignn") -> "ALIGNNWrapper":
        """
        Load a pretrained ALIGNN model from jarvis.
        
        Args:
            model_name: Name of pretrained model from JARVIS
        
        Returns:
            ALIGNNWrapper instance
        """
        from alignn.pretrained import get_figshare_model
        
        model, config = get_figshare_model(model_name)
        
        wrapper = cls(device="cuda" if torch.cuda.is_available() else "cpu")
        wrapper.model = model
        wrapper.config = config
        wrapper.model.to(wrapper.device)
        wrapper.model.eval()
        
        print(f"Loaded pretrained ALIGNN: {model_name}")
        return wrapper
    
    @torch.no_grad()
    def predict_structure(self, structure) -> Dict[str, float]:
        """
        Predict properties for a single structure.
        
        Args:
            structure: pymatgen Structure
        
        Returns:
            Dict with prediction, e.g. {'energy': 0.123}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")
        
        from alignn.graphs import Graph
        
        # Convert to JARVIS atoms
        atoms = pymatgen_to_jarvis(structure)
        
        # Build graph
        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=8.0,
            max_neighbors=12,
        )
        
        # Move to device
        g = g.to(self.device)
        lg = lg.to(self.device)
        
        # Predict
        out = self.model([g, lg])
        
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out
        
        return {"energy": float(pred.cpu().numpy().flatten()[0])}
    
    @torch.no_grad()
    def predict_batch(
        self,
        structures: List,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Predict properties for a batch of structures.
        
        Args:
            structures: List of pymatgen Structures
            batch_size: Batch size for inference
        
        Returns:
            [N] array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = []
        
        for structure in structures:
            try:
                pred = self.predict_structure(structure)
                predictions.append(pred["energy"])
            except Exception as e:
                warnings.warn(f"Failed to predict: {e}")
                predictions.append(np.nan)
        
        return np.array(predictions)
