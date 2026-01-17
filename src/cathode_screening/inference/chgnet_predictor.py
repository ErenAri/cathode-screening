"""
CHGNet predictor for E_hull prediction.

Uses the fine-tuned CHGNet model for high-accuracy predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from pymatgen.core import Structure


class CHGNetPredictor:
    """Load and run fine-tuned CHGNet for E_hull prediction."""
    
    def __init__(
        self,
        model,
        device: torch.device,
    ) -> None:
        """Initialize with loaded model."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "CHGNetPredictor":
        """Load CHGNet from checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pth.tar checkpoint file
            device: Device to load model on ("cuda" or "cpu")
        
        Returns:
            CHGNetPredictor instance
        """
        from chgnet.model import CHGNet
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        
        # Load pretrained CHGNet
        model = CHGNet.load()
        
        # Load fine-tuned weights
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device_obj)
            
            # Handle different checkpoint formats
            if "model" in ckpt and "state_dict" in ckpt["model"]:
                model.load_state_dict(ckpt["model"]["state_dict"])
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        
        return cls(model=model, device=device_obj)
    
    @classmethod
    def from_directory(
        cls,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
    ) -> "CHGNetPredictor":
        """Load CHGNet from model directory.
        
        Args:
            model_dir: Directory containing CHGNet checkpoint
            device: Device to load model on
            
        Returns:
            CHGNetPredictor instance
        """
        model_dir = Path(model_dir)
        
        # Look for best checkpoint first, then final
        best_ckpts = list(model_dir.glob("bestE_*.pth.tar"))
        if best_ckpts:
            checkpoint_path = best_ckpts[0]
        elif (model_dir / "chgnet_best.pt").exists():
            checkpoint_path = model_dir / "chgnet_best.pt"
        elif (model_dir / "chgnet_final.pt").exists():
            checkpoint_path = model_dir / "chgnet_final.pt"
        else:
            raise FileNotFoundError(f"No CHGNet checkpoint found in {model_dir}")
        
        return cls.from_checkpoint(checkpoint_path, device)
    
    def predict_structure(
        self,
        structure: Structure,
    ) -> Tuple[float, float]:
        """Predict E_hull for a single structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Tuple of (prediction, uncertainty)
            Note: CHGNet doesn't provide uncertainty, so uncertainty is 0
        """
        with torch.no_grad():
            result = self.model.predict_structure(
                structure,
                return_site_energies=False,
            )
        
        e_hull = result.get("e", 0.0)
        
        # CHGNet doesn't provide uncertainty natively
        # Could implement ensemble for uncertainty
        uncertainty = 0.0
        
        return float(e_hull), float(uncertainty)
    
    def predict_from_cif(
        self,
        cif_path: Union[str, Path],
    ) -> Tuple[float, float]:
        """Predict E_hull from CIF file.
        
        Args:
            cif_path: Path to CIF file
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        structure = Structure.from_file(cif_path)
        return self.predict_structure(structure)
    
    def predict_from_cif_string(
        self,
        cif_string: str,
    ) -> Tuple[float, float]:
        """Predict E_hull from CIF string.
        
        Args:
            cif_string: CIF file contents as string
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        structure = Structure.from_str(cif_string, fmt="cif")
        return self.predict_structure(structure)
    
    def predict_batch(
        self,
        structures: list[Structure],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict E_hull for batch of structures.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            Tuple of (predictions array, uncertainties array)
        """
        predictions = []
        uncertainties = []
        
        for struct in structures:
            try:
                pred, unc = self.predict_structure(struct)
                predictions.append(pred)
                uncertainties.append(unc)
            except Exception as e:
                print(f"Warning: Failed to predict structure: {e}")
                predictions.append(np.nan)
                uncertainties.append(np.nan)
        
        return np.array(predictions), np.array(uncertainties)
