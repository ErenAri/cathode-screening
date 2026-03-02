"""
v1-Li-Cathode CHGNet Ensemble Production Interface.

Provides decision-grade predictions with:
- predicted E_hull (μ)
- uncertainty (σ)
- calibrated interval
- rank position
- recommendation: KEEP / MAYBE / KILL

Dataset: Li-O-TM cathodes (SOAP-LOCO ∩ Li)
Metrics: MAE=0.032 eV/atom, EF@1%=6.66x, False Kill Rate<0.3%
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import logging
import numpy as np
import torch
from pymatgen.core import Structure

from cathode_screening.common.serialization import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class CathodeScreenResult:
    """Decision-grade screening result for a cathode material."""
    
    material_id: Optional[str]
    formula: str
    pred_ehull: float  # μ (predicted E_hull in eV/atom)
    uncertainty: float  # σ (epistemic uncertainty)
    ci_lower: float  # Calibrated interval lower bound
    ci_upper: float  # Calibrated interval upper bound
    rank: int  # Position in ranked list (1 = best)
    recommendation: str  # KEEP / MAYBE / KILL
    confidence: str  # HIGH / MEDIUM / LOW
    
    def to_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "formula": self.formula,
            "pred_ehull": round(self.pred_ehull, 4),
            "uncertainty": round(self.uncertainty, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "rank": self.rank,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


class LiCathodeScreener:
    """
    v1-Li-Cathode Decision-Grade Screening Engine.
    
    Industrial claim:
    "On unseen Li-cathode chemistries, our system recovers 55% of all 
    ultra-stable (Ehull≤10 meV) materials within the top-100 candidates 
    — a 6.6× enrichment over random — while maintaining <0.3% false-discard rate."
    
    Valid for: Li-containing oxide cathode materials only.
    """
    
    # Policy thresholds (tuned on SOAP-LOCO validation)
    KEEP_THRESHOLD = 0.05  # μ < 0.05 eV → KEEP
    KILL_THRESHOLD = 0.15  # μ > 0.15 eV → KILL
    HIGH_UNCERTAINTY = 0.02  # σ > 0.02 → lower confidence
    
    # Calibration factor (empirical from SOAP-LOCO test set)
    # 95% CI ≈ μ ± 2.0 * σ based on ensemble coverage
    CALIBRATION_FACTOR = 2.0
    
    def __init__(
        self,
        model_dir: Union[str, Path] = "data/artifacts/chgnet_ensemble",
        device: Optional[str] = None,
    ):
        """Initialize the Li-cathode screener.
        
        Args:
            model_dir: Directory containing the 5 ensemble checkpoints
            device: "cuda" or "cpu" (auto-detected if None)
        """
        from chgnet.model import CHGNet
        
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ensemble
        self.models = []
        seeds = [42, 123, 456, 789, 1024]
        
        for seed in seeds:
            seed_dir = self.model_dir / f"seed_{seed}"
            ckpt_path = self._get_best_checkpoint(seed_dir)
            
            if ckpt_path:
                model = CHGNet.load()
                ckpt = safe_torch_load(ckpt_path, device=torch.device(self.device))
                model.load_state_dict(ckpt["model"]["state_dict"])
                model.to(self.device)
                model.eval()
                self.models.append(model)
        
        if not self.models:
            raise RuntimeError(f"No models found in {model_dir}")
        
        logger.info("LiCathodeScreener v1 initialized: %d models on %s", len(self.models), self.device)
    
    def _get_best_checkpoint(self, model_dir: Path) -> Optional[Path]:
        """Get best checkpoint by MAE."""
        best_ckpts = list(model_dir.glob("bestE_*.pth.tar"))
        if not best_ckpts:
            return None
        
        def parse_mae(p):
            m = re.search(r"_e(\d+)_", p.name)
            return int(m.group(1)) if m else float("inf")
        
        return sorted(best_ckpts, key=parse_mae)[0]
    
    def _make_recommendation(
        self, 
        mu: float, 
        sigma: float,
    ) -> Tuple[str, str]:
        """Generate recommendation and confidence level.
        
        Returns:
            (recommendation, confidence)
        """
        # Determine confidence based on uncertainty
        if sigma > self.HIGH_UNCERTAINTY:
            confidence = "LOW"
        elif sigma > self.HIGH_UNCERTAINTY / 2:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"
        
        # Decision policy
        if mu < self.KEEP_THRESHOLD and sigma < self.HIGH_UNCERTAINTY:
            return "KEEP", confidence
        elif mu > self.KILL_THRESHOLD:
            return "KILL", confidence
        else:
            return "MAYBE", confidence
    
    def predict_structure(
        self,
        structure: Structure,
        material_id: Optional[str] = None,
    ) -> CathodeScreenResult:
        """Predict E_hull for a single structure.
        
        Args:
            structure: pymatgen Structure object
            material_id: Optional material identifier
            
        Returns:
            CathodeScreenResult with full decision info
        """
        # Validate Li content
        elements = [str(s.specie) for s in structure.sites]
        if "Li" not in elements:
            raise ValueError("Structure must contain Li for v1-Li-Cathode model")
        
        # Get predictions from all models
        preds = []
        for model in self.models:
            try:
                result = model.predict_structure(structure)
                preds.append(float(result["e"]))
            except (RuntimeError, KeyError, ValueError) as exc:
                logger.warning("Model prediction failed for %s: %s", material_id, exc)
                preds.append(float("nan"))
        
        preds = np.array(preds)
        valid_preds = preds[~np.isnan(preds)]
        
        if len(valid_preds) == 0:
            raise RuntimeError("All model predictions failed")
        
        # Ensemble statistics
        mu = float(np.mean(valid_preds))
        sigma = float(np.std(valid_preds))
        
        # Calibrated interval
        ci_lower = mu - self.CALIBRATION_FACTOR * sigma
        ci_upper = mu + self.CALIBRATION_FACTOR * sigma
        
        # Recommendation
        recommendation, confidence = self._make_recommendation(mu, sigma)
        
        return CathodeScreenResult(
            material_id=material_id,
            formula=structure.formula,
            pred_ehull=mu,
            uncertainty=sigma,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            rank=0,  # Set later in batch
            recommendation=recommendation,
            confidence=confidence,
        )
    
    def screen_batch(
        self,
        structures: List[Structure],
        material_ids: Optional[List[str]] = None,
    ) -> List[CathodeScreenResult]:
        """Screen a batch of structures with ranking.
        
        Args:
            structures: List of pymatgen Structure objects
            material_ids: Optional list of material identifiers
            
        Returns:
            List of CathodeScreenResult, sorted by pred_ehull (best first)
        """
        if material_ids is None:
            material_ids = [None] * len(structures)
        
        results = []
        for i, (struct, mid) in enumerate(zip(structures, material_ids)):
            try:
                result = self.predict_structure(struct, mid)
                results.append(result)
            except (RuntimeError, KeyError, ValueError) as exc:
                logger.warning("Failed to predict %s: %s", mid or i, exc)
        
        # Sort by predicted E_hull (ascending)
        results.sort(key=lambda r: r.pred_ehull)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def from_cif(
        self,
        cif_path: Union[str, Path],
        material_id: Optional[str] = None,
    ) -> CathodeScreenResult:
        """Predict from CIF file."""
        structure = Structure.from_file(cif_path)
        return self.predict_structure(structure, material_id)
    
    def from_cif_string(
        self,
        cif_string: str,
        material_id: Optional[str] = None,
    ) -> CathodeScreenResult:
        """Predict from CIF string."""
        structure = Structure.from_str(cif_string, fmt="cif")
        return self.predict_structure(structure, material_id)


# Convenience function
def screen_li_cathode(
    structure: Structure,
    material_id: Optional[str] = None,
    model_dir: str = "data/artifacts/chgnet_ensemble",
) -> CathodeScreenResult:
    """Quick screening for a single Li-cathode structure.
    
    Example:
        >>> from pymatgen.core import Structure
        >>> struct = Structure.from_file("LiCoO2.cif")
        >>> result = screen_li_cathode(struct)
        >>> print(f"{result.recommendation}: E_hull={result.pred_ehull:.3f}±{result.uncertainty:.3f}")
    """
    screener = LiCathodeScreener(model_dir)
    return screener.predict_structure(structure, material_id)
