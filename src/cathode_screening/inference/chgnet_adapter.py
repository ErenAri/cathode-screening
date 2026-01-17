"""
CHGNet adapter for decision-grade predictions.

Wraps LiCathodeScreener to provide the same interface as DecisionPredictor,
enabling seamless substitution in the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from pymatgen.core import Structure

from cathode_screening.inference.decision_predictor import DecisionOutput
from cathode_screening.inference.li_cathode_screener import LiCathodeScreener


# Stability thresholds (eV/atom) - matches DecisionPredictor
THRESH_STABLE = 0.05
THRESH_METASTABLE = 0.10


class CHGNetDecisionService:
    """
    Decision-grade service using CHGNet ensemble.
    
    Drop-in replacement for DecisionService when using CHGNet model.
    """
    
    def __init__(
        self,
        screener: LiCathodeScreener,
        default_mode: str = "dft_followup",
    ):
        self.screener = screener
        self.default_mode = default_mode
        
        # OOD scoring not available in CHGNet (single embedding model)
        # Use uncertainty-based heuristic instead
        self.ood_sigma_threshold = 0.03
    
    @classmethod
    def from_env(cls) -> "CHGNetDecisionService":
        """Load CHGNet service from environment configuration."""
        artifacts_dir = Path(os.getenv("CATHODE_ARTIFACTS_DIR", "data/artifacts"))
        chgnet_dir = artifacts_dir / "chgnet_ensemble"
        
        # Check if CHGNet ensemble exists
        if not chgnet_dir.exists():
            raise FileNotFoundError(
                f"CHGNet ensemble not found at {chgnet_dir}. "
                "Set CATHODE_MODEL_TYPE=cgcnn to use CGCNN instead."
            )
        
        mode = os.getenv("CATHODE_DEFAULT_MODE", "dft_followup")
        
        device_name = os.getenv("CATHODE_DEVICE", "auto").lower()
        if device_name == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_name
        
        screener = LiCathodeScreener(model_dir=chgnet_dir, device=device)
        return cls(screener=screener, default_mode=mode)
    
    def predict_structure(
        self,
        structure: Structure,
        material_id: str,
        formula: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> DecisionOutput:
        """Predict E_hull for a structure using CHGNet ensemble."""
        actual_mode = mode or self.default_mode
        actual_formula = formula or structure.composition.reduced_formula
        
        # Get CHGNet prediction
        result = self.screener.predict_structure(structure, material_id)
        
        # Map recommendation to decision
        if result.recommendation == "KEEP":
            decision = "KEEP"
        elif result.recommendation == "KILL":
            decision = "KILL"
        else:
            decision = "MAYBE"
        
        # Map confidence
        if result.confidence == "HIGH":
            decision_confidence = 0.9
        elif result.confidence == "MEDIUM":
            decision_confidence = 0.5
        else:
            decision_confidence = 0.2
        
        # OOD detection via uncertainty threshold
        ood_flag = result.uncertainty > self.ood_sigma_threshold
        ood_score = min(1.0, result.uncertainty / (2 * self.ood_sigma_threshold))
        
        # Generate explanation
        if ood_flag:
            explanation = f"High uncertainty (σ={result.uncertainty:.3f}). Manual review recommended."
        elif decision == "KEEP":
            explanation = f"Confidently stable: μ={result.pred_ehull:.3f} ± {result.uncertainty:.3f} eV"
        elif decision == "KILL":
            explanation = f"Confidently unstable: μ={result.pred_ehull:.3f} ± {result.uncertainty:.3f} eV"
        else:
            explanation = f"Uncertain: μ={result.pred_ehull:.3f} ± {result.uncertainty:.3f} eV"
        
        return DecisionOutput(
            material_id=material_id,
            formula=actual_formula,
            ehull_pred=result.pred_ehull,
            ehull_lower=result.ci_lower,
            ehull_upper=result.ci_upper,
            uncertainty_aleatoric=0.0,  # CHGNet uses ensemble epistemic only
            uncertainty_epistemic=result.uncertainty,
            uncertainty_total=result.uncertainty,
            ood_score=ood_score,
            ood_flag=ood_flag,
            ood_gates={
                "comp": False,  # Not available
                "emb": False,   # Not available
                "disagree": result.uncertainty > self.ood_sigma_threshold,
            },
            decision=decision,
            decision_confidence=decision_confidence,
            decision_mode=actual_mode,
            explanation=explanation,
            p_stable=None,
            p_metastable=None,
        )
    
    def predict_structures(
        self,
        structures: Sequence[Structure],
        material_ids: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
    ) -> List[DecisionOutput]:
        """Batch prediction for multiple structures."""
        if not structures:
            return []
        
        n = len(structures)
        if material_ids is None:
            material_ids = [f"material-{idx}" for idx in range(n)]
        if formulas is None:
            formulas = [s.composition.reduced_formula for s in structures]
        
        if len(material_ids) != n or len(formulas) != n:
            raise ValueError("Material IDs and formulas must match structure count")
        
        results = []
        for i in range(n):
            try:
                output = self.predict_structure(
                    structure=structures[i],
                    material_id=material_ids[i],
                    formula=formulas[i],
                    mode=mode,
                )
                results.append(output)
            except Exception as e:
                print(f"Warning: Failed to predict {material_ids[i]}: {e}")
        
        return results
