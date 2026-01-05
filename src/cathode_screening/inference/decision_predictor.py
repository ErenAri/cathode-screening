"""
Decision-grade predictor for cathode screening.

Combines:
- Deep ensemble (K=5)
- Conformal calibration
- OOD gating
- Utility-optimized thresholds

Usage:
    from cathode_screening.inference.decision_predictor import DecisionPredictor
    
    predictor = DecisionPredictor.from_artifacts("data/artifacts")
    result = predictor.predict("mp-12345", "LiCoO2", "path/to/graph.npz", mode="dft_followup")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cathode_screening.inference.ood import OODGate, OODGateResult
from cathode_screening.evaluation.conformal import (
    ConformalCalibrator,
    load_calibration_params,
    apply_conformal_calibration,
)


# Stability thresholds (eV/atom)
THRESH_STABLE = 0.05
THRESH_METASTABLE = 0.10


@dataclass
class DecisionOutput:
    """Complete output from decision predictor."""
    
    # Identifiers
    material_id: str
    formula: str
    
    # Point estimates (eV)
    ehull_pred: float           # Ensemble q50
    ehull_lower: float          # Calibrated q10
    ehull_upper: float          # Calibrated q90
    
    # Uncertainty decomposition
    uncertainty_aleatoric: float  # From interval width (irreducible)
    uncertainty_epistemic: float  # From ensemble disagreement (model uncertainty)
    uncertainty_total: float      # Combined
    
    # OOD assessment
    ood_score: float              # [0, 1], higher = more OOD
    ood_flag: bool                # True if any gate triggered
    ood_gates: Dict[str, bool] = field(default_factory=dict)
    
    # Decision
    decision: str                 # KEEP / MAYBE / KILL
    decision_confidence: float    # [0, 1] based on margin to thresholds
    decision_mode: str            # "dft_followup" or "experimental"
    
    # Explanation
    explanation: str = ""         # Human-readable rationale
    
    # Raw predictions (for debugging)
    q50_per_member: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "material_id": self.material_id,
            "formula": self.formula,
            "ehull_pred": self.ehull_pred,
            "ehull_lower": self.ehull_lower,
            "ehull_upper": self.ehull_upper,
            "uncertainty_aleatoric": self.uncertainty_aleatoric,
            "uncertainty_epistemic": self.uncertainty_epistemic,
            "uncertainty_total": self.uncertainty_total,
            "ood_score": self.ood_score,
            "ood_flag": self.ood_flag,
            "ood_gates": self.ood_gates,
            "decision": self.decision,
            "decision_confidence": self.decision_confidence,
            "decision_mode": self.decision_mode,
            "explanation": self.explanation,
        }


class Normalizer:
    """Target normalizer (matches training)."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
    
    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class DecisionPredictor:
    """
    Decision-grade predictor combining ensemble, calibration, OOD gating, and thresholds.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        normalizers: List[Normalizer],
        calibrator: Optional[ConformalCalibrator],
        ood_gate: OODGate,
        thresholds: Dict[str, Dict],
        device: torch.device,
    ):
        self.models = models
        self.normalizers = normalizers
        self.calibrator = calibrator
        self.ood_gate = ood_gate
        self.thresholds = thresholds
        self.device = device
    
    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        device: Optional[torch.device] = None,
    ) -> "DecisionPredictor":
        """Load predictor from saved artifacts."""
        artifact_dir = Path(artifact_dir)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ensemble
        ensemble_dir = artifact_dir / "ensemble"
        meta_path = ensemble_dir / "ensemble_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        models = []
        normalizers = []
        
        for member in meta["members"]:
            ckpt_path = Path(member["checkpoint"])
            ckpt = torch.load(ckpt_path, map_location=device)
            cfg = ckpt["cfg"]
            
            # Reconstruct model (import from training script)
            from scripts.training_imports import CGCNN
            
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
        
        # Load calibration params (use new conformal module if available)
        calibrator = None
        calib_path = artifact_dir / "calibration" / "conformal_params.json"
        if calib_path.exists():
            try:
                calibrator = ConformalCalibrator.from_file(calib_path)
                print(f"Loaded conformal calibration: Δ_upper={calibrator.delta_upper:.4f}, Δ_lower={calibrator.delta_lower:.4f}")
            except Exception as e:
                print(f"Warning: Could not load calibration params: {e}")
        
        # Load OOD gate
        ood_gate = OODGate.from_artifacts(artifact_dir / "ood")
        
        # Load thresholds
        thresholds = {}
        for mode in ["dft_followup", "experimental"]:
            thresh_path = artifact_dir / "thresholds" / f"thresholds_{mode.split('_')[0]}.json"
            if thresh_path.exists():
                with open(thresh_path, "r", encoding="utf-8") as f:
                    thresholds[mode] = json.load(f)
        
        # Fallback thresholds if not found
        if "dft_followup" not in thresholds:
            thresholds["dft_followup"] = {"tau_keep": 0.07, "tau_kill": 0.12}
        if "experimental" not in thresholds:
            thresholds["experimental"] = {"tau_keep": 0.04, "tau_kill": 0.09}
        
        return cls(
            models=models,
            normalizers=normalizers,
            calibrator=calibrator,
            ood_gate=ood_gate,
            thresholds=thresholds,
            device=device,
        )
    
    def _load_graph(self, graph_npz_path: str | Path) -> Tuple[torch.Tensor, ...]:
        """Load graph from NPZ file."""
        z = np.load(graph_npz_path)
        x = torch.from_numpy(z["x"]).to(self.device)
        src = torch.from_numpy(z["edge_src"]).long().to(self.device)
        dst = torch.from_numpy(z["edge_dst"]).long().to(self.device)
        e = torch.from_numpy(z["e_attr"]).to(self.device)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        return x, src, dst, e, batch
    
    def _get_embedding(self, model: nn.Module, x, src, dst, e, batch) -> np.ndarray:
        """Extract graph embedding from model (after pooling, before heads)."""
        x = model.embed(x)
        for blk in model.blocks:
            x = blk(x, src, dst, e)
        g = model.pool(x, batch)
        return g.detach().cpu().numpy()
    
    @torch.no_grad()
    def predict(
        self,
        material_id: str,
        formula: str,
        graph_npz_path: str | Path,
        mode: str = "dft_followup",
    ) -> DecisionOutput:
        """
        Make decision prediction for a single material.
        
        Args:
            material_id: Unique identifier
            formula: Chemical formula (e.g., "LiCoO2")
            graph_npz_path: Path to preprocessed graph NPZ
            mode: "dft_followup" or "experimental"
        
        Returns:
            DecisionOutput with prediction, uncertainty, OOD assessment, and decision
        """
        if mode not in self.thresholds:
            raise ValueError(f"Unknown mode: {mode}. Use 'dft_followup' or 'experimental'")
        
        # Load graph
        x, src, dst, e, batch = self._load_graph(graph_npz_path)
        
        # Run ensemble inference
        q10_raw, q50_raw, q90_raw = [], [], []
        embeddings = []
        
        for k, (model, norm) in enumerate(zip(self.models, self.normalizers)):
            model.eval()
            q10, q50, q90, _, _ = model(x, src, dst, e, batch)
            
            # Denormalize
            q10 = norm.denorm(q10).cpu().item()
            q50 = norm.denorm(q50).cpu().item()
            q90 = norm.denorm(q90).cpu().item()
            
            q10_raw.append(q10)
            q50_raw.append(q50)
            q90_raw.append(q90)
            
            # Get embedding
            emb = self._get_embedding(model, x, src, dst, e, batch)
            embeddings.append(emb.flatten())
        
        # Ensemble aggregation
        q50_ens = float(np.mean(q50_raw))
        q10_ens = float(np.mean(q10_raw))
        q90_ens = float(np.mean(q90_raw))
        
        # Apply conformal calibration if available
        if self.calibrator is not None:
            q10_cal_arr, q90_cal_arr = self.calibrator.calibrate(
                np.array([q10_ens]), 
                np.array([q90_ens])
            )
            q10_cal = float(q10_cal_arr[0])
            q90_cal = float(q90_cal_arr[0])
        else:
            # No calibration - use raw ensemble bounds
            q10_cal = q10_ens
            q90_cal = q90_ens
        
        # Uncertainty decomposition
        # Aleatoric: average interval width / 2.56 (80% interval → std conversion)
        sigma_ale = float(np.mean([q90_raw[k] - q10_raw[k] for k in range(len(self.models))]) / 2.56)
        # Epistemic: std of median predictions
        sigma_epi = float(np.std(q50_raw))
        # Total
        sigma_tot = float(np.sqrt(sigma_ale**2 + sigma_epi**2))
        
        # OOD gating
        embedding_concat = np.concatenate(embeddings)
        ood_result = self.ood_gate.score(formula, embedding_concat, q50_raw)
        
        # Decision logic
        thresh = self.thresholds[mode]
        tau_keep = thresh["tau_keep"]
        tau_kill = thresh["tau_kill"]
        
        # Force MAYBE for OOD inputs (conservative)
        if ood_result.flag:
            decision = "MAYBE"
            decision_confidence = 0.0
            triggered = [g for g in ["comp", "emb", "disagree"] if ood_result.to_dict()[f"flag_{g}"]]
            explanation = f"OOD detected ({', '.join(triggered)}). Manual review recommended."
        elif q90_cal < tau_keep:
            decision = "KEEP"
            margin = (tau_keep - q90_cal) / tau_keep
            decision_confidence = min(1.0, margin * 2)
            explanation = f"Confidently stable: q90={q90_cal:.3f} eV < τ_keep={tau_keep:.3f} eV"
        elif q10_cal > tau_kill:
            decision = "KILL"
            margin = (q10_cal - tau_kill) / tau_kill
            decision_confidence = min(1.0, margin * 2)
            explanation = f"Confidently unstable: q10={q10_cal:.3f} eV > τ_kill={tau_kill:.3f} eV"
        else:
            decision = "MAYBE"
            # Confidence based on how centered in MAYBE region
            maybe_width = tau_kill - tau_keep
            dist_to_keep = q90_cal - tau_keep
            dist_to_kill = tau_kill - q10_cal
            decision_confidence = 0.5 - 0.5 * abs(dist_to_keep - dist_to_kill) / maybe_width
            explanation = f"Uncertain: interval [{q10_cal:.3f}, {q90_cal:.3f}] overlaps decision region"
        
        return DecisionOutput(
            material_id=material_id,
            formula=formula,
            ehull_pred=q50_ens,
            ehull_lower=q10_cal,
            ehull_upper=q90_cal,
            uncertainty_aleatoric=sigma_ale,
            uncertainty_epistemic=sigma_epi,
            uncertainty_total=sigma_tot,
            ood_score=ood_result.ood_score,
            ood_flag=ood_result.flag,
            ood_gates={
                "comp": ood_result.flag_comp,
                "emb": ood_result.flag_emb,
                "disagree": ood_result.flag_disagree,
            },
            decision=decision,
            decision_confidence=decision_confidence,
            decision_mode=mode,
            explanation=explanation,
            q50_per_member=q50_raw,
        )
    
    def predict_batch(
        self,
        materials: List[Dict],  # [{"material_id": ..., "formula": ..., "graph_npz": ...}, ...]
        mode: str = "dft_followup",
    ) -> List[DecisionOutput]:
        """Batch prediction for multiple materials."""
        return [
            self.predict(m["material_id"], m["formula"], m["graph_npz"], mode)
            for m in materials
        ]


def generate_report(outputs: List[DecisionOutput]) -> Dict:
    """Generate summary report from batch predictions."""
    
    n = len(outputs)
    n_keep = sum(1 for o in outputs if o.decision == "KEEP")
    n_maybe = sum(1 for o in outputs if o.decision == "MAYBE")
    n_kill = sum(1 for o in outputs if o.decision == "KILL")
    n_ood = sum(1 for o in outputs if o.ood_flag)
    
    ehull_preds = [o.ehull_pred for o in outputs]
    uncertainties = [o.uncertainty_total for o in outputs]
    
    keep_outputs = [o for o in outputs if o.decision == "KEEP"]
    
    return {
        "total": n,
        "decisions": {
            "KEEP": n_keep,
            "MAYBE": n_maybe,
            "KILL": n_kill,
            "keep_frac": n_keep / n if n > 0 else 0,
            "kill_frac": n_kill / n if n > 0 else 0,
        },
        "ood": {
            "n_flagged": n_ood,
            "frac_flagged": n_ood / n if n > 0 else 0,
        },
        "predictions": {
            "ehull_mean": float(np.mean(ehull_preds)),
            "ehull_std": float(np.std(ehull_preds)),
            "uncertainty_mean": float(np.mean(uncertainties)),
        },
        "top_candidates": [
            {
                "material_id": o.material_id,
                "formula": o.formula,
                "ehull_pred": o.ehull_pred,
                "confidence": o.decision_confidence,
            }
            for o in sorted(keep_outputs, key=lambda x: x.ehull_pred)[:10]
        ],
    }
