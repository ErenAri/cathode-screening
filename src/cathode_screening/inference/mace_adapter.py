"""
MACE adapter for decision-grade predictions.

Wraps a 5-member MACE-MP-0 fine-tuned ensemble to provide the same interface
as CHGNetDecisionService, enabling seamless substitution in the API.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from pymatgen.core import Structure

from cathode_screening.inference.decision_predictor import DecisionOutput
from cathode_screening.models.mace_finetune.model import MACEFineTuner
from cathode_screening.models.mace_finetune.data import _compute_neighbors

logger = logging.getLogger(__name__)

# Stability thresholds (eV)
THRESH_STABLE = 0.05
THRESH_METASTABLE = 0.10


class _Normalizer:
    """Simple target normalizer matching training."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.std = d["std"]


class MACEDecisionService:
    """Decision-grade service using MACE-MP-0 fine-tuned ensemble."""

    def __init__(
        self,
        models: List[torch.nn.Module],
        normalizers: List[_Normalizer],
        z_table: Optional[list],
        r_max: float = 5.0,
        conformal_delta: float = 0.0,
        default_mode: str = "dft_followup",
        device: str = "cpu",
    ):
        self.models = models
        self.normalizers = normalizers
        self.z_table = z_table
        self.r_max = r_max
        self.conformal_delta = conformal_delta
        self.default_mode = default_mode
        self.device = device

    @classmethod
    def from_env(cls) -> "MACEDecisionService":
        """Load MACE ensemble from environment configuration."""
        artifacts_dir = Path(os.getenv("CATHODE_ARTIFACTS_DIR", "data/artifacts"))
        ensemble_dir = artifacts_dir / "models" / "mace_ensemble_v1"

        if not ensemble_dir.exists():
            # Also check directly under artifacts_dir
            ensemble_dir = artifacts_dir / "mace_ensemble_v1"
        if not ensemble_dir.exists():
            raise FileNotFoundError(
                f"MACE ensemble not found at {ensemble_dir}. "
                "Set CATHODE_MODEL_TYPE=chgnet to use CHGNet instead."
            )

        meta_path = ensemble_dir / "ensemble_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        device_name = os.getenv("CATHODE_DEVICE", "auto").lower()
        if device_name == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_name

        mode = os.getenv("CATHODE_DEFAULT_MODE", "dft_followup")

        logger.info("Loading MACE ensemble from %s (%d members)", ensemble_dir, len(meta["members"]))

        models = []
        normalizers = []
        z_table = None
        r_max = 5.0

        for member_info in meta["members"]:
            ckpt_path = Path(member_info["checkpoint"])
            if not ckpt_path.exists():
                ckpt_path = ensemble_dir / ckpt_path
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt["cfg"]

            model = MACEFineTuner.from_pretrained(
                model_name=cfg["model"].get("backbone", "medium"),
                head_dim=cfg["model"].get("head_dim", 128),
                dropout=cfg["model"].get("dropout", 0.1),
                freeze_backbone=cfg["model"].get("freeze_backbone", True),
                unfreeze_last_n=cfg["model"].get("unfreeze_last_n", 1),
            )
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()

            norm = _Normalizer()
            if "normalizer" in ckpt:
                norm.load_state_dict(ckpt["normalizer"])

            models.append(model)
            normalizers.append(norm)

            if z_table is None:
                z_table = getattr(model, "z_table", None)
            r_max = cfg["model"].get("r_max", 5.0)

            idx = member_info.get("member_idx", "?")
            logger.info("  Loaded member %s", idx)

        # Load conformal calibration
        conformal_delta = 0.0
        cal_path = ensemble_dir / "calibration" / "conformal_params.json"
        if cal_path.exists():
            with open(cal_path, "r") as f:
                cal = json.load(f)
            conformal_delta = cal.get("delta_upper", 0.0)
            logger.info("  Conformal delta: %.4f", conformal_delta)

        return cls(
            models=models,
            normalizers=normalizers,
            z_table=z_table,
            r_max=r_max,
            conformal_delta=conformal_delta,
            default_mode=mode,
            device=device,
        )

    def _structure_to_input(self, structure: Structure) -> dict:
        """Convert a pymatgen Structure to MACE input dict."""
        positions = structure.cart_coords.astype(np.float32)
        cell = np.array(structure.lattice.matrix, dtype=np.float32)
        atomic_numbers = np.array([s.specie.Z for s in structure], dtype=int)

        # One-hot encode atoms
        if self.z_table is not None:
            z_to_idx = {z: i for i, z in enumerate(self.z_table)}
            n_species = len(self.z_table)
            node_attrs = np.zeros((len(atomic_numbers), n_species), dtype=np.float32)
            for i, z in enumerate(atomic_numbers):
                if z in z_to_idx:
                    node_attrs[i, z_to_idx[z]] = 1.0
                else:
                    logger.warning("Atomic number %d not in z_table, using zero vector", z)
        else:
            node_attrs = np.eye(len(atomic_numbers), dtype=np.float32)

        # Compute neighbor list
        edge_src, edge_dst, cart_shifts, unit_shifts = _compute_neighbors(
            positions, cell, cutoff=self.r_max
        )

        n_atoms = len(atomic_numbers)
        return {
            "positions": torch.tensor(positions, dtype=torch.float32),
            "node_attrs": torch.tensor(node_attrs, dtype=torch.float32),
            "edge_index": torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long),
            "shifts": torch.tensor(cart_shifts, dtype=torch.float32),
            "unit_shifts": torch.tensor(unit_shifts, dtype=torch.float32),
            "cell": torch.tensor(cell, dtype=torch.float32).unsqueeze(0),
            "batch": torch.zeros(n_atoms, dtype=torch.long),
            "ptr": torch.tensor([0, n_atoms], dtype=torch.long),
        }

    @torch.no_grad()
    def _ensemble_predict(self, data_dict: dict):
        """Run all ensemble members and aggregate predictions."""
        data_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }

        q10_all, q50_all, q90_all, ps_all, pm_all = [], [], [], [], []

        for model, norm in zip(self.models, self.normalizers):
            # Each model may mutate the dict, so re-create device tensors
            d = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
            q10, q50, q90, logit_s, logit_m = model(d)
            q10_all.append(norm.denorm(q10).cpu().item())
            q50_all.append(norm.denorm(q50).cpu().item())
            q90_all.append(norm.denorm(q90).cpu().item())
            ps_all.append(torch.sigmoid(logit_s).cpu().item())
            pm_all.append(torch.sigmoid(logit_m).cpu().item())

        q10_mean = float(np.mean(q10_all))
        q50_mean = float(np.mean(q50_all))
        q90_mean = float(np.mean(q90_all))
        p_stable = float(np.mean(ps_all))
        p_meta = float(np.mean(pm_all))
        epistemic_std = float(np.std(q50_all, ddof=1)) if len(q50_all) > 1 else 0.0

        # Apply conformal calibration
        q10_cal = q10_mean - self.conformal_delta
        q90_cal = q90_mean + self.conformal_delta

        return q10_cal, q50_mean, q90_cal, p_stable, p_meta, epistemic_std, q50_all

    def predict_structure(
        self,
        structure: Structure,
        material_id: str,
        formula: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> DecisionOutput:
        """Predict E_hull for a structure using MACE ensemble."""
        actual_mode = mode or self.default_mode
        actual_formula = formula or structure.composition.reduced_formula

        data_dict = self._structure_to_input(structure)
        q10, q50, q90, p_stable, p_meta, eps_std, q50_members = self._ensemble_predict(data_dict)

        # Decision policy
        if q90 < THRESH_STABLE and p_stable > 0.8:
            decision = "KEEP"
        elif q90 < THRESH_METASTABLE and p_stable > 0.7:
            decision = "KEEP"
        elif q10 > THRESH_METASTABLE:
            decision = "KILL"
        else:
            decision = "MAYBE"

        # Confidence from margin to threshold
        if decision == "KEEP":
            margin = THRESH_STABLE - q90
            confidence = min(1.0, max(0.0, margin / 0.05))
        elif decision == "KILL":
            margin = q10 - THRESH_METASTABLE
            confidence = min(1.0, max(0.0, margin / 0.10))
        else:
            confidence = 0.3

        # OOD heuristic based on epistemic uncertainty
        ood_threshold = 0.03
        ood_flag = eps_std > ood_threshold
        ood_score = min(1.0, eps_std / (2 * ood_threshold))

        aleatoric = (q90 - q10) / 2.0
        total_unc = float(np.sqrt(aleatoric**2 + eps_std**2))

        # Explanation
        if ood_flag:
            explanation = f"High ensemble disagreement (sigma={eps_std:.3f}). Manual review recommended."
        elif decision == "KEEP":
            explanation = f"Confidently stable: q50={q50:.3f}, q90={q90:.3f} eV, p_stable={p_stable:.2f}"
        elif decision == "KILL":
            explanation = f"Confidently unstable: q50={q50:.3f}, q10={q10:.3f} eV"
        else:
            explanation = f"Uncertain: q50={q50:.3f} [{q10:.3f}, {q90:.3f}] eV"

        return DecisionOutput(
            material_id=material_id,
            formula=actual_formula,
            ehull_pred=q50,
            ehull_lower=q10,
            ehull_upper=q90,
            uncertainty_aleatoric=aleatoric,
            uncertainty_epistemic=eps_std,
            uncertainty_total=total_unc,
            ood_score=ood_score,
            ood_flag=ood_flag,
            ood_gates={"disagree": ood_flag},
            decision=decision,
            decision_confidence=confidence,
            decision_mode=actual_mode,
            explanation=explanation,
            p_stable=p_stable,
            p_metastable=p_meta,
            q50_per_member=q50_members,
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
            material_ids = [f"material-{i}" for i in range(n)]
        if formulas is None:
            formulas = [s.composition.reduced_formula for s in structures]
        return [
            self.predict_structure(structures[i], material_ids[i], formulas[i], mode)
            for i in range(n)
        ]
