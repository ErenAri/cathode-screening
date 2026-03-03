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
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cathode_screening.common.serialization import safe_torch_load, Normalizer
from cathode_screening.inference.artifact_manifest import (
    validate_manifest,
    verify_manifest_signature,
)
from cathode_screening.inference.ood import OODGate
from cathode_screening.evaluation.conformal import (
    ConformalCalibrator,
)
from cathode_screening.models.cgcnn.model import CGCNN

logger = logging.getLogger(__name__)

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

    # Decision
    decision: str                 # KEEP / MAYBE / KILL
    decision_confidence: float    # [0, 1] based on margin to thresholds
    decision_mode: str            # "dft_followup" or "experimental"

    # Explanation
    explanation: str = ""         # Human-readable rationale

    # Optional probabilities (if available)
    p_stable: Optional[float] = None
    p_metastable: Optional[float] = None

    # OOD details
    ood_gates: Dict[str, bool] = field(default_factory=dict)

    # Raw predictions (for debugging)
    q50_per_member: List[float] = field(default_factory=list)

    # Multi-property predictions (analytical, from structure/composition)
    cathode_properties: Optional[Dict] = None

    def to_dict(self) -> Dict:
        result = {
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
            "p_stable": self.p_stable,
            "p_metastable": self.p_metastable,
        }
        if self.cathode_properties is not None:
            result["cathode_properties"] = self.cathode_properties
        return result


def _normalize_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def _looks_like_windows_abs(value: str) -> bool:
    return len(value) > 1 and value[1] == ":"


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

        manifest_path = artifact_dir / "manifest.json"
        validate_manifest_flag = os.getenv("CATHODE_VALIDATE_ARTIFACTS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        strict_manifest = os.getenv("CATHODE_STRICT_ARTIFACTS", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        require_signature = os.getenv("CATHODE_REQUIRE_MANIFEST_SIGNATURE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if validate_manifest_flag and manifest_path.exists():
            issues = validate_manifest(manifest_path, artifact_dir)
            if issues:
                detail = ", ".join(issues[:5])
                if len(issues) > 5:
                    detail = f"{detail} (+{len(issues) - 5} more)"
                message = f"Artifact manifest validation failed ({len(issues)} issues): {detail}"
                if strict_manifest:
                    raise RuntimeError(message)
                logger.warning(message)
        if require_signature and not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for signature check: {manifest_path}")
        if manifest_path.exists():
            signature_path = os.getenv("CATHODE_MANIFEST_SIGNATURE_PATH")
            signature_path = Path(signature_path) if signature_path else manifest_path.with_suffix(".sig")
            signing_key = os.getenv("CATHODE_MANIFEST_HMAC_KEY")
            if require_signature or signing_key:
                if not signing_key:
                    raise RuntimeError("CATHODE_MANIFEST_HMAC_KEY must be set for signature verification")
                if not signature_path.exists():
                    if require_signature:
                        raise FileNotFoundError(f"Manifest signature not found: {signature_path}")
                    logger.warning("Manifest signature not found at %s", signature_path)
                else:
                    ok = verify_manifest_signature(manifest_path, signature_path, signing_key)
                    if not ok:
                        message = f"Manifest signature verification failed: {signature_path}"
                        if require_signature:
                            raise RuntimeError(message)
                        logger.warning(message)
        
        # Load ensemble metadata (artifact root or direct ensemble dir)
        ensemble_dir = artifact_dir
        meta_path = ensemble_dir / "ensemble_meta.json"
        if not meta_path.exists():
            ensemble_dir = artifact_dir / "ensemble"
            meta_path = ensemble_dir / "ensemble_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found under {artifact_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        models = []
        normalizers = []
        
        for member in meta["members"]:
            raw_ckpt = member.get("checkpoint")
            if not raw_ckpt:
                raise ValueError("Missing checkpoint entry in ensemble metadata")

            raw_ckpt_str = str(raw_ckpt)
            ckpt_path = _normalize_path(raw_ckpt_str)
            candidates = [ckpt_path]
            if not ckpt_path.is_absolute() and not _looks_like_windows_abs(raw_ckpt_str):
                candidates.append(ensemble_dir / ckpt_path)
                if len(ckpt_path.parts) >= 2 and ckpt_path.parts[:2] == ("data", "artifacts"):
                    candidates.append(artifact_dir / Path(*ckpt_path.parts[2:]))
                candidates.append(artifact_dir / ckpt_path)

            for candidate in candidates:
                if candidate.exists():
                    ckpt_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried: {', '.join(str(c) for c in candidates)}"
                )
            ckpt = safe_torch_load(ckpt_path, device)
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
        
        # Load calibration params (global conformal)
        calibrator = None
        calib_candidates = [
            artifact_dir / "calibration" / "ensemble" / "conformal_params.json",
            artifact_dir / "calibration" / "conformal_params.json",
            artifact_dir / "calibration_global" / "conformal_params.json",
        ]
        for calib_path in calib_candidates:
            if calib_path.exists():
                try:
                    calibrator = ConformalCalibrator.from_file(calib_path)
                    logger.info(
                        "Loaded conformal calibration from %s: "
                        "delta_upper=%.4f, delta_lower=%.4f",
                        calib_path, calibrator.delta_upper, calibrator.delta_lower,
                    )
                except Exception as exc:
                    logger.warning("Could not load calibration params: %s", exc)
                break

        # Load OOD gate
        ood_dir = artifact_dir / "ood"
        if (ood_dir / "ensemble").exists():
            ood_dir = ood_dir / "ensemble"
        ood_gate = OODGate.from_artifacts(ood_dir)
        
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
    
    def _prepare_graph(
        self,
        graph: str | Path | Dict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load graph from NPZ file or in-memory dict."""
        x_np, src_np, dst_np, e_np = self._load_graph_arrays(graph)

        x = torch.from_numpy(x_np).to(self.device)
        src = torch.from_numpy(src_np).long().to(self.device)
        dst = torch.from_numpy(dst_np).long().to(self.device)
        e = torch.from_numpy(e_np).to(self.device)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        return x, src, dst, e, batch

    def _load_graph_arrays(
        self,
        graph: str | Path | Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(graph, (str, Path)):
            with np.load(graph) as z:
                x_np = np.asarray(z["x"], dtype=np.float32)
                src_np = np.asarray(z["edge_src"], dtype=np.int64)
                dst_np = np.asarray(z["edge_dst"], dtype=np.int64)
                e_np = np.asarray(z["e_attr"], dtype=np.float32)
        else:
            x_np = np.asarray(graph["x"], dtype=np.float32)
            src_np = np.asarray(graph["edge_src"], dtype=np.int64)
            dst_np = np.asarray(graph["edge_dst"], dtype=np.int64)
            e_np = np.asarray(graph["e_attr"], dtype=np.float32)
        return x_np, src_np, dst_np, e_np

    def _prepare_batch_graphs(
        self,
        graphs: List[str | Path | Dict[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not graphs:
            raise ValueError("No graphs provided for prediction")

        x_list = []
        src_list = []
        dst_list = []
        e_list = []
        batch_list = []
        node_offset = 0

        for i, graph in enumerate(graphs):
            x_np, src_np, dst_np, e_np = self._load_graph_arrays(graph)
            if x_np.ndim != 2 or e_np.ndim != 2:
                raise ValueError("Graph tensors must be 2D for nodes and edges")

            x_t = torch.from_numpy(x_np)
            src_t = torch.from_numpy(src_np).long() + node_offset
            dst_t = torch.from_numpy(dst_np).long() + node_offset
            e_t = torch.from_numpy(e_np)

            x_list.append(x_t)
            src_list.append(src_t)
            dst_list.append(dst_t)
            e_list.append(e_t)
            batch_list.append(torch.full((x_t.shape[0],), i, dtype=torch.long))

            node_offset += x_t.shape[0]

        x = torch.cat(x_list, dim=0).to(self.device)
        src = torch.cat(src_list, dim=0).long().to(self.device)
        dst = torch.cat(dst_list, dim=0).long().to(self.device)
        e = torch.cat(e_list, dim=0).to(self.device)
        batch = torch.cat(batch_list, dim=0).long().to(self.device)
        return x, src, dst, e, batch

    def _validate_graph_dims(self, x: torch.Tensor, e: torch.Tensor) -> None:
        if not self.models:
            return
        model = self.models[0]
        expected_node = getattr(model.embed, "in_features", None)
        expected_edge = None
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            node_dim = model.embed.out_features
            expected_edge = model.blocks[0].lin_msg.in_features - (2 * node_dim)

        if x.ndim != 2:
            raise ValueError(f"Expected node features to be 2D, got shape {tuple(x.shape)}")
        if e.ndim != 2:
            raise ValueError(f"Expected edge features to be 2D, got shape {tuple(e.shape)}")
        if expected_node is not None and x.shape[1] != expected_node:
            raise ValueError(
                f"Node feature dim mismatch: got {x.shape[1]}, expected {expected_node}"
            )
        if expected_edge is not None and e.shape[1] != expected_edge:
            raise ValueError(
                f"Edge feature dim mismatch: got {e.shape[1]}, expected {expected_edge}"
            )
    
    def _get_embedding(self, model: nn.Module, x, src, dst, e, batch) -> np.ndarray:
        """Extract graph embedding from model (after pooling, before heads)."""
        if hasattr(model, "get_embedding"):
            g = model.get_embedding(x, src, dst, e, batch)
        else:
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
        graph_npz_path: str | Path | Dict[str, np.ndarray],
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
        x, src, dst, e, batch = self._prepare_graph(graph_npz_path)
        self._validate_graph_dims(x, e)
        
        # Run ensemble inference
        q10_raw, q50_raw, q90_raw = [], [], []
        p_stable_raw, p_meta_raw = [], []
        embeddings = []

        for k, (model, norm) in enumerate(zip(self.models, self.normalizers)):
            model.eval()
            q10, q50, q90, logit_stable, logit_meta = model(x, src, dst, e, batch)
            
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

            # Probabilities
            p_stable_raw.append(float(torch.sigmoid(logit_stable).cpu().item()))
            p_meta_raw.append(float(torch.sigmoid(logit_meta).cpu().item()))
        
        # Ensemble aggregation
        q50_ens = float(np.mean(q50_raw))
        q10_ens = float(np.mean(q10_raw))
        q90_ens = float(np.mean(q90_raw))
        
        # Aggregate probabilities
        p_stable = float(np.mean(p_stable_raw)) if p_stable_raw else None
        p_metastable = float(np.mean(p_meta_raw)) if p_meta_raw else None

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
        embedding_concat = np.concatenate(embeddings) if embeddings else None
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
            p_stable=p_stable,
            p_metastable=p_metastable,
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        materials: List[Dict],  # [{"material_id": ..., "formula": ..., "graph_npz": ...}, ...]
        mode: str = "dft_followup",
    ) -> List[DecisionOutput]:
        """Batch prediction for multiple materials."""
        if not materials:
            return []

        if mode not in self.thresholds:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'dft_followup' or 'experimental'"
            )

        material_ids = [m["material_id"] for m in materials]
        formulas = [m["formula"] for m in materials]
        graphs = [m["graph_npz"] for m in materials]

        x, src, dst, e, batch = self._prepare_batch_graphs(graphs)
        self._validate_graph_dims(x, e)

        q10_all = []
        q50_all = []
        q90_all = []
        p_stable_all = []
        p_meta_all = []
        embeddings_all = []

        for model, norm in zip(self.models, self.normalizers):
            model.eval()
            q10, q50, q90, logit_stable, logit_meta = model(x, src, dst, e, batch)

            q10_all.append(norm.denorm(q10).cpu().numpy())
            q50_all.append(norm.denorm(q50).cpu().numpy())
            q90_all.append(norm.denorm(q90).cpu().numpy())

            p_stable_all.append(torch.sigmoid(logit_stable).cpu().numpy())
            p_meta_all.append(torch.sigmoid(logit_meta).cpu().numpy())

            embeddings_all.append(self._get_embedding(model, x, src, dst, e, batch))

        q10_all = np.stack(q10_all, axis=0)
        q50_all = np.stack(q50_all, axis=0)
        q90_all = np.stack(q90_all, axis=0)
        p_stable_all = np.stack(p_stable_all, axis=0)
        p_meta_all = np.stack(p_meta_all, axis=0)

        q10_mean = np.mean(q10_all, axis=0)
        q50_mean = np.mean(q50_all, axis=0)
        q90_mean = np.mean(q90_all, axis=0)

        p_stable_mean = np.mean(p_stable_all, axis=0) if p_stable_all.size else None
        p_meta_mean = np.mean(p_meta_all, axis=0) if p_meta_all.size else None

        if self.calibrator is not None:
            q10_cal_arr, q90_cal_arr = self.calibrator.calibrate(q10_mean, q90_mean)
        else:
            q10_cal_arr = q10_mean
            q90_cal_arr = q90_mean

        sigma_ale = np.mean(q90_all - q10_all, axis=0) / 2.56
        sigma_epi = np.std(q50_all, axis=0)
        sigma_tot = np.sqrt(sigma_ale**2 + sigma_epi**2)

        outputs: List[DecisionOutput] = []

        for i, (material_id, formula) in enumerate(zip(material_ids, formulas)):
            q50_ens = float(q50_mean[i])
            q10_cal = float(q10_cal_arr[i])
            q90_cal = float(q90_cal_arr[i])

            q50_members = [float(v) for v in q50_all[:, i]]
            if embeddings_all:
                embedding_concat = np.concatenate(
                    [np.ravel(embeddings_all[k][i]) for k in range(len(embeddings_all))]
                )
            else:
                embedding_concat = None

            ood_result = self.ood_gate.score(formula, embedding_concat, q50_members)

            thresh = self.thresholds[mode]
            tau_keep = thresh["tau_keep"]
            tau_kill = thresh["tau_kill"]

            if ood_result.flag:
                decision = "MAYBE"
                decision_confidence = 0.0
                triggered = [
                    g for g in ["comp", "emb", "disagree"]
                    if ood_result.to_dict().get(f"flag_{g}")
                ]
                explanation = (
                    f"OOD detected ({', '.join(triggered)}). Manual review recommended."
                )
            elif q90_cal < tau_keep:
                decision = "KEEP"
                margin = (tau_keep - q90_cal) / tau_keep
                decision_confidence = min(1.0, margin * 2)
                explanation = (
                    f"Confidently stable: q90={q90_cal:.3f} eV < Ï„_keep={tau_keep:.3f} eV"
                )
            elif q10_cal > tau_kill:
                decision = "KILL"
                margin = (q10_cal - tau_kill) / tau_kill
                decision_confidence = min(1.0, margin * 2)
                explanation = (
                    f"Confidently unstable: q10={q10_cal:.3f} eV > Ï„_kill={tau_kill:.3f} eV"
                )
            else:
                decision = "MAYBE"
                maybe_width = tau_kill - tau_keep
                dist_to_keep = q90_cal - tau_keep
                dist_to_kill = tau_kill - q10_cal
                decision_confidence = 0.5 - 0.5 * abs(dist_to_keep - dist_to_kill) / maybe_width
                explanation = (
                    f"Uncertain: interval [{q10_cal:.3f}, {q90_cal:.3f}] overlaps decision region"
                )

            outputs.append(
                DecisionOutput(
                    material_id=material_id,
                    formula=formula,
                    ehull_pred=q50_ens,
                    ehull_lower=q10_cal,
                    ehull_upper=q90_cal,
                    uncertainty_aleatoric=float(sigma_ale[i]),
                    uncertainty_epistemic=float(sigma_epi[i]),
                    uncertainty_total=float(sigma_tot[i]),
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
                    q50_per_member=q50_members,
                    p_stable=float(p_stable_mean[i]) if p_stable_mean is not None else None,
                    p_metastable=float(p_meta_mean[i]) if p_meta_mean is not None else None,
                )
            )

        return outputs


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
