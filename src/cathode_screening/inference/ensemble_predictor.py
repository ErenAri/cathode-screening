from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cathode_screening.common.serialization import safe_torch_load, Normalizer
from cathode_screening.models.cgcnn.model import CGCNN

logger = logging.getLogger(__name__)


def _normalize_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def _looks_like_windows_abs(value: str) -> bool:
    return len(value) > 1 and value[1] == ":"


def _build_model_from_cfg(cfg: dict, ckpt: dict, device: torch.device) -> nn.Module:
    """Instantiate a model from checkpoint config and load its weights."""
    model_type = str(cfg["model"].get("type", "cgcnn"))

    if model_type == "mace":
        from cathode_screening.models.mace_finetune.model import MACEFineTuner

        model = MACEFineTuner.from_pretrained(
            model_name=str(cfg["model"].get("backbone", "medium")),
            head_dim=int(cfg["model"].get("head_dim", 128)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            freeze_backbone=True,  # Always freeze at inference
            unfreeze_last_n=0,
            device=str(device),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = CGCNN(
            node_in=int(cfg["model"].get("node_in_dim", 6)),
            node_dim=int(cfg["model"]["node_embed_dim"]),
            edge_dim=int(cfg["model"]["edge_rbf_bins"]),
            layers=int(cfg["model"]["message_passing_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            pooling=str(cfg["model"]["pooling"]),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

    model.eval()
    return model


class EnsemblePredictor:
    """Load and run a CGCNN or MACE ensemble for fast screening."""

    def __init__(
        self,
        models: List[nn.Module],
        normalizers: List[Normalizer],
        device: torch.device,
        model_type: str = "cgcnn",
    ) -> None:
        self.models = models
        self.normalizers = normalizers
        self.device = device
        self.model_type = model_type

    @classmethod
    def from_directory(
        cls,
        ensemble_dir: str | Path,
        device: Optional[str | torch.device] = None,
    ) -> "EnsemblePredictor":
        ensemble_dir = Path(ensemble_dir)
        if device is None or device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        meta_path = ensemble_dir / "ensemble_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        models: List[nn.Module] = []
        normalizers: List[Normalizer] = []
        detected_type: Optional[str] = None

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
                    candidates.append(ensemble_dir.parent / Path(*ckpt_path.parts[2:]))

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

            model = _build_model_from_cfg(cfg, ckpt, device)
            models.append(model)

            # Track model type (all members must be the same type)
            member_type = str(cfg["model"].get("type", "cgcnn"))
            if detected_type is None:
                detected_type = member_type
            elif detected_type != member_type:
                raise ValueError(
                    f"Mixed model types in ensemble: {detected_type} vs {member_type}"
                )

            normalizer = Normalizer()
            if "normalizer" in ckpt:
                normalizer.load_state_dict(ckpt["normalizer"])
            normalizers.append(normalizer)

        return cls(
            models=models,
            normalizers=normalizers,
            device=device,
            model_type=detected_type or "cgcnn",
        )

    # ------------------------------------------------------------------
    # CGCNN graph loading / batching
    # ------------------------------------------------------------------
    def _load_graph(
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

    def _batch_graphs(
        self,
        graphs: List[str | Path | Dict[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_list = []
        src_list = []
        dst_list = []
        e_list = []
        batch_list = []

        node_offset = 0
        for i, graph in enumerate(graphs):
            x_np, src_np, dst_np, e_np = self._load_graph(graph)
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
        batch = torch.cat(batch_list, dim=0).to(self.device)
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

        if expected_node is not None and x.shape[1] != expected_node:
            raise ValueError(
                f"Node feature dim mismatch: got {x.shape[1]}, expected {expected_node}"
            )
        if expected_edge is not None and e.shape[1] != expected_edge:
            raise ValueError(
                f"Edge feature dim mismatch: got {e.shape[1]}, expected {expected_edge}"
            )

    # ------------------------------------------------------------------
    # MACE structure loading / batching
    # ------------------------------------------------------------------
    def _batch_structures(
        self,
        structures: List[str | Path | Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Batch structure NPZ files into a MACE-compatible data dict."""
        from cathode_screening.models.mace_finetune.data import _compute_neighbors

        # Infer z_table from first model
        z_table = getattr(self.models[0], "z_table", None) or list(range(1, 95))
        z_to_idx = {z: i for i, z in enumerate(z_table)}
        r_max = 5.0  # MACE default

        positions_list: list = []
        node_attrs_list: list = []
        edge_index_list: list = []
        shifts_list: list = []
        unit_shifts_list: list = []
        cells: list = []
        batch_idx: list = []
        ptr = [0]

        node_offset = 0
        for i, struct in enumerate(structures):
            if isinstance(struct, (str, Path)):
                with np.load(struct) as z:
                    atomic_numbers = z["atomic_numbers"]
                    positions = z["positions"].astype(np.float64)
                    cell = z["cell"].astype(np.float64)
            else:
                atomic_numbers = np.asarray(struct["atomic_numbers"])
                positions = np.asarray(struct["positions"], dtype=np.float64)
                cell = np.asarray(struct["cell"], dtype=np.float64)

            edge_src, edge_dst, cart_shifts, u_shifts = _compute_neighbors(
                positions, cell, r_max
            )

            n_atoms = len(atomic_numbers)
            n_types = len(z_table)
            na = np.zeros((n_atoms, n_types), dtype=np.float64)
            for j, z_val in enumerate(atomic_numbers):
                idx = z_to_idx.get(int(z_val))
                if idx is not None:
                    na[j, idx] = 1.0

            positions_list.append(torch.tensor(positions, dtype=torch.float64))
            node_attrs_list.append(torch.tensor(na, dtype=torch.float64))

            ei = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
            ei += node_offset
            edge_index_list.append(ei)

            shifts_list.append(torch.tensor(cart_shifts, dtype=torch.float64))
            unit_shifts_list.append(torch.tensor(u_shifts, dtype=torch.float64))
            cells.append(torch.tensor(cell, dtype=torch.float64))
            batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))

            node_offset += n_atoms
            ptr.append(node_offset)

        data: Dict[str, torch.Tensor] = {
            "positions": torch.cat(positions_list).to(self.device),
            "node_attrs": torch.cat(node_attrs_list).to(self.device),
            "edge_index": torch.cat(edge_index_list, dim=1).to(self.device),
            "shifts": torch.cat(shifts_list).to(self.device),
            "unit_shifts": torch.cat(unit_shifts_list).to(self.device),
            "cell": torch.stack(cells).to(self.device),
            "batch": torch.cat(batch_idx).to(self.device),
            "ptr": torch.tensor(ptr, dtype=torch.long).to(self.device),
        }
        return data

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(
        self,
        graphs: List[str | Path | Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if not graphs:
            raise ValueError("No graphs provided for prediction")

        # Prepare batched input based on model type
        if self.model_type == "mace":
            batch_data = self._batch_structures(graphs)
        else:
            x, src, dst, e, batch = self._batch_graphs(graphs)
            self._validate_graph_dims(x, e)

        q10_all = []
        q50_all = []
        q90_all = []
        p_stable_all = []
        p_meta_all = []

        for model, normalizer in zip(self.models, self.normalizers):
            model.eval()
            if self.model_type == "mace":
                q10, q50, q90, logit_stable, logit_meta = model(batch_data)
            else:
                q10, q50, q90, logit_stable, logit_meta = model(x, src, dst, e, batch)

            q10 = normalizer.denorm(q10).cpu().numpy()
            q50 = normalizer.denorm(q50).cpu().numpy()
            q90 = normalizer.denorm(q90).cpu().numpy()

            p_stable = torch.sigmoid(logit_stable).cpu().numpy()
            p_meta = torch.sigmoid(logit_meta).cpu().numpy()

            q10_all.append(q10)
            q50_all.append(q50)
            q90_all.append(q90)
            p_stable_all.append(p_stable)
            p_meta_all.append(p_meta)

        q10_all = np.stack(q10_all, axis=0)
        q50_all = np.stack(q50_all, axis=0)
        q90_all = np.stack(q90_all, axis=0)
        p_stable_all = np.stack(p_stable_all, axis=0)
        p_meta_all = np.stack(p_meta_all, axis=0)

        q10_mean = np.mean(q10_all, axis=0)
        q50_mean = np.mean(q50_all, axis=0)
        q90_mean = np.mean(q90_all, axis=0)

        if len(self.models) > 1:
            epistemic_var = np.var(q50_all, axis=0, ddof=1)
        else:
            epistemic_var = np.zeros_like(q50_mean)
        epistemic_std = np.sqrt(epistemic_var)

        p_stable_mean = np.mean(p_stable_all, axis=0)
        p_meta_mean = np.mean(p_meta_all, axis=0)

        return {
            "q10": q10_mean,
            "q50": q50_mean,
            "q90": q90_mean,
            "epistemic_var": epistemic_var,
            "epistemic_std": epistemic_std,
            "p_stable": p_stable_mean,
            "p_metastable": p_meta_mean,
        }

    def predict(
        self,
        graph: str | Path | Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        outputs = self.predict_batch([graph])
        return {k: float(v[0]) for k, v in outputs.items()}
