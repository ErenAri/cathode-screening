from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cathode_screening.common.serialization import safe_torch_load
from cathode_screening.models.cgcnn.model import CGCNN


class Normalizer:
    """Target normalizer (matches training)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def load_state_dict(self, state_dict: Dict) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def _normalize_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def _looks_like_windows_abs(value: str) -> bool:
    return len(value) > 1 and value[1] == ":"


class EnsemblePredictor:
    """Load and run a CGCNN ensemble for fast screening."""

    def __init__(
        self,
        models: List[nn.Module],
        normalizers: List[Normalizer],
        device: torch.device,
    ) -> None:
        self.models = models
        self.normalizers = normalizers
        self.device = device

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
            models.append(model)

            normalizer = Normalizer()
            if "normalizer" in ckpt:
                normalizer.load_state_dict(ckpt["normalizer"])
            normalizers.append(normalizer)

        return cls(models=models, normalizers=normalizers, device=device)

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

    @torch.no_grad()
    def predict_batch(
        self,
        graphs: List[str | Path | Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if not graphs:
            raise ValueError("No graphs provided for prediction")

        x, src, dst, e, batch = self._batch_graphs(graphs)
        self._validate_graph_dims(x, e)

        q10_all = []
        q50_all = []
        q90_all = []
        p_stable_all = []
        p_meta_all = []

        for model, normalizer in zip(self.models, self.normalizers):
            model.eval()
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
