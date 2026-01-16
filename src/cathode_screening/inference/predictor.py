from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from pymatgen.core import Structure

from cathode_screening.datasets.featurize.graph_builder import GraphBuilder, GraphConfig
from cathode_screening.inference.decision_predictor import DecisionOutput, DecisionPredictor


def _env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class DecisionService:
    """High-level inference service for decision-grade predictions."""

    predictor: DecisionPredictor
    graph_builder: GraphBuilder
    default_mode: str = "dft_followup"

    @classmethod
    def from_env(cls) -> "DecisionService":
        artifacts_dir = Path(os.getenv("CATHODE_ARTIFACTS_DIR", "data/artifacts"))
        mode = os.getenv("CATHODE_DEFAULT_MODE", "dft_followup")

        device_name = os.getenv("CATHODE_DEVICE", "auto").lower()
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        graph_config = GraphConfig(
            cutoff=_env_float(os.getenv("CATHODE_GRAPH_CUTOFF"), 8.0),
            max_neighbors=_env_int(os.getenv("CATHODE_GRAPH_MAX_NEIGHBORS"), 12),
            rbf_bins=_env_int(os.getenv("CATHODE_GRAPH_RBF_BINS"), 41),
            rbf_gamma=_env_float(os.getenv("CATHODE_GRAPH_RBF_GAMMA"), 10.0),
            use_voronoi_features=_env_bool(os.getenv("CATHODE_GRAPH_USE_VORONOI"), False),
            use_angle_features=_env_bool(os.getenv("CATHODE_GRAPH_USE_ANGLES"), False),
        )

        predictor = DecisionPredictor.from_artifacts(artifacts_dir, device=device)
        graph_builder = GraphBuilder(graph_config)
        return cls(predictor=predictor, graph_builder=graph_builder, default_mode=mode)

    def predict_structure(
        self,
        structure: Structure,
        material_id: str,
        formula: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> DecisionOutput:
        graph = self.graph_builder.build(structure)
        return self.predictor.predict(
            material_id=material_id,
            formula=formula or structure.composition.reduced_formula,
            graph_npz_path=graph,
            mode=mode or self.default_mode,
        )

    def predict_structures(
        self,
        structures: Sequence[Structure],
        material_ids: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
    ) -> List[DecisionOutput]:
        if not structures:
            return []

        n = len(structures)
        if material_ids is None:
            material_ids = [f"material-{idx}" for idx in range(n)]
        if formulas is None:
            formulas = [s.composition.reduced_formula for s in structures]

        if len(material_ids) != n or len(formulas) != n:
            raise ValueError("Material IDs and formulas must match structure count")

        graphs = [self.graph_builder.build(structure) for structure in structures]
        materials = [
            {
                "material_id": material_ids[i],
                "formula": formulas[i],
                "graph_npz": graphs[i],
            }
            for i in range(n)
        ]
        return self.predictor.predict_batch(materials, mode=mode or self.default_mode)
