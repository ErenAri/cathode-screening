"""
Model registry for tracking, versioning, and promoting model artifacts.

Supports:
    - MLflow backend (production)
    - Local JSON registry (development / offline)

Each registered model includes:
    - Version, architecture, training config
    - Governance metrics (Spearman, coverage, precision, false-kill)
    - Artifact paths and checksums
    - Promotion status (staging → production → archived)

Configuration:
    CATHODE_REGISTRY_BACKEND=mlflow|local  (default: local)
    CATHODE_MLFLOW_TRACKING_URI=http://mlflow:5000
    CATHODE_REGISTRY_DIR=data/model_registry
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ModelVersion:
    """Represents a single registered model version."""

    def __init__(
        self,
        version: str,
        model_type: str,
        architecture: str,
        training_config: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_path: str,
        artifact_checksum: Optional[str] = None,
        stage: str = "staging",
        registered_at: Optional[str] = None,
        promoted_at: Optional[str] = None,
        notes: str = "",
    ):
        self.version = version
        self.model_type = model_type
        self.architecture = architecture
        self.training_config = training_config
        self.metrics = metrics
        self.artifact_path = artifact_path
        self.artifact_checksum = artifact_checksum
        self.stage = stage  # staging | production | archived
        self.registered_at = registered_at or datetime.now(timezone.utc).isoformat()
        self.promoted_at = promoted_at
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "artifact_checksum": self.artifact_checksum,
            "stage": self.stage,
            "registered_at": self.registered_at,
            "promoted_at": self.promoted_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        return cls(**d)

    def passes_governance(self) -> bool:
        """Check if this model passes minimum governance gates."""
        m = self.metrics
        checks = [
            m.get("spearman_val", 0) > 0.5,
            m.get("spearman_test", 0) > 0.5,
            m.get("coverage_test", 0) >= 0.90,
            m.get("false_kill_rate", 1) < 0.02,
            m.get("keep_precision", 0) > 0.85,
            m.get("test_mae", 1) < 0.050,
        ]
        return all(checks)


# ---------------------------------------------------------------------------
# Local JSON registry
# ---------------------------------------------------------------------------


class LocalModelRegistry:
    """File-based model registry using JSON."""

    def __init__(self, registry_dir: Optional[str] = None):
        self.registry_dir = Path(
            registry_dir or os.getenv("CATHODE_REGISTRY_DIR", "data/model_registry")
        )
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "registry.json"

    def _load_index(self) -> Dict[str, Any]:
        if not self._index_path.exists():
            return {"models": {}, "current_production": None}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _save_index(self, index: Dict[str, Any]) -> None:
        self._index_path.write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    def register(self, model: ModelVersion) -> ModelVersion:
        """Register a new model version."""
        index = self._load_index()

        if model.version in index["models"]:
            raise ValueError(f"Model version {model.version} already registered")

        # Compute artifact checksum if not provided
        if model.artifact_checksum is None:
            artifact = Path(model.artifact_path)
            if artifact.exists() and artifact.is_dir():
                model.artifact_checksum = self._dir_checksum(artifact)
            elif artifact.exists():
                model.artifact_checksum = self._file_checksum(artifact)

        index["models"][model.version] = model.to_dict()
        self._save_index(index)

        # Save version-specific metadata
        version_dir = self.registry_dir / model.version
        version_dir.mkdir(exist_ok=True)
        (version_dir / "metadata.json").write_text(
            json.dumps(model.to_dict(), indent=2), encoding="utf-8"
        )

        logger.info("Registered model %s (stage=%s)", model.version, model.stage)
        return model

    def promote(self, version: str, to_stage: str = "production") -> ModelVersion:
        """Promote a model version to a new stage.

        When promoting to 'production', the previous production model
        is automatically archived.
        """
        index = self._load_index()
        if version not in index["models"]:
            raise ValueError(f"Model version {version} not found")

        model_data = index["models"][version]
        model = ModelVersion.from_dict(model_data)

        if to_stage == "production":
            # Governance gate
            if not model.passes_governance():
                raise ValueError(
                    f"Model {version} does not pass governance gates. "
                    f"Cannot promote to production."
                )

            # Archive current production
            current_prod = index.get("current_production")
            if current_prod and current_prod in index["models"]:
                index["models"][current_prod]["stage"] = "archived"

            index["current_production"] = version

        model.stage = to_stage
        model.promoted_at = datetime.now(timezone.utc).isoformat()
        index["models"][version] = model.to_dict()
        self._save_index(index)

        logger.info("Promoted model %s to %s", version, to_stage)
        return model

    def get_production(self) -> Optional[ModelVersion]:
        """Get the current production model."""
        index = self._load_index()
        version = index.get("current_production")
        if version and version in index["models"]:
            return ModelVersion.from_dict(index["models"][version])
        return None

    def list_versions(self, stage: Optional[str] = None) -> List[ModelVersion]:
        """List all registered model versions, optionally filtered by stage."""
        index = self._load_index()
        versions = []
        for data in index["models"].values():
            mv = ModelVersion.from_dict(data)
            if stage is None or mv.stage == stage:
                versions.append(mv)
        return sorted(versions, key=lambda m: m.registered_at, reverse=True)

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        index = self._load_index()
        if version in index["models"]:
            return ModelVersion.from_dict(index["models"][version])
        return None

    @staticmethod
    def _file_checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    @staticmethod
    def _dir_checksum(path: Path) -> str:
        h = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(str(f.stat().st_size).encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# MLflow registry adapter
# ---------------------------------------------------------------------------


class MLflowModelRegistry:
    """MLflow-backed model registry for production deployments."""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "CATHODE_MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

    def _client(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            return mlflow.tracking.MlflowClient(self.tracking_uri)
        except ImportError:
            raise RuntimeError(
                "mlflow is required for MLflow registry. "
                "Install with: pip install mlflow"
            )

    def register(self, model: ModelVersion) -> ModelVersion:
        """Register a model in MLflow Model Registry."""
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)

        with mlflow.start_run(run_name=f"register-{model.version}"):
            # Log metrics
            for key, value in model.metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log params
            mlflow.log_param("model_type", model.model_type)
            mlflow.log_param("architecture", model.architecture)
            mlflow.log_param("version", model.version)

            # Log training config
            for key, value in model.training_config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"train_{key}", value)

            # Log artifact
            artifact = Path(model.artifact_path)
            if artifact.exists():
                if artifact.is_dir():
                    mlflow.log_artifacts(str(artifact), "model")
                else:
                    mlflow.log_artifact(str(artifact))

            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

            result = mlflow.register_model(
                model_uri,
                "cathode-screening-ensemble",
            )
            logger.info(
                "Registered model %s in MLflow (version %s)",
                model.version, result.version,
            )

        return model

    def promote(self, version: str, to_stage: str = "Production") -> ModelVersion:
        """Transition a model version stage in MLflow."""
        client = self._client()
        # MLflow uses capitalized stage names
        stage_map = {
            "staging": "Staging",
            "production": "Production",
            "archived": "Archived",
        }
        mlflow_stage = stage_map.get(to_stage, to_stage)

        client.transition_model_version_stage(
            name="cathode-screening-ensemble",
            version=version,
            stage=mlflow_stage,
        )
        logger.info("Promoted model version %s to %s in MLflow", version, mlflow_stage)
        return ModelVersion(version=version, model_type="", architecture="",
                          training_config={}, metrics={}, artifact_path="",
                          stage=to_stage)

    def get_production(self) -> Optional[ModelVersion]:
        """Get the current Production model from MLflow."""
        client = self._client()
        versions = client.get_latest_versions(
            "cathode-screening-ensemble", stages=["Production"]
        )
        if not versions:
            return None
        mv = versions[0]
        return ModelVersion(
            version=str(mv.version),
            model_type="mace",
            architecture="MACE-MP-0 ensemble",
            training_config={},
            metrics={},
            artifact_path=mv.source,
            stage="production",
        )

    def list_versions(self, stage: Optional[str] = None) -> List[ModelVersion]:
        """List model versions from MLflow."""
        client = self._client()
        stages = [stage.capitalize()] if stage else None
        if stages:
            versions = client.get_latest_versions(
                "cathode-screening-ensemble", stages=stages
            )
        else:
            versions = client.search_model_versions("name='cathode-screening-ensemble'")

        return [
            ModelVersion(
                version=str(v.version),
                model_type="mace",
                architecture="MACE-MP-0 ensemble",
                training_config={},
                metrics={},
                artifact_path=v.source or "",
                stage=v.current_stage.lower() if v.current_stage else "none",
            )
            for v in versions
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_registry():
    """Get the configured model registry backend."""
    backend = os.getenv("CATHODE_REGISTRY_BACKEND", "local").strip().lower()

    if backend == "mlflow":
        return MLflowModelRegistry()
    return LocalModelRegistry()
