from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from typing import Dict

logger = logging.getLogger(__name__)


class Normalizer:
    """Target normalizer for model predictions (Z-score denormalization).

    Shared implementation used by ensemble and decision predictors.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def load_state_dict(self, state_dict: Dict) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _add_numpy_safe_globals() -> None:
    try:
        import numpy as np
        from numpy.core.multiarray import scalar
    except Exception:
        return

    safe = [scalar, np.dtype]
    for dtype_name in ("float64", "float32", "int64", "int32", "bool"):
        try:
            safe.append(np.dtype(dtype_name).__class__)
        except Exception:
            continue

    try:
        torch.serialization.add_safe_globals(safe)
    except Exception:
        return


def safe_torch_load(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> Any:
    map_location = device if device is not None else "cpu"
    allow_unsafe = _env_bool("CATHODE_ALLOW_UNSAFE_TORCH_LOAD", False)
    _add_numpy_safe_globals()

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        if not allow_unsafe:
            raise RuntimeError(
                "torch.load weights_only is unavailable; set "
                "CATHODE_ALLOW_UNSAFE_TORCH_LOAD=true to allow unsafe loading."
            ) from exc
        logger.warning("Falling back to unsafe torch.load for %s", path)
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        msg = str(exc).lower()
        weights_only_error = (
            "weights only load failed" in msg
            or "weightsunpickler error" in msg
            or "weights_only" in msg
            or "unpickler" in msg
            or "pickle" in msg
            or "_rebuild" in msg
        )
        if not weights_only_error:
            raise
        if not allow_unsafe:
            raise RuntimeError(
                "Safe torch.load failed; set CATHODE_ALLOW_UNSAFE_TORCH_LOAD=true "
                "to allow unsafe loading."
            ) from exc
        logger.warning("Safe torch.load failed for %s (%s). Using unsafe load.", path, msg)
        return torch.load(path, map_location=map_location, weights_only=False)
