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


_NUMPY_SAFE_GLOBALS_ADDED = False


def _add_numpy_safe_globals() -> bool:
    """Register numpy types as safe for torch.load(weights_only=True).

    Returns True if registration succeeded (PyTorch ≥ 2.4), False otherwise.
    On PyTorch < 2.4, ``add_safe_globals`` does not exist and the caller
    must fall back to ``weights_only=False`` for checkpoints that contain
    numpy scalars.
    """
    global _NUMPY_SAFE_GLOBALS_ADDED
    if _NUMPY_SAFE_GLOBALS_ADDED:
        return True

    if not hasattr(torch.serialization, "add_safe_globals"):
        return False

    try:
        import numpy as np
        from numpy.core.multiarray import scalar
    except Exception:
        return False

    safe = [scalar, np.dtype]
    for dtype_name in ("float64", "float32", "int64", "int32", "bool"):
        try:
            safe.append(np.dtype(dtype_name).__class__)
        except Exception:
            continue

    try:
        torch.serialization.add_safe_globals(safe)
        _NUMPY_SAFE_GLOBALS_ADDED = True
        return True
    except Exception:
        return False


def safe_torch_load(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> Any:
    """Load a torch checkpoint with the safest available method.

    Strategy:
      1. Try ``weights_only=True`` (safest).
      2. If that fails because numpy types aren't whitelisted and
         ``add_safe_globals`` is unavailable (PyTorch < 2.4), fall back
         to ``weights_only=False`` for **our own artifact paths only**
         (the checkpoints are trusted, first-party artifacts).
      3. If ``CATHODE_ALLOW_UNSAFE_TORCH_LOAD=true`` is set, always
         allow the fallback regardless of the error type.
    """
    map_location = device if device is not None else "cpu"
    allow_unsafe = _env_bool("CATHODE_ALLOW_UNSAFE_TORCH_LOAD", False)
    has_safe_globals = _add_numpy_safe_globals()

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        # PyTorch too old for weights_only kwarg
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

        # On PyTorch < 2.4, add_safe_globals is unavailable, so numpy-
        # containing checkpoints always fail with weights_only=True.
        # These are our own training artifacts — safe to load.
        if not has_safe_globals:
            logger.info(
                "PyTorch %s lacks add_safe_globals; loading %s with weights_only=False "
                "(trusted first-party artifact).",
                torch.__version__, Path(path).name,
            )
            return torch.load(path, map_location=map_location, weights_only=False)

        if not allow_unsafe:
            raise RuntimeError(
                "Safe torch.load failed; set CATHODE_ALLOW_UNSAFE_TORCH_LOAD=true "
                "to allow unsafe loading."
            ) from exc
        logger.warning("Safe torch.load failed for %s (%s). Using unsafe load.", path, msg)
        return torch.load(path, map_location=map_location, weights_only=False)
