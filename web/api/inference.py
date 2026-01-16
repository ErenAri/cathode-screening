"""
Inference service for CathodeScreen API.
Uses the decision-grade pipeline from the core package.
"""
from __future__ import annotations

from typing import Optional
import sys
from pathlib import Path

# Ensure core package is importable when running from web/api
SRC_PATH = Path(__file__).parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cathode_screening.inference.predictor import DecisionService

_predictor: Optional[DecisionService] = None


def get_predictor() -> DecisionService:
    """Get or create the global decision-grade predictor."""
    global _predictor
    if _predictor is None:
        _predictor = DecisionService.from_env()
    return _predictor
