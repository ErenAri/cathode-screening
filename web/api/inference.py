"""
Inference service for CathodeScreen API.
Supports both CHGNet (v1-Li-Cathode) and CGCNN (legacy) models.
"""
from __future__ import annotations

import os
from typing import Optional, Union
import sys
from pathlib import Path

# Ensure core package is importable when running from web/api
SRC_PATH = Path(__file__).parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import types for type hints
from cathode_screening.inference.predictor import DecisionService
from cathode_screening.inference.chgnet_adapter import CHGNetDecisionService

# Type alias for model service
ModelService = Union[DecisionService, CHGNetDecisionService]

_predictor: Optional[ModelService] = None


def get_model_type() -> str:
    """Get configured model type from environment."""
    return os.getenv("CATHODE_MODEL_TYPE", "chgnet").strip().lower()


def get_predictor() -> ModelService:
    """Get or create the global decision-grade predictor.
    
    Model type is determined by CATHODE_MODEL_TYPE env var:
    - "chgnet" (default): v1-Li-Cathode CHGNet ensemble
    - "cgcnn": Legacy CGCNN ensemble
    
    Returns:
        DecisionService or CHGNetDecisionService instance
    """
    global _predictor
    
    if _predictor is None:
        model_type = get_model_type()
        
        if model_type == "chgnet":
            print(f"Loading CHGNet v1-Li-Cathode ensemble...")
            _predictor = CHGNetDecisionService.from_env()
        elif model_type == "cgcnn":
            print(f"Loading CGCNN legacy ensemble...")
            _predictor = DecisionService.from_env()
        else:
            raise ValueError(
                f"Unknown CATHODE_MODEL_TYPE: {model_type}. "
                "Use 'chgnet' or 'cgcnn'"
            )
        
        print(f"Model loaded: {type(_predictor).__name__}")
    
    return _predictor


def reset_predictor() -> None:
    """Clear cached predictor (for testing)."""
    global _predictor
    _predictor = None
