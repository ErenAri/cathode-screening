"""
Schema definitions for inference outputs.

Defines structured output types for predictions and decisions,
including calibration source tracking.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Literal


Decision = Literal["KEEP", "MAYBE", "KILL"]
GateLevel = Literal["IN", "BORDERLINE", "OOD"]
CalibrationSource = Literal["cluster", "global"]


@dataclass
class PredictionOutput:
    """
    Complete prediction output for a single material.
    
    Contains model predictions, calibrated bounds, uncertainty estimates,
    and decision outputs with calibration source tracking.
    """
    # Material identifier
    material_id: str
    
    # Raw model predictions
    q10: float          # 10th percentile (uncalibrated)
    q50: float          # Median prediction
    q90: float          # 90th percentile (uncalibrated)
    
    # Calibrated bounds
    q10_cal: float      # Calibrated lower bound
    q90_cal: float      # Calibrated upper bound
    
    # Uncertainty estimates
    epistemic_std: float      # Ensemble disagreement
    aleatoric_std: float      # Model's internal uncertainty estimate
    
    # OOD detection
    gate_level: GateLevel     # "IN", "BORDERLINE", "OOD"
    
    # Calibration source tracking
    calibrated_source: CalibrationSource   # "cluster" or "global"
    is_cluster_calibrated: bool            # True if per-cluster delta used
    
    # Ranking
    ranking_score: float      # q50 + lambda * epistemic_std
    
    # Decision outputs
    decision: Decision        # "KEEP", "KILL", "MAYBE"
    
    # Optional fields (must come after required fields)
    cluster_id: Optional[str] = None       # Cluster ID if available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionOutput":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class BatchPredictionSummary:
    """
    Summary statistics for a batch of predictions.
    
    Useful for monitoring and debugging.
    """
    n_total: int
    n_keep: int
    n_maybe: int
    n_kill: int
    
    # Calibration source breakdown
    n_cluster_calibrated: int
    n_global_calibrated: int
    
    # Coverage of cluster calibration
    cluster_calibration_rate: float
    
    # Gate level breakdown
    n_in: int
    n_borderline: int
    n_ood: int
    
    def __str__(self) -> str:
        lines = [
            "Batch Prediction Summary",
            "=" * 40,
            f"Total: {self.n_total}",
            f"Decisions: KEEP={self.n_keep}, MAYBE={self.n_maybe}, KILL={self.n_kill}",
            f"Calibration: cluster={self.n_cluster_calibrated} ({self.cluster_calibration_rate:.1%}), "
            f"global={self.n_global_calibrated}",
            f"Gate levels: IN={self.n_in}, BORDERLINE={self.n_borderline}, OOD={self.n_ood}",
        ]
        return "\n".join(lines)
