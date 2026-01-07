"""
Tiered screening pipeline.

Two-stage screening: fast CGCNN for initial ranking,
refined ALIGNN for top-k candidates.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure


class TieredScreener:
    """
    Two-stage screening pipeline for cathode materials.
    
    Stage 1 (CGCNN): Fast screening of all candidates
    Stage 2 (ALIGNN): Refined predictions on top-k
    
    Usage:
        screener = TieredScreener(
            cgcnn_ensemble_dir="artifacts/models/ensemble_xxx",
            alignn_checkpoint="artifacts/models/alignn/best.pt",
        )
        results = screener.screen(structures, top_k=100)
    """
    
    def __init__(
        self,
        cgcnn_ensemble_dir: Optional[str] = None,
        alignn_checkpoint: Optional[str] = None,
        alignn_pretrained: str = "jv_formation_energy_peratom_alignn",
        device: str = "cuda",
        top_k: int = 100,
    ):
        """
        Initialize tiered screener.
        
        Args:
            cgcnn_ensemble_dir: Path to CGCNN ensemble directory
            alignn_checkpoint: Path to fine-tuned ALIGNN checkpoint
            alignn_pretrained: Name of pretrained ALIGNN model (if no checkpoint)
            device: Device for inference
            top_k: Number of top candidates for ALIGNN refinement
        """
        self.device = device
        self.top_k = top_k
        
        self.cgcnn_ensemble = None
        self.alignn_model = None
        
        # Load CGCNN ensemble if provided
        if cgcnn_ensemble_dir:
            self._load_cgcnn_ensemble(cgcnn_ensemble_dir)
        
        # Load ALIGNN model
        if alignn_checkpoint:
            self._load_alignn_checkpoint(alignn_checkpoint)
        elif alignn_pretrained:
            self._load_alignn_pretrained(alignn_pretrained)
    
    def _load_cgcnn_ensemble(self, ensemble_dir: str) -> None:
        """Load CGCNN ensemble from directory."""
        # Import here to avoid circular dependency
        from cathode_screening.inference.ensemble_predictor import EnsemblePredictor
        
        self.cgcnn_ensemble = EnsemblePredictor.from_directory(
            ensemble_dir,
            device=self.device,
        )
        print(f"Loaded CGCNN ensemble from {ensemble_dir}")
    
    def _load_alignn_checkpoint(self, checkpoint_path: str) -> None:
        """Load ALIGNN from checkpoint."""
        from cathode_screening.models.alignn.wrapper import ALIGNNWrapper
        
        self.alignn_model = ALIGNNWrapper(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        print(f"Loaded ALIGNN from {checkpoint_path}")
    
    def _load_alignn_pretrained(self, model_name: str) -> None:
        """Load pretrained ALIGNN."""
        from cathode_screening.models.alignn.wrapper import ALIGNNWrapper
        
        self.alignn_model = ALIGNNWrapper.from_pretrained(model_name)
        print(f"Loaded pretrained ALIGNN: {model_name}")
    
    def screen_stage1_cgcnn(
        self,
        material_ids: List[str],
        graphs: List,
    ) -> pd.DataFrame:
        """
        Stage 1: Fast CGCNN screening of all candidates.
        
        Args:
            material_ids: List of material IDs
            graphs: List of graph data for CGCNN
        
        Returns:
            DataFrame with CGCNN predictions, sorted by q50 ascending
        """
        if self.cgcnn_ensemble is None:
            raise RuntimeError("CGCNN ensemble not loaded")
        
        start = time.time()
        
        # Run ensemble inference
        predictions = self.cgcnn_ensemble.predict_batch(graphs)
        
        elapsed = time.time() - start
        print(f"Stage 1 (CGCNN): {len(material_ids)} samples in {elapsed:.1f}s "
              f"({elapsed/len(material_ids)*1000:.1f}ms/sample)")
        
        # Build results DataFrame
        df = pd.DataFrame({
            "material_id": material_ids,
            "cgcnn_q50": predictions["q50"],
            "cgcnn_q10": predictions["q10"],
            "cgcnn_q90": predictions["q90"],
            "cgcnn_epistemic_std": predictions.get("epistemic_std", np.nan),
            "cgcnn_p_stable": predictions.get("p_stable", np.nan),
        })
        
        # Sort by predicted E_hull (lower = more promising)
        df = df.sort_values("cgcnn_q50", ascending=True).reset_index(drop=True)
        df["cgcnn_rank"] = range(1, len(df) + 1)
        
        return df
    
    def screen_stage2_alignn(
        self,
        df: pd.DataFrame,
        structures: Dict[str, Structure],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Stage 2: ALIGNN refinement on top-k candidates.
        
        Args:
            df: DataFrame from Stage 1 with CGCNN predictions
            structures: Dict mapping material_id -> pymatgen Structure
            top_k: Number of top candidates to refine (default: self.top_k)
        
        Returns:
            DataFrame with ALIGNN predictions for top-k
        """
        if self.alignn_model is None:
            raise RuntimeError("ALIGNN model not loaded")
        
        top_k = top_k or self.top_k
        
        # Get top-k candidates
        top_df = df.head(top_k).copy()
        top_ids = top_df["material_id"].tolist()
        
        print(f"Stage 2 (ALIGNN): Refining top {len(top_ids)} candidates...")
        
        start = time.time()
        
        # Get structures for top-k
        top_structures = [structures[mid] for mid in top_ids]
        
        # Run ALIGNN predictions
        alignn_preds = self.alignn_model.predict_batch(top_structures)
        
        elapsed = time.time() - start
        print(f"Stage 2 (ALIGNN): {len(top_ids)} samples in {elapsed:.1f}s "
              f"({elapsed/len(top_ids)*1000:.1f}ms/sample)")
        
        # Add ALIGNN predictions
        top_df["alignn_pred"] = alignn_preds
        
        # Compute combined score (weighted average)
        # Weight ALIGNN more since it's more accurate
        alpha = 0.7  # ALIGNN weight
        top_df["combined_score"] = (
            alpha * top_df["alignn_pred"] +
            (1 - alpha) * top_df["cgcnn_q50"]
        )
        
        # Re-rank by combined score
        top_df = top_df.sort_values("combined_score", ascending=True).reset_index(drop=True)
        top_df["final_rank"] = range(1, len(top_df) + 1)
        
        return top_df
    
    def screen(
        self,
        material_ids: List[str],
        graphs: List,
        structures: Dict[str, Structure],
        top_k: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full two-stage screening pipeline.
        
        Args:
            material_ids: List of material IDs
            graphs: List of graph data for CGCNN
            structures: Dict mapping material_id -> pymatgen Structure
            top_k: Number of candidates for ALIGNN refinement
        
        Returns:
            Tuple of (full_results_df, top_k_refined_df)
        """
        # Stage 1: CGCNN screening
        full_df = self.screen_stage1_cgcnn(material_ids, graphs)
        
        # Stage 2: ALIGNN refinement on top-k
        refined_df = self.screen_stage2_alignn(full_df, structures, top_k)
        
        return full_df, refined_df


def compute_tiered_metrics(
    refined_df: pd.DataFrame,
    y_true: np.ndarray,
    threshold_stable: float = 0.05,
) -> Dict[str, float]:
    """
    Compute metrics for tiered screening results.
    
    Args:
        refined_df: DataFrame from tiered screening
        y_true: Ground truth E_hull values
        threshold_stable: Stability threshold
    
    Returns:
        Dict with precision, recall, DAF metrics
    """
    from cathode_screening.evaluation.topk import discovery_acceleration_factor
    
    # Map material_ids to y_true
    # Assuming refined_df has material_id and y_true columns
    
    metrics = {}
    
    # Precision in top-k
    is_stable = refined_df["y_true"] < threshold_stable if "y_true" in refined_df.columns else None
    
    if is_stable is not None:
        for k in [10, 25, 50]:
            if k <= len(refined_df):
                prec_k = is_stable.head(k).mean()
                metrics[f"precision_at_{k}"] = float(prec_k)
    
    return metrics
