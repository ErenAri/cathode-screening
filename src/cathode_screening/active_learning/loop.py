"""
Active Learning Loop for cathode materials discovery.

Orchestrates the iterative process of:
1. Predict on unlabeled pool
2. Select informative candidates via acquisition function
3. Query oracle (DFT/experiment) for labels
4. Add to training set and retrain
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ALState:
    """State of the active learning loop."""
    
    iteration: int = 0
    n_queries: int = 0
    
    # Indices
    train_indices: List[int] = field(default_factory=list)
    queried_indices: List[int] = field(default_factory=list)
    
    # History
    cumulative_stable_found: List[int] = field(default_factory=list)
    cumulative_queries: List[int] = field(default_factory=list)
    
    # Per-iteration metrics
    iteration_metrics: List[Dict] = field(default_factory=list)


class ActiveLearningLoop:
    """
    Active learning loop for efficient materials discovery.
    
    Example usage:
        al = ActiveLearningLoop(
            pool_df=candidates_df,
            oracle_fn=lambda ids: get_dft_labels(ids),
            acquisition_fn="expected_improvement",
        )
        
        for _ in range(10):
            al.step(batch_size=10)
            print(f"Found {al.state.cumulative_stable_found[-1]} stable materials")
    """
    
    ACQUISITION_FUNCTIONS = {
        "uncertainty": "uncertainty_sampling",
        "expected_improvement": "expected_improvement",
        "ucb": "upper_confidence_bound",
        "pi": "probability_of_improvement",
        "random": "random_sampling",
    }
    
    def __init__(
        self,
        pool_df: pd.DataFrame,
        oracle_fn: Callable[[List[str]], Dict[str, float]],
        predictor: Optional[object] = None,
        acquisition_fn: str = "expected_improvement",
        stability_threshold: float = 0.05,
        initial_train_size: int = 100,
        seed: int = 42,
    ):
        """
        Initialize active learning loop.
        
        Args:
            pool_df: DataFrame with candidate materials (must have 'material_id')
            oracle_fn: Function to query labels: fn(material_ids) -> {id: e_hull}
            predictor: Model predictor (if None, uses simulated predictions)
            acquisition_fn: Name of acquisition function
            stability_threshold: E_hull threshold for stability (eV/atom)
            initial_train_size: Initial random training set size
            seed: Random seed
        """
        self.pool_df = pool_df.copy().reset_index(drop=True)
        self.oracle_fn = oracle_fn
        self.predictor = predictor
        self.acquisition_fn = acquisition_fn
        self.stability_threshold = stability_threshold
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Initialize state
        self.state = ALState()
        
        # Initialize with random sample
        if initial_train_size > 0:
            self._initialize_random_train(initial_train_size)
    
    def _initialize_random_train(self, n: int) -> None:
        """Initialize with random training samples."""
        n_pool = len(self.pool_df)
        initial_idx = self.rng.choice(n_pool, size=min(n, n_pool), replace=False)
        
        self.state.train_indices = list(initial_idx)
        self.state.queried_indices = list(initial_idx)
        self.state.n_queries = len(initial_idx)
        
        # Get initial labels
        initial_ids = self.pool_df.loc[initial_idx, "material_id"].tolist()
        self._query_oracle(initial_ids)
        
        # Record initial state
        self._record_metrics()
    
    def _query_oracle(self, material_ids: List[str]) -> Dict[str, float]:
        """Query oracle for labels."""
        labels = self.oracle_fn(material_ids)
        
        # Update pool_df with labels
        for mid, label in labels.items():
            idx = self.pool_df[self.pool_df["material_id"] == mid].index
            if len(idx) > 0:
                self.pool_df.loc[idx[0], "y_true"] = label
        
        return labels
    
    def _get_predictions(self) -> pd.DataFrame:
        """Get model predictions for unlabeled pool."""
        # Get unlabeled indices
        all_indices = set(range(len(self.pool_df)))
        unlabeled_indices = list(all_indices - set(self.state.queried_indices))
        
        if len(unlabeled_indices) == 0:
            return pd.DataFrame()
        
        unlabeled_df = self.pool_df.loc[unlabeled_indices].copy()
        
        if self.predictor is not None:
            # Use actual model predictions
            preds = self.predictor.predict(unlabeled_df)
            unlabeled_df["pred_mean"] = preds.get("q50", preds.get("mean"))
            unlabeled_df["pred_std"] = preds.get("epistemic_std", preds.get("std", 0.1))
        else:
            # Simulated predictions for testing
            # Use actual labels with noise
            if "y_true" in unlabeled_df.columns:
                true_vals = unlabeled_df["y_true"].values
                noise = self.rng.normal(0, 0.1, len(true_vals))
                unlabeled_df["pred_mean"] = true_vals + noise
                unlabeled_df["pred_std"] = np.abs(noise) + 0.05
            else:
                unlabeled_df["pred_mean"] = self.rng.uniform(0, 1, len(unlabeled_df))
                unlabeled_df["pred_std"] = 0.1
        
        unlabeled_df["pool_index"] = unlabeled_indices
        return unlabeled_df
    
    def _select_candidates(
        self,
        predictions_df: pd.DataFrame,
        batch_size: int,
    ) -> List[int]:
        """Select candidates using acquisition function."""
        from cathode_screening.active_learning import acquisition
        
        if len(predictions_df) == 0:
            return []
        
        mean_pred = predictions_df["pred_mean"].values
        std_pred = predictions_df["pred_std"].values
        pool_indices = predictions_df["pool_index"].values
        
        k = min(batch_size, len(predictions_df))
        
        if self.acquisition_fn == "uncertainty":
            selected_local = acquisition.uncertainty_sampling(std_pred, k)
        elif self.acquisition_fn == "expected_improvement":
            selected_local = acquisition.expected_improvement(
                mean_pred, std_pred, self.stability_threshold, k
            )
        elif self.acquisition_fn == "ucb":
            selected_local = acquisition.upper_confidence_bound(
                mean_pred, std_pred, kappa=2.0, k=k
            )
        elif self.acquisition_fn == "pi":
            selected_local = acquisition.probability_of_improvement(
                mean_pred, std_pred, self.stability_threshold, k
            )
        elif self.acquisition_fn == "random":
            selected_local = acquisition.random_sampling(len(predictions_df), k, self.seed)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_fn}")
        
        # Convert to pool indices
        return [pool_indices[i] for i in selected_local]
    
    def _record_metrics(self) -> None:
        """Record current metrics."""
        # Count stable materials found
        queried_df = self.pool_df.loc[self.state.queried_indices]
        
        if "y_true" in queried_df.columns:
            n_stable = (queried_df["y_true"] < self.stability_threshold).sum()
        else:
            n_stable = 0
        
        self.state.cumulative_stable_found.append(n_stable)
        self.state.cumulative_queries.append(self.state.n_queries)
        
        # Per-iteration metrics
        self.state.iteration_metrics.append({
            "iteration": self.state.iteration,
            "n_queries": self.state.n_queries,
            "n_stable_found": n_stable,
            "pool_remaining": len(self.pool_df) - len(self.state.queried_indices),
        })
    
    def step(self, batch_size: int = 10) -> Dict:
        """
        Run one iteration of active learning.
        
        Args:
            batch_size: Number of candidates to query
        
        Returns:
            Dict with iteration results
        """
        self.state.iteration += 1
        
        # Get predictions
        predictions_df = self._get_predictions()
        
        if len(predictions_df) == 0:
            return {"status": "pool_exhausted", "iteration": self.state.iteration}
        
        # Select candidates
        selected_indices = self._select_candidates(predictions_df, batch_size)
        
        if len(selected_indices) == 0:
            return {"status": "no_candidates", "iteration": self.state.iteration}
        
        # Query oracle
        selected_ids = self.pool_df.loc[selected_indices, "material_id"].tolist()
        labels = self._query_oracle(selected_ids)
        
        # Update state
        self.state.queried_indices.extend(selected_indices)
        self.state.train_indices.extend(selected_indices)
        self.state.n_queries += len(selected_indices)
        
        # Record metrics
        self._record_metrics()
        
        # Count stable in this batch
        n_stable_batch = sum(1 for v in labels.values() if v < self.stability_threshold)
        
        return {
            "status": "ok",
            "iteration": self.state.iteration,
            "batch_size": len(selected_indices),
            "n_stable_batch": n_stable_batch,
            "cumulative_stable": self.state.cumulative_stable_found[-1],
            "cumulative_queries": self.state.n_queries,
        }
    
    def run(
        self,
        n_iterations: int,
        batch_size: int = 10,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run multiple iterations of active learning.
        
        Args:
            n_iterations: Number of AL iterations
            batch_size: Candidates per iteration
            verbose: Print progress
        
        Returns:
            DataFrame with iteration history
        """
        if verbose:
            print(f"Running {n_iterations} AL iterations with batch_size={batch_size}")
            print(f"Acquisition function: {self.acquisition_fn}")
            print(f"Initial training size: {len(self.state.train_indices)}")
            print("-" * 50)
        
        for i in range(n_iterations):
            result = self.step(batch_size)
            
            if result["status"] != "ok":
                if verbose:
                    print(f"Stopping: {result['status']}")
                break
            
            if verbose:
                print(
                    f"Iter {result['iteration']:3d} | "
                    f"Queries: {result['cumulative_queries']:4d} | "
                    f"Stable found: {result['cumulative_stable']}"
                )
        
        return pd.DataFrame(self.state.iteration_metrics)
    
    def get_discovery_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cumulative discovery curve."""
        return (
            np.array(self.state.cumulative_queries),
            np.array(self.state.cumulative_stable_found),
        )
