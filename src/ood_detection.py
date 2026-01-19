"""
Out-of-Distribution (OOD) Detection for CHGNet Predictions.

Detects when a material is unlike the training data, signaling
that predictions may be unreliable.

Methods:
1. Ensemble variance - disagreement between model predictions
2. Embedding distance - Mahalanobis distance from training distribution
3. Elemental check - presence of unseen elements
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

import torch
from pymatgen.core import Structure

# Training set elements for Li-cathode materials
TRAINING_ELEMENTS = {
    "Li", "O", "Co", "Ni", "Mn", "Fe", "V", "Ti", "Al", "Mg", 
    "Zn", "Cu", "Cr", "Na", "K", "Ca", "P", "S", "F", "Cl",
    "Mo", "W", "Nb", "Ta", "Zr", "Y", "La", "Ce", "Nd", "Sm"
}


@dataclass
class OODResult:
    """Result of OOD detection analysis."""
    is_ood: bool
    ood_score: float  # 0-1, higher = more out-of-distribution
    ensemble_variance: float
    embedding_distance: float
    has_unseen_elements: bool
    unseen_elements: List[str]
    confidence_level: str  # "high", "medium", "low"


class OODDetector:
    """
    Detects out-of-distribution samples using multiple signals.
    
    Usage:
        detector = OODDetector.load("checkpoints/ood_detector.pkl")
        result = detector.detect(structure, model_predictions)
        if result.is_ood:
            print(f"Warning: {result.unseen_elements}")
    """
    
    def __init__(
        self,
        training_embeddings: Optional[np.ndarray] = None,
        training_mean: Optional[np.ndarray] = None,
        training_cov_inv: Optional[np.ndarray] = None,
        ood_threshold: float = 0.7,
        variance_threshold: float = 0.01,
    ):
        """
        Initialize OOD detector.
        
        Args:
            training_embeddings: Embeddings of training structures
            training_mean: Mean of training embeddings
            training_cov_inv: Inverse covariance matrix
            ood_threshold: Threshold for OOD score (0-1)
            variance_threshold: Threshold for ensemble variance
        """
        self.training_embeddings = training_embeddings
        self.training_mean = training_mean
        self.training_cov_inv = training_cov_inv
        self.ood_threshold = ood_threshold
        self.variance_threshold = variance_threshold
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit the detector on training embeddings.
        
        Args:
            embeddings: (n_samples, embedding_dim) array
        """
        self.training_embeddings = embeddings
        self.training_mean = np.mean(embeddings, axis=0)
        
        # Compute covariance with regularization for stability
        cov = np.cov(embeddings.T)
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
        self.training_cov_inv = np.linalg.inv(cov)
    
    def mahalanobis_distance(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from training distribution.
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            Mahalanobis distance
        """
        if self.training_mean is None or self.training_cov_inv is None:
            return 0.0
        
        diff = embedding - self.training_mean
        dist = np.sqrt(diff @ self.training_cov_inv @ diff)
        return float(dist)
    
    def ensemble_variance(self, predictions: List[float]) -> float:
        """
        Compute variance across ensemble predictions.
        
        Args:
            predictions: List of predictions from ensemble models
            
        Returns:
            Variance of predictions
        """
        if len(predictions) < 2:
            return 0.0
        return float(np.var(predictions))
    
    def check_elements(self, structure: Structure) -> Tuple[bool, List[str]]:
        """
        Check if structure contains unseen elements.
        
        Args:
            structure: pymatgen Structure
            
        Returns:
            (has_unseen, list of unseen element symbols)
        """
        structure_elements = set(str(s.specie) for s in structure.sites)
        unseen = structure_elements - TRAINING_ELEMENTS
        return len(unseen) > 0, list(unseen)
    
    def compute_ood_score(
        self,
        ensemble_variance: float,
        embedding_distance: float,
        has_unseen_elements: bool,
    ) -> float:
        """
        Compute combined OOD score (0-1).
        
        Higher scores indicate more out-of-distribution samples.
        """
        # Normalize embedding distance (typical range: 0-20)
        normalized_distance = min(embedding_distance / 15.0, 1.0)
        
        # Normalize variance (typical range: 0-0.05)
        normalized_variance = min(ensemble_variance / 0.02, 1.0)
        
        # Combine signals
        if has_unseen_elements:
            # Unseen elements = definitely OOD
            return 1.0
        
        # Weighted combination
        score = 0.5 * normalized_distance + 0.5 * normalized_variance
        return min(score, 1.0)
    
    def detect(
        self,
        structure: Structure,
        predictions: List[float],
        embedding: Optional[np.ndarray] = None,
    ) -> OODResult:
        """
        Perform full OOD detection analysis.
        
        Args:
            structure: pymatgen Structure to analyze
            predictions: List of predictions from ensemble models
            embedding: Optional embedding vector for the structure
            
        Returns:
            OODResult with all detection metrics
        """
        # Check elements
        has_unseen, unseen_list = self.check_elements(structure)
        
        # Compute ensemble variance
        variance = self.ensemble_variance(predictions)
        
        # Compute embedding distance
        if embedding is not None and self.training_mean is not None:
            distance = self.mahalanobis_distance(embedding)
        else:
            distance = 0.0
        
        # Compute OOD score
        ood_score = self.compute_ood_score(variance, distance, has_unseen)
        
        # Determine if OOD
        is_ood = ood_score >= self.ood_threshold or has_unseen
        
        # Determine confidence level
        if ood_score < 0.3:
            confidence = "high"
        elif ood_score < 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return OODResult(
            is_ood=is_ood,
            ood_score=ood_score,
            ensemble_variance=variance,
            embedding_distance=distance,
            has_unseen_elements=has_unseen,
            unseen_elements=unseen_list,
            confidence_level=confidence,
        )
    
    def save(self, path: str):
        """Save detector to file."""
        save_dict = {
            "training_embeddings": self.training_embeddings,
            "training_mean": self.training_mean,
            "training_cov_inv": self.training_cov_inv,
            "ood_threshold": self.ood_threshold,
            "variance_threshold": self.variance_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "OODDetector":
        """Load detector from file."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        return cls(**save_dict)


def extract_chgnet_embedding(
    model,
    structure: Structure,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract embedding from CHGNet model for a structure.
    
    Args:
        model: CHGNet model
        structure: pymatgen Structure
        device: torch device
        
    Returns:
        Embedding vector (numpy array)
    """
    from chgnet.graph import CrystalGraphConverter
    
    # Convert structure to graph
    converter = CrystalGraphConverter()
    graph = converter(structure)
    
    # Get embedding from model (before final prediction layer)
    model.eval()
    with torch.no_grad():
        # This extracts the atom feature aggregation
        graph = graph.to(device)
        
        # Forward through encoder layers
        atom_embeddings = model.atom_embedding(graph)
        for layer in model.bond_graph_layers:
            atom_embeddings = layer(atom_embeddings, graph)
        
        # Global pooling
        global_embedding = atom_embeddings.mean(dim=0)
        
    return global_embedding.cpu().numpy()


def build_ood_detector_from_training(
    model,
    training_structures: List[Structure],
    save_path: str = "checkpoints/ood_detector.pkl",
    device: str = "cpu",
) -> OODDetector:
    """
    Build OOD detector from training data.
    
    Args:
        model: Trained CHGNet model
        training_structures: List of training structures
        save_path: Where to save the detector
        device: torch device
        
    Returns:
        Fitted OODDetector
    """
    print(f"Building OOD detector from {len(training_structures)} structures...")
    
    # Extract embeddings for all training structures
    embeddings = []
    for i, struct in enumerate(training_structures):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(training_structures)}...")
        try:
            emb = extract_chgnet_embedding(model, struct, device)
            embeddings.append(emb)
        except Exception as e:
            continue
    
    embeddings = np.array(embeddings)
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Fit detector
    detector = OODDetector()
    detector.fit(embeddings)
    
    # Save
    detector.save(save_path)
    print(f"Saved OOD detector to {save_path}")
    
    return detector
