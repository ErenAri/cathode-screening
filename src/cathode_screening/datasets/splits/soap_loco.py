"""
SOAP-LOCO (Smooth Overlap of Atomic Positions - Leave One Cluster Out) splitting.

Implements chemistry-aware data splits based on structural similarity using
SOAP descriptors. This prevents information leakage between train/test sets
by ensuring entire chemical families are held out together.

References:
    - Bartók et al., "On representing chemical environments" PRB 87, 184115 (2013)
    - Research doc: "SOAP-LOCO recommended as industrial standard"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

try:
    from dscribe.descriptors import SOAP
    from pymatgen.core import Structure
    DSCRIBE_AVAILABLE = True
except ImportError:
    DSCRIBE_AVAILABLE = False
    SOAP = None
    Structure = None


def check_dscribe_available() -> bool:
    """Check if dscribe is available for SOAP computation."""
    return DSCRIBE_AVAILABLE


def compute_soap_descriptors(
    structures: List["Structure"],
    r_cut: float = 6.0,
    n_max: int = 6,
    l_max: int = 4,
    species: Optional[List[str]] = None,
    average: str = "inner",
    sparse: bool = False,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Compute SOAP descriptors for a list of pymatgen Structures.
    
    SOAP encodes the local atomic environment as a smooth, rotation-invariant
    fingerprint. We average over all atoms to get a global structure descriptor.
    
    Args:
        structures: List of pymatgen Structure objects
        r_cut: Cutoff radius for local environment (Å)
        n_max: Number of radial basis functions
        l_max: Maximum degree of spherical harmonics
        species: List of element symbols (auto-detected if None)
        average: Averaging mode ("inner" = average over atoms)
        sparse: Whether to return sparse output
        batch_size: Number of structures to process at once (for memory efficiency)
    
    Returns:
        [N, D] array of SOAP descriptors (float32), where N = len(structures)
    
    Raises:
        ImportError: If dscribe is not installed
    """
    if not DSCRIBE_AVAILABLE:
        raise ImportError(
            "dscribe is required for SOAP descriptors. "
            "Install with: pip install dscribe"
        )
    
    # Auto-detect species if not provided
    if species is None:
        all_species = set()
        for struct in structures:
            all_species.update([str(s) for s in struct.species])
        species = sorted(list(all_species))
    
    # Initialize SOAP descriptor
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        sparse=sparse,
        periodic=True,
        dtype="float32",  # Use float32 to reduce memory by 2x
    )
    
    # Convert pymatgen to ASE for dscribe compatibility
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    
    # Process in batches to avoid OOM
    n_structures = len(structures)
    n_batches = (n_structures + batch_size - 1) // batch_size
    
    all_descriptors = []
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_structures)
        
        batch_structures = structures[start:end]
        batch_atoms = [adaptor.get_atoms(s) for s in batch_structures]
        
        # Compute SOAP for batch
        batch_descriptors = soap.create(batch_atoms)
        all_descriptors.append(batch_descriptors.astype(np.float32))
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"    Processed {end}/{n_structures} structures...")
    
    return np.vstack(all_descriptors)


def compute_soap_similarity_matrix(
    descriptors: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix from SOAP descriptors.
    
    Uses normalized dot product (cosine similarity) which corresponds
    to the SOAP kernel.
    
    Args:
        descriptors: [N, D] SOAP descriptors
    
    Returns:
        [N, N] similarity matrix (1 = identical, 0 = orthogonal)
    """
    # Normalize descriptors
    normed = normalize(descriptors)
    
    # Cosine similarity = dot product of normalized vectors
    similarity = normed @ normed.T
    
    return similarity


def soap_cluster(
    descriptors: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Cluster structures by SOAP similarity using K-Means.
    
    Args:
        descriptors: [N, D] SOAP descriptors
        n_clusters: Number of clusters
        seed: Random seed for reproducibility
    
    Returns:
        [N] cluster assignment for each structure
    """
    # Normalize for better K-Means behavior
    normed = normalize(descriptors)
    
    # Determine number of clusters (can't exceed unique structures)
    unique = np.unique(normed, axis=0).shape[0]
    k = min(n_clusters, unique)
    
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_ids = km.fit_predict(normed)
    
    return cluster_ids


def soap_loco_split(
    df: pd.DataFrame,
    structures: List["Structure"],
    n_clusters: int = 20,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    soap_params: Optional[Dict] = None,
) -> Dict[str, List[str]]:
    """
    Create SOAP-LOCO splits for train/val/test.
    
    Leave-One-Cluster-Out splitting ensures that entire chemical families
    (defined by structural similarity) are held out together. This simulates
    genuine discovery scenarios where the model must generalize to novel chemistries.
    
    Args:
        df: DataFrame with 'material_id' column
        structures: List of pymatgen Structures (same order as df)
        n_clusters: Number of SOAP clusters
        train_frac: Fraction for training
        val_frac: Fraction for validation
        seed: Random seed
        soap_params: Optional SOAP hyperparameters (r_cut, n_max, l_max)
    
    Returns:
        Dictionary with 'train', 'val', 'test' lists of material_ids
    """
    if not DSCRIBE_AVAILABLE:
        raise ImportError(
            "dscribe is required for SOAP-LOCO splitting. "
            "Install with: pip install dscribe"
        )
    
    soap_params = soap_params or {}
    
    print(f"Computing SOAP descriptors for {len(structures)} structures...")
    descriptors = compute_soap_descriptors(structures, **soap_params)
    
    print(f"Clustering into {n_clusters} groups by structural similarity...")
    cluster_ids = soap_cluster(descriptors, n_clusters=n_clusters, seed=seed)
    
    df = df.copy()
    df["soap_cluster"] = cluster_ids
    
    # Shuffle cluster order and assign to splits
    clusters = df["soap_cluster"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_ids, val_ids, test_ids = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}
    
    for cid in clusters:
        mids = df.loc[df["soap_cluster"] == cid, "material_id"].tolist()
        if counts["train"] < n_train:
            train_ids.extend(mids)
            counts["train"] += len(mids)
        elif counts["val"] < n_val:
            val_ids.extend(mids)
            counts["val"] += len(mids)
        else:
            test_ids.extend(mids)
            counts["test"] += len(mids)
    
    # Shuffle within splits
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    
    print(f"  Train: {len(train_ids)} ({100*len(train_ids)/n:.1f}%)")
    print(f"  Val:   {len(val_ids)} ({100*len(val_ids)/n:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({100*len(test_ids)/n:.1f}%)")
    
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def compute_split_novelty(
    train_descriptors: np.ndarray,
    test_descriptors: np.ndarray,
) -> Dict[str, float]:
    """
    Compute novelty metrics between train and test sets.
    
    Higher novelty = more challenging generalization task.
    
    Args:
        train_descriptors: [N_train, D] SOAP descriptors
        test_descriptors: [N_test, D] SOAP descriptors
    
    Returns:
        Dictionary with novelty statistics:
        - mean_min_distance: Average distance from test to nearest train
        - max_min_distance: Maximum distance from test to nearest train
        - fraction_ood: Fraction of test samples far from train
    """
    # Normalize
    train_normed = normalize(train_descriptors)
    test_normed = normalize(test_descriptors)
    
    # Compute similarity (cosine)
    similarity = test_normed @ train_normed.T  # [N_test, N_train]
    
    # Max similarity to any training sample
    max_sim = np.max(similarity, axis=1)  # [N_test]
    
    # Convert to distance (1 - similarity)
    min_dist = 1 - max_sim
    
    # OOD threshold: similarity < 0.9 → distance > 0.1
    ood_threshold = 0.1
    
    return {
        "mean_min_distance": float(np.mean(min_dist)),
        "max_min_distance": float(np.max(min_dist)),
        "median_min_distance": float(np.median(min_dist)),
        "fraction_ood": float(np.mean(min_dist > ood_threshold)),
    }
