"""
Prepare cathode data for CHGNet fine-tuning.

Downloads crystal structures from Materials Project and prepares them
for CHGNet training with E_hull as the target property.

Usage:
    python scripts/20_prepare_chgnet_data.py
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set MP API key from environment
MP_API_KEY = os.getenv("MP_API_KEY")
if not MP_API_KEY:
    raise ValueError("MP_API_KEY environment variable not set")


@dataclass
class StructureData:
    """Container for structure and properties."""
    material_id: str
    structure: object  # pymatgen Structure
    energy_above_hull: float
    formation_energy_per_atom: float
    formula: str


def load_material_ids(
    parquet_path: str = "data/processed/mp/processed_mp_cathodes_v1.parquet",
) -> pd.DataFrame:
    """Load processed cathode data with material IDs and targets."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} materials from {parquet_path}")
    return df


def fetch_structures_batch(
    material_ids: List[str],
    batch_size: int = 100,
) -> Dict[str, object]:
    """Fetch structures from Materials Project API in batches."""
    from mp_api.client import MPRester
    
    structures = {}
    
    with MPRester(MP_API_KEY) as mpr:
        for i in tqdm(range(0, len(material_ids), batch_size), desc="Fetching structures"):
            batch_ids = material_ids[i:i + batch_size]
            
            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=["material_id", "structure", "energy_above_hull", 
                            "formation_energy_per_atom", "formula_pretty"]
                )
                
                for doc in docs:
                    structures[doc.material_id] = {
                        "structure": doc.structure,
                        "energy_above_hull": doc.energy_above_hull,
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "formula": doc.formula_pretty,
                    }
            except Exception as e:
                print(f"Error fetching batch starting at {i}: {e}")
                continue
    
    return structures


def prepare_chgnet_dataset(
    df: pd.DataFrame,
    structures: Dict[str, object],
    output_dir: str = "data/processed/chgnet",
) -> Tuple[List, np.ndarray]:
    """Prepare dataset for CHGNet training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    structure_list = []
    energy_list = []
    metadata = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
        mid = row["material_id"]
        
        if mid not in structures:
            continue
        
        struct_data = structures[mid]
        structure_list.append(struct_data["structure"])
        energy_list.append(row["energy_above_hull"])  # Use our computed E_hull
        metadata.append({
            "material_id": mid,
            "formula": struct_data["formula"],
            "energy_above_hull": row["energy_above_hull"],
            "formation_energy_per_atom": row.get("formation_energy_per_atom", 0),
        })
    
    print(f"Prepared {len(structure_list)} structures")
    
    # Save structures and metadata
    with open(output_path / "structures.pkl", "wb") as f:
        pickle.dump(structure_list, f)
    
    np.save(output_path / "energies.npy", np.array(energy_list))
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_path}")
    
    return structure_list, np.array(energy_list)


def create_train_val_test_splits(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random train/val/test splits."""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def load_soap_loco_splits(
    split_file: str = "data/splits/mp/splits_mp_cathodes_v1_soap-loco.json",
) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """Load existing SOAP-LOCO splits if available."""
    split_path = Path(split_file)
    
    if not split_path.exists():
        print(f"SOAP-LOCO splits not found at {split_path}, will create random splits")
        return None
    
    with open(split_path) as f:
        splits = json.load(f)
    
    train_ids = splits.get("train", [])
    val_ids = splits.get("val", [])
    test_ids = splits.get("test", [])
    
    print(f"Loaded SOAP-LOCO splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return train_ids, val_ids, test_ids


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare CHGNet training data")
    parser.add_argument("--input", default="data/processed/mp/processed_mp_cathodes_v1.parquet")
    parser.add_argument("--output", default="data/processed/chgnet")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cache", default="data/cache/mp_structures.pkl",
                        help="Cache file for fetched structures")
    args = parser.parse_args()
    
    # Load material IDs
    df = load_material_ids(args.input)
    material_ids = df["material_id"].tolist()
    
    # Check for cached structures
    cache_path = Path(args.cache)
    if cache_path.exists():
        print(f"Loading cached structures from {cache_path}")
        with open(cache_path, "rb") as f:
            structures = pickle.load(f)
        print(f"Loaded {len(structures)} cached structures")
    else:
        # Fetch from Materials Project
        print("Fetching structures from Materials Project...")
        structures = fetch_structures_batch(material_ids, args.batch_size)
        
        # Cache for future use
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(structures, f)
        print(f"Cached {len(structures)} structures to {cache_path}")
    
    # Prepare dataset
    structure_list, energies = prepare_chgnet_dataset(df, structures, args.output)
    
    # Create or load splits
    soap_splits = load_soap_loco_splits()
    
    output_path = Path(args.output)
    
    if soap_splits:
        # Use SOAP-LOCO splits
        train_ids, val_ids, test_ids = soap_splits
        
        # Create index mapping
        id_to_idx = {mid: i for i, mid in enumerate([m["material_id"] for m in json.load(open(output_path / "metadata.json"))])}
        
        train_idx = np.array([id_to_idx[mid] for mid in train_ids if mid in id_to_idx])
        val_idx = np.array([id_to_idx[mid] for mid in val_ids if mid in id_to_idx])
        test_idx = np.array([id_to_idx[mid] for mid in test_ids if mid in id_to_idx])
    else:
        # Create random splits
        train_idx, val_idx, test_idx = create_train_val_test_splits(len(structure_list))
    
    # Save splits
    np.save(output_path / "train_idx.npy", train_idx)
    np.save(output_path / "val_idx.npy", val_idx)
    np.save(output_path / "test_idx.npy", test_idx)
    
    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    print(f"\nOutput directory: {output_path}")


if __name__ == "__main__":
    main()
