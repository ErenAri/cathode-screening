"""
Regenerate database predictions using CHGNet v1-Li-Cathode ensemble.

This script runs predictions on all Li-cathode materials and saves
results in the format expected by the API database endpoint.

Usage:
    python scripts/27_regenerate_database.py
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from cathode_screening.inference.li_cathode_screener import LiCathodeScreener


def load_all_structures():
    """Load all Li-cathode structures from the dataset."""
    data_path = Path("data/processed/chgnet_soap_loco")
    
    structures = pickle.load(open(data_path / "structures.pkl", "rb"))
    energies = np.load(data_path / "energies.npy")
    metadata = json.load(open(data_path / "metadata.json"))
    
    print(f"Loaded {len(structures)} structures")
    return structures, energies, metadata


def regenerate_predictions():
    """Regenerate all predictions using CHGNet."""
    
    print("=" * 70)
    print("REGENERATING DATABASE: CHGNet v1-Li-Cathode")
    print("=" * 70)
    
    # Load screener
    print("\n[1/4] Loading CHGNet ensemble...")
    screener = LiCathodeScreener("data/artifacts/chgnet_ensemble")
    
    # Load structures
    print("\n[2/4] Loading all structures...")
    structures, true_energies, metadata = load_all_structures()
    
    # Run predictions in batches
    print("\n[3/4] Running predictions...")
    results = []
    batch_size = 50
    
    for i in tqdm(range(0, len(structures), batch_size), desc="Predicting"):
        batch_end = min(i + batch_size, len(structures))
        batch_structures = structures[i:batch_end]
        batch_metadata = metadata[i:batch_end]
        batch_true = true_energies[i:batch_end]
        
        for j, (struct, meta, true_ehull) in enumerate(zip(batch_structures, batch_metadata, batch_true)):
            try:
                result = screener.predict_structure(struct, meta.get("material_id", f"idx_{i+j}"))
                
                results.append({
                    "material_id": result.material_id,
                    "formula": result.formula,
                    "ehull_true": float(true_ehull),
                    "ehull_pred": result.pred_ehull,
                    "uncertainty": result.uncertainty,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "recommendation": result.recommendation,
                    "confidence": result.confidence,
                })
            except Exception as e:
                print(f"  Warning: Failed on {meta.get('material_id', i+j)}: {e}")
    
    # Save results
    print("\n[4/4] Saving predictions...")
    df = pd.DataFrame(results)
    
    # Save main database file
    output_path = Path("data/predictions/chgnet_database.parquet")
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} predictions to {output_path}")
    
    # Also save as CSV for debugging
    csv_path = Path("data/predictions/chgnet_database.csv")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total predictions: {len(df)}")
    print(f"MAE: {np.abs(df['ehull_true'] - df['ehull_pred']).mean():.4f} eV/atom")
    print(f"\nRecommendation distribution:")
    print(df['recommendation'].value_counts())
    
    return df


if __name__ == "__main__":
    regenerate_predictions()
