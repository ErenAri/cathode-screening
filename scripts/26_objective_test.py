"""
Objective Testing Script for v1-Li-Cathode CHGNet Ensemble.

Downloads test materials from Materials Project, runs predictions,
and compares with ground truth E_hull values.

Usage:
    python scripts/26_objective_test.py
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Set environment
os.environ['CATHODE_MODEL_TYPE'] = 'chgnet'
os.environ['CATHODE_ARTIFACTS_DIR'] = 'data/artifacts'

from mp_api.client import MPRester
from cathode_screening.inference.li_cathode_screener import LiCathodeScreener, CathodeScreenResult


# Test materials with known E_hull (from Materials Project)
TEST_MATERIALS = [
    # Well-known stable Li-cathodes
    {"id": "mp-22526", "name": "LiCoO2", "expected_class": "KEEP"},
    {"id": "mp-19017", "name": "LiFePO4", "expected_class": "KEEP"},
    {"id": "mp-18767", "name": "LiMnO2", "expected_class": "KEEP"},
    # Meta-stable / borderline
    {"id": "mp-756364", "name": "LiMnNiO", "expected_class": "MAYBE"},
    # Less stable
    {"id": "mp-770295", "name": "Li2MnO3", "expected_class": "MAYBE"},
]


def fetch_structures_from_mp(material_ids: List[str]) -> Dict[str, Tuple]:
    """Fetch structures and ground truth E_hull from Materials Project."""
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("Warning: MP_API_KEY not set. Using cached data if available.")
        return {}
    
    materials = {}
    with MPRester(api_key) as mpr:
        for mid in material_ids:
            try:
                doc = mpr.materials.summary.get_data_by_id(mid)
                materials[mid] = {
                    "structure": doc.structure,
                    "formula": doc.formula_pretty,
                    "ehull_true": doc.energy_above_hull,
                }
                print(f"  Fetched {mid}: {doc.formula_pretty}, E_hull={doc.energy_above_hull:.4f} eV")
            except Exception as e:
                print(f"  Failed to fetch {mid}: {e}")
    
    return materials


def run_objective_test():
    """Run objective tests comparing predictions with ground truth."""
    
    print("=" * 70)
    print("OBJECTIVE TEST: v1-Li-Cathode CHGNet Ensemble")
    print("=" * 70)
    
    # Load screener
    print("\n[1/4] Loading CHGNet ensemble...")
    screener = LiCathodeScreener("data/artifacts/chgnet_ensemble")
    
    # Fetch test structures
    print("\n[2/4] Fetching test structures from Materials Project...")
    material_ids = [m["id"] for m in TEST_MATERIALS]
    materials = fetch_structures_from_mp(material_ids)
    
    if not materials:
        print("No materials fetched. Using cached test data...")
        # Fall back to cached test data
        data_path = Path("data/processed/chgnet_soap_loco")
        if data_path.exists():
            structures = pickle.load(open(data_path / "structures.pkl", "rb"))
            energies = np.load(data_path / "energies.npy")
            test_idx = np.load(data_path / "test_idx.npy")
            metadata = json.load(open(data_path / "metadata.json"))
            
            # Use first 5 test samples
            for i in range(min(5, len(test_idx))):
                idx = test_idx[i]
                mid = metadata[idx]["material_id"]
                materials[mid] = {
                    "structure": structures[idx],
                    "formula": structures[idx].composition.reduced_formula,
                    "ehull_true": energies[idx],
                }
            print(f"  Loaded {len(materials)} cached test samples")
    
    # Run predictions
    print("\n[3/4] Running predictions...")
    results = []
    
    for mid, data in materials.items():
        try:
            result = screener.predict_structure(data["structure"], mid)
            
            error = result.pred_ehull - data["ehull_true"]
            abs_error = abs(error)
            in_interval = result.ci_lower <= data["ehull_true"] <= result.ci_upper
            
            results.append({
                "material_id": mid,
                "formula": data["formula"],
                "ehull_true": data["ehull_true"],
                "ehull_pred": result.pred_ehull,
                "uncertainty": result.uncertainty,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "recommendation": result.recommendation,
                "error": error,
                "abs_error": abs_error,
                "in_interval": in_interval,
            })
            
            status = "✓" if in_interval else "✗"
            print(f"  {status} {mid} ({data['formula']}): "
                  f"true={data['ehull_true']:.4f}, pred={result.pred_ehull:.4f}±{result.uncertainty:.4f}, "
                  f"rec={result.recommendation}")
            
        except Exception as e:
            print(f"  ✗ {mid}: FAILED - {e}")
    
    # Summary statistics
    print("\n[4/4] Objective Review Summary")
    print("=" * 70)
    
    if results:
        df = pd.DataFrame(results)
        
        mae = df["abs_error"].mean()
        rmse = np.sqrt((df["error"] ** 2).mean())
        coverage = df["in_interval"].mean()
        
        print(f"\n📊 Error Metrics:")
        print(f"   MAE:  {mae:.4f} eV/atom")
        print(f"   RMSE: {rmse:.4f} eV/atom")
        
        print(f"\n📈 Calibration:")
        print(f"   95% CI Coverage: {coverage:.1%} (expect ~95%)")
        
        print(f"\n🎯 Decision Distribution:")
        for rec in ["KEEP", "MAYBE", "KILL"]:
            count = (df["recommendation"] == rec).sum()
            print(f"   {rec}: {count}")
        
        print("\n📋 Detailed Results:")
        print(df[["material_id", "formula", "ehull_true", "ehull_pred", "error", "recommendation", "in_interval"]].to_string(index=False))
        
        # Save results
        output_path = Path("data/artifacts/chgnet_ensemble/objective_test_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Pass/Fail summary
        print("\n" + "=" * 70)
        if mae < 0.05 and coverage > 0.8:
            print("✅ OBJECTIVE TEST PASSED")
            print(f"   MAE {mae:.4f} < 0.05 threshold")
            print(f"   Coverage {coverage:.1%} > 80% threshold")
        else:
            print("⚠️ OBJECTIVE TEST: REVIEW NEEDED")
            if mae >= 0.05:
                print(f"   MAE {mae:.4f} exceeds 0.05 threshold")
            if coverage <= 0.8:
                print(f"   Coverage {coverage:.1%} below 80% threshold")
    else:
        print("No results to analyze.")


if __name__ == "__main__":
    run_objective_test()
