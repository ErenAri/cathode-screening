"""
Phase 7B: Update Materials Project Li-cathode data (2024).

Downloads latest Li-cathode materials from Materials Project API
to capture new entries since the original training data was collected.

Requires: MP_API_KEY environment variable

Usage:
    export MP_API_KEY="your_api_key"
    python scripts/29_update_mp_data.py

Output:
    data/external/mp_2024/mp_li_cathodes_2024.parquet
    data/external/mp_2024/structures.pkl
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Check for MP API
try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False
    print("Warning: mp_api not installed. Run: pip install mp_api")

# Output directories
OUTPUT_DIR = Path("data/external/mp_2024")

# Transition metals for cathodes
TRANSITION_METALS = ["Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"]


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str:
    """Get MP API key from environment."""
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise ValueError(
            "MP_API_KEY not set. Get your key at https://materialsproject.org/api"
        )
    return api_key


def query_mp_li_cathodes(
    api_key: str,
    e_hull_max: float = 0.5,
) -> Tuple[pd.DataFrame, List]:
    """
    Query Materials Project for Li-containing oxide cathode materials.
    
    Returns:
        DataFrame with material properties
        List of pymatgen Structure objects
    """
    if not HAS_MP_API:
        raise RuntimeError("mp_api package required. Install with: pip install mp_api")
    
    all_materials = []
    all_structures = []
    
    with MPRester(api_key) as mpr:
        # Query for Li-O-TM materials
        for tm in tqdm(TRANSITION_METALS, desc="Querying MP for TM families"):
            try:
                # Search for materials containing Li, O, and the transition metal
                docs = mpr.materials.summary.search(
                    elements=["Li", "O", tm],
                    energy_above_hull=(0, e_hull_max),
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "composition",
                        "energy_above_hull",
                        "formation_energy_per_atom",
                        "band_gap",
                        "is_stable",
                        "nsites",
                        "volume",
                        "density",
                        "symmetry",
                        "structure",
                    ],
                )
                
                for doc in docs:
                    material_data = {
                        "material_id": str(doc.material_id),
                        "formula": doc.formula_pretty,
                        "e_hull": doc.energy_above_hull,
                        "formation_energy": doc.formation_energy_per_atom,
                        "band_gap": doc.band_gap,
                        "is_stable": doc.is_stable,
                        "nsites": doc.nsites,
                        "volume": doc.volume,
                        "density": doc.density,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                        "tm_element": tm,
                    }
                    
                    all_materials.append(material_data)
                    
                    if doc.structure:
                        all_structures.append({
                            "material_id": str(doc.material_id),
                            "structure": doc.structure,
                        })
                
                print(f"  {tm}: {len([m for m in all_materials if m['tm_element'] == tm])} materials")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  Error querying {tm}: {e}")
                continue
    
    # Remove duplicates by material_id
    seen_ids = set()
    unique_materials = []
    unique_structures = []
    
    for mat in all_materials:
        if mat["material_id"] not in seen_ids:
            seen_ids.add(mat["material_id"])
            unique_materials.append(mat)
    
    for struct in all_structures:
        if struct["material_id"] in seen_ids:
            unique_structures.append(struct)
    
    df = pd.DataFrame(unique_materials)
    print(f"\nTotal unique Li-TM-O materials from MP: {len(df)}")
    
    return df, unique_structures


def filter_new_materials(
    df_new: pd.DataFrame,
    existing_ids_path: Path = Path("data/processed/chgnet_soap_loco/metadata.json"),
) -> pd.DataFrame:
    """
    Filter to only materials not in existing training set.
    """
    if existing_ids_path.exists():
        with open(existing_ids_path) as f:
            existing_metadata = json.load(f)
        existing_ids = {m["material_id"] for m in existing_metadata}
        
        df_filtered = df_new[~df_new["material_id"].isin(existing_ids)]
        print(f"Filtered: {len(df_new)} -> {len(df_filtered)} (removed {len(df_new) - len(df_filtered)} existing)")
        return df_filtered
    
    return df_new


def save_results(df: pd.DataFrame, structures: List) -> None:
    """Save downloaded data."""
    # Save metadata
    parquet_path = OUTPUT_DIR / "mp_li_cathodes_2024.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved metadata to {parquet_path}")
    
    # Save structures
    structures_path = OUTPUT_DIR / "structures.pkl"
    with open(structures_path, "wb") as f:
        pickle.dump(structures, f)
    print(f"Saved {len(structures)} structures to {structures_path}")
    
    # Save summary
    summary = {
        "total_materials": len(df),
        "structures_downloaded": len(structures),
        "e_hull_range": [float(df["e_hull"].min()), float(df["e_hull"].max())],
        "download_date": time.strftime("%Y-%m-%d"),
    }
    
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("Phase 7B: Materials Project 2024 Update")
    print("=" * 60)
    
    setup_directories()
    
    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Step 1: Query MP for Li-TM-O materials
    print("\n[1/3] Querying Materials Project...")
    df, structures = query_mp_li_cathodes(api_key)
    
    if len(df) == 0:
        print("No materials found. Check API key and connectivity.")
        return
    
    # Step 2: Filter out existing materials
    print("\n[2/3] Filtering new materials...")
    df_new = filter_new_materials(df)
    
    # Step 3: Save results
    print("\n[3/3] Saving results...")
    save_results(df_new, structures)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Total materials: {len(df_new)}")
    print(f"  Structures: {len(structures)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
