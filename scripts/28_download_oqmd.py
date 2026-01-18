"""
Phase 7A: Download Li-cathode materials from OQMD.

Uses the OQMD REST API via qmpy_rester to download:
- Li-containing oxide materials
- Stability data (E_hull / stability)
- Crystal structures

Target: ~15,000 additional Li-cathode materials to expand training set.

Usage:
    pip install qmpy_rester requests
    python scripts/28_download_oqmd.py

Output:
    data/external/oqmd/oqmd_li_cathodes.parquet
    data/external/oqmd/structures/
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# OQMD API endpoints
OQMD_API_BASE = "http://oqmd.org/oqmdapi/formationenergy"
OQMD_STRUCTURE_API = "http://oqmd.org/oqmdapi/entry"

# Transition metals commonly found in cathodes
TRANSITION_METALS = ["Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"]

# Output directories
OUTPUT_DIR = Path("data/external/oqmd")
STRUCTURES_DIR = OUTPUT_DIR / "structures"


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)


def query_oqmd_li_oxides(
    limit: int = 1000,
    offset: int = 0,
    stability_max: float = 0.5,  # eV/atom
) -> Dict:
    """
    Query OQMD for Li-containing oxide materials.
    
    Uses the formation energy endpoint with filters.
    """
    params = {
        "filter": f"element_set=Li AND element_set=O AND stability<{stability_max}",
        "fields": "name,entry_id,composition,spacegroup,volume,natoms,stability,delta_e,band_gap",
        "limit": limit,
        "offset": offset,
        "format": "json",
    }
    
    try:
        response = requests.get(OQMD_API_BASE, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API error: {e}")
        return {"data": [], "meta": {"more_data_available": False}}


def query_oqmd_by_composition(
    elements: List[str],
    limit: int = 500,
    offset: int = 0,
) -> Dict:
    """
    Query OQMD for materials containing specific elements.
    
    Args:
        elements: List of element symbols (e.g., ["Li", "Co", "O"])
        limit: Number of results per page
        offset: Pagination offset
    
    Returns:
        JSON response with material data
    """
    element_filter = " AND ".join([f"element_set={el}" for el in elements])
    
    params = {
        "filter": f"{element_filter} AND stability<0.5",
        "fields": "name,entry_id,composition,spacegroup,volume,natoms,stability,delta_e,band_gap,prototype",
        "limit": limit,
        "offset": offset,
        "format": "json",
    }
    
    try:
        response = requests.get(OQMD_API_BASE, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API error for {elements}: {e}")
        return {"data": [], "meta": {"more_data_available": False}}


def download_structure(entry_id: int) -> Optional[Dict]:
    """
    Download structure data for a specific entry.
    
    Returns structure in a format compatible with pymatgen.
    """
    url = f"{OQMD_STRUCTURE_API}/{entry_id}"
    
    try:
        response = requests.get(url, params={"format": "json"}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def download_all_li_cathodes() -> pd.DataFrame:
    """
    Download all Li-containing oxide cathode materials from OQMD.
    
    Iterates through combinations of Li + O + transition metals.
    """
    all_materials = []
    seen_ids = set()
    
    # Query Li-O-TM for each transition metal
    for tm in tqdm(TRANSITION_METALS, desc="Querying TM families"):
        elements = ["Li", "O", tm]
        offset = 0
        
        while True:
            result = query_oqmd_by_composition(elements, limit=500, offset=offset)
            data = result.get("data", [])
            
            if not data:
                break
            
            for entry in data:
                entry_id = entry.get("entry_id")
                if entry_id and entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    all_materials.append({
                        "oqmd_id": entry_id,
                        "name": entry.get("name"),
                        "composition": entry.get("composition"),
                        "spacegroup": entry.get("spacegroup"),
                        "volume": entry.get("volume"),
                        "natoms": entry.get("natoms"),
                        "stability_oqmd": entry.get("stability"),  # E_hull equivalent
                        "delta_e": entry.get("delta_e"),  # Formation energy
                        "band_gap": entry.get("band_gap"),
                        "prototype": entry.get("prototype"),
                        "tm_element": tm,
                    })
            
            # Check for more pages
            more_data = result.get("meta", {}).get("more_data_available", False)
            if not more_data:
                break
            
            offset += 500
            time.sleep(0.5)  # Rate limiting
        
        print(f"  {tm}: {len([m for m in all_materials if m['tm_element'] == tm])} materials")
    
    df = pd.DataFrame(all_materials)
    print(f"\nTotal unique Li-TM-O materials: {len(df)}")
    
    return df


def download_structures_for_subset(
    df: pd.DataFrame,
    max_structures: int = 5000,
) -> List[Dict]:
    """
    Download crystal structures for a subset of materials.
    
    Prioritizes low-stability (more stable) materials.
    """
    # Sort by stability and take top materials
    df_sorted = df.sort_values("stability_oqmd").head(max_structures)
    
    structures = []
    
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="Downloading structures"):
        entry_id = row["oqmd_id"]
        structure_data = download_structure(entry_id)
        
        if structure_data:
            structures.append({
                "oqmd_id": entry_id,
                "structure": structure_data,
            })
        
        time.sleep(0.2)  # Rate limiting
    
    return structures


def save_results(df: pd.DataFrame, structures: List[Dict]) -> None:
    """Save downloaded data."""
    # Save metadata
    parquet_path = OUTPUT_DIR / "oqmd_li_cathodes.parquet"
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
        "transition_metals": TRANSITION_METALS,
        "stability_range": [df["stability_oqmd"].min(), df["stability_oqmd"].max()],
        "download_date": time.strftime("%Y-%m-%d"),
    }
    
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("Phase 7A: OQMD Li-Cathode Download")
    print("=" * 60)
    
    setup_directories()
    
    # Step 1: Download metadata for all Li-TM-O materials
    print("\n[1/3] Querying OQMD for Li-TM-O materials...")
    df = download_all_li_cathodes()
    
    if len(df) == 0:
        print("No materials found. Check API connectivity.")
        return
    
    # Step 2: Download structures for most stable materials
    print("\n[2/3] Downloading structures for top materials...")
    structures = download_structures_for_subset(df, max_structures=5000)
    
    # Step 3: Save results
    print("\n[3/3] Saving results...")
    save_results(df, structures)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Total materials: {len(df)}")
    print(f"  Structures: {len(structures)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
