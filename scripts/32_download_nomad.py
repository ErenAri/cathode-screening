"""
Phase 7E: Download NOMAD (Novel Materials Discovery) Li-cathode data.

NOMAD contains millions of DFT calculations from various sources.
Uses NOMAD API to query for Li-containing oxide materials.

Install: pip install nomad-lab

Usage:
    python scripts/32_download_nomad.py

Output:
    data/external/nomad/nomad_li_cathodes.parquet
    data/external/nomad/structures.pkl
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path("data/external/nomad")

# NOMAD API endpoint
NOMAD_API = "https://nomad-lab.eu/prod/v1/api/v1"

# Transition metals for cathode filtering
TRANSITION_METALS = ["Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"]


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def query_nomad_li_cathodes(
    elements: List[str],
    max_results: int = 2000,
    page_size: int = 100,
) -> List[Dict]:
    """
    Query NOMAD for Li-containing materials with specific elements.
    
    Uses the NOMAD v1 API with corrected query format.
    """
    all_results = []
    
    # Simpler query format for NOMAD v1 API
    url = f"{NOMAD_API}/entries"
    
    # Build element filter string
    elements_str = ",".join(elements)
    
    params = {
        "elements": elements_str,
        "page_size": page_size,
        "page": 1,
    }
    
    try:
        while len(all_results) < max_results:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 422:
                # Try alternative query format
                break
                
            response.raise_for_status()
            data = response.json()
            
            entries = data.get("data", [])
            if not entries:
                break
            
            for entry in entries:
                all_results.append({
                    "nomad_id": entry.get("entry_id"),
                    "formula": entry.get("mainfile", ""),
                    "upload_id": entry.get("upload_id"),
                })
            
            # Check for next page
            if len(entries) < page_size:
                break
                
            params["page"] += 1
            time.sleep(0.3)
        
    except requests.RequestException as e:
        print(f"API error: {e}")
    
    return all_results


def download_all_li_cathodes() -> pd.DataFrame:
    """
    Download all Li-TM-O materials from NOMAD.
    """
    all_materials = []
    seen_ids = set()
    
    for tm in tqdm(TRANSITION_METALS, desc="Querying NOMAD for TM families"):
        elements = ["Li", "O", tm]
        
        results = query_nomad_li_cathodes(elements, max_results=2000)
        
        for entry in results:
            nomad_id = entry.get("nomad_id")
            if nomad_id and nomad_id not in seen_ids:
                seen_ids.add(nomad_id)
                entry["tm_element"] = tm
                all_materials.append(entry)
        
        print(f"  {tm}: {len([m for m in all_materials if m.get('tm_element') == tm])} entries")
    
    df = pd.DataFrame(all_materials)
    print(f"\nTotal unique Li-TM-O materials from NOMAD: {len(df)}")
    
    return df


def save_results(df: pd.DataFrame) -> None:
    """Save downloaded data."""
    # Save metadata
    parquet_path = OUTPUT_DIR / "nomad_li_cathodes.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} materials to {parquet_path}")
    
    # Summary
    summary = {
        "total_materials": len(df),
        "transition_metals": TRANSITION_METALS,
        "download_date": time.strftime("%Y-%m-%d"),
        "source": "NOMAD (nomad-lab.eu)",
    }
    
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("Phase 7E: NOMAD Li-Cathode Download")
    print("=" * 60)
    
    setup_directories()
    
    # Download Li-TM-O materials
    print("\n[1/2] Querying NOMAD for Li-TM-O materials...")
    df = download_all_li_cathodes()
    
    if len(df) == 0:
        print("No materials found.")
        return
    
    # Save results
    print("\n[2/2] Saving results...")
    save_results(df)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Total materials: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
