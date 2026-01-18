"""
Phase 7F: Download JARVIS-DFT Li-cathode data.

JARVIS-DFT contains ~40,000 3D materials with DFT-calculated properties.
Uses direct download from Figshare/NIST.

Usage:
    python scripts/33_download_jarvis.py

Output:
    data/external/jarvis/jarvis_li_cathodes.parquet
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path("data/external/jarvis")

# JARVIS dataset URL (from NIST/Figshare)
JARVIS_URL = "https://figshare.com/ndownloader/files/40357663"  # jarvis_dft_3d.json

# Transition metals for cathode filtering
TRANSITION_METALS = {"Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"}


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_jarvis_data() -> Optional[List[Dict]]:
    """Download JARVIS-DFT 3D materials dataset using jarvis-tools package."""
    json_path = OUTPUT_DIR / "jarvis_dft_3d.json"
    
    # First try jarvis-tools package (most reliable)
    try:
        from jarvis.db.figshare import data as jarvis_data
        print("Loading JARVIS via jarvis-tools package...")
        data = jarvis_data("dft_3d")
        print(f"  Total JARVIS materials: {len(data)}")
        return data
    except ImportError:
        print("jarvis-tools not installed. Install with: pip install jarvis-tools")
        print("Trying direct download as fallback...")
    except Exception as e:
        print(f"jarvis-tools failed: {e}")
        print("Trying direct download as fallback...")
    
    # Fallback: direct download with browser headers
    if not json_path.exists():
        print("Downloading JARVIS-DFT dataset...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        try:
            response = requests.get(JARVIS_URL, headers=headers, timeout=300, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(json_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        except requests.RequestException as e:
            print(f"Download error: {e}")
            print("\nTo fix: pip install jarvis-tools")
            return None
    
    print("Loading JARVIS data from file...")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print(f"  Total JARVIS materials: {len(data)}")
    return data


def filter_li_cathodes(data: List[Dict]) -> pd.DataFrame:
    """
    Filter JARVIS dataset for Li-containing oxide cathode materials.
    """
    print("Filtering for Li-cathode materials...")
    
    li_cathodes = []
    
    for entry in tqdm(data, desc="Filtering"):
        formula = entry.get("formula", "")
        elements = entry.get("elements", [])
        
        # Check for Li and O
        if "Li" not in elements or "O" not in elements:
            continue
        
        # Check for transition metal
        has_tm = any(tm in elements for tm in TRANSITION_METALS)
        if not has_tm:
            continue
        
        # Get stability data
        e_hull = entry.get("ehull")  # JARVIS provides E_hull
        formation_energy = entry.get("formation_energy_peratom")
        
        # Filter for reasonable stability
        if e_hull is not None and e_hull > 0.5:
            continue
        
        li_cathodes.append({
            "jarvis_id": entry.get("jid"),
            "formula": formula,
            "e_hull": e_hull,
            "formation_energy": formation_energy,
            "band_gap": entry.get("optb88vdw_bandgap"),
            "spacegroup": entry.get("spg_symbol"),
            "elements": elements,
            "natoms": entry.get("nat"),
        })
    
    df = pd.DataFrame(li_cathodes)
    print(f"  Filtered: {len(data)} -> {len(df)} Li-cathode materials")
    
    return df


def save_results(df: pd.DataFrame) -> None:
    """Save filtered JARVIS data."""
    # Save metadata
    parquet_path = OUTPUT_DIR / "jarvis_li_cathodes.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} materials to {parquet_path}")
    
    # Summary
    summary = {
        "total_materials": len(df),
        "e_hull_range": [
            float(df["e_hull"].min()) if df["e_hull"].notna().any() else None,
            float(df["e_hull"].max()) if df["e_hull"].notna().any() else None,
        ],
        "download_date": time.strftime("%Y-%m-%d"),
        "source": "JARVIS-DFT (NIST)",
    }
    
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("Phase 7F: JARVIS-DFT Li-Cathode Download")
    print("=" * 60)
    
    setup_directories()
    
    # Step 1: Download JARVIS data
    print("\n[1/3] Downloading JARVIS-DFT...")
    data = download_jarvis_data()
    
    if data is None:
        print("Failed to download JARVIS data.")
        return
    
    # Step 2: Filter for Li-cathodes
    print("\n[2/3] Filtering Li-cathode materials...")
    df = filter_li_cathodes(data)
    
    if len(df) == 0:
        print("No Li-cathode materials found.")
        return
    
    # Step 3: Save results
    print("\n[3/3] Saving results...")
    save_results(df)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Li-cathode materials: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
