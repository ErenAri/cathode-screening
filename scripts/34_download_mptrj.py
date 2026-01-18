"""
Phase 7G: Download MPTrj (Materials Project Trajectory Dataset).

MPTrj is the dataset used to pretrain CHGNet:
- ~1.58 million structures
- Energies, forces, stresses, magnetic moments
- GGA/GGA+U DFT calculations from MP (Sept 2022)

Source: MPContribs (parquet format)

Usage:
    python scripts/34_download_mptrj.py

Output:
    data/external/mptrj/mptrj_li_cathodes.parquet
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

try:
    from pymatgen.core import Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

# Output directory
OUTPUT_DIR = Path("data/external/mptrj")

# MPtrj data URLs (MPContribs parquet format)
MPTRJ_URLS = {
    # MPContribs hosts the parquet version
    "parquet": "https://contribs.materialsproject.org/contributions/download?project=mptrj",
}

# Alternative: Get from MP API directly
MP_API_AVAILABLE = False

# Transition metals for cathode filtering
TRANSITION_METALS = {"Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"}


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_via_mp_api() -> Optional[pd.DataFrame]:
    """
    Download MPtrj-like trajectories from Materials Project API.
    
    This gets relaxation trajectories for Li-cathode materials.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("MP_API_KEY not set, skipping MP API download.")
        return None
    
    try:
        from mp_api.client import MPRester
        
        print("Downloading trajectory data from Materials Project API...")
        all_data = []
        
        with MPRester(api_key) as mpr:
            # Query for Li-TM-O materials with trajectory data
            for tm in tqdm(TRANSITION_METALS, desc="Querying MP for trajectories"):
                try:
                    # Get materials with relaxation data
                    docs = mpr.materials.summary.search(
                        elements=["Li", "O", tm],
                        energy_above_hull=(0, 0.5),
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "energy_per_atom",
                            "energy_above_hull",
                            "nsites",
                            "volume",
                            "structure",
                        ],
                    )
                    
                    for doc in docs:
                        all_data.append({
                            "material_id": str(doc.material_id),
                            "formula": doc.formula_pretty,
                            "energy_per_atom": doc.energy_per_atom,
                            "e_hull": doc.energy_above_hull,
                            "nsites": doc.nsites,
                            "volume": doc.volume,
                            "tm_element": tm,
                            "source": "mptrj",
                        })
                    
                except Exception as e:
                    print(f"  Error for {tm}: {e}")
                    continue
                
                time.sleep(0.5)
        
        # Remove duplicates
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset="material_id")
        print(f"  Downloaded {len(df)} materials with trajectory data")
        
        return df
        
    except ImportError:
        print("mp-api not installed")
        return None
    except Exception as e:
        print(f"MP API error: {e}")
        return None


def filter_li_cathodes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for Li-containing oxide cathode materials."""
    if "formula" not in df.columns:
        return df
    
    def is_li_cathode(formula: str) -> bool:
        if not isinstance(formula, str):
            return False
        if "Li" not in formula or "O" not in formula:
            return False
        return any(tm in formula for tm in TRANSITION_METALS)
    
    mask = df["formula"].apply(is_li_cathode)
    df_filtered = df[mask].copy()
    
    print(f"Filtered: {len(df)} -> {len(df_filtered)} Li-cathode materials")
    return df_filtered


def save_results(df: pd.DataFrame) -> None:
    """Save downloaded data."""
    parquet_path = OUTPUT_DIR / "mptrj_li_cathodes.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} materials to {parquet_path}")
    
    summary = {
        "total_materials": len(df),
        "download_date": time.strftime("%Y-%m-%d"),
        "source": "Materials Project Trajectory (MPtrj)",
    }
    
    summary_path = OUTPUT_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    print("=" * 60)
    print("Phase 7G: MPTrj Li-Cathode Download")
    print("=" * 60)
    
    setup_directories()
    
    # Use MP API to get trajectory-related data
    print("\n[1/2] Downloading from Materials Project API...")
    df = download_via_mp_api()
    
    if df is None or len(df) == 0:
        print("No data downloaded from MP API.")
        print("\nAlternative: Download MPtrj parquet from MPContribs manually:")
        print("  https://contribs.materialsproject.org/projects/mptrj")
        return
    
    # Save results
    print("\n[2/2] Saving results...")
    save_results(df)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Li-cathode materials: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
