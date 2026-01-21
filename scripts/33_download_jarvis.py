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
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm import tqdm
from pymatgen.core import Structure

# Output directory
OUTPUT_DIR = Path("data/external/jarvis")

# JARVIS dataset source (Figshare article)
JARVIS_FIGSHARE_ARTICLE = "6815699"

# Transition metals for cathode filtering
TRANSITION_METALS = {"Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"}


def setup_directories() -> None:
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _select_figshare_file(files: List[Dict]) -> Optional[Dict]:
    """Pick the most suitable JARVIS DFT 3D file from figshare listing."""
    jdft_candidates = []
    for f in files:
        name = f.get("name", "").lower()
        if "jdft_3d" in name:
            jdft_candidates.append(f)
    candidates = jdft_candidates or files

    def date_key(entry: Dict) -> tuple:
        name = entry.get("name", "")
        m = re.search(r"jdft_3d-(\d+)-(\d+)-(\d{4})", name)
        if m:
            month, day, year = m.groups()
            return (int(year), int(month), int(day))
        return (0, 0, 0)

    # Prefer latest jdft_3d by date; fallback to largest file size.
    if jdft_candidates:
        return sorted(jdft_candidates, key=date_key, reverse=True)[0]
    return sorted(candidates, key=lambda x: x.get("size", 0), reverse=True)[0] if candidates else None


def _download_file(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    response = requests.get(url, headers=headers, timeout=300, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest_path.exists() and dest_path.stat().st_size > 0


def download_jarvis_data() -> Optional[List[Dict]]:
    """Download JARVIS-DFT 3D materials dataset using jarvis-tools package."""
    json_path = OUTPUT_DIR / "jarvis_dft_3d.json"
    existing_jsons = list(OUTPUT_DIR.glob("jdft_3d-*.json"))
    if existing_jsons:
        json_path = max(existing_jsons, key=lambda p: p.stat().st_mtime)
    
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
    
    # Remove empty/corrupt placeholder if present
    if json_path.exists() and json_path.stat().st_size == 0:
        json_path.unlink()

    # Fallback: download via Figshare API to avoid ndownloader 202 responses
    if not json_path.exists():
        print("Downloading JARVIS-DFT dataset via Figshare API...")
        api_url = f"https://api.figshare.com/v2/articles/{JARVIS_FIGSHARE_ARTICLE}"
        try:
            resp = requests.get(api_url, timeout=60)
            resp.raise_for_status()
            article = resp.json()
            file_entry = _select_figshare_file(article.get("files", []))
            if not file_entry:
                print("No downloadable files found in Figshare article.")
                return None
            download_url = file_entry.get("download_url")
            file_name = file_entry.get("name", "jarvis_dft_3d.json")
            download_path = OUTPUT_DIR / file_name
            if not _download_file(download_url, download_path):
                print("Downloaded file is empty.")
                return None

            # If zip, extract JSON
            if ZipFile is not None and ZipFile and str(download_path).lower().endswith(".zip"):
                with ZipFile(download_path, "r") as zf:
                    json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
                    if not json_names:
                        print("Zip file contains no JSON payloads.")
                        return None
                    # Extract the largest JSON file
                    json_name = max(json_names, key=lambda n: zf.getinfo(n).file_size)
                    zf.extract(json_name, OUTPUT_DIR)
                    json_path = OUTPUT_DIR / json_name
            else:
                json_path = download_path

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

        if not elements:
            atoms = entry.get("atoms")
            if isinstance(atoms, dict):
                elements = atoms.get("elements", []) or []
        if not elements:
            # Some JARVIS files store structure dicts under final_str
            struct_dict = entry.get("final_str")
            if isinstance(struct_dict, dict):
                try:
                    struct = Structure.from_dict(struct_dict)
                    elements = [el.symbol for el in struct.composition.elements]
                    formula = struct.composition.reduced_formula
                except Exception:
                    elements = []

        # Check for Li and O
        if "Li" not in elements or "O" not in elements:
            continue

        # Check for transition metal
        has_tm = any(tm in elements for tm in TRANSITION_METALS)
        if not has_tm:
            continue

        # Get stability data (try common keys)
        e_hull = entry.get("ehull")
        if e_hull is None:
            for key in ("e_above_hull", "energy_above_hull", "stability"):
                if key in entry:
                    e_hull = entry.get(key)
                    break

        formation_energy = entry.get("formation_energy_peratom") or entry.get("form_enp")
        
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
