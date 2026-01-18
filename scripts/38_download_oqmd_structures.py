"""
Download crystal structures from OQMD (Open Quantum Materials Database).
Uses the qmpy REST API to fetch structures for Li-cathode compositions.
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# OQMD REST API endpoint
OQMD_API = "http://oqmd.org/oqmdapi/formationenergy"

def get_li_cathode_compositions() -> List[str]:
    """Get Li-cathode compositions from our existing data."""
    import pandas as pd
    
    compositions = set()
    
    oqmd_path = Path("data/external/oqmd/oqmd_li_cathodes.parquet")
    if oqmd_path.exists():
        df = pd.read_parquet(oqmd_path)
        if "composition" in df.columns:
            compositions.update(df["composition"].dropna().tolist())
        if "name" in df.columns:
            compositions.update(df["name"].dropna().tolist())
    
    print(f"Found {len(compositions)} unique compositions")
    return list(compositions)


def download_oqmd_structures():
    """Download structures from OQMD API."""
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compositions = get_li_cathode_compositions()
    
    training_data = []
    
    print("Downloading structures from OQMD...")
    for comp in tqdm(compositions[:500]):  # Limit to 500 for speed
        try:
            # Query OQMD API
            params = {
                "composition": comp,
                "fields": "entry_id,name,spacegroup,volume,natoms,delta_e,stability",
            }
            
            response = requests.get(OQMD_API, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for entry in data.get("data", []):
                    # OQMD API doesn't return full structure by default
                    # We need to query individual entries for structure
                    entry_id = entry.get("entry_id")
                    if entry_id:
                        training_data.append({
                            "source": "oqmd",
                            "entry_id": entry_id,
                            "name": entry.get("name"),
                            "spacegroup": entry.get("spacegroup"),
                            "volume": entry.get("volume"),
                            "natoms": entry.get("natoms"),
                            "formation_energy": entry.get("delta_e"),
                            "stability": entry.get("stability"),
                        })
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error for {comp}: {e}")
            continue
    
    print(f"Downloaded {len(training_data)} entries from OQMD")
    
    # Note: OQMD REST API is limited. For full structures, 
    # consider using the downloadable database dump instead.
    
    output_path = output_dir / "oqmd_training_data.json"
    with open(output_path, "w") as f:
        json.dump(training_data, f)
    print(f"Saved to {output_path}")
    
    print("\nNOTE: OQMD REST API has limitations. For full training data,")
    print("consider downloading the OQMD dump from: http://oqmd.org/download/")
    
    return training_data


if __name__ == "__main__":
    download_oqmd_structures()
