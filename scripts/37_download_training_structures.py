"""
Download full crystal structures for Li-cathode materials from Materials Project.
This creates training data with structures, energies, and forces for CHGNet fine-tuning.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

def download_mp_structures():
    """Download structures from Materials Project API."""
    from mp_api.client import MPRester
    
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise ValueError("MP_API_KEY not found in environment")
    
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect unique material IDs from our parquet files
    material_ids = set()
    
    # From mptrj Li-cathodes
    mptrj_path = Path("data/external/mptrj/mptrj_li_cathodes.parquet")
    if mptrj_path.exists():
        df = pd.read_parquet(mptrj_path)
        if "material_id" in df.columns:
            material_ids.update(df["material_id"].dropna().tolist())
        print(f"MPTrj Li-cathodes: {len(df)} materials")
    
    # From MP 2024 Li-cathodes
    mp_path = Path("data/external/mp_2024/mp_li_cathodes_2024.parquet")
    if mp_path.exists():
        df = pd.read_parquet(mp_path)
        if "material_id" in df.columns:
            material_ids.update(df["material_id"].dropna().tolist())
        print(f"MP 2024 Li-cathodes: {len(df)} materials")
    
    print(f"\nTotal unique material IDs: {len(material_ids)}")
    
    # Download from MP API
    training_data = []
    
    with MPRester(api_key) as mpr:
        print("\nDownloading structures from Materials Project...")
        
        # Convert to list and batch process
        mp_ids = [mid for mid in material_ids if mid and str(mid).startswith("mp-")]
        print(f"Valid MP IDs: {len(mp_ids)}")
        
        # Download in batches
        batch_size = 100
        for i in tqdm(range(0, len(mp_ids), batch_size)):
            batch_ids = mp_ids[i:i+batch_size]
            
            try:
                # Get structure data with energies
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=["material_id", "structure", "energy_per_atom", 
                            "formation_energy_per_atom", "band_gap", "is_stable"]
                )
                
                for doc in docs:
                    if doc.structure is not None:
                        training_data.append({
                            "material_id": str(doc.material_id),
                            "structure": doc.structure.as_dict(),
                            "energy_per_atom": doc.energy_per_atom,
                            "formation_energy": doc.formation_energy_per_atom,
                            "band_gap": doc.band_gap,
                            "is_stable": doc.is_stable,
                        })
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
    
    print(f"\nDownloaded {len(training_data)} structures with energies")
    
    # Save as JSON for training
    output_path = output_dir / "li_cathode_structures.json"
    with open(output_path, "w") as f:
        json.dump(training_data, f)
    print(f"Saved to {output_path}")
    
    # Also save as pickle for faster loading
    pickle_path = output_dir / "li_cathode_structures.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved to {pickle_path}")
    
    return training_data

if __name__ == "__main__":
    download_mp_structures()
