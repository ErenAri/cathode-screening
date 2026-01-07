#!/usr/bin/env python
"""
Train ALIGNN model on cathode screening data.

Uses the SOAP-LOCO splits and trains ALIGNN for E_hull prediction.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from jarvis.core.atoms import Atoms as JarvisAtoms
from pymatgen.core import Structure


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pymatgen_to_jarvis(structure):
    """Convert pymatgen Structure to JARVIS Atoms."""
    return JarvisAtoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[str(s) for s in structure.species],
        cartesian=False,
    )


def prepare_alignn_dataset(
    raw_parquet: Path,
    splits_json: Path,
    output_dir: Path,
    target_col: str = "energy_above_hull",
):
    """
    Prepare dataset in ALIGNN format.
    
    Creates id_prop.csv and POSCAR files for each structure.
    """
    print(f"Loading raw data from {raw_parquet}...")
    df = pd.read_parquet(raw_parquet)
    
    print(f"Loading splits from {splits_json}...")
    with open(splits_json, "r") as f:
        splits = json.load(f)
    
    # Filter to materials in splits
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    df = df[df["material_id"].isin(all_ids)].reset_index(drop=True)
    print(f"Filtered to {len(df)} materials in splits")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Prepare each split
    for split_name, material_ids in splits.items():
        split_dir = output_dir / split_name
        split_df = df[df["material_id"].isin(material_ids)].copy()
        
        if len(split_df) == 0:
            print(f"  {split_name}: 0 materials (skipping)")
            continue
        
        print(f"  {split_name}: {len(split_df)} materials")
        
        # Write id_prop.csv
        id_prop_path = split_dir / "id_prop.csv"
        id_prop_data = []
        
        for idx, row in split_df.iterrows():
            mid = row["material_id"]
            target = row[target_col]
            
            # Parse structure from CIF
            try:
                struct = Structure.from_str(row["cif"], fmt="cif")
                atoms = pymatgen_to_jarvis(struct)
                
                # Write POSCAR
                poscar_path = split_dir / f"{mid}.vasp"
                atoms.write_poscar(str(poscar_path))
                
                id_prop_data.append({
                    "id": mid,
                    "target": target,
                })
            except Exception as e:
                print(f"    Warning: Failed to process {mid}: {e}")
        
        # Write id_prop.csv
        id_prop_df = pd.DataFrame(id_prop_data)
        id_prop_df.to_csv(id_prop_path, index=False, header=False)
        print(f"    Wrote {len(id_prop_df)} entries to {id_prop_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ALIGNN training data")
    parser.add_argument("--config", type=str, default="configs/train_cgcnn_ehull_soap_loco.yaml")
    parser.add_argument("--output-dir", type=str, default="data/alignn_format")
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    raw_parquet = Path("data/raw/mp/raw_mp_cathodes_v1.parquet")
    splits_json = Path(cfg["data"]["splits_manifest"])
    output_dir = Path(args.output_dir)
    
    prepare_alignn_dataset(
        raw_parquet=raw_parquet,
        splits_json=splits_json,
        output_dir=output_dir,
        target_col=cfg["data"]["target"]["name"],
    )
    
    print(f"\nDone! ALIGNN training data saved to {output_dir}")
    print("\nTo train ALIGNN, run:")
    print(f"  alignn_train --train_folder {output_dir}/train --val_folder {output_dir}/val --test_folder {output_dir}/test --output_dir artifacts/models/alignn --epochs 100")


if __name__ == "__main__":
    main()
