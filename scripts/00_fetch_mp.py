import os
import json
import time
import argparse
from pathlib import Path

import yaml
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_key(cfg: dict) -> str:
    env_name = cfg["mp"].get("api_key_env", "MP_API_KEY")
    key = os.getenv(env_name) or os.getenv("PMG_MAPI_KEY")
    if not key:
        raise RuntimeError(f"Missing API key. Set {env_name} or PMG_MAPI_KEY.")
    return key


def to_cif(structure) -> str:
    return structure.to(fmt="cif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["dataset"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"raw_{cfg['dataset']['name']}.parquet"

    key = env_key(cfg)
    fields = [
        "material_id",
        "formula_pretty",
        "elements",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "structure",
        "volume",
    ]

    records = []
    with MPRester(key, endpoint=cfg["mp"].get("base_url", "https://api.materialsproject.org")) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li", "O"],
            fields=fields,
            chunk_size=cfg["mp"].get("page_size", 500),
            num_chunks=None,
        )
        for i, d in enumerate(docs):
            if args.limit and i >= args.limit:
                break
            s = d.structure
            cif = to_cif(s) if s is not None else None
            records.append(
                {
                    "material_id": d.material_id,
                    "formula_pretty": d.formula_pretty,
                    "elements": [str(e) for e in d.elements] if d.elements else None,
                    "energy_above_hull": d.energy_above_hull,
                    "formation_energy_per_atom": d.formation_energy_per_atom,
                    "band_gap": d.band_gap,
                    "volume": d.volume,
                    "cif": cif,
                }
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)

    meta = {
        "dataset": cfg["dataset"]["name"],
        "rows": int(len(df)),
        "path": str(out_path),
        "fields": fields,
    }
    (out_dir / f"raw_{cfg['dataset']['name']}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
