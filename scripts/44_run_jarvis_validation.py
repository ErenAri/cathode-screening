import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from tqdm import tqdm
from chgnet.model import CHGNet
from pymatgen.core import Structure

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from calibrated_predictor import CalibratedPredictor
from grounded_win import grounded_win_report


def patch_chgnet_make_graph():
    """Patch CHGNet cygraph to coerce indices to int64 on Windows."""
    try:
        import numpy as np
        import chgnet.graph.converter
        from chgnet.graph import cygraph
    except Exception as exc:
        print(f"Warning: failed to patch CHGNet make_graph ({exc})")
        return

    original_make_graph = cygraph.make_graph

    def make_graph_patched(center_index, num_edges, neighbor_index, image, distance, n_atoms):
        center_index = np.ascontiguousarray(center_index, dtype=np.int64)
        neighbor_index = np.ascontiguousarray(neighbor_index, dtype=np.int64)
        image = np.ascontiguousarray(image, dtype=np.int64)
        distance = np.ascontiguousarray(distance, dtype=np.float64)
        return original_make_graph(
            center_index, int(num_edges), neighbor_index, image, distance, int(n_atoms)
        )

    chgnet.graph.converter.make_graph = make_graph_patched


def load_mp_val_set(data_path: Path, split_path: Path):
    print(f"Loading MP dataset from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data_dict = {d["material_id"]: d for d in data if "material_id" in d}
    print(f"Loaded {len(data_dict)} MP structures.")

    print(f"Loading splits from {split_path}...")
    with open(split_path, "r") as f:
        splits = json.load(f)

    val_set = []
    for mid in splits["val"]:
        if mid not in data_dict:
            continue
        item = data_dict[mid]
        struct = Structure.from_dict(item["structure"])
        val_set.append(
            {
                "material_id": mid,
                "structure": struct,
                "is_stable": item["is_stable"],
            }
        )
    print(f"Prepared MP val set: {len(val_set)}")
    return val_set


def _jarvis_to_structure(entry: dict) -> Structure:
    atoms = entry.get("atoms")
    if isinstance(atoms, dict):
        lattice = atoms.get("lattice_mat")
        coords = atoms.get("coords")
        elements = atoms.get("elements")
        if lattice is not None and coords is not None and elements is not None:
            cartesian = bool(atoms.get("cartesian", False))
            return Structure(lattice, elements, coords, coords_are_cartesian=cartesian)
    final_str = entry.get("final_str")
    if isinstance(final_str, dict):
        return Structure.from_dict(final_str)
    raise ValueError("No structure data in JARVIS entry")


def load_jarvis_set(jarvis_json: Path, stable_thresh: float):
    print(f"Loading JARVIS JSON from {jarvis_json}...")
    with open(jarvis_json, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} JARVIS rows.")

    test_set = []
    missing = 0
    for entry in tqdm(data, desc="Filtering JARVIS"):
        elements = []
        atoms = entry.get("atoms")
        if isinstance(atoms, dict):
            elements = atoms.get("elements", []) or []
        if "Li" not in elements or "O" not in elements:
            continue
        has_tm = any(el in {"Fe", "Co", "Ni", "Mn", "V", "Cr", "Ti", "Cu", "Zn", "Mo", "W"} for el in elements)
        if not has_tm:
            continue

        e_hull = entry.get("ehull")
        if e_hull is None:
            continue

        try:
            struct = _jarvis_to_structure(entry)
        except Exception:
            missing += 1
            continue

        test_set.append(
            {
                "jarvis_id": entry.get("jid"),
                "structure": struct,
                "is_stable": float(e_hull) <= stable_thresh,
                "e_hull": float(e_hull),
            }
        )
    print(f"Prepared JARVIS test set: {len(test_set)} (missing structures: {missing})")
    return test_set


def load_ensemble(checkpoint_dir, n_models=None, device="cpu"):
    models = []
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    checkpoint_dir = Path(checkpoint_dir)
    print(f"Loading models from {checkpoint_dir} to {device}...")

    model_subdirs = sorted([p for p in checkpoint_dir.glob("finetune_model_*") if p.is_dir()])
    if not model_subdirs:
        model_subdirs = sorted([p for p in checkpoint_dir.glob("seed_*") if p.is_dir()])
    if model_subdirs:
        if n_models is not None:
            model_subdirs = model_subdirs[:n_models]
        for i, model_subdir in enumerate(model_subdirs):
            ckpts = list(model_subdir.glob("bestE_*.pth.tar"))
            if not ckpts:
                ckpts = list(model_subdir.glob("*.pth.tar"))
            if not ckpts:
                print(f"Warning: No checkpoint found for model {i}")
                continue
            ckpt_path = sorted(ckpts)[-1]
            print(f"  Model {i}: {ckpt_path.name}")
            model = CHGNet.from_file(str(ckpt_path))
            model.to(device)
            model.eval()
            models.append(model)
    else:
        ckpts = sorted(checkpoint_dir.glob("bestE_*.pth.tar"))
        if not ckpts:
            ckpts = sorted(checkpoint_dir.glob("*.pth.tar"))
        if n_models is not None:
            ckpts = ckpts[:n_models]
        if not ckpts:
            print("Warning: No checkpoints found in directory.")
            return models
        for i, ckpt_path in enumerate(ckpts):
            print(f"  Model {i}: {ckpt_path.name}")
            model = CHGNet.from_file(str(ckpt_path))
            model.to(device)
            model.eval()
            models.append(model)

    return models


def predict_ensemble(models, structures, device="cpu", batch_size=32):
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Running inference on {len(structures)} structures...")

    n_models = len(models)
    n_structs = len(structures)
    all_model_preds = np.zeros((n_models, n_structs))

    for i, model in enumerate(models):
        print(f"  Model {i+1}/{n_models} predicting...")
        if device != "cpu":
            model.to(device)
        with torch.no_grad():
            preds = model.predict_structure(
                structures,
                task="e",
                batch_size=batch_size,
            )
        for j, res in enumerate(preds):
            e_per_atom = res["e"]
            if hasattr(e_per_atom, "item"):
                e_per_atom = e_per_atom.item()
            if not getattr(model, "is_intensive", True):
                e_per_atom /= len(structures[j])
            all_model_preds[i, j] = e_per_atom
        if device != "cpu":
            model.to("cpu")
            torch.cuda.empty_cache()

    return all_model_preds.T.tolist()


def build_candidate_table(test_set, test_preds, calibrator):
    records = []
    for item, preds in zip(test_set, test_preds):
        mu = float(np.mean(preds))
        sigma = float(np.std(preds))
        p_stable = float(calibrator.predict_p_stable(preds))
        struct = item["structure"]
        records.append(
            {
                "jarvis_id": item.get("jarvis_id"),
                "formula": struct.composition.reduced_formula,
                "p_stable": p_stable,
                "mu": mu,
                "sigma": sigma,
                "e_hull": item.get("e_hull"),
                "is_stable": item.get("is_stable"),
            }
        )
    df = pd.DataFrame(records)
    df = df.sort_values("p_stable", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _parse_mix_counts(value: str):
    try:
        parts = [int(p.strip()) for p in value.split(",")]
    except Exception as exc:
        raise ValueError(f"Invalid CATHODE_DFT_MIX '{value}': {exc}")
    if len(parts) != 3 or any(p < 0 for p in parts):
        raise ValueError("CATHODE_DFT_MIX must be three non-negative integers like '40,5,5'")
    return parts


def main():
    default_base = Path(__file__).resolve().parents[1]
    base_dir = Path(os.getenv("CATHODE_BASE_DIR", str(default_base)))

    mp_data_path = Path(
        os.getenv("CATHODE_MP_DATA_PATH", str(base_dir / "data/training/li_cathode_structures.pkl"))
    )
    mp_split_path = Path(
        os.getenv("CATHODE_MP_SPLIT_PATH", str(base_dir / "data/splits/mp/splits_mp_cathodes_v1_soap-loco.json"))
    )
    jarvis_json = Path(
        os.getenv("CATHODE_JARVIS_JSON", str(base_dir / "data/external/jarvis/jdft_3d-12-12-2022.json"))
    )
    ckpt_dir = Path(os.getenv("CATHODE_CKPT_DIR", str(base_dir / "checkpoints/gcp_h100/ehull_oqmd_v1")))
    output_path = Path(os.getenv("CATHODE_REPORT_PATH", str(base_dir / "reports/grounded_win_jarvis.json")))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    patch_chgnet_make_graph()

    stable_thresh = float(os.getenv("CATHODE_STABLE_THRESH", "0.05"))
    device = os.getenv("CATHODE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.getenv("CATHODE_BATCH_SIZE", "32"))
    n_models_env = os.getenv("CATHODE_N_MODELS")
    n_models = int(n_models_env) if n_models_env else None
    feature_mode = os.getenv("CATHODE_CALIBRATOR_MODE", "mu").strip().lower()

    val_set = load_mp_val_set(mp_data_path, mp_split_path)
    test_set = load_jarvis_set(jarvis_json, stable_thresh)
    if len(test_set) == 0:
        print("Error: JARVIS test set empty.")
        return

    models = load_ensemble(ckpt_dir, n_models=n_models, device="cpu")
    print(f"Predict device: {device} | batch_size={batch_size}")

    print("\n--- Calibration (MP Val Set) ---")
    val_structs = [x["structure"] for x in val_set]
    val_labels = [x["is_stable"] for x in val_set]
    val_preds = predict_ensemble(models, val_structs, device=device, batch_size=batch_size)

    val_means = np.array([np.mean(p) for p in val_preds], dtype=float)
    val_stds = np.array([np.std(p) for p in val_preds], dtype=float)
    print(
        "Val stats: "
        f"mean_min={np.min(val_means):.3f}, mean_max={np.max(val_means):.3f}, "
        f"mean_mean={np.mean(val_means):.3f}, std_min={np.min(val_stds):.6f}, "
        f"std_max={np.max(val_stds):.6f}, std_mean={np.mean(val_stds):.6f}"
    )

    calibrator = CalibratedPredictor(threshold=0.0, feature_mode=feature_mode)
    calibrator.fit(val_preds, val_labels)
    if isinstance(calibrator.platt_a, list):
        a_str = ",".join(f"{a:.3f}" for a in calibrator.platt_a)
    else:
        a_str = f"{calibrator.platt_a:.3f}"
    print(f"Calibrator mode: {feature_mode}")
    print(f"Calibrator fitted: a={a_str}, b={calibrator.platt_b:.3f}")

    print("\n--- Evaluation (JARVIS Test Set) ---")
    test_structs = [x["structure"] for x in test_set]
    test_preds = predict_ensemble(models, test_structs, device=device, batch_size=batch_size)

    class MockEnsemble:
        def __init__(self, all_preds):
            self.preds = all_preds
            self.idx = 0

        def predict(self, _):
            res = self.preds[self.idx]
            self.idx += 1
            return res

    mock_ens = MockEnsemble(test_preds)
    report = grounded_win_report(test_set, mock_ens, calibrator)

    print("\n=== Grounded Win Report (JARVIS) ===")
    print(json.dumps(report, indent=2))

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {output_path}")

    debug = {
        "stable_threshold": stable_thresh,
        "n_val": len(val_set),
        "n_test": len(test_set),
        "val_mean_min": float(np.min(val_means)),
        "val_mean_max": float(np.max(val_means)),
        "val_mean_mean": float(np.mean(val_means)),
        "val_std_min": float(np.min(val_stds)),
        "val_std_max": float(np.max(val_stds)),
        "val_std_mean": float(np.mean(val_stds)),
        "calibrator_mode": feature_mode,
        "calibrator_a": calibrator.platt_a,
        "calibrator_b": calibrator.platt_b,
    }
    debug_path = output_path.with_name("grounded_win_jarvis_debug.json")
    with open(debug_path, "w") as f:
        json.dump(debug, f, indent=2)
    print(f"Saved debug report to {debug_path}")

    dft_budget = int(os.getenv("CATHODE_DFT_BUDGET", "0"))
    if dft_budget > 0:
        batch_path = Path(
            os.getenv(
                "CATHODE_DFT_BATCH_PATH",
                str(output_path.with_name(f"dft_batch_jarvis_{dft_budget}.csv")),
            )
        )
        df = build_candidate_table(test_set, test_preds, calibrator)
        dft_budget = min(dft_budget, len(df))

        strategy = os.getenv("CATHODE_DFT_STRATEGY", "top").strip().lower()
        if strategy == "mix":
            mix_value = os.getenv("CATHODE_DFT_MIX", "40,5,5")
            top_n, border_n, rand_n = _parse_mix_counts(mix_value)
            total_n = top_n + border_n + rand_n
            if total_n != dft_budget:
                raise ValueError(
                    f"CATHODE_DFT_MIX sums to {total_n}, but budget is {dft_budget}"
                )

            top = df.head(top_n).copy()
            border_start = top_n
            border_end = min(top_n + border_n, len(df))
            borderline = df.iloc[border_start:border_end].copy()

            remaining = df.drop(top.index.union(borderline.index))
            rand_n = min(rand_n, len(remaining))
            seed = int(os.getenv("CATHODE_DFT_SEED", "42"))
            rng = np.random.default_rng(seed)
            random = remaining.sample(n=rand_n, random_state=rng.integers(0, 2**32 - 1))

            top["selection"] = "top"
            borderline["selection"] = "borderline"
            random["selection"] = "random"

            batch = pd.concat([top, borderline, random], ignore_index=True)
        else:
            batch = df.head(dft_budget).copy()

        batch.to_csv(batch_path, index=False)
        hit_rate = float(batch["is_stable"].mean()) if "is_stable" in batch else 0.0
        print(f"Saved DFT batch ({dft_budget}) to {batch_path}")
        print(f"Expected stable hit rate (JARVIS labels): {hit_rate:.1%}")


if __name__ == "__main__":
    main()
