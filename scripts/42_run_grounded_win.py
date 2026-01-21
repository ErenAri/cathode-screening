
import os
import json
import pickle
import glob
import numpy as np
import torch
from pathlib import Path
import sys
from tqdm import tqdm
from chgnet.model import CHGNet
from pymatgen.core import Structure

# Import our new modules
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

def load_dataset(data_path):
    print(f"Loading dataset from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} structures.")
    # Index by material_id
    data_dict = {d["material_id"]: d for d in data if "material_id" in d}
    print(f"Indexed {len(data_dict)} items by material_id.")
    return data_dict

def load_splits(split_path):
    print(f"Loading splits from {split_path}...")
    with open(split_path, "r") as f:
        splits = json.load(f)
    print(f"Split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    return splits

def load_ensemble(checkpoint_dir, n_models=None, device="cpu"):
    models = []
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    checkpoint_dir = Path(checkpoint_dir)
    print(f"Loading models from {checkpoint_dir} to {device}...")

    # Newer training saves checkpoints directly in the directory.
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
    """
    Predict energy for a list of structures using ensemble.
    Returns list of [pred_1, pred_2, ...] for each structure.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Running inference on {len(structures)} structures...")
    
    # Batching could be faster, but CHGNet default predict is per-structure or list
    # predict_structure supports list of structures.
    
    # We can run each model on all structures
    # But memory might be issue if N is large. structure list is small (1-2k) so ok.
    
    # Organize by model first to batch?
    # chgnet.predict_structure(structures, batch_size=...)
    
    n_models = len(models)
    n_structs = len(structures)
    
    all_model_preds = np.zeros((n_models, n_structs))
    
    for i, model in enumerate(models):
        print(f"  Model {i+1}/{n_models} predicting...")
        if device != "cpu":
            model.to(device)
        # CHGNet prediction returns dict or list of dicts
        # predict_structure with return_site_energies=False, return_atom_feas=False
        # Custom batch size support if needed, but for now simple sequential or use internal batching if supported
        # CHGNet predict_structure usually takes batch_size for list of structures
        with torch.no_grad():
            preds = model.predict_structure(
                structures,
                task="e",
                batch_size=batch_size,
            )
        # Extract energies 'e' (eV/atom usually or total eV? check wrapper)
        # CHGNet.predict_structure returns 'e' as energy per atom if task was energy per atom?
        # Standard CHGNet output 'e' is TOTAL energy in eV.
        # We need energy per atom for comparison/thresholds usually.
        # But 'is_stable' is the ground truth.
        
        # NOTE: If we use Energy Per Atom, we must divide by len(s).
        
        for j, res in enumerate(preds):
            # CHGNet returns intensive energy when model.is_intensive is True
            e_per_atom = res["e"]
            if hasattr(e_per_atom, "item"):
                e_per_atom = e_per_atom.item()
            if not getattr(model, "is_intensive", True):
                e_per_atom /= len(structures[j])
            all_model_preds[i, j] = e_per_atom
        if device != "cpu":
            model.to("cpu")
            torch.cuda.empty_cache()
            
    # Transpose to get list of preds per structure
    return all_model_preds.T.tolist()

def main():
    default_base = Path(__file__).resolve().parents[1]
    BASE_DIR = Path(os.getenv("CATHODE_BASE_DIR", str(default_base)))
    DATA_PATH = Path(os.getenv("CATHODE_DATA_PATH", str(BASE_DIR / "data/training/li_cathode_structures.pkl")))
    SPLIT_PATH = Path(os.getenv("CATHODE_SPLIT_PATH", str(BASE_DIR / "data/splits/mp/splits_mp_cathodes_v1_soap-loco.json")))
    CKPT_DIR = Path(os.getenv("CATHODE_CKPT_DIR", str(BASE_DIR / "checkpoints/gcp_h100")))
    OUTPUT_PATH = Path(os.getenv("CATHODE_REPORT_PATH", str(BASE_DIR / "reports/grounded_win_h100.json")))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Required on Windows for CHGNet cygraph dtype mismatch.
    patch_chgnet_make_graph()
    
    # 1. Load Data
    data_dict = load_dataset(DATA_PATH)
    splits = load_splits(SPLIT_PATH)
    
    # 2. Prepare Sets
    def get_subset(split_name):
        subset = []
        for mid in splits[split_name]:
            if mid in data_dict:
                item = data_dict[mid]
                # Reconstruct Structure object
                s_dict = item["structure"]
                struct = Structure.from_dict(s_dict)
                subset.append({
                    "material_id": mid,
                    "structure": struct,
                    "is_stable": item["is_stable"],
                    # Optional: "e_hull": item.get("energy_above_hull")
                })
        return subset

    val_set = get_subset("val")
    test_set = get_subset("test")
    print(f"Prepared: Val={len(val_set)}, Test={len(test_set)}")
    
    if len(test_set) == 0:
        print("Error: Test set empty! Check matching of IDs.")
        return

    # 3. Load Models (keep on CPU; move to GPU per model during inference)
    n_models_env = os.getenv("CATHODE_N_MODELS")
    n_models = int(n_models_env) if n_models_env else None
    models = load_ensemble(CKPT_DIR, n_models=n_models, device="cpu")
    device = os.getenv("CATHODE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.getenv("CATHODE_BATCH_SIZE", "32"))
    print(f"Predict device: {device} | batch_size={batch_size}")
    
    # 4. Predict on Val (for Calibration)
    print("\n--- Calibration (Val Set) ---")
    val_structs = [x["structure"] for x in val_set]
    val_labels = [x["is_stable"] for x in val_set]
    val_preds = predict_ensemble(models, val_structs, device=device, batch_size=batch_size)

    # Quick diagnostics on val predictions for calibrator fit
    val_means = np.array([np.mean(p) for p in val_preds], dtype=float)
    val_stds = np.array([np.std(p) for p in val_preds], dtype=float)
    print(
        "Val stats: "
        f"mean_min={np.min(val_means):.3f}, mean_max={np.max(val_means):.3f}, "
        f"mean_mean={np.mean(val_means):.3f}, std_min={np.min(val_stds):.6f}, "
        f"std_max={np.max(val_stds):.6f}, std_mean={np.mean(val_stds):.6f}"
    )
    
    # 5. Fit Calibrator
    # Default to mean-energy calibration (feature_mode="mu") for stability
    # with small ensemble std. Override with CATHODE_CALIBRATOR_MODE if needed.
    
    feature_mode = os.getenv("CATHODE_CALIBRATOR_MODE", "mu").strip().lower()
    calibrator = CalibratedPredictor(threshold=0.0, feature_mode=feature_mode)
    calibrator.fit(val_preds, val_labels)
    if isinstance(calibrator.platt_a, list):
        a_str = ",".join(f"{a:.3f}" for a in calibrator.platt_a)
    else:
        a_str = f"{calibrator.platt_a:.3f}"
    print(f"Calibrator mode: {feature_mode}")
    print(f"Calibrator fitted: a={a_str}, b={calibrator.platt_b:.3f}")
    
    # 6. Predict on Test
    print("\n--- Evaluation (Test Set) ---")
    test_structs = [x["structure"] for x in test_set]
    test_preds = predict_ensemble(models, test_structs, device=device, batch_size=batch_size)
    
    # 7. Generate Report
    # We need to construct test_dataset list with preds? 
    # No, predict_ensemble returns lists of preds.
    # grounded_win_report logic:
    #   p = calibrator.predict_p_stable(ensemble.predict(s))
    # We already have ensemble preds.
    
    # We can't pass ensemble object easily if we pre-computed preds.
    # Let's wrap pre-computed preds in a mock ensemble or modify grounded_win_report?
    # Modification is cleaner but let's just make a MockEnsemble for compat.
    
    class MockEnsemble:
        def __init__(self, all_preds):
            self.preds = all_preds # List of lists
            self.idx = 0
            
        def predict(self, struct):
            # This relies on sequential calls match order
            # grounded_win_report iterates test_set in order
            p = self.preds[self.idx]
            self.idx += 1
            return p
            
    mock_ensemble = MockEnsemble(test_preds)
    
    report = grounded_win_report(
        test_set, 
        mock_ensemble, 
        calibrator, 
        n_random=200
    )
    
    # 8. Output
    print("\n=== Grounded Win Report (GCP H100) ===")
    print(json.dumps(report, indent=2))
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {OUTPUT_PATH}")

    # 9. Debug artifacts for ranking sanity checks
    test_labels = np.array([x["is_stable"] for x in test_set]).astype(int)
    test_p_stable = np.array([calibrator.predict_p_stable(p) for p in test_preds])
    order_desc = np.argsort(test_p_stable)[::-1]
    order_asc = np.argsort(test_p_stable)

    def topk_summary(order, k=20):
        out = []
        for i, idx in enumerate(order[:k], start=1):
            out.append({
                "rank": i,
                "material_id": test_set[idx]["material_id"],
                "p_stable": float(test_p_stable[idx]),
                "is_stable": int(test_labels[idx]),
            })
        return out

    debug = {
        "n_test": int(len(test_labels)),
        "n_stable": int(test_labels.sum()),
        "calibrator_mode": feature_mode,
        "calibrator_threshold": float(calibrator.threshold),
        "val_pred_stats": {
            "mean_min": float(np.min(val_means)),
            "mean_max": float(np.max(val_means)),
            "mean_mean": float(np.mean(val_means)),
            "std_min": float(np.min(val_stds)),
            "std_max": float(np.max(val_stds)),
            "std_mean": float(np.mean(val_stds)),
        },
        "p_stable_min": float(np.nanmin(test_p_stable)),
        "p_stable_max": float(np.nanmax(test_p_stable)),
        "p_stable_mean": float(np.nanmean(test_p_stable)),
        "p_stable_stable_mean": float(np.nanmean(test_p_stable[test_labels == 1]))
        if np.any(test_labels == 1) else None,
        "p_stable_unstable_mean": float(np.nanmean(test_p_stable[test_labels == 0]))
        if np.any(test_labels == 0) else None,
        "n_nonfinite": int(np.sum(~np.isfinite(test_p_stable))),
        "top20_desc": topk_summary(order_desc, k=20),
        "top20_asc": topk_summary(order_asc, k=20),
        "p_stable": test_p_stable.tolist(),
        "y_true": test_labels.tolist(),
    }

    debug_path = OUTPUT_PATH.with_name("grounded_win_h100_debug.json")
    with open(debug_path, "w") as f:
        json.dump(debug, f, indent=2)
    print(f"Saved debug report to {debug_path}")

if __name__ == "__main__":
    import sys
    # Add src to path to import modules
    sys.path.append(str(Path("c:/projects/cathode-screening/src")))
    main()
