#!/usr/bin/env python
"""
Run simulated active learning experiment.

Uses the test set as a simulated oracle to compare different
acquisition strategies for materials discovery.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_oracle(pool_df: pd.DataFrame) -> callable:
    """Create oracle function from labeled data."""
    label_map = dict(zip(pool_df["material_id"], pool_df["y_true"]))
    
    def oracle(material_ids: List[str]) -> Dict[str, float]:
        return {mid: label_map.get(mid, np.nan) for mid in material_ids}
    
    return oracle


def run_simulation(
    pool_df: pd.DataFrame,
    acquisition_fn: str,
    n_iterations: int = 50,
    batch_size: int = 10,
    initial_train_size: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run AL simulation with given acquisition function."""
    from cathode_screening.active_learning.loop import ActiveLearningLoop
    
    oracle = create_oracle(pool_df)
    
    al = ActiveLearningLoop(
        pool_df=pool_df,
        oracle_fn=oracle,
        predictor=None,  # Use simulated predictions
        acquisition_fn=acquisition_fn,
        stability_threshold=0.05,
        initial_train_size=initial_train_size,
        seed=seed,
    )
    
    history = al.run(n_iterations, batch_size, verbose=False)
    history["acquisition"] = acquisition_fn
    
    return history


def plot_discovery_curves(
    results: Dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "Active Learning Discovery Curves",
):
    """Plot cumulative discovery curves for different strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "random": "#888888",
        "uncertainty": "#1f77b4",
        "expected_improvement": "#2ca02c",
        "ucb": "#ff7f0e",
        "pi": "#9467bd",
    }
    
    for name, df in results.items():
        color = colors.get(name, "#000000")
        ax.plot(
            df["n_queries"],
            df["n_stable_found"],
            label=name.replace("_", " ").title(),
            color=color,
            linewidth=2,
        )
    
    # Plot random baseline
    ax.set_xlabel("Number of DFT Queries", fontsize=12)
    ax.set_ylabel("Cumulative Stable Materials Found", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved discovery curves to {output_path}")


def compute_metrics(
    results: Dict[str, pd.DataFrame],
    n_total_stable: int,
) -> pd.DataFrame:
    """Compute summary metrics for each strategy."""
    metrics = []
    
    for name, df in results.items():
        final = df.iloc[-1]
        
        # Area Under Discovery Curve (normalized)
        audc = np.trapz(df["n_stable_found"], df["n_queries"])
        max_audc = final["n_queries"] * n_total_stable
        audc_normalized = audc / max_audc if max_audc > 0 else 0
        
        # Queries to find 50% of stable materials
        target = n_total_stable * 0.5
        queries_50 = df[df["n_stable_found"] >= target]["n_queries"].min()
        
        metrics.append({
            "strategy": name,
            "final_stable_found": final["n_stable_found"],
            "total_queries": final["n_queries"],
            "AUDC_normalized": audc_normalized,
            "queries_to_50pct": queries_50,
        })
    
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="Run AL simulation")
    parser.add_argument("--config", type=str, default="configs/train_cgcnn_ehull_soap_loco.yaml")
    parser.add_argument("--predictions", type=str, default="data/predictions/ensemble_soap_loco_test.parquet")
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data/reports/active_learning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions (which have ground truth labels)
    print(f"Loading predictions from {args.predictions}...")
    pred_df = pd.read_parquet(args.predictions)
    
    # Prepare pool
    pool_df = pred_df[["material_id", "y_true"]].copy()
    n_total_stable = (pool_df["y_true"] < 0.05).sum()
    
    print(f"Pool size: {len(pool_df)}")
    print(f"Total stable materials: {n_total_stable}")
    print(f"Stability rate: {100*n_total_stable/len(pool_df):.1f}%")
    print()
    
    # Run simulations for each strategy
    strategies = ["random", "uncertainty", "expected_improvement", "ucb", "pi"]
    results = {}
    
    for strategy in strategies:
        print(f"Running {strategy}...")
        results[strategy] = run_simulation(
            pool_df=pool_df.copy(),
            acquisition_fn=strategy,
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
            initial_train_size=args.initial_size,
            seed=args.seed,
        )
    
    # Plot discovery curves
    plot_discovery_curves(
        results,
        output_dir / "discovery_curves.png",
        title=f"AL Discovery Curves (batch={args.batch_size})",
    )
    
    # Compute metrics
    metrics_df = compute_metrics(results, n_total_stable)
    print("\n" + "=" * 60)
    print("Summary Metrics")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nSaved metrics to {output_dir / 'metrics.csv'}")
    
    # Save full results
    for name, df in results.items():
        df.to_csv(output_dir / f"history_{name}.csv", index=False)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
