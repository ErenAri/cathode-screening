"""Test benchmark functions with simulated data."""
import numpy as np
import sys
sys.path.insert(0, 'scripts')

# Import benchmark functions
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("benchmark", "scripts/39_benchmark_model.py")
benchmark = module_from_spec(spec)
spec.loader.exec_module(benchmark)

# Simulate prediction data
np.random.seed(42)
n_samples = 1000

# Ground truth: energy per atom with some stable (<= 0) and unstable (> 0)
targets = np.random.randn(n_samples) * 0.1

# Simulated predictions with some error (~30 meV MAE)
predictions = targets + np.random.randn(n_samples) * 0.03

# Ensemble uncertainties
uncertainties = np.abs(np.random.randn(n_samples)) * 0.02

print("=" * 60)
print("Testing Benchmark Suite with Simulated Data")
print("=" * 60)
print(f"n_samples: {n_samples}")
print(f"Stable (<=0): {(targets <= 0).sum()} ({(targets <= 0).mean()*100:.1f}%)")
print(f"True MAE: {np.mean(np.abs(predictions - targets)):.4f} eV/atom")

# Run benchmark
results = benchmark.run_benchmark(
    predictions=predictions,
    targets=targets,
    uncertainties=uncertainties,
    stability_threshold=0.0
)

# Print report
benchmark.print_benchmark_report(results)

# Save results
benchmark.save_benchmark_report(results, "test_benchmark_results.json")
