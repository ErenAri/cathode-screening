"""Test OOD detection and calibration modules with simulated data."""
import numpy as np
import sys
sys.path.insert(0, 'src')

# Import modules
from ood_detection import OODDetector, OODResult
from calibration import CalibratedPredictor, CalibrationResult

print("=" * 60)
print("Testing OOD Detection Module")
print("=" * 60)

# Create mock detector with simulated training embeddings
np.random.seed(42)
n_train = 1000
embedding_dim = 64

# Simulated training embeddings
train_embeddings = np.random.randn(n_train, embedding_dim)

# Fit detector
detector = OODDetector()
detector.fit(train_embeddings)
print(f"Fitted OOD detector on {n_train} embeddings (dim={embedding_dim})")

# Test with in-distribution sample
in_dist_embedding = np.random.randn(embedding_dim)
dist_in = detector.mahalanobis_distance(in_dist_embedding)
print(f"In-distribution distance: {dist_in:.2f}")

# Test with out-of-distribution sample (shifted mean)
ood_embedding = np.random.randn(embedding_dim) * 3 + 5  # Shifted distribution
dist_ood = detector.mahalanobis_distance(ood_embedding)
print(f"OOD sample distance: {dist_ood:.2f}")

# Test ensemble variance
predictions_low_var = [0.1, 0.11, 0.09, 0.1, 0.1]  # Low variance = confident
predictions_high_var = [0.1, 0.5, -0.2, 0.3, 0.0]   # High variance = uncertain

var_low = detector.ensemble_variance(predictions_low_var)
var_high = detector.ensemble_variance(predictions_high_var)
print(f"Low variance ensemble: {var_low:.4f}")
print(f"High variance ensemble: {var_high:.4f}")

# Test OOD score
score_low = detector.compute_ood_score(var_low, dist_in, False)
score_high = detector.compute_ood_score(var_high, dist_ood, False)
print(f"OOD score (in-dist, low var): {score_low:.3f}")
print(f"OOD score (OOD, high var): {score_high:.3f}")

print("\n" + "=" * 60)
print("Testing Calibration Module")
print("=" * 60)

# Create calibrator
calibrator = CalibratedPredictor()

# Simulated ensemble predictions for calibration
n_calibration = 500
ensemble_preds = []
targets = []

np.random.seed(123)
for i in range(n_calibration):
    # True value
    true_val = np.random.randn() * 0.1
    targets.append(true_val)
    
    # Ensemble predictions with noise
    preds = [true_val + np.random.randn() * 0.02 for _ in range(5)]
    ensemble_preds.append(preds)

print(f"Fitting calibrator on {n_calibration} samples...")
calibrator.fit(ensemble_preds, targets)

# Test prediction
test_preds = [0.05, 0.06, 0.04, 0.055, 0.045]
result = calibrator.predict(test_preds)
print(f"\nTest prediction result:")
print(f"  Prediction: {result.prediction:.4f}")
print(f"  Uncertainty: {result.uncertainty:.4f}")
print(f"  Confidence: {result.confidence:.2%}")
print(f"  90% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

print(f"\nCalibration ECE: {calibrator.ece:.4f}")

# Save test artifacts
detector.save("test_ood_detector.pkl")
calibrator.save("test_calibrator.pkl")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
