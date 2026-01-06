"""
Decision-level calibration tracking.

Measures whether model confidence aligns with actual decision outcomes.
For example: "Of materials we said 'KILL' at 90% confidence, 
what fraction were actually unstable?"

This goes beyond point-estimate calibration to directly measure
decision quality as recommended in the research document.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DecisionCalibrationStats:
    """Statistics for decision-level calibration."""
    
    # Per-decision type stats
    keep_count: int = 0
    keep_correct: int = 0  # KEEP and actually stable
    
    maybe_count: int = 0
    
    kill_count: int = 0
    kill_correct: int = 0  # KILL and actually unstable
    
    # Confidence-binned stats
    confidence_bins: Dict[str, Dict] = field(default_factory=dict)
    
    @property
    def keep_precision(self) -> float:
        """P(stable | KEEP decision)"""
        return self.keep_correct / self.keep_count if self.keep_count > 0 else float('nan')
    
    @property
    def kill_precision(self) -> float:
        """P(unstable | KILL decision)"""
        return self.kill_correct / self.kill_count if self.kill_count > 0 else float('nan')
    
    @property
    def kill_error_rate(self) -> float:
        """P(stable | KILL decision) - the costly error"""
        if self.kill_count == 0:
            return float('nan')
        return 1.0 - self.kill_precision
    
    @property
    def decisive_rate(self) -> float:
        """Fraction of decisions that are KEEP or KILL (not MAYBE)"""
        total = self.keep_count + self.maybe_count + self.kill_count
        if total == 0:
            return float('nan')
        return (self.keep_count + self.kill_count) / total


def compute_decision_calibration(
    decisions: np.ndarray,
    y_true: np.ndarray,
    p_stable: Optional[np.ndarray] = None,
    confidence_bins: int = 5,
    threshold_stable: float = 0.05,
) -> DecisionCalibrationStats:
    """
    Compute decision-level calibration statistics.
    
    Args:
        decisions: [N] array of "KEEP"/"MAYBE"/"KILL" decisions
        y_true: [N] ground truth E_hull values
        p_stable: [N] predicted stability probabilities (optional)
        confidence_bins: Number of bins for confidence-stratified analysis
        threshold_stable: E_hull threshold for stability
    
    Returns:
        DecisionCalibrationStats with precision and calibration info
    """
    is_stable = y_true < threshold_stable
    
    stats = DecisionCalibrationStats()
    
    # Count by decision type
    keep_mask = decisions == "KEEP"
    maybe_mask = decisions == "MAYBE"
    kill_mask = decisions == "KILL"
    
    stats.keep_count = int(keep_mask.sum())
    stats.keep_correct = int((keep_mask & is_stable).sum())
    
    stats.maybe_count = int(maybe_mask.sum())
    
    stats.kill_count = int(kill_mask.sum())
    stats.kill_correct = int((kill_mask & ~is_stable).sum())
    
    # Confidence-binned calibration (if p_stable provided)
    if p_stable is not None:
        bin_edges = np.linspace(0, 1, confidence_bins + 1)
        
        for i in range(confidence_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_label = f"{lo:.1f}-{hi:.1f}"
            
            # Samples in this confidence bin
            if i == confidence_bins - 1:
                bin_mask = (p_stable >= lo) & (p_stable <= hi)
            else:
                bin_mask = (p_stable >= lo) & (p_stable < hi)
            
            n_bin = bin_mask.sum()
            if n_bin == 0:
                continue
            
            # Decision distribution in this bin
            bin_keep = (bin_mask & keep_mask).sum()
            bin_maybe = (bin_mask & maybe_mask).sum()
            bin_kill = (bin_mask & kill_mask).sum()
            
            # Actual stability rate
            actual_stable_rate = is_stable[bin_mask].mean()
            
            # Mean predicted confidence
            mean_confidence = p_stable[bin_mask].mean()
            
            stats.confidence_bins[bin_label] = {
                "n": int(n_bin),
                "mean_confidence": float(mean_confidence),
                "actual_stable_rate": float(actual_stable_rate),
                "calibration_error": float(abs(mean_confidence - actual_stable_rate)),
                "n_keep": int(bin_keep),
                "n_maybe": int(bin_maybe),
                "n_kill": int(bin_kill),
            }
    
    return stats


def compute_decision_level_ece(
    decisions: np.ndarray,
    y_true: np.ndarray,
    p_stable: np.ndarray,
    threshold_stable: float = 0.05,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute ECE stratified by decision type.
    
    This answers: "Is the model well-calibrated for the samples it 
    confidently KEEP vs KILL?"
    
    Args:
        decisions: [N] array of decisions
        y_true: [N] ground truth E_hull
        p_stable: [N] predicted stability probability
        threshold_stable: Stability threshold
        n_bins: Number of calibration bins
    
    Returns:
        Dict with ECE per decision type
    """
    is_stable = y_true < threshold_stable
    
    results = {}
    
    for decision_type in ["KEEP", "MAYBE", "KILL"]:
        mask = decisions == decision_type
        n = mask.sum()
        
        if n < 10:
            results[f"ece_{decision_type.lower()}"] = float('nan')
            continue
        
        p_sub = p_stable[mask]
        y_sub = is_stable[mask].astype(float)
        
        # Compute ECE for this subset
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            
            if i == n_bins - 1:
                bin_mask = (p_sub >= lo) & (p_sub <= hi)
            else:
                bin_mask = (p_sub >= lo) & (p_sub < hi)
            
            n_bin = bin_mask.sum()
            if n_bin == 0:
                continue
            
            mean_conf = p_sub[bin_mask].mean()
            mean_acc = y_sub[bin_mask].mean()
            
            ece += (n_bin / n) * abs(mean_conf - mean_acc)
        
        results[f"ece_{decision_type.lower()}"] = float(ece)
    
    return results


def format_decision_calibration_report(
    stats: DecisionCalibrationStats,
    ece_by_decision: Optional[Dict[str, float]] = None,
) -> str:
    """
    Format decision calibration as readable report.
    
    Args:
        stats: Output from compute_decision_calibration()
        ece_by_decision: Optional ECE per decision type
    
    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "DECISION-LEVEL CALIBRATION REPORT",
        "=" * 60,
        "",
        "Decision Distribution:",
        f"  KEEP:  {stats.keep_count:5d}  (precision: {stats.keep_precision:.1%})",
        f"  MAYBE: {stats.maybe_count:5d}",
        f"  KILL:  {stats.kill_count:5d}  (precision: {stats.kill_precision:.1%})",
        "",
        f"Decisive Rate: {stats.decisive_rate:.1%}",
        f"KILL Error Rate (missed stable): {stats.kill_error_rate:.2%}",
        "",
    ]
    
    if ece_by_decision:
        lines.extend([
            "ECE by Decision Type:",
            f"  KEEP samples:  {ece_by_decision.get('ece_keep', float('nan')):.4f}",
            f"  MAYBE samples: {ece_by_decision.get('ece_maybe', float('nan')):.4f}",
            f"  KILL samples:  {ece_by_decision.get('ece_kill', float('nan')):.4f}",
            "",
        ])
    
    if stats.confidence_bins:
        lines.extend([
            "Calibration by Confidence Bin:",
            "-" * 60,
            f"{'Bin':<12} {'N':>6} {'Conf':>8} {'Actual':>8} {'|Err|':>8} {'KEEP':>6} {'KILL':>6}",
            "-" * 60,
        ])
        
        for bin_label, bin_data in sorted(stats.confidence_bins.items()):
            lines.append(
                f"{bin_label:<12} {bin_data['n']:>6} "
                f"{bin_data['mean_confidence']:>8.3f} "
                f"{bin_data['actual_stable_rate']:>8.3f} "
                f"{bin_data['calibration_error']:>8.3f} "
                f"{bin_data['n_keep']:>6} {bin_data['n_kill']:>6}"
            )
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
