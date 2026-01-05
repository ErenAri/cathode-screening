"""
Utility-optimized decision policy for cathode screening.

Two operating modes:
A) DFT-followup: false KEEP is cheap (just run DFT), false KILL is expensive (miss discovery)
B) Experimental: false KEEP is expensive (waste synthesis resources)

Cost Model:
    E[Cost] = C_keep_fp * P(KEEP | unstable) * P(unstable)
            + C_kill_fn * P(KILL | stable) * P(stable)  
            + C_maybe * P(MAYBE)

Policy Family:
    KEEP if q90_cal <= T_keep AND gate_level != OOD [AND p_stable >= P_keep]
    KILL if q10_cal >= T_kill [AND p_stable <= P_kill]
    else MAYBE

Usage:
    from cathode_screening.inference.decision_policy import (
        DecisionPolicy, CostModel, tune_policy_thresholds
    )
    
    policy = DecisionPolicy.for_mode("dft_followup")
    decision = policy.decide(q10_cal, q50, q90_cal, p_stable, gate_level)
    
    # Tune on validation
    best_policy, report = tune_policy_thresholds(val_data, mode="dft_followup")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

# Stability thresholds (ground truth)
THRESH_STABLE = 0.05      # E_hull < 0.05 = stable
THRESH_METASTABLE = 0.10  # E_hull < 0.10 = metastable

GateLevel = Literal["IN", "BORDERLINE", "OOD"]
Decision = Literal["KEEP", "MAYBE", "KILL"]


@dataclass
class CostModel:
    """
    Cost model for decision outcomes.
    
    Costs are relative (can scale arbitrarily, only ratios matter).
    Convention: C_maybe = 1.0 as baseline.
    """
    # Cost of false KEEP (KEEP an unstable material)
    # DFT mode: low (just waste a DFT run)
    # Experimental mode: high (waste synthesis)
    C_keep_fp: float
    
    # Cost of false KILL (KILL a stable material)
    # Both modes: high (miss a discovery)
    C_kill_fn: float
    
    # Cost of MAYBE (requires human review / further analysis)
    # Baseline cost
    C_maybe: float = 1.0
    
    # Cost of correct KEEP (KEEP a truly stable material)
    # Negative = benefit
    C_keep_tp: float = -2.0
    
    # Cost of correct KILL (KILL a truly unstable material)
    # Small benefit (saved effort)
    C_kill_tn: float = -0.5
    
    @classmethod
    def for_mode(cls, mode: str) -> "CostModel":
        """Get cost model for operating mode."""
        if mode == "dft_followup":
            # False KEEP is cheap (just run DFT, ~$10 compute)
            # False KILL is expensive (miss potential discovery, ~$10000 value)
            return cls(
                C_keep_fp=0.5,    # Low: just a wasted DFT
                C_kill_fn=10.0,   # High: missed discovery
                C_maybe=1.0,
                C_keep_tp=-2.0,   # Good: found a candidate
                C_kill_tn=-0.5,   # Small win: saved a DFT
            )
        elif mode == "experimental":
            # False KEEP is expensive (waste synthesis, ~$1000)
            # False KILL is expensive (miss discovery, ~$10000)
            return cls(
                C_keep_fp=5.0,    # High: wasted synthesis
                C_kill_fn=10.0,   # High: missed discovery
                C_maybe=1.0,
                C_keep_tp=-3.0,   # Very good: synthesize winner
                C_kill_tn=-0.2,   # Small win: avoided bad synthesis
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def expected_cost(
        self,
        decisions: np.ndarray,
        y_true: np.ndarray,
        thresh_stable: float = THRESH_STABLE,
    ) -> float:
        """
        Compute expected cost for a set of decisions.
        
        Args:
            decisions: [N] array of "KEEP"/"MAYBE"/"KILL"
            y_true: [N] ground truth E_hull
        
        Returns:
            Total expected cost
        """
        is_stable = y_true < thresh_stable
        
        keep_mask = decisions == "KEEP"
        kill_mask = decisions == "KILL"
        maybe_mask = decisions == "MAYBE"
        
        # True/False positives/negatives
        keep_tp = (keep_mask & is_stable).sum()      # Correct KEEP
        keep_fp = (keep_mask & ~is_stable).sum()     # False KEEP
        kill_tn = (kill_mask & ~is_stable).sum()     # Correct KILL
        kill_fn = (kill_mask & is_stable).sum()      # False KILL
        n_maybe = maybe_mask.sum()
        
        cost = (
            self.C_keep_tp * keep_tp +
            self.C_keep_fp * keep_fp +
            self.C_kill_tn * kill_tn +
            self.C_kill_fn * kill_fn +
            self.C_maybe * n_maybe
        )
        
        return float(cost)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PolicyThresholds:
    """Tunable thresholds for decision policy."""
    # Primary thresholds (quantile-based)
    T_keep: float = 0.05      # KEEP if q90_cal <= T_keep
    T_kill: float = 0.15      # KILL if q10_cal >= T_kill
    
    # Optional probability thresholds
    P_keep: Optional[float] = None   # AND p_stable >= P_keep
    P_kill: Optional[float] = None   # AND p_stable <= P_kill
    
    # OOD handling
    allow_keep_borderline: bool = True   # Allow KEEP when BORDERLINE?
    allow_keep_ood: bool = False         # Allow KEEP when OOD?
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "PolicyThresholds":
        return cls(**d)


@dataclass
class PolicyReport:
    """Report from policy evaluation or tuning."""
    mode: str
    thresholds: PolicyThresholds
    cost_model: CostModel
    
    # Outcomes
    n_total: int
    n_keep: int
    n_maybe: int
    n_kill: int
    
    # Error rates
    fn_rate: float          # P(KILL | stable) - missed discoveries
    fp_rate: float          # P(KEEP | unstable) - false keeps
    
    # Costs
    expected_cost: float
    cost_per_sample: float
    
    # Enrichment
    ef_at_k: Dict[int, float] = field(default_factory=dict)  # Enrichment factor
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = {
            "mode": self.mode,
            "thresholds": self.thresholds.to_dict(),
            "cost_model": self.cost_model.to_dict(),
            "n_total": self.n_total,
            "n_keep": self.n_keep,
            "n_maybe": self.n_maybe,
            "n_kill": self.n_kill,
            "fn_rate": self.fn_rate,
            "fp_rate": self.fp_rate,
            "expected_cost": self.expected_cost,
            "cost_per_sample": self.cost_per_sample,
            "ef_at_k": self.ef_at_k,
            "precision_at_k": self.precision_at_k,
        }
        return d
    
    def __str__(self) -> str:
        lines = [
            f"Policy Report ({self.mode})",
            "=" * 40,
            f"Thresholds: T_keep={self.thresholds.T_keep:.3f}, T_kill={self.thresholds.T_kill:.3f}",
            f"Decisions: KEEP={self.n_keep}, MAYBE={self.n_maybe}, KILL={self.n_kill}",
            f"FN rate (missed): {self.fn_rate:.2%}",
            f"FP rate (false keep): {self.fp_rate:.2%}",
            f"Expected cost: {self.expected_cost:.2f} ({self.cost_per_sample:.3f}/sample)",
        ]
        if self.ef_at_k:
            ef_str = ", ".join(f"EF@{k}={v:.2f}" for k, v in sorted(self.ef_at_k.items()))
            lines.append(f"Enrichment: {ef_str}")
        return "\n".join(lines)


class DecisionPolicy:
    """
    Utility-optimized decision policy.
    
    Makes KEEP/MAYBE/KILL decisions based on calibrated predictions
    and OOD gate level, optimized for a specific cost model.
    """
    
    def __init__(
        self,
        thresholds: PolicyThresholds,
        cost_model: CostModel,
        mode: str = "dft_followup",
    ):
        self.thresholds = thresholds
        self.cost_model = cost_model
        self.mode = mode
    
    @classmethod
    def for_mode(cls, mode: str) -> "DecisionPolicy":
        """Create policy with default thresholds for mode."""
        cost_model = CostModel.for_mode(mode)
        
        if mode == "dft_followup":
            # More permissive KEEP (cheap false positives)
            thresholds = PolicyThresholds(
                T_keep=0.06,    # Slightly above stable threshold
                T_kill=0.15,    # Conservative KILL
                P_keep=0.6,     # Moderate confidence required
                allow_keep_borderline=True,
            )
        else:  # experimental
            # Stricter KEEP (expensive false positives)
            thresholds = PolicyThresholds(
                T_keep=0.04,    # Below stable threshold
                T_kill=0.12,    # More aggressive KILL
                P_keep=0.8,     # High confidence required
                allow_keep_borderline=False,
            )
        
        return cls(thresholds, cost_model, mode)
    
    def decide(
        self,
        q10_cal: float,
        q50: float,
        q90_cal: float,
        p_stable: Optional[float] = None,
        gate_level: GateLevel = "IN",
    ) -> Decision:
        """
        Make decision for a single sample.
        
        Args:
            q10_cal: Calibrated lower bound (10th percentile)
            q50: Point estimate (median)
            q90_cal: Calibrated upper bound (90th percentile)
            p_stable: Probability of E_hull < 0.05 (optional)
            gate_level: OOD gate level ("IN", "BORDERLINE", "OOD")
        
        Returns:
            Decision: "KEEP", "MAYBE", or "KILL"
        """
        t = self.thresholds
        
        # Check KEEP conditions
        keep_quantile_ok = q90_cal <= t.T_keep
        keep_prob_ok = (t.P_keep is None) or (p_stable is not None and p_stable >= t.P_keep)
        
        # OOD check for KEEP
        if gate_level == "OOD":
            keep_ood_ok = t.allow_keep_ood
        elif gate_level == "BORDERLINE":
            keep_ood_ok = t.allow_keep_borderline
        else:
            keep_ood_ok = True
        
        if keep_quantile_ok and keep_prob_ok and keep_ood_ok:
            return "KEEP"
        
        # Check KILL conditions
        kill_quantile_ok = q10_cal >= t.T_kill
        kill_prob_ok = (t.P_kill is None) or (p_stable is not None and p_stable <= t.P_kill)
        
        if kill_quantile_ok and kill_prob_ok:
            return "KILL"
        
        # Default to MAYBE
        return "MAYBE"
    
    def decide_batch(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        p_stable: Optional[np.ndarray] = None,
        gate_levels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make decisions for a batch of samples."""
        n = len(q10_cal)
        decisions = []
        
        for i in range(n):
            ps = p_stable[i] if p_stable is not None else None
            gl = gate_levels[i] if gate_levels is not None else "IN"
            
            d = self.decide(q10_cal[i], q50[i], q90_cal[i], ps, gl)
            decisions.append(d)
        
        return np.array(decisions)
    
    def evaluate(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        y_true: np.ndarray,
        p_stable: Optional[np.ndarray] = None,
        gate_levels: Optional[np.ndarray] = None,
    ) -> PolicyReport:
        """
        Evaluate policy on a dataset with ground truth.
        
        Returns:
            PolicyReport with metrics
        """
        decisions = self.decide_batch(q10_cal, q50, q90_cal, p_stable, gate_levels)
        
        is_stable = y_true < THRESH_STABLE
        n_stable = is_stable.sum()
        n_unstable = (~is_stable).sum()
        
        keep_mask = decisions == "KEEP"
        kill_mask = decisions == "KILL"
        maybe_mask = decisions == "MAYBE"
        
        # Error rates
        fn_rate = (kill_mask & is_stable).sum() / max(n_stable, 1)
        fp_rate = (keep_mask & ~is_stable).sum() / max(keep_mask.sum(), 1)
        
        # Cost
        expected_cost = self.cost_model.expected_cost(decisions, y_true)
        
        # Enrichment factor at K
        # EF@K = (fraction stable in top K by q90) / (fraction stable overall)
        base_rate = n_stable / len(y_true)
        ef_at_k = {}
        precision_at_k = {}
        
        sorted_idx = np.argsort(q90_cal)  # Sort by q90 ascending (best first)
        for k in [10, 25, 50, 100]:
            if k <= len(y_true):
                top_k_idx = sorted_idx[:k]
                prec_k = is_stable[top_k_idx].mean()
                ef_k = prec_k / base_rate if base_rate > 0 else 0
                ef_at_k[k] = float(ef_k)
                precision_at_k[k] = float(prec_k)
        
        return PolicyReport(
            mode=self.mode,
            thresholds=self.thresholds,
            cost_model=self.cost_model,
            n_total=len(y_true),
            n_keep=int(keep_mask.sum()),
            n_maybe=int(maybe_mask.sum()),
            n_kill=int(kill_mask.sum()),
            fn_rate=float(fn_rate),
            fp_rate=float(fp_rate),
            expected_cost=expected_cost,
            cost_per_sample=expected_cost / len(y_true),
            ef_at_k=ef_at_k,
            precision_at_k=precision_at_k,
        )
    
    def save(self, path: Path) -> None:
        """Save policy to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "mode": self.mode,
            "thresholds": self.thresholds.to_dict(),
            "cost_model": self.cost_model.to_dict(),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "DecisionPolicy":
        """Load policy from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        thresholds = PolicyThresholds.from_dict(data["thresholds"])
        cost_model = CostModel(**data["cost_model"])
        
        return cls(thresholds, cost_model, data["mode"])


def tune_policy_thresholds(
    q10_cal: np.ndarray,
    q50: np.ndarray,
    q90_cal: np.ndarray,
    y_true: np.ndarray,
    mode: str = "dft_followup",
    p_stable: Optional[np.ndarray] = None,
    gate_levels: Optional[np.ndarray] = None,
    T_keep_range: Tuple[float, float] = (0.02, 0.10),
    T_kill_range: Tuple[float, float] = (0.08, 0.20),
    P_keep_range: Optional[Tuple[float, float]] = (0.5, 0.9),
    n_grid: int = 20,
    max_fn_rate: Optional[float] = None,
) -> Tuple[DecisionPolicy, PolicyReport]:
    """
    Tune policy thresholds to minimize expected cost on validation data.
    
    Grid search over (T_keep, T_kill, P_keep) combinations.
    
    Args:
        q10_cal, q50, q90_cal: Calibrated predictions
        y_true: Ground truth E_hull
        mode: Operating mode ("dft_followup" or "experimental")
        p_stable: Optional stability probabilities
        gate_levels: Optional OOD gate levels
        T_keep_range: Range for T_keep threshold (default: 0.02 to 0.10)
        T_kill_range: Range for T_kill threshold (default: 0.08 to 0.20)
        P_keep_range: Range for P_keep threshold (None to disable)
        n_grid: Number of grid points per dimension
        max_fn_rate: Max allowed FN rate (default: 0.10 for dft, 0.05 for experimental)
    
    Returns:
        Tuple of (best_policy, best_report)
    """
    cost_model = CostModel.for_mode(mode)
    
    # Default max FN rate based on mode
    if max_fn_rate is None:
        max_fn_rate = 0.10 if mode == "dft_followup" else 0.05
    
    # Generate grid
    T_keep_vals = np.linspace(T_keep_range[0], T_keep_range[1], n_grid)
    T_kill_vals = np.linspace(T_kill_range[0], T_kill_range[1], n_grid)
    
    if P_keep_range is not None and p_stable is not None:
        P_keep_vals = np.linspace(P_keep_range[0], P_keep_range[1], n_grid // 2)
    else:
        P_keep_vals = [None]
    
    best_cost = float("inf")
    best_policy = None
    best_report = None
    
    # Default OOD settings based on mode
    allow_borderline = (mode == "dft_followup")
    
    for T_keep in T_keep_vals:
        for T_kill in T_kill_vals:
            # Skip invalid combinations
            if T_keep >= T_kill:
                continue
            
            for P_keep in P_keep_vals:
                thresholds = PolicyThresholds(
                    T_keep=float(T_keep),
                    T_kill=float(T_kill),
                    P_keep=float(P_keep) if P_keep is not None else None,
                    allow_keep_borderline=allow_borderline,
                    allow_keep_ood=False,
                )
                
                policy = DecisionPolicy(thresholds, cost_model, mode)
                report = policy.evaluate(
                    q10_cal, q50, q90_cal, y_true, p_stable, gate_levels
                )
                
                # Constraint: FN rate must be below threshold
                if report.fn_rate > max_fn_rate:
                    continue
                
                if report.expected_cost < best_cost:
                    best_cost = report.expected_cost
                    best_policy = policy
                    best_report = report
    
    if best_policy is None:
        # Fallback to default
        print("Warning: No valid policy found, using defaults")
        best_policy = DecisionPolicy.for_mode(mode)
        best_report = best_policy.evaluate(
            q10_cal, q50, q90_cal, y_true, p_stable, gate_levels
        )
    
    return best_policy, best_report


def save_policy_report(report: PolicyReport, path: Path) -> None:
    """Save policy report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
