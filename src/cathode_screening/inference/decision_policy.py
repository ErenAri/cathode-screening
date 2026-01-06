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
    """
    Tunable thresholds for two-layer decision policy.
    
    Layer 1 (Ranking): priority_score = q50 + lambda_rank * epistemic_std
                       Lower score = higher priority in DFT queue
    
    Layer 2 (Safety):  KILL only if:
                         q10_cal >= T_kill
                         AND epistemic_std <= sigma_max (ensemble agrees)
                         AND gate_level == "IN" (in-distribution)
                         AND calibrated_source == "cluster" (if require_cluster_for_kill)
    
    Minimal False KILL Rule:
        - T_kill = 0.12 (conservative: 2.4× stability threshold)
        - sigma_max = P20(epistemic_std | IN, val) ≈ 0.03
        - require_cluster_for_kill = True (only KILL cluster-calibrated samples)
        - Expected: <0.1% false KILL rate
    """
    # Primary thresholds (quantile-based)
    T_keep: float = 0.05      # KEEP if q90_cal <= T_keep (disabled in DFT mode)
    T_kill: float = 0.12      # KILL if q10_cal >= T_kill (conservative: 2.4x stable threshold)
    
    # Epistemic uncertainty threshold for KILL
    # Only KILL when ensemble agrees (low uncertainty)
    # Set via compute_adaptive_sigma_max() with percentile=20
    sigma_max: float = 0.03   # KILL requires epistemic_std <= sigma_max (P20 default)
    sigma_max_percentile: int = 20  # Percentile of IN-distribution for sigma_max
    
    # Ranking parameters
    # ranking_score = q50 + lambda_rank * epistemic_std
    lambda_rank: float = 0.5  # Uncertainty penalty in ranking score
    
    # Optional probability thresholds
    P_keep: Optional[float] = None   # AND p_stable >= P_keep
    P_kill: Optional[float] = None   # AND p_stable <= P_kill
    
    # OOD handling for KEEP decisions
    allow_keep_borderline: bool = True   # Allow KEEP when BORDERLINE?
    allow_keep_ood: bool = False         # Allow KEEP when OOD?
    
    # OOD handling for KILL decisions (safety overlay)
    # Default: KILL only when gate_level == "IN" (strictest safety)
    block_kill_ood: bool = True          # Block KILL when OOD?
    block_kill_borderline: bool = True   # Block KILL when BORDERLINE?
    
    # Calibration source requirements for KILL (minimal false KILL)
    # If True, only allow KILL for cluster-calibrated samples
    require_cluster_for_kill: bool = True
    
    # Global calibration fallback strictness multipliers
    # When calibrated_source=="global" AND require_cluster_for_kill==False
    global_keep_factor: float = 0.8    # T_keep_effective = T_keep * factor
    global_sigma_factor: float = 0.5   # sigma_max_effective = sigma_max * factor (stricter)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "PolicyThresholds":
        # Handle missing keys for backward compatibility
        defaults = {
            'T_kill': 0.12,
            'sigma_max': 0.03,
            'sigma_max_percentile': 20,
            'lambda_rank': 0.5,
            'block_kill_ood': True,
            'block_kill_borderline': True,
            'require_cluster_for_kill': True,
            'global_keep_factor': 0.8,
            'global_sigma_factor': 0.5,
        }
        for k, v in defaults.items():
            if k not in d:
                d[k] = v
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
        """
        Create policy with default thresholds for mode.
        
        DFT-followup mode (Minimal False KILL):
          - No auto-KEEP (all materials go to ranked queue for DFT validation)
          - Conservative KILL: q10_cal >= 0.12 AND epistemic_std <= 0.03 (P20)
                              AND gate == IN AND calibrated_source == cluster
          - Ranking: priority_score = q50 + 0.5 * epistemic_std
          - Expected false KILL: <0.1%
        
        Experimental mode:
          - Stricter KEEP allowed (high confidence only)
          - More aggressive KILL
        """
        cost_model = CostModel.for_mode(mode)
        
        if mode == "dft_followup":
            # Two-layer policy: ranking + safety overlay
            # No auto-KEEP in DFT mode (T_keep=0 disables it)
            # Minimal False KILL: require cluster calibration + P20 sigma
            thresholds = PolicyThresholds(
                T_keep=0.0,       # Disabled: no auto-KEEP in DFT mode
                T_kill=0.12,      # Conservative: 2.4x stability threshold
                sigma_max=0.03,   # P20 of IN-distribution (stricter)
                sigma_max_percentile=20,
                lambda_rank=0.5,  # Mild uncertainty penalty in ranking
                P_keep=None,      # Not used (no KEEP)
                allow_keep_borderline=True,
                allow_keep_ood=False,
                block_kill_ood=True,           # Don't auto-kill novel materials
                block_kill_borderline=True,    # Don't auto-kill borderline
                require_cluster_for_kill=True, # Only KILL cluster-calibrated samples
            )
        else:  # experimental
            # Stricter KEEP (expensive false positives)
            thresholds = PolicyThresholds(
                T_keep=0.04,      # Below stable threshold
                T_kill=0.12,      # More aggressive KILL
                sigma_max=0.03,   # Stricter epistemic requirement
                sigma_max_percentile=20,
                lambda_rank=1.0,  # Higher uncertainty penalty
                P_keep=0.8,       # High confidence required for KEEP
                allow_keep_borderline=False,
                allow_keep_ood=False,
                block_kill_ood=True,
                block_kill_borderline=True,  # KILL only for IN samples
            )
        
        return cls(thresholds, cost_model, mode)
    
    def decide(
        self,
        q10_cal: float,
        q50: float,
        q90_cal: float,
        p_stable: Optional[float] = None,
        gate_level: GateLevel = "IN",
        epistemic_std: Optional[float] = None,
        calibrated_source: str = "global",
        is_cluster_calibrated: bool = False,
    ) -> Decision:
        """
        Make decision for a single sample (two-layer policy).
        
        Layer 1 (Ranking): All non-KILL materials go to ranked queue
                           Priority score = q50 + lambda * epistemic_std
        
        Layer 2 (Safety):  KILL if q10_cal >= T_kill 
                                AND epistemic_std < sigma_max (ensemble agrees)
                                AND gate_level != OOD (in-distribution)
        
        Calibration Source Handling:
            - If calibrated_source=="global" AND gate_level!="IN": force MAYBE
            - If calibrated_source=="global": stricter KEEP/KILL thresholds
        
        Args:
            q10_cal: Calibrated lower bound (10th percentile)
            q50: Point estimate (median)
            q90_cal: Calibrated upper bound (90th percentile)
            p_stable: Probability of E_hull < 0.05 (optional)
            gate_level: OOD gate level ("IN", "BORDERLINE", "OOD")
            epistemic_std: Ensemble disagreement (optional but recommended)
            calibrated_source: "cluster" or "global" calibration used
            is_cluster_calibrated: Whether per-cluster calibration was available
        
        Returns:
            Decision: "KEEP", "MAYBE", or "KILL"
        """
        t = self.thresholds
        
        # ================================================================
        # CALIBRATION SOURCE CHECK - Force MAYBE for low-confidence cases
        # ================================================================
        # Minimal false KILL: require cluster calibration for KILL
        if t.require_cluster_for_kill and not is_cluster_calibrated:
            # Global-calibrated samples cannot be KILLed (safety)
            # They go to MAYBE → DFT queue
            pass  # Continue to check KEEP, but KILL will be blocked below
        
        # Global calibration with non-IN gate: insufficient confidence
        if calibrated_source == "global" and gate_level != "IN":
            return "MAYBE"
        
        # Compute effective thresholds based on calibration source
        # Global fallback → stricter requirements (if KILL allowed at all)
        if is_cluster_calibrated:
            effective_sigma_max = t.sigma_max
            effective_T_keep = t.T_keep
        else:
            effective_sigma_max = t.sigma_max * t.global_sigma_factor
            effective_T_keep = t.T_keep * t.global_keep_factor
        
        # ================================================================
        # LAYER 2: SAFETY OVERLAY - Check KILL first (hard gate)
        # ================================================================
        # Minimal False KILL Rule - KILL requires ALL of:
        #   1. q10_cal >= T_kill (even optimistic bound says unstable)
        #   2. epistemic_std <= sigma_max (ensemble strongly agrees, P20)
        #   3. gate_level == "IN" (strictly in-distribution only)
        #   4. calibrated_source == "cluster" (if require_cluster_for_kill)
        
        kill_quantile_ok = q10_cal >= t.T_kill
        
        # Epistemic check: only kill if ensemble agrees (low uncertainty)
        # Use effective_sigma_max (stricter for global calibration)
        if epistemic_std is not None:
            kill_epistemic_ok = epistemic_std <= effective_sigma_max
        else:
            kill_epistemic_ok = False  # Require epistemic info for KILL (safety)
        
        # Gate check: KILL only for IN samples (strictest safety)
        # OOD or BORDERLINE → always MAYBE (never auto-kill uncertain cases)
        if gate_level == "OOD" and t.block_kill_ood:
            kill_gate_ok = False
        elif gate_level == "BORDERLINE" and t.block_kill_borderline:
            kill_gate_ok = False
        else:
            kill_gate_ok = True
        
        # Calibration source check: require cluster calibration for KILL
        if t.require_cluster_for_kill:
            kill_calibration_ok = is_cluster_calibrated
        else:
            kill_calibration_ok = True
        
        # Probability check (optional)
        kill_prob_ok = (t.P_kill is None) or (p_stable is not None and p_stable <= t.P_kill)
        
        if kill_quantile_ok and kill_epistemic_ok and kill_gate_ok and kill_calibration_ok and kill_prob_ok:
            return "KILL"
        
        # ================================================================
        # LAYER 1: Check KEEP (only if T_keep > 0, disabled in DFT mode)
        # ================================================================
        # Use effective_T_keep (stricter for global calibration)
        if effective_T_keep > 0:
            keep_quantile_ok = q90_cal <= effective_T_keep
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
        
        # ================================================================
        # Default: MAYBE (goes to ranked DFT queue)
        # ================================================================
        return "MAYBE"
    
    def decide_batch(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        p_stable: Optional[np.ndarray] = None,
        gate_levels: Optional[np.ndarray] = None,
        epistemic_std: Optional[np.ndarray] = None,
        calibrated_sources: Optional[List[str]] = None,
        is_cluster_calibrated: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make decisions for a batch of samples."""
        n = len(q10_cal)
        decisions = []
        
        for i in range(n):
            ps = p_stable[i] if p_stable is not None else None
            gl = gate_levels[i] if gate_levels is not None else "IN"
            es = epistemic_std[i] if epistemic_std is not None else None
            cs = calibrated_sources[i] if calibrated_sources is not None else "global"
            ic = is_cluster_calibrated[i] if is_cluster_calibrated is not None else False
            
            d = self.decide(q10_cal[i], q50[i], q90_cal[i], ps, gl, es, cs, ic)
            decisions.append(d)
        
        return np.array(decisions)
    
    def compute_ranking_scores(
        self,
        q50: np.ndarray,
        epistemic_std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ranking scores for DFT queue prioritization.
        
        Score = q50 + lambda_rank * epistemic_std
        Lower score = higher priority (more likely stable + more confident)
        
        Args:
            q50: Point estimates of E_hull
            epistemic_std: Ensemble disagreement (optional)
        
        Returns:
            Ranking scores (lower = higher priority)
        """
        if epistemic_std is not None:
            return q50 + self.thresholds.lambda_rank * epistemic_std
        else:
            return q50.copy()
    
    def evaluate(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        y_true: np.ndarray,
        p_stable: Optional[np.ndarray] = None,
        gate_levels: Optional[np.ndarray] = None,
        epistemic_std: Optional[np.ndarray] = None,
        calibrated_sources: Optional[List[str]] = None,
        is_cluster_calibrated: Optional[np.ndarray] = None,
    ) -> PolicyReport:
        """
        Evaluate policy on a dataset with ground truth.
        
        Returns:
            PolicyReport with metrics
        """
        decisions = self.decide_batch(
            q10_cal, q50, q90_cal, p_stable, gate_levels, epistemic_std,
            calibrated_sources, is_cluster_calibrated
        )
        
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
        # EF@K = (fraction stable in top K by q50) / (fraction stable overall)
        # IMPORTANT: Use q50 (point estimate), NOT q90_cal (interval bound)
        # q90_cal selects narrow intervals, q50 selects low predicted E_hull
        base_rate = n_stable / len(y_true)
        ef_at_k = {}
        precision_at_k = {}
        
        sorted_idx = np.argsort(q50)  # Sort by q50 ascending (best = lowest E_hull)
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


def compute_adaptive_sigma_max(
    epistemic_std: np.ndarray,
    gate_levels: Optional[np.ndarray] = None,
    percentile: float = 20.0,
) -> float:
    """
    Compute adaptive sigma_max from percentile of IN-distribution epistemic uncertainty.
    
    Using P20 means we only KILL when the model is in its most confident quintile.
    This adapts to the ensemble's natural uncertainty scale rather than using
    an arbitrary absolute threshold.
    
    Minimal False KILL Rule:
        - P20 is stricter than P25, reducing false KILL rate
        - Only the lowest 20% uncertainty samples can be KILLed
        - Typical value: ~0.025-0.035 eV
    
    Args:
        epistemic_std: Ensemble disagreement (std of predictions)
        gate_levels: OOD gate levels ("IN", "BORDERLINE", "OOD")
        percentile: Percentile to use (default: 20th = most confident quintile)
    
    Returns:
        Adaptive sigma_max threshold
    """
    if gate_levels is not None:
        # Only use IN-distribution samples for calibration
        in_mask = gate_levels == "IN"
        if in_mask.sum() > 10:
            return float(np.percentile(epistemic_std[in_mask], percentile))
    
    # Fallback: use all samples
    return float(np.percentile(epistemic_std, percentile))


def tune_policy_thresholds(
    q10_cal: np.ndarray,
    q50: np.ndarray,
    q90_cal: np.ndarray,
    y_true: np.ndarray,
    mode: str = "dft_followup",
    p_stable: Optional[np.ndarray] = None,
    gate_levels: Optional[np.ndarray] = None,
    epistemic_std: Optional[np.ndarray] = None,
    calibrated_sources: Optional[List[str]] = None,
    is_cluster_calibrated: Optional[np.ndarray] = None,
    T_keep_range: Tuple[float, float] = (0.05, 0.30),
    T_kill_range: Tuple[float, float] = (0.10, 0.15),
    sigma_max_range: Tuple[float, float] = (0.02, 0.05),
    sigma_max_percentile: Optional[float] = 20.0,
    P_keep_range: Optional[Tuple[float, float]] = None,
    n_grid: int = 15,
    max_fn_rate: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[DecisionPolicy, PolicyReport]:
    """
    Tune two-layer policy thresholds to minimize expected cost.
    
    Grid search over (T_keep, T_kill, sigma_max) combinations.
    
    For DFT mode: T_keep=0 by default (no auto-KEEP), focuses on T_kill and sigma_max.
    For experimental mode: T_keep grid is denser in [0.10, 0.25] range.
    
    sigma_max can be set adaptively using percentile of IN-distribution epistemic_std,
    which adapts to the ensemble's natural uncertainty scale.
    
    Args:
        q10_cal, q50, q90_cal: Calibrated predictions
        y_true: Ground truth E_hull
        mode: Operating mode ("dft_followup" or "experimental")
        p_stable: Optional stability probabilities
        gate_levels: Optional OOD gate levels
        epistemic_std: Ensemble disagreement (recommended for KILL safety)
        T_keep_range: Range for T_keep threshold (default: 0.05 to 0.30)
        T_kill_range: Range for T_kill threshold (default: 0.08 to 0.15)
        sigma_max_range: Range for epistemic threshold (default: 0.03 to 0.08)
        sigma_max_percentile: If set, compute sigma_max as this percentile of
                              IN-distribution epistemic_std (default: 25 = most confident quartile).
                              Set to None to use grid search over sigma_max_range.
        P_keep_range: Range for P_keep threshold (None to disable)
        n_grid: Number of grid points per dimension
        max_fn_rate: Max allowed FN rate (default: 0.005 for dft, 0.01 for experimental)
        verbose: If True, log KEEP/MAYBE/KILL counts per grid point
    
    Returns:
        Tuple of (best_policy, best_report)
    """
    cost_model = CostModel.for_mode(mode)
    
    # Default max FN rate: < 0.5% for DFT mode (key constraint!)
    if max_fn_rate is None:
        max_fn_rate = 0.005 if mode == "dft_followup" else 0.01
    
    # Compute adaptive sigma_max if percentile specified and epistemic_std available
    if sigma_max_percentile is not None and epistemic_std is not None:
        adaptive_sigma = compute_adaptive_sigma_max(epistemic_std, gate_levels, sigma_max_percentile)
        sigma_max_vals = [adaptive_sigma]
        if verbose:
            print(f"Using adaptive sigma_max = {adaptive_sigma:.4f} (P{sigma_max_percentile:.0f} of IN-distribution)")
    else:
        sigma_max_vals = np.linspace(sigma_max_range[0], sigma_max_range[1], n_grid // 2)
    
    # Generate grid with denser sampling in the critical T_keep range [0.10, 0.25]
    if mode == "dft_followup":
        # In DFT mode, T_keep=0 (disabled), only tune T_kill and sigma_max
        T_keep_vals = [0.0]
    else:
        # Build T_keep grid: sparse outside [0.10, 0.25], dense inside
        lo, hi = T_keep_range
        # Points below 0.10
        below_dense = np.linspace(lo, min(0.10, hi), max(3, n_grid // 5), endpoint=False)
        # Dense region [0.10, 0.25]
        dense_lo = max(lo, 0.10)
        dense_hi = min(hi, 0.25)
        dense_points = np.linspace(dense_lo, dense_hi, max(5, n_grid // 2))
        # Points above 0.25
        above_dense = np.linspace(max(0.25, lo), hi, max(3, n_grid // 5))[1:]  # skip first (overlap)
        
        T_keep_vals = np.unique(np.concatenate([below_dense, dense_points, above_dense]))
        T_keep_vals = T_keep_vals[(T_keep_vals >= lo) & (T_keep_vals <= hi)]
    
    T_kill_vals = np.linspace(T_kill_range[0], T_kill_range[1], n_grid)
    
    # sigma_max_vals already set above (either adaptive or grid)
    
    if P_keep_range is not None and p_stable is not None:
        P_keep_vals = np.linspace(P_keep_range[0], P_keep_range[1], n_grid // 2)
    else:
        P_keep_vals = [None]
    
    best_cost = float("inf")
    best_policy = None
    best_report = None
    
    # Default OOD settings based on mode
    allow_borderline = (mode == "dft_followup")
    
    # Track if any grid point yields KEEP > 0
    any_keep_found = False
    grid_results = []
    
    for T_keep in T_keep_vals:
        for T_kill in T_kill_vals:
            # Skip invalid: T_keep must be < T_kill (unless T_keep disabled)
            if T_keep > 0 and T_keep >= T_kill:
                continue
            
            for sigma_max in sigma_max_vals:
                for P_keep in P_keep_vals:
                    thresholds = PolicyThresholds(
                        T_keep=float(T_keep),
                        T_kill=float(T_kill),
                        sigma_max=float(sigma_max),
                        lambda_rank=0.5,
                        P_keep=float(P_keep) if P_keep is not None else None,
                        allow_keep_borderline=allow_borderline,
                        allow_keep_ood=False,
                        block_kill_ood=True,
                        block_kill_borderline=True,  # KILL only for gate_level=="IN"
                        require_cluster_for_kill=True,  # Minimal false KILL: cluster-only
                    )
                    
                    policy = DecisionPolicy(thresholds, cost_model, mode)
                    report = policy.evaluate(
                        q10_cal, q50, q90_cal, y_true, p_stable, gate_levels, epistemic_std,
                        calibrated_sources, is_cluster_calibrated
                    )
                    
                    # Track KEEP presence
                    if report.n_keep > 0:
                        any_keep_found = True
                    
                    # Verbose logging
                    if verbose:
                        grid_results.append({
                            "T_keep": T_keep,
                            "T_kill": T_kill,
                            "sigma_max": sigma_max,
                            "n_keep": report.n_keep,
                            "n_maybe": report.n_maybe,
                            "n_kill": report.n_kill,
                            "fn_rate": report.fn_rate,
                            "cost": report.expected_cost,
                        })
                    
                    # Hard constraint: FN rate must be below threshold
                    if report.fn_rate > max_fn_rate:
                        continue
                    
                    if report.expected_cost < best_cost:
                        best_cost = report.expected_cost
                        best_policy = policy
                        best_report = report
    
    # Sanity check: warn if no KEEP decisions at any grid point
    if mode != "dft_followup" and not any_keep_found:
        print(f"\\n*** WARNING: No KEEP decisions at any of {len(T_keep_vals)} T_keep values ***")
        print(f"    T_keep grid: {T_keep_vals[:5]}... (min q90_cal in data may be higher)")
        print(f"    Consider increasing T_keep_range upper bound")
    
    # Print verbose summary
    if verbose and grid_results:
        # Show sample of grid results
        print(f"\\nGrid search summary ({len(grid_results)} configurations evaluated):")
        # Find configs with most KEEP
        sorted_by_keep = sorted(grid_results, key=lambda x: -x["n_keep"])[:5]
        print("Top 5 by KEEP count:")
        for r in sorted_by_keep:
            print(f"  T_keep={r['T_keep']:.3f}, T_kill={r['T_kill']:.3f}: "
                  f"KEEP={r['n_keep']}, MAYBE={r['n_maybe']}, KILL={r['n_kill']}, "
                  f"FN={r['fn_rate']:.2%}, cost={r['cost']:.1f}")
    
    if best_policy is None:
        # Fallback to default (may violate FN constraint)
        print(f"Warning: No policy achieves FN rate < {max_fn_rate:.2%}, using defaults")
        best_policy = DecisionPolicy.for_mode(mode)
        best_report = best_policy.evaluate(
            q10_cal, q50, q90_cal, y_true, p_stable, gate_levels, epistemic_std,
            calibrated_sources, is_cluster_calibrated
        )
    
    return best_policy, best_report


def save_policy_report(report: PolicyReport, path: Path) -> None:
    """Save policy report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
