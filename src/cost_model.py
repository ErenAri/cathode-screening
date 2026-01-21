from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple

@dataclass
class CostModel:
    """
    Economic reasoning for discovery campaigns.
    
    Model: Utility v0 (Yield vs Cost only)
    Utility(K) = Value * Expected_Yield(K) - Cost * K
    
    Future v1 will include miss-risk penalties (Recall).
    """
    
    # User-editable parameters
    value_per_stable: float = 100.0    # Value of one stable structure
    cost_per_dft: float = 10.0          # Cost of one DFT calculation
    
    # Derived/Fitted parameters
    prevalence: Optional[float] = None
    precision_curve: Optional[Callable[[int], float]] = None
    
    @classmethod
    def from_calibration(cls, labels: List[bool], **kwargs):
        """Initialize from calibration data (sets prevalence)."""
        if not labels:
            raise ValueError("Empty labels list")
        prevalence = sum(labels) / len(labels)
        return cls(prevalence=prevalence, **kwargs)
        
    def estimate_positives(self, search_space_size: int) -> int:
        """Estimate total number of stable structures in search space."""
        if self.prevalence is None:
            raise ValueError("Prevalence not set")
        return int(self.prevalence * search_space_size)
    
    def utility(self, K: int) -> float:
        """
        Calculate Utility at budget K.
        Utility = Value * (Precision@K * K) - Cost * K
        """
        if K <= 0: return 0.0
        
        # Use curve if available, else worst-case (random prevalence)
        # In a real engine, precision_curve would come from validation set
        p_k = self.precision_curve(K) if self.precision_curve else self.prevalence
        if p_k is None: p_k = 0.0 # Should not happen if init correctly
        
        expected_yield = p_k * K
        return self.value_per_stable * expected_yield - self.cost_per_dft * K
        
    def optimal_budget(self, max_K: int) -> int:
        """
        Find budget K that maximizes Utility.
        argmax_{K in 1..max_K} Utility(K)
        """
        if max_K <= 0: return 0
        
        best_k = 1
        best_u = self.utility(1)
        
        for k in range(2, max_K + 1):
            u = self.utility(k)
            if u > best_u:
                best_u = u
                best_k = k
                
        # If even best utility is negative, should we return 0? 
        # For now assume user wants to do *something* if max_K > 0, 
        # or we could return 0 if best_u < 0. Let's return best_k.
        return best_k
    
    def summary(self, K: int) -> str:
        """Human-readable summary of expected return."""
        p_k = self.precision_curve(K) if self.precision_curve else self.prevalence
        exp_yield = p_k * K
        cost = self.cost_per_dft * K
        return f"Budget: {K} | Expected Yield: {exp_yield:.1f} | Cost: ${cost:.0f} | ROI: {((exp_yield*self.value_per_stable)/cost - 1)*100:.0f}%"
