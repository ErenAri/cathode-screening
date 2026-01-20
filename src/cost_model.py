"""
Utility model v0 for discovery decisioning.

Provides expected utility for KEEP / MAYBE / KILL based on p_stable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityV0:
    value_keep: float = 10.0
    value_kill: float = 1.0
    cost_false_keep: float = 2.0
    cost_miss: float = 20.0
    cost_review: float = 1.0

    def expected_utility(self, p_stable: float, action: str) -> float:
        p = max(0.0, min(1.0, float(p_stable)))
        p_unstable = 1.0 - p

        if action == "KEEP":
            return p * self.value_keep - p_unstable * self.cost_false_keep
        if action == "KILL":
            return -p * self.cost_miss + p_unstable * self.value_kill
        if action == "MAYBE":
            return -self.cost_review

        raise ValueError(f"Unknown action: {action}")
