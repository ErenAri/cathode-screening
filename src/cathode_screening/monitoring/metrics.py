from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value

    @property
    def variance(self) -> Optional[float]:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> Optional[float]:
        var = self.variance
        if var is None:
            return None
        return var ** 0.5

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "count": self.count,
            "mean": self.mean if self.count else None,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
        }


class MetricsCollector:
    """In-memory metrics collector for prediction monitoring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self.decision_counts: Dict[str, int] = {"KEEP": 0, "MAYBE": 0, "KILL": 0}
        self.mode_counts: Dict[str, int] = {}
        self.ood_flag_count = 0
        self.latency_stats = RunningStats()
        self.ood_score_stats = RunningStats()
        self.epistemic_stats = RunningStats()
        self.ehull_stats = RunningStats()

    def record_request(self, count: int = 1) -> None:
        count = max(int(count), 0)
        if count == 0:
            return
        with self._lock:
            self.request_count += count

    def record_error(self, count: int = 1) -> None:
        count = max(int(count), 0)
        if count == 0:
            return
        with self._lock:
            self.error_count += count

    def record_prediction(
        self,
        decision: str,
        decision_mode: Optional[str] = None,
        ood_flag: Optional[bool] = None,
        ood_score: Optional[float] = None,
        uncertainty_epistemic: Optional[float] = None,
        ehull_pred: Optional[float] = None,
        latency_s: Optional[float] = None,
    ) -> None:
        with self._lock:
            if decision in self.decision_counts:
                self.decision_counts[decision] += 1
            else:
                self.decision_counts[decision] = 1

            if decision_mode:
                self.mode_counts[decision_mode] = self.mode_counts.get(decision_mode, 0) + 1

            if ood_flag:
                self.ood_flag_count += 1

            if ood_score is not None:
                self.ood_score_stats.update(float(ood_score))

            if uncertainty_epistemic is not None:
                self.epistemic_stats.update(float(uncertainty_epistemic))

            if ehull_pred is not None:
                self.ehull_stats.update(float(ehull_pred))

            if latency_s is not None:
                self.latency_stats.update(float(latency_s))

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            uptime = time.time() - self.started_at
            return {
                "uptime_s": uptime,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "decision_counts": dict(self.decision_counts),
                "mode_counts": dict(self.mode_counts),
                "ood_flag_count": self.ood_flag_count,
                "latency_s": self.latency_stats.to_dict(),
                "ood_score": self.ood_score_stats.to_dict(),
                "uncertainty_epistemic": self.epistemic_stats.to_dict(),
                "ehull_pred": self.ehull_stats.to_dict(),
            }
