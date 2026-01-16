from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque, Dict


class RateLimiter:
    """Simple sliding-window rate limiter (in-memory, per process)."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, key: str, cost: int = 1) -> bool:
        if self.max_requests <= 0:
            return True
        cost = max(int(cost), 1)
        now = time.time()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) + cost > self.max_requests:
                return False
            hits.extend([now] * cost)
            return True
