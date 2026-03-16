"""
GPU-aware dynamic batcher for ensemble inference.

Accumulates incoming prediction requests and dispatches them as
optimally-sized batches to minimize GPU idle time while keeping
latency bounded.

Design:
    - Incoming requests are queued via `submit()`
    - A background thread flushes the queue when batch is full or timeout expires
    - Each flush runs all ensemble members on the full batch in one forward pass
    - Results are distributed back to waiting callers via futures

Configuration:
    CATHODE_BATCH_SIZE=32          Max structures per batch
    CATHODE_BATCH_TIMEOUT_MS=200   Max wait before flushing partial batch
    CATHODE_DEVICE=cuda             GPU device
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class _BatchItem:
    """A single structure waiting for inference."""
    structure: Structure
    material_id: str
    formula: str
    future: Future
    submitted_at: float


class DynamicBatcher:
    """Accumulates structures and runs batched GPU inference.

    Thread-safe: multiple API handlers can call `submit()` concurrently.
    A single background thread handles batch formation and dispatch.
    """

    def __init__(
        self,
        predictor,
        max_batch_size: int = 32,
        max_wait_ms: float = 200.0,
        device: str = "cuda",
    ):
        self.predictor = predictor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device

        self._queue: List[_BatchItem] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._shutdown = False

        # Stats
        self._batches_dispatched = 0
        self._total_items = 0

        # Start background flush thread
        self._thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="gpu-batcher",
        )
        self._thread.start()
        logger.info(
            "DynamicBatcher started: max_batch=%d, timeout=%dms, device=%s",
            max_batch_size, max_wait_ms, device,
        )

    def submit(
        self,
        structure: Structure,
        material_id: str,
        formula: Optional[str] = None,
    ) -> Future:
        """Submit a structure for batched inference.

        Returns a Future that resolves to a DecisionOutput.
        """
        future: Future = Future()
        item = _BatchItem(
            structure=structure,
            material_id=material_id,
            formula=formula or structure.composition.reduced_formula,
            future=future,
            submitted_at=time.perf_counter(),
        )

        with self._lock:
            self._queue.append(item)
            if len(self._queue) >= self.max_batch_size:
                self._event.set()  # Wake up flush thread immediately

        return future

    def _flush_loop(self) -> None:
        """Background loop that flushes accumulated batches."""
        while not self._shutdown:
            # Wait for batch full or timeout
            self._event.wait(timeout=self.max_wait_ms / 1000.0)
            self._event.clear()

            # Grab current queue
            with self._lock:
                if not self._queue:
                    continue
                batch = self._queue[: self.max_batch_size]
                self._queue = self._queue[self.max_batch_size:]

            # Process batch
            self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: List[_BatchItem]) -> None:
        """Run inference on a batch and resolve futures."""
        start = time.perf_counter()
        structures = [item.structure for item in batch]
        material_ids = [item.material_id for item in batch]
        formulas = [item.formula for item in batch]

        try:
            results = self.predictor.predict_structures(
                structures,
                material_ids=material_ids,
                formulas=formulas,
            )

            for item, result in zip(batch, results):
                item.future.set_result(result)

            latency = (time.perf_counter() - start) * 1000
            self._batches_dispatched += 1
            self._total_items += len(batch)

            logger.debug(
                "Batch %d: %d items in %.1fms (%.1fms/item)",
                self._batches_dispatched,
                len(batch),
                latency,
                latency / len(batch),
            )

        except Exception as exc:
            # Fail all futures in the batch
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            logger.error("Batch inference failed: %s", exc)

    def stats(self) -> Dict[str, Any]:
        """Return batcher statistics."""
        with self._lock:
            queue_depth = len(self._queue)
        return {
            "batches_dispatched": self._batches_dispatched,
            "total_items": self._total_items,
            "queue_depth": queue_depth,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "device": self.device,
        }

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the batcher, flushing remaining items."""
        self._shutdown = True
        self._event.set()
        if wait:
            self._thread.join(timeout=5.0)

        # Flush remaining
        with self._lock:
            remaining = list(self._queue)
            self._queue.clear()

        if remaining:
            self._dispatch_batch(remaining)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_batcher_instance: Optional[DynamicBatcher] = None


def get_batcher(predictor=None) -> Optional[DynamicBatcher]:
    """Get or create the global dynamic batcher.

    Only activates if CATHODE_ENABLE_BATCHING=true and a GPU is available.
    Falls back to synchronous inference otherwise.
    """
    global _batcher_instance

    enable = os.getenv("CATHODE_ENABLE_BATCHING", "false").strip().lower() in {"1", "true", "yes"}
    if not enable:
        return None

    if _batcher_instance is not None:
        return _batcher_instance

    if predictor is None:
        return None

    device = os.getenv("CATHODE_DEVICE", "auto").lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_batch = int(os.getenv("CATHODE_BATCH_SIZE", "32"))
    max_wait = float(os.getenv("CATHODE_BATCH_TIMEOUT_MS", "200"))

    _batcher_instance = DynamicBatcher(
        predictor=predictor,
        max_batch_size=max_batch,
        max_wait_ms=max_wait,
        device=device,
    )
    return _batcher_instance
