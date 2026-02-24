"""
Stage Profiler — context-manager based per-stage performance capture.

Usage:
    profiler = StageProfiler()

    with profiler.stage(StageName.PARSE, input_count=1):
        parsed = parser.parse(instruction)
    profiler.set_output_count(StageName.PARSE, output_count=1)

    metrics = profiler.collect()      # list[StageMetric]
    total   = profiler.total_metric() # StageMetric for TOTAL

Does NOT touch any generation logic — purely observational.
"""

from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from typing import Any, Generator

from src.observability.models import StageMetric, StageName


class StageProfiler:
    """
    Capture timing and approximate memory delta per pipeline stage.

    Thread-safe for sequential stage execution (one stage at a time).
    """

    def __init__(self) -> None:
        self._metrics: dict[StageName, StageMetric] = {}
        self._total_start: float | None = None
        self._total_end: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_total(self) -> None:
        """Mark the start of the entire pipeline."""
        self._total_start = time.perf_counter()

    def end_total(self) -> None:
        """Mark the end of the entire pipeline."""
        self._total_end = time.perf_counter()

    @contextmanager
    def stage(
        self,
        name: StageName,
        *,
        input_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that captures timing and memory for a stage.

        Example::

            with profiler.stage(StageName.GENERATE, input_count=10):
                workflow = generator.generate(...)
        """
        mem_before = self._current_rss_kb()
        t0 = time.perf_counter()

        yield  # <-- the actual stage code runs here

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        mem_after = self._current_rss_kb()

        self._metrics[name] = StageMetric(
            stage=name,
            duration_ms=elapsed_ms,
            memory_delta_kb=round(mem_after - mem_before, 1),
            input_count=input_count,
            output_count=0,  # filled later via set_output_count
            metadata=metadata or {},
        )

    def set_output_count(self, name: StageName, output_count: int) -> None:
        """Set the output count for a stage after it completes."""
        if name in self._metrics:
            self._metrics[name].output_count = output_count

    def add_stage_metadata(self, name: StageName, key: str, value: Any) -> None:
        """Attach extra metadata to a captured stage."""
        if name in self._metrics:
            self._metrics[name].metadata[key] = value

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self) -> list[StageMetric]:
        """Return all captured stage metrics (excluding TOTAL)."""
        return [
            self._metrics[name]
            for name in StageName
            if name in self._metrics and name != StageName.TOTAL
        ]

    def total_metric(self) -> StageMetric:
        """Compute the TOTAL stage metric from start/end marks."""
        if self._total_start is not None and self._total_end is not None:
            duration = round((self._total_end - self._total_start) * 1000, 3)
        else:
            duration = sum(m.duration_ms for m in self._metrics.values())

        return StageMetric(
            stage=StageName.TOTAL,
            duration_ms=duration,
        )

    def get(self, name: StageName) -> StageMetric | None:
        """Get the metric for a specific stage, or None."""
        return self._metrics.get(name)

    def reset(self) -> None:
        """Clear all captured metrics."""
        self._metrics.clear()
        self._total_start = None
        self._total_end = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _current_rss_kb() -> float:
        """Get current process RSS in kilobytes (macOS/Linux)."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS reports in bytes, Linux in KB
            rss = usage.ru_maxrss
            import sys
            if sys.platform == "darwin":
                return rss / 1024.0
            return float(rss)
        except Exception:
            return 0.0
