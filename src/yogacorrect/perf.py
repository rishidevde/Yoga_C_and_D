"""Performance monitoring and latency profiling module.

Instruments pipeline execution, tracks stage timings (capture, inference,
evaluate, plan, render), and computes rolling p50 and p95 latency percentiles.
"""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional

import numpy as np


class PerformanceProfiler:
    """Sliding-window latency profiler across pipeline stages."""

    def __init__(self, window_size: int = 120) -> None:
        self.window_size = window_size
        self._stage_timings: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )
        self._current_frame_marks: Dict[str, float] = {}

    def mark(self, stage_name: str) -> None:
        """Mark completion time of a stage for the current frame."""
        self._current_frame_marks[stage_name] = time.perf_counter()

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Directly record elapsed time for a stage in milliseconds."""
        self._stage_timings[stage_name].append(duration_ms)

    def record_delta(self, stage_name: str, start_mark: str, end_mark: str) -> None:
        """Record duration between two existing timestamps in milliseconds."""
        if start_mark in self._current_frame_marks and end_mark in self._current_frame_marks:
            dt_ms = (self._current_frame_marks[end_mark] - self._current_frame_marks[start_mark]) * 1000.0
            self._stage_timings[stage_name].append(dt_ms)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute p50 and p95 metrics for each tracked stage in milliseconds."""
        stats: Dict[str, Dict[str, float]] = {}
        for stage, deq in self._stage_timings.items():
            if len(deq) == 0:
                continue
            arr = np.array(deq)
            stats[stage] = {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }
        return stats

    def format_summary(self) -> str:
        """Format latency stats as readable text summary."""
        stats = self.get_stats()
        if not stats:
            return "No performance data collected."

        lines = ["--- Pipeline Latency (ms) ---"]
        for stage, m in stats.items():
            lines.append(f"{stage:15s}: p50={m['p50']:5.1f}ms  p95={m['p95']:5.1f}ms (mean={m['mean']:5.1f}ms)")
        return "\n".join(lines)
