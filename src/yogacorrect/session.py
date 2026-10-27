"""Session state management module.

Tracks the active pose, alignment hold timer, user calibration state,
and session statistics.
"""

from __future__ import annotations

import time
from typing import Optional


class YogaSession:
    """Maintains runtime state and timers for a yoga practice session."""

    def __init__(
        self,
        initial_pose_key: str = "triangle",
        hold_seconds_required: float = 3.0,
    ) -> None:
        self.pose_key = initial_pose_key
        self.hold_seconds_required = hold_seconds_required

        self.session_start_time = time.perf_counter()
        self.frames_processed = 0

        self.aligned_start_time: Optional[float] = None
        self.is_aligned = False
        self.aligned_count = 0

    def update_alignment(self, num_errors: int, current_time: Optional[float] = None) -> bool:
        """Update alignment hold timer based on current frame errors.

        Args:
            num_errors: Number of failing rules in the current frame.
            current_time: Current timestamp (optional, defaults to perf_counter).

        Returns:
            True if the user has maintained alignment for the required duration.
        """
        now = current_time if current_time is not None else time.perf_counter()
        self.frames_processed += 1

        if num_errors == 0:
            if self.aligned_start_time is None:
                self.aligned_start_time = now
            elapsed = now - self.aligned_start_time
            if elapsed >= self.hold_seconds_required:
                if not self.is_aligned:
                    self.aligned_count += 1
                self.is_aligned = True
        else:
            self.aligned_start_time = None
            self.is_aligned = False

        return self.is_aligned

    def get_hold_progress(self, current_time: Optional[float] = None) -> float:
        """Return hold progress as float between 0.0 and 1.0."""
        if self.is_aligned:
            return 1.0
        if self.aligned_start_time is None:
            return 0.0
        now = current_time if current_time is not None else time.perf_counter()
        return min(1.0, (now - self.aligned_start_time) / self.hold_seconds_required)

    def switch_pose(self, new_pose_key: str, hold_seconds: float = 3.0) -> None:
        """Switch current pose and reset alignment state."""
        self.pose_key = new_pose_key
        self.hold_seconds_required = hold_seconds
        self.aligned_start_time = None
        self.is_aligned = False

    def reset_alignment(self) -> None:
        """Reset alignment timer."""
        self.aligned_start_time = None
        self.is_aligned = False
