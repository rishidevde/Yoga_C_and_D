"""Calibration module for neutral-stance bone length measurement.

Captures stable neutral-stance frames over a brief hold period (e.g. 3 seconds)
and calculates individual bone lengths (upper arm, forearm, torso, thigh, shin)
in metric 3D coordinates. Used for scaling reference poses and ghost limbs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from yogacorrect.types import LandmarkIndex, Landmarks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationProfile:
    """Stores calibrated user bone lengths in 3D world meters."""

    torso_length: float
    left_upper_arm: float
    right_upper_arm: float
    left_forearm: float
    right_forearm: float
    left_thigh: float
    right_thigh: float
    left_shin: float
    right_shin: float

    def get_segment_length(self, parent_idx: int, child_idx: int) -> Optional[float]:
        """Lookup calibrated length between two connected joints."""
        pair = (min(parent_idx, child_idx), max(parent_idx, child_idx))
        mapping = {
            (11, 13): self.left_upper_arm,
            (13, 15): self.left_forearm,
            (12, 14): self.right_upper_arm,
            (14, 16): self.right_forearm,
            (23, 25): self.left_thigh,
            (25, 27): self.left_shin,
            (24, 26): self.right_thigh,
            (26, 28): self.right_shin,
        }
        return mapping.get(pair)


class BodyCalibrator:
    """Accumulates stable frames during neutral stance and computes bone lengths."""

    def __init__(
        self,
        target_hold_seconds: float = 2.5,
        min_visibility: float = 0.6,
        max_motion_threshold: float = 0.05,
    ) -> None:
        self.target_hold_seconds = target_hold_seconds
        self.min_visibility = min_visibility
        self.max_motion_threshold = max_motion_threshold

        self._start_time: Optional[float] = None
        self._collected_world_landmarks: List[np.ndarray] = []
        self._prev_world: Optional[np.ndarray] = None
        self._is_calibrated = False
        self._profile: Optional[CalibrationProfile] = None

    def reset(self) -> None:
        """Reset calibration session."""
        self._start_time = None
        self._collected_world_landmarks.clear()
        self._prev_world = None
        self._is_calibrated = False
        self._profile = None

    def update(self, landmarks: Landmarks) -> Tuple[bool, float, str]:
        """Update calibration state with incoming frame landmarks.

        Args:
            landmarks: User's detected Landmarks.

        Returns:
            Tuple of (is_complete, progress_0_to_1, status_message).
        """
        if self._is_calibrated and self._profile is not None:
            return True, 1.0, "Calibration Complete"

        # Check full-body visibility (shoulders, hips, knees, ankles)
        essential_joints = [11, 12, 23, 24, 25, 26, 27, 28]
        if any(landmarks.visibility[j] < self.min_visibility for j in essential_joints):
            self.reset()
            return False, 0.0, "Step back so your full body is visible"

        now = time.perf_counter()

        # Check motion stability
        world = landmarks.world
        if self._prev_world is not None:
            motion = float(np.mean(np.linalg.norm(world[essential_joints] - self._prev_world[essential_joints], axis=1)))
            if motion > self.max_motion_threshold:
                # User moved too much, reset timer
                self._start_time = now
                self._collected_world_landmarks.clear()
                self._prev_world = world.copy()
                return False, 0.0, "Hold steady and stand still"

        self._prev_world = world.copy()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time
        progress = min(1.0, elapsed / self.target_hold_seconds)
        self._collected_world_landmarks.append(world.copy())

        if progress >= 1.0 and len(self._collected_world_landmarks) >= 15:
            self._profile = self._compute_profile(self._collected_world_landmarks)
            self._is_calibrated = True
            logger.info("Body calibration successful: %s", self._profile)
            return True, 1.0, "Calibration Complete!"

        return False, progress, f"Calibrating... {int(progress * 100)}%"

    def _compute_profile(self, samples: List[np.ndarray]) -> CalibrationProfile:
        """Compute median bone lengths across collected stable samples."""
        stack = np.array(samples)  # (N, 33, 3)

        def seg_len(j1: int, j2: int) -> float:
            dists = np.linalg.norm(stack[:, j1, :] - stack[:, j2, :], axis=1)
            return float(np.median(dists))

        # Torso: distance between shoulder midpoint and hip midpoint
        mid_shoulders = (stack[:, 11, :] + stack[:, 12, :]) / 2.0
        mid_hips = (stack[:, 23, :] + stack[:, 24, :]) / 2.0
        torso_len = float(np.median(np.linalg.norm(mid_shoulders - mid_hips, axis=1)))

        return CalibrationProfile(
            torso_length=torso_len,
            left_upper_arm=seg_len(11, 13),
            right_upper_arm=seg_len(12, 14),
            left_forearm=seg_len(13, 15),
            right_forearm=seg_len(14, 16),
            left_thigh=seg_len(23, 25),
            right_thigh=seg_len(24, 26),
            left_shin=seg_len(25, 27),
            right_shin=seg_len(26, 28),
        )

    @property
    def profile(self) -> Optional[CalibrationProfile]:
        """Return the completed CalibrationProfile if calibrated."""
        return self._profile
