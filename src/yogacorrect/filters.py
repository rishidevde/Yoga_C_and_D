"""Signal filtering and smoothing module.

Implements the One-Euro filter to eliminate landmark jitter without introducing
perceptible lag, along with visibility gating to mask low-confidence joints.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from yogacorrect.types import Landmarks


class OneEuroFilter:
    """One-Euro filter for low-latency 1D or ND signal smoothing.

    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Human-Computer Interaction", CHI 2012.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize OneEuroFilter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing at rest.
            beta: Speed coefficient. Higher = less lag during fast movement.
            d_cutoff: Cutoff frequency for derivative (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Compute exponential smoothing factor alpha."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: np.ndarray | list | float, t: float) -> np.ndarray:
        """Filter a new measurement at timestamp t.

        Args:
            x: Input signal (scalar, 1D, or ND array).
            t: Timestamp in seconds.

        Returns:
            Smoothed signal array of the same shape as x.
        """
        x_curr = np.asarray(x, dtype=np.float64)

        if self._x_prev is None:
            self._x_prev = x_curr.copy()
            self._dx_prev = np.zeros_like(x_curr)
            self._t_prev = float(t)
            return x_curr

        dt = max(float(t - (self._t_prev or t)), 1e-4)

        # 1. Filtered derivative (dx)
        dx_curr = (x_curr - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx_curr + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # 2. Adaptive cutoff frequency based on velocity magnitude
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Filtered signal (x)
        a = self._alpha(cutoff, dt)
        x_hat = a * x_curr + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._t_prev = float(t)

        return x_hat

    def reset(self) -> None:
        """Clear filter history."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class LandmarkFilter:
    """Applies One-Euro filtering across all 33 pose landmarks.

    Smooths image coordinates, 3D world coordinates, and preserves visibility.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.image_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.world_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(self, landmarks: Landmarks) -> Landmarks:
        """Smooth landmarks using One-Euro filtering.

        Args:
            landmarks: Raw detected Landmarks.

        Returns:
            Smoothed Landmarks.
        """
        smoothed_image = self.image_filter.filter(landmarks.image, landmarks.timestamp).astype(
            np.float32
        )
        smoothed_world = self.world_filter.filter(landmarks.world, landmarks.timestamp).astype(
            np.float32
        )

        return Landmarks(
            timestamp=landmarks.timestamp,
            image=smoothed_image,
            world=smoothed_world,
            visibility=landmarks.visibility.copy(),
        )

    def reset(self) -> None:
        """Reset internal filters."""
        self.image_filter.reset()
        self.world_filter.reset()
