"""Unit tests for signal filtering and One-Euro smoothing."""

import numpy as np
import pytest

from yogacorrect.filters import LandmarkFilter, OneEuroFilter
from yogacorrect.types import Landmarks


class TestOneEuroFilter:
    """Tests for OneEuroFilter signal smoothing."""

    def test_static_signal_noise_reduction(self):
        filter_1e = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        true_value = 10.0
        # Add random noise around true_value
        np.random.seed(42)
        raw_values = [true_value + np.random.normal(0, 0.5) for _ in range(50)]

        filtered_values = []
        for i, val in enumerate(raw_values):
            t = i * (1.0 / 30.0)
            filtered = filter_1e.filter(val, t)
            filtered_values.append(filtered)

        raw_std = np.std(raw_values[10:])
        filt_std = np.std(filtered_values[10:])
        # Filtered signal must have lower variance than raw noisy signal
        assert filt_std < raw_std

    def test_dynamic_signal_tracking(self):
        filter_1e = OneEuroFilter(min_cutoff=1.0, beta=10.0)
        # Linear ramp: signal moves by 1.0 each second
        for i in range(30):
            t = i * (1.0 / 30.0)
            val = t * 10.0
            filtered = filter_1e.filter(val, t)

        # At high speed, lag should be minimal
        assert abs(float(filtered) - 29.0 * (1.0 / 30.0) * 10.0) < 1.0


class TestLandmarkFilter:
    """Tests for LandmarkFilter across 33 keypoints."""

    def test_landmark_filter_preserves_shape_and_visibility(self):
        lm_filter = LandmarkFilter()
        img = np.random.randn(33, 3).astype(np.float32)
        world = np.random.randn(33, 3).astype(np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.8

        lm = Landmarks(timestamp=0.1, image=img, world=world, visibility=vis)
        filtered_lm = lm_filter.filter(lm)

        assert filtered_lm.image.shape == (33, 3)
        assert filtered_lm.world.shape == (33, 3)
        assert np.allclose(filtered_lm.visibility, vis)
