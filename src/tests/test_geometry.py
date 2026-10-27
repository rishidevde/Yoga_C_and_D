"""Unit tests for geometry calculations and limb rotations."""

import math
import numpy as np
import pytest

from yogacorrect.geometry import angle_deg, rotate_limb, batch_angles


class TestAngleDeg:
    """Tests for angle_deg function."""

    def test_right_angle_2d(self):
        a = (0, 1)
        b = (0, 0)
        c = (1, 0)
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 90.0

    def test_straight_line_180(self):
        a = (-1, 0, 0)
        b = (0, 0, 0)
        c = (1, 0, 0)
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 180.0

    def test_straight_line_0(self):
        a = (1, 0, 0)
        b = (0, 0, 0)
        c = (2, 0, 0)
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 0.0

    def test_45_degrees(self):
        a = (1, 0)
        b = (0, 0)
        c = (1, 1)
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 45.0

    def test_60_degrees_equilateral(self):
        # Equilateral triangle with side length 2
        a = (0, 0)
        b = (2, 0)
        c = (1, np.sqrt(3))
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 60.0

    def test_zero_length_segment_returns_nan(self):
        a = (0, 0, 0)
        b = (0, 0, 0)
        c = (1, 0, 0)
        assert math.isnan(angle_deg(a, b, c))

        a2 = (1, 1, 1)
        b2 = (2, 2, 2)
        c2 = (2, 2, 2)
        assert math.isnan(angle_deg(a2, b2, c2))

    def test_3d_perpendicular(self):
        a = (1, 0, 0)
        b = (0, 0, 0)
        c = (0, 0, 1)
        deg = angle_deg(a, b, c)
        assert pytest.approx(deg, 0.001) == 90.0


class TestRotateLimb:
    """Tests for rotate_limb (ghost limb generation)."""

    def test_rotate_45_to_90(self):
        a = (0, 1, 0)
        b = (0, 0, 0)
        c = (1, 1, 0)  # initial angle is 45 deg
        initial_deg = angle_deg(a, b, c)
        assert pytest.approx(initial_deg, 0.001) == 45.0

        c_prime = rotate_limb(a, b, c, target_deg=90.0)
        assert c_prime is not None

        new_deg = angle_deg(a, b, c_prime)
        assert pytest.approx(new_deg, 0.001) == 90.0

    def test_bone_length_preserved(self):
        a = (0.5, 0.2, 0.1)
        b = (0.2, 0.2, 0.0)
        c = (0.2, 0.8, -0.1)
        orig_len = np.linalg.norm(np.array(c) - np.array(b))

        c_prime = rotate_limb(a, b, c, target_deg=135.0)
        assert c_prime is not None
        new_len = np.linalg.norm(c_prime - np.array(b))
        assert pytest.approx(new_len, 0.001) == orig_len

    def test_collinear_without_hint_returns_none(self):
        a = (1, 0, 0)
        b = (0, 0, 0)
        c = (2, 0, 0)  # Collinear 0 deg
        res = rotate_limb(a, b, c, target_deg=90.0, hint_axis=None)
        assert res is None

    def test_collinear_with_hint_rotates_correctly(self):
        a = (-1, 0, 0)
        b = (0, 0, 0)
        c = (1, 0, 0)  # 180 deg
        # hint axis z-axis (0, 0, 1)
        c_prime = rotate_limb(a, b, c, target_deg=90.0, hint_axis=(0, 0, 1))
        assert c_prime is not None
        new_deg = angle_deg(a, b, c_prime)
        assert pytest.approx(new_deg, 0.001) == 90.0


class TestBatchAngles:
    """Tests for vectorised batch_angles calculation."""

    def test_batch_matches_single(self):
        # Generate dummy 33 landmarks
        world = np.random.RandomState(42).randn(33, 3)
        triplets = np.array([
            [11, 13, 15],
            [12, 14, 16],
            [23, 25, 27],
            [24, 26, 28],
        ])

        batch_res = batch_angles(world, triplets)
        assert len(batch_res) == len(triplets)

        for i, (a_idx, b_idx, c_idx) in enumerate(triplets):
            single_res = angle_deg(world[a_idx], world[b_idx], world[c_idx])
            assert pytest.approx(batch_res[i], 0.0001) == single_res
