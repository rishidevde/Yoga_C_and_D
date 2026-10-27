"""Unit tests for the evaluator module."""

import numpy as np
import pytest

from yogacorrect.evaluator import PoseEvaluator, load_pose_rules
from yogacorrect.types import Landmarks, Rule


@pytest.fixture
def dummy_landmarks():
    """Create a dummy Landmarks instance with 33 points."""
    img = np.zeros((33, 3), dtype=np.float32)
    world = np.zeros((33, 3), dtype=np.float32)
    vis = np.ones(33, dtype=np.float32) * 0.9

    # Set up a right angle at left elbow (joint 13):
    # shoulder 11 at (0, 1, 0), elbow 13 at (0, 0, 0), wrist 15 at (1, 0, 0)
    world[11] = [0.0, 1.0, 0.0]
    world[13] = [0.0, 0.0, 0.0]
    world[15] = [1.0, 0.0, 0.0]

    # Set up straight leg at left knee (joint 25):
    # hip 23 at (0, 1, 0), knee 25 at (0, 0.5, 0), ankle 27 at (0, 0, 0) -> 180 deg
    world[23] = [0.0, 1.0, 0.0]
    world[25] = [0.0, 0.5, 0.0]
    world[27] = [0.0, 0.0, 0.0]

    return Landmarks(timestamp=1.0, image=img, world=world, visibility=vis)


class TestLoadPoseRules:
    """Test loading rules from YAML."""

    def test_load_triangle(self):
        meta, rules = load_pose_rules("configs/poses.yaml", "triangle")
        assert meta["key"] == "triangle"
        assert len(rules) >= 4
        # Verify first rule
        r0 = rules[0]
        assert r0.name == "front_knee_straight"
        assert len(r0.joints) == 3
        assert r0.safety is True

    def test_load_unknown_pose_raises(self):
        with pytest.raises(KeyError):
            load_pose_rules("configs/poses.yaml", "non_existent_pose")


class TestPoseEvaluator:
    """Test rule evaluation on landmarks."""

    def test_aligned_pose_returns_no_errors(self, dummy_landmarks):
        rule = Rule(
            name="left_arm_90",
            kind="angle",
            joints=(11, 13, 15),
            target=90.0,
            tolerance=10.0,
        )
        evaluator = PoseEvaluator(rules=[rule])
        errors = evaluator.evaluate(dummy_landmarks)
        assert len(errors) == 0

    def test_error_detected_when_exceeding_tolerance(self, dummy_landmarks):
        # Target 180 deg, but elbow is at 90 deg -> delta is 90 deg -> error
        rule = Rule(
            name="straighten_arm",
            kind="angle",
            joints=(11, 13, 15),
            target=180.0,
            tolerance=15.0,
            cue_low="Extend arm more",
        )
        evaluator = PoseEvaluator(rules=[rule])
        errors = evaluator.evaluate(dummy_landmarks)
        assert len(errors) == 1
        assert errors[0].rule.name == "straighten_arm"
        assert pytest.approx(errors[0].value, 0.1) == 90.0
        assert errors[0].delta > 0  # target - value = 180 - 90 = 90

    def test_visibility_gate_skips_occluded_joints(self, dummy_landmarks):
        # Mark wrist (15) as low visibility
        dummy_landmarks.visibility[15] = 0.2

        rule = Rule(
            name="check_arm",
            kind="angle",
            joints=(11, 13, 15),
            target=180.0,
            tolerance=10.0,
        )
        evaluator = PoseEvaluator(rules=[rule], min_visibility=0.5)
        errors = evaluator.evaluate(dummy_landmarks)
        assert len(errors) == 0  # Skipped because visibility 0.2 < 0.5

    def test_safety_rule_ranked_first(self, dummy_landmarks):
        # Non-safety rule with massive deviation
        non_safety = Rule(
            name="wrist_angle",
            kind="angle",
            joints=(11, 13, 15),
            target=180.0,
            tolerance=5.0,
            weight=10.0,
            safety=False,
        )
        # Safety rule with moderate deviation
        # Target knee 150, measured is 180 -> deviation 30
        safety = Rule(
            name="knee_safety",
            kind="angle",
            joints=(23, 25, 27),
            target=150.0,
            tolerance=10.0,
            weight=1.0,
            safety=True,
        )

        evaluator = PoseEvaluator(rules=[non_safety, safety], max_errors=2)
        errors = evaluator.evaluate(dummy_landmarks)
        assert len(errors) == 2
        # Safety rule must come first despite lower score
        assert errors[0].rule.name == "knee_safety"
        assert errors[1].rule.name == "wrist_angle"


class TestBilateralSwapHysteresis:
    """Test that bilateral side detection is stable and threshold-gated."""

    def _make_lm(self, lwy: float, rwy: float) -> Landmarks:
        """Create Landmarks with specific wrist y-coordinates."""
        img = np.zeros((33, 3), dtype=np.float32)
        world = np.zeros((33, 3), dtype=np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.9
        img[15, 1] = lwy   # left wrist y
        img[16, 1] = rwy   # right wrist y
        # Triangle rule needs visible hip/knee/ankle joints — set them plausibly
        for idx in [23, 24, 25, 26, 27, 28, 11, 12, 15, 16]:
            world[idx] = [0.0, 0.0, 0.0]
        return Landmarks(timestamp=0.0, image=img, world=world, visibility=vis)

    def _triangle_rule(self) -> Rule:
        return Rule(
            name="front_knee_straight",
            kind="angle",
            joints=(23, 25, 27),
            target=180.0,
            tolerance=10.0,
            safety=True,
        )

    def test_no_swap_when_wrist_diff_below_threshold(self):
        """Wrist diff < 0.15 must NOT trigger a swap (avoids T-pose oscillation)."""
        rule = self._triangle_rule()
        ev = PoseEvaluator(rules=[rule], pose_key="triangle")

        # Feed 20 frames where left wrist is slightly higher (diff=0.05 < threshold)
        for _ in range(20):
            lm = self._make_lm(lwy=0.40, rwy=0.45)  # diff = 0.05
            ev.evaluate(lm)

        # Active side must remain at default 'left_bend' — no swap committed
        assert ev._active_side == "left_bend"

    def test_swap_commits_after_hysteresis_frames(self):
        """Wrist diff > 0.15 for >= 8 frames must commit to 'right_bend'."""
        rule = self._triangle_rule()
        ev = PoseEvaluator(rules=[rule], pose_key="triangle")

        # Feed 10 frames with clear right bend (diff = 0.30 > 0.15)
        for _ in range(10):
            lm = self._make_lm(lwy=0.30, rwy=0.60)  # diff = 0.30
            ev.evaluate(lm)

        assert ev._active_side == "right_bend"

    def test_ambiguous_zone_preserves_prior_side(self):
        """During ambiguous T-pose zone, side must not change."""
        rule = self._triangle_rule()
        ev = PoseEvaluator(rules=[rule], pose_key="triangle")

        # First commit to right bend
        for _ in range(10):
            ev.evaluate(self._make_lm(lwy=0.30, rwy=0.60))
        assert ev._active_side == "right_bend"

        # Now go ambiguous (diff < threshold)
        for _ in range(30):
            ev.evaluate(self._make_lm(lwy=0.40, rwy=0.42))  # diff=0.02

        # Should still be right_bend — ambiguous zone must not flip the side
        assert ev._active_side == "right_bend"

    def test_reset_side_detection_clears_state(self):
        """reset_side_detection() must clear all bilateral state."""
        rule = self._triangle_rule()
        ev = PoseEvaluator(rules=[rule], pose_key="triangle")

        for _ in range(10):
            ev.evaluate(self._make_lm(lwy=0.30, rwy=0.60))
        assert ev._active_side == "right_bend"

        ev.reset_side_detection()
        assert ev._active_side == "left_bend"
        assert ev._swap_state_frames == 0
        assert ev._noswap_state_frames == 0
