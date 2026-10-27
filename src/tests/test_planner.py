"""Unit tests for the planner module."""

import numpy as np
import pytest

from yogacorrect.planner import CorrectionPlanner
from yogacorrect.types import Error, Landmarks, Rule


@pytest.fixture
def arm_error():
    """Create an error where elbow is at 45 deg, target is 90 deg."""
    rule = Rule(
        name="elbow_bend",
        kind="angle",
        joints=(11, 13, 15),
        target=90.0,
        tolerance=10.0,
        cue_low="Bend your elbow more",
        cue_high="Straighten your elbow slightly",
    )
    return Error(rule=rule, value=45.0, delta=45.0, score=4.5)


@pytest.fixture
def arm_landmarks():
    """Create landmarks corresponding to a 45 deg elbow angle."""
    img = np.zeros((33, 3), dtype=np.float32)
    world = np.zeros((33, 3), dtype=np.float32)
    vis = np.ones(33, dtype=np.float32)

    # Shoulder 11, Elbow 13 (vertex), Wrist 15
    # b (elbow) at (0.5, 0.5) in image, (0, 0, 0) in world
    # a (shoulder) at (0.5, 0.2) in image, (0, -0.3, 0) in world
    # c (wrist) at (0.7, 0.5) in image, (0.2, 0.2, 0) in world
    world[11] = [0.0, -0.3, 0.0]
    world[13] = [0.0, 0.0, 0.0]
    world[15] = [0.2, 0.2, 0.0]

    img[11] = [0.5, 0.2, 0.0]
    img[13] = [0.5, 0.5, 0.0]
    img[15] = [0.7, 0.5, 0.0]

    return Landmarks(timestamp=1.0, image=img, world=world, visibility=vis)


class TestCorrectionPlanner:
    """Test CorrectionPlanner generation and hysteresis."""

    def test_plan_generates_correction(self, arm_error, arm_landmarks):
        planner = CorrectionPlanner(fail_threshold=1)
        corrections = planner.plan(
            errors=[arm_error],
            landmarks=arm_landmarks,
            frame_width=640,
            frame_height=480,
            apply_hysteresis=False,
        )
        assert len(corrections) == 1
        c = corrections[0]
        assert c.joint_idx == 13
        assert "45° → 90°" in c.label
        assert c.cue == "Bend your elbow more"
        assert len(c.ghost_px) == 2
        assert len(c.current_px) == 2

    def test_hysteresis_delays_activation(self, arm_error, arm_landmarks):
        planner = CorrectionPlanner(fail_threshold=3, ok_threshold=2)
        # Frames 1 and 2: should not trigger yet
        c1 = planner.plan([arm_error], arm_landmarks, 640, 480, apply_hysteresis=True)
        assert len(c1) == 0

        c2 = planner.plan([arm_error], arm_landmarks, 640, 480, apply_hysteresis=True)
        assert len(c2) == 0

        # Frame 3: threshold reached -> triggers!
        c3 = planner.plan([arm_error], arm_landmarks, 640, 480, apply_hysteresis=True)
        assert len(c3) == 1

        # Now pass for 1 frame -> stays active because ok_threshold is 2
        c4 = planner.plan([], arm_landmarks, 640, 480, apply_hysteresis=True, all_rules_names=["elbow_bend"])
        assert len(c4) == 0

        # Pass for 2nd frame -> ok_count reaches 2, deactivates
        planner.plan([], arm_landmarks, 640, 480, apply_hysteresis=True, all_rules_names=["elbow_bend"])
        assert planner._rule_states["elbow_bend"]["state"] == "OK"
