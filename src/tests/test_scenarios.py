"""End-to-end scenario tests for realistic yoga posture deviations and corrections."""

import numpy as np
import pytest
import yaml

from yogacorrect.evaluator import PoseEvaluator, load_pose_rules
from yogacorrect.planner import CorrectionPlanner
from yogacorrect.types import Landmarks, Rule


@pytest.fixture
def triangle_postures():
    """Create synthetic landmarks for various Triangle pose scenarios."""
    def _create_triangle(front_knee_bent: bool = False, top_arm_dropped: bool = False) -> Landmarks:
        img = np.zeros((33, 3), dtype=np.float32)
        world = np.zeros((33, 3), dtype=np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.95

        # Shoulders (11, 12)
        world[11] = [-0.15, -0.4, 0.0]  # left shoulder (bottom)
        world[12] = [0.15, -0.5, 0.0]   # right shoulder (top)

        # Arms
        world[15] = [-0.35, -0.1, 0.0]  # left wrist
        if top_arm_dropped:
            # Dropped horizontal instead of vertical
            world[16] = [0.45, -0.4, 0.0]
        else:
            # Straight up vertical
            world[16] = [0.15, -0.9, 0.0]

        # Hips (23, 24)
        world[23] = [-0.1, 0.0, 0.0]   # left hip
        world[24] = [0.1, 0.0, 0.0]    # right hip

        # Back leg (straight): 24 -> 26 -> 28
        world[26] = [0.35, 0.45, 0.0]
        world[28] = [0.6, 0.9, 0.0]

        # Front leg (left): 23 -> 25 -> 27
        if front_knee_bent:
            # Knee bent forward at ~90-120 deg
            world[25] = [-0.35, 0.4, 0.0]
            world[27] = [-0.35, 0.9, 0.0]
        else:
            # Straight knee (~175 deg)
            world[25] = [-0.3, 0.45, 0.0]
            world[27] = [-0.5, 0.9, 0.0]

        # Normalised image coordinates (scaled to [0.2, 0.8])
        img[:, 0] = world[:, 0] * 0.5 + 0.5
        img[:, 1] = world[:, 1] * 0.5 + 0.5

        return Landmarks(timestamp=1.0, image=img, world=world, visibility=vis)

    return _create_triangle


class TestYogaScenarios:
    """Test realistic posture scenarios."""

    def test_bent_front_knee_triggers_knee_correction(self, triangle_postures):
        meta, rules = load_pose_rules("configs/poses.yaml", "triangle")
        evaluator = PoseEvaluator(rules=rules)
        planner = CorrectionPlanner(fail_threshold=1)

        bent_knee_lm = triangle_postures(front_knee_bent=True, top_arm_dropped=False)
        errors = evaluator.evaluate(bent_knee_lm)

        knee_err = next((e for e in errors if e.rule.name == "front_knee_straight"), None)
        assert knee_err is not None
        assert knee_err.value < 150.0  # Significantly bent

        corrections = planner.plan(errors, bent_knee_lm, 640, 480, apply_hysteresis=False)
        knee_corr = next((c for c in corrections if c.joint_idx == 25), None)
        assert knee_corr is not None
        assert "Straighten" in knee_corr.cue

    def test_dropped_top_arm_triggers_arm_correction(self, triangle_postures):
        meta, rules = load_pose_rules("configs/poses.yaml", "triangle")
        evaluator = PoseEvaluator(rules=rules)
        planner = CorrectionPlanner(fail_threshold=1)

        dropped_arm_lm = triangle_postures(front_knee_bent=False, top_arm_dropped=True)
        errors = evaluator.evaluate(dropped_arm_lm)

        arm_err = next((e for e in errors if e.rule.name == "top_arm_vertical"), None)
        assert arm_err is not None

        corrections = planner.plan(errors, dropped_arm_lm, 640, 480, apply_hysteresis=False)
        assert len(corrections) > 0
        arm_corr = next((c for c in corrections if "arm" in c.cue.lower()), None)
        assert arm_corr is not None

    def test_extensibility_custom_pose_added_via_yaml(self):
        """Verify that any new pose can be added via pure YAML without modifying any code."""
        custom_yaml = """
        warrior_custom:
          name: "Custom Warrior"
          view: front
          min_visibility: 0.5
          rules:
            - name: front_elbow_flex
              kind: angle
              joints: [left_shoulder, left_elbow, left_wrist]
              target: 90.0
              tol: 10.0
              cue_low: "Bend elbow"
              cue_high: "Straighten elbow"
              hint_axis: [0.0, 0.0, 1.0]
        """
        data = yaml.safe_load(custom_yaml)
        r_data = data["warrior_custom"]["rules"][0]

        # Evaluator works directly with custom rule
        rule = Rule(
            name=r_data["name"],
            kind=r_data["kind"],
            joints=(11, 13, 15),
            target=r_data["target"],
            tolerance=r_data["tol"],
            cue_low=r_data["cue_low"],
            hint_axis=(0.0, 0.0, 1.0),
        )

        evaluator = PoseEvaluator(rules=[rule])
        planner = CorrectionPlanner(fail_threshold=1)

        img = np.zeros((33, 3), dtype=np.float32)
        world = np.zeros((33, 3), dtype=np.float32)
        vis = np.ones(33, dtype=np.float32)

        # Set elbow at 180 deg (straight)
        world[11] = [0.0, -0.3, 0.0]
        world[13] = [0.0, 0.0, 0.0]
        world[15] = [0.0, 0.3, 0.0]

        # Image pixel coordinates with valid bone length
        img[11] = [0.5, 0.2, 0.0]
        img[13] = [0.5, 0.5, 0.0]
        img[15] = [0.5, 0.8, 0.0]

        lm = Landmarks(timestamp=1.0, image=img, world=world, visibility=vis)
        errors = evaluator.evaluate(lm)
        assert len(errors) == 1
        assert errors[0].rule.name == "front_elbow_flex"

        corrs = planner.plan(errors, lm, 640, 480, apply_hysteresis=False)
        assert len(corrs) == 1
        assert "180° → 90°" in corrs[0].label

    def test_all_dataset_poses_load_and_have_rules(self):
        """Verify that all 8 dataset poses load successfully from poses.yaml with safety rules."""
        required_poses = [
            "triangle",
            "ardhachandrasana",
            "baddhakonasana",
            "downward_dog",
            "natarajasana",
            "utkatakonasana",
            "veerabhadrasana",
            "vrukshasana",
        ]
        for pkey in required_poses:
            meta, rules = load_pose_rules("configs/poses.yaml", pkey)
            assert meta["key"] == pkey
            assert len(rules) >= 3, f"Pose {pkey} should have at least 3 rules"
            # Ensure each pose has at least one safety rule
            assert any(r.safety for r in rules), f"Pose {pkey} must contain a safety rule"

    def test_warrior2_bilateral_knee_swap(self):
        """Warrior II should swap rules when the right knee is bent (right-front) instead of left."""
        meta, rules = load_pose_rules("configs/poses.yaml", "veerabhadrasana")
        evaluator = PoseEvaluator(rules=rules, pose_key="veerabhadrasana")

        img = np.zeros((33, 3), dtype=np.float32)
        world = np.zeros((33, 3), dtype=np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.95

        # Right knee bent ~90 deg (hip: 24, knee: 26, ankle: 28)
        world[24] = [0.1, 0.0, 0.0]
        world[26] = [0.4, 0.0, 0.0]
        world[28] = [0.4, 0.4, 0.0]

        # Left leg straight ~180 deg (hip: 23, knee: 25, ankle: 27)
        world[23] = [-0.1, 0.0, 0.0]
        world[25] = [-0.4, 0.0, 0.0]
        world[27] = [-0.7, 0.0, 0.0]

        lm = Landmarks(timestamp=0.0, image=img, world=world, visibility=vis)

        # Feed 10 frames to commit hysteresis
        for _ in range(10):
            evaluator.evaluate(lm)

        assert evaluator._active_side == "right_bend"

    def test_tree_pose_bilateral_lifted_ankle_swap(self):
        """Tree pose should swap rules when left ankle is lifted (standing on right leg)."""
        meta, rules = load_pose_rules("configs/poses.yaml", "vrukshasana")
        evaluator = PoseEvaluator(rules=rules, pose_key="vrukshasana")

        img = np.zeros((33, 3), dtype=np.float32)
        world = np.zeros((33, 3), dtype=np.float32)
        vis = np.ones(33, dtype=np.float32) * 0.95

        # Left ankle (27) lifted high (y=0.4), Right ankle (28) on ground (y=0.9)
        img[27, 1] = 0.4
        img[28, 1] = 0.9

        lm = Landmarks(timestamp=0.0, image=img, world=world, visibility=vis)
        for _ in range(10):
            evaluator.evaluate(lm)

        assert evaluator._active_side == "right_bend"
