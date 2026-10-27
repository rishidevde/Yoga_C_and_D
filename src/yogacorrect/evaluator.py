"""Rule-based pose evaluation engine.

Loads pose definitions from YAML and evaluates user landmarks against target
angles, tolerances, and visibility thresholds. Produces a ranked list of
errors prioritizing safety rules and high-impact deviations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from yogacorrect.geometry import angle_deg, batch_angles
from yogacorrect.types import Error, Landmarks, Rule, LANDMARK_NAMES

logger = logging.getLogger(__name__)


def load_pose_rules(
    yaml_path: str | Path,
    pose_key: str,
) -> Tuple[Dict[str, Any], List[Rule]]:
    """Load configuration and rule definitions for a specific pose from YAML.

    Args:
        yaml_path: Path to poses.yaml.
        pose_key: Pose identifier (e.g. "triangle", "bhujangasana").

    Returns:
        Tuple of (pose_metadata_dict, list_of_rules).

    Raises:
        KeyError: If pose_key is not found in the YAML file.
        FileNotFoundError: If yaml_path does not exist.
    """
    path = Path(yaml_path)
    if not path.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        candidate = base_dir / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Poses configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if pose_key not in data:
        raise KeyError(
            f"Pose '{pose_key}' not found in {path}. Available poses: {list(data.keys())}"
        )

    pose_cfg = data[pose_key]
    metadata = {
        "key": pose_key,
        "name": pose_cfg.get("name", pose_key.title()),
        "view": pose_cfg.get("view", "front"),
        "min_visibility": float(pose_cfg.get("min_visibility", 0.5)),
        "hold_seconds": float(pose_cfg.get("hold_seconds", 3.0)),
    }

    rules: List[Rule] = []
    for r_data in pose_cfg.get("rules", []):
        # Resolve joint names (e.g. "left_hip") to landmark indices (e.g. 23)
        raw_joints = r_data["joints"]
        resolved_joints: List[int] = []
        for j in raw_joints:
            if isinstance(j, int):
                resolved_joints.append(j)
            else:
                j_lower = str(j).strip().lower()
                if j_lower not in LANDMARK_NAMES:
                    raise ValueError(f"Unknown landmark name '{j}' in rule '{r_data.get('name')}'")
                resolved_joints.append(LANDMARK_NAMES[j_lower])

        hint_axis = None
        if "hint_axis" in r_data and r_data["hint_axis"] is not None:
            hint_axis = tuple(float(x) for x in r_data["hint_axis"])

        rule = Rule(
            name=r_data["name"],
            kind=r_data.get("kind", "angle"),
            joints=tuple(resolved_joints),
            target=float(r_data["target"]),
            tolerance=float(r_data["tol"]),
            weight=float(r_data.get("weight", 1.0)),
            safety=bool(r_data.get("safety", False)),
            cue_low=r_data.get("cue_low", ""),
            cue_high=r_data.get("cue_high", ""),
            hint_axis=hint_axis,
        )
        rules.append(rule)

    return metadata, rules


SYMMETRIC_SWAP = {
    11: 12, 12: 11,  # shoulders
    13: 14, 14: 13,  # elbows
    15: 16, 16: 15,  # wrists
    17: 18, 18: 17,  # pinkies
    19: 20, 20: 19,  # index fingers
    21: 22, 22: 21,  # thumbs
    23: 24, 24: 23,  # hips
    25: 26, 26: 25,  # knees
    27: 28, 28: 27,  # ankles
    29: 30, 30: 29,  # heels
    31: 32, 32: 31,  # foot index
}


def _swap_text_side(text: str) -> str:
    """Swap left and right words in cue text."""
    return (
        text.replace("left", "__TMP__")
        .replace("Left", "__TMP_CAP__")
        .replace("right", "left")
        .replace("Right", "Left")
        .replace("__TMP__", "right")
        .replace("__TMP_CAP__", "Right")
    )


class PoseEvaluator:
    """Evaluates Landmarks against a list of Rules.

    Ranks errors by importance (safety first, then score = weight * error / tolerance),
    and filters conflicting corrections. Automatically adapts bilateral poses to the active side.
    """

    def __init__(
        self,
        rules: List[Rule],
        pose_key: str = "triangle",
        min_visibility: float = 0.5,
        max_errors: int = 3,
    ) -> None:
        """Initialize PoseEvaluator.

        Args:
            rules: List of Rule objects to check.
            pose_key: Name of the active pose.
            min_visibility: Landmark visibility threshold (below which rule is skipped).
            max_errors: Maximum number of failing rules to return per frame.
        """
        self.rules = rules
        self.pose_key = pose_key.lower()
        self.min_visibility = min_visibility
        self.max_errors = max_errors

        # Bilateral side-detection hysteresis state (used for triangle-class poses)
        self._swap_state_frames: int = 0   # frames confirming user bends RIGHT
        self._noswap_state_frames: int = 0  # frames confirming user bends LEFT
        self._active_side: str = "left_bend"  # current stable side

    def reset_side_detection(self) -> None:
        """Reset bilateral side-detection state (call on pose change)."""
        self._swap_state_frames = 0
        self._noswap_state_frames = 0
        self._active_side = "left_bend"

    def _swap_rule_side(self, rule: Rule) -> Rule:
        """Produce bilateral mirrored rule (left <-> right)."""
        swapped_joints = tuple(SYMMETRIC_SWAP.get(j, j) for j in rule.joints)
        hint = rule.hint_axis
        if hint is not None:
            # Mirror X-axis for rotation hint
            hint = (-hint[0], hint[1], hint[2])
        return Rule(
            name=f"{rule.name}_mirror",
            kind=rule.kind,
            joints=swapped_joints,
            target=rule.target,
            tolerance=rule.tolerance,
            weight=rule.weight,
            safety=rule.safety,
            cue_low=_swap_text_side(rule.cue_low),
            cue_high=_swap_text_side(rule.cue_high),
            hint_axis=hint,
        )

    def evaluate(self, landmarks: Landmarks) -> List[Error]:
        """Evaluate landmarks against active rules.

        Args:
            landmarks: Detected user landmarks.

        Returns:
            List of Error objects sorted by priority (safety first, then highest score).
            Empty list if pose is fully aligned.
        """
        errors: List[Error] = []

        # ---------------------------------------------------------------
        # Adaptive bilateral side detection with hysteresis
        # Prevents frame-by-frame oscillation when both wrists are near
        # the same height (e.g. during T-pose preparation).
        # ---------------------------------------------------------------
        eval_rules = self.rules
        if self.pose_key == "triangle":
            lwy = landmarks.image[15, 1]  # left wrist y (normalised; smaller = higher)
            rwy = landmarks.image[16, 1]  # right wrist y
            wrist_diff = rwy - lwy        # positive → left wrist is higher (right bend)

            # Minimum threshold: only update counters when difference is unambiguous
            MIN_SWAP_THRESHOLD = 0.15

            if wrist_diff > MIN_SWAP_THRESHOLD:
                # Left wrist clearly above right → user bending to their RIGHT
                self._swap_state_frames += 1
                self._noswap_state_frames = 0
            elif wrist_diff < -MIN_SWAP_THRESHOLD:
                # Right wrist clearly above left → user bending to their LEFT
                self._noswap_state_frames += 1
                self._swap_state_frames = 0
            # else: ambiguous zone — hold existing side without resetting counters

            # Only commit a side switch after SIDE_HYSTERESIS consecutive clear frames
            SIDE_HYSTERESIS = 8
            if self._swap_state_frames >= SIDE_HYSTERESIS:
                self._active_side = "right_bend"
            elif self._noswap_state_frames >= SIDE_HYSTERESIS:
                self._active_side = "left_bend"
            # else: keep _active_side unchanged

            if self._active_side == "right_bend":
                eval_rules = [self._swap_rule_side(r) for r in self.rules]

        for rule in eval_rules:
            # 1. Visibility Gate: Check if all joints required by this rule are visible
            joints_vis = [landmarks.visibility[j] for j in rule.joints]
            if any(v < self.min_visibility for v in joints_vis):
                continue

            # 2. Compute measured value based on rule kind
            if rule.kind == "angle":
                if len(rule.joints) != 3:
                    continue
                a_idx, b_idx, c_idx = rule.joints
                # Compute angle from 3D world coordinates
                val = angle_deg(
                    landmarks.world[a_idx],
                    landmarks.world[b_idx],
                    landmarks.world[c_idx],
                )
                if np.isnan(val):
                    continue

            elif rule.kind == "depth_diff":
                if len(rule.joints) != 2:
                    continue
                j1, j2 = rule.joints
                val = abs(float(landmarks.world[j1][2] - landmarks.world[j2][2]))

            elif rule.kind == "ratio":
                if len(rule.joints) != 4:
                    continue
                j1, j2, j3, j4 = rule.joints
                d1 = np.linalg.norm(landmarks.world[j1] - landmarks.world[j2])
                d2 = np.linalg.norm(landmarks.world[j3] - landmarks.world[j4])
                val = float(d1 / (d2 + 1e-9))

            else:
                continue

            # 3. Check deviation from target
            delta = rule.target - val
            err_mag = abs(delta)

            if err_mag > rule.tolerance:
                score = rule.weight * (err_mag / (rule.tolerance + 1e-6))
                errors.append(
                    Error(
                        rule=rule,
                        value=val,
                        delta=delta,
                        score=score,
                    )
                )

        # 4. Rank errors:
        # Priority: safety rules first (safety=True), then highest score descending
        errors.sort(key=lambda e: (not e.rule.safety, -e.score))

        # 5. Prevent contradictory corrections on the same distal joint:
        filtered_errors: List[Error] = []
        used_distal: set[int] = set()

        for err in errors:
            if err.rule.kind == "angle" and len(err.rule.joints) >= 3:
                distal = err.rule.joints[2]
                if distal in used_distal:
                    continue
                used_distal.add(distal)

            filtered_errors.append(err)

            if len(filtered_errors) >= self.max_errors:
                break

        return filtered_errors
