"""Correction planner module.

Transforms evaluated errors into concrete visual corrections (ghost limbs and
directional arrows). Applies hysteresis state tracking to prevent flickering,
computes Rodrigues rotation for ghost limb positioning, and generates user cues.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from yogacorrect.geometry import rotate_limb, rotate_limb_2d
from yogacorrect.types import Correction, Error, Landmarks

logger = logging.getLogger(__name__)


class CorrectionPlanner:
    """Plans visual corrections from errors with hysteresis smoothing.

    Hysteresis prevents flickering:
    - An error must persist for `fail_threshold` consecutive frames to trigger an arrow.
    - An active correction persists until the error is within dead zone (< 0.7 * tol)
      for `ok_threshold` frames.
    """

    def __init__(
        self,
        fail_threshold: int = 5,
        ok_threshold: int = 8,
        max_corrections: int = 2,
    ) -> None:
        """Initialize CorrectionPlanner.

        Args:
            fail_threshold: Consecutive failing frames before showing correction.
            ok_threshold: Consecutive passing frames before removing correction.
            max_corrections: Maximum number of corrections displayed simultaneously.
        """
        self.fail_threshold = fail_threshold
        self.ok_threshold = ok_threshold
        self.max_corrections = max_corrections

        # Hysteresis state per rule name:
        # {rule_name: {"state": "OK" | "FAILING", "fail_count": int, "ok_count": int}}
        self._rule_states: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Reset hysteresis state tracking (e.g. on pose change)."""
        self._rule_states.clear()

    def update_hysteresis(self, all_rules_names: List[str], active_errors: List[Error]) -> List[Error]:
        """Filter errors using temporal hysteresis to eliminate flickers.

        Args:
            all_rules_names: Names of all rules evaluated in the pose.
            active_errors: Errors that failed tolerance in the current frame.

        Returns:
            List of Errors that are confirmed FAILING by hysteresis.
        """
        failing_rule_map = {e.rule.name: e for e in active_errors}
        confirmed_errors: List[Error] = []

        for name in all_rules_names:
            if name not in self._rule_states:
                self._rule_states[name] = {
                    "state": "OK",
                    "fail_count": 0,
                    "ok_count": 0,
                }
            st = self._rule_states[name]

            if name in failing_rule_map:
                st["fail_count"] += 1
                st["ok_count"] = 0
                if st["state"] == "OK" and st["fail_count"] >= self.fail_threshold:
                    st["state"] = "FAILING"
            else:
                st["ok_count"] += 1
                st["fail_count"] = 0
                if st["state"] == "FAILING" and st["ok_count"] >= self.ok_threshold:
                    st["state"] = "OK"

            # If rule is confirmed failing, retain its error
            if st["state"] == "FAILING" and name in failing_rule_map:
                confirmed_errors.append(failing_rule_map[name])

        return confirmed_errors

    def plan(
        self,
        errors: List[Error],
        landmarks: Landmarks,
        frame_width: int,
        frame_height: int,
        apply_hysteresis: bool = True,
        all_rules_names: Optional[List[str]] = None,
    ) -> List[Correction]:
        """Generate visual corrections for detected errors.

        Args:
            errors: Failing errors from evaluator.
            landmarks: Detected user landmarks.
            frame_width: Width of image in pixels.
            frame_height: Height of image in pixels.
            apply_hysteresis: Whether to apply temporal smoothing.
            all_rules_names: Optional list of all pose rule names for hysteresis.

        Returns:
            List of Correction instructions ready for rendering.
        """
        if apply_hysteresis:
            rule_names = (
                all_rules_names
                if all_rules_names is not None
                else [e.rule.name for e in errors]
            )
            filtered_errors = self.update_hysteresis(rule_names, errors)
        else:
            filtered_errors = errors

        corrections: List[Correction] = []

        for err in filtered_errors:
            rule = err.rule
            if rule.kind != "angle" or len(rule.joints) != 3:
                continue

            a_idx, b_idx, c_idx = rule.joints
            # b is vertex/pivot, c is distal joint (to move)
            a_px = (
                int(landmarks.image[a_idx, 0] * frame_width),
                int(landmarks.image[a_idx, 1] * frame_height),
            )
            b_px = (
                int(landmarks.image[b_idx, 0] * frame_width),
                int(landmarks.image[b_idx, 1] * frame_height),
            )
            c_px = (
                int(landmarks.image[c_idx, 0] * frame_width),
                int(landmarks.image[c_idx, 1] * frame_height),
            )

            # Ensure non-degenerate segment
            bone_len_px = float(np.hypot(c_px[0] - b_px[0], c_px[1] - b_px[1]))
            if bone_len_px < 5:
                continue

            # Compute target ghost position in 2D screen coordinates
            c_target = rotate_limb_2d(a_px, b_px, c_px, rule.target)
            ghost_px = (int(round(c_target[0])), int(round(c_target[1])))

            # 4. Generate label and cue text
            measured_deg = int(round(err.value))
            target_deg = int(round(rule.target))
            label = f"{measured_deg}° → {target_deg}°"

            # Choose cue based on sign of delta:
            # delta = target - value > 0 means measured is smaller than target
            cue = rule.cue_low if err.delta > 0 else rule.cue_high
            if not cue:
                cue = rule.cue_low or rule.cue_high

            # Build full-limb chain for visualization.
            # For torso/bend rules the 3-joint chain (shoulder→hip→knee) renders
            # as a confusing body-size triangle, so we only show the pivot→distal
            # segment (hip→target_knee) which clearly communicates "tilt from hip".
            is_torso_rule = any(k in rule.name.lower() for k in ["torso", "bend", "spine", "chest"])
            if is_torso_rule:
                target_chain = [b_px, ghost_px]   # pivot → target endpoint only
                actual_chain = [b_px, c_px]        # pivot → actual endpoint only
            else:
                target_chain = [a_px, b_px, ghost_px]  # full limb: anchor→pivot→target
                actual_chain = [a_px, b_px, c_px]      # full limb: anchor→pivot→actual

            corrections.append(
                Correction(
                    joint_idx=b_idx,
                    current_px=c_px,
                    ghost_px=ghost_px,
                    ghost_joint_px=b_px,
                    label=label,
                    cue=cue,
                    target_chain_px=target_chain,
                    actual_chain_px=actual_chain,
                )
            )

            if len(corrections) >= self.max_corrections:
                break

        return corrections
