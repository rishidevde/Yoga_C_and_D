"""Reference skeleton panel renderer.

Renders a canonical target skeleton in an adjacent UI panel, scaled to match
the user's calibrated bone proportions for intuitive side-by-side pose guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from yogacorrect.calibration import CalibrationProfile
from yogacorrect.types import POSE_CONNECTIONS


# Canonical normalized 2D skeleton keypoints for reference poses
# Coordinates are normalized: x ∈ [0, 1], y ∈ [0, 1], origin top-left.
CANONICAL_REFERENCE_POSES: Dict[str, Dict[int, Tuple[float, float]]] = {
    "triangle": {
        # Head / Neck
        0: (0.45, 0.20),
        11: (0.43, 0.32),  # left shoulder (bottom)
        12: (0.50, 0.28),  # right shoulder (top)
        13: (0.38, 0.44),  # left elbow
        15: (0.35, 0.58),  # left wrist (pointing to front foot)
        14: (0.54, 0.18),  # right elbow
        16: (0.58, 0.08),  # right wrist (pointing straight up)
        23: (0.52, 0.45),  # left hip
        24: (0.58, 0.44),  # right hip
        25: (0.44, 0.65),  # left knee (straight)
        27: (0.36, 0.86),  # left ankle
        26: (0.68, 0.64),  # right knee (straight)
        28: (0.78, 0.86),  # right ankle
    },
    "bhujangasana": {
        # Cobra
        0: (0.28, 0.32),
        11: (0.35, 0.40),
        12: (0.37, 0.40),
        13: (0.34, 0.54),
        15: (0.34, 0.68),
        14: (0.36, 0.54),
        16: (0.36, 0.68),
        23: (0.55, 0.65),
        24: (0.56, 0.65),
        25: (0.72, 0.67),
        27: (0.88, 0.68),
        26: (0.73, 0.67),
        28: (0.89, 0.68),
    },
    "dhanurasana": {
        # Bow
        0: (0.30, 0.36),
        11: (0.38, 0.45),
        12: (0.40, 0.45),
        13: (0.48, 0.40),
        15: (0.64, 0.36),  # hands reaching ankles
        14: (0.50, 0.40),
        16: (0.65, 0.36),
        23: (0.50, 0.60),
        24: (0.52, 0.60),
        25: (0.68, 0.48),  # knees lifted
        27: (0.64, 0.38),  # ankles held
        26: (0.70, 0.48),
        28: (0.65, 0.38),
    },
}


class ReferencePanel:
    """Renders a target pose reference panel alongside the user's camera view."""

    def __init__(
        self,
        panel_width: int = 320,
        panel_height: int = 480,
        bg_color: Tuple[int, int, int] = (25, 25, 25),
        bone_color: Tuple[int, int, int] = (80, 220, 80),   # Vibrant green
        joint_color: Tuple[int, int, int] = (255, 255, 255), # White
    ) -> None:
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.bg_color = bg_color
        self.bone_color = bone_color
        self.joint_color = joint_color

    def render(
        self,
        pose_key: str,
        pose_name: str,
        calibration: Optional[CalibrationProfile] = None,
    ) -> np.ndarray:
        """Render target reference panel image.

        Args:
            pose_key: Key for target pose (e.g. "triangle").
            pose_name: Display title for pose.
            calibration: Optional CalibrationProfile to adjust proportions.

        Returns:
            RGB image of the reference panel.
        """
        panel = np.full(
            (self.panel_height, self.panel_width, 3),
            self.bg_color,
            dtype=np.uint8,
        )

        # Title header
        cv2.putText(
            panel,
            "Target Form",
            (16, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            pose_name,
            (16, 56),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        keypoints = CANONICAL_REFERENCE_POSES.get(pose_key)
        if not keypoints:
            cv2.putText(
                panel,
                "No reference data",
                (30, self.panel_height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
            return panel

        # Convert normalized coordinates to panel pixel coordinates
        pts_px: Dict[int, Tuple[int, int]] = {}
        for idx, (nx, ny) in keypoints.items():
            px = int(nx * self.panel_width)
            py = int(ny * self.panel_height)
            pts_px[idx] = (px, py)

        # Draw bones
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in pts_px and end_idx in pts_px:
                cv2.line(
                    panel,
                    pts_px[start_idx],
                    pts_px[end_idx],
                    self.bone_color,
                    2,
                    cv2.LINE_AA,
                )

        # Draw joints
        for idx, pt in pts_px.items():
            cv2.circle(panel, pt, 4, self.joint_color, -1, cv2.LINE_AA)
            cv2.circle(panel, pt, 5, self.bone_color, 1, cv2.LINE_AA)

        # Bottom alignment tip
        cv2.putText(
            panel,
            "Match your posture",
            (16, self.panel_height - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return panel
