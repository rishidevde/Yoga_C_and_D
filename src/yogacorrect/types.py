"""Core data structures for the yoga pose correction pipeline.

This module defines the shared types used across all pipeline stages:
camera → estimator → evaluator → planner → renderer.

Conventions:
    - All angles are in degrees, unsigned [0, 180].
    - Image coordinates are normalised [0, 1], origin at top-left.
    - World coordinates are in metres, hip-centred (MediaPipe convention).
    - Visibility scores range from 0.0 (invisible) to 1.0 (fully visible).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmark indices (33 landmarks)
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------

class LandmarkIndex(enum.IntEnum):
    """MediaPipe Pose landmark indices.

    Each member maps to a specific body joint detected by MediaPipe Pose.
    Used throughout the codebase instead of raw integers for clarity.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# String name → index lookup for YAML config parsing.
# Keys are lowercase with underscores: ``"left_shoulder"`` → ``11``.
LANDMARK_NAMES: dict[str, int] = {
    member.name.lower(): member.value for member in LandmarkIndex
}

# Total number of MediaPipe Pose landmarks.
NUM_LANDMARKS: int = 33

# Skeleton connections for drawing — each tuple is ``(start_idx, end_idx)``.
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Upper body
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # Hands
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Lower body
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    # Feet
    (27, 29), (27, 31), (29, 31),
    (28, 30), (28, 32), (30, 32),
]


# ---------------------------------------------------------------------------
# Pipeline data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Landmarks:
    """Detected pose landmarks for a single frame.

    Attributes:
        timestamp: Capture time in seconds (``time.perf_counter``).
        image:     Normalised image coordinates, shape ``(33, 3)`` —
                   x ∈ [0, 1] (left→right), y ∈ [0, 1] (top→bottom), z (depth).
        world:     Metric world coordinates, shape ``(33, 3)`` — metres,
                   hip-centred.
        visibility: Per-landmark confidence, shape ``(33,)`` — 0.0 … 1.0.
    """

    timestamp: float
    image: np.ndarray       # (33, 3) float32
    world: np.ndarray       # (33, 3) float32
    visibility: np.ndarray  # (33,)   float32


@dataclass(frozen=True)
class Rule:
    """A single correction rule loaded from ``poses.yaml``.

    For ``kind="angle"`` the vertex (pivot) is always ``joints[1]``
    (the middle element).

    Attributes:
        name:      Human-readable identifier, e.g. ``"front_knee_straight"``.
        kind:      Rule type — ``"angle"`` | ``"axis_angle"`` | ``"depth_diff"``
                   | ``"ratio"``.
        joints:    Tuple of landmark indices.  For ``"angle"`` rules the vertex
                   is ``joints[1]``.
        target:    Desired value (degrees, metres, or dimensionless).
        tolerance: Acceptable deviation from *target*.
        weight:    Ranking weight (higher = more important).
        safety:    Safety rules are always shown first, regardless of score.
        cue_low:   Text shown when ``measured < target``.
        cue_high:  Text shown when ``measured > target``.
        hint_axis: Optional rotation-axis hint for collinear limbs, ``(x, y, z)``.
    """

    name: str
    kind: str
    joints: tuple[int, ...]
    target: float
    tolerance: float
    weight: float = 1.0
    safety: bool = False
    cue_low: str = ""
    cue_high: str = ""
    hint_axis: Optional[tuple[float, float, float]] = None


@dataclass
class Error:
    """A failing rule together with its measured deviation.

    Attributes:
        rule:  The rule that was violated.
        value: Measured value (degrees / metres / dimensionless).
        delta: Signed difference ``target − value`` (positive ⇒ need more).
        score: Ranking score ``weight × |delta| / tolerance``.
    """

    rule: Rule
    value: float
    delta: float
    score: float


@dataclass
class Correction:
    """Visual correction instruction consumed by the renderer.

    All pixel coordinates are in the **original** (un-mirrored) frame space;
    the renderer handles mirroring.

    Attributes:
        joint_idx:      Landmark index of the vertex joint.
        current_px:     Pixel position of the *current* limb endpoint (distal).
        ghost_px:       Pixel position of the *corrected* limb endpoint (distal).
        ghost_joint_px: Pixel position of the vertex (pivot) joint.
        label:          Angle-change label, e.g. ``"45° → 90°"``.
        cue:            Verbal correction cue for the user.
        target_chain_px: Ordered list of pixel points for the full target limb
                        (e.g. [hip_px, knee_px, target_ankle_px] for a leg correction).
                        If empty, falls back to just [ghost_joint_px, ghost_px].
        actual_chain_px: Ordered list of pixel points for the user's actual limb
                        corresponding to target_chain_px.
    """

    joint_idx: int
    current_px: tuple[int, int]
    ghost_px: tuple[int, int]
    ghost_joint_px: tuple[int, int]
    label: str
    cue: str
    target_chain_px: list[tuple[int, int]] = None
    actual_chain_px: list[tuple[int, int]] = None

    def __post_init__(self):
        if self.target_chain_px is None:
            self.target_chain_px = [self.ghost_joint_px, self.ghost_px]
        if self.actual_chain_px is None:
            self.actual_chain_px = [self.ghost_joint_px, self.current_px]
