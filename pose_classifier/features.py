"""Feature extraction engine for yoga pose classification.

Extracts a scale-, position-, and camera-distance invariant feature vector (24 dimensions)
from 33 MediaPipe pose landmarks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Landmark Indices from MediaPipe Pose
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

FEATURE_NAMES = [
    "angle_left_elbow",
    "angle_right_elbow",
    "angle_left_shoulder",
    "angle_right_shoulder",
    "angle_left_knee",
    "angle_right_knee",
    "angle_left_hip",
    "angle_right_hip",
    "angle_inter_leg",
    "angle_torso_tilt",
    "ratio_left_wrist_y",
    "ratio_right_wrist_y",
    "ratio_left_ankle_y",
    "ratio_right_ankle_y",
    "ratio_left_knee_y",
    "ratio_right_knee_y",
    "ratio_wrist_span",
    "ratio_ankle_span",
    "ratio_knee_span",
    "ratio_shoulder_hip_width",
    "ratio_wrist_diff_y",
    "ratio_ankle_diff_y",
    "ratio_head_hip_y",
    "ratio_aspect_ratio",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 24


def calc_angle_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate unsigned angle in degrees at vertex p2 between p2->p1 and p2->p3."""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    dot = np.dot(v1, v2) / (norm1 * norm2)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def extract_feature_vector(pts: np.ndarray) -> np.ndarray:
    """Extract a 24-dimensional invariant feature vector from (33, 2) or (33, 3) landmarks.

    Args:
        pts: Array of shape (33, 2) or (33, 3) containing landmark coordinates.

    Returns:
        1D float32 numpy array of shape (24,) with normalized invariant features.
    """
    p = pts[:, :2]  # Work in 2D image-plane coordinates (x, y)

    # 1. Joint angles (normalized to [0, 1] by dividing by 180.0)
    ang_l_elb = calc_angle_2d(p[LEFT_SHOULDER], p[LEFT_ELBOW], p[LEFT_WRIST]) / 180.0
    ang_r_elb = calc_angle_2d(p[RIGHT_SHOULDER], p[RIGHT_ELBOW], p[RIGHT_WRIST]) / 180.0
    ang_l_sho = calc_angle_2d(p[LEFT_ELBOW], p[LEFT_SHOULDER], p[LEFT_HIP]) / 180.0
    ang_r_sho = calc_angle_2d(p[RIGHT_ELBOW], p[RIGHT_SHOULDER], p[RIGHT_HIP]) / 180.0
    ang_l_kne = calc_angle_2d(p[LEFT_HIP], p[LEFT_KNEE], p[LEFT_ANKLE]) / 180.0
    ang_r_kne = calc_angle_2d(p[RIGHT_HIP], p[RIGHT_KNEE], p[RIGHT_ANKLE]) / 180.0
    ang_l_hip = calc_angle_2d(p[LEFT_SHOULDER], p[LEFT_HIP], p[LEFT_KNEE]) / 180.0
    ang_r_hip = calc_angle_2d(p[RIGHT_SHOULDER], p[RIGHT_HIP], p[RIGHT_KNEE]) / 180.0
    ang_leg_split = calc_angle_2d(p[LEFT_ANKLE], (p[LEFT_HIP] + p[RIGHT_HIP]) * 0.5, p[RIGHT_ANKLE]) / 180.0

    # Torso tilt relative to vertical (0 = vertical upright, 90 = horizontal)
    mid_shoulder = (p[LEFT_SHOULDER] + p[RIGHT_SHOULDER]) * 0.5
    mid_hip = (p[LEFT_HIP] + p[RIGHT_HIP]) * 0.5
    spine_vec = mid_shoulder - mid_hip  # y points down in image coords
    # Angle with vertical [0, -1]
    spine_len = np.linalg.norm(spine_vec)
    if spine_len > 1e-6:
        # Cosine with (0, -1) (upward vertical)
        cos_tilt = (-spine_vec[1]) / spine_len
        ang_torso_tilt = math.degrees(math.acos(np.clip(cos_tilt, -1.0, 1.0))) / 180.0
    else:
        ang_torso_tilt = 0.0

    # Scale normalizer: Torso height (mid_shoulder to mid_hip distance)
    torso_len = max(float(spine_len), 0.05)

    # 2. Relative vertical heights (signed, normalized by torso length)
    # y increases downward, so (mid_hip_y - joint_y) is positive when joint is above hip
    rel_l_wrist_y = float(mid_hip[1] - p[LEFT_WRIST, 1]) / torso_len
    rel_r_wrist_y = float(mid_hip[1] - p[RIGHT_WRIST, 1]) / torso_len
    rel_l_ankle_y = float(mid_hip[1] - p[LEFT_ANKLE, 1]) / torso_len
    rel_r_ankle_y = float(mid_hip[1] - p[RIGHT_ANKLE, 1]) / torso_len
    rel_l_knee_y = float(mid_hip[1] - p[LEFT_KNEE, 1]) / torso_len
    rel_r_knee_y = float(mid_hip[1] - p[RIGHT_KNEE, 1]) / torso_len

    # 3. Distance & Span Ratios (normalized by torso length)
    wrist_span = float(np.linalg.norm(p[LEFT_WRIST] - p[RIGHT_WRIST])) / torso_len
    ankle_span = float(np.linalg.norm(p[LEFT_ANKLE] - p[RIGHT_ANKLE])) / torso_len
    knee_span = float(np.linalg.norm(p[LEFT_KNEE] - p[RIGHT_KNEE])) / torso_len

    hip_width = max(float(np.linalg.norm(p[LEFT_HIP] - p[RIGHT_HIP])), 0.02)
    sho_width = float(np.linalg.norm(p[LEFT_SHOULDER] - p[RIGHT_SHOULDER]))
    sho_hip_ratio = sho_width / hip_width

    # Wrist & ankle vertical differentials (shows reach/bend direction)
    wrist_diff_y = float(p[RIGHT_WRIST, 1] - p[LEFT_WRIST, 1]) / torso_len
    ankle_diff_y = float(p[RIGHT_ANKLE, 1] - p[LEFT_ANKLE, 1]) / torso_len
    head_hip_y = float(mid_hip[1] - p[NOSE, 1]) / torso_len

    # Overall bounding box aspect ratio
    min_x, max_x = np.min(p[:, 0]), np.max(p[:, 0])
    min_y, max_y = np.min(p[:, 1]), np.max(p[:, 1])
    bbox_h = max(max_y - min_y, 0.05)
    aspect_ratio = (max_x - min_x) / bbox_h

    features = np.array([
        ang_l_elb,
        ang_r_elb,
        ang_l_sho,
        ang_r_sho,
        ang_l_kne,
        ang_r_kne,
        ang_l_hip,
        ang_r_hip,
        ang_leg_split,
        ang_torso_tilt,
        rel_l_wrist_y,
        rel_r_wrist_y,
        rel_l_ankle_y,
        rel_r_ankle_y,
        rel_l_knee_y,
        rel_r_knee_y,
        wrist_span,
        ankle_span,
        knee_span,
        sho_hip_ratio,
        wrist_diff_y,
        ankle_diff_y,
        head_hip_y,
        aspect_ratio,
    ], dtype=np.float32)

    return features


def extract_from_dataframe_row(row: Any) -> np.ndarray:
    """Extract features from a CSV DataFrame row where columns are named NOSE_x, NOSE_y, etc."""
    LANDMARK_KEY_LIST = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR",
        "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
        "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]
    pts = np.zeros((33, 2), dtype=np.float32)
    for idx, name in enumerate(LANDMARK_KEY_LIST):
        x_col = f"{name}_x"
        y_col = f"{name}_y"
        if x_col in row and y_col in row:
            pts[idx, 0] = float(row[x_col])
            pts[idx, 1] = float(row[y_col])
    return extract_feature_vector(pts)
