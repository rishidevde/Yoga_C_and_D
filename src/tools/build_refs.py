"""Tool to compute clean target angles and tolerances from raw landmark datasets.

Reads raw landmark CSVs (e.g. Dataset_Triangle.csv), recomputes 3D joint angles
using the unsigned 0-180 arccos formulation, calculates median (target) and
scaled Median Absolute Deviation (MAD) for tolerance, and outputs YAML configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from yogacorrect.geometry import angle_deg
from yogacorrect.types import LandmarkIndex


# Definition of standard angles to extract for yoga poses
POSE_ANGLE_DEFINITIONS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "triangle": {
        "front_knee_straight": (23, 25, 27),     # left hip - knee - ankle
        "back_knee_straight": (24, 26, 28),      # right hip - knee - ankle
        "top_arm_extension": (11, 12, 16),       # left shoulder - right shoulder - right wrist
        "bottom_arm_alignment": (12, 11, 15),    # right shoulder - left shoulder - left wrist
        "torso_lateral_bend": (11, 23, 25),      # left shoulder - left hip - left knee
    }
}


def compute_dataset_angles(
    csv_path: str | Path,
    angle_defs: Dict[str, Tuple[int, int, int]],
    min_visibility: float = 0.5,
) -> Dict[str, List[float]]:
    """Extract joint angles across all valid rows in a dataset CSV."""
    df = pd.read_csv(csv_path)

    # MediaPipe landmark names in CSV
    names = [m.name for m in LandmarkIndex]
    results: Dict[str, List[float]] = {name: [] for name in angle_defs}

    for _, row in df.iterrows():
        # Parse 33 landmarks
        landmarks_3d = np.zeros((33, 3), dtype=np.float64)
        visibilities = np.zeros(33, dtype=np.float64)

        valid_row = True
        for idx, name in enumerate(names):
            x_col = f"{name}_x"
            y_col = f"{name}_y"
            z_col = f"{name}_z"
            v_col = f"{name}_vis"

            if x_col not in row:
                valid_row = False
                break

            landmarks_3d[idx] = [row[x_col], row[y_col], row[z_col]]
            visibilities[idx] = row.get(v_col, 1.0)

        if not valid_row:
            continue

        for angle_name, (a_idx, b_idx, c_idx) in angle_defs.items():
            if (
                visibilities[a_idx] >= min_visibility
                and visibilities[b_idx] >= min_visibility
                and visibilities[c_idx] >= min_visibility
            ):
                deg = angle_deg(landmarks_3d[a_idx], landmarks_3d[b_idx], landmarks_3d[c_idx])
                if not np.isnan(deg):
                    results[angle_name].append(deg)

    return results


def summarize_targets(angles_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute median target and robust tolerance (MAD)."""
    summary: Dict[str, Dict[str, float]] = {}

    for name, values in angles_dict.items():
        if not values:
            continue
        arr = np.array(values)
        median = float(np.median(arr))
        # Scaled Median Absolute Deviation (MAD) as robust standard deviation
        mad = float(np.median(np.abs(arr - median)) * 1.4826)
        # Minimum tolerance 8 deg, maximum 25 deg
        tolerance = float(np.clip(round(mad, 1), 8.0, 25.0))

        summary[name] = {
            "target": round(median, 1),
            "tol": tolerance,
            "sample_count": len(arr),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract target angles from dataset CSV.")
    parser.add_argument(
        "--csv",
        default="Yoga_Poses-Dataset/Results/Dataset_Triangle.csv",
        help="Path to landmark CSV",
    )
    parser.add_argument("--pose", default="triangle", help="Pose key")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Check relative to repo root
        root_cand = Path(r"c:\Yoga_F") / args.csv
        if root_cand.exists():
            csv_path = root_cand

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    defs = POSE_ANGLE_DEFINITIONS.get(args.pose, {})
    if not defs:
        print(f"No angle definitions configured for pose: {args.pose}")
        return

    print(f"Processing {csv_path} for pose '{args.pose}'...")
    angles = compute_dataset_angles(csv_path, defs)
    summary = summarize_targets(angles)

    print("\n--- Computed Targets & Tolerances ---")
    for rule_name, stats in summary.items():
        print(f"Rule: {rule_name:25s} Target: {stats['target']:6.1f}°  Tol: ±{stats['tol']:4.1f}°  (N={stats['sample_count']})")


if __name__ == "__main__":
    main()
