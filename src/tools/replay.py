"""Tool to replay recorded landmark sessions through the evaluation pipeline.

Reads session JSON Lines files, passes recorded landmarks through the evaluator,
planner, and renderer, outputting frame-by-frame correction metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yogacorrect.evaluator import PoseEvaluator, load_pose_rules
from yogacorrect.planner import CorrectionPlanner
from yogacorrect.types import Landmarks


def replay_session(
    json_path: str | Path,
    pose_key: str = "triangle",
    poses_config: str = "configs/poses.yaml",
) -> None:
    """Replay recorded landmarks through evaluation pipeline."""
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    pose_meta, rules = load_pose_rules(poses_config, pose_key)
    evaluator = PoseEvaluator(rules=rules)
    planner = CorrectionPlanner(fail_threshold=2, ok_threshold=2)

    total_frames = 0
    pose_frames = 0
    aligned_frames = 0
    failing_rule_counts = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total_frames += 1

            if not record.get("has_pose"):
                continue

            pose_frames += 1
            lm = Landmarks(
                timestamp=record["timestamp"],
                image=np.array(record["image_coords"], dtype=np.float32),
                world=np.array(record["world_coords"], dtype=np.float32),
                visibility=np.array(record["visibility"], dtype=np.float32),
            )

            errors = evaluator.evaluate(lm)
            corrections = planner.plan(errors, lm, 640, 480, apply_hysteresis=True)

            if len(corrections) == 0:
                aligned_frames += 1
            else:
                for corr in corrections:
                    failing_rule_counts[corr.cue] = failing_rule_counts.get(corr.cue, 0) + 1

    print("\n--- Session Replay Summary ---")
    print(f"Total Frames:    {total_frames}")
    print(f"Pose Detected:   {pose_frames} ({(pose_frames / max(1, total_frames)) * 100:.1f}%)")
    print(f"Aligned Frames:  {aligned_frames} ({(aligned_frames / max(1, pose_frames)) * 100:.1f}%)")
    print("\nTop Corrections Triggered:")
    for cue, count in sorted(failing_rule_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  [{count:3d} frames] {cue}")


def main():
    parser = argparse.ArgumentParser(description="Replay recorded landmark session.")
    parser.add_argument("json_file", help="Path to recording .jsonl file")
    parser.add_argument("--pose", default="triangle", help="Target pose key")
    parser.add_argument("--config", default="configs/poses.yaml", help="Poses config YAML")
    args = parser.parse_args()

    replay_session(args.json_file, args.pose, args.config)


if __name__ == "__main__":
    main()
