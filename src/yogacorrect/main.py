"""Main application entry point for Yoga Pose Correction System.

Orchestrates camera capture, pose estimation, rule evaluation, correction planning,
and real-time UI rendering with interactive keyboard controls.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from yogacorrect.camera import CameraCapture
from yogacorrect.estimator import PoseEstimator
from yogacorrect.evaluator import PoseEvaluator, load_pose_rules
from yogacorrect.planner import CorrectionPlanner
from yogacorrect.renderer import PoseRenderer
from yogacorrect.types import Correction, Landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yogacorrect")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Yoga Pose Detection and Visual Correction System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera device index (e.g. 0) or path to a video file for testing.",
    )
    parser.add_argument(
        "--pose",
        default="triangle",
        help="Initial yoga pose to evaluate (e.g. triangle, ardhachandrasana, veerabhadrasana, etc.).",
    )
    parser.add_argument(
        "--poses-config",
        default="configs/poses.yaml",
        help="Path to YAML file containing pose definitions and rules.",
    )
    parser.add_argument(
        "--app-config",
        default="configs/app.yaml",
        help="Path to application configuration YAML.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror view.",
    )
    return parser.parse_args()


def load_app_config(config_path: str | Path) -> dict:
    """Load app.yaml settings."""
    path = Path(config_path)
    if not path.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        candidate = base_dir / path
        if candidate.exists():
            path = candidate

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> int:
    """Main application loop."""
    args = parse_args()
    app_cfg = load_app_config(args.app_config)

    # 1. Parse video source (integer index or file path)
    source = int(args.source) if args.source.isdigit() else args.source

    cam_cfg = app_cfg.get("camera", {})
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    fps_target = int(cam_cfg.get("fps", 30))

    logger.info("Initializing camera capture on source: %s", source)
    camera = CameraCapture(source=source, width=width, height=height, fps=fps_target)
    if not camera.is_opened():
        logger.error("Failed to open camera/video source: %s", source)
        return 1

    # 2. Initialize Pose Estimator
    model_cfg = app_cfg.get("model", {})
    model_path = model_cfg.get("path", "models/pose_landmarker_lite.task")
    running_mode = model_cfg.get("running_mode", "LIVE_STREAM")
    if isinstance(source, str):
        running_mode = "VIDEO"

    logger.info("Initializing PoseEstimator (%s mode)...", running_mode)
    try:
        estimator = PoseEstimator(
            model_path=model_path,
            min_detection_confidence=float(model_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(model_cfg.get("min_tracking_confidence", 0.5)),
            running_mode=running_mode,
        )
    except Exception as exc:
        logger.error("Failed to initialize PoseEstimator: %s", exc)
        camera.release()
        return 1

    # 3. Setup Pose rules, Evaluator, and Planner
    current_pose_key = args.pose
    poses_config_path = args.poses_config

    try:
        pose_meta, rules = load_pose_rules(poses_config_path, current_pose_key)
    except Exception as exc:
        logger.error("Failed to load pose rules: %s", exc)
        camera.release()
        estimator.close()
        return 1

    evaluator = PoseEvaluator(
        rules=rules,
        pose_key=current_pose_key,
        min_visibility=float(pose_meta.get("min_visibility", 0.5)),
        max_errors=int(app_cfg.get("correction", {}).get("max_corrections_shown", 2)),
    )

    corr_cfg = app_cfg.get("correction", {})
    planner = CorrectionPlanner(
        fail_threshold=int(corr_cfg.get("hysteresis_fail_frames", 4)),
        ok_threshold=int(corr_cfg.get("hysteresis_ok_frames", 6)),
        max_corrections=int(corr_cfg.get("max_corrections_shown", 2)),
    )

    # 4. Setup Renderer & Window
    mirror = not args.no_mirror and bool(app_cfg.get("ui", {}).get("mirror_display", True))
    renderer = PoseRenderer(mirror=mirror)

    ui_cfg = app_cfg.get("ui", {})
    display_w = int(ui_cfg.get("display_width", 1280))
    display_h = int(ui_cfg.get("display_height", 720))
    window_name = ui_cfg.get("window_name", "Yoga Pose Correction")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    logger.info("=======================================================")
    logger.info("Yoga Pose Correction System running!")
    logger.info("Selected Pose: %s", pose_meta["name"])
    logger.info("Controls:")
    logger.info("  [1] Triangle Pose (Trikonasana)")
    logger.info("  [2] Cobra Pose (Bhujangasana)")
    logger.info("  [3] Bow Pose (Dhanurasana)")
    logger.info("  [f] Toggle Fullscreen")
    logger.info("  [r] Reset pose state")
    logger.info("  [q] or [ESC] to quit")
    logger.info("=======================================================")

    # State variables
    fps_time = time.perf_counter()
    fps_current = 0.0
    aligned_start_time: Optional[float] = None
    is_aligned = False
    is_fullscreen = False
    hold_duration_required = float(pose_meta.get("hold_seconds", 3.0))
    last_captured_ts = -1.0

    try:
        while True:
            frame, t_capture = camera.get_frame()
            if frame is None or t_capture == last_captured_ts:
                time.sleep(0.003)
                continue
            last_captured_ts = t_capture

            # Compute smoothed FPS
            now = time.perf_counter()
            dt = now - fps_time
            fps_time = now
            if dt > 0:
                fps_current = 0.9 * fps_current + 0.1 * (1.0 / dt)

            # Estimate pose landmarks on un-flipped frame
            landmarks = estimator.process_frame(frame, t_capture)
            person_detected = landmarks is not None

            corrections: List[Correction] = []

            # -------------------------------------------------------------
            # Prepare High-Definition Display Frame
            # -------------------------------------------------------------
            # In mirror mode, flip the frame horizontally so user looks into a mirror
            if mirror:
                disp_frame = cv2.flip(frame, 1)
            else:
                disp_frame = frame.copy()

            # Upscale display frame to HD (e.g. 1280x720) for large, readable room view
            if disp_frame.shape[1] != display_w or disp_frame.shape[0] != display_h:
                disp_frame = cv2.resize(
                    disp_frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR
                )

            h_disp, w_disp = disp_frame.shape[:2]

            if person_detected and landmarks is not None:
                # Evaluate active rules (with automatic bilateral side adaptation)
                errors = evaluator.evaluate(landmarks)

                # Plan corrections in display coordinates
                all_rule_names = [r.name for r in evaluator.rules]
                corrections = planner.plan(
                    errors=errors,
                    landmarks=landmarks,
                    frame_width=w_disp,
                    frame_height=h_disp,
                    apply_hysteresis=True,
                    all_rules_names=all_rule_names,
                )

                # Alignment state tracking
                if len(corrections) == 0 and len(errors) == 0:
                    if aligned_start_time is None:
                        aligned_start_time = now
                    elif (now - aligned_start_time) >= hold_duration_required:
                        is_aligned = True
                else:
                    aligned_start_time = None
                    is_aligned = False
            else:
                aligned_start_time = None
                is_aligned = False

            # Draw visual feedback
            if person_detected and landmarks is not None:
                renderer.draw_skeleton(disp_frame, landmarks)
                renderer.draw_corrections(disp_frame, corrections)

            renderer.draw_hud(
                frame=disp_frame,
                pose_name=pose_meta["name"],
                corrections=corrections,
                fps=fps_current,
                is_aligned=is_aligned,
                person_detected=person_detected,
            )

            cv2.imshow(window_name, disp_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                logger.info("Fullscreen toggled: %s", is_fullscreen)

            elif key in (
                ord("1"), ord("2"), ord("3"), ord("4"), ord("5"),
                ord("6"), ord("7"), ord("8"), ord("9"), ord("0"),
                ord("n"), ord("p"),
            ):
                pose_order = [
                    "triangle",
                    "ardhachandrasana",
                    "baddhakonasana",
                    "downward_dog",
                    "natarajasana",
                    "utkatakonasana",
                    "veerabhadrasana",
                    "vrukshasana",
                    "bhujangasana",
                    "dhanurasana",
                ]
                pose_map = {
                    ord("1"): "triangle",
                    ord("2"): "ardhachandrasana",
                    ord("3"): "baddhakonasana",
                    ord("4"): "downward_dog",
                    ord("5"): "natarajasana",
                    ord("6"): "utkatakonasana",
                    ord("7"): "veerabhadrasana",
                    ord("8"): "vrukshasana",
                    ord("9"): "bhujangasana",
                    ord("0"): "dhanurasana",
                }
                if key == ord("n"):
                    curr_idx = pose_order.index(current_pose_key) if current_pose_key in pose_order else -1
                    new_pose_key = pose_order[(curr_idx + 1) % len(pose_order)]
                elif key == ord("p"):
                    curr_idx = pose_order.index(current_pose_key) if current_pose_key in pose_order else 0
                    new_pose_key = pose_order[(curr_idx - 1) % len(pose_order)]
                else:
                    new_pose_key = pose_map[key]

                if new_pose_key != current_pose_key:
                    try:
                        pose_meta, rules = load_pose_rules(poses_config_path, new_pose_key)
                        current_pose_key = new_pose_key
                        evaluator = PoseEvaluator(
                            rules=rules,
                            pose_key=current_pose_key,
                            min_visibility=float(pose_meta.get("min_visibility", 0.5)),
                        )
                        evaluator.reset_side_detection()
                        planner.reset()
                        aligned_start_time = None
                        is_aligned = False
                        hold_duration_required = float(pose_meta.get("hold_seconds", 3.0))
                        logger.info("Switched to pose: %s", pose_meta["name"])
                    except Exception as err:
                        logger.error("Error switching pose: %s", err)

            elif key == ord("r"):
                planner.reset()
                aligned_start_time = None
                is_aligned = False
                logger.info("Pose alignment timer reset.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        camera.release()
        estimator.close()
        cv2.destroyAllWindows()
        logger.info("Application cleanly terminated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
