"""Tool to record landmark sessions and video for reproducible testing.

Records frames and extracts MediaPipe landmarks, saving them to JSON lines
and optional MP4 video files. Allows deterministic replay without a physical webcam.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from yogacorrect.camera import CameraCapture
from yogacorrect.estimator import PoseEstimator


def record_session(
    source: int | str = 0,
    duration_sec: float = 10.0,
    output_dir: str = "data/recordings",
    record_video: bool = True,
) -> None:
    """Record session frames and extracted landmarks."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    json_file = out_path / f"session_{timestamp_str}.jsonl"
    video_file = out_path / f"session_{timestamp_str}.mp4"

    camera = CameraCapture(source=source, width=640, height=480)
    if not camera.is_opened():
        print(f"Error opening camera source: {source}")
        return

    estimator = PoseEstimator(running_mode="LIVE_STREAM")

    writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))

    print(f"Recording session for {duration_sec}s to {out_path}...")
    start_time = time.perf_counter()
    frame_count = 0

    with open(json_file, "w", encoding="utf-8") as f_json:
        try:
            while (time.perf_counter() - start_time) < duration_sec:
                frame, ts = camera.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                if writer:
                    writer.write(frame)

                landmarks = estimator.process_frame(frame, ts)
                record = {
                    "frame_idx": frame_count,
                    "timestamp": ts,
                    "has_pose": landmarks is not None,
                }
                if landmarks is not None:
                    record["image_coords"] = landmarks.image.tolist()
                    record["world_coords"] = landmarks.world.tolist()
                    record["visibility"] = landmarks.visibility.tolist()

                f_json.write(json.dumps(record) + "\n")

                elapsed = time.perf_counter() - start_time
                print(f"\rRecorded {frame_count} frames ({elapsed:.1f}s / {duration_sec}s)...", end="")

        finally:
            camera.release()
            estimator.close()
            if writer:
                writer.release()

    print(f"\nRecording complete! Saved {frame_count} frames to:")
    print(f"  Landmarks: {json_file}")
    if record_video:
        print(f"  Video:     {video_file}")


def main():
    parser = argparse.ArgumentParser(description="Record yoga practice session.")
    parser.add_argument("--source", default="0", help="Camera index or video file")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--out-dir", default="data/recordings", help="Output directory")
    parser.add_argument("--no-video", action="store_true", help="Do not save MP4 video")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    record_session(src, args.duration, args.out_dir, not args.no_video)


if __name__ == "__main__":
    main()
