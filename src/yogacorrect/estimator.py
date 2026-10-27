"""MediaPipe PoseLandmarker wrapper.

Wraps the MediaPipe PoseLandmarker Task API to extract 33 pose landmarks
in normalised image coordinates and metric 3D world coordinates.
Supports LIVE_STREAM mode with async callbacks for real-time tracking.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from yogacorrect.types import Landmarks, NUM_LANDMARKS

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Wrapper around MediaPipe PoseLandmarker.

    Extracts 33 body landmarks per frame with normalised image coordinates,
    metric world 3D coordinates (metres, hip-centred), and visibility scores.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "LIVE_STREAM",
    ) -> None:
        """Initialise PoseEstimator.

        Args:
            model_path: Path to the .task model file.
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            running_mode: "LIVE_STREAM" or "VIDEO".
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            # Resolve relative to project root or src
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir / self.model_path
            if candidate.exists():
                self.model_path = candidate

        if not self.model_path.exists():
            raise FileNotFoundError(f"PoseLandmarker model not found at: {self.model_path}")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.running_mode_str = running_mode.upper()

        self._lock = threading.Lock()
        self._latest_landmarks: Optional[Landmarks] = None
        self._landmarker: Optional[vision.PoseLandmarker] = None
        self._frame_count = 0
        self._last_ts_ms = -1
        self._last_input_timestamp = -1.0

        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Create PoseLandmarker instance with configured options."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        if self.running_mode_str == "LIVE_STREAM":
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                result_callback=self._live_stream_callback,
            )
        else:
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        logger.info(
            "PoseEstimator initialized in %s mode with model %s",
            self.running_mode_str,
            self.model_path,
        )

    def _live_stream_callback(
        self,
        result: vision.PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Async callback invoked by MediaPipe LIVE_STREAM mode."""
        t_capture = timestamp_ms / 1000.0
        parsed = self._parse_result(result, t_capture)
        with self._lock:
            self._latest_landmarks = parsed

    def _parse_result(
        self,
        result: vision.PoseLandmarkerResult,
        timestamp: float,
    ) -> Optional[Landmarks]:
        """Convert MediaPipe PoseLandmarkerResult into Landmarks dataclass."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        img_lm = result.pose_landmarks[0]
        world_lm = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0
            else None
        )

        n = min(len(img_lm), NUM_LANDMARKS)
        image_coords = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        world_coords = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        visibility = np.zeros(NUM_LANDMARKS, dtype=np.float32)

        for i in range(n):
            lm = img_lm[i]
            image_coords[i] = [lm.x, lm.y, lm.z]
            visibility[i] = getattr(lm, "visibility", 1.0) or 0.0

            if world_lm is not None and i < len(world_lm):
                w = world_lm[i]
                world_coords[i] = [w.x, w.y, w.z]
            else:
                # Fallback to normalized image coordinates if world coords unavailable
                world_coords[i] = [lm.x, lm.y, lm.z]

        return Landmarks(
            timestamp=timestamp,
            image=image_coords,
            world=world_coords,
            visibility=visibility,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
    ) -> Optional[Landmarks]:
        """Process a frame and return detected landmarks.

        In LIVE_STREAM mode, submits the frame asynchronously and returns
        the latest available landmarks.
        In VIDEO mode, runs detection synchronously and returns the result.

        Args:
            frame_bgr: Input image in OpenCV BGR format.
            timestamp: Capture timestamp in seconds.

        Returns:
            Landmarks dataclass if a pose is detected, else None.
        """
        if self._landmarker is None:
            return None

        # Convert OpenCV BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_count += 1
        # MediaPipe requires strictly monotonically increasing integer timestamp in milliseconds
        ts_ms = int(timestamp * 1000)
        if ts_ms <= 0:
            ts_ms = int(time.perf_counter() * 1000)

        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        if self.running_mode_str == "LIVE_STREAM":
            self._landmarker.detect_async(mp_image, ts_ms)
            with self._lock:
                return self._latest_landmarks
        else:
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
            return self._parse_result(result, timestamp)

    def get_latest_landmarks(self) -> Optional[Landmarks]:
        """Get the most recently computed landmarks (LIVE_STREAM mode)."""
        with self._lock:
            return self._latest_landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        logger.info("PoseEstimator closed.")
