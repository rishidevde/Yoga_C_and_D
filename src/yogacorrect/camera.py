"""Threaded camera capture module for low-latency frame acquisition.

Runs a background thread that continuously grabs frames from the camera,
keeping only the latest frame (queue size 1) to avoid stale-frame lag.
Supports webcam index or video file path for testing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """Threaded camera reader maintaining the single freshest frame.

    Attributes:
        source: Camera device index (int) or video file path (str).
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested frames per second.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_capture()

    def _init_capture(self) -> None:
        """Initialize OpenCV VideoCapture with optimal latency settings."""
        if isinstance(self.source, int):
            # Windows directshow is faster for webcams
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.warning("Could not open video source: %s", self.source)
            return

        # Attempt to set MJPG format and resolution for speed
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera capture started for source: %s", self.source)

    def _capture_loop(self) -> None:
        """Background thread grabbing frames as fast as possible."""
        while self._running and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    # Loop video file for continuous testing
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.005)
                    continue

            now = time.perf_counter()
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = now

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Retrieve the freshest frame and its capture timestamp.

        Returns:
            Tuple of (frame, capture_timestamp). Frame is None if no frame is available.
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def is_opened(self) -> bool:
        """Check if camera device or video is opened."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Stop background thread and release camera resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera capture released.")
