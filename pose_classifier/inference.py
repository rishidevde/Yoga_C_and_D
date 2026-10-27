"""Real-time Auto Pose Detector and Inference Engine.

Wraps trained model with temporal debouncing, confidence threshold gating,
and idle/transition detection for seamless live webcam integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np

from pose_classifier.dataset import ID_TO_LABEL, LABEL_TO_ID
from pose_classifier.features import extract_feature_vector

logger = logging.getLogger(__name__)


class AutoPoseDetector:
    """Real-time automatic yoga pose detector with temporal hysteresis.

    Prevents flickering during transitions and avoids false triggers when user
    is in neutral/idle stance.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_confidence: float = 0.70,
        debounce_frames: int = 8,
    ) -> None:
        """Initialize AutoPoseDetector.

        Args:
            model_path: Path to serialized Scikit-Learn .pkl model.
            min_confidence: Minimum prediction probability threshold (0.0 to 1.0).
            debounce_frames: Number of consecutive consistent frames required to switch pose.
        """
        if model_path is None:
            default_path = Path(__file__).resolve().parent / "models" / "pose_classifier.pkl"
            self.model_path = default_path
        else:
            self.model_path = Path(model_path)

        self.min_confidence = min_confidence
        self.debounce_frames = debounce_frames

        self.model: Optional[Any] = None
        self._load_model()

        # State tracking
        self.current_pose: Optional[str] = None
        self.candidate_pose: Optional[str] = None
        self.candidate_streak: int = 0
        self.last_confidence: float = 0.0
        self.is_transition: bool = True

    def _load_model(self) -> None:
        """Load model from disk."""
        if not self.model_path.exists():
            logger.warning("Classifier model not found at %s. Please train model first.", self.model_path)
            self.model = None
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info("AutoPoseDetector successfully loaded model from %s", self.model_path)
        except Exception as err:
            logger.error("Failed to load model from %s: %s", self.model_path, err)
            self.model = None

    def reset(self) -> None:
        """Reset state tracking."""
        self.current_pose = None
        self.candidate_pose = None
        self.candidate_streak = 0
        self.last_confidence = 0.0
        self.is_transition = True

    def update(
        self,
        landmarks_or_pts: Any,
    ) -> Tuple[Optional[str], float, bool]:
        """Process a new frame and update the active pose detection.

        Args:
            landmarks_or_pts: Either a `Landmarks` instance (with `.image` attribute)
                              or a numpy array of shape (33, 2) or (33, 3).

        Returns:
            Tuple of:
                - `pose_key`: Active confirmed pose (e.g. 'triangle') or None if idle.
                - `confidence`: Prediction confidence (0.0 to 1.0).
                - `is_transition`: True if transitioning or in idle stance.
        """
        if self.model is None:
            return None, 0.0, True

        # Extract landmark points array
        if hasattr(landmarks_or_pts, "image"):
            pts = landmarks_or_pts.image
        else:
            pts = np.asarray(landmarks_or_pts)

        if pts.shape[0] < 29:
            return self.current_pose, 0.0, True

        # 1. Extract 24-dimensional scale/position invariant features
        features = extract_feature_vector(pts)
        features_2d = features.reshape(1, -1)

        # 2. Predict probability distribution
        try:
            probs = self.model.predict_proba(features_2d)[0]
            best_id = int(np.argmax(probs))
            confidence = float(probs[best_id])
            predicted_label = ID_TO_LABEL.get(best_id, "idle")
        except Exception as err:
            logger.warning("Inference prediction error: %s", err)
            return self.current_pose, 0.0, True

        self.last_confidence = confidence

        # 3. Handle Idle / Transition
        if predicted_label == "idle" or confidence < self.min_confidence:
            # Drop candidate streak
            self.candidate_streak = 0
            self.candidate_pose = None
            self.is_transition = True
            return self.current_pose, confidence, True

        # 4. Temporal Debounce Buffer
        if predicted_label == self.candidate_pose:
            self.candidate_streak += 1
        else:
            self.candidate_pose = predicted_label
            self.candidate_streak = 1

        # Commit switch after debounce_frames
        if self.candidate_streak >= self.debounce_frames:
            if self.current_pose != predicted_label:
                logger.info("Auto-switched pose to: %s (conf: %.2f)", predicted_label, confidence)
                self.current_pose = predicted_label
            self.is_transition = False

        return self.current_pose, confidence, self.is_transition
