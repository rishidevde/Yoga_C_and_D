"""Pose Classifier & Auto-Detector Package.

Provides scale- and position-invariant feature extraction, lightweight ML models,
HPC training pipelines (PyTorch + CUDA / Scikit-Learn), and real-time temporal
debounce inference for automatic yoga pose detection.
"""

from pose_classifier.features import extract_feature_vector, FEATURE_NAMES
from pose_classifier.inference import AutoPoseDetector

__all__ = ["extract_feature_vector", "FEATURE_NAMES", "AutoPoseDetector"]
