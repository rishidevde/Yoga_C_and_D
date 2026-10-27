"""Yoga Pose Detection and Correction System.

Real-time webcam application that detects yoga poses and provides
visual correction feedback using ghost limbs and directional arrows.

Modules:
    types       - Core data structures (Landmarks, Rule, Error, Correction)
    geometry    - Angle computation and limb rotation
    camera      - Threaded webcam capture
    estimator   - MediaPipe PoseLandmarker wrapper
    evaluator   - Rule-based pose evaluation
    planner     - Ghost point and correction planning
    renderer    - OpenCV drawing and UI
    main        - Application entry point
"""

__version__ = "0.1.0"
