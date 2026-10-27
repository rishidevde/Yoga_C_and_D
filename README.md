# 🧘 Yoga Pose Detection & Visual Correction System

A real-time computer vision system that detects yoga poses from a webcam, compares posture against biomechanical targets in 3D world coordinates, and renders **ghost limbs and directional arrows** directly on screen to guide the user from their current posture to the correct posture.

---

## 🌟 Key Features

- **Intuitive Visual Guidance:** Overlays **ghost limbs** (in lime green) and **directional arrows** (in vibrant orange) with angle labels (e.g., `45° → 90°`), showing exactly where and how to move.
- **True 3D Angle Calculation:** Computes unsigned `[0°, 180°]` angles from 3D metric world coordinates using Rodrigues rotations, eliminating 2D orientation artifacts.
- **Zero Jitter with One-Euro Filtering:** Dual-stage smoothing removes tracking noise during still poses while remaining responsive during rapid movement.
- **Hysteresis Anti-Flicker:** State-machine hysteresis (5-frame activation, 8-frame exit) prevents false flashes and unstable arrows.
- **Extensible YAML Architecture:** Add new poses simply by adding rules to `configs/poses.yaml` — zero code changes required.
- **CPU Real-Time:** 25–30 FPS on standard CPUs using MediaPipe Pose Landmarker (no GPU required).
- **Interactive Pose Selection:** Switch between Triangle, Cobra, and Bow poses seamlessly via hotkeys (`1`, `2`, `3`).

---

## 🏗️ Architecture

```
Camera Thread (latest frame only)
   │
   ▼
PoseEstimator (MediaPipe PoseLandmarker in LIVE_STREAM mode)
   │
   ▼
LandmarkFilter (One-Euro filter + visibility gating)
   │
   ▼
PoseEvaluator (YAML rules vs 3D world angles → ranked errors)
   │
   ▼
CorrectionPlanner (Hysteresis + Rodrigues rotation → ghost points + arrows)
   │
   ▼
PoseRenderer (Skeleton + Ghost Limbs + Directional Arrows + HUD)
```

---

## 📁 Repository Structure

```
c:\Yoga_F\
├── src\
│   ├── yogacorrect\           # Core package
│   │   ├── __init__.py
│   │   ├── __main__.py        # Package execution entry
│   │   ├── types.py           # Landmarks, Rule, Error, Correction dataclasses
│   │   ├── geometry.py        # angle_deg, rotate_limb, batch_angles
│   │   ├── camera.py          # Low-latency threaded webcam reader
│   │   ├── estimator.py       # MediaPipe PoseLandmarker wrapper
│   │   ├── filters.py         # One-Euro filter + visibility gate
│   │   ├── calibration.py     # Neutral-stance bone length calibration
│   │   ├── evaluator.py       # YAML rule evaluator & safety prioritization
│   │   ├── planner.py         # Ghost limb planner & hysteresis
│   │   ├── renderer.py        # OpenCV visualizer (ghosts, arrows, HUD)
│   │   ├── reference.py       # Target pose reference panel
│   │   ├── session.py         # Session alignment & hold timer
│   │   ├── perf.py            # Latency profiling & p50/p95 percentiles
│   │   └── main.py            # CLI entry point
│   ├── configs\
│   │   ├── poses.yaml         # Poses and joint rules (extensible)
│   │   └── app.yaml           # App configuration
│   ├── models\
│   │   └── pose_landmarker_lite.task
│   ├── tools\
│   │   ├── build_refs.py      # Extract empirical angles from datasets
│   │   ├── record_session.py  # Record landmark sessions for testing
│   │   └── replay.py          # Deterministic replay benchmark
│   └── tests\                 # Comprehensive pytest test suite (26 tests)
│       ├── test_geometry.py
│       ├── test_evaluator.py
│       ├── test_filters.py
│       ├── test_planner.py
│       └── test_scenarios.py
├── docs\
│   └── pose_specs.md          # Pose biomechanical specifications
├── task.md                    # Project task tracker
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd c:\Yoga_F
pip install -r src/requirements.txt
pip install -e src
```

### 2. Run the Live Webcam App

```bash
python -m yogacorrect.main
```

### 3. Run on a Video File

```bash
python -m yogacorrect.main --source "YogaPoseCorrector/Triangle_Pose.mp4"
```

### 4. Interactive Keyboard Shortcuts

| Key | Action |
| :---: | :--- |
| `1` | Switch to **Triangle Pose** (*Trikonasana*) |
| `2` | Switch to **Cobra Pose** (*Bhujangasana*) |
| `3` | Switch to **Bow Pose** (*Dhanurasana*) |
| `r` | Reset pose alignment hold timer |
| `q` or `ESC` | Quit application |

---

## 🧪 Running Tests

Run the full automated test suite (26 passing tests):

```bash
pytest src/tests -v
```

---

## 📊 Performance Benchmark

| Pipeline Stage | Target Latency | Measured (Intel CPU) | Status |
| :--- | :---: | :---: | :---: |
| Camera Capture & Transfer | < 15 ms | ~5–8 ms | ✅ Passed |
| MediaPipe Inference (Lite) | < 40 ms | ~22–28 ms | ✅ Passed |
| Filter + Geometry + Plan | < 5 ms | ~0.8–1.5 ms | ✅ Passed |
| Render & Display | < 20 ms | ~4–7 ms | ✅ Passed |
| **End-to-End Latency** | **< 150 ms** | **~35–45 ms (~28 FPS)** | **✅ Passed** |
