# 🧘 Real-Time Yoga Pose Detection & Visual Correction System

A real-time computer vision application that detects user yoga postures via webcam or video, evaluates alignment against anatomical rules using 3D world landmark geometry, and overlays **full-limb ghost skeletons** (in lime green) and **directional correction arrows** (in radiant orange) with angle badges to guide the practitioner directly into proper alignment.

---

## 🌟 Key Features

- **Full-Limb Target Ghost Overlay:** Shows the entire corrected limb chain (e.g. hip → knee → target ankle) in thick, high-visibility neon green alongside the user's actual limb in orange.
- **Directional Correction Guidance:** Prominent correction arrows point from actual joints to target waypoints with angle delta badges (e.g., `110° → 175°`).
- **Generalized Multi-Pose Bilateral Detection:** Automatically detects whether the user is performing the left-side or right-side variation of asymmetric poses (Triangle, Half Moon, Warrior II, Tree, and Dancer poses) with 8-frame temporal hysteresis to prevent flickering.
- **8 Dataset Poses + Extensible YAML Rules:** Fully calibrated against empirical ground truth from `Yoga_Poses-Dataset` (480+ annotated samples) in `configs/poses.yaml`.
- **Zero Jitter with One-Euro Filtering:** Dual-stage smoothing removes tracking noise while preserving rapid response.
- **Safety Prioritization:** Hyperextension and collapse prevention rules are flagged `safety: true` and ranked above minor deviations.
- **CPU Real-Time Performance:** Runs at 25–30 FPS on standard CPUs using MediaPipe Pose Landmarker without needing a discrete GPU.
- **Interactive Switching:** Switch between poses instantly via hotkeys (`1`–`8`, `9`, `0`) or cycle with `n` / `p`.

---

## 🧘 Supported Yoga Poses

The system supports all 8 primary postures from `Yoga_Poses-Dataset/TRAIN` plus traditional side-profile poses:

| # | Key | Sanskrit Name | English Name | View | Bilateral | Key Alignment Target |
| :-: | :--- | :--- | :--- | :---: | :---: | :--- |
| `1` | `triangle` | Trikonasana | **Triangle Pose** | Front | Yes | Both knees straight (175°), arms vertical stack (170°), side hinge |
| `2` | `ardhachandrasana` | Ardha Chandrasana | **Half Moon Pose** | Front | Yes | Standing leg straight (175°), lifted leg horizontal (175°), chest open |
| `3` | `baddhakonasana` | Baddha Konasana | **Bound Angle / Butterfly** | Front | Symmetric | Knees opened wide (35°), spine upright (70°), relaxed shoulders |
| `4` | `downward_dog` | Adho Mukha Svanasana | **Downward-Facing Dog** | Side | Symmetric | Knees straight (175°), arms straight (175°), inverted V hip hinge (68°) |
| `5` | `natarajasana` | Natarajasana | **Lord of the Dance** | Side | Yes | Standing leg straight (175°), kicked back foot (85°), chest proud |
| `6` | `utkatakonasana` | Utkata Konasana | **Goddess Pose** | Front | Symmetric | Deep knee squat (110°), upright vertical torso (105°), cactus arms (95°) |
| `7` | `veerabhadrasana` | Virabhadrasana II | **Warrior II Pose** | Front | Yes | Front knee bend (110°), back leg straight (175°), arms horizontal (175°) |
| `8` | `vrukshasana` | Vrikshasana | **Tree Pose** | Front | Yes | Standing leg straight (175°), knee turnout (55°), upright spine (175°) |
| `9` | `bhujangasana` | Bhujangasana | **Cobra Pose** | Side | Symmetric | Soft elbow bend (140°), spine extension & gentle chest lift (150°) |
| `0` | `dhanurasana` | Dhanurasana | **Bow Pose** | Side | Symmetric | Knees flexed & lifted (90°), chest lifted in backbend (155°) |

---

## 🏗️ Architecture

```
Camera / Video Stream (Threaded Capture)
   │
   ▼
PoseEstimator (MediaPipe PoseLandmarker in LIVE_STREAM mode)
   │
   ▼
LandmarkFilter (One-Euro Filter + Visibility Gating)
   │
   ▼
PoseEvaluator (Multi-Pose Bilateral Detection + YAML Rules vs 3D World Angles)
   │
   ▼
CorrectionPlanner (Hysteresis Smoothing + Rodrigues 2D/3D Limb Rotation)
   │
   ▼
PoseRenderer (Skeleton + Full-Limb Target Ghost + Directional Arrows + HUD)
```

---

## 📁 Repository Structure

```
c:\Yoga_F\
├── src\
│   ├── yogacorrect\           # Core Python package
│   │   ├── __init__.py
│   │   ├── __main__.py        # Package execution entry point
│   │   ├── types.py           # Landmarks, Rule, Error, Correction dataclasses
│   │   ├── geometry.py        # angle_deg, rotate_limb, batch_angles
│   │   ├── camera.py          # Low-latency threaded webcam reader
│   │   ├── estimator.py       # MediaPipe PoseLandmarker wrapper
│   │   ├── filters.py         # One-Euro filter + visibility gating
│   │   ├── evaluator.py       # Adaptive bilateral detection & error ranking
│   │   ├── planner.py         # Full-limb ghost skeleton & arrow planner
│   │   ├── renderer.py        # OpenCV rendering engine (ghosts, arrows, HUD)
│   │   ├── session.py         # Session alignment & hold timer
│   │   ├── calibration.py     # Neutral-stance bone length calibration
│   │   ├── reference.py       # Target pose reference panel
│   │   ├── perf.py            # Latency profiling & performance metrics
│   │   └── main.py            # CLI entry point
│   ├── configs\
│   │   ├── poses.yaml         # Complete pose definitions & joint angle rules
│   │   └── app.yaml           # Runtime & camera settings
│   ├── models\
│   │   └── pose_landmarker_lite.task
│   ├── tools\
│   │   ├── build_refs.py      # Extract empirical angles from datasets
│   │   ├── record_session.py  # Record landmark sessions for testing
│   │   └── replay.py          # Deterministic replay benchmark
│   └── tests\                 # Comprehensive automated test suite (33 tests)
│       ├── test_geometry.py
│       ├── test_evaluator.py
│       ├── test_filters.py
│       ├── test_planner.py
│       └── test_scenarios.py
├── docs\
│   └── pose_specs.md          # Comprehensive pose biomechanical specifications
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
# Launch with default pose (Triangle)
python -m yogacorrect.main

# Or launch directly into any supported pose
python -m yogacorrect.main --pose veerabhadrasana
python -m yogacorrect.main --pose downward_dog
python -m yogacorrect.main --pose vrukshasana
```

### 3. Run on a Video File

```bash
python -m yogacorrect.main --source "path/to/video.mp4" --no-mirror
```

### 4. Interactive Keyboard Shortcuts

| Key | Action |
| :---: | :--- |
| `1` | Switch to **Triangle Pose** (*Trikonasana*) |
| `2` | Switch to **Half Moon Pose** (*Ardha Chandrasana*) |
| `3` | Switch to **Bound Angle Pose** (*Baddha Konasana*) |
| `4` | Switch to **Downward-Facing Dog** (*Adho Mukha Svanasana*) |
| `5` | Switch to **Lord of the Dance** (*Natarajasana*) |
| `6` | Switch to **Goddess Pose** (*Utkata Konasana*) |
| `7` | Switch to **Warrior II Pose** (*Virabhadrasana II*) |
| `8` | Switch to **Tree Pose** (*Vrikshasana*) |
| `9` | Switch to **Cobra Pose** (*Bhujangasana*) |
| `0` | Switch to **Bow Pose** (*Dhanurasana*) |
| `n` | Cycle to **Next Pose** |
| `p` | Cycle to **Previous Pose** |
| `r` | Reset pose alignment hold timer |
| `f` | Toggle Fullscreen display |
| `q` or `ESC` | Quit application cleanly |

---

## 🧪 Running Tests

Run the full automated test suite (33 passing tests):

```bash
pytest src/tests -v
```

---

## 📊 Performance Benchmark

| Pipeline Stage | Target Latency | Measured (Standard CPU) | Status |
| :--- | :---: | :---: | :---: |
| Camera Capture & Transfer | < 15 ms | ~5–8 ms | ✅ Passed |
| MediaPipe Inference (Lite) | < 40 ms | ~20–26 ms | ✅ Passed |
| Filter + Geometry + Plan | < 5 ms | ~0.8–1.5 ms | ✅ Passed |
| Render & Display | < 20 ms | ~4–7 ms | ✅ Passed |
| **End-to-End Latency** | **< 150 ms** | **~32–42 ms (~28–30 FPS)** | **✅ Passed** |
