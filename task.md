# Yoga Pose Detection & Correction — Task Tracker

> **Tech Stack:** Python 3.12 | MediaPipe | OpenCV | NumPy | PyYAML  
> **Window:** Oct 27 – Nov 30, 2026  
> **Status:** 🔴 Not Started · 🟡 In Progress · 🟢 Done

---

## Phase 1 — Vertical Slice (Oct 27–28)
> Goal: Camera → MediaPipe → Triangle pose → ONE working ghost limb + arrow

### 1.0 Project Setup
- [x] 🟢 Create folder structure (`src/yogacorrect/`, `configs/`, `models/`, `assets/`, `data/`, `tools/`, `tests/`)
- [x] 🟢 Create `requirements.txt` (mediapipe, opencv-python, numpy, pyyaml, pytest)
- [x] 🟢 Create `pyproject.toml` with package config
- [x] 🟢 Create `src/yogacorrect/__init__.py`
- [x] 🟢 Install dependencies (`pip install -r requirements.txt`)
- [x] 🟢 Download MediaPipe `pose_landmarker_lite.task` model into `models/`

### 1.1 Types Module
- [x] 🟢 Create `src/yogacorrect/types.py`
- [x] 🟢 Define `Landmarks` dataclass (timestamp, img coords, world coords, visibility)
- [x] 🟢 Define `Rule` dataclass (name, kind, joints, target, tol, weight, safety, cues, hint_axis)
- [x] 🟢 Define `Error` dataclass (rule, value, delta, score)
- [x] 🟢 Define `Correction` dataclass (joint_idx, current_px, ghost_px, ghost_joint_px, label, cue)
- [x] 🟢 Define MediaPipe landmark name → index mapping dict (`LANDMARK_NAMES`, `POSE_CONNECTIONS`)

### 1.2 Geometry Module
- [x] 🟢 Create `src/yogacorrect/geometry.py`
- [x] 🟢 Implement `angle_deg(a, b, c)` — unsigned 0–180° angle at vertex b
- [x] 🟢 Implement `rotate_limb(a, b, c, target_deg, hint_axis)` — Rodrigues rotation → ghost point
- [x] 🟢 Implement `batch_angles(world_landmarks, triplets)` — vectorized angle computation for all rules
- [x] 🟢 Create `tests/test_geometry.py`
- [x] 🟢 Test: angle_deg with known angles (90°, 180°, 45°, 60°)
- [x] 🟢 Test: angle_deg with zero-length vectors → returns NaN
- [x] 🟢 Test: angle_deg with collinear points → returns 0° or 180°
- [x] 🟢 Test: rotate_limb output angle equals target
- [x] 🟢 Test: rotate_limb preserves bone length
- [x] 🟢 Test: rotate_limb collinear case with/without hint_axis
- [x] 🟢 Run tests and all pass ✓

### 1.3 Camera Module
- [x] 🟢 Create `src/yogacorrect/camera.py`
- [x] 🟢 Implement `CameraCapture` class with threaded capture loop
- [x] 🟢 Size-1 queue (drop old frames, always latest)
- [x] 🟢 Configure 640×480 resolution, MJPG format, buffer size 1
- [x] 🟢 `get_frame()` method returns latest frame + timestamp
- [x] 🟢 `release()` method for cleanup
- [x] 🟢 Test: camera and video file reading verified

### 1.4 Estimator Module
- [x] 🟢 Create `src/yogacorrect/estimator.py`
- [x] 🟢 Implement `PoseEstimator` class wrapping MediaPipe PoseLandmarker Task API
- [x] 🟢 Configure LIVE_STREAM and VIDEO modes with tracking
- [x] 🟢 Extract 33 landmarks: image coords (normalized x,y,z) + world coords (metres) + visibility
- [x] 🟢 Pack results into `Landmarks` dataclass
- [x] 🟢 Handle "no person detected" case gracefully
- [x] 🟢 Test: estimator processes video frames and returns landmarks

### 1.5 Config — Triangle Pose Rules
- [x] 🟢 Create `configs/poses.yaml`
- [x] 🟢 Define Triangle pose rules with target values:
  - [x] 🟢 `front_knee_straight` — left_hip/left_knee/left_ankle (175° ± 12°)
  - [x] 🟢 `back_knee_straight` — right_hip/right_knee/right_ankle (175° ± 12°)
  - [x] 🟢 `top_arm_vertical` — left_shoulder/right_shoulder/right_wrist (170° ± 15°)
  - [x] 🟢 `bottom_arm_alignment` — right_shoulder/left_shoulder/left_wrist (170° ± 15°)
  - [x] 🟢 `torso_lateral_bend` — left_shoulder/left_hip/left_knee (140° ± 20°)
- [x] 🟢 Create `configs/app.yaml` with camera, model path, UI defaults

### 1.6 Evaluator Module
- [x] 🟢 Create `src/yogacorrect/evaluator.py`
- [x] 🟢 Implement `load_pose_rules(yaml_path, pose_name)` → list of `Rule` objects
- [x] 🟢 Implement `evaluate(landmarks, rules)`:
  - [x] 🟢 Skip rules where needed joints have visibility < min_visibility
  - [x] 🟢 Compute angle for each rule using 3D world coordinates
  - [x] 🟢 Compare against target ± tolerance
  - [x] 🟢 Compute score = weight × |error| / tolerance
  - [x] 🟢 Return ranked list of `Error` objects (safety rules first, then by score)
  - [x] 🟢 Limit to top 2 failing rules (no contradictory arrows on shared joints)
- [x] 🟢 Create `tests/test_evaluator.py` (6 passing tests)

### 1.7 Planner Module
- [x] 🟢 Create `src/yogacorrect/planner.py`
- [x] 🟢 Implement `plan(errors, landmarks, frame_width, frame_height)`:
  - [x] 🟢 For each error: call `rotate_limb()` → get ghost point in world coords
  - [x] 🟢 Project world ghost point → screen pixel coordinates preserving bone length
  - [x] 🟢 Build `Correction` object with current_px, ghost_px, angle label, cue text
- [x] 🟢 Choose cue_low or cue_high based on sign of delta
- [x] 🟢 Return list of `Correction` objects
- [x] 🟢 Create `tests/test_planner.py` (passing tests)

### 1.8 Renderer Module
- [x] 🟢 Create `src/yogacorrect/renderer.py`
- [x] 🟢 Implement `draw_skeleton(frame, landmarks)` — draw user's live skeleton (anti-aliased lines)
- [x] 🟢 Implement `draw_corrections(frame, corrections)`:
  - [x] 🟢 Draw ghost limb: lime green dashed line (joint → ghost point)
  - [x] 🟢 Draw arrow: vibrant orange arrow (current limb end → ghost point)
  - [x] 🟢 Draw angle label badge: "45° → 90°" near arrow midpoint
  - [x] 🟢 Draw text cue banner at bottom of screen
- [x] 🟢 Draw "Pose Aligned! Keep holding" when all rules pass
- [x] 🟢 Draw top HUD with pose title, status, and FPS counter
- [x] 🟢 Support horizontal mirroring for webcam display

### 1.9 Main Entry Point
- [x] 🟢 Create `src/yogacorrect/main.py`
- [x] 🟢 Parse command-line args (source, pose, configs, no-mirror)
- [x] 🟢 Initialize: Camera → Estimator → load rules → Evaluator → Planner → Renderer
- [x] 🟢 Main loop: capture → estimate → evaluate → plan → render → display
- [x] 🟢 Handle keyboard controls ('1', '2', '3' for pose switching, 'r' for reset, 'q'/ESC to quit)
- [x] 🟢 Clean shutdown (release camera, close windows)
- [x] 🟢 **MILESTONE: Triangle pose verified with working ghost arrows on video** ✓

---

## Phase 2 — Stability & Multi-Pose (Oct 29 – Nov 3)
> Goal: Smooth, stable corrections for Triangle + Cobra + Bow

### 2.1 Filters Module
- [x] 🟢 Create `src/yogacorrect/filters.py`
- [x] 🟢 Implement `OneEuroFilter` class (min_cutoff=1.0, beta=5.0, d_cutoff=1.0)
- [x] 🟢 Implement `LandmarkFilter` — applies One-Euro to all 33 landmarks
- [x] 🟢 Visibility gating integrated into `evaluator.py`
- [x] 🟢 Create `tests/test_filters.py` (3 passing tests)

### 2.2 Planner v2 — Hysteresis
- [x] 🟢 Add hysteresis state tracking to `planner.py`:
  - [x] 🟢 Rule becomes "failing" only after error > tol for N=5 consecutive frames
  - [x] 🟢 Rule becomes "ok" only after error < 0.7×tol for M=8 consecutive frames
  - [x] 🟢 Tested in `test_planner.py`

### 2.3 Calibration Module
- [x] 🟢 Create `src/yogacorrect/calibration.py`
- [x] 🟢 Implement neutral-stance capture with motion detection and visibility checks
- [x] 🟢 Compute per-segment bone lengths (upper arm, forearm, thigh, shin, torso)
- [x] 🟢 `CalibrationProfile` dataclass

### 2.4 Multi-Pose Support
- [x] 🟢 Add Cobra (Bhujangasana) rules to `poses.yaml`:
  - [x] 🟢 Elbow angle, spine extension, chest lift
- [x] 🟢 Add Bow (Dhanurasana) rules to `poses.yaml`:
  - [x] 🟢 Knee bend, chest lift
- [x] 🟢 Interactive pose switching via hotkeys ('1', '2', '3') in `main.py`

### 2.5 Build Reference Tool
- [x] 🟢 Create `tools/build_refs.py`
- [x] 🟢 Read raw landmark CSV from `Yoga_Poses-Dataset/Results/Dataset_Triangle.csv`
- [x] 🟢 Recompute angles as unsigned 0–180° using `angle_deg()`
- [x] 🟢 Compute median target + scaled MAD tolerance
- [x] 🟢 Successfully executed and verified on Triangle dataset

---

## Phase 3 — Full UI & Polish (Nov 4–10)
> Goal: Reference skeleton, session management, performance profiling

### 3.1 & 3.2 Reference Panel
- [x] 🟢 Create `src/yogacorrect/reference.py`
- [x] 🟢 Canonical reference poses for Triangle, Cobra, and Bow
- [x] 🟢 `ReferencePanel` renderer

### 3.3 Session Module
- [x] 🟢 Create `src/yogacorrect/session.py`
- [x] 🟢 Track active pose, hold timer, alignment count

### 3.4 Performance Module
- [x] 🟢 Create `src/yogacorrect/perf.py`
- [x] 🟢 Sliding window latency profiler (capture, inference, evaluate, plan, render)
- [x] 🟢 Compute rolling p50 and p95 latencies

---

## Phase 4 — Testing & Documentation (Nov 11–18)
> Goal: Comprehensive testing, replay system, documentation

### 4.1 Record & Replay Tools
- [x] 🟢 Create `tools/record_session.py` — record landmarks + timestamps
- [x] 🟢 Create `tools/replay.py` — deterministic session replay

### 4.2 Scenario Testing
- [x] 🟢 Create `tests/test_scenarios.py` (3 comprehensive tests):
  - [x] 🟢 Bent front knee triggers knee correction
  - [x] 🟢 Dropped top arm triggers top arm correction
  - [x] 🟢 Dynamic YAML extensibility test passes

### 4.4 Documentation
- [x] 🟢 Create `docs/pose_specs.md` — specifications, joint angles, cues
- [x] 🟢 Create `README.md` — system overview, setup, usage instructions

---

---

## Phase 5 — Visual Feedback Overhaul (Oct 27)
> Goal: Full-limb ghost skeleton, arm rule detection fix, multi-cue HUD

### 5.1 Detection Fix — Arm Rule Suppression Bug
- [x] 🟢 Fixed `used_joints` conflict filter in `evaluator.py` — now only tracks **distal joint** (`joints[2]`) not all joints; prevents arm rule being blocked by knee rules sharing shoulder joints
- [x] 🟢 Increased `max_errors` from 2 → 3 in `PoseEvaluator` default and `app.yaml`
- [x] 🟢 `top_arm_vertical_mirror` rule now correctly fires (val=96-103° vs target=170°) when arm is brought down

### 5.2 Full-Limb Ghost Skeleton
- [x] 🟢 Added `target_chain_px` and `actual_chain_px` to `Correction` dataclass (`types.py`)
- [x] 🟢 Updated `planner.py` to build full `[a_px, b_px, ghost_px]` / `[a_px, b_px, c_px]` chains
- [x] 🟢 `renderer.draw_corrections()` now draws full polyline ghost limb (green) + full actual limb (orange)
- [x] 🟢 Removed tiny single-segment ghost line — now full hip→knee→target_ankle chain visible
- [x] 🟢 Full-limb rendering works for all poses and any future YAML-added poses

### 5.3 Visual Improvements
- [x] 🟢 **Full actual limb** in orange polyline (hip→knee→actual_ankle) — clear "where you are"
- [x] 🟢 **Full target limb** in bright green polyline (hip→knee→target_ankle) — clear "where to go"
- [x] 🟢 Badge stagger — angle badges offset by correction index to prevent overlap
- [x] 🟢 **Two-line HUD** banner: Priority 1 (gold, large) + Priority 2 (blue-white, smaller)
- [x] 🟢 Target endpoint: large green filled circle with white bullseye
- [x] 🟢 Current endpoint: large orange filled circle with white bullseye
- [x] 🟢 Directional arrow from current → target endpoint (thick 9px outline + 5px orange core)

### 5.4 Tests & Verification
- [x] 🟢 **26 / 26 tests still pass** after all changes
- [x] 🟢 Visual smoke test on `pose-2.mp4`: all 3 corrections detected at sec 13-15 (`front knee`, `back leg`, `top arm`)

---

## Summary of Progress
- **All Core Modules Implemented & Tested**: 100% Python (`yogacorrect` package)
- **Automated Tests**: **30 / 30 Passing Tests** across `geometry`, `evaluator`, `planner`, `filters`, `scenarios`, and bilateral hysteresis.
- **Extensibility**: Fully YAML-driven (`configs/poses.yaml`).
- **Live & Video Verified**: Tested on actual yoga video — 3 simultaneous corrections correctly detected and visualized as full-limb ghost overlays.
- **Arm detection fixed**: `top_arm_vertical` rule now fires correctly alongside knee corrections.
- **Full-limb visualization**: Orange actual limb + green target limb gives clear body-wide guidance from any distance.

---

## Phase 6 — Bilateral Swap Stability & Torso Clarity (pose3.mp4 fixes)

### 6.1 Bug: Wrong hand at sec 8 — bilateral swap misfires
- [x] ✅ Added `MIN_SWAP_THRESHOLD = 0.15` — swap only fires when `abs(rwy - lwy) > 0.15`
- [x] ✅ Added `SIDE_HYSTERESIS = 8` frames — side only commits after 8 consecutive clear frames
- [x] ✅ Ambiguous zone (`diff < 0.15`) now **preserves** prior committed side without resetting counters
- [x] ✅ Added `reset_side_detection()` method to clear bilateral state on pose change
- [x] ✅ `_swap_state_frames`, `_noswap_state_frames`, `_active_side` state properly initialised in `__init__`

### 6.2 Bug: No corrections at sec 12–20 — hysteresis silenced by oscillating rule names
- [x] ✅ Stable side (bilateral fix above) prevents rule name from alternating `front_knee_straight` ↔ `_mirror` every frame
- [x] ✅ Reduced `hysteresis_fail_frames: 4 → 2` in `app.yaml` — corrections appear faster

### 6.3 Bug: "70° → 140°" torso cue is confusing
- [x] ✅ Torso/bend rules now render **only pivot→distal** (hip→target_knee) — no more body-sized triangle
- [x] ✅ Detection: `"torso" in rule.name or "bend" in rule.name` — automatic for any future YAML torso rule
- [x] ✅ Improved cue: `"Lean your torso further sideways — reach longer through your top arm"` / `"Stand taller from the hip — reduce the side bend"`

### 6.4 Tests
- [x] ✅ Added `TestBilateralSwapHysteresis` class with 4 new tests covering all hysteresis edge cases
- [x] ✅ **30 / 30 tests passing** after all Phase 6 changes
