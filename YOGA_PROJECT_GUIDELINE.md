# Yoga Pose Detection and Visual Correction: Project Guideline

Course: CSIT334 Human Computer Interaction. Window: Sep 24 to Nov 30, 2026.
This is the single reference for the whole build: what we make, how it works, folder layout, module contracts, algorithms, speed, testing and schedule.

Status labels used below: **[validated]** means the code was run against test cases while writing this guideline. **[target]** means a goal we must measure ourselves. Nothing marked [target] has been measured yet.

---

## 1. What we are building

A real-time webcam app that watches a person do a yoga pose and shows exactly how to fix it.

Instead of only a skeleton or joint-angle numbers, the screen shows:
1. The user's live skeleton.
2. A **ghost limb** at the correct angle, sized to the user's own bone lengths.
3. An **arrow** from the current joint position to the ghost joint position, labelled with the angle change (for example 45 to 90 degrees).
4. One short text cue ("Raise your right arm").
5. A second panel with an animated correct-pose reference scaled to the user.

Scope:
- Static poses with full correction: **Bhujangasana (Cobra), Dhanurasana (Bow), Trikonasana (Triangle)**.
- **Surya Namaskar**: recognize which of the 12 stages the user is in and advance automatically (no deep per-stage correction).
- Everything runs locally. No video is stored or uploaded.

Not in scope: medical advice, injury diagnosis, multi-person, a custom-trained deep network.

## 2. Approach in one paragraph

MediaPipe Pose Landmarker (pretrained) gives 33 body landmarks per frame. We smooth them, compute joint angles from 3D world landmarks, compare against per-pose angle ranges stored in a YAML file, rank the errors, and for the top one or two failing joints rotate the limb to the target angle to get the ghost point and arrow. **No network training is needed.** The only learning-like part is optional (Surya stage recognition can use a tiny k-NN or random forest, or just a rule-based state machine).

## 3. End-to-end real-time cycle

```
Camera thread (latest frame only)
   -> Inference thread: MediaPipe PoseLandmarker (LIVE_STREAM, async)
   -> Landmark post-process: visibility gate + One-Euro smoothing
   -> Angle engine: 3D angles (2D fallback), vectorized
   -> Evaluator: rules vs poses.yaml -> ranked errors
   -> Correction planner: ghost point, arrow, cue, dead zone + hysteresis
   -> Renderer (main thread): two-panel OpenCV UI
   -> next frame
```

Design rules:
- Three threads: capture, inference, render. Queues have size 1 and drop old items, so the display never shows stale data.
- Run inference on the **unflipped** frame. Mirror only for display (draw at x -> 1 - x). This avoids left/right label bugs.
- Each frame carries a capture timestamp, so we can measure real camera-to-screen latency.

## 4. Folder structure

```
yoga-pose-correction/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── poses.yaml              # per-pose joints, targets, tolerances, cues, view
│   ├── surya.yaml              # 12 stages: angle signature + preview keyframes
│   └── app.yaml                # camera res, model path, thresholds, UI
├── models/
│   └── pose_landmarker_lite.task   # downloaded from MediaPipe docs (lite/full)
├── assets/
│   └── reference_keyframes/    # reference skeleton keyframes per pose/stage (json)
├── data/
│   ├── raw/                    # downloaded datasets (Manoj dataset etc.), gitignored
│   ├── recordings/             # recorded landmark/video sessions (consented), gitignored
│   └── labels/                 # instructor labels (csv)
├── src/yogacorrect/
│   ├── __init__.py
│   ├── main.py                 # entry point, loop, pose selection
│   ├── types.py                # dataclasses: Landmarks, Rule, Error, Correction
│   ├── camera.py               # threaded capture, latest-frame-only
│   ├── estimator.py            # MediaPipe wrapper (LIVE_STREAM)
│   ├── filters.py              # One-Euro filter, visibility gate
│   ├── geometry.py             # angle_deg, rotate_limb, batch angles, projection
│   ├── calibration.py          # neutral-stance bone lengths
│   ├── evaluator.py            # rules -> errors, ranking, safety override
│   ├── planner.py              # ghost point, arrow, cue, hysteresis
│   ├── renderer.py             # two-panel UI, ghost, arrows, text
│   ├── reference.py            # Bezier-interpolated reference skeleton
│   ├── surya_fsm.py            # 12-stage state machine
│   ├── session.py              # state: selected pose, mode, timers
│   └── perf.py                 # per-stage timing, p50/p95 logging
├── tools/
│   ├── build_refs.py           # dataset CSVs -> clean 0-180 deg ranges -> poses.yaml
│   ├── record_session.py       # record landmarks (+ optional video) with consent
│   ├── replay.py               # replay recorded landmarks through the pipeline
│   ├── label_tool.py           # interval-sampled frames for instructor labelling
│   └── eval_accuracy.py        # accuracy, Cohen's kappa, calibrated vs generic
├── tests/
│   ├── test_geometry.py
│   ├── test_filters.py
│   ├── test_evaluator.py
│   ├── test_planner.py
│   ├── test_surya_fsm.py
│   └── fixtures/               # synthetic and recorded landmark sets
└── docs/
    ├── GUIDELINE.md            # this file
    ├── pose_specs.md           # key joints, mistakes, cues per pose
    └── evaluation.md           # method + results
```

Requirements (versions to be pinned after install; check mediapipe wheel support for your Python version): `mediapipe`, `opencv-python`, `numpy`, `pyyaml`, `pytest`; tools only: `pandas`, `scikit-learn`.

## 5. Data structures (`types.py`)

```python
@dataclass
class Landmarks:
    t: float                 # capture timestamp (perf_counter)
    img: np.ndarray          # (33,3) normalized image coords x,y,z
    world: np.ndarray        # (33,3) metric world coords (metres, hip-centred)
    vis: np.ndarray          # (33,) visibility 0..1

@dataclass
class Rule:
    name: str                # e.g. "front_knee_straight"
    kind: str                # "angle" | "axis_angle" | "depth_diff" | "ratio"
    joints: tuple            # landmark names, e.g. (hip, knee, ankle); vertex = middle
    target: float            # degrees (or ratio)
    tol: float               # degrees
    weight: float            # ranking weight
    safety: bool             # safety rules override ranking
    cue_low: str             # text if value < target
    cue_high: str            # text if value > target
    hint_axis: tuple | None  # rotation axis when limb is collinear

@dataclass
class Error:
    rule: Rule
    value: float
    delta: float             # signed: target - value
    score: float             # ranking score

@dataclass
class Correction:
    joint_idx: int
    from_px: tuple
    to_px: tuple
    label: str               # "45 -> 90"
    cue: str
```

## 6. Config format (`poses.yaml`)

All numbers below are **placeholders** to show the shape. Real values are generated by `tools/build_refs.py` from data and then checked by an instructor.

```yaml
trikonasana:
  view: front              # supported camera view
  min_visibility: 0.6
  hold_seconds: 3
  rules:
    - name: front_knee_straight
      kind: angle
      joints: [left_hip, left_knee, left_ankle]
      target: 175          # placeholder
      tol: 8
      weight: 1.0
      safety: true
      cue_low: "Straighten your front knee"
      cue_high: "Soften your front knee"
    - name: torso_not_twisted
      kind: depth_diff     # |z(left_shoulder) - z(right_shoulder)| in world units
      joints: [left_shoulder, right_shoulder]
      target: 0.0
      tol: 0.06            # placeholder
      weight: 0.8
      cue_high: "Open your chest toward the front"
```

Supported rule kinds:
- `angle`: unsigned angle at the middle joint (0 to 180).
- `axis_angle`: angle between a segment and a world axis (for example spine vs vertical).
- `depth_diff`: difference in world z of two joints (detects torso twist toward the camera).
- `ratio`: normalized distance (for example hand-to-ankle over leg length).

## 7. Core algorithms

### 7.1 Landmarks and indices
Use MediaPipe indices: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, heels 29/30, foot index 31/32. Names map to indices in `types.py` once.

### 7.2 Angles (`geometry.py`) [validated]

```python
EPS = 1e-9

def angle_deg(a, b, c):
    """Unsigned angle at b (0..180) formed by a-b-c. 2D or 3D."""
    u = np.asarray(a, float) - np.asarray(b, float)
    v = np.asarray(c, float) - np.asarray(b, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))
```

Always unsigned 0 to 180. The dataset's original 0 to 360 angles depend on which side faces the camera and must not be used.

For speed, compute all rule angles in one vectorized call: stack the triplets into index arrays `A, B, C`, then `u = W[A]-W[B]`, `v = W[C]-W[B]`, and use `einsum` for dot products and norms. One NumPy pass for all rules.

### 7.3 Ghost limb: rotate the distal segment to the target angle [validated]

```python
def rotate_limb(a, b, c, target_deg, hint_axis=None):
    """Return C' with angle(a,b,C') == target_deg and |BC'| == |BC|."""
    a, b, c = (np.asarray(p, float) for p in (a, b, c))
    u, v = a - b, c - b
    cur = angle_deg(a, b, c)
    n = np.cross(u, v)
    if np.linalg.norm(n) < 1e-6:          # limb collinear: plane undefined
        if hint_axis is None:
            return None
        n = np.asarray(hint_axis, float)
    n = n / np.linalg.norm(n)
    phi = math.radians(target_deg - cur)
    v2 = v * math.cos(phi) + np.cross(n, v) * math.sin(phi)   # Rodrigues, n perpendicular to v
    return b + v2
```

Checked: 45 to 90 and 20/90/135 to 45/90/170 all land on the target angle, bone length is preserved, the collinear case returns None without a hint and works with one, and zero-length vectors return NaN safely.

### 7.4 World-3D to screen
Do the rotation in world coordinates (metric, stable), then map the new point to pixels:
1. Pixel scale `s = |shoulder_mid - hip_mid|_px / |shoulder_mid - hip_mid|_world`.
2. `to_px = joint_px + s * (C'_world - joint_world)[x, y]` (flip y sign if world y points up in your convention; verify on a sample).
3. Draw the ghost segment from `joint_px` to `to_px` and the arrow from the current limb end to `to_px`.

Caveat: this is an approximation because the camera is a perspective projection. It is fine for guidance and must be stated in the limitations.

### 7.5 Smoothing: One-Euro filter [validated]

```python
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=5.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-6)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x
```

Tuning result on normalized (0 to 1) coordinates at 30 FPS: with `min_cutoff=1.0, beta=5`, still-signal noise dropped from 0.010 to about 0.004, and tracking lag stayed under 0.02 at 1 unit/s motion. The common default `beta=0.007` was too low for this scale (lag 0.15). Re-tune on real webcam recordings. Use one filter per landmark (world and image), or filter all landmarks as one array.

### 7.6 Evaluator (`evaluator.py`)
For each rule: skip it if any needed joint has visibility below `min_visibility` (never guess). Else compute `value`, then `delta = target - value`, `err = |delta|`.
- Rule passes if `err <= tol`.
- Ranking score: `weight * err / tol`. Safety rules are listed first regardless of score.
- Return the top 2 failing rules. Never show contradictory arrows for connected joints: if two failing rules share the same joint, keep the higher-ranked one.

### 7.7 Planner: stability (`planner.py`)
Raw per-frame errors flicker. Use hysteresis:
- A rule becomes "failing" only after `err > tol` for N consecutive frames (start with N=5).
- It becomes "ok" only after `err < 0.7 * tol` for M frames (start with M=8).
- Arrows are hidden while within the dead zone (`err <= tol`).
- Smooth the ghost endpoint with the same One-Euro filter so the arrow does not jitter.
- Pose "aligned" requires all rules ok for `hold_seconds` (Bow needs a minimum hold; Triangle needs the front knee not overbent).

### 7.8 Calibration (`calibration.py`)
The user stands neutral facing the camera for about 3 seconds. Average per-segment world lengths (upper arm, forearm, thigh, shin, torso) over the stable frames and keep the median. Use these lengths to size ghost limbs and the reference panel, because single-frame lengths are foreshortened. Reject calibration if visibility is low or motion is high, and ask to repeat.

### 7.9 Reference panel (`reference.py`)
Store reference keyframes per pose (joint positions, normalized to torso). Animate between keyframes with a cubic Bezier ease-in-out (`t' = 3t^2 - 2t^3` for simple smoothstep, or a full cubic Bezier if desired). Scale the reference skeleton to the user's calibrated bone lengths and draw it in the right panel.

### 7.10 Surya Namaskar state machine (`surya_fsm.py`)
The 12 stages in order: Pranamasana, Hastauttanasana, Hasta Padasana, Ashwa Sanchalanasana, Parvatasana, Ashtanga Namaskara, Bhujangasana, Parvatasana, Ashwa Sanchalanasana, Hasta Padasana, Hastauttanasana, Pranamasana.
- Each stage has an angle signature (joint, target, tolerance) in `surya.yaml`.
- Per frame: compute a match score for the expected next stage (and the current one).
- Advance only if the next stage matches for K frames **and** mean joint velocity is below a threshold (so we don't trigger on transitions).
- Show a preview of the next stage and a peripheral cue.
- Fatigue timeout: if one stage is held beyond a limit, prompt the user.
- Skip cascade: if the user skips ahead, fall back to a tutorial screen and resume from the detected stage.
- If rules are not reliable enough, add a small k-NN or random forest on joint angles. It trains in seconds on CPU and needs no deep learning.

## 8. Making it fast (and faster)

Biggest wins first.

1. **Use LIVE_STREAM (or VIDEO) mode, never IMAGE mode.** In video modes MediaPipe tracks from the previous frame instead of re-detecting every frame. This is the single biggest speed lever.
2. **Lite model first** (`pose_landmarker_lite`), move to full only if accuracy needs it. `num_poses=1`. Do not request segmentation masks.
3. **Capture at 640x480 (or 960x540).** The model resizes internally, so higher resolution only adds copy cost. Request MJPG from the webcam if it supports it, and set the capture buffer size to 1.
4. **Threaded pipeline with size-1 queues**, dropping stale frames. Never let rendering block capture.
5. **Async inference** (LIVE_STREAM callback) so the render thread keeps drawing at display rate with the latest result.
6. **One vectorized angle pass** for all rules; preallocated `float32` arrays; no per-frame Python loops over landmarks.
7. **Drawing cost:** draw the ghost and arrows on a single overlay and blend once; avoid full-frame copies more than once per frame; use anti-aliased lines only for the few arrows that matter.
8. **Skip work that does not change:** evaluate rules only when new landmarks arrive; render the cached overlay if no new result.
9. **Hardware:** the GPU delegate may help depending on platform and build (check the MediaPipe docs for your OS). On CPU, keep other heavy apps closed.
10. **Browser option (later):** MediaPipe's JavaScript version with WebGL is often the fastest end to end if this ever becomes a web app. Not needed for the course version.
11. **Measure everything** (`perf.py`): timestamps at capture, inference done, evaluate done, render done. Log p50 and p95 per stage and end-to-end. Optimize only what the numbers show.

Latency budget, **[target]**, to be replaced by measured values:

| Stage | Target |
|---|---|
| Capture and transfer | under 15 ms |
| MediaPipe inference (lite) | under 40 ms |
| Smoothing + angles + evaluate + plan | under 5 ms |
| Render + display | under 20 ms |
| End to end | under 150 ms at 25+ FPS |

Develop without a webcam by recording landmarks once (`record_session.py`) and replaying (`replay.py`). That makes tests and benchmarks deterministic and repeatable.

## 9. Pose specifications to write first (`docs/pose_specs.md`)

For each pose list: camera view, required landmarks, 3 to 6 rules, common mistakes, cues.
- **Trikonasana (front view):** front knee not overbent, legs apart, torso lateral bend, shoulders stacked (depth rule), top arm vertical.
- **Bhujangasana (side view):** elbow angle and spine extension in a safe range, hips down, shoulders away from ears.
- **Dhanurasana (side view):** knee and hip angles together, chest lift, hold duration.

Side-view poses make 2D unreliable, so these rely on world landmarks. State this in the limitations.

## 10. Building references from data

1. `tools/build_refs.py` reads the Manoj dataset landmark CSVs (33 landmarks x, y, z, visibility).
2. Recomputes angles with `angle_deg` (unsigned 0 to 180) on world-like or normalized coordinates, normalizes for the facing side, drops low-visibility rows.
3. Per pose and joint: median as target, tolerance from the spread (for example median absolute deviation scaled), capped to a sane range.
4. Writes a draft `poses.yaml`. An instructor reviews every range before it is trusted.
5. Dataset limits: about 60 images per pose, unknown image provenance, and only Triangle overlaps with the three target poses. Cobra and Bow need our own recorded reference frames.

## 11. Testing

- **Unit:** geometry (known angles, zero-length, collinear), filters, evaluator thresholds and boundaries, planner hysteresis, Surya transitions.
- **Property checks:** rotate_limb output angle equals target, bone length preserved.
- **Replay tests:** recorded landmark sequences through the full pipeline, expected errors and cues asserted.
- **Scenario tests:** correct pose, one wrong arm, one wrong leg, multiple errors, occlusion, leaving frame, different body sizes and distances, lighting changes.
- **Performance test:** replay at full speed and assert stage timings.

## 12. Evaluation plan

- **Static pose accuracy:** frames sampled at intervals, labelled correct/incorrect by two instructors. Report inter-rater agreement (Cohen's kappa) and our accuracy against agreed labels. Target above 95% **[target]**; report the real number whatever it is.
- **Surya stage transitions:** target above 85% **[target]**.
- **Baseline:** generic thresholds from 50th-percentile anthropometric data versus calibrated thresholds.
- **Usability:** 10 to 12 novices, counterbalanced order. SUS (target 80+), NASA-TLX, time-to-correction (static poses only), and structured qualitative feedback. Pilot scale: report trends, not significance.
- **Privacy:** local processing, informed consent before any recording, no video retained unless agreed.

## 13. Schedule

| Dates | Goal |
|---|---|
| Oct 27 (today) | Repo skeleton, camera + estimator + live skeleton, `geometry.py` with tests, Triangle rules, first working arrow and ghost limb |
| Oct 28 to 31 | Smoothing, visibility gating, calibration, Cobra and Bow rules, evaluator and planner with hysteresis (Milestone 1) |
| Nov 1 to 7 | Two-panel UI, reference panel with Bezier animation, measured latency |
| Nov 8 to 18 | Surya state machine, next-stage preview, fatigue timeout, skip-cascade fallback, polish (Milestone 2) |
| Nov 19 to 25 | User testing and instructor labelling |
| Nov 26 to 30 | Analysis, report, demo video, submission |

## 14. Limitations to state honestly

- Single user only; not tested with multiple people in frame.
- Thresholds follow body proportions, not individual flexibility.
- Reference scaling is linear in limb length; it ignores nonlinear biomechanics (for example center-of-mass shifts).
- MediaPipe depth is approximate from a 2D webcam; the 2D fallback reduces but does not remove this.
- World-to-screen mapping is an approximation of perspective projection.
- Pilot-sized sample (N=12): trends only.
- Not a replacement for in-person instruction, especially with pre-existing injuries. Feedback is posture guidance, not medical advice.

## 15. Definition of done

- A user selects a pose and gets a live correction session from start to aligned state.
- At least one common error per static pose produces a ghost limb, arrow and cue.
- Low-confidence landmarks never trigger corrections.
- Surya Namaskar tracks stages and advances automatically.
- Measured latency and FPS are reported.
- Unit and replay tests pass; evaluation results and failure cases are documented.
- The demo runs from startup to a successful correction without editing any data by hand.
