# Yoga Pose Specifications and Correction Guide

This document describes the anatomical specifications, key joint landmarks, common postural deviations, and correction cues for the target yoga poses.

---

## 1. Triangle Pose (*Trikonasana*)

### Camera View
- **Supported View:** Front view (facing the camera directly)
- **Minimum Visibility:** 0.5 across essential limb joints

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Front Knee Straight** | Left Hip (23) → Left Knee (25) → Left Ankle (27) | **175°** | ±12° | Yes | *"Straighten your front knee"* | *"Do not hyperextend your front knee"* |
| **Back Knee Straight** | Right Hip (24) → Right Knee (26) → Right Ankle (28) | **175°** | ±12° | Yes | *"Keep your back leg straight and strong"* | *"Do not hyperextend your back knee"* |
| **Top Arm Vertical** | Left Shoulder (11) → Right Shoulder (12) → Right Wrist (16) | **170°** | ±15° | No | *"Reach your top arm straight up toward ceiling"* | *"Align your top arm with your shoulders"* |
| **Bottom Arm Alignment**| Right Shoulder (12) → Left Shoulder (11) → Left Wrist (15) | **170°** | ±15° | No | *"Extend your bottom hand down toward shin"* | *"Keep bottom arm aligned with chest"* |
| **Torso Lateral Bend** | Left Shoulder (11) → Left Hip (23) → Left Knee (25) | **140°** | ±20° | No | *"Lengthen both sides of your torso"* | *"Hinge deeper from your front hip crease"* |

### Common Postural Deviations
1. **Front Knee Sagging/Bent:** Beginners often collapse the front knee under body weight.
   - *Visual Feedback:* Directional arrow pulls knee back into alignment.
2. **Top Arm Dropped Forward:** Trailing arm collapses forward instead of stacking vertically over shoulders.
   - *Visual Feedback:* Ghost limb projects at 90°/170° upwards; orange arrow directs hand upward.

---

## 2. Cobra Pose (*Bhujangasana*)

### Camera View
- **Supported View:** Side profile (90° to camera line of sight)
- **Minimum Visibility:** 0.4

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Elbows Soft Bend** | Shoulder (11) → Elbow (13) → Wrist (15) | **140°** | ±20° | Yes | *"Do not bend elbows too much; open chest"* | *"Keep elbows slightly bent, don't lock arms"* |
| **Spine Extension** | Ear (7) → Shoulder (11) → Hip (23) | **150°** | ±25° | Yes | *"Lift your chest forward and up gently"* | *"Avoid compressing your lower back"* |

---

## 3. Bow Pose (*Dhanurasana*)

### Camera View
- **Supported View:** Side profile
- **Minimum Visibility:** 0.4

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Knee Bend & Lift** | Hip (23) → Knee (25) → Ankle (27) | **90°** | ±25° | Yes | *"Kick your feet up and back into hands"* | *"Keep knees hip-width apart and active"* |
| **Chest Lift** | Shoulder (11) → Hip (23) → Knee (25) | **155°** | ±20° | No | *"Lift your chest and thighs higher"* | *"Maintain smooth breathing and steady hold"* |

---

## 4. How to Add a New Pose

To introduce a new pose without touching any Python code:
1. Open `configs/poses.yaml`.
2. Append a new block with your pose key:
```yaml
warrior2:
  name: "Warrior II (Virabhadrasana II)"
  view: front
  min_visibility: 0.5
  hold_seconds: 3.0
  rules:
    - name: front_knee_bend
      kind: angle
      joints: [left_hip, left_knee, left_ankle]
      target: 90.0
      tol: 12.0
      weight: 1.0
      safety: true
      cue_low: "Bend front knee deeper to 90 degrees"
      cue_high: "Don't let knee move past your ankle"
```
3. Run `python -m yogacorrect.main --pose warrior2`.
