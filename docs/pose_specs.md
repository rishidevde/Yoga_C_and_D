# Yoga Pose Specifications and Correction Guide

This document describes the anatomical specifications, key joint landmarks, common postural deviations, and correction cues for the target yoga poses supported in the system.

All targets and tolerances are derived from statistical ground-truth analysis of `Yoga_Poses-Dataset/Results` and anatomical yoga guidelines.

---

## 1. Triangle Pose (*Trikonasana*)

### Camera View & Setup
- **Supported View:** Front view (facing camera directly)
- **Bilateral:** Yes (adaptive side detection via wrist height differential `rwy - lwy`)
- **Minimum Visibility:** 0.5 across essential limb joints
- **Hotkey:** `1`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Front Knee Straight** | Hip (23) → Knee (25) → Ankle (27) | **175°** | ±15° | Yes | *"Straighten your front knee"* | *"Do not hyperextend your front knee"* |
| **Back Knee Straight** | Hip (24) → Knee (26) → Ankle (28) | **175°** | ±15° | Yes | *"Keep your back leg straight and strong"* | *"Do not hyperextend your back knee"* |
| **Top Arm Vertical** | Shoulder (11) → Shoulder (12) → Wrist (16) | **170°** | ±20° | No | *"Reach your top arm straight up toward ceiling"* | *"Align your top arm with your shoulders"* |
| **Bottom Arm Alignment**| Shoulder (12) → Shoulder (11) → Wrist (15) | **170°** | ±20° | No | *"Extend your bottom hand down toward shin"* | *"Keep bottom arm aligned with chest"* |
| **Torso Lateral Bend** | Shoulder (11) → Hip (23) → Knee (25) | **60°** | ±22° | No | *"Hinge further sideways from front hip crease"* | *"Stand taller from hip — reduce the side bend"* |

---

## 2. Half Moon Pose (*Ardha Chandrasana*)

### Camera View & Setup
- **Supported View:** Front view
- **Bilateral:** Yes (detected via wrist height differential)
- **Minimum Visibility:** 0.45
- **Hotkey:** `2`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing Knee Straight** | Standing Hip → Knee → Ankle | **175°** | ±16° | Yes | *"Straighten your standing leg firmly"* | *"Do not hyperextend your standing knee"* |
| **Lifted Leg Straight** | Lifted Hip → Knee → Ankle | **175°** | ±18° | No | *"Extend your lifted leg straight through the heel"* | *"Keep lifted leg active and parallel to floor"* |
| **Top Arm Vertical** | Bottom Shoulder → Top Shoulder → Top Wrist | **170°** | ±20° | No | *"Reach your top arm straight up toward ceiling"* | *"Stack your top shoulder directly above bottom shoulder"* |
| **Bottom Arm Grounded** | Top Shoulder → Bottom Shoulder → Bottom Wrist | **165°** | ±22° | No | *"Extend bottom hand toward floor under your shoulder"* | *"Keep bottom arm firm and chest open"* |
| **Torso Alignment** | Shoulder → Hip → Lifted Ankle | **165°** | ±25° | No | *"Lift back leg to hip height, lengthening through spine"* | *"Maintain a straight horizontal line from head to lifted heel"* |

---

## 3. Bound Angle Pose (*Baddha Konasana*)

### Camera View & Setup
- **Supported View:** Front view (seated on floor/mat)
- **Bilateral:** Symmetric
- **Minimum Visibility:** 0.40
- **Hotkey:** `3`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Left Knee Flexion** | Left Hip (23) → Knee (25) → Ankle (27) | **35°** | ±22° | No | *"Ease left knee outwards comfortably"* | *"Draw left heel closer toward your pelvis"* |
| **Right Knee Flexion** | Right Hip (24) → Knee (26) → Ankle (28) | **35°** | ±22° | No | *"Ease right knee outwards comfortably"* | *"Draw right heel closer toward your pelvis"* |
| **Spine Upright** | Left Shoulder (11) → Left Hip (23) → Left Knee (25) | **70°** | ±22° | Yes | *"Sit taller through crown of head; avoid hunching"* | *"Hinge forward gently from hips with a flat back"* |
| **Shoulders Relaxed** | Left Elbow (13) → Left Shoulder (11) → Left Hip (23) | **10°** | ±18° | No | *"Keep arms resting naturally alongside torso"* | *"Relax shoulders away from your ears and lower arms"* |

---

## 4. Downward-Facing Dog (*Adho Mukha Svanasana*)

### Camera View & Setup
- **Supported View:** Side profile view (45°–90° to camera)
- **Bilateral:** Symmetric
- **Minimum Visibility:** 0.40
- **Hotkey:** `4`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Knees Straight** | Left Hip (23) → Knee (25) → Ankle (27) | **175°** | ±16° | Yes | *"Press thighs back to straighten your legs"* | *"Do not hyperextend your knees"* |
| **Arms Straight** | Left Shoulder (11) → Elbow (13) → Wrist (15) | **175°** | ±16° | Yes | *"Push the ground away to fully straighten arms"* | *"Keep micro-bend in elbows to prevent hyperextension"* |
| **Hip Inverted V** | Left Shoulder (11) → Left Hip (23) → Left Ankle (27) | **68°** | ±22° | No | *"Send hips higher toward ceiling to form an inverted V"* | *"Press weight back toward feet and lengthen torso"* |
| **Shoulder Opening** | Left Wrist (15) → Left Shoulder (11) → Left Hip (23) | **168°** | ±22° | No | *"Open shoulders and press chest gently toward thighs"* | *"Avoid sinking excessively through shoulders"* |

---

## 5. Lord of the Dance (*Natarajasana*)

### Camera View & Setup
- **Supported View:** Side / diagonal view
- **Bilateral:** Yes (detected via lifted foot height)
- **Minimum Visibility:** 0.45
- **Hotkey:** `5`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing Knee Straight** | Standing Hip → Knee → Ankle | **175°** | ±16° | Yes | *"Straighten your standing leg firmly"* | *"Do not hyperextend standing knee"* |
| **Lifted Knee Bend** | Lifted Hip → Knee → Ankle | **85°** | ±25° | No | *"Kick back foot higher and back into hand"* | *"Bend lifted knee and lift thigh upward"* |
| **Reaching Arm Forward** | Standing Hip → Standing Shoulder → Standing Wrist | **165°** | ±25° | No | *"Reach front arm forward and slightly upward"* | *"Lower front arm parallel to floor"* |
| **Chest Lift Backbend** | Standing Shoulder → Standing Hip → Standing Knee | **145°** | ±25° | No | *"Lift chest forward and proud as you kick back"* | *"Keep hips square and spine elongated"* |

---

## 6. Goddess Pose (*Utkata Konasana*)

### Camera View & Setup
- **Supported View:** Front view
- **Bilateral:** Symmetric
- **Minimum Visibility:** 0.45
- **Hotkey:** `6`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Left Knee Squat** | Left Hip (23) → Knee (25) → Ankle (27) | **110°** | ±25° | Yes | *"Do not collapse knees inward; press out over toes"* | *"Sink hips lower toward knee height"* |
| **Right Knee Squat** | Right Hip (24) → Knee (26) → Ankle (28) | **110°** | ±25° | Yes | *"Do not collapse knees inward; press out over toes"* | *"Sink hips lower toward knee height"* |
| **Left Arm Cactus** | Left Shoulder (11) → Elbow (13) → Wrist (15) | **95°** | ±30° | No | *"Raise left forearm upright into 90-degree cactus shape"* | *"Bend left elbow to 90 degrees with palm forward"* |
| **Right Arm Cactus** | Right Shoulder (12) → Elbow (14) → Wrist (16) | **95°** | ±30° | No | *"Raise right forearm upright into 90-degree cactus shape"* | *"Bend right elbow to 90 degrees with palm forward"* |
| **Torso Vertical** | Left Shoulder (11) → Left Hip (23) → Left Knee (25) | **105°** | ±22° | No | *"Stack shoulders directly above hips; avoid forward lean"* | *"Keep spine tall, tailbone tucked down"* |

---

## 7. Warrior II Pose (*Virabhadrasana II*)

### Camera View & Setup
- **Supported View:** Front view
- **Bilateral:** Yes (detected via knee angle comparison: bent front knee vs straight back knee)
- **Minimum Visibility:** 0.50
- **Hotkey:** `7`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Front Knee Bend** | Front Hip → Front Knee → Front Ankle | **110°** | ±22° | Yes | *"Knee collapsing past toes — keep knee stacked above ankle"* | *"Bend front knee deeper toward 90 degrees"* |
| **Back Knee Straight** | Back Hip → Back Knee → Back Ankle | **175°** | ±16° | Yes | *"Straighten back leg firmly and ground through back foot"* | *"Do not hyperextend back knee"* |
| **Front Arm Horizontal** | Back Shoulder → Front Shoulder → Front Wrist | **175°** | ±18° | No | *"Lift front arm parallel to floor at shoulder height"* | *"Lower front arm to shoulder height"* |
| **Back Arm Horizontal** | Front Shoulder → Back Shoulder → Back Wrist | **175°** | ±18° | No | *"Lift back arm parallel to floor at shoulder height"* | *"Lower back arm to shoulder height"* |
| **Torso Upright** | Front Shoulder → Front Hip → Front Knee | **105°** | ±22° | No | *"Do not lean forward over front thigh; center your torso"* | *"Stack shoulders directly over hips"* |

---

## 8. Tree Pose (*Vrikshasana*)

### Camera View & Setup
- **Supported View:** Front view
- **Bilateral:** Yes (detected via lifted foot height)
- **Minimum Visibility:** 0.45
- **Hotkey:** `8`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing Knee Straight** | Standing Hip → Knee → Ankle | **175°** | ±16° | Yes | *"Straighten standing leg and ground through your foot"* | *"Do not hyperextend standing knee"* |
| **Tree Knee Turnout** | Bent Hip → Knee → Ankle | **55°** | ±28° | No | *"Open bent knee outward without twisting pelvis"* | *"Place bent foot higher on inner thigh or calf"* |
| **Torso Vertical** | Standing Shoulder → Standing Hip → Standing Knee | **175°** | ±20° | No | *"Stand tall; do not lean toward the standing hip"* | *"Stack shoulders over hips and lengthen spine"* |

---

## 9. Cobra Pose (*Bhujangasana*)

### Camera View & Setup
- **Supported View:** Side profile (90° to camera line of sight)
- **Minimum Visibility:** 0.40
- **Hotkey:** `9`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Elbows Soft Bend** | Shoulder (11) → Elbow (13) → Wrist (15) | **140°** | ±20° | Yes | *"Do not bend elbows too much; open chest"* | *"Keep elbows slightly bent, don't lock arms"* |
| **Spine Extension** | Ear (7) → Shoulder (11) → Hip (23) | **150°** | ±25° | Yes | *"Lift your chest forward and up gently"* | *"Avoid compressing your lower back"* |

---

## 10. Bow Pose (*Dhanurasana*)

### Camera View & Setup
- **Supported View:** Side profile view
- **Minimum Visibility:** 0.40
- **Hotkey:** `0`

### Key Joint Rules
| Joint Rule | Anatomical Landmarks | Target Angle | Tolerance | Safety | Cue Low | Cue High |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Knee Bend & Lift** | Hip (23) → Knee (25) → Ankle (27) | **90°** | ±25° | Yes | *"Kick your feet up and back into hands"* | *"Keep knees hip-width apart and active"* |
| **Chest Lift** | Shoulder (11) → Hip (23) → Knee (25) | **155°** | ±20° | No | *"Lift your chest and thighs higher"* | *"Maintain smooth breathing and steady hold"* |

---

## Interactive Hotkey Controls Reference
- **Keys `1`–`8`**: Instant switch to any of the 8 poses from `Yoga_Poses-Dataset/TRAIN`
- **Keys `9` and `0`**: Switch to Cobra and Bow poses
- **Key `n`**: Cycle to **Next Pose**
- **Key `p`**: Cycle to **Previous Pose**
- **Key `r`**: Reset alignment hold timer and correction planner
- **Key `f`**: Toggle Fullscreen display
- **Key `q` or `ESC`**: Quit cleanly
