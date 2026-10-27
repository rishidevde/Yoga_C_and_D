"""OpenCV rendering engine for live skeleton, ghost limbs, arrows, and UI cues.

Follows project specifications:
- Bold, high-visibility skeleton with distinguished hand and foot keypoints.
- Prominent virtual target limbs (synthetic dummy rays) in vibrant lime green.
- Large, high-contrast directional arrows in radiant orange with target waypoint beacons.
- Large readable HUD and text cue banners legible from 8–10 feet away.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from yogacorrect.types import Correction, Landmarks, POSE_CONNECTIONS


class PoseRenderer:
    """Renders user skeleton, ghost limb corrections, directional arrows, and HUD."""

    def __init__(
        self,
        mirror: bool = True,
        skeleton_color: Tuple[int, int, int] = (240, 180, 50),   # BGR: Bright cyan/blue
        joint_color: Tuple[int, int, int] = (255, 255, 255),      # BGR: White
        ghost_color: Tuple[int, int, int] = (0, 255, 60),         # BGR: Vibrant neon green
        ghost_halo: Tuple[int, int, int] = (0, 160, 40),          # BGR: Target glow halo
        arrow_color: Tuple[int, int, int] = (0, 140, 255),        # BGR: Radiant orange
        min_visibility: float = 0.5,
    ) -> None:
        """Initialize PoseRenderer.

        Args:
            mirror: Whether display coordinates are horizontally mirrored.
            skeleton_color: BGR color for skeleton bones.
            joint_color: BGR color for landmark dots.
            ghost_color: BGR color for ghost target limbs.
            ghost_halo: BGR color for ghost limb glow halo.
            arrow_color: BGR color for correction arrows.
            min_visibility: Minimum visibility to draw a joint.
        """
        self.mirror = mirror
        self.skeleton_color = skeleton_color
        self.joint_color = joint_color
        self.ghost_color = ghost_color
        self.ghost_halo = ghost_halo
        self.arrow_color = arrow_color
        self.min_visibility = min_visibility

    def _map_pt(self, px: Tuple[int, int], width: int) -> Tuple[int, int]:
        """Optionally mirror pixel coordinate horizontally."""
        if self.mirror:
            return (width - 1 - px[0], px[1])
        return px

    def draw_skeleton(self, frame: np.ndarray, landmarks: Landmarks) -> None:
        """Draw live skeleton lines and joint dots with hand & foot highlights."""
        h, w = frame.shape[:2]

        # 1. Draw Skeleton Bones
        for start_idx, end_idx in POSE_CONNECTIONS:
            if (
                landmarks.visibility[start_idx] < self.min_visibility
                or landmarks.visibility[end_idx] < self.min_visibility
            ):
                continue

            pt1 = (
                int(landmarks.image[start_idx, 0] * w),
                int(landmarks.image[start_idx, 1] * h),
            )
            pt2 = (
                int(landmarks.image[end_idx, 0] * w),
                int(landmarks.image[end_idx, 1] * h),
            )

            pt1_m = self._map_pt(pt1, w)
            pt2_m = self._map_pt(pt2, w)

            # High-visibility bone with dark border outline for contrast against clothing
            cv2.line(frame, pt1_m, pt2_m, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.line(frame, pt1_m, pt2_m, self.skeleton_color, 3, cv2.LINE_AA)

        # 2. Draw Body & Hand/Foot Keypoints
        hand_wrist_indices = {15, 16}
        hand_finger_indices = {17, 18, 19, 20, 21, 22}
        foot_indices = {27, 28, 29, 30, 31, 32}

        for idx in range(len(landmarks.image)):
            if landmarks.visibility[idx] < self.min_visibility:
                continue

            pt = (
                int(landmarks.image[idx, 0] * w),
                int(landmarks.image[idx, 1] * h),
            )
            pt_m = self._map_pt(pt, w)

            if idx in hand_wrist_indices:
                # Wrists: prominent bright cyan ring
                cv2.circle(frame, pt_m, 8, (255, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, pt_m, 9, (20, 20, 20), 2, cv2.LINE_AA)
            elif idx in hand_finger_indices:
                # Fingers: yellow dots
                cv2.circle(frame, pt_m, 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt_m, 6, (20, 20, 20), 1, cv2.LINE_AA)
            elif idx in foot_indices:
                # Feet: purple/magenta rings
                cv2.circle(frame, pt_m, 6, (200, 50, 220), -1, cv2.LINE_AA)
                cv2.circle(frame, pt_m, 7, (20, 20, 20), 1, cv2.LINE_AA)
            else:
                # Standard joints: white dot with dark outline
                cv2.circle(frame, pt_m, 5, self.joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt_m, 6, (20, 20, 20), 1, cv2.LINE_AA)

    def draw_corrections(
        self,
        frame: np.ndarray,
        corrections: List[Correction],
    ) -> None:
        """Draw full-limb target ghost, actual limb highlight, and directional arrows."""
        h, w = frame.shape[:2]

        for corr_idx, corr in enumerate(corrections):
            # Map all chain points for mirroring
            target_chain = [self._map_pt(pt, w) for pt in corr.target_chain_px]
            actual_chain = [self._map_pt(pt, w) for pt in corr.actual_chain_px]
            curr_m = self._map_pt(corr.current_px, w)
            ghost_m = self._map_pt(corr.ghost_px, w)

            # -----------------------------------------------------------------
            # 1. Full Target Limb (Ghost Skeleton) — thick green polyline
            # -----------------------------------------------------------------
            if len(target_chain) >= 2:
                pts_arr = np.array(target_chain, dtype=np.int32)
                # Outer dark border for contrast
                cv2.polylines(frame, [pts_arr], False, (15, 15, 15), 10, cv2.LINE_AA)
                # Bright green core limb
                cv2.polylines(frame, [pts_arr], False, self.ghost_color, 6, cv2.LINE_AA)

                # Draw joint circles along the target chain
                for i, pt in enumerate(target_chain):
                    if i == len(target_chain) - 1:
                        # Target endpoint — large filled green circle with white ring
                        cv2.circle(frame, pt, 14, (15, 15, 15), -1, cv2.LINE_AA)
                        cv2.circle(frame, pt, 12, self.ghost_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    else:
                        # Intermediate joint — small green ring
                        cv2.circle(frame, pt, 7, self.ghost_color, 2, cv2.LINE_AA)

            # -----------------------------------------------------------------
            # 2. Actual Limb Highlight — full orange polyline showing "where you are"
            # -----------------------------------------------------------------
            if len(actual_chain) >= 2:
                act_arr = np.array(actual_chain, dtype=np.int32)
                # Draw full actual limb in orange so user clearly sees their current position
                cv2.polylines(frame, [act_arr], False, (15, 15, 15), 8, cv2.LINE_AA)
                cv2.polylines(frame, [act_arr], False, self.arrow_color, 5, cv2.LINE_AA)

            # Current endpoint — orange filled circle with white bullseye
            cv2.circle(frame, curr_m, 13, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(frame, curr_m, 11, self.arrow_color, -1, cv2.LINE_AA)
            cv2.circle(frame, curr_m, 5, (255, 255, 255), -1, cv2.LINE_AA)

            # -----------------------------------------------------------------
            # 3. Directional Correction Arrow (Current → Target)
            # -----------------------------------------------------------------
            arrow_dist = math.hypot(ghost_m[0] - curr_m[0], ghost_m[1] - curr_m[1])
            if arrow_dist > 12:
                # Black outline arrow for contrast
                cv2.arrowedLine(
                    frame, curr_m, ghost_m,
                    (15, 15, 15), 9, cv2.LINE_AA, tipLength=0.3,
                )
                # Vibrant orange core arrow
                cv2.arrowedLine(
                    frame, curr_m, ghost_m,
                    self.arrow_color, 5, cv2.LINE_AA, tipLength=0.3,
                )

            # -----------------------------------------------------------------
            # 4. Angle Adjustment Badge
            # -----------------------------------------------------------------
            mid_x = (curr_m[0] + ghost_m[0]) // 2
            mid_y = (curr_m[1] + ghost_m[1]) // 2
            # Stagger badge vertically by correction index to prevent overlap
            badge_offset_y = -30 - corr_idx * 36
            self._draw_badge(
                frame=frame,
                text=corr.label,
                pos=(mid_x - 40, mid_y + badge_offset_y),
                bg_color=(20, 20, 20),
                border_color=self.arrow_color,
                text_color=(255, 255, 255),
                scale=0.65,
            )

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist < 1e-3:
            return

        dashes = int(dist / dash_length)
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist

        for i in range(0, dashes, 2):
            s = int(i * dash_length)
            e = int(min((i + 1) * dash_length, dist))
            p1 = (int(pt1[0] + dx * s), int(pt1[1] + dy * s))
            p2 = (int(pt1[0] + dx * e), int(pt1[1] + dy * e))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    def _draw_badge(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int] = (25, 25, 25),
        border_color: Tuple[int, int, int] = (0, 140, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        scale: float = 0.55,
    ) -> None:
        """Draw text with rounded dark background pill for high contrast."""
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        x, y = pos
        pad = 6
        h, w = frame.shape[:2]
        x = max(pad, min(x, w - tw - pad * 2))
        y = max(th + pad, min(y, h - pad))

        # Solid background rectangle
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            bg_color,
            -1,
        )
        # Border
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            border_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        pose_name: str,
        corrections: List[Correction],
        fps: float,
        is_aligned: bool = False,
        person_detected: bool = True,
    ) -> None:
        """Draw top status bar and bottom instruction cue."""
        h, w = frame.shape[:2]

        # 1. Top HUD Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Pose title (large font)
        cv2.putText(
            frame,
            f"Pose: {pose_name}",
            (20, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Status badge (top right)
        if not person_detected:
            status_text = "Step into camera view"
            status_color = (0, 165, 255)  # Orange
        elif is_aligned:
            status_text = "Pose Aligned! Keep holding"
            status_color = (50, 255, 80)  # Bright Green
        else:
            status_text = f"Adjusting ({len(corrections)} to fix)"
            status_color = (0, 220, 255)  # Vibrant Yellow

        (sw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(
            frame,
            status_text,
            (w - sw - 20, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # FPS indicator
        cv2.putText(
            frame,
            f"{int(fps)} FPS",
            (w // 2 - 35, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # 2. Bottom Instruction Cue Banner — shows top 2 failing cues (100px tall)
        if corrections and len(corrections) > 0:
            bot_overlay = frame.copy()
            cv2.rectangle(bot_overlay, (0, h - 100), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(bot_overlay, 0.85, frame, 0.15, 0, frame)

            # Priority cue (line 1) — most important correction
            cue0 = f"1. {corrections[0].cue}  [{corrections[0].label}]"
            cv2.putText(
                frame, cue0, (24, h - 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (0, 225, 255),  # Vibrant gold
                2, cv2.LINE_AA,
            )

            # Secondary cue (line 2) — if there's a second correction
            if len(corrections) > 1:
                cue1 = f"2. {corrections[1].cue}  [{corrections[1].label}]"
                cv2.putText(
                    frame, cue1, (24, h - 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (180, 200, 255),  # Lighter blue-white for secondary
                    1, cv2.LINE_AA,
                )

        elif is_aligned:
            bot_overlay = frame.copy()
            cv2.rectangle(bot_overlay, (0, h - 70), (w, h), (10, 45, 10), -1)
            cv2.addWeighted(bot_overlay, 0.85, frame, 0.15, 0, frame)

            cv2.putText(
                frame,
                "PERFECT ALIGNMENT! HOLD THE POSE",
                (24, h - 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (80, 255, 80),  # Bright green
                2,
                cv2.LINE_AA,
            )
