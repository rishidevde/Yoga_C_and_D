"""Geometry utilities for angle computation and limb rotation.

All public functions work with 2-D **or** 3-D points and return unsigned
angles in the range [0, 180] degrees.

Key functions:
    angle_deg    — angle at a vertex formed by three points
    rotate_limb  — rotate a limb segment to a target angle (Rodrigues)
    batch_angles — vectorised angle computation for many triplets at once
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Small epsilon to guard against division by zero.
_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Single-triplet angle
# ---------------------------------------------------------------------------

def angle_deg(
    a: np.ndarray | list | tuple,
    b: np.ndarray | list | tuple,
    c: np.ndarray | list | tuple,
) -> float:
    """Unsigned angle at vertex **b** formed by rays b→a and b→c.

    Works for 2-D ``(x, y)`` or 3-D ``(x, y, z)`` points.

    Args:
        a: First endpoint.
        b: Vertex (the point *at* which the angle is measured).
        c: Second endpoint.

    Returns:
        Angle in degrees ∈ [0, 180].
        Returns ``nan`` if either segment has near-zero length.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    u = a - b
    v = c - b

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < _EPS or norm_v < _EPS:
        return float("nan")

    cos_angle = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Limb rotation (ghost-point generation)
# ---------------------------------------------------------------------------

def rotate_limb_2d(
    a: np.ndarray | list | tuple,
    b: np.ndarray | list | tuple,
    c: np.ndarray | list | tuple,
    target_deg: float,
) -> np.ndarray:
    """Rotate point c around vertex b so that angle(a, b, c') == target_deg in 2D.

    Preserves bone length |c - b|. Operates directly in the 2D image plane,
    eliminating monocular depth ambiguity for front-facing pose displays.

    Args:
        a: Anchor/proximal point (e.g. shoulder).
        b: Vertex/pivot point (e.g. elbow).
        c: Distal point to rotate (e.g. wrist).
        target_deg: Desired angle at b in degrees [0, 180].

    Returns:
        Rotated 2D point as ndarray(2,), or c if degenerate.
    """
    a_pt = np.asarray(a[:2], dtype=np.float64)
    b_pt = np.asarray(b[:2], dtype=np.float64)
    c_pt = np.asarray(c[:2], dtype=np.float64)

    u = a_pt - b_pt
    v = c_pt - b_pt
    cur_deg = angle_deg(a_pt, b_pt, c_pt)
    bone_len = float(np.linalg.norm(v))
    if bone_len < _EPS or np.isnan(cur_deg):
        return c_pt

    # 2D cross product indicates rotational orientation
    cp = float(u[0] * v[1] - u[1] * v[0])
    sign = 1.0 if cp >= 0 else -1.0

    delta_deg = target_deg - cur_deg
    phi = math.radians(delta_deg) * sign

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    v_rot = np.array([
        v[0] * cos_phi - v[1] * sin_phi,
        v[0] * sin_phi + v[1] * cos_phi,
    ])
    norm_rot = float(np.linalg.norm(v_rot))
    if norm_rot < _EPS:
        return c_pt

    v_rot = (v_rot / norm_rot) * bone_len
    return b_pt + v_rot


def rotate_limb(
    a: np.ndarray | list | tuple,
    b: np.ndarray | list | tuple,
    c: np.ndarray | list | tuple,
    target_deg: float,
    hint_axis: Optional[tuple[float, float, float] | np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Rotate point **c** around vertex **b** so that ``angle(a, b, c') == target_deg``.

    The rotation preserves ``|b − c|`` (bone length) and operates in the plane
    defined by a, b, c.  Uses the **Rodrigues rotation formula**.

    Args:
        a:          Proximal point (e.g. shoulder).
        b:          Vertex / pivot (e.g. elbow).
        c:          Distal point to rotate (e.g. wrist).
        target_deg: Desired angle at *b* in degrees [0, 180].
        hint_axis:  Fallback rotation axis when a-b-c are collinear and the
                    rotation plane is undefined.  Shape ``(3,)``.

    Returns:
        New position for *c* as ``ndarray(3,)``, or ``None`` when the limb is
        collinear and no *hint_axis* was provided (rotation is ambiguous).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    u = a - b  # b → a
    v = c - b  # b → c

    current_deg = angle_deg(a, b, c)
    if math.isnan(current_deg):
        return None

    # Compute the rotation-plane normal.
    n = np.cross(u, v)
    n_len = np.linalg.norm(n)

    if n_len < 1e-6:
        # Collinear case — the plane is ambiguous.
        if hint_axis is None:
            return None
        n = np.asarray(hint_axis, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len < _EPS:
            return None

    n = n / np.linalg.norm(n)

    # Angle to rotate v by (radians).
    phi = math.radians(target_deg - current_deg)

    # Rodrigues formula (simplified because n ⊥ v when n = cross(u,v)/‖…‖):
    #   v' = v·cos φ  +  (n × v)·sin φ
    v_rot = v * math.cos(phi) + np.cross(n, v) * math.sin(phi)

    return b + v_rot


# ---------------------------------------------------------------------------
# Vectorised batch computation
# ---------------------------------------------------------------------------

def batch_angles(
    world: np.ndarray,
    triplets: np.ndarray,
) -> np.ndarray:
    """Compute unsigned angles for multiple joint triplets in one pass.

    Uses ``einsum`` for dot products and ``linalg.norm`` along axis 1 to
    avoid Python-level loops, giving a significant speed-up when evaluating
    many rules per frame.

    Args:
        world:    Landmark world coordinates, shape ``(33, 3)``.
        triplets: Joint index triplets, shape ``(N, 3)``.
                  Each row ``(a, b, c)`` defines a vertex at *b*.

    Returns:
        Angles in degrees, shape ``(N,)``.
        Entries are ``nan`` for degenerate (zero-length) segments.
    """
    triplets = np.asarray(triplets, dtype=np.intp)

    a_pts = world[triplets[:, 0]]  # (N, 3)
    b_pts = world[triplets[:, 1]]  # (N, 3)
    c_pts = world[triplets[:, 2]]  # (N, 3)

    u = a_pts - b_pts  # (N, 3)
    v = c_pts - b_pts  # (N, 3)

    norm_u = np.linalg.norm(u, axis=1)  # (N,)
    norm_v = np.linalg.norm(v, axis=1)  # (N,)

    dots = np.einsum("ij,ij->i", u, v)  # (N,)

    denom = norm_u * norm_v
    valid_mask = denom > _EPS

    angles = np.full(len(triplets), float("nan"), dtype=np.float64)
    cos_vals = np.clip(dots[valid_mask] / denom[valid_mask], -1.0, 1.0)
    angles[valid_mask] = np.degrees(np.arccos(cos_vals))

    return angles
