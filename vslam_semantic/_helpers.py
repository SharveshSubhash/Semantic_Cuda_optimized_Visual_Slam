# SPDX-License-Identifier: Apache-2.0
"""
vslam_semantic._helpers
========================

Pure numerical and convention helpers used across the package. **No ROS
dependencies** - the only third-party import is numpy, so this module is
safe to import in any environment (unit tests, off-board analysis,
documentation generation).

If you find yourself adding a function here that touches rclpy, geometry_msgs,
or any other ROS type, move it to the node module that uses it instead.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import numpy as np


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Uses the standard right-handed convention matching tf2 / ROS REP-103.
    Identity quaternion (0, 0, 0, 1) maps to the 3x3 identity matrix.
    The quaternion is normalised implicitly via the `2/n` scale factor,
    so unnormalised inputs produce a valid (but non-orthonormal) matrix
    rather than blowing up.

    A degenerate (near-zero norm) quaternion is treated as identity to
    avoid producing NaNs - this matters because cuVSLAM occasionally
    publishes a zero-quaternion during the first frame after init.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,            xz + wy],
        [xy + wz,               1.0 - (xx + zz),    yz - wx],
        [xz - wy,               yz + wx,            1.0 - (xx + yy)],
    ])


def triangulate_bearing_rays(
        rays: List[Tuple[np.ndarray, np.ndarray]],
        fallback_range: float = 2.0) -> np.ndarray:
    """Closed-form least-squares intersection of a set of bearing rays.

    Each ray is `(origin, unit_direction)`. The returned point `p`
    minimises the sum of squared perpendicular distances to all rays:

        sum_i (I - d_i d_i^T) (p - o_i) = 0
        => (sum_i (I - d_i d_i^T)) p = sum_i (I - d_i d_i^T) o_i

    With a single ray, falls back to placing the point at
    `fallback_range` metres along that ray. With parallel rays
    (singular system), falls back to the same single-ray rule using
    the most recent ray. Never raises.
    """
    if len(rays) < 2:
        o, d = rays[-1]
        return o + d * fallback_range

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        o, d = rays[-1]
        return o + d * fallback_range


def first_attr(obj: Any, names: Iterable[str]) -> Any:
    """Return the first attribute in `names` present on `obj` (else None).

    Useful for reading values out of objects whose field names have
    drifted across versions - e.g. NanoOWL's ``OwlDecodeOutput`` is
    spelled ``boxes/scores/labels`` in current main, but earlier forks
    and the HuggingFace OWL-ViT object use ``pred_boxes/pred_scores/
    pred_labels``. Lets us decode either without hard-coding a spelling.
    """
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None
