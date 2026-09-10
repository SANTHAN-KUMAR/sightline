"""Telemetry-predicted image motion: the §5.6 rule 4 fallback when optical flow fails over rippling water.

Given two `Telemetry` samples and the camera `Intrinsics`, the mapping between the two images of a **flat ground
plane** is exactly a homography

    H = K2 (R21 + t21 n1^T / d1) K1^-1                                     (Hartley & Zisserman 13.1)

with `R21` the rotation from camera 1's optical frame to camera 2's, `t21` camera 1's origin expressed in camera
2's frame, `n1` the plane normal in camera 1's frame and `d1` the plane's distance from camera 1. Nothing here is
learned or estimated from pixels — it is the analytic warp §5.6 asks for ("the previous frame's boxes warped
analytically from attitude and GPS change").

**Frame convention.** `Telemetry.q_gimbal` is read exactly the way the frozen schema reads it. `Telemetry.
gimbal_pitch_deg()` returns `aerospace_pitch(q_gimbal) - 90`, so the identity quaternion means **nadir**, not
"looking forward". `OPTICAL_TO_GIMBAL_REF` below is the basis that makes that true, and `rotation_optical_to_ned`
is the only place the convention lives.

This helper belongs morally to the geolocation lane (F13, `sightline/geo/`), which owns the image->ground chain.
It is duplicated here because F11 and F13 are built in parallel and CONTRACTS.md forbids cross-lane writes. If
F13 publishes an equivalent `rotation_optical_to_ned` in `sightline/common/`, delete this one and import theirs —
there must not be two conventions in the repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sightline.common import geodesy
from sightline.schemas import Intrinsics, Telemetry

#: Camera optical axes (x = image right, y = image down, z = into the scene) expressed in the gimbal's reference
#: frame, i.e. the frame `q_gimbal` rotates into NED. Columns are the images of the optical x, y, z axes:
#:   optical z (view direction) -> NED down   (nadir when q_gimbal is identity, per `Telemetry.gimbal_pitch_deg`)
#:   optical y (image bottom)   -> NED south  (so image "up" is north for an unrotated nadir camera)
#:   optical x (image right)    -> NED east
OPTICAL_TO_GIMBAL_REF = np.array(
    [
        [0.0, -1.0, 0.0],  # north component of (x_opt, y_opt, z_opt)
        [1.0, 0.0, 0.0],  # east
        [0.0, 0.0, 1.0],  # down
    ],
    dtype=float,
)


def rotation_optical_to_ned(tel: Telemetry) -> np.ndarray:
    """3x3 rotation taking a vector in the camera's optical frame to NED.

    Honours `Telemetry.gimbal_is_earth_referenced`: when False the gimbal quaternion is camera->body and must be
    composed with `q_body` (body->NED).
    """
    r_gimbal = geodesy.quat_to_rot(tel.q_gimbal)
    if not tel.gimbal_is_earth_referenced:
        r_gimbal = geodesy.quat_to_rot(tel.q_body) @ r_gimbal
    return r_gimbal @ OPTICAL_TO_GIMBAL_REF


def camera_matrix(intr: Intrinsics) -> np.ndarray:
    return np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=float)


def ned_offset(a: Telemetry, b: Telemetry) -> np.ndarray:
    """Position of `b` relative to `a` in local NED metres (north, east, down).

    Uses `ned_m` when both samples carry the simulator's native NED (exact); otherwise the WGS-84 offset.
    """
    if a.ned_m is not None and b.ned_m is not None:
        return np.array(b.ned_m, dtype=float) - np.array(a.ned_m, dtype=float)
    north, east = geodesy.ne_between(a.lat, a.lon, b.lat, b.lon)
    return np.array([north, east, -(b.alt_msl_m - a.alt_msl_m)], dtype=float)


def project_ground_point(
    tel: Telemetry, intr: Intrinsics, north_m: float, east_m: float, down_m: float
) -> tuple[float, float] | None:
    """Project a NED point given relative to the camera onto the image. `None` when it is behind the camera.

    Independent of `ground_plane_homography` on purpose: the tests use it as a second opinion on the homography.
    """
    r = rotation_optical_to_ned(tel)
    p_opt = r.T @ np.array([north_m, east_m, down_m], dtype=float)
    if p_opt[2] <= 1e-9:
        return None
    return (intr.fx * p_opt[0] / p_opt[2] + intr.cx, intr.fy * p_opt[1] / p_opt[2] + intr.cy)


def ground_plane_homography(
    prev: Telemetry,
    cur: Telemetry,
    intr: Intrinsics,
    *,
    intr_cur: Intrinsics | None = None,
    agl_m: float | None = None,
) -> np.ndarray | None:
    """3x3 homography mapping PREVIOUS-frame pixels to CURRENT-frame pixels for points on the ground plane.

    Args:
        prev, cur: the two telemetry samples (previous frame first).
        intr: intrinsics of the previous frame; `intr_cur` defaults to the same camera.
        agl_m: height of the camera above the assumed ground plane at the previous frame. Defaults to
            `prev.agl_m`. Over flood water this is the water surface, per §5.7 step 5.

    Returns `None` when the geometry cannot support a plane-induced warp (no AGL, camera below the plane).
    """
    height = prev.agl_m if agl_m is None else agl_m
    if not math.isfinite(height) or height <= 1e-3:
        return None
    k1 = camera_matrix(intr)
    k2 = camera_matrix(intr_cur if intr_cur is not None else intr)

    r1 = rotation_optical_to_ned(prev)
    r2 = rotation_optical_to_ned(cur)
    r21 = r2.T @ r1  # optical(prev) -> optical(cur)

    p2 = ned_offset(prev, cur)  # camera 2 relative to camera 1, in NED
    t21 = r2.T @ (-p2)  # camera 1's origin, expressed in camera 2's optical frame

    n1 = r1.T @ np.array([0.0, 0.0, 1.0])  # plane normal (NED down) in camera 1's optical frame
    m = r21 + np.outer(t21, n1) / height
    h = k2 @ m @ np.linalg.inv(k1)
    if not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        return None
    return h / h[2, 2]


@dataclass(slots=True)
class AffineFit:
    """A 2x3 affine least-squares approximation of a homography, with the price of the approximation."""

    affine: np.ndarray  # (2, 3) float32, previous-frame pixels -> current-frame pixels
    residual_px: float  # worst sampled point's error against the exact homography
    rms_px: float


def affine_from_homography(h: np.ndarray, width_px: int, height_px: int, *, grid: int = 5) -> AffineFit:
    """Fit the best 2x3 affine to a homography over the image rectangle.

    BoT-SORT's camera-motion compensation warps Kalman states with a 2x3 affine, so a homography has to be
    reduced before it can be handed over. The reduction is exact for a nadir camera at constant altitude
    (the plane-induced warp is then a similarity) and loses accuracy as the view goes oblique — `residual_px`
    says how much, so a caller can refuse the fallback instead of trusting it silently.
    """
    xs = np.linspace(0.0, float(width_px), grid)
    ys = np.linspace(0.0, float(height_px), grid)
    gx, gy = np.meshgrid(xs, ys)
    src = np.column_stack([gx.ravel(), gy.ravel()])

    hom = np.column_stack([src, np.ones(len(src))]) @ h.T
    w = hom[:, 2]
    ok = np.abs(w) > 1e-9
    dst = np.full((len(src), 2), np.nan)
    dst[ok] = hom[ok, :2] / w[ok, None]

    keep = ok & np.all(np.isfinite(dst), axis=1)
    if keep.sum() < 3:
        return AffineFit(np.eye(2, 3, dtype=np.float32), float("inf"), float("inf"))

    a = np.column_stack([src[keep], np.ones(keep.sum())])
    sol, *_ = np.linalg.lstsq(a, dst[keep], rcond=None)  # (3, 2)
    affine = sol.T.astype(np.float32)  # (2, 3)

    pred = a @ sol
    err = np.linalg.norm(pred - dst[keep], axis=1)
    return AffineFit(affine, float(err.max()), float(np.sqrt(np.mean(err**2))))


def telemetry_affine(
    prev: Telemetry,
    cur: Telemetry,
    intr: Intrinsics,
    *,
    agl_m: float | None = None,
) -> AffineFit | None:
    """`ground_plane_homography` reduced to the 2x3 affine BoT-SORT's CMC consumes. `None` if unavailable."""
    h = ground_plane_homography(prev, cur, intr, agl_m=agl_m)
    if h is None:
        return None
    return affine_from_homography(h, intr.width_px, intr.height_px)


def affine_disagreement_px(a: np.ndarray, b: np.ndarray, width_px: int, height_px: int) -> float:
    """Largest distance between where two 2x3 affines send the frame's corners and centre, in pixels.

    This is the honest way to compare camera-motion estimates: matrix norms mean nothing, pixels do.
    """
    pts = np.array(
        [
            [0.0, 0.0],
            [width_px, 0.0],
            [0.0, height_px],
            [width_px, height_px],
            [width_px / 2.0, height_px / 2.0],
        ],
        dtype=float,
    )
    hom = np.column_stack([pts, np.ones(len(pts))])
    pa = hom @ np.asarray(a, dtype=float).T
    pb = hom @ np.asarray(b, dtype=float).T
    return float(np.linalg.norm(pa - pb, axis=1).max())
