"""The camera's ground footprint per frame, from `Telemetry` + `Intrinsics` (SOLUTION_DOC §5.7 geometry, App. B).

The coverage map needs *which cells were in view and at what GSD*, which is the same ray-plane intersection the
geolocation lane (F13, `sightline/geo/`) uses to project a detection pixel to the ground.

DELIBERATE DUPLICATION, reported to the orchestrator: this module re-implements the ray-plane step rather than
importing `sightline/geo/`, because the two lanes are written in parallel and `docs/CONTRACTS.md` forbids
cross-lane imports for anything but `schemas.py` and `common/`. The shared piece is small (build the rotation,
build the ray, intersect z = agl) and both sides use `sightline.common.geodesy.quat_to_rot`, so they cannot drift
on the quaternion convention. If the geo lane later exports a `project_pixel()` with a matching signature, the
functions here should collapse onto it; until then the duplication is *intentional and known*.

Conventions, forced by `schemas.Telemetry.gimbal_pitch_deg()`:
  * `q_gimbal` rotates the camera's OPTICAL frame (x right, y down, z forward along the optical axis) into NED;
  * the identity quaternion is therefore nadir (optical z -> NED down), which is exactly what
    `gimbal_pitch_deg()` reports (it returns `quat_pitch - 90`, i.e. -90 for the identity);
  * at nadir with gimbal yaw psi, image +u points along bearing psi and image +v along bearing psi + 90.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sightline.common.geodesy import euler_to_quat, quat_to_rot
from sightline.schemas import Intrinsics, Telemetry

#: §5.7 step 5 rejects a ray whose downward component is <= 0.1 ("near_horizon"). A footprint corner that grazes
#: the horizon is clipped to this instead of being dropped, so an oblique frame stays usable and bounded.
MIN_DOWN_COMPONENT = 0.1


def gimbal_quat(pitch_deg: float, yaw_deg: float = 0.0) -> tuple[float, float, float, float]:
    """Build a `q_gimbal` for a gimbal pitch (-90 = nadir, 0 = horizon) and an azimuth the camera looks toward.

    Round-trips exactly through `Telemetry.gimbal_pitch_deg()`, which is the contract this has to satisfy.
    """
    return euler_to_quat(0.0, pitch_deg + 90.0, yaw_deg)


def ground_offsets_ne(uv: np.ndarray, intr: Intrinsics, rot_cam_to_ned: np.ndarray, agl_m: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the rays through pixels `uv` (N, 2) with the horizontal plane `agl_m` below the camera.

    Returns (offsets_ne (N, 2) in metres from the camera's nadir point, clipped (N,) bool).
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d_cam = np.column_stack([(uv[:, 0] - intr.cx) / intr.fx, (uv[:, 1] - intr.cy) / intr.fy, np.ones(len(uv))])
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    d_ned = d_cam @ rot_cam_to_ned.T  # (N, 3) north, east, down
    dz = d_ned[:, 2]
    clipped = dz < MIN_DOWN_COMPONENT
    t = agl_m / np.where(clipped, MIN_DOWN_COMPONENT, dz)
    return np.column_stack([t * d_ned[:, 0], t * d_ned[:, 1]]), clipped


@dataclass(slots=True)
class Footprint:
    """One frame's ground footprint, in metres north/east of the camera's own position.

    `poly_ne_m` is the image border projected to the ground: for a pinhole on a plane the border maps to straight
    lines, so the four corners are exact (no edge sampling needed) unless a corner had to be clipped.
    """

    poly_ne_m: np.ndarray  # (4, 2) north, east offsets from the telemetry lat/lon
    agl_m: float
    fx_px: float
    off_nadir_deg: float  # of the principal ray
    gsd_nadir_m: float  # agl / fx: the best GSD anywhere in the frame
    centre_ne_m: tuple[float, float]
    clipped: bool = False  # a corner grazed the horizon and was clipped to MIN_DOWN_COMPONENT
    valid: bool = True
    reject_reason: str = ""

    def area_m2(self) -> float:
        p = self.poly_ne_m
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def gsd_at(self, north_m: np.ndarray, east_m: np.ndarray) -> np.ndarray:
        """Per-point GSD on the ground plane: the geometric mean of the cross-range (r/f) and along-range
        (r^2/(f h)) ground samples, i.e. r^1.5 / (f sqrt(h)). Reduces exactly to h/f at nadir."""
        r = np.sqrt(np.asarray(north_m, float) ** 2 + np.asarray(east_m, float) ** 2 + self.agl_m**2)
        return (r**1.5) / (self.fx_px * math.sqrt(max(self.agl_m, 1e-6)))

    def max_off_nadir_deg(self) -> float:
        d = np.hypot(self.poly_ne_m[:, 0], self.poly_ne_m[:, 1])
        return float(math.degrees(math.atan2(d.max(), max(self.agl_m, 1e-6))))


def ground_footprint(tel: Telemetry, intr: Intrinsics, ground_asl_m: float | None = None) -> Footprint:
    """Project the image border onto the horizontal ground plane. `ground_asl_m` defaults to `tel.agl_m` below.

    A flat local ground plane is the §5.7 `flat_plane` method; a DEM would move the corners but not the machinery.
    """
    agl = tel.agl_m if ground_asl_m is None else (tel.alt_msl_m - ground_asl_m)
    rot = quat_to_rot(tel.q_gimbal)
    if not tel.gimbal_is_earth_referenced:  # camera -> body -> NED
        rot = quat_to_rot(tel.q_body) @ rot
    empty = np.zeros((4, 2))
    if agl <= 0.0:
        return Footprint(empty, 0.0, intr.fx, 0.0, 0.0, (0.0, 0.0), False, False, "camera at or below ground plane")
    principal = rot @ np.array([0.0, 0.0, 1.0])
    off_nadir = math.degrees(math.acos(max(-1.0, min(1.0, float(principal[2])))))
    w, h = float(intr.width_px), float(intr.height_px)
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    poly, clipped = ground_offsets_ne(corners, intr, rot, agl)
    centre, _ = ground_offsets_ne(np.array([[intr.cx, intr.cy]]), intr, rot, agl)
    return Footprint(
        poly_ne_m=poly,
        agl_m=float(agl),
        fx_px=float(intr.fx),
        off_nadir_deg=off_nadir,
        gsd_nadir_m=float(agl / intr.fx),
        centre_ne_m=(float(centre[0, 0]), float(centre[0, 1])),
        clipped=bool(clipped.any()),
        valid=True,
        reject_reason="",
    )


def swath_m(intr: Intrinsics, agl_m: float) -> float:
    """Appendix B: nadir footprint width = 2 h tan(HFOV/2) = agl * width_px / fx."""
    return agl_m * intr.width_px / intr.fx


def along_track_m(intr: Intrinsics, agl_m: float) -> float:
    """Nadir footprint height = 2 h tan(VFOV/2) = agl * height_px / fy."""
    return agl_m * intr.height_px / intr.fy
