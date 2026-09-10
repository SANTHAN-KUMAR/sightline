"""The geolocation chain: image pixel -> optical -> camera FRD -> body -> NED -> geodetic (SOLUTION_DOC §5.7).

Frames, spelled out once so nothing downstream has to guess
-----------------------------------------------------------
* **image**  ``(u, v)`` pixels, origin top-left.
* **optical** OpenCV convention: ``x`` right, ``y`` down, ``z`` forward along the boresight. This is the frame
  `cv2.undistortPoints` works in.
* **camera FRD** aerospace convention: ``x`` forward (boresight), ``y`` right, ``z`` down. Step 3 of the doc's
  chain is exactly the re-order ``(x, y, z)_frd = (z, x, y)_optical``.
* **body FRD** the airframe: ``x`` nose, ``y`` starboard, ``z`` belly.
* **NED** north, east, down, local-level at the vehicle.

`Telemetry.q_gimbal` is applied to the **camera FRD** vector by default (`ChainConfig.gimbal_frame = "frd"`),
because `common.geodesy.euler_to_quat` — the only quaternion constructor in this repo — is documented as
"body(FRD) -> NED", so every producer in this project builds `q_gimbal = euler_to_quat(roll, pitch, yaw)` with
pitch = -90 for nadir. `gimbal_frame = "optical"` is available for a producer that hands over an
optical -> NED quaternion instead (the reading `schemas.Telemetry`'s docstring suggests); both are unit-tested.

`Telemetry.gimbal_is_earth_referenced` decides composition (doc step 4):
* **True** (DJI SRT/XMP, and the simulator, whose camera pose is exact): ``d_ned = R(q_gimbal) · d_cam``.
* **False** (MAVLink `GIMBAL_DEVICE_ATTITUDE_STATUS` with a body-frame flag): ``d_ned = R(q_body) · R(q_gimbal) · d_cam``.

What this module does NOT do
----------------------------
It never averages a sim number with a real one, and it never deletes anything (guardrail R10). A ray that
cannot be intersected comes back as a `GeoFix` with ``valid = False`` and a `reject_reason`, never as an
exception and never silently dropped.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sightline.common.geodesy import offset_ne, offset_ne_geodesic, quat_to_rot
from sightline.geo.budget import CONSUMER, MAX_OFF_NADIR_DEG, BudgetInputs, error_terms, h_acc_m
from sightline.geo.dem import GLO30_SIGMA_M, DemSampler
from sightline.schemas import Detection, GeoFix, Intrinsics, Telemetry

__all__ = [
    "ChainConfig",
    "OriginGeopoint",
    "FLOODVALLEY_ORIGIN",
    "REJECTED_H_ACC_M",
    "AIRSIM_EARTH_RADIUS_M",
    "project_pixel",
    "project_pixel_ne",
    "project_detection",
    "project_detections",
    "airsim_ned_to_geodetic",
    "footprint_ned",
    "gsd_cm_px",
    "error_breakdown",
    "pixel_ray_ned",
    "ray_optical",
    "optical_to_frd",
    "boresight_ned",
    "off_nadir_deg",
    "intrinsics_from_fov",
    "hfov_from_dfov_deg",
    "effective_hfov_deg",
    "hfov_ambiguity_m",
    "geodetic_from_ned",
    "ned_from_geodetic",
    "load_scene_origin",
]

#: `h_acc_m` written onto a rejected fix. Deliberately huge and finite (not inf/NaN, which break JSON and the
#: map's circle radius) so a consumer that ignores `valid` degrades to "useless circle", never "tight circle".
REJECTED_H_ACC_M = 9999.0


# --- scenario anchoring (the simulator's NED <-> geodetic bridge) -------------------------------------------
@dataclass(frozen=True, slots=True)
class OriginGeopoint:
    """AirSim's `OriginGeopoint`. Cosys anchors it at the **UE world origin**, not the PlayerStart
    (docs/CONTEXT.md §5, measured to 0.3 m), and the FloodValley terrain uses UE X = north, UE Y = east,
    Z up — i.e. AirSim NED with down = -Z.
    """

    lat: float
    lon: float
    alt_m: float

    def to_geodetic(self, ned_m: tuple[float, float, float]) -> tuple[float, float, float]:
        n, e, d = ned_m
        lat, lon = offset_ne_geodesic(self.lat, self.lon, n, e)
        return lat, lon, self.alt_m - d

    def to_ned(self, lat: float, lon: float, alt_msl_m: float) -> tuple[float, float, float]:
        from sightline.common.geodesy import ne_between

        n, e = ne_between(self.lat, self.lon, lat, lon)
        return n, e, self.alt_m - alt_msl_m


#: docs/CONTEXT.md §5: "OriginGeopoint = 11.4870 N, 76.1450 E, 1046.007 m" (UE world Z = 0 for FloodValley).
FLOODVALLEY_ORIGIN = OriginGeopoint(11.4870, 76.1450, 1046.0069580078125)


def load_scene_origin(path: str | Path | None = None) -> OriginGeopoint:
    """Read the origin from `data/scene/flood_valley.json` if it is present, else the recorded constant."""
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "scene" / "flood_valley.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        c = j["map_centre_geopoint"]
        return OriginGeopoint(float(c["lat"]), float(c["lon"]), float(j["base_z_m"]))
    except Exception:
        return FLOODVALLEY_ORIGIN


def geodetic_from_ned(ned_m: tuple[float, float, float], origin: OriginGeopoint = FLOODVALLEY_ORIGIN):
    """Simulator NED -> (lat, lon, alt_msl_m). Offered here so ingest/mission do not re-derive it."""
    return origin.to_geodetic(ned_m)


def ned_from_geodetic(lat: float, lon: float, alt_msl_m: float, origin: OriginGeopoint = FLOODVALLEY_ORIGIN):
    return origin.to_ned(lat, lon, alt_msl_m)


#: Cosys-AirSim's earth radius. `Plugins/AirSim/Source/AirLib/include/common/common_utils/Utils.hpp:49`.
AIRSIM_EARTH_RADIUS_M = 6378137.0


def airsim_ned_to_geodetic(
    ned_m: tuple[float, float, float], origin: OriginGeopoint = FLOODVALLEY_ORIGIN
) -> tuple[float, float, float]:
    """Exact port of `EarthUtils::nedToGeodetic` (AirLib `common/EarthUtils.hpp:291-307`).

    **The simulator's lat/lon are not WGS-84 geodetic.** AirSim projects NED with an azimuthal-equidistant
    projection on a *sphere* of radius 6 378 137 m (the WGS-84 semi-major axis), so a northward offset is
    0.633 % too long compared with the true meridian arc at 11.487 N: **+0.63 m per 100 m north, +6.30 m per
    km north** (east is only 0.013 % off). Measured, not assumed — and it explains the 0.93 m gap between the
    launch-site geopoint recorded in `data/scene/flood_valley.json` and a WGS-84 geodesic from the origin.

    Consequence for the evaluation harness: a sim fix produced with `datum="wgs84"` must never be differenced
    against a sim ground-truth geopoint produced by AirSim. Compare in local NED (`project_pixel_ne`), or set
    `ChainConfig(datum="airsim_sphere", origin=...)` so both sides share the simulator's datum.
    """
    n, e, d = (float(v) for v in ned_m)
    x, y = n / AIRSIM_EARTH_RADIUS_M, e / AIRSIM_EARTH_RADIUS_M
    c = math.hypot(x, y)
    alt = origin.alt_m - d
    if c < 1e-12:
        return origin.lat, origin.lon, alt
    sc, cc = math.sin(c), math.cos(c)
    sl, cl = math.sin(math.radians(origin.lat)), math.cos(math.radians(origin.lat))
    lat = math.asin(cc * sl + (x * sc * cl) / c)
    lon = math.radians(origin.lon) + math.atan2(y * sc, c * cl * cc - x * sl * sc)
    return math.degrees(lat), math.degrees(lon), alt


# --- step 1: intrinsics, and the 4K-crop ambiguity ---------------------------------------------------------
def hfov_from_dfov_deg(dfov_deg: float, aspect_w: float, aspect_h: float) -> float:
    """Diagonal FOV -> horizontal FOV for a given frame aspect (doc step 1: "DJI publishes *diagonal* FOV")."""
    diag = math.hypot(aspect_w, aspect_h)
    f = (diag / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)
    return math.degrees(2.0 * math.atan((aspect_w / 2.0) / f))


def effective_hfov_deg(
    dfov_deg: float,
    *,
    native_aspect: tuple[float, float] = (4.0, 3.0),
    video_aspect: tuple[float, float] = (16.0, 9.0),
    mode: str = "full_width",
    zoom: float = 1.0,
) -> float:
    """Resolve the doc's "4K 16:9 video may be a crop of the 4:3 sensor" ambiguity into ONE number.

    `mode`:
      * ``"native"``      the published DFOV already describes the video frame — no crop is involved.
      * ``"full_width"``  the video keeps the full sensor **width** and cuts the top and bottom, so the HFOV is
                          the native sensor's HFOV. This is the usual 4:3 sensor -> 16:9 video path.
      * ``"full_height"`` the video keeps the full sensor **height** and cuts the sides, so the HFOV shrinks.
                          Only possible when the video aspect is *narrower* than the sensor; asking for it the
                          other way round (4:3 sensor -> 16:9 video) raises, because no crop can widen a lens.
    `zoom` is any further linear digital crop (2.0 = half the width kept).

    The caller must still pass the result into `intrinsics_from_fov` explicitly — the point of this function is
    that the assumption is written down at the call site rather than buried in a constant. Day-1 test in the
    doc: measure the effective HFOV, or read the SRT's per-frame `focal_len`.
    """
    aw, ah = native_aspect
    vw, vh = video_aspect
    if mode == "native":
        # the DFOV describes the video frame itself: focal and half-width both in video-frame units
        f = (math.hypot(vw, vh) / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)
        half_w = vw / 2.0
    else:
        # the DFOV describes the native sensor; work in native-sensor units
        f = (math.hypot(aw, ah) / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)
        if mode == "full_width":
            half_w = aw / 2.0  # top/bottom cropped -> HFOV unchanged
        elif mode == "full_height":
            half_w = (ah * vw / vh) / 2.0  # sides cropped -> HFOV shrinks
            if half_w > aw / 2.0 + 1e-12:
                raise ValueError(
                    f"video aspect {vw}:{vh} is wider than the sensor's {aw}:{ah}; a crop cannot widen the "
                    "field of view. Use mode='full_width' or mode='native'."
                )
        else:
            raise ValueError(f"unknown crop mode {mode!r}")
    return math.degrees(2.0 * math.atan((half_w / max(zoom, 1e-9)) / f))


def intrinsics_from_fov(
    width_px: int,
    height_px: int,
    effective_hfov_deg: float,
    *,
    dist: tuple[float, ...] = (),
    source: str = "fov",
    square_pixels: bool = True,
) -> Intrinsics:
    """§5.7 step 1, with the effective HFOV as an **explicit, required** argument.

    `square_pixels=True` sets fy = fx (correct for every sensor here); pass False to derive fy from the frame
    height, which is what you want only if the video is anamorphic.
    """
    if not 0.0 < effective_hfov_deg < 180.0:
        raise ValueError(f"effective_hfov_deg out of range: {effective_hfov_deg}")
    fx = (width_px / 2.0) / math.tan(math.radians(effective_hfov_deg) / 2.0)
    # A non-square-pixel camera would need its VFOV measured separately; nothing in this project has one, so
    # `square_pixels=False` is refused rather than silently guessed from the frame height.
    if not square_pixels:
        raise NotImplementedError("anamorphic video needs a measured VFOV; build the Intrinsics directly")
    return Intrinsics(width_px, height_px, fx, fx, width_px / 2.0, height_px / 2.0, tuple(dist), source)


def hfov_ambiguity_m(
    width_px: int,
    height_px: int,
    hfov_a_deg: float,
    hfov_b_deg: float,
    h_agl_m: float,
    u_px: float | None = None,
    v_px: float | None = None,
) -> float:
    """Ground distance between the fixes the two candidate HFOVs give for the same pixel, at nadir.

    This is the cost of guessing wrong in step 1, in metres — the number that decides whether the day-1 HFOV
    measurement is worth doing. Defaults to the mid-right frame edge, the worst in-frame case for HFOV error.
    """
    u = width_px - 0.5 if u_px is None else u_px
    v = height_px / 2.0 if v_px is None else v_px
    out = []
    for hfov in (hfov_a_deg, hfov_b_deg):
        intr = intrinsics_from_fov(width_px, height_px, hfov)
        du, dv = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
        out.append((h_agl_m * du, h_agl_m * dv))  # nadir: ground offset = h * tan(angle) componentwise
    return math.hypot(out[0][0] - out[1][0], out[0][1] - out[1][1])


# --- steps 2-4: the ray ------------------------------------------------------------------------------------
def ray_optical(u_px: float, v_px: float, intr: Intrinsics, undistort: bool = True) -> np.ndarray:
    """§5.7 steps 2-3: undistorted unit ray in the OPTICAL frame (x right, y down, z forward).

    Skipping the Brown-Conrady undistortion costs up to ~1 deg at the frame edge of a wide lens (doc step 2);
    `undistort=False` exists to measure exactly that, not as a shortcut.
    """
    if undistort and intr.dist:
        import cv2

        k = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        d = np.array(intr.dist, dtype=np.float64).reshape(1, -1)
        src = np.array([[[float(u_px), float(v_px)]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(src, k, d).reshape(2)
    else:
        xn = (float(u_px) - intr.cx) / intr.fx
        yn = (float(v_px) - intr.cy) / intr.fy
    d3 = np.array([xn, yn, 1.0], dtype=float)
    return d3 / np.linalg.norm(d3)


def optical_to_frd(d_opt: np.ndarray) -> np.ndarray:
    """(x right, y down, z forward) -> (x forward, y right, z down). The doc's "re-order to forward-right-down"."""
    return np.array([d_opt[2], d_opt[0], d_opt[1]], dtype=float)


def pixel_ray_ned(u_px: float, v_px: float, intr: Intrinsics, tel: Telemetry, cfg: "ChainConfig") -> np.ndarray:
    """§5.7 steps 2-4: a unit ray in NED for this pixel."""
    d_opt = ray_optical(u_px, v_px, intr, cfg.undistort)
    d_cam = d_opt if cfg.gimbal_frame == "optical" else optical_to_frd(d_opt)
    d = quat_to_rot(tel.q_gimbal) @ d_cam
    if not tel.gimbal_is_earth_referenced:
        d = quat_to_rot(tel.q_body) @ d
    n = float(np.linalg.norm(d))
    return d / n if n else d


def boresight_ned(tel: Telemetry, cfg: "ChainConfig | None" = None) -> np.ndarray:
    """The camera's optical axis in NED. Nadir -> (0, 0, 1)."""
    c = cfg or DEFAULT_CONFIG
    d_cam = np.array([0.0, 0.0, 1.0]) if c.gimbal_frame == "optical" else np.array([1.0, 0.0, 0.0])
    d = quat_to_rot(tel.q_gimbal) @ d_cam
    if not tel.gimbal_is_earth_referenced:
        d = quat_to_rot(tel.q_body) @ d
    return d / float(np.linalg.norm(d))


def off_nadir_deg(d_ned: np.ndarray) -> float:
    """θ, the ray-to-vertical angle, in degrees. This is the angle the error budget is evaluated at.

    Note this is NOT `Telemetry.gimbal_pitch_deg()`: that convenience returns -90 for every gimbal quaternion
    whose 3-2-1 pitch is 0, so it reads -90 ("straight down") for a -45 deg oblique too. Reported to the
    orchestrator; this lane computes θ from the ray and never calls it.
    """
    dz = float(np.clip(d_ned[2] / float(np.linalg.norm(d_ned)), -1.0, 1.0))
    return math.degrees(math.acos(dz))


# --- step 5: ground intersection ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChainConfig:
    """Everything the chain needs beyond the frozen schema types."""

    method: str = "flat_plane"  # "flat_plane" | "dem" | "water_plane"
    gimbal_frame: str = "frd"  # "frd" (default, see module docstring) | "optical"
    undistort: bool = True
    anchor: str = "foot"  # which pixel of a Detection to project: "foot" (ground contact) | "centre"
    min_dz: float = 0.1  # doc step 5: reject rays with d_z <= 0.1
    # datum for step 6. "wgs84" is the doc's `pyproj.Geod.fwd` and is right for real footage; "airsim_sphere"
    # reproduces Cosys-AirSim's spherical projection so sim fixes share the simulator's frame (see
    # `airsim_ned_to_geodetic`). `origin` makes the sim path exact when `Telemetry.ned_m` is present.
    datum: str = "wgs84"
    origin: OriginGeopoint | None = None
    # DEM
    dem: DemSampler | None = None
    dem_step_m: float = 3.0  # doc: "ray-march in 2-5 m steps"
    dem_max_range_m: float = 4000.0
    dem_bisect_m: float = 0.05
    dem_sigma_m: float = GLO30_SIGMA_M
    dem_fallback: bool = True  # a DEM miss falls back to the flat plane rather than rejecting
    # water
    water_level_asl_m: float | None = None  # absolute flood surface, when known (sim: `flood_level_asl_m`)
    water_offset_m: float = 0.0  # manual "the water is this far below the barometric AGL reference"
    flood_depth_m: float = 0.0  # doc: "treat DEM error as at least the flood depth"
    # budget
    budget: BudgetInputs = CONSUMER
    use_reported_gnss: bool = True  # take δp from `Telemetry.h_acc_m` rather than the preset
    use_reported_speed: bool = True  # take v for the sync term from `Telemetry.vel_ned_ms`
    use_camera_focal: bool = True  # take f for the pixel term from `Intrinsics.fx`


DEFAULT_CONFIG = ChainConfig()


def _plane_height(tel: Telemetry, cfg: ChainConfig) -> tuple[float, float, float, str]:
    """(camera height above the surface, surface ASL height, altitude 1σ, method) for the non-DEM methods."""
    if cfg.method == "water_plane":
        if cfg.water_level_asl_m is not None:
            h = float(tel.alt_msl_m) - float(cfg.water_level_asl_m)
            surface = float(cfg.water_level_asl_m)
        elif tel.flood_level_asl_m is not None:
            h = float(tel.alt_msl_m) - float(tel.flood_level_asl_m)
            surface = float(tel.flood_level_asl_m)
        else:
            h = float(tel.agl_m) + float(cfg.water_offset_m)
            surface = float(tel.alt_msl_m) - h
        # doc: over flood water, treat the vertical error as at least the flood depth
        sigma = max(cfg.budget.agl_sigma_m, float(cfg.flood_depth_m))
        return h, surface, sigma, "water_plane"
    h = float(tel.agl_m)
    return h, float(tel.alt_msl_m) - h, cfg.budget.agl_sigma_m, "flat_plane"


def _march_dem(
    tel: Telemetry, d: np.ndarray, cfg: ChainConfig
) -> tuple[float, float, float, float] | None:
    """Ray-march the DEM in `dem_step_m` steps and bisect the crossing (doc step 5).

    Returns ``(north_m, east_m, ground_asl_m, slant_range_m)`` or None when the ray never crosses terrain
    inside the tile / the range cap, or the camera starts below the DEM surface.
    """
    dem = cfg.dem
    assert dem is not None
    lat0, lon0, alt0 = float(tel.lat), float(tel.lon), float(tel.alt_msl_m)
    dn, de, dd = (float(v) for v in d)

    ts = np.arange(0.0, float(cfg.dem_max_range_m) + cfg.dem_step_m, float(cfg.dem_step_m))
    north, east = ts * dn, ts * de
    # Local flat offsets for SAMPLING only: they agree with the geodesic to < 1 cm at these ranges
    # (common/geodesy docstring) and the reported fix below uses the true geodesic.
    m_lat = 111_132.954 - 559.822 * math.cos(2 * math.radians(lat0))
    m_lon = 111_320.0 * math.cos(math.radians(lat0))
    lats = lat0 + north / m_lat
    lons = lon0 + east / m_lon
    ground = dem.elevations(lats, lons)
    ray_alt = alt0 - ts * dd
    f = ray_alt - ground  # > 0 while the ray is above terrain

    ok = np.isfinite(f)
    if not ok[0] or f[0] <= 0.0:
        return None  # camera outside the tile, or already below the surface: caller decides
    # first index where the ray is at or below terrain, before leaving the tile
    below = np.flatnonzero(ok & (f <= 0.0))
    lost = np.flatnonzero(~ok)
    if below.size == 0:
        return None
    i = int(below[0])
    if lost.size and int(lost[0]) < i:
        return None  # ran off the tile before hitting anything

    lo, hi = float(ts[i - 1]), float(ts[i])
    for _ in range(64):
        if hi - lo <= cfg.dem_bisect_m:
            break
        mid = 0.5 * (lo + hi)
        g = float(dem.elevations(np.array([lat0 + mid * dn / m_lat]), np.array([lon0 + mid * de / m_lon]))[0])
        if not math.isfinite(g):
            return None
        if (alt0 - mid * dd) - g > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return t * dn, t * de, alt0 - t * dd, t


# --- the public entry points -------------------------------------------------------------------------------
def _reject(tel: Telemetry, theta: float, reason: str, method: str, cfg: ChainConfig) -> GeoFix:
    return GeoFix(
        lat=float(tel.lat),
        lon=float(tel.lon),
        alt_msl_m=float(tel.alt_msl_m),
        h_acc_m=REJECTED_H_ACC_M,
        off_nadir_deg=theta,
        method=method,  # type: ignore[arg-type]
        h_acc_basis=f"budget_v1/{cfg.budget.name}",
        dem_source=cfg.dem.name if cfg.dem is not None else "",
        agl_m=float(tel.agl_m),
        slant_range_m=0.0,
        valid=False,
        reject_reason=reason,
    )


def project_pixel(
    u_px: float, v_px: float, tel: Telemetry, intr: Intrinsics, cfg: ChainConfig = DEFAULT_CONFIG
) -> GeoFix:
    """Project one pixel to the ground. The whole of §5.7 steps 1-7, for a single pixel.

    Never raises on bad geometry: a near-horizon ray, a non-positive AGL or a failed DEM march come back as a
    `GeoFix` with `valid = False` and a `reject_reason`.
    """
    return project_pixel_ne(u_px, v_px, tel, intr, cfg)[0]


def project_pixel_ne(
    u_px: float, v_px: float, tel: Telemetry, intr: Intrinsics, cfg: ChainConfig = DEFAULT_CONFIG
) -> tuple[GeoFix, float, float]:
    """`project_pixel` plus the LOCAL ``(north_m, east_m)`` offset of the fix from the camera.

    The local offset carries no datum, so it is the exact way for the evaluation harness to compare a sim fix
    with sim ground truth (which AirSim reports in NED). Use it instead of differencing lat/lon whenever both
    sides come from the simulator — see `airsim_ned_to_geodetic`.
    """
    d = pixel_ray_ned(u_px, v_px, intr, tel, cfg)
    theta = off_nadir_deg(d)
    dz = float(d[2])

    # step 5a: reject rays that point at or above the horizon
    if dz <= cfg.min_dz:
        return _reject(tel, theta, "near_horizon", cfg.method, cfg), 0.0, 0.0

    method = cfg.method
    dem_source = ""
    if cfg.method == "dem":
        if cfg.dem is None:
            raise ValueError("ChainConfig.method='dem' needs ChainConfig.dem")
        hit = _march_dem(tel, d, cfg)
        if hit is None:
            if not cfg.dem_fallback:
                return _reject(tel, theta, "dem_miss", "dem", cfg), 0.0, 0.0
            dem_source = f"{cfg.dem.name}(miss:fallback_flat_plane)"
            h, surface_asl, sigma_h, method = _plane_height(tel, dataclasses.replace(cfg, method="flat_plane"))
            if h <= 0.0:
                return _reject(tel, theta, "no_agl", method, cfg), 0.0, 0.0
            t = h / dz
            north, east = t * float(d[0]), t * float(d[1])
            slant = t
        else:
            north, east, surface_asl, slant = hit
            h = float(tel.alt_msl_m) - surface_asl
            sigma_h = cfg.dem_sigma_m
            dem_source = cfg.dem.name
    else:
        h, surface_asl, sigma_h, method = _plane_height(tel, cfg)
        if h <= 0.0:
            return _reject(tel, theta, "no_agl", method, cfg), 0.0, 0.0
        t = h / dz  # doc step 5: t = h / d_z
        north, east = t * float(d[0]), t * float(d[1])
        slant = t
        if cfg.dem is not None and cfg.method == "water_plane":
            dem_source = f"{cfg.dem.name}(flood_depth)"

    # step 6: north/east -> lat/lon. "wgs84" is the doc's true geodesic; "airsim_sphere" stays in the
    # simulator's frame (see `airsim_ned_to_geodetic` for why the two differ by 0.63 m per 100 m north).
    if cfg.datum == "airsim_sphere":
        origin = cfg.origin or FLOODVALLEY_ORIGIN
        if tel.ned_m is not None:
            tn, te, td = tel.ned_m
            lat, lon, _ = airsim_ned_to_geodetic((tn + north, te + east, td + h), origin)
        else:  # no NED available: apply the same projection with the camera as its own origin
            lat, lon, _ = airsim_ned_to_geodetic(
                (north, east, 0.0), OriginGeopoint(float(tel.lat), float(tel.lon), float(tel.alt_msl_m))
            )
    elif cfg.datum == "wgs84":
        lat, lon = offset_ne_geodesic(float(tel.lat), float(tel.lon), north, east)
    else:
        raise ValueError(f"unknown datum {cfg.datum!r} (use 'wgs84' or 'airsim_sphere')")

    # step 7: h_acc for THIS pixel's geometry
    inp = cfg.budget.with_(agl_sigma_m=sigma_h)
    if cfg.use_camera_focal:
        inp = inp.with_(focal_px=float(intr.fx))
    if cfg.use_reported_gnss and tel.h_acc_m > 0.0:
        inp = inp.with_(gnss_h_m=float(tel.h_acc_m))
    if cfg.use_reported_speed:
        vn, ve, _ = tel.vel_ned_ms
        inp = inp.with_(speed_ms=float(math.hypot(vn, ve)))
    acc = h_acc_m(h, theta, inp)

    fix = GeoFix(
        lat=lat,
        lon=lon,
        alt_msl_m=surface_asl,
        h_acc_m=acc,
        off_nadir_deg=theta,
        method=method,  # type: ignore[arg-type]
        h_acc_basis=f"budget_v1/{inp.name}",
        dem_source=dem_source,
        agl_m=h,
        slant_range_m=slant,
        valid=True,
        reject_reason="",
    )
    return fix, north, east


def project_detection(
    det: Detection, tel: Telemetry, intr: Intrinsics, cfg: ChainConfig = DEFAULT_CONFIG
) -> GeoFix:
    """Project a detection box. `ChainConfig.anchor` picks the pixel: "foot" (bottom-centre, a standing body's
    ground contact — the schema's own recommendation) or "centre"."""
    u, v = det.foot_px() if cfg.anchor == "foot" else det.centre_px()
    return project_pixel(u, v, tel, intr, cfg)


def project_detections(
    dets: list[Detection], tel: Telemetry, intr: Intrinsics, cfg: ChainConfig = DEFAULT_CONFIG
) -> list[GeoFix]:
    return [project_detection(d, tel, intr, cfg) for d in dets]


# --- diagnostics other lanes and the eval harness can use ---------------------------------------------------
def footprint_ned(intr: Intrinsics, tel: Telemetry, cfg: ChainConfig = DEFAULT_CONFIG) -> list[tuple[float, float]]:
    """The four image corners projected to the ground as local (north, east) offsets; [] if any corner is
    rejected. The coverage lane (F16) can build its swath from this."""
    out: list[tuple[float, float]] = []
    corners = (
        (0.5, 0.5),
        (intr.width_px - 0.5, 0.5),
        (intr.width_px - 0.5, intr.height_px - 0.5),
        (0.5, intr.height_px - 0.5),
    )
    for u, v in corners:
        fix, north, east = project_pixel_ne(u, v, tel, intr, cfg)
        if not fix.valid:
            return []
        out.append((north, east))
    return out


def gsd_cm_px(h_agl_m: float, intr: Intrinsics, off_nadir: float = 0.0) -> float:
    """Ground sample distance in cm/px for this geometry (triage stores it on every record)."""
    th = math.radians(min(abs(off_nadir), MAX_OFF_NADIR_DEG))
    return 100.0 * h_agl_m / (intr.fx * math.cos(th) ** 2)


def error_breakdown(fix: GeoFix, cfg: ChainConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """The per-term metres behind a published `h_acc_m` — for the UI's "why is this circle this big" popup."""
    inp = cfg.budget.with_(agl_sigma_m=cfg.dem_sigma_m if fix.method == "dem" else cfg.budget.agl_sigma_m)
    return error_terms(fix.agl_m, fix.off_nadir_deg, inp)
