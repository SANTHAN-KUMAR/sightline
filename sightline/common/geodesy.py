"""Shared WGS-84 geodesy. Orchestrator-owned so the geo, dedup, coverage and map lanes agree to the metre.

`pyproj.Geod` is the reference implementation (§5.7 step 6: "never use a single 111 km per degree constant for
longitude"). The pure-numpy fallbacks exist so modules stay importable and unit-testable when pyproj is not
loaded; they agree with pyproj to < 1 cm over the few-kilometre offsets this project uses.
"""

from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

_GEOD = None


def _geod():
    global _GEOD
    if _GEOD is None:
        from pyproj import Geod

        _GEOD = Geod(ellps="WGS84")
    return _GEOD


def meridian_radius_m(lat_deg: float) -> float:
    """Radius of curvature in the meridian (north-south) at this latitude."""
    s = math.sin(math.radians(lat_deg))
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s) ** 1.5


def normal_radius_m(lat_deg: float) -> float:
    """Radius of curvature in the prime vertical (east-west) at this latitude."""
    s = math.sin(math.radians(lat_deg))
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)


def offset_ne(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Move (north_m, east_m) from a point; returns (lat, lon). Exact enough for < 10 km offsets."""
    dlat = north_m / meridian_radius_m(lat_deg)
    lat2 = lat_deg + math.degrees(dlat)
    dlon = east_m / (normal_radius_m(lat_deg) * math.cos(math.radians(lat_deg)))
    return lat2, lon_deg + math.degrees(dlon)


def offset_ne_geodesic(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """The §5.7 step 6 path: a true WGS-84 geodesic via pyproj. Needs the `geo` dependency group."""
    az = math.degrees(math.atan2(east_m, north_m))
    dist = math.hypot(north_m, east_m)
    lon2, lat2, _ = _geod().fwd(lon_deg, lat_deg, az, dist)
    return lat2, lon2


def ne_between(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Local (north_m, east_m) of point 2 relative to point 1."""
    north = math.radians(lat2 - lat1) * meridian_radius_m((lat1 + lat2) / 2.0)
    east = math.radians(lon2 - lon1) * normal_radius_m((lat1 + lat2) / 2.0) * math.cos(math.radians((lat1 + lat2) / 2))
    return north, east


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a sphere of the local ellipsoid radius. Used for dedup radii (§5.6)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    r = math.sqrt(meridian_radius_m((lat1 + lat2) / 2) * normal_radius_m((lat1 + lat2) / 2))
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def haversine_matrix_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise distances in metres. DBSCAN with metric='haversine' wants RADIANS and returns radians, so
    prefer `radians_stack` + eps/R for clustering; this helper is for reporting and tests."""
    p = np.radians(np.asarray(lat, dtype=float))[:, None]
    l = np.radians(np.asarray(lon, dtype=float))[:, None]
    dp, dl = p - p.T, l - l.T
    a = np.sin(dp / 2) ** 2 + np.cos(p) * np.cos(p.T) * np.sin(dl / 2) ** 2
    r = math.sqrt(meridian_radius_m(float(np.mean(lat))) * normal_radius_m(float(np.mean(lat))))
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def radians_stack(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(N, 2) array in radians, the input `sklearn.cluster.DBSCAN(metric='haversine')` requires."""
    return np.column_stack([np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))])


def earth_radius_m(lat_deg: float) -> float:
    """Gaussian mean radius at a latitude — divide a metre eps by this to get DBSCAN's haversine eps."""
    return math.sqrt(meridian_radius_m(lat_deg) * normal_radius_m(lat_deg))


def quat_to_rot(q: tuple[float, float, float, float]) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix. Rotates a vector from the quaternion's source frame to its target."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Aerospace 3-2-1 (yaw, then pitch, then roll) body(FRD) -> NED, as (w, x, y, z)."""
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_to_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Inverse of `euler_to_quat`; returns (roll_deg, pitch_deg, yaw_deg)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def slerp(q0: tuple[float, ...], q1: tuple[float, ...], t: float) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion interpolation — the §5.4 attitude alignment step."""
    a, b = np.array(q0, dtype=float), np.array(q1, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
        r /= np.linalg.norm(r)
        return tuple(float(v) for v in r)  # type: ignore[return-value]
    th = math.acos(d)
    r = (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
    return tuple(float(v) for v in r)  # type: ignore[return-value]
