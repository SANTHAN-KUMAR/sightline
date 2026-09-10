"""Live flight layers for the map (§5.9: "the drone's live position and footprint, the planned pattern, and
the flight track"), and the wire encoder/decoder for `Record`.

`MissionState` is a small thread-safe holder the **orchestrator's mission lane (F2/F3)** pushes into; the API
turns it into GeoJSON and the WebSocket pushes a ``mission`` frame on every update. Nothing here flies
anything — this lane never touches AirSim.

    state.update_pose(t)                      # t: sightline.schemas.Telemetry
    state.set_route(route.to_dict())          # B6's route JSON (docs/lanes/coverage_plan_eval.md §5.2)
    state.set_plan([(lat, lon), ...], name="segment-1 boustrophedon")   # fallback when there is no route
    state.set_footprint([[lon, lat], ...])    # from the geo/coverage lane, or derived nadir (below)

The planned pattern comes from the plan lane (B6) as `Route.to_dict()`: ``product == "sightline.plan.route"``,
``waypoints[i]`` carrying ``seq`` / ``action`` / ``lat`` / ``lon`` / ``alt_asl_m`` / ``agl_m`` / ``speed_ms`` /
``gimbal_pitch_deg`` / ``segment_id`` / ``pass_id`` / ``reason``. :meth:`MissionState.set_route` keeps every one
of those fields on the GeoJSON waypoint Features so the map can show *why* a leg exists, and keeps ``pattern``,
``params`` and ``totals`` on the plan FeatureCollection.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from sightline.common.geodesy import offset_ne
from sightline.schemas import Intrinsics, Telemetry

__all__ = ["MissionState", "nadir_footprint", "camera_footprint", "ROUTE_PRODUCT",
           "route_to_plan_geojson"]

#: The plan lane's route JSON product key (`sightline/plan/waypoints.py`, `Route.to_dict()`).
ROUTE_PRODUCT = "sightline.plan.route"

#: Waypoint fields carried straight through onto the GeoJSON Feature (B6's contract, §5.2 of its lane report).
_WAYPOINT_FIELDS = ("seq", "action", "alt_asl_m", "agl_m", "speed_ms", "gimbal_pitch_deg", "yaw_deg",
                    "dwell_s", "orbit_radius_m", "segment_id", "pass_id", "reason")


def route_to_plan_geojson(route: dict[str, Any]) -> dict[str, Any]:
    """B6 route JSON -> the plan FeatureCollection the map draws (LineString + one Point per waypoint).

    Raises ValueError when the payload is not a route: a silently-empty plan layer is exactly the failure
    this lane's quality gate exists to catch.
    """
    if str(route.get("product") or "") != ROUTE_PRODUCT:
        raise ValueError(f"not a {ROUTE_PRODUCT} payload: product={route.get('product')!r}")
    wps = list(route.get("waypoints") or [])
    if not wps:
        raise ValueError("route has no waypoints")
    line = [[float(w["lon"]), float(w["lat"])] for w in wps]
    params = dict(route.get("params") or {})
    name = (f"{route.get('pattern', 'route')} · {params.get('agl_m', '?')} m AGL"
            f" · {params.get('presentation', '?')}")
    feats: list[dict[str, Any]] = [{
        "type": "Feature",
        "properties": {"kind": "plan", "name": name, "pattern": route.get("pattern", ""),
                       "domain": route.get("domain", ""), "aborted": bool(route.get("aborted", False)),
                       "abort_reason": route.get("abort_reason", ""),
                       "totals": route.get("totals", {}), "notes": route.get("notes", [])},
        "geometry": {"type": "LineString", "coordinates": line},
    }]
    for w in wps:
        props: dict[str, Any] = {"kind": "waypoint"}
        for f in _WAYPOINT_FIELDS:
            if f in w:
                props[f] = w[f]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [float(w["lon"]), float(w["lat"])]}})
    return {"type": "FeatureCollection", "product": ROUTE_PRODUCT, "pattern": route.get("pattern", ""),
            "params": params, "totals": route.get("totals", {}), "features": feats}


def nadir_footprint(lat: float, lon: float, agl_m: float, intr: Intrinsics, yaw_deg: float = 0.0
                    ) -> list[list[float]]:
    """Ground rectangle seen by a perfectly nadir camera, as a closed [lon, lat] ring.

    Exact only at pitch = -90 deg. The general oblique footprint is the geo lane's (B2) job; the map takes
    whatever polygon it is handed and only falls back to this.
    """
    half_e = agl_m * math.tan(math.radians(intr.hfov_deg()) / 2.0)
    vfov = math.degrees(2.0 * math.atan((intr.height_px / 2.0) / intr.fy))
    half_n = agl_m * math.tan(math.radians(vfov) / 2.0)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    ring = []
    for dn, de in ((half_n, -half_e), (half_n, half_e), (-half_n, half_e), (-half_n, -half_e)):
        rn, re = dn * c - de * s, dn * s + de * c  # rotate the rectangle into the heading
        la, lo = offset_ne(lat, lon, rn, re)
        ring.append([lo, la])
    ring.append(ring[0])
    return ring


def camera_footprint(tel: Telemetry, intr: Intrinsics) -> list[list[float]] | None:
    """The REAL projected footprint, via the coverage lane's `ground_footprint` — the same function that
    decides which cells the POD raster credits, so the drawn quadrilateral and the swept cells agree.

    Returns a closed ``[lon, lat]`` ring, or None when the coverage package is unavailable or the ray misses
    the ground plane (near-horizon). :func:`nadir_footprint` is the fallback and is only exact at pitch -90
    **and** gimbal yaw 0; the survey gimbal is yawed +90 from the heading, so the fallback is 90 deg out.
    """
    try:
        from sightline.coverage.footprint import ground_footprint
    except Exception:  # the coverage lane is optional for the API to boot
        return None
    fp = ground_footprint(tel, intr)
    if not fp.valid:
        return None
    ring: list[list[float]] = []
    for north_m, east_m in fp.poly_ne_m:
        lat, lon = offset_ne(tel.lat, tel.lon, float(north_m), float(east_m))
        ring.append([lon, lat])
    if ring:
        ring.append(ring[0])
    return ring


class MissionState:
    """Drone pose + camera footprint + flight track + planned pattern, as GeoJSON for the map."""

    def __init__(self, track_maxlen: int = 4000):
        self._lock = threading.Lock()
        self._track: deque[tuple[float, float, float, str]] = deque(maxlen=track_maxlen)
        self._pose: dict[str, Any] | None = None
        self._footprint: list[list[float]] | None = None
        self._plan: list[tuple[float, float]] = []
        self._plan_name = ""
        self._plan_fc: dict[str, Any] | None = None  # set by set_route(): B6's route, kept whole
        self._route: dict[str, Any] | None = None
        self.updated_utc = 0.0
        self.intrinsics: Intrinsics | None = None

    # ---- producers -----------------------------------------------------------------------------------
    def update_pose(self, t: Telemetry, *, footprint: list[list[float]] | None = None) -> None:
        yaw = math.degrees(
            math.atan2(
                2.0 * (t.q_body[0] * t.q_body[3] + t.q_body[1] * t.q_body[2]),
                1.0 - 2.0 * (t.q_body[2] ** 2 + t.q_body[3] ** 2),
            )
        )
        with self._lock:
            self._pose = {
                "t_utc": t.t_utc,
                "lat": t.lat,
                "lon": t.lon,
                "alt_msl_m": t.alt_msl_m,
                "agl_m": t.agl_m,
                "yaw_deg": yaw,
                "gimbal_pitch_deg": t.gimbal_pitch_deg(),
                "mode": t.mode,
                "speed_ms": math.sqrt(sum(v * v for v in t.vel_ned_ms)),
                "h_acc_m": t.h_acc_m,
                "clip_id": t.clip_id,
                "frame_idx": t.frame_idx,
            }
            self._track.append((t.lon, t.lat, t.t_utc, t.mode))
            if footprint is not None:
                self._footprint = footprint
            elif self.intrinsics is not None and t.agl_m > 0:
                self._footprint = (camera_footprint(t, self.intrinsics)
                                   or nadir_footprint(t.lat, t.lon, t.agl_m, self.intrinsics, yaw))
            self.updated_utc = time.time()

    def set_footprint(self, ring: list[list[float]]) -> None:
        with self._lock:
            self._footprint = list(ring)
            self.updated_utc = time.time()

    def set_plan(self, waypoints: list[tuple[float, float]], *, name: str = "") -> None:
        with self._lock:
            self._plan = [(float(a), float(b)) for a, b in waypoints]
            self._plan_name = name
            self._plan_fc = None
            self.updated_utc = time.time()

    def set_route(self, route: dict[str, Any]) -> dict[str, Any]:
        """Adopt a plan-lane route JSON (`Route.to_dict()`) as the planned pattern. Returns the plan FC."""
        fc = route_to_plan_geojson(route)
        with self._lock:
            self._route = route
            self._plan_fc = fc
            self._plan = [(float(w["lat"]), float(w["lon"])) for w in route["waypoints"]]
            self._plan_name = fc["features"][0]["properties"]["name"]
            self.updated_utc = time.time()
        return fc

    def set_route_file(self, path: str | Path) -> dict[str, Any]:
        """Load a route JSON the plan lane wrote to disk."""
        return self.set_route(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def route(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._route) if self._route else None

    # ---- consumer ------------------------------------------------------------------------------------
    def as_geojson(self) -> dict[str, Any]:
        """One object with four keys, each an RFC 7946 FeatureCollection (or null when unknown)."""
        with self._lock:
            pose = dict(self._pose) if self._pose else None
            track = list(self._track)
            fp = list(self._footprint) if self._footprint else None
            plan = list(self._plan)
            plan_name = self._plan_name
            plan_fc = self._plan_fc
            updated = self.updated_utc
        drone = None
        if pose:
            drone = {
                "type": "Feature",
                "properties": pose,
                "geometry": {"type": "Point", "coordinates": [round(pose["lon"], 6), round(pose["lat"], 6)]},
            }
        footprint = (
            {"type": "Feature", "properties": {"kind": "camera_footprint"},
             "geometry": {"type": "Polygon", "coordinates": [fp]}}
            if fp
            else None
        )
        # Split the track on mode changes so MANUAL segments can be coloured differently (§7 step 3).
        segs: list[dict[str, Any]] = []
        cur: list[list[float]] = []
        cur_mode = track[0][3] if track else "AUTO"
        for lon, lat, _t, mode in track:
            if mode != cur_mode and cur:
                segs.append({"type": "Feature", "properties": {"mode": cur_mode},
                             "geometry": {"type": "LineString", "coordinates": cur}})
                cur = [cur[-1]]
                cur_mode = mode
            cur.append([round(lon, 6), round(lat, 6)])
        if len(cur) > 1:
            segs.append({"type": "Feature", "properties": {"mode": cur_mode},
                         "geometry": {"type": "LineString", "coordinates": cur}})
        if plan_fc is None and plan:
            plan_fc = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"kind": "plan", "name": plan_name},
                     "geometry": {"type": "LineString", "coordinates": [[lo, la] for la, lo in plan]}},
                    *[
                        {"type": "Feature", "properties": {"kind": "waypoint", "seq": i + 1},
                         "geometry": {"type": "Point", "coordinates": [lo, la]}}
                        for i, (la, lo) in enumerate(plan)
                    ],
                ],
            }
        return {
            "updated_utc": updated,
            "drone": drone,
            "footprint": footprint,
            "track": {"type": "FeatureCollection", "features": segs},
            "plan": plan_fc,
            "track_points": len(track),
        }
