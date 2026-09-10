"""The waypoint format the mission lane flies, and the terrain sampler that turns AGL into ASL.

**THE CONTRACT** with `sightline/mission/` (orchestrator-owned). A `Route` serialises to::

    {
      "schema_version": "1.0.0",
      "product": "sightline.plan.route",
      "domain": "sim",
      "pattern": "boustrophedon",             # boustrophedon | expanding_square | orbit | revisit | composite
      "params":  {...},                        # everything needed to regenerate this route
      "frame":   {"centre_lat": 11.487, "centre_lon": 76.145, "base_z_m": 1046.007,
                  "axes": "north = UE +X, east = UE +Y",
                  "ue_from_local": "X_cm = north_m*100, Y_cm = east_m*100, Z_cm = (alt_asl_m - base_z_m)*100"},
      "totals":  {"n_waypoints": 42, "length_m": 5120.0, "duration_s": 780.0},
      "waypoints": [ {
          "seq": 0, "action": "goto",          # goto | orbit | loiter | hold | rtl | land
          "north_m": -320.0, "east_m": 118.0,  # SCENE frame: metres north/east of the map centre
          "alt_asl_m": 1112.4, "agl_m": 55.0,
          "lat": 11.48412, "lon": 76.14608,    # WGS-84, lat/lon order (GeoJSON below is lon/lat)
          "ue_cm": [-32000.0, 11800.0, 6639.3],
          "speed_ms": 7.0, "gimbal_pitch_deg": -90.0, "yaw_deg": null,   # null = face the direction of travel
          "dwell_s": 0.0, "orbit_radius_m": 0.0,
          "segment_id": "S03", "pass_id": 0, "reason": "boustrophedon line 3/12"
      }, ... ]
    }

Every waypoint carries BOTH the scenario's local frame and lat/lon, as the lane brief requires. `ue_cm` is the
same point in the simulator's own units, using the `ue_import` rule from `data/scene/flood_valley.json`, so the
mission lane never has to redo the conversion. `yaw_deg` is null wherever the vehicle should simply face its
direction of travel; the gimbal yaw needed to put the wide axis of the frame across-track is a separate concern
handled by `patterns.gimbal_yaw_for_heading`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from sightline.coverage.grid import SceneFrame
from sightline.schemas import SCHEMA_VERSION

Action = str  # "goto" | "orbit" | "loiter" | "hold" | "rtl" | "land"


# --- terrain ----------------------------------------------------------------------------------------------
@dataclass(slots=True)
class TerrainSampler:
    """Ground height ASL at a scene-NE point, from `data/scene/flood_valley_height.npy`.

    The array is indexed [north, east] with [0, 0] at the SW corner of the scene square (that is how
    `tools/scene/gen_terrain.py` writes it), and it spans `size_m` metres in each direction.
    """

    height: np.ndarray
    size_m: float
    default_asl_m: float = 0.0

    @classmethod
    def from_scene(cls, height_npy: str | Path, scene: SceneFrame) -> "TerrainSampler":
        h = np.load(Path(height_npy))
        return cls(np.asarray(h, dtype=np.float32), scene.size_m, float(h.mean()))

    @classmethod
    def flat(cls, asl_m: float) -> "TerrainSampler":
        return cls(np.full((2, 2), asl_m, dtype=np.float32), 1.0, asl_m)

    def __call__(self, north_m: float, east_m: float) -> float:
        n = self.height.shape[0]
        if n < 2:
            return self.default_asl_m
        u = (north_m + self.size_m / 2.0) / self.size_m * (n - 1)
        v = (east_m + self.size_m / 2.0) / self.size_m * (n - 1)
        i = int(min(max(round(u), 0), n - 1))
        j = int(min(max(round(v), 0), self.height.shape[1] - 1))
        return float(self.height[i, j])


TerrainFn = Callable[[float, float], float]


# --- waypoints --------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Waypoint:
    """One commanded point. Scene-NE, ASL/AGL, lat/lon and UE centimetres, all consistent by construction."""

    seq: int
    north_m: float
    east_m: float
    alt_asl_m: float
    agl_m: float
    lat: float
    lon: float
    speed_ms: float = 7.0
    gimbal_pitch_deg: float = -90.0
    yaw_deg: float | None = None
    action: Action = "goto"
    dwell_s: float = 0.0
    orbit_radius_m: float = 0.0
    segment_id: str = ""
    pass_id: int = 0
    reason: str = ""

    def ne(self) -> tuple[float, float]:
        return (self.north_m, self.east_m)

    def to_dict(self, scene: SceneFrame | None = None) -> dict[str, Any]:
        d = asdict(self)
        if scene is not None:
            d["ue_cm"] = list(scene.ue_cm(self.north_m, self.east_m, self.alt_asl_m))
        return d


def make_waypoint(seq: int, north_m: float, east_m: float, agl_m: float, scene: SceneFrame,
                  terrain: TerrainFn | None = None, **kw: Any) -> Waypoint:
    ground = terrain(north_m, east_m) if terrain is not None else scene.base_z_m
    lat, lon = scene.to_latlon(north_m, east_m)
    return Waypoint(seq, float(north_m), float(east_m), float(ground + agl_m), float(agl_m), lat, lon, **kw)


@dataclass(slots=True)
class Route:
    """An ordered waypoint list plus the metadata the mission lane and the UI need."""

    waypoints: list[Waypoint] = field(default_factory=list)
    pattern: str = "composite"
    params: dict[str, Any] = field(default_factory=dict)
    scene: SceneFrame | None = None
    notes: list[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""

    def __len__(self) -> int:
        return len(self.waypoints)

    def length_m(self) -> float:
        wp = self.waypoints
        return float(sum(math.dist(wp[i].ne(), wp[i + 1].ne()) for i in range(len(wp) - 1)))

    def duration_s(self, cruise_ms: float | None = None) -> float:
        """Straight-line time plus dwells. `cruise_ms` overrides the per-waypoint speed when given."""
        wp, total = self.waypoints, 0.0
        for i in range(len(wp) - 1):
            v = cruise_ms if cruise_ms else max(wp[i + 1].speed_ms, 0.1)
            total += math.dist(wp[i].ne(), wp[i + 1].ne()) / v
        return float(total + sum(w.dwell_s for w in wp))

    def renumber(self) -> "Route":
        for i, w in enumerate(self.waypoints):
            w.seq = i
        return self

    def extend(self, others: Iterable[Waypoint]) -> "Route":
        self.waypoints.extend(others)
        return self.renumber()

    def to_dict(self) -> dict[str, Any]:
        sc = self.scene
        frame = {}
        if sc is not None:
            frame = {"centre_lat": sc.centre_lat, "centre_lon": sc.centre_lon, "base_z_m": sc.base_z_m,
                     "size_m": sc.size_m, "axes": "north = UE +X, east = UE +Y",
                     "ue_from_local": "X_cm = north_m*100, Y_cm = east_m*100, Z_cm = (alt_asl_m - base_z_m)*100"}
        return {
            "schema_version": SCHEMA_VERSION,
            "product": "sightline.plan.route",
            "domain": "sim",
            "pattern": self.pattern,
            "params": self.params,
            "frame": frame,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "notes": self.notes,
            "totals": {"n_waypoints": len(self.waypoints), "length_m": round(self.length_m(), 2),
                       "duration_s": round(self.duration_s(), 1)},
            "waypoints": [w.to_dict(sc) for w in self.waypoints],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    def to_geojson(self) -> dict[str, Any]:
        """A LineString of the track plus one Point per waypoint. RFC 7946 order: [lon, lat, alt]."""
        line = [[round(w.lon, 7), round(w.lat, 7), round(w.alt_asl_m, 2)] for w in self.waypoints]
        feats: list[dict[str, Any]] = [{
            "type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
            "properties": {"pattern": self.pattern, "domain": "sim", "length_m": round(self.length_m(), 2)},
        }]
        for w in self.waypoints:
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point",
                                       "coordinates": [round(w.lon, 7), round(w.lat, 7), round(w.alt_asl_m, 2)]},
                          "properties": {k: v for k, v in w.to_dict().items() if k not in ("lat", "lon")}})
        return {"type": "FeatureCollection", "schema_version": SCHEMA_VERSION, "features": feats}
