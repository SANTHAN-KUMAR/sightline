"""F2 constraints: battery with an RTL reserve, geofence, the 120 m ceiling, and optional operator no-go areas.

§5.2 behaviour 4: "simulated battery budget with return-to-launch reserve, a geofence around the area of
operations, the 120 m ceiling, and a no-fly buffer around the buried-polygon areas **only if the commander sets
one (not automatic)**."

These are real constraints, not annotations: `apply()` walks the route, and the moment the vehicle could no longer
reach home inside the reserve it truncates the route and appends the RTL leg. A route that had to be shortened
says so in `abort_reason`, and the reason is meant to be shown, not swallowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from sightline.coverage.grid import SceneFrame
from sightline.coverage.presentation import LEGAL_CEILING_M
from sightline.plan.patterns import polygon_area_m2
from sightline.plan.waypoints import Route, TerrainFn, make_waypoint

#: A small multirotor's usable endurance in the field, deliberately conservative (§5.2 is silent on the airframe).
DEFAULT_ENDURANCE_S = 25.0 * 60.0
DEFAULT_RESERVE_FRAC = 0.20
#: Below this the drone is in ground effect and the survey geometry stops making sense.
DEFAULT_MIN_AGL_M = 15.0


@dataclass(slots=True)
class Battery:
    """Endurance budget in seconds of flight, with a fraction held back for the return leg."""

    endurance_s: float = DEFAULT_ENDURANCE_S
    reserve_frac: float = DEFAULT_RESERVE_FRAC
    spent_s: float = 0.0
    rtl_speed_ms: float = 12.0

    @property
    def usable_s(self) -> float:
        return self.endurance_s * (1.0 - self.reserve_frac)

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.usable_s - self.spent_s)

    def time_home_s(self, from_ne: Sequence[float], home_ne: Sequence[float]) -> float:
        return math.dist(tuple(from_ne), tuple(home_ne)) / max(self.rtl_speed_ms, 0.1)

    def can_continue(self, elapsed_s: float, from_ne: Sequence[float], home_ne: Sequence[float]) -> bool:
        return elapsed_s + self.time_home_s(from_ne, home_ne) <= self.usable_s


@dataclass(slots=True)
class Constraints:
    """Everything that can prune or shorten a plan. `no_go_ne` is empty unless a commander sets one."""

    home_ne: tuple[float, float] = (0.0, 0.0)
    geofence_ne: np.ndarray | None = None
    ceiling_m: float = LEGAL_CEILING_M
    min_agl_m: float = DEFAULT_MIN_AGL_M
    battery: Battery = field(default_factory=Battery)
    no_go_ne: list[np.ndarray] = field(default_factory=list)  # §5.2: only if the commander sets one
    scene: SceneFrame | None = None
    terrain: TerrainFn | None = None

    # -- checks ------------------------------------------------------------------------------------------
    def inside_geofence(self, ne: Sequence[float]) -> bool:
        if self.geofence_ne is None:
            return True
        from sightline.coverage.grid import points_in_polygon

        return bool(points_in_polygon(self.geofence_ne, np.array([[ne[0], ne[1]]]))[0])

    def inside_no_go(self, ne: Sequence[float]) -> bool:
        from sightline.coverage.grid import points_in_polygon

        pt = np.array([[ne[0], ne[1]]])
        return any(bool(points_in_polygon(p, pt)[0]) for p in self.no_go_ne)

    def clamp_altitude(self, agl_m: float) -> float:
        return float(min(max(agl_m, self.min_agl_m), self.ceiling_m))

    def clip_polygon(self, poly_ne: np.ndarray) -> np.ndarray | None:
        """Intersect a survey polygon with the geofence and subtract any operator no-go areas."""
        try:
            from shapely.geometry import Polygon
        except ImportError:  # pragma: no cover - shapely is pinned
            return poly_ne
        g = Polygon(np.asarray(poly_ne, dtype=float))
        if not g.is_valid:
            g = g.buffer(0)
        if self.geofence_ne is not None:
            g = g.intersection(Polygon(np.asarray(self.geofence_ne, dtype=float)))
        for ng in self.no_go_ne:
            g = g.difference(Polygon(np.asarray(ng, dtype=float)))
        if g.is_empty:
            return None
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        return np.array(g.exterior.coords[:-1], dtype=float)

    # -- application -------------------------------------------------------------------------------------
    def apply(self, route: Route, start_ne: Sequence[float] | None = None, elapsed_s: float = 0.0,
              append_rtl: bool = True) -> Route:
        """Truncate `route` at the last waypoint from which RTL is still affordable, then append the RTL leg.

        Waypoints outside the geofence or inside a commander-set no-go area are dropped (and counted). Altitudes
        are clamped into [min_agl, ceiling]. The returned Route is a NEW object; the input is not mutated.
        """
        home = self.home_ne
        pos = tuple(start_ne) if start_ne is not None else home
        out = Route(pattern=route.pattern, scene=route.scene or self.scene, params=dict(route.params),
                    notes=list(route.notes))
        t = float(elapsed_s)
        dropped_fence = dropped_nogo = clamped = 0
        for wp in route.waypoints:
            ne = wp.ne()
            if not self.inside_geofence(ne):
                dropped_fence += 1
                continue
            if self.inside_no_go(ne):
                dropped_nogo += 1
                continue
            leg = math.dist(pos, ne) / max(wp.speed_ms, 0.1)
            if not self.battery.can_continue(t + leg + wp.dwell_s, ne, home):
                out.aborted = True
                out.abort_reason = (
                    f"battery reserve: continuing to waypoint {wp.seq} would need "
                    f"{t + leg + wp.dwell_s + self.battery.time_home_s(ne, home):.0f} s of the "
                    f"{self.battery.usable_s:.0f} s usable budget ({self.battery.reserve_frac * 100:.0f} % held "
                    f"for return)")
                break
            new = replace(wp)
            agl = self.clamp_altitude(wp.agl_m)
            if abs(agl - wp.agl_m) > 1e-9:
                clamped += 1
                ground = new.alt_asl_m - new.agl_m
                new.agl_m, new.alt_asl_m = agl, ground + agl
                new.reason = (new.reason + f" [altitude clamped to {agl:.0f} m AGL]").strip()
            out.waypoints.append(new)
            t += leg + wp.dwell_s
            pos = ne
        if dropped_fence:
            out.notes.append(f"{dropped_fence} waypoints dropped: outside the geofence")
        if dropped_nogo:
            out.notes.append(f"{dropped_nogo} waypoints dropped: inside an operator no-go area")
        if clamped:
            out.notes.append(f"{clamped} waypoints clamped into [{self.min_agl_m:.0f}, {self.ceiling_m:.0f}] m AGL")
        if append_rtl and (out.waypoints or out.aborted):
            last = out.waypoints[-1] if out.waypoints else None
            agl = self.clamp_altitude(last.agl_m if last else self.min_agl_m)
            scene = out.scene
            if scene is not None:
                out.waypoints.append(make_waypoint(
                    len(out.waypoints), home[0], home[1], agl, scene, self.terrain,
                    speed_ms=self.battery.rtl_speed_ms, action="rtl",
                    reason=out.abort_reason or "pattern complete: return to launch"))
        out.params["constraints"] = self.as_dict()
        out.params["elapsed_s_at_end"] = round(t, 1)
        return out.renumber()

    def as_dict(self) -> dict:
        return {
            "home_ne": list(self.home_ne),
            "ceiling_m": self.ceiling_m,
            "min_agl_m": self.min_agl_m,
            "geofence_area_m2": None if self.geofence_ne is None else round(polygon_area_m2(self.geofence_ne), 1),
            "no_go_areas": len(self.no_go_ne),
            "battery": {"endurance_s": self.battery.endurance_s, "reserve_frac": self.battery.reserve_frac,
                        "usable_s": self.battery.usable_s, "rtl_speed_ms": self.battery.rtl_speed_ms},
        }


def time_to_fly_route(route: Route, cruise_ms: float | None = None) -> float:
    return route.duration_s(cruise_ms)
