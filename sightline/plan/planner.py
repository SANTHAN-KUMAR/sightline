"""F2b — the decision planner: Koopman allocation plus greedy action scoring (SOLUTION_DOC §5.3a, Appendix B).

Two layers, exactly as the doc specifies.

**Layer 1, strategic allocation (closed form).** Koopman/Stone:

    c_i* = max(ln p_i - lambda, 0),  lambda chosen so that sum c_i* = the effort available

a water-filling solution: cells whose prior falls below `e^lambda` get **zero** effort. It answers "how many
minutes does each segment deserve before anyone takes off", in milliseconds, by bisection on lambda.

**Layer 2, tactical scoring (greedy, replanned every 30-60 s).**

    value(a) = [ sum_cells POA(cell) * dPOD_mix(cell, a) ] / ( t_transit(a) + t_execute(a) )
    dPOD_j   = e^(-k C_before) - e^(-k C_after),   C_after = C_before + q_pass(cell, j, a)
    dPOD_mix = sum_j w_j(zone) * dPOD_j                                              (§5.3b)

with the candidate set of §5.3a: continue the current swath; survey segment k at an altitude band and a sensor
band; orbit and dwell on a candidate record; loiter until time T (the pre-dawn thermal window); return to launch.
Hard constraints prune the set **before** scoring.

**Graceful degradation is the safety property, and it is tested.** With a flat prior and nothing searched, every
cell has the same `p_i` and the same dPOD, so the argmax reduces to "cover the nearest unsearched area
efficiently" — the plain boustrophedon. There is no separate fallback mode: `Decision.route` under a flat prior is
byte-for-byte the route `patterns.boustrophedon_route` produces on its own.

**Explainability.** Every decision keeps its ranked candidates with their scores and a one-line reason, because
"chose Segment B at 0.031 expected finds per minute over continuing swath 7 at 0.004" is the product.

**Guardrail.** Cells inside burial polygons contribute zero dPOD by construction (`CoverageMap.delta_pod` zeroes
them), so the planner never spends effort pretending to clear them, and nothing here marks anything complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sightline.coverage.accumulate import CoverageMap
from sightline.coverage.grid import SceneFrame, scene_poly_to_grid
from sightline.coverage.presentation import SIM_RGB_4K, CameraModel
from sightline.coverage.quality import Conditions, SliceTable
from sightline.plan.constraints import Constraints
from sightline.plan.patterns import (
    DEFAULT_SIDE_OVERLAP,
    PatternSpec,
    boustrophedon_route,
    orbit_route,
    polygon_area_m2,
)
from sightline.plan.segments import Segment
from sightline.plan.waypoints import Route, TerrainFn

#: §5.3a candidate set: "change altitude band (30 / 45 / 60 / 90 / 120 m)".
ALTITUDE_BANDS_M: tuple[float, ...] = (30.0, 45.0, 60.0, 90.0, 120.0)
#: The altitude a confirmation orbit descends to; §5.3b's limb-only ceiling for a 4K wide camera is 24 m.
CONFIRM_AGL_M = 25.0
TOP_N_EXPLAINED = 3


# --- Layer 1: Koopman / Stone optimal allocation ----------------------------------------------------------
def koopman_allocation(prior: np.ndarray, budget: float, mask: np.ndarray | None = None,
                       iters: int = 200) -> tuple[np.ndarray, float]:
    """`c_i* = max(ln p_i - lambda, 0)` with lambda set so the allocation sums to `budget`.

    Returns (allocation, lambda). Cells with zero prior, and cells outside `mask`, receive zero.
    A FLAT prior gives a flat allocation — the property the degradation argument rests on.
    """
    p = np.asarray(prior, dtype=np.float64)
    live = np.ones(p.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    live = live & (p > 0.0)
    if budget <= 0.0 or not live.any():
        return np.zeros_like(p, dtype=np.float32), float("inf")
    lp = np.log(np.where(live, p, 1.0))
    hi = float(lp[live].max())
    lo = float(lp[live].min()) - budget / max(int(live.sum()), 1) - 1.0

    def total(lam: float) -> float:
        return float(np.clip(lp - lam, 0.0, None)[live].sum())

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if total(mid) > budget:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    c = np.where(live, np.clip(lp - lam, 0.0, None), 0.0)
    s = float(c.sum())
    if s > 0:  # exact budget: bisection converges to machine noise, so normalise the last epsilon away
        c = c * (budget / s)
    return c.astype(np.float32), float(lam)


def allocate_segment_minutes(prior: np.ndarray, segments: Sequence[Segment], cmap: CoverageMap, scene: SceneFrame,
                             total_minutes: float) -> dict[str, float]:
    """Layer 1 expressed the way a commander reads it: minutes of flight per segment, before anyone takes off."""
    alloc, _ = koopman_allocation(prior, 1.0, mask=~cmap.cannot_clear)
    out: dict[str, float] = {}
    for s in segments:
        out[s.seg_id] = float(alloc[s.mask(cmap, scene)].sum() * total_minutes)
    return out


# --- Layer 2: candidates ----------------------------------------------------------------------------------
_DROP = object()  # sentinel: a candidate parameter that has no place in a serialised decision log


def _jsonable(v: Any) -> Any:
    """Reduce one candidate parameter to something `json.dumps` accepts, or `_DROP`."""
    if isinstance(v, np.ndarray):
        return _DROP
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if isinstance(v, Segment):
        return v.seg_id
    if isinstance(v, PatternSpec):
        return {"camera": v.camera.name, "agl_m": v.agl_m, "presentation": v.presentation,
                "side_overlap": v.side_overlap, "sweep_width_m": round(v.sweep_width(), 2),
                "spacing_m": round(v.spacing(), 2), "speed_ms": round(v.speed(), 2), "band": v.band}
    if isinstance(v, Route):
        return {"pattern": v.pattern, "n_waypoints": len(v)}
    if isinstance(v, Candidate):
        return v.label
    if isinstance(v, PendingRecord):
        return {"record_id": v.record_id, "ne": list(v.ne), "score": v.score, "radius_m": v.radius_m}
    if isinstance(v, dict):
        return {str(k): x for k, x in ((k, _jsonable(x)) for k, x in v.items()) if x is not _DROP}
    if isinstance(v, (list, tuple)):
        return [x for x in (_jsonable(x) for x in v) if x is not _DROP]
    return str(v)


@dataclass(slots=True)
class Candidate:
    """One scored action. `value` is expected finds per MINUTE (the doc's formula is per second; stated here)."""

    kind: str  # "survey" | "continue" | "orbit" | "loiter" | "rtl"
    label: str
    value: float = 0.0
    gain: float = 0.0
    t_transit_s: float = 0.0
    t_execute_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    feasible: bool = True
    infeasible_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe. The decision timeline is a logged, serialised product, so nothing here may be an object
        `json.dumps` cannot write: segments, pattern specs and routes are reduced to the identifier that names
        them, arrays are dropped, and numpy scalars become Python floats."""
        return {"kind": self.kind, "label": self.label, "value": round(self.value, 6),
                "gain": round(self.gain, 6), "t_transit_s": round(self.t_transit_s, 1),
                "t_execute_s": round(self.t_execute_s, 1), "reason": self.reason,
                "feasible": self.feasible, "infeasible_reason": self.infeasible_reason,
                "params": {k: v for k, v in ((k, _jsonable(v)) for k, v in self.params.items())
                           if v is not _DROP}}


@dataclass(slots=True)
class PendingRecord:
    """A candidate record the planner may choose to orbit and confirm (§5.2 behaviour 2)."""

    record_id: str
    ne: tuple[float, float]
    score: float = 1.0
    radius_m: float = 30.0


@dataclass(slots=True)
class PlannerState:
    """Everything the planner reads. It writes nothing back — `decide()` is a pure function of this."""

    cmap: CoverageMap
    scene: SceneFrame
    poa: np.ndarray
    segments: list[Segment]
    position_ne: tuple[float, float] = (0.0, 0.0)
    elapsed_s: float = 0.0
    local_hour: float | None = None
    weather: str = "dry"
    constraints: Constraints | None = None
    camera: CameraModel = SIM_RGB_4K
    thermal_camera: CameraModel | None = None
    terrain: TerrainFn | None = None
    slice_table: SliceTable = field(default_factory=SliceTable)
    current_segment: Segment | None = None
    current_route: Route | None = None
    pending_records: list[PendingRecord] = field(default_factory=list)
    cruise_ms: float = 12.0
    side_overlap: float = DEFAULT_SIDE_OVERLAP


@dataclass(slots=True)
class Decision:
    chosen: Candidate | None
    ranked: list[Candidate]
    explanation: str
    route: Route | None = None
    allocation_minutes: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen.as_dict() if self.chosen else None,
                "ranked": [c.as_dict() for c in self.ranked[:TOP_N_EXPLAINED]],
                "explanation": self.explanation, "allocation_minutes": self.allocation_minutes,
                "n_waypoints": len(self.route) if self.route else 0, "domain": "sim"}


class DecisionPlanner:
    """The §5.3a planner. `altitudes` and `bands` define the candidate set; the defaults are the doc's."""

    def __init__(self, altitudes_m: Sequence[float] = (45.0, 60.0, 90.0), bands: Sequence[str] = ("rgb",),
                 presentation: str = "body", max_segments: int = 4, dawn_hour: float = 5.0,
                 min_px: float = 20.0) -> None:
        self.altitudes_m = tuple(altitudes_m)
        self.bands = tuple(bands)
        self.presentation = presentation
        self.max_segments = int(max_segments)
        self.dawn_hour = float(dawn_hour)
        self.min_px = float(min_px)

    # -- q_pass for a whole action ------------------------------------------------------------------------
    def _q_for(self, st: PlannerState, agl_m: float, band: str, hour: float | None,
               spec: PatternSpec) -> dict[str, float]:
        cam = spec.camera
        gsd = cam.gsd_m(agl_m)
        cond = Conditions(band=band, time_of_day=_tod(hour), local_hour=hour, weather=st.weather,
                          speed_ms=spec.speed(), exposure_s=spec.exposure_s)
        return {p: float(st.slice_table.lookup(p, agl_m, cond, gsd, 0.0).value) for p in st.cmap.presentations}

    def _spec(self, st: PlannerState, agl_m: float, band: str) -> PatternSpec:
        cam = st.camera if band == "rgb" or st.thermal_camera is None else st.thermal_camera
        return PatternSpec(camera=cam, agl_m=agl_m, presentation=self.presentation,
                           side_overlap=st.side_overlap, band=band)

    def _gain_over(self, st: PlannerState, mask: np.ndarray, q: dict[str, float]) -> float:
        d = st.cmap.delta_pod_mixture(q)
        return float((np.asarray(st.poa, dtype=np.float32) * d)[mask].sum())

    # -- candidate builders --------------------------------------------------------------------------------
    def _survey_candidates(self, st: PlannerState, hour: float | None, prefix: str = "") -> list[Candidate]:
        out: list[Candidate] = []
        segs = sorted(st.segments, key=lambda s: -s.poa_mass)[: self.max_segments]
        for seg in segs:
            mask = seg.mask(st.cmap, st.scene)
            if not mask.any():
                continue
            area = polygon_area_m2(seg.poly_ne)
            t_transit = math.dist(st.position_ne, seg.centroid_ne()) / max(st.cruise_ms, 0.1)
            for agl in self.altitudes_m:
                for band in self.bands:
                    spec = self._spec(st, agl, band)
                    if spec.sweep_width() <= 0.0:
                        continue
                    q = self._q_for(st, agl, band, hour, spec)
                    gain = self._gain_over(st, mask, q)
                    t_exec = area / max(spec.area_rate_m2_s(), 1e-6)
                    value = gain / max((t_transit + t_exec) / 60.0, 1e-9)
                    c = Candidate("survey", f"{prefix}survey {seg.seg_id} at {agl:.0f} m {band}", value, gain,
                                  t_transit, t_exec,
                                  {"segment": seg, "agl_m": agl, "band": band, "spec": spec, "q": q,
                                   "hour": hour},
                                  reason=(f"{int(mask.sum())} cells, POA {float(np.asarray(st.poa)[mask].sum()):.4f}, "
                                          f"q_body {q.get(self.presentation, 0.0):.2f}, sweep "
                                          f"{spec.sweep_width():.0f} m, {t_exec / 60.0:.1f} min to fly"))
                    self._check_feasible(st, c, seg.centroid_ne(), agl)
                    out.append(c)
        return out

    def _continue_candidate(self, st: PlannerState, hour: float | None) -> Candidate | None:
        if st.current_route is None or not st.current_route.waypoints or st.current_segment is None:
            return None
        rt = st.current_route
        agl = float(rt.params.get("agl_m", self.altitudes_m[0]))
        band = str(rt.params.get("band", self.bands[0]))
        spec = self._spec(st, agl, band)
        q = self._q_for(st, agl, band, hour, spec)
        mask = st.current_segment.mask(st.cmap, st.scene)
        gain = self._gain_over(st, mask, q)
        nxt = rt.waypoints[0].ne()
        t_transit = math.dist(st.position_ne, nxt) / max(st.cruise_ms, 0.1)
        t_exec = rt.duration_s()
        c = Candidate("continue", f"continue the current swath in {st.current_segment.seg_id}",
                      gain / max((t_transit + t_exec) / 60.0, 1e-9), gain, t_transit, t_exec,
                      {"route": rt, "segment": st.current_segment, "agl_m": agl, "band": band, "q": q},
                      reason=f"{len(rt)} waypoints left, {t_exec / 60.0:.1f} min")
        self._check_feasible(st, c, nxt, agl)
        return c

    def _orbit_candidates(self, st: PlannerState, hour: float | None) -> list[Candidate]:
        out: list[Candidate] = []
        grid = st.cmap.any_grid
        nn, ee = st.cmap.cell_centres()
        for rec in st.pending_records:
            spec = self._spec(st, CONFIRM_AGL_M, "rgb")
            if spec.sweep_width() <= 0.0:
                continue
            g = scene_poly_to_grid(grid, st.scene, np.array([[rec.ne[0], rec.ne[1]]]))[0]
            mask = ((nn - g[0]) ** 2 + (ee - g[1]) ** 2) <= rec.radius_m**2
            if not mask.any():
                continue
            q = self._q_for(st, CONFIRM_AGL_M, "rgb", hour, spec)
            gain = self._gain_over(st, mask, q) * max(rec.score, 0.0)
            t_transit = math.dist(st.position_ne, rec.ne) / max(st.cruise_ms, 0.1)
            t_exec = 2.0 * math.pi * rec.radius_m / 3.0 + 8 * 2.0
            c = Candidate("orbit", f"orbit and confirm {rec.record_id} at {CONFIRM_AGL_M:.0f} m",
                          gain / max((t_transit + t_exec) / 60.0, 1e-9), gain, t_transit, t_exec,
                          {"record": rec, "agl_m": CONFIRM_AGL_M, "q": q},
                          reason=(f"descend-to-confirm: q_limb_only "
                                  f"{q.get('limb_only', 0.0):.2f} at {CONFIRM_AGL_M:.0f} m versus "
                                  f"{self._q_for(st, self.altitudes_m[-1], 'rgb', hour, self._spec(st, self.altitudes_m[-1], 'rgb')).get('limb_only', 0.0):.2f} at "
                                  f"{self.altitudes_m[-1]:.0f} m"))
            self._check_feasible(st, c, rec.ne, CONFIRM_AGL_M)
            out.append(c)
        return out

    def _loiter_candidate(self, st: PlannerState) -> Candidate | None:
        """§5.3a behaviour 3: loiter until the pre-dawn thermal window can outscore flying now."""
        if st.local_hour is None or st.thermal_camera is None:
            return None
        wait_h = (self.dawn_hour - st.local_hour) % 24.0
        if wait_h <= 0.0 or wait_h > 12.0:
            return None
        wait_s = wait_h * 3600.0
        future = self._survey_candidates(st, self.dawn_hour, prefix="after loiter: ")
        future = [c for c in future if c.feasible]
        if not future:
            return None
        best = max(future, key=lambda c: c.gain / max(c.t_transit_s + c.t_execute_s, 1e-9))
        total = wait_s + best.t_transit_s + best.t_execute_s
        return Candidate("loiter", f"loiter until {self.dawn_hour:04.1f} local, then {best.label}",
                         best.gain / max(total / 60.0, 1e-9), best.gain, wait_s, best.t_execute_s,
                         {"until_hour": self.dawn_hour, "then": best},
                         reason=(f"waiting {wait_h:.1f} h buys q_body {best.params['q'].get(self.presentation, 0):.2f} "
                                 f"in the pre-dawn thermal window instead of flying now"),
                         feasible=True)

    def _check_feasible(self, st: PlannerState, c: Candidate, target_ne: Sequence[float], agl_m: float) -> None:
        con = st.constraints
        if con is None:
            return
        if agl_m > con.ceiling_m:
            c.feasible, c.infeasible_reason = False, f"above the {con.ceiling_m:.0f} m ceiling"
            return
        if not con.inside_geofence(target_ne):
            c.feasible, c.infeasible_reason = False, "outside the geofence"
            return
        if con.inside_no_go(target_ne):
            c.feasible, c.infeasible_reason = False, "inside an operator no-go area"
            return
        need = st.elapsed_s + c.t_transit_s + c.t_execute_s + con.battery.time_home_s(target_ne, con.home_ne)
        if need > con.battery.usable_s:
            c.feasible = False
            c.infeasible_reason = (f"battery: needs {need / 60.0:.1f} min of the "
                                   f"{con.battery.usable_s / 60.0:.1f} min usable budget")

    # -- the decision --------------------------------------------------------------------------------------
    def decide(self, st: PlannerState, build_route: bool = True, total_minutes: float = 0.0) -> Decision:
        cands: list[Candidate] = []
        cont = self._continue_candidate(st, st.local_hour)
        if cont is not None:
            cands.append(cont)
        cands += self._survey_candidates(st, st.local_hour)
        cands += self._orbit_candidates(st, st.local_hour)
        loiter = self._loiter_candidate(st)
        if loiter is not None:
            cands.append(loiter)
        cands.append(Candidate("rtl", "return to launch", 0.0, 0.0,
                               math.dist(st.position_ne, st.constraints.home_ne) / max(st.cruise_ms, 0.1)
                               if st.constraints else 0.0, 0.0, {},
                               reason="always available; the only action when nothing else is affordable"))
        feasible = [c for c in cands if c.feasible]
        ranked = sorted(feasible, key=lambda c: -c.value) + sorted([c for c in cands if not c.feasible],
                                                                  key=lambda c: -c.value)
        chosen = ranked[0] if ranked and ranked[0].feasible else next((c for c in ranked if c.kind == "rtl"), None)
        expl = _explain(chosen, ranked)
        alloc = (allocate_segment_minutes(st.poa, st.segments, st.cmap, st.scene, total_minutes)
                 if total_minutes > 0 else {})
        route = self.route_for(st, chosen) if (build_route and chosen is not None) else None
        return Decision(chosen, ranked, expl, route, alloc)

    def route_for(self, st: PlannerState, c: Candidate) -> Route | None:
        """Turn a chosen candidate into waypoints. A `survey` produces exactly `boustrophedon_route`'s output."""
        if c.kind == "continue":
            return c.params["route"]
        if c.kind == "survey":
            seg: Segment = c.params["segment"]
            spec: PatternSpec = c.params["spec"]
            poly = seg.poly_ne
            if st.constraints is not None:
                clipped = st.constraints.clip_polygon(poly)
                if clipped is None:
                    return None
                poly = clipped
            rt = boustrophedon_route(poly, spec.camera, spec.agl_m, st.scene, presentation=self.presentation,
                                     side_overlap=spec.side_overlap, terrain=st.terrain,
                                     segment_id=seg.seg_id, min_px=self.min_px)
            rt.params["band"] = c.params["band"]
            rt.params["chosen_by"] = c.label
            rt.params["decision_reason"] = c.reason
            return st.constraints.apply(rt, st.position_ne, st.elapsed_s) if st.constraints else rt
        if c.kind == "orbit":
            rec: PendingRecord = c.params["record"]
            rt = orbit_route(rec.ne, rec.radius_m, CONFIRM_AGL_M, st.scene, terrain=st.terrain,
                             record_id=rec.record_id)
            return st.constraints.apply(rt, st.position_ne, st.elapsed_s) if st.constraints else rt
        if c.kind == "loiter":
            return None
        if c.kind == "rtl" and st.constraints is not None:
            rt = Route(pattern="rtl", scene=st.scene, params={"reason": c.reason})
            return st.constraints.apply(rt, st.position_ne, st.elapsed_s)
        return None


def _tod(hour: float | None) -> str:
    from sightline.coverage.accumulate import time_of_day_label

    return time_of_day_label(hour)


def _explain(chosen: Candidate | None, ranked: list[Candidate]) -> str:
    if chosen is None:
        return "no feasible action"
    others = [c for c in ranked if c is not chosen][: TOP_N_EXPLAINED - 1]
    tail = "; ".join(f"{c.label} at {c.value:.4f}" for c in others)
    s = f"chose {chosen.label} at {chosen.value:.4f} expected finds per minute"
    if tail:
        s += f" over {tail}"
    if chosen.value < 1e-6:
        s += " — every candidate scores about zero, so re-flying adds nothing measurable"
    return s + " (simulation)"
