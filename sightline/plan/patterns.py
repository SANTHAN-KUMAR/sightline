"""F2 — coverage patterns: boustrophedon, expanding square and orbit (SOLUTION_DOC §5.2, Appendix B).

Fields2Cover was the doc's first choice for the geometric layer and was **rejected on this project**: it does not
build on Windows (`docs/HANDBOOK.md` §6). The doc's own fallback applies — "the boustrophedon generator is ~60
lines and is written by the team" — and that is what this module is.

Nothing here hard-codes a line spacing. The chain is the one in Appendix B:

    GSD          = 2 h tan(HFOV/2) / W_px
    swath        = 2 h tan(HFOV/2)
    sweep width  = the part of the swath where the target still spans >= min_px, which shrinks toward the frame
                   edge because the slant range grows (this is narrower than the swath for marginal presentations)
    line spacing = sweep width x (1 - side overlap)
    speed limit  = max_blur_px x GSD / t_exp

so changing the altitude, the camera or the presentation being searched for changes the pattern, which is the
whole point of tying the planner to the coverage model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sightline.coverage.grid import SceneFrame
from sightline.coverage.presentation import CRITICAL_DIM_M, MIN_PX_FOR_RECALL, CameraModel
from sightline.plan.waypoints import Route, TerrainFn, Waypoint, make_waypoint

#: §5.2: "spaced by the camera swath at the chosen altitude with 20-30 % side overlap".
DEFAULT_SIDE_OVERLAP = 0.25
#: §2.5: "keep shutter <= GSD / ground speed"; one pixel of smear is the budget.
MAX_BLUR_PX = 1.0


# --- 1. the geometry chain --------------------------------------------------------------------------------
def swath_m(camera: CameraModel, agl_m: float) -> float:
    """Appendix B: full nadir footprint width = 2 h tan(HFOV/2)."""
    return camera.swath_m(agl_m)


def sweep_width_m(camera: CameraModel, agl_m: float, presentation: str = "body",
                  min_px: float = MIN_PX_FOR_RECALL) -> float:
    """The across-track width over which `presentation` still spans `min_px`, capped by the geometric swath.

    GSD grows away from nadir as r^1.5 / (f sqrt(h)) (the geometric mean of the cross- and along-range samples),
    so the usable half-width x solves  (h^2 + x^2)^(3/4) / (f sqrt(h)) = L / min_px. For a well-resolved target
    the whole swath qualifies and this returns the swath; for a marginal one it returns less, and the pattern
    tightens on its own.
    """
    geo = swath_m(camera, agl_m)
    length_m = CRITICAL_DIM_M.get(presentation, 1.70)
    if length_m <= 0.0 or agl_m <= 0.0:
        return 0.0
    r_max = (length_m * camera.fx_px * math.sqrt(agl_m) / min_px) ** (2.0 / 3.0)
    if r_max <= agl_m:
        return 0.0  # not even the nadir pixel resolves the target at this altitude
    return float(min(geo, 2.0 * math.sqrt(r_max * r_max - agl_m * agl_m)))


def line_spacing_m(sweep_width: float, side_overlap: float = DEFAULT_SIDE_OVERLAP) -> float:
    """Appendix B: line spacing = width x (1 - side overlap)."""
    return float(sweep_width * (1.0 - min(max(side_overlap, 0.0), 0.95)))


def speed_limit_ms(camera: CameraModel, agl_m: float, exposure_s: float = 1.0 / 500.0,
                   max_blur_px: float = MAX_BLUR_PX) -> float:
    """Appendix B: blur_px = v t_exp / GSD, so v <= max_blur_px x GSD / t_exp."""
    return float(max_blur_px * camera.gsd_m(agl_m) / max(exposure_s, 1e-9))


def altitude_for_min_px(camera: CameraModel, presentation: str, min_px: float = MIN_PX_FOR_RECALL,
                        legal_cap: float = 120.0) -> float:
    """The §5.3b ceiling: the highest AGL at which `presentation` still spans `min_px` at nadir."""
    return camera.ceiling_m(presentation, min_px, legal_cap)


def gimbal_yaw_for_heading(heading_deg: float) -> float:
    """Point the WIDE axis of the frame across-track.

    At nadir the image +u axis lies **perpendicular** to the gimbal yaw: at yaw 0 the frame is north-up and
    +u is due EAST. The wide axis is therefore across-track when the gimbal yaw EQUALS the heading.

    This returned `heading + 90` until 2026-09-11, compensating for a quarter-turn in
    `coverage/footprint.py`, which applied the identity `q_gimbal` of a nadir camera to an OPTICAL ray and so
    mapped image-right to north. With that fixed at the root, the +90 became a double rotation. The truth was
    settled by measurement rather than by argument: across 110 boxes whose survivors have known world
    positions, image-right is due east (median residual 1.37 m; the next-best hypothesis 15.8 m).
    """
    return heading_deg % 360.0


def bearing_deg(from_ne: Sequence[float], to_ne: Sequence[float]) -> float:
    return math.degrees(math.atan2(to_ne[1] - from_ne[1], to_ne[0] - from_ne[0])) % 360.0


# --- 2. boustrophedon -------------------------------------------------------------------------------------
@dataclass(slots=True)
class PatternGeometry:
    """What the spacing calculation decided, kept so the report and the UI can state it."""

    sweep_width_m: float
    nominal_spacing_m: float
    actual_spacing_m: float
    n_lines: int
    side_overlap_nominal: float
    side_overlap_actual: float
    heading_deg: float
    speed_limit_ms: float
    gsd_m: float
    covers_polygon: bool = True
    #: Measured, not assumed: the share of the polygon further than sweep_width/2 from every flown line
    #: (`uncovered_fraction`). `covers_polygon` is this being zero, so the field cannot claim a coverage the
    #: geometry does not deliver — a concave segment can leave a gap even at the requested side overlap.
    uncovered_fraction: float = 0.0
    gap_sample_step_m: float = 0.0
    #: Worst distance, in metres, by which an uncovered sample exceeds sweep_width/2 — the size of the gap, not
    #: just its area. A corner sliver and a missed strip both show as a fraction; only this separates them.
    worst_gap_m: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of `pts` (N, 2) to each segment a[k]->b[k] (K, 2). Returns (N, K)."""
    d = b - a  # (K, 2)
    ll = (d * d).sum(axis=1)  # (K,)
    ap = pts[:, None, :] - a[None, :, :]  # (N, K, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ll > 0, (ap * d[None, :, :]).sum(axis=2) / np.where(ll > 0, ll, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2)


def uncovered_fraction(poly_ne: np.ndarray, lines: Sequence[tuple[tuple[float, float], tuple[float, float]]],
                       sweep_width: float, max_samples: int = 4096) -> tuple[float, float, float]:
    """Share of the polygon further than `sweep_width / 2` from EVERY flown line, by area sampling.

    This is the honest form of "no gaps": it measures against the flown line *segments*, so a line that had to
    be clipped short gets no credit for ground it never overflew. Returns (fraction, sample step m, worst
    excess distance beyond sweep_width/2 in m).

    **The known residual, stated rather than hidden.** Lines are clipped to the polygon, so the swath stops at
    the polygon edge. For a polygon flown along a heading that is not one of its own edges, the strip nearest a
    corner can therefore miss a sliver *beyond the end of the last line* — bounded by the corner's reach past
    that endpoint, typically well under a metre on a rectangular segment. Real coverage planners close this with
    a headland overrun (flying past the boundary before turning); that is deliberately NOT done here because the
    geofence check in `constraints.Constraints.apply` drops waypoints outside the permitted area, and an overrun
    would put every line end outside a geofence-clipped segment. The sliver is reported instead of being
    designed away, and the coverage raster (F16) never claims it, because that raster is built from real
    footprints rather than from this pattern.
    """
    poly = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    if not lines or sweep_width <= 0.0:
        return 1.0, 0.0, float("inf")
    n0, n1 = float(poly[:, 0].min()), float(poly[:, 0].max())
    e0, e1 = float(poly[:, 1].min()), float(poly[:, 1].max())
    area = max((n1 - n0) * (e1 - e0), 1e-9)
    step = max(math.sqrt(area / max(max_samples, 16)), sweep_width / 16.0, 1e-3)
    ns = np.arange(n0 + step / 2.0, n1, step)
    es = np.arange(e0 + step / 2.0, e1, step)
    if ns.size == 0 or es.size == 0:
        return 0.0, float(step), 0.0
    pts = np.column_stack([np.repeat(ns, es.size), np.tile(es, ns.size)])
    from sightline.coverage.grid import points_in_polygon

    inside = points_in_polygon(poly, pts)
    if not inside.any():
        return 0.0, float(step), 0.0
    pts = pts[inside]
    a = np.asarray([ln[0] for ln in lines], dtype=float)
    b = np.asarray([ln[1] for ln in lines], dtype=float)
    excess = _point_segment_distance(pts, a, b).min(axis=1) - sweep_width / 2.0
    return float((excess > 0.0).mean()), float(step), float(max(excess.max(), 0.0))


def _rotate(pts: np.ndarray, theta: float, inverse: bool = False) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if inverse:
        return np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c])
    return np.column_stack([p[:, 0] * c + p[:, 1] * s, -p[:, 0] * s + p[:, 1] * c])


def _clip_line(poly_ne: np.ndarray, a_ne: tuple[float, float], b_ne: tuple[float, float]
               ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Intersect a segment with a simple polygon; returns the inside parts. Uses shapely when available."""
    try:
        from shapely.geometry import LineString, Polygon

        inter = LineString([a_ne, b_ne]).intersection(Polygon(np.asarray(poly_ne, dtype=float)))
        parts: list[tuple[tuple[float, float], tuple[float, float]]] = []
        geoms = getattr(inter, "geoms", None)
        for g in (list(geoms) if geoms is not None else [inter]):
            if g.is_empty or g.geom_type != "LineString":
                continue
            cs = list(g.coords)
            parts.append(((cs[0][0], cs[0][1]), (cs[-1][0], cs[-1][1])))
        return parts
    except ImportError:  # pragma: no cover - shapely is a pinned dependency
        return [(a_ne, b_ne)]


def boustrophedon_lines(poly_ne: np.ndarray, sweep_width: float, side_overlap: float = DEFAULT_SIDE_OVERLAP,
                        heading_deg: float = 0.0) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]],
                                                           PatternGeometry]:
    """Serpentine lines over a polygon, spaced so the swaths overlap by AT LEAST `side_overlap` and leave no gap.

    The nominal spacing is `sweep_width x (1 - side_overlap)`. An integer number of lines rarely divides the
    polygon exactly, so the lines are distributed evenly between `y_min + W/2` and `y_max - W/2`: the first and
    last swaths touch the polygon edges exactly, and the realised spacing is <= the nominal one, i.e. the realised
    overlap is >= the requested one. Under-covering is never traded for a rounder number.
    """
    poly = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    theta = math.radians(heading_deg)
    pr = _rotate(poly, theta)
    ymin, ymax = float(pr[:, 1].min()), float(pr[:, 1].max())
    xmin, xmax = float(pr[:, 0].min()), float(pr[:, 0].max())
    nominal = line_spacing_m(sweep_width, side_overlap)
    span = (ymax - ymin) - sweep_width
    if sweep_width <= 0.0:
        raise ValueError("sweep width must be positive; check the camera, altitude and presentation")
    if span <= 0.0:
        ys, actual, n = [0.5 * (ymin + ymax)], 0.0, 1
    else:
        n = int(math.ceil(span / max(nominal, 1e-6))) + 1
        actual = span / (n - 1)
        ys = [ymin + sweep_width / 2.0 + i * actual for i in range(n)]
    pad = 1.0 + 0.01 * max(1.0, xmax - xmin)
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i, y in enumerate(ys):
        a = _rotate(np.array([[xmin - pad, y]]), theta, inverse=True)[0]
        b = _rotate(np.array([[xmax + pad, y]]), theta, inverse=True)[0]
        parts = _clip_line(poly, (a[0], a[1]), (b[0], b[1]))
        parts_pr = []
        for p, q in parts:
            pp = _rotate(np.array([p, q]), theta)
            parts_pr.append((pp[0], pp[1]) if pp[0][0] <= pp[1][0] else (pp[1], pp[0]))
        parts_pr.sort(key=lambda seg: seg[0][0])
        if i % 2 == 1:
            parts_pr = [(q, p) for p, q in reversed(parts_pr)]
        for p, q in parts_pr:
            back = _rotate(np.array([p, q]), theta, inverse=True)
            lines.append(((float(back[0][0]), float(back[0][1])), (float(back[1][0]), float(back[1][1]))))
    uncovered, step, worst = uncovered_fraction(poly, lines, sweep_width)
    geom = PatternGeometry(
        sweep_width_m=float(sweep_width),
        nominal_spacing_m=float(nominal),
        actual_spacing_m=float(actual if span > 0 else 0.0),
        n_lines=int(n),
        side_overlap_nominal=float(side_overlap),
        side_overlap_actual=float(1.0 - (actual / sweep_width)) if span > 0 else 1.0,
        heading_deg=float(heading_deg % 360.0),
        speed_limit_ms=float("nan"),
        gsd_m=float("nan"),
        covers_polygon=uncovered <= 0.0,
        uncovered_fraction=uncovered,
        gap_sample_step_m=step,
        worst_gap_m=worst,
    )
    return lines, geom


def boustrophedon_route(poly_ne: np.ndarray, camera: CameraModel, agl_m: float, scene: SceneFrame,
                        presentation: str = "body", side_overlap: float = DEFAULT_SIDE_OVERLAP,
                        heading_deg: float | None = None, speed_ms: float | None = None,
                        exposure_s: float = 1.0 / 500.0, terrain: TerrainFn | None = None,
                        segment_id: str = "", pass_id: int = 0, min_px: float = MIN_PX_FOR_RECALL) -> Route:
    """A full lawnmower `Route` over `poly_ne` (scene-NE metres), with the spacing derived, never assumed.

    `heading_deg` defaults to the polygon's own long axis, which minimises the number of turns.
    """
    poly = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    heading = principal_axis_heading(poly) if heading_deg is None else heading_deg
    width = sweep_width_m(camera, agl_m, presentation, min_px)
    lines, geom = boustrophedon_lines(poly, width, side_overlap, heading)
    geom.speed_limit_ms = speed_limit_ms(camera, agl_m, exposure_s)
    geom.gsd_m = camera.gsd_m(agl_m)
    v = min(speed_ms, geom.speed_limit_ms) if speed_ms else min(8.0, geom.speed_limit_ms)
    route = Route(pattern="boustrophedon", scene=scene, params={
        "camera": camera.name, "agl_m": agl_m, "presentation": presentation, "min_px": min_px,
        "side_overlap": side_overlap, "heading_deg": geom.heading_deg, "speed_ms": v,
        "exposure_s": exposure_s, "geometry": geom.as_dict(), "segment_id": segment_id,
    })
    seq = 0
    for idx, (a, b) in enumerate(lines):
        hdg = bearing_deg(a, b)
        for point, tag in ((a, "start"), (b, "end")):
            route.waypoints.append(make_waypoint(
                seq, point[0], point[1], agl_m, scene, terrain, speed_ms=v, gimbal_pitch_deg=-90.0,
                yaw_deg=None, action="goto", segment_id=segment_id, pass_id=pass_id,
                reason=f"boustrophedon line {idx + 1}/{len(lines)} {tag}, "
                       f"sweep {width:.1f} m, spacing {geom.actual_spacing_m:.1f} m, gimbal yaw "
                       f"{gimbal_yaw_for_heading(hdg):.0f} deg"))
            seq += 1
    route.notes.append(
        f"{geom.n_lines} lines at {geom.actual_spacing_m:.1f} m ({geom.side_overlap_actual * 100:.0f} % overlap, "
        f"requested {side_overlap * 100:.0f} %), sweep width {width:.1f} m for '{presentation}' at {agl_m:.0f} m, "
        f"speed {v:.1f} m/s (blur limit {geom.speed_limit_ms:.1f} m/s)")
    if not geom.covers_polygon:
        route.notes.append(
            f"measured gap: {geom.uncovered_fraction * 100:.2f} % of this polygon lies more than half a sweep "
            f"from any flown line (worst {geom.worst_gap_m:.1f} m past the swath edge). Flying the polygon's "
            f"long axis, or a tighter overlap, closes it; the coverage map will show it either way.")
    return route


def principal_axis_heading(poly_ne: np.ndarray) -> float:
    """Heading (degrees from north) of the polygon's long axis, from the covariance of its vertices."""
    p = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    c = p - p.mean(axis=0)
    if len(c) < 3:
        return 0.0
    w, v = np.linalg.eigh(np.cov(c.T))
    axis = v[:, int(np.argmax(w))]
    return float(math.degrees(math.atan2(axis[1], axis[0])) % 180.0)


# --- 3. expanding square ----------------------------------------------------------------------------------
def expanding_square_route(centre_ne: Sequence[float], camera: CameraModel, agl_m: float, scene: SceneFrame,
                           n_legs: int = 12, presentation: str = "body",
                           side_overlap: float = DEFAULT_SIDE_OVERLAP, start_heading_deg: float = 0.0,
                           speed_ms: float | None = None, exposure_s: float = 1.0 / 500.0,
                           terrain: TerrainFn | None = None, segment_id: str = "", pass_id: int = 0,
                           max_radius_m: float | None = None) -> Route:
    """§5.2: "Expanding-square and sector patterns for a last-known-position start."

    Leg lengths run s, s, 2s, 2s, 3s, 3s, ... with 90 degree turns, where s is the same derived line spacing the
    lawnmower uses, so the two patterns search at the same density.
    """
    width = sweep_width_m(camera, agl_m, presentation)
    s = line_spacing_m(width, side_overlap)
    v_limit = speed_limit_ms(camera, agl_m, exposure_s)
    v = min(speed_ms, v_limit) if speed_ms else min(8.0, v_limit)
    route = Route(pattern="expanding_square", scene=scene, params={
        "camera": camera.name, "agl_m": agl_m, "presentation": presentation, "spacing_m": s,
        "sweep_width_m": width, "n_legs": n_legs, "start_heading_deg": start_heading_deg, "speed_ms": v,
        "segment_id": segment_id})
    n, e = float(centre_ne[0]), float(centre_ne[1])
    route.waypoints.append(make_waypoint(0, n, e, agl_m, scene, terrain, speed_ms=v, segment_id=segment_id,
                                         pass_id=pass_id, reason="expanding square datum"))
    heading = start_heading_deg
    for leg in range(1, n_legs + 1):
        length = s * ((leg + 1) // 2)
        n += length * math.cos(math.radians(heading))
        e += length * math.sin(math.radians(heading))
        if max_radius_m is not None and math.dist((n, e), centre_ne) > max_radius_m:
            route.notes.append(f"stopped at leg {leg}: max radius {max_radius_m:.0f} m reached")
            break
        route.waypoints.append(make_waypoint(
            len(route.waypoints), n, e, agl_m, scene, terrain, speed_ms=v, segment_id=segment_id, pass_id=pass_id,
            reason=f"expanding square leg {leg}, length {length:.0f} m, spacing {s:.1f} m"))
        heading = (heading + 90.0) % 360.0
    route.notes.append(f"spacing {s:.1f} m from a {width:.1f} m sweep width at {agl_m:.0f} m")
    return route


# --- 4. orbit on detection --------------------------------------------------------------------------------
def orbit_route(centre_ne: Sequence[float], radius_m: float, agl_m: float, scene: SceneFrame,
                n_points: int = 8, dwell_s: float = 2.0, speed_ms: float = 3.0,
                terrain: TerrainFn | None = None, record_id: str = "", pass_id: int = 0,
                start_heading_deg: float = 0.0) -> Route:
    """§5.2 behaviour 2: orbit and dwell on a candidate, camera locked on the target.

    Dwelling is what "shrinks the random part of the geolocation error and produces the best evidence thumbnail".
    The gimbal pitch is the depression angle onto the centre; the vehicle yaw faces the centre.
    """
    pitch = -math.degrees(math.atan2(agl_m, max(radius_m, 1e-6)))
    route = Route(pattern="orbit", scene=scene, params={
        "centre_ne": [float(centre_ne[0]), float(centre_ne[1])], "radius_m": radius_m, "agl_m": agl_m,
        "n_points": n_points, "dwell_s": dwell_s, "gimbal_pitch_deg": pitch, "record_id": record_id})
    for i in range(n_points):
        a = math.radians(start_heading_deg + 360.0 * i / n_points)
        n = centre_ne[0] + radius_m * math.cos(a)
        e = centre_ne[1] + radius_m * math.sin(a)
        route.waypoints.append(make_waypoint(
            i, n, e, agl_m, scene, terrain, speed_ms=speed_ms, gimbal_pitch_deg=pitch,
            yaw_deg=bearing_deg((n, e), centre_ne), action="orbit", dwell_s=dwell_s,
            orbit_radius_m=radius_m, pass_id=pass_id,
            reason=f"orbit {record_id or 'candidate'} at {radius_m:.0f} m, pitch {pitch:.0f} deg"))
    return route


# --- 5. polygon helpers -----------------------------------------------------------------------------------
def rect_polygon(centre_ne: Sequence[float], north_m: float, east_m: float) -> np.ndarray:
    n, e = float(centre_ne[0]), float(centre_ne[1])
    hn, he = north_m / 2.0, east_m / 2.0
    return np.array([[n - hn, e - he], [n + hn, e - he], [n + hn, e + he], [n - hn, e + he]], dtype=float)


def polygon_area_m2(poly_ne: np.ndarray) -> float:
    p = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid_ne(poly_ne: np.ndarray) -> tuple[float, float]:
    p = np.asarray(poly_ne, dtype=float).reshape(-1, 2)
    return (float(p[:, 0].mean()), float(p[:, 1].mean()))


@dataclass(slots=True)
class PatternSpec:
    """A reusable description of "how we survey", so the planner can vary altitude and band without duplication."""

    camera: CameraModel
    agl_m: float = 55.0
    presentation: str = "body"
    side_overlap: float = DEFAULT_SIDE_OVERLAP
    exposure_s: float = 1.0 / 500.0
    speed_ms: float | None = None
    band: str = "rgb"
    extra: dict[str, Any] = field(default_factory=dict)

    def sweep_width(self) -> float:
        return sweep_width_m(self.camera, self.agl_m, self.presentation)

    def spacing(self) -> float:
        return line_spacing_m(self.sweep_width(), self.side_overlap)

    def speed(self) -> float:
        limit = speed_limit_ms(self.camera, self.agl_m, self.exposure_s)
        return min(self.speed_ms, limit) if self.speed_ms else min(8.0, limit)

    def area_rate_m2_s(self) -> float:
        """Ground newly swept per second: line spacing x ground speed. Drives every `t_execute` in the planner."""
        return self.spacing() * self.speed()
