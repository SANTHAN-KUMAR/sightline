"""F2 / F2b — patterns, constraints and the decision planner, against SOLUTION_DOC §5.2, §5.3a, §5.3b, App. B.

The property that matters most here is the one the doc calls the safety property: **the planner has no fallback
branch — the boustrophedon is what it outputs when the prior is flat.** That is tested by building the plain
pattern independently and comparing waypoint for waypoint.

    D:\\Tools\\uv\\uv.exe run pytest tests/test_plan.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sightline.coverage.accumulate import ZONE_CODE, CoverageMap
from sightline.coverage.grid import SceneFrame
from sightline.coverage.presentation import LEGAL_CEILING_M, M3T_WIDE_4K, SIM_RGB_4K, CameraModel
from sightline.plan import patterns as pat
from sightline.plan.constraints import DEFAULT_MIN_AGL_M, Battery, Constraints
from sightline.plan.planner import (
    CONFIRM_AGL_M,
    Candidate,
    DecisionPlanner,
    PendingRecord,
    PlannerState,
    allocate_segment_minutes,
    koopman_allocation,
)
from sightline.plan.revisit import RevisitQueue, from_coverage, from_records
from sightline.plan.segments import Segment, assignment_record, auto_segments, recommend, segments_geojson
from sightline.plan.waypoints import Route, TerrainSampler, make_waypoint
from sightline.schemas import SCHEMA_VERSION, Record

SCENE = SceneFrame(11.4870, 76.1450, 1046.007, 800.0)
CAM = SIM_RGB_4K


def _cmap(cell_m: float = 20.0, presentations=("body", "limb_only")) -> CoverageMap:
    return CoverageMap.for_scene(SCENE, cell_m=cell_m, presentations=presentations)


def _flat_poa(cmap: CoverageMap) -> np.ndarray:
    n = cmap.shape[0] * cmap.shape[1]
    return np.full(cmap.shape, 1.0 / n, dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------
# 1. the geometry chain: nothing is hard-coded (Appendix B)
# ---------------------------------------------------------------------------------------------------------
def test_line_spacing_is_derived_from_sweep_width_altitude_and_gsd():
    # Appendix B: swath = 2 h tan(HFOV/2); GSD = swath / W_px; spacing = sweep x (1 - side overlap)
    for agl in (30.0, 45.0, 60.0, 90.0, 120.0):
        swath = 2.0 * agl * math.tan(math.radians(CAM.hfov_deg) / 2.0)
        assert pat.swath_m(CAM, agl) == pytest.approx(swath)
        assert CAM.gsd_m(agl) == pytest.approx(swath / CAM.width_px)
        w = pat.sweep_width_m(CAM, agl, "body")
        assert pat.line_spacing_m(w, 0.25) == pytest.approx(w * 0.75)
        assert pat.line_spacing_m(w, 0.30) == pytest.approx(w * 0.70)
        # v <= max_blur_px * GSD / t_exp
        assert pat.speed_limit_ms(CAM, agl, 1.0 / 500.0) == pytest.approx(CAM.gsd_m(agl) * 500.0)

    # doubling the altitude doubles the spacing; a different camera gives a different spacing at the same height
    s60, s120 = (pat.line_spacing_m(pat.sweep_width_m(CAM, h, "body")) for h in (60.0, 120.0))
    assert s120 == pytest.approx(2.0 * s60)
    assert pat.line_spacing_m(pat.sweep_width_m(M3T_WIDE_4K, 60.0, "body")) != pytest.approx(s60)
    # the spacing is never a round constant anyone typed
    assert s60 not in (50.0, 60.0, 70.0, 100.0)


def test_sweep_width_narrows_for_marginal_presentations_and_vanishes_above_the_ceiling():
    """§5.3b: a 60 m pass is complete for a body and nearly blind to a limb, and the pattern must know it."""
    at60 = {p: pat.sweep_width_m(CAM, 60.0, p) for p in ("body", "upright", "head_only", "limb_only")}
    assert at60["body"] == pytest.approx(pat.swath_m(CAM, 60.0)), "a body resolves across the whole swath"
    assert at60["limb_only"] == 0.0, "a hand does not resolve at all at 60 m: no usable sweep"
    assert at60["head_only"] == 0.0
    # descend and the limb layer becomes searchable, but over a narrower strip than the raw swath
    low = pat.sweep_width_m(CAM, 18.0, "limb_only")
    assert 0.0 < low < pat.swath_m(CAM, 18.0)
    assert pat.altitude_for_min_px(M3T_WIDE_4K, "limb_only") == pytest.approx(24.0, abs=0.5)
    assert pat.altitude_for_min_px(M3T_WIDE_4K, "body") == pytest.approx(LEGAL_CEILING_M)
    with pytest.raises(ValueError):
        pat.boustrophedon_lines(pat.rect_polygon((0, 0), 100.0, 100.0), 0.0)


def test_boustrophedon_covers_the_polygon_with_at_least_the_requested_overlap():
    poly = pat.rect_polygon((0.0, 0.0), 400.0, 300.0)
    for heading in (0.0, 30.0, 90.0, 143.0):
        for agl, overlap in ((45.0, 0.20), (60.0, 0.25), (90.0, 0.30)):
            w = pat.sweep_width_m(CAM, agl, "body")
            lines, geom = pat.boustrophedon_lines(poly, w, overlap, heading_deg=heading)
            assert geom.sweep_width_m == pytest.approx(w)
            assert geom.nominal_spacing_m == pytest.approx(w * (1.0 - overlap))
            assert geom.actual_spacing_m <= geom.nominal_spacing_m + 1e-9, "never widen to a rounder number"
            assert geom.side_overlap_actual >= overlap - 1e-9, "realised overlap is at least what was asked"
            assert geom.gap_sample_step_m <= w / 8.0
            # Measured, not assumed. Lines are clipped to the polygon, so flying across a corner can leave a
            # sliver beyond the last line's endpoint (documented in `uncovered_fraction`). It must stay a
            # sliver: under a tenth of a percent of the area and under a metre past the swath edge — never a
            # missed strip, which would be of order the spacing.
            assert geom.uncovered_fraction < 0.001, (heading, agl, geom)
            assert geom.worst_gap_m < 1.0, (heading, agl, geom)

    # flown along the polygon's own long axis — the default the router picks — there is no gap at all
    for agl, overlap in ((45.0, 0.20), (60.0, 0.25), (90.0, 0.30)):
        w = pat.sweep_width_m(CAM, agl, "body")
        _, geom = pat.boustrophedon_lines(poly, w, overlap, heading_deg=pat.principal_axis_heading(poly))
        assert geom.covers_polygon and geom.uncovered_fraction == 0.0 and geom.worst_gap_m == 0.0

    # the strips also tile the polygon's across-track extent with no gap, checked in the rotated frame
    w = pat.sweep_width_m(CAM, 60.0, "body")
    lines, geom = pat.boustrophedon_lines(poly, w, 0.25, heading_deg=0.0)
    ys = sorted({round(a[1], 6) for a, _ in lines})
    assert len(ys) == geom.n_lines == 4
    assert min(ys) == pytest.approx(-150.0 + w / 2.0)
    assert max(ys) == pytest.approx(150.0 - w / 2.0)
    assert max(np.diff(ys)) <= w, "consecutive swaths must touch or overlap"
    assert np.allclose(np.diff(ys), geom.actual_spacing_m)


def test_the_gap_check_is_falsifiable():
    """A concave segment CAN leave a gap at a legal overlap; `covers_polygon` has to be able to say so."""
    plus = np.array([[-30, -150], [30, -150], [30, -30], [150, -30], [150, 30], [30, 30],
                     [30, 150], [-30, 150], [-30, 30], [-150, 30], [-150, -30], [-30, -30]], dtype=float)
    _lines, wide = pat.boustrophedon_lines(plus, 200.0, 0.25, heading_deg=90.0)
    assert wide.n_lines == 2 and not wide.covers_polygon
    assert wide.uncovered_fraction > 0.05 and wide.worst_gap_m > 10.0, "a missed STRIP, not a sliver"
    assert any("measured gap" in n for n in
               pat.boustrophedon_route(plus, CAM, 120.0, SCENE, heading_deg=90.0).notes)
    _, tight = pat.boustrophedon_lines(plus, 120.0, 0.25, heading_deg=90.0)
    assert tight.covers_polygon and tight.uncovered_fraction == 0.0
    frac, _, worst = pat.uncovered_fraction(plus, [], 100.0)
    assert frac == 1.0 and math.isinf(worst), "no lines means nothing is covered"


def test_boustrophedon_route_is_a_serpentine_over_the_polygon_and_states_its_own_geometry():
    poly = pat.rect_polygon((0.0, 0.0), 400.0, 300.0)
    rt = pat.boustrophedon_route(poly, CAM, 60.0, SCENE, presentation="body")
    assert rt.pattern == "boustrophedon"
    assert len(rt) == 8 == 2 * rt.params["geometry"]["n_lines"]
    assert rt.params["heading_deg"] == pytest.approx(pat.principal_axis_heading(poly)) == 0.0

    # serpentine: consecutive lines run in opposite directions
    legs = [(rt.waypoints[i], rt.waypoints[i + 1]) for i in range(0, len(rt), 2)]
    dirs = [math.copysign(1.0, b.north_m - a.north_m) for a, b in legs]
    assert dirs == [1.0, -1.0, 1.0, -1.0]
    # every waypoint explains itself and is flown nadir at the blur-limited speed
    assert all(w.gimbal_pitch_deg == -90.0 for w in rt.waypoints)
    assert all("boustrophedon line" in w.reason and "sweep" in w.reason for w in rt.waypoints)
    assert all(w.speed_ms <= pat.speed_limit_ms(CAM, 60.0) for w in rt.waypoints)
    assert "sweep width" in rt.notes[0] and "overlap" in rt.notes[0]
    # Gimbal yaw puts the WIDE axis across track, and at nadir the image +u axis is PERPENDICULAR to the
    # gimbal yaw (at yaw 0 the frame is north-up and +u is due EAST), so the yaw equals the heading. This
    # asserted `heading + 90` until 2026-09-11, which was the compensator for a quarter-turn in
    # coverage/footprint.py rather than a property of the camera. Measured across 110 boxes with known
    # survivor world positions: image-right is due east (residual 1.37 m vs 15.8 m for the next hypothesis).
    assert pat.gimbal_yaw_for_heading(0.0) == 0.0 and pat.gimbal_yaw_for_heading(300.0) == 300.0
    # the long axis is chosen by default, so the pattern turns as little as possible
    wide_poly = pat.rect_polygon((0.0, 0.0), 100.0, 600.0)
    assert pat.principal_axis_heading(wide_poly) == pytest.approx(90.0)
    assert len(pat.boustrophedon_route(wide_poly, CAM, 60.0, SCENE)) < \
        len(pat.boustrophedon_route(wide_poly, CAM, 60.0, SCENE, heading_deg=0.0))


def test_expanding_square_legs_and_turns_are_correct():
    """§5.2: expanding square from a last-known position, at the SAME density as the lawnmower."""
    rt = pat.expanding_square_route((0.0, 0.0), CAM, 45.0, SCENE, n_legs=8, start_heading_deg=0.0)
    s = pat.line_spacing_m(pat.sweep_width_m(CAM, 45.0, "body"))
    assert rt.params["spacing_m"] == pytest.approx(s)
    assert len(rt) == 9  # datum + 8 legs

    pts = [w.ne() for w in rt.waypoints]
    legs = [(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    lengths = [math.hypot(*d) for d in legs]
    assert lengths == pytest.approx([s, s, 2 * s, 2 * s, 3 * s, 3 * s, 4 * s, 4 * s])
    headings = [math.degrees(math.atan2(d[1], d[0])) % 360.0 for d in legs]
    assert headings == pytest.approx([0.0, 90.0, 180.0, 270.0, 0.0, 90.0, 180.0, 270.0])
    assert all(w.reason for w in rt.waypoints)

    capped = pat.expanding_square_route((0.0, 0.0), CAM, 45.0, SCENE, n_legs=20, max_radius_m=120.0)
    assert len(capped) < 21 and any("max radius" in n for n in capped.notes)
    assert max(math.dist(w.ne(), (0.0, 0.0)) for w in capped.waypoints) <= 120.0


def test_orbit_route_looks_down_at_the_candidate():
    rt = pat.orbit_route((10.0, 20.0), radius_m=30.0, agl_m=40.0, scene=SCENE, n_points=8, dwell_s=2.0,
                         record_id="rec-1")
    assert rt.pattern == "orbit" and len(rt) == 8
    pitch = -math.degrees(math.atan2(40.0, 30.0))
    assert all(w.gimbal_pitch_deg == pytest.approx(pitch) for w in rt.waypoints)
    assert all(math.dist(w.ne(), (10.0, 20.0)) == pytest.approx(30.0) for w in rt.waypoints)
    assert all(w.dwell_s == 2.0 and w.action == "orbit" for w in rt.waypoints)
    for w in rt.waypoints:  # the vehicle faces the centre
        assert w.yaw_deg == pytest.approx(pat.bearing_deg(w.ne(), (10.0, 20.0)))
    assert rt.duration_s() >= 8 * 2.0


# ---------------------------------------------------------------------------------------------------------
# 2. constraints really prune, shorten and abort (§5.2 behaviour 4)
# ---------------------------------------------------------------------------------------------------------
def test_battery_reserve_truncates_the_pattern_and_appends_the_return_leg():
    rt = pat.boustrophedon_route(pat.rect_polygon((0.0, 0.0), 400.0, 300.0), CAM, 60.0, SCENE)
    generous = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=100000.0))
    full = generous.apply(rt, start_ne=(0.0, 0.0))
    assert not full.aborted and len(full) == len(rt) + 1 and full.waypoints[-1].action == "rtl"

    tight = Constraints(home_ne=(0.0, 0.0), scene=SCENE,
                        battery=Battery(endurance_s=300.0, reserve_frac=0.2, rtl_speed_ms=12.0))
    short = tight.apply(rt, start_ne=(0.0, 0.0))
    assert short.aborted and "battery reserve" in short.abort_reason
    assert 0 < len(short) < len(full), "the pattern is shortened, not silently flown"
    assert short.waypoints[-1].action == "rtl" and short.abort_reason in short.waypoints[-1].reason
    # the kept prefix really is affordable: elapsed + time home <= the usable budget
    last = short.waypoints[-2]
    t = short.params["elapsed_s_at_end"]
    assert t + tight.battery.time_home_s(last.ne(), (0.0, 0.0)) <= tight.battery.usable_s + 1e-6
    assert tight.battery.usable_s == pytest.approx(300.0 * 0.8)
    assert rt.aborted is False and len(rt) == 8, "apply() must not mutate its input"


def test_geofence_and_no_go_areas_prune_and_then_abort():
    rt = pat.boustrophedon_route(pat.rect_polygon((0.0, 0.0), 400.0, 300.0), CAM, 60.0, SCENE)
    # a fence that keeps only part of the pattern -> shortened, reason in the notes
    partial = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=100000.0),
                          geofence_ne=pat.rect_polygon((0.0, -80.0), 500.0, 120.0))
    out = partial.apply(rt, start_ne=(0.0, 0.0))
    assert 0 < len(out) < len(rt) + 1
    assert any("outside the geofence" in n for n in out.notes)
    assert all(partial.inside_geofence(w.ne()) for w in out.waypoints if w.action != "rtl")

    # a fence that keeps nothing -> the caller must be told the pattern could not be flown at all
    elsewhere = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=100000.0),
                            geofence_ne=pat.rect_polygon((900.0, 900.0), 50.0, 50.0))
    dead = elsewhere.apply(rt, start_ne=(0.0, 0.0))
    assert dead.aborted and "nothing left to fly" in dead.abort_reason

    # an operator no-go area (never automatic: §5.2) drops what falls inside it
    nogo = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=100000.0),
                       no_go_ne=[pat.rect_polygon((0.0, 0.0), 1000.0, 120.0)])
    pruned = nogo.apply(rt, start_ne=(0.0, 0.0))
    assert any("no-go area" in n for n in pruned.notes)
    assert not any(nogo.inside_no_go(w.ne()) for w in pruned.waypoints if w.action != "rtl")
    assert Constraints().no_go_ne == [], "no-go buffers are never created automatically"


def test_clipping_the_survey_polygon_shortens_the_pattern_instead_of_shredding_it():
    poly = pat.rect_polygon((0.0, 0.0), 400.0, 300.0)
    con = Constraints(home_ne=(0.0, 0.0), scene=SCENE,
                      geofence_ne=pat.rect_polygon((0.0, 0.0), 200.0, 200.0),
                      no_go_ne=[pat.rect_polygon((80.0, 80.0), 40.0, 40.0)])
    clipped = con.clip_polygon(poly)
    assert clipped is not None
    assert pat.polygon_area_m2(clipped) < pat.polygon_area_m2(poly)
    inside = pat.boustrophedon_route(clipped, CAM, 60.0, SCENE)
    assert len(inside) <= len(pat.boustrophedon_route(poly, CAM, 60.0, SCENE))
    assert con.clip_polygon(pat.rect_polygon((5000.0, 5000.0), 10.0, 10.0)) is None


def test_altitude_is_clamped_into_the_legal_band():
    con = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=100000.0))
    assert con.ceiling_m == LEGAL_CEILING_M == 120.0
    high = pat.boustrophedon_route(pat.rect_polygon((0.0, 0.0), 200.0, 200.0), CAM, 150.0, SCENE)
    out = con.apply(high, start_ne=(0.0, 0.0))
    assert all(w.agl_m <= 120.0 for w in out.waypoints)
    assert any("clamped" in n for n in out.notes)
    assert any("altitude clamped" in w.reason for w in out.waypoints if w.action != "rtl")
    assert con.clamp_altitude(5.0) == DEFAULT_MIN_AGL_M and con.clamp_altitude(999.0) == 120.0
    # the ASL height moves with the clamp, so AGL and ASL stay consistent
    w = out.waypoints[0]
    assert w.alt_asl_m == pytest.approx(SCENE.base_z_m + w.agl_m)


# ---------------------------------------------------------------------------------------------------------
# 3. Layer 1: Koopman / Stone optimal allocation (§5.3a, Appendix B)
# ---------------------------------------------------------------------------------------------------------
def test_koopman_allocation_is_the_water_filling_solution():
    rng = np.random.default_rng(3)
    p = rng.random((8, 8)) ** 3 + 1e-6
    p /= p.sum()
    budget = 12.0
    c, lam = koopman_allocation(p, budget)
    assert c.sum() == pytest.approx(budget, rel=1e-5)
    want = np.clip(np.log(p) - lam, 0.0, None)
    scale = budget / want.sum()
    np.testing.assert_allclose(c, want * scale, rtol=1e-4, atol=1e-6)
    # cells whose prior falls below e^lambda receive exactly zero
    below = p < math.exp(lam)
    assert below.any() and c[below].max() == 0.0
    assert (c[~below] > 0.0).all()
    # more prior never means less effort
    order = np.argsort(p.ravel())
    assert (np.diff(c.ravel()[order]) >= -1e-6).all()


def test_a_flat_prior_gives_a_flat_allocation():
    p = np.full((10, 10), 0.01, dtype=np.float32)
    c, _ = koopman_allocation(p, 50.0)
    assert np.unique(np.round(c, 6)).size == 1
    assert c.sum() == pytest.approx(50.0, rel=1e-5)
    # a zero budget, or nothing to search, allocates nothing rather than dividing by zero
    assert koopman_allocation(p, 0.0)[0].max() == 0.0
    assert koopman_allocation(np.zeros((4, 4)), 10.0)[0].max() == 0.0


def test_allocation_skips_burial_polygons_and_reports_minutes_per_segment():
    cmap = _cmap(cell_m=25.0)
    cmap.add_burial_polygons([np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])])
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    minutes = allocate_segment_minutes(poa, segs, cmap, SCENE, total_minutes=40.0)
    assert sum(minutes.values()) == pytest.approx(40.0, rel=1e-3)
    alloc, _ = koopman_allocation(poa, 1.0, mask=~cmap.cannot_clear)
    assert alloc[cmap.cannot_clear].max() == 0.0, "no minutes are budgeted for ground the air cannot clear"


# ---------------------------------------------------------------------------------------------------------
# 4. Layer 2: the decision planner (§5.3a) — including the degradation property
# ---------------------------------------------------------------------------------------------------------
def _state(cmap, poa, segments, position_ne=(0.0, 0.0), **kw) -> PlannerState:
    kw.setdefault("constraints", Constraints(home_ne=(0.0, 0.0), scene=SCENE,
                                             battery=Battery(endurance_s=6000.0)))
    kw.setdefault("local_hour", 12.0)
    return PlannerState(cmap=cmap, scene=SCENE, poa=poa, segments=segments, position_ne=position_ne,
                        camera=CAM, **kw)


def test_planner_degrades_to_the_plain_pattern_when_the_prior_is_flat():
    """§5.3a: "the pattern is what the planner outputs when it has no information to be clever with"."""
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    assert len({round(s.poa_mass, 9) for s in segs}) == 1, "a flat prior gives every segment the same mass"

    # start next to segment C so the tie is broken by transit time: "cover the nearest unsearched area"
    nearest = min(segs, key=lambda s: math.dist((190.0, -190.0), s.centroid_ne()))
    st = _state(cmap, poa, segs, position_ne=(190.0, -190.0))
    dec = DecisionPlanner(altitudes_m=(60.0,), bands=("rgb",)).decide(st)

    assert dec.chosen.kind == "survey"
    assert dec.chosen.params["segment"].seg_id == nearest.seg_id
    plain = pat.boustrophedon_route(nearest.poly_ne, CAM, 60.0, SCENE, presentation="body",
                                    side_overlap=st.side_overlap, segment_id=nearest.seg_id, min_px=20.0)
    flown = [w for w in dec.route.waypoints if w.action != "rtl"]
    assert len(flown) == len(plain)
    for a, b in zip(flown, plain.waypoints):
        assert (a.north_m, a.east_m, a.agl_m) == pytest.approx((b.north_m, b.east_m, b.agl_m))
        assert a.reason == b.reason, "the decision route IS the pattern, not a re-derivation of it"
    assert dec.route.params["geometry"] == plain.params["geometry"]


def test_planner_follows_the_prior_as_soon_as_it_stops_being_flat():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    far = max(segs, key=lambda s: math.dist((190.0, -190.0), s.centroid_ne()))

    hot = np.full(cmap.shape, 1e-6, dtype=np.float32)
    hot[far.mask(cmap, SCENE)] = 1.0
    hot /= hot.sum()
    for s in segs:
        s.poa_mass = float(hot[s.mask(cmap, SCENE)].sum())
    st = _state(cmap, hot, segs, position_ne=(190.0, -190.0))
    dec = DecisionPlanner(altitudes_m=(60.0,), bands=("rgb",)).decide(st)
    assert dec.chosen.params["segment"].seg_id == far.seg_id, "it abandons the nearest for the informative"
    assert dec.chosen.value > 0.0


def test_every_candidate_carries_a_number_and_a_reason():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    st = _state(cmap, poa, segs, pending_records=[PendingRecord("rec-7", (100.0, 100.0), score=0.8)])
    dec = DecisionPlanner(altitudes_m=(45.0, 60.0, 90.0), bands=("rgb",)).decide(st, total_minutes=30.0)

    assert len(dec.ranked) >= 4
    kinds = {c.kind for c in dec.ranked}
    assert {"survey", "orbit", "rtl"} <= kinds
    for c in dec.ranked:
        assert c.reason, f"{c.label} has no explanation"
        assert math.isfinite(c.value) and math.isfinite(c.gain)
        assert c.feasible or c.infeasible_reason
    # ranked descending among the feasible ones, infeasible last
    feasible = [c for c in dec.ranked if c.feasible]
    assert [c.value for c in feasible] == sorted((c.value for c in feasible), reverse=True)
    assert dec.ranked.index(dec.chosen) == 0
    assert "chose " in dec.explanation and "expected finds per minute" in dec.explanation
    assert "(simulation)" in dec.explanation
    d = dec.as_dict()
    assert d["domain"] == "sim" and len(d["ranked"]) <= 3 and d["chosen"]["reason"]
    assert json.dumps(d)  # the timeline has to be able to serialise it
    # the chosen action's reason travels with the waypoints the vehicle will fly
    assert dec.route.params["decision_reason"] == dec.chosen.reason
    assert dec.route.params["chosen_by"] == dec.chosen.label


def test_planner_declines_to_refly_a_saturated_area_and_says_so_numerically():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    fresh = DecisionPlanner(altitudes_m=(60.0,), bands=("rgb",)).decide(_state(cmap, poa, segs))
    for g in cmap.grids.values():
        g.coverage[:] = 25.0
    cmap._recompute()
    tired = DecisionPlanner(altitudes_m=(60.0,), bands=("rgb",)).decide(_state(cmap, poa, segs))

    assert fresh.chosen.value > 1e-4
    assert tired.chosen.value < 1e-6, "a repeat pass in identical conditions scores about zero (§5.3a)"
    assert "adds nothing measurable" in tired.explanation
    assert tired.chosen is not None, "it still proposes something; it never closes the segment"


def test_planner_is_never_paid_for_a_burial_polygon():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    victim = segs[0]
    grid_poly = np.array([[0.0, 0.0], [cmap.shape[0] * 25.0, 0.0],
                          [cmap.shape[0] * 25.0, cmap.shape[1] * 25.0], [0.0, cmap.shape[1] * 25.0]])
    cmap.set_cannot_clear(victim.mask(cmap, SCENE))
    planner = DecisionPlanner(altitudes_m=(60.0,), bands=("rgb",))
    cands = planner._survey_candidates(_state(cmap, poa, segs), 12.0)
    buried = [c for c in cands if c.params["segment"].seg_id == victim.seg_id]
    others = [c for c in cands if c.params["segment"].seg_id != victim.seg_id]
    assert buried and all(c.gain == 0.0 for c in buried)
    assert others and max(c.gain for c in others) > 0.0
    assert grid_poly.shape == (4, 2)


def test_infeasible_actions_are_pruned_before_scoring_and_rtl_is_always_left():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    broke = Constraints(home_ne=(0.0, 0.0), scene=SCENE, battery=Battery(endurance_s=60.0))
    st = _state(cmap, poa, segs, constraints=broke)
    dec = DecisionPlanner(altitudes_m=(60.0, 150.0), bands=("rgb",)).decide(st)
    assert dec.chosen.kind == "rtl", "with no affordable action the only thing left is to come home"
    ceiling = [c for c in dec.ranked if c.params.get("agl_m") == 150.0]
    assert ceiling and all(not c.feasible and "ceiling" in c.infeasible_reason for c in ceiling)
    assert any("battery" in c.infeasible_reason for c in dec.ranked if not c.feasible)


def test_loiter_until_the_predawn_thermal_window_is_a_scorable_action():
    """§5.3a behaviour 3: it schedules the thermal pass for pre-dawn on its own, from q_pass alone."""
    thermal = CameraModel("thermal 640", 640, 512, 45.0, "thermal", "tests")
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    st = _state(cmap, poa, segs, local_hour=2.0, thermal_camera=thermal)
    planner = DecisionPlanner(altitudes_m=(45.0,), bands=("rgb", "thermal"), dawn_hour=5.0)
    cand = planner._loiter_candidate(st)
    assert cand is not None and cand.kind == "loiter"
    assert cand.t_transit_s == pytest.approx(3.0 * 3600.0)
    assert "pre-dawn" in cand.reason
    assert planner._loiter_candidate(_state(cmap, poa, segs, local_hour=None)) is None


def test_descend_to_confirm_falls_out_of_the_coverage_model():
    """§5.3b: an orbit at 25 m buys limb-only quality a 90 m survey cannot; the reason must state both."""
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    st = _state(cmap, poa, segs, pending_records=[PendingRecord("rec-1", (50.0, 50.0), score=1.0, radius_m=60.0)])
    orbits = DecisionPlanner(altitudes_m=(45.0, 90.0), bands=("rgb",))._orbit_candidates(st, 12.0)
    assert orbits and orbits[0].params["agl_m"] == CONFIRM_AGL_M == 25.0
    assert "descend-to-confirm" in orbits[0].reason
    assert orbits[0].params["q"]["limb_only"] > 0.0
    high = DecisionPlanner(altitudes_m=(90.0,))._q_for(
        st, 90.0, "rgb", 12.0, DecisionPlanner()._spec(st, 90.0, "rgb"))
    # at 90 m a hand is ~5 px: a cue at best. At 25 m it is ~18 px, which is why the descent scores.
    assert orbits[0].params["q"]["limb_only"] > 5.0 * high["limb_only"]
    assert pat.sweep_width_m(CAM, 90.0, "limb_only") == 0.0, "and no survey line would even be planned"


# ---------------------------------------------------------------------------------------------------------
# 5. the waypoint output format (the contract with sightline/mission/)
# ---------------------------------------------------------------------------------------------------------
def test_waypoint_output_format():
    terrain = TerrainSampler.flat(1100.0)
    rt = pat.boustrophedon_route(pat.rect_polygon((0.0, 0.0), 200.0, 200.0), CAM, 55.0, SCENE,
                                 terrain=terrain, segment_id="S03", pass_id=2)
    d = rt.to_dict()
    assert set(d) == {"schema_version", "product", "domain", "pattern", "params", "frame", "aborted",
                      "abort_reason", "notes", "totals", "waypoints"}
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["product"] == "sightline.plan.route" and d["domain"] == "sim"
    assert d["frame"]["centre_lat"] == SCENE.centre_lat and d["frame"]["base_z_m"] == SCENE.base_z_m
    assert d["frame"]["axes"] == "north = UE +X, east = UE +Y"
    assert d["totals"]["n_waypoints"] == len(rt) == len(d["waypoints"])
    assert d["totals"]["length_m"] == pytest.approx(rt.length_m(), abs=0.01)

    w = d["waypoints"][0]
    assert set(w) >= {"seq", "action", "north_m", "east_m", "alt_asl_m", "agl_m", "lat", "lon", "ue_cm",
                      "speed_ms", "gimbal_pitch_deg", "yaw_deg", "dwell_s", "orbit_radius_m", "segment_id",
                      "pass_id", "reason"}
    assert w["segment_id"] == "S03" and w["pass_id"] == 2 and w["action"] == "goto"
    assert w["yaw_deg"] is None, "null means: face the direction of travel"
    # every waypoint carries BOTH the local frame and lat/lon, and they agree
    for wd in d["waypoints"]:
        assert wd["alt_asl_m"] == pytest.approx(1100.0 + wd["agl_m"])
        lat, lon = SCENE.to_latlon(wd["north_m"], wd["east_m"])
        assert (wd["lat"], wd["lon"]) == pytest.approx((lat, lon))
        assert wd["ue_cm"] == pytest.approx([wd["north_m"] * 100.0, wd["east_m"] * 100.0,
                                             (wd["alt_asl_m"] - SCENE.base_z_m) * 100.0])
    assert [wd["seq"] for wd in d["waypoints"]] == list(range(len(rt)))
    assert json.dumps(d)

    gj = rt.to_geojson()
    assert gj["features"][0]["geometry"]["type"] == "LineString"
    assert gj["features"][0]["properties"]["domain"] == "sim"
    lon, lat, alt = gj["features"][0]["geometry"]["coordinates"][0]
    assert 76.0 < lon < 76.3 and 11.0 < lat < 11.9, "RFC 7946 is [lon, lat, alt]"
    assert alt == pytest.approx(rt.waypoints[0].alt_asl_m, abs=0.01)
    assert len(gj["features"]) == 1 + len(rt)


def test_terrain_sampler_puts_agl_above_the_ground_not_above_the_datum():
    h = np.zeros((9, 9), dtype=np.float32)
    h[4:, :] = 50.0
    sampler = TerrainSampler(h, size_m=800.0, default_asl_m=0.0)
    low = make_waypoint(0, -300.0, 0.0, 45.0, SCENE, sampler)
    high = make_waypoint(1, 300.0, 0.0, 45.0, SCENE, sampler)
    assert low.alt_asl_m == pytest.approx(45.0)
    assert high.alt_asl_m == pytest.approx(95.0)
    assert low.agl_m == high.agl_m == 45.0
    flat = make_waypoint(2, 0.0, 0.0, 45.0, SCENE, None)
    assert flat.alt_asl_m == pytest.approx(SCENE.base_z_m + 45.0)


# ---------------------------------------------------------------------------------------------------------
# 6. segments, assignment records and the revisit queue
# ---------------------------------------------------------------------------------------------------------
def test_auto_segments_rank_by_prior_mass_and_export_as_geojson():
    cmap = _cmap(cell_m=25.0)
    poa = np.zeros(cmap.shape, dtype=np.float32)
    poa[: cmap.shape[0] // 2, : cmap.shape[1] // 2] = 1.0
    poa /= poa.sum()
    cmap.set_zones(np.full(cmap.shape, ZONE_CODE["fan"], dtype=np.uint8))
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    assert [s.priority for s in segs] == [1, 2, 3, 4]
    assert segs[0].poa_mass == pytest.approx(1.0, abs=0.02)
    assert segs[0].seg_id == "Segment A" and segs[0].zone == "fan"
    assert "I" not in "".join(s.seg_id for s in segs), "I and O read as 1 and 0 on a radio"

    gj = segments_geojson(segs, SCENE)
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 4
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "a GeoJSON polygon ring is closed"
    assert 76.0 < ring[0][0] < 76.3 and 11.0 < ring[0][1] < 11.9
    assert all(f["properties"]["domain"] == "sim" for f in gj["features"])


def test_assignment_record_recommends_from_the_numbers_and_never_closes_a_segment():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    segs = auto_segments(cmap, SCENE, poa=poa, n_rows=2, n_cols=2)
    seg = segs[0]

    fresh = assignment_record(seg, cmap, SCENE, poa=poa, camera=CAM, local_hour=12.0)
    assert fresh.domain == "sim" and "not yet flown" in fresh.recommendation
    assert set(fresh.pod) == set(cmap.presentations)
    assert "cleared" not in str(fresh).lower()
    assert "pre-dawn thermal pass" in fresh.recommendation, "it beats the crossover window on its own"

    # a well-searched body layer and a thin limb layer must produce the §5.3b sentence, with a real altitude
    m = seg.mask(cmap, SCENE)
    cmap.grids["body"].coverage[m] = 3.0
    cmap._recompute()
    seg.passes_flown = 2
    flown = assignment_record(seg, cmap, SCENE, poa=poa, camera=CAM, local_hour=12.0)
    assert flown.pod["body"] > 0.9 and flown.pod["limb_only"] == 0.0
    # the recommended altitude is derived from THIS camera's limb-only ceiling (22 m for the sim 4K at
    # 75.5 deg; the document's 24 m is the M3T wide 4K), not typed in
    assert "limb_only" in flown.recommendation
    assert f"{CAM.ceiling_m('limb_only'):.0f} m" in flown.recommendation
    assert CAM.ceiling_m("limb_only") == pytest.approx(22.0, abs=0.5)
    assert "(simulation)" in flown.recommendation

    # a segment over a burial polygon says what it is, and still is not closed
    cmap.set_cannot_clear(m)
    buried = assignment_record(seg, cmap, SCENE, poa=poa, camera=CAM, local_hour=12.0)
    assert buried.cannot_clear_fraction == pytest.approx(1.0)
    assert "aerial search cannot clear it" in buried.recommendation
    assert "radar" in buried.recommendation or "canine" in buried.recommendation
    assert "cleared" not in buried.recommendation


def test_recommend_never_emits_a_closing_sentence():
    seg = Segment("Segment A", pat.rect_polygon((0.0, 0.0), 100.0, 100.0), passes_flown=9)
    for pod in ({"body": 0.99, "limb_only": 0.99}, {"body": 0.0}, {"body": 0.7, "limb_only": 0.05}):
        text = recommend(seg, pod, 0.0, camera=CAM, local_hour=12.0, thermal_flown=True)
        assert "(simulation)" in text
        for banned in ("cleared", "complete", "finished", "done", "no further"):
            assert banned not in text.lower(), text


def test_revisit_queue_is_ordered_by_residual_pos_and_skips_burial_polygons():
    cmap = _cmap(cell_m=25.0)
    poa = _flat_poa(cmap)
    n = cmap.shape[0]
    cmap.grids["body"].coverage[: n // 2, :] = 4.0  # the north half is well searched
    cmap._recompute()
    cmap.add_burial_polygons([np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])])

    items = from_coverage(cmap, SCENE, poa=poa, presentation="body", pod_threshold=0.5, min_cells=4)
    assert items and all(it.kind == "low_pod" for it in items)
    assert [it.priority for it in items] == sorted((it.priority for it in items), reverse=True)
    assert all("residual POS" in it.reason for it in items)
    # nothing queued lands inside the burial polygon
    from sightline.coverage.grid import cell_of_latlon
    for it in items:
        lat, lon = SCENE.to_latlon(*it.ne)
        cell = cell_of_latlon(cmap.any_grid, lat, lon)
        assert cell is None or not cmap.cannot_clear[cell]
    assert from_coverage(cmap, SCENE, poa=poa, pod_threshold=0.0) == []

    q = RevisitQueue()
    q.extend(items)
    rt = q.to_route(SCENE, camera=CAM)
    assert len(rt) > 0 and rt.pattern == "revisit"
    assert all("revisit:" in w.reason for w in rt.waypoints)
    assert "residual POS = POA x (1 - POD)" in rt.notes[0]
    assert q.pop().priority == max(it.priority for it in items)


def test_stale_records_are_revisited_and_dismissed_ones_are_neither_flown_nor_deleted():
    now = 1_780_000_000.0
    recs = [
        Record(lat=11.4875, lon=76.1455, status="stale", score=2.0, confidence=0.4, last_seen_utc=now - 60),
        Record(lat=11.4880, lon=76.1460, status="confirmed", score=1.0, confidence=0.9,
               last_seen_utc=now - 3600),
        Record(lat=11.4885, lon=76.1465, status="confirmed", score=9.0, confidence=0.5, last_seen_utc=now - 5),
        Record(lat=11.4890, lon=76.1470, status="dismissed", score=9.9, confidence=0.1,
               last_seen_utc=now - 9999, dismissed_reason="operator: roofing sheet"),
    ]
    items = from_records(recs, SCENE, now_utc=now)
    ids = {it.record_id for it in items}
    assert recs[0].record_id in ids and recs[1].record_id in ids
    assert recs[2].record_id not in ids, "recently seen and not stale"
    assert recs[3].record_id not in ids, "a dismissed record stops attracting effort..."
    assert len(recs) == 4 and recs[3].dismissed_reason, "...but is still in the log, with its reason (R10)"
    rt = RevisitQueue(items).to_route(SCENE)
    assert all(w.action == "orbit" for w in rt.waypoints)


def test_route_helpers():
    rt = Route(pattern="composite", scene=SCENE)
    assert len(rt) == 0 and rt.length_m() == 0.0 and rt.duration_s() == 0.0
    rt.extend([make_waypoint(0, 0.0, 0.0, 40.0, SCENE, None, speed_ms=10.0),
               make_waypoint(1, 100.0, 0.0, 40.0, SCENE, None, speed_ms=10.0, dwell_s=5.0)])
    assert rt.length_m() == pytest.approx(100.0)
    assert rt.duration_s() == pytest.approx(100.0 / 10.0 + 5.0)
    assert rt.duration_s(cruise_ms=20.0) == pytest.approx(5.0 + 5.0)
    assert [w.seq for w in rt.waypoints] == [0, 1]
    assert pat.polygon_area_m2(pat.rect_polygon((0.0, 0.0), 20.0, 30.0)) == pytest.approx(600.0)
    assert pat.polygon_centroid_ne(pat.rect_polygon((5.0, 7.0), 20.0, 30.0)) == pytest.approx((5.0, 7.0))
    assert pat.bearing_deg((0.0, 0.0), (0.0, 10.0)) == pytest.approx(90.0)


def test_candidate_serialisation_drops_arrays_but_keeps_the_reason():
    c = Candidate("survey", "survey Segment A at 60 m rgb", 0.031, 0.12, 30.0, 300.0,
                  {"mask": np.zeros((2, 2)), "agl_m": 60.0}, reason="because")
    d = c.as_dict()
    assert "mask" not in d["params"] and d["params"]["agl_m"] == 60.0
    assert d["reason"] == "because" and json.dumps(d)
