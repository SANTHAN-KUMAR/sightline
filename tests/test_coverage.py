"""F16 / F16b — the search-quality map, verified against SOLUTION_DOC §5.3, §5.3a, §5.3b, §2.7 and Appendix B.

Every number asserted here is either printed in the solution document or derivable on paper from it. Nothing in
this file calls the simulator, the editor, a GPU or a model: the coverage lane is pure numpy over `Telemetry` +
`Intrinsics`, so it is provable offline (docs/CONTRACTS.md §3).

    D:\\Tools\\uv\\uv.exe run pytest tests/test_coverage.py -q
"""

from __future__ import annotations

import io
import json
import math
import tokenize
from pathlib import Path

import numpy as np
import pytest

from sightline.coverage import calibrate as cal
from sightline.coverage.accumulate import (
    ZONE_CODE,
    ZONE_NAMES,
    CoverageMap,
    conditions_from_telemetry,
    time_of_day_label,
)
from sightline.coverage.export import (
    CANNOT_CLEAR_LABEL,
    EFFECTIVE_LAYER,
    banded_geojson,
    export_coverage,
    manifest,
    overlay_rgba,
)
from sightline.coverage.footprint import along_track_m, gimbal_quat, ground_footprint, swath_m
from sightline.coverage.grid import (
    SceneFrame,
    cell_centres_m,
    cell_of_latlon,
    grid_ne_to_latlon,
    latlon_to_grid_ne,
    make_grid,
    points_in_polygon,
    polygon_cell_weights,
    rasterise_polygons,
)
from sightline.coverage.presentation import (
    CRITICAL_DIM_M,
    LEGAL_CEILING_M,
    M3T_THERMAL_640,
    M3T_WIDE_4K,
    MIN_PX_FOR_RECALL,
    SIM_RGB_4K,
    ZERO_LAYER_PRESENTATIONS,
    CameraModel,
    altitude_ceiling_table,
    mix_for_zone,
    update_mix_from_observations,
)
from sightline.coverage.prior import (
    PRIOR_FLOOR,
    LastKnownPosition,
    bayesian_update,
    build_prior,
    probability_of_success,
)
from sightline.coverage.quality import (
    R_MAX,
    Conditions,
    SliceTable,
    analytic_recall,
    analytic_recall_array,
    recall_from_px,
    recall_from_px_array,
    slice_recall_array,
)
from sightline.schemas import PRESENTATIONS, CoverageGrid, Telemetry

REPO = Path(__file__).resolve().parents[1]

# A camera chosen so the nadir footprint lands on exact grid-cell boundaries: fx = fy = 1200 px
# (HFOV = 2 atan(600/1200) = 53.13 deg), so at 60 m AGL the footprint is 60 m across track x 30 m along track,
# which is 12 x 6 cells of 5 m. Every "exactly these cells" assertion below rests on that.
PROBE_CAM = CameraModel("test 1200px", 1200, 600, math.degrees(2.0 * math.atan(0.5)), "rgb", "tests")
PROBE_AGL = 60.0
CELL_M = 5.0
DAY = Conditions(band="rgb", time_of_day="day", weather="dry", speed_ms=6.0, exposure_s=1.0 / 500.0)


def _map(n=40, presentations=("body", "limb_only"), cell_m=CELL_M):
    return CoverageMap.create(11.4800, 76.1400, cell_m, n, n, presentations=presentations)


def _tel_over(cmap: CoverageMap, north_m: float, east_m: float, agl_m: float = PROBE_AGL,
              yaw_deg: float = 0.0, pitch_deg: float = -90.0, frame_idx: int = 0) -> Telemetry:
    """Telemetry putting the camera over grid-NE (north_m, east_m) of this map's SW corner."""
    g = cmap.any_grid
    lat, lon = grid_ne_to_latlon(g, north_m, east_m)
    return Telemetry(t_utc=0.0, lat=lat, lon=lon, alt_msl_m=1000.0 + agl_m, agl_m=agl_m,
                     q_gimbal=gimbal_quat(pitch_deg, yaw_deg), frame_idx=frame_idx)


# ---------------------------------------------------------------------------------------------------------
# 1. geometry: the footprint is Appendix B, not an approximation of it
# ---------------------------------------------------------------------------------------------------------
def test_gimbal_quat_round_trips_through_the_schema():
    """`footprint.gimbal_quat` must be the inverse of `Telemetry.gimbal_pitch_deg` — the frozen contract."""
    for pitch in (-90.0, -75.0, -45.0, -20.0):
        for yaw in (0.0, 37.0, 180.0, 300.0):
            tel = Telemetry(0.0, 11.0, 76.0, 100.0, 60.0, q_gimbal=gimbal_quat(pitch, yaw))
            assert tel.gimbal_pitch_deg() == pytest.approx(pitch, abs=1e-9)


def test_nadir_footprint_is_the_appendix_b_rectangle():
    intr = PROBE_CAM.intrinsics()
    tel = Telemetry(0.0, 11.487, 76.145, 1060.0, PROBE_AGL, q_gimbal=gimbal_quat(-90.0))
    fp = ground_footprint(tel, intr)

    assert fp.valid and not fp.clipped
    assert fp.off_nadir_deg == pytest.approx(0.0, abs=1e-9)
    # Appendix B: width = 2 h tan(HFOV/2) = agl * W_px / fx, height = agl * H_px / fy
    #
    # `poly_ne_m` columns are [north, east]. ACROSS-TRACK is EAST: `survey.py` flies north-south legs and
    # spaces them in east by `W = 2 * alt * tan(hfov/2)`, the WIDE swath. This test previously read `across`
    # from column 0 (north) and so asserted the wide axis lay north-south - encoding a quarter-turn bug in
    # `ground_footprint`, which applied the identity `q_gimbal` of a nadir camera to an OPTICAL ray and
    # mapped image-right to north.
    #
    # Settled by measurement, not argument: across 110 boxes whose survivors have known world positions,
    # image-right is due EAST (median residual 1.37 m; the next-best hypothesis 15.8 m, an order of
    # magnitude worse). See docs/CONTEXT.md.
    across = fp.poly_ne_m[:, 1].max() - fp.poly_ne_m[:, 1].min()   # east
    along = fp.poly_ne_m[:, 0].max() - fp.poly_ne_m[:, 0].min()    # north
    assert across == pytest.approx(swath_m(intr, PROBE_AGL)) == pytest.approx(60.0)
    assert along == pytest.approx(along_track_m(intr, PROBE_AGL)) == pytest.approx(30.0)
    assert fp.area_m2() == pytest.approx(60.0 * 30.0)
    # GSD = 2 h tan(HFOV/2) / W_px = agl / fx
    assert fp.gsd_nadir_m == pytest.approx(PROBE_AGL / intr.fx) == pytest.approx(0.05)
    assert PROBE_CAM.gsd_m(PROBE_AGL) == pytest.approx(0.05)
    # and it reduces to exactly h/f directly under the camera
    assert float(fp.gsd_at(np.array([0.0]), np.array([0.0]))[0]) == pytest.approx(0.05)


def test_oblique_footprint_grows_and_is_clipped_near_the_horizon():
    intr = PROBE_CAM.intrinsics()
    nadir = ground_footprint(Telemetry(0.0, 11.487, 76.145, 1060.0, 60.0, q_gimbal=gimbal_quat(-90.0)), intr)
    oblique = ground_footprint(Telemetry(0.0, 11.487, 76.145, 1060.0, 60.0, q_gimbal=gimbal_quat(-40.0)), intr)
    assert oblique.area_m2() > nadir.area_m2()
    assert oblique.off_nadir_deg == pytest.approx(50.0, abs=1e-6)
    horizon = ground_footprint(Telemetry(0.0, 11.487, 76.145, 1060.0, 60.0, q_gimbal=gimbal_quat(-2.0)), intr)
    assert horizon.clipped, "a frame pointed at the horizon must be clipped, not unbounded"
    assert np.isfinite(horizon.poly_ne_m).all()
    below = ground_footprint(Telemetry(0.0, 11.487, 76.145, 1060.0, 0.0, q_gimbal=gimbal_quat(-90.0)), intr)
    assert not below.valid and below.reject_reason


def test_grid_and_scene_frames_round_trip():
    g = make_grid(11.4800, 76.1400, 10.0, 20, 20)
    # cell midpoints, not boundaries: `offset_ne` (cos lat1) and `ne_between` (cos mean lat) are
    # different approximations, so a point sitting exactly on a cell edge can round either way.
    for north, east in ((5.0, 5.0), (55.0, 135.0), (195.0, 15.0)):
        lat, lon = grid_ne_to_latlon(g, north, east)
        back_n, back_e = latlon_to_grid_ne(g, lat, lon)
        # `common/geodesy.py` claims agreement to < 1 cm over the few-km offsets this project uses; the
        # forward step uses cos(lat1) and the inverse cos(mean lat), so they differ by less than that.
        assert back_n == pytest.approx(north, abs=0.01)
        assert back_e == pytest.approx(east, abs=0.01)
        assert cell_of_latlon(g, lat, lon) == (int(north // 10.0), int(east // 10.0))
    assert cell_of_latlon(g, *grid_ne_to_latlon(g, -50.0, 5.0)) is None

    scene = SceneFrame(11.4870, 76.1450, 1046.0, 2048.0)
    lat, lon = scene.to_latlon(120.0, -300.0)
    n, e = scene.to_scene_ne(lat, lon)
    assert (n, e) == pytest.approx((120.0, -300.0), abs=0.01)
    assert scene.ue_cm(120.0, -300.0, 1106.0) == pytest.approx((12000.0, -30000.0, 6000.0))


def test_polygon_rasterisation_is_area_weighted():
    g = make_grid(11.48, 76.14, 10.0, 10, 10)
    # a 20 m x 20 m square on exact cell boundaries -> four cells at weight 1
    poly = np.array([[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]])
    _win, w = polygon_cell_weights(g, poly, supersample=4)
    assert w[w > 0].sum() == pytest.approx(4.0)
    assert set(np.unique(w[w > 0]).tolist()) == {1.0}
    # a half-cell offset gives fractional edge weights that still sum to the true area in cells
    poly2 = poly + 5.0
    _, w2 = polygon_cell_weights(g, poly2, supersample=8)
    assert w2.sum() == pytest.approx(4.0, abs=0.05)
    assert 0.0 < w2.max() <= 1.0
    mask = rasterise_polygons(g, [poly])
    assert mask.sum() == 4
    assert points_in_polygon(poly, np.array([[30.0, 30.0], [10.0, 10.0]])).tolist() == [True, False]


# ---------------------------------------------------------------------------------------------------------
# 2. the core claim: one nadir pass raises POD in exactly the covered cells and nowhere else
# ---------------------------------------------------------------------------------------------------------
def test_single_nadir_pass_raises_pod_in_exactly_the_covered_cells():
    cmap = _map(n=40)
    intr = PROBE_CAM.intrinsics()
    # Camera over grid-NE (100, 100). ACROSS-TRACK IS EAST (see test_nadir_footprint_is_the_appendix_b_
    # rectangle), so the 60 m x 30 m footprint is east [70, 130) and north [85, 115) - rows 17..22 and
    # columns 14..25 of a 5 m grid. Still exactly 72 cells; nothing else may move. This block was
    # transposed while `ground_footprint` mapped image-right to north.
    fc = cmap.add_frame(_tel_over(cmap, 100.0, 100.0), intr, DAY, pass_id=0)
    assert fc.skipped_reason == ""
    cmap.end_pass()

    cov, pod = cmap.coverage("body"), cmap.pod("body")
    expected = np.zeros(cmap.shape, dtype=bool)
    expected[17:23, 14:26] = True      # rows = north (30 m), cols = east (60 m)
    assert expected.sum() == 72
    np.testing.assert_array_equal(cov > 0.0, expected)
    np.testing.assert_array_equal(pod > 0.0, expected)
    assert cov[~expected].max() == 0.0 and pod[~expected].max() == 0.0
    assert fc.cells_touched == 72

    # POD is exactly the closed form of §5.3, cell by cell
    k = cmap.grids["body"].k
    np.testing.assert_allclose(pod, 1.0 - np.exp(-k * cov), rtol=0, atol=1e-6)
    # one look can never search a cell completely: q <= R_MAX (quality.py) and POD < 1
    assert cov.max() <= R_MAX and pod.max() < 1.0
    assert fc.q_max["body"] == pytest.approx(float(cov.max()))
    # the best GSD in the frame is the nadir one; the worst is at a corner
    assert fc.best_gsd_m == pytest.approx(PROBE_CAM.gsd_m(PROBE_AGL), rel=0.05)
    assert fc.worst_gsd_m > fc.best_gsd_m


def test_the_covered_cells_are_the_cells_the_camera_actually_saw():
    """The raised set must equal an independent rasterisation of the projected footprint polygon."""
    cmap = _map(n=40)
    intr = PROBE_CAM.intrinsics()
    tel = _tel_over(cmap, 100.0, 100.0, yaw_deg=35.0)
    cmap.add_frame(tel, intr, DAY, pass_id=0)
    cmap.end_pass()

    fp = ground_footprint(tel, intr)
    cam_n, cam_e = latlon_to_grid_ne(cmap.any_grid, tel.lat, tel.lon)
    win, w = polygon_cell_weights(cmap.any_grid, fp.poly_ne_m + np.array([cam_n, cam_e]), supersample=3)
    seen = np.zeros(cmap.shape, dtype=bool)
    seen[win.slice()] = w > 0.0
    np.testing.assert_array_equal(cmap.coverage("body") > 0.0, seen)


def test_a_frame_outside_the_grid_contributes_nothing():
    cmap = _map(n=20)
    fc = cmap.add_frame(_tel_over(cmap, 5000.0, 5000.0), PROBE_CAM.intrinsics(), DAY, pass_id=0)
    assert fc.skipped_reason == "footprint outside the grid"
    cmap.end_pass()
    assert cmap.coverage("body").max() == 0.0


# ---------------------------------------------------------------------------------------------------------
# 3. POD composition: 1 - exp(-k C) over repeated passes, checked against the closed form by hand
# ---------------------------------------------------------------------------------------------------------
def test_pod_composes_as_the_closed_form_over_repeated_passes():
    # A low-quality look (RGB at night, §2.3: "very nearly blind") so six passes stay clear of the POD_MAX
    # clamp and the raw closed form is what is being checked. Saturation has its own test below.
    night = Conditions(band="rgb", time_of_day="night", weather="dry", speed_ms=6.0, exposure_s=1.0 / 500.0)
    cmap = _map(n=40)
    intr = PROBE_CAM.intrinsics()
    tel = _tel_over(cmap, 100.0, 100.0)
    k = cmap.grids["body"].k

    cmap.add_frame(tel, intr, night, pass_id=0)
    cmap.end_pass()
    q = float(cmap.coverage("body")[20, 20])  # a fully-interior cell of the footprint
    assert 0.0 < q < 0.2
    pods = [float(cmap.pod("body")[20, 20])]

    for p in range(1, 6):
        cmap.add_frame(tel, intr, night, pass_id=p)
        cmap.end_pass()
        n = p + 1
        # C after n identical passes is exactly n q, and POD is the closed form of that
        assert float(cmap.coverage("body")[20, 20]) == pytest.approx(n * q, rel=1e-5)
        assert float(cmap.pod("body")[20, 20]) == pytest.approx(1.0 - math.exp(-k * n * q), abs=1e-6)
        pods.append(float(cmap.pod("body")[20, 20]))

    assert max(pods) < cal.POD_MAX, "this test must exercise the unclamped closed form"
    # strictly increasing, with a strictly shrinking increment (diminishing returns)
    d = np.diff(pods)
    assert (d > 0).all()
    assert (np.diff(d) < 0).all()
    assert pods[-1] < 1.0


def test_k_makes_n_passes_read_as_n_independent_looks():
    """§5.3 + calibrate.k_for_reference_quality: 1 - exp(-k n q) == 1 - (1 - q)^n at the reference quality."""
    for q in (0.10, 0.45, 0.9044):
        k = cal.k_for_reference_quality(q)
        assert 1.0 - math.exp(-k * q) == pytest.approx(q, abs=1e-9), "one pass reads back the detector's recall"
        for n in (1, 2, 3, 7):
            assert 1.0 - math.exp(-k * n * q) == pytest.approx(1.0 - (1.0 - q) ** n, abs=1e-9)
    assert cal.K_MIN <= cal.k_for_reference_quality(0.999999) <= cal.K_MAX
    assert cal.k_for_reference_quality(1e-9) == pytest.approx(cal.K_MIN)  # never below random search


def test_pod_saturates_towards_one_without_reaching_it():
    cmap = _map(n=24)
    intr = PROBE_CAM.intrinsics()
    tel = _tel_over(cmap, 60.0, 60.0)
    for p in range(200):
        cmap.add_frame(tel, intr, DAY, pass_id=p)
        cmap.end_pass()
    pod = cmap.pod("body")
    assert pod.max() == pytest.approx(cal.POD_MAX)
    assert cal.POD_MAX < 1.0
    assert (pod < 1.0).all(), "a cell may never read as fully searched (guardrail R10)"
    assert cmap.coverage("body").max() > 50.0, "effort keeps accumulating; only the DISPLAY saturates"
    assert cmap.pod_effective().max() <= cal.POD_MAX


def test_effort_is_accumulated_per_pass_not_per_frame():
    """§5.3: C = sum over PASSES. Ten frames of the same ground in one pass may not claim ten times the search."""
    intr = PROBE_CAM.intrinsics()
    one, ten = _map(n=30), _map(n=30)
    tel = _tel_over(one, 75.0, 75.0)
    one.add_frame(tel, intr, DAY, pass_id=0)
    one.end_pass()
    for i in range(10):
        ten.add_frame(_tel_over(ten, 75.0, 75.0, frame_idx=i), intr, DAY, pass_id=0)
    summary = ten.end_pass()
    assert summary.frames == 10
    np.testing.assert_allclose(one.coverage("body"), ten.coverage("body"), rtol=0, atol=1e-7)
    # ... but ten separate PASSES do accumulate
    for p in range(1, 10):
        ten.add_frame(_tel_over(ten, 75.0, 75.0), intr, DAY, pass_id=p)
        ten.end_pass()
    assert float(ten.coverage("body")[15, 15]) == pytest.approx(10.0 * float(one.coverage("body")[15, 15]),
                                                                rel=1e-5)


# ---------------------------------------------------------------------------------------------------------
# 4. §2.7 + guardrail R10: a burial cell keeps its floor POD under unlimited effort
# ---------------------------------------------------------------------------------------------------------
def test_burial_cells_keep_their_floor_pod_under_unlimited_effort():
    cmap = _map(n=40)
    intr = PROBE_CAM.intrinsics()
    # a burial polygon over grid-NE north [90, 110), east [90, 110): rows 18..21, cols 18..21
    poly = np.array([[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]])
    cmap.add_burial_polygons([poly])
    buried = cmap.cannot_clear
    assert buried.sum() == 16

    tel = _tel_over(cmap, 100.0, 100.0)
    floors = []
    for p in range(50):
        fc = cmap.add_frame(tel, intr, DAY, pass_id=p)
        cmap.end_pass()
        floors.append(float(cmap.pod("body")[buried].max()))
        assert float(fc.q_max["body"]) >= 0.0

    assert set(floors) == {0.0}, "no number of passes may raise the POD of a burial cell"
    assert cmap.coverage("body")[buried].max() == 0.0
    assert cmap.pod("body")[buried].max() == 0.0
    assert cmap.pod_effective()[buried].max() == 0.0
    # the surrounding cells the same frames covered DID rise, so the mask is doing the work, not an empty frame
    assert cmap.pod("body")[~buried].max() > 0.5
    # visibility was zeroed there too, so an independent consumer reaches the same answer
    for p in cmap.presentations:
        assert cmap.visibility[p][buried].max() == 0.0
    # ... and even a consumer that recomputes POD off the frozen schema type sees zero, because C is zero
    g = cmap.grids["body"]
    g.recompute_pod()
    assert g.pod[buried].max() == 0.0


def test_drawing_a_burial_polygon_late_retracts_the_effort_already_claimed():
    """§2.7 rule 1 is "mark, never clear": a polygon drawn after the flight must undo the claim, not keep it."""
    cmap = _map(n=40)
    intr = PROBE_CAM.intrinsics()
    cmap.add_frame(_tel_over(cmap, 100.0, 100.0), intr, DAY, pass_id=0)
    cmap.end_pass()
    assert cmap.pod("body")[19, 19] > 0.0
    cmap.add_burial_polygons([np.array([[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]])])
    assert cmap.pod("body")[19, 19] == 0.0
    assert cmap.coverage("body")[19, 19] == 0.0


def test_the_buried_presentation_layer_is_identically_zero_by_construction():
    """§5.3b: fully buried is the presentation whose layer is zero everywhere — derived, not a painted label."""
    cmap = _map(n=20, presentations=("body", "buried"))
    intr = PROBE_CAM.intrinsics()
    for p in range(5):
        cmap.add_frame(_tel_over(cmap, 50.0, 50.0), intr, DAY, pass_id=p)
        cmap.end_pass()
    assert cmap.coverage("buried").max() == 0.0
    assert cmap.pod("buried").max() == 0.0
    assert cmap.coverage("body").max() > 0.0
    assert CRITICAL_DIM_M["buried"] == 0.0 and "buried" in ZERO_LAYER_PRESENTATIONS
    assert analytic_recall("buried", 0.01, DAY).value == 0.0
    # setting a visibility raster cannot switch the layer back on
    cmap.set_visibility("buried", np.ones(cmap.shape, dtype=np.float32))
    assert cmap.visibility["buried"].max() == 0.0


def test_the_lane_has_no_cleared_state_anywhere():
    """Guardrail R10, checked in the code rather than in the prose: no identifier means "searched, done"."""
    allowed_with_clear = {
        "cannot_clear", "cannot_clear_label", "cannot_clear_fraction", "clearable_cells", "cannot_clear_cells",
        "set_cannot_clear", "CANNOT_CLEAR_LABEL", "clear_polygon", "clip_polygon",
    }
    seen: set[str] = set()
    files = sorted((REPO / "sightline" / "coverage").glob("*.py")) + \
        sorted((REPO / "sightline" / "plan").glob("*.py"))
    assert len(files) >= 14
    for path in files:
        src = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.NAME:
                continue
            low = tok.string.lower()
            assert "cleared" not in low, f"{path.name}:{tok.start[0]} defines/uses {tok.string!r}"
            if "clear" in low:
                seen.add(tok.string)
    assert seen <= allowed_with_clear, f"unexpected clear-ish identifiers: {sorted(seen - allowed_with_clear)}"
    assert not hasattr(CoverageGrid, "cleared")
    assert not any("clear" in a and "cannot" not in a for a in dir(CoverageMap) if not a.startswith("_"))


# ---------------------------------------------------------------------------------------------------------
# 5. §5.3b: the layers are independent, and they reproduce the document's own tables
# ---------------------------------------------------------------------------------------------------------
def test_presentation_layers_are_independent():
    cmap = _map(n=40, presentations=("body", "limb_only"))
    intr = SIM_RGB_4K.intrinsics()
    tel = _tel_over(cmap, 100.0, 100.0)
    cmap.add_frame(tel, intr, DAY, pass_id=0)
    cmap.end_pass()

    body, limb = cmap.coverage("body"), cmap.coverage("limb_only")
    assert body.max() > 0.0 and limb.max() > 0.0
    assert body.max() > 3.0 * limb.max(), "a 60 m pass searches far harder for a body than for a hand"
    assert cmap.grids["body"].k != cmap.grids["limb_only"].k, "each layer carries its own k (§5.3b)"
    assert body is not limb and not np.shares_memory(body, limb)

    # writing one layer's visibility must not touch the other
    v = np.zeros(cmap.shape, dtype=np.float32)
    cmap.set_visibility("limb_only", v)
    before = body.copy()
    cmap.add_frame(tel, intr, DAY, pass_id=1)
    cmap.end_pass()
    assert cmap.coverage("limb_only").max() == pytest.approx(limb.max()), "V=0 stops the limb layer accruing"
    assert (cmap.coverage("body") > before).any(), "the body layer kept accruing"

    # and the §5.3b sentence the map exists to say is generated from those two numbers
    s = cmap.statement("body", "limb_only")
    assert "searched for body" in s and "limb_only" in s and "simulation" in s and "cleared" not in s


def test_altitude_ceilings_reproduce_the_document_table():
    """§5.3b: 'Highest altitude still >= 20 px, 4K wide' — 120 / 120 / 60 / 60 / 33 / 24 m, and thermal beside it."""
    wide = altitude_ceiling_table(M3T_WIDE_4K)
    assert wide["body"] == pytest.approx(LEGAL_CEILING_M) == pytest.approx(120.0)
    assert wide["cluster"] == pytest.approx(120.0)
    assert wide["upright"] == pytest.approx(60.0, abs=0.5)
    assert wide["wading"] == pytest.approx(60.0, abs=0.5)
    assert wide["head_only"] == pytest.approx(33.0, abs=0.5)
    assert wide["limb_only"] == pytest.approx(24.0, abs=0.5)
    assert wide["buried"] == 0.0, "fully buried: never"
    assert MIN_PX_FOR_RECALL == 20.0, "the ceilings above are the >= 20 px rule of §2.5"
    assert M3T_WIDE_4K.geometric_ceiling_m("body") == pytest.approx(227.0, abs=1.0)  # doc: "geometric ~227 m"

    thermal = altitude_ceiling_table(M3T_THERMAL_640)
    assert thermal["body"] == pytest.approx(64.0, abs=1.0)
    assert thermal["upright"] == pytest.approx(17.0, abs=0.5)
    assert thermal["limb_only"] == pytest.approx(7.0, abs=0.5)

    # §5.3b pixels-on-target row: prone body 150 / 76 / 50 / 38 px at 30 / 60 / 90 / 120 m
    px = [M3T_WIDE_4K.px_on_target(CRITICAL_DIM_M["body"], h) for h in (30, 60, 90, 120)]
    assert [round(v) for v in px] == [151, 76, 50, 38]
    limb = [M3T_WIDE_4K.px_on_target(CRITICAL_DIM_M["limb_only"], h) for h in (30, 60, 90, 120)]
    assert [round(v) for v in limb] == [16, 8, 5, 4]
    assert set(CRITICAL_DIM_M) == set(PRESENTATIONS)


def test_recall_model_sits_on_its_published_anchors_and_is_labelled_a_model():
    assert recall_from_px(20.0) == pytest.approx(0.70, abs=1e-9)   # §2.5 design floor
    assert recall_from_px(60.0) == pytest.approx(0.90, abs=1e-9)   # HERIDAL/AIR anchor
    assert recall_from_px(1e6) < R_MAX + 1e-9 and recall_from_px(1e6) > 0.96
    assert recall_from_px(10.0) == pytest.approx(0.47, abs=0.01)   # docstring's AIResQ sanity check
    assert recall_from_px(150.0) == pytest.approx(0.95, abs=0.01)
    assert recall_from_px(0.0) == 0.0 and recall_from_px(-3.0) == 0.0
    # sub-8-px taper: strictly below the bare logistic
    assert recall_from_px(5.0) < 0.5 * recall_from_px(10.0)
    px = np.array([0.0, 4.0, 8.0, 20.0, 60.0, 500.0])
    np.testing.assert_allclose(recall_from_px_array(px), [recall_from_px(float(p)) for p in px], atol=1e-12)

    est = analytic_recall("body", SIM_RGB_4K.gsd_m(60.0), DAY)
    assert est.measured is False and "NOT a measured slice" in est.basis
    assert 0.85 < est.value < 0.95
    arr = analytic_recall_array("body", np.full((2, 2), SIM_RGB_4K.gsd_m(60.0)), DAY)
    np.testing.assert_allclose(arr, est.value, atol=1e-12)


def test_condition_factors_move_recall_in_the_documented_directions():
    gsd = SIM_RGB_4K.gsd_m(60.0)
    day_rgb = analytic_recall("body", gsd, Conditions(band="rgb", time_of_day="day")).value
    night_rgb = analytic_recall("body", gsd, Conditions(band="rgb", time_of_day="night")).value
    night_ir = analytic_recall("body", gsd, Conditions(band="thermal", time_of_day="night")).value
    dawn_ir = analytic_recall("body", gsd, Conditions(band="thermal", time_of_day="dawn")).value
    crossover = analytic_recall("body", gsd, Conditions(band="thermal", time_of_day="dawn",
                                                       local_hour=6.9)).value
    assert night_rgb < 0.2 * day_rgb, "RGB is very nearly blind at night (§2.3)"
    assert night_ir > 5 * night_rgb, "thermal is the night sensor"
    assert crossover < 0.5 * dawn_ir, "the ~06:50 crossover window costs the thermal band (§2.5)"
    fog = analytic_recall("body", gsd, Conditions(band="rgb", time_of_day="day", weather="fog")).value
    rain = analytic_recall("body", gsd, Conditions(band="rgb", time_of_day="day", weather="heavy_rain")).value
    assert fog < rain < day_rgb
    # motion blur is folded into the effective resolution: a slow shutter costs pixels
    slow = analytic_recall("body", gsd, Conditions(speed_ms=12.0, exposure_s=1.0 / 60.0)).value
    assert slow < 0.6 * day_rgb


def test_a_measured_slice_overrides_the_model_only_when_it_has_enough_samples():
    gsd = np.full((3, 3), SIM_RGB_4K.gsd_m(60.0))
    fat = SliceTable(source="unit"); fat.put("body", 60.0, "rgb", "day", 0.55, 100)
    arr, est = slice_recall_array(fat, "body", 60.0, DAY, gsd)
    assert est.measured and est.n == 100 and est.value == pytest.approx(0.55)
    assert float(arr.max()) == pytest.approx(0.55, abs=1e-6)

    thin = SliceTable(); thin.put("body", 60.0, "rgb", "day", 0.55, 5)
    arr2, est2 = slice_recall_array(thin, "body", 60.0, DAY, gsd)
    assert not est2.measured, "a slice below min_n must fall back to the labelled model, not be trusted"
    assert float(arr2.max()) == pytest.approx(analytic_recall("body", SIM_RGB_4K.gsd_m(60.0), DAY).value)
    # a measured slice can never claim a perfect detector either
    hot = SliceTable(); hot.put("body", 60.0, "rgb", "day", 1.0, 500)
    _, est3 = slice_recall_array(hot, "body", 60.0, DAY, gsd)
    assert est3.value <= R_MAX


def test_zone_mixture_is_a_proper_distribution_and_learns_from_observations():
    for zone in ("settlement", "channel", "fan", "hillslope", "unknown", "nonsense"):
        mix = mix_for_zone(zone)
        assert sum(mix.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in mix.values())
    two = mix_for_zone("fan", ("body", "limb_only"))
    assert sum(two.values()) == pytest.approx(1.0) and set(two) == {"body", "limb_only"}
    # §5.3b: a fan that turns out to be producing limb-only detections re-weights its own map
    post = update_mix_from_observations("fan", {"limb_only": 40})
    assert post["limb_only"] > mix_for_zone("fan")["limb_only"]
    assert sum(post.values()) == pytest.approx(1.0)
    assert mix_for_zone("fan")["limb_only"] == pytest.approx(0.25), "ZONE_MIX must not be mutated"


def test_pod_effective_is_the_zone_weighted_mixture_of_the_layers():
    cmap = _map(n=20, presentations=("body", "limb_only"))
    cmap.set_zones(np.full(cmap.shape, ZONE_CODE["fan"], dtype=np.uint8))
    cmap.add_frame(_tel_over(cmap, 50.0, 50.0), SIM_RGB_4K.intrinsics(), DAY, pass_id=0)
    cmap.end_pass()
    w = mix_for_zone("fan", ("body", "limb_only"))
    expect = w["body"] * cmap.pod("body") + w["limb_only"] * cmap.pod("limb_only")
    np.testing.assert_allclose(cmap.pod_effective(), expect, atol=1e-6)
    # the mixture lies between the two layers everywhere, so it can never flatter the weaker one
    hot = cmap.pod("body") > 0
    assert (cmap.pod_effective()[hot] < cmap.pod("body")[hot]).all()
    assert (cmap.pod_effective()[hot] > cmap.pod("limb_only")[hot]).all()
    assert cmap.zone_at(0, 0) == "fan" and ZONE_NAMES[0] == "unknown"


def test_delta_pod_diminishes_and_is_zero_inside_burial_polygons():
    """§5.3a: dPOD = e^(-kC_before) - e^(-kC_after), so a repeat pass in identical conditions scores ~0."""
    cmap = _map(n=20)
    cmap.add_burial_polygons([np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])])
    g = cmap.grids["body"]
    first = cmap.delta_pod("body", 0.9)
    live = ~cmap.cannot_clear
    assert first[live].max() > 0.5
    assert first[cmap.cannot_clear].max() == 0.0, "the planner can never be paid for a burial cell"

    g.coverage[:] = 5.0
    g.coverage[cmap.cannot_clear] = 0.0
    later = cmap.delta_pod("body", 0.9)
    assert later[live].max() < 0.01 * first[live].max(), "diminishing returns, numerically"
    # the closed form, independently
    np.testing.assert_allclose(later[live], (np.exp(-g.k * 5.0) - np.exp(-g.k * 5.9)), atol=1e-6)
    mix = cmap.delta_pod_mixture({"body": 0.9, "limb_only": 0.2})
    assert mix[cmap.cannot_clear].max() == 0.0
    assert mix[live].max() > 0.0


# ---------------------------------------------------------------------------------------------------------
# 6. calibration: k is fitted where there is data, and labelled a placeholder where there is not
# ---------------------------------------------------------------------------------------------------------
def test_shipped_k_is_labelled_derived_not_measured():
    for name, kv in cal.DEFAULT_K.items():
        assert kv.measured is False, f"{name}: the shipped k must never claim to be calibrated"
        assert "PLACEHOLDER" in kv.basis or "zero-layer" in kv.basis
        assert cal.K_MIN <= kv.k <= cal.K_MAX
    assert cal.k_for("body") == cal.DEFAULT_K["body"].k
    assert cal.k_for("not-a-presentation") == cal.K_MIN
    # the derivation is reproducible from the model at the stated operating point
    q = analytic_recall("body", SIM_RGB_4K.gsd_m(cal.NOMINAL_AGL_M), cal.NOMINAL_CONDITIONS).value
    assert cal.DEFAULT_K["body"].k == pytest.approx(cal.k_for_reference_quality(q))
    assert cal.DEFAULT_K["body"].q_ref == pytest.approx(q)


def test_calibrate_k_from_outcomes_recovers_a_known_k():
    """The real §5.12 path: fit k to (coverage, found) pairs by maximum likelihood."""
    rng = np.random.default_rng(11)
    true_k = 2.0
    coverage = rng.uniform(0.05, 2.0, size=6000)
    found = rng.random(6000) < (1.0 - np.exp(-true_k * coverage))
    kv = cal.calibrate_k_from_outcomes(coverage, found)
    assert kv.measured is True and kv.n == 6000
    assert kv.k == pytest.approx(true_k, abs=0.12)
    # a detector that finds nothing pushes k to the floor; one that finds everything to the ceiling
    assert cal.calibrate_k_from_outcomes(coverage, np.zeros(6000, bool)).k == pytest.approx(cal.K_MIN, abs=1e-3)
    assert cal.calibrate_k_from_outcomes(coverage, np.ones(6000, bool)).k == pytest.approx(cal.K_MAX, abs=1e-3)
    with pytest.raises(ValueError):
        cal.calibrate_k_from_outcomes(np.zeros(0), np.zeros(0))


def test_reliability_diagram_detects_an_honest_and_a_dishonest_map():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=4000)
    honest = cal.reliability_diagram(p, rng.random(4000) < p, n_bins=5)
    assert cal.expected_calibration_error(honest) < 0.05
    assert sum(b.n for b in honest) == 4000, "every sample lands in exactly one bin"
    liar = cal.reliability_diagram(p, rng.random(4000) < p * 0.4, n_bins=5)
    assert cal.expected_calibration_error(liar) > 0.15
    lo, hi = cal.wilson_interval(7, 10)
    assert 0.0 < lo < 0.7 < hi < 1.0
    assert cal.wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        cal.reliability_diagram(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------------------------------------
# 7. the prior and the Bayesian update (Appendix B)
# ---------------------------------------------------------------------------------------------------------
def test_prior_is_a_probability_mass_and_a_last_known_position_dominates_it():
    cmap = _map(n=32, cell_m=10.0)
    codes = np.full(cmap.shape, ZONE_CODE["hillslope"], dtype=np.uint8)
    codes[:, :8] = ZONE_CODE["settlement"]
    cmap.set_zones(codes)
    flat = build_prior(cmap)
    assert flat.total.sum() == pytest.approx(1.0, abs=1e-5)
    assert flat.total[0, 0] > flat.total[0, 20], "the settlement zone outranks the hillslope (§2.4)"

    lat, lon = grid_ne_to_latlon(cmap.any_grid, 155.0, 255.0)
    withlkp = build_prior(cmap, lkps=[LastKnownPosition(lat, lon, sigma_m=30.0, label="phone")])
    assert withlkp.total.sum() == pytest.approx(1.0, abs=1e-5)
    assert withlkp.lkp.sum() == pytest.approx(40.0, rel=0.05), "an LKP contributes its stated mass"
    pin = np.unravel_index(int(np.argmax(withlkp.lkp)), cmap.shape)
    assert pin == (15, 25)
    assert withlkp.total[pin] > 4.0 * flat.total[pin], "the pin dominates its own neighbourhood"
    assert withlkp.total[pin] == withlkp.total[cmap.zone_codes == ZONE_CODE["hillslope"]].max()
    assert "1 last-known-position pins" in " ".join(withlkp.notes)
    assert set(withlkp.as_dict()) == {"zone", "buildings", "high_ground", "channel", "lkp", "total"}


def test_prior_weights_flooded_buildings_and_channel_bends(tmp_path):
    """§5.3: buildings INSIDE the flood footprint, weighted by roof area and storeys; bends on the channel."""
    scene = SceneFrame(11.4870, 76.1450, 1046.0, 640.0)
    cmap = CoverageMap.for_scene(scene, cell_m=10.0)
    codes = np.full(cmap.shape, ZONE_CODE["settlement"], dtype=np.uint8)
    cmap.set_zones(codes)
    settlement = tmp_path / "settlement.json"
    settlement.write_text(json.dumps({
        "archetypes": {"small": {"length_m": 8.0, "width_m": 6.0, "storeys": 1},
                       "big": {"length_m": 12.0, "width_m": 10.0, "storeys": 2}},
        "houses": [
            {"archetype": "big", "north_m": 100.0, "east_m": 0.0, "flood_depth_m": 3.0},
            {"archetype": "small", "north_m": -100.0, "east_m": 0.0, "flood_depth_m": 1.0},
            {"archetype": "big", "north_m": 0.0, "east_m": 200.0, "flood_depth_m": 0.0},  # dry: not counted
        ]}), encoding="utf-8")
    # a channel that runs straight and then turns sharply: the bend must be the hotter node
    centreline = [{"x_m": e, "y_m": -250.0} for e in np.linspace(-250.0, 0.0, 12)]
    centreline += [{"x_m": 0.0, "y_m": n} for n in np.linspace(-240.0, 0.0, 12)]

    layers = build_prior(cmap, settlement_json=settlement, channel_centreline=centreline, scene=scene)
    assert layers.total.sum() == pytest.approx(1.0, abs=1e-5)
    assert "2 flooded buildings" in " ".join(layers.notes)
    assert "24 channel centreline nodes" in " ".join(layers.notes)

    def cell(n_m, e_m):
        lat, lon = scene.to_latlon(n_m, e_m)
        return cell_of_latlon(cmap.any_grid, lat, lon)

    big, small, dry = cell(100.0, 0.0), cell(-100.0, 0.0), cell(0.0, 200.0)
    assert layers.buildings[big] > layers.buildings[small], "roof area x storeys x flood depth"
    assert layers.buildings[dry] == 0.0, "a dry building is outside the flood footprint"
    assert layers.channel[cell(-250.0, 0.0)] > layers.channel[cell(-250.0, -240.0)], "the bend outranks the reach"
    assert layers.channel.sum() > 0.0 and layers.high_ground.sum() > 0.0
    np.testing.assert_allclose(layers.total.sum(), 1.0, atol=1e-5)


def test_prior_does_not_change_with_the_raster_resolution():
    """§5.3 allows 5-10 m cells. Evidence layers carry a fixed mass; a per-cell zone weight must not out-vote
    them four times harder just because the raster got finer."""
    shares = {}
    for cell_m, n in ((10.0, 32), (5.0, 64)):
        cmap = _map(n=n, cell_m=cell_m)
        codes = np.full(cmap.shape, ZONE_CODE["hillslope"], dtype=np.uint8)
        codes[:, : n // 4] = ZONE_CODE["settlement"]
        cmap.set_zones(codes)
        lat, lon = grid_ne_to_latlon(cmap.any_grid, 155.0, 255.0)
        layers = build_prior(cmap, lkps=[LastKnownPosition(lat, lon, sigma_m=30.0)])
        raw = layers.zone + layers.lkp
        shares[cell_m] = float(layers.lkp.sum() / raw.sum())
    assert shares[10.0] == pytest.approx(shares[5.0], rel=0.02), shares


def test_burial_cells_never_attract_prior_mass():
    cmap = _map(n=24, cell_m=10.0)
    cmap.add_burial_polygons([np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]])])
    prior = build_prior(cmap)
    assert prior.total[cmap.cannot_clear].max() <= prior.total[~cmap.cannot_clear].min()
    assert prior.total.sum() == pytest.approx(1.0, abs=1e-5)


def test_bayesian_update_never_drives_a_cell_to_zero():
    poa = np.full((8, 8), 1.0 / 64.0, dtype=np.float32)
    pod = np.zeros((8, 8), dtype=np.float32)
    pod[0, 0] = cal.POD_MAX
    post = bayesian_update(poa, pod)
    assert post.sum() == pytest.approx(1.0, abs=1e-6)
    assert post[0, 0] > 0.0 and post[0, 0] < poa[0, 0]
    assert post[0, 1] > poa[0, 1], "mass moves to the cells that were NOT searched"
    assert post[0, 1] / post[0, 0] == pytest.approx(1.0 / (1.0 - cal.POD_MAX), rel=1e-3)
    # even an impossible POD of exactly 1 leaves probability behind: "the system recommends, never closes"
    saturated = bayesian_update(poa, np.ones((8, 8), dtype=np.float32))
    assert (saturated > 0.0).all() and saturated.sum() == pytest.approx(1.0, abs=1e-6)
    assert PRIOR_FLOOR > 0.0
    pos = probability_of_success(poa, pod)
    assert pos[0, 0] == pytest.approx(poa[0, 0] * cal.POD_MAX)


# ---------------------------------------------------------------------------------------------------------
# 8. the export contract the map lane (B5) reads
# ---------------------------------------------------------------------------------------------------------
def test_export_contract(tmp_path):
    cmap = _map(n=16, cell_m=10.0)
    cmap.add_frame(_tel_over(cmap, 80.0, 80.0), SIM_RGB_4K.intrinsics(), DAY, pass_id=0)
    cmap.end_pass()
    cmap.add_burial_polygons([np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]])])

    man = export_coverage(cmap, tmp_path, stem="coverage")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["coverage.json", "coverage_body.geojson", "coverage_body.png",
                     "coverage_effective.geojson", "coverage_effective.png",
                     "coverage_limb_only.geojson", "coverage_limb_only.png"]
    assert json.loads((tmp_path / "coverage.json").read_text()) == man

    assert man["product"] == "sightline.coverage" and man["domain"] == "sim"
    assert man["grid"]["row_order"] == "image row 0 is NORTH; array row 0 is SOUTH"
    assert man["cannot_clear_label"] == CANNOT_CLEAR_LABEL
    assert "cleared" not in man["legend_note"] or "never a cleared flag" in man["legend_note"]
    assert man["bounds"]["north"] > man["bounds"]["south"] and man["bounds"]["east"] > man["bounds"]["west"]
    assert man["coordinates"][0] == [man["bounds"]["west"], man["bounds"]["north"]]
    assert [lx["layer"] for lx in man["layers"]] == ["body", "limb_only", EFFECTIVE_LAYER]
    for lx in man["layers"]:
        assert lx["stats"]["domain"] == "sim"
        assert lx["k_is_measured"] is False
        assert lx["stats"]["cannot_clear_cells"] == int(cmap.cannot_clear.sum())
        assert lx["image_url"].endswith(".png") and lx["geojson_url"].endswith(".geojson")
    assert man["passes"] == [{"pass_id": 0, "frames": 1, "cells": man["passes"][0]["cells"], "mode": "AUTO"}]

    # the PNG is north-up and transparent exactly where aerial search cannot clear
    rgba = overlay_rgba(cmap.pod("body"), cmap.cannot_clear)
    assert rgba.shape == (cmap.shape[0], cmap.shape[1], 4)
    np.testing.assert_array_equal(rgba[..., 3] == 0, np.flipud(cmap.cannot_clear))
    bright = np.unravel_index(int(np.argmax(cmap.pod("body"))), cmap.shape)
    assert rgba[cmap.shape[0] - 1 - bright[0], bright[1], 3] != 0

    gj = json.loads((tmp_path / "coverage_body.geojson").read_text())
    assert gj["type"] == "FeatureCollection" and gj["layer"] == "body"
    lons = [c[0] for f in gj["features"] for ring in f["geometry"]["coordinates"] for c in ring]
    lats = [c[1] for f in gj["features"] for ring in f["geometry"]["coordinates"] for c in ring]
    assert 76.0 < min(lons) and max(lons) < 76.3, "RFC 7946 order is [lon, lat]"
    assert 11.0 < min(lats) and max(lats) < 11.9
    assert all(f["properties"]["domain"] == "sim" for f in gj["features"])
    hatched = [f for f in gj["features"] if f["properties"].get("cannot_clear")]
    assert hatched and all(f["properties"]["label"] == CANNOT_CLEAR_LABEL for f in hatched)


def test_manifest_and_summary_never_report_a_number_without_its_domain():
    cmap = _map(n=8, cell_m=10.0)
    assert cmap.summary()["domain"] == "sim"
    assert manifest(cmap)["domain"] == "sim"
    assert all(lx["stats"]["domain"] == "sim" for lx in manifest(cmap)["layers"])
    assert cmap.searched_fraction("body") == 0.0
    empty = banded_geojson(cmap.any_grid, cmap.pod("body"), cmap.cannot_clear, layer="body")
    assert empty["type"] == "FeatureCollection"


# ---------------------------------------------------------------------------------------------------------
# 9. small helpers the accumulator depends on
# ---------------------------------------------------------------------------------------------------------
def test_conditions_are_read_off_the_telemetry_the_ingest_lane_emits():
    tel = Telemetry(0.0, 11.487, 76.145, 1100.0, 60.0, vel_ned_ms=(3.0, 4.0, 0.0),
                    weather={"rain": 0.6, "fog": 0.0}, time_of_day="2026-09-10T05:40:00")
    cond = conditions_from_telemetry(tel, band="thermal")
    assert cond.band == "thermal" and cond.time_of_day == "dawn"
    assert cond.local_hour == pytest.approx(5.0 + 40.0 / 60.0)
    assert cond.weather == "heavy_rain"
    assert cond.speed_ms == pytest.approx(5.0)
    assert cond.blur_px(0.05) == pytest.approx(5.0 * (1 / 500) / 0.05)
    assert [time_of_day_label(h) for h in (6.0, 12.0, 18.0, 23.0, None)] == \
        ["dawn", "day", "dusk", "night", "day"]


def test_cell_centres_and_shape_helpers():
    cmap = _map(n=12, cell_m=10.0)
    n1, e1 = cell_centres_m(cmap.any_grid)
    assert n1[0] == pytest.approx(5.0) and n1[-1] == pytest.approx(115.0)
    np.testing.assert_allclose(n1, e1)
    nn, ee = cmap.cell_centres()
    assert nn.shape == ee.shape == cmap.shape
    assert cmap.presentations == ("body", "limb_only")
    with pytest.raises(ValueError):
        cmap.set_zones(np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        cmap.set_cannot_clear(np.zeros((3, 3), dtype=bool))
