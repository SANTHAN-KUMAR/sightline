"""Tests for `sightline.eval.context` — the measured terrain type under a box (SOLUTION_DOC 5.12).

Runnable offline: every geometric test builds its own synthetic scene. The last group re-measures the real
flown campaign and is skipped when `_artifacts/dataset/` is not present, so a fresh clone still goes green.

Each test is written so that it FAILS if the behaviour it protects is removed — the substitution-vs-ray test
and the vegetation test in particular both reproduce defects that were real in this module and were caught by
measuring against ground truth rather than by reading the code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sightline.eval.context import (
    CONTEXTS,
    PROJECTION_P90_M,
    ContextError,
    SceneGeometry,
    classify,
    project_pixel,
)

REPO = Path(__file__).resolve().parents[1]
F_PX, CX, CY = 2548.723, 1920.0, 1080.0


def flat_scene(ground_asl: float = 1000.0, water_asl: float = 990.0, size_m: float = 2048.0,
               grid: int = 65) -> SceneGeometry:
    """A featureless plain above the flood, so anything found is something a test put there."""
    return SceneGeometry(water_asl_m=water_asl, height=np.full((grid, grid), ground_asl, dtype=float),
                         cell_m=size_m / (grid - 1), size_m=size_m)


# --------------------------------------------------------------------------- projection

def test_image_right_is_east_and_image_down_is_south():
    """The axis convention, which was 90 degrees out in `coverage/footprint.py` until 2026-09-11."""
    e, n = project_pixel(CX + 1000.0, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                         target_asl_m=1000.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert e > 0 and abs(n) < 1e-9, "image-right must be EAST"
    e, n = project_pixel(CX, CY + 1000.0, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                         target_asl_m=1000.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert n < 0 and abs(e) < 1e-9, "image-down must be SOUTH"


def test_scale_is_the_range_to_the_target_not_to_the_ground():
    """A roof 10 m up is imaged at a smaller scale than the ground beside it."""
    kw = dict(drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    e_ground, _ = project_pixel(CX + 1000.0, CY, target_asl_m=1000.0, **kw)
    e_roof, _ = project_pixel(CX + 1000.0, CY, target_asl_m=1010.0, **kw)
    assert e_roof < e_ground
    assert e_ground / e_roof == pytest.approx(45.0 / 35.0, rel=1e-9)


def test_camera_at_or_below_the_surface_is_an_error_not_a_silent_flip():
    with pytest.raises(ContextError):
        project_pixel(CX, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1000.0,
                      target_asl_m=1000.0, f_px=F_PX, cx_px=CX, cy_px=CY)


# --------------------------------------------------------------------------- the ray, not a substitution

def test_a_tall_target_far_off_axis_resolves_onto_itself_not_the_ground_beside_it():
    """The parallax case. Projecting a corner pixel onto the TERRAIN put roof survivors up to 77 m out."""
    geom = flat_scene()
    # A building 10 m tall, sitting where the ray through a far-off-axis pixel actually meets 1010 m.
    u = CX + 1700.0
    e_roof, n_roof = project_pixel(u, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                                   target_asl_m=1010.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    geom.add(east_m=e_roof, north_m=n_roof, r_major_m=5.0, r_minor_m=4.0, top_asl_m=1010.0,
             context="structure", source="House_test")

    fix = classify([u - 20, CY - 20, u + 20, CY + 20], geom, drone_east_m=0.0, drone_north_m=0.0,
                   drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert fix.context == "structure"
    assert fix.source == "House_test"
    assert math.hypot(fix.east_m - e_roof, fix.north_m - n_roof) < 0.5
    # ... and the terrain-referenced projection would have been metres away, which is the whole point.
    e_flat, _ = project_pixel(u, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                              target_asl_m=1000.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert abs(e_flat - e_roof) > 4.0


def test_vegetation_cannot_move_the_fix_but_can_still_be_reported():
    """A crown on the slant ray is an OCCLUSION, measured elsewhere from masks — not a thing to stand on.

    Letting crowns drive the position solve dragged 26 of 140 roof survivors onto neighbouring canopy and
    moved their fixes by up to 14 m, against a scene gate that independently asserts no unoccluded survivor
    sits under a crown at all.
    """
    geom = flat_scene()
    u = CX + 1700.0
    e_roof, n_roof = project_pixel(u, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                                   target_asl_m=1010.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    geom.add(east_m=e_roof, north_m=n_roof, r_major_m=5.0, r_minor_m=4.0, top_asl_m=1010.0,
             context="structure", source="House_test")
    # A big crown, taller than the house, sitting on the ray where it passes 1025 m.
    e_c, n_c = project_pixel(u, CY, drone_east_m=0.0, drone_north_m=0.0, drone_asl_m=1045.0,
                             target_asl_m=1025.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    geom.add(east_m=e_c, north_m=n_c, r_major_m=12.0, r_minor_m=10.0, top_asl_m=1025.0,
             context="vegetation", source="Tree_test")

    fix = classify([u - 20, CY - 20, u + 20, CY + 20], geom, drone_east_m=0.0, drone_north_m=0.0,
                   drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert math.hypot(fix.east_m - e_roof, fix.north_m - n_roof) < 0.5, "a crown must not move the fix"

    # But a crown standing OVER the settled position is the terrain type there.
    under = flat_scene()
    under.add(east_m=0.0, north_m=0.0, r_major_m=12.0, r_minor_m=12.0, top_asl_m=1020.0,
              context="vegetation", source="Tree_over")
    fix2 = classify([CX - 5, CY - 5, CX + 5, CY + 5], under, drone_east_m=0.0, drone_north_m=0.0,
                    drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert fix2.context == "vegetation"


def test_a_surface_above_the_camera_is_never_hit():
    """This valley climbs to 1170 m; a hillslope crown can stand above a drone 35 m over the settlement."""
    geom = flat_scene()
    geom.add(east_m=0.0, north_m=0.0, r_major_m=50.0, r_minor_m=50.0, top_asl_m=1100.0,
             context="structure", source="Above_camera")
    fix = classify([CX - 5, CY - 5, CX + 5, CY + 5], geom, drone_east_m=0.0, drone_north_m=0.0,
                   drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert fix.context == "open_ground"


# --------------------------------------------------------------------------- surfaces

def test_flood_covers_terrain_below_it_and_nothing_above_it():
    geom = flat_scene(ground_asl=1000.0, water_asl=1010.0)
    assert geom.surface_at(0.0, 0.0).context == "water"
    dry = flat_scene(ground_asl=1000.0, water_asl=990.0)
    assert dry.surface_at(0.0, 0.0).context == "open_ground"


def test_the_topmost_surface_wins_and_ties_break_towards_structure():
    geom = flat_scene()
    geom.add(east_m=0.0, north_m=0.0, r_major_m=6.0, r_minor_m=6.0, top_asl_m=1003.0,
             context="debris", source="pile")
    assert geom.surface_at(0.0, 0.0).context == "debris"
    geom.add(east_m=0.0, north_m=0.0, r_major_m=6.0, r_minor_m=6.0, top_asl_m=1009.0,
             context="structure", source="house")
    assert geom.surface_at(0.0, 0.0).context == "structure"
    # Coplanar: the more structured class wins, without either being above the other.
    tie = flat_scene()
    tie.add(east_m=0.0, north_m=0.0, r_major_m=6.0, r_minor_m=6.0, top_asl_m=1004.0,
            context="debris", source="pile")
    tie.add(east_m=0.0, north_m=0.0, r_major_m=6.0, r_minor_m=6.0, top_asl_m=1004.1,
            context="structure", source="house")
    assert tie.surface_at(0.0, 0.0).context == "structure"


def test_footprints_are_ellipses_in_the_items_own_yawed_frame():
    geom = flat_scene()
    geom.add(east_m=0.0, north_m=0.0, r_major_m=10.0, r_minor_m=2.0, top_asl_m=1005.0,
             context="structure", source="long_house", yaw_deg=0.0)
    assert geom.surface_at(8.0, 0.0).context == "structure"   # along the major axis
    assert geom.surface_at(0.0, 8.0).context == "open_ground"  # across the minor axis
    turned = flat_scene()
    turned.add(east_m=0.0, north_m=0.0, r_major_m=10.0, r_minor_m=2.0, top_asl_m=1005.0,
               context="structure", source="long_house", yaw_deg=90.0)
    assert turned.surface_at(0.0, 8.0).context == "structure"
    assert turned.surface_at(8.0, 0.0).context == "open_ground"


def test_an_unknown_context_is_refused_at_the_boundary():
    geom = flat_scene()
    with pytest.raises(ContextError):
        geom.add(east_m=0.0, north_m=0.0, r_major_m=1.0, r_minor_m=1.0, top_asl_m=1001.0,
                 context="roof", source="x")  # 5.12's wording, but not the GtBox vocabulary


# --------------------------------------------------------------------------- ambiguity

def test_ambiguity_fires_at_a_boundary_and_not_in_the_open():
    geom = flat_scene()
    geom.add(east_m=0.0, north_m=0.0, r_major_m=4.0, r_minor_m=4.0, top_asl_m=1004.0,
             context="structure", source="small_house")
    open_fix = classify([CX - 5, CY - 5, CX + 5, CY + 5], flat_scene(), drone_east_m=500.0,
                        drone_north_m=500.0, drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert not open_fix.ambiguous and open_fix.dominant_fraction == 1.0

    # A box on the roof EDGE: the ambiguity disc straddles house and ground.
    edge = classify([CX - 5, CY - 5, CX + 5, CY + 5], geom, drone_east_m=-4.0, drone_north_m=0.0,
                    drone_asl_m=1045.0, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert edge.ambiguous
    assert {c for c, _ in edge.mix} >= {"structure", "open_ground"}


def test_the_mix_is_a_descending_distribution_that_sums_to_one():
    geom = flat_scene()
    geom.add(east_m=0.0, north_m=0.0, r_major_m=3.0, r_minor_m=3.0, top_asl_m=1004.0,
             context="structure", source="h")
    mix = geom.mix_within(0.0, 0.0, PROJECTION_P90_M)
    assert sum(f for _, f in mix) == pytest.approx(1.0)
    assert [f for _, f in mix] == sorted((f for _, f in mix), reverse=True)
    assert all(c in CONTEXTS for c, _ in mix)


# --------------------------------------------------------------------------- the real scene

@pytest.mark.skipif(not (REPO / "data" / "scene" / "actors.json").is_file(), reason="scene not generated")
def test_the_heightfield_axis_order_is_decided_by_measurement():
    """A silent transpose is exactly the defect that left the coverage footprint 90 degrees out."""
    geom = SceneGeometry.load()
    assert geom.terrain_check_m < 2.0, "heightfield disagrees with the known survivor elevations"
    assert geom.n_items > 5000, f"only {geom.n_items} placed items loaded"

    # And the check has teeth: transposing the heightfield must be caught, not absorbed.
    bad = SceneGeometry(water_asl_m=geom.water_asl_m, height=geom.height, cell_m=geom.cell_m,
                        size_m=geom.size_m)
    bad.north_major = not geom.north_major
    actors = json.loads((REPO / "data/scene/actors.json").read_text(encoding="utf-8"))["actors"]
    probes = [(a["east_m"], a["north_m"], a["base_asl_m"] - a.get("ground_offset_cm", 0.0) / 100.0)
              for a in actors
              if not (str(a.get("group") or "").startswith("roof") or "roof" in str(a.get("note") or ""))]
    wrong = float(np.median([abs(bad.terrain_asl(e, n) - z) for e, n, z in probes]))
    assert wrong > 5 * geom.terrain_check_m, (
        f"the wrong axis order is only {wrong:.2f} m off against {geom.terrain_check_m:.2f} m — "
        f"this scene cannot distinguish them, so the orientation check proves nothing")


def _campaign_fixes(pass_name: str):
    root = REPO / "_artifacts" / "dataset" / pass_name
    if not (root / "telemetry.csv").is_file():
        return None
    cam = json.loads((REPO / "data/scene/camera_survey.json").read_text(encoding="utf-8"))
    actors = {a["id"]: a for a in
              json.loads((REPO / "data/scene/actors.json").read_text(encoding="utf-8"))["actors"]}
    tel = {}
    with (root / "telemetry.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            tel[int(r["frame_idx"])] = r
    geom = SceneGeometry.load()
    out = []
    for p in sorted((root / "labels").glob("*.json")):
        idx = int(p.stem.rsplit("_", 1)[1])
        t = tel.get(idx)
        if t is None:
            continue
        for m in json.loads(p.read_text(encoding="utf-8")):
            a = actors.get(m["actor_id"])
            if a is None or m["size_px"] < 8:
                continue
            fix = classify(m["bbox_px"], geom, drone_east_m=float(t["east_m"]),
                           drone_north_m=float(t["north_m"]), drone_asl_m=float(t["alt_msl_m"]),
                           f_px=cam["f_px"], cx_px=cam["cx"], cy_px=cam["cy"])
            out.append((m, a, fix, geom))
    return out


@pytest.mark.skipif(_campaign_fixes("seed23_alt35") is None, reason="no flown campaign on disk")
def test_measured_context_agrees_with_independent_ground_truth_on_the_flown_data():
    """Two claims actors.json makes that the classifier never reads: who is on a roof, and who is in water."""
    rows = _campaign_fixes("seed23_alt35")
    assert rows, "campaign present but produced no boxes"

    roof = [(m, a, f) for m, a, f in ((m, a, f) for m, a, f, _ in rows)
            if str(a.get("group") or "").startswith("roof") or "roof" in str(a.get("note") or "")]
    assert len(roof) >= 20, f"only {len(roof)} roof observations to test with"
    hit = sum(1 for _, _, f in roof if f.context == "structure")
    assert hit / len(roof) >= 0.85, f"only {hit}/{len(roof)} roof survivors measured 'structure'"

    geom = rows[0][3]
    sub = [(m, f) for m, _, f, _ in rows if m["submersion"] in ("partial", "half", "head_only")]
    if sub:
        over = sum(1 for _, f in sub if geom.terrain_asl(f.east_m, f.north_m) < geom.water_asl_m)
        assert over == len(sub), f"only {over}/{len(sub)} submerged survivors landed over flooded terrain"


@pytest.mark.skipif(_campaign_fixes("seed23_alt35") is None, reason="no flown campaign on disk")
def test_the_measured_context_disagrees_with_the_assumed_one():
    """The finding this module exists for: `_context_for(zone, submersion)` is not what is on the ground.

    If this ever passes trivially — because someone wired the measured value back into the assumption — the
    assertion below fails and says so, rather than the disagreement quietly disappearing.
    """
    def assumed(zone: str, submersion: str) -> str:
        if submersion in ("partial", "half", "head_only"):
            return "water"
        return {"settlement": "structure", "fan": "debris", "channel": "water",
                "hillslope": "vegetation"}.get(zone, "")

    rows = _campaign_fixes("seed23_alt35")
    disagree = sum(1 for m, _, f, _ in rows if f.context != assumed(m["zone"], m["submersion"]))
    assert disagree > 0.2 * len(rows), (
        f"only {disagree}/{len(rows)} boxes disagree with the zone-derived assumption; if the assumption is "
        f"now measured, delete this test and _context_for with it")


# --------------------------------------------------------------------------- 5.12 rows

def _match_result(pred, gt, minutes=2.0):
    """A MatchResult built by hand, so the row logic is tested and not the matcher."""
    from sightline.eval.detection import MatchResult
    from sightline.eval.groundtruth import GtBox, GtFrame
    from sightline.schemas import Detection

    frames, gt_boxes, gt_frames, gt_score = {}, [], [], []
    for fidx, bbox, ctx, found in gt:
        fr = frames.setdefault(fidx, GtFrame(frame_idx=fidx, camera_east_m=0.0, camera_north_m=0.0,
                                             camera_asl_m=1045.0))
        b = GtBox(bbox_px=bbox, frame_idx=fidx, context=ctx)
        fr.boxes.append(b)
        gt_boxes.append(b)
        gt_frames.append(fr)
        gt_score.append(1.0 if found else -1.0)
    dets, p_score, p_out, p_frame = [], [], [], []
    for fidx, bbox, score, outcome in pred:
        frames.setdefault(fidx, GtFrame(frame_idx=fidx, camera_east_m=0.0, camera_north_m=0.0,
                                        camera_asl_m=1045.0))
        dets.append(Detection(bbox_px=tuple(bbox), score=score, frame_idx=fidx))
        p_score.append(score)
        p_out.append(outcome)
        p_frame.append(fidx)

    res = MatchResult(iou_thr=0.5, cls="human", minutes=minutes, n_frames=len(frames))
    res.gt_boxes, res.gt_frames = gt_boxes, gt_frames
    res.gt_match_score = np.asarray(gt_score, dtype=float)
    res.gt_match_iou = np.zeros(len(gt_boxes))
    res.gt_frame = np.asarray([f.frame_idx for f in gt_frames], dtype=int)
    res.gt_local = np.arange(len(gt_boxes))
    res.detections = dets
    res.pred_score = np.asarray(p_score, dtype=float)
    res.pred_outcome = np.asarray(p_out, dtype="<U8")
    res.pred_frame = np.asarray(p_frame, dtype=int)
    res.pred_gt = np.full(len(dets), -1, dtype=int)

    class _DS:
        pass
    ds = _DS()
    ds.frames = list(frames.values())
    return res, ds


def _scene_with_a_house_at_nadir():
    """A house directly under the camera and open water to the east, both at known pixel positions."""
    geom = flat_scene(ground_asl=1000.0, water_asl=1002.0)  # terrain drowned -> water everywhere...
    geom.add(east_m=0.0, north_m=0.0, r_major_m=8.0, r_minor_m=8.0, top_asl_m=1006.0,
             context="structure", source="House_000")       # ...except on the roof
    return geom


def test_false_positives_get_a_terrain_type_which_ground_truth_could_never_give_them():
    """The point of the module: FP/min per terrain type, for detections with no ground truth behind them."""
    from sightline.eval.context import context_rows
    from sightline.eval.slicing import make_slice

    geom = _scene_with_a_house_at_nadir()
    # One FP on the roof (nadir), one far east over water. No ground truth at all.
    res, ds = _match_result(pred=[(0, (CX - 10, CY - 10, CX + 10, CY + 10), 0.9, "fp"),
                                  (0, (CX + 1600, CY - 10, CX + 1620, CY + 10), 0.9, "fp")],
                            gt=[], minutes=2.0)
    ms = context_rows(res, ds, 0.5, make_slice("sim"), geom, f_px=F_PX, cx_px=CX, cy_px=CY)
    got = {r.slice.context: r.value for r in ms.rows if r.name.startswith("fp_per_min")}
    assert got.get("structure") == pytest.approx(0.5), got   # 1 FP / 2 min
    assert got.get("water") == pytest.approx(0.5), got
    assert "open_ground" not in got


def test_a_frame_without_a_camera_pose_is_reported_unmeasured_not_guessed():
    from sightline.eval.context import context_rows, measure_prediction_contexts
    from sightline.eval.slicing import make_slice

    geom = _scene_with_a_house_at_nadir()
    res, ds = _match_result(pred=[(0, (CX - 10, CY - 10, CX + 10, CY + 10), 0.9, "fp")], gt=[])
    for f in ds.frames:
        f.camera_east_m = f.camera_north_m = f.camera_asl_m = None
    assert list(measure_prediction_contexts(res, ds, geom, f_px=F_PX, cx_px=CX, cy_px=CY)) == [""]
    ms = context_rows(res, ds, 0.5, make_slice("sim"), geom, f_px=F_PX, cx_px=CX, cy_px=CY)
    cov = [r for r in ms.rows if r.name == "fp_context_coverage"]
    assert cov and cov[0].value == 0.0
    assert not [r for r in ms.rows if r.name.startswith("fp_per_min")]


def test_recall_and_fp_per_min_are_reported_on_the_same_terrain_axis():
    from sightline.eval.context import context_rows
    from sightline.eval.slicing import make_slice

    geom = _scene_with_a_house_at_nadir()
    res, ds = _match_result(
        pred=[(0, (CX + 1600, CY - 10, CX + 1620, CY + 10), 0.9, "fp")],
        gt=[(0, (CX - 10, CY - 10, CX + 10, CY + 10), "structure", True),
            (0, (CX - 40, CY - 40, CX - 20, CY - 20), "structure", False)],
        minutes=2.0)
    ms = context_rows(res, ds, 0.5, make_slice("sim"), geom, f_px=F_PX, cx_px=CX, cy_px=CY)
    rec = {r.slice.context: r.value for r in ms.rows if r.name.startswith("recall")}
    fp = {r.slice.context: r.value for r in ms.rows if r.name.startswith("fp_per_min")}
    assert rec["structure"] == pytest.approx(0.5), rec   # 1 of 2 roof survivors found
    assert fp["water"] == pytest.approx(0.5), fp         # the FP landed on water, not on the roof
    assert fp.get("structure", 0.0) == 0.0


def test_ground_truth_contexts_are_re_measured_so_both_halves_share_one_axis():
    from sightline.eval.context import measure_dataset_contexts

    geom = _scene_with_a_house_at_nadir()
    res, ds = _match_result(pred=[], gt=[(0, (CX - 10, CY - 10, CX + 10, CY + 10), "debris", True)])
    assert ds.frames[0].boxes[0].context == "debris"       # what the zone label asserted
    stats = measure_dataset_contexts(ds, geom, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert ds.frames[0].boxes[0].context == "structure"    # what is actually there
    assert stats == {"measured": 1, "changed": 1, "unmeasured": 0}


def test_boxes_on_a_poseless_frame_are_left_alone_rather_than_blanked():
    from sightline.eval.context import measure_dataset_contexts

    geom = _scene_with_a_house_at_nadir()
    res, ds = _match_result(pred=[], gt=[(0, (CX - 10, CY - 10, CX + 10, CY + 10), "debris", True)])
    ds.frames[0].camera_east_m = None
    stats = measure_dataset_contexts(ds, geom, f_px=F_PX, cx_px=CX, cy_px=CY)
    assert ds.frames[0].boxes[0].context == "debris", "an unmeasurable box must keep what it had"
    assert stats["unmeasured"] == 1 and stats["measured"] == 0
