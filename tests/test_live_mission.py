"""The live demo loop (`sightline.mission.live`) and the F3 takeover state machine.

Four things are checked, and every one of them is written so that it CAN fail:

1. **The live path and the offline path produce the same records from the same frames.** They share
   `sightline.pipeline.FramePipeline`, and this is the test that keeps it that way. `test_the_comparison_can_fail`
   sabotages one path and proves the comparison notices, so a green result means something.
2. **R10: no code path deletes a record or marks anything done.** Source scan of the whole mission lane plus a
   behavioural check that the record count never falls while a run is in flight, plus a planted violation to
   prove the scanner fires.
3. **F3.** The state machine table from SOLUTION_DOC §5.2, the immediacy of takeover, the "sticks centred
   ≥ 1 s" rule on hand-back, and the mode being visible in the telemetry and on the map.
4. **The map hop.** A record pushed by the live runner's own `C2Client` reaches a WebSocket client on the
   real server, and the arrival is what the latency number is measured from.
"""

from __future__ import annotations

import csv
import json
import math
import socket
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from sightline.api.mission_feed import MissionState
from sightline.geo import ChainConfig, project_pixel_ne
from sightline.mission import live as livemod
from sightline.mission.pattern import MIN_HITS, MIN_HITS_WINDOW_S, cadence_verdict
from sightline.mission.takeover import (MODES, ControlInput, KeyboardSource, NullSource, TakeoverMachine,
                                        VehicleAuthority)
from sightline.pipeline import (FramePipeline, nadir_gimbal_quat, record_fingerprint,
                                to_geo_gimbal)
from sightline.schemas import Intrinsics, Record, Telemetry
from sightline.track.config import TrackerConfig

CAL = json.loads((REPO / "data/scene/camera_survey.json").read_text())
F_PX, CX, CY = float(CAL["f_px"]), float(CAL["cx"]), float(CAL["cy"])
HFOV = float(CAL["hfov_deg"])
W_PX, H_PX = 3840, 2160
HOME_LAT, HOME_LON, GROUND_ASL = 10.05, 76.42, 1060.0


# --- a synthetic capture run, with a cadence that can actually confirm a track -------------------------------
def _pixel_solver(tel: Telemetry, intr: Intrinsics):
    """Invert the REAL geolocation chain so a fixture box lands where a real survivor would.

    Probing `project_pixel_ne` at three pixels and inverting the resulting affine map means the fixture
    cannot silently disagree with whatever convention the geo lane uses - if that lane changes its axes, the
    fixture follows it instead of quietly producing boxes over empty ground.
    """
    cfg = ChainConfig()
    tg = to_geo_gimbal(tel)          # the same conversion FramePipeline makes; see pipeline.to_geo_gimbal
    f0, n0, e0 = project_pixel_ne(CX, CY, tg, intr, cfg)
    _, n1, e1 = project_pixel_ne(CX + 100.0, CY, tg, intr, cfg)
    _, n2, e2 = project_pixel_ne(CX, CY + 100.0, tg, intr, cfg)
    assert f0.valid, "the fixture's own nadir ray must project"
    a, b = (n1 - n0) / 100.0, (n2 - n0) / 100.0
    c, d = (e1 - e0) / 100.0, (e2 - e0) / 100.0
    det = a * d - b * c
    assert abs(det) > 1e-9, "degenerate pixel->ground map"

    def solve(dn: float, de: float) -> tuple[float, float]:
        rn, re = dn - n0, de - e0
        du = (d * rn - b * re) / det
        dv = (-c * rn + a * re) / det
        return CX + du, CY + dv

    return solve


def make_capture_run(root: Path, *, n_frames: int = 42, alt: float = 45.0, speed: float = 12.0,
                     shutter: float = 9.8, clip_id: str = "fixture", box_px: float = 62.0,
                     survivors: list[tuple[float, float]] | None = None) -> Path:
    """Write telemetry.csv + labels/*.json for a straight north leg over a handful of survivors.

    The cadence is derived from the tracker's own confirmation gate, not chosen: `assert` below refuses to
    build a fixture that could not produce a track, which is exactly the 2026-09-11 defect.
    """
    from sightline.common.geodesy import offset_ne

    verdict = cadence_verdict(alt, speed, shutter, HFOV, W_PX, H_PX)
    assert verdict.ok, f"the fixture's own cadence cannot confirm a track: {verdict.reasons}"

    survivors = survivors or [(60.0, -12.0), (150.0, 9.0), (240.0, -4.0), (300.0, 15.0)]
    (root / "labels").mkdir(parents=True, exist_ok=True)
    dt = shutter / speed
    t0 = 1_780_000_000.0
    intr = Intrinsics(width_px=W_PX, height_px=H_PX, fx=F_PX, fy=F_PX, cx=CX, cy=CY, source="calibration")
    rows = []
    for k in range(n_frames):
        north, east = k * shutter, 0.0
        lat, lon = offset_ne(HOME_LAT, HOME_LON, north, east)
        tel = Telemetry(t_utc=t0 + k * dt, lat=lat, lon=lon, alt_msl_m=GROUND_ASL + alt, agl_m=alt,
                        q_body=(1.0, 0.0, 0.0, 0.0), q_gimbal=livemod.NADIR_Q,
                        gimbal_is_earth_referenced=True, mode="AUTO", clip_id=clip_id, frame_idx=k,
                        flood_level_asl_m=GROUND_ASL)
        solve = _pixel_solver(tel, intr)
        labels = []
        for i, (sn, se) in enumerate(survivors):
            u, v = solve(sn - north, se - east)
            if not (box_px < u < W_PX - box_px and box_px < v < H_PX - box_px):
                continue
            labels.append({
                "actor_id": i, "name": f"Human_{i:03d}", "cls": "human",
                "bbox_px": [round(u - box_px / 2, 2), round(v - box_px / 2, 2),
                            round(u + box_px / 2, 2), round(v + box_px / 2, 2)],
                "visible_px": int(box_px * box_px * 0.4), "size_px": box_px,
                "pose": "prone" if i % 2 else "standing",
                "submersion": "dry", "occlusion": 0, "zone": "settlement", "group": 0,
                "aerially_detectable": True,
            })
        (root / "labels" / f"{clip_id}_{k:05d}.json").write_text(json.dumps(labels, indent=1),
                                                                 encoding="utf-8")
        rows.append([k, tel.t_utc, clip_id, round(east, 2), round(north, 2), round(tel.alt_msl_m, 2),
                     round(alt, 2), lat, lon, 1.0, 0.0, 0.0, 0.0, -90.0, round(HFOV, 3), W_PX, H_PX,
                     round(alt / F_PX * 100.0, 4), "AUTO", round(GROUND_ASL, 3), len(labels), speed])
    with (root / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(livemod.TELEMETRY_COLUMNS)
        w.writerows(rows)
    return root


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return make_capture_run(tmp_path_factory.mktemp("live_clip"))


def _live_args(clip_dir: Path, out: Path, **over):
    a = livemod.build_parser().parse_args(
        ["--replay", str(clip_dir), "--out", str(out), "--no-c2", "--no-probe", "--pace", "0",
         "--detector", "truth", "--no-images"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _run_live(clip_dir: Path, out: Path, **over) -> tuple[int, livemod.LiveMission]:
    m = livemod.LiveMission(_live_args(clip_dir, out, **over))
    return m.run(), m


# --- the gimbal frame: the defect that made every survey produce zero tracks ---------------------------------
def test_the_nadir_quaternion_is_the_contractual_one():
    """`Telemetry.q_gimbal` is frozen as optical->NED, so nadir is the IDENTITY quaternion and the schema's
    own helper reports -90 for it. The hand-rolled `(0.7071, 0, -0.7071, 0)` reads back as -180."""
    from sightline.ingest.spec import gimbal_quat_from_euler

    q = nadir_gimbal_quat()
    assert q == pytest.approx(gimbal_quat_from_euler(0.0, -90.0, 0.0))
    t = Telemetry(t_utc=0.0, lat=HOME_LAT, lon=HOME_LON, alt_msl_m=1105.0, agl_m=45.0, q_gimbal=q)
    assert t.gimbal_pitch_deg() == pytest.approx(-90.0)
    bad = Telemetry(t_utc=0.0, lat=HOME_LAT, lon=HOME_LON, alt_msl_m=1105.0, agl_m=45.0,
                    q_gimbal=(0.70710678, 0.0, -0.70710678, 0.0))
    assert abs(bad.gimbal_pitch_deg()) > 179.0, "the old literal is supposed to be the WRONG one"


def test_geo_and_tracker_agree_on_where_a_pixel_lands():
    """THE cross-lane check. `sightline.geo` reads `q_gimbal` as camera-FRD->NED and `sightline.track` reads
    it as optical->NED; `pipeline.to_geo_gimbal` is the single place that reconciles them. Before this,
    the tracker believed a nadir camera was looking due north, its telemetry homography was ~identity
    against 555 px of real image motion per frame, and a survey produced detections, geolocations and then
    ZERO tracks - which is exactly what the 285-frame run of 2026-09-11 did.
    """
    from sightline.track.geometry import project_ground_point

    intr = Intrinsics(width_px=W_PX, height_px=H_PX, fx=F_PX, fy=F_PX, cx=CX, cy=CY)
    tel = Telemetry(t_utc=0.0, lat=HOME_LAT, lon=HOME_LON, alt_msl_m=1105.0, agl_m=45.0,
                    q_gimbal=nadir_gimbal_quat(), gimbal_is_earth_referenced=True)
    for du, dv in ((100.0, 0.0), (0.0, 100.0), (-250.0, 400.0), (600.0, -300.0)):
        fix, north, east = project_pixel_ne(CX + du, CY + dv, to_geo_gimbal(tel), intr, ChainConfig())
        assert fix.valid
        back = project_ground_point(tel, intr, north, east, tel.agl_m)
        assert back is not None, f"the tracker cannot see the point the geo lane projected ({du}, {dv})"
        assert back[0] == pytest.approx(CX + du, abs=0.01), f"u disagrees at ({du}, {dv})"
        assert back[1] == pytest.approx(CY + dv, abs=0.01), f"v disagrees at ({du}, {dv})"


def test_the_tracker_can_follow_a_survey_frame_step():
    """The telemetry homography must move the image by the distance the drone actually flew. 9.8 m at 45 m
    AGL with f = 2548.7 px is 555 px; the broken convention produced 42."""
    from sightline.track.geometry import telemetry_affine

    from sightline.common.geodesy import offset_ne

    intr = Intrinsics(width_px=W_PX, height_px=H_PX, fx=F_PX, fy=F_PX, cx=CX, cy=CY)
    step_m = 9.8
    made = []
    for north in (0.0, step_m):
        lat, lon = offset_ne(HOME_LAT, HOME_LON, north, 0.0)
        made.append(Telemetry(t_utc=north, lat=lat, lon=lon, alt_msl_m=1105.0, agl_m=45.0,
                              q_gimbal=nadir_gimbal_quat(), gimbal_is_earth_referenced=True))
    fit = telemetry_affine(made[0], made[1], intr)
    assert fit is not None
    expect_px = step_m / 45.0 * F_PX
    moved = math.hypot(float(fit.affine[0, 2]), float(fit.affine[1, 2]))
    assert moved == pytest.approx(expect_px, rel=0.02), (
        f"the telemetry homography moved the image {moved:.0f} px for a {step_m} m step; "
        f"the geometry says {expect_px:.0f} px")
    assert fit.residual_px < 1.0, f"a nadir step should reduce to an affine exactly, residual {fit.residual_px}"


def test_camera_motion_compensation_is_not_optional_for_a_survey(clip: Path, tmp_path: Path):
    """Turning CMC off must produce ZERO tracks on this cadence. If this ever passes with tracks, the
    survey geometry has changed and the 555-px-per-frame reasoning needs redoing."""
    from sightline.pipeline import run as offline_run

    off = offline_run(clip, tmp_path / "nocmc", cmc_enabled=False)
    on = offline_run(clip, tmp_path / "cmc", cmc_enabled=True)
    assert off["stages"]["detections"] == on["stages"]["detections"] > 0
    assert off["stages"]["tracks"] == 0, (
        "CMC-off produced tracks; the image motion per frame must have changed, so the default and the "
        "comment in FramePipeline need re-deriving")
    assert on["stages"]["tracks"] >= 1


# --- 0. the fixture is worth testing against ----------------------------------------------------------------
def test_fixture_produces_a_real_track_and_records(clip: Path, tmp_path: Path):
    """A fixture that yields no records would make every comparison below vacuously true."""
    from sightline.pipeline import run as offline_run

    m = offline_run(clip, tmp_path / "off0")
    assert m["stages"]["detections"] > 0, "fixture emitted no boxes"
    assert m["stages"]["tracks"] >= 1, f"fixture confirmed no track: {m['stages']}"
    assert m["stages"]["ranked"] >= 1, f"fixture produced no records: {m['stages']}"


# --- 1. the live path and the offline path may not drift ----------------------------------------------------
def test_live_and_offline_agree(clip: Path, tmp_path: Path):
    """THE anti-drift check: same frames in, same records out, on every field a commander would act on."""
    from sightline.pipeline import run as offline_run

    rc, mission = _run_live(clip, tmp_path / "live")
    assert rc == 0
    live_records = mission.pipe.close()
    assert live_records, "the live path produced no records at all"

    offline_run(clip, tmp_path / "off")
    offline = json.loads((tmp_path / "off" / "records.geojson").read_text())["features"]
    diff = livemod.compare_record_sets(live_records, offline)
    assert diff["match"], "live and offline disagree:\n  " + "\n  ".join(diff["differences"][:12])
    assert len(live_records) == len(offline) >= 1


def test_the_comparison_can_fail(clip: Path, tmp_path: Path):
    """Sabotage one path and the comparison must notice. Without this, a green result proves nothing."""
    from sightline.pipeline import run as offline_run

    _, sabotaged = _run_live(clip, tmp_path / "live_sab", cmc=False)
    offline_run(clip, tmp_path / "off_sab")
    offline = json.loads((tmp_path / "off_sab" / "records.geojson").read_text())["features"]
    diff = livemod.compare_record_sets(sabotaged.pipe.close(), offline)
    assert not diff["match"], (
        "a live run with a DIFFERENT tracker configuration compared equal to the offline run - the "
        "comparison is not actually comparing anything")


def test_live_run_is_itself_a_replayable_capture_run(clip: Path, tmp_path: Path):
    """`--verify` replays the run's own written frames and must agree. This is the shipped self-check."""
    rc, mission = _run_live(clip, tmp_path / "live_v", verify=True)
    assert rc == 0, "the shipped --verify self-check failed"
    v = json.loads((mission.out / "verify.json").read_text())
    assert v["match"], v["differences"][:10]
    assert v["live_records"] == v["offline_records"] >= 1


def test_records_appear_before_the_run_ends(clip: Path, tmp_path: Path):
    """The whole point: a record exists while frames are still arriving, not only at close()."""
    m = livemod.LiveMission(_live_args(clip, tmp_path / "live_early"))
    m.open_telemetry()
    first_record_frame, total = None, 0
    for fr in m.replay_frames():
        res = m.pipe.process(frame_idx=fr.frame_idx, telemetry=fr.telemetry, intrinsics=fr.intrinsics,
                             labels=fr.labels, capture_ms=fr.capture_ms)
        m.write_frame(fr, res)
        total += 1
        if res.records and first_record_frame is None:
            first_record_frame = fr.frame_idx
    m._tf.close()
    assert first_record_frame is not None, "no record was produced during the run at all"
    assert first_record_frame < total - 1, (
        f"the first record only appeared on the last frame ({first_record_frame} of {total}); "
        "that is a batch pipeline wearing a live pipeline's clothes")


def test_stage_latency_is_measured_and_attributed(clip: Path, tmp_path: Path):
    from sightline.pipeline import latency_report

    _, m = _run_live(clip, tmp_path / "live_lat")
    rep = latency_report(m.pipe.latencies, domain="sim", label="unit test")
    assert rep["domain"] == "sim", "a latency number without its domain is a bug (hard rule 5)"
    assert rep["n"] == m.frames > 0
    assert rep["dominant_stage"] in ("capture", "detect", "geo", "track", "dedup", "triage", "publish")
    for st in ("detect", "geo", "track", "dedup", "triage"):
        assert rep["stages"][st]["median_ms"] >= 0.0
    assert sum(rep["stages"][s]["median_ms"] for s in ("detect", "geo", "track", "dedup", "triage")) > 0.0


def test_triage_actually_scores_the_records(clip: Path, tmp_path: Path):
    """`rank_records(records, None)` sorts WITHOUT scoring, so every record keeps score 0.0 and every
    component its default - a "triage list" of `P(living) 0.00` cards in arbitrary order, which is exactly
    what the first live run put on the map. The context must always be built."""
    _, m = _run_live(clip, tmp_path / "live_triage")
    ranked = m.pipe.close()
    assert ranked, "no records to score"
    assert all(r.score > 0.0 for r in ranked),         f"triage produced zero scores: {[round(r.score, 4) for r in ranked]}"
    assert all(r.components.p_living > 0.0 for r in ranked), "p_living is zero on every record"
    assert all(r.components.count_bonus > 1.0 for r in ranked),         "count_bonus is 1.0, so the components are defaults - score_record never ran"
    assert all(r.components.urgency_class != "unknown" for r in ranked),         "no urgency class was assigned"
    assert m.pipe.incident_t0_utc is not None


def test_triage_scores_do_not_depend_on_wall_clock(clip: Path, tmp_path: Path):
    """`now_utc` is pinned to the FRAME clock. If it were wall-clock, a replay run minutes later would
    score differently and `--verify` would fail for a reason that has nothing to do with the pipeline."""
    from sightline.pipeline import run as offline_run

    a = offline_run(clip, tmp_path / "t1", incident_hours_ago=3.0)
    time.sleep(1.1)
    b = offline_run(clip, tmp_path / "t2", incident_hours_ago=3.0)
    ra = json.loads((tmp_path / "t1" / "records.geojson").read_text())["features"]
    rb = json.loads((tmp_path / "t2" / "records.geojson").read_text())["features"]
    assert a["stages"] == b["stages"]
    sa = sorted(round(f["properties"]["score"], 9) for f in ra)
    sb = sorted(round(f["properties"]["score"], 9) for f in rb)
    assert sa == sb, "scores moved between two runs over identical frames - the clock leaked in"
    assert all(x > 0.0 for x in sa)


def test_an_older_incident_is_scored_differently(clip: Path, tmp_path: Path):
    """The survival curves must actually respond to elapsed time, or the parameter is decoration."""
    from sightline.pipeline import run as offline_run

    offline_run(clip, tmp_path / "fresh", incident_hours_ago=0.0)
    offline_run(clip, tmp_path / "old", incident_hours_ago=48.0)
    fresh = sorted(f["properties"]["score"] for f in
                   json.loads((tmp_path / "fresh" / "records.geojson").read_text())["features"])
    old = sorted(f["properties"]["score"] for f in
                 json.loads((tmp_path / "old" / "records.geojson").read_text())["features"])
    assert fresh != old, "48 hours of elapsed time changed nothing in the triage score"


# --- F2 constraints reach the thing that flies --------------------------------------------------------------
def test_the_live_runner_refuses_a_plan_above_the_ceiling(tmp_path: Path):
    """`sightline/mission/safety.py` exists because the planner that knows about geofences and the planner
    that FLIES were not connected. This is the join: the flight code must actually consult it."""
    a = livemod.build_parser().parse_args(
        ["--out", str(tmp_path / "ceiling"), "--no-c2", "--no-probe", "--dry-run",
         "--alt", "130", "--no-track-check"])
    m = livemod.LiveMission(a)
    assert m.safety is not None, "the safety check did not run at all"
    assert not m.safety.ok
    assert any(v.kind == "ceiling" for v in m.safety.violations), m.safety.summary()
    assert m.run() == 4, "the runner flew a plan that breaks the 120 m ceiling"


def test_a_legal_plan_passes_the_constraint_check(tmp_path: Path):
    a = livemod.build_parser().parse_args(
        ["--out", str(tmp_path / "legal"), "--no-c2", "--no-probe", "--dry-run", "--alt", "45"])
    m = livemod.LiveMission(a)
    assert m.safety is not None and m.safety.ok, m.safety.summary() if m.safety else "no verdict"
    assert m.safety.total_legs > 0 and m.safety.est_usable_s > 0
    assert m.run() == 0


def test_ignore_safety_still_records_the_violation(tmp_path: Path):
    """An override must never make the breach invisible; R10's spirit is that nothing is quietly concluded."""
    a = livemod.build_parser().parse_args(
        ["--out", str(tmp_path / "ign"), "--no-c2", "--no-probe", "--dry-run",
         "--alt", "130", "--no-track-check", "--ignore-safety"])
    m = livemod.LiveMission(a)
    assert m.run() == 0
    assert not m.safety.ok and any(v.kind == "ceiling" for v in m.safety.violations)


# --- 2. guardrail R10 ---------------------------------------------------------------------------------------
def test_r10_no_delete_shaped_code_in_the_live_lane():
    """Source scan of everything this lane owns. An empty result is the pass condition."""
    from sightline.triage.guardrails import scan_lane_sources, scan_source

    found = scan_lane_sources(REPO, dirs=("sightline/mission", "tools/live"))
    found += scan_source(REPO / "sightline" / "pipeline.py")
    assert not found, "R10 violations:\n" + "\n".join(f"  {v.path}:{v.line_no} {v.kind}: {v.line.strip()}"
                                                      for v in found)


def test_r10_scanner_actually_fires(tmp_path: Path):
    """Plant a violation and prove the scan above is not just returning an empty list for free."""
    from sightline.triage.guardrails import scan_source

    p = tmp_path / "planted.py"
    p.write_text("def go(records):\n    records.pop(0)\n", encoding="utf-8")
    assert scan_source(p), "the R10 scanner did not catch a planted `.pop()`"


def test_r10_record_count_never_falls_during_a_run(clip: Path, tmp_path: Path):
    """Behavioural R10: once a record exists it stays, frame after frame, for the whole flight."""
    m = livemod.LiveMission(_live_args(clip, tmp_path / "live_r10"))
    m.open_telemetry()
    seen_ids: set[str] = set()
    counts = []
    for fr in m.replay_frames():
        res = m.pipe.process(frame_idx=fr.frame_idx, telemetry=fr.telemetry, intrinsics=fr.intrinsics,
                             labels=fr.labels)
        m.write_frame(fr, res)
        ids = {r.record_id for r in res.records}
        assert seen_ids <= ids | {r.record_id for r in m.pipe.dedup.all_records()}, (
            "a record that existed on an earlier frame is gone")
        seen_ids |= ids
        counts.append(len(m.pipe.dedup.all_records()))
    m._tf.close()
    assert counts and counts == sorted(counts), f"the record count went DOWN during the run: {counts}"
    assert len(m.pipe.dedup.all_records()) >= len(seen_ids)


def test_r10_a_dismissed_record_survives_the_store(tmp_path: Path):
    """Dismissal keeps the row and demands a reason; there is no delete on the way to the map."""
    from sightline.store import GuardrailError, RecordStore

    store = RecordStore(tmp_path / "r10.db")
    rec = Record(cls="human", lat=HOME_LAT, lon=HOME_LON, status="candidate", score=0.5)
    store.put(rec)
    with pytest.raises(GuardrailError):
        store.put(Record(record_id=rec.record_id, status="dismissed", version=2))
    store.dismiss(rec.record_id, reason="operator: it is a tarpaulin", by="IC-1")
    assert store.get(rec.record_id) is not None
    assert len(store.history(rec.record_id)) >= 2
    assert not hasattr(store, "delete") and not hasattr(store, "purge")
    store.close()


# --- 3. F3: the takeover state machine ----------------------------------------------------------------------
def test_mode_vocabulary_matches_the_frozen_schema():
    import typing

    from sightline.schemas import FlightMode

    assert set(MODES) == set(typing.get_args(FlightMode))


def test_survey_cadence_constants_match_the_tracker():
    cfg = TrackerConfig()
    assert MIN_HITS == cfg.min_hits and MIN_HITS_WINDOW_S == cfg.min_hits_window_s


def test_takeover_is_immediate_on_one_poll():
    """§5.2: a stick past the deadband hands over. No debounce, no confirmation, no second poll."""
    m = TakeoverMachine(clock=lambda: 0.0)
    tr = m.poll(ControlInput(t=10.0, roll=m.deadband + 0.01, valid=True))
    assert tr is not None and m.mode == "MANUAL" and tr.reason == "stick deflection"
    assert tr.t_utc == 10.0 and len(m.transitions) == 1


def test_a_stick_inside_the_deadband_does_not_take_over():
    m = TakeoverMachine()
    assert m.poll(ControlInput(t=1.0, roll=m.deadband - 0.001, pitch=0.0, valid=True)) is None
    assert m.mode == "AUTO"


def test_takeover_button_works_with_centred_sticks():
    m = TakeoverMachine()
    tr = m.poll(ControlInput(t=1.0, takeover=True, valid=True))
    assert tr is not None and m.mode == "MANUAL" and tr.reason == "TAKEOVER button"


def test_resume_needs_the_sticks_centred_for_a_second():
    """§5.2: "RESUME button (and sticks centred >= 1 s)". Pressing it early must be refused AND logged."""
    m = TakeoverMachine(centred_s=1.0)
    m.poll(ControlInput(t=0.0, roll=0.9, valid=True))
    assert m.mode == "MANUAL"
    m.poll(ControlInput(t=0.5, roll=0.9, valid=True))                       # still deflected
    assert m.poll(ControlInput(t=0.6, roll=0.9, resume=True, valid=True)) is None
    m.poll(ControlInput(t=1.0, valid=True))                                 # sticks centred; clock starts
    assert m.poll(ControlInput(t=1.5, resume=True, valid=True)) is None     # only 0.5 s
    assert m.mode == "MANUAL"
    tr = m.poll(ControlInput(t=2.1, resume=True, valid=True))               # 1.1 s
    assert tr is not None and m.mode == "AUTO" and m.resume_pending
    assert len(m.refusals) == 2, m.refusals


def test_resume_sets_resume_pending_so_the_mission_replans_instead_of_restarting():
    m = TakeoverMachine()
    m.poll(ControlInput(t=0.0, takeover=True, valid=True))
    m.poll(ControlInput(t=1.0, valid=True))
    m.poll(ControlInput(t=2.5, resume=True, valid=True))
    assert m.mode == "AUTO" and m.resume_pending is True


def test_hold_and_back_to_auto():
    m = TakeoverMachine()
    assert m.poll(ControlInput(t=0.0, hold=True, valid=True)).to_mode == "HOLD"
    m.poll(ControlInput(t=0.1, valid=True))
    assert m.poll(ControlInput(t=1.5, resume=True, valid=True)).to_mode == "AUTO"


@pytest.mark.parametrize("start", ["AUTO", "MANUAL", "HOLD", "RTL"])
def test_rtl_is_available_from_every_state(start: str):
    """§7 step 3: "HOLD and RTL are always available"."""
    m = TakeoverMachine(initial=start)
    tr = m.poll(ControlInput(t=1.0, rtl=True, valid=True))
    assert m.mode == "RTL"
    if start != "RTL":
        assert tr is not None and tr.reason == "RTL button"


def test_rtl_is_not_a_trap():
    """A mis-pressed RTL must be recoverable, or the operator has lost the aircraft to a button."""
    m = TakeoverMachine(initial="RTL")
    assert m.poll(ControlInput(t=1.0, roll=0.9, valid=True)).to_mode == "MANUAL"


def test_no_controller_means_no_transitions():
    """An absent gamepad must not read as a pilot. This is how a demo hovers for ever in MANUAL."""
    m = TakeoverMachine()
    src = NullSource()
    for i in range(50):
        assert m.poll(src.read(float(i))) is None
    assert m.mode == "AUTO" and not m.transitions and m.polls_valid == 0


def test_mission_forced_transitions_are_logged_too():
    """Pattern complete / low battery / geofence are mission decisions and must show in the same log."""
    m = TakeoverMachine()
    m.force("RTL", "pattern complete", t=99.0)
    assert m.mode == "RTL" and m.transitions[-1].reason == "pattern complete"
    assert m.transitions[-1].t_utc == 99.0


def test_every_transition_is_timestamped_and_serialisable():
    m = TakeoverMachine()
    m.poll(ControlInput(t=1.0, takeover=True, valid=True))
    m.poll(ControlInput(t=2.0, valid=True))
    m.poll(ControlInput(t=3.5, resume=True, valid=True))
    rows = [t.as_dict() for t in m.transitions]
    assert json.loads(json.dumps(rows)) == rows
    assert [r["t_utc"] for r in rows] == sorted(r["t_utc"] for r in rows)
    assert all(r["from"] in MODES and r["to"] in MODES and r["reason"] for r in rows)


def test_seconds_by_mode_add_up():
    """Coverage has to be attributable to AUTO vs MANUAL (§5.2), which needs the time in each."""
    m = TakeoverMachine()
    m.poll(ControlInput(t=0.0, valid=True))
    m.poll(ControlInput(t=10.0, takeover=True, valid=True))     # AUTO 0..10
    m.poll(ControlInput(t=11.0, valid=True))
    m.poll(ControlInput(t=14.0, resume=True, valid=True))       # MANUAL 10..14
    assert m.seconds_in("AUTO", now=20.0) == pytest.approx(10.0 + 6.0)
    assert m.seconds_in("MANUAL", now=20.0) == pytest.approx(4.0)


class _FakeClient:
    """Records the RPC calls the authority makes, in order, with timestamps."""

    def __init__(self):
        self.calls: list[tuple[float, str]] = []

    def enableApiControl(self, on, vehicle=""):          # noqa: N802 - AirSim's own spelling
        self.calls.append((time.perf_counter(), f"enableApiControl({on})"))

    def hoverAsync(self):                                 # noqa: N802
        self.calls.append((time.perf_counter(), "hover"))


def test_handover_releases_the_vehicle_on_the_same_transition():
    """The pilot is already moving the sticks: `enableApiControl(False)` must happen now, not next loop."""
    c = _FakeClient()
    m = TakeoverMachine()
    auth = VehicleAuthority(c, hover_fn=c.hoverAsync)
    tr = m.poll(ControlInput(t=1.0, roll=0.9, valid=True))
    out = auth.apply(tr)
    assert [n for _, n in c.calls] == ["enableApiControl(False)"]
    assert out["did"] == ["api_control=False"] and not out["errors"]


def test_handback_takes_the_vehicle_back():
    c = _FakeClient()
    m = TakeoverMachine()
    auth = VehicleAuthority(c, hover_fn=c.hoverAsync)
    auth.apply(m.poll(ControlInput(t=1.0, roll=0.9, valid=True)))
    m.poll(ControlInput(t=2.0, valid=True))
    auth.apply(m.poll(ControlInput(t=3.5, resume=True, valid=True)))
    assert [n for _, n in c.calls] == ["enableApiControl(False)", "enableApiControl(True)"]


def test_authority_survives_a_failing_rpc():
    """A dead link must not take the state machine down with it; the failure is recorded instead."""
    class Broken:
        def enableApiControl(self, on, vehicle=""):       # noqa: N802
            raise RuntimeError("rpc down")

    auth = VehicleAuthority(Broken())
    m = TakeoverMachine()
    out = auth.apply(m.poll(ControlInput(t=1.0, roll=0.9, valid=True)))
    assert out["did"] == [] and out["errors"] and m.mode == "MANUAL"


# --- the gamepad, against the physical device -------------------------------------------------------------
def _pad_or_skip():
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        pytest.skip("no physical gamepad attached")


def test_an_untouched_pad_does_not_take_over():
    """The regression for the bug the real hardware caught: `pitch` was mapped to a TRIGGER, which rests at
    -1.0, so opening the source reported full deflection and took the aircraft into MANUAL with nobody
    touching it. Everything about it looked right until a physical pad was read."""
    from sightline.mission.takeover import PygameGamepadSource

    _pad_or_skip()
    src = PygameGamepadSource(0)
    try:
        m = TakeoverMachine()
        for i in range(40):
            m.poll(src.read(float(i) * 0.02))
        assert m.polls_valid == 40, "the pad was not readable at all"
        assert m.mode == "AUTO" and not m.transitions, (
            f"an untouched pad took over: {[str(t) for t in m.transitions]}, "
            f"rest axes {src.rest_axes}")
    finally:
        src.close()


def test_a_trigger_mapped_as_a_stick_is_refused_not_flown():
    """The guard that turns that bug into a named failure instead of a phantom pilot."""
    from sightline.mission.takeover import PygameGamepadSource

    _pad_or_skip()
    with pytest.raises(RuntimeError, match="TRIGGER, not a stick"):
        PygameGamepadSource(0, axes={"roll": 3, "pitch": 4, "yaw": 0, "throttle": 1})


def test_the_pad_source_says_what_it_has_and_has_not_proven():
    from sightline.mission.takeover import PygameGamepadSource

    _pad_or_skip()
    src = PygameGamepadSource(0)
    try:
        d = src.describe()
        assert d["verified_against_hardware"] is True          # axes were read from a real pad
        assert d["buttons_verified_against_hardware"] is False  # nobody has pressed them
        assert len(d["rest_axes"]) >= 4
    finally:
        src.close()


def test_keyboard_fallback_reports_itself_as_untested_hardware():
    """The report has to be able to say plainly that no physical pad was involved."""
    kb = KeyboardSource()
    d = kb.describe()
    assert d["source"] == "keyboard" and d["verified_against_hardware"] is False
    assert "no physical gamepad" in d["note"].lower()


# --- mode visibility: telemetry and the map -----------------------------------------------------------------
def test_mode_is_on_the_telemetry_and_on_the_map():
    state = MissionState()
    state.intrinsics = Intrinsics(width_px=W_PX, height_px=H_PX, fx=F_PX, fy=F_PX, cx=CX, cy=CY)
    for i, mode in enumerate(("AUTO", "AUTO", "MANUAL", "MANUAL", "AUTO")):
        state.update_pose(Telemetry(t_utc=1000.0 + i, lat=HOME_LAT + i * 1e-4, lon=HOME_LON,
                                    alt_msl_m=GROUND_ASL + 45, agl_m=45.0, q_gimbal=livemod.NADIR_Q,
                                    mode=mode, clip_id="c", frame_idx=i))
    gj = state.as_geojson()
    assert gj["drone"]["properties"]["mode"] == "AUTO"
    modes = [f["properties"]["mode"] for f in gj["track"]["features"]]
    assert "MANUAL" in modes and "AUTO" in modes, f"the track was not split by mode: {modes}"


def test_live_run_writes_the_mode_into_its_telemetry_csv(clip: Path, tmp_path: Path):
    _, m = _run_live(clip, tmp_path / "live_mode")
    rows = list(csv.DictReader((m.out / "telemetry.csv").open(newline="", encoding="utf-8")))
    assert rows and all(r["mode"] in MODES for r in rows)
    assert (m.out / "mode_log.json").exists()


def test_detector_provenance_is_stamped_everywhere(clip: Path, tmp_path: Path):
    """A truth number must be unmistakable for a detection number, in every artefact the run leaves."""
    _, m = _run_live(clip, tmp_path / "live_prov")
    card = json.loads((m.out / "data_card.json").read_text())
    assert card["detector"] == "truth" and card["domain"] == "sim"
    assert "not detection results" in card["detector_note"].lower()
    assert card["source"] == "replay", "a replay-driven run must not describe itself as a flight"
    assert card["latency"]["domain"] == "sim"
    assert card["noise_injected"] is False


# --- 4. the map hop, on a real server -----------------------------------------------------------------------
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def c2_server(tmp_path: Path):
    """A real uvicorn process-in-a-thread, because `C2Client` and `MapArrivalProbe` speak real sockets."""
    import uvicorn

    from sightline.api.app import create_app
    from sightline.store import RecordStore

    store = RecordStore(tmp_path / "c2.db")
    app = create_app(store, mission=MissionState(), repo_root=REPO, serve_static=False)
    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            livemod.C2Client.probe(url)
            break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("the C2 test server never came up")
    yield url, store
    server.should_exit = True
    th.join(timeout=5)
    store.close()


def test_a_pushed_record_reaches_a_map_client_and_the_latency_is_measured(c2_server, tmp_path: Path):
    """The demo's actual claim: the runner enqueues a record and a map client sees it, in milliseconds."""
    url, store = c2_server
    probe = livemod.MapArrivalProbe(url).start()
    assert not probe.error, probe.error
    c2 = livemod.C2Client(url, outbox_dir=tmp_path / "ob")
    try:
        rec = Record(cls="human", lat=HOME_LAT, lon=HOME_LON, status="candidate", score=0.7,
                     n_observations=3, source={"detector": "truth", "domain": "sim"})
        t0 = time.perf_counter()
        c2.push_records([rec])
        arrived = None
        while time.perf_counter() - t0 < 15.0 and arrived is None:
            arrived = probe.arrival_of(rec.record_id, rec.version)
            time.sleep(0.005)
        assert arrived is not None, "the record never reached the map's WebSocket"
        ms = (arrived - t0) * 1e3
        assert 0.0 < ms < 15_000.0
        # `not_before` must refuse an arrival that predates the capture, or the reported latency goes
        # NEGATIVE - which is exactly what the first measured run printed ("median -1010.0 ms").
        assert probe.arrival_of(rec.record_id, rec.version, not_before=arrived + 1.0) is None
        assert store.get(rec.record_id) is not None, "the record is on the map but not in the log"
    finally:
        c2.close()
        probe.stop()


def test_reported_map_latency_is_never_negative(clip: Path, tmp_path: Path, c2_server):
    """A latency measurement that can go negative is worse than none: it was reported as a headline number
    on the first live run. Every resolved sample must be >= 0 and the unresolved ones must be COUNTED,
    not quietly dropped."""
    url, _ = c2_server
    rc, m = _run_live(clip, tmp_path / "live_lat2", no_c2=False, c2=url, no_probe=False)
    assert rc == 0
    assert all(x >= 0.0 for x in m.map_latency_ms),         f"negative capture->map latency: {[x for x in m.map_latency_ms if x < 0][:5]}"
    card = json.loads((m.out / "data_card.json").read_text())
    e2e = card["capture_to_map_ms"]
    assert e2e is not None and e2e["n"] > 0, "nothing reached the map at all"
    assert e2e["min_ms"] >= 0.0 and e2e["median_ms"] >= 0.0
    assert "unresolved_pushes" in e2e


def test_the_outbox_drains_and_the_server_never_rejects_a_push(clip: Path, tmp_path: Path, c2_server):
    """Every record the live loop pushes must actually reach the map.

    The first measured run enqueued 1019 uploads for 32 records and delivered FOUR: the change detector
    fired on the decaying triage score, so the same version was re-sent with different content, the C2's
    idempotent upsert rejected it 409, and the uploader retried that one job for ever with the whole queue
    behind it. A green result here means the queue emptied and nothing was refused.
    """
    url, store = c2_server
    rc, m = _run_live(clip, tmp_path / "live_ob", no_c2=False, c2=url, no_probe=False)
    assert rc == 0
    # Read the card, not the live client: `run()` closes the outbox (and its SQLite handle) on the way out.
    st = json.loads((m.out / "data_card.json").read_text())["c2"]
    assert st.get("failures", 0) == 0, f"the C2 refused a push: {st.get('last_error')!r}"
    assert st.get("depth", 0) == 0, f"the outbox did not drain: {st}"
    assert st["records_pushed"] == st["sent"], f"enqueued {st['records_pushed']} but sent {st['sent']}"
    n_records = len(m.pipe.dedup.all_records())
    assert n_records > 0
    assert st["records_pushed"] <= 6 * n_records, (
        f"{st['records_pushed']} uploads for {n_records} records - the change detector is firing on "
        f"something that varies every frame")
    assert store.stats()["records"] == n_records


def test_a_record_is_not_pushed_twice_at_the_same_version(tmp_path: Path, c2_server):
    """The guard that makes a 409 impossible even if a future change detector misbehaves."""
    url, _ = c2_server
    c2 = livemod.C2Client(url, outbox_dir=tmp_path / "ob3")
    try:
        rec = Record(cls="human", lat=HOME_LAT, lon=HOME_LON, score=0.4, version=1)
        c2.push_records([rec])
        rec.score = 0.9                      # same version, different content: a stale write
        c2.push_records([rec])
        assert c2.records_pushed == 1 and c2.records_skipped == 1
        rec.version = 2
        c2.push_records([rec])
        assert c2.records_pushed == 2
        assert c2.drain()["failures"] == 0
    finally:
        c2.close()


def test_the_map_probe_can_fail(c2_server):
    """Point the probe at a port with nothing on it: it must report the failure, not claim success."""
    url, _ = c2_server
    dead = livemod.MapArrivalProbe(f"http://127.0.0.1:{_free_port()}").start()
    dead.connected.wait(timeout=5)
    assert dead.error, "the arrival probe reported no error while connecting to nothing"
    dead.stop()


def test_pose_push_puts_the_mode_on_the_map(c2_server, tmp_path: Path):
    """§7 step 3: the mode chip flips to MANUAL. This is the hop that makes that true."""
    import httpx

    url, _ = c2_server
    c2 = livemod.C2Client(url, outbox_dir=tmp_path / "ob2")
    try:
        tel = Telemetry(t_utc=time.time(), lat=HOME_LAT, lon=HOME_LON, alt_msl_m=GROUND_ASL + 45,
                        agl_m=45.0, q_gimbal=livemod.NADIR_Q, mode="MANUAL", clip_id="c", frame_idx=1)
        c2.push_pose(tel)
        assert c2.pose_errors == 0
        m = httpx.get(url + "/api/mission", timeout=5).json()
        assert m["drone"]["properties"]["mode"] == "MANUAL"
    finally:
        c2.close()


# --- the fingerprint that decides what gets pushed ----------------------------------------------------------
def test_a_version_bump_alone_is_not_a_record_change():
    """`Deduplicator` bumps every record's version on every re-cluster; the map must not be told each time."""
    a = Record(cls="human", lat=1.0, lon=2.0, n_observations=3, version=1)
    b = Record(record_id=a.record_id, cls="human", lat=1.0, lon=2.0, n_observations=3, version=57)
    assert record_fingerprint(a) == record_fingerprint(b)
    c = Record(record_id=a.record_id, cls="human", lat=1.0, lon=2.0, n_observations=4, version=1)
    assert record_fingerprint(a) != record_fingerprint(c), "a new observation IS a change"


def test_the_decaying_score_is_not_a_record_change():
    """The triage score falls with time since the incident. If that counted as a change, every record would
    be re-pushed on every frame - and re-pushed at the SAME version, which the C2 rejects with 409."""
    a = Record(cls="human", lat=1.0, lon=2.0, score=0.5, priority_rank=3, version=4)
    b = Record(record_id=a.record_id, cls="human", lat=1.0, lon=2.0, score=0.31, priority_rank=3, version=4)
    assert record_fingerprint(a) == record_fingerprint(b), (
        "a decaying score is being treated as new information about the survivor")
    c = Record(record_id=a.record_id, cls="human", lat=1.0, lon=2.0, score=0.5, priority_rank=7, version=4)
    assert record_fingerprint(a) != record_fingerprint(c), (
        "a rank change must reach the map: it sizes, labels and sorts the triage list by it")


def test_frame_pipeline_refuses_a_detector_it_cannot_run():
    with pytest.raises(ValueError):
        FramePipeline(detector="magic")
    fp = FramePipeline(detector="truth")
    with pytest.raises(ValueError):
        fp.process(frame_idx=0,
                   telemetry=Telemetry(t_utc=0.0, lat=0.0, lon=0.0, alt_msl_m=0.0, agl_m=45.0),
                   intrinsics=Intrinsics(W_PX, H_PX, F_PX, F_PX, CX, CY))
