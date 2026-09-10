"""Scenario data for the map, the headless check and the demo.

**Every number here is SIMULATED and labelled as such** (hard rule 5 / §5.12): each record carries
``source = {"domain": "sim", "synthetic": True, ...}`` and ``notes`` saying so, and the coverage manifest
carries ``"domain": "sim"`` on the manifest and again inside every layer's statistics.

Two halves, and they are not the same kind of thing:

* **the records are synthetic** — hand-authored cases so the triage UI, the score-component popup and the
  WebSocket can be exercised before the detection/geolocation lanes land. Nothing here measures anything.
* **the flight, the coverage raster and the planned pattern are computed by the real lanes**: the pattern
  comes from ``sightline.plan.boustrophedon_route`` (real spacing from the real camera model), the search
  quality from ``sightline.coverage.CoverageMap`` fed with the telemetry of the part of that route already
  flown, and the export from ``sightline.coverage.export_coverage`` — B6's own contract writer. The map is
  therefore reading the production format, not a mock of it. Its POD numbers are still model-driven: B6's
  ``k`` is derived, not calibrated, and ``k_is_measured`` is false in the manifest, which the page prints.

Replace the record half by pointing the API at the store the pipeline writes to.
"""

from __future__ import annotations

import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from sightline.api.mission_feed import MissionState
from sightline.common.geodesy import offset_ne
from sightline.schemas import (
    Evidence,
    Record,
    ScoreComponents,
    Telemetry,
)
from sightline.store import RecordStore

__all__ = ["ORIGIN_LAT", "ORIGIN_LON", "SCENE_JSON", "build_route", "route_frames", "seed_store",
           "seed_coverage", "seed_mission", "seed_all"]

#: docs/CONTEXT.md: OriginGeopoint of the FloodValley scenario (Chooralmala / Mundakkai).
ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 11.4870, 76.1450, 1046.007
SCENE_JSON = Path(__file__).resolve().parents[2] / "data" / "scene" / "flood_valley.json"

#: The search box, in scene-NE metres about the map centre, and the survey altitude.
SEARCH_BOX_N_M, SEARCH_BOX_E_M, SURVEY_AGL_M, SURVEY_SPEED_MS = 900.0, 700.0, 55.0, 8.0
#: How much of the planned pattern has been flown when the demo opens (§7: the sortie is in progress).
FLOWN_FRACTION = 0.46
#: Telemetry sample spacing along the track. 30 m at 55 m AGL is ~3 frames per swath width.
FRAME_STEP_M = 30.0
#: "Aerial search cannot clear" areas (§2.7), as closed lat/lon rings: the debris fan and one buried lot.
BURIAL_RINGS_LATLON: tuple[tuple[tuple[float, float], ...], ...] = (
    ((11.48430, 76.14180), (11.48430, 76.14420), (11.48620, 76.14420), (11.48620, 76.14180)),
    ((11.48870, 76.14790), (11.48870, 76.14930), (11.48960, 76.14930), (11.48960, 76.14790)),
)

_CASES: list[dict[str, Any]] = [
    dict(cls="human", urgency_class="immersed", posture="half_submerged", submersion="head_only",
         zone="channel", p_living=0.91, w_class=0.86, urgency=2.4, count=1, thermal_hot=True,
         thermal_c=31.4, motion="moving", n_obs=17, h_acc=2.9, north=-190.0, east=240.0, occl=1,
         note="head-only in the channel; thermal agreed"),
    dict(cls="human", urgency_class="trapped", posture="trapped", submersion="wet", zone="settlement",
         p_living=0.83, w_class=0.95, urgency=1.9, count=2, thermal_hot=True, thermal_c=30.1,
         motion="still", n_obs=11, h_acc=3.4, north=120.0, east=-160.0, occl=2,
         note="two bodies under a collapsed eave"),
    dict(cls="human", urgency_class="stranded", posture="standing", submersion="dry", zone="fan",
         p_living=0.88, w_class=0.99, urgency=1.4, count=3, thermal_hot=False, thermal_c=None,
         motion="moving", n_obs=23, h_acc=2.5, north=380.0, east=95.0, occl=0,
         note="group waving on the deposit fan"),
    dict(cls="human", urgency_class="stranded", posture="prone", submersion="wet", zone="hillslope",
         p_living=0.62, w_class=0.97, urgency=1.3, count=1, thermal_hot=False, thermal_c=None,
         motion="still", n_obs=6, h_acc=4.6, north=-420.0, east=-330.0, occl=2,
         note="prone, heavy canopy occlusion; needs a second pass"),
    dict(cls="human", urgency_class="immersed", posture="half_submerged", submersion="half",
         zone="channel", p_living=0.74, w_class=0.90, urgency=2.1, count=1, thermal_hot=True,
         thermal_c=29.6, motion="still", n_obs=9, h_acc=3.1, north=-60.0, east=430.0, occl=1,
         note="half submerged against debris"),
    dict(cls="animal", urgency_class="animal", posture="standing", submersion="dry", zone="fan",
         p_living=0.79, w_class=0.30, urgency=1.0, count=4, thermal_hot=True, thermal_c=33.2,
         motion="moving", n_obs=14, h_acc=2.7, north=520.0, east=-480.0, occl=0,
         note="cattle on high ground"),
    dict(cls="human", urgency_class="unknown", posture="unknown", submersion="unknown", zone="unknown",
         p_living=0.41, w_class=1.00, urgency=1.0, count=1, thermal_hot=False, thermal_c=None,
         motion="unknown", n_obs=3, h_acc=6.2, north=-560.0, east=520.0, occl=2,
         note="low-confidence candidate, 3 hits only"),
    dict(cls="human", urgency_class="stranded", posture="sitting", submersion="dry", zone="settlement",
         p_living=0.55, w_class=0.98, urgency=1.2, count=1, thermal_hot=False, thermal_c=None,
         motion="still", n_obs=5, h_acc=5.1, north=250.0, east=520.0, occl=1,
         note="dismissed after review: mannequin in a shopfront"),
]


def _thumb_bytes(idx: int, case: dict[str, Any], size: int = 192) -> bytes:
    """A synthetic evidence crop. Clearly marked SIM so a screenshot can never be mistaken for real data."""
    from PIL import Image, ImageDraw

    rng = random.Random(1000 + idx)
    img = Image.new("RGB", (size, size), (70, 82, 66))
    d = ImageDraw.Draw(img)
    for _ in range(900):  # mottled ground
        x, y = rng.randrange(size), rng.randrange(size)
        r = rng.randrange(2, 9)
        g = rng.randrange(-22, 22)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(70 + g, 82 + g, 66 + g))
    if case["zone"] == "channel":  # muddy water band
        d.rectangle([0, size * 0.45, size, size], fill=(96, 88, 66))
    body = (206, 176, 148) if case["cls"] == "human" else (120, 92, 62)
    cx, cy = size // 2, size // 2
    if case["posture"] in ("prone", "half_submerged", "supine"):
        d.ellipse([cx - 34, cy - 12, cx + 34, cy + 12], fill=body)
    else:
        d.ellipse([cx - 11, cy - 30, cx + 11, cy - 8], fill=body)
        d.rectangle([cx - 13, cy - 8, cx + 13, cy + 30], fill=body)
    x1, y1, x2, y2 = cx - 40, cy - 38, cx + 40, cy + 38
    d.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
    d.rectangle([0, 0, size - 1, 15], fill=(0, 0, 0))
    d.text((4, 3), f"SIM  {case['cls']}  conf {case['p_living']:.2f}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return buf.getvalue()


def seed_store(store: RecordStore, *, n: int | None = None, t0: float | None = None) -> list[Record]:
    """Write synthetic records + evidence thumbnails into the log. Returns them, ranked."""
    t0 = t0 or time.time()
    cases = _CASES[: n or len(_CASES)]
    built: list[Record] = []
    for i, c in enumerate(cases):
        lat, lon = offset_ne(ORIGIN_LAT, ORIGIN_LON, c["north"], c["east"])
        comps = ScoreComponents(
            p_living=c["p_living"],
            w_class=c["w_class"],
            urgency=c["urgency"],
            count_bonus=1.0 + 0.1 * c["count"],
            urgency_class=c["urgency_class"],
            elapsed_h=round(2.0 + 0.8 * i, 2),
            thermal_boost=1.15 if c["thermal_hot"] else 1.0,
            motion_boost=1.1 if c["motion"] == "moving" else 1.0,
            posture_promoted=c["urgency_class"] in ("immersed", "trapped"),
        )
        rid = f"SIM-{i + 1:03d}"
        rec = Record(
            record_id=rid,
            cluster_id=i,
            status="confirmed" if c["n_obs"] >= 5 else "candidate",
            cls=c["cls"],
            lat=lat,
            lon=lon,
            alt_msl_m=ORIGIN_ALT + 3.0 * math.sin(i),
            h_acc_m=c["h_acc"],
            h_acc_basis="budget_v1",
            method="flat_plane",
            agl_m=52.0,
            off_nadir_deg=round(4.0 + 3.0 * i % 17, 1),
            confidence=round(1.0 - (1.0 - c["p_living"]) ** max(1, c["n_obs"] // 4), 4),
            confidence_max_det=c["p_living"],
            score=round(comps.total(), 6),
            components=comps,
            n_observations=c["n_obs"],
            n_tracks_merged=1 + (i % 3),
            seen_in_passes=[1] if i % 2 else [1, 2],
            first_seen_utc=t0 - 900 + 30 * i,
            last_seen_utc=t0 - 120 + 5 * i,
            motion_state=c["motion"] if c["motion"] in ("still", "moving") else "unknown",
            motion_displacement_m=2.4 if c["motion"] == "moving" else 0.2,
            motion_window_s=12.0,
            count_estimate=c["count"],
            count_min=max(1, c["count"] - 1),
            count_max=c["count"] + 1,
            posture=c["posture"],
            posture_conf=0.71,
            submersion=c["submersion"],
            submersion_conf=0.66,
            occlusion=c["occl"],
            modality="fused" if c["thermal_hot"] else "rgb",
            thermal_hot=c["thermal_hot"],
            thermal_c=c["thermal_c"],
            pixel_size_px=round(26.0 + 4.0 * i, 1),
            gsd_cm_px=1.9,
            zone=c["zone"],
            source={"platform": "sim", "sim": True, "domain": "sim", "synthetic": True,
                    "clip_id": "SIM_FLOODVALLEY_001", "aoi_id": "wayanad-demo",
                    "telemetry": "cosysairsim"},
            notes=f"SIMULATED demo record (not a measurement). {c['note']}",
        )
        uri = store.put_thumbnail(rid, 1, _thumb_bytes(i, c), ext=".jpg")
        rec.evidence = [
            Evidence(thumb_uri=uri, clip_id="SIM_FLOODVALLEY_001", frame_idx=1200 + 37 * i,
                     frame_time_utc=t0 - 300 + 7 * i, bbox_px=(1820.0 + 9 * i, 940.0 + 11 * i,
                                                               1868.0 + 9 * i, 1010.0 + 11 * i),
                     det_conf=c["p_living"], camera="fused" if c["thermal_hot"] else "rgb")
        ]
        built.append(rec)

    built.sort(key=lambda r: r.score, reverse=True)
    for rank, rec in enumerate(built, start=1):
        rec.priority_rank = rank
        store.put(rec, actor="demo_seed", reason="synthetic seed")
    # One dismissal, so the map's "dismissed" layer and the R10 audit trail are both visible.
    store.dismiss(built[-1].record_id, reason="reviewed: mannequin in a shopfront, not a person",
                  by="commander")
    return store.records()


# --- the flight: the real plan lane, then the real coverage lane -------------------------------------------
def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def _nadir_gimbal(heading_deg: float = 0.0) -> tuple[float, float, float, float]:
    """camera(optical) -> NED at pitch -90 (nadir), wide axis across track.

    Built with the coverage lane's own `gimbal_quat`, which is defined to round-trip through
    `Telemetry.gimbal_pitch_deg()`. Hand-rolling this quaternion is a trap: the obvious
    "rotate -90 about Y" reads back as -180 deg and the footprint comes out oblique and huge —
    which is exactly what the first render of this scenario showed (see docs/lanes/store_api_map.md).
    """
    from sightline.coverage.footprint import gimbal_quat
    from sightline.plan.patterns import gimbal_yaw_for_heading

    return gimbal_quat(-90.0, gimbal_yaw_for_heading(heading_deg))


def build_route(*, agl_m: float = SURVEY_AGL_M, speed_ms: float = SURVEY_SPEED_MS,
                segment_id: str = "S01") -> dict[str, Any]:
    """The planned pattern, from the PLAN LANE (F2). Returns `Route.to_dict()` — B6's route JSON verbatim.

    The line spacing is derived by `sightline.plan` from the simulator's own camera model and the "body"
    presentation's sweep width; nothing about the pattern is hand-drawn here.
    """
    from sightline.coverage.grid import SceneFrame
    from sightline.coverage.presentation import SIM_RGB_4K
    from sightline.plan import boustrophedon_route, rect_polygon

    scene = SceneFrame.from_json(SCENE_JSON)
    poly = rect_polygon((0.0, 0.0), SEARCH_BOX_N_M, SEARCH_BOX_E_M)
    route = boustrophedon_route(poly, SIM_RGB_4K, agl_m, scene, presentation="body", side_overlap=0.25,
                                speed_ms=speed_ms, segment_id=segment_id, pass_id=0)
    return route.to_dict()


def route_frames(route: dict[str, Any], *, flown_fraction: float = FLOWN_FRACTION,
                 step_m: float = FRAME_STEP_M, t_end: float | None = None) -> list[Telemetry]:
    """Telemetry for the part of `route` already flown, sampled every `step_m` along track.

    One stretch is logged as a MANUAL takeover so the map's track colouring and §7 step 3 have something
    real to show. Timestamps end at `t_end` (default: now), so the sortie reads as in-progress.
    """
    wps = list(route["waypoints"])
    n_legs = max(1, int(round((len(wps) - 1) * max(0.02, min(1.0, flown_fraction)))))
    legs = list(zip(wps, wps[1:]))[:n_legs]
    speed = float(route["params"].get("speed_ms", SURVEY_SPEED_MS)) or SURVEY_SPEED_MS
    out: list[Telemetry] = []
    fi = 0
    for leg_i, (a, b) in enumerate(legs):
        dn, de = b["north_m"] - a["north_m"], b["east_m"] - a["east_m"]
        dist = math.hypot(dn, de)
        n = max(1, int(dist / step_m))
        yaw = math.degrees(math.atan2(de, dn))
        for s in range(n):
            f = s / n
            fi += 1
            manual = leg_i == 2 and 0.15 <= f <= 0.8  # one logged operator takeover
            out.append(Telemetry(
                t_utc=0.0, lat=a["lat"] + (b["lat"] - a["lat"]) * f, lon=a["lon"] + (b["lon"] - a["lon"]) * f,
                alt_msl_m=float(a["alt_asl_m"]), agl_m=float(a["agl_m"]),
                q_body=_yaw_quat(yaw), q_gimbal=_nadir_gimbal(yaw), gimbal_is_earth_referenced=True,
                vel_ned_ms=(speed * math.cos(math.radians(yaw)), speed * math.sin(math.radians(yaw)), 0.0),
                h_acc_m=2.5, mode="MANUAL" if manual else "AUTO",
                clip_id="SIM_FLOODVALLEY_001", frame_idx=fi,
                weather={"rain": 0.0, "fog": 0.0, "wind_ms": 3.0, "cloud": 0.35}, time_of_day="day"))
    dt = step_m / speed
    t0 = (t_end if t_end is not None else time.time()) - dt * len(out)
    for i, tel in enumerate(out):
        tel.t_utc = t0 + dt * i
    return out


def seed_coverage(out_dir: str | Path, frames: list[Telemetry] | None = None, *,
                  cell_m: float = 20.0, run_id: str = "SIM_FLOODVALLEY_001") -> dict[str, Any]:
    """Accumulate REAL search quality from `frames` and export it through B6's own `export_coverage`.

    Returns B6's manifest. Every POD number in it is `domain: "sim"` and model-driven: B6 reports
    `k_is_measured = false` until F19 fits it, and the map prints that.
    """
    from sightline.coverage import CoverageMap, export_coverage
    from sightline.coverage.grid import SceneFrame, polygons_from_latlon
    from sightline.coverage.presentation import SIM_RGB_4K

    frames = frames if frames is not None else route_frames(build_route())
    scene = SceneFrame.from_json(SCENE_JSON)
    cmap = CoverageMap.for_scene(scene, cell_m=cell_m, presentations=("body", "limb_only"))
    cmap.add_burial_polygons(polygons_from_latlon(cmap.any_grid, [list(r) for r in BURIAL_RINGS_LATLON]))
    intr = SIM_RGB_4K.intrinsics()
    auto = [t for t in frames if t.mode == "AUTO"]
    manual = [t for t in frames if t.mode != "AUTO"]
    cmap.add_pass(((t, intr) for t in auto), pass_id=0, mode="AUTO")
    if manual:  # the takeover is its own pass, so the map can attribute coverage to the mode (§5.12)
        cmap.add_pass(((t, intr) for t in manual), pass_id=1, mode="MANUAL")
    man = export_coverage(cmap, out_dir, stem="coverage")
    man["run_id"] = run_id
    (Path(out_dir) / "coverage.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def seed_mission(mission: MissionState, route: dict[str, Any] | None = None,
                 frames: list[Telemetry] | None = None) -> MissionState:
    """Adopt the real route as the planned pattern and replay the flown telemetry into the live layers."""
    from sightline.coverage.presentation import SIM_RGB_4K

    route = route if route is not None else build_route()
    frames = frames if frames is not None else route_frames(route)
    mission.intrinsics = SIM_RGB_4K.intrinsics()
    mission.set_route(route)
    for tel in frames:
        mission.update_pose(tel)
    return mission


def seed_all(store: RecordStore, coverage_dir: str | Path, mission: MissionState | None = None
             ) -> dict[str, Any]:
    """One coherent scenario: synthetic records, a real plan, a real flown track and a real POD raster."""
    recs = seed_store(store)
    route = build_route()
    frames = route_frames(route)
    man = seed_coverage(coverage_dir, frames)
    mission = seed_mission(mission or MissionState(), route, frames)
    return {"records": len(recs), "coverage": man, "mission": mission, "route": route,
            "frames": len(frames), "domain": "sim", "records_are_synthetic": True}
