"""Synthetic data for the map, the headless check and the demo.

**Every number here is SIMULATED and labelled as such** (hard rule 5 / §5.12): each record carries
``source = {"domain": "sim", "synthetic": True, ...}`` and ``notes`` saying so, and the coverage overlay's
sidecar carries ``"domain": "sim"``. Nothing in this module measures anything — it exists so the UI, the
WebSocket and the coverage contract can be exercised end to end before the pipeline lanes land.

Replace it by pointing the API at the real record store the pipeline writes to.
"""

from __future__ import annotations

import io
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from sightline.api.coverage_feed import grid_to_overlay
from sightline.api.mission_feed import MissionState
from sightline.common.geodesy import offset_ne
from sightline.schemas import (
    CoverageGrid,
    Evidence,
    Intrinsics,
    Record,
    ScoreComponents,
    Telemetry,
)
from sightline.store import RecordStore

__all__ = ["ORIGIN_LAT", "ORIGIN_LON", "seed_store", "seed_coverage", "seed_mission", "seed_all"]

#: docs/CONTEXT.md: OriginGeopoint of the FloodValley scenario (Chooralmala / Mundakkai).
ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 11.4870, 76.1450, 1046.007

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


def seed_coverage(out_dir: str | Path, *, cell_m: float = 20.0, n: int = 80,
                  lanes: int = 7, run_id: str = "SIM_FLOODVALLEY_001") -> dict[str, Any]:
    """A boustrophedon's worth of swept coverage plus one 'aerial search cannot clear' burial polygon."""
    south, west = offset_ne(ORIGIN_LAT, ORIGIN_LON, -n * cell_m / 2, -n * cell_m / 2)
    grid = CoverageGrid.empty(south, west, cell_m, n, n, presentation="body", k=1.0)
    swath_cells = 4.5
    for li in range(lanes):
        row = (li + 0.5) * (n / lanes)
        for i in range(n):
            d = abs(i - row)
            if d > swath_cells * 1.6:
                continue
            gain = math.exp(-(d / swath_cells) ** 2) * (0.95 if li % 2 == 0 else 0.72)
            j_hi = n if li < lanes - 2 else int(n * 0.55)  # the last lanes are only half flown
            grid.coverage[i, :j_hi] += gain
    # A light second pass over the middle third only, so unswept ground stays visibly unswept.
    band = slice(int(n * 0.33), int(n * 0.66))
    grid.coverage[:, band] += 0.35 * np.exp(-((np.linspace(-2, 2, n)[:, None]) ** 2))
    grid.recompute_pod()
    cc = np.zeros((n, n), dtype=bool)
    cc[int(n * 0.55): int(n * 0.72), int(n * 0.18): int(n * 0.40)] = True  # debris fan / burial polygon
    cc[int(n * 0.24): int(n * 0.32), int(n * 0.62): int(n * 0.78)] = True
    grid.cannot_clear = cc
    return grid_to_overlay(grid, out_dir, run_id=run_id, domain="sim")


def seed_mission(mission: MissionState, *, cell_m: float = 20.0, n: int = 80, lanes: int = 7,
                 progress: float = 0.42) -> MissionState:
    """Planned boustrophedon + the part of it already flown, ending at the live pose."""
    mission.intrinsics = Intrinsics.from_hfov(3840, 2160, 84.0, source="sim")
    half = n * cell_m / 2
    plan: list[tuple[float, float]] = []
    for li in range(lanes):
        north = -half + (li + 0.5) * (2 * half / lanes)
        ends = [(-half, half), (half, -half)][li % 2]
        for east in ends:
            plan.append(offset_ne(ORIGIN_LAT, ORIGIN_LON, north, east))
    mission.set_plan(plan, name="FloodValley segment 1 — boustrophedon, 52 m AGL, 7 m/s")

    flown = max(2, int(len(plan) * progress))
    t = time.time() - 600
    for k in range(flown - 1):
        (la1, lo1), (la2, lo2) = plan[k], plan[k + 1]
        for s in range(0, 21):
            f = s / 20.0
            la, lo = la1 + (la2 - la1) * f, lo1 + (lo2 - lo1) * f
            mode = "MANUAL" if 2.0 <= k + f <= 2.7 else "AUTO"  # one logged takeover (§7 step 3)
            yaw = math.degrees(math.atan2(lo2 - lo1, la2 - la1))
            q = (math.cos(math.radians(yaw) / 2), 0.0, 0.0, math.sin(math.radians(yaw) / 2))
            t += 1.5
            mission.update_pose(
                Telemetry(t_utc=t, lat=la, lon=lo, alt_msl_m=ORIGIN_ALT + 52.0, agl_m=52.0,
                          q_body=q, vel_ned_ms=(6.0, 1.0, 0.0), mode=mode,
                          clip_id="SIM_FLOODVALLEY_001", frame_idx=k * 21 + s, h_acc_m=2.5)
            )
    return mission


def seed_all(store: RecordStore, coverage_dir: str | Path, mission: MissionState | None = None
             ) -> dict[str, Any]:
    recs = seed_store(store)
    side = seed_coverage(coverage_dir)
    mission = seed_mission(mission or MissionState())
    return {"records": len(recs), "coverage": side, "mission": mission,
            "domain": "sim", "synthetic": True}
