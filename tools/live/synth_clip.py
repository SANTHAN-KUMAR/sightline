"""A capture run with REAL geometry and NO renderer: telemetry + labels for the live loop, editor shut.

    uv run python tools/live/synth_clip.py --out _artifacts/live/clip_seed23 --alt 45 --speed 12

**What this is.** The real scenario (`data/scene/actors.json`, `flood_valley.json`), the real flight plan
(`sightline.mission.pattern.build_plan`), the real terrain follower, the real camera calibration and the real
geolocation chain, flown host-side at the real ground speed and shutter spacing. For every frame it works out
which of the 71 survivors are inside the footprint and where they land in the image, and writes the same
`telemetry.csv` + `labels/*.json` a flown capture writes.

**What this is NOT.** There are no images. Nothing was rendered, nothing was flown, no simulator ran. It is a
harness for the live loop, the C2 map, the WebSocket and the latency instrumentation - the parts that do not
need pixels - and every artefact it produces is stamped ``source: "synthetic_telemetry"`` so it can never be
mistaken for a flight. It cannot exercise `--detector rgb`, and it says so rather than pretending.

Use it when the Unreal editor is unavailable (another lane holds the single-writer lock) or to reproduce a
live-path bug without a 25-minute flight. The real demo is `sightline.mission.live` against PIE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.common.geodesy import offset_ne  # noqa: E402
from sightline.geo import ChainConfig, project_pixel_ne  # noqa: E402
from sightline.mission.live import TELEMETRY_COLUMNS  # noqa: E402
from sightline.mission.pattern import Scenario, build_plan, cadence_verdict  # noqa: E402
from sightline.pipeline import nadir_gimbal_quat, to_geo_gimbal  # noqa: E402
from sightline.schemas import Intrinsics, Telemetry  # noqa: E402

W_PX, H_PX = 3840, 2160


def pixel_solver(tel: Telemetry, intr: Intrinsics, cfg: ChainConfig):
    """Invert the REAL geo chain at this pose: (north, east) offset -> (u, v). Probed, never hard-coded."""
    tg = to_geo_gimbal(tel)
    f0, n0, e0 = project_pixel_ne(intr.cx, intr.cy, tg, intr, cfg)
    if not f0.valid:
        return None
    _, n1, e1 = project_pixel_ne(intr.cx + 100.0, intr.cy, tg, intr, cfg)
    _, n2, e2 = project_pixel_ne(intr.cx, intr.cy + 100.0, tg, intr, cfg)
    a, b = (n1 - n0) / 100.0, (n2 - n0) / 100.0
    c, d = (e1 - e0) / 100.0, (e2 - e0) / 100.0
    det = a * d - b * c
    if abs(det) < 1e-9:
        return None

    def solve(dn: float, de: float) -> tuple[float, float]:
        rn, re = dn - n0, de - e0
        return intr.cx + (d * rn - b * re) / det, intr.cy + (-c * rn + a * re) / det

    return solve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="_artifacts/live/synth_clip")
    ap.add_argument("--alt", type=float, default=45.0)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--plan", choices=("box", "patches"), default="patches")
    ap.add_argument("--shutter-m", type=float, default=0.0, help="0 = derive from the confirmation gate")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--patches", type=int, default=0, help="only fly the first N patches (0 = all legs)")
    a = ap.parse_args()

    scn = Scenario.load()
    if a.shutter_m > 0:
        shutter = a.shutter_m
    else:
        from tools.capture.campaign import shutter_m as derive

        shutter = float(derive(a.alt, a.speed, scn.hfov_deg))
    verdict = cadence_verdict(a.alt, a.speed, shutter, scn.hfov_deg)
    if not verdict.ok:
        print(verdict.message())
        return 2
    plan = build_plan(scn, alt_m=a.alt, speed_ms=a.speed, shutter_m=shutter, plan=a.plan)
    legs = plan.legs
    if a.patches:
        per = max(1, len(legs) // max(1, len(plan.patches)))
        legs = legs[: per * a.patches]
    print(plan.summary())
    print(verdict.message())

    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    (out / "labels").mkdir(parents=True, exist_ok=True)
    clip_id = f"synth_seed{scn.seed}_alt{int(a.alt)}"
    cfg = ChainConfig()
    intr = Intrinsics(width_px=W_PX, height_px=H_PX, fx=scn.f_px, fy=scn.f_px,
                      cx=float(scn.cal["cx"]), cy=float(scn.cal["cy"]), source="calibration")
    home = scn.home
    home_geo = home["geopoint"]        # the pad's WGS-84 fix, verified in PIE against the GPS
    actors = [x for x in scn.truth["actors"] if x["aerially_detectable"]]

    rows: list[list] = []
    k, t_utc, seen = 0, 1_789_000_000.0, set()
    dt = shutter / a.speed
    for leg in legs:
        if k >= a.max_frames:
            break
        n, step = leg.north_start_m, math.copysign(shutter, leg.north_end_m - leg.north_start_m)
        while (n <= leg.north_end_m if step > 0 else n >= leg.north_end_m) and k < a.max_frames:
            east = leg.east_m
            asl = scn.terrain.surface_asl(east, n) + a.alt
            agl = asl - scn.terrain.surface_asl(east, n)
            lat, lon = offset_ne(float(home_geo["lat"]), float(home_geo["lon"]),
                                 n - float(home["north_m"]), east - float(home["east_m"]))
            tel = Telemetry(t_utc=t_utc, lat=lat, lon=lon, alt_msl_m=asl, agl_m=agl,
                            q_body=(1.0, 0.0, 0.0, 0.0), q_gimbal=nadir_gimbal_quat(),
                            gimbal_is_earth_referenced=True, mode="AUTO", clip_id=clip_id, frame_idx=k,
                            flood_level_asl_m=scn.water_level_m,
                            vel_ned_ms=(float(step / dt), 0.0, 0.0))
            solve = pixel_solver(tel, intr, cfg)
            labels = []
            if solve is not None:
                for act in actors:
                    u, v = solve(float(act["north_m"]) - n, float(act["east_m"]) - east)
                    box = 1.8 / max(agl, 1e-6) * scn.f_px          # a ~1.8 m person at this AGL, in pixels
                    if not (box < u < W_PX - box and box < v < H_PX - box):
                        continue
                    labels.append({
                        "actor_id": int(act["id"]), "name": f"Human_{int(act['id']):03d}",
                        "cls": act.get("cls", "human"),
                        "bbox_px": [round(u - box / 2, 2), round(v - box / 2, 2),
                                    round(u + box / 2, 2), round(v + box / 2, 2)],
                        "visible_px": int(box * box * 0.4), "size_px": round(box, 1),
                        "pose": act.get("pose", "unknown"), "submersion": act.get("submersion", "unknown"),
                        "occlusion": act.get("occlusion", 0), "zone": act.get("zone", "unknown"),
                        "group": act.get("group", 0), "aerially_detectable": True,
                    })
                    seen.add(int(act["id"]))
            (out / "labels" / f"{clip_id}_{k:05d}.json").write_text(json.dumps(labels, indent=1),
                                                                    encoding="utf-8")
            rows.append([k, t_utc, clip_id, round(east, 2), round(n, 2), round(asl, 2), round(agl, 2),
                         lat, lon, 1.0, 0.0, 0.0, 0.0, -90.0, round(scn.hfov_deg, 3), W_PX, H_PX,
                         round(scn.gsd_cm_px(agl), 4), "AUTO", round(scn.water_level_m, 3), len(labels),
                         round(a.speed, 2)])
            k += 1
            t_utc += dt
            n += step

    with (out / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TELEMETRY_COLUMNS)
        w.writerows(rows)
    card = {
        "product": "tools/live/synth_clip.py", "clip_id": clip_id, "domain": "sim",
        "source": "synthetic_telemetry",
        "WARNING": "NO IMAGES AND NO RENDERER. Telemetry and labels are computed from the real scenario, the "
                   "real flight plan, the real terrain and the real geolocation chain; nothing was flown and "
                   "nothing was photographed. Cannot be used with --detector rgb, and no number from it is a "
                   "detection result or a capture result.",
        "scenario_seed": scn.seed, "altitude_m_agl": a.alt, "speed_ms": a.speed, "shutter_m": shutter,
        "plan": plan.plan, "legs_flown": len(legs), "legs_planned": len(plan.legs), "frames": k,
        "total_boxes": int(sum(r[-2] for r in rows)), "unique_actors_seen": len(seen),
        "detectable_total": len(actors), "randomisation": "off (5.5c)",
    }
    (out / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"\n{k} frames, {card['total_boxes']} boxes, {len(seen)}/{len(actors)} survivors -> {out}")
    print("SYNTHETIC TELEMETRY, NO IMAGES: this exercises the live loop and the map, not the capture path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
