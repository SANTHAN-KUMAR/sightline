"""F2 + F5: fly a real boustrophedon coverage pattern and capture the dataset from the actual flight.

    uv run python -m sightline.mission.survey --alt 45 --speed 10 --out _artifacts/dataset/flight_alt45

Why this replaces waypoint teleporting. The first capture runner moved the drone with `simSetVehiclePose` /
`simSetKinematics` at 731 waypoints. That is using a physics-simulated multirotor as a camera dolly, and every
failure it produced came from fighting the solver rather than from the scene:

    * teleporting a rigid body each frame drove it to 2.2e8 rad/s and a 17 m/s descent;
    * the survey camera sits 30 cm BELOW the body, so the tumbling airframe swung into shot and filled frames
      with propellers - it silently ruined a 731-frame dataset that passed every programmatic check;
    * a pose set while paused is ignored; set while running it lands one call late;
    * even after zeroing the motion, each teleport impulse rolled the body up to 25.4 deg.

Flying the pattern removes all of it at once, and it is what SOLUTION_DOC F2 asks for anyway. The telemetry
then describes a real trajectory - real attitude, real velocity, real motion between frames - which is what
sections 5.4 and 5.7 assume and what the tracker's camera-motion compensation is designed for.

The drone holds constant AGL over the higher of ground or flood surface: the valley walls reach 1169.9 m ASL
while the flood surface is 1061.7, so a fixed height above the water flies into the hillside.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "scene"))
from tools.capture.labels import attach_truth, labels_from_mask, to_yolo  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import cosysairsim as airsim


def connect():
    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()
    return c


def grab(c, want_ir: bool = False) -> dict:
    """Raw AirSim buffers are RGB (measured); OpenCV is BGR, so the swap happens once, on write."""
    req = [airsim.ImageRequest("survey", airsim.ImageType.Scene, False, False),
           airsim.ImageRequest("survey", airsim.ImageType.Segmentation, False, False)]
    if want_ir:
        req.append(airsim.ImageRequest("survey", airsim.ImageType.Infrared, False, False))
    with contextlib.redirect_stdout(io.StringIO()):
        res = c.simGetImages(req)
    keys = ("scene", "seg", "ir")
    out = {}
    for r, k in zip(res, keys):
        out[k] = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alt", type=float, default=45.0, help="metres AGL")
    ap.add_argument("--speed", type=float, default=10.0, help="m/s ground speed")
    ap.add_argument("--out", default="_artifacts/dataset/flight")
    ap.add_argument("--quantile", type=float, default=0.05,
                    help="trim the survivor bounding box to keep the flight a sane length")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--shutter-m", type=float, default=26.0, help="capture every N metres of track")
    ap.add_argument("--thermal", action="store_true", help="also capture the Infrared pass")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--alt-tol-m", type=float, default=8.0,
                    help="do not capture while more than this far off the commanded AGL")
    ap.add_argument("--max-tilt-deg", type=float, default=8.0,
                    help="do not capture while the airframe is banked beyond this")
    a = ap.parse_args()

    scene = json.loads((REPO / "data/scene/flood_valley.json").read_text())
    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    cal = json.loads((REPO / "data/scene/camera_survey.json").read_text())
    home, water = scene["launch_site"], truth["water_level_m"]
    f_px = float(cal["f_px"])

    import gen_terrain as gt
    t = gt.build(scene["size_m"], scene["cell_m"], scene["seed"])
    H, N, CELL, SIZE = t["height"], t["n"], t["cell_m"], t["size_m"]
    X0 = -SIZE / 2.0

    def surface_asl(e, n):
        i = int(round((e - X0) / CELL)); j = int(round((n - X0) / CELL))
        return max(float(H[min(max(j, 0), N - 1)][min(max(i, 0), N - 1)]), water)

    det = [x for x in truth["actors"] if x["aerially_detectable"]]
    es = np.array([x["east_m"] for x in det]); ns = np.array([x["north_m"] for x in det])
    q = a.quantile
    e0, e1 = np.quantile(es, [q, 1 - q]); n0, n1 = np.quantile(ns, [q, 1 - q])
    e0, e1, n0, n1 = e0 - 35, e1 + 35, n0 - 35, n1 + 35

    W = 2.0 * a.alt * math.tan(math.radians(float(cal["hfov_deg"])) / 2.0)
    d_line = W * (1.0 - a.overlap)
    lines = int(math.ceil((e1 - e0) / d_line)) + 1
    legs = []
    for i in range(lines):
        e = e0 + i * d_line
        if e > e1 + d_line:
            break
        legs.append((e, n0, n1) if i % 2 == 0 else (e, n1, n0))
    track_km = sum(abs(b - c_) for _, b, c_ in legs) / 1000.0
    inside = int(((es >= e0) & (es <= e1) & (ns >= n0) & (ns <= n1)).sum())
    print(f"survey box east[{e0:.0f},{e1:.0f}] north[{n0:.0f},{n1:.0f}] holds {inside}/{len(det)} survivors")
    print(f"{len(legs)} lines, {d_line:.0f} m apart, {track_km:.1f} km of track at {a.speed:.0f} m/s "
          f"-> ~{track_km * 1000 / a.speed / 60:.1f} min, shutter every {a.shutter_m:.0f} m")

    out = REPO / a.out
    for d in ("images", "labels", "masks") + (("ir",) if a.thermal else ()):
        (out / d).mkdir(parents=True, exist_ok=True)

    c = connect()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    with contextlib.redirect_stdout(io.StringIO()):
        grab(c, a.thermal)                                  # warm-up frame, discarded (unconverted buffers)

    # --- PRE-FLIGHT: is the instance segmentation healthy? -------------------------------------------------
    # A capture is only as good as its instance colours. `tools/capture/thermal_ids.py` assigns a SHARED
    # segmentation id per temperature (331 objects got one id), which collapses distinct instances onto one
    # colour: the flood plane then renders in a survivor's colour, and 78 % of the resulting labels are
    # physically impossible - a "person" spanning 67.8 x 38.1 m of ground. Nothing downstream noticed until a
    # human looked at a frame. Refuse to fly rather than produce that again.
    _pal: dict[tuple, list[str]] = {}
    for _i, _n in enumerate(names):
        _pal.setdefault(tuple(int(v) for v in cmap[_i]), []).append(_n)
    _dupes = {c_: ns for c_, ns in _pal.items()
              if len(ns) > 1 and any(x.startswith(("Human_", "Animal_")) for x in ns)}
    if _dupes:
        print("\nFATAL: instance segmentation is degenerate - actors share a colour with other objects:")
        for c_, ns in list(_dupes.items())[:5]:
            print(f"   colour {c_} used by {len(ns)} objects: {ns[:4]}")
        print("\n   Restart PIE to regenerate unique instance ids (InitialInstanceSegmentation in settings),")
        print("   and never run tools/capture/thermal_ids.py on a PIE session you intend to capture from.")
        raise SystemExit(2)
    print(f"pre-flight: {len(names)} instances, every actor colour unique")

    c.enableApiControl(True)
    c.armDisarm(True)
    print("\ntaking off...")
    c.takeoffAsync(timeout_sec=20).join()

    clip_id = f"flight_seed{truth['seed']}_alt{int(a.alt)}"
    tf = (out / "telemetry.csv").open("w", newline="", encoding="utf-8")
    tw = csv.writer(tf)
    tw.writerow(["frame_idx", "t_utc", "clip_id", "east_m", "north_m", "alt_msl_m", "agl_m", "lat", "lon",
                 "q_w", "q_x", "q_y", "q_z", "gimbal_pitch_deg", "hfov_deg", "width_px", "height_px",
                 "gsd_cm_px", "mode", "flood_level_asl_m", "n_labels", "speed_ms"])

    k = 0
    total = 0
    agls: list[float] = []
    seen: set[int] = set()
    t_start = time.time()
    last_shot = None
    try:
        for li, (e, na, nb) in enumerate(legs):
            if (time.time() - t_start) / 60.0 > a.max_minutes:
                print(f"  stopping: {a.max_minutes} min budget reached at line {li}/{len(legs)}")
                break
            # Fly the WHOLE leg in one command and shoot on the move. Stopping at every shutter point made the
            # drone accelerate and brake 22 times per line (5 min for two lines) and removed the very motion the
            # tracker's camera-motion compensation exists to handle. A continuous leg is faster AND a truer pass.
            asl_a = surface_asl(e, na) + a.alt
            c.moveToPositionAsync(float(na - home["north_m"]), float(e - home["east_m"]),
                                  float(-(asl_a - home["ground_asl_m"])), a.speed, timeout_sec=120).join()
            asl_b = surface_asl(e, nb) + a.alt
            fut = c.moveToPositionAsync(float(nb - home["north_m"]), float(e - home["east_m"]),
                                        float(-(asl_b - home["ground_asl_m"])), a.speed, timeout_sec=400,
                                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=airsim.YawMode(False, 0))
            print(f"  line {li + 1}/{len(legs)} east {e:7.1f}  north {na:.0f} -> {nb:.0f}")
            shots, last_n, t_leg = 0, na, time.time()
            while True:
                st = c.simGetGroundTruthKinematics()
                cur_n = home["north_m"] + st.position.x_val
                cur_e = home["east_m"] + st.position.y_val
                reached = (cur_n >= nb - 2.0) if nb > na else (cur_n <= nb + 2.0)
                # A real survey does not release the shutter mid-bank, and the validator flags any frame
                # captured beyond +-12 deg. The drone banks on acceleration and at leg ends, so gate on the
                # body attitude: wait for it to level rather than recording a tilted frame.
                _o = st.orientation
                _roll = math.degrees(math.atan2(2 * (_o.w_val * _o.x_val + _o.y_val * _o.z_val),
                                                1 - 2 * (_o.x_val ** 2 + _o.y_val ** 2)))
                _pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (_o.w_val * _o.y_val
                                                                       - _o.z_val * _o.x_val)))))
                _agl = (home["ground_asl_m"] - st.position.z_val) - surface_asl(cur_e, cur_n)
                # and do not shoot when the drone is far off its commanded height: an audit found AGL ranging
                # 18.7-121.4 m in a run labelled "45 m", which corrupts GSD, scale and geolocation alike.
                on_alt = abs(_agl - a.alt) <= a.alt_tol_m
                level = max(abs(_roll), abs(_pitch)) <= a.max_tilt_deg and on_alt
                if level and (shots == 0 or abs(cur_n - last_n) >= a.shutter_m):
                    g = grab(c, a.thermal)
                    labs = attach_truth(labels_from_mask(g["seg"], names, cmap), truth)
                    gp = c.getMultirotorState().gps_location
                    o, v = st.orientation, st.linear_velocity
                    spd = math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2)
                    asl = home["ground_asl_m"] - st.position.z_val
                    stem = f"{clip_id}_{k:05d}"
                    cv2.imwrite(str(out / "images" / f"{stem}.png"), g["scene"][:, :, ::-1])
                    cv2.imwrite(str(out / "masks" / f"{stem}.png"), g["seg"][:, :, ::-1])
                    if a.thermal and "ir" in g:
                        cv2.imwrite(str(out / "ir" / f"{stem}.png"), g["ir"][:, :, ::-1])
                    (out / "labels" / f"{stem}.txt").write_text(
                        "\n".join(to_yolo(labs, g["seg"].shape[1], g["seg"].shape[0])), encoding="utf-8")
                    (out / "labels" / f"{stem}.json").write_text(json.dumps(
                        [{"actor_id": m.actor_id, "name": m.name, "cls": m.cls, "bbox_px": list(m.bbox_px),
                          "visible_px": m.visible_px, "size_px": m.size_px, "pose": m.pose,
                          "submersion": m.submersion, "occlusion": m.occlusion, "zone": m.zone,
                          "group": m.group, "aerially_detectable": m.aerially_detectable} for m in labs],
                        indent=1), encoding="utf-8")
                    # GSD must come from the MEASURED height above the surface, not the commanded --alt. The
                    # drone does not hold the setpoint exactly over rising ground, and writing the constant
                    # made every GSD and every geolocation number wrong by however much it was off.
                    agl_now = asl - surface_asl(cur_e, cur_n)
                    tw.writerow([k, time.time(), clip_id, round(cur_e, 2), round(cur_n, 2), round(asl, 2),
                                 round(agl_now, 2), gp.latitude, gp.longitude,
                                 round(o.w_val, 6), round(o.x_val, 6), round(o.y_val, 6), round(o.z_val, 6),
                                 -90.0, round(float(cal["hfov_deg"]), 3), g["seg"].shape[1], g["seg"].shape[0],
                                 round(agl_now / f_px * 100.0, 4), "AUTO", round(water, 3), len(labs),
                                 round(spd, 2)])
                    agls.append(agl_now)
                    total += len(labs)
                    seen.update(m.actor_id for m in labs)
                    k += 1
                    shots += 1
                    last_shot = stem
                    last_n = cur_n
                if reached or time.time() - t_leg > 400:
                    break
                time.sleep(0.05)
            fut.join()
            print(f"    {shots} shots in {time.time() - t_leg:.0f}s, {len(seen)} unique survivors so far")
        print(f"\nreturning to launch...")
        c.moveToPositionAsync(0.0, 0.0, -(surface_asl(home["east_m"], home["north_m"]) + a.alt
                                          - home["ground_asl_m"]), a.speed, timeout_sec=120).join()
    finally:
        tf.close()
        try:
            c.landAsync(timeout_sec=30)
            c.armDisarm(False)
            c.enableApiControl(False)
        except Exception:
            pass

    detset = {x["id"] for x in det}
    card = {"clip_id": clip_id, "scenario_seed": truth["seed"], "altitude_m_agl": a.alt,
            "speed_ms": a.speed, "shutter_m": a.shutter_m, "frames": k, "total_boxes": total,
            "unique_actors_seen": len(seen), "detectable_in_box": inside,
            "detectable_seen": len(seen & detset), "buried_seen": sorted(seen - detset),
            "f_px": f_px,
            "gsd_cm_px_nominal": round(a.alt / f_px * 100.0, 4),
            "agl_m_measured": ({"min": round(min(agls), 1), "median": round(sorted(agls)[len(agls) // 2], 1),
                                "max": round(max(agls), 1)} if agls else None),
            "gsd_note": "per-frame gsd_cm_px in telemetry.csv is from MEASURED AGL, not the commanded altitude",
            "capture": "flown coverage pattern (F2), not teleported",
            "domain": "sim", "randomisation": "off (5.5c)", "split_rule": "by scenario seed",
            "minutes": round((time.time() - t_start) / 60.0, 1), "last": last_shot}
    (out / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"\n{k} frames, {total} boxes, {len(seen & detset)}/{inside} survivors in the box seen, "
          f"{card['minutes']} min")
    print(f"written to {out}")
    print("NOW LOOK: uv run python tools/capture/contact_sheet.py " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
