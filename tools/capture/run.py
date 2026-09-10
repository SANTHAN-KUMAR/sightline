"""F5: capture a labelled synthetic dataset from the FloodValley scenario.

    uv run python tools/capture/run.py --alt 45 --out _artifacts/dataset/seed23_alt45

Requires a running PIE (or packaged build) started with `-settings=sim/settings/dataset.json` (4K survey camera).

Design decisions, stated because they matter for how the numbers should be read:

* **Waypoints are teleports, not flights.** `simSetVehiclePose` places the drone exactly on each waypoint, so a
  run is reproducible frame for frame and costs seconds instead of minutes. Flight dynamics are exercised by the
  live-demo path (F2/F3), not by dataset generation; SOLUTION_DOC 6.2 asks for waypoint capture, and a settled
  drone is what a survey pass photographs anyway. A small fixed attitude perturbation per waypoint keeps the
  telemetry from being unrealistically perfect.
* **The first `simGetImages` of a fresh engine process returns UNCONVERTED buffers** - depth in 0..0.6 instead of
  metres - with correct dimensions and no error (docs/HANDBOOK.md section 5). A warm-up frame is taken and
  discarded before anything is written, and depth is sanity-checked against the known altitude.
* **The visible-extent box is the training box** (6.3) and is read exactly from the instance mask.
* Splits are BY SCENARIO SEED, never by frame (5.5c). This script writes one scenario; run it again with a
  different `data/scene/actors.json` seed for the held-out split.
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

with contextlib.redirect_stdout(io.StringIO()):     # cosysairsim prints on import/connect
    import cosysairsim as airsim


def connect():
    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()
    return c


def boustrophedon(e0, e1, n0, n1, alt_m, hfov_deg, aspect, overlap=0.2):
    """Lawnmower waypoints whose spacing comes from the camera footprint, not a magic number."""
    w = 2.0 * alt_m * math.tan(math.radians(hfov_deg) / 2.0)
    h = w / aspect
    d_line = w * (1.0 - overlap)
    d_step = h * (1.0 - overlap)
    lines = max(1, int(math.ceil((e1 - e0) / d_line)) + 1)
    pts = []
    for i in range(lines):
        e = e0 + i * d_line
        if e > e1 + d_line:
            break
        steps = max(1, int(math.ceil((n1 - n0) / d_step)) + 1)
        rng = range(steps) if i % 2 == 0 else range(steps - 1, -1, -1)
        for j in rng:
            n = n0 + j * d_step
            if n > n1 + d_step:
                continue
            pts.append((e, n))
    return pts, (w, h)


def park(c) -> None:
    """Leave the drone at rest on the pad, disarmed. Always call this, including on failure: an airborne,
    armed drone with no setpoint tumbles until someone stops PIE."""
    try:
        c.enableApiControl(False)
        c.armDisarm(False)
        ks = airsim.KinematicsState()
        ks.position = airsim.Vector3r(0.0, 0.0, 0.0)
        ks.orientation = airsim.euler_to_quaternion(0.0, 0.0, 0.0)
        for f in ("linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration"):
            setattr(ks, f, airsim.Vector3r(0.0, 0.0, 0.0))
        c.simPause(False)
        c.simSetKinematics(ks, True)
    except Exception as exc:                      # never mask the real failure with a cleanup error
        print(f"  (could not park the drone: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=45.0, help="metres above the flood surface")
    ap.add_argument("--out", default="_artifacts/dataset/run")
    ap.add_argument("--margin", type=float, default=40.0, help="metres of margin around the actors")
    ap.add_argument("--limit", type=int, default=0, help="stop after N waypoints (0 = all)")
    ap.add_argument("--seed", type=int, default=7, help="seed for the attitude perturbation")
    ap.add_argument("--keep-empty-every", type=int, default=6,
                    help="write 1 in N frames that contain no survivor, as true negatives (0 = none)")
    args = ap.parse_args()

    scene = json.loads((REPO / "data/scene/flood_valley.json").read_text())
    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    home = scene["launch_site"]
    water = truth["water_level_m"]
    rng = np.random.default_rng(args.seed)

    # TERRAIN FOLLOWING. Flying a fixed height above the FLOOD SURFACE puts the camera inside the valley
    # walls: the terrain over the sweep reaches 1169.9 m ASL while water + 45 m is only 1106.7, so 30.3 % of
    # the area was above the camera and produced blank frames, and much of the rest was photographed at
    # 9-23 m AGL instead of 45 (wrong scale, wrong GSD, wrong telemetry). A survey drone holds AGL, so the
    # camera height is taken above whichever is higher, the ground or the flood surface.
    import gen_terrain as gt                                    # noqa: PLC0415  (host-side, has numpy)
    _t = gt.build(scene["size_m"], scene["cell_m"], scene["seed"])
    _h, _n, _cell, _size = _t["height"], _t["n"], _t["cell_m"], _t["size_m"]
    _xs0 = -_size / 2.0

    def surface_asl(east_m: float, north_m: float) -> float:
        i = int(round((east_m - _xs0) / _cell))
        j = int(round((north_m - _xs0) / _cell))
        g = float(_h[min(max(j, 0), _n - 1)][min(max(i, 0), _n - 1)])
        return max(g, water)

    out = REPO / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    c = connect()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    # Capture teleports the camera; SimpleFlight must not also be trying to fly. Armed with API control, its
    # 60 ms hover watchdog fights every teleport, and a script that exits mid-sweep leaves the drone airborne
    # with no setpoint - which is what left it tumbling on screen between runs.
    c.enableApiControl(False)
    c.armDisarm(False)

    # Camera geometry: use the CALIBRATED focal length, not simGetCameraInfo().fov.
    # Measured 2026-09-10: simGetCameraInfo reports 89.904 deg for this camera while the true rendered HFOV is
    # 73.98 deg (f_px 2548.7, residual RMS 5.6 px against known survivor positions). Trusting the reported FOV
    # would put a 21% error into every GSD, footprint and geolocation number downstream.
    cal_path = REPO / "data/scene/camera_survey.json"
    if not cal_path.exists():
        raise SystemExit("run tools/capture/calibrate_camera.py first: simGetCameraInfo().fov is not the "
                         "rendered FOV and every GSD depends on the calibrated f_px")
    cal = json.loads(cal_path.read_text())
    f_px = float(cal["f_px"])
    hfov = float(cal["hfov_deg"])

    # --- warm-up frame, discarded: the first simGetImages of a process returns unconverted buffers ---------
    req = [airsim.ImageRequest("survey", airsim.ImageType.Scene, False, False)]
    with contextlib.redirect_stdout(io.StringIO()):
        c.simGetImages(req)
    time.sleep(0.2)
    c.simPause(False)

    e = [a["east_m"] for a in truth["actors"]]
    n = [a["north_m"] for a in truth["actors"]]
    e0, e1 = min(e) - args.margin, max(e) + args.margin
    n0, n1 = min(n) - args.margin, max(n) + args.margin

    probe = _grab(c, names, cmap)
    H, W = probe["scene"].shape[:2]
    if (W, H) != (cal["width"], cal["height"]):
        raise SystemExit(f"calibration is for {cal['width']}x{cal['height']} but the camera is "
                         f"{W}x{H}: re-run calibrate_camera.py")
    aspect = W / H
    pts, (fw, fh) = boustrophedon(e0, e1, n0, n1, args.alt, hfov, aspect)
    if args.limit:
        pts = pts[: args.limit]
    gsd = args.alt / f_px * 100.0            # cm per pixel at the flood surface
    with contextlib.redirect_stdout(io.StringIO()):
        _ci = c.simGetCameraInfo("survey")
    _o = _ci.pose.orientation
    _pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (_o.w_val * _o.y_val - _o.z_val * _o.x_val)))))
    if abs(_pitch + 90.0) > 3.0:
        raise SystemExit(f"survey camera is not nadir (pitch {_pitch:.2f} deg): the gimbal has not settled")
    print(f"frame {W}x{H}  calibrated f_px {f_px:.1f} (HFOV {hfov:.2f} deg)  "
          f"footprint {fw:.1f}x{fh:.1f} m  GSD {gsd:.3f} cm/px")
    _sample = [surface_asl(x, y) for x, y in pts[::37]]
    print(f"area east[{e0:.0f},{e1:.0f}] north[{n0:.0f},{n1:.0f}]  -> {len(pts)} waypoints at "
          f"{args.alt:.0f} m AGL (terrain-following; surface {min(_sample):.0f}-{max(_sample):.0f} m ASL)")

    tele_path = out / "telemetry.csv"
    tf = tele_path.open("w", newline="", encoding="utf-8")
    tw = csv.writer(tf)
    tw.writerow(["frame_idx", "t_utc", "clip_id", "east_m", "north_m", "alt_msl_m", "agl_m",
                 "lat", "lon", "q_w", "q_x", "q_y", "q_z", "gimbal_pitch_deg", "hfov_deg",
                 "width_px", "height_px", "gsd_cm_px", "mode", "flood_level_asl_m", "n_labels"])

    clip_id = f"sim_seed{truth['seed']}_alt{int(args.alt)}"
    _parked = False
    total_labels, seen, empty_skipped, written, skipped_pose = 0, set(), 0, 0, 0
    t0 = time.time()
    for k, (ee, nn) in enumerate(pts):
        # Teleport: NED is relative to the PlayerStart, x=north, y=east, z=down.
        ned_n = nn - home["north_m"]
        ned_e = ee - home["east_m"]
        cam_asl = surface_asl(ee, nn) + args.alt
        ned_d = -(cam_asl - home["ground_asl_m"])
        # a realistic survey attitude rather than a perfectly level one
        roll, pitch = rng.normal(0, 0.8), rng.normal(0, 0.8)
        # cosysairsim 3.4.1 spells it euler_to_quaternion(roll, pitch, yaw) in RADIANS.
        q = airsim.euler_to_quaternion(math.radians(roll), math.radians(pitch), 0.0)
        # Measured 2026-09-10, and all three facts matter:
        #  * a pose set while the sim is PAUSED is ignored entirely - the vehicle never moves, and every frame
        #    comes out identical (8 frames from 8 different waypoints were byte-alike, taken from the ground
        #    beside a survivor);
        #  * a pose set while running applies ASYNCHRONOUSLY, landing one call late, so capturing immediately
        #    photographs the PREVIOUS waypoint;
        #  * simSetVehiclePose does not clear the body's motion, so teleporting each frame under a live solver
        #    spun it up to 2.2e8 rad/s. The airframe then tumbled around the camera (mounted 30 cm below it)
        #    and filled most frames with propellers - which is what wrecked the first dataset.
        # So: move while RUNNING with the motion zeroed, wait until the pose has actually taken, then pause to
        # capture a still frame.
        ks = airsim.KinematicsState()
        ks.position = airsim.Vector3r(ned_n, ned_e, ned_d)
        ks.orientation = q
        for _f in ("linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration"):
            setattr(ks, _f, airsim.Vector3r(0.0, 0.0, 0.0))
        if not _settle(c, ks, ned_n, ned_e, ned_d, q):
            skipped_pose += 1
            continue
        c.simPause(True)
        try:
            g = _grab(c, names, cmap)
        finally:
            c.simPause(False)

        labs = attach_truth(labels_from_mask(g["seg"], names, cmap), truth)
        if k < 3 or k % 200 == 0:
            frac = _airframe_fraction(g["seg"], names, cmap)
            if frac > 0.02:
                raise SystemExit(f"waypoint {k}: the drone's own airframe covers {100 * frac:.1f}% of the "
                                 "frame - the body is tumbling around the camera again")
        # The sweep covers a lot of open flood water. Keep every frame that has a survivor, plus a regular
        # sample of empty ones as true negatives - a detector trained without negatives over-fires on glint
        # and debris, which is exactly the false-positive slice section 5.5 warns about. Skipped frames are
        # still flown and still counted, so coverage statistics stay honest.
        if not labs and (args.keep_empty_every <= 0 or k % args.keep_empty_every):
            empty_skipped += 1
            continue
        st = c.simGetGroundTruthKinematics()
        gp = c.getMultirotorState().gps_location
        stem = f"{clip_id}_{k:05d}"
        # raw AirSim buffers are RGB; cv2 writes BGR, so swap on the way out or every saved
        # frame has red and blue exchanged.
        cv2.imwrite(str(out / "images" / f"{stem}.png"), g["scene"][:, :, ::-1])
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), g["seg"][:, :, ::-1])
        (out / "labels" / f"{stem}.txt").write_text("\n".join(to_yolo(labs, W, H)), encoding="utf-8")
        (out / "labels" / f"{stem}.json").write_text(json.dumps(
            [{"actor_id": m.actor_id, "name": m.name, "cls": m.cls, "bbox_px": list(m.bbox_px),
              "visible_px": m.visible_px, "size_px": m.size_px, "pose": m.pose,
              "submersion": m.submersion, "occlusion": m.occlusion, "zone": m.zone, "group": m.group,
              "aerially_detectable": m.aerially_detectable} for m in labs], indent=1), encoding="utf-8")

        tw.writerow([k, time.time(), clip_id, round(ee, 2), round(nn, 2), round(cam_asl, 2),
                     round(args.alt, 2), gp.latitude, gp.longitude,
                     round(st.orientation.w_val, 6), round(st.orientation.x_val, 6),
                     round(st.orientation.y_val, 6), round(st.orientation.z_val, 6),
                     -90.0, round(hfov, 3), W, H, round(gsd, 3), "AUTO", round(water, 3), len(labs)])
        total_labels += len(labs)
        written += 1
        seen.update(m.actor_id for m in labs)
        if k % 10 == 0 or k == len(pts) - 1:
            print(f"  [{k + 1:4d}/{len(pts)}] {stem}  labels={len(labs):3d}  "
                  f"unique so far={len(seen):3d}  {time.time() - t0:5.1f}s")
    tf.close()
    park(c)
    _parked = True

    detectable = {a["id"] for a in truth["actors"] if a["aerially_detectable"]}
    card = {
        "clip_id": clip_id, "scenario_seed": truth["seed"], "terrain_seed": truth["terrain_seed"],
        "altitude_m": args.alt, "frames": len(pts), "width": W, "height": H,
        "hfov_deg": round(hfov, 3), "f_px": f_px, "gsd_cm_px": round(gsd, 3),
        "camera_calibration": "data/scene/camera_survey.json",
        "footprint_m": [round(fw, 1), round(fh, 1)],
        "total_boxes": total_labels,
        "unique_actors_seen": len(seen),
        "actors_total": len(truth["actors"]),
        "actors_aerially_detectable": len(detectable),
        "detectable_seen": len(seen & detectable),
        "buried_seen": sorted(seen - detectable),
        "domain": "sim",
        "randomisation": "off (SOLUTION_DOC 5.5c)",
        "split_rule": "by scenario seed, never by frame",
        "label_rule": "visible extent from the instance mask (6.3)",
    }
    (out / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"\n{len(pts)} frames, {total_labels} boxes, {len(seen)}/{len(truth['actors'])} actors seen "
          f"({len(seen & detectable)}/{len(detectable)} of the aerially detectable ones)")
    if seen - detectable:
        print(f"  WARNING: buried actors appeared in the mask: {sorted(seen - detectable)}")
    print(f"written to {out}")
    return 0


def _airframe_fraction(seg, names, cmap):
    """Fraction of the frame taken by the drone itself. The camera sits 30 cm below the body, so a tumbling
    airframe swings into shot and silently ruins a whole dataset."""
    from tools.capture.labels import palette_rgb
    pal = palette_rgb(cmap)
    idx = [i for i, n in enumerate(names) if "Drone" in n or "Propeller" in n]
    if not idx:
        return 0.0
    m = np.zeros(seg.shape[:2], bool)
    for i in idx:
        m |= np.all(seg == pal[i], axis=2)
    return float(m.mean())


def _settle(c, ks, n, e, d, q, tol_m=0.30, tol_deg=1.0, tol_w=0.05, tries=40):
    """Hold the vehicle at the commanded pose until it is ACTUALLY there and level.

    Two separate traps, both measured 2026-09-10:
      * the teleport applies asynchronously, so capturing straight away photographs the previous waypoint;
      * the solver treats each teleport as an impulse, so the body lurches - commanding +-0.8 deg of survey
        attitude produced rolls of up to 25.4 deg and 2.4 rad/s of spin between waypoints. That is the
        "aggressive flipping" seen on screen, and at 25 deg the airframe can swing into the camera's view.
    Re-applying the zeroed kinematics each iteration makes the correction win, and the exit condition checks
    position, attitude AND angular rate rather than position alone.
    """
    want_roll, want_pitch = _rp_deg(q)
    for _ in range(tries):
        c.simSetKinematics(ks, True)
        k = c.simGetGroundTruthKinematics()
        p, o, w = k.position, k.orientation, k.angular_velocity
        if (abs(p.x_val - n) < tol_m and abs(p.y_val - e) < tol_m and abs(p.z_val - d) < tol_m):
            roll, pitch = _rp_deg(o)
            spin = math.sqrt(w.x_val ** 2 + w.y_val ** 2 + w.z_val ** 2)
            if abs(roll - want_roll) < tol_deg and abs(pitch - want_pitch) < tol_deg and spin < tol_w:
                return True
        time.sleep(0.02)
    return False


def _rp_deg(o):
    """(roll, pitch) in degrees from a quaternion-like with w/x/y/z members."""
    w, x, y, z = o.w_val, o.x_val, o.y_val, o.z_val
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return roll, math.degrees(math.asin(s))


def _grab(c, names, cmap) -> dict:
    req = [airsim.ImageRequest("survey", airsim.ImageType.Scene, False, False),
           airsim.ImageRequest("survey", airsim.ImageType.Segmentation, False, False)]
    with contextlib.redirect_stdout(io.StringIO()):
        res = c.simGetImages(req)
    out = {}
    for r, key in zip(res, ("scene", "seg")):
        a = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        out[key] = a.reshape(r.height, r.width, 3)
    return out


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        # A crash or Ctrl-C mid-sweep must not leave an armed drone airborne.
        try:
            park(connect())
        except Exception:
            pass
        raise
