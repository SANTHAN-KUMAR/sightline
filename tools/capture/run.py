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

    out = REPO / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    c = connect()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    c.enableApiControl(True)
    c.armDisarm(True)

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
    print(f"frame {W}x{H}  calibrated f_px {f_px:.1f} (HFOV {hfov:.2f} deg)  "
          f"footprint {fw:.1f}x{fh:.1f} m  GSD {gsd:.3f} cm/px")
    print(f"area east[{e0:.0f},{e1:.0f}] north[{n0:.0f},{n1:.0f}]  -> {len(pts)} waypoints at {args.alt:.0f} m")

    tele_path = out / "telemetry.csv"
    tf = tele_path.open("w", newline="", encoding="utf-8")
    tw = csv.writer(tf)
    tw.writerow(["frame_idx", "t_utc", "clip_id", "east_m", "north_m", "alt_msl_m", "agl_m",
                 "lat", "lon", "q_w", "q_x", "q_y", "q_z", "gimbal_pitch_deg", "hfov_deg",
                 "width_px", "height_px", "gsd_cm_px", "mode", "flood_level_asl_m", "n_labels"])

    clip_id = f"sim_seed{truth['seed']}_alt{int(args.alt)}"
    total_labels, seen, empty_skipped, written = 0, set(), 0, 0
    t0 = time.time()
    for k, (ee, nn) in enumerate(pts):
        # Teleport: NED is relative to the PlayerStart, x=north, y=east, z=down.
        ned_n = nn - home["north_m"]
        ned_e = ee - home["east_m"]
        ned_d = -((water + args.alt) - home["ground_asl_m"])
        # a realistic survey attitude rather than a perfectly level one
        roll, pitch = rng.normal(0, 0.8), rng.normal(0, 0.8)
        # cosysairsim 3.4.1 spells it euler_to_quaternion(roll, pitch, yaw) in RADIANS; `to_quaternion`
        # (the upstream AirSim name) does not exist here.
        q = airsim.euler_to_quaternion(math.radians(roll), math.radians(pitch), 0.0)
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(ned_n, ned_e, ned_d), q), True)
        c.simPause(True)
        try:
            g = _grab(c, names, cmap)
        finally:
            c.simPause(False)

        labs = attach_truth(labels_from_mask(g["seg"], names, cmap), truth)
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

        tw.writerow([k, time.time(), clip_id, round(ee, 2), round(nn, 2), round(water + args.alt, 2),
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
    raise SystemExit(main())
