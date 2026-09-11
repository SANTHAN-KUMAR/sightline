"""Recover the missing depth channel for already-captured frames and re-derive their labels. PIE ON.

    uv run python tools/capture/refine_labels_with_depth.py _artifacts/dataset/seed23_alt35            # report
    uv run python tools/capture/refine_labels_with_depth.py _artifacts/dataset/seed23_alt35 --write
    uv run python tools/capture/refine_labels_with_depth.py _artifacts/dataset/seed23_alt55 --de-max 8

WHY THIS IS NOT A RE-FLIGHT. Cosys-AirSim renders the instance mask with the InstancedFoliage and
InstancedGrass show flags off (`Source/Annotation/ObjectAnnotator.cpp:SetViewForAnnotationRender`). Every
plant is a HISM instance and every rubble slab an ISM instance, so the mask sees through all of them and
reports survivors underneath as fully visible. Measured across the two completed passes, 18 of 313 boxes sit
on scenery with no subject visible in RGB at all.

The fix needs the depth buffer, which does render that geometry - but depth is pure geometry. It does not
depend on the sun angle, the weather, or when the frame was taken, and this scene is static. `telemetry.csv`
already records the exact camera pose of every frame. So the missing channel can be recovered by returning
the camera to a recorded pose and capturing depth alone: the RGB, the masks and the flight all stay exactly
as they were.

`simSetVehiclePose` is used deliberately here, and this is the one job it is right for. `survey.py` documents
at length why teleporting is wrong DURING a capture - it drove the airframe to 2.2e8 rad/s and filled a
731-frame dataset with the drone's own propellers. None of that applies to parking a stationary camera for a
single geometry read.

IT VERIFIES ITS OWN POSE RATHER THAN TRUSTING IT. A pose set while the sim is running lands one call late
(docs/CONTEXT.md section 7), so a frame captured at the wrong place would be silently mislabelled - the exact
class of failure this project keeps being bitten by. Every revisit therefore captures the SEGMENTATION mask
as well and requires it to reproduce the stored one: the actor's own silhouette must match at IoU >= MIN_IOU.
A frame that fails is refused and left untouched, never guessed at.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "scene"))

from sightline.mission.pattern import Scenario, connect, grab  # noqa: E402
from tools.capture.check_mask_visibility import RIM_PX, contrast_de  # noqa: E402
from tools.capture.labels import DEPTH_SLACK_ABOVE_M, DEPTH_SLACK_BELOW_M  # noqa: E402

MIN_IOU = 0.90        # the revisited mask must reproduce the stored silhouette this well to be trusted
SETTLE_S = 0.06
PARK_TOL_M = 0.05     # how exactly the camera must return to the recorded pose
PARK_TRIES = 8


def park_at(c, airsim, x: float, y: float, z: float, q, *, tol_m: float = PARK_TOL_M,
            tries: int = PARK_TRIES) -> tuple[bool, float, object]:
    """Put the camera back on a recorded pose and FREEZE it there. Returns (ok, error_m, kinematics).

    Two things fight this, and both were measured rather than guessed:

    * **Gravity.** `simSetVehiclePose` teleports but does not suspend physics, and `grab()` spends several
      hundred milliseconds pulling two 4K buffers. Left running, the drone fell 14.6 m between the pose and
      the shutter - which moved the target 63 px and put the silhouette IoU at exactly 0.000.
    * **The one-tick lag.** A pose set while the sim runs lands one call late (docs/CONTEXT.md section 7),
      so a single set-and-read reports the previous position.

    So: set the pose while running (twice, for the lag), pause to stop the fall, then MEASURE where the
    camera actually is and re-aim by the residual. Pausing alone still left 0.87 m of drop - 2.5 % of a 35 m
    altitude, which is a 24 px scale error at the frame edge. The loop closes that to `tol_m`.
    """
    err, st = float("inf"), None
    dz = 0.0
    for _ in range(tries):
        c.simPause(False)
        pose = airsim.Pose(airsim.Vector3r(x, y, z + dz), q)
        c.simSetVehiclePose(pose, True)
        time.sleep(SETTLE_S)
        c.simSetVehiclePose(pose, True)          # the second set is the one that lands
        c.simPause(True)
        st = c.simGetGroundTruthKinematics()
        ex, ey, ez = st.position.x_val - x, st.position.y_val - y, st.position.z_val - (z + dz)
        err = float((ex * ex + ey * ey + ez * ez) ** 0.5)
        if err <= tol_m:
            return True, err, st
        dz -= ez                                  # aim high by exactly what the fall cost us
    return False, err, st
MIN_VISIBLE_FRACTION = 0.04   # below this the "visible" pixels are specks through gaps, not a view of a body
MIN_VISIBLE_PX = 12


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--de-max", type=float, default=8.0,
                    help="revisit frames holding a box whose RGB silhouette contrast is below this")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run

    pal = json.loads((run / "segmentation_palette.json").read_text())["actor_rgb"]
    tel = {int(r["frame_idx"]): r
           for r in csv.DictReader((run / "telemetry.csv").open(newline="", encoding="utf-8"))}

    # --- which frames need a revisit -------------------------------------------------------------------
    suspects: dict[str, list[dict]] = {}
    for lf in sorted(glob.glob(str(run / "labels" / "*.json"))):
        labs = json.loads(Path(lf).read_text())
        if not labs:
            continue
        stem = Path(lf).stem
        img = cv2.imread(str(run / "images" / f"{stem}.jpg"))[:, :, ::-1]
        msk = cv2.imread(str(run / "masks" / f"{stem}.png"))[:, :, ::-1]
        for L in labs:
            x1, y1, x2, y2 = L["bbox_px"]
            p = RIM_PX + 2
            X1, Y1 = max(0, x1 - p), max(0, y1 - p)
            X2, Y2 = min(img.shape[1] - 1, x2 + p), min(img.shape[0] - 1, y2 + p)
            actor = np.all(msk[Y1:Y2 + 1, X1:X2 + 1]
                           == np.array(pal[str(L["actor_id"])], np.uint8), axis=2)
            de, _, _ = contrast_de(np.ascontiguousarray(img[Y1:Y2 + 1, X1:X2 + 1]), actor)
            if np.isfinite(de) and de < a.de_max:
                suspects.setdefault(stem, []).append({"label": L, "de": float(de)})
    if not suspects:
        print(f"no frame in {run.name} holds a box below dE {a.de_max} - nothing to revisit")
        return 0
    print(f"{len(suspects)} frame(s) to revisit in {run.name}, "
          f"{sum(len(v) for v in suspects.values())} suspect box(es)")

    scn = Scenario.load()
    home, truth = scn.home, scn.truth
    by_id = {x["id"]: x for x in truth["actors"]}

    import cosysairsim as airsim  # noqa: PLC0415

    c = connect()
    c.enableApiControl(True)
    grab(c, want_depth=True)                                   # warm-up; proves ImageType 1 is live

    changed, refused, report = 0, [], []
    for stem in sorted(suspects):
        idx = int(stem.rsplit("_", 1)[1])
        t = tel.get(idx)
        if t is None:
            refused.append((stem, "no telemetry row"))
            continue
        x = float(t["north_m"]) - home["north_m"]
        y = float(t["east_m"]) - home["east_m"]
        z = -(float(t["alt_msl_m"]) - home["ground_asl_m"])
        q = airsim.Quaternionr(float(t["q_x"]), float(t["q_y"]), float(t["q_z"]), float(t["q_w"]))
        stored = cv2.imread(str(run / "masks" / f"{stem}.png"))[:, :, ::-1]
        probe = np.array(pal[str(suspects[stem][0]["label"]["actor_id"])], np.uint8)
        want_m = np.all(stored == probe, axis=2)
        gsd = float(t["gsd_cm_px"]) / 100.0

        # SOLVE for the shutter pose rather than trusting the recorded one. `survey.py` samples
        # `simGetGroundTruthKinematics()` at the top of its loop and only then calls `grab()`, which spends
        # a few hundred milliseconds pulling two 4K buffers - at 11 m/s the aircraft has moved by the time
        # the shutter actually fires. The residual is sub-metre, but on a 120 px target a 0.3 m error is a
        # 24 px shift and IoU collapses. So: park, look at where the silhouette landed against the stored
        # one, and re-aim by that difference. Image axes are x = +East/gsd, y = -North/gsd (measured), and
        # AirSim NED is x = north, y = east.
        ok, err, st, g, got_iou, shift = False, float("inf"), None, None, 0.0, (0.0, 0.0)
        dx_m, dy_m = 0.0, 0.0
        for attempt in range(3):
            ok, err, st = park_at(c, airsim, x + dx_m, y + dy_m, z, q)
            try:
                if not ok:
                    break
                g = grab(c, want_depth=True)
            finally:
                c.simPause(False)
            have = np.all(g["seg"] == probe, axis=2)
            if not have.any() or not want_m.any():
                break
            got_iou = iou(want_m, have)
            if got_iou >= MIN_IOU:
                break
            wy, wx = np.where(want_m)
            hy, hx = np.where(have)
            sx = float(hx.mean() - wx.mean())          # revisited minus stored, in pixels
            sy = float(hy.mean() - wy.mean())
            shift = (sx, sy)
            dy_m += sx * gsd                           # actor too far EAST -> move the camera east
            dx_m -= sy * gsd                           # actor too far SOUTH -> move the camera south
        if not ok:
            refused.append((stem, f"could not park the camera: {err:.2f} m off the recorded pose"))
            print(f"  {stem[-5:]} REFUSED: camera parked {err:.2f} m off the recorded pose")
            continue
        if g is None:
            refused.append((stem, "no frame captured"))
            continue

        # The altitude the depth window is measured against must be where the camera ACTUALLY is, not where
        # the telemetry says it was: a residual of a few centimetres is fine, but reading one and using the
        # other is how a systematic bias creeps in unnoticed.
        cam_alt = home["ground_asl_m"] - st.position.z_val
        labs = json.loads((run / "labels" / f"{stem}.json").read_text())

        # --- prove the camera is where the telemetry says before believing anything it renders ---------
        bad_pose = []
        for s in suspects[stem]:
            col = np.array(pal[str(s["label"]["actor_id"])], np.uint8)
            got = iou(np.all(stored == col, axis=2), np.all(g["seg"] == col, axis=2))
            if got < MIN_IOU:
                bad_pose.append((s["label"]["name"], got))
        if bad_pose:
            refused.append((stem, f"mask IoU {['%s %.2f' % b for b in bad_pose]}"))
            print(f"  {stem[-5:]} REFUSED: revisited mask does not reproduce the stored one {bad_pose}")
            continue

        for s in suspects[stem]:
            L = s["label"]
            act = by_id[L["actor_id"]]
            col = np.array(pal[str(L["actor_id"])], np.uint8)
            amodal = np.all(stored == col, axis=2)
            d_ground = cam_alt - float(act["base_asl_m"])
            body = float(act["height_cm"]) / 100.0
            vis = (amodal & np.isfinite(g["depth"])
                   & (g["depth"] >= d_ground - body - DEPTH_SLACK_ABOVE_M)
                   & (g["depth"] <= d_ground + DEPTH_SLACK_BELOW_M))
            n_am, n_vis = int(amodal.sum()), int(vis.sum())
            vf = n_vis / n_am if n_am else 0.0
            hide = (n_vis < MIN_VISIBLE_PX) or (vf < MIN_VISIBLE_FRACTION)

            tgt = next(q for q in labs if q["actor_id"] == L["actor_id"])
            tgt["visible_fraction"] = round(vf, 4)
            tgt["occlusion"] = 0 if vf >= 0.99 else (1 if vf >= 0.5 else 2)
            tgt["occlusion_basis"] = "measured from the DepthPlanar buffer at the recorded camera pose"
            if hide:
                # NOT deleted. Section 6.3's `ignore` is exactly this case - neither a recall target nor a
                # false positive - and sightline/eval/groundtruth.py already honours it.
                tgt["ignore"] = True
                tgt["ignore_reason"] = (
                    "no line of sight: the instance mask is blind to instanced foliage and rubble "
                    "(Cosys-AirSim disables the InstancedFoliage/InstancedGrass show flags), so this box "
                    "was written over an occluder. Depth at the recorded pose shows "
                    f"{100 * vf:.1f}% of the body reaching the camera.")
            else:
                ys, xs = np.where(vis)
                tgt["amodal_bbox_px"] = list(tgt["bbox_px"])
                tgt["bbox_px"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                tgt["visible_px"] = n_vis
                tgt["size_px"] = int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            report.append(dict(stem=stem, name=L["name"], pose=L["pose"], de=round(s["de"], 2),
                               vf=round(vf, 4), amodal_px=n_am, visible_px=n_vis,
                               action="ignore" if hide else "tightened"))
            print(f"  {stem[-5:]} {L['name']:11s} {L['pose']:14s} dE{s['de']:5.2f}  "
                  f"vf={vf:6.3f} ({n_vis}/{n_am} px)  -> {'IGNORE' if hide else 'tightened'}")

        if a.write:
            (run / "labels" / f"{stem}.json").write_text(json.dumps(labs, indent=1), encoding="utf-8")
            changed += 1

    outp = run / "depth_refinement.json"
    outp.write_text(json.dumps({
        "by": "tools/capture/refine_labels_with_depth.py",
        "why": "the instance mask cannot see instanced foliage or rubble; depth can. Re-derived at the "
               "camera poses already recorded in telemetry.csv - no re-flight, no new RGB or masks.",
        "de_max": a.de_max, "min_visible_fraction": MIN_VISIBLE_FRACTION, "min_visible_px": MIN_VISIBLE_PX,
        "mask_iou_required": MIN_IOU, "written": bool(a.write),
        "refused": [{"stem": s, "why": w} for s, w in refused], "boxes": report,
    }, indent=1), encoding="utf-8")

    n_ign = sum(1 for r in report if r["action"] == "ignore")
    print(f"\n{len(report)} box(es) re-derived: {n_ign} flagged ignore, {len(report) - n_ign} tightened")
    if refused:
        print(f"{len(refused)} frame(s) REFUSED - pose could not be reproduced, labels left untouched:")
        for s, w in refused:
            print(f"  {s}: {w}")
    print(f"audit trail -> {outp}")
    if a.write:
        print(f"written: {changed} label file(s)")
    else:
        print("(report only - pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
