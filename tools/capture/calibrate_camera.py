"""Calibrate the survey camera's focal length against known survivor positions, and validate the nadir
projection chain at the same time (SOLUTION_DOC 5.7 step 1).

    uv run python tools/capture/calibrate_camera.py

Why this exists: `simGetCameraInfo("survey").fov` reports 89.9 deg while `sim/settings/dataset.json` asks for
75.5, and measuring a building's roof gives a third answer because a two-storey roof is ~6.5 m nearer the camera
than the flood surface the altitude is quoted against. Every geolocation number descends from f_px, so it is
measured here rather than taken on trust.

Method: teleport the nadir camera to a set of poses, read each survivor's instance-mask centroid, and least-
squares fit f_px to the exact ground truth in `data/scene/actors.json`:

    u = cx + f * (east_actor  - east_cam ) / d
    v = cy - f * (north_actor - north_cam) / d          d = alt_cam - asl_actor

The residual of that fit is a direct check of the projection chain: if the coordinate convention, the camera
mounting or the altitude reference were wrong, the residual would be large and structured rather than sub-pixel.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.capture.labels import actor_index, palette_rgb  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import cosysairsim as airsim


def main() -> int:
    scene = json.loads((REPO / "data/scene/flood_valley.json").read_text())
    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    home = scene["launch_site"]
    water = truth["water_level_m"]
    by_id = {a["id"]: a for a in truth["actors"]}

    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    pal = palette_rgb(cmap)
    idx = actor_index(names)
    req = [airsim.ImageRequest("survey", airsim.ImageType.Segmentation, False, False)]
    with contextlib.redirect_stdout(io.StringIO()):
        c.simGetImages(req)                                   # warm-up frame, discarded

    # Aim at a spread of survivors so the fit sees the whole field, not just the centre.
    aims = [a for a in truth["actors"] if a["aerially_detectable"]][::7][:12]
    rows = []
    for a in aims:
        for alt in (45.0, 60.0):
            cam_n, cam_e = a["north_m"], a["east_m"]
            cam_asl = water + alt
            ned = (cam_n - home["north_m"], cam_e - home["east_m"], -(cam_asl - home["ground_asl_m"]))
            c.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(*ned), airsim.euler_to_quaternion(0, 0, 0)), True)
            time.sleep(0.35)
            c.simPause(True)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = c.simGetImages(req)[0]
            finally:
                c.simPause(False)
            m = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
            H, W = m.shape[:2]
            for i, (aid, _nm, _cls) in idx.items():
                t = by_id.get(aid)
                if t is None:
                    continue
                mask = np.all(m == pal[i], axis=2)
                n_px = int(mask.sum())
                if n_px < 25:                                  # too few pixels for a stable centroid
                    continue
                ys, xs = np.where(mask)
                u, v = float(xs.mean()), float(ys.mean())
                # Project the actor's CENTROID height, not its feet: the mask centroid sits at mid-body.
                asl = t["base_asl_m"] + 0.5 * t["height_cm"] / 100.0
                d = cam_asl - asl
                if d <= 5.0:
                    continue
                rows.append((u - W / 2.0, v - H / 2.0,
                             (t["east_m"] - cam_e) / d, (t["north_m"] - cam_n) / d, n_px, alt))
    if len(rows) < 20:
        print(f"only {len(rows)} observations - not enough to calibrate")
        return 1

    du = np.array([r[0] for r in rows]); dv = np.array([r[1] for r in rows])
    ge = np.array([r[2] for r in rows]); gn = np.array([r[3] for r in rows])
    # single focal length, both axes: u = f*ge, v = -f*gn
    A = np.concatenate([ge, -gn]); b = np.concatenate([du, dv])
    f = float(A @ b / (A @ A))
    res = b - f * A
    rms = float(np.sqrt(np.mean(res ** 2)))
    W = int(r.width); H = int(r.height)
    hfov = 2 * math.degrees(math.atan((W / 2.0) / f))
    vfov = 2 * math.degrees(math.atan((H / 2.0) / f))
    print(f"observations: {len(rows)} over {len(aims)} aim points x 2 altitudes")
    print(f"calibrated f_px = {f:.2f}   residual RMS = {rms:.2f} px   max |res| = {np.abs(res).max():.1f} px")
    print(f"  -> HFOV {hfov:.3f} deg, VFOV {vfov:.3f} deg  (frame {W}x{H})")
    print(f"  simGetCameraInfo reports {float(c.simGetCameraInfo('survey').fov):.3f} deg "
          f"= f_px {(W / 2.0) / math.tan(math.radians(float(c.simGetCameraInfo('survey').fov)) / 2):.1f}")
    for a_deg in (75.5,):
        print(f"  settings FOV_Degrees {a_deg} = f_px {(W / 2.0) / math.tan(math.radians(a_deg) / 2):.1f}")
    for alt in (45.0, 60.0):
        print(f"  GSD at {alt:.0f} m = {alt / f * 100:.3f} cm/px, "
              f"footprint {W * alt / f:.1f} x {H * alt / f:.1f} m")

    out = {
        "camera": "survey", "width": W, "height": H,
        "f_px": round(f, 3), "cx": W / 2.0, "cy": H / 2.0,
        "hfov_deg": round(hfov, 4), "vfov_deg": round(vfov, 4),
        "residual_rms_px": round(rms, 3), "observations": len(rows),
        "method": "least squares against data/scene/actors.json ground truth, nadir pinhole",
        "note": ("simGetCameraInfo().fov is NOT the rendered FOV for this camera; use f_px from this file. "
                 "Residual RMS is also the validation of the projection chain."),
    }
    (REPO / "data/scene/camera_survey.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to data/scene/camera_survey.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
