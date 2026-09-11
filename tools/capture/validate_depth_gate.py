"""Fly to named survivors and prove the depth gate keeps the visible ones and drops the hidden ones. PIE ON.

    uv run python tools/capture/validate_depth_gate.py
    uv run python tools/capture/validate_depth_gate.py --alt 35 --ids 20,44,49,53

This is the gate on the gate. `apply_depth_visibility` is about to decide which survivors exist in a
re-captured dataset, so before spending hours re-flying, it has to be shown working against the actual
renderer on actual survivors - not against the synthetic frames in `tests/test_depth_visibility.py`.

The expectation is stated per survivor BEFORE the flight, from `check_mask_visibility.py`'s measurement of
the silhouette contrast in the already-captured seed23_alt35 run:

  * survivors whose boxes sat on pure foliage with no edge in RGB (dE < 5) MUST be dropped;
  * survivors that were plainly visible in the same run MUST be kept, with visible_fraction > 0.

A gate that drops everything would also "fix" the foliage boxes, and would be worthless. Both halves matter.
"""

from __future__ import annotations

import argparse
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
from tools.capture.labels import (apply_depth_visibility, attach_truth,  # noqa: E402
                                  labels_from_mask)

#: Measured on seed23_alt35 by tools/capture/check_mask_visibility.py. HIDDEN = the box sat on leaf texture
#: with no silhouette in RGB; VISIBLE = a person was plainly there.
EXPECT = {20: "hidden", 53: "hidden", 49: "hidden", 44: "visible", 12: "visible", 13: "visible"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=35.0)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--ids", default="", help="comma-separated actor ids (default: the measured set)")
    ap.add_argument("--settle-s", type=float, default=1.5)
    a = ap.parse_args()

    want = ([int(x) for x in a.ids.split(",") if x.strip()] if a.ids else sorted(EXPECT))
    scn = Scenario.load()
    home, truth = scn.home, scn.truth
    surface_asl = scn.terrain.surface_asl
    by_id = {x["id"]: x for x in truth["actors"]}
    missing = [i for i in want if i not in by_id]
    if missing:
        print(f"FAIL: actor ids not in actors.json: {missing}")
        return 2

    c = connect()
    names = c.simListInstanceSegmentationObjects()
    cmap = c.simGetSegmentationColorMap()
    grab(c, want_depth=True)                                  # warm-up; also proves ImageType 1 is live

    c.enableApiControl(True)
    c.armDisarm(True)
    print(f"taking off to {a.alt:.0f} m AGL ...")
    c.takeoffAsync(timeout_sec=30).join()

    out = REPO / "_artifacts" / "depth_gate_validation"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tiles: list[np.ndarray] = []

    try:
        for aid in want:
            act = by_id[aid]
            e, n = float(act["east_m"]), float(act["north_m"])
            z = -(surface_asl(e, n) + a.alt - home["ground_asl_m"])
            c.moveToPositionAsync(n - home["north_m"], e - home["east_m"], z, a.speed,
                                  timeout_sec=180).join()
            time.sleep(a.settle_s)                            # let the gimbal settle before the shutter

            st = c.simGetGroundTruthKinematics()
            asl = home["ground_asl_m"] - st.position.z_val
            g = grab(c, want_depth=True)
            raw = attach_truth(labels_from_mask(g["seg"], names, cmap), truth)
            mine_raw = [m for m in raw if m.actor_id == aid]
            kept, hidden = apply_depth_visibility(raw, g["seg"], g["depth"],
                                                  cam_alt_asl_m=asl, actors_json=truth)
            mine_k = [m for m in kept if m.actor_id == aid]
            mine_h = [m for m in hidden if m.actor_id == aid]

            if not mine_raw:
                verdict, vf, amodal = "not-in-mask", None, 0
            else:
                amodal = int(np.all(g["seg"] == np.array(mine_raw[0].rgb, np.uint8), axis=2).sum())
                m = (mine_k or mine_h)[0]
                vf = m.visible_fraction
                verdict = "kept" if mine_k else "dropped"

            exp = EXPECT.get(aid, "?")
            ok = (exp == "hidden" and verdict == "dropped") or (exp == "visible" and verdict == "kept")
            rows.append(dict(id=aid, pose=act["pose"], zone=act["zone"], sub=act["submersion"],
                             expect=exp, verdict=verdict, vf=vf, amodal_px=amodal, ok=ok))
            print(f"  Human_{aid:03d} {act['pose']:14s} {act['zone']:10s} expect {exp:8s} -> {verdict:12s} "
                  f"vf={'n/a' if vf is None else f'{vf:.3f}'}  amodal {amodal} px  "
                  f"{'OK' if ok else '<-- MISMATCH'}")

            if mine_raw:
                x1, y1, x2, y2 = mine_raw[0].amodal_bbox_px or mine_raw[0].bbox_px
                p = 40
                X1, Y1 = max(0, x1 - p), max(0, y1 - p)
                X2, Y2 = min(g["seg"].shape[1] - 1, x2 + p), min(g["seg"].shape[0] - 1, y2 + p)
                rgb = g["scene"][Y1:Y2 + 1, X1:X2 + 1, ::-1]
                msk = g["seg"][Y1:Y2 + 1, X1:X2 + 1, ::-1]
                dep = g["depth"][Y1:Y2 + 1, X1:X2 + 1]
                lo, hi = np.percentile(dep[np.isfinite(dep)], [2, 98]) if np.isfinite(dep).any() else (0, 1)
                dv = cv2.applyColorMap(
                    (np.clip((dep - lo) / max(1e-6, hi - lo), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_TURBO)
                t = np.hstack([cv2.resize(rgb, (200, 200)),
                               cv2.resize(msk, (200, 200), interpolation=cv2.INTER_NEAREST),
                               cv2.resize(dv, (200, 200))])
                cv2.putText(t, f"{aid} {exp}->{verdict}", (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
                tiles.append(t)
    finally:
        with open(out / "results.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        try:
            c.landAsync(timeout_sec=30)
            c.armDisarm(False)
            c.enableApiControl(False)
        except Exception:                                     # noqa: BLE001
            pass

    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(out / "gate_validation.png"), grid)
        print(f"\nwrote {out / 'gate_validation.png'}  (per tile: RGB | mask | depth) - LOOK AT IT")

    bad = [r for r in rows if not r["ok"]]
    n_kept = sum(1 for r in rows if r["verdict"] == "kept")
    print(f"\n{len(rows)} survivors: {n_kept} kept, {len(rows) - n_kept} dropped/absent")
    if not n_kept:
        print("FAIL: the gate kept nothing. A gate that deletes every survivor also removes the foliage\n"
              "      boxes and is still useless. Both halves have to work.")
        return 1
    if bad:
        print(f"FAIL: {len(bad)} survivor(s) did not match the pre-stated expectation: "
              f"{[r['id'] for r in bad]}")
        return 1
    print("PASS: every survivor matched the expectation measured independently from RGB silhouette contrast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
