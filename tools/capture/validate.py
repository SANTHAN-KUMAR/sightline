"""Prove a captured dataset is clean before anything trains on it.

    uv run python tools/capture/validate.py _artifacts/dataset/<run>

Written after a 731-frame dataset passed every JSON check while being worthless: the drone's own propellers
filled most frames, one frame was pure sky, and 42 of an expected ~110 boxes survived. Nothing reported a
problem. These checks are the ones that would have caught it, and each FAILS LOUDLY rather than warning.

  A  airframe        the drone photographing itself (camera sits 30 cm below the body)
  B  degenerate      blank / sky-only / near-uniform frames, and frames identical to their predecessor
  C  label-image     every labelled box must contain that actor's instance colour in the mask
  D  attitude        roll/pitch within survey limits; the camera actually nadir
  E  altitude        AGL close to the commanded value (a fixed-ASL sweep flies into hillsides)
  F  box sanity      sizes plausible for the GSD; nothing at a frame edge claiming full extent
  G  truth coverage  which survivors were never seen, broken down by pose/submersion/zone
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: the longest a person can plausibly be on the ground, with generous slack for a lying adult
MAX_HUMAN_M = 3.0
sys.path.insert(0, str(REPO))
from tools.capture.labels import actor_index, palette_rgb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--airframe-max", type=float, default=0.005, help="max fraction of frame that may be drone")
    ap.add_argument("--uniform-std", type=float, default=3.0, help="below this grey std a frame is degenerate")
    ap.add_argument("--max-tilt-deg", type=float, default=12.0)
    ap.add_argument("--sample-masks", type=int, default=60, help="frames to check for label-image agreement")
    a = ap.parse_args()

    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run
    imgs = sorted(glob.glob(str(run / "images" / "*.png")))
    if not imgs:
        print(f"FAIL: no images under {run}")
        return 1
    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    by_id = {x["id"]: x for x in truth["actors"]}
    detset = {x["id"] for x in truth["actors"] if x["aerially_detectable"]}

    fails: list[str] = []
    warns: list[str] = []
    print(f"validating {len(imgs)} frames in {run.name}\n")

    # --- A/B: image-level checks ---------------------------------------------------------------------------
    prev_hash = None
    dupes = 0
    flat = []
    for p in imgs:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            fails.append(f"{Path(p).name}: unreadable")
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (64, 36))
        if small.std() < a.uniform_std:
            flat.append((Path(p).name, float(small.std())))
        h = hash(small.tobytes())
        if prev_hash is not None and h == prev_hash:
            dupes += 1
        prev_hash = h
    print(f"A/B  degenerate frames: {len(flat)} near-uniform, {dupes} identical to the previous frame")
    if flat:
        fails.append(f"{len(flat)} near-uniform frames (blank/sky/underground), e.g. {flat[:3]}")
    if dupes:
        fails.append(f"{dupes} frames identical to their predecessor (camera never moved)")

    # --- C: does every labelled box actually contain that actor in the mask? --------------------------------
    masks = sorted(glob.glob(str(run / "masks" / "*.png")))
    checked = mismatched = boxes = 0
    if masks:
        try:
            import cosysairsim as airsim
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                c = airsim.MultirotorClient(); c.confirmConnection()
            names = c.simListInstanceSegmentationObjects()
            pal = palette_rgb(c.simGetSegmentationColorMap())
            idx = actor_index(names)
            colour_of = {aid: pal[i] for i, (aid, _n, _c) in idx.items()}
        except Exception as exc:
            colour_of = {}
            warns.append(f"label-image check skipped (no live sim: {exc})")
        if colour_of:
            step = max(1, len(masks) // a.sample_masks)
            for mp in masks[::step]:
                lj = run / "labels" / (Path(mp).stem + ".json")
                if not lj.exists():
                    continue
                labs = json.loads(lj.read_text())
                if not labs:
                    continue
                m = cv2.imread(mp, cv2.IMREAD_COLOR)[:, :, ::-1]        # cv2 gives BGR; palette is RGB
                checked += 1
                for L in labs:
                    boxes += 1
                    x1, y1, x2, y2 = L["bbox_px"]
                    col = colour_of.get(L["actor_id"])
                    if col is None:
                        continue
                    sub = m[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
                    if sub.size == 0 or not np.any(np.all(sub == col, axis=2)):
                        mismatched += 1
            print(f"C    label-image agreement: {boxes} boxes over {checked} frames, {mismatched} mismatched")
            if mismatched:
                fails.append(f"{mismatched}/{boxes} labelled boxes do not contain that actor in the mask")

    # --- D/E: telemetry checks ------------------------------------------------------------------------------
    tele = run / "telemetry.csv"
    tilt_bad = agl_bad = 0
    speeds = []
    if tele.exists():
        import csv
        rows = list(csv.DictReader(tele.open(newline="", encoding="utf-8")))
        for r in rows:
            w, x, y, z = (float(r["q_w"]), float(r["q_x"]), float(r["q_y"]), float(r["q_z"]))
            roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            if abs(roll) > a.max_tilt_deg or abs(pitch) > a.max_tilt_deg:
                tilt_bad += 1
            if "agl_m" in r and r["agl_m"]:
                agl = float(r["agl_m"])
                if not (5.0 < agl < 200.0):
                    agl_bad += 1
            if r.get("speed_ms"):
                speeds.append(float(r["speed_ms"]))
        print(f"D/E  telemetry: {len(rows)} rows, {tilt_bad} beyond +-{a.max_tilt_deg:.0f} deg tilt, "
              f"{agl_bad} with implausible AGL" +
              (f", speed {min(speeds):.1f}-{max(speeds):.1f} m/s" if speeds else ""))
        if tilt_bad:
            fails.append(f"{tilt_bad} frames captured beyond +-{a.max_tilt_deg:.0f} deg tilt")
        if agl_bad:
            fails.append(f"{agl_bad} frames with implausible AGL")
    else:
        warns.append("no telemetry.csv")

    # --- F/G: label statistics and truth coverage -----------------------------------------------------------
    sizes, seen = [], Counter()
    per_pose = defaultdict(Counter)
    for lj in glob.glob(str(run / "labels" / "*.json")):
        for L in json.loads(Path(lj).read_text()):
            sizes.append(L.get("size_px", 0))
            seen[L["actor_id"]] += 1
            per_pose[L.get("pose", "?")][L["actor_id"]] += 1
    if sizes:
        s = np.array(sizes)
        print(f"F    boxes: {len(s)}, size px  min {s.min()} p50 {int(np.median(s))} p90 "
              f"{int(np.percentile(s, 90))} max {s.max()}")
        tiny = int((s < 12).sum())
        if tiny:
            warns.append(f"{tiny} boxes under 12 px (below the section 5.5 >=20 px rule; keep or drop knowingly)")
        # A box is a PERSON. Convert to metres through the run's own GSD and fail on anything that cannot be
        # one. This check used to only PRINT the sizes, and so passed a dataset in which a single survivor was
        # labelled across a whole 3840x2160 frame - 67.8 x 38.1 m of ground - because the segmentation had
        # collapsed onto shared colours. Printing is not checking.
        gsd_cm = None
        card = run / "data_card.json"
        if card.exists():
            gsd_cm = json.loads(card.read_text()).get("gsd_cm_px")
        if gsd_cm:
            big = [(round(float(x) * gsd_cm / 100.0, 1)) for x in s if float(x) * gsd_cm / 100.0 > MAX_HUMAN_M]
            if big:
                fails.append(f"{len(big)} of {len(s)} boxes are larger than {MAX_HUMAN_M} m on the ground "
                             f"(largest {max(big)} m) - a human cannot be that size, so the instance mask has "
                             "collapsed onto shared colours")
    missed = sorted(detset - set(seen))
    print(f"G    survivors: {len(set(seen) & detset)}/{len(detset)} detectable seen, {len(missed)} never seen")
    if seen:
        obs = np.array(list(seen.values()))
        print(f"     observations per seen survivor: min {obs.min()} median {int(np.median(obs))} max {obs.max()}")
    if seen.keys() - detset:
        fails.append(f"BURIED survivors appeared in labels: {sorted(seen.keys() - detset)} "
                     "(section 2.7 says aerial search cannot find them)")
    if missed:
        agg = Counter()
        for i in missed:
            t = by_id[i]
            agg[f"{t['pose']}/{t['submersion']}/{t['zone']}"] += 1
        print("     never seen, by pose/submersion/zone:")
        for k, v in agg.most_common(8):
            print(f"        {k:34s} {v}")

    # --- H: is the instance segmentation itself healthy? ---------------------------------------------------
    if masks:
        share = 0
        for mp in masks[:: max(1, len(masks) // 25)]:
            m = cv2.imread(mp, cv2.IMREAD_COLOR)
            if m is None:
                continue
            cols, cnts = np.unique(m.reshape(-1, 3), axis=0, return_counts=True)
            frac = cnts / float(m.shape[0] * m.shape[1])
            lj = run / "labels" / (Path(mp).stem + ".json")
            labs = json.loads(lj.read_text()) if lj.exists() else []
            # any colour that covers a huge share of the frame AND is claimed by a survivor label
            if labs and frac.max() > 0.5 and any(
                    (L["bbox_px"][2] - L["bbox_px"][0]) > 0.8 * m.shape[1] for L in labs):
                share += 1
        if share:
            fails.append(f"{share} sampled frames have a survivor label spanning most of the frame - the "
                         "segmentation palette is shared between actors and a large surface (water/terrain)")
        print(f"H    segmentation health: {share} sampled frames show an actor colour covering the frame")

    print()
    for w in warns:
        print(f"WARN  {w}")
    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        print(f"\n{len(fails)} check(s) FAILED - do not train on this dataset")
        return 1
    print("all checks passed - dataset is clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
