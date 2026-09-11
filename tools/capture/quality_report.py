"""Decide whether a captured dataset is fit to TRAIN on, and say why not when it isn't.

    uv run python tools/capture/quality_report.py _artifacts/dataset/<run>

`validate.py` answers "is this dataset corrupt?". This answers the harder question: "will training on it
produce a model worth having?" A dataset can be perfectly self-consistent and still be useless - too few
targets, all of one posture, targets below the resolvable size, or so much frame-to-frame overlap that a
random split leaks the same survivor into train and val.

Everything here is reported per SOLUTION_DOC's slice grid and stamped `domain=sim` (hard rule 5). Nothing is
averaged across domains, and no number is emitted without its sample size.

Verdicts:
  FIT        train on it
  MARGINAL   trainable, but a named weakness will show up in the evaluation - proceed knowingly
  UNFIT      do not train; the reasons are listed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MIN_TRAINABLE_PX = 20      # SOLUTION_DOC 5.5: the ">= 20 px on target" rule
MIN_BOXES = 300            # below this a fine-tune cannot learn a class
MIN_UNIQUE_ACTORS = 25     # variety of subject, not just variety of frames
MAX_HUMAN_M = 3.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    ap.add_argument("--in-campaign", action="store_true",
                    help="this run is one pass of a multi-run campaign, so the two CAMPAIGN-scope checks "
                         "(pooled box count, >=2 scenario seeds) are reported here and enforced by "
                         "dataset_gate.py across all the runs, where they can actually be evaluated")
    a = ap.parse_args()
    in_campaign = bool(a.in_campaign)
    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run
    from tools.capture.labels import truth_for_run  # noqa: PLC0415
    truth, _truth_src = truth_for_run(run)
    print(f"ground truth: {_truth_src}")
    by_id = {x["id"]: x for x in truth["actors"]}
    detset = {x["id"] for x in truth["actors"] if x["aerially_detectable"]}
    card = json.loads((run / "data_card.json").read_text()) if (run / "data_card.json").exists() else {}

    labels = sorted(glob.glob(str(run / "labels" / "*.json")))
    imgs = sorted(glob.glob(str(run / "images" / "*.png"))
                  + glob.glob(str(run / "images" / "*.jpg")))
    if not labels:
        print("UNFIT: no labels")
        return 1

    boxes = []
    per_frame = []
    for lj in labels:
        L = json.loads(Path(lj).read_text())
        per_frame.append((Path(lj).stem, len(L)))
        for b in L:
            boxes.append(b)

    print(f"=== dataset quality report: {run.name}   domain=sim ===\n")
    print(f"frames {len(imgs)}   labelled boxes {len(boxes)}   "
          f"frames with >=1 box {sum(1 for _, n in per_frame if n)} "
          f"({100 * sum(1 for _, n in per_frame if n) / max(1, len(per_frame)):.0f}%)")

    fails, warns = [], []

    # --- 1. is there enough signal at all? -----------------------------------------------------------------
    # An `ignore` box is neither a training target nor a recall target (section 6.3), so it cannot make a
    # buried survivor "found" and cannot count towards volume. Actor 55 is genuinely buried and genuinely
    # covered - check_occlusion_truth.py confirms "everybody buried is covered" - but the instance mask is
    # blind to the ISM rubble slab on top, so the box exists and is flagged. validate.py already skips these;
    # this file did not, which failed a run whose ground truth is correct.
    seen = Counter(b["actor_id"] for b in boxes if not b.get("ignore"))
    n_uniq = len(set(seen) & detset)
    print(f"\n1  VOLUME")
    print(f"   boxes {len(boxes)} (need >= {MIN_BOXES})")
    print(f"   unique survivors seen {n_uniq}/{len(detset)} (need >= {MIN_UNIQUE_ACTORS})")
    if seen:
        obs = np.array(list(seen.values()))
        print(f"   observations per survivor: min {obs.min()} median {int(np.median(obs))} max {obs.max()}")
    if len(boxes) < MIN_BOXES:
        # SCOPE. The fine-tune trains on the POOLED campaign, not on one flight, so a per-run floor asks a
        # question this run cannot answer. `dataset_gate.py` enforces the campaign-wide floor across runs.
        # Failing it here failed runs whose data was correct, which is how a gate teaches people to ignore it.
        (warns if in_campaign else fails).append(
            f"only {len(boxes)} boxes; a fine-tune needs >= {MIN_BOXES}"
            + (" [campaign scope: dataset_gate.py checks the pooled total]" if in_campaign else ""))
    if n_uniq < MIN_UNIQUE_ACTORS:
        fails.append(f"only {n_uniq} distinct survivors ever seen; the model would memorise a handful of people")

    # --- 2. are the targets resolvable? --------------------------------------------------------------------
    px = np.array([b.get("size_px", 0) for b in boxes], dtype=float)
    gsd = card.get("gsd_cm_px_nominal") or card.get("gsd_cm_px")
    print(f"\n2  TARGET SIZE  (SOLUTION_DOC 5.5: >= {MIN_TRAINABLE_PX} px on target)")
    if len(px):
        print(f"   px: min {px.min():.0f}  p10 {np.percentile(px,10):.0f}  p50 {np.median(px):.0f}  "
              f"p90 {np.percentile(px,90):.0f}  max {px.max():.0f}")
        frac_ok = float((px >= MIN_TRAINABLE_PX).mean())
        print(f"   at or above {MIN_TRAINABLE_PX} px: {100*frac_ok:.1f}%")
        if gsd:
            m = px * gsd / 100.0
            print(f"   implied ground size: p50 {np.median(m):.2f} m  max {m.max():.2f} m")
            impossible = int((m > MAX_HUMAN_M).sum())
            if impossible:
                fails.append(f"{impossible} boxes imply a person larger than {MAX_HUMAN_M} m on the ground")
        if frac_ok < 0.5:
            fails.append(f"only {100*frac_ok:.0f}% of targets reach {MIN_TRAINABLE_PX} px - most are below the "
                         "resolvable limit, so recall would be capped by physics, not by the model")
        elif frac_ok < 0.8:
            warns.append(f"{100*(1-frac_ok):.0f}% of targets are under {MIN_TRAINABLE_PX} px")

    # --- 3. class balance across the slice grid ------------------------------------------------------------
    print("\n3  BALANCE  (n boxes / n distinct survivors)")
    for key in ("pose", "submersion", "zone", "occlusion"):
        c_box, c_act = Counter(), defaultdict(set)
        for b in boxes:
            v = str(b.get(key))
            c_box[v] += 1
            c_act[v].add(b["actor_id"])
        print(f"   by {key}:")
        for v, n in c_box.most_common():
            print(f"      {v:18s} {n:5d} boxes / {len(c_act[v]):3d} survivors")
        missing = []
        if key == "pose":
            missing = [p["pose"] for p in truth["actors"]
                       if p["aerially_detectable"] and p["pose"] not in c_box]
        if missing:
            warns.append(f"postures never captured: {sorted(set(missing))}")

    # --- 4. split integrity: can we make an honest val set? ------------------------------------------------
    print("\n4  SPLIT INTEGRITY  (SOLUTION_DOC 5.5c: split by scenario seed, never by frame)")
    seeds = {card.get("scenario_seed", truth.get("seed"))}
    print(f"   scenario seeds present: {sorted(x for x in seeds if x is not None)}")
    if len(seeds) < 2:
        # SCOPE, same as the box floor above. `seeds` is read from ONE data card, so a single flight can
        # never hold two - this could not pass per-run however good the data was. "Is there an honest
        # held-out split?" is a campaign question and `dataset_gate.py` answers it across runs. Standalone
        # it still FAILS, because one run genuinely cannot be trained on.
        (warns if in_campaign else fails).append(
            "only ONE scenario seed exists, so there is no honest held-out split. Splitting these frames "
            "randomly leaks the same survivors, poses and terrain into val and inflates every number. "
            "Capture a second seed before training."
            + (" [campaign scope: dataset_gate.py checks seeds across runs]" if in_campaign else ""))

    # --- 5. frame-to-frame redundancy ---------------------------------------------------------------------
    print("\n5  REDUNDANCY")
    tele = run / "telemetry.csv"
    if tele.exists():
        rows = list(csv.DictReader(tele.open(newline="", encoding="utf-8")))
        pos = [(float(r["east_m"]), float(r["north_m"])) for r in rows]
        d = [math.dist(pos[i], pos[i + 1]) for i in range(len(pos) - 1)]
        if d:
            print(f"   consecutive shot spacing: median {np.median(d):.1f} m  min {min(d):.1f} m")
        agl = [float(r["agl_m"]) for r in rows if r.get("agl_m")]
        if agl:
            print(f"   AGL: min {min(agl):.1f}  median {np.median(agl):.1f}  max {max(agl):.1f} m")
            if max(agl) - min(agl) > 30:
                warns.append(f"AGL varies {min(agl):.0f}-{max(agl):.0f} m; scale varies more than the "
                             "nominal slice claims")
        gsds = [float(r["gsd_cm_px"]) for r in rows if r.get("gsd_cm_px")]
        if gsds and (max(gsds) - min(gsds)) / max(1e-9, np.median(gsds)) > 0.05:
            print(f"   per-frame GSD spans {min(gsds):.3f}-{max(gsds):.3f} cm/px")

    # --- 6. image quality ---------------------------------------------------------------------------------
    print("\n6  IMAGE QUALITY  (sampled)")
    step = max(1, len(imgs) // 40)
    means, blurs, clipped = [], [], []
    for p in imgs[::step]:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        means.append(float(g.mean()))
        blurs.append(float(cv2.Laplacian(cv2.resize(g, (960, 540)), cv2.CV_64F).var()))
        clipped.append(float((g >= 254).mean()))
    if means:
        print(f"   brightness: p10 {np.percentile(means,10):.0f} p50 {np.median(means):.0f} "
              f"p90 {np.percentile(means,90):.0f} (0-255)")
        print(f"   focus (laplacian var): p10 {np.percentile(blurs,10):.0f} p50 {np.median(blurs):.0f}")
        print(f"   blown highlights: p90 {100*np.percentile(clipped,90):.2f}% of pixels")
        if np.median(means) < 40:
            warns.append(f"frames are dark (median brightness {np.median(means):.0f}/255)")
        if np.percentile(clipped, 90) > 0.05:
            warns.append("more than 5% of pixels clipped to white in the brightest decile of frames")

    # --- 6b. near-duplicate frames -------------------------------------------------------------------------
    # Exact duplicates are caught by validate.py. NEAR-duplicates are the subtler problem: with heavy
    # along-track overlap two consecutive frames can be 95 % the same picture, which inflates the apparent
    # size of the dataset without adding information and, if they straddle a split, leaks. A 64-bit dHash
    # over an 8x8 downsample finds them cheaply; the threshold is Hamming distance, so 0 is "identical to
    # the eye" and <=3 is "the same view nudged".
    print("\n6b DUPLICATES")
    hashes = []
    for pth in imgs[::max(1, len(imgs) // 300)]:
        im = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        sm = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (sm[:, 1:] > sm[:, :-1]).flatten()
        hashes.append((Path(pth).name, int("".join("1" if b else "0" for b in bits), 2)))
    near = ident = 0
    for i in range(1, len(hashes)):
        dist = bin(hashes[i][1] ^ hashes[i - 1][1]).count("1")
        if dist == 0:
            ident += 1
        elif dist <= 3:
            near += 1
    n_pairs = max(1, len(hashes) - 1)
    print(f"   sampled {len(hashes)} frames: {ident} visually identical to the previous, "
          f"{near} near-identical (dHash distance <= 3)")
    if ident:
        fails.append(f"{ident} sampled frames are visually IDENTICAL to their predecessor - the camera did "
                     f"not move, so those frames add nothing and inflate the dataset")
    if (ident + near) / n_pairs > 0.30:
        warns.append(f"{100 * (ident + near) / n_pairs:.0f}% of consecutive frames are near-identical; the "
                     f"along-track overlap is buying repetition rather than new views")

    # --- 6c. the campaign's OWN slice axes ------------------------------------------------------------------
    # Altitude and lighting condition are slice axes (SOLUTION_DOC 5.12), and the whole point of a compact
    # campaign is that it covers SITUATIONS rather than piling up frames. If a run carries a data card, say
    # which situation it is, so a reader can see the coverage rather than assume it.
    if card:
        print("\n6c SITUATION")
        print(f"   condition {card.get('condition')}   time_of_day {card.get('time_of_day')}   "
              f"weather {card.get('weather')}")
        print(f"   altitude {card.get('altitude_m_agl')} m AGL   "
              f"nominal GSD {card.get('gsd_cm_px_nominal')} cm/px   plan {card.get('plan')}")

    # --- 7. ground-truth consistency ----------------------------------------------------------------------
    print("\n7  GROUND-TRUTH CONSISTENCY")
    bad_truth = 0
    for b in boxes:
        t = by_id.get(b["actor_id"])
        if t and (b.get("pose") != t["pose"] or b.get("submersion") != t["submersion"]):
            bad_truth += 1
    print(f"   label attributes disagreeing with actors.json: {bad_truth}")
    if bad_truth:
        fails.append(f"{bad_truth} labels disagree with data/scene/actors.json - the label writer and the "
                     "generator are out of step")
    leaked = sorted(set(seen) - detset)
    if leaked:
        fails.append(f"survivors marked aerially undetectable appear in labels: {leaked}")
    print(f"   buried survivors leaked into labels: {len(leaked)}")

    # --- verdict ------------------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    for w in warns:
        print(f"WARN   {w}")
    for f in fails:
        print(f"FAIL   {f}")
    verdict = "UNFIT" if fails else ("MARGINAL" if warns else "FIT")
    print(f"\nVERDICT: {verdict} for training  (domain=sim, n_boxes={len(boxes)}, n_survivors={n_uniq})")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"run": run.name, "verdict": verdict, "boxes": len(boxes), "unique_survivors": n_uniq,
             "fails": fails, "warns": warns, "domain": "sim"}, indent=1), encoding="utf-8")
    return 0 if verdict != "UNFIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
