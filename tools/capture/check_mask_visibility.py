"""Every labelled box must contain a subject that is actually VISIBLE in RGB, not only in the mask.

    uv run python tools/capture/check_mask_visibility.py _artifacts/dataset/<run>
    uv run python tools/capture/check_mask_visibility.py _artifacts/dataset/<run> --dump 12

`validate.py` check C proves the mask carries the actor's colour inside the box. That is necessary and it is
not sufficient. The instance-segmentation pass and the RGB pass are rendered by different material paths, and
alpha-tested foliage is the known divergence: a fern that writes depth and colour in RGB may not write the
segmentation buffer at all. When that happens the mask reports a survivor the camera cannot see, the
auto-labeller writes a confident box over pure leaf texture, and the detector is trained to hallucinate a
person from foliage. Nothing else in the pipeline would notice - the box is tight, the colour matches, the
size is plausible, and validate.py passes.

THE TEST. A subject that is really rendered has a **silhouette**: crossing the mask boundary in the RGB frame
changes the colour. A subject that is occluded but leaking into the mask does not - both sides of the boundary
are the same leaf. So this measures, per box, the mean colour step across the mask contour:

    rim_in   = RGB of pixels just INSIDE the actor mask, within RIM_PX of the contour
    rim_out  = RGB of pixels just OUTSIDE it, within RIM_PX
    contrast = || mean(rim_in) - mean(rim_out) ||  in CIE Lab, which is perceptually uniform so a dark-on-dark
               step is not scored as small merely because the scene is dark

This deliberately does NOT measure how different the person is from the background overall. A person in mud
colours lying on mud is a HARD case and belongs in the training set; the test must not throw it away. What it
catches is the case where there is no step at all because there is no person in the picture.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

RIM_PX = 2            # band width either side of the mask contour
MIN_RIM_PX = 12       # below this the rim is too small to average; the box is reported, not judged
FAIL_DE = 2.0         # CIE dE76 below which the boundary carries no step at all -> subject not visible
WARN_DE = 5.0         # a genuinely hard low-contrast case; expected and wanted, but counted


def contrast_de(rgb: np.ndarray, actor: np.ndarray) -> tuple[float, int, int]:
    """Mean Lab distance across the mask contour. `actor` is a bool mask the same HxW as `rgb`."""
    k = np.ones((2 * RIM_PX + 1, 2 * RIM_PX + 1), np.uint8)
    a8 = actor.astype(np.uint8)
    inner = cv2.erode(a8, k, iterations=1).astype(bool)
    outer = cv2.dilate(a8, k, iterations=1).astype(bool)
    rim_in = actor & ~inner            # inside the mask, near the edge
    rim_out = outer & ~actor           # outside the mask, near the edge
    n_in, n_out = int(rim_in.sum()), int(rim_out.sum())
    if n_in < MIN_RIM_PX or n_out < MIN_RIM_PX:
        return float("nan"), n_in, n_out
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(lab[rim_in].mean(0) - lab[rim_out].mean(0))
    return float(d), n_in, n_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dump", type=int, default=0, help="write the N worst boxes as RGB|mask pairs")
    ap.add_argument("--fail-de", type=float, default=FAIL_DE)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run

    pal_f = run / "segmentation_palette.json"
    if not pal_f.exists():
        print(f"FAIL: {pal_f} missing - cannot map actor ids to mask colours, so this check cannot run.")
        return 2
    pal = json.loads(pal_f.read_text())["actor_rgb"]

    rows: list[dict] = []
    no_rim = 0
    for lf in sorted(glob.glob(str(run / "labels" / "*.json"))):
        labs = json.loads(Path(lf).read_text())
        if not labs:
            continue
        stem = Path(lf).stem
        img = cv2.imread(str(run / "images" / f"{stem}.jpg"))
        msk = cv2.imread(str(run / "masks" / f"{stem}.png"))
        if img is None or msk is None:
            print(f"FAIL: frame or mask missing for {stem}")
            return 2
        img = img[:, :, ::-1]                                   # BGR -> RGB
        msk = msk[:, :, ::-1]
        for L in labs:
            if L.get("ignore") or L.get("uncertain"):
                continue                                        # judgement withheld by section 6.3
            x1, y1, x2, y2 = L["bbox_px"]
            p = RIM_PX + 2
            X1, Y1 = max(0, x1 - p), max(0, y1 - p)
            X2, Y2 = min(img.shape[1] - 1, x2 + p), min(img.shape[0] - 1, y2 + p)
            sub_i = np.ascontiguousarray(img[Y1:Y2 + 1, X1:X2 + 1])
            sub_m = msk[Y1:Y2 + 1, X1:X2 + 1]
            col = np.array(pal[str(L["actor_id"])], dtype=np.uint8)
            actor = np.all(sub_m == col, axis=2)
            de, n_in, n_out = contrast_de(sub_i, actor)
            if np.isnan(de):
                no_rim += 1
                continue
            rows.append(dict(stem=stem, name=L["name"], pose=L["pose"], sub=L["submersion"],
                             size=float(L.get("size_px", 0)), occ=L.get("occlusion", -1), de=de,
                             box=[X1, Y1, X2, Y2]))

    if not rows:
        print("FAIL: no box was large enough to measure. That is not a pass.")
        return 2

    de = np.array([r["de"] for r in rows])
    rows.sort(key=lambda r: r["de"])
    bad = [r for r in rows if r["de"] < a.fail_de]
    warn = [r for r in rows if a.fail_de <= r["de"] < WARN_DE]

    print(f"{len(rows)} scored boxes ({no_rim} too small for a rim, reported not judged)")
    print(f"  silhouette contrast dE76: min {de.min():.1f}  p05 {np.percentile(de, 5):.1f}  "
          f"median {np.median(de):.1f}  max {de.max():.1f}")
    print(f"  < {a.fail_de} (no edge at all -> subject not visible in RGB): {len(bad)}")
    print(f"  {a.fail_de}-{WARN_DE} (hard low-contrast case, wanted): {len(warn)}")
    print("")
    print("weakest 10:")
    for r in rows[:10]:
        print(f"  dE {r['de']:5.1f}  {r['stem'][-5:]}  {r['name']:11s} "
              f"{r['pose']:14s}/{r['sub']:10s} {r['size']:4.0f}px occ{r['occ']}")

    if a.dump:
        out = run / "visibility_worst.png"
        tiles = []
        for r in rows[:a.dump]:
            X1, Y1, X2, Y2 = r["box"]
            im = cv2.imread(str(run / "images" / f"{r['stem']}.jpg"))[Y1:Y2 + 1, X1:X2 + 1]
            mk = cv2.imread(str(run / "masks" / f"{r['stem']}.png"))[Y1:Y2 + 1, X1:X2 + 1]
            t = np.hstack([cv2.resize(im, (180, 180)),
                           cv2.resize(mk, (180, 180), interpolation=cv2.INTER_NEAREST)])
            cv2.putText(t, f"dE{r['de']:.1f} {r['pose'][:9]}", (3, 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out), grid)
        print("")
        print(f"wrote {out}  (per tile: RGB | mask) - LOOK AT IT")

    if bad:
        print("")
        print(f"FAIL: {len(bad)} box(es) have no silhouette edge in RGB. The mask claims a subject the")
        print( "      camera cannot see - most likely alpha-tested foliage that renders in RGB but not in")
        print( "      the segmentation pass. Training on these teaches the detector to invent people.")
        return 1
    print("")
    print("PASS: every scored box has a measurable silhouette in the RGB frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
