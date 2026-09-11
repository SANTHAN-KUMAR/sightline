"""Draw a contact sheet of captured frames with their labels overlaid, so the dataset can be EYEBALLED.

    uv run python tools/capture/contact_sheet.py _artifacts/dataset/<run> [--n 12]

This exists because a capture can pass every programmatic check and still be wrong. On 2026-09-10 a run
produced 731 frames whose object names, instance colours and box maths all verified, while the airframe was
tumbling at 2.2e8 rad/s and only 42 of an expected ~110 boxes were recovered. Nothing in the JSON said so.
Look at the sheet before trusting a dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=12, help="tiles on the sheet")
    ap.add_argument("--tile", type=int, default=520)
    ap.add_argument("--out", default="")
    ap.add_argument("--only-labelled", action="store_true", default=True)
    a = ap.parse_args()

    run = Path(a.run) if os.path.isabs(a.run) else REPO / a.run
    imgs = sorted(glob.glob(str(run / "images" / "*.png"))
                  + glob.glob(str(run / "images" / "*.jpg")))
    if not imgs:
        print(f"no images under {run}")
        return 1

    withbox, empty = [], []
    for p in imgs:
        lj = run / "labels" / (Path(p).stem + ".json")
        labs = json.loads(lj.read_text()) if lj.exists() else []
        (withbox if labs else empty).append((p, labs))
    print(f"{len(imgs)} frames: {len(withbox)} with boxes, {len(empty)} empty")

    # prefer the frames that actually contain survivors, then pad with a few negatives for context
    pick = withbox[:: max(1, len(withbox) // max(1, a.n - 2))][: a.n - 2] if withbox else []
    pick += empty[:: max(1, len(empty) // 2)][:2]
    pick = pick[: a.n]
    if not pick:
        print("nothing to draw")
        return 1

    cols = 4
    rows = (len(pick) + cols - 1) // cols
    T = a.tile
    sheet = np.full((rows * T, cols * T, 3), 32, np.uint8)
    for k, (p, labs) in enumerate(pick):
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        s = T / max(W, H)
        for m in labs:
            x1, y1, x2, y2 = m["bbox_px"]
            # draw on the full-res frame so thin boxes survive the downscale
            pad = max(6, int(0.25 * max(x2 - x1, y2 - y1)))
            cv2.rectangle(im, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 255, 0), max(2, int(3 / s)))
            cv2.putText(im, f"{m['pose']}/{m['submersion']} {m['size_px']}px",
                        (x1 - pad, max(24, y1 - pad - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, 1.2 / s), (0, 255, 0), max(2, int(3 / s)))
        small = cv2.resize(im, (int(W * s), int(H * s)))
        r, c = divmod(k, cols)
        y0, x0 = r * T + (T - small.shape[0]) // 2, c * T + (T - small.shape[1]) // 2
        sheet[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.putText(sheet, f"{Path(p).stem}  [{len(labs)}]", (c * T + 8, r * T + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    out = Path(a.out) if a.out else run / "contact_sheet.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("OPEN IT. A dataset that passes every JSON check can still be wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
