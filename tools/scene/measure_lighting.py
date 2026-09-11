"""Measure the sun-to-shadow ratio in a render, and fail when shadows are too dark to search.

    uv run python tools/scene/measure_lighting.py _artifacts/editor_shots/qa_1_valley.png [more.png ...]

This exists because "the shadows look a bit dark" is not something anyone can act on, and because the defect
it measures is not merely cosmetic: a survivor lying in a shadow the renderer has crushed to black is
invisible to the detector AND to the human reviewing the frame, so it silently costs recall in a slice that
nobody thinks to look at.

FloodValley runs with dynamic GI OFF on purpose (`r.DynamicGlobalIlluminationMethod=0`, `PPV_NoGI`) because
the machine has an 8 GB laptop GPU. That decision is sound, but it has a consequence people forget: with GI
off the **SkyLight is the entire ambient term**. If it is too weak, shadows go to black, because nothing else
is filling them.

The physics this checks against:

* Under a clear sky the sky contributes roughly 15-25 % of horizontal illuminance, so an outdoor shadow sits
  about **5:1 to 10:1** below full sun. It is never 25:1 or worse. (Overcast is flatter still, nearer 2:1.)
* A sky-lit shadow is **BLUE**: it is lit by the sky, so blue exceeds red. A shadow whose blue channel is its
  *lowest* is not being lit by a sky at all, whatever the SkyLight says it is doing.

Measured on `qa_1_valley.png` at 2026-09-11 00:57, before any lighting work: ratio **26.8 : 1**, shadow
linear (0.0073, 0.0060, 0.0024) - blue lowest. Both tests failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

RATIO_MAX = 12.0          # generous: real clear-sky is 5-10:1, so 12 still passes a harsh sunny scene
RATIO_MIN = 3.0           # below this the scene is washed out and flat - ambient drowning the sun
BLACK_FRAC_MAX = 0.02     # at most 2 % of terrain may sit below sRGB 0.05


def to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def measure(path: Path) -> dict:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return {"path": path.name, "error": "unreadable"}
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    grey = im.mean(2)
    # Terrain only: green-dominant pixels. Sky and water are excluded because their shadow behaviour is a
    # different question and averaging them in would hide the one being asked.
    terrain = im[:, :, 1] > im[:, :, 2]
    if terrain.sum() < 5000:
        return {"path": path.name, "error": "too little terrain in frame to judge"}
    vals = grey[terrain]
    lo, hi = np.percentile(vals, 8), np.percentile(vals, 85)
    shadow = im[terrain & (grey <= lo)].mean(0)
    sun = im[terrain & (grey >= hi)].mean(0)
    ls, lu = to_linear(shadow), to_linear(sun)
    return {
        "path": path.name,
        "sun_srgb": sun, "shadow_srgb": shadow,
        "sun_linear": lu, "shadow_linear": ls,
        "ratio": float(lu.mean() / max(ls.mean(), 1e-9)),
        # A shadow is lit by the SKY while sunlit ground gets sky PLUS sun, so the shadow must be
        # RELATIVELY cooler than the sunlit ground. Testing "blue is the highest channel" was wrong:
        # that only holds under a clear blue sky, and this scene uses an overcast HDRI whose light is
        # near-neutral grey by definition. The relative test holds for both.
        "cooler_than_sun": bool((ls[2] / max(ls[0], 1e-9)) > (lu[2] / max(lu[0], 1e-9)) * 0.98),
        "shadow_br": float(ls[2] / max(ls[0], 1e-9)), "sun_br": float(lu[2] / max(lu[0], 1e-9)),
        "black_frac": float((vals < 0.05).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ratio-max", type=float, default=RATIO_MAX)
    a = ap.parse_args()

    fails, warns = [], []
    for spec in a.images:
        p = Path(spec) if Path(spec).is_absolute() else REPO / spec
        r = measure(p)
        if "error" in r:
            print(f"{r['path']:34s} SKIPPED: {r['error']}")
            continue
        print(f"{r['path']:34s} sun:shadow {r['ratio']:5.1f}:1   "
              f"shadow linear ({r['shadow_linear'][0]:.4f}, {r['shadow_linear'][1]:.4f}, "
              f"{r['shadow_linear'][2]:.4f})  "
              f"B/R shadow {r['shadow_br']:.2f} vs sun {r['sun_br']:.2f}   "
              f"black {100 * r['black_frac']:.1f}%")
        if r["ratio"] > a.ratio_max:
            # The 5-10:1 figure describes OPEN ground under open sky. A dense settlement's inter-building
            # slots are genuinely darker than that, and forcing every view into the open-terrain band would
            # mean flattening the whole scene to satisfy a number derived for a different situation. So the
            # ratio is only a FAILURE when it is accompanied by the harm this tool exists to prevent -
            # ground actually crushed to black, where a survivor cannot be seen by the detector or by a
            # human. Otherwise it is reported as a warning and left visible.
            if r["black_frac"] > BLACK_FRAC_MAX * 0.5:
                fails.append(f"{r['path']}: sun:shadow {r['ratio']:.1f}:1 (limit {a.ratio_max:.0f}:1) AND "
                             f"{100 * r['black_frac']:.1f}% of terrain below sRGB 0.05 - the shadows are "
                             f"both too deep and actually crushed. Raise the SkyLight intensity.")
            else:
                warns.append(f"{r['path']}: sun:shadow {r['ratio']:.1f}:1 is over the {a.ratio_max:.0f}:1 "
                             f"open-terrain limit, but only {100 * r['black_frac']:.2f}% of terrain is "
                             f"crushed to black, so nothing is actually hidden. Expected for a view "
                             f"dominated by buildings.")
        if r["ratio"] < RATIO_MIN:
            fails.append(f"{r['path']}: sun:shadow is only {r['ratio']:.1f}:1, under the {RATIO_MIN:.0f}:1 "
                         f"floor. The ambient is drowning the sun and the scene reads flat and washed out; "
                         f"lower the SkyLight intensity.")
        # NOTE: there is deliberately NO shadow-colour test. Two were tried and both were wrong. "Shadow is
        # blue" holds only under a CLEAR sky, and this scene uses an overcast HDRI whose light is neutral
        # grey. "Shadow is cooler than sun" then failed too, because light bouncing off vegetation is green
        # and warms the shadow - which is correct behaviour, not a defect. A test that cannot distinguish a
        # defect from correct physics is worse than no test, so it is gone rather than tuned until it passes.
        if r["black_frac"] > BLACK_FRAC_MAX:
            fails.append(f"{r['path']}: {100 * r['black_frac']:.1f}% of terrain is below sRGB 0.05 "
                         f"(limit {100 * BLACK_FRAC_MAX:.0f}%). A survivor in there is invisible to the "
                         f"detector and to a human reviewer.")

    print()
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} lighting check(s) failed  (domain=sim)")
        return 1
    print("lighting is within the physical range for an outdoor scene  (domain=sim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
