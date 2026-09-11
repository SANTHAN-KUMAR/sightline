"""Do the scenario presets actually change the rendered world, or only the label? PIE ON, nothing flying.

    uv run python tools/live/_check_scenarios.py

A "variation" that does not change the pixels is worse than no variation: it invites a report claiming
coverage of dawn, rain and dusk that is really the same clear-midday distribution five times over. Section
5.5c is explicit that a slice is only a slice if the thing it names is actually different.

There is already a reason to doubt it. `dataset_gate.py` measured `clear_morning` at luma 0.640 and
`clear_midday` at 0.649 across two captured passes - a 1.4 % difference, which is not what a six-hour change
in sun angle looks like. Either the sun is not moving, or the scene's lighting is not driven by it.

So: set each scenario, grab a frame through the same camera the mission uses, and MEASURE luma, contrast and
the red/blue ratio (a warm low sun raises R/B; an overcast or high sun does not). Writes a contact strip so
the answer can also be seen rather than only computed.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "live"))

from demo import SCENARIOS, apply_scenario  # noqa: E402

import cosysairsim as airsim  # noqa: E402

SETTLE_S = 6.0          # the sun sweep and any weather need a moment to take effect and the frame to settle


def grab() -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()
        req = [airsim.ImageRequest("survey", airsim.ImageType.Scene, False, False)]
        c.simGetImages(req)                                  # warm-up: the first call is unconverted
        r = c.simGetImages(req)[0]
    return np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)


def stats(rgb: np.ndarray) -> dict[str, float]:
    f = rgb.astype(np.float32) / 255.0
    luma = float((0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]).mean())
    return {
        "luma": round(luma, 4),
        "contrast": round(float(f.std()), 4),
        "r_over_b": round(float(f[:, :, 0].mean() / max(1e-6, f[:, :, 2].mean())), 4),
        "p05": round(float(np.percentile(f, 5)), 4),
        "p95": round(float(np.percentile(f, 95)), 4),
    }


if __name__ == "__main__":
    tiles, rows = [], []
    for name in ("midday", "dawn", "dusk", "rain", "high"):
        desc = apply_scenario(name)
        time.sleep(SETTLE_S)
        rgb = grab()
        st = stats(rgb)
        rows.append((name, st))
        print(f"  {name:7s} {desc[:52]:54s} luma {st['luma']:.4f}  contrast {st['contrast']:.4f}  "
              f"R/B {st['r_over_b']:.4f}  p05 {st['p05']:.3f}  p95 {st['p95']:.3f}")
        t = cv2.resize(rgb[:, :, ::-1], (420, 236))
        cv2.rectangle(t, (0, 0), (420, 20), (20, 20, 24), -1)
        cv2.putText(t, f"{name}  luma {st['luma']:.3f}  R/B {st['r_over_b']:.3f}", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 235), 1, cv2.LINE_AA)
        tiles.append(t)

    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    out = REPO / "_artifacts/scenario_check.png"
    cv2.imwrite(str(out), grid)
    print(f"\nwrote {out} - LOOK AT IT")

    lum = [s["luma"] for _, s in rows]
    rb = [s["r_over_b"] for _, s in rows]
    spread_l = (max(lum) - min(lum)) / max(1e-6, np.mean(lum))
    spread_rb = (max(rb) - min(rb)) / max(1e-6, np.mean(rb))
    print(f"\nluma spread {100 * spread_l:.1f} %   R/B spread {100 * spread_rb:.1f} %")
    if spread_l < 0.08 and spread_rb < 0.05:
        print("VERDICT: the scenarios are COSMETIC. The rendered world barely changes, so these are not\n"
              "         five slices - they are one slice with five labels. Do not report them as coverage.")
        raise SystemExit(1)
    print("VERDICT: the scenarios produce measurably different imagery.")
