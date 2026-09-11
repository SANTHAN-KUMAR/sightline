"""Does the DEPTH pass render the foliage that the SEGMENTATION pass is configured to skip? (PIE ON)

    uv run python tools/capture/check_depth_sees_foliage.py
    uv run python tools/capture/check_depth_sees_foliage.py --east 82.6 --north 760.1 --agl 35

Cosys-AirSim's annotation renderer disables two engine show flags on purpose
(`Source/Annotation/ObjectAnnotator.cpp:SetViewForAnnotationRender`):

    show_flags.SetInstancedFoliage(false);
    show_flags.SetInstancedGrass(false);

Every plant in this scene lives on a HierarchicalInstancedStaticMeshComponent - that was forced on us by the
Windows commit limit, which killed the editor at ~5,500 individual actors. So the instance mask renders the
terrain straight through the canopy, the auto-labeller reads a full unbroken silhouette for a survivor lying
under a fern, and the box it writes covers pure leaf texture. Measured on seed23_alt35: 7 of 110 boxes had no
silhouette in RGB at all, and every vegetation-occluded survivor carries a full-body box where the guideline
(section 1) requires a visible-extent box.

DepthPlanar is a different render path - the ordinary scene depth buffer, which Nanite and instanced meshes
both write. If depth sees the canopy, occlusion becomes measurable per pixel without patching and rebuilding
the plugin:

    a mask pixel claiming actor A is REALLY visible  <=>  depth(pixel) is not closer than A's own surface

This script decides that question by looking, and prints the numbers behind the picture. It does not modify
anything.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

with contextlib.redirect_stdout(io.StringIO()):
    import cosysairsim as airsim

CAM = "survey"
OUT = REPO / "_artifacts" / "depth_vs_seg.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=CAM)
    a = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()

    req = [airsim.ImageRequest(a.camera, airsim.ImageType.Scene, False, False),
           airsim.ImageRequest(a.camera, airsim.ImageType.Segmentation, False, False),
           airsim.ImageRequest(a.camera, airsim.ImageType.DepthPlanar, True, False)]
    with contextlib.redirect_stdout(io.StringIO()):
        c.simGetImages(req)                 # warm-up: the first call of a fresh engine is unconverted
        res = c.simGetImages(req)

    scene, seg, dep = res
    if dep.width == 0 or not dep.image_data_float:
        print("FAIL: DepthPlanar returned an empty buffer. The capture entry for ImageType 1 is declared in\n"
              "      the settings profile the editor was STARTED with, so an empty buffer means the camera\n"
              "      name is wrong or the profile in use is not sim/settings/dataset.json.")
        return 2

    rgb = np.frombuffer(scene.image_data_uint8, np.uint8).reshape(scene.height, scene.width, 3)
    msk = np.frombuffer(seg.image_data_uint8, np.uint8).reshape(seg.height, seg.width, 3)
    d = np.array(dep.image_data_float, np.float32).reshape(dep.height, dep.width)

    finite = np.isfinite(d) & (d > 0) & (d < 1e4)
    print(f"scene {scene.width}x{scene.height}   seg {seg.width}x{seg.height}   depth {dep.width}x{dep.height}")
    print(f"depth: {100 * finite.mean():.1f}% finite, range {d[finite].min():.1f}-{d[finite].max():.1f} m, "
          f"median {np.median(d[finite]):.1f} m")

    cols = np.unique(msk.reshape(-1, 3), axis=0)
    print(f"segmentation: {len(cols)} distinct colours in this frame")

    # --- the test -------------------------------------------------------------------------------------
    # Foliage is high-frequency geometry. If depth renders it, depth has structure exactly where the
    # segmentation mask is flat. Measure both on the SAME pixels: the largest flat mask region.
    big = max((tuple(int(x) for x in cc) for cc in cols),
              key=lambda cc: int(np.all(msk == np.array(cc, np.uint8), axis=2).sum()))
    flat = np.all(msk == np.array(big, np.uint8), axis=2) & finite
    print(f"largest flat mask region {big}: {100 * flat.mean():.1f}% of frame")

    # local depth roughness = |depth - 5x5 median|, in metres, inside that flat region.
    # ksize is 5 because OpenCV's medianBlur only accepts CV_32F at ksize 3 or 5.
    med = cv2.medianBlur(d, 5)
    rough = np.abs(d - med)
    r_flat = rough[flat]
    print(f"  depth roughness inside it: median {np.median(r_flat):.3f} m  "
          f"p95 {np.percentile(r_flat, 95):.2f} m  max {r_flat.max():.1f} m")
    struct = float((r_flat > 0.5).mean())
    print(f"  fraction of those pixels standing >0.5 m off the local median: {100 * struct:.1f}%")

    hsv = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] > 60)
    gf = green & flat
    if gf.sum() > 1000:
        print(f"  of those pixels, the {100 * gf.sum() / max(1, flat.sum()):.1f}% that look like foliage in RGB "
              f"have depth roughness median {np.median(rough[gf]):.3f} m "
              f"vs {np.median(rough[flat & ~green]):.3f} m for the rest")

    dv = np.zeros_like(d)
    dv[finite] = d[finite]
    lo, hi = np.percentile(dv[finite], [2, 98])
    dvis = np.clip((dv - lo) / max(1e-6, hi - lo), 0, 1)
    dvis = (cv2.applyColorMap((dvis * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    h = 620
    w = int(scene.width * h / scene.height)
    panel = np.hstack([cv2.resize(rgb[:, :, ::-1], (w, h)),
                       cv2.resize(msk[:, :, ::-1], (w, h), interpolation=cv2.INTER_NEAREST),
                       cv2.resize(dvis, (w, h))])
    cv2.imwrite(str(OUT), panel)
    print("")
    print(f"wrote {OUT}  (RGB | segmentation | depth) - LOOK AT IT")

    if struct < 0.02:
        print("")
        print("VERDICT: depth is as flat as the mask. Depth does NOT see the foliage either, so per-pixel")
        print("         visibility cannot be recovered this way and the plugin show flags must be patched.")
        return 1
    print("")
    print("VERDICT: depth resolves geometry the segmentation mask is blind to. Per-pixel visibility can be")
    print("         computed as depth(pixel) vs the actor's own surface range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
