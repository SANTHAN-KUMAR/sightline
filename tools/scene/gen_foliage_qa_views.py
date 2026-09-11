"""Pick the camera poses the foliage-material QA renders use, and write them to data/scene/foliage_qa_views.json.

    uv run python tools/scene/gen_foliage_qa_views.py        # host side: needs numpy

Host-side only: it reads `data/scene/vegetation.json` (the layout that is already placed and verified) and
picks views. It never touches the editor and it never moves a tree.

WHY THE VIEW IS CHOSEN AND NOT HARD-CODED
-----------------------------------------
The check this feeds (`tools/scene/check_foliage_materials.py`) compares the luminance of CANOPY pixels with
the luminance of SUNLIT GROUND pixels *in the same frame*, because that is exactly the evidence the red-team
audit gave for the defect: "sunlit grass brilliant green, canopy beside it near-black at the same exposure".

That comparison needs a frame that actually contains both. The densest patch of canopy in the valley is the
wrong choice -- at 100 % closure there is no ground left to compare against -- and so is a sparse one, where a
handful of crowns give a noisy canopy sample. So the nadir view is placed over the hillslope window whose
canopy closure is CLOSEST TO `TARGET_CLOSURE`, subject to a minimum tree count.

Closure is rasterised from the crown ellipse of every instance the layout places (major/minor radius times the
instance's own scale), on the same 2 m grid `gen_vegetation.py` used to report 44.9 % over the ROI, so the
number here is comparable with that one.

The oblique view targets the largest jacaranda crown in the same neighbourhood. Its azimuth is left to the
editor script, which reads the DirectionalLight and puts the camera on the anti-sun side so the crown is
BACKLIT -- backlit crowns going black is the specific symptom two-sided-foliage + subsurface is supposed to
fix, so that is the frame where the fix has to show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VEG = REPO / "data/scene/vegetation.json"
OUT = REPO / "data/scene/foliage_qa_views.json"

STEP_M = 2.0            # crown raster cell, same as gen_vegetation.py's canopy-closure raster
NADIR_AGL_M = 45.0      # survey altitude (docs/SOLUTION_DOC.md survey slice); the dataset is flown at this
FOV_DEG = 74.0          # same lens qa_shots.py uses for its nadir, so the frames are comparable
TARGET_CLOSURE = 0.55   # see the docstring: enough canopy to measure, enough ground to compare against
MIN_TREES_IN_VIEW = 30
ZONE = "hillslope"      # the brief asks for "45 m nadir over hillslope canopy"


def main() -> int:
    plan = json.loads(VEG.read_text())
    base_z = plan["base_z_m"]
    cat = plan["catalogue"]
    items = plan["items"]

    trees = [t for t in items if t["zone"] == ZONE]
    if not trees:
        raise SystemExit(f"no trees in zone {ZONE!r}")
    print(f"{len(trees):,} trees in zone {ZONE!r} (of {len(items):,} total)")

    e = np.array([t["east_m"] for t in trees])
    n = np.array([t["north_m"] for t in trees])
    # crown radius: from the CATALOGUE times the instance's own scale, not the layout's cached crown_r_m --
    # the same reasoning check_vegetation.py uses so a generator that miscomputed its own field cannot hide.
    rmaj = np.array([cat[t["pid"]]["crown_major_m"] * 0.5 * t["scale"] for t in trees])
    ground = np.array([t["ground_asl_m"] for t in trees])
    height = np.array([cat[t["pid"]]["height_m"] * t["scale"] for t in trees])
    top = np.array([t["base_asl_m"] for t in trees]) + height

    e0, e1 = e.min() - 40, e.max() + 40
    n0, n1 = n.min() - 40, n.max() + 40
    nx = int(np.ceil((e1 - e0) / STEP_M))
    ny = int(np.ceil((n1 - n0) / STEP_M))
    cover = np.zeros((ny, nx), dtype=bool)
    for ei, ni, ri in zip(e, n, rmaj):
        ix0 = max(0, int((ei - ri - e0) / STEP_M))
        ix1 = min(nx, int((ei + ri - e0) / STEP_M) + 1)
        iy0 = max(0, int((ni - ri - n0) / STEP_M))
        iy1 = min(ny, int((ni + ri - n0) / STEP_M) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        gx = e0 + (np.arange(ix0, ix1) + 0.5) * STEP_M
        gy = n0 + (np.arange(iy0, iy1) + 0.5) * STEP_M
        dd = ((gx[None, :] - ei) ** 2 + (gy[:, None] - ni) ** 2) <= ri * ri
        cover[iy0:iy1, ix0:ix1] |= dd
    print(f"crown raster {nx} x {ny} cells at {STEP_M} m; zone closure {cover.mean() * 100:.1f} %")

    # the ground footprint one nadir frame sees, in cells
    half_m = NADIR_AGL_M * np.tan(np.radians(FOV_DEG / 2.0))
    win = max(3, int(round(2 * half_m / STEP_M)))
    print(f"a {NADIR_AGL_M:.0f} m nadir at {FOV_DEG:.0f} deg fov sees {2 * half_m:.1f} m across "
          f"({win} cells) -- the window scored below")

    # box-filter the coverage with a summed-area table so every candidate window is scored, not sampled
    sat = np.pad(cover.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, ny - win)
    xs = np.arange(0, nx - win)
    if len(ys) == 0 or len(xs) == 0:
        raise SystemExit("zone is smaller than one frame footprint")
    box = (sat[np.ix_(ys + win, xs + win)] - sat[np.ix_(ys, xs + win)]
           - sat[np.ix_(ys + win, xs)] + sat[np.ix_(ys, xs)])
    closure = box / float(win * win)

    # tree count per window, same footprint
    cy = ((n - n0) / STEP_M).astype(int)
    cx = ((e - e0) / STEP_M).astype(int)
    cnt = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(cnt, (np.clip(cy, 0, ny - 1), np.clip(cx, 0, nx - 1)), 1)
    csat = np.pad(cnt, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    tcount = (csat[np.ix_(ys + win, xs + win)] - csat[np.ix_(ys, xs + win)]
              - csat[np.ix_(ys + win, xs)] + csat[np.ix_(ys, xs)])

    score = np.abs(closure - TARGET_CLOSURE)
    score[tcount < MIN_TREES_IN_VIEW] = 9.9                 # too few trees: canopy sample would be noise
    iy, ix = np.unravel_index(np.argmin(score), score.shape)
    if score[iy, ix] > 9.0:
        raise SystemExit(f"no window in {ZONE!r} has >= {MIN_TREES_IN_VIEW} trees")
    got_closure = float(closure[iy, ix])
    got_trees = int(tcount[iy, ix])
    ce = e0 + (xs[ix] + win / 2.0) * STEP_M
    cn = n0 + (ys[iy] + win / 2.0) * STEP_M

    # local ground and canopy top inside the footprint, so the camera clears the tallest crown by a real margin
    inwin = (np.abs(e - ce) <= half_m) & (np.abs(n - cn) <= half_m)
    g_mean = float(ground[inwin].mean())
    top_max = float(top[inwin].max())
    cam_asl = max(g_mean + NADIR_AGL_M, top_max + 20.0)
    print(f"nadir window: closure {got_closure * 100:.1f} % ({got_trees} trees), centre "
          f"E {ce:.1f} N {cn:.1f}, ground {g_mean:.1f} m ASL, tallest crown top {top_max:.1f} m ASL")
    print(f"  camera {cam_asl:.1f} m ASL = {cam_asl - g_mean:.1f} m above local ground, "
          f"{cam_asl - top_max:.1f} m above the tallest crown")

    # --- the crown for the oblique: the biggest jacaranda near that window --------------------------------
    jac = [(i, t) for i, t in enumerate(trees) if t["pid"] == "jacaranda_tree"]
    if not jac:
        raise SystemExit("no jacaranda in the hillslope zone")
    # prefer big and near the nadir window, so both views describe the same piece of forest
    def crown_score(pair):
        i, t = pair
        d = np.hypot(t["east_m"] - ce, t["north_m"] - cn)
        return (-height[i], d)
    jac.sort(key=crown_score)
    i, t = jac[0]
    crown_top = float(top[i])
    crown_h = float(height[i])
    crown_r = float(rmaj[i])
    # crown centre: photogrammetry trees carry their crown in the upper half of the mesh
    crown_ctr_asl = float(t["base_asl_m"]) + crown_h * 0.68
    print(f"oblique target: {t['name']} {t['pid']} scale {t['scale']:.2f}, {crown_h:.1f} m tall, "
          f"crown r {crown_r:.1f} m, centre {crown_ctr_asl:.1f} m ASL at E {t['east_m']:.1f} N {t['north_m']:.1f}")

    out = {
        "generated_by": "tools/scene/gen_foliage_qa_views.py",
        "base_z_m": base_z,
        "fov_deg": FOV_DEG,
        "zone": ZONE,
        "nadir": {
            "east_m": round(float(ce), 2), "north_m": round(float(cn), 2),
            "cam_asl_m": round(float(cam_asl), 2),
            "agl_m": round(float(cam_asl - g_mean), 2),
            "ground_asl_m": round(g_mean, 2),
            "footprint_m": round(float(2 * half_m), 1),
            "expected_closure": round(got_closure, 4),
            "trees_in_view": got_trees,
        },
        "crown": {
            "name": t["name"], "pid": t["pid"],
            "east_m": t["east_m"], "north_m": t["north_m"],
            "centre_asl_m": round(crown_ctr_asl, 2),
            "top_asl_m": round(crown_top, 2),
            "radius_m": round(crown_r, 2),
            "height_m": round(crown_h, 2),
            "scale": t["scale"],
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
