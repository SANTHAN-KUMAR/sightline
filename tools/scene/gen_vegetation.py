"""Generate the FloodValley tree canopy and understorey layout (SOLUTION_DOC 2.2; scene-realism lane).

    uv run python tools/scene/gen_vegetation.py [--seed 67] [--max-trees 4600]
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_vegetation.py').read())"

Writes `data/scene/vegetation.json` (the layout) and `data/scene/vegetation_canopy.png` (a canopy-closure map
you can LOOK at without opening the editor). No editor, no numpy in the editor: this is the host half.

WHY THE SCENE NEEDS THIS
------------------------
The reference photograph of a real flooded neighbourhood is roughly half broadleaf canopy. FloodValley has
none: `gen_props.py` scatters `fern_02` on the hillslopes and that is the entire vegetation story. Bare
hillsides are the single largest difference between our nadir frames and the photo.

THE TREES ARE PHOTOGRAMMETRY SCANS. MEASURE THEM, DO NOT ASSUME THEM.
Straight from the glTF accessors (`count`, per-attribute `min`/`max`), both variants:

    id                full scan tris   lite tris   height    crown    lite .bin
    island_tree_01        1,599,403     162,400     5.03 m    4.82 m     6.1 MB
    island_tree_02        1,072,213     106,821     3.41 m    4.21 m     4.1 MB
    island_tree_03        2,085,320     111,522     2.62 m    2.97 m     4.4 MB
    jacaranda_tree        3,863,832     352,726    19.47 m   24.42 m    13.6 MB
    tree_small_02         2,062,487     156,172     4.56 m    4.29 m     6.0 MB

The assets lane decimates each scan to `<pid>_2k_lite.gltf` and imports THAT, so the lite variant is the
geometry the level carries and the one this budget is built from; the full scan is measured only as a
cross-check. If a lite variant is missing the generator falls back to the full scan and says so.

(glTF is Y-up, so height = bbox.y and the crown spread is bbox.x/bbox.z; UE's importer maps
UE(x, y, z) = glTF(x, z, y), which is exactly what `data/scene/props.json` already records for
`dead_tree_trunk`: glTF bbox (3.05, 0.29, 0.28) -> UE size_m [3.05, 0.28, 0.29]. When props.json carries the
species, its editor-measured `tris` and `size_m` win - unless they disagree with the glTF by more than 2x,
which would mean Nanite is reporting a FALLBACK mesh, and then the glTF is treated as truth.)

Half-canopy closure over the 1.37 km2 the drone surveys takes ~4,400 crowns (measured below), which even at
the lite counts is **1.08 BILLION instanced source triangles from 889,641 unique**. These meshes are unusable
without Nanite: a plain static mesh has no LOD chain here, so every instance would rasterise its full LOD0.
`build_vegetation.py` therefore turns Nanite ON for the tree meshes only (`import_assets.py` turns it OFF for
every prop, which is right for the 12-53 k rocks and wrong for these), asserts that it took, and REFUSES to
place anything if it did not and the layout exceeds `NO_NANITE_MAX_TRIS`. That refusal is a check that can fail.

HOW THE LAYOUT IS BUILT
-----------------------
Per zone, a target canopy CLOSURE (fraction of ground under a crown) is turned into a Poisson intensity
    lambda = -ln(1 - closure) / mean_crown_area
because random crowns overlap: closure = 1 - exp(-lambda * A), not lambda * A. Each 4 m terrain cell then
draws Poisson(lambda * 16 m2) trees at uniform positions inside it, so the result is a real random field
rather than a grid, and it is exactly reproducible from the seed.

    hillslope   dry ground above the flood line, outside terrace/fan/channel   closure 0.72   dense forest
    garden      the settlement terrace above the water: Kerala homegardens     closure 0.55
    flooded     the same homegardens with their floor under water              closure 0.55
    bank        cut banks either side of the channel                           closure 0.35
    floodplain  drowned valley floor outside the terrace: riparian trees       closure 0.20
    fan         fresh debris-flow deposit, 2-7 m thick: scoured and buried     closure 0.08
    channel     scoured clean by the flow                                      closure 0.02

Measured result at the default seed: 45.6 % canopy closure over the 1.37 km2 ROI and 42.1 % over the
settlement box. It is not 50 % because two large zones are deliberately bare - the deposit fan (22.7 % of the
ROI, 8 % closure) and the channel - which is what a debris-flow flood actually looks like from the air.

Drowned trees lean DOWNSTREAM (the valley drains towards -north) by 6-22 degrees, and one in five is
"undercut" - its root plate scoured out, so it has settled 0.4-1.3 m. In 0.5-3.7 m of water a 3-5 m island
tree then shows only its crown, which is the "crowns still emerging from shallow water" the brief asks for;
the generator reports the emergent-height distribution so that claim is a number, not an adjective.

CLEARANCES (all enforced, all reported)
    house footprint + 2.0 m     a trunk may not stand in a wall; crowns may overhang, which is realistic
    survivor + 3.0 m            a crown over a survivor marked `aerially_detectable` would contradict the
                                ground truth, exactly like the foam quads in gen_foam.py
    launch pad 70 m             sim_fly treats contact with anything not named "Ground" as a failed takeoff
    slope < 46 deg              trees do not grow on cut faces

The triangle budget is spent nearest the camera first: candidates are ranked by distance to the closest house
or survivor, so if `--max-trees` bites, it thins the far hillslopes and not the settlement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_terrain as gt  # noqa: E402  (rebuild the same seeded height grid the level was imported from)

OUT = gt.OUT
PH = gt.REPO / "_downloads" / "assets" / "polyhaven"

# Poly Haven ids, their role in the canopy, and the scale range applied on top of the scanned size.
# Kerala canopy trees (mango, jack, rain tree) are 15-25 m; the jacaranda scan is 19.5 m, so it carries the
# canopy. The island trees are 3-5 m scans and are scaled up to 6-9 m as the mid-storey / homegarden layer.
SPECIES = {
    "jacaranda_tree": {"layer": "canopy", "scale": (0.85, 1.30)},
    "island_tree_01": {"layer": "mid", "scale": (1.15, 1.80)},
    "island_tree_02": {"layer": "mid", "scale": (1.25, 1.95)},
    "island_tree_03": {"layer": "shrub", "scale": (1.20, 2.10)},
    "tree_small_02": {"layer": "mid", "scale": (1.15, 1.80)},
}
GROUND_SPECIES = ["fern_02"]

# zone -> (target canopy closure, species mix). Mixes are normalised, and any species whose source files are
# incomplete is dropped and its weight redistributed, so a half-finished download degrades instead of crashing.
ZONES = {
    "hillslope": (0.72, {"jacaranda_tree": 0.56, "island_tree_01": 0.14, "island_tree_02": 0.11,
                         "island_tree_03": 0.07, "tree_small_02": 0.12}),
    "garden": (0.55, {"jacaranda_tree": 0.50, "island_tree_01": 0.18, "island_tree_02": 0.14,
                      "island_tree_03": 0.06, "tree_small_02": 0.12}),
    "bank": (0.35, {"jacaranda_tree": 0.40, "island_tree_01": 0.24, "island_tree_02": 0.18,
                    "island_tree_03": 0.08, "tree_small_02": 0.10}),
    # The flood does not fell a mature homegarden: it drowns its floor. Over the terrace the pre-flood canopy
    # is still standing, which is exactly what the reference photo shows, so this closure is close to the dry
    # garden's. Only the scoured channel and the fresh 2-7 m fan deposit actually lose their trees.
    "flooded": (0.55, {"jacaranda_tree": 0.44, "island_tree_01": 0.22, "island_tree_02": 0.18,
                       "island_tree_03": 0.06, "tree_small_02": 0.10}),
    # Drowned valley floor outside the settlement terrace: riparian trees the overbank flow reached but did
    # not scour. Without this zone 7.2 % of the ROI (measured) fell through every mask and rendered as a bare
    # halo all the way round the water - visible immediately in vegetation_canopy.png, invisible in any count.
    "floodplain": (0.20, {"jacaranda_tree": 0.28, "island_tree_01": 0.30, "island_tree_02": 0.24,
                          "island_tree_03": 0.08, "tree_small_02": 0.10}),
    "fan": (0.08, {"jacaranda_tree": 0.20, "island_tree_01": 0.28, "island_tree_02": 0.24,
                   "island_tree_03": 0.18, "tree_small_02": 0.10}),
    "channel": (0.02, {"island_tree_01": 0.40, "island_tree_02": 0.35, "island_tree_03": 0.25}),
}
ZONE_ORDER = ["hillslope", "garden", "bank", "flooded", "floodplain", "fan", "channel"]

CLEAR_HOUSE_M = 2.0
CLEAR_SURVIVOR_M = 3.0
CLEAR_PAD_M = 70.0
MAX_SLOPE_DEG = 46.0      # Western Ghats forest holds well past 40 deg; 46 leaves only the cut faces bare
SINK_CM = 10.0            # push each trunk this far into the ground so micro-relief never shows daylight
# If Nanite cannot be turned on, this is all the tree geometry the scene may carry. 8 M triangles is roughly
# what the rest of FloodValley already costs (terrain 524 k + debris 4.15 M + damage 212 k + houses/foam).
NO_NANITE_MAX_TRIS = 8_000_000


# --------------------------------------------------------------------------------------------------------
# source measurement: read the glTF header instead of trusting a catalogue
# --------------------------------------------------------------------------------------------------------
def measure_gltf(pid: str) -> dict:
    """Exact triangle count and bounding box straight out of the glTF accessors, plus a file-integrity check.

    The accessors carry `count` and per-attribute `min`/`max`, so this is the real number the importer will
    produce - no guessing, and no need for the editor. It also catches a half-finished download: the buffer
    URI must exist and be at least `byteLength` bytes, which is how `tree_small_02` was caught with a 9 KB
    glTF and no .bin at all.
    """
    # The assets lane decimates the tree scans to `<pid>_2k_lite.gltf` (jacaranda 3,863,832 -> 352,726
    # triangles, 208 MB -> 13.6 MB) and imports THOSE, so the lite variant is the geometry the level actually
    # carries and the one the triangle budget must be built from. Fall back to the full scan if it is absent.
    path = PH / pid / f"{pid}_2k_lite.gltf"
    variant = "lite"
    if not path.is_file():
        path = PH / pid / f"{pid}_2k.gltf"
        variant = "full"
    if not path.is_file():
        return {"pid": pid, "ok": False, "why": f"missing {path.name}"}
    g = json.loads(path.read_text())
    for buf in g.get("buffers", []):
        uri = buf.get("uri")
        if uri is None:
            continue
        bin_path = path.parent / uri
        if not bin_path.is_file():
            return {"pid": pid, "ok": False, "why": f"missing buffer {uri} (download incomplete)"}
        if bin_path.stat().st_size < buf["byteLength"]:
            return {"pid": pid, "ok": False,
                    "why": f"{uri} is {bin_path.stat().st_size} B, glTF declares {buf['byteLength']} B"}
    acc = g["accessors"]
    tris = verts = 0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for mesh in g.get("meshes", []):
        for prim in mesh["primitives"]:
            tris += acc[prim["indices"]]["count"] // 3
            pos = acc[prim["attributes"]["POSITION"]]
            verts += pos["count"]
            for k in range(3):
                lo[k] = min(lo[k], pos["min"][k])
                hi[k] = max(hi[k], pos["max"][k])
    dx, dy, dz = (hi[k] - lo[k] for k in range(3))
    # glTF is Y-up. UE's importer lands it as UE(x, y, z) = glTF(x, z, y) - verified against props.json's
    # dead_tree_trunk, whose glTF bbox (3.05, 0.29, 0.28) is recorded there as UE size_m [3.05, 0.28, 0.29].
    return {
        "pid": pid, "ok": True, "variant": variant, "tris": int(tris), "verts": int(verts),
        "height_m": round(float(dy), 3),
        "crown_major_m": round(float(max(dx, dz)), 3), "crown_minor_m": round(float(min(dx, dz)), 3),
        "materials": [m.get("name", "") for m in g.get("materials", [])],
        "gltf": str(path),
        # /Game/Sightline/Props/<pid>/<pid>_2k/StaticMeshes/<glTF node name> is the convention props.json
        # already uses; build_vegetation.py treats this as a HINT and resolves the real asset by folder.
        "asset_hint": f"/Game/Sightline/Props/{pid}/{pid}_2k/StaticMeshes/"
                      + (g.get("nodes", [{}])[0].get("name") or pid),
    }


def catalogue(props_json: Path) -> tuple[dict, list[str]]:
    """Measured tree/ground catalogue. props.json wins when it already carries the asset (the assets lane owns
    that file); otherwise the glTF on disk is measured directly so this lane never has to wait for it."""
    known = {}
    if props_json.is_file():
        known = json.loads(props_json.read_text()).get("props", {})
    cat, notes = {}, []
    for pid in list(SPECIES) + GROUND_SPECIES:
        m = measure_gltf(pid)
        if not m["ok"]:
            notes.append(f"{pid}: SKIPPED - {m['why']}")
            continue
        entry = dict(m)
        entry.pop("ok")
        entry["tris_gltf"] = m["tris"]          # kept even when props.json overrides `tris`, so the checker
                                                # can re-measure the source and compare like with like
        if pid in known and known[pid].get("meshes"):
            meshes = known[pid]["meshes"]
            big = max(meshes, key=lambda x: x.get("tris", 0))
            entry["asset_hint"] = big["asset"]
            entry["in_props_json"] = True
            if big.get("tris"):
                entry["tris_imported"] = int(big["tris"])
                # props.json is written from inside the editor by import_assets.py, so its count is what UE
                # actually holds. Prefer it over our own glTF count for the budget when the two agree to
                # within a factor of 2; a wild disagreement means Nanite is reporting a FALLBACK mesh
                # (docs/CONTEXT.md: a 524 k-triangle terrain reported 347) and we must not trust it.
                if 0.5 <= entry["tris_imported"] / max(1, entry["tris"]) <= 2.0:
                    entry["tris"] = entry["tris_imported"]
                else:
                    entry["tris_note"] = ("props.json says {:,} but the glTF has {:,}: treating the glTF as "
                                          "truth (a Nanite fallback count?)".format(
                                              entry["tris_imported"], entry["tris"]))
            if big.get("size_m"):
                sx, sy, sz = big["size_m"]
                entry["height_m"] = round(float(sz), 3)
                entry["crown_major_m"] = round(float(max(sx, sy)), 3)
                entry["crown_minor_m"] = round(float(min(sx, sy)), 3)
            entry["sub_meshes"] = [x["asset"] for x in meshes]
        else:
            entry["in_props_json"] = False
        entry["layer"] = SPECIES.get(pid, {}).get("layer", "ground")
        cat[pid] = entry
    return cat, notes


# --------------------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------------------
def crown_area_m2(entry: dict, scale: float) -> float:
    """Projected crown footprint, as an ellipse with the scanned major/minor spread."""
    return math.pi * 0.25 * entry["crown_major_m"] * entry["crown_minor_m"] * scale * scale


def lean_rotator(tilt_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """(roll, pitch, yaw) that tilts a trunk `tilt_deg` towards world azimuth `azimuth_deg`.

    Azimuth is measured from UE +X (north) towards UE +Y (east), the same convention every other spawner in
    this project uses. A tree is rotationally arbitrary about its own trunk, so the lean is expressed with
    PITCH ALONE and the yaw is spent aiming it - that avoids UE's roll-sign convention entirely.

    UE is left-handed: a positive pitch raises +X (nose up), which tips local +Z towards -X. So a pitch of
    -tilt with yaw = azimuth tips the trunk towards +X rotated by the yaw, i.e. towards the azimuth.
    If a render ever shows the drowned trees leaning UPSTREAM, flip PITCH_SIGN in build_vegetation.py; it is
    one constant and it changes nothing else.
    """
    return (0.0, -float(tilt_deg), float(azimuth_deg) % 360.0)


def build(seed: int, max_trees: int, roi_margin_m: float, ground_cover: bool) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    actors = json.loads((OUT / "actors.json").read_text())["actors"]
    cat, notes = catalogue(OUT / "props.json")
    trees = {k: v for k, v in cat.items() if k in SPECIES}
    if not trees:
        raise SystemExit("no usable tree source found under _downloads/assets/polyhaven:\n  " + "\n  ".join(notes))

    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    base_z = float(h.min())
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")          # X = east, Y = north, indexed [j = north, i = east]
    gy, gx = np.gradient(h, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    # --- zone masks -------------------------------------------------------------------------------------
    dry = h > water + 0.35
    ok_slope = slope_deg < MAX_SLOPE_DEG
    L = meta["launch_site"]
    far_pad = np.hypot(X - L["east_m"], Y - L["north_m"]) > CLEAR_PAD_M
    terrace, fan, chan, bank = s["terrace"], s["fan"], s["in_channel"], s["bank"]
    masks = {
        "channel": chan & ok_slope,
        "bank": bank & ~chan & dry & ok_slope,
        "fan": fan & ~chan & ~bank & dry & ok_slope,
        "flooded": terrace & ~chan & ~bank & ~fan & ~dry,
        "garden": terrace & ~chan & ~bank & ~fan & dry & ok_slope,
    }
    claimed = np.zeros_like(chan)
    for k in ("channel", "bank", "fan", "flooded", "garden"):
        claimed |= masks[k]
    masks["floodplain"] = ~claimed & ~dry & ~terrace & ok_slope
    claimed |= masks["floodplain"]
    masks["hillslope"] = ~claimed & dry & ok_slope & ~terrace
    for k in masks:
        masks[k] = masks[k] & far_pad

    # --- region of interest: the box the survey actually covers, plus a margin ---------------------------
    pts = np.array([[b["east_m"], b["north_m"]] for b in town["houses"]]
                   + [[a["east_m"], a["north_m"]] for a in actors])
    roi = {"east_m": [float(pts[:, 0].min() - roi_margin_m), float(pts[:, 0].max() + roi_margin_m)],
           "north_m": [float(pts[:, 1].min() - roi_margin_m), float(pts[:, 1].max() + roi_margin_m)],
           "margin_m": roi_margin_m}
    in_roi = ((X >= roi["east_m"][0]) & (X <= roi["east_m"][1])
              & (Y >= roi["north_m"][0]) & (Y <= roi["north_m"][1]))
    roi["area_km2"] = round(float((roi["east_m"][1] - roi["east_m"][0])
                                  * (roi["north_m"][1] - roi["north_m"][0]) / 1e6), 4)
    for k in masks:
        masks[k] = masks[k] & in_roi

    # --- clearance tests --------------------------------------------------------------------------------
    arch = town["archetypes"]
    hb = np.array([[b["east_m"], b["north_m"], math.radians(b["yaw_deg"]),
                    arch[b["archetype"]]["length_m"] / 2 + CLEAR_HOUSE_M,
                    arch[b["archetype"]]["width_m"] / 2 + CLEAR_HOUSE_M] for b in town["houses"]])
    sv = np.array([[a["east_m"], a["north_m"]] for a in actors])

    def clear_of_houses(e: float, nn: float) -> bool:
        de, dn = e - hb[:, 0], nn - hb[:, 1]
        u = de * np.cos(hb[:, 2]) + dn * np.sin(hb[:, 2])
        v = -de * np.sin(hb[:, 2]) + dn * np.cos(hb[:, 2])
        return not bool(np.any((np.abs(u) < hb[:, 3]) & (np.abs(v) < hb[:, 4])))

    def clear_of_survivors(e: float, nn: float, r: float = CLEAR_SURVIVOR_M) -> bool:
        return not bool(np.any(np.hypot(sv[:, 0] - e, sv[:, 1] - nn) < r))

    def ground_at(e: float, nn: float) -> float:
        i = int(round((e + size / 2) / cell))
        j = int(round((nn + size / 2) / cell))
        return float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])

    # --- candidate generation ---------------------------------------------------------------------------
    cand: list[dict] = []
    rejected = {"house": 0, "survivor": 0, "drowned": 0}
    cell_area = cell * cell
    zone_stats = {}
    for zone in ZONE_ORDER:
        closure, mix = ZONES[zone]
        mix = {k: w for k, w in mix.items() if k in trees}
        tot = sum(mix.values())
        if not tot:
            continue
        names = sorted(mix)                                   # sorted => deterministic regardless of dict order
        probs = np.array([mix[k] / tot for k in names])
        mean_scale = np.array([sum(SPECIES[k]["scale"]) / 2 for k in names])
        mean_area = float(sum(p * crown_area_m2(trees[k], sc)
                              for p, k, sc in zip(probs, names, mean_scale, strict=True)))
        lam = -math.log(max(1e-6, 1.0 - closure)) / mean_area     # crowns overlap: closure = 1 - exp(-lam*A)
        cells = np.argwhere(masks[zone])
        if not len(cells):
            zone_stats[zone] = {"cells": 0, "placed": 0}
            continue
        counts = rng.poisson(lam * cell_area, size=len(cells))
        placed = 0
        for (j, i), k in zip(cells, counts, strict=True):
            for _ in range(int(k)):
                e = float(xs[i]) + float(rng.uniform(-cell / 2, cell / 2))
                nn = float(xs[j]) + float(rng.uniform(-cell / 2, cell / 2))
                if not clear_of_houses(e, nn):
                    rejected["house"] += 1
                    continue
                if not clear_of_survivors(e, nn):
                    rejected["survivor"] += 1
                    continue
                pid = names[int(rng.choice(len(names), p=probs))]
                lo, hi = SPECIES[pid]["scale"]
                sc = float(rng.uniform(lo, hi))
                gz = ground_at(e, nn)
                depth = max(0.0, water - gz)
                tilt, undercut = 0.0, 0.0
                if zone in ("flooded", "bank", "channel") and depth > 0.15:
                    tilt = float(rng.uniform(6.0, 22.0))
                    if rng.random() < 0.20:                      # root plate scoured out; the tree settled
                        undercut = float(rng.uniform(0.4, 1.3))
                elif rng.random() < 0.25:
                    tilt = float(rng.uniform(1.0, 5.0))           # ordinary lean towards the light
                azim = 180.0 + float(rng.normal(0, 25.0)) if tilt > 5.5 else float(rng.uniform(0, 360))
                roll, pitch, yaw = lean_rotator(tilt, azim)
                base = gz - undercut - SINK_CM / 100.0
                top = base + trees[pid]["height_m"] * sc * math.cos(math.radians(tilt))
                # A tree whose crown is under the surface is 1-4 M triangles of geometry that renders as
                # nothing but a smudge under opaque flood water. The drowned valley floor is up to 6.7 m
                # deep (measured), so this rejects the small species there instead of paying for them.
                if top < water + 0.35:
                    rejected["drowned"] += 1
                    continue
                cand.append({
                    "pid": pid, "zone": zone, "east_m": round(e, 2), "north_m": round(nn, 2),
                    "base_asl_m": round(base, 3), "ground_asl_m": round(gz, 3),
                    "yaw_deg": round(yaw, 1), "pitch_deg": round(pitch, 2),
                    "roll_deg": round(roll, 2), "scale": round(sc, 3),
                    "tilt_deg": round(tilt, 1), "undercut_m": round(undercut, 2),
                    "crown_r_m": round(0.5 * trees[pid]["crown_major_m"] * sc, 2),
                    "crown_r_minor_m": round(0.5 * trees[pid]["crown_minor_m"] * sc, 2),
                    "flood_depth_m": round(depth, 2),
                    "emergent_m": round(top - water, 2) if depth > 0.05 else None,
                    "tris": int(trees[pid]["tris"]),
                })
                placed += 1
        zone_stats[zone] = {"cells": int(len(cells)), "target_closure": closure,
                            "mean_crown_m2": round(mean_area, 1),
                            "lambda_per_ha": round(lam * 1e4, 1), "placed": placed}

    # --- spend the budget nearest the camera first -------------------------------------------------------
    if cand:
        ce = np.array([c["east_m"] for c in cand])
        cn = np.array([c["north_m"] for c in cand])
        d2 = np.min((ce[:, None] - pts[None, :, 0]) ** 2 + (cn[:, None] - pts[None, :, 1]) ** 2, axis=1)
        order = np.lexsort((ce, cn, d2))                  # d2 first, ties broken deterministically
        cand = [cand[k] for k in order]
    dropped = 0
    if max_trees and len(cand) > max_trees:
        dropped = len(cand) - max_trees
        cand = cand[:max_trees]

    items = []
    for c in sorted(cand, key=lambda x: (x["north_m"], x["east_m"])):
        c = dict(c)
        c["id"] = len(items)
        c["name"] = f"Tree_{len(items):04d}"
        c["kind"] = "tree"
        c["asset_hint"] = trees[c["pid"]]["asset_hint"]
        items.append(c)

    # --- understorey ------------------------------------------------------------------------------------
    ground_items = []
    if ground_cover and any(g in cat for g in GROUND_SPECIES):
        gid = next(g for g in GROUND_SPECIES if g in cat)
        sub = cat[gid].get("sub_meshes") or [cat[gid]["asset_hint"]]
        tris_each = max(1, int(cat[gid]["tris"] / max(1, len(sub))))
        tree_xy = np.array([[t["east_m"], t["north_m"]] for t in items]) if items else np.zeros((0, 2))
        # riparian scrub in the 0-4 m band above the flood line, plus undergrowth under the canopy
        band = (h > water + 0.1) & (h < water + 4.0) & ok_slope & in_roi & far_pad
        under = (masks["hillslope"] | masks["garden"]) & in_roi
        for mask, note, prob in ((band, "riparian scrub above the flood line", 0.10),
                                 (under, "undergrowth under the canopy", 0.035)):
            cells = np.argwhere(mask)
            take = rng.random(len(cells)) < prob
            for (j, i) in cells[take]:
                e = float(xs[i]) + float(rng.uniform(-cell / 2, cell / 2))
                nn = float(xs[j]) + float(rng.uniform(-cell / 2, cell / 2))
                if not clear_of_houses(e, nn) or not clear_of_survivors(e, nn, 2.0):
                    continue
                if "undergrowth" in note and len(tree_xy):
                    if float(np.min(np.hypot(tree_xy[:, 0] - e, tree_xy[:, 1] - nn))) > 6.0:
                        continue                              # only under a crown, not a second fern scatter
                asset = sub[int(rng.integers(len(sub)))]
                ground_items.append({
                    "id": len(ground_items), "name": f"Under_{len(ground_items):04d}", "kind": "ground",
                    "pid": gid, "asset_hint": asset, "zone": "understorey",
                    "east_m": round(e, 2), "north_m": round(nn, 2),
                    "base_asl_m": round(ground_at(e, nn) - 0.04, 3),
                    "yaw_deg": round(float(rng.uniform(0, 360)), 1), "pitch_deg": 0.0, "roll_deg": 0.0,
                    "scale": round(float(rng.uniform(0.9, 1.9)), 3), "note": note, "tris": tris_each,
                })

    # --- achieved canopy closure, rasterised at 2 m over the ROI ------------------------------------------
    step = 2.0
    ge = np.arange(roi["east_m"][0], roi["east_m"][1], step)
    gn = np.arange(roi["north_m"][0], roi["north_m"][1], step)
    cover = np.zeros((len(gn), len(ge)), dtype=bool)
    for t in items:
        r = t["crown_r_m"]
        i0 = max(0, int((t["east_m"] - r - roi["east_m"][0]) / step))
        i1 = min(len(ge), int((t["east_m"] + r - roi["east_m"][0]) / step) + 1)
        j0 = max(0, int((t["north_m"] - r - roi["north_m"][0]) / step))
        j1 = min(len(gn), int((t["north_m"] + r - roi["north_m"][0]) / step) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        dd = ((ge[i0:i1][None, :] - t["east_m"]) / r) ** 2 + \
             ((gn[j0:j1][:, None] - t["north_m"]) / max(t["crown_r_minor_m"], 0.1)) ** 2
        cover[j0:j1, i0:i1] |= dd <= 1.0
    closure_roi = float(cover.mean())
    # closure inside the settlement box alone: that is what a nadir frame over the houses actually sees
    hx = np.array([b["east_m"] for b in town["houses"]])
    hy = np.array([b["north_m"] for b in town["houses"]])
    sel_e = (ge >= hx.min() - 40) & (ge <= hx.max() + 40)
    sel_n = (gn >= hy.min() - 40) & (gn <= hy.max() + 40)
    closure_town = float(cover[np.ix_(sel_n, sel_e)].mean()) if sel_e.any() and sel_n.any() else 0.0

    tris_total = sum(t["tris"] for t in items)
    unique_tris = sum(trees[p]["tris"] for p in sorted({t["pid"] for t in items}))
    emerg = [t["emergent_m"] for t in items if t["emergent_m"] is not None]
    by_pid = {p: sum(1 for t in items if t["pid"] == p) for p in sorted({t["pid"] for t in items})}
    by_zone = {z: sum(1 for t in items if t["zone"] == z) for z in ZONE_ORDER}

    return {
        "generated_by": "tools/scene/gen_vegetation.py", "seed": seed, "terrain_seed": meta["seed"],
        "settlement_seed": town["seed"], "water_level_m": water, "base_z_m": base_z,
        "sink_cm": SINK_CM, "clearances_m": {"house": CLEAR_HOUSE_M, "survivor": CLEAR_SURVIVOR_M,
                                             "launch_pad": CLEAR_PAD_M, "max_slope_deg": MAX_SLOPE_DEG},
        "catalogue": cat, "catalogue_notes": notes,
        "roi": roi, "zones": zone_stats,
        "nanite": {
            "required": True,
            "reason": "the tree scans are 1.07-3.86 M triangles each; without Nanite the placed instances "
                      "would rasterise {:,} triangles".format(tris_total),
            "no_nanite_max_tris": NO_NANITE_MAX_TRIS,
            "unique_mesh_tris": unique_tris,
        },
        "counts": {
            "trees": len(items), "trees_dropped_over_budget": dropped,
            "rejected_house_clearance": rejected["house"],
            "rejected_survivor_clearance": rejected["survivor"],
            "rejected_crown_below_water": rejected["drowned"],
            "by_species": by_pid, "by_zone": by_zone,
            "instanced_source_triangles": tris_total,
            "unique_mesh_triangles": unique_tris,
            "ground_cover": len(ground_items),
            "ground_cover_triangles": sum(g["tris"] for g in ground_items),
        },
        "canopy": {
            "closure_roi": round(closure_roi, 4), "closure_settlement": round(closure_town, 4),
            "raster_step_m": step, "roi_area_km2": roi["area_km2"],
        },
        "emergent_m": ({"n": len(emerg), "min": round(min(emerg), 2), "p10": round(float(np.percentile(emerg, 10)), 2),
                        "p50": round(float(np.percentile(emerg, 50)), 2),
                        "p90": round(float(np.percentile(emerg, 90)), 2), "max": round(max(emerg), 2)}
                       if emerg else None),
        "items": items, "ground_items": ground_items,
        "_cover_png": cover,
    }


def write_preview(d: dict, path: Path) -> None:
    """Canopy-closure map: grey = ROI, green = under a crown, blue = flood, red dots = houses.

    This is the artifact to LOOK at before anything is imported - it shows immediately whether the canopy
    thins in the channel, hugs the settlement and covers the hillslopes, none of which a count can tell you.
    """
    from PIL import Image, ImageDraw
    cover = d.pop("_cover_png")
    roi = d["roi"]
    meta = json.loads((OUT / "flood_valley.json").read_text())
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, size, n = s["height"], s["size_m"], s["n"]
    ge = np.linspace(roi["east_m"][0], roi["east_m"][1], cover.shape[1])
    gn = np.linspace(roi["north_m"][0], roi["north_m"][1], cover.shape[0])
    ii = np.clip(((ge + size / 2) / s["cell_m"]).astype(int), 0, n - 1)
    jj = np.clip(((gn + size / 2) / s["cell_m"]).astype(int), 0, n - 1)
    hh = h[np.ix_(jj, ii)]
    shade = np.clip((hh - hh.min()) / max(1e-3, hh.max() - hh.min()), 0, 1) * 0.45 + 0.25
    rgb = np.stack([shade, shade, shade], -1)
    wet = hh < s["water_level"]
    rgb[wet] = rgb[wet] * 0.35 + np.array([0.13, 0.30, 0.36]) * 0.65
    rgb[cover] = rgb[cover] * 0.30 + np.array([0.16, 0.42, 0.14]) * 0.70
    img = Image.fromarray((np.flipud(rgb) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for b in json.loads((OUT / "settlement.json").read_text())["houses"]:
        px = (b["east_m"] - roi["east_m"][0]) / (roi["east_m"][1] - roi["east_m"][0]) * cover.shape[1]
        py = cover.shape[0] - (b["north_m"] - roi["north_m"][0]) / (roi["north_m"][1] - roi["north_m"][0]) * cover.shape[0]
        dr.rectangle([px - 2, py - 2, px + 2, py + 2], fill=(220, 60, 40))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--max-trees", type=int, default=4600, help="0 = no cap; the cap thins the far hillslopes")
    ap.add_argument("--roi-margin", type=float, default=110.0, help="metres beyond the houses/survivors box")
    ap.add_argument("--no-ground-cover", action="store_true")
    a = ap.parse_args()
    d = build(a.seed, a.max_trees, a.roi_margin, not a.no_ground_cover)
    write_preview(d, OUT / "vegetation_canopy.png")
    (OUT / "vegetation.json").write_text(json.dumps(d, indent=1))
    c, k = d["counts"], d["canopy"]
    for line in d["catalogue_notes"]:
        print("  !", line)
    print(f"trees: {c['trees']:,} placed ({c['trees_dropped_over_budget']:,} dropped over --max-trees), "
          f"{c['ground_cover']:,} understorey clumps")
    print(f"  by species: {c['by_species']}")
    print(f"  by zone:    {c['by_zone']}")
    print(f"  rejected:   {c['rejected_house_clearance']} house-clearance, "
          f"{c['rejected_survivor_clearance']} survivor-clearance, "
          f"{c['rejected_crown_below_water']} crown below the water surface")
    print(f"canopy closure: {k['closure_roi'] * 100:.1f} % of the {k['roi_area_km2']} km2 ROI, "
          f"{k['closure_settlement'] * 100:.1f} % over the settlement")
    print(f"triangles: {c['instanced_source_triangles']:,} instanced source "
          f"({c['unique_mesh_triangles']:,} unique - Nanite stores it ONCE), "
          f"understorey {c['ground_cover_triangles']:,}")
    if d["emergent_m"]:
        print(f"emergent height above water (drowned trees, n={d['emergent_m']['n']}): "
              f"p10 {d['emergent_m']['p10']} m, p50 {d['emergent_m']['p50']} m, p90 {d['emergent_m']['p90']} m")
    print(f"written to {OUT / 'vegetation.json'} and {OUT / 'vegetation_canopy.png'}")


if __name__ == "__main__":
    main()
