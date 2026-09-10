"""Generate the FloodValley waterline foam as a real ribbon of geometry hugging the shoreline.

    uv run python tools/scene/gen_foam.py [--water-level ASL] [--spacing 4.0]
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_foam.py').read())"

Writes `data/scene/foam.obj` (CENTIMETRES, x = east, y = north, z = (asl - base_z) * 100, exactly the convention
`gen_terrain.py` uses, so the actor is placed at the origin with yaw +90 like `Ground`) and `data/scene/foam.json`.

WHY GEOMETRY AND NOT A LAYER IN M_FloodWater
--------------------------------------------
Foam only exists where the water meets something. Painting it into `M_FloodWater` would mean either a baked
shoreline mask texture or a screen-space trick, and either way it would mean CLEARING AND REBUILDING that
material - which `build_materials.py` owns. Two scripts rebuilding one material graph silently undo each other,
so the foam is a separate, additive actor instead: `build_materials.py` and `build_foam.py` never touch the same
asset, and the foam can be regenerated, retuned or deleted without risking the water.

HOW THE RIBBON IS BUILT
-----------------------
* A cell of the height grid is a SHORELINE cell when the water level falls between the minimum and the maximum
  of its four corners. Those cells are thinned to one quad per `--spacing` metres by a grid hash (deterministic,
  and O(n) instead of the O(n^2) of a greedy spacing pass).
* Each quad is oriented by the local terrain gradient: `across` runs downhill (towards the water), `along` runs
  parallel to the shore, and the centre is pushed 30 % of the ribbon width downhill so most of the quad lies
  over water rather than buried in the bank.
* UV0 carries (metres along the shore, 0..1 across the ribbon). The material turns the second component into a
  soft edge falloff `saturate(1 - abs(2v - 1))`, and each quad's strength is baked in by SHRINKING its v range
  about 0.5 - which costs nothing and, because the falloff is symmetric about v = 0.5, is immune to the V flip
  that UE's OBJ importer applies. The u origin is jittered per quad so no two patches show the same bubbles.
* Extra, stronger quads pile foam against the UPSTREAM wall of every flooded house and around any obstacle that
  stands at the waterline: that is where a real flood puts its foam and its wrack.
* Quads never come within `--clear-survivors` metres of a survivor. Foam is a separate actor with its own
  Cosys instance-segmentation colour, and a quad lying across a half-submerged survivor would corrupt the very
  masks the auto-labels are made from.

Every quad is a flat horizontal rectangle at `water_level + 3 cm` with a 0-3 cm jitter, so overlapping patches
are never exactly coplanar (which would z-fight) and the offset is invisible from survey altitude.
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
import gen_terrain as gt  # noqa: E402  (rebuild the same seeded height grid instead of reading the gitignored npy)

OUT = gt.OUT
RIBBON_W_M = 7.0          # across-shore width of a shoreline quad; the material needs this to scale the texture
RIBBON_L_M = 6.0          # along-shore length
Z_LIFT_CM = 3.0


class Ribbon:
    """Accumulates flat foam quads and writes them as one OBJ (single `usemtl foam` group)."""

    def __init__(self) -> None:
        self.v: list[tuple[float, float, float]] = []
        self.vt: list[tuple[float, float]] = []
        self.f: list[tuple[int, int, int]] = []

    def quad(self, ce, cn, z_m, e1, e2, along_m, across_m, strength, u0, base_z):
        """e1 = along-shore unit vector, e2 = across-shore (downhill) unit vector, both in (east, north)."""
        v_lo, v_hi = 0.5 - 0.5 * strength, 0.5 + 0.5 * strength
        ha, hb = along_m / 2.0, across_m / 2.0
        base = len(self.v)
        for su, sv, u, vv in ((-1, -1, u0, v_lo), (+1, -1, u0 + along_m, v_lo),
                              (+1, +1, u0 + along_m, v_hi), (-1, +1, u0, v_hi)):
            e = ce + su * ha * e1[0] + sv * hb * e2[0]
            n = cn + su * ha * e1[1] + sv * hb * e2[1]
            self.v.append((e * 100.0, n * 100.0, (z_m - base_z) * 100.0))
            self.vt.append((u, vv))
        a, b, c, d = base + 1, base + 2, base + 3, base + 4
        self.f.append((a, b, c))
        self.f.append((a, c, d))

    def write(self, path: Path) -> int:
        lines = ["# Sightline waterline foam (generated, centimetres, x = east, y = north)"]
        lines += [f"v {x:.1f} {y:.1f} {z:.1f}" for x, y, z in self.v]
        lines += [f"vt {u:.4f} {v:.4f}" for u, v in self.vt]
        lines.append("usemtl foam")
        lines += [f"f {i}/{i} {j}/{j} {k}/{k}" for i, j, k in self.f]
        path.write_text("\n".join(lines), encoding="utf-8")
        return len(self.f)


def perp(e2):
    """The along-shore axis, chosen so that (along, across, +z) is right-handed and every quad faces +z."""
    return (e2[1], -e2[0])


def build(spacing_m: float, water_level: float | None, clear_survivors_m: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text()) if (OUT / "settlement.json").exists() else None
    actors = json.loads((OUT / "actors.json").read_text())["actors"] if (OUT / "actors.json").exists() else []
    props = json.loads((OUT / "props_layout.json").read_text()) if (OUT / "props_layout.json").exists() else None

    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    base_z = meta["base_z_m"]
    wl = float(water_level if water_level is not None else meta["water_level_m"])
    xs = np.linspace(-size / 2, size / 2, n)

    # --- shoreline cells: the water level falls between the min and the max of the cell's four corners -----
    c00, c10, c01, c11 = h[:-1, :-1], h[1:, :-1], h[:-1, 1:], h[1:, 1:]
    lo = np.minimum(np.minimum(c00, c10), np.minimum(c01, c11))
    hi = np.maximum(np.maximum(c00, c10), np.maximum(c01, c11))
    cross = (lo < wl) & (hi > wl)
    dhdn, dhde = np.gradient(h, cell)                    # h is indexed [j = north, i = east]
    slope = np.hypot(dhde, dhdn)

    idx = np.argwhere(cross)
    rng.shuffle(idx)
    # thin by a grid hash: one quad per `spacing_m` cell, deterministic and O(n)
    seen, kept = set(), []
    for j, i in idx:
        e = float(xs[i]) + cell / 2.0
        nn = float(xs[j]) + cell / 2.0
        key = (int(math.floor(e / spacing_m)), int(math.floor(nn / spacing_m)))
        if key in seen:
            continue
        seen.add(key)
        kept.append((int(j), int(i), e, nn))

    def near_survivor(e, nn, r):
        r2 = r * r
        return any((a["east_m"] - e) ** 2 + (a["north_m"] - nn) ** 2 < r2 for a in actors)

    rb = Ribbon()
    shore_slopes = []
    n_shore = 0
    for j, i, e, nn in kept:
        if near_survivor(e, nn, clear_survivors_m):
            continue
        gx, gy = float(dhde[j, i]), float(dhdn[j, i])
        g = math.hypot(gx, gy)
        if g < 1e-6:
            e2 = (1.0, 0.0)
        else:
            e2 = (-gx / g, -gy / g)                      # downhill = towards the water
        e1 = perp(e2)
        shore_slopes.append(g)
        # a gentle margin holds a wide raft of foam; a steep cut bank holds a thin line of it
        strength = float(np.clip(0.95 - 0.80 * min(g, 1.0), 0.35, 0.95)) * float(rng.uniform(0.82, 1.0))
        z = wl + (Z_LIFT_CM + float(rng.uniform(0.0, 3.0))) / 100.0
        rb.quad(e + e2[0] * 0.30 * RIBBON_W_M, nn + e2[1] * 0.30 * RIBBON_W_M, z, e1, e2,
                RIBBON_L_M * float(rng.uniform(0.85, 1.15)), RIBBON_W_M, strength,
                float(rng.uniform(0.0, 40.0)), base_z)
        n_shore += 1

    # --- foam piled against the upstream wall of every flooded house --------------------------------------
    n_house = 0
    if town:
        arch = town["archetypes"]
        for b in town["houses"]:
            if b["flood_depth_m"] <= 0.3:
                continue
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            # the four outward wall normals in world (east, north); take the one facing most upstream (+north)
            cand = [((math.cos(th), math.sin(th)), a["length_m"] / 2, a["width_m"]),
                    ((-math.cos(th), -math.sin(th)), a["length_m"] / 2, a["width_m"]),
                    ((-math.sin(th), math.cos(th)), a["width_m"] / 2, a["length_m"]),
                    ((math.sin(th), -math.cos(th)), a["width_m"] / 2, a["length_m"])]
            e2, off, wall_len = max(cand, key=lambda t: t[0][1])
            e1 = perp(e2)
            for k in range(3):
                t = (k - 1) * wall_len / 3.2
                ce = b["east_m"] + e2[0] * (off + 1.3) + e1[0] * t
                cn = b["north_m"] + e2[1] * (off + 1.3) + e1[1] * t
                if near_survivor(ce, cn, clear_survivors_m):
                    continue
                z = wl + (Z_LIFT_CM + float(rng.uniform(0.0, 3.0))) / 100.0
                rb.quad(ce, cn, z, e1, e2, wall_len / 2.6, 4.6, float(rng.uniform(0.80, 1.0)),
                        float(rng.uniform(0.0, 40.0)), base_z)
                n_house += 1

    # --- a collar around anything grounded that stands at the waterline -----------------------------------
    n_obst = 0
    if props:
        for it in props["items"]:
            if it["floats"] or not (wl - 1.0 <= it["base_asl_m"] <= wl + 0.2):
                continue
            e2 = (0.0, 1.0)                              # the flow runs south, so foam piles on the north face
            e1 = perp(e2)
            ce, cn = it["east_m"], it["north_m"] + 0.8
            if near_survivor(ce, cn, clear_survivors_m):
                continue
            z = wl + (Z_LIFT_CM + float(rng.uniform(0.0, 3.0))) / 100.0
            rb.quad(ce, cn, z, e1, e2, 2.4, 2.8, float(rng.uniform(0.6, 0.9)),
                    float(rng.uniform(0.0, 40.0)), base_z)
            n_obst += 1

    tris = rb.write(OUT / "foam.obj")
    sl = np.array(shore_slopes) if shore_slopes else np.zeros(1)
    # Expected WORLD bounds of the actor: the OBJ carries (x = east, y = north), UE's importer flips it to
    # (UE X = x, UE Y = -y) and the actor's yaw +90 then maps it to (UE X = north, UE Y = east) - identical to
    # the terrain. build_foam.py asserts the spawned actor's real bounds against these, which is the only way to
    # catch a wrong transform without looking at a render.
    ex = [v[0] for v in rb.v] or [0.0]
    ny = [v[1] for v in rb.v] or [0.0]
    nz = [v[2] for v in rb.v] or [0.0]
    return {
        "generated_by": "tools/scene/gen_foam.py", "seed": seed, "terrain_seed": meta["seed"],
        "water_level_m": wl, "base_z_m": base_z,
        "ribbon_width_m": RIBBON_W_M, "ribbon_length_m": RIBBON_L_M, "z_lift_cm": Z_LIFT_CM,
        "spacing_m": spacing_m, "clear_survivors_m": clear_survivors_m,
        "obj": "foam.obj",
        "ue_actor": {"location_cm": [0.0, 0.0, 0.0], "yaw_deg": 90.0,
                     "bounds_cm": {"x_north": [round(min(ny), 1), round(max(ny), 1)],
                                   "y_east": [round(min(ex), 1), round(max(ex), 1)],
                                   "z": [round(min(nz), 1), round(max(nz), 1)]}},
        "counts": {
            "shoreline_cells": int(cross.sum()), "quads_after_thinning": len(kept),
            "quads_shoreline": n_shore, "quads_houses": n_house, "quads_obstacles": n_obst,
            "quads_total": tris // 2, "triangles": tris, "vertices": len(rb.v),
            "approx_shoreline_km": round(len(kept) * spacing_m / 1000.0, 2),
        },
        "shoreline_slope": {"mean": round(float(sl.mean()), 4), "p10": round(float(np.percentile(sl, 10)), 4),
                            "p50": round(float(np.percentile(sl, 50)), 4),
                            "p90": round(float(np.percentile(sl, 90)), 4)},
        # exact grid NODES (not cell centres) so build_foam.py can line-trace the east/north mapping against
        # the generator's own height and catch a wrong actor transform before anything is captured
        "probe_points": [{"east_m": round(float(xs[i]), 2), "north_m": round(float(xs[j]), 2),
                          "terrain_asl_m": round(float(h[j, i]), 3)}
                         for j, i, _e, _n in kept[:8]],
    }


def main(spacing_m, water_level, clear_survivors_m, seed):
    d = build(spacing_m, water_level, clear_survivors_m, seed)
    (OUT / "foam.json").write_text(json.dumps(d, indent=1))
    c = d["counts"]
    print(f"shoreline cells: {c['shoreline_cells']:,} -> {c['quads_after_thinning']:,} quads after thinning "
          f"at {d['spacing_m']} m (~{c['approx_shoreline_km']} km of shoreline)")
    print(f"quads: {c['quads_shoreline']:,} shoreline + {c['quads_houses']} upstream-of-house + "
          f"{c['quads_obstacles']} obstacle = {c['quads_total']:,}")
    print(f"triangles: {c['triangles']:,}  vertices: {c['vertices']:,}")
    print(f"shoreline slope (dz/dx): {d['shoreline_slope']}")
    print(f"written to {OUT / 'foam.obj'} and {OUT / 'foam.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=4.0, help="one foam quad per this many metres of shoreline")
    ap.add_argument("--water-level", type=float, default=None, help="ASL metres; defaults to flood_valley.json")
    ap.add_argument("--clear-survivors", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    main(a.spacing, a.water_level, a.clear_survivors, a.seed)
