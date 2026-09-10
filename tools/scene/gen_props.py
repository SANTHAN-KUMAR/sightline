"""Generate the FloodValley debris field (SOLUTION_DOC 2.2, 2.3): where flood debris actually ends up.

    uv run python tools/scene/gen_props.py [--seed 31]

Writes `data/scene/props_layout.json`. Debris is placed by the transport physics of a debris-flow-fed flood,
not scattered uniformly, because the arrangement is what makes a scene read as a disaster:

  * deposit fan (upstream)  the flow loses capacity as it spreads, so it drops its coarsest load FIRST:
                            boulders and whole trunks high on the fan, cobbles and branches further down.
  * wrack line              the single most recognisable feature of a real flood. Buoyant debris - drums,
                            crates, plastics, sheeting - rafts downstream until it jams against the first
                            obstacle and piles there. Here that is the upstream face of every flooded house.
  * open water              a thin scatter of floating items caught in eddies, denser near the channel margin.
  * channel                 scoured clean: fast water leaves bare bed, so almost nothing rests in it.
  * hillslopes              undisturbed vegetation above the flood line; the fern is the only asset available.

Floating items are placed AT the flood surface; grounded items sit on the terrain. Nothing is placed inside a
house footprint.
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
import gen_terrain as gt  # noqa: E402

OUT = gt.OUT

# role -> (floats?, typical scale range, how far it travels)
ROLE = {
    "boulder": dict(floats=False, scale=(0.9, 2.6)),
    "woody": dict(floats=True, scale=(0.8, 1.8)),
    "household": dict(floats=True, scale=(0.9, 1.3)),
    "vehicle": dict(floats=False, scale=(1.0, 1.0)),
    "vegetation": dict(floats=False, scale=(0.8, 1.6)),
}


def build(seed: int, max_tris: int = 6_500_000) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    props = json.loads((OUT / "props.json").read_text())["props"]

    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    xs = np.linspace(-size / 2, size / 2, n)

    def ground(e, nn):
        i = int(round((e + size / 2) / cell)); j = int(round((nn + size / 2) / cell))
        return float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])

    by_role: dict[str, list] = {}
    for name, p in props.items():
        by_role.setdefault(p["role"], []).extend(
            (name, m["asset"], m.get("size_m", [1, 1, 1]), m.get("tris", 0)) for m in p["meshes"])
    if not by_role:
        raise SystemExit("props.json has no meshes: run tools/scene/import_assets.py first")

    houses = town["houses"]
    arch = town["archetypes"]
    items: list[dict] = []
    tris = 0

    def add(kind, role, asset, e, nn, asl, yaw, scale, note, mesh_tris):
        """Returns False once the triangle budget is spent.

        The Poly Haven scans are 12k-53k triangles each and there is no LOD chain, so an unbounded scatter
        blows past what 8 GB of VRAM will hold: the first pass produced 1574 items / 19.4M triangles. The
        budget is enforced here rather than discovered as a crash or a 4 fps editor.
        """
        nonlocal tris
        if tris + mesh_tris > max_tris:
            return False
        items.append({
            "id": len(items), "name": f"Debris_{kind}_{len(items):04d}",
            "role": role, "asset": asset,
            "east_m": round(float(e), 2), "north_m": round(float(nn), 2), "base_asl_m": round(float(asl), 3),
            "yaw_deg": round(float(yaw) % 360.0, 1),
            "pitch_deg": round(float(rng.normal(0, 6)), 1), "roll_deg": round(float(rng.normal(0, 6)), 1),
            "scale": round(float(scale), 3), "floats": bool(asl >= water - 0.05), "note": note,
        })
        tris += mesh_tris
        return True

    def pick(role):
        """Bias towards the cheaper meshes of a role: at survey altitude a 53k-triangle rock and a 12k one are
        the same handful of pixels, so the extra geometry buys nothing."""
        pool = sorted(by_role.get(role) or by_role["household"], key=lambda t: t[3])
        k = int(rng.integers(0, max(1, int(len(pool) * 0.7)))) if len(pool) > 2 else int(rng.integers(len(pool)))
        return pool[k]

    def clear_of_houses(e, nn, margin=1.0):
        for b in houses:
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = e - b["east_m"], nn - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 + margin and abs(v) < a["width_m"] / 2 + margin:
                return False
        return True

    # --- 1. deposit fan: coarse load dropped first, fining downstream ---------------------------------------
    fan = np.argwhere(s["fan"] & ~s["in_channel"])
    rng.shuffle(fan)
    fan_n = [float(xs[j]) for j, _ in fan]
    apex = max(fan_n) if fan_n else 0.0
    placed = 0
    for j, i in fan:
        if placed >= 240:
            break
        e, nn = float(xs[i]), float(xs[j])
        gz = ground(e, nn)
        if gz < water - 0.2 or not clear_of_houses(e, nn):
            continue
        # downstream fraction: 0 at the apex, 1 at the toe of the fan
        f = 0.0 if apex == min(fan_n) else float(np.clip((apex - nn) / max(1.0, apex - min(fan_n)), 0, 1))
        if rng.random() > 0.55 - 0.25 * f:          # thinning towards the toe
            continue
        r = rng.random()
        role = "boulder" if r < 0.62 - 0.30 * f else "woody"
        kind, asset, _sz, mt = pick(role)
        lo, hi = ROLE[role]["scale"]
        sc = float(rng.uniform(lo, hi)) * (1.0 - 0.45 * f)      # coarse high, fine low
        add(kind, role, asset, e, nn, gz, rng.uniform(0, 360), sc,
            "dropped by the debris flow, coarse near the apex", mt)
        placed += 1

    # --- 2. the wrack line: rafted debris jammed against the upstream wall of each flooded house ------------
    # The flow runs downstream (towards -north), so the upstream face is the +north side of each building.
    for b in houses:
        if b["flood_depth_m"] <= 0.4:
            continue
        a = arch[b["archetype"]]
        th = math.radians(b["yaw_deg"])
        count = int(rng.integers(2, 6))
        for _ in range(count):
            # along the upstream wall, with a little pile-up depth
            u = float(rng.uniform(-0.55, 0.55)) * a["length_m"]
            v = a["width_m"] / 2 + float(rng.uniform(0.4, 2.2))
            e = b["east_m"] + u * math.cos(th) - v * math.sin(th)
            nn = b["north_m"] + u * math.sin(th) + v * math.cos(th)
            role = "household" if rng.random() < 0.72 else "woody"
            kind, asset, _sz, mt = pick(role)
            lo, hi = ROLE[role]["scale"]
            add(kind, role, asset, e, nn, water, rng.uniform(0, 360), float(rng.uniform(lo, hi)),
                f"wrack line against house {b['id']}", mt)

    # --- 3. open water: a thinner scatter of floating items, denser near the channel margin ----------------
    wet = np.argwhere(s["terrace"] & ~s["in_channel"])
    rng.shuffle(wet)
    placed = 0
    for j, i in wet:
        if placed >= 130:
            break
        e, nn = float(xs[i]), float(xs[j])
        if ground(e, nn) > water - 0.3 or not clear_of_houses(e, nn, 2.0):
            continue
        if rng.random() > 0.16:
            continue
        role = "household" if rng.random() < 0.8 else "woody"
        kind, asset, _sz, mt = pick(role)
        lo, hi = ROLE[role]["scale"]
        add(kind, role, asset, e, nn, water, rng.uniform(0, 360), float(rng.uniform(lo, hi)),
            "floating in open water", mt)
        placed += 1

    # --- 4. vehicles: swept against obstacles, a strong scale cue from the air ------------------------------
    veh = by_role.get("vehicle", [])
    if veh:
        flooded = [b for b in houses if b["flood_depth_m"] > 0.5]
        rng.shuffle(flooded)
        for b in flooded[:6]:
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            v = a["width_m"] / 2 + 3.4
            e = b["east_m"] - v * math.sin(th) + float(rng.uniform(-4, 4))
            nn = b["north_m"] + v * math.cos(th) + float(rng.uniform(-2, 2))
            for kind, asset, _sz, mt in veh:
                add(kind, "vehicle", asset, e, nn, water - 0.55, b["yaw_deg"] + rng.uniform(-40, 40), 1.0,
                    f"vehicle swept against house {b['id']}", mt)

    # --- 5. vegetation on the dry hillslopes ---------------------------------------------------------------
    veg = by_role.get("vegetation", [])
    if veg:
        dry = np.argwhere(~s["fan"] & ~s["in_channel"] & ~s["terrace"] & ~s["bank"])
        rng.shuffle(dry)
        placed = 0
        for j, i in dry:
            if placed >= 260:
                break
            e, nn = float(xs[i]), float(xs[j])
            gz = ground(e, nn)
            if gz < water + 1.5:
                continue
            if rng.random() > 0.30:
                continue
            kind, asset, _sz, mt = veg[int(rng.integers(len(veg)))]
            lo, hi = ROLE["vegetation"]["scale"]
            add(kind, "vegetation", asset, e, nn, gz, rng.uniform(0, 360), float(rng.uniform(lo, hi)),
                "hillslope vegetation above the flood line", mt)
            placed += 1

    roles = sorted({it["role"] for it in items})
    return {
        "generated_by": "tools/scene/gen_props.py", "seed": seed, "terrain_seed": meta["seed"],
        "water_level_m": water, "base_z_m": meta["base_z_m"],
        "counts": {"total": len(items),
                   "by_role": {r: sum(1 for it in items if it["role"] == r) for r in roles},
                   "floating": sum(1 for it in items if it["floats"]),
                   "grounded": sum(1 for it in items if not it["floats"]),
                   "approx_triangles": int(tris), "triangle_budget": int(max_tris)},
        "items": items,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--max-tris", type=int, default=6_500_000)
    a = ap.parse_args()
    d = build(a.seed, a.max_tris)
    (OUT / "props_layout.json").write_text(json.dumps(d, indent=1))
    c = d["counts"]
    print(f"debris items: {c['total']}  (floating {c['floating']}, grounded {c['grounded']})")
    print(f"  by role: {c['by_role']}")
    print(f"  approx triangles: {c['approx_triangles']:,}")
    print(f"written to {OUT / 'props_layout.json'}")
