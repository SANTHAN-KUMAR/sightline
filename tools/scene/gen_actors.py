"""Generate the FloodValley survivor/animal layout (SOLUTION_DOC 2.3 rows 2-3, 6.2).

    uv run python tools/scene/gen_actors.py [--seed 23]

Writes `data/scene/actors.json`: the seeded placement plan AND the ground truth the whole evaluation is measured
against. Every actor carries posture, submersion, expected occlusion, zone, and whether it is findable from the
air at all. Randomisation is OFF by default (SOLUTION_DOC 5.5c): one seed produces one fixed scenario, and the
train/held-out split is BY SCENARIO SEED, so generating a second scenario is just a different --seed.

Placement follows the causal model of section 2.3 rather than uniform random scatter:
  * flooded settlement terrace  people on roofs and upper floors, some in groups (the count_estimate case),
                                some clinging in the water beside a wall
  * channel and water margin    half-submerged and head-only cases, the hardest RGB slice
  * deposit fan                 prone / supine / trapped, the debris-flow casualties
  * dry high ground             standing and waving, the easy slice that anchors recall

Section 2.7's burial boundary is respected honestly: actors marked `aerially_detectable: false` are buried under
debris and CANNOT be found from the air. They stay in the ground truth so the coverage map can hatch those cells
as "aerial search cannot clear" (guardrail R10) instead of counting them as searched.
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
import gen_terrain as gt  # noqa: E402  same seeded terrain, so masks and heights match the imported mesh

OUT = gt.OUT
REPO = HERE.parents[1]

# Building geometry, kept in step with gen_buildings.py (imported rather than duplicated where possible).
STOREY_H, PLINTH = 3.0, 0.45
ROOF_PITCH_DEG = 26.0


def roof_top_m(arch: dict) -> float:
    """Height above the house's base at which a person standing on the roof has their feet."""
    h = PLINTH + arch["storeys"] * STOREY_H
    if arch["roof"] == "gable":
        return h + 0.35                                    # on the slope, not the ridge
    if arch["roof"] == "mono":
        return h + 0.30
    return h + 0.20                                        # flat slab


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    poses = json.loads((OUT / "poses.json").read_text())
    chars = sorted(poses["characters"])
    adults = [c for c in chars if "Adult" in c]
    children = [c for c in chars if "Child" in c]

    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    xs = np.linspace(-size / 2, size / 2, n)

    def ground(east_m: float, north_m: float) -> float:
        i = int(round((east_m + size / 2) / cell))
        j = int(round((north_m + size / 2) / cell))
        return float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])

    actors: list[dict] = []

    def add(cls, char, pose, east, north, base_asl, yaw, zone, occl, detectable=True, group=None, note=""):
        g = poses["characters"][char][pose]
        # Submersion from the pelvis height against the flood surface, exactly as SOLUTION_DOC 5.5a defines it.
        # `head_z_cm` and `height_cm` are the pose's own FK geometry, so this is consistent per posture.
        feet = base_asl
        head = feet + g["height_cm"] / 100.0
        pelvis = feet + 0.55 * g["height_cm"] / 100.0
        depth = water - feet
        if depth <= 0.02:
            sub = "dry"
        elif head <= water:
            sub = "unknown"                                  # fully under: not visible, handled by detectable
        elif pelvis >= water:
            sub = "wet"
        elif head - water < 0.25:
            sub = "head_only"
        elif depth >= 0.5 * (head - feet):
            sub = "half"
        else:
            sub = "partial"
        actors.append({
            "id": len(actors), "name": f"{'Human' if cls == 'human' else 'Animal'}_{len(actors):03d}",
            "cls": cls, "character": char, "pose": pose,
            "east_m": round(float(east), 2), "north_m": round(float(north), 2),
            "base_asl_m": round(float(base_asl), 3), "yaw_deg": round(float(yaw) % 360.0, 1),
            "ground_offset_cm": g["ground_offset_cm"],
            "zone": zone, "submersion": sub, "occlusion": int(occl),
            "flood_depth_m": round(float(max(0.0, depth)), 2),
            "height_cm": g["height_cm"], "footprint_cm": g["bbox_cm"][:2],
            "aerially_detectable": bool(detectable), "group": group, "note": note,
            "asset": f"/Game/Sightline/Characters/Rocketbox/{char}/{char}",
            "pose_asset": g["asset"],
        })

    # --- 1. flooded settlement: people on roofs, the signature presentation (2.3 row 2) --------------------
    flooded = [b for b in town["houses"] if b["flood_depth_m"] > 0.3]
    dry_houses = [b for b in town["houses"] if b["flood_depth_m"] <= 0.3]
    rng.shuffle(flooded)
    arch = town["archetypes"]
    n_roof = min(len(flooded), 26)
    for k, b in enumerate(flooded[:n_roof]):
        a = arch[b["archetype"]]
        top = b["base_asl_m"] + roof_top_m(a)
        # a group of 2-4 on roughly every fourth roof: this is what count_estimate has to recover
        size_g = int(rng.choice([2, 3, 4])) if k % 4 == 0 else 1
        gid = f"roof{b['id']}" if size_g > 1 else None
        for m in range(size_g):
            pose = str(rng.choice(["sitting", "standing", "waving", "sitting"]))
            char = str(rng.choice(children if rng.random() < 0.18 else adults))
            # keep them on the slab, inside the footprint
            du = float(rng.uniform(-0.35, 0.35)) * a["length_m"]
            dv = float(rng.uniform(-0.3, 0.3)) * a["width_m"]
            th = math.radians(b["yaw_deg"])
            e = b["east_m"] + du * math.cos(th) - dv * math.sin(th)
            nn = b["north_m"] + du * math.sin(th) + dv * math.cos(th)
            add("human", char, pose, e, nn, top, rng.uniform(0, 360), "settlement",
                occl=0 if a["roof"] != "gable" else 1, group=gid,
                note="on roof" + (f", group of {size_g}" if size_g > 1 else ""))

    # --- 2. in the water beside a wall: half-submerged / head-only, the hardest RGB slice ------------------
    for b in flooded[n_roof:n_roof + 10]:
        a = arch[b["archetype"]]
        th = math.radians(b["yaw_deg"])
        off = a["width_m"] / 2 + 1.1
        e = b["east_m"] - off * math.sin(th)
        nn = b["north_m"] + off * math.cos(th)
        gz = ground(e, nn)
        # stand them on the bed so the water cuts the body: the pose puts the head just above the surface
        stand = water - float(rng.uniform(1.25, 1.62))
        add("human", str(rng.choice(adults)), "half_submerged", e, nn, max(gz, stand),
            rng.uniform(0, 360), "settlement", occl=1, note="clinging beside a wall, in the water")

    # --- 3. deposit fan: prone / supine / trapped casualties (2.3 row 3) -----------------------------------
    fan = np.argwhere(s["fan"] & ~s["in_channel"])
    rng.shuffle(fan)
    placed = 0
    for j, i in fan:
        if placed >= 14:
            break
        e, nn = float(xs[i]), float(xs[j])
        gz = ground(e, nn)
        if gz < water + 0.4:                                  # keep the fan casualties out of the water
            continue
        if any((a["east_m"] - e) ** 2 + (a["north_m"] - nn) ** 2 < 400 for a in actors):
            continue
        buried = placed % 7 == 6                              # a minority are under the deposit: NOT findable
        pose = "trapped" if buried or placed % 3 == 2 else str(rng.choice(["prone", "supine"]))
        add("human", str(rng.choice(adults)), pose, e, nn, gz, rng.uniform(0, 360), "fan",
            occl=2 if buried else int(rng.choice([0, 1, 1])), detectable=not buried,
            note="buried under debris: aerial search cannot clear (2.7)" if buried else "casualty on the deposit fan")
        placed += 1

    # --- 4. dry high ground: the easy slice that anchors the recall number ---------------------------------
    for b in dry_houses[:8]:
        e = b["east_m"] + float(rng.uniform(-9, 9))
        nn = b["north_m"] + float(rng.uniform(-9, 9))
        add("human", str(rng.choice(adults)), str(rng.choice(["standing", "waving"])), e, nn,
            ground(e, nn), rng.uniform(0, 360), "settlement", occl=0, note="on dry ground beside a house")

    return {
        "generated_by": "tools/scene/gen_actors.py",
        "seed": seed, "terrain_seed": meta["seed"], "water_level_m": water,
        "base_z_m": meta["base_z_m"],
        "note": "ground truth for F5 auto-labels; UE cm = (north*100, east*100, (asl - base_z)*100)",
        "counts": {
            "total": len(actors),
            "aerially_detectable": sum(1 for a in actors if a["aerially_detectable"]),
            "buried_not_detectable": sum(1 for a in actors if not a["aerially_detectable"]),
            "by_pose": {p: sum(1 for a in actors if a["pose"] == p) for p in sorted({a["pose"] for a in actors})},
            "by_submersion": {v: sum(1 for a in actors if a["submersion"] == v)
                              for v in sorted({a["submersion"] for a in actors})},
            "by_zone": {v: sum(1 for a in actors if a["zone"] == v) for v in sorted({a["zone"] for a in actors})},
            "by_occlusion": {str(v): sum(1 for a in actors if a["occlusion"] == v) for v in (0, 1, 2)},
            "groups": len({a["group"] for a in actors if a["group"]}),
        },
        "actors": actors,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    data = build(args.seed)
    (OUT / "actors.json").write_text(json.dumps(data, indent=1))
    c = data["counts"]
    print(f"actors: {c['total']}  (detectable {c['aerially_detectable']}, buried {c['buried_not_detectable']}, "
          f"groups {c['groups']})")
    for k in ("by_pose", "by_submersion", "by_zone", "by_occlusion"):
        print(f"  {k:15s} {c[k]}")
    print(f"written to {OUT / 'actors.json'}")
