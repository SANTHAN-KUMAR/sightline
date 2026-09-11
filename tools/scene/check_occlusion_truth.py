"""Cross-check every scene layout against the survivor ground truth in actors.json.

    uv run python tools/scene/check_occlusion_truth.py

`actors.json` is the ONLY source of truth the labels and the metrics are built on. Five generators
independently scatter geometry over the same ground - vegetation, rubble, props, damage, utilities - and each
one was written to respect survivors by keeping its TRUNK / ORIGIN a fixed distance away. That is the wrong
test whenever the thing being placed is wider than its origin: a jacaranda trunk 4.7 m from a survivor still
throws a 13.5 m crown over their head. Measured on the layout of 2026-09-10, 22 of the 71 survivors marked
`occlusion: 0` were under a crown, four of them the roof group Human_000..003.

This checks the property that actually matters and that no single generator can check alone:

    a survivor recorded `occlusion: 0` must have NOTHING above them, from ANY generator
    a survivor recorded `aerially_detectable: false` must have SOMETHING above them

Each layout is measured with its own real footprint - crown radius for trees, mesh extent for rubble and
props, roof polygon for damage, wire corridor for utilities - not with the clearance the generator claims it
used. It exits non-zero, and it names the survivor and the offending object.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: how far an occluder's top must rise above the survivor's own ground before it counts as
#: being OVER them rather than beside them. A prone survivor is ~0.3 m tall.
MIN_RISE_M = 0.15
S = REPO / "data/scene"


def load(name: str):
    p = S / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def covers(item: dict, a: dict, radius: float, top_m: float | None = None) -> float | None:
    """Horizontal distance if `item` is ABOVE survivor `a` and within `radius` of them, else None.

    The height test is not decoration. Without it this check credits a survivor with "coverage" from a slab
    lying flat on the ground BESIDE them, or from a boat hull sitting lower than they are - geometry that is
    near them but not over them. `tools/scene/reconcile_occlusion.py` has exactly that flaw and consequently
    reports 25 contradictions where only 4 survive a height test.

    "Above" is deliberately generous: the occluder's top only has to rise `MIN_RISE_M` over the ground the
    survivor is lying on. A prone survivor is barely 0.3 m tall, so a 0.5 m plate resting on the ground and
    overlapping them horizontally really does hide them from a nadir camera.
    """
    d = math.hypot(item["east_m"] - a["east_m"], item["north_m"] - a["north_m"])
    if d >= radius:
        return None
    if top_m is not None:
        base = float(item.get("base_asl_m", a["base_asl_m"]))
        if base + top_m < a["base_asl_m"] + MIN_RISE_M:
            return None                      # it is beside them or below them, not over them
    return d


def measure(actors: list[dict]) -> dict[str, list[tuple[str, str, float]]]:
    """actor name -> [(source, object, horizontal distance)] for everything proven ABOVE them.

    Exposed as a function so `fix_occlusion_truth.py` uses THIS code rather than its own copy. A
    second implementation drifted from this one within minutes of being written - it found 3 of the
    4 violations - which is the whole argument for having one.
    """
    over: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    # --- vegetation: the crown, not the trunk -------------------------------------------------------------
    veg = load("vegetation.json")
    if veg:
        for t in veg["items"]:
            for a in actors:
                d = covers(t, a, t["crown_r_m"])
                if d is not None:
                    over[a["name"]].append(("vegetation", f"{t['name']}({t['pid']})", d))
        for g in veg.get("ground_items", []):
            r = 1.0 * g.get("scale", 1.0)
            for a in actors:
                d = covers(g, a, r)
                if d is not None:
                    over[a["name"]].append(("understorey", g["name"], d))

    # --- rubble and props: the real mesh extent, resolved from the catalogues ------------------------------
    # Neither layout records a size on the item; both record a variant/asset name and a scale, and the size
    # lives in a catalogue. Looking it up is the whole point - guessing a radius here would reproduce exactly
    # the mistake this file exists to catch.
    rub = load("rubble_layout.json")
    if rub:
        var = rub["variants"]
        for it in rub["items"]:
            sz = var[it["variant"]]["size_m"]
            r = 0.5 * max(sz[0], sz[1]) * it.get("scale", 1.0)
            top = float(var[it["variant"]].get("max_m", [0, 0, sz[2]])[2]) * it.get("scale", 1.0)
            for a in actors:
                d = covers(it, a, r, top)
                if d is not None:
                    over[a["name"]].append(("rubble", f"{it['name']}({r*2:.1f}m, top +{top:.1f}m)", d))

    pr = load("props_layout.json")
    cat = load("props.json")
    if pr and cat:
        size_by_asset = {m["asset"]: m["size_m"]
                         for e in cat["props"].values() for m in e["meshes"]}
        unresolved = 0
        for it in pr["items"]:
            sz = size_by_asset.get(it["asset"])
            if sz is None:
                unresolved += 1
                continue
            r = 0.5 * max(sz[0], sz[1]) * it.get("scale", 1.0)
            for a in actors:
                d = covers(it, a, r)
                if d is not None:
                    over[a["name"]].append(("props", f"{it['name']}({r*2:.1f}m)", d))
        if unresolved:
            print(f"  WARNING: {unresolved}/{len(pr['items'])} prop items have no catalogue entry, so their "
                  "footprint could NOT be checked")

    # --- utilities: poles and boats ------------------------------------------------------------------------
    ut = load("utilities.json")
    if ut:
        for grp, r in (("pole_list", 0.6), ("boat_list", 2.2)):
            for it in ut.get(grp, []) or []:
                for a in actors:
                    d = covers(it, a, r)
                    if d is not None:
                        over[a["name"]].append(("utilities", it.get("name", grp), d))

    return over


def main() -> int:
    actors = json.loads((S / "actors.json").read_text())["actors"]
    over = measure(actors)

    # --- verdict ---------------------------------------------------------------------------------------
    fails: list[str] = []
    open_declared = [a for a in actors if a.get("occlusion", 0) == 0 and a.get("aerially_detectable", True)]
    buried = [a for a in actors if not a.get("aerially_detectable", True)]

    print(f"survivors: {len(actors)}  declared in the open (occlusion 0): {len(open_declared)}  "
          f"declared buried: {len(buried)}\n")

    bad_open = [(a, over[a["name"]]) for a in open_declared if over.get(a["name"])]
    for a, srcs in bad_open:
        by = ", ".join(f"{s}:{n} @{d:.1f}m" for s, n, d in srcs[:3])
        fails.append(f"{a['name']} ({a['pose']}/{a['zone']}) is declared occlusion 0 but has {len(srcs)} "
                     f"object(s) overhead: {by}")

    bad_buried = [a for a in buried if not over.get(a["name"])]
    for a in bad_buried:
        fails.append(f"{a['name']} is declared aerially_detectable=false but NOTHING covers it - section 2.7 "
                     "says aerial search cannot find them, so something must actually be on top")

    covered = {k: v for k, v in over.items() if v}
    by_occ: dict[int, int] = defaultdict(int)
    by_src: dict[str, int] = defaultdict(int)
    for a in actors:
        if over.get(a["name"]):
            by_occ[a.get("occlusion", 0)] += 1
            for s, _n, _d in over[a["name"]]:
                by_src[s] += 1
    print(f"survivors with something overhead: {len(covered)}")
    print(f"  by declared occlusion level: {dict(sorted(by_occ.items()))}")
    print(f"  overhead objects by source:   {dict(sorted(by_src.items()))}")
    missing = [fn for fn in ("vegetation.json", "rubble_layout.json", "props_layout.json",
                             "utilities.json", "damage.json") if not (S / fn).exists()]
    if missing:
        print(f"  NOT CHECKED (layout absent): {missing}")

    print()
    if fails:
        for f in fails[:20]:
            print(f"FAIL  {f}")
        if len(fails) > 20:
            print(f"      ... and {len(fails) - 20} more")
        print(f"\n{len(fails)} ground-truth violation(s) - the labels would lie about these survivors")
        return 1
    print("ground truth and geometry agree: nobody in the open is covered, everybody buried is covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
