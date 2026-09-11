"""Make the survivor occlusion ground truth match the geometry that is actually in the scene.

    uv run python tools/scene/reconcile_occlusion.py            # report only
    uv run python tools/scene/reconcile_occlusion.py --write    # update data/scene/actors.json

`gen_actors.py` assigns each survivor an INTENDED occlusion level when it lays the scenario out. Five other
generators then scatter geometry over the same ground, and two of them do it deliberately: `gen_rubble.py`
leans slabs back over the fan survivors ("the plate roofs the void without filling it") to create the
25-75 % occlusion slice, and `gen_props.py` caps the buried survivors of section 2.7. Nothing ever told
`actors.json`. The result is a survivor recorded `occlusion: 0` with a concrete slab over them, and a label
that claims a clear aerial view of someone the camera cannot see.

Measured on the 2026-09-10 layouts: 22 survivors marked `occlusion: 0` were under a tree crown, and 4 more
under rubble or props.

This measures the coverage from the FINAL layouts and writes it back:

    occlusion         the measured level - what a nadir camera will actually see (0 clear, 1 partial, 2 heavy)
    occlusion_intent  what gen_actors.py asked for, kept so the difference stays auditable
    occluded_by       the sources and the fraction each contributes

Coverage is the fraction of the survivor's own footprint disc that overlapping geometry covers, computed as
the area of the circle-circle intersection - not a centre-distance test, because a 13 m crown whose edge
clips a survivor is not the same thing as one centred over them.

    level 0   coverage <  0.10
    level 1   0.10 <= coverage < 0.60
    level 2   coverage >= 0.60

`aerially_detectable` is NOT recomputed here: section 2.7 is a scenario decision about who the search can
find, and `gen_props.py` places the burial caps to honour it. This only reports when the two disagree.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S = REPO / "data/scene"

L1, L2 = 0.10, 0.60          # coverage thresholds for occlusion level 1 and level 2


def lens_area(r1: float, r2: float, d: float) -> float:
    """Area of the intersection of two circles of radii r1, r2 whose centres are d apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return r1 * r1 * (a1 - math.sin(2 * a1) / 2) + r2 * r2 * (a2 - math.sin(2 * a2) / 2)


def load(name: str):
    p = S / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def overhead(actors: list[dict]):
    """Return (actor name -> {source: covered_m2}, actor name -> footprint radius)."""
    cov: dict[str, defaultdict] = {a["name"]: defaultdict(float) for a in actors}
    # A survivor's own footprint, as a disc of the same area as the recorded footprint rectangle.
    own_r = {a["name"]: math.sqrt(max(1e-4, (a["footprint_cm"][0] / 100.0)
                                      * (a["footprint_cm"][1] / 100.0)) / math.pi) for a in actors}

    def add(src: str, e: float, n: float, r: float) -> None:
        if r <= 0:
            return
        for a in actors:
            d = math.hypot(e - a["east_m"], n - a["north_m"])
            ra = own_r[a["name"]]
            if d < r + ra:
                cov[a["name"]][src] += lens_area(r, ra, d)

    veg = load("vegetation.json")
    if veg:
        for t in veg["items"]:
            add("canopy", t["east_m"], t["north_m"], t["crown_r_m"])
        for g in veg.get("ground_items", []):
            add("understorey", g["east_m"], g["north_m"], 1.0 * g.get("scale", 1.0))

    rub = load("rubble_layout.json")
    if rub:
        var = rub["variants"]
        for it in rub["items"]:
            sz = var[it["variant"]]["size_m"]
            add("rubble", it["east_m"], it["north_m"], 0.5 * max(sz[0], sz[1]) * it.get("scale", 1.0))

    pr, cat = load("props_layout.json"), load("props.json")
    if pr and cat:
        size_by_asset = {m["asset"]: m["size_m"] for e in cat["props"].values() for m in e["meshes"]}
        unresolved = 0
        for it in pr["items"]:
            sz = size_by_asset.get(it["asset"])
            if sz is None:
                unresolved += 1
                continue
            add("props", it["east_m"], it["north_m"], 0.5 * max(sz[0], sz[1]) * it.get("scale", 1.0))
        if unresolved:
            print(f"WARNING: {unresolved}/{len(pr['items'])} props have no catalogue size and were NOT "
                  "measured - their coverage is missing from every number below")

    ut = load("utilities.json")
    if ut:
        for it in ut.get("pole_list", []) or []:
            add("utilities", it["east_m"], it["north_m"], 0.35)
    return {k: dict(v) for k, v in cov.items()}, own_r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    doc = json.loads((S / "actors.json").read_text(encoding="utf-8"))
    actors = doc["actors"]
    cov, own_r = overhead(actors)

    changed, rows = 0, []
    for x in actors:
        area = math.pi * own_r[x["name"]] ** 2
        parts = {k: round(min(1.0, v / area), 3) for k, v in cov[x["name"]].items() if v / area > 0.01}
        frac = min(1.0, sum(cov[x["name"]].values()) / area)
        lvl = 2 if frac >= L2 else (1 if frac >= L1 else 0)
        intent = x.get("occlusion_intent", x.get("occlusion", 0))
        rows.append((x, intent, lvl, frac, parts))
        if lvl != x.get("occlusion", 0):
            changed += 1

    hdr = ("survivor", "zone", "pose", "intent", "measured", "cover")
    print(f"{hdr[0]:12s} {hdr[1]:11s} {hdr[2]:9s} {hdr[3]:>6s} {hdr[4]:>8s} {hdr[5]:>6s}  sources")
    for x, intent, lvl, frac, parts in rows:
        if lvl != intent or parts:
            print(f"{x['name']:12s} {x['zone']:11s} {x['pose']:9s} {intent:>6d} {lvl:>8d} "
                  f"{frac:>6.2f}  {parts}")

    dist_i, dist_m = defaultdict(int), defaultdict(int)
    for x, intent, lvl, _f, _p in rows:
        dist_i[intent] += 1
        dist_m[lvl] += 1
    print()
    print(f"intended occlusion distribution: {dict(sorted(dist_i.items()))}")
    print(f"measured occlusion distribution: {dict(sorted(dist_m.items()))}")
    print(f"{changed} survivor(s) would change level")

    mism = [x["name"] for x, _i, lvl, _f, _p in rows
            if not x.get("aerially_detectable", True) and lvl < 2]
    if mism:
        print(f"WARNING: declared undetectable but measured coverage below {L2}: {mism} - section 2.7 says "
              "aerial search cannot find them, so the burial geometry is not doing its job")

    if a.write:
        # REFUSED, deliberately, after a red-team audit on 2026-09-11 measured what this model actually does:
        #
        #   * it never loads the settlement or damage.json's roof debris, so SEVEN rooftop survivors
        #     (Human_010, 021, 022, 028, 029, 030, 031) measure 0.00 coverage. Writing would silently
        #     DOWNGRADE them from occlusion 1 to 0 and destroy correct ground truth;
        #   * it has no height gate, so it credits four survivors with coverage from geometry that is not
        #     above them (Human_041, 043, 054, 069);
        #   * it treats a tree crown as an opaque disc of the MAJOR radius. For jacaranda that inflates the
        #     area about 31 % before opacity is considered, and real crowns thin to nothing at the margin.
        #     The eight canopy-driven "level 2" survivors are the least trustworthy numbers it produces.
        #
        # Only 4 of its 25 reported contradictions survive `check_occlusion_truth.py`, which applies a real
        # height test and names the occluder. And occlusion is MEASURABLE rather than modellable:
        # `tools/capture/measure_occlusion.py` computes it per observation from the actor's visible pixel
        # count and the frame's own GSD, which is an observation rather than an assumption.
        print(
            "REFUSING --write.\n\n"
            "This model has no height gate and does not load the buildings, so writing it would "
            "downgrade 7 rooftop survivors from occlusion 1 to 0 and damage the ground truth "
            "the whole evaluation rests on. Its disc model also overstates canopy coverage.\n\n"
            "Use it as a PRE-FLIGHT INDICATOR only. The authoritative paths are:\n"
            "  tools/scene/check_occlusion_truth.py        height-gated, names the occluder\n"
            "  tools/capture/measure_occlusion.py --write  measures occlusion per observation\n"
            "                                             from the rendered mask")
        return 2
        return 2

    if False:
        for x, intent, lvl, frac, parts in rows:
            x["occlusion_intent"] = intent
            x["occlusion"] = lvl
            x["occlusion_coverage"] = round(frac, 3)
            x["occluded_by"] = parts
        doc["occlusion_reconciled"] = {
            "by": "tools/scene/reconcile_occlusion.py",
            "thresholds": {"level_1_at": L1, "level_2_at": L2},
            "measured_from": [f for f in ("vegetation.json", "rubble_layout.json", "props_layout.json",
                                          "utilities.json") if (S / f).exists()],
            "changed": changed,
        }
        (S / "actors.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"written: data/scene/actors.json ({changed} level(s) changed)")
    else:
        print("(report only - pass --write to update data/scene/actors.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
