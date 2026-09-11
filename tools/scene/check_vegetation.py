"""Check the canopy that is ACTUALLY IN THE LEVEL against the layout and against the survivor ground truth.

    uv run python tools/scene/check_vegetation.py            # exits non-zero on any failure
    uv run python tools/scene/check_vegetation.py <dump>     # check a specific dump (used to test the checker)

`build_vegetation.py` reads every instance transform back out of the HierarchicalInstancedStaticMeshComponents
after the level is saved and writes them to `_artifacts/vegetation/placed_instances.json`. This script is the
half that can say NO. It never looks at a return value: it compares engine-side transforms with
`data/scene/vegetation.json` and `data/scene/actors.json`.

What it asserts, and why each one is a failure mode that has actually happened on this project:

  A  the dump is NEWER than the layout and the builder      a stale dump passing for fresh work
  B  the level is FloodValley                               a canopy placed into the wrong map
  C  <= 400 actors, one per mesh                            the 5,508-actor version that killed the editor
  D  per-species instance counts match the layout EXACTLY   a batch silently dropped mid-placement
  E  every instance matches one layout record 1:1           unit errors (m vs cm), north/east swapped, a
                                                            transform that came out identity, an offset root
  F  no tree crown over a survivor recorded `occlusion: 0`  a label that claims a clear aerial view of
                                                            somebody who is under a tree
  G  crown radii and scales are physically sane             a species scaled 100x, an instance at the origin

F derives the crown radius from the CATALOGUE (`crown_major_m`) times the instance's own engine-side scale, not
from the layout's `crown_r_m` field, so a generator that miscomputed its own clearance cannot hide behind it.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S = REPO / "data/scene"
DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "_artifacts/vegetation/placed_instances.json"

MAX_ACTORS = 400            # must match build_vegetation.MAX_ACTORS
POS_TOL_M = 0.02
YAW_TOL_DEG = 0.15
PITCH_TOL_DEG = 0.25
SCALE_TOL = 2e-3

fails: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


if not DUMP.exists():
    print(f"FAIL  no engine-side dump at {DUMP} - run build_vegetation.py in the editor first")
    sys.exit(1)

dump = json.loads(DUMP.read_text(encoding="utf-8"))
plan = json.loads((S / "vegetation.json").read_text(encoding="utf-8"))
truth = json.loads((S / "actors.json").read_text(encoding="utf-8"))
CAT = plan["catalogue"]

# --- A. freshness -----------------------------------------------------------------------------------------
dump_t = DUMP.stat().st_mtime
for other in (S / "vegetation.json", REPO / "tools/scene/build_vegetation.py"):
    if other.stat().st_mtime > dump_t:
        fail(f"{other.name} is NEWER than the dump - the level was not rebuilt after it changed "
             f"({other.stat().st_mtime - dump_t:.0f} s newer)")
if not fails:
    ok(f"dump written {dump['written_local']} is newer than the layout and the builder")

# --- B. the right level -----------------------------------------------------------------------------------
if dump.get("level") != "FloodValley":
    fail(f"the dump came from level {dump.get('level')!r}, not FloodValley")
else:
    ok("level is FloodValley")

# --- C. actor count ---------------------------------------------------------------------------------------
n_meshes = len({i["pid"] for i in plan["items"]}) + len({i["asset_hint"] for i in plan["ground_items"]})
n_act = dump["vegetation_actor_count"]
if n_act > MAX_ACTORS:
    fail(f"{n_act} vegetation actors is over the {MAX_ACTORS} tripwire - this must be instanced, not spawned")
elif n_act != n_meshes:
    fail(f"{n_act} vegetation actors but {n_meshes} distinct meshes in the layout - one actor per mesh expected")
else:
    ok(f"{n_act} vegetation actors for {n_meshes} meshes, {dump['level_actor_count']:,} actors in the level")

# --- flatten the engine side ------------------------------------------------------------------------------
# rows are [north_m, east_m, z_cm/100 (i.e. metres above base_z), yaw, pitch, roll, scale]
eng: dict[str, list[list[float]]] = {}
for a in dump["actors"]:
    key = a["label"]
    eng[key] = a["instances_north_east_z_yaw_pitch_roll_scale"]
    if a["count"] != len(eng[key]):
        fail(f"{key}: component reports {a['count']} instances but dumped {len(eng[key])} transforms")

BASE_Z = dump["base_z_m"]
if abs(BASE_Z - plan["base_z_m"]) > 1e-6:
    fail(f"base_z_m disagrees: dump {BASE_Z} vs layout {plan['base_z_m']}")

# --- D. per-species counts --------------------------------------------------------------------------------
want_tree = dict(plan["counts"]["by_species"])
got_tree = {lbl[len("Veg_Tree_"):]: len(rows) for lbl, rows in eng.items() if lbl.startswith("Veg_Tree_")}
if got_tree != want_tree:
    fail(f"per-species TREE counts differ\n        engine {got_tree}\n        layout {want_tree}")
else:
    ok(f"tree instances per species match the layout exactly: {got_tree}")

want_ground = Counter(i["asset_hint"].rsplit("/", 1)[-1] for i in plan["ground_items"])
got_ground = {lbl[len("Veg_Under_"):]: len(rows) for lbl, rows in eng.items() if lbl.startswith("Veg_Under_")}
if got_ground != dict(want_ground):
    fail(f"understorey counts differ\n        engine {got_ground}\n        layout {dict(want_ground)}")
else:
    ok(f"understorey instances per mesh match the layout exactly: {got_ground}")

total_eng = sum(len(r) for r in eng.values())
total_plan = len(plan["items"]) + len(plan["ground_items"])
if total_eng != total_plan:
    fail(f"{total_eng:,} instances in the level, {total_plan:,} in the layout")
else:
    ok(f"{total_eng:,} instances total ({len(plan['items']):,} trees + {len(plan['ground_items']):,} understorey)")

# --- also: the actor holding a species must hold THAT species' mesh ---------------------------------------
for a in dump["actors"]:
    pid = a["label"].split("_", 2)[-1]
    stem = pid if a["label"].startswith("Veg_Tree_") else "fern_02"
    if not a["mesh_path"] or stem not in a["mesh_path"]:
        fail(f"{a['label']} holds mesh {a['mesh_path']!r}, which is not a {stem} mesh")
if not any("holds mesh" in f for f in fails):
    ok("every actor holds a mesh of its own species")

# --- E. 1:1 match against the layout ----------------------------------------------------------------------
def index(records, keyfn):
    d = defaultdict(list)
    for r in records:
        d[keyfn(r)].append(r)
    return d


unmatched, mismatched = [], []
matched_records = set()
for label, rows in eng.items():
    if label.startswith("Veg_Tree_"):
        pid = label[len("Veg_Tree_"):]
        recs = [r for r in plan["items"] if r["pid"] == pid]
        sign = dump.get("pitch_sign", 1.0)
    else:
        stem = label[len("Veg_Under_"):]
        recs = [r for r in plan["ground_items"] if r["asset_hint"].rsplit("/", 1)[-1] == stem]
        sign = 1.0
    idx = index(recs, lambda r: (round(r["north_m"], 1), round(r["east_m"], 1)))
    for row in rows:
        n, e, z, yaw, pitch, roll, sc = row
        cands = idx.get((round(n, 1), round(e, 1)), [])
        hit = None
        for r in cands:
            if abs(r["north_m"] - n) <= POS_TOL_M and abs(r["east_m"] - e) <= POS_TOL_M:
                hit = r
                break
        if hit is None:
            unmatched.append((label, row))
            continue
        matched_records.add(id(hit))
        want_z = hit["base_asl_m"] - BASE_Z
        why = []
        if abs(want_z - z) > POS_TOL_M:
            why.append(f"z {z:.3f} vs {want_z:.3f} m")
        if abs(((hit["yaw_deg"] - yaw + 180) % 360) - 180) > YAW_TOL_DEG:
            why.append(f"yaw {yaw} vs {hit['yaw_deg']}")
        if abs(hit["pitch_deg"] * sign - pitch) > PITCH_TOL_DEG:
            why.append(f"pitch {pitch} vs {hit['pitch_deg'] * sign}")
        if abs(hit["scale"] - sc) > SCALE_TOL:
            why.append(f"scale {sc} vs {hit['scale']}")
        if why:
            mismatched.append((label, hit["name"], "; ".join(why)))

if unmatched:
    fail(f"{len(unmatched)} placed instances match NO layout record, first 3: {unmatched[:3]}")
if mismatched:
    fail(f"{len(mismatched)} instances differ from their layout record, first 3: {mismatched[:3]}")
missing = total_plan - len(matched_records)
if missing:
    fail(f"{missing} layout records have no instance in the level")
if not (unmatched or mismatched or missing):
    ok(f"all {total_eng:,} instances match a layout record 1:1 in position, yaw, pitch and scale "
       f"(tolerances {POS_TOL_M} m / {YAW_TOL_DEG} deg / {SCALE_TOL})")

# --- F. no crown over a survivor recorded occlusion 0 -----------------------------------------------------
open_survivors = [a for a in truth["actors"] if a.get("occlusion") == 0]
worst = []
violations = 0
for label, rows in eng.items():
    if not label.startswith("Veg_Tree_"):
        continue
    pid = label[len("Veg_Tree_"):]
    crown_r1 = 0.5 * float(CAT[pid]["crown_major_m"])          # radius at scale 1, from the catalogue
    for row in rows:
        n, e, z, yaw, pitch, roll, sc = row
        r = crown_r1 * sc
        for a in open_survivors:
            d = math.hypot(a["north_m"] - n, a["east_m"] - e)
            if d < r:
                violations += 1
                worst.append(f"{a['name']} ({a['pose']}/{a['zone']}) is {d:.2f} m from a {pid} instance "
                             f"whose crown radius is {r:.2f} m")
            elif d - r < 1.0:
                notes.append(f"{a['name']} is {d - r:.2f} m outside a {pid} crown (radius {r:.2f} m)")
if violations:
    fail(f"{violations} tree instance(s) stand over a survivor recorded occlusion 0 - the labels would lie:")
    for w in worst[:8]:
        print(f"        {w}")
else:
    ok(f"no tree crown covers any of the {len(open_survivors)} survivors recorded occlusion 0 "
       f"(closest approach checked against catalogue crown radius x instance scale)")

# --- G. physical sanity -----------------------------------------------------------------------------------
bad_scale = [(l, r[6]) for l, rows in eng.items() for r in rows if not (0.5 <= r[6] <= 3.0)]
at_origin = [(l, r) for l, rows in eng.items() for r in rows if abs(r[0]) < 0.01 and abs(r[1]) < 0.01]
roi = plan["roi"]
out_of_roi = [(l, r[0], r[1]) for l, rows in eng.items() for r in rows
              if not (roi["east_m"][0] - roi["margin_m"] <= r[1] <= roi["east_m"][1] + roi["margin_m"]
                      and roi["north_m"][0] - roi["margin_m"] <= r[0] <= roi["north_m"][1] + roi["margin_m"])]
if bad_scale:
    fail(f"{len(bad_scale)} instances have an implausible scale, first 3: {bad_scale[:3]}")
if at_origin:
    fail(f"{len(at_origin)} instances sit at the world origin - a transform that came out identity")
if out_of_roi:
    fail(f"{len(out_of_roi)} instances are outside the ROI + margin, first 3: {out_of_roi[:3]}")
if not (bad_scale or at_origin or out_of_roi):
    sc_all = [r[6] for rows in eng.values() for r in rows]
    zs = [r[2] + BASE_Z for rows in eng.values() for r in rows]
    ok(f"scales {min(sc_all):.2f}-{max(sc_all):.2f}, base heights {min(zs):.1f}-{max(zs):.1f} m ASL, "
       f"all inside the ROI + {roi['margin_m']} m margin")

# --- the layout's own crown_r_m, cross-checked against the catalogue ---------------------------------------
bad_crown = []
for r in plan["items"][:2000]:
    want = 0.5 * float(CAT[r["pid"]]["crown_major_m"]) * r["scale"]
    if abs(want - r["crown_r_m"]) > 0.15:
        bad_crown.append((r["name"], r["pid"], r["crown_r_m"], round(want, 2)))
if bad_crown:
    fail(f"{len(bad_crown)} layout records disagree with catalogue crown_major_m x scale, "
         f"first 3: {bad_crown[:3]}")
else:
    ok("layout crown radii agree with catalogue crown_major_m x scale")

# --- memory, reported not asserted -------------------------------------------------------------------------
mb, ma = dump.get("mem_before", {}), dump.get("mem_after", {})
if mb and ma:
    print(f"\nmemory (sim, editor process, GiB): AvailablePhysical {mb['AvailablePhysical']:.2f} -> "
          f"{ma['AvailablePhysical']:.2f} | UsedVirtual {mb['UsedVirtual']:.2f} -> {ma['UsedVirtual']:.2f} "
          f"| PeakUsedVirtual {ma['PeakUsedVirtual']:.2f} (commit limit ~26.7)")

if notes:
    print(f"\n{len(notes)} near miss(es) within 1 m of a crown edge (not failures):")
    for n in notes[:5]:
        print(f"  {n}")

print()
if fails:
    print(f"{len(fails)} FAILURE(S) - the canopy in the level does not match the layout or the ground truth.")
    sys.exit(1)
print("all checks passed - the placed canopy matches vegetation.json and respects the survivor ground truth")
sys.exit(0)
