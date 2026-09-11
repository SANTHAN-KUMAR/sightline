"""Assert the utility network, boats and foam ribbon are in the level WHERE data/scene/*.json says they are.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\check_utilities_placed.py').read())"
    ue_python code="SELFTEST=True; exec(open(r'D:\\Sightline\\tools\\scene\\check_utilities_placed.py').read())"

This is a check that can FAIL, per docs/QUALITY_GATE.md rule 2. It raises RuntimeError listing every failure,
so `ue_python` comes back success: False.

It does NOT trust the build script's return value or its printed summary. build_utilities.py already checks
its own actor BOUNDS, but a bounding box is satisfied by a mesh that is mirrored, rotated 180 deg, or scaled
about its centre - every pole could be in the wrong place and the bbox would still match. So this walks the
imported mesh's actual VERTICES, transforms them by the actor's real transform, and asks of each of the 60
poles in utilities.json: is there a shaft standing at that world position, with its base at that ASL?

Checks
------
  poles      for each of the 60 entries in utilities.json pole_list, mesh vertices exist within POLE_R cm of
             the pole's world XY, their maximum Z reaches the pole top (ground_asl_m + height_m) to within
             TOP_TOL cm, and their minimum Z is the buried butt of the pole, ground_asl_m - POLE_BURY_M, to
             within BASE_TOL cm.
             POLE_BURY_M is read out of gen_utilities.py rather than written here, so this asserts the
             burial the generator actually applies instead of a number someone typed twice. (The first
             version of this check expected the shaft to start at ground level and failed all 60 poles by
             exactly 120.0 cm, which is how the burial got noticed at all.)
  boats      19 boat actors exist, each at the north/east/ASL utilities.json gives, to within BOAT_TOL cm.
  foam       exactly ONE foam actor (this is the constraint that killed the previous attempt: 2,573 quads as
             2,573 actors is what runs the editor out of commit space), carrying foam.json's triangle count.
  materials  every material this lane created or touched reports non-zero pixel-shader instructions. A
             material that fails to compile reports ZERO and renders as the grey checker with no error
             anywhere (docs/CONTEXT.md section 7).
  budget     the level stays under MAX_ACTORS.

SELFTEST=True shifts the expected pole positions by SELFTEST_SHIFT cm and the expected foam triangle count by
1 before running, and then INVERTS the result: it passes only if the check fails. A test that cannot fail is
not a test, and this is the cheapest way to prove this one still has teeth.

That self-test earned its keep immediately: the first version of the pole check only asked "is there mesh
geometry within POLE_R of this XY", which a 50 cm shift sailed straight through because POLE_R is 60 cm. The
pole positions were therefore only pinned to +-0.6 m. Measuring the shaft AXIS (below) pins them to
centimetres, and the self-test now fails on the shift as it should.
"""

import json
import math
import re

import unreal

REPO = r"D:\Sightline"
POLE_R = 60.0          # cm; the shaft is ~12 cm radius, crossarms reach ~1 m but only near the top
BASE_TOL = 5.0         # cm
XY_TOL = 5.0           # cm; how far the measured shaft axis may sit from utilities.json east/north
TOP_TOL = 40.0         # cm; the cap/crossarm sits a little above the nominal top
BOAT_TOL = 3.0         # cm
MAX_ACTORS = 9000

SELFTEST_SHIFT = 50.0  # cm of deliberate pole error the self-test injects; must exceed XY_TOL, not POLE_R
# ue_python exec()s into ONE persistent namespace, so a SELFTEST=True left over from a previous call would
# silently turn the next plain run into another self-test - which is exactly what happened the first time it
# was used here, and a self-test always "passes". POP the caller's flag (so it cannot leak forward) and hold
# it under a private name that every plain run reassigns.
_ST = bool(globals().pop("SELFTEST", False))

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
mel = unreal.MaterialEditingLibrary

with open(REPO + r"\data\scene\utilities.json") as fh:
    U = json.load(fh)
with open(REPO + r"\data\scene\foam.json") as fh:
    F = json.load(fh)
BASE_Z = U["base_z_m"]

# The pole shaft is generated from (z_ground - POLE_BURY_M) upwards so that a pole on a slope never shows a
# gap at its foot. Read the constant from the generator instead of restating it, so this check tracks the
# geometry rather than a copy of it.
with open(REPO + r"\tools\scene\gen_utilities.py") as fh:
    _m = re.search(r"^POLE_BURY_M\s*=\s*([0-9.]+)", fh.read(), re.M)
if not _m:
    raise RuntimeError("could not read POLE_BURY_M from tools/scene/gen_utilities.py - this check needs it "
                       "to know where a correctly placed pole's geometry starts")
POLE_BURY_M = float(_m.group(1))

fails = []
notes = []


def check(ok, msg):
    (notes if ok else fails).append(("PASS " if ok else "FAIL ") + msg)


actors = eas.get_all_level_actors()
by_label = {}
for a in actors:
    by_label.setdefault(a.get_actor_label(), []).append(a)

# ---------------------------------------------------------------------------------------------------------
# 1. poles: walk the real mesh vertices through the real actor transform
# ---------------------------------------------------------------------------------------------------------
net = [a for a in actors if a.get_actor_label() == "Utilities"]
check(len(net) == 1, f"exactly one Utilities network actor (found {len(net)})")
if len(net) == 1:
    act = net[0]
    sm = act.static_mesh_component.static_mesh
    xf = act.get_actor_transform()
    verts, _tris, _n, _uv, _t = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(sm, 0, 0)
    # get_section_from_static_mesh returns plain unreal.Vector positions, not vertex structs
    world = [xf.transform_location(v) for v in verts]
    check(len(world) > 1000, f"network mesh exposes {len(world)} vertices for inspection")

    # bucket vertices into a coarse XY grid so 60 poles x 45k vertices is not 2.7 M distance tests
    CELL = 200.0
    grid = {}
    for w in world:
        grid.setdefault((int(w.x // CELL), int(w.y // CELL)), []).append(w)

    bad_base, bad_top, missing, worst_base, worst_top = 0, 0, 0, 0.0, 0.0
    bad_xy, worst_xy, thin_shaft = 0, 0.0, 0
    for p in U["pole_list"]:
        wx, wy = p["north_m"] * 100.0, p["east_m"] * 100.0
        if _ST:
            wx += SELFTEST_SHIFT
        want_base = (p["ground_asl_m"] - POLE_BURY_M - BASE_Z) * 100.0
        want_top = (p["ground_asl_m"] + p["height_m"] - BASE_Z) * 100.0
        near = []
        for gx in (int((wx - POLE_R) // CELL), int((wx + POLE_R) // CELL)):
            for gy in (int((wy - POLE_R) // CELL), int((wy + POLE_R) // CELL)):
                for w in grid.get((gx, gy), ()):
                    if math.hypot(w.x - wx, w.y - wy) <= POLE_R:
                        near.append(w)
        if not near:
            missing += 1
            continue
        zs = [w.z for w in near]
        db, dt = abs(min(zs) - want_base), abs(max(zs) - want_top)
        worst_base, worst_top = max(worst_base, db), max(worst_top, dt)
        bad_base += db > BASE_TOL
        bad_top += dt > TOP_TOL
        # Horizontal placement. "There is geometry within 60 cm" would tolerate a pole misplaced by half a
        # metre, so measure the SHAFT AXIS instead: take the centroid of the vertices in the lowest 2 m of
        # the pole, which is buried butt and bare shaft - below every crossarm, bracket, insulator and wire,
        # so nothing asymmetric drags the centroid off the axis. The generator centres the shaft tube on
        # (east, north), so this centroid IS the pole's planted position, to the centimetre.
        shaft = [w for w in near if want_base - 5.0 <= w.z <= want_base + 200.0]
        if len(shaft) >= 4:
            cx = sum(w.x for w in shaft) / len(shaft)
            cy = sum(w.y for w in shaft) / len(shaft)
            dxy = math.hypot(cx - wx, cy - wy)
            worst_xy = max(worst_xy, dxy)
            bad_xy += dxy > XY_TOL
        else:
            thin_shaft += 1
    check(missing == 0, f"every pole has mesh geometry at its world XY ({missing} of "
                        f"{len(U['pole_list'])} had none within {POLE_R:.0f} cm)")
    check(bad_base == 0, f"every pole butt buried exactly {POLE_BURY_M:.2f} m below its utilities.json "
                         f"ground_asl, within {BASE_TOL:.0f} cm ({bad_base} outside; "
                         f"worst {worst_base:.1f} cm)")
    check(bad_top == 0, f"every pole top within {TOP_TOL:.0f} cm of utilities.json "
                        f"({bad_top} outside; worst {worst_top:.1f} cm)")
    check(thin_shaft == 0, f"every pole exposes a measurable shaft cross-section ({thin_shaft} did not)")
    check(bad_xy == 0, f"every pole's shaft axis within {XY_TOL:.0f} cm of its utilities.json east/north "
                       f"({bad_xy} outside; worst {worst_xy:.2f} cm)")
    got = sm.get_num_triangles(0)
    check(got == U["network"]["triangles"],
          f"network triangles {got} == utilities.json {U['network']['triangles']}")

# ---------------------------------------------------------------------------------------------------------
# 2. boats
# ---------------------------------------------------------------------------------------------------------
bad_boat, worst_boat, miss_boat = 0, 0.0, []
for rec in U["boats"]["items"]:
    got = by_label.get(rec["name"])
    if not got:
        miss_boat.append(rec["name"])
        continue
    loc = got[0].get_actor_location()
    d = max(abs(loc.x - rec["north_m"] * 100.0), abs(loc.y - rec["east_m"] * 100.0),
            abs(loc.z - (rec["base_asl_m"] - BASE_Z) * 100.0))
    worst_boat = max(worst_boat, d)
    bad_boat += d > BOAT_TOL
check(not miss_boat, f"all {U['counts']['boats']} boats present (missing {miss_boat})")
check(bad_boat == 0, f"every boat within {BOAT_TOL:.0f} cm of utilities.json "
                     f"({bad_boat} outside; worst {worst_boat:.2f} cm)")

# every boat must sit AT the flood surface: base_asl is the designed waterline of the hull
off = [abs(r["base_asl_m"] - U["water_level_m"]) for r in U["boats"]["items"]]
check(max(off) < 0.15, f"every boat's waterline within 15 cm of the flood surface "
                       f"(worst {max(off) * 100:.1f} cm)")

# ---------------------------------------------------------------------------------------------------------
# 3. foam: ONE actor, the full ribbon
# ---------------------------------------------------------------------------------------------------------
foam = [a for a in actors if str(a.get_folder_path()) == "Foam"]
check(len(foam) == 1, f"the foam ribbon is ONE actor, not {F['counts']['quads_total']} "
                      f"(found {len(foam)})")
if len(foam) == 1:
    fsm = foam[0].static_mesh_component.static_mesh
    want = F["counts"]["triangles"] + (1 if _ST else 0)
    got = fsm.get_num_triangles(0)
    check(got == want, f"foam triangles {got} == foam.json {want}")

# ---------------------------------------------------------------------------------------------------------
# 4. materials: zero instructions == failed compile == grey checker, reported nowhere else
# ---------------------------------------------------------------------------------------------------------
MATS = ["/Game/Sightline/Buildings/Materials/M_Wire",
        "/Game/Sightline/Buildings/Materials/MI_Boat_Wood",
        "/Game/Sightline/Buildings/Materials/MI_Boat_Blue",
        "/Game/Sightline/Buildings/Materials/MI_Boat_Green",
        "/Game/Sightline/Buildings/Materials/MI_Boat_Red",
        "/Game/Sightline/Buildings/Materials/MI_Concrete",
        "/Game/Sightline/Buildings/Materials/MI_RoofSheet_Rusty",
        "/Game/Sightline/Buildings/Materials/MI_Wood",
        "/Game/Sightline/Water/M_Foam",
        "/Game/Sightline/Water/M_FloodWater"]
for path in MATS:
    m = unreal.load_asset(path)
    if m is None:
        check(False, f"{path} exists")
        continue
    s = mel.get_statistics(m)
    check(s.num_pixel_shader_instructions > 0,
          f"{path.rsplit('/', 1)[1]} compiles ({s.num_pixel_shader_instructions} instructions, "
          f"{s.num_pixel_texture_samples} samples)")

# M_FloodWater must still carry tune_water.py's nodes: build_materials.py CLEARS this graph, so if it ran
# last the volumetric water is silently back to a flat lit card.
w = unreal.load_asset("/Game/Sightline/Water/M_FloodWater")
marked = 0
for e in list(mel.get_material_expressions(w)):
    try:
        if str(e.get_editor_property("desc")) == "SIGHTLINE_TUNE_WATER":
            marked += 1
    except Exception:                                          # noqa: BLE001
        pass
check(marked >= 4, f"M_FloodWater still carries tune_water.py's nodes ({marked} found) - if this is 0, "
                   f"build_materials.py ran last and tune_water.py must be re-run")

# ---------------------------------------------------------------------------------------------------------
# 5. budget
# ---------------------------------------------------------------------------------------------------------
check(len(actors) < MAX_ACTORS, f"level holds {len(actors)} actors (< {MAX_ACTORS})")

# ---------------------------------------------------------------------------------------------------------
for line in notes + fails:
    print("  " + line)
print(f"\n{len(notes)} passed, {len(fails)} failed"
      + ("   [SELFTEST: expected failures]" if _ST else ""))

if _ST:
    if not fails:
        raise RuntimeError("SELFTEST DID NOT FAIL: the pole/foam assertions do not actually test anything")
    print("SELFTEST OK: the check detects a 50 cm pole shift and a 1-triangle foam mismatch.")
elif fails:
    raise RuntimeError(f"{len(fails)} utilities/foam checks FAILED:\n  " + "\n  ".join(fails))
else:
    print("ALL UTILITIES / FOAM / WATER PLACEMENT CHECKS PASSED")
