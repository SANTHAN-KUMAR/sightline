"""Apply the seeded flood damage from data/scene/damage.json to the FloodValley settlement (editor, PIE OFF).

    uv run python tools/scene/gen_damage.py            # host side: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_damage.py').read())"

Run AFTER `build_buildings.py` (it needs the House_### actors and the MI_* material instances) and, for the
tide line, after `build_roof_materials.py`.

Fully idempotent, and safe to run repeatedly:
  1. every House_### actor is first RESET to its pristine archetype mesh from settlement.json, so a house that
     was damaged by an earlier run and is not in the current plan really does become pristine again;
  2. the damaged variants are then swapped in with `set_static_mesh` - the actor, its label, its object name and
     its transform are untouched. NOTHING is renamed: renaming onto a name a destroyed-but-not-yet-collected
     actor still holds is a FATAL engine error (Obj.cpp:383) and crashed the editor twice on 2026-09-10;
  3. roof debris is cleared and replaced by outliner folder, the same pattern as build_props.py, but placed as
     `HierarchicalInstancedStaticMeshComponent` instances - the 67 pieces use 15 distinct meshes, so they cost
     15 actors instead of 67. See build_rubble.py's docstring for why every actor matters on this machine.

One new material instance is created here, `MI_Rubble` (Poly Haven brown_mud_rocks_01, already imported by
build_materials.py as T_brown_mud_rocks_01_D/_N/_ARM). It parents to M_PBR_Master, so it inherits the anti-tiling
and the tide line for free. Every material this script touches is asserted to compile: a material that fails
renders as the grey WorldGridMaterial checker and reports nothing at all through the Python API.
"""

import json
import math

import unreal

REPO = r"D:\Sightline"
MATDIR = "/Game/Sightline/Buildings/Materials"
DMG_PKG = "/Game/Sightline/Buildings/Damaged"
FOLDER = "Damage"
MAX_ACTORS = 400                # hard tripwire: the roof debris is instanced, never one actor per piece

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)

if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports, spawns and saves are unreliable while PIE runs")

with open(REPO + r"\data\scene\damage.json") as fh:
    plan = json.load(fh)
with open(REPO + r"\data\scene\settlement.json") as fh:
    town = json.load(fh)
BASE_Z = plan["base_z_m"]


def assert_compiles(mat):
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB.")
    return s


# --- MI_Rubble -------------------------------------------------------------------------------------------
master = unreal.load_asset(f"{MATDIR}/M_PBR_Master")
if master is None:
    raise RuntimeError("M_PBR_Master is missing - run tools/scene/build_buildings.py first")
rubble_path = f"{MATDIR}/MI_Rubble"
if eal.does_asset_exist(rubble_path):
    rubble = unreal.load_asset(rubble_path)
else:
    rubble = tools.create_asset("MI_Rubble", MATDIR, unreal.MaterialInstanceConstant,
                                unreal.MaterialInstanceConstantFactoryNew())
mel.set_material_instance_parent(rubble, master)
RUB = "/Game/Sightline/Textures/brown_mud_rocks_01/T_brown_mud_rocks_01_"
for pname, suffix in (("BaseColor", "D"), ("Normal", "N"), ("ARM", "ARM")):
    tex = unreal.load_asset(RUB + suffix)
    if tex is None:
        raise RuntimeError(f"{RUB + suffix} is missing - run tools/scene/build_materials.py first")
    mel.set_material_instance_texture_parameter_value(rubble, pname, tex)
mel.set_material_instance_vector_parameter_value(rubble, "Tint", unreal.LinearColor(0.95, 0.92, 0.88, 1.0))
mel.update_material_instance(rubble)
eal.save_asset(rubble_path)
assert_compiles(rubble)

WALLS = [unreal.load_asset(f"{MATDIR}/MI_Wall_{n}") for n in ("Cream", "Yellow", "Mint", "Weathered")]
SLOT = {
    "wall": WALLS[0],
    "roof_tile": unreal.load_asset(f"{MATDIR}/MI_RoofTile"),
    "roof_sheet": unreal.load_asset(f"{MATDIR}/MI_RoofSheet_Rusty"),
    "concrete": unreal.load_asset(f"{MATDIR}/MI_Concrete"),
    "wood": unreal.load_asset(f"{MATDIR}/MI_Wood"),
    "window": unreal.load_asset(f"{MATDIR}/M_Window"),
    "rubble": rubble,
}
missing = [k for k, v in SLOT.items() if v is None] + [i for i, w in enumerate(WALLS) if w is None]
if missing:
    raise RuntimeError(f"missing building materials {missing} - run tools/scene/build_buildings.py first")
for m in list(SLOT.values()) + WALLS:
    assert_compiles(m)


def import_mesh(path, name, expect_tris):
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", DMG_PKG), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", False)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    sm = unreal.load_asset(f"{DMG_PKG}/{name}")
    if sm is None:
        raise RuntimeError(f"import failed: {path}")
    ns = sm.get_editor_property("nanite_settings")          # imported meshes arrive with Nanite ON; the triangle
    ns.set_editor_property("enabled", False)                # count then reports the fallback mesh instead
    sm.set_editor_property("nanite_settings", ns)
    bs = sm.get_editor_property("body_setup")
    bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    sm.set_editor_property("body_setup", bs)
    slots = []
    for i, sl in enumerate(sm.get_editor_property("static_materials")):
        nm = str(sl.get_editor_property("material_slot_name"))
        key = next((k for k in SLOT if nm.lower().startswith(k)), None)
        if key is None:
            raise RuntimeError(f"{name}: unknown material slot {nm!r}")
        sm.set_material(i, SLOT[key])
        slots.append(key)
    eal.save_asset(sm.get_path_name())
    # The point of this check is the Nanite trap: an imported mesh arrives with Nanite ON and get_num_triangles
    # then reports the FALLBACK mesh (347 triangles for the 524k-triangle terrain). A 2 % tolerance still catches
    # that by three orders of magnitude while allowing the importer to weld away a degenerate or two.
    got = sm.get_num_triangles(0)
    if abs(got - expect_tris) > max(4, 0.02 * expect_tris):
        raise RuntimeError(f"{name} has {got} triangles, generator says {expect_tris} (Nanite fallback?)")
    return sm, slots


# --- import the damaged variants -------------------------------------------------------------------------
# The pristine archetypes are symmetric in u and v, so nothing in this project has ever proved whether UE's OBJ
# importer keeps the file's origin or re-pivots to the bounding-box centre. These meshes are NOT symmetric
# (rubble spills outside the footprint by up to 3 m), so a re-pivot would slide the house sideways. Measure it
# per mesh and hand back a LOCAL correction, which apply() rotates into world by the actor's own yaw.
meshes, slot_index, fix_local = {}, {}, {}
for vname, v in sorted(plan["variants"].items()):
    sm, slots = import_mesh(fr"{REPO}\data\scene\{v['obj']}".replace("/", "\\"), f"SM_{vname}", v["triangles"])
    meshes[vname], slot_index[vname] = sm, slots
    cx, cy, cz = v["bbox_centre_obj_cm"]
    want = unreal.Vector(cx, -cy, cz)               # UE's importer flips y: UE local X = obj x, Y = -obj y
    try:
        got = sm.get_bounds().origin
        d = unreal.Vector(want.x - got.x, want.y - got.y, want.z - got.z)
    except Exception as exc:                        # noqa: BLE001 - if the API is unavailable, assume no shift
        print(f"  ! {vname}: get_bounds() unavailable ({exc}); assuming the importer keeps the OBJ origin")
        d = unreal.Vector(0, 0, 0)
    fix_local[vname] = d
    if max(abs(d.x), abs(d.y), abs(d.z)) > 1.0:
        print(f"  {vname}: importer re-pivoted, correcting by local ({d.x:.1f}, {d.y:.1f}, {d.z:.1f}) cm")
_sample = {k: (meshes[k].get_num_triangles(0), slot_index[k]) for k in sorted(meshes)[:2]}
print("imported", len(meshes), "damaged variants; sample:", _sample)

# --- pristine archetype meshes, for the reset ------------------------------------------------------------
pristine, pristine_slots = {}, {}
for aname in town["archetypes"]:
    sm = unreal.load_asset(f"/Game/Sightline/Buildings/SM_{aname}")
    if sm is None:
        raise RuntimeError(f"SM_{aname} is missing - run tools/scene/build_buildings.py first")
    pristine[aname] = sm
    keys = []
    for sl in sm.get_editor_property("static_materials"):
        nm = str(sl.get_editor_property("material_slot_name"))
        key = next((k for k in SLOT if nm.lower().startswith(k)), None)
        if key is None:
            raise RuntimeError(f"SM_{aname}: unknown material slot {nm!r}")
        keys.append(key)
    pristine_slots[aname] = keys


def apply(act, hs, sm, slots, fix=None):
    """Swap the mesh, re-assign EVERY slot (a stale override at an index would survive the swap), and re-assert
    the actor's location from settlement.json plus any pivot correction, so the script is self-repairing."""
    comp = act.static_mesh_component
    comp.set_static_mesh(sm)
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    for i, key in enumerate(slots):
        comp.set_material(i, WALLS[hs["wall_tint"]] if key == "wall" else SLOT[key])
    loc = unreal.Vector(hs["north_m"] * 100.0, hs["east_m"] * 100.0, (hs["base_asl_m"] - BASE_Z) * 100.0)
    if fix is not None and max(abs(fix.x), abs(fix.y), abs(fix.z)) > 1.0:
        yaw = math.radians(90.0 - hs["yaw_deg"])          # the actor rotation build_buildings.py used
        loc = unreal.Vector(loc.x + fix.x * math.cos(yaw) - fix.y * math.sin(yaw),
                            loc.y + fix.x * math.sin(yaw) + fix.y * math.cos(yaw), loc.z + fix.z)
    act.set_actor_location(loc, False, False)


houses = {a.get_actor_label(): a for a in eas.get_all_level_actors()
          if a.get_actor_label().startswith("House_")}
if not houses:
    raise RuntimeError("no House_### actors in the level - run tools/scene/build_buildings.py first")

by_id = {b["id"]: b for b in town["houses"]}

# 1. reset every house to pristine (so a plan that no longer damages a house really un-damages it)
for hs in town["houses"]:
    act = houses.get(f"House_{hs['id']:03d}")
    if act is None:
        continue
    apply(act, hs, pristine[hs["archetype"]], pristine_slots[hs["archetype"]])
    act.tags = [str(t) for t in act.tags if str(t) != "Damaged" and not str(t).startswith("dmg:")]

# 2. swap in the damaged variants
damaged, missing_actors = 0, []
for h in plan["houses"]:
    act = houses.get(f"House_{h['id']:03d}")
    if act is None:
        missing_actors.append(h["id"])
        continue
    hs = by_id[h["id"]]
    apply(act, hs, meshes[h["variant"]], slot_index[h["variant"]], fix_local[h["variant"]])
    act.tags = [str(t) for t in act.tags] + ["Damaged", f"dmg:{h['variant']}"]
    damaged += 1
if missing_actors:
    raise RuntimeError(f"{len(missing_actors)} planned houses have no actor, first: {missing_actors[:5]}")

# --- 3. debris rafted onto the flat roofs ----------------------------------------------------------------
removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

# INSTANCES, NOT ACTORS. The 67 pieces use only 15 distinct meshes, so they cost 15 HISM actors instead of 67
# AActors. See build_rubble.py's docstring: the editor died at ~5,500 actors on the Windows COMMIT limit
# (AvailableVirtual 0.00 GiB, UsedVirtual 27.48 GiB) while 0.46 GiB of physical RAM was still free, and every
# actor this lane does not create is commit that the vegetation lane's instances can have instead.
# unreal.Transform's `rotation` field is a QUAT, not a Rotator (UE 5.8). Assign through set_editor_property,
# convert the rotator explicitly, and prove the builder before trusting it 67 times.
def _rot_to_quat(rot):
    try:
        return rot.quaternion()
    except Exception:                                               # noqa: BLE001
        return unreal.MathLibrary.conv_rotator_to_quaternion(rot)


def _quat_to_rot(q):
    try:
        return q.rotator()
    except Exception:                                               # noqa: BLE001
        return unreal.MathLibrary.conv_quaternion_to_rotator(q)


def debris_xform(rec):
    t = unreal.Transform()
    t.set_editor_property("translation", unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                                                       (rec["base_asl_m"] - BASE_Z) * 100.0))
    t.set_editor_property("rotation", _rot_to_quat(
        unreal.Rotator(rec["roll_deg"], rec["pitch_deg"], rec["yaw_deg"])))
    s = float(rec["scale"])
    t.set_editor_property("scale3d", unreal.Vector(s, s, s))
    return t


_p = {"north_m": 12.34, "east_m": -56.78, "base_asl_m": BASE_Z + 9.0,
      "roll_deg": 3.0, "pitch_deg": -11.0, "yaw_deg": 50.0, "scale": 1.25}
_t = debris_xform(_p)
_tr = _t.get_editor_property("translation")
_rr = _quat_to_rot(_t.get_editor_property("rotation"))
_sc = _t.get_editor_property("scale3d")
if (abs(_tr.x - 1234.0) > 0.5 or abs(_tr.y + 5678.0) > 0.5 or abs(_tr.z - 900.0) > 0.5
        or abs(_sc.x - 1.25) > 1e-3 or abs(_rr.roll - 3.0) > 0.05 or abs(_rr.pitch + 11.0) > 0.05
        or abs(_rr.yaw - 50.0) > 0.05):
    raise RuntimeError(f"transform builder is wrong: t=({_tr.x:.1f},{_tr.y:.1f},{_tr.z:.1f}) "
                       f"rot=(roll {_rr.roll:.2f}, pitch {_rr.pitch:.2f}, yaw {_rr.yaw:.2f}) scale {_sc.x:.3f}")

cache, absent, byasset = {}, set(), {}
for rec in plan["roof_debris"]:
    if rec["asset"] not in cache:
        cache[rec["asset"]] = unreal.load_asset(rec["asset"])
    if cache[rec["asset"]] is None:
        absent.add(rec["asset"])
        continue
    byasset.setdefault(rec["asset"], []).append(rec)
if absent:
    raise RuntimeError(f"{len(absent)} roof-debris meshes are not in the project, first: {sorted(absent)[:3]}")

if len(byasset) > MAX_ACTORS:
    raise RuntimeError(f"{len(byasset)} debris actors is over the {MAX_ACTORS} tripwire. NOTHING WAS PLACED.")

placed, comps = 0, 0
for asset, recs in sorted(byasset.items()):
    label = "RoofDebris_" + asset.rsplit("/", 1)[-1]
    act = eas.spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
    if act is None:
        raise RuntimeError(f"could not spawn the container actor for {label}")
    # UE 5.8's Python bindings do NOT expose Actor.add_component_by_class (checked live: it is absent from
    # dir(unreal.Actor) and the call raises AttributeError). SubobjectDataSubsystem is the path the Details
    # panel's "+ Add Component" takes and it produces an INSTANCE component that serialises with the actor.
    # Same route as build_rubble.py and build_vegetation.py.
    handles = sds.k2_gather_subobject_data_for_instance(act)
    new_handle, why = sds.add_new_subobject(unreal.AddNewSubobjectParams(
        parent_handle=handles[0], new_class=unreal.HierarchicalInstancedStaticMeshComponent,
        blueprint_context=None, conform_transform_to_parent=True))
    if str(why):
        raise RuntimeError(f"{label}: add_new_subobject refused: {why}")
    sds.rename_subobject(new_handle, "Debris")
    comp = act.get_component_by_class(unreal.HierarchicalInstancedStaticMeshComponent)
    if comp is None:
        raise RuntimeError(f"{label}: no HISM on the actor after add_new_subobject - the debris would "
                           f"silently not exist")
    try:
        comp.set_static_mesh(cache[asset])
    except Exception:                                                   # noqa: BLE001
        comp.set_editor_property("static_mesh", cache[asset])
    if comp.get_editor_property("static_mesh") != cache[asset]:
        raise RuntimeError(f"{label}: the HISM did not take the mesh {cache[asset].get_name()}")
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    comp.set_collision_profile_name("NoCollision")
    comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    act.set_actor_label(label)                                          # label only - never rename()
    act.set_folder_path(FOLDER)
    act.tags = ["Debris", "roof"] + sorted({f"house:{r['house_id']}" for r in recs})
    al, ar, asx = act.get_actor_location(), act.get_actor_rotation(), act.get_actor_scale3d()
    if (abs(al.x) + abs(al.y) + abs(al.z) > 1.0 or abs(ar.pitch) + abs(ar.yaw) + abs(ar.roll) > 0.01
            or abs(asx.x - 1) + abs(asx.y - 1) + abs(asx.z - 1) > 1e-3):
        raise RuntimeError(f"{label}: container actor is not at the identity")

    xf = [debris_xform(r) for r in recs]
    comp.add_instances(xf, False, True)         # world_space=True: the transforms are world coordinates
    if comp.get_instance_count() != len(recs):
        raise RuntimeError(f"{label}: asked for {len(recs)} instances, component reports "
                           f"{comp.get_instance_count()}")
    # prove the instances are where the plan says, in world space - a count alone passes on a pile at (0,0,0)
    want = xf[0].get_editor_property("translation")
    got = comp.get_instance_transform(0, True).get_editor_property("translation")
    if max(abs(got.x - want.x), abs(got.y - want.y), abs(got.z - want.z)) > 1.0:
        raise RuntimeError(f"{label}: instance 0 reads back at ({got.x:.1f}, {got.y:.1f}, {got.z:.1f}) cm, the "
                           f"plan says ({want.x:.1f}, {want.y:.1f}, {want.z:.1f}) cm")
    placed += len(recs)
    comps += 1

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != comps:
    raise RuntimeError(f"spawned {comps} debris instancer actors but the {FOLDER} folder holds {n_now}")
if placed != len(plan["roof_debris"]):
    raise RuntimeError(f"placed {placed} debris instances, the plan has {len(plan['roof_debris'])}")

c = plan["counts"]
saved = les.save_current_level()
print(f"damaged {damaged} / {c['houses_total']} houses ({c['by_archetype']})")
print(f"  by variant: {c['by_variant']}")
print(f"roof debris: removed {removed}, placed {placed} INSTANCES across {comps} HISM actors "
      f"(~{c['roof_debris_triangles']:,} tris)")
print(f"net added triangles: {c['net_added_triangles']:,} | level saved {saved}")
print(f"  total level actors now: {len(eas.get_all_level_actors())}")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images - the API reports success for a scene that "
      "renders wrong.")
