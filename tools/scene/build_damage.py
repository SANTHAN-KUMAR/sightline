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
  3. roof debris is cleared and replaced by outliner folder, the same pattern as build_props.py.

One new material instance is created here, `MI_Rubble` (Poly Haven brown_mud_rocks_01, already imported by
build_materials.py as T_brown_mud_rocks_01_D/_N/_ARM). It parents to M_PBR_Master, so it inherits the anti-tiling
and the tide line for free. Every material this script touches is asserted to compile: a material that fails
renders as the grey WorldGridMaterial checker and reports nothing at all through the Python API.
"""

import json

import unreal

REPO = r"D:\Sightline"
MATDIR = "/Game/Sightline/Buildings/Materials"
DMG_PKG = "/Game/Sightline/Buildings/Damaged"
FOLDER = "Damage"

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

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
    got = sm.get_num_triangles(0)
    if got != expect_tris:
        raise RuntimeError(f"{name} has {got} triangles, generator says {expect_tris} (Nanite fallback?)")
    return sm, slots


# --- import the damaged variants -------------------------------------------------------------------------
meshes, slot_index = {}, {}
for vname, v in sorted(plan["variants"].items()):
    sm, slots = import_mesh(fr"{REPO}\data\scene\{v['obj']}".replace("/", "\\"), f"SM_{vname}", v["triangles"])
    meshes[vname], slot_index[vname] = sm, slots
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


def apply(act, sm, slots, wall_tint):
    comp = act.static_mesh_component
    comp.set_static_mesh(sm)
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    for i, key in enumerate(slots):
        comp.set_material(i, WALLS[wall_tint] if key == "wall" else SLOT[key])


houses = {a.get_actor_label(): a for a in eas.get_all_level_actors()
          if a.get_actor_label().startswith("House_")}
if not houses:
    raise RuntimeError("no House_### actors in the level - run tools/scene/build_buildings.py first")

# 1. reset every house to pristine (so a plan that no longer damages a house really un-damages it)
for hs in town["houses"]:
    act = houses.get(f"House_{hs['id']:03d}")
    if act is None:
        continue
    apply(act, pristine[hs["archetype"]], pristine_slots[hs["archetype"]], hs["wall_tint"])
    act.tags = [str(t) for t in act.tags if str(t) != "Damaged" and not str(t).startswith("dmg:")]

# 2. swap in the damaged variants
damaged, missing_actors = 0, []
for h in plan["houses"]:
    act = houses.get(f"House_{h['id']:03d}")
    if act is None:
        missing_actors.append(h["id"])
        continue
    apply(act, meshes[h["variant"]], slot_index[h["variant"]], h["wall_tint"])
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

cache, absent, placed = {}, set(), 0
for rec in plan["roof_debris"]:
    mesh = cache.get(rec["asset"])
    if mesh is None:
        mesh = unreal.load_asset(rec["asset"])
        cache[rec["asset"]] = mesh
    if mesh is None:
        absent.add(rec["asset"])
        continue
    loc = unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0, (rec["base_asl_m"] - BASE_Z) * 100.0)
    act = eas.spawn_actor_from_object(mesh, loc, unreal.Rotator(rec["roll_deg"], rec["pitch_deg"], rec["yaw_deg"]))
    if act is None:
        continue
    act.set_actor_scale3d(unreal.Vector(rec["scale"], rec["scale"], rec["scale"]))
    act.set_actor_label(rec["name"])
    act.set_folder_path(FOLDER)
    act.tags = ["Debris", "roof", f"house:{rec['house_id']}"]
    act.static_mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
    placed += 1
if absent:
    raise RuntimeError(f"{len(absent)} roof-debris meshes are not in the project, first: {sorted(absent)[:3]}")

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != placed:
    raise RuntimeError(f"placed {placed} roof-debris actors but the {FOLDER} folder holds {n_now}")

c = plan["counts"]
saved = les.save_current_level()
print(f"damaged {damaged} / {c['houses_total']} houses ({c['by_archetype']})")
print(f"  by variant: {c['by_variant']}")
print(f"roof debris: removed {removed}, placed {placed} (~{c['roof_debris_triangles']:,} tris)")
print(f"net added triangles: {c['net_added_triangles']:,} | level saved {saved}")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images - the API reports success for a scene that "
      "renders wrong.")
