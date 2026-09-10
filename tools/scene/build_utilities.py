"""Import and place the FloodValley utility network and boats (running editor, PIE OFF). Idempotent.

    uv run python tools/scene/gen_utilities.py                      # host side: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_utilities.py').read())"

Run AFTER `build_buildings.py` (this script reuses its material instances and the layout is clearance-checked
against its houses). It creates exactly one new material, `M_Wire`, and four boat-paint INSTANCES of the
existing `M_PBR_Master`; it never clears or rebuilds a graph another script owns, which is the rule that keeps
`build_materials.py`, `build_roof_materials.py`, `build_foam.py` and this script from silently undoing each
other.

Material slots come straight from the OBJ `usemtl` groups:
    concrete  -> MI_Concrete          pole shafts and ceramic insulators
    roof_sheet-> MI_RoofSheet_Rusty   steel crossarms, light brackets, the punt's hull
    wood      -> MI_Wood              boat planking, gunwales, thwarts (overridden per boat, see below)
    wire      -> M_Wire               conductors and service drops

Boat hulls are painted per actor by overriding the `wood` slot with one of four instances of M_PBR_Master
that copy MI_Wood's textures and vary only `Tint`. Copying the textures off the existing instance (rather than
hard-coding a texture path) means this cannot drift from whatever `build_buildings.py` decided weathered
planks are: `/Game/Sightline/Textures/...` naming differs between the terrain sets (`_N`) and the building
sets (`_NRM`), and guessing it wrong is a silent grey material.

The network is ONE actor at the world origin with yaw +90, exactly like `Ground` and `FloodFoam`, because its
OBJ is written in world coordinates (a catenary spans two poles). Its bounds are asserted against the
generator's own bbox: the mesh is far from centred (east 9-455 m, north -281..261 m), so a wrong yaw or a
re-pivoting importer shows up as a hundreds-of-metres error instead of as a wrong-looking capture.

Collision: the network and the boats are NoCollision. `sim_fly` fails a takeoff on contact with any actor not
named "Ground", and a 3.6 cm wire strung across the settlement at 8 m is exactly the sort of thing a descent
would clip.
"""

import json

import unreal

REPO = r"D:\Sightline"
FOLDER = "Utilities"
MATDIR = "/Game/Sightline/Buildings/Materials"
UDIR = "/Game/Sightline/Utilities"

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports, saves and material recompiles are unreliable while PIE runs")

with open(REPO + r"\data\scene\utilities.json") as fh:
    plan = json.load(fh)
BASE_Z = plan["base_z_m"]


def assert_compiles(mat, where=""):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    through Python except zeroed statistics (docs/CONTEXT.md section 7)."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0:
        raise RuntimeError(f"{where}{mat.get_name()} FAILED TO COMPILE (0 instructions, "
                           f"{s.num_pixel_texture_samples} texture samples)")
    return s


def reuse_or_create(path, cls, factory):
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    d, n = path.rsplit("/", 1)
    a = tools.create_asset(n, d, cls, factory)
    if a is None:
        raise RuntimeError(f"could not create {path}")
    return a


# --- M_Wire: the only material this script owns -------------------------------------------------------------
# Insulated LV aerial bundled cable is a black polymer sheath, not bare aluminium: metallic 0, roughness 0.55.
# It is 2 px wide at survey altitude, so this is a constant-colour material on purpose - a texture would
# alias, and there is nothing to see at that size but a dark line.
wire = reuse_or_create(f"{MATDIR}/M_Wire", unreal.Material, unreal.MaterialFactoryNew())
for e in list(mel.get_material_expressions(wire)):
    mel.delete_material_expression(wire, e)
if mel.get_num_material_expressions(wire):
    raise RuntimeError("M_Wire graph did not clear")
_c = mel.create_material_expression(wire, unreal.MaterialExpressionConstant3Vector, -300, 0)
_c.set_editor_property("constant", unreal.LinearColor(0.016, 0.016, 0.018, 1.0))
_r = mel.create_material_expression(wire, unreal.MaterialExpressionConstant, -300, 140)
_r.set_editor_property("r", 0.55)
_sp = mel.create_material_expression(wire, unreal.MaterialExpressionConstant, -300, 260)
_sp.set_editor_property("r", 0.35)
mel.connect_material_property(_c, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(_r, "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(_sp, "", unreal.MaterialProperty.MP_SPECULAR)
mel.recompile_material(wire)
eal.save_asset(wire.get_path_name())
assert_compiles(wire)

# --- slot map: everything else is reused ---------------------------------------------------------------------
SLOT = {
    "concrete": unreal.load_asset(f"{MATDIR}/MI_Concrete"),
    "roof_sheet": unreal.load_asset(f"{MATDIR}/MI_RoofSheet_Rusty"),
    "wood": unreal.load_asset(f"{MATDIR}/MI_Wood"),
    "wall": unreal.load_asset(f"{MATDIR}/MI_Wall_Cream"),
    "window": unreal.load_asset(f"{MATDIR}/M_Window"),
    "wire": wire,
}
missing = sorted(k for k, v in SLOT.items() if v is None)
if missing:
    raise RuntimeError(f"missing building materials {missing} - run tools/scene/build_buildings.py first")

# --- boat paint: instances of M_PBR_Master that copy MI_Wood's textures and vary only the tint ---------------
master = unreal.load_asset(f"{MATDIR}/M_PBR_Master")
if master is None:
    raise RuntimeError("M_PBR_Master is missing - run tools/scene/build_buildings.py first")
wood_tex = {}
for pname in ("BaseColor", "Normal", "ARM"):
    t = mel.get_material_instance_texture_parameter_value(SLOT["wood"], pname)
    if t is None:
        raise RuntimeError(f"MI_Wood has no {pname} texture override - re-run build_buildings.py")
    wood_tex[pname] = t

PAINT = {                       # small craft in Kerala are painted; a bare-plank boat is the exception
    "MI_Boat_Wood": (1.00, 0.96, 0.90),
    "MI_Boat_Blue": (0.30, 0.52, 0.95),
    "MI_Boat_Green": (0.32, 0.78, 0.48),
    "MI_Boat_Red": (0.88, 0.30, 0.26),
}
paints = []
for name, tint in PAINT.items():
    path = f"{MATDIR}/{name}"
    inst = reuse_or_create(path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, master)
    for pname, tex in wood_tex.items():
        mel.set_material_instance_texture_parameter_value(inst, pname, tex)
    mel.set_material_instance_vector_parameter_value(inst, "Tint", unreal.LinearColor(*tint, 1.0))
    mel.update_material_instance(inst)
    eal.save_asset(path)
    assert_compiles(inst, "boat paint: ")
    paints.append(inst)


def import_obj(rel_obj, name):
    t = unreal.AssetImportTask()
    for k, v in (("filename", REPO + "\\data\\scene\\" + rel_obj.replace("/", "\\")),
                 ("destination_path", UDIR), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", False)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    sm = unreal.load_asset(f"{UDIR}/{name}")
    if sm is None:
        raise RuntimeError(f"{rel_obj} did not import - run tools/scene/gen_utilities.py first")
    ns = sm.get_editor_property("nanite_settings")      # imported meshes arrive with Nanite ON and
    ns.set_editor_property("enabled", False)            # get_num_triangles then reports the fallback mesh
    sm.set_editor_property("nanite_settings", ns)
    slots = []
    for i, sl in enumerate(sm.get_editor_property("static_materials")):
        nm = str(sl.get_editor_property("material_slot_name"))
        key = next((k for k in SLOT if nm.lower().startswith(k)), None)
        if key is None:
            raise RuntimeError(f"{name}: unknown material slot {nm!r}")
        sm.set_material(i, SLOT[key])
        slots.append(key)
    eal.save_asset(sm.get_path_name())
    return sm, slots


# --- the network ---------------------------------------------------------------------------------------------
net = plan["network"]
sm_net, net_slots = import_obj(net["obj"], "SM_Utilities")
got, want = sm_net.get_num_triangles(0), net["triangles"]
if abs(got - want) > max(4, 0.02 * want):
    raise RuntimeError(f"SM_Utilities has {got} triangles, the generator wrote {want} (Nanite fallback?)")

removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

act = eas.spawn_actor_from_object(sm_net, unreal.Vector(0, 0, 0),
                                  unreal.Rotator(0.0, 0.0, net["ue_actor"]["yaw_deg"]))
act.set_actor_label("Utilities")
act.set_folder_path(FOLDER)
act.tags = ["Utilities", "Network"]
act.static_mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
act.static_mesh_component.set_collision_profile_name("NoCollision")

# The OBJ is far from centred, so this catches a wrong yaw or a re-pivoting importer as a huge error.
b = net["ue_actor"]["bounds_cm"]
want_c = ((b["x_north"][0] + b["x_north"][1]) / 2, (b["y_east"][0] + b["y_east"][1]) / 2)
half = ((b["x_north"][1] - b["x_north"][0]) / 2, (b["y_east"][1] - b["y_east"][0]) / 2)
origin, extent = act.get_actor_bounds(False)
if abs(extent.x - half[0]) > 200.0 or abs(extent.y - half[1]) > 200.0:
    raise RuntimeError(
        f"Utilities has the wrong footprint: extent ({extent.x:.0f}, {extent.y:.0f}) cm, expected "
        f"({half[0]:.0f}, {half[1]:.0f}). The actor yaw ({net['ue_actor']['yaw_deg']}) or the OBJ axes are "
        f"wrong - do not capture anything until this matches.")
shift = unreal.Vector(want_c[0] - origin.x, want_c[1] - origin.y, 0.0)
if abs(shift.x) > 1.0 or abs(shift.y) > 1.0:
    print(f"  re-pivoted on import: shifting Utilities by ({shift.x:.1f}, {shift.y:.1f}) cm")
    loc = act.get_actor_location()
    act.set_actor_location(unreal.Vector(loc.x + shift.x, loc.y + shift.y, loc.z), False, False)
    origin, extent = act.get_actor_bounds(False)
errs = [abs(origin.x - want_c[0]), abs(origin.y - want_c[1]),
        abs(extent.x - half[0]), abs(extent.y - half[1])]
if max(errs) > 200.0:
    raise RuntimeError(f"Utilities is still misplaced: origin ({origin.x:.0f}, {origin.y:.0f}) extent "
                       f"({extent.x:.0f}, {extent.y:.0f}), expected origin "
                       f"({want_c[0]:.0f}, {want_c[1]:.0f}) extent ({half[0]:.0f}, {half[1]:.0f})")

# --- the boats -------------------------------------------------------------------------------------------------
boat_mesh, boat_slots = {}, {}
for name, v in sorted(plan["boats"]["variants"].items()):
    sm, slots = import_obj(v["obj"], f"SM_Boat_{name}")
    g, w = sm.get_num_triangles(0), v["triangles"]
    if abs(g - w) > max(4, 0.02 * w):
        raise RuntimeError(f"SM_Boat_{name} has {g} triangles, the generator wrote {w} (Nanite fallback?)")
    boat_mesh[name], boat_slots[name] = sm, slots

placed_boats = 0
for rec in plan["boats"]["items"]:
    sm = boat_mesh[rec["variant"]]
    loc = unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                        (rec["base_asl_m"] - BASE_Z) * 100.0)
    # gen_buildings/build_buildings convention: yaw_deg is measured from east towards north, UE yaw = 90 - it
    a2 = eas.spawn_actor_from_object(sm, loc, unreal.Rotator(rec["roll_deg"], rec["pitch_deg"],
                                                             90.0 - rec["yaw_deg"]))
    a2.set_actor_label(rec["name"])
    a2.set_folder_path(FOLDER)
    a2.tags = ["Utilities", "Boat", rec["variant"]]
    comp = a2.static_mesh_component
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)     # so a future flood_level.py can float them
    comp.set_collision_profile_name("NoCollision")
    if "wood" in boat_slots[rec["variant"]]:
        comp.set_material(boat_slots[rec["variant"]].index("wood"), paints[rec["id"] % len(paints)])
    placed_boats += 1

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != placed_boats + 1:
    raise RuntimeError(f"placed {placed_boats + 1} but the {FOLDER} folder holds {n_now}")
for _m in [wire] + paints + [SLOT[k] for k in ("concrete", "roof_sheet", "wood")]:
    assert_compiles(_m)

saved = les.save_current_level()
c, sv = plan["counts"], plan["survey"]
print(f"utilities: removed {removed}; placed 1 network actor + {placed_boats} boats | level saved {saved}")
print(f"  {c['poles']} poles ({net['poles_in_water']} standing in the flood) in {net['chains']} lines, "
      f"{c['spans']} spans x {net['conductors_per_span']} conductors, {c['service_drops']} service drops")
print(f"  network {c['network_triangles']:,} tris (slots {net_slots}); boats {c['boat_triangles']:,} tris; "
      f"total {c['total_triangles']:,}")
print(f"  wire {net['wire_diameter_m'] * 100:.1f} cm = {sv['wire_px']} px at the survey GSD "
      f"({sv['gsd_cm_per_px']} cm/px); bounds check max error {max(errs):.0f} cm")
print("NOW LOOK: run tools/scene/qa_shots.py. Check that the wires read as continuous lines and not as dotted "
      "ones (if dotted, raise WIRE_D_M in gen_utilities.py and regenerate) and that the boats sit ON the "
      "water, not in it or above it (if wrong, the flood level moved: re-run gen_utilities.py).")
