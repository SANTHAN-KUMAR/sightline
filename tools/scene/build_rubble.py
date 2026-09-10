"""Import, texture and place the FloodValley structural-collapse rubble field (running editor, PIE OFF). Idempotent.

    D:\\Tools\\uv\\uv.exe run python tools\\scene\\gen_rubble.py        # host side first: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_rubble.py').read())"

WHAT IT DOES
  1. resolves the six textures it needs from the sets ALREADY in the project (nothing is downloaded) and asserts
     their sRGB/compression settings, because a normal map left as sRGB makes the whole material fail to compile
     and every rubble actor renders as the grey WorldGridMaterial checker with no error anywhere;
  2. builds `M_Rubble_Master` and seven instances in /Game/Sightline/Rubble/Materials, and ASSERTS that each one
     compiles (`MaterialEditingLibrary.get_statistics`: a failed compile reports zero instructions and nothing
     else). The master carries a Dust parameter - Reference B's palette is "dust-covered, desaturated tan/grey,
     very low colour contrast", which is a material effect, not a texture choice;
  3. imports the 40 generated OBJs, turns **Nanite off** (imports arrive with it ON, and `get_num_triangles`
     then reports the fallback mesh - 347 triangles for a 524k-triangle asset), and checks the imported triangle
     count against the count `gen_rubble.py` recorded;
  4. places `data/scene/rubble_layout.json` and saves the level.

OWNERSHIP. This script owns `/Game/Sightline/Rubble/**` and NOTHING else. It deliberately does not touch
`M_PBR_Master` (owned by build_buildings.py / build_roof_materials.py) or the terrain and water materials
(build_materials.py): two scripts clearing and rebuilding the same graph silently undo each other.

NAMING. Like `build_props.py`, this NEVER renames an actor. Renaming onto a name a destroyed-but-not-yet-
collected actor still holds is a FATAL engine error (Obj.cpp:383) that crashed the editor twice on 2026-09-10.
Rubble is never a detection target, so its object names do not need to be stable; the field is cleared by its
outliner folder instead.

COLLISION is OFF by default (`RUBBLE_COLLISION = False`). Two reasons, both concrete: complex collision for
~2.4M triangles is real memory on a 16 GB / 8 GB-VRAM machine, and the sightline server fails a flight on any
collision report whose actor is not named "Ground", so a drone descending over the fan would fail on rubble
contact. Set it True if something needs to line-trace onto the rubble surface (e.g. re-seating survivors on it).
"""

import json

import unreal

REPO = r"D:\Sightline"
FOLDER = "Rubble"
MATDIR = "/Game/Sightline/Rubble/Materials"
MESHDIR = "/Game/Sightline/Rubble"
TEXROOT = "/Game/Sightline/Textures"
RUBBLE_COLLISION = False        # see the docstring; True gives query-only complex collision

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports come back as 32 px placeholders and saves return False in PIE")

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    meta = json.load(fh)
with open(REPO + r"\data\scene\rubble_layout.json") as fh:
    plan = json.load(fh)
BASE_Z = meta["base_z_m"]


# --- textures -------------------------------------------------------------------------------------------------
# The two suffix conventions in this project are real: build_materials.py wrote the terrain normals as `_N`,
# build_buildings.py wrote the building normals as `_NRM`. Resolve both instead of guessing.
def tex(pid, kind):
    names = {"col": [f"T_{pid}_D"], "nrm": [f"T_{pid}_NRM", f"T_{pid}_N"], "arm": [f"T_{pid}_ARM"]}[kind]
    for nm in names:
        p = f"{TEXROOT}/{pid}/{nm}"
        if eal.does_asset_exist(p):
            t = unreal.load_asset(p)
            if kind == "nrm":
                ok = (not t.get_editor_property("srgb")
                      and t.get_editor_property("compression_settings")
                      == unreal.TextureCompressionSettings.TC_NORMALMAP)
            elif kind == "arm":
                ok = (not t.get_editor_property("srgb")
                      and t.get_editor_property("compression_settings")
                      == unreal.TextureCompressionSettings.TC_MASKS)
            else:
                ok = bool(t.get_editor_property("srgb"))
            if not ok:
                raise RuntimeError(
                    f"{p} has the wrong sampler settings for a {kind} map (srgb="
                    f"{t.get_editor_property('srgb')}, compression="
                    f"{t.get_editor_property('compression_settings')}). Bound to a mismatched sampler type the "
                    f"WHOLE material fails to compile and renders as the grey checker. Fix the texture first.")
            return t
    raise RuntimeError(f"missing texture: none of {[f'{TEXROOT}/{pid}/{n}' for n in names]} exist")


SETS = {"concrete": "dirty_concrete", "masonry": "brown_mud_rocks_01", "rebar": "rusty_corrugated_iron",
        "fabric": "painted_plaster_wall", "wood": "weathered_planks", "metal": "rusty_corrugated_iron"}
TEX = {k: {kind: tex(pid, kind) for kind in ("col", "nrm", "arm")} for k, pid in SETS.items()}
print("textures resolved:", {k: TEX[k]["nrm"].get_name() for k in TEX})


def reuse_or_create(path, cls, factory):
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    d, n = path.rsplit("/", 1)
    a = tools.create_asset(n, d, cls, factory)
    if a is None:
        raise RuntimeError(f"create_asset returned None for {path} (is an actor still referencing it?)")
    return a


def assert_compiles(mat):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    through the Python API except zeroed statistics. Copied from build_buildings.py on purpose."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB.")
    return s


# --- master material ------------------------------------------------------------------------------------------
master = reuse_or_create(f"{MATDIR}/M_Rubble_Master", unreal.Material, unreal.MaterialFactoryNew())
for e in list(mel.get_material_expressions(master)):
    mel.delete_material_expression(master, e)
mel.delete_unused_expressions(master)
if mel.get_num_material_expressions(master):
    # delete_all_material_expressions does NOT clear everything (a terrain graph grew 64 -> 95 -> 111 nodes over
    # three reruns), so the count is asserted rather than trusted.
    raise RuntimeError(f"master graph did not clear: {mel.get_num_material_expressions(master)} nodes left")

_y = [0]


def node(cls, x=-800, **props):
    _y[0] += 130
    e = mel.create_material_expression(master, cls, x, _y[0])
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect {a.get_name()}.{ao or 'out'} -> {b.get_name()}.{bi}")


S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS
uv0 = node(unreal.MaterialExpressionTextureCoordinate, -1500)
uvs = node(unreal.MaterialExpressionScalarParameter, -1500, parameter_name="UVScale", default_value=1.0)
uvm = node(unreal.MaterialExpressionMultiply, -1300)
link(uv0, "", uvm, "A")
link(uvs, "", uvm, "B")

# Every parameter's DEFAULT texture matches its sampler type. The engine's own DefaultDiffuse is sRGB and under
# SAMPLERTYPE_MASKS it silently killed every building material on 2026-09-10.
col = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="BaseColor",
           texture=TEX["concrete"]["col"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_COLOR,
           sampler_source=S)
nrm = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="Normal",
           texture=TEX["concrete"]["nrm"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL,
           sampler_source=S)
arm = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="ARM",
           texture=TEX["concrete"]["arm"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_MASKS,
           sampler_source=S)
for t in (col, nrm, arm):
    link(uvm, "", t, "UVs")

tint = node(unreal.MaterialExpressionVectorParameter, -900, parameter_name="Tint",
            default_value=unreal.LinearColor(1, 1, 1, 1))
tcol = node(unreal.MaterialExpressionMultiply, -700)
link(col, "", tcol, "A")
link(tint, "", tcol, "B")

# Dust. SCENE_REFERENCE Reference B: "dust-covered, desaturated tan/grey, very low colour contrast". Concrete
# dust also flattens the surface normal and raises roughness, so all three are driven from one scalar.
dust_a = node(unreal.MaterialExpressionScalarParameter, -900, parameter_name="DustAmount", default_value=0.4)
dust_c = node(unreal.MaterialExpressionVectorParameter, -900, parameter_name="DustColor",
              default_value=unreal.LinearColor(0.74, 0.70, 0.62, 1.0))
base_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(tcol, "", base_lerp, "A")
link(dust_c, "", base_lerp, "B")
link(dust_a, "", base_lerp, "Alpha")

flat_n = node(unreal.MaterialExpressionConstant3Vector, -900)
flat_n.set_editor_property("constant", unreal.LinearColor(0.0, 0.0, 1.0, 1.0))
n_damp = node(unreal.MaterialExpressionMultiply, -700)
n_damp_k = node(unreal.MaterialExpressionConstant, -900)
n_damp_k.set_editor_property("r", 0.45)
link(dust_a, "", n_damp, "A")
link(n_damp_k, "", n_damp, "B")
n_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(nrm, "", n_lerp, "A")
link(flat_n, "", n_lerp, "B")
link(n_damp, "", n_lerp, "Alpha")

rough_dust = node(unreal.MaterialExpressionConstant, -900)
rough_dust.set_editor_property("r", 0.93)
r_damp = node(unreal.MaterialExpressionMultiply, -700)
r_damp_k = node(unreal.MaterialExpressionConstant, -900)
r_damp_k.set_editor_property("r", 0.7)
link(dust_a, "", r_damp, "A")
link(r_damp_k, "", r_damp, "B")
r_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(arm, "G", r_lerp, "A")
link(rough_dust, "", r_lerp, "B")
link(r_damp, "", r_lerp, "Alpha")

mel.connect_material_property(base_lerp, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(n_lerp, "", unreal.MaterialProperty.MP_NORMAL)
mel.connect_material_property(arm, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
mel.connect_material_property(r_lerp, "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(arm, "B", unreal.MaterialProperty.MP_METALLIC)
mel.recompile_material(master)
eal.save_asset(master.get_path_name())
_s = assert_compiles(master)
print(f"M_Rubble_Master compiles: {_s.num_pixel_shader_instructions} instructions, "
      f"{_s.num_pixel_texture_samples} texture samples")


def mi(name, texkey, tint_rgb, dust, uv_scale=1.0, dust_rgb=(0.74, 0.70, 0.62)):
    path = f"{MATDIR}/{name}"
    inst = reuse_or_create(path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, master)
    mel.set_material_instance_texture_parameter_value(inst, "BaseColor", TEX[texkey]["col"])
    mel.set_material_instance_texture_parameter_value(inst, "Normal", TEX[texkey]["nrm"])
    mel.set_material_instance_texture_parameter_value(inst, "ARM", TEX[texkey]["arm"])
    mel.set_material_instance_vector_parameter_value(inst, "Tint", unreal.LinearColor(*tint_rgb, 1.0))
    mel.set_material_instance_vector_parameter_value(inst, "DustColor", unreal.LinearColor(*dust_rgb, 1.0))
    mel.set_material_instance_scalar_parameter_value(inst, "DustAmount", dust)
    mel.set_material_instance_scalar_parameter_value(inst, "UVScale", uv_scale)
    mel.update_material_instance(inst)     # rebuild cached uniform expressions, else the render lags the asset
    eal.save_asset(path)
    return inst


SLOT = {
    "concrete": mi("MI_Rubble_Concrete", "concrete", (0.98, 0.96, 0.92), 0.50),
    "masonry": mi("MI_Rubble_Masonry", "masonry", (0.96, 0.90, 0.83), 0.42),
    "rebar": mi("MI_Rubble_Rebar", "rebar", (0.80, 0.52, 0.34), 0.18),
    "wood": mi("MI_Rubble_Wood", "wood", (0.92, 0.86, 0.78), 0.38),
    "metal": mi("MI_Rubble_Metal", "metal", (0.88, 0.88, 0.90), 0.32),
    "fabric": mi("MI_Rubble_Fabric_A", "fabric", (0.24, 0.40, 0.66), 0.26),
}
FABRIC_B = mi("MI_Rubble_Fabric_B", "fabric", (0.66, 0.26, 0.22), 0.26)
for _n, _m in list(SLOT.items()) + [("fabric_b", FABRIC_B)]:
    assert_compiles(_m)
print(f"all {len(SLOT) + 1} rubble materials compile")


# --- meshes -----------------------------------------------------------------------------------------------------
def import_file(path, dest, name):
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", dest), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", True)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.get_editor_property("imported_object_paths")]


meshes, tri_report, bad_tris, unknown = {}, {}, [], []
for name, info in plan["variants"].items():
    objs = import_file(fr"{REPO}\data\scene\rubble\{name}.obj", MESHDIR, f"SM_{name}")
    sm = next((o for o in objs if isinstance(o, unreal.StaticMesh)), None)
    if sm is None:
        raise RuntimeError(f"{name}.obj did not import as a StaticMesh (got {objs})")
    ns = sm.get_editor_property("nanite_settings")     # imports arrive with Nanite ON; the project runs GI=None
    ns.set_editor_property("enabled", False)
    sm.set_editor_property("nanite_settings", ns)
    if RUBBLE_COLLISION:
        bs = sm.get_editor_property("body_setup")
        bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        sm.set_editor_property("body_setup", bs)
    for i, sl in enumerate(sm.get_editor_property("static_materials")):
        nm = str(sl.get_editor_property("material_slot_name")).lower()
        key = next((k for k in SLOT if nm.startswith(k)), None)
        if key is None:
            unknown.append((name, nm))
            continue
        sm.set_material(i, SLOT[key])
    eal.save_asset(sm.get_path_name())
    got = sm.get_num_triangles(0)
    tri_report[name] = (got, info["tris"])
    # A Nanite-enabled mesh reports its FALLBACK here (347 triangles for a 524k asset), so this doubles as proof
    # that Nanite really is off. Anything else means the import lost geometry.
    if abs(got - info["tris"]) > max(8, 0.02 * info["tris"]):
        bad_tris.append((name, got, info["tris"]))
    meshes[name] = sm

if unknown:
    raise RuntimeError(f"unknown material slots (expected {sorted(SLOT)}): {unknown[:5]}")
if bad_tris:
    raise RuntimeError(
        "imported triangle counts do not match the generator (name, imported, expected): "
        f"{bad_tris[:5]} -- either Nanite is still on (the fallback mesh is reported) or the import lost "
        "geometry. Do NOT place this field.")
print(f"imported {len(meshes)} rubble meshes, Nanite off, triangle counts match the generator "
      f"(total {sum(v[0] for v in tri_report.values()):,})")


# --- place --------------------------------------------------------------------------------------------------
removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

placed = 0
for rec in plan["items"]:
    sm = meshes[rec["variant"]]
    loc = unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0, (rec["base_asl_m"] - BASE_Z) * 100.0)
    # yaw_deg is measured from east towards north (the gen_buildings convention), and the OBJ importer flips
    # handedness, so the actor yaw is 90 - yaw_deg exactly as in build_buildings.py.
    rot = unreal.Rotator(rec["roll_deg"], rec["pitch_deg"], 90.0 - rec["yaw_deg"])
    act = eas.spawn_actor_from_object(sm, loc, rot)
    if act is None:
        raise RuntimeError(f"spawn failed for {rec['name']} ({rec['variant']})")
    act.set_actor_scale3d(unreal.Vector(rec["scale"], rec["scale"], rec["scale"]))
    act.set_actor_label(rec["name"])          # label only - never rename(); see the docstring
    act.set_folder_path(FOLDER)
    act.tags = ["Rubble", rec["group"], rec["family"]]
    comp = act.static_mesh_component
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    if RUBBLE_COLLISION:
        comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
    else:
        comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    if rec["family"] == "belonging" and rec["id"] % 2:
        for i, sl in enumerate(sm.get_editor_property("static_materials")):
            if str(sl.get_editor_property("material_slot_name")).lower().startswith("fabric"):
                comp.set_material(i, FABRIC_B)
    placed += 1

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != placed:
    raise RuntimeError(f"placed {placed} but the {FOLDER} folder holds {n_now}")

saved = les.save_current_level()
c = plan["counts"]
print(f"removed {removed}, placed {placed} rubble actors (~{c['approx_triangles']:,} triangles, budget "
      f"{c['triangle_budget']:,}) | collision {'QUERY_ONLY' if RUBBLE_COLLISION else 'OFF'} | saved {saved}")
print(f"  by group : {c['by_group']}")
print(f"  by family: {c['by_family']}")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images. Check specifically that the rubble is NOT "
      "grey checker (material compile), that the slabs are metre-scale and not 100x small (OBJ units), and "
      "that the survivors on the fan sit in voids rather than inside a pile.")
