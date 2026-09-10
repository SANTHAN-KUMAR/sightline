"""Texture, import and place the generated FloodValley settlement (running editor, PIE OFF). Idempotent.

    uv run python tools/scene/gen_buildings.py          # host: meshes + layout
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_buildings.py').read())"

Materials: one PBR master (`M_PBR_Master`: BaseColor/Normal/ARM texture parameters on UV0, Tint, UVScale) and an
instance per Poly Haven CC0 set. The OBJ UVs are already in texture tiles, so nothing stretches. Four plaster tints
give the painted-house variety of Kerala settlements; the choice is seeded in settlement.json (fixed scene content,
not training-time randomisation, §5.5c).
Placement: UE (X, Y, Z) cm = (north*100, east*100, (asl - base_z)*100) (flood_valley.json `ue_import`). The OBJ
import mirrors local y like the terrain; archetypes are symmetric enough that only the door side flips, and the
yaw maps as UE yaw = 90 - theta (theta measured from east towards north).
Houses are STATIC (they never move) and collide with complex geometry so spawners can trace onto roofs.
"""

import json

import unreal

REPO = r"D:\Sightline"
PH = REPO + r"\_downloads\assets\polyhaven"
eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
with open(REPO + r"\data\scene\flood_valley.json") as f:
    meta = json.load(f)
with open(REPO + r"\data\scene\settlement.json") as f:
    town = json.load(f)
BASE = meta["base_z_m"]
MATDIR = "/Game/Sightline/Buildings/Materials"


def import_file(path, dest, name=None, save=True):
    t = unreal.AssetImportTask()
    props = [("filename", path), ("destination_path", dest), ("automated", True), ("replace_existing", True),
             ("save", save)]
    if name:
        props.append(("destination_name", name))
    for k, v in props:
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.get_editor_property("imported_object_paths")]


def tex_set(pid):
    dest = f"/Game/Sightline/Textures/{pid}"
    out = {}
    for key, suffix, kind in (("col", "diff", "color"), ("nrm", "nor_dx", "normal"), ("arm", "arm", "arm")):
        path = f"{dest}/T_{pid}_{key.upper() if key != 'col' else 'D'}"
        if not eal.does_asset_exist(path):
            import_file(fr"{PH}\{pid}\{pid}_{suffix}_2k.jpg", dest, path.rsplit("/", 1)[1], save=False)
        tex = unreal.load_asset(path)
        if kind == "normal":
            tex.set_editor_property("srgb", False)
            tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        elif kind == "arm":
            tex.set_editor_property("srgb", False)
            tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
        eal.save_asset(path)
        out[key] = tex
    return out


def reuse_or_create(path, cls, factory):
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    d, n = path.rsplit("/", 1)
    return tools.create_asset(n, d, cls, factory)


# --- master material ------------------------------------------------------------------------------------
master = reuse_or_create(f"{MATDIR}/M_PBR_Master", unreal.Material, unreal.MaterialFactoryNew())
for e in list(mel.get_material_expressions(master)):
    mel.delete_material_expression(master, e)
if mel.get_num_material_expressions(master):
    raise RuntimeError("master material graph did not clear")
y = [0]


def node(cls, **props):
    y[0] += 120
    e = mel.create_material_expression(master, cls, -700, y[0])
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect {a.get_name()}.{ao} -> {b.get_name()}.{bi}")


# A TextureSampleParameter2D's DEFAULT texture must match its sampler type or the whole material fails to
# compile and every instance silently renders as the grey WorldGridMaterial checker (seen 2026-09-10: the
# engine's DefaultDiffuse is sRGB, which SAMPLERTYPE_MASKS rejects). Default each parameter to a real texture
# of the right kind instead of an engine placeholder.
placeholder = unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_D")
flatn = unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_NRM")
arm_placeholder = unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_ARM")
assert placeholder and flatn and arm_placeholder, "import the building textures before building materials"
uv0 = node(unreal.MaterialExpressionTextureCoordinate)
uvs = node(unreal.MaterialExpressionScalarParameter, parameter_name="UVScale", default_value=1.0)
uvm = node(unreal.MaterialExpressionMultiply)
link(uv0, "", uvm, "A")
link(uvs, "", uvm, "B")
S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS
col = node(unreal.MaterialExpressionTextureSampleParameter2D, parameter_name="BaseColor", texture=placeholder,
           sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, sampler_source=S)
nrm = node(unreal.MaterialExpressionTextureSampleParameter2D, parameter_name="Normal", texture=flatn,
           sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, sampler_source=S)
arm = node(unreal.MaterialExpressionTextureSampleParameter2D, parameter_name="ARM", texture=arm_placeholder,
           sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_MASKS, sampler_source=S)
for t in (col, nrm, arm):
    link(uvm, "", t, "UVs")
tint = node(unreal.MaterialExpressionVectorParameter, parameter_name="Tint", default_value=unreal.LinearColor(1, 1, 1, 1))
tcol = node(unreal.MaterialExpressionMultiply)
link(col, "", tcol, "A")
link(tint, "", tcol, "B")
mel.connect_material_property(tcol, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(nrm, "", unreal.MaterialProperty.MP_NORMAL)
mel.connect_material_property(arm, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
mel.connect_material_property(arm, "G", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(arm, "B", unreal.MaterialProperty.MP_METALLIC)
mel.recompile_material(master)
eal.save_asset(master.get_path_name())


def assert_compiles(mat):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    through the Python API except zeroed statistics. Fail loudly here instead of discovering it in a capture."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB."
        )
    return s


_s = assert_compiles(master)
print(f"M_PBR_Master compiles: {_s.num_pixel_shader_instructions} instructions, "
      f"{_s.num_pixel_texture_samples} texture samples")


def mi(name, pid, tint_rgb=(1, 1, 1)):
    path = f"{MATDIR}/{name}"
    inst = reuse_or_create(path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, master)
    t = tex_set(pid)
    mel.set_material_instance_texture_parameter_value(inst, "BaseColor", t["col"])
    mel.set_material_instance_texture_parameter_value(inst, "Normal", t["nrm"])
    mel.set_material_instance_texture_parameter_value(inst, "ARM", t["arm"])
    mel.set_material_instance_vector_parameter_value(inst, "Tint", unreal.LinearColor(*tint_rgb, 1.0))
    mel.update_material_instance(inst)  # rebuild cached uniform expressions, else the render lags the asset
    eal.save_asset(path)
    return inst


# Kerala painted plaster: off-white, pale yellow, pale blue-green, pale pink (seeded per house)
WALLS = [mi("MI_Wall_Cream", "painted_plaster_wall", (1.0, 0.97, 0.88)),
         mi("MI_Wall_Yellow", "painted_plaster_wall", (1.0, 0.9, 0.62)),
         mi("MI_Wall_Mint", "painted_plaster_wall", (0.78, 0.95, 0.88)),
         mi("MI_Wall_Weathered", "worn_mossy_plasterwall")]
SLOT = {"wall": WALLS[0], "roof_tile": mi("MI_RoofTile", "clay_roof_tiles"),
        "roof_sheet": mi("MI_RoofSheet_Rusty", "rusty_corrugated_iron"),
        "concrete": mi("MI_Concrete", "dirty_concrete"), "wood": mi("MI_Wood", "weathered_planks")}
win = reuse_or_create(f"{MATDIR}/M_Window", unreal.Material, unreal.MaterialFactoryNew())
for e in list(mel.get_material_expressions(win)):
    mel.delete_material_expression(win, e)
wc = mel.create_material_expression(win, unreal.MaterialExpressionConstant3Vector, -300, 0)
wc.set_editor_property("constant", unreal.LinearColor(0.015, 0.02, 0.025, 1))
wr = mel.create_material_expression(win, unreal.MaterialExpressionConstant, -300, 150)
wr.set_editor_property("r", 0.12)
mel.connect_material_property(wc, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(wr, "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.recompile_material(win)
eal.save_asset(win.get_path_name())
SLOT["window"] = win

# --- archetype meshes ---------------------------------------------------------------------------------------
meshes, slot_index = {}, {}
for arch in town["archetypes"]:
    objs = import_file(fr"{REPO}\data\scene\buildings\{arch}.obj", "/Game/Sightline/Buildings", f"SM_{arch}")
    sm = [o for o in objs if isinstance(o, unreal.StaticMesh)][0]
    ns = sm.get_editor_property("nanite_settings")
    ns.set_editor_property("enabled", False)
    sm.set_editor_property("nanite_settings", ns)
    bs = sm.get_editor_property("body_setup")
    bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    sm.set_editor_property("body_setup", bs)
    names = []
    for i, sl in enumerate(sm.get_editor_property("static_materials")):
        nm = str(sl.get_editor_property("material_slot_name"))
        key = next((k for k in SLOT if nm.lower().startswith(k)), None)
        if key is None:
            raise RuntimeError(f"{arch}: unknown material slot {nm!r}")
        sm.set_material(i, SLOT[key])
        names.append(key)
    slot_index[arch] = names
    eal.save_asset(sm.get_path_name())
    meshes[arch] = sm
print("archetypes", {a: (m.get_num_triangles(0), slot_index[a]) for a, m in meshes.items()})

# --- place ----------------------------------------------------------------------------------------------------
for a in eas.get_all_level_actors():
    if a.get_actor_label().startswith("House_"):
        eas.destroy_actor(a)
for hs in town["houses"]:
    loc = unreal.Vector(hs["north_m"] * 100.0, hs["east_m"] * 100.0, (hs["base_asl_m"] - BASE) * 100.0)
    act = eas.spawn_actor_from_object(meshes[hs["archetype"]], loc, unreal.Rotator(0, 0, 90.0 - hs["yaw_deg"]))
    act.set_actor_label(f"House_{hs['id']:03d}")
    act.set_folder_path("Settlement")
    comp = act.static_mesh_component
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    wall_slot = slot_index[hs["archetype"]].index("wall")
    comp.set_material(wall_slot, WALLS[hs["wall_tint"]])
    act.tags = ["Building", hs["zone"], hs["archetype"]]
for _n, _m in list(SLOT.items()) + [(f"wall{i}", w) for i, w in enumerate(WALLS)]:
    assert_compiles(_m)
print(f"all {len(SLOT) + len(WALLS)} building materials compile")
print("houses placed", len(town["houses"]), "| level saved", les.save_current_level())
