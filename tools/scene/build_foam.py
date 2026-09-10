"""Build the FloodValley waterline foam: import the ambientCG Foam textures, build M_Foam, place the ribbon.

    uv run python tools/scene/gen_foam.py               # host side: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_foam.py').read())"

Editor must be up with PIE STOPPED. Idempotent: the material and the mesh are reused in place and the actor is
cleared by outliner folder, so nothing is ever renamed (renaming onto a name a destroyed-but-not-yet-collected
actor still holds is a FATAL engine error, Obj.cpp:383, and crashed the editor twice on 2026-09-10).

This script does NOT touch `M_FloodWater` or `M_FloodValleyTerrain`: `build_materials.py` owns those, and two
scripts clearing and rebuilding one material graph silently undo each other. The foam is an additive actor.

MEASURED CONSTANTS
------------------
`opacity_mask_clip_value = 0.12`. The alpha test keeps a texel when `Foam001_Opacity * FoamGain * edge falloff *
patchiness` exceeds the clip value, so the clip value alone sets how solid the foam reads. Measured coverage of
the source map (2048 px, and of its 4th mip, which is roughly what 45 m nadir at 1.77 cm/px samples):

    clip   0.08   0.10   0.12   0.14   0.16   0.20   0.24   0.30   0.36   0.48   0.60
    cover  0.730  0.679  0.630  0.581  0.533  0.431  0.351  0.248  0.164  0.052  0.011
    (mip 0; the 4th mip tracks it to within 0.02, so the alpha test does not thin out with distance)

0.12 gives 63 % coverage at the centre of the ribbon, ~35 % where the edge falloff has dropped to 0.5, and
nothing at all below falloff 0.25 - a band that fades out instead of ending at a line. FoamGain (default 1.0) is
the per-instance knob that shifts the whole curve without touching the material.

`FoamTint` default (2.08, 1.67, 1.38). Foam001's colour map is a scan of foam on dark water: over the texels the
alpha test keeps, its mean is sRGB (0.479, 0.510, 0.517) = linear (0.202, 0.233, 0.240), i.e. mid-grey with a
BLUE cast. River foam on a silt-laden monsoon flood is a dirty warm cream, so the tint has to invert that
ordering, not merely brighten it: these multipliers land on linear (0.42, 0.39, 0.33).

`FoamRoughness` default 0.75. The set ships a roughness map whose mean is 0.17 - that is a wet *water* surface,
not foam; a raft of bubbles is a diffuse, rough mass. The map is deliberately not used.

Macro patchiness uses `brown_mud_rocks_01_ARM.G` at mip 6, remapped `saturate((G - 0.44) * 4.8)`: measured
mean 0.545, std 0.068, p10 0.463, p90 0.631, min 0.349, max 0.804 - the only channel among the imported sets
that is both centred and broad (see build_roof_materials.py for the full table and for why this matters).
"""

import json

import unreal

REPO = r"D:\Sightline"
AC = REPO + r"\_downloads\assets\ambientcg"
TEXDIR = "/Game/Sightline/Water/Foam"
MATDIR = "/Game/Sightline/Water"
FOLDER = "Foam"
MACRO_TEX = "/Game/Sightline/Textures/brown_mud_rocks_01/T_brown_mud_rocks_01_ARM"
MACRO_T, MACRO_K = 0.44, 4.8
CLIP = 0.12

mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS
MST = unreal.MaterialSamplerType

if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports, material recompiles and saves are unreliable while PIE runs")

with open(REPO + r"\data\scene\foam.json") as fh:
    plan = json.load(fh)
with open(REPO + r"\data\scene\flood_valley.json") as fh:
    meta = json.load(fh)
BASE_Z = meta["base_z_m"]


# --- textures --------------------------------------------------------------------------------------------
def import_tex(path, name, kind):
    dest = TEXDIR
    full = f"{dest}/{name}"
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", dest), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", False)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    tex = unreal.load_asset(full)
    if tex is None:
        raise RuntimeError(f"import failed: {path}")
    if kind == "normal":
        tex.set_editor_property("srgb", False)
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
    elif kind == "mask":
        tex.set_editor_property("srgb", False)
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
    eal.save_asset(full)
    return tex


TEX = {}
for setname in ("Foam001", "Foam002"):
    TEX[setname] = {
        "col": import_tex(fr"{AC}\{setname}\{setname}_2K-JPG_Color.jpg", f"T_{setname}_D", "color"),
        "opa": import_tex(fr"{AC}\{setname}\{setname}_2K-JPG_Opacity.jpg", f"T_{setname}_O", "mask"),
        "nrm": import_tex(fr"{AC}\{setname}\{setname}_2K-JPG_NormalDX.jpg", f"T_{setname}_N", "normal"),
    }
print("foam textures:", {k: v["col"].blueprint_get_size_x() for k, v in TEX.items()})
macro_tex = unreal.load_asset(MACRO_TEX)
if macro_tex is None:
    raise RuntimeError(f"{MACRO_TEX} is missing - run tools/scene/build_materials.py first")


# --- material --------------------------------------------------------------------------------------------
def new_material(path):
    """Reuse + clear: deleting a material a level actor references is refused, and delete_all_material_
    expressions leaves nodes behind (measured 64 -> 95 -> 111 over three reruns)."""
    if eal.does_asset_exist(path):
        mat = unreal.load_asset(path)
        for e in list(mel.get_material_expressions(mat)):
            mel.delete_material_expression(mat, e)
        mel.delete_all_material_expressions(mat)
        left = mel.get_num_material_expressions(mat)
        if left:
            raise RuntimeError(f"{path}: {left} expressions survived clearing")
        return mat
    d, n = path.rsplit("/", 1)
    return tools.create_asset(n, d, unreal.Material, unreal.MaterialFactoryNew())


m = new_material(f"{MATDIR}/M_Foam")
m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
m.set_editor_property("opacity_mask_clip_value", CLIP)
m.set_editor_property("two_sided", False)
Y = [0]


def node(cls, x=-900, **props):
    Y[0] += 80
    e = mel.create_material_expression(m, cls, x, Y[0])
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect failed: {a.get_name()}.{ao or 'out'} -> {b.get_name()}.{bi}")


def op(cls, a, b, a_out="", b_out="", x=-600):
    e = node(cls, x)
    for src, out, pin in ((a, a_out, "A"), (b, b_out, "B")):
        if isinstance(src, (int, float)):
            e.set_editor_property("const_" + pin.lower(), float(src))
        else:
            link(src, out, e, pin)
    return e


def sat(a, a_out=""):
    e = node(unreal.MaterialExpressionSaturate, -400)
    link(a, a_out, e, "")
    return e


def scalar(name, val):
    return node(unreal.MaterialExpressionScalarParameter, -1400, parameter_name=name, default_value=val)


MUL, ADD, SUB, DIV = (unreal.MaterialExpressionMultiply, unreal.MaterialExpressionAdd,
                      unreal.MaterialExpressionSubtract, unreal.MaterialExpressionDivide)

# UV0 of the generated ribbon is (metres along the shore, 0..1 across it). Scale u by 1/FoamTileM and v by
# RibbonWidthM/FoamTileM so the foam texture keeps its real-world size in both directions.
uv0 = node(unreal.MaterialExpressionTextureCoordinate, -1400)
tile = scalar("FoamTileM", 2.2)
su = op(DIV, 1.0, tile)
sv = op(DIV, scalar("RibbonWidthM", plan["ribbon_width_m"]), tile)
suv = node(unreal.MaterialExpressionAppendVector, -800)
link(su, "", suv, "A")
link(sv, "", suv, "B")
fuv = op(MUL, uv0, suv)

# A TextureSampleParameter2D's DEFAULT texture must match its sampler type or the WHOLE material fails to
# compile and renders as the grey WorldGridMaterial checker (an sRGB default under SAMPLERTYPE_MASKS did exactly
# that on 2026-09-10). Foam002 is imported with the same settings, so switching a parameter is safe.
def tex_param(name, tex, stype):
    e = node(unreal.MaterialExpressionTextureSampleParameter2D, -1000, parameter_name=name, texture=tex,
             sampler_type=stype, sampler_source=S)
    link(fuv, "", e, "UVs")
    return e


col = tex_param("FoamColor", TEX["Foam001"]["col"], MST.SAMPLERTYPE_COLOR)
opa = tex_param("FoamOpacity", TEX["Foam001"]["opa"], MST.SAMPLERTYPE_MASKS)
nrm = tex_param("FoamNormal", TEX["Foam001"]["nrm"], MST.SAMPLERTYPE_NORMAL)

# across-ribbon edge falloff, from UV0.v. saturate(1 - |2v - 1|) is symmetric about v = 0.5, which is why the
# generator can bake a quad's strength into a SHRUNK v range - and why UE's OBJ V flip changes nothing.
vco = node(unreal.MaterialExpressionComponentMask, -1200, r=False, g=True, b=False, a=False)
link(uv0, "", vco, "")
absn = node(unreal.MaterialExpressionAbs, -500)
link(op(SUB, op(MUL, vco, 2.0), 1.0), "", absn, "")
falloff = sat(op(SUB, 1.0, absn))

# large-scale patchiness so the band is not a uniform stripe (measured channel: see the docstring)
wxy = node(unreal.MaterialExpressionComponentMask, -1300, r=True, g=True, b=False, a=False)
link(node(unreal.MaterialExpressionWorldPosition, -1400), "", wxy, "")
mac_s = node(unreal.MaterialExpressionTextureSample, -1000, texture=macro_tex, sampler_type=MST.SAMPLERTYPE_MASKS,
             sampler_source=S, mip_value_mode=unreal.TextureMipValueMode.TMVM_MIP_LEVEL, const_mip_value=6)
link(op(DIV, wxy, scalar("FoamMacroCm", 3400.0)), "", mac_s, "UVs")
patch = op(ADD, op(MUL, sat(op(MUL, op(SUB, mac_s, MACRO_T, "G"), MACRO_K)), 0.55), 0.45)   # 0.45 .. 1.00

opacity = op(MUL, op(MUL, op(MUL, opa, scalar("FoamGain", 1.0), "R"), falloff), patch)
mel.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY_MASK)

tint = node(unreal.MaterialExpressionVectorParameter, -1400, parameter_name="FoamTint",
            default_value=unreal.LinearColor(2.08, 1.67, 1.38, 1.0))
mel.connect_material_property(op(MUL, col, tint), "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(scalar("FoamRoughness", 0.75), "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(scalar("FoamSpecular", 0.30), "", unreal.MaterialProperty.MP_SPECULAR)
flat = node(unreal.MaterialExpressionConstant3Vector, -600, constant=unreal.LinearColor(0, 0, 1, 1))
nlerp = node(unreal.MaterialExpressionLinearInterpolate, -300)
link(flat, "", nlerp, "A")
link(nrm, "", nlerp, "B")
link(scalar("FoamNormalStrength", 0.5), "", nlerp, "Alpha")
mel.connect_material_property(nlerp, "", unreal.MaterialProperty.MP_NORMAL)
mel.recompile_material(m)
eal.save_asset(m.get_path_name())


def assert_compiles(mat):
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB.")
    return s


_s = assert_compiles(m)
print(f"M_Foam compiles: {_s.num_pixel_shader_instructions} instructions, "
      f"{_s.num_pixel_texture_samples} texture samples, clip {CLIP}")

inst_path = f"{MATDIR}/MI_Foam_Shore"
if eal.does_asset_exist(inst_path):
    inst = unreal.load_asset(inst_path)
else:
    inst = tools.create_asset("MI_Foam_Shore", MATDIR, unreal.MaterialInstanceConstant,
                              unreal.MaterialInstanceConstantFactoryNew())
mel.set_material_instance_parent(inst, m)
for pname, tex in (("FoamColor", TEX["Foam001"]["col"]), ("FoamOpacity", TEX["Foam001"]["opa"]),
                   ("FoamNormal", TEX["Foam001"]["nrm"])):
    mel.set_material_instance_texture_parameter_value(inst, pname, tex)
mel.set_material_instance_scalar_parameter_value(inst, "RibbonWidthM", plan["ribbon_width_m"])
mel.update_material_instance(inst)
eal.save_asset(inst_path)
assert_compiles(inst)

# --- mesh ------------------------------------------------------------------------------------------------
t = unreal.AssetImportTask()
for k, v in (("filename", REPO + "\\data\\scene\\" + plan["obj"].replace("/", "\\")),
             ("destination_path", MATDIR), ("destination_name", "SM_FloodFoam"),
             ("automated", True), ("replace_existing", True), ("save", False)):
    t.set_editor_property(k, v)
tools.import_asset_tasks([t])
sm = unreal.load_asset(f"{MATDIR}/SM_FloodFoam")
if sm is None:
    raise RuntimeError("foam mesh import failed - run tools/scene/gen_foam.py first")
ns = sm.get_editor_property("nanite_settings")          # imported meshes arrive with Nanite ON and the triangle
ns.set_editor_property("enabled", False)                # count then reports the fallback mesh instead
sm.set_editor_property("nanite_settings", ns)
for _i, _sl in enumerate(sm.get_editor_property("static_materials")):
    sm.set_material(_i, inst)                           # the OBJ has one `usemtl foam` group, but be safe
eal.save_asset(sm.get_path_name())
# A 2 % tolerance still catches the Nanite fallback (which reports hundreds of triangles, not thousands) while
# allowing the importer to weld away a degenerate or two.
got, want_t = sm.get_num_triangles(0), plan["counts"]["triangles"]
if abs(got - want_t) > max(4, 0.02 * want_t):
    raise RuntimeError(f"SM_FloodFoam has {got} triangles, generator says {want_t} (Nanite fallback?)")

# --- actor -----------------------------------------------------------------------------------------------
removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

act = eas.spawn_actor_from_object(sm, unreal.Vector(0, 0, 0),
                                  unreal.Rotator(0.0, 0.0, plan["ue_actor"]["yaw_deg"]))
act.set_actor_label("FloodFoam")
act.set_folder_path(FOLDER)
act.tags = ["Foam", "Water"]
comp = act.static_mesh_component
comp.set_mobility(unreal.ComponentMobility.MOVABLE)     # so a future flood_level.py can lift it with the water
comp.set_collision_profile_name("NoCollision")          # never let foam block sim_fly or a spawner's trace
comp.set_editor_property("cast_shadow", False)          # a flat sheet at the surface must not shade the water

# --- verify the transform: the ONLY way to catch a wrong yaw or a shifted pivot without a render ----------
# The terrain OBJ happens to be centred on the world origin, so nothing in this project has ever proved whether
# UE's OBJ importer keeps the file's origin or re-pivots to the bounding-box centre. The foam ribbon is NOT
# centred (east -276..488 m, north +/-1026 m), so it would expose the difference as a ~106 m offset. Rather than
# fail on it, measure the actor's real bounds and translate it onto the expected ones: a pure translation is
# valid whichever way the importer behaves, and the EXTENT check below still catches a wrong yaw.
b = plan["ue_actor"]["bounds_cm"]
want = ((b["x_north"][0] + b["x_north"][1]) / 2, (b["y_east"][0] + b["y_east"][1]) / 2)
half = ((b["x_north"][1] - b["x_north"][0]) / 2, (b["y_east"][1] - b["y_east"][0]) / 2)
origin, extent = act.get_actor_bounds(False)
if abs(extent.x - half[0]) > 200.0 or abs(extent.y - half[1]) > 200.0:
    raise RuntimeError(
        f"FloodFoam has the wrong footprint: extent ({extent.x:.0f}, {extent.y:.0f}) cm, expected "
        f"({half[0]:.0f}, {half[1]:.0f}). The actor yaw ({plan['ue_actor']['yaw_deg']}) or the OBJ axes are "
        f"wrong - do NOT capture anything until this matches.")
shift = unreal.Vector(want[0] - origin.x, want[1] - origin.y, 0.0)
if abs(shift.x) > 1.0 or abs(shift.y) > 1.0:
    print(f"  re-pivoted on import: shifting FloodFoam by ({shift.x:.1f}, {shift.y:.1f}) cm")
    loc = act.get_actor_location()
    act.set_actor_location(unreal.Vector(loc.x + shift.x, loc.y + shift.y, loc.z), False, False)
    origin, extent = act.get_actor_bounds(False)
errs = [abs(origin.x - want[0]), abs(origin.y - want[1]),
        abs(extent.x - half[0]), abs(extent.y - half[1])]
if max(errs) > 200.0:
    raise RuntimeError(
        f"FloodFoam is still in the wrong place after the shift: origin ({origin.x:.0f}, {origin.y:.0f}) "
        f"extent ({extent.x:.0f}, {extent.y:.0f}) cm, expected origin ({want[0]:.0f}, {want[1]:.0f}) "
        f"extent ({half[0]:.0f}, {half[1]:.0f}).")


def trace_z_cm(north_m, east_m):
    hit = unreal.SystemLibrary.line_trace_single(
        world, unreal.Vector(north_m * 100.0, east_m * 100.0, 100000),
        unreal.Vector(north_m * 100.0, east_m * 100.0, -100000), unreal.TraceTypeQuery.ECC_VISIBILITY, True, [],
        unreal.DrawDebugTrace.NONE, True)
    return hit.to_tuple()[4].z if hit else None


# secondary check: the terrain in the level must be the same seed this foam was generated from. A probe may
# legitimately land on a debris prop or a house, so require a majority rather than all of them.
ok, probes = 0, []
for p in plan["probe_points"]:
    z = trace_z_cm(p["north_m"], p["east_m"])
    want_z = (p["terrain_asl_m"] - BASE_Z) * 100.0
    probes.append(None if z is None else round(z - want_z, 1))
    if z is not None and abs(z - want_z) < 40.0:
        ok += 1
if ok < 5:
    raise RuntimeError(f"only {ok}/{len(plan['probe_points'])} shoreline probes match the generator's terrain "
                       f"(deltas in cm: {probes}). The level's terrain is not this seed - re-run gen_terrain.py "
                       f"and build_flood_valley.py.")

saved = les.save_current_level()
c = plan["counts"]
print(f"foam: removed {removed} old actor(s), placed 1 (SM_FloodFoam, {c['quads_total']:,} quads / "
      f"{c['triangles']:,} tris) covering ~{c['approx_shoreline_km']} km of shoreline")
print(f"  {c['quads_shoreline']:,} shoreline + {c['quads_houses']} upstream-of-house + "
      f"{c['quads_obstacles']} obstacle quads; water level {plan['water_level_m']:.3f} m ASL")
print(f"  bounds check OK (max error {max(errs):.0f} cm); {ok}/{len(plan['probe_points'])} terrain probes "
      f"matched (deltas cm: {probes})")
print(f"level saved {saved}")
print("NOW LOOK: run tools/scene/qa_shots.py. If the foam edge is too crunchy raise FoamGain on MI_Foam_Shore; "
      "if there is too much of it, lower FoamGain (each 0.1 of gain moves coverage ~5 points).")
