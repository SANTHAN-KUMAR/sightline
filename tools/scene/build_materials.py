"""Import the CC0 texture sets and build the FloodValley terrain and flood-water materials (running editor, PIE off).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_materials.py').read())"

Sources: Poly Haven CC0 2K sets in _downloads/assets (manifest: _downloads/assets/MANIFEST.md) and the engine
Water plugin's tiling normals. Randomisation is OFF by default (SOLUTION_DOC §5.5c): one fixed material per zone.

M_FloodValleyTerrain, all world-space (no reliance on the OBJ's UV orientation):
  hillslopes  lush grass broken up by forest-floor leaf litter (large-scale mask, kills tiling)
  steep > ~30 deg  red laterite soil and stones (Western Ghats cut slopes)
  zone R (deposit fan)  mud with embedded rocks (debris-flow deposit, §2.2 zone 1)
  zone G/B (terrace, channel banks) + a band up to ~2 m above flood stage  wet brown silt (fresh flood deposit)
  The zone mask T_FloodValley_Zones is sampled from world position: u = east/2048 + 0.5, v = 0.5 - north/2048
  (the PNG is written north-up by gen_terrain.py).
M_FloodWater (single-layer water): silt-laden scattering/absorption as parameters (turbidity slice, §2.3 row 1),
  two panning tiling normals for surface ripples.
"""

import json

import unreal

REPO = r"D:\Sightline"
ASSETS = REPO + r"\_downloads\assets\polyhaven"
mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
with open(REPO + r"\data\scene\flood_valley.json") as f:
    meta = json.load(f)
SIZE_CM = meta["size_m"] * 100.0
FLOOD_Z_CM = meta["ue_import"]["flood_water_z_cm"]

LAYERS = {  # key: (poly haven id, diffuse file suffix, tile size in metres)
    "grass": ("leafy_grass", "diff", 4.0),
    "forest": ("forest_leaves_02", "diffuse", 4.0),
    "laterite": ("red_laterite_soil_stones", "diff", 5.0),
    "mud": ("brown_mud_rocks_01", "diff", 5.0),
    "silt": ("brown_mud_02", "diff", 4.0),
}


def import_tex(path, dest, name, kind):
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", dest), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", False)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    tex = unreal.load_asset(f"{dest}/{name}")
    if kind == "normal":
        tex.set_editor_property("srgb", False)
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
    elif kind == "arm":
        tex.set_editor_property("srgb", False)
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
    eal.save_asset(tex.get_path_name())
    return tex


textures = {}
for key, (pid, dsuf, _) in LAYERS.items():
    dest = f"/Game/Sightline/Textures/{pid}"
    textures[key] = {
        "col": import_tex(fr"{ASSETS}\{pid}\{pid}_{dsuf}_2k.jpg", dest, f"T_{pid}_D", "color"),
        "nrm": import_tex(fr"{ASSETS}\{pid}\{pid}_nor_dx_2k.jpg", dest, f"T_{pid}_N", "normal"),
        "arm": import_tex(fr"{ASSETS}\{pid}\{pid}_arm_2k.jpg", dest, f"T_{pid}_ARM", "arm"),
    }
print("textures imported:", {k: v["col"].blueprint_get_size_x() for k, v in textures.items()})


def new_material(path):
    """Reuse + clear an existing material: deleting one that a level actor references is refused (create_asset
    then returns None), and reuse keeps every reference valid."""
    if eal.does_asset_exist(path):
        mat = unreal.load_asset(path)
        # delete_all_material_expressions leaves nodes behind (measured: 64 -> 95 -> 111 over three reruns), so
        # delete each expression explicitly and insist the graph is empty before rebuilding.
        for e in list(mel.get_material_expressions(mat)):
            mel.delete_material_expression(mat, e)
        mel.delete_all_material_expressions(mat)
        left = mel.get_num_material_expressions(mat)
        if left:
            raise RuntimeError(f"{path}: {left} expressions survived clearing; graph would accumulate junk")
        return mat
    d, n = path.rsplit("/", 1)
    return tools.create_asset(n, d, unreal.Material, unreal.MaterialFactoryNew())


class G:
    """Tiny graph builder over MaterialEditingLibrary; x/y only for readability in the editor."""

    def __init__(self, m):
        self.m, self.y = m, 0

    def node(self, cls, x=-1200, **props):
        self.y += 90
        e = mel.create_material_expression(self.m, cls, x, self.y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def link(self, a, a_out, b, b_in):
        if not mel.connect_material_expressions(a, a_out, b, b_in):
            raise RuntimeError(f"connect failed: {a.get_name()}.{a_out} -> {b.get_name()}.{b_in}")

    def const(self, v):
        return self.node(unreal.MaterialExpressionConstant, r=v)

    def op(self, cls, a, b, a_out="", b_out=""):
        e = self.node(cls, -400)
        for src, out, pin in ((a, a_out, "A"), (b, b_out, "B")):
            if isinstance(src, (int, float)):
                e.set_editor_property("const_" + pin.lower(), float(src))
            else:
                self.link(src, out, e, pin)
        return e

    def lerp(self, a, b, alpha, alpha_out=""):
        e = self.node(unreal.MaterialExpressionLinearInterpolate, -200)
        self.link(a, "", e, "A")
        self.link(b, "", e, "B")
        self.link(alpha, alpha_out, e, "Alpha")
        return e

    def sat(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionSaturate, -300)
        self.link(a, a_out, e, "")
        return e

    def tex(self, t, uv, stype, shared=unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS):
        e = self.node(unreal.MaterialExpressionTextureSample, -800, texture=t, sampler_type=stype, sampler_source=shared)
        self.link(uv, "", e, "UVs")
        return e


# ============================== terrain ==============================
m = new_material("/Game/Sightline/Terrain/M_FloodValleyTerrain")
g = G(m)
wp = g.node(unreal.MaterialExpressionWorldPosition, -1600)
wx = g.node(unreal.MaterialExpressionComponentMask, -1500, r=True, g=False, b=False, a=False)
wy = g.node(unreal.MaterialExpressionComponentMask, -1500, r=False, g=True, b=False, a=False)
wz = g.node(unreal.MaterialExpressionComponentMask, -1500, r=False, g=False, b=True, a=False)
wxy = g.node(unreal.MaterialExpressionComponentMask, -1500, r=True, g=True, b=False, a=False)
for msk in (wx, wy, wz, wxy):
    g.link(wp, "", msk, "")

# zone mask UV from world position (UE X = north, Y = east)
u = g.op(unreal.MaterialExpressionAdd, g.op(unreal.MaterialExpressionMultiply, wy, 1.0 / SIZE_CM), 0.5)
v = g.op(unreal.MaterialExpressionAdd, g.op(unreal.MaterialExpressionMultiply, wx, -1.0 / SIZE_CM), 0.5)
zuv = g.node(unreal.MaterialExpressionAppendVector, -1000)
g.link(u, "", zuv, "A")
g.link(v, "", zuv, "B")
zones = g.tex(unreal.load_asset("/Game/Sightline/Terrain/T_FloodValley_Zones"), zuv,
              unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, unreal.SamplerSourceMode.SSM_CLAMP_WORLD_GROUP_SETTINGS)

# Smooth macro noise (~40 m blobs): a LOW MIP (level 7 = 16 px) of a texture, so it is a soft field rather than
# fine detail. It drives both the anti-tiling blend weight and a large-scale brightness variation.
noise_uv = g.op(unreal.MaterialExpressionMultiply, wxy, 1.0 / 4100.0)
noise = g.node(unreal.MaterialExpressionTextureSample, -800, texture=textures["forest"]["col"],
               sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_COLOR,
               sampler_source=unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS,
               mip_value_mode=unreal.TextureMipValueMode.TMVM_MIP_LEVEL, const_mip_value=7)
g.link(noise_uv, "", noise, "UVs")
noise_r = g.sat(g.op(unreal.MaterialExpressionMultiply, g.op(unreal.MaterialExpressionSubtract, noise, 0.25, "R"), 3.0))
blend_w = g.op(unreal.MaterialExpressionAdd, g.op(unreal.MaterialExpressionMultiply, noise_r, 0.5), 0.25)  # 0.25..0.75

layer = {}
for key, (pid, _, tile_m) in LAYERS.items():
    # Anti-tiling, measured need: a 4 m tile repeated visibly every ~110 px at the 45 m survey altitude (capture
    # 20260910-212806). Tiles are doubled, and colour, normal AND ARM are each blended with a copy at a
    # non-integer 0.137x scale (offset so the two grids never align), weighted by the macro noise, so no constant
    # period survives. Shared wrap samplers keep the 31 samples under the sampler limit.
    uv = g.op(unreal.MaterialExpressionMultiply, wxy, 1.0 / (tile_m * 200.0))
    uv_far = g.op(unreal.MaterialExpressionAdd, g.op(unreal.MaterialExpressionMultiply, uv, 0.137), 0.371)
    t = textures[key]
    maps = []
    for tex_key, stype in (("col", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR),
                           ("nrm", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL),
                           ("arm", unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)):
        maps.append(g.lerp(g.tex(t[tex_key], uv, stype), g.tex(t[tex_key], uv_far, stype), blend_w))
    if key == "grass":
        # leafy_grass is a DRY scan: measured linear means R=0.324 G=0.239 B=0.109, i.e. yellow-brown (R > G).
        # Monsoon Kerala hillsides are saturated green, so the tint has to invert that ordering, not merely
        # nudge it: (0.35, 1.0, 0.45) lands near linear (0.11, 0.24, 0.05).
        tint = g.node(unreal.MaterialExpressionVectorParameter, -600, parameter_name="GrassTint",
                      default_value=unreal.LinearColor(0.35, 1.0, 0.45, 1.0))
        maps[0] = g.op(unreal.MaterialExpressionMultiply, maps[0], tint)
    layer[key] = tuple(maps)

# masks
macro_uv = g.op(unreal.MaterialExpressionMultiply, wxy, 1.0 / 9000.0)       # ~90 m blobs of leaf litter
# The mask channel must be a field that actually VARIES around the threshold. forest_leaves_02's ARM green is
# roughness, and leaf litter is uniformly rough: measured p10=0.878 / p50=0.937, so sat((G-0.55)*4) evaluated to
# 1.0 over the whole map and leaf litter replaced 100 % of the grass layer (the hillsides rendered dry orange,
# and GrassTint had no visible effect at all). brown_mud_rocks_01's ARM green is genuinely broad
# (mean 0.545, std 0.243); sat((G-0.55)*3) gives ~27 % coverage, which is the patchiness intended here.
macro = g.tex(textures["mud"]["arm"], macro_uv, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
forest_a = g.sat(g.op(unreal.MaterialExpressionMultiply, g.op(unreal.MaterialExpressionSubtract, macro, 0.55, "G"), 3.0))
nws = g.node(unreal.MaterialExpressionVertexNormalWS, -1600)
nz = g.node(unreal.MaterialExpressionComponentMask, -1500, r=False, g=False, b=True, a=False)
g.link(nws, "", nz, "")
steep_a = g.sat(g.op(unreal.MaterialExpressionMultiply, g.op(unreal.MaterialExpressionSubtract, 0.9, nz), 8.0))
# Fresh silt line: fully silt up to ~0.6 m above flood stage, fading out by ~1.2 m (a 2 m band covered most of a
# gentle bank in capture 20260910-212806).
flood_z = g.node(unreal.MaterialExpressionScalarParameter, -1600, parameter_name="SiltLineZ", default_value=FLOOD_Z_CM + 120.0)
band_a = g.sat(g.op(unreal.MaterialExpressionDivide, g.op(unreal.MaterialExpressionSubtract, flood_z, wz), 60.0))
silt_a = g.op(unreal.MaterialExpressionMax, g.op(unreal.MaterialExpressionMax, zones, zones, "G", "B"), band_a)
fan_a = g.sat(g.op(unreal.MaterialExpressionMultiply, g.op(unreal.MaterialExpressionSubtract, zones, 0.1, "R"), 1.5))


def blend(a, b, alpha):
    return tuple(g.lerp(a[i], b[i], alpha) for i in range(3))


out = blend(layer["grass"], layer["forest"], forest_a)
out = blend(out, layer["laterite"], steep_a)
out = blend(out, layer["mud"], fan_a)
out = blend(out, layer["silt"], silt_a)
col, nrm, arm = out
mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(nrm, "", unreal.MaterialProperty.MP_NORMAL)
armm = {c: g.node(unreal.MaterialExpressionComponentMask, -100, r=(c == "r"), g=(c == "g"), b=False, a=False) for c in "rg"}
for c, e in armm.items():
    g.link(arm, "", e, "")
mel.connect_material_property(armm["r"], "", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
mel.connect_material_property(armm["g"], "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.recompile_material(m)
eal.save_asset(m.get_path_name())
print("terrain material nodes:", mel.get_num_material_expressions(m))

# ============================== water ==============================
w = new_material("/Game/Sightline/Water/M_FloodWater")
w.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_SINGLE_LAYER_WATER)
g = G(w)


def vparam(name, rgb):
    return g.node(unreal.MaterialExpressionVectorParameter, -700, parameter_name=name,
                  default_value=unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))


def sparam(name, val):
    return g.node(unreal.MaterialExpressionScalarParameter, -700, parameter_name=name, default_value=val)


# Silt-laden monsoon water (§2.3 row 1): extinction ~10-20 /m => opaque within 0.05-0.3 m.
# The single-layer-water surface tint tends to scattering/(scattering+absorption). The first pass used a dark
# BaseColor (0.09, 0.065, 0.035) with absorption above scattering, which rendered the flood as near-black and
# made the 60 m nadir frame - the exact view the detector trains on - a featureless dark field. Heavy suspended
# silt is bright: these values give albedo (0.85, 0.73, 0.43), a cafe-au-lait flood surface.
slw = g.node(unreal.MaterialExpressionSingleLayerWaterMaterialOutput, 0)
for src, pin in ((vparam("Scattering", (7.0, 5.0, 2.6)), "ScatteringCoefficients"),
                 (vparam("Absorption", (1.2, 1.9, 3.4)), "AbsorptionCoefficients"),
                 (sparam("PhaseG", 0.35), "PhaseG"),
                 (sparam("ColorScaleBehindWater", 0.0), "ColorScaleBehindWater")):
    g.link(src, "", slw, pin)
mel.connect_material_property(vparam("BaseColor", (0.36, 0.27, 0.17)), "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(sparam("Roughness", 0.07), "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(sparam("Specular", 0.5), "", unreal.MaterialProperty.MP_SPECULAR)

wn = unreal.load_asset("/Water/Textures/Normals/T_Water_TilingNormal_Waves_02")
if wn is None:
    raise RuntimeError("engine water normal not found at /Water/Textures/Normals/T_Water_TilingNormal_Waves_02")
wwp = g.node(unreal.MaterialExpressionWorldPosition, -1600)
wxy2 = g.node(unreal.MaterialExpressionComponentMask, -1500, r=True, g=True, b=False, a=False)
g.link(wwp, "", wxy2, "")
normals = []
for scale_m, sx, sy in ((6.0, 0.035, 0.02), (15.0, -0.015, 0.03)):   # flow is north -> south
    uv = g.op(unreal.MaterialExpressionMultiply, wxy2, 1.0 / (scale_m * 100.0))
    pan = g.node(unreal.MaterialExpressionPanner, -1000, speed_x=sx, speed_y=sy)
    g.link(uv, "", pan, "Coordinate")
    normals.append(g.tex(wn, pan, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL))
# BlendAngleCorrectedNormals is a material FUNCTION, not an expression class; sum + normalise is the cheap blend.
nsum = g.node(unreal.MaterialExpressionNormalize, -500)
g.link(g.op(unreal.MaterialExpressionAdd, normals[0], normals[1]), "", nsum, "VectorInput")
flat = g.node(unreal.MaterialExpressionConstant3Vector, -500, constant=unreal.LinearColor(0, 0, 1, 1))
strength = sparam("NormalStrength", 0.35)  # 0.6 read as large dark blobs under a low dawn sun (capture 20260910-212927)
nfinal = g.lerp(flat, nsum, strength)
mel.connect_material_property(nfinal, "", unreal.MaterialProperty.MP_NORMAL)
mel.recompile_material(w)
eal.save_asset(w.get_path_name())
print("water material nodes:", mel.get_num_material_expressions(w))

# ============================== assign ==============================
actors = {a.get_actor_label(): a for a in eas.get_all_level_actors()}
actors["Ground"].static_mesh_component.set_material(0, m)
sm = unreal.load_asset("/Game/Sightline/Terrain/SM_FloodValley")
sm.set_material(0, m)
eal.save_asset(sm.get_path_name())
actors["FloodWater"].static_mesh_component.set_material(0, w)
print("level saved", les.save_current_level())
