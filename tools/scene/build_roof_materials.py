"""Upgrade the settlement's shared master material `M_PBR_Master` in place: anti-tiling + a flood tide line.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_roof_materials.py').read())"

Run AFTER `build_buildings.py` (which creates the master and its instances) and with PIE STOPPED. Idempotent:
it rebuilds the master graph from scratch every time, and restores every instance's texture/vector overrides
afterwards, so re-running it cannot lose an instance's textures.

WHY
---
Roofs are the largest surfaces a nadir survey camera sees. `build_buildings.py` maps the generated OBJ UVs
straight through (`TILE_M`: wall 2.0 m, roof_tile 1.5 m, roof_sheet 2.0 m), so at the survey GSD of 1.77 cm/px a
1.5 m clay-tile tile repeats every 85 px and 73 houses of five archetypes show the *identical* texture, in a grid.
Measured low-frequency content of the diffuse maps (std of the 64 px mip = the blotch scale that survives to
survey altitude, sRGB luma):

    clay_roof_tiles        0.0432      <- the worst offender, and it is on 46 % of the roofs
    worn_mossy_plasterwall 0.0648
    dirty_concrete         0.0516
    rusty_corrugated_iron  0.0215
    weathered_planks       0.0212
    painted_plaster_wall   0.0110      <- almost flat; anti-tiling buys little here

Three independent period-breakers are added, all driven from world space so nothing depends on the OBJ UVs:

1. **Per-object UV jitter.** `ObjectPositionWS` is constant across one actor, so `frac(dot(objPos.xy, k))` is a
   free per-house random UV offset with ZERO distortion (unlike a smoothly varying warp, which shears the tile
   grid). This alone stops 73 houses from sharing one pixel-identical roof.
2. **Two UV scales blended by a low-frequency macro field**, exactly the treatment the terrain got in
   `build_materials.py`.
3. **Macro brightness variation** from the same field: period-free by construction and, at survey altitude, the
   single most effective cue against "same roof, 73 times".

MEASURED macro-field choice (the terrain bug was a mask sampled from a roughness channel whose 10th percentile
was already 0.878, so `saturate((G-0.55)*4)` was 1.0 everywhere). Statistics below are of the 32 px mip -- the
level this material actually samples -- of the raw JPEG, which is what a TC_MASKS texture (sRGB OFF) returns:

    texture / channel                mean    std    p10    p50    p90    min    max
    brown_mud_rocks_01 ARM.G        0.545  0.068  0.463  0.541  0.631  0.349  0.804   <- CHOSEN
    clay_roof_tiles    ARM.R        0.651  0.051  0.580  0.651  0.718  0.506  0.784
    red_laterite       ARM.R        0.626  0.020  0.604  0.624  0.651  0.569  0.765
    brown_mud_02       ARM.G        0.951  0.028  0.914  0.957  0.980  0.828  1.000   <- saturated, useless
    rusty_corrugated   ARM.G        0.936  0.004  0.933  0.937  0.941  0.925  0.945   <- dead flat, useless
    painted_plaster    ARM.G        0.919  0.002  0.918  0.918  0.922  0.910  0.929   <- dead flat, useless

`brown_mud_rocks_01_ARM.G` is the only channel that is both centred and broad, so the remap is
`saturate((G - 0.44) * 4.8)`: it puts the measured p10/p90 (0.463 / 0.631) at 0.11 / 0.92 and the mean at 0.50,
i.e. the field really does swing over the full 0..1 range instead of clamping to one end.

FAR-UV MODE, also measured. 1-D profiles of the source maps (std of the row/column means, and the dominant
period of each profile) show how structured each material is:

    rusty_corrugated_iron normal:  std along u 0.0694, along v 0.0005; 59 cycles along u
    clay_roof_tiles       normal:  std along u 0.0746, along v 0.0083; 27 cycles along u, 16 along v
    (everything else is unstructured)

So the two roof materials are strict lattices. For them a *scale* change would blend two different ridge pitches
and mush the corrugations/courses in oblique views, while an offset of an INTEGER number of lattice periods
(20/59 = 0.339 for the iron, 9/27 and 5/16 for the tiles) keeps the ridges perfectly in phase and still moves the
macro blotches to a different part of the map. Those two instances therefore get `UVScaleFar = 1.0` plus a
lattice-locked `UVFarOffset`; every other instance gets the true two-scale blend (`UVScaleFar = 0.53`).

TIDE LINE. The master also gains a world-Z silt band (`SiltLineZ`, default = the flood surface + 8 cm) with mud
staining below it and a darker accent at the line itself, broken up by a second, finer macro sample so the line is
not laser-straight across 73 buildings. It lives here, not in `build_damage.py`, because `M_PBR_Master` must have
exactly ONE owner: two scripts clearing and rebuilding the same graph would silently undo each other.
Silt colour is `T_brown_mud_02_D` (measured linear mean RGB 0.0802 / 0.0633 / 0.0421 = dark wet mud); the default
`SiltTint` of (1.50, 1.45, 1.35) lifts it to linear (0.120, 0.092, 0.057), a light grey-brown dried-silt stain.

Everything is a material parameter, so the orchestrator can retune from a render without rebuilding the graph.
"""

import json

import unreal

REPO = r"D:\Sightline"
MATDIR = "/Game/Sightline/Buildings/Materials"
MASTER = f"{MATDIR}/M_PBR_Master"
MACRO_TEX = "/Game/Sightline/Textures/brown_mud_rocks_01/T_brown_mud_rocks_01_ARM"
SILT_TEX = "/Game/Sightline/Textures/brown_mud_02/T_brown_mud_02_D"

# saturate((G - MACRO_T) * MACRO_K): see the measured table in the docstring.
MACRO_T, MACRO_K = 0.44, 4.8

mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS
MST = unreal.MaterialSamplerType

if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: material recompiles and saves are unreliable while PIE runs")

if not eal.does_asset_exist(MASTER):
    raise RuntimeError(f"{MASTER} does not exist - run tools/scene/build_buildings.py first")

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    meta = json.load(fh)
FLOOD_Z_CM = meta["ue_import"]["flood_water_z_cm"]

master = unreal.load_asset(MASTER)
macro_tex = unreal.load_asset(MACRO_TEX)
silt_tex = unreal.load_asset(SILT_TEX)
if macro_tex is None or silt_tex is None:
    raise RuntimeError(f"missing source textures: {MACRO_TEX} / {SILT_TEX} (run build_materials.py first)")


# ---------------------------------------------------------------------------------------------------------
# 0. capture every instance override BEFORE the graph is cleared, so nothing can be lost by the rebuild
# ---------------------------------------------------------------------------------------------------------
def short(asset):
    """The asset's bare name. Taken from the package path rather than get_name() so that a name like
    'MI_Concrete.MI_Concrete' can never silently miss the per-instance lookups below."""
    return asset.get_path_name().rsplit("/", 1)[-1].split(".")[0]


def list_instances():
    out = []
    for p in eal.list_assets(MATDIR, recursive=False, include_folder=False):
        a = unreal.load_asset(p)
        if isinstance(a, unreal.MaterialInstanceConstant):
            out.append(a)
    return sorted(out, key=short)


def capture(inst):
    """Read back the instance's own texture / vector / scalar overrides by struct, with a by-name fallback."""
    got = {"tex": {}, "vec": {}, "scal": {}}
    for prop, key, getter in (("texture_parameter_values", "tex", "get_material_instance_texture_parameter_value"),
                              ("vector_parameter_values", "vec", "get_material_instance_vector_parameter_value"),
                              ("scalar_parameter_values", "scal", "get_material_instance_scalar_parameter_value")):
        try:
            for pv in inst.get_editor_property(prop):
                name = str(pv.get_editor_property("parameter_info").get_editor_property("name"))
                got[key][name] = pv.get_editor_property("parameter_value")
        except Exception as exc:                                       # noqa: BLE001 - editor API varies
            print(f"  ! {inst.get_name()}: could not read {prop} ({exc}); falling back to known names")
            fn = getattr(mel, getter, None)
            if fn is None:
                continue
            for name in (("BaseColor", "Normal", "ARM") if key == "tex" else ("Tint",) if key == "vec" else ()):
                try:
                    v = fn(inst, name)
                except Exception:                                      # noqa: BLE001
                    v = None
                if v is not None:
                    got[key][name] = v
    return got


INSTANCES = list_instances()
if not INSTANCES:
    raise RuntimeError(f"no material instances under {MATDIR} - run build_buildings.py first")
BEFORE = {short(i): capture(i) for i in INSTANCES}
for n, g in BEFORE.items():
    if not g["tex"]:
        raise RuntimeError(f"{n} has no texture overrides to preserve; refusing to rebuild the master blind")
print("captured instance overrides:", {n: sorted(g["tex"]) for n, g in BEFORE.items()})

_s0 = mel.get_statistics(master)
print(f"before: M_PBR_Master {_s0.num_pixel_shader_instructions} instructions, "
      f"{_s0.num_pixel_texture_samples} texture samples")


# ---------------------------------------------------------------------------------------------------------
# 1. clear the graph (delete_all_material_expressions leaves nodes behind - measured 2026-09-10)
# ---------------------------------------------------------------------------------------------------------
for e in list(mel.get_material_expressions(master)):
    mel.delete_material_expression(master, e)
mel.delete_all_material_expressions(master)
left = mel.get_num_material_expressions(master)
if left:
    raise RuntimeError(f"{MASTER}: {left} expressions survived clearing; the graph would accumulate junk")


class G:
    """Tiny graph builder over MaterialEditingLibrary (x/y are only for readability in the editor)."""

    def __init__(self, m):
        self.m, self.y = m, 0

    def node(self, cls, x=-1200, **props):
        self.y += 80
        e = mel.create_material_expression(self.m, cls, x, self.y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def link(self, a, a_out, b, b_in):
        if not mel.connect_material_expressions(a, a_out, b, b_in):
            raise RuntimeError(f"connect failed: {a.get_name()}.{a_out or 'out'} -> {b.get_name()}.{b_in}")

    def op(self, cls, a, b, a_out="", b_out="", x=-600):
        e = self.node(cls, x)
        for src, out, pin in ((a, a_out, "A"), (b, b_out, "B")):
            if isinstance(src, (int, float)):
                e.set_editor_property("const_" + pin.lower(), float(src))
            else:
                self.link(src, out, e, pin)
        return e

    def mul(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionMultiply, a, b, a_out, b_out)

    def add(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionAdd, a, b, a_out, b_out)

    def sub(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionSubtract, a, b, a_out, b_out)

    def div(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionDivide, a, b, a_out, b_out)

    def sat(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionSaturate, -500)
        self.link(a, a_out, e, "")
        return e

    def absx(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionAbs, -500)
        self.link(a, a_out, e, "")
        return e

    def frac(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionFrac, -900)
        self.link(a, a_out, e, "")
        return e

    def lerp(self, a, b, alpha, a_out="", b_out="", alpha_out=""):
        e = self.node(unreal.MaterialExpressionLinearInterpolate, -300)
        for src, out, pin in ((a, a_out, "A"), (b, b_out, "B"), (alpha, alpha_out, "Alpha")):
            if isinstance(src, (int, float)):
                e.set_editor_property("const_" + ("alpha" if pin == "Alpha" else pin.lower()), float(src))
            else:
                self.link(src, out, e, pin)
        return e

    def maxx(self, a, b, a_out="", b_out=""):
        return self.op(unreal.MaterialExpressionMax, a, b, a_out, b_out)

    def mask(self, a, r=False, g=False, b=False, a_out=""):
        e = self.node(unreal.MaterialExpressionComponentMask, -400, r=r, g=g, b=b, a=False)
        self.link(a, a_out, e, "")
        return e

    def append(self, a, b, a_out="", b_out=""):
        e = self.node(unreal.MaterialExpressionAppendVector, -900)
        self.link(a, a_out, e, "A")
        self.link(b, b_out, e, "B")
        return e

    def dot(self, a, b):
        e = self.node(unreal.MaterialExpressionDotProduct, -1000)
        self.link(a, "", e, "A")
        self.link(b, "", e, "B")
        return e

    def scalar(self, name, val, x=-1600):
        return self.node(unreal.MaterialExpressionScalarParameter, x, parameter_name=name, default_value=val)

    def vector(self, name, rgba, x=-1600):
        return self.node(unreal.MaterialExpressionVectorParameter, x, parameter_name=name,
                         default_value=unreal.LinearColor(*rgba))

    def c2(self, x_, y_):
        return self.node(unreal.MaterialExpressionConstant2Vector, -1300, r=x_, g=y_)


g = G(master)

# --- probe how a TextureObject connects to a TextureSample -------------------------------------------------
# Two TextureSampleParameter2D nodes sharing one parameter name also works, but one TextureObjectParameter fed
# into two plain samples is unambiguous. The input pin is called "Tex" in UE 5.x; probe it rather than assume,
# and fall back to duplicate parameter nodes if the probe finds nothing.
TEX_OBJ_CLS = getattr(unreal, "MaterialExpressionTextureObjectParameter", None)


def _probe_tex_pin():
    if TEX_OBJ_CLS is None:
        return None
    tmp = mel.create_material_expression(master, TEX_OBJ_CLS, -3200, -600)
    smp = mel.create_material_expression(master, unreal.MaterialExpressionTextureSample, -3200, -480)
    found = None
    for cand in ("Tex", "TextureObject", "Texture"):
        if mel.connect_material_expressions(tmp, "", smp, cand):
            found = cand
            break
    mel.delete_material_expression(master, smp)
    mel.delete_material_expression(master, tmp)
    return found


TEX_PIN = _probe_tex_pin()
print("TextureSample texture-object pin:", TEX_PIN or "(none: using duplicate TextureSampleParameter2D nodes)")

# A TextureSampleParameter2D's DEFAULT texture must match its sampler type or the WHOLE material fails to compile
# and every instance silently renders as the grey WorldGridMaterial checker (2026-09-10: the engine's sRGB
# DefaultDiffuse under SAMPLERTYPE_MASKS did exactly that). Default each parameter to a real texture of the right
# kind, the same ones build_buildings.py used.
DEFAULTS = {
    "BaseColor": (unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_D"), MST.SAMPLERTYPE_COLOR),
    "Normal": (unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_NRM"), MST.SAMPLERTYPE_NORMAL),
    "ARM": (unreal.load_asset("/Game/Sightline/Textures/dirty_concrete/T_dirty_concrete_ARM"), MST.SAMPLERTYPE_MASKS),
}
for k, (t, _st) in DEFAULTS.items():
    if t is None:
        raise RuntimeError(f"default texture for parameter {k} is missing; re-run build_buildings.py")


def tex_param_pair(name, uv_near, uv_far):
    """Sample one texture PARAMETER at two UV sets and return (near_sample, far_sample)."""
    tex, stype = DEFAULTS[name]
    if TEX_PIN:
        obj = g.node(TEX_OBJ_CLS, -1900, parameter_name=name, texture=tex, sampler_type=stype)
        out = []
        for uv in (uv_near, uv_far):
            s = g.node(unreal.MaterialExpressionTextureSample, -1000, texture=tex, sampler_type=stype,
                       sampler_source=S)
            g.link(uv, "", s, "UVs")
            g.link(obj, "", s, TEX_PIN)
            out.append(s)
        return out
    out = []
    for uv in (uv_near, uv_far):
        s = g.node(unreal.MaterialExpressionTextureSampleParameter2D, -1000, parameter_name=name, texture=tex,
                   sampler_type=stype, sampler_source=S)
        g.link(uv, "", s, "UVs")
        out.append(s)
    return out


# --- world position ---------------------------------------------------------------------------------------
wp = g.node(unreal.MaterialExpressionWorldPosition, -2400)
wxy = g.mask(wp, r=True, g=True)
wz = g.mask(wp, b=True)

# --- 1. per-object UV jitter (constant across one actor: no distortion at all) ------------------------------
obj_pos_cls = getattr(unreal, "MaterialExpressionObjectPositionWS", None)
uv_scale = g.scalar("UVScale", 1.0)
uv0 = g.mul(g.node(unreal.MaterialExpressionTextureCoordinate, -2400), uv_scale)
if obj_pos_cls is None:
    print("  ! MaterialExpressionObjectPositionWS not available: per-object UV jitter skipped")
    uv_near = uv0
else:
    opxy = g.mask(g.node(obj_pos_cls, -2400), r=True, g=True)
    # two irrational-ish gradients in 1/cm: houses are >= 17 m apart, so dot() moves by >= 19 units between
    # neighbours and frac() decorrelates them completely.
    ju = g.frac(g.dot(opxy, g.c2(0.011300, 0.007100)))
    jv = g.frac(g.dot(opxy, g.c2(0.007900, -0.013100)))
    uv_near = g.add(uv0, g.mul(g.append(ju, jv), g.scalar("UVJitter", 1.0)))

# NOTE the ComponentMask: a VectorParameter's default output is float3, and the material compiler refuses
# "Arithmetic between types float2 and float3", so the offset must be masked down to RG before it meets a UV.
uv_far = g.add(g.mul(uv_near, g.scalar("UVScaleFar", 0.53)),
               g.mask(g.vector("UVFarOffset", (0.371, 0.183, 0.0, 1.0)), r=True, g=True))

# --- 2. macro fields --------------------------------------------------------------------------------------
def macro(name_cm, default_cm, mip):
    """A soft low-frequency world-space field: a low MIP of a MASKS texture, remapped by the measured constants."""
    uv = g.div(wxy, g.scalar(name_cm, default_cm))
    s = g.node(unreal.MaterialExpressionTextureSample, -1000, texture=macro_tex, sampler_type=MST.SAMPLERTYPE_MASKS,
               sampler_source=S, mip_value_mode=unreal.TextureMipValueMode.TMVM_MIP_LEVEL, const_mip_value=mip)
    g.link(uv, "", s, "UVs")
    return g.sat(g.mul(g.sub(s, MACRO_T, "G"), MACRO_K))


mA = macro("MacroTileCm", 2600.0, 6)          # ~26 m blobs: the anti-tiling weight and the brightness variation
mB = macro("MacroFineTileCm", 1200.0, 6)      # ~12 m: breaks up the tide line and the mud staining

blend_w = g.mul(g.add(g.mul(mA, 0.5), 0.25), g.scalar("AntiTileStrength", 1.0))   # 0.25 .. 0.75 at strength 1

# --- 3. the three maps, each blended between the two UV sets -----------------------------------------------
col_n, col_f = tex_param_pair("BaseColor", uv_near, uv_far)
nrm_n, nrm_f = tex_param_pair("Normal", uv_near, uv_far)
arm_n, arm_f = tex_param_pair("ARM", uv_near, uv_far)
col = g.lerp(col_n, col_f, blend_w)
nrm = g.lerp(nrm_n, nrm_f, blend_w)
arm = g.lerp(arm_n, arm_f, blend_w)

# macro brightness variation: 1 +/- MacroTintStrength, period-free by construction
macro_tint = g.add(g.mul(g.mul(g.sub(mA, 0.5), g.scalar("MacroTintStrength", 0.12)), 2.0), 1.0)
base = g.mul(g.mul(col, g.vector("Tint", (1.0, 1.0, 1.0, 1.0))), macro_tint)

# --- 4. flood tide line and mud staining -------------------------------------------------------------------
# The line wobbles with the fine macro field so it is not perfectly horizontal across the whole settlement.
line_z = g.add(g.scalar("SiltLineZ", FLOOD_Z_CM + 8.0),
               g.mul(g.sub(mB, 0.5), g.scalar("SiltWobbleCm", 22.0)))
below = g.sat(g.div(g.sub(line_z, wz), g.scalar("SiltFadeCm", 90.0)))            # 0 at the line, 1 well below
band = g.sat(g.sub(1.0, g.div(g.absx(g.sub(wz, line_z)), g.scalar("SiltBandCm", 22.0))))  # accent at the line
streak = g.sat(g.add(g.mul(mB, 0.9), 0.55))                                     # 0.55 .. 1.0 patchiness
silt_a = g.sat(g.mul(g.mul(g.maxx(below, band), streak), g.scalar("SiltAmount", 0.85)))

silt_uv = g.mul(uv_near, g.scalar("SiltUVScale", 0.75))
silt_s = g.node(unreal.MaterialExpressionTextureSample, -1000, texture=silt_tex,
                sampler_type=MST.SAMPLERTYPE_COLOR, sampler_source=S)
g.link(silt_uv, "", silt_s, "UVs")
silt_col = g.mul(silt_s, g.vector("SiltTint", (1.50, 1.45, 1.35, 1.0)))

# --- 5. outputs --------------------------------------------------------------------------------------------
mel.connect_material_property(g.lerp(base, silt_col, silt_a), "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(nrm, "", unreal.MaterialProperty.MP_NORMAL)
mel.connect_material_property(g.mask(arm, r=True), "", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
mel.connect_material_property(g.lerp(g.mask(arm, g=True), g.scalar("SiltRoughness", 0.86), silt_a), "",
                              unreal.MaterialProperty.MP_ROUGHNESS)
# wet silt is never metallic: without this the rusty-iron roofs keep a metal sheen under the waterline
mel.connect_material_property(g.lerp(g.mask(arm, b=True), 0.0, silt_a), "", unreal.MaterialProperty.MP_METALLIC)

mel.recompile_material(master)
eal.save_asset(master.get_path_name())


def assert_compiles(mat):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    through the Python API except zeroed statistics. Fail loudly here instead of discovering it in a capture."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB.")
    return s


_s1 = assert_compiles(master)
print(f"after:  M_PBR_Master {_s1.num_pixel_shader_instructions} instructions, "
      f"{_s1.num_pixel_texture_samples} texture samples, {mel.get_num_material_expressions(master)} nodes")


# ---------------------------------------------------------------------------------------------------------
# 6. restore every instance override, then set the measured per-instance anti-tiling mode
# ---------------------------------------------------------------------------------------------------------
# Lattice-locked far offsets for the two structured roof maps (see the docstring): an offset of an integer
# number of lattice periods keeps the ridges in phase while moving the macro blotches somewhere else.
LATTICE = {
    "MI_RoofTile": (1.0, (9.0 / 27.0, 5.0 / 16.0)),          # clay tiles: 27 columns x 16 courses (measured)
    "MI_RoofSheet_Rusty": (1.0, (20.0 / 59.0, 0.410)),       # corrugated iron: 59 corrugations along u, flat along v
}
# Anti-tiling strength scaled by the measured 64 px-mip blotchiness of each set's diffuse map: painted plaster
# barely repeats (0.0110), mossy plaster repeats most (0.0648).
STRENGTH = {
    "MI_Wall_Cream": 0.45, "MI_Wall_Yellow": 0.45, "MI_Wall_Mint": 0.45,
    "MI_Wall_Weathered": 1.0, "MI_Concrete": 1.0, "MI_Wood": 0.7,
    "MI_RoofTile": 1.0, "MI_RoofSheet_Rusty": 0.8,
}

report, unknown = {}, []
for inst in INSTANCES:
    name = short(inst)
    if name not in STRENGTH:
        unknown.append(name)
    mel.set_material_instance_parent(inst, master)
    got = BEFORE[name]
    for pname, tex in got["tex"].items():
        mel.set_material_instance_texture_parameter_value(inst, pname, tex)
    for pname, vec in got["vec"].items():
        mel.set_material_instance_vector_parameter_value(inst, pname, vec)
    for pname, val in got["scal"].items():
        mel.set_material_instance_scalar_parameter_value(inst, pname, val)
    mel.set_material_instance_scalar_parameter_value(inst, "AntiTileStrength", STRENGTH.get(name, 1.0))
    if name in LATTICE:
        far, (ou, ov) = LATTICE[name]
        mel.set_material_instance_scalar_parameter_value(inst, "UVScaleFar", far)
        mel.set_material_instance_vector_parameter_value(inst, "UVFarOffset", unreal.LinearColor(ou, ov, 0.0, 1.0))
    mel.update_material_instance(inst)      # rebuild cached uniform expressions, else the render lags the asset
    eal.save_asset(inst.get_path_name())

    # The instance must still resolve to the SAME textures. Checking only for None is not enough: a lost
    # override falls back to the parent's dirty-concrete default, which is a texture, so every wall and roof
    # would quietly render as grey concrete. Compare the package paths.
    resolved = {}
    for pname, want_tex in got["tex"].items():
        v = mel.get_material_instance_texture_parameter_value(inst, pname)
        if v is None:
            raise RuntimeError(f"{name}.{pname} no longer resolves to a texture after the master rebuild")
        if want_tex is not None and v.get_path_name() != want_tex.get_path_name():
            raise RuntimeError(f"{name}.{pname} now resolves to {v.get_path_name()} but was "
                               f"{want_tex.get_path_name()}: the override was lost in the rebuild")
        resolved[pname] = v.get_name()
    s = assert_compiles(inst)
    report[name] = {"instructions": s.num_pixel_shader_instructions,
                    "samples": s.num_pixel_texture_samples,
                    "anti_tile": round(STRENGTH.get(name, 1.0), 2),
                    "mode": "lattice-locked offset" if name in LATTICE else "two-scale",
                    "textures": resolved}

for n in sorted(report):
    r = report[n]
    print(f"  {n:22s} {r['instructions']:4d} instr {r['samples']:2d} samples  anti-tile {r['anti_tile']:.2f} "
          f"({r['mode']})  {r['textures']}")
if unknown:
    print(f"  ! {len(unknown)} instances are not in the measured STRENGTH table and got the 1.0 default: "
          f"{unknown} (harmless, but check the names if a new instance was added)")
print(f"all {len(report)} instances compile and resolve their textures")
print("NOW LOOK: run tools/scene/qa_shots.py and open qa_2_settlement.png and qa_4_nadir45.png.")
