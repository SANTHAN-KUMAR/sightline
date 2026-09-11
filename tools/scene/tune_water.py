"""Re-tune M_FloodWater in place: make the water volume actually render, and give it a green-teal, depth-
graded colour instead of a flat silty tan (running editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\tune_water.py').read())"

`build_materials.py` OWNS this material: it clears and rebuilds the graph from scratch. This script only
edits parameter defaults and adds a handful of clearly-marked nodes on top, so **it must be re-run every time
build_materials.py runs** - exactly the relationship build_roof_materials.py has with build_buildings.py.
Every node this script creates carries `desc = "SIGHTLINE_TUNE_WATER"`, and a re-run replaces that set and
nothing else.

================================================================================================
1. THE WATER VOLUME WAS NOT BEING RENDERED AT ALL
================================================================================================
`BasePassPixelShader.usf` lines 1140-1141 (UE 5.8, D:\\UE_5.8\\Engine\\Shaders\\Private):

        const float BaseMaterialCoverageOverWater = Opacity;
        const float WaterVisibility = 1.0 - BaseMaterialCoverageOverWater;

and `SingleLayerWaterShading.ush:238`:

        Output.Luminance = WaterVisibility * (ScatteredLuminance + Transmittance * (...));

`Material.cpp:7977` makes MP_Opacity ACTIVE for MSM_SingleLayerWater even in an Opaque blend mode, and
`MaterialAttributeDefinitionMap.cpp:401` gives an unconnected Opacity the default **1.0**. build_materials.py
never connects Opacity, so `WaterVisibility` is 0 and the entire single-layer-water term - every one of the
Scattering / Absorption / PhaseG / ColorScaleBehindWater values it so carefully sets - is multiplied by zero
and thrown away. What actually rendered was the DefaultLit surface: `BaseColor` as a Lambertian diffuse lobe
plus a specular highlight. A flat lit card. That is the "uniform silty tan" in the reference comparison, and
no amount of tuning the scattering could ever have changed it.
(The Substrate path reaches the same place: `SubstrateLegacyConversion.ush:633` passes Opacity through as the
SLW BSDF's TopMaterialOpacity, and `Substrate.ush:1964` blends BaseColor over the water volume with it.)

FIX: connect MP_Opacity to `lerp(WaterOpacity, WaterOpacitySilt, silt_mix)`. Opacity is now the fraction of
surface film - scum, foam sheen, floating silt - drawn over the water, so 0.03 of clear water rising to 0.22
in the heaviest silt plumes. That both switches the volume on and keeps the existing plume work visible as a
surface film rather than as the water's entire colour.

================================================================================================
2. THE COEFFICIENTS WERE 100x TOO LARGE: THE UNIT IS 1/cm, NOT 1/m
================================================================================================
`MaterialExpressionSingleLayerWaterMaterialOutput.h`: "Valid range is [0,+inf[. **Unit is 1/cm.**" and the
shader multiplies them by `WaterVolumeDepth`, a scene-depth difference in UE units (centimetres).

The old values - Scattering (7.0, 5.0, 2.6), Absorption (1.2, 1.9, 3.4) - are therefore an extinction of
**820 per metre** in red: optical depth 1 at 1.2 mm. Even with the volume switched on, `Transmittance` would
be zero everywhere, `SafeScatteringAmount` would collapse to the constant single-scattering albedo, and the
flood would be one flat colour with no depth variation anywhere - the second half of the same symptom.

NEW VALUES, chosen from water optics rather than from a colour picker:

    turbidity   a turbid monsoon flood has a Secchi depth of roughly 0.35 m; the standard relation
                K_d * z_SD ~= 1.44 gives an effective attenuation of ~4.1 /m in the green
    hue         green-teal is what a turbid inland flood IS: pure water absorbs strongly in the red
                (0.35 /m at 650 nm) and CDOM/yellow substance absorbs strongly in the blue, so the green
                channel is the one that survives. Single-scattering albedo w = S/(S+A) per channel:

                                extinction /m        albedo w = S/(S+A)      per-cm S            per-cm A
                clear water     R 6.2 G 4.1 B 5.0    R 0.34 G 0.62 B 0.52    .0211 .0254 .0260   .0409 .0156 .0240
                silt plume      R 7.5 G 5.0 B 6.3    R 0.39 G 0.60 B 0.39    .0293 .0300 .0246   .0458 .0200 .0384

                Clear water is 0.55 : 1.00 : 0.84 - green-cyan. The plume is 0.65 : 1.00 : 0.65 - warmer and
                browner, and more opaque, exactly what a fresh sediment plume looks like from the air.
    depth       green transmittance is 0.51 at 0.17 m, 0.13 at 0.50 m, 0.017 at 1.0 m and 3e-4 at 2.0 m. The
                settlement stands in 0.5-3.7 m of water (settlement.json), so the deep terrace reads as opaque
                teal while submerged roads, kerbs, garden walls and the shallow margins show through - which
                is the depth variation the reference photograph has and our render did not.

================================================================================================
3. THE SCENE BEHIND THE WATER WAS BEING MULTIPLIED BY ZERO
================================================================================================
`ColorScaleBehindWater` was 0.0. The shader uses
`lerp(1.0, param, saturate(WaterVolumeDepth * 0.02))`, so beyond 50 cm of depth EVERYTHING behind the surface
was blacked out. Set to 1.0: the drowned ground is now attenuated by the physical transmittance and nothing
else, which is the other half of the depth cue.

================================================================================================
4. SMALLER, ALL REASONED
================================================================================================
    PhaseG        0.35 -> 0.10   Schlick's phase is evaluated between the sun and the refracted view ray. For
                                 a nadir camera that geometry is backscatter, where a forward lobe (g>0) is
                                 small, so 0.35 was quietly starving the sun's contribution and making the
                                 result depend on the sun azimuth. Near-isotropic is stable across the whole
                                 survey and matches the weight the ambient term already uses (1/4pi).
    Specular      0.50 -> 0.25   UE's F0 = 0.08 * Specular. Water at IOR 1.33 has F0 = 0.02, i.e. Specular
                                 0.25. 0.50 was doubling every reflection, including the sky.
    Roughness     0.06 -> 0.035  a sharper sun disc in the glint; the glitter PATH comes from wave slopes.
    RoughnessSilt 0.16 -> 0.19   scum kills the mirror where the plume is heaviest.
    NormalStrength 0.35 -> 0.50  broadens the glitter. The earlier note that 0.6 "read as large dark blobs
                                 under a low dawn sun" was measured while the surface was an opaque DIFFUSE
                                 card, where the normal modulates N.L directly; once Opacity drops to 0.03
                                 the normal drives refraction and specular instead, where the same amplitude
                                 reads as ripple rather than as blotches. Still the first knob to lower if the
                                 water looks lumpy.
    BaseColor     (0.30,0.215,0.125) -> (0.052,0.070,0.060)   these are no longer the water. They are the 3 %
    BaseColorSilt (0.46,0.355,0.225) -> (0.165,0.150,0.112)   surface film, so they must be near-black water
                                 sheen and pale silt scum, not cafe-au-lait.

The plume/streak field built by build_materials.py is NOT removed - it is promoted. It still drives BaseColor
and Roughness, and it now also drives Opacity and (when the graph can be introspected) the scattering and
absorption coefficients themselves, so a silt plume is a genuinely different WATER, not just a different tint.
"""

import json

import unreal

REPO = r"D:\Sightline"
MARK = "SIGHTLINE_TUNE_WATER"
WATER = "/Game/Sightline/Water/M_FloodWater"

mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: material recompiles and saves are unreliable while PIE runs")

w = unreal.load_asset(WATER)
if w is None:
    raise RuntimeError(f"{WATER} is missing - run tools/scene/build_materials.py first")

# --------------------------------------------------------------------------------------------------------
# new values. Vector params are RGB; scalars are floats. Units: Scattering/Absorption are 1/cm (engine header)
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# TURBIDITY: measured from the render on 2026-09-11 (utilities lane), not from the table above.
#
# At the coefficients derived in section 2 the flood renders as a FLAT, FEATURELESS GREEN FIELD from 45 m
# nadir - _artifacts/editor_shots/util_10_nadir_gsd.png, taken at the real survey GSD, is a uniform green
# with nothing but sun glint in it. Nothing of the drowned ground is readable at ANY depth, and even the
# 5-15 cm of water at the very shoreline is fully opaque (util_9_shallow.png: the bank goes brown mud ->
# teal with no submerged slope visible at all). The hue was right; the depth scale was not.
#
# Reference A (docs/SCENE_REFERENCE.md) is explicit that "depth [is] readable through it near edges". Water
# you can read depth through is water with a Secchi depth near 1 m, not the 0.35 m section 2 assumed:
#     K_d ~= 1.44 / z_SD    ->    z_SD 0.35 m -> 4.1 /m (green, as derived)
#                                 z_SD 1.05 m -> 1.37 /m
# so every extinction coefficient is divided by 3.0. Dividing Scattering AND Absorption by the SAME factor
# leaves the single-scattering albedo w = S/(S+A) unchanged in every channel, so the hue, the albedo and the
# clear/plume contrast derived above all survive untouched and only the DEPTH SCALE moves:
#     green transmittance   0.79 at 0.17 m   0.51 at 0.50 m   0.26 at 1.0 m   0.065 at 2.0 m  0.017 at 3.0 m
#     new extinction /m     clear R 2.07 G 1.37 B 1.67        plume R 2.50 G 1.67 B 2.10
# The settlement stands in 0.5-3.7 m (settlement.json), so submerged roads, kerbs, garden walls and the
# shallow margins now show through while the deep terrace still reads as opaque teal - which is the depth
# variation the reference photograph has. This is the knob the docstring's own "NOW LOOK" line names for
# this exact symptom: "if the shallows do not show the ground, lower every extinction value by the same
# factor".
# --------------------------------------------------------------------------------------------------------
TURBIDITY_DIVISOR = 3.0

VEC = {
    # 2026-09-11: re-derived from the RENDER against the reference photographs, not from a colour theory.
    # The previous pair gave albedo 0.34 / 0.62 / 0.52 - green-cyan - and once the SkyLight was fixed and
    # started delivering real ambient, `qa_4_nadir45.png` read as pale milky JADE: a swimming pool, not a
    # monsoon flood. Both photographs in docs/SCENE_REFERENCE.md show brown-olive silty water.
    # Silt is a mineral suspension: it scatters long wavelengths and absorbs short ones, so red must SURVIVE
    # and blue must die. New albedo w = S/(S+A): R 0.56, G 0.44, B 0.17.
    # Extinction is deliberately left near the old magnitude (1.8 / 1.8 / 2.4 per m after the divisor) so the
    # depth cue the utilities lane established - submerged walls readable near the margin - is preserved.
    "Scattering": (0.0300, 0.0240, 0.0120),      # extinction /m 5.4 5.4 7.2, albedo 0.56 0.44 0.17
    "Absorption": (0.0240, 0.0300, 0.0600),      # (both scaled by TURBIDITY_DIVISOR below)
    "BaseColor": (0.052, 0.070, 0.060),          # surface film only, at 3 % coverage - NOT scaled
    "BaseColorSilt": (0.165, 0.150, 0.112),
}
SCAL = {
    "PhaseG": 0.10,
    "ColorScaleBehindWater": 1.0,
    "Roughness": 0.035,
    "RoughnessSilt": 0.19,
    "Specular": 0.25,                            # F0 = 0.08 * 0.25 = 0.02 = water at IOR 1.33
    "NormalStrength": 0.50,
}
# added by this script
NEW_VEC = {
    # The silt plume is the same physics pushed further: browner still and more opaque than clear water.
    "ScatteringSilt": (0.0380, 0.0290, 0.0130),  # albedo 0.59 0.45 0.16, and more opaque than clear
    "AbsorptionSilt": (0.0260, 0.0350, 0.0680),
}
NEW_SCAL = {"WaterOpacity": 0.03, "WaterOpacitySilt": 0.22}
REQUIRED = set(VEC) | set(SCAL)

# Apply the turbidity divisor to the four optical coefficients ONLY. The BaseColor pair is a surface film,
# not a volume coefficient, and scaling it would darken the scum for no reason.
_OPTICAL = ("Scattering", "Absorption", "ScatteringSilt", "AbsorptionSilt")
for _d in (VEC, NEW_VEC):
    for _k in list(_d):
        if _k in _OPTICAL:
            _d[_k] = tuple(round(c / TURBIDITY_DIVISOR, 6) for c in _d[_k])
_ext = {n: [(VEC if n == "clear" else NEW_VEC)[f"Scattering{s}"][i]
            + (VEC if n == "clear" else NEW_VEC)[f"Absorption{s}"][i]
            for i in range(3)] for n, s in (("clear", ""), ("plume", "Silt"))}
print(f"turbidity /{TURBIDITY_DIVISOR}: extinction /m clear "
      f"R {_ext['clear'][0] * 100:.2f} G {_ext['clear'][1] * 100:.2f} B {_ext['clear'][2] * 100:.2f}; "
      f"plume R {_ext['plume'][0] * 100:.2f} G {_ext['plume'][1] * 100:.2f} B {_ext['plume'][2] * 100:.2f}")
import math as _math
for _z in (0.17, 0.5, 1.0, 2.0, 3.0):
    print(f"    green transmittance at {_z:.2f} m: {_math.exp(-_ext['clear'][1] * 100 * _z):.3f}")

# --------------------------------------------------------------------------------------------------------
# 1. find what build_materials.py left behind, and refuse to guess if it is not what we expect
# --------------------------------------------------------------------------------------------------------
def marker_of(e):
    """`UMaterialExpression::Desc` is the node comment field; it is the cheapest durable marker for "this
    script put this node here". If the Python API does not expose it, say so instead of silently letting a
    second run stack a duplicate lerp on every pin."""
    try:
        return str(e.get_editor_property("desc"))
    except Exception:                                       # noqa: BLE001
        return None


MARKABLE = True          # cleared below if ANY expression refuses to expose `desc`
params, slw, stale = {}, None, []
for e in list(mel.get_material_expressions(w)):
    m = marker_of(e)
    if m is None:
        MARKABLE = False
    if m == MARK:
        stale.append(e)
        continue
    if isinstance(e, unreal.MaterialExpressionSingleLayerWaterMaterialOutput):
        slw = e
    if isinstance(e, (unreal.MaterialExpressionVectorParameter, unreal.MaterialExpressionScalarParameter)):
        params[str(e.get_editor_property("parameter_name"))] = e
absent = sorted(REQUIRED - set(params))
if absent:
    raise RuntimeError(f"{WATER} does not carry the parameters this script tunes: {absent}. "
                       f"build_materials.py has changed - re-read it before editing these values.")
if slw is None:
    raise RuntimeError(f"{WATER} has no SingleLayerWaterMaterialOutput node; it is not a water material")
if not MARKABLE and (set(NEW_VEC) | set(NEW_SCAL)) & set(params):
    raise RuntimeError(
        "this material already carries tune_water's parameters but MaterialExpression.desc is not readable "
        "through Python in this build, so the previous run's lerp nodes cannot be identified and a second "
        "pass would stack duplicates on the Opacity and Scattering pins. Re-run build_materials.py to get a "
        "clean graph, then run this script once.")

for name, rgb in VEC.items():
    params[name].set_editor_property("default_value", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
for name, val in SCAL.items():
    params[name].set_editor_property("default_value", float(val))

# --------------------------------------------------------------------------------------------------------
# 2. recover the existing silt/plume field so the new nodes ride on it instead of duplicating it
# --------------------------------------------------------------------------------------------------------
# build_materials.py connects lerp(BaseColor, BaseColorSilt, silt_mix) to MP_BASE_COLOR, so the node on the
# BaseColor property is that lerp and its Alpha input is the plume field. Reading an ExpressionInput's
# `expression` is not a documented Python API, so it is probed rather than assumed: without it the script
# still runs, with constant coefficients and a constant surface coverage.
silt_mix, how = None, "constant (could not introspect the plume field)"
try:
    lerp = mel.get_material_property_input_node(w, unreal.MaterialProperty.MP_BASE_COLOR)
    if isinstance(lerp, unreal.MaterialExpressionLinearInterpolate):
        alpha = lerp.get_editor_property("alpha")
        cand = getattr(alpha, "expression", None)
        if cand is None and hasattr(alpha, "get_editor_property"):
            cand = alpha.get_editor_property("expression")
        if isinstance(cand, unreal.MaterialExpression):
            silt_mix, how = cand, f"plume field {cand.get_name()}"
except Exception as exc:                                    # noqa: BLE001 - a probe, not a control path
    how = f"constant (probe raised {type(exc).__name__}: {exc})"

created = []


def node(cls, x, y, **props):
    e = mel.create_material_expression(w, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    if MARKABLE:
        e.set_editor_property("desc", MARK)                 # so a re-run replaces exactly this set
    created.append(e)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect failed: {a.get_name()}.{ao} -> {b.get_name()}.{bi}")


def mix(a, b, y):
    """lerp(a, b, silt_mix), or just `a` when the plume field could not be recovered."""
    if silt_mix is None:
        return a
    e = node(unreal.MaterialExpressionLinearInterpolate, 600, y)
    link(a, "", e, "A")
    link(b, "", e, "B")
    link(silt_mix, "", e, "Alpha")
    return e


# 2a. surface coverage -> MP_Opacity. THIS is the change that switches the water volume on.
op0 = node(unreal.MaterialExpressionScalarParameter, 300, 900,
           parameter_name="WaterOpacity", default_value=NEW_SCAL["WaterOpacity"])
op1 = node(unreal.MaterialExpressionScalarParameter, 300, 1000,
           parameter_name="WaterOpacitySilt", default_value=NEW_SCAL["WaterOpacitySilt"])
mel.connect_material_property(mix(op0, op1, 950), "", unreal.MaterialProperty.MP_OPACITY)

# 2b. the plume is a different water, not just a different tint: modulate the coefficients too
sc1 = node(unreal.MaterialExpressionVectorParameter, 300, 1120, parameter_name="ScatteringSilt",
           default_value=unreal.LinearColor(*NEW_VEC["ScatteringSilt"], 1.0))
ab1 = node(unreal.MaterialExpressionVectorParameter, 300, 1240, parameter_name="AbsorptionSilt",
           default_value=unreal.LinearColor(*NEW_VEC["AbsorptionSilt"], 1.0))
link(mix(params["Scattering"], sc1, 1160), "", slw, "ScatteringCoefficients")
link(mix(params["Absorption"], ab1, 1280), "", slw, "AbsorptionCoefficients")

# 3. only now remove the previous run's nodes, so a failure above can never leave the SLW pins dangling
for e in stale:
    mel.delete_material_expression(w, e)

mel.recompile_material(w)
eal.save_asset(w.get_path_name())

s = mel.get_statistics(w)
if s.num_pixel_shader_instructions == 0:
    raise RuntimeError("M_FloodWater FAILED TO COMPILE after tuning (0 instructions). Nothing else in the "
                       "scene changed; revert by re-running build_materials.py.")

meta = json.loads(open(REPO + r"\data\scene\flood_valley.json").read())
print(f"M_FloodWater retuned: {len(created)} nodes added, {len(stale)} from a previous run removed, "
      f"{mel.get_num_material_expressions(w)} expressions total")
print(f"  silt modulation: {how}")
print(f"  Opacity connected -> WaterVisibility = 1 - Opacity is now {1 - NEW_SCAL['WaterOpacity']:.2f} "
      f"instead of 0.00 (the volume was previously multiplied away entirely)")
# Derived from the values this run actually wrote, NOT hard-coded. The previous version of these two lines
# printed the pre-TURBIDITY_DIVISOR figures as literals, so after any change to the coefficients the script
# confidently reported numbers that were no longer true of the material it had just written.
_alb = [VEC["Scattering"][i] / (VEC["Scattering"][i] + VEC["Absorption"][i]) for i in range(3)]
print(f"  extinction /m: clear R {_ext['clear'][0] * 100:.2f} G {_ext['clear'][1] * 100:.2f} "
      f"B {_ext['clear'][2] * 100:.2f} (albedo {_alb[0]:.2f}/{_alb[1]:.2f}/{_alb[2]:.2f}), "
      f"silt R {_ext['plume'][0] * 100:.2f} G {_ext['plume'][1] * 100:.2f} B {_ext['plume'][2] * 100:.2f}; "
      f"green transmittance {_math.exp(-_ext['clear'][1] * 100 * 0.5):.3f} at 0.50 m, "
      f"{_math.exp(-_ext['clear'][1] * 100 * 1.0):.3f} at 1.0 m "
      f"(turbidity divisor {TURBIDITY_DIVISOR})")
print(f"  compiles: {s.num_pixel_shader_instructions} instructions, {s.num_pixel_texture_samples} "
      f"texture samples")
print(f"  flood surface sits at z = {meta['ue_import']['flood_water_z_cm']:.1f} cm")
print("NOW LOOK: run tools/scene/qa_shots.py. Expect green-teal water that goes brown-through-clear at the "
      "shallow margins. If it is too dark, raise Scattering (all three channels together); if the hue is too "
      "green, raise Absorption.G; if the shallows do not show the ground, lower every extinction value "
      "(Scattering + Absorption) by the same factor. This script must be re-run after build_materials.py.")
