"""Take the fluorescent green out of M_FloodValleyTerrain and break up its flatness (editor, PIE OFF).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\tune_terrain.py').read())"

`build_materials.py` OWNS this material and rebuilds its graph from scratch, so - exactly like
`tune_water.py` and `build_roof_materials.py` - this script only edits parameter defaults and adds a small,
clearly-marked set of nodes on top, and **it must be re-run every time build_materials.py runs**. Every node
it creates carries `desc = "SIGHTLINE_TUNE_TERRAIN"`, and a re-run replaces exactly that set.

================================================================================================
1. THE GRASS IS ABOUT TWICE AS SATURATED AS VEGETATION CAN BE
================================================================================================
Measured off `_artifacts/editor_shots/qa_1_valley.png` (2026-09-11 00:57:50), sunlit hillside:

    rendered linear   R 0.054  G 0.159  B 0.032      G/R 2.95   G/B 5.00

Real vegetation seen from the air sits at roughly R 0.06-0.09, G 0.09-0.13, B 0.04-0.06 - G/R about 1.3-1.5
and G/B about 2.0-2.5 - and atmospheric haze only ever pushes those ratios DOWN, never up. The valley reads
as a golf course, and no amount of vegetation placed on top fixes the ground colour between the trees.

The cause is in the material, not the lighting. `build_materials.py` measured leafy_grass's linear albedo as
(0.324, 0.239, 0.109) - a dry, yellow-brown scan - and corrected it with

    GrassTint = (0.35, 1.00, 0.45)   ->  product (0.113, 0.239, 0.049)   G/R 2.11  G/B 4.87

which inverts the R > G ordering as intended but overshoots badly: it does not merely make the scan green, it
makes it more green than vegetation gets. Rendering adds a little more on top (2.95 measured against 2.11 in
the material), so the tint has to land comfortably inside the real range rather than at its edge:

    GrassTint = (0.26, 0.50, 0.50)   ->  product (0.084, 0.120, 0.055)   G/R 1.42  G/B 2.19

That is the same overall brightness, in the middle of the measured range for aerial vegetation.

================================================================================================
2. THE WHOLE 4.2 km2 IS ONE FLAT COLOUR
================================================================================================
The graph already computes a smooth macro-noise field (`blend_w`, ~40 m blobs off a mip-7 sample) and the
comment says it drives "the anti-tiling blend weight AND a large-scale brightness variation" - but it is only
ever used as the anti-tiling lerp alpha. Nothing varies the colour at large scale, which is why the hillsides
read as a single painted surface at survey altitude.

This adds that variation as a separate, coarser field: a procedural Noise at roughly 130 m, remapped to
multiply base colour by 1 +- MACRO_AMP. Real hillsides vary far more than this between wet hollows and dry
spurs; +-18 % is deliberately conservative because it multiplies EVERY training frame and an over-strong
field would show up as blotches the detector could learn.

Nothing here touches the normal, the roughness, the zone masks, the silt line or the layer blends.
"""

from __future__ import annotations

import unreal

TERRAIN = "/Game/Sightline/Terrain/M_FloodValleyTerrain"
MARK = "SIGHTLINE_TUNE_TERRAIN"

#: linear albedo leafy_grass actually scans at, measured by build_materials.py
GRASS_ALBEDO = (0.324, 0.239, 0.109)
NEW_GRASS_TINT = (0.26, 0.50, 0.50)
MACRO_AMP = 0.18
MACRO_SCALE_M = 130.0
#: 0 keeps the colour, 1 is greyscale. 0.22 takes the Mars-red out of the laterite without
#: greying the scene, and matches the direction aerial haze pushes real terrain.
DESATURATE = 0.22

mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary

m = unreal.load_asset(TERRAIN)
if m is None:
    raise RuntimeError(f"{TERRAIN} not found - run build_materials.py first")

before = mel.get_statistics(m)
print(f"before: {before.num_pixel_shader_instructions} instructions, "
      f"{before.num_samplers} samplers, {mel.get_num_material_expressions(m)} expressions")


def marker_of(e):
    """`desc` is the node comment; it is the cheapest durable marker for "this script put this here"."""
    try:
        return str(e.get_editor_property("desc"))
    except Exception:                                        # noqa: BLE001
        return None


MARKABLE = True
params, stale = {}, []
for e in list(mel.get_material_expressions(m)):
    mk = marker_of(e)
    if mk is None:
        MARKABLE = False
    if mk == MARK:
        stale.append(e)
        continue
    if isinstance(e, (unreal.MaterialExpressionVectorParameter,
                      unreal.MaterialExpressionScalarParameter)):
        params[str(e.get_editor_property("parameter_name"))] = e

if "GrassTint" not in params:
    raise RuntimeError(f"{TERRAIN} has no GrassTint parameter; build_materials.py has changed - re-read it "
                       f"before editing. Parameters present: {sorted(params)}")
if stale and not MARKABLE:
    raise RuntimeError("this material carries nodes from a previous run but MaterialExpression.desc is not "
                       "readable in this build, so they cannot be identified and a second pass would stack "
                       "duplicate multiplies on BaseColor. Re-run build_materials.py, then run this once.")

# --- 1. the tint ---------------------------------------------------------------------------------------
gt = params["GrassTint"]
# COPY the floats out before writing. `get_editor_property` hands back a reference to the live struct, so
# holding on to it and then setting the property mutates the very object you saved - the "before" value
# silently becomes the "after" value and the script reports `0.26,0.50,0.50 -> 0.26,0.50,0.50`, which is a
# lie that looks like a no-op.
_o = gt.get_editor_property("default_value")
old = (float(_o.r), float(_o.g), float(_o.b))
gt.set_editor_property("default_value",
                       unreal.LinearColor(NEW_GRASS_TINT[0], NEW_GRASS_TINT[1], NEW_GRASS_TINT[2], 1.0))


def ratios(tint):
    p = [GRASS_ALBEDO[i] * tint[i] for i in range(3)]
    return p, p[1] / max(p[0], 1e-9), p[1] / max(p[2], 1e-9)


p_old, gr_old, gb_old = ratios(old)
p_new, gr_new, gb_new = ratios(NEW_GRASS_TINT)
print(f"GrassTint {old[0]:.2f},{old[1]:.2f},{old[2]:.2f} -> "
      f"{NEW_GRASS_TINT[0]:.2f},{NEW_GRASS_TINT[1]:.2f},{NEW_GRASS_TINT[2]:.2f}")
print(f"  grass linear was ({p_old[0]:.3f},{p_old[1]:.3f},{p_old[2]:.3f})  G/R {gr_old:.2f}  G/B {gb_old:.2f}")
print(f"  grass linear now ({p_new[0]:.3f},{p_new[1]:.3f},{p_new[2]:.3f})  G/R {gr_new:.2f}  G/B {gb_new:.2f}")
print("  real aerial vegetation: G/R 1.3-1.5, G/B 2.0-2.5")

# --- 2. remember what BaseColor was ORIGINALLY fed by, before deleting anything -------------------------
# A re-run deletes this script's own nodes, and one of them is the multiply feeding BaseColor - so after the
# delete, `get_material_property_input_node` returns None and the script cannot find its way back to the
# graph build_materials.py wrote. The original source node's NAME is therefore recorded in a sidecar the
# first time and read back on every re-run. Reading an ExpressionInput's `expression` would avoid this, but
# that is not a documented Python API (tune_water.py had to probe for it and it raised).
import os
# NOT `__file__`: these scripts are run with `exec(open(...).read())` inside the editor, where __file__ is
# undefined and the NameError aborts the whole tune.
_side = r"D:/Sightline/_artifacts/scene_terrain_basecolor_src.txt"
_cur0 = mel.get_material_property_input_node(m, unreal.MaterialProperty.MP_BASE_COLOR)
if _cur0 is not None and marker_of(_cur0) != MARK:
    os.makedirs(os.path.dirname(_side), exist_ok=True)
    with open(_side, "w", encoding="utf-8") as fh:
        fh.write(_cur0.get_name())
    print(f"  BaseColor source recorded: {_cur0.get_name()}")

for e in stale:
    mel.delete_material_expression(m, e)
if stale:
    print(f"removed {len(stale)} node(s) from a previous run")

created = []


def node(cls, x, y, **props):
    e = mel.create_material_expression(m, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    if MARKABLE:
        e.set_editor_property("desc", MARK)
    created.append(e)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect failed: {a.get_name()}.{ao} -> {b.get_name()}.{bi}")


cur = mel.get_material_property_input_node(m, unreal.MaterialProperty.MP_BASE_COLOR)
if cur is None or marker_of(cur) == MARK:
    if not os.path.exists(_side):
        raise RuntimeError("BaseColor has nothing connected and no recorded source. Re-run "
                           "build_materials.py to get a clean graph, then run this once.")
    want = open(_side, encoding="utf-8").read().strip()
    cur = next((e for e in mel.get_material_expressions(m) if e.get_name() == want), None)
    if cur is None:
        raise RuntimeError(f"the recorded BaseColor source {want!r} is no longer in the graph. Re-run "
                           f"build_materials.py, then run this once.")
    print(f"  reconnecting to the recorded BaseColor source: {want}")

# The Noise node's Position pin is OPTIONAL: left unconnected it uses absolute world position, which is
# exactly what a macro terrain field wants. Connecting a WorldPosition node to it is refused
# (`connect failed: MaterialExpressionWorldPosition -> MaterialExpressionNoise.Position`), so don't.
nz = node(unreal.MaterialExpressionNoise, -650, 1400,
          # Verified against sim/SightlineSim/Intermediate/PythonStub/unreal.py: the property is
          # `noise_function`, NOT `function`. Guessing it would have raised in the editor exactly the way
          # `unreal.BlendMode.BM_OPAQUE` did - the stub file is the cheapest way to check a UE API name.
          scale=1.0 / (MACRO_SCALE_M * 100.0), quality=1, levels=2, output_min=-1.0, output_max=1.0,
          noise_function=unreal.NoiseFunction.NOISEFUNCTION_SIMPLEX_TEX)
amp = node(unreal.MaterialExpressionMultiply, -450, 1400, const_b=MACRO_AMP)
link(nz, "", amp, "A")
one = node(unreal.MaterialExpressionAdd, -300, 1400, const_b=1.0)
link(amp, "", one, "A")
mul = node(unreal.MaterialExpressionMultiply, -150, 1200)
link(cur, "", mul, "A")
link(one, "", mul, "B")

# --- 2b. pull the saturation down ------------------------------------------------------------------------
# The steep-slope laterite layer reads as Mars-orange at valley scale (qa_1_valley.png, 2026-09-11 01:31)
# and the red team independently flagged the terrain as over-saturated. The laterite scan is genuinely that
# red; rather than add a per-layer tint to a graph another script owns, the whole terrain base colour is
# desaturated slightly. Aerial haze desaturates real terrain anyway, so this is the physically right
# direction as well as the one that touches the fewest nodes. Fraction 0 keeps the colour, 1 is greyscale.
sat = node(unreal.MaterialExpressionDesaturation, 0, 1200)
link(mul, "", sat, "")
frac = node(unreal.MaterialExpressionConstant, -150, 1330, r=DESATURATE)
link(frac, "", sat, "Fraction")
mel.connect_material_property(sat, "", unreal.MaterialProperty.MP_BASE_COLOR)
print(f"macro colour variation: 1 +- {MACRO_AMP:.2f} at ~{MACRO_SCALE_M:.0f} m, "
      f"{len(created)} node(s) added")

# --- 3. it has to COMPILE, and a failed compile reports zero instructions and nothing else ----------------
mel.recompile_material(m)
after = mel.get_statistics(m)
if after.num_pixel_shader_instructions <= 0:
    raise RuntimeError("M_FloodValleyTerrain FAILED TO COMPILE after tuning - it will render as the grey "
                       "WorldGridMaterial checker. Re-run build_materials.py to recover a clean graph.")
eal.save_asset(m.get_path_name())
print(f"after:  {after.num_pixel_shader_instructions} instructions, {after.num_samplers} samplers, "
      f"{mel.get_num_material_expressions(m)} expressions")
print("\nNOW LOOK: render qa_1_valley and qa_4_nadir45 and compare against docs/SCENE_REFERENCE.md. The "
      "hillsides should read drab olive with visible large-scale variation, not one flat green. If they are "
      "still too green, lower GrassTint.G; if they have gone grey, raise it. Re-run after build_materials.py.")
