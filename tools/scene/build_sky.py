"""Give FloodValley a VISIBLE sky that is the same sky that lights it (running editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_sky.py').read())"
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_sky.py').read())"     # render
    uv run python tools/scene/check_sky.py                                            # the check that can fail

THE DEFECT
----------
The level is lit by a Poly Haven CC0 **overcast** HDRI (`overcast_soil_puresky_2k`), chosen deliberately:
both reference photographs are diffuse overcast, and an overcast sky fills shadows honestly (that HDRI is
what took `qa_2_settlement` from 239:1 down to 17.3:1). But the HDRI only ever drove the SkyLight's
*ambient*. What the camera actually SAW was the `SkyAtmosphere`'s own Rayleigh sky: a flat navy gradient.

Measured on `_artifacts/editor_shots/qa_1_valley.png` before this script existed:

    visible sky   mean linear RGB (0.023, 0.100, 0.248)   B/R = 10.8
    driving HDRI  mean linear RGB (1.445, 1.429, 1.433)   B/R =  0.99   (upper hemisphere)

So the sky the camera saw was **11x more blue-over-red than the sky doing the lighting**. That is not a
matter of taste: a clear-sky background over overcast ambient is internally inconsistent, and it is wrong in
every oblique frame the dataset captures.

THE FIX, AND WHY IT CANNOT DISTURB THE LIGHTING
-----------------------------------------------
An unlit sky-dome mesh sampling the SAME TextureCube with the view ray, so the background IS the light probe.

The SkyLight is untouched, and that is enforced rather than hoped for: this script copies its five
load-bearing values out BEFORE it does anything (`get_editor_property` hands back a live reference, so a
"before" reading taken late is really an "after" reading), and asserts them again at the end. In particular
`real_time_capture` must stay **False** - True makes UE re-capture the scene every frame and ignore the
specified cubemap, which was the original bug.

The dome's only tunable is `SkyBrightness`, a SCALAR that multiplies all three channels equally. It can
therefore change how bright the sky reads but it **cannot change the sky's colour**, so no amount of
tuning it can break the sky-matches-light property that `check_sky.py` tests. That is the point of making
it a scalar rather than a tint.

THREE THINGS THAT WOULD FAIL SILENTLY
-------------------------------------
1. **`is_sky` must be True on the material.** Without it the dome is treated as ordinary geometry and the
   SkyAtmosphere's aerial perspective paints ~40 km of atmosphere over it - which would put the navy
   straight back, on top of the HDRI, and it would look like the script simply had no effect. `is_sky` is
   only honoured on Unlit + Opaque materials, so the shading model and blend mode are asserted too.
2. **A translucent or lit sky material still renders**, just wrongly, so a compile check is not enough:
   `MaterialEditingLibrary.get_statistics` is asserted non-zero for instructions AND texture samples,
   because a sky dome that samples no texture is a flat colour that would still pass "it compiled".
3. **The cubemap must be a TextureCube**, not a Texture2D. A Texture2D assigned to a cube sampler makes the
   material fail to compile, and a material that fails to compile renders as the grey checker - which at
   sky scale would read as "overcast" to a careless eye. Asserted by type.

WHAT THIS ADDS TO THE SCENE, HONESTLY
-------------------------------------
One actor, one 40 km sphere, one unlit material with one cubemap fetch. It casts no shadow, affects no
distance fields and no indirect lighting. It costs no VRAM beyond the 2k cubemap that was already resident
for the SkyLight. But it IS geometry where there used to be none, so it now has a depth value (~40 km) and
could take an instance-segmentation id; `render_custom_depth` is explicitly forced off here so Cosys does
not stencil it, and that is flagged in TRACKER for the dataset lane to confirm in a real capture.
"""

import ctypes
import time

import unreal

# --- tunables ------------------------------------------------------------------------------------------
CUBE = "/Game/Sightline/Sky/overcast_soil_puresky_2k"
MAT_PATH = "/Game/Sightline/Sky/M_SkyDome"
DOME_NAME = "SkyDome"
DOME_RADIUS_CM = 4_000_000.0        # 40 km: encloses the 2 km valley and the 450 m demo camera with margin
SKY_BRIGHTNESS = 1.0                # scalar only - see the docstring; tuned by measurement in check_sky.py
DOME_MESH_CANDIDATES = ("/Engine/EngineSky/SM_SkySphere", "/Engine/BasicShapes/Sphere",
                        "/Engine/EditorMeshes/EditorSphere")

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


# --- memory (the editor has already died once at 27.49 GiB of commit) ------------------------------------
class _MEMSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem():
    G = 1024.0 ** 3
    ms = _MEMSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return {"AvailablePhysical": ms.ullAvailPhys / G, "AvailableVirtual": ms.ullAvailPageFile / G,
            "CommitLimit": ms.ullTotalPageFile / G}


def memline(tag):
    m = mem()
    print(f"  [mem] {tag:24s} AvailablePhysical {m['AvailablePhysical']:.2f} GiB  "
          f"AvailableVirtual {m['AvailableVirtual']:.2f} GiB  (commit limit {m['CommitLimit']:.1f} GiB)")
    return m


if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first - imports, saves and material edits fail silently while PIE runs")

world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
print(f"build_sky.py on {world.get_name()} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
m0 = memline("before")

# --- 0. copy the SkyLight's load-bearing values OUT, before touching anything ----------------------------
# docs/CONTEXT.md: get_editor_property returns a REFERENCE to a live struct, so a "before" reading taken
# after a write silently becomes the "after" reading. These five are scalars/enums/objects rather than
# structs, but the same discipline is applied because the whole point is to prove they did not move.
skylights = [a for a in eas.get_all_level_actors() if isinstance(a, unreal.SkyLight)]
if len(skylights) != 1:
    raise RuntimeError(f"expected exactly 1 SkyLight in the level, found {len(skylights)}")
slc = skylights[0].get_editor_property("light_component")      # NOT sky_light_component - verified in stub


def skylight_state():
    cm = slc.get_editor_property("cubemap")
    return {
        "source_type": str(slc.get_editor_property("source_type")),
        "cubemap": None if cm is None else cm.get_path_name(),
        "real_time_capture": bool(slc.get_editor_property("real_time_capture")),
        "lower_hemisphere_is_black": bool(slc.get_editor_property("lower_hemisphere_is_black")),
        "intensity": float(slc.get_editor_property("intensity")),
    }


BEFORE = skylight_state()
print("  SkyLight BEFORE (must be identical at the end):")
for k, v in BEFORE.items():
    print(f"    {k:28s} {v}")

# --- 1. the cubemap ---------------------------------------------------------------------------------------
cube = unreal.load_asset(CUBE)
if cube is None:
    raise RuntimeError(f"{CUBE} is not in the project. The SkyLight cannot be driving it either.")
if not isinstance(cube, unreal.TextureCube):
    raise RuntimeError(f"{CUBE} is a {type(cube).__name__}, not a TextureCube. A Texture2D in a cube sampler "
                       f"makes the material fail to compile and render as the grey checker.")
if BEFORE["cubemap"] != cube.get_path_name():
    print(f"  !! the SkyLight is driven by {BEFORE['cubemap']}, not {cube.get_path_name()} - the visible sky "
          f"is about to be built from a DIFFERENT sky than the one lighting the scene")
print(f"  cubemap {cube.get_name()}: TextureCube, srgb={cube.get_editor_property('srgb')}, "
      f"compression={cube.get_editor_property('compression_settings')}")

# --- 2. the sky material ----------------------------------------------------------------------------------
pkg, name = MAT_PATH.rsplit("/", 1)
if eal.does_asset_exist(MAT_PATH):
    mat = unreal.load_asset(MAT_PATH)
    # Reuse rather than delete: deleting a referenced material fails with an ensure + callstack
    # (docs/CONTEXT.md section 7). Clear its graph instead, and assert the clear actually took -
    # delete_all_material_expressions is documented in CONTEXT as leaving nodes behind.
    for e in list(mel.get_material_expressions(mat)):
        mel.delete_material_expression(mat, e)
    left = len(mel.get_material_expressions(mat))
    if left:
        raise RuntimeError(f"{MAT_PATH} still has {left} expressions after deleting them all")
else:
    mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, pkg, unreal.Material, unreal.MaterialFactoryNew())

mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)     # BLEND_, not BM_ (CONTEXT section 7)
mat.set_editor_property("two_sided", True)                               # the camera is INSIDE the dome
mat.set_editor_property("is_sky", True)                                  # no aerial perspective on the sky

# view ray = -CameraVectorWS. CameraVectorWS points FROM the shaded pixel TO the camera, so the direction
# the camera is actually looking is its negative; that is the vector a sky cubemap must be sampled with.
cam = mel.create_material_expression(mat, unreal.MaterialExpressionCameraVectorWS, -900, 0)
neg = mel.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -900, 180)
neg.set_editor_property("constant", unreal.LinearColor(-1.0, -1.0, -1.0, 1.0))
ray = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -650, 60)

samp = mel.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameterCube, -420, 0)
samp.set_editor_property("parameter_name", "SkyCube")
samp.set_editor_property("texture", cube)
# The HDRI is linear-light data, not an sRGB-encoded colour texture. SAMPLERTYPE_COLOR would apply an
# sRGB->linear decode to values that are already linear and darken the sky non-uniformly.
samp.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)

bright = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -420, 260)
bright.set_editor_property("parameter_name", "SkyBrightness")
bright.set_editor_property("default_value", SKY_BRIGHTNESS)
out = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -180, 60)

def wire(src, out_name, dst, candidates):
    """Connect by a pin name RESOLVED from the engine, not guessed.

    The first version of this script wired the cube sampler's UV input as "Coordinates" - which is the C++
    member name and reads perfectly plausibly - and `connect_material_expressions` simply returned False.
    The pin is actually called "UVs". A returned False is easy to miss in a list of six connections, and an
    unwired sky dome renders solid black, so the name is looked up and the result is asserted per wire.
    """
    names = list(mel.get_material_expression_input_names(dst))
    for c in candidates:
        if c in names:
            if not mel.connect_material_expressions(src, out_name, dst, c):
                raise RuntimeError(f"connecting {type(src).__name__}.{out_name!r} -> "
                                   f"{type(dst).__name__}.{c!r} returned False")
            return c
    raise RuntimeError(f"none of {candidates} is an input of {type(dst).__name__}; it has {names}")


pins = [
    wire(cam, "", ray, ["A"]),
    wire(neg, "", ray, ["B"]),
    wire(ray, "", samp, ["UVs", "Coordinates"]),
    wire(samp, "RGB", out, ["A"]),
    wire(bright, "", out, ["B"]),
]
if not mel.connect_material_property(out, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR):
    raise RuntimeError("connecting the sky graph to MP_EMISSIVE_COLOR returned False - the dome would be "
                       "black, and black is not obviously wrong at a glance in a night-time frame")
print(f"  wired pins: {pins} -> MP_EMISSIVE_COLOR")

mel.recompile_material(mat)
eal.save_asset(MAT_PATH)

st = mel.get_statistics(mat)
if st.num_pixel_shader_instructions == 0:
    raise RuntimeError(f"{MAT_PATH} FAILED TO COMPILE (0 instructions) - it would render as the grey checker")
if st.num_pixel_texture_samples == 0:
    raise RuntimeError(f"{MAT_PATH} compiles but samples NO texture - the sky would be a flat colour, which "
                       f"is exactly the failure this script exists to remove")
# Prove the three flags survived the save, by reloading rather than trusting the setters.
rl = unreal.load_asset(MAT_PATH)
flags = {"is_sky": bool(rl.get_editor_property("is_sky")),
         "shading_model": str(rl.get_editor_property("shading_model")),
         "blend_mode": str(rl.get_editor_property("blend_mode")),
         "two_sided": bool(rl.get_editor_property("two_sided"))}
if not flags["is_sky"]:
    raise RuntimeError("is_sky did not stick; the SkyAtmosphere would paint 40 km of aerial perspective "
                       "over the dome and the navy would come straight back")
if "MSM_UNLIT" not in flags["shading_model"] or "BLEND_OPAQUE" not in flags["blend_mode"]:
    raise RuntimeError(f"is_sky is only honoured on Unlit+Opaque materials, and this is {flags}")
print(f"  {MAT_PATH}: {st.num_pixel_shader_instructions} instructions, "
      f"{st.num_pixel_texture_samples} texture samples, {flags}")

# --- 3. the dome mesh -------------------------------------------------------------------------------------
dome_mesh, mesh_path = None, None
for p in DOME_MESH_CANDIDATES:
    a = unreal.load_asset(p)
    if isinstance(a, unreal.StaticMesh):
        dome_mesh, mesh_path = a, p
        break
if dome_mesh is None:
    raise RuntimeError(f"no sphere mesh found among {DOME_MESH_CANDIDATES}")
bb = dome_mesh.get_bounding_box()
mesh_radius = max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z) / 2.0
if mesh_radius <= 0:
    raise RuntimeError(f"{mesh_path} has zero extent")
scale = DOME_RADIUS_CM / mesh_radius
print(f"  dome mesh {mesh_path}: radius {mesh_radius:.0f} cm -> scale {scale:.1f} "
      f"= {DOME_RADIUS_CM / 100000:.0f} km radius")

# --- 4. the dome actor ------------------------------------------------------------------------------------
existing = [a for a in eas.get_all_level_actors() if a.get_name() == DOME_NAME or a.get_actor_label() == DOME_NAME]
for a in existing[1:]:
    eas.destroy_actor(a)
if existing:
    dome = existing[0]
    print(f"  reusing existing {DOME_NAME}")
else:
    dome = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
    dome.set_actor_label(DOME_NAME)
    dome.rename(DOME_NAME)        # Cosys names objects by GetName(), not the editor label (HANDBOOK section 5)
    print(f"  spawned {DOME_NAME}")

comp = dome.get_editor_property("static_mesh_component")
comp.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
comp.set_editor_property("static_mesh", dome_mesh)
comp.set_material(0, mat)
comp.set_editor_property("cast_shadow", False)                      # a 40 km shadow caster would be absurd
comp.set_editor_property("affect_distance_field_lighting", False)
comp.set_editor_property("affect_dynamic_indirect_lighting", False)
comp.set_editor_property("render_custom_depth", False)              # keep it out of Cosys' stencil pass
comp.set_editor_property("cast_shadow_as_two_sided", False)
dome.set_actor_location(unreal.Vector(0, 0, 0), False, False)
dome.set_actor_scale3d(unreal.Vector(scale, scale, scale))
dome.set_folder_path("Sky")
try:
    comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
except Exception as exc:                                            # noqa: BLE001 - report, do not hide
    print(f"  !! could not clear collision on the dome: {exc}")

# --- 4b. stop the height fog from repainting the dome navy -------------------------------------------------
# The dome alone did NOT fix the sky, and the reason is worth writing down because it is not guessable.
# `is_sky` removes AERIAL PERSPECTIVE from the dome but, in the engine's own words, "Height and Volumetric
# fog effects will still be applied". This level's ExponentialHeightFog has fog_inscattering_luminance
# (0,0,0) with sky_atmosphere_ambient_contribution_color_scale 1.0, so it takes its colour FROM the
# SkyAtmosphere - i.e. navy - and at the dome's 40 km depth the fog is fully saturated, so it replaced the
# dome completely everywhere except a thin strip near the zenith.
#
# Isolated by experiment rather than by reasoning, in the same 450 m frame, band y60-190 (domain=sim):
#     dome + fog + atmosphere      RGB (0.021, 0.108, 0.268)   B/R 12.81   <- the navy
#     dome + atmosphere, fog OFF   RGB (0.511, 0.510, 0.505)   B/R  0.99   <- the HDRI, exactly
#     dome, fog OFF, atmosphere OFF  identical to the line above           <- atmosphere adds nothing now
#     dome + fog, atmosphere OFF   RGB (0.000, 0.000, 0.000)   B/R  1.00   <- fog with no colour source
# The third line is what proves the dome is doing the work and the second isolates the fog as the cause.
#
# The fix is a distance cutoff rather than a colour change: every piece of real geometry in this level is
# within ~3 km, and the dome is the ONLY thing beyond that, so cutting fog off at 20 km leaves the terrain's
# aerial haze exactly as the other lanes tuned it and removes fog only from the sky. A colour change would
# have altered every distant terrain pixel too, which is not this lane's to move.
fogs = [a for a in eas.get_all_level_actors() if isinstance(a, unreal.ExponentialHeightFog)]
if len(fogs) != 1:
    raise RuntimeError(f"expected exactly 1 ExponentialHeightFog, found {len(fogs)}")
fogc = fogs[0].get_editor_property("component")
FOG_CUTOFF_CM = 2_000_000.0        # 20 km: >> the 3 km scene, << the 40 km dome
fog_before = float(fogc.get_editor_property("fog_cutoff_distance"))
fogc.set_editor_property("fog_cutoff_distance", FOG_CUTOFF_CM)
fog_after = float(unreal.load_object(None, fogs[0].get_path_name()).get_editor_property("component")
                  .get_editor_property("fog_cutoff_distance"))
if fog_after != FOG_CUTOFF_CM:
    raise RuntimeError(f"fog_cutoff_distance did not take: wanted {FOG_CUTOFF_CM}, got {fog_after}")
if DOME_RADIUS_CM <= FOG_CUTOFF_CM:
    raise RuntimeError(f"the dome at {DOME_RADIUS_CM:.0f} cm is INSIDE the fog cutoff {FOG_CUTOFF_CM:.0f} cm, "
                       f"so the fog will still repaint it")
print(f"  fog_cutoff_distance {fog_before:.0f} -> {fog_after:.0f} cm ({fog_after / 100000:.0f} km); dome sits "
      f"at {DOME_RADIUS_CM / 100000:.0f} km, scene geometry within ~3 km, so only the sky loses its fog")

# --- 5. prove the SkyLight did not move -------------------------------------------------------------------
AFTER = skylight_state()
drift = {k: (BEFORE[k], AFTER[k]) for k in BEFORE if BEFORE[k] != AFTER[k]}
if drift:
    raise RuntimeError(f"the SkyLight CHANGED while building the sky dome: {drift}. Its settings were "
                       f"derived by measurement and are load-bearing; nothing here may touch them.")
print("  SkyLight AFTER: identical on all five load-bearing values (real_time_capture still "
      f"{AFTER['real_time_capture']}, intensity {AFTER['intensity']}, {AFTER['source_type']})")

# --- 6. save ----------------------------------------------------------------------------------------------
if not les.save_current_level():
    raise RuntimeError("save_current_level() returned False - the dome is NOT persisted")
m1 = memline("after")
print(f"\nsky dome built: {DOME_NAME} r={DOME_RADIUS_CM / 100000:.0f} km, {MAT_PATH}, SkyBrightness="
      f"{SKY_BRIGHTNESS}, cube={cube.get_name()}")
print(f"  memory delta AvailablePhysical {m1['AvailablePhysical'] - m0['AvailablePhysical']:+.2f} GiB, "
      f"AvailableVirtual {m1['AvailableVirtual'] - m0['AvailableVirtual']:+.2f} GiB")
print("NOW RENDER AND LOOK:  ue_python exec tools/scene/qa_sky.py   then read the PNGs")
