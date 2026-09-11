"""Put sky light back into the shadows without turning GI on (running editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\tune_lighting.py').read())"
    uv run python tools/scene/measure_lighting.py _artifacts/editor_shots/qa_*.png   # then RE-MEASURE

================================================================================================
THE DEFECT, MEASURED
================================================================================================
`tools/scene/measure_lighting.py` on the 2026-09-11 00:57 renders:

    qa_1_valley.png      sun:shadow  26.8 : 1     3.3 % of terrain below sRGB 0.05
    qa_4_nadir45.png     sun:shadow  45.0 : 1     8.1 %      <- the survey altitude
    qa_2_settlement.png  sun:shadow 239.5 : 1    29.0 %

Outdoors under a clear sky the sky supplies 15-25 % of horizontal illuminance, so a shadow sits about 5:1 to
10:1 below full sun. 239:1 is not a shadow, it is a hole. And in every one of those renders the shadow's
BLUE channel is its LOWEST, when a sky-lit shadow is blue-dominated - so the shadows are not being lit by a
sky at all.

This is not a cosmetic complaint. A survivor lying in a shadow crushed to black is invisible to the detector
and to the human reviewing the frame, while `actors.json` still records them as `occlusion: 0` and the
evaluation still counts them as a miss. It costs recall in a slice nobody thinks to look at.

================================================================================================
WHY IT HAPPENS, AND WHY THE FIX IS FREE
================================================================================================
FloodValley runs with dynamic GI OFF on purpose - `r.DynamicGlobalIlluminationMethod=0` plus a `PPV_NoGI`
post-process volume - because the machine has an 8 GB laptop GPU (docs/CONTEXT.md, doc section 4). That is
the right call and this script does NOT change it.

But with GI off, the **SkyLight is the entire ambient term**. Nothing else fills a shadow. A SkyLight is not
GI: it is a single ambient cubemap lookup, it costs effectively nothing at runtime, and it works perfectly
well with Lumen disabled. So the shadows can be fixed without spending a byte of the VRAM budget.

Two things are done here, both physical:

1. **Intensity.** Raised so the sun:shadow ratio lands in the 5-10:1 band. From `ratio = D/A + 1`, moving
   45:1 to 8:1 needs the ambient about 6x stronger, which is where SKY_GAIN starts. It is a first
   iteration, not a final answer - re-render, re-measure, adjust.
2. **The lower hemisphere.** `lower_hemisphere_is_black = True` zeroes everything arriving from below the
   horizon. With GI on, bounce off the ground would supply that. With GI off, nothing does, which is exactly
   why these shadows are neutral-dark instead of sky-blue. Setting a non-black lower-hemisphere colour is
   the standard, cheap stand-in for the missing ground bounce, and it is physically motivated: the terrain
   reflects roughly 10-20 % of what lands on it.

If the SkyLight is capturing the scene and `BP_Sky_Sphere` is hidden (it is, in this level), the capture can
be of a black sky - which would explain the missing blue directly. The script reports what it finds instead
of assuming, and recaptures afterwards.
"""

from __future__ import annotations

import unreal

#: First iteration, derived from the 45:1 measured at the survey altitude: ratio = D/A + 1, so 45 -> 8 needs
#: about 6x the ambient. Re-measure after rendering and adjust rather than trusting this number.
# Measured in the editor 2026-09-11 01:24: SkyLight intensity 1.0, source SLS_CAPTURED_SCENE,
# real_time_capture True, cubemap None, lower_hemisphere_is_black True - against a Sun at intensity 8.0.
# The ambient was not WEAK, it was ABSENT: a captured-scene SkyLight in a level whose BP_Sky_Sphere is
# hidden captures black. So the fix is the cubemap, and the gain starts at 1.0 - sky 1.0 against sun 8.0
# lands near 9:1, inside the 5-10:1 target, and multiplying it up would only flatten the scene.
SKY_GAIN = 1.0
#: Ground bounce stand-in. Terrain linear albedo measured at roughly (0.09, 0.12, 0.05); this is that,
#: scaled down, so the fill reads as light off wet ground rather than as a second sky.
LOWER_HEMI = (0.055, 0.070, 0.045)

#: Poly Haven CC0 HDRIs, downloaded and md5-verified by hand into _downloads/assets/polyhaven/hdri/.
#: `overcast_soil_puresky` is chosen because BOTH reference photographs in docs/SCENE_REFERENCE.md are lit
#: by diffuse overcast light. That is not only the right look - it is the right PHYSICS: an overcast sky
#: fills shadows from the whole hemisphere, so the sun:shadow ratio comes down honestly instead of being
#: forced down by inventing ambient that has no source.
HDRI_DIR = "D:/Sightline/_downloads/assets/polyhaven/hdri"   # forward slashes: no escape to mangle
HDRI = "overcast_soil_puresky_2k"                # alt: kloofendal_48d_partly_cloudy_puresky_2k
SKY_PKG = "/Game/Sightline/Sky"


def import_hdri(name):
    """Import <name>.hdr as a TextureCube and PROVE it is one.

    UE decides between Texture2D and TextureCube from the source image's aspect and metadata. If it lands on
    Texture2D the SkyLight's `cubemap` assignment is silently ignored - the property accepts it, the log says
    nothing, and the shadows stay exactly as black as they were. So the type is asserted, not assumed.
    """
    import os
    path = f"{SKY_PKG}/T_{name}"
    got = unreal.load_asset(path)
    if got is None:
        src = os.path.join(HDRI_DIR, f"{name}.hdr")
        if not os.path.exists(src):
            raise RuntimeError(f"{src} not found. Fetch it first (CC0, Poly Haven).")
        t = unreal.AssetImportTask()
        for k, v in (("filename", src), ("destination_path", SKY_PKG), ("automated", True),
                     ("replace_existing", True), ("save", True)):
            t.set_editor_property(k, v)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
        paths = list(t.get_editor_property("imported_object_paths") or [])
        if not paths:
            raise RuntimeError(f"importing {src} produced no asset")
        got = unreal.load_asset(paths[0])
    if not isinstance(got, unreal.TextureCube):
        raise RuntimeError(f"{path} imported as {type(got).__name__}, not TextureCube. Assigning it to the "
                           f"SkyLight would be accepted and then silently ignored, and the shadows would "
                           f"stay black. Re-import the .hdr as a long/lat cubemap.")
    print(f"  HDRI: {got.get_path_name()} ({type(got).__name__})")
    return got


eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = eas.get_all_level_actors()

sky = [a for a in actors if isinstance(a, unreal.SkyLight)]
suns = [a for a in actors if isinstance(a, unreal.DirectionalLight)]
if not sky:
    raise RuntimeError("no SkyLight in the level. With GI off there is then NO ambient term at all, and "
                       "every shadow is black by construction. build_flood_valley.py should have placed one.")
if len(sky) > 1:
    print(f"WARNING: {len(sky)} SkyLights in the level; tuning all of them")

for s in sky:
    c = s.get_editor_property("light_component")
    before = {
        "intensity": float(c.get_editor_property("intensity")),
        "lower_black": bool(c.get_editor_property("lower_hemisphere_is_black")),
        "indirect": float(c.get_editor_property("indirect_lighting_intensity")),
        "volumetric": float(c.get_editor_property("volumetric_scattering_intensity")),
        "mobility": str(s.get_editor_property("root_component").get_editor_property("mobility")),
        "cubemap_res": int(c.get_editor_property("cubemap_resolution")),
    }
    try:
        before["source_type"] = str(c.get_editor_property("source_type"))
        before["real_time"] = bool(c.get_editor_property("real_time_capture"))
    except Exception as exc:                                 # noqa: BLE001 - report, do not guess
        before["source_type"] = f"(unreadable: {exc})"
    print(f"{s.get_actor_label()} BEFORE: {before}")

    # A real sky to capture. Every shadow measured on 2026-09-11 had BLUE as its LOWEST channel, which
    # means the ambient was not coming from a sky at all - the level's BP_Sky_Sphere is hidden, so a
    # captured-scene SkyLight had nothing but blackness to capture.
    cube = import_hdri(HDRI)
    c.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
    c.set_editor_property("cubemap", cube)
    if c.get_editor_property("cubemap") != cube:
        raise RuntimeError("the cubemap did not take - the SkyLight is still sourcing from the scene")

    c.set_editor_property("intensity", before["intensity"] * SKY_GAIN)
    # The missing ground bounce. This is the change that makes a shadow read as sky-lit rather than as a hole.
    c.set_editor_property("lower_hemisphere_is_black", False)
    c.set_editor_property("lower_hemisphere_color",
                          unreal.LinearColor(LOWER_HEMI[0], LOWER_HEMI[1], LOWER_HEMI[2], 1.0))
    try:
        c.recapture_sky()
        print("  recaptured the sky")
    except Exception as exc:                                 # noqa: BLE001
        print(f"  recapture_sky failed ({exc}); if the SkyLight is set to a specified cubemap this is fine")

    after_i = float(c.get_editor_property("intensity"))
    if abs(after_i - before["intensity"] * SKY_GAIN) > 1e-3:
        raise RuntimeError(f"intensity did not take: asked for {before['intensity'] * SKY_GAIN}, "
                           f"read back {after_i}. Is the SkyLight's mobility STATIC and the level locked?")
    print(f"  intensity {before['intensity']:.3f} -> {after_i:.3f}  (x{SKY_GAIN:g}), "
          f"lower hemisphere {before['lower_black']} -> False {LOWER_HEMI}")

for d in suns:
    dc = d.get_editor_property("light_component")
    print(f"{d.get_actor_label()} (unchanged): intensity {float(dc.get_editor_property('intensity')):.1f}, "
          f"rotation {d.get_actor_rotation()}")

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
les.save_current_level()
print(f"\nlevel saved. {len(sky)} SkyLight(s) tuned, {len(suns)} DirectionalLight(s) left alone.\n"
      "NOW RE-RENDER AND RE-MEASURE - this is one iteration, not an answer:\n"
      "  ue_python exec tools/scene/qa_shots.py\n"
      "  uv run python tools/scene/measure_lighting.py _artifacts/editor_shots/qa_1_valley.png "
      "_artifacts/editor_shots/qa_4_nadir45.png _artifacts/editor_shots/qa_2_settlement.png\n"
      "Target: sun:shadow between 5:1 and 10:1, shadow BLUE-dominated, under 2 % of terrain below sRGB 0.05.\n"
      "If it is still too dark raise SKY_GAIN; if the scene has gone flat and milky, lower it.")
