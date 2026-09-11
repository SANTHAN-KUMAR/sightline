"""Render the fixed views the sky lane is judged on (running editor, PIE OFF).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_sky.py').read())"
    uv run python tools/scene/check_sky.py            # host side: the check that can FAIL

Writes to _artifacts/sky_lane/. Four views, each chosen so that the thing being measured is the ONLY thing
in the frame - which is what makes the measurement in check_sky.py non-circular:

  sky_up      pitch +28 deg from the demo camera: 100 % sky, no terrain, no water, no horizon. The sky's
              colour can therefore be measured directly instead of being segmented out of a mixed frame by
              a colour rule that would itself have to assume what the sky looks like.
  sky_horizon pitch -3 deg: the horizon band, where an HDRI dome shows fog contamination and a seam if the
              dome is wrong. Not measured numerically; it is there to be LOOKED at.
  lod_far     EXACTLY the qa_1_valley camera from tools/scene/qa_shots.py (-600, -200, 450 m, pitch -22,
              yaw 35, FOV 75). Same camera means the before/after pair is an A/B, not two different shots.
  lod_near    nadir 45 m over the densest tree cluster - survey altitude, where the vegetation lane
              confirmed the crowns are FULL. This is the REFERENCE the distant crowns are compared against,
              rendered in the same build and the same exposure, so the comparison is not against a
              remembered number from a different scene.

Set NAME_SUFFIX in globals() before exec to tag a sweep variant, e.g.

    NAME_SUFFIX = "_mppe0.25"
    exec(open(r'D:\\Sightline\\tools\\scene\\qa_sky.py').read())
"""

import json
import os

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\sky_lane"
W, H = 1600, 900
SUFFIX = globals().get("NAME_SUFFIX", "")

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    BASE_Z = json.load(fh)["base_z_m"]
with open(REPO + r"\data\scene\vegetation.json") as fh:
    veg = json.load(fh)


def ue(north_m, east_m, asl_m):
    return unreal.Vector(north_m * 100.0, east_m * 100.0, (asl_m - BASE_Z) * 100.0)


# Densest DRY tree cluster, on a 40 m grid. Dry (hillslope/floodplain/garden) on purpose: a drowned crown
# has its lower half under water, which would make the reference darker than a real crown and quietly bias
# the LOD comparison in the fix's favour.
CELL = 40.0
bins = {}
for it in veg["items"]:
    if it["zone"] in ("flooded", "channel"):
        continue
    bins.setdefault((int(it["east_m"] // CELL), int(it["north_m"] // CELL)), []).append(it)
key = max(bins, key=lambda k: len(bins[k]))
cluster = bins[key]
cn = sum(i["north_m"] for i in cluster) / len(cluster)
ce = sum(i["east_m"] for i in cluster) / len(cluster)
cg = sum(i["ground_asl_m"] for i in cluster) / len(cluster)
print(f"densest dry cluster: {len(cluster)} trees at north {cn:.0f} m / east {ce:.0f} m, ground {cg:.1f} m ASL")

rt = unreal.RenderingLibrary.create_render_target2d(world, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("QA_SKY_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)

# The qa_1_valley camera, copied from tools/scene/qa_shots.py. If that file ever changes these must too, or
# the A/B stops being an A/B.
VALLEY_LOC, VALLEY_ROT, VALLEY_FOV = unreal.Vector(-60000, -20000, 45000), (0, -22, 35), 75.0


def shot(name, loc, rot, fov=75.0):
    comp.set_editor_property("fov_angle", fov)
    cap.set_actor_location_and_rotation(loc, unreal.Rotator(*rot), False, False)
    comp.capture_scene()
    comp.capture_scene()                     # second pass: temporal effects settle (as in qa_shots.py)
    fn = name + SUFFIX + ".png"
    unreal.RenderingLibrary.export_render_target(world, rt, OUT, fn)
    print(f"  {OUT}\\{fn}")


try:
    shot("sky_up", VALLEY_LOC, (0, 28, 35), VALLEY_FOV)
    shot("sky_horizon", VALLEY_LOC, (0, -3, 35), VALLEY_FOV)
    shot("lod_far", VALLEY_LOC, VALLEY_ROT, VALLEY_FOV)
    shot("lod_near", ue(cn, ce, cg + 45.0), (0, -90, 0), 74.0)
finally:
    eas.destroy_actor(cap)

meta = {"suffix": SUFFIX, "cluster_trees": len(cluster), "cluster_north_m": cn, "cluster_east_m": ce,
        "cluster_ground_asl_m": cg,
        "valley_cam": {"loc": [VALLEY_LOC.x, VALLEY_LOC.y, VALLEY_LOC.z], "rot": list(VALLEY_ROT),
                       "fov": VALLEY_FOV}}
with open(os.path.join(OUT, f"qa_sky_meta{SUFFIX}.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print(f"4 views written to {OUT} (suffix {SUFFIX!r}). OPEN THEM - a return value is not a look.")
