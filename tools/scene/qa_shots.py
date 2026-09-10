"""Render a fixed set of QA views of FloodValley so a human (or Claude) can LOOK at the scene.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_shots.py').read())"

Run this after EVERY change to the level, materials or actors, and actually open the images. The API will
happily report success for a scene that renders wrong: on 2026-09-10 every survivor was verified by object
name, by instance colour and by measured mask size, and all of it passed while all 71 rendered as bind-pose
T shapes, because `set_animation()` does not serialise. Only a screenshot showed it.

Writes PNGs to _artifacts/editor_shots/qa_*.png using a SceneCapture2D (SCS_FINAL_COLOR_LDR), which applies the
level's real lighting and post-processing.
"""

import json
import os

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\editor_shots"
W, H = 1600, 900

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    scene = json.load(fh)
with open(REPO + r"\data\scene\actors.json") as fh:
    truth = json.load(fh)
BASE_Z = scene["base_z_m"]


def ue(north_m, east_m, asl_m):
    return unreal.Vector(north_m * 100.0, east_m * 100.0, (asl_m - BASE_Z) * 100.0)


rt = unreal.RenderingLibrary.create_render_target2d(world, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("QA_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)


def shot(name, loc, rot, fov=75.0):
    comp.set_editor_property("fov_angle", fov)
    cap.set_actor_location_and_rotation(loc, unreal.Rotator(*rot), False, False)
    comp.capture_scene()
    comp.capture_scene()                       # second pass: temporal effects settle
    unreal.RenderingLibrary.export_render_target(world, rt, OUT, name + ".png")
    print(f"  {OUT}\\{name}.png")


try:
    # 1. the whole valley
    shot("qa_1_valley", unreal.Vector(-60000, -20000, 45000), (0, -22, 35))
    # 2. the flooded settlement, oblique
    shot("qa_2_settlement", unreal.Vector(2000, 9000, 3000), (0, -12, 35))

    # 3-5. close-ups of one survivor per posture family, so poses can actually be judged
    want = ("standing", "prone", "sitting", "waving", "half_submerged", "trapped")
    picked = {}
    for a in truth["actors"]:
        if a["pose"] in want and a["pose"] not in picked:
            picked[a["pose"]] = a
    for i, (pose, a) in enumerate(sorted(picked.items())):
        target = ue(a["north_m"], a["east_m"], a["base_asl_m"] + 0.9)
        # 6 m away, 20 deg above horizontal: close enough to see the limbs
        cam = unreal.Vector(target.x - 560, target.y - 200, target.z + 240)
        shot(f"qa_3_pose_{pose}", cam, (0, -18, 20), fov=40.0)

    # 6. nadir at survey altitude over the densest survivor cluster - what the detector actually sees
    es = [a["east_m"] for a in truth["actors"]]
    ns = [a["north_m"] for a in truth["actors"]]
    best, bn, be = -1, 0.0, 0.0
    for a in truth["actors"]:
        k = sum(1 for e2, n2 in zip(es, ns) if abs(e2 - a["east_m"]) < 30 and abs(n2 - a["north_m"]) < 30)
        if k > best:
            best, bn, be = k, a["north_m"], a["east_m"]
    shot("qa_4_nadir45", ue(bn, be, truth["water_level_m"] + 45.0), (0, -90, 0), fov=74.0)
finally:
    eas.destroy_actor(cap)

# Write a manifest so "I looked at it" becomes an auditable fact rather than a claim. It records WHEN the
# render happened and WHAT was in the level at that moment; tools/scene/assert_qa_fresh.py then fails if any
# scene script or scene data file is NEWER than this, i.e. if the scene changed after anyone last looked.
import hashlib
import time as _time

shots = sorted(f for f in os.listdir(OUT) if f.startswith("qa_") and f.endswith(".png"))
_all = eas.get_all_level_actors()
manifest = {
    "rendered_utc": _time.time(),
    "rendered_local": _time.strftime("%Y-%m-%d %H:%M:%S"),
    "level": world.get_name(),
    "actor_count": len(_all),
    "survivors": sum(1 for a in _all if a.get_name().startswith("Human_")),
    "houses": sum(1 for a in _all if str(a.get_folder_path()) == "Settlement"),
    "debris": sum(1 for a in _all if str(a.get_folder_path()) == "Debris"),
    "images": [],
}
for f in shots:
    fp = os.path.join(OUT, f)
    with open(fp, "rb") as fh:
        manifest["images"].append({"file": f, "bytes": os.path.getsize(fp),
                                   "sha1": hashlib.sha1(fh.read()).hexdigest()[:16]})
with open(os.path.join(OUT, "qa_manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)

print(f"\n{len(shots)} QA shots + qa_manifest.json written to {OUT}")
print(f"  level {manifest['level']}: {manifest['actor_count']} actors "
      f"({manifest['survivors']} survivors, {manifest['houses']} houses, {manifest['debris']} debris)")
print("OPEN THE IMAGES. Do not infer the scene is correct from API return values.")
