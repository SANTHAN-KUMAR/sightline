"""Place the FloodValley debris field from data/scene/props_layout.json (running editor, PIE off). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_props.py').read())"

Run `uv run python tools/scene/gen_props.py` first (host side: it needs numpy).

Deliberately does NOT rename actors. Renaming onto a name that a destroyed-but-not-yet-collected actor still
holds is a FATAL engine error (Obj.cpp:383) and crashed the editor twice on 2026-09-10. Debris is never a
detection target, so its object names do not need to be stable - only the survivors' do. Actors are found and
cleared by their outliner folder instead, which needs no rename at all.
"""

import json

import unreal

REPO = r"D:\Sightline"
FOLDER = "Debris"

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: spawns and saves are unreliable while PIE runs")

with open(REPO + r"\data\scene\props_layout.json") as fh:
    plan = json.load(fh)
BASE_Z = plan["base_z_m"]

removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

cache, missing, placed = {}, set(), 0
for rec in plan["items"]:
    path = rec["asset"]
    mesh = cache.get(path)
    if mesh is None:
        mesh = unreal.load_asset(path)
        cache[path] = mesh
    if mesh is None:
        missing.add(path)
        continue
    loc = unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                        (rec["base_asl_m"] - BASE_Z) * 100.0)
    rot = unreal.Rotator(rec["roll_deg"], rec["pitch_deg"], rec["yaw_deg"])
    act = eas.spawn_actor_from_object(mesh, loc, rot)
    if act is None:
        continue
    act.set_actor_scale3d(unreal.Vector(rec["scale"], rec["scale"], rec["scale"]))
    act.set_actor_label(rec["name"])
    act.set_folder_path(FOLDER)
    act.tags = ["Debris", rec["role"], "floating" if rec["floats"] else "grounded"]
    act.static_mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
    placed += 1

if missing:
    raise RuntimeError(f"{len(missing)} debris meshes are not in the project, first: {sorted(missing)[:3]}")

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != placed:
    raise RuntimeError(f"placed {placed} but the Debris folder holds {n_now}")

saved = les.save_current_level()
c = plan["counts"]
print(f"removed {removed}, placed {placed} debris actors "
      f"({c['floating']} floating, {c['grounded']} grounded, ~{c['approx_triangles']:,} tris) | saved {saved}")
print(f"  by role: {c['by_role']}")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images.")
