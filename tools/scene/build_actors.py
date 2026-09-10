"""Spawn the FloodValley survivors from data/scene/actors.json (running editor, PIE off). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_actors.py').read())"

Run `uv run python tools/scene/gen_actors.py` first (host side: it needs numpy, the editor has none), and
`build_poses.py` before that (it writes the pose assets this reads).

Non-obvious requirements, all from docs/HANDBOOK.md section 5:
  * actors must be MOVABLE - `simSetObjectPose` silently returns False on static actors, and the runtime
    re-posing and flood-level work both need it;
  * Cosys-AirSim identifies objects by `GetName()`, NOT the editor label, so every actor is renamed to
    `Human_<id>` / `Animal_<id>`; the auto-label pipeline keys on exactly those names;
  * the level must be saved from a script, and saves are unreliable while PIE runs.
"""

import json

import unreal

REPO = r"D:\Sightline"
FOLDER = "Survivors"

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: spawns and saves are unreliable while PIE runs")

with open(REPO + r"\data\scene\actors.json") as fh:
    plan = json.load(fh)
BASE_Z = plan["base_z_m"]

# --- idempotent: clear any previous spawn -----------------------------------------------------------------
removed = 0
for a in list(eas.get_all_level_actors()):
    nm = a.get_name()
    if nm.startswith(("Human_", "Animal_")) or str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1

missing, spawned = [], 0
for rec in plan["actors"]:
    mesh = unreal.load_asset(rec["asset"])
    pose = unreal.load_asset(rec["pose_asset"])
    if not mesh or not pose:
        missing.append((rec["name"], rec["asset"] if not mesh else rec["pose_asset"]))
        continue
    # UE (X, Y, Z) cm = (north*100, east*100, (asl - base_z)*100); ground_offset_cm seats the pose on the surface.
    loc = unreal.Vector(rec["north_m"] * 100.0,
                        rec["east_m"] * 100.0,
                        (rec["base_asl_m"] - BASE_Z) * 100.0 + rec["ground_offset_cm"])
    act = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, loc, unreal.Rotator(0, 0, rec["yaw_deg"]))
    c = act.skeletal_mesh_component
    c.set_skeletal_mesh_asset(mesh)
    c.set_mobility(unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    c.set_animation(pose)
    c.play(False)                                   # hold frame 0: these are static postures, not motion
    act.rename(rec["name"])                         # GetName() is what Cosys reports, not the label
    act.set_actor_label(rec["name"])
    act.set_folder_path(FOLDER)
    act.tags = [rec["cls"], rec["pose"], rec["zone"], rec["submersion"],
                "detectable" if rec["aerially_detectable"] else "buried"]
    spawned += 1

if missing:
    raise RuntimeError(f"{len(missing)} actors could not be spawned, first: {missing[:3]}")

names = {a.get_name() for a in eas.get_all_level_actors() if a.get_name().startswith(("Human_", "Animal_"))}
expected = {r["name"] for r in plan["actors"]}
if names != expected:
    raise RuntimeError(f"object-name mismatch: missing {sorted(expected - names)[:5]}, extra {sorted(names - expected)[:5]}")

saved = les.save_current_level()
c = plan["counts"]
print(f"removed {removed}, spawned {spawned} actors ({c['aerially_detectable']} detectable, "
      f"{c['buried_not_detectable']} buried), all names verified | level saved {saved}")
print(f"  poses     {c['by_pose']}")
print(f"  submersion{c['by_submersion']}")
