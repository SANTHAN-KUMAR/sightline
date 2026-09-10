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

# --- idempotent by UPDATE IN PLACE, never destroy-and-rename ------------------------------------------------
# Destroying an actor does not free its object name (the UObject lives until garbage collection), and renaming
# a fresh actor onto a taken name is a FATAL engine error rather than an exception:
#   "Renaming an object (...Human_000) on top of an existing object (...) is not allowed"  Obj.cpp:383
# That crashed the editor twice on 2026-09-10 - once on the live name, then again on the `DEAD_` name a
# previous crashed run had already saved into the level. Renaming around it just moves the collision.
# So: reuse the actor that already owns each name and only update its mesh/pose/transform. No rename, no
# collision, and the level's object names stay stable across rebuilds (which Cosys identifies objects by).
existing = {a.get_name(): a for a in eas.get_all_level_actors()
            if a.get_name().startswith(("Human_", "Animal_"))}
# sweep up debris from earlier crashed runs
removed = 0
for a in list(eas.get_all_level_actors()):
    if a.get_name().startswith("DEAD_"):
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

missing, spawned, reused = [], 0, 0
planned_names = {r["name"] for r in plan["actors"]}
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
    act = existing.get(rec["name"])
    if act is None:
        act = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, loc, unreal.Rotator(0, 0, rec["yaw_deg"]))
        act.rename(rec["name"])                     # only ever on a genuinely new actor: no name is taken
        spawned += 1
    else:
        act.set_actor_location_and_rotation(loc, unreal.Rotator(0, 0, rec["yaw_deg"]), False, False)
        reused += 1
    c = act.skeletal_mesh_component
    c.set_skeletal_mesh_asset(mesh)
    c.set_mobility(unreal.ComponentMobility.MOVABLE)
    c.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    # `set_animation()` only sets TRANSIENT runtime state: it leaves `animation_data.anim_to_play` as None, so
    # the pose is lost the moment the level is saved and reloaded and every survivor reverts to the bind
    # (A-)pose - visible in-game while the editor still looked right. The serialised struct is what persists.
    ad = unreal.SingleAnimationPlayData()
    ad.set_editor_property("anim_to_play", pose)
    ad.set_editor_property("saved_playing", False)   # hold frame 0: these are static postures, not motion
    ad.set_editor_property("saved_looping", False)
    ad.set_editor_property("saved_position", 0.0)
    c.set_editor_property("animation_data", ad)
    c.set_animation(pose)
    c.play(False)
    act.set_actor_label(rec["name"])
    act.set_folder_path(FOLDER)
    act.tags = [rec["cls"], rec["pose"], rec["zone"], rec["submersion"],
                "detectable" if rec["aerially_detectable"] else "buried"]

for nm, a in existing.items():
    if nm not in planned_names:
        eas.destroy_actor(a)
        removed += 1

if missing:
    raise RuntimeError(f"{len(missing)} actors could not be spawned, first: {missing[:3]}")

names = {a.get_name() for a in eas.get_all_level_actors() if a.get_name().startswith(("Human_", "Animal_"))}
expected = {r["name"] for r in plan["actors"]}
if names != expected:
    raise RuntimeError(f"object-name mismatch: missing {sorted(expected - names)[:5]}, extra {sorted(names - expected)[:5]}")

# Every survivor must carry a SERIALISED pose, or it renders as a bind-pose T shape at runtime.
unposed = [a.get_name() for a in eas.get_all_level_actors()
           if a.get_name().startswith(("Human_", "Animal_"))
           and a.skeletal_mesh_component.get_editor_property("animation_data").anim_to_play is None]
if unposed:
    raise RuntimeError(f"{len(unposed)} actors have no persisted pose (first: {unposed[:5]})")

# A stray un-renamed skeletal actor is an unlabelled human in frame: a false negative in every training image.
stray = [a.get_name() for a in eas.get_all_level_actors()
         if isinstance(a, unreal.SkeletalMeshActor) and not a.get_name().startswith(("Human_", "Animal_"))]
if stray:
    raise RuntimeError(f"unlabelled skeletal actors left in the level: {stray[:5]}")

saved = les.save_current_level()
c = plan["counts"]
print(f"reused {reused}, spawned {spawned}, removed {removed} | {c['aerially_detectable']} detectable, "
      f"{c['buried_not_detectable']} buried | names + persisted poses verified | level saved {saved}")
print(f"  poses     {c['by_pose']}")
print(f"  submersion{c['by_submersion']}")
