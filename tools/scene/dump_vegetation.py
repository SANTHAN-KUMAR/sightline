"""Read every canopy instance transform back OUT of the level and write it to disk (running editor, PIE off).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\dump_vegetation.py').read())"
    ue_python code="RELOAD_FIRST=True; exec(open(r'D:\\Sightline\\tools\\scene\\dump_vegetation.py').read())"

`build_vegetation.py` runs this itself after it saves. Run it a SECOND time with `RELOAD_FIRST = True` to
reload the map from disk before reading: that is the only way to prove the instances actually serialised into
the .umap rather than merely existing in the editor's memory, which is exactly the class of silent failure
docs/CONTEXT.md section 7 is about. `tools/scene/check_vegetation.py` (host side) consumes the file and exits
non-zero.

Writes `_artifacts/vegetation/placed_instances.json`. Nothing here modifies the level.
"""

import json
import os
import time

import unreal

REPO = r"D:\Sightline"
FOLDER = "Vegetation"
# A DRY RUN MUST NOT WRITE THE REAL ARTIFACT. `tools/scene/check_realism.py` execs this file against a
# mock UE with `__name__ == "__dry_run__"`. Once that mock became complete enough to run this script to
# completion, the dry run began overwriting the real dump with mock content (/Game/Mock/mesh_LOD0, level
# "get_editor_world"), and check_vegetation.py then failed against a canopy that was in fact correct.
# A gate that a dry run can corrupt is not a gate.
_DRY = __name__ == "__dry_run__"
DUMP = REPO + (r"\_artifacts\vegetation\placed_instances_DRYRUN.json" if _DRY
               else r"\_artifacts\vegetation\placed_instances.json")
MAP = "/Game/Sightline/Maps/FloodValley"

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")

if globals().get("RELOAD_FIRST"):
    print(f"reloading {MAP} from disk to prove the instances serialised ...")
    les.load_level(MAP)

world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _quat_to_rot(q):
    try:
        return q.rotator()
    except Exception:
        return unreal.MathLibrary.conv_quaternion_to_rotator(q)


with open(REPO + r"\data\scene\vegetation.json") as fh:
    BASE_Z = json.load(fh)["base_z_m"]

all_actors = eas.get_all_level_actors()
veg = [a for a in all_actors if str(a.get_folder_path()) == FOLDER]
dump = {"written_utc": time.time(), "written_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level": world.get_name(), "base_z_m": BASE_Z,
        "pitch_sign": globals().get("PITCH_SIGN", 1.0),
        "reloaded_from_disk": bool(globals().get("RELOAD_FIRST")),
        "level_actor_count": len(all_actors), "vegetation_actor_count": len(veg),
        "mem_before": globals().get("MEM0", {}), "mem_after": globals().get("MEM1", {}), "actors": []}

for a in veg:
    for c in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
        rows = []
        for i in range(c.get_instance_count()):
            r = c.get_instance_transform(i, True)           # True = world space
            t = next(x for x in (r if isinstance(r, (tuple, list)) else [r])
                     if isinstance(x, unreal.Transform))
            tr, sc = t.get_editor_property("translation"), t.get_editor_property("scale3d")
            ro = _quat_to_rot(t.get_editor_property("rotation"))
            rows.append([round(tr.x / 100.0, 3), round(tr.y / 100.0, 3), round(tr.z / 100.0, 3),
                         round(ro.yaw, 2), round(ro.pitch, 2), round(ro.roll, 2), round(sc.x, 4)])
        sm = c.get_editor_property("static_mesh")
        dump["actors"].append({"label": a.get_actor_label(), "class": c.get_class().get_name(),
                               "mesh": sm.get_name() if sm else None,
                               "mesh_path": sm.get_path_name() if sm else None,
                               "count": c.get_instance_count(),
                               "instances_north_east_z_yaw_pitch_roll_scale": rows})

os.makedirs(os.path.dirname(DUMP), exist_ok=True)
with open(DUMP, "w") as fh:
    json.dump(dump, fh)
tot = sum(a["count"] for a in dump["actors"])
print(f"dumped {tot:,} instances from {len(veg)} actors "
      f"({'AFTER a reload from disk' if dump['reloaded_from_disk'] else 'from the live editor'}) -> {DUMP}")
for a in dump["actors"]:
    print(f"  {a['label']:34s} {a['count']:5d}  {a['mesh']}")
print(f"level {dump['level']} holds {dump['level_actor_count']:,} actors")
print("NEXT (host side): uv run python tools/scene/check_vegetation.py")
