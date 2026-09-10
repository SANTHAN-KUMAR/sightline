"""Import the downloaded photoreal props and humans into the project (running editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\import_assets.py').read())"

Sources (licences in _downloads/assets/MANIFEST.md): Poly Haven CC0 glTF props (2K textures) and Microsoft
Rocketbox rigged humans (MIT, FBX + TGA). Destinations:
  /Game/Sightline/Props/<id>/            static meshes: Nanite OFF (§4), simple box collision for small props
  /Game/Sightline/Characters/Rocketbox/<name>/   skeletal meshes + materials/textures
Props are grouped by role for the debris/distractor spawners (§2.3 rows 4 and 16); the group table is written to
data/scene/props.json so host-side spawner code can read it without touching the editor.
"""

import json
import os

import unreal

REPO = r"D:\Sightline"
SRC = REPO + r"\_downloads\assets"
eal = unreal.EditorAssetLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
sme = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports and saves are unreliable while PIE runs")

# role -> Poly Haven ids. Roles follow §2.3: boulders/rubble on the fan, woody debris, household debris that
# floats or collects at the waterline (and doubles as human-like distractors), vehicles.
PROPS = {
    "boulder": ["rock_07", "rock_09", "stone_01", "rock_moss_set_01", "rock_moss_set_02", "namaqualand_boulders_01"],
    "woody": ["dead_tree_trunk", "dead_quiver_trunk", "tree_stump_01", "tree_stump_02", "dry_branches_medium_01"],
    "household": ["Barrel_01", "barrel_03", "wooden_crate_01", "wooden_crate_02", "plastic_crate_01", "cardboard_box_01",
                  "plastic_jerrycan", "metal_jerrycan", "plastic_bottle_gallon", "plastic_container", "trashbag",
                  "plastic_monobloc_chair_01", "old_tyre", "rollershutter_door"],
    "vehicle": ["covered_car"],
    "vegetation": ["fern_02"],
}
HUMANS = ["Adults/Female_Adult_04", "Adults/Female_Adult_11", "Adults/Female_Adult_17", "Adults/Male_Adult_04",
          "Adults/Male_Adult_08", "Adults/Male_Adult_13", "Adults/Male_Adult_15", "Children/Female_Child_02",
          "Children/Male_Child_02"]


def import_file(path, dest):
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", dest), ("automated", True),
                 ("replace_existing", True), ("save", True)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.get_editor_property("imported_object_paths")]


report = {"props": {}, "humans": {}, "failures": []}
for role, ids in PROPS.items():
    for pid in ids:
        src = os.path.join(SRC, "polyhaven", pid, f"{pid}_2k.gltf")
        try:
            objs = import_file(src, f"/Game/Sightline/Props/{pid}")
            meshes = [o for o in objs if isinstance(o, unreal.StaticMesh)]
            if not meshes:
                raise RuntimeError(f"no static mesh in {[type(o).__name__ for o in objs]}")
            entry = []
            for sm in meshes:
                ns = sm.get_editor_property("nanite_settings")
                if ns.get_editor_property("enabled"):
                    ns.set_editor_property("enabled", False)
                    sm.set_editor_property("nanite_settings", ns)
                if sme.get_simple_collision_count(sm) == 0:
                    sme.add_simple_collisions(sm, unreal.ScriptingCollisionShapeType.BOX)
                eal.save_asset(sm.get_path_name())
                b = sm.get_bounding_box()
                size_m = [round((b.max.x - b.min.x) / 100, 2), round((b.max.y - b.min.y) / 100, 2),
                          round((b.max.z - b.min.z) / 100, 2)]
                entry.append({"asset": sm.get_path_name().split(".")[0], "tris": sm.get_num_triangles(0),
                              "size_m": size_m})
            report["props"][pid] = {"role": role, "meshes": entry}
        except Exception as e:
            report["failures"].append(f"{pid}: {e}")

for rel in HUMANS:
    name = rel.split("/")[-1]
    src = os.path.join(SRC, "rocketbox", *rel.split("/"), "Export", f"{name}.fbx")
    try:
        objs = import_file(src, f"/Game/Sightline/Characters/Rocketbox/{name}")
        sk = [o for o in objs if isinstance(o, unreal.SkeletalMesh)]
        if not sk:
            raise RuntimeError(f"no skeletal mesh in {[type(o).__name__ for o in objs]}")
        b = sk[0].get_bounds().box_extent
        report["humans"][name] = {"asset": sk[0].get_path_name().split(".")[0],
                                  "height_m": round(b.z * 2 / 100, 2),
                                  "imported": sorted({type(o).__name__ for o in objs})}
    except Exception as e:
        report["failures"].append(f"{name}: {e}")

with open(REPO + r"\data\scene\props.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps({"props": len(report["props"]), "humans": len(report["humans"]), "failures": report["failures"]}))
for pid, v in report["props"].items():
    print(pid, v["role"], [(m["tris"], m["size_m"]) for m in v["meshes"]])
for n, v in report["humans"].items():
    print(n, v)
