"""Import the downloaded photoreal props and humans into the project (running editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\import_assets.py').read())"

Sources (licences in _downloads/assets/MANIFEST.md): Poly Haven CC0 glTF props (2K textures) and Microsoft
Rocketbox rigged humans (MIT, FBX + TGA). Destinations:
  /Game/Sightline/Props/<id>/            static meshes: Nanite OFF (§4), simple box collision for small props
  /Game/Sightline/Characters/Rocketbox/<name>/   skeletal meshes + materials/textures
Props are grouped by role for the debris/distractor spawners (§2.3 rows 4 and 16); the group table is written to
data/scene/props.json so host-side spawner code can read it without touching the editor.

The environment set (trees, ground cover, utility poles, boat, waterfront, bark debris) is listed separately in
`data/scene/env_assets.json` so that the host-side check `tools/scene/check_env_assets.py` and this editor-side
importer read the SAME table. Trees are imported from the `*_2k_lite.gltf` built by tools/scene/gltf_tools.py
(the Poly Haven originals are 1.1M-3.9M triangles each, LOD0 only), and entries marked `"lods": true` get a
4-step reduction chain here, because that is what makes a canopy affordable at 8 GB of VRAM.
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


# role -> (build an LOD chain?, add simple box collision?). Foliage gets LODs and NO collision: a survivor must
# never be blocked by a leaf, and a collision box the size of a 19 m crown would swallow the whole street.
NO_COLLISION = {"tree", "ground_cover"}
LOD_PERCENTS = [1.0, 0.40, 0.15, 0.06]


def build_lods(sm):
    """4-step reduction chain with automatic screen sizes. Returns None on success, else the error text."""
    # UE 5.8: FStaticMeshReductionOptions{bAutoComputeLODScreenSize, ReductionSettings[]} and
    # FStaticMeshReductionSettings{PercentTriangles, ScreenSize}
    # (Engine/Source/Editor/StaticMeshEditor/Public/StaticMeshEditorSubsystemHelpers.h)
    try:
        settings = []
        for pct in LOD_PERCENTS:
            s = unreal.StaticMeshReductionSettings()
            s.set_editor_property("percent_triangles", pct)
            settings.append(s)
        opts = unreal.StaticMeshReductionOptions()
        opts.set_editor_property("auto_compute_lod_screen_size", True)
        opts.set_editor_property("reduction_settings", settings)
        n = sme.set_lods(sm, opts)
        return None if (n is None or n >= 0) else f"set_lods returned {n}"
    except Exception as e:
        return str(e)


def finish_mesh(sm, role):
    """Nanite off (§4), collision per role, LODs where asked, then save. Returns the props.json entry."""
    ns = sm.get_editor_property("nanite_settings")
    if ns.get_editor_property("enabled"):
        ns.set_editor_property("enabled", False)
        sm.set_editor_property("nanite_settings", ns)
    if role not in NO_COLLISION and sme.get_simple_collision_count(sm) == 0:
        sme.add_simple_collisions(sm, unreal.ScriptingCollisionShapeType.BOX)
    eal.save_asset(sm.get_path_name())
    b = sm.get_bounding_box()
    return {"asset": sm.get_path_name().split(".")[0], "tris": sm.get_num_triangles(0),
            "size_m": [round((b.max.x - b.min.x) / 100, 2), round((b.max.y - b.min.y) / 100, 2),
                       round((b.max.z - b.min.z) / 100, 2)]}


report = {"props": {}, "humans": {}, "failures": []}
for role, ids in PROPS.items():
    for pid in ids:
        src = os.path.join(SRC, "polyhaven", pid, f"{pid}_2k.gltf")
        try:
            objs = import_file(src, f"/Game/Sightline/Props/{pid}")
            meshes = [o for o in objs if isinstance(o, unreal.StaticMesh)]
            if not meshes:
                raise RuntimeError(f"no static mesh in {[type(o).__name__ for o in objs]}")
            report["props"][pid] = {"role": role, "meshes": [finish_mesh(sm, role) for sm in meshes]}
        except Exception as e:
            report["failures"].append(f"{pid}: {e}")

# --- environment set: canopy, ground cover, utility poles, boat, waterfront, bark debris -------------------
with open(REPO + r"\data\scene\env_assets.json") as fh:
    ENV = json.load(fh)["assets"]
for e in ENV:
    src = os.path.join(SRC, *e["source"].split("/"))
    try:
        if not os.path.exists(src):
            raise RuntimeError(f"source missing: {src} "
                               f"(run tools/scene/fetch_env_assets.py, gen_boat.py, gltf_tools.py lite)")
        objs = import_file(src, f"/Game/Sightline/Props/{e['id']}")
        meshes = [o for o in objs if isinstance(o, unreal.StaticMesh)]
        if not meshes:
            raise RuntimeError(f"no static mesh in {[type(o).__name__ for o in objs]}")
        entry, lod_errors = [], []
        for sm in meshes:
            if e.get("lods"):
                err = build_lods(sm)
                if err:
                    lod_errors.append(err)
            rec = finish_mesh(sm, e["role"])
            try:
                rec["lod_count"] = sme.get_lod_count(sm)
            except Exception:
                rec["lod_count"] = None
            entry.append(rec)
        report["props"][e["id"]] = {"role": e["role"], "meshes": entry, "source": e["source"],
                                    "lods_requested": bool(e.get("lods"))}
        if lod_errors:
            report["failures"].append(f"{e['id']}: LOD build failed: {sorted(set(lod_errors))[0]}")
    except Exception as e2:
        report["failures"].append(f"{e['id']}: {e2}")

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
    tris = sum(m["tris"] for m in v["meshes"])
    lods = sorted({m.get("lod_count") for m in v["meshes"]} - {None})
    print(f"{pid:28s} {v['role']:13s} {len(v['meshes']):>3d} mesh {tris:>9,} tris"
          + (f"  LODs {lods}" if lods else "")
          + ("" if len(v["meshes"]) > 6 else
             "  " + str([(m["tris"], m["size_m"]) for m in v["meshes"]])))
for n, v in report["humans"].items():
    print(n, v)
