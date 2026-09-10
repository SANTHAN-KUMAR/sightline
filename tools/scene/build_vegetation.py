"""Import the Poly Haven tree scans and place the FloodValley canopy (running editor, PIE OFF). Idempotent.

    uv run python tools/scene/gen_vegetation.py                     # host side: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_vegetation.py').read())"

Run AFTER `import_assets.py` (it may have already imported the trees, with the wrong settings for them) and
after `build_buildings.py` / `build_actors.py` (the layout is clearance-checked against the houses and the
survivors that those two place). Nothing here touches `M_FloodValleyTerrain`, `M_FloodWater`, `M_PBR_Master`
or `M_Foam`: this script owns only the tree assets it imports and the `Vegetation` outliner folder.

THREE THINGS THIS SCRIPT FIXES THAT ALL FAIL SILENTLY
-----------------------------------------------------
1. **Nanite must be ON for the trees.** They are photogrammetry scans decimated to 107-353 k triangles each
   (measured from the glTF accessors by gen_vegetation.py) and the layout places 4,377 of them: 1.08 BILLION
   source triangles. Nanite stores that geometry once per mesh (889,641 unique triangles) and rasterises clusters
   sized to the pixels on screen, so the instances are nearly free; without it the renderer would have to push
   every instanced triangle. `import_assets.py` turns Nanite OFF for every prop - correct for a 12-53 k rock,
   catastrophic here - so this script turns it back on for the tree meshes and ASSERTS that it took. If it did
   not, and the raw instanced count exceeds NO_NANITE_MAX_TRIS, the script refuses to place anything.

2. **The leaf materials arrive TRANSLUCENT.** Every Poly Haven tree glTF declares its leaf material
   `"alphaMode": "BLEND"`, yet every `*_diff_2k.jpg` is a 3-channel RGB JPEG with no alpha channel at all
   (checked with PIL on all five sets). The transparency is therefore a no-op in the source and a disaster in
   UE: a translucent material is NOT rendered by Nanite (it silently falls back to the fallback mesh, i.e.
   bald trees), it does not write the depth/stencil that Cosys-AirSim's instance segmentation reads, and
   thousands of translucent crowns is unbounded overdraw. Every tree material is forced to Opaque here.

3. **The leaves are single-sided.** The glTF sets `doubleSided: true` on all three slots; if the importer
   drops it, half of every crown disappears when viewed from above. Forced to two-sided here.

Trees are spawned with NO COLLISION on purpose. `sim_fly` fails a takeoff on contact with any actor not named
"Ground" (docs/CONTEXT.md), and a spawner line trace that lands on a leaf would put a survivor in mid-air. The
canopy is a visual and an occlusion feature, not a physical one; the layout already keeps 70 m clear of the
launch pad so nothing near the ground station changes either way.

Actors are cleared by outliner FOLDER, never renamed: renaming onto a name a destroyed-but-uncollected actor
still holds is a fatal engine error (Obj.cpp:383) and crashed this editor twice on 2026-09-10.
"""

import json
import os

import unreal

REPO = r"D:\Sightline"
PH = REPO + r"\_downloads\assets\polyhaven"
FOLDER = "Vegetation"
# If a render ever shows the drowned trees leaning UPSTREAM (towards +north), flip this to +1.0. It is the one
# sign in the whole layout that cannot be checked without looking at a picture: UE is left-handed, so a
# positive pitch raises +X and tips local +Z towards -X, which with yaw = lean azimuth leans the trunk along
# the azimuth. Nothing else depends on it.
PITCH_SIGN = 1.0

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports, saves and material recompiles are unreliable while PIE runs")

with open(REPO + r"\data\scene\vegetation.json") as fh:
    plan = json.load(fh)
BASE_Z = plan["base_z_m"]
CAT = plan["catalogue"]
print(f"build_vegetation: {len(set(i['pid'] for i in plan['items']))} tree meshes "
      f"({plan['counts']['unique_mesh_triangles']:,} unique triangles, ~34 MB of decimated source buffers) "
      f"and {plan['counts']['trees'] + plan['counts']['ground_cover']:,} actors to spawn.")
print("  The spawn loop, not the import, is the slow part at this actor count; expect minutes and a large "
      "level save. If the editor struggles, re-run gen_vegetation.py with a smaller --max-trees.")


def import_gltf(pid):
    t = unreal.AssetImportTask()
    for k, v in (("filename", os.path.join(PH, pid, f"{pid}_2k.gltf")),
                 ("destination_path", f"/Game/Sightline/Props/{pid}"),
                 ("automated", True), ("replace_existing", True), ("save", True)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.get_editor_property("imported_object_paths")]


def bbox_volume(sm):
    b = sm.get_bounding_box()
    return (b.max.x - b.min.x) * (b.max.y - b.min.y) * (b.max.z - b.min.z)


def resolve_mesh(pid, hint):
    """The generator only writes a HINT path, because the imported mesh is named after the glTF node and that
    naming differs between Poly Haven exports. Resolve for real: hint -> anything already under the prop
    folder -> import. Picks the biggest mesh by bounding volume, which (unlike get_num_triangles) is not
    affected by Nanite reporting the fallback mesh."""
    if hint and eal.does_asset_exist(hint):
        sm = unreal.load_asset(hint)
        if isinstance(sm, unreal.StaticMesh):
            return sm, "hint"
    folder = f"/Game/Sightline/Props/{pid}"
    for source in ("existing", "imported"):
        if source == "imported":
            import_gltf(pid)
        if eal.does_directory_exist(folder):
            found = []
            for p in eal.list_assets(folder, recursive=True, include_folder=False):
                a = unreal.load_asset(p)
                if isinstance(a, unreal.StaticMesh):
                    found.append(a)
            if found:
                return max(found, key=bbox_volume), source
    raise RuntimeError(f"{pid}: no static mesh under {folder} and the glTF import produced none "
                       f"(is {PH}\\{pid}\\{pid}_2k.gltf complete?)")


def force_opaque_two_sided(mat):
    """Poly Haven tree leaf materials import TRANSLUCENT (glTF alphaMode BLEND) even though the base colour
    JPEGs carry no alpha. Translucent kills Nanite, breaks the segmentation mask and costs unbounded overdraw.
    Handles both a plain Material and a MaterialInstanceConstant (which needs base_property_overrides)."""
    changed = []
    if isinstance(mat, unreal.MaterialInstanceConstant):
        ov = mat.get_editor_property("base_property_overrides")
        if str(ov.get_editor_property("blend_mode")) != str(unreal.BlendMode.BM_OPAQUE) or \
                not ov.get_editor_property("override_blend_mode"):
            ov.set_editor_property("override_blend_mode", True)
            ov.set_editor_property("blend_mode", unreal.BlendMode.BM_OPAQUE)
            changed.append("blend=Opaque")
        if not ov.get_editor_property("two_sided") or not ov.get_editor_property("override_two_sided"):
            ov.set_editor_property("override_two_sided", True)
            ov.set_editor_property("two_sided", True)
            changed.append("two_sided")
        mat.set_editor_property("base_property_overrides", ov)
        mel.update_material_instance(mat)
    elif isinstance(mat, unreal.Material):
        if str(mat.get_editor_property("blend_mode")) != str(unreal.BlendMode.BM_OPAQUE):
            mat.set_editor_property("blend_mode", unreal.BlendMode.BM_OPAQUE)
            changed.append("blend=Opaque")
        if not mat.get_editor_property("two_sided"):
            mat.set_editor_property("two_sided", True)
            changed.append("two_sided")
        if changed:
            mel.recompile_material(mat)
    if changed:
        eal.save_asset(mat.get_path_name())
    return changed


def assert_compiles(mat, where):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    except zeroed statistics (docs/CONTEXT.md section 7). Zero instructions is fatal; zero texture samples is
    printed loudly but not fatal, because a legitimately constant slot would report it too."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0:
        raise RuntimeError(f"{where}: material {mat.get_name()} FAILED TO COMPILE (0 instructions). "
                           f"Check its sampler types against its default textures.")
    if s.num_pixel_texture_samples == 0:
        print(f"  !! {where}: {mat.get_name()} compiles but samples NO texture - it will render flat")
    return s


# --- 1. resolve, import and configure the tree meshes -------------------------------------------------------
used = sorted({it["pid"] for it in plan["items"]})
meshes, mesh_report, nanite_bad = {}, [], []
for pid in used:
    sm, how = resolve_mesh(pid, CAT[pid].get("asset_hint"))
    ns = sm.get_editor_property("nanite_settings")
    if not ns.get_editor_property("enabled"):
        ns.set_editor_property("enabled", True)
        sm.set_editor_property("nanite_settings", ns)
    mats = []
    for i, slot in enumerate(sm.get_editor_property("static_materials")):
        mi = slot.get_editor_property("material_interface")
        if mi is None:
            raise RuntimeError(f"{pid}: material slot {i} "
                               f"({slot.get_editor_property('material_slot_name')}) is EMPTY after import")
        mats.append(mi)
    fixed = []
    for mi in dict.fromkeys(mats):
        fixed += [f"{mi.get_name()}:{c}" for c in force_opaque_two_sided(mi)]
    eal.save_asset(sm.get_path_name())
    ns2 = unreal.load_asset(sm.get_path_name()).get_editor_property("nanite_settings")
    on = bool(ns2.get_editor_property("enabled"))
    if not on:
        nanite_bad.append(pid)
    for mi in dict.fromkeys(mats):
        assert_compiles(mi, pid)
    meshes[pid] = sm
    b = sm.get_bounding_box()
    mesh_report.append(
        f"  {pid:16s} {how:8s} nanite={'ON' if on else 'OFF'} slots={len(mats)} "
        f"size_m=({(b.max.x - b.min.x) / 100:.1f},{(b.max.y - b.min.y) / 100:.1f},{(b.max.z - b.min.z) / 100:.1f}) "
        f"src_tris={CAT[pid]['tris']:,} fallback_tris={sm.get_num_triangles(0):,}"
        + (f" fixed[{', '.join(fixed)}]" if fixed else ""))
print("tree meshes:")
print("\n".join(mesh_report))

raw_tris = plan["counts"]["instanced_source_triangles"]
if nanite_bad and raw_tris > plan["nanite"]["no_nanite_max_tris"]:
    raise RuntimeError(
        f"Nanite is OFF on {nanite_bad} and this layout instances {raw_tris:,} source triangles, over the "
        f"{plan['nanite']['no_nanite_max_tris']:,} non-Nanite ceiling. Placing it would make the editor "
        f"unusable. Enable Nanite on those meshes (or re-run gen_vegetation.py with a much smaller "
        f"--max-trees) before continuing. NOTHING WAS PLACED.")
if nanite_bad:
    print(f"  !! Nanite is OFF on {nanite_bad} but the layout is small enough to survive it")

# ground cover meshes stay NON-Nanite: fern_02's four meshes are 784-2,384 triangles, where Nanite is all cost
ground_meshes = {}
for it in plan.get("ground_items", []):
    p = it["asset_hint"]
    if p not in ground_meshes:
        m = unreal.load_asset(p)
        if m is None:
            raise RuntimeError(f"understorey mesh {p} is not in the project - run import_assets.py first")
        ground_meshes[p] = m

# --- 2. clear the folder ------------------------------------------------------------------------------------
removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

# --- 3. place -----------------------------------------------------------------------------------------------
placed = 0
for rec in plan["items"] + plan.get("ground_items", []):
    mesh = meshes.get(rec["pid"]) if rec["kind"] == "tree" else ground_meshes.get(rec["asset_hint"])
    if mesh is None:
        raise RuntimeError(f"no mesh resolved for {rec['name']} ({rec['pid']})")
    loc = unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                        (rec["base_asl_m"] - BASE_Z) * 100.0)
    rot = unreal.Rotator(rec["roll_deg"], rec["pitch_deg"] * PITCH_SIGN, rec["yaw_deg"])
    act = eas.spawn_actor_from_object(mesh, loc, rot)
    if act is None:
        raise RuntimeError(f"spawn failed for {rec['name']}")
    act.set_actor_scale3d(unreal.Vector(rec["scale"], rec["scale"], rec["scale"]))
    act.set_actor_label(rec["name"])
    act.set_folder_path(FOLDER)
    act.tags = ["Vegetation", rec["kind"], rec["pid"], rec["zone"]]
    comp = act.static_mesh_component
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    comp.set_collision_profile_name("NoCollision")
    placed += 1
    if placed % 500 == 0:
        print(f"  ... {placed} placed")

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != placed:
    raise RuntimeError(f"placed {placed} but the {FOLDER} folder holds {n_now}")

saved = les.save_current_level()
c, k = plan["counts"], plan["canopy"]
print(f"vegetation: removed {removed}, placed {placed} ({c['trees']:,} trees + {c['ground_cover']:,} "
      f"understorey) | level saved {saved}")
print(f"  by species: {c['by_species']}")
print(f"  by zone:    {c['by_zone']}")
print(f"  canopy closure {k['closure_roi'] * 100:.1f} % of the {k['roi_area_km2']} km2 ROI, "
      f"{k['closure_settlement'] * 100:.1f} % over the settlement")
print(f"  {raw_tris:,} instanced source triangles from {c['unique_mesh_triangles']:,} unique "
      f"(Nanite stores each mesh once)")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images. Specifically check that (a) the crowns are "
      "LEAFY, not bald - bald means the leaf material is still translucent and Nanite fell back; (b) the "
      "drowned trees lean downstream (towards -north / south), else flip PITCH_SIGN at the top of this file.")
