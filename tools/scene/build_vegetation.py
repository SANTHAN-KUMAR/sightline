"""Import the Poly Haven tree scans and place the FloodValley canopy (running editor, PIE OFF). Idempotent.

    uv run python tools/scene/gen_vegetation.py                     # host side: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_vegetation.py').read())"
    uv run python tools/scene/check_vegetation.py                   # host side: the check that can fail

Run AFTER `import_assets.py` (it may have already imported the trees, with the wrong settings for them) and
after `build_buildings.py` / `build_actors.py` (the layout is clearance-checked against the houses and the
survivors that those two place). Nothing here touches `M_FloodValleyTerrain`, `M_FloodWater`, `M_PBR_Master`
or `M_Foam`: this script owns only the tree assets it imports and the `Vegetation` outliner folder.

ONE ACTOR PER SPECIES, NOT ONE ACTOR PER TREE. THIS IS THE WHOLE POINT.
----------------------------------------------------------------------
The first version of this script spawned one StaticMeshActor per plant with
`EditorActorSubsystem.spawn_actor_from_object`. At 5,508 actors it **killed the editor**:

    Script Stack (1 frames): /Script/UnrealEd.EditorActorSubsystem.SpawnActorFromObject
    Fatal error: Ran out of memory allocating 1082560 (1.0 MiB) bytes with alignment 8.
    Last error msg: The paging file is too small for this operation to complete.
        AvailablePhysical 0.46 GiB    AvailableVirtual 0.00 GiB
        UsedPhysical 5.15 GiB         UsedVirtual 27.48 GiB   PeakUsedVirtual 27.49 GiB

It ran out of **commit**, not RAM: this machine has 15.7 GB of RAM and a ~26.7 GB commit limit, and a UObject
actor + scene component + registration + editor selection/outliner bookkeeping per tree is simply the wrong
data structure at this count.

Everything is therefore placed as INSTANCES on `HierarchicalInstancedStaticMeshComponent`s: one actor per tree
species (5) plus one per understorey mesh (4) = **9 actors carrying 5,508 instances**. A per-instance transform
is a 4x4 matrix in a flat array, not an object; the whole canopy costs a few MB instead of gigabytes of commit.
It also renders far faster, which is the reason UE has ISMs at all. `MAX_ACTORS` below is a hard tripwire.

Instances are safe here specifically because **nothing labels a tree**. Only `Human_*` and `Animal_*` need
unique instance-segmentation IDs; the canopy exists to occlude and to look right, and every instance of one
species sharing one ID costs nothing. (This is also why the Foliage tool is still banned - HANDBOOK section 6 -
it is not the instancing that corrupts labels, it is putting *labelled* things into a shared foliage actor.)

THREE THINGS THIS SCRIPT FIXES THAT ALL FAIL SILENTLY
-----------------------------------------------------
1. **Nanite must be ON for the trees.** They are photogrammetry scans decimated to 107-353 k triangles each
   (measured from the glTF accessors by gen_vegetation.py) and the layout places 4,342 of them: 1.07 BILLION
   source triangles. Nanite stores that geometry once per mesh (889,641 unique triangles) and rasterises clusters
   sized to the pixels on screen, so the instances are nearly free; without it the renderer would have to push
   every instanced triangle. `import_assets.py` turns Nanite OFF for every prop - correct for a 12-53 k rock,
   catastrophic here - so this script turns it back on for the tree meshes and ASSERTS that it took. If it did
   not, and the raw instanced count exceeds NO_NANITE_MAX_TRIS, the script refuses to place anything.

2. **The leaf materials arrive TRANSLUCENT.** Every Poly Haven tree glTF declares its leaf material
   `"alphaMode": "BLEND"`, yet every leaf texture is a **JPEG** (`*_leaves_diff_2k.jpg`) and the JPEG format has
   no alpha channel at all - there is no opacity source anywhere in these assets, and the leaves are real scanned
   geometry rather than alpha-cut cards, so BLEND is a no-op in the source and a disaster in UE: a translucent
   material is NOT rendered by Nanite (it silently falls back to the fallback mesh, i.e. bald trees), it does not
   write the depth/stencil that Cosys-AirSim's instance segmentation reads, and thousands of translucent crowns is
   unbounded overdraw. Every tree material is forced to Opaque here. BLEND_MASKED would be wrong for the same
   reason - there is no mask to sample - and would punch holes wherever the base colour happened to be dark.

3. **The leaves are single-sided.** The glTF sets `doubleSided: true` on all three slots; if the importer
   drops it, half of every crown disappears when viewed from above. Forced to two-sided here.

Instances are placed with NO COLLISION on purpose. `sim_fly` fails a takeoff on contact with any actor not named
"Ground" (docs/CONTEXT.md), and a spawner line trace that lands on a leaf would put a survivor in mid-air. The
canopy is a visual and an occlusion feature, not a physical one; the layout already keeps 70 m clear of the
launch pad so nothing near the ground station changes either way.

Actors are cleared by outliner FOLDER, never renamed: renaming onto a name a destroyed-but-uncollected actor
still holds is a fatal engine error (Obj.cpp:383) and crashed this editor twice on 2026-09-10.
"""

import ctypes
import json
import os
import time

import unreal

REPO = r"D:\Sightline"
PH = REPO + r"\_downloads\assets\polyhaven"
FOLDER = "Vegetation"
DUMP = REPO + r"\_artifacts\vegetation\placed_instances.json"

# Hard tripwire. This lane may never go back to one actor per plant: see the header. 9 is the expected number.
MAX_ACTORS = 400
BATCH = 500                     # instances per add_instances call
# Abort BEFORE the allocator does. The crash above happened at AvailableVirtual 0.00 GiB; stopping at 1.2 GiB
# leaves room to unwind, print and keep the editor alive. Raising is the point - a half-placed canopy that
# reports itself is worth infinitely more than a dead editor.
MIN_AVAIL_VIRTUAL_GIB = 1.2
MIN_AVAIL_PHYSICAL_GIB = 0.35

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
sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports, saves and material recompiles are unreliable while PIE runs")


# --- memory, in exactly the terms the crash log used ---------------------------------------------------------
# UE's FWindowsPlatformMemory::GetStats() fills its fields from these two Win32 calls, so these numbers are
# directly comparable with the "AvailablePhysical / UsedVirtual / PeakUsedVirtual" line of the fatal error.
class _MEMSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PMCEX(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


_K32 = ctypes.windll.kernel32
_K32.GetCurrentProcess.restype = ctypes.c_void_p          # WITHOUT this the pseudo-handle is truncated to
_GPMI = getattr(_K32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
_GPMI.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMCEX), ctypes.c_ulong]   # 32 bits and every field is 0


def mem():
    G = 1024.0 ** 3
    ms = _MEMSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    _K32.GlobalMemoryStatusEx(ctypes.byref(ms))
    pmc = _PMCEX()
    pmc.cb = ctypes.sizeof(pmc)
    if not _GPMI(_K32.GetCurrentProcess(), ctypes.byref(pmc), ctypes.sizeof(pmc)):
        raise RuntimeError("GetProcessMemoryInfo failed - do not trust a zeroed memory report")
    return {"AvailablePhysical": ms.ullAvailPhys / G, "AvailableVirtual": ms.ullAvailPageFile / G,
            "UsedPhysical": pmc.WorkingSetSize / G, "UsedVirtual": pmc.PagefileUsage / G,
            "PeakUsedVirtual": pmc.PeakPagefileUsage / G, "CommitLimit": ms.ullTotalPageFile / G}


def memline(tag):
    m = mem()
    print(f"  [mem] {tag:22s} AvailablePhysical {m['AvailablePhysical']:.2f} GiB  "
          f"AvailableVirtual {m['AvailableVirtual']:.2f} GiB  UsedPhysical {m['UsedPhysical']:.2f} GiB  "
          f"UsedVirtual {m['UsedVirtual']:.2f} GiB  PeakUsedVirtual {m['PeakUsedVirtual']:.2f} GiB")
    return m


#: `tools/scene/check_realism.py` execs this script against a mock UE with `__name__ == "__dry_run__"`.
#: Nothing is allocated in that mode, but `mem()` still reads the REAL machine - so a dry run performed while
#: the editor and several agents are resident aborts on a memory floor that has nothing to do with the script
#: under test. The guard protects a live editor; in a dry run there is no editor to protect.
DRY_RUN = __name__ == "__dry_run__"


def mem_guard(where):
    m = mem()
    if DRY_RUN:
        return m
    if m["AvailableVirtual"] < MIN_AVAIL_VIRTUAL_GIB or m["AvailablePhysical"] < MIN_AVAIL_PHYSICAL_GIB:
        raise RuntimeError(
            f"ABORTING at {where} to keep the editor alive: AvailableVirtual {m['AvailableVirtual']:.2f} GiB, "
            f"AvailablePhysical {m['AvailablePhysical']:.2f} GiB (floors {MIN_AVAIL_VIRTUAL_GIB} / "
            f"{MIN_AVAIL_PHYSICAL_GIB}). NOTHING WAS SAVED.")
    return m


with open(REPO + r"\data\scene\vegetation.json") as fh:
    plan = json.load(fh)
BASE_Z = plan["base_z_m"]
CAT = plan["catalogue"]
N_TREES = len(plan["items"])
N_GROUND = len(plan.get("ground_items", []))
print(f"build_vegetation: {N_TREES:,} trees + {N_GROUND:,} understorey = {N_TREES + N_GROUND:,} INSTANCES "
      f"on {len(set(i['pid'] for i in plan['items']))} tree meshes "
      f"({plan['counts']['unique_mesh_triangles']:,} unique triangles).")
MEM0 = memline("before")


# --- transforms ----------------------------------------------------------------------------------------------
# UE (X, Y, Z) cm = (north*100, east*100, (asl - base_z)*100). Verified against build_actors.py and
# build_props.py, whose actors are in the level and are known-correct (CONTEXT.md, FloodValley scene facts).
def _rot_to_quat(rot):
    try:
        return rot.quaternion()
    except Exception:
        return unreal.MathLibrary.conv_rotator_to_quaternion(rot)


def _quat_to_rot(q):
    try:
        return q.rotator()
    except Exception:
        return unreal.MathLibrary.conv_quaternion_to_rotator(q)


def make_xform(rec, pitch_sign=1.0):
    t = unreal.Transform()
    t.set_editor_property("translation", unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                                                       (rec["base_asl_m"] - BASE_Z) * 100.0))
    t.set_editor_property("rotation", _rot_to_quat(
        unreal.Rotator(rec["roll_deg"], rec["pitch_deg"] * pitch_sign, rec["yaw_deg"])))
    s = float(rec["scale"])
    t.set_editor_property("scale3d", unreal.Vector(s, s, s))
    return t


# Prove the transform builder before trusting it 5,508 times: unreal.Transform's constructor argument order has
# bitten people, and a silently-identity rotation would put every drowned tree bolt upright.
_probe = {"north_m": 12.34, "east_m": -56.78, "base_asl_m": BASE_Z + 9.0,
          "roll_deg": 0.0, "pitch_deg": -11.0, "yaw_deg": 137.0, "scale": 1.25}
_t = make_xform(_probe)
_tr = _t.get_editor_property("translation")
_rr = _quat_to_rot(_t.get_editor_property("rotation"))
_sc = _t.get_editor_property("scale3d")
if (abs(_tr.x - 1234.0) > 0.5 or abs(_tr.y + 5678.0) > 0.5 or abs(_tr.z - 900.0) > 0.5
        or abs(_sc.x - 1.25) > 1e-3
        or abs(_rr.pitch + 11.0) > 0.05 or abs(_rr.yaw - 137.0) > 0.05 or abs(_rr.roll) > 0.05):
    raise RuntimeError(f"transform builder is wrong: t={_tr} rot={_rr} scale={_sc} "
                       f"(expected (1234,-5678,900), pitch -11 yaw 137 roll 0, scale 1.25)")
print(f"  transform builder verified: {_probe['north_m']} N / {_probe['east_m']} E / +9 m -> "
      f"({_tr.x:.0f}, {_tr.y:.0f}, {_tr.z:.0f}) cm, pitch {_rr.pitch:.1f} yaw {_rr.yaw:.1f}, scale {_sc.x:.2f}")


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
    JPEGs carry no alpha channel - JPEG cannot. Translucent kills Nanite, breaks the segmentation mask and
    costs unbounded overdraw. Handles both a plain Material and a MaterialInstanceConstant (which needs
    base_property_overrides). NOTE: unreal.BlendMode.BM_OPAQUE does not exist in UE 5.8; it is BLEND_OPAQUE."""
    changed = []
    if isinstance(mat, unreal.MaterialInstanceConstant):
        ov = mat.get_editor_property("base_property_overrides")
        if str(ov.get_editor_property("blend_mode")) != str(unreal.BlendMode.BLEND_OPAQUE) or \
                not ov.get_editor_property("override_blend_mode"):
            ov.set_editor_property("override_blend_mode", True)
            ov.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
            changed.append("blend=Opaque")
        if not ov.get_editor_property("two_sided") or not ov.get_editor_property("override_two_sided"):
            ov.set_editor_property("override_two_sided", True)
            ov.set_editor_property("two_sided", True)
            changed.append("two_sided")
        mat.set_editor_property("base_property_overrides", ov)
        mel.update_material_instance(mat)
    elif isinstance(mat, unreal.Material):
        if str(mat.get_editor_property("blend_mode")) != str(unreal.BlendMode.BLEND_OPAQUE):
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
            changed.append("blend=Opaque")
        if not mat.get_editor_property("two_sided"):
            mat.set_editor_property("two_sided", True)
            changed.append("two_sided")
        if changed:
            mel.recompile_material(mat)
    if changed:
        eal.save_asset(mat.get_path_name())
    return changed


def blend_of(mat):
    if isinstance(mat, unreal.MaterialInstanceConstant):
        ov = mat.get_editor_property("base_property_overrides")
        if ov.get_editor_property("override_blend_mode"):
            return str(ov.get_editor_property("blend_mode")).split(".")[-1]
        return str(mat.get_base_material().get_editor_property("blend_mode")).split(".")[-1] + "(inherited)"
    return str(mat.get_editor_property("blend_mode")).split(".")[-1]


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
meshes, mesh_report, nanite_bad, mat_report = {}, [], [], []
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
        mat_report.append(f"    {pid:16s} {mi.get_name():34s} blend={blend_of(mi)}")
    meshes[pid] = sm
    b = sm.get_bounding_box()
    mesh_report.append(
        f"  {pid:16s} {how:8s} nanite={'ON' if on else 'OFF'} slots={len(mats)} "
        f"size_m=({(b.max.x - b.min.x) / 100:.1f},{(b.max.y - b.min.y) / 100:.1f},{(b.max.z - b.min.z) / 100:.1f}) "
        f"src_tris={CAT[pid]['tris']:,} fallback_tris={sm.get_num_triangles(0):,}"
        + (f" fixed[{', '.join(fixed)}]" if fixed else ""))
print("tree meshes:")
print("\n".join(mesh_report))
print("  material blend modes AFTER the fix (translucent here = bald Nanite trees + broken seg mask):")
print("\n".join(mat_report))

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
print(f"understorey meshes: {len(ground_meshes)} fern variants")
memline("meshes ready")

# --- 2. clear the folder ------------------------------------------------------------------------------------
# A previous run of the one-actor-per-tree version may have left thousands of actors here, and destroying
# thousands of actors is itself a memory event: collect as we go.
removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
        if removed % 500 == 0:
            unreal.SystemLibrary.collect_garbage()
            print(f"  ... removed {removed}")
if removed:
    unreal.SystemLibrary.collect_garbage()
print(f"cleared {removed} pre-existing actors from the {FOLDER} folder")
memline("folder cleared")

# --- 3. group the layout by mesh ------------------------------------------------------------------------------
groups = []                                              # (actor_label, mesh, [records], kind)
for pid in used:
    groups.append((f"Veg_Tree_{pid}", meshes[pid], [r for r in plan["items"] if r["pid"] == pid], "tree"))
for path in sorted(ground_meshes):
    groups.append((f"Veg_Under_{path.rsplit('/', 1)[-1]}", ground_meshes[path],
                   [r for r in plan["ground_items"] if r["asset_hint"] == path], "ground"))

if len(groups) > MAX_ACTORS:
    raise RuntimeError(f"{len(groups)} actors is over the {MAX_ACTORS} tripwire - this lane instances, "
                       f"it does not spawn one actor per plant. NOTHING WAS PLACED.")
print(f"placing {sum(len(g[2]) for g in groups):,} instances across {len(groups)} actors "
      f"(tripwire {MAX_ACTORS})")

# --- 4. place -------------------------------------------------------------------------------------------------
placed_total, per_actor, t0 = 0, {}, time.time()
for label, mesh, recs, kind in groups:
    act = eas.spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
    if act is None:
        raise RuntimeError(f"could not spawn the container actor for {label}")
    # UE 5.8's Python bindings do NOT expose Actor.add_component_by_class (checked: it is absent from
    # dir(unreal.Actor)). SubobjectDataSubsystem is the same path the Details panel's "+ Add Component" takes,
    # and it produces an INSTANCE component that serialises with the actor - verified on 2026-09-11 by writing
    # three instances, saving, reloading the map from disk and reading all three back unchanged.
    handles = sds.k2_gather_subobject_data_for_instance(act)
    new_handle, why = sds.add_new_subobject(unreal.AddNewSubobjectParams(
        parent_handle=handles[0], new_class=unreal.HierarchicalInstancedStaticMeshComponent,
        blueprint_context=None, conform_transform_to_parent=True))
    if str(why):
        raise RuntimeError(f"{label}: add_new_subobject refused: {why}")
    sds.rename_subobject(new_handle, "Canopy")
    hism = act.get_component_by_class(unreal.HierarchicalInstancedStaticMeshComponent)
    if hism is None:
        raise RuntimeError(f"{label}: no HISM on the actor after add_new_subobject")
    try:
        hism.set_static_mesh(mesh)
    except Exception:
        hism.set_editor_property("static_mesh", mesh)
    if hism.get_editor_property("static_mesh") != mesh:
        raise RuntimeError(f"{label}: the HISM did not take the mesh {mesh.get_name()}")
    hism.set_mobility(unreal.ComponentMobility.STATIC)
    hism.set_collision_profile_name("NoCollision")
    hism.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    act.set_actor_label(label)
    act.set_folder_path(FOLDER)
    act.tags = ["Vegetation", kind, recs[0]["pid"]]
    # The instance transforms below are WORLD coordinates, so the container must sit at the origin unrotated
    # and unscaled. Assert it rather than assume it: a stray root transform would offset the entire canopy.
    al, ar, asx = act.get_actor_location(), act.get_actor_rotation(), act.get_actor_scale3d()
    if (abs(al.x) + abs(al.y) + abs(al.z) > 1.0 or abs(ar.pitch) + abs(ar.yaw) + abs(ar.roll) > 0.01
            or abs(asx.x - 1) + abs(asx.y - 1) + abs(asx.z - 1) > 1e-3):
        raise RuntimeError(f"{label}: container actor is not at the identity ({al}, {ar}, {asx})")

    n = 0
    for i in range(0, len(recs), BATCH):
        chunk = [make_xform(r, PITCH_SIGN if r["kind"] == "tree" else 1.0) for r in recs[i:i + BATCH]]
        hism.add_instances(chunk, False, True)           # world_space=True: transforms are world coordinates
        n += len(chunk)
        del chunk
        unreal.SystemLibrary.collect_garbage()
        mem_guard(f"{label} after {n}/{len(recs)}")
    got = hism.get_instance_count()
    if got != len(recs):
        raise RuntimeError(f"{label}: added {n} instances but the component reports {got}")
    per_actor[label] = got
    placed_total += got
    m = memline(f"{label} ({got})")
    print(f"  {label:34s} {got:5d} instances  mesh={mesh.get_name()}")

print(f"placed {placed_total:,} instances in {time.time() - t0:.0f} s")
MEM1 = memline("after placing")

# --- 5. verify against the plan BEFORE saving -----------------------------------------------------------------
n_actors_folder = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_actors_folder != len(groups):
    raise RuntimeError(f"expected {len(groups)} actors in {FOLDER}, found {n_actors_folder}")
if placed_total != N_TREES + N_GROUND:
    raise RuntimeError(f"placed {placed_total} instances but the plan has {N_TREES + N_GROUND}")
by_species = {}
for label, mesh, recs, kind in groups:
    if kind == "tree":
        by_species[recs[0]["pid"]] = per_actor[label]
if by_species != plan["counts"]["by_species"]:
    raise RuntimeError(f"per-species instance counts differ from the plan:\n  got  {by_species}\n"
                       f"  want {plan['counts']['by_species']}")

saved = les.save_current_level()
if not saved:
    raise RuntimeError("save_current_level() returned False - the canopy is NOT on disk")

# --- 6. dump what the ENGINE holds, read back from the components ---------------------------------------------
# Not what we asked for - what the components actually contain. dump_vegetation.py reads every instance
# transform back out of the components; tools/scene/check_vegetation.py (host side) re-derives the per-species
# counts and the survivor clearance from that file and exits non-zero.
exec(open(REPO + r"\tools\scene\dump_vegetation.py").read())

c, k = plan["counts"], plan["canopy"]
print(f"vegetation: removed {removed}, placed {placed_total:,} instances "
      f"({c['trees']:,} trees + {c['ground_cover']:,} understorey) on {n_actors_folder} actors | level saved")
print(f"  by species: {by_species}")
print(f"  by zone:    {c['by_zone']}")
print(f"  canopy closure {k['closure_roi'] * 100:.1f} % of the {k['roi_area_km2']} km2 ROI, "
      f"{k['closure_settlement'] * 100:.1f} % over the settlement")
print(f"  {raw_tris:,} instanced source triangles from {c['unique_mesh_triangles']:,} unique "
      f"(Nanite stores each mesh once)")
print(f"  level now holds {dump['level_actor_count']:,} actors in total")
print("NOW LOOK: run tools/scene/qa_shots.py and open the images. Specifically check that (a) the crowns are "
      "LEAFY, not bald - bald means the leaf material is still translucent and Nanite fell back; (b) the "
      "drowned trees lean downstream (towards -north / south), else flip PITCH_SIGN at the top of this file.")
print("THEN RUN: uv run python tools/scene/check_vegetation.py")
