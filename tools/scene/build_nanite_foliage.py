"""Rebuild the tree meshes' Nanite data so distant crowns keep their volume (PIE OFF). Idempotent.

    ue_python code="MODE='VOXELIZE'; exec(open(r'D:\\Sightline\\tools\\scene\\build_nanite_foliage.py').read())"
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_sky.py').read())"
    uv run python tools/scene/check_sky.py

WHY A MESH SETTING AND NOT A CVAR
---------------------------------
`tools/scene/sweep_nanite_foliage.py` measured the two runtime hypotheses on the 450 m demo frame. Metric is
the darkest quartile's G/R in a fixed canopy crop - bark is red-dominant, leaves are green-dominant - with
the 45 m survey-altitude crowns as the same-run reference (domain=sim):

    r.Nanite.MaxPixelsPerEdge 1.0 (default)   G/R 0.723   canopy coverage 32.3 %
    r.Nanite.MaxPixelsPerEdge 0.5             G/R 0.835   canopy coverage 49.2 %
    r.Nanite.MaxPixelsPerEdge 0.25            G/R 0.918   canopy coverage 66.4 %
    r.Nanite.MaxPixelsPerEdge 0.1             G/R 0.949   canopy coverage 70.1 %
    r.Nanite.Streaming.StreamingPoolSize 1024 G/R 0.722   canopy coverage 31.6 %   <- no effect at all

So the cause is screen-space LOD, NOT streaming residency: doubling the streaming pool moved the number by
0.001, which is noise. Screen-space LOD is also subject to hard diminishing returns - going from 0.25 to 0.1
costs 6.25x the rasterised geometry for +0.031 of G/R - and it is a GLOBAL cost paid on every frame the
dataset captures at 35-80 m, where the crowns were already correct. Buying the wide demo shot with a
permanent tax on the dataset render would be the wrong trade.

The build-time lever has no runtime cost. UE 5.8's `NaniteShapePreservation` enum exists for exactly this:

    NONE          "Do not attempt to preserve the object's shape in the distance."      <- current setting
    PRESERVE_AREA "Try to maintain the same surface area at all distances (Legacy foliage technique)."
    VOXELIZE      "Simplify triangles to voxels in the distance to preserve the perceived volume of the
                   object. Useful for foliage that thins out otherwise."

That last sentence is the engine's own description of this exact defect. It is a rebuild of the Nanite data,
paid once at build time and stored in the asset.

THE TRAP THIS SCRIPT AVOIDS
---------------------------
`get_editor_property("nanite_settings")` returns a REFERENCE to the live struct (docs/CONTEXT.md section 7),
so mutating it and then reading a "before" value gives you the "after" value and the diff looks empty. Every
field is therefore COPIED OUT into a plain dict first, and the struct that gets written is CONSTRUCTED FRESH
from that dict with exactly one field changed - which also proves nothing else moved. The result is read
back from a reloaded asset, not from the object we just wrote.
"""

import ctypes
import time

import unreal

MODE = globals().get("MODE", "VOXELIZE")            # NONE | PRESERVE_AREA | VOXELIZE
VOXEL_OPACITY = globals().get("VOXEL_OPACITY", None)  # None = leave as is
MIN_AVAIL_VIRTUAL_GIB = 0.80
MIN_AVAIL_PHYSICAL_GIB = 0.40

#: Fields copied out and reconstructed. Everything MeshNaniteSettings.__init__ accepts.
FIELDS = ("enabled", "explicit_tangents", "lerp_u_vs", "separable", "voxel_ndf", "voxel_opacity",
          "shape_preservation", "position_precision", "normal_precision", "tangent_precision",
          "bone_weight_precision", "keep_percent_triangles", "trim_relative_error", "generate_fallback",
          "fallback_target", "fallback_percent_triangles", "fallback_relative_error",
          "max_edge_length_factor", "num_rays", "voxel_level", "ray_back_up", "displacement_uv_channel")


class _MEMSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem():
    G = 1024.0 ** 3
    ms = _MEMSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / G, ms.ullAvailPageFile / G


def memline(tag):
    ap, av = mem()
    print(f"  [mem] {tag:28s} AvailablePhysical {ap:.2f} GiB  AvailableVirtual {av:.2f} GiB")
    return ap, av


les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first - asset builds and saves fail silently while PIE runs")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
smes = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
eal = unreal.EditorAssetLibrary

want = getattr(unreal.NaniteShapePreservation, MODE)
print(f"build_nanite_foliage.py MODE={MODE} ({want}) at {time.strftime('%H:%M:%S')}")
memline("before")

# Resolve the meshes from the LEVEL, not from data/scene/vegetation.json. The generator's asset_hint points
# at "<name>_2k_lite" paths that do not exist in this project - resolve_mesh() fell through to "<name>_2k" at
# import time and the JSON was never updated. Applying settings to the hint paths would have silently
# configured nothing at all, and every assertion here would still have passed.
meshes = []
for a in eas.get_all_level_actors():
    if str(a.get_folder_path()) != "Vegetation":
        continue
    for c in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
        sm = c.get_editor_property("static_mesh")
        if sm is not None and sm.get_editor_property("nanite_settings").get_editor_property("enabled"):
            if sm not in meshes:
                meshes.append(sm)
if not meshes:
    raise RuntimeError("no Nanite-enabled meshes found on the Vegetation HISM actors - nothing to rebuild")
print(f"  {len(meshes)} Nanite tree meshes resolved from the level's HISM components")

changed, report = [], []
for sm in meshes:
    ap, av = mem()
    if av < MIN_AVAIL_VIRTUAL_GIB or ap < MIN_AVAIL_PHYSICAL_GIB:
        raise RuntimeError(f"ABORTING before {sm.get_name()}: AvailableVirtual {av:.2f} GiB / "
                           f"AvailablePhysical {ap:.2f} GiB is under the floor "
                           f"({MIN_AVAIL_VIRTUAL_GIB}/{MIN_AVAIL_PHYSICAL_GIB}). "
                           f"{len(changed)} mesh(es) were already rebuilt and saved.")
    live = sm.get_editor_property("nanite_settings")
    snap = {f: live.get_editor_property(f) for f in FIELDS}      # COPY OUT before anything is written
    before_mode = snap["shape_preservation"]
    if before_mode == want and (VOXEL_OPACITY is None or snap["voxel_opacity"] == VOXEL_OPACITY):
        report.append(f"  {sm.get_name():22s} already {MODE}, skipped")
        continue

    fresh = dict(snap)
    fresh["shape_preservation"] = want
    if VOXEL_OPACITY is not None:
        fresh["voxel_opacity"] = VOXEL_OPACITY
    ns = unreal.MeshNaniteSettings(**fresh)

    t0 = time.perf_counter()
    smes.set_nanite_settings(sm, ns, True)                       # True = apply, i.e. rebuild the Nanite data
    if not eal.save_asset(sm.get_path_name()):
        raise RuntimeError(f"save_asset returned False for {sm.get_path_name()} - the rebuild is NOT on disk")
    dt = time.perf_counter() - t0

    # Read back from a RELOADED asset, not from the struct we just handed the engine.
    rl = unreal.load_asset(sm.get_path_name()).get_editor_property("nanite_settings")
    got = rl.get_editor_property("shape_preservation")
    if got != want:
        raise RuntimeError(f"{sm.get_name()}: shape_preservation is {got} after the write, wanted {want}")
    drift = {f: (snap[f], rl.get_editor_property(f)) for f in FIELDS
             if f not in ("shape_preservation", "voxel_opacity") and snap[f] != rl.get_editor_property(f)}
    if drift:
        raise RuntimeError(f"{sm.get_name()}: rebuilding changed fields it should not have: {drift}")
    changed.append(sm.get_name())
    ap2, av2 = mem()
    report.append(f"  {sm.get_name():22s} {before_mode.name} -> {got.name}  "
                  f"nanite_tris={sm.get_num_nanite_triangles():>9,}  rebuild {dt:6.1f} s  "
                  f"availPhys {ap2:.2f} GiB  availVirt {av2:.2f} GiB")

print("\n".join(report))
memline("after")
print(f"\n{len(changed)} mesh(es) rebuilt to shape_preservation={MODE}: {changed}")
print("NOW RENDER AND LOOK:  ue_python exec tools/scene/qa_sky.py   then  uv run python tools/scene/check_sky.py")
