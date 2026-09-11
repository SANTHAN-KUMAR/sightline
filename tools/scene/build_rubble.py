"""Import, texture and place the FloodValley structural-collapse rubble field (running editor, PIE OFF). Idempotent.

    D:\\Tools\\uv\\uv.exe run python tools\\scene\\gen_rubble.py        # host side first: it needs numpy
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_rubble.py').read())"

WHAT IT DOES
  1. resolves the six textures it needs from the sets ALREADY in the project (nothing is downloaded) and asserts
     their sRGB/compression settings, because a normal map left as sRGB makes the whole material fail to compile
     and every rubble actor renders as the grey WorldGridMaterial checker with no error anywhere;
  2. builds `M_Rubble_Master` and seven instances in /Game/Sightline/Rubble/Materials, and ASSERTS that each one
     compiles (`MaterialEditingLibrary.get_statistics`: a failed compile reports zero instructions and nothing
     else). The master carries a Dust parameter - Reference B's palette is "dust-covered, desaturated tan/grey,
     very low colour contrast", which is a material effect, not a texture choice;
  3. imports the 40 generated OBJs, turns **Nanite off** (imports arrive with it ON, and `get_num_triangles`
     then reports the fallback mesh - 347 triangles for a 524k-triangle asset), and checks the imported triangle
     count against the count `gen_rubble.py` recorded;
  4. places `data/scene/rubble_layout.json` as INSTANCES (see below) and saves the level.

INSTANCES, NOT ACTORS - the constraint that killed the previous attempt.
An earlier lane spawned one StaticMeshActor per item and the editor died at ~5,500 actors:

    Fatal error: Ran out of memory allocating 1082560 (1.0 MiB) bytes with alignment 8.
    Last error msg: The paging file is too small for this operation to complete.
        AvailablePhysical 0.46 GiB   AvailableVirtual 0.00 GiB   UsedVirtual 27.48 GiB

That is the Windows COMMIT limit (26.7 GiB on this 15.7 GiB machine), not RAM. It is spent by the per-AActor
cost - the UObject, its USceneComponent, registration, render-state and outliner entries - which is thousands
of times the cost of one instance. So the 1,799 rubble items become one
`HierarchicalInstancedStaticMeshComponent` per mesh variant (44 actors, one extra for each belonging variant
that carries the second fabric colour), and each item is one FTransform inside it. HISM also gives per-instance
frustum culling and instance LOD, which matters here because Nanite is deliberately OFF on these meshes.

NAMING. Like `build_props.py`, this NEVER renames an actor. Renaming onto a name a destroyed-but-not-yet-
collected actor still holds is a FATAL engine error (Obj.cpp:383) that crashed the editor twice on 2026-09-10.
Rubble is never a detection target, so its object names do not need to be stable; the field is cleared by its
outliner folder instead. `set_actor_label` is a label, not a rename, and is the only naming call here.

COLLISION is OFF by default (`RUBBLE_COLLISION = False`). Two reasons, both concrete: complex collision for
~2.4M triangles is real memory on a 16 GB / 8 GB-VRAM machine, and the sightline server fails a flight on any
collision report whose actor is not named "Ground", so a drone descending over the fan would fail on rubble
contact. Set it True if something needs to line-trace onto the rubble surface (e.g. re-seating survivors on it).
"""

import ctypes
import json

import unreal

REPO = r"D:\Sightline"
FOLDER = "Rubble"
MATDIR = "/Game/Sightline/Rubble/Materials"
MESHDIR = "/Game/Sightline/Rubble"
TEXROOT = "/Game/Sightline/Textures"
RUBBLE_COLLISION = False        # see the docstring; True gives query-only complex collision
MAX_ACTORS = 400                # hard tripwire: this lane instances, it never spawns one actor per item
BATCH = 500                     # instances per add_instances call
MIN_AVAIL_VIRTUAL_GIB = 1.2     # abort BEFORE the allocator does; the crash happened at 0.00 GiB
MIN_AVAIL_PHYSICAL_GIB = 0.35

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: imports come back as 32 px placeholders and saves return False in PIE")


# --- memory, in exactly the terms the crash log used ----------------------------------------------------------
# Same helper as build_vegetation.py, and deliberately identical: UE's FWindowsPlatformMemory::GetStats() fills
# its fields from these two Win32 calls, so these numbers are directly comparable with the
# "AvailablePhysical / AvailableVirtual / UsedVirtual / PeakUsedVirtual" line of the fatal error that killed the
# actor-per-item attempt.
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
_K32.GetCurrentProcess.restype = ctypes.c_void_p          # without this the pseudo-handle truncates to 32 bits
_GPMI = getattr(_K32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
_GPMI.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMCEX), ctypes.c_ulong]      # and every field reads back 0


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
    print(f"  [mem] {tag:26s} AvailablePhysical {m['AvailablePhysical']:.2f} GiB  "
          f"AvailableVirtual {m['AvailableVirtual']:.2f} GiB  UsedPhysical {m['UsedPhysical']:.2f} GiB  "
          f"UsedVirtual {m['UsedVirtual']:.2f} GiB  PeakUsedVirtual {m['PeakUsedVirtual']:.2f} GiB")
    return m


def mem_guard(where):
    m = mem()
    if m["AvailableVirtual"] < MIN_AVAIL_VIRTUAL_GIB or m["AvailablePhysical"] < MIN_AVAIL_PHYSICAL_GIB:
        raise RuntimeError(
            f"ABORTING at {where} to keep the editor alive: AvailableVirtual {m['AvailableVirtual']:.2f} GiB, "
            f"AvailablePhysical {m['AvailablePhysical']:.2f} GiB (floors {MIN_AVAIL_VIRTUAL_GIB} / "
            f"{MIN_AVAIL_PHYSICAL_GIB}). NOTHING WAS SAVED.")
    return m


MEM0 = memline("before")

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    meta = json.load(fh)
with open(REPO + r"\data\scene\rubble_layout.json") as fh:
    plan = json.load(fh)
BASE_Z = meta["base_z_m"]


# --- textures -------------------------------------------------------------------------------------------------
# The two suffix conventions in this project are real: build_materials.py wrote the terrain normals as `_N`,
# build_buildings.py wrote the building normals as `_NRM`. Resolve both instead of guessing.
def tex(pid, kind):
    names = {"col": [f"T_{pid}_D"], "nrm": [f"T_{pid}_NRM", f"T_{pid}_N"], "arm": [f"T_{pid}_ARM"]}[kind]
    for nm in names:
        p = f"{TEXROOT}/{pid}/{nm}"
        if eal.does_asset_exist(p):
            t = unreal.load_asset(p)
            if kind == "nrm":
                ok = (not t.get_editor_property("srgb")
                      and t.get_editor_property("compression_settings")
                      == unreal.TextureCompressionSettings.TC_NORMALMAP)
            elif kind == "arm":
                ok = (not t.get_editor_property("srgb")
                      and t.get_editor_property("compression_settings")
                      == unreal.TextureCompressionSettings.TC_MASKS)
            else:
                ok = bool(t.get_editor_property("srgb"))
            if not ok:
                raise RuntimeError(
                    f"{p} has the wrong sampler settings for a {kind} map (srgb="
                    f"{t.get_editor_property('srgb')}, compression="
                    f"{t.get_editor_property('compression_settings')}). Bound to a mismatched sampler type the "
                    f"WHOLE material fails to compile and renders as the grey checker. Fix the texture first.")
            return t
    raise RuntimeError(f"missing texture: none of {[f'{TEXROOT}/{pid}/{n}' for n in names]} exist")


# DUST COLOUR IS LINEAR, NOT sRGB. This was wrong on the first run and the render proved it: the dust was
# authored as (0.74, 0.70, 0.62), which reads as a pleasant tan in sRGB but is what the shader consumes
# LINEARLY - linear 0.74 is sRGB 0.885, near white. dirty_concrete's own albedo is linear (0.257, 0.236, 0.196)
# (measured from dirty_concrete_diff_2k.jpg), so a 50 % lerp to linear 0.74 landed the slabs at ~0.49 linear
# = sRGB 0.73 before any sun, and blew out to featureless white in daylight - the opposite of Reference B's
# "desaturated tan/grey, very low colour contrast". These are the same tan converted properly:
#     sRGB (0.68, 0.66, 0.61) -> linear ((c + 0.055) / 1.055) ** 2.4
# The dust is also close to NEUTRAL, not tan. dirty_concrete's own albedo is already warm (linear chroma 0.061
# on a 0.230 mean) and the scene's sun is low and warm, so a tan dust on top rendered the slabs at sRGB
# saturation 0.126 - measured, and over the 0.10 that "very low colour contrast" allows. Grey dust plus a
# faintly cool tint lets the scan's own warmth through without stacking three warm sources.
DUST_LINEAR = (0.40, 0.385, 0.36)

SETS = {"concrete": "dirty_concrete", "masonry": "brown_mud_rocks_01", "rebar": "rusty_corrugated_iron",
        "fabric": "painted_plaster_wall", "wood": "weathered_planks", "metal": "rusty_corrugated_iron"}
TEX = {k: {kind: tex(pid, kind) for kind in ("col", "nrm", "arm")} for k, pid in SETS.items()}
print("textures resolved:", {k: TEX[k]["nrm"].get_name() for k in TEX})


def reuse_or_create(path, cls, factory):
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    d, n = path.rsplit("/", 1)
    a = tools.create_asset(n, d, cls, factory)
    if a is None:
        raise RuntimeError(f"create_asset returned None for {path} (is an actor still referencing it?)")
    return a


def assert_compiles(mat):
    """A material that fails to compile renders as the grey WorldGridMaterial checker and reports NOTHING
    through the Python API except zeroed statistics. Copied from build_buildings.py on purpose."""
    s = mel.get_statistics(mat)
    if s.num_pixel_shader_instructions == 0 or s.num_pixel_texture_samples == 0:
        raise RuntimeError(
            f"{mat.get_name()} FAILED TO COMPILE (instructions={s.num_pixel_shader_instructions}, "
            f"texture samples={s.num_pixel_texture_samples}). Check sampler type vs default texture sRGB.")
    return s


# --- master material ------------------------------------------------------------------------------------------
master = reuse_or_create(f"{MATDIR}/M_Rubble_Master", unreal.Material, unreal.MaterialFactoryNew())
for e in list(mel.get_material_expressions(master)):
    mel.delete_material_expression(master, e)
mel.delete_unused_expressions(master)
if mel.get_num_material_expressions(master):
    # delete_all_material_expressions does NOT clear everything (a terrain graph grew 64 -> 95 -> 111 nodes over
    # three reruns), so the count is asserted rather than trusted.
    raise RuntimeError(f"master graph did not clear: {mel.get_num_material_expressions(master)} nodes left")

_y = [0]


def node(cls, x=-800, **props):
    _y[0] += 130
    e = mel.create_material_expression(master, cls, x, _y[0])
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(a, ao, b, bi):
    if not mel.connect_material_expressions(a, ao, b, bi):
        raise RuntimeError(f"connect {a.get_name()}.{ao or 'out'} -> {b.get_name()}.{bi}")


S = unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS
uv0 = node(unreal.MaterialExpressionTextureCoordinate, -1500)
uvs = node(unreal.MaterialExpressionScalarParameter, -1500, parameter_name="UVScale", default_value=1.0)
uvm = node(unreal.MaterialExpressionMultiply, -1300)
link(uv0, "", uvm, "A")
link(uvs, "", uvm, "B")

# Every parameter's DEFAULT texture matches its sampler type. The engine's own DefaultDiffuse is sRGB and under
# SAMPLERTYPE_MASKS it silently killed every building material on 2026-09-10.
col = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="BaseColor",
           texture=TEX["concrete"]["col"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_COLOR,
           sampler_source=S)
nrm = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="Normal",
           texture=TEX["concrete"]["nrm"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL,
           sampler_source=S)
arm = node(unreal.MaterialExpressionTextureSampleParameter2D, -1100, parameter_name="ARM",
           texture=TEX["concrete"]["arm"], sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_MASKS,
           sampler_source=S)
for t in (col, nrm, arm):
    link(uvm, "", t, "UVs")

tint = node(unreal.MaterialExpressionVectorParameter, -900, parameter_name="Tint",
            default_value=unreal.LinearColor(1, 1, 1, 1))
tcol = node(unreal.MaterialExpressionMultiply, -700)
link(col, "", tcol, "A")
link(tint, "", tcol, "B")

# Dust. SCENE_REFERENCE Reference B: "dust-covered, desaturated tan/grey, very low colour contrast". Concrete
# dust also flattens the surface normal and raises roughness, so all three are driven from one scalar.
dust_a = node(unreal.MaterialExpressionScalarParameter, -900, parameter_name="DustAmount", default_value=0.38)
dust_c = node(unreal.MaterialExpressionVectorParameter, -900, parameter_name="DustColor",
              default_value=unreal.LinearColor(*DUST_LINEAR, 1.0))
base_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(tcol, "", base_lerp, "A")
link(dust_c, "", base_lerp, "B")
link(dust_a, "", base_lerp, "Alpha")

flat_n = node(unreal.MaterialExpressionConstant3Vector, -900)
flat_n.set_editor_property("constant", unreal.LinearColor(0.0, 0.0, 1.0, 1.0))
n_damp = node(unreal.MaterialExpressionMultiply, -700)
n_damp_k = node(unreal.MaterialExpressionConstant, -900)
n_damp_k.set_editor_property("r", 0.45)
link(dust_a, "", n_damp, "A")
link(n_damp_k, "", n_damp, "B")
n_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(nrm, "", n_lerp, "A")
link(flat_n, "", n_lerp, "B")
link(n_damp, "", n_lerp, "Alpha")

rough_dust = node(unreal.MaterialExpressionConstant, -900)
rough_dust.set_editor_property("r", 0.93)
r_damp = node(unreal.MaterialExpressionMultiply, -700)
r_damp_k = node(unreal.MaterialExpressionConstant, -900)
r_damp_k.set_editor_property("r", 0.7)
link(dust_a, "", r_damp, "A")
link(r_damp_k, "", r_damp, "B")
r_lerp = node(unreal.MaterialExpressionLinearInterpolate, -500)
link(arm, "G", r_lerp, "A")
link(rough_dust, "", r_lerp, "B")
link(r_damp, "", r_lerp, "Alpha")

mel.connect_material_property(base_lerp, "", unreal.MaterialProperty.MP_BASE_COLOR)
mel.connect_material_property(n_lerp, "", unreal.MaterialProperty.MP_NORMAL)
mel.connect_material_property(arm, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
mel.connect_material_property(r_lerp, "", unreal.MaterialProperty.MP_ROUGHNESS)
mel.connect_material_property(arm, "B", unreal.MaterialProperty.MP_METALLIC)
mel.recompile_material(master)
eal.save_asset(master.get_path_name())
_s = assert_compiles(master)
print(f"M_Rubble_Master compiles: {_s.num_pixel_shader_instructions} instructions, "
      f"{_s.num_pixel_texture_samples} texture samples")


def mi(name, texkey, tint_rgb, dust, uv_scale=1.0, dust_rgb=DUST_LINEAR):
    path = f"{MATDIR}/{name}"
    inst = reuse_or_create(path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(inst, master)
    mel.set_material_instance_texture_parameter_value(inst, "BaseColor", TEX[texkey]["col"])
    mel.set_material_instance_texture_parameter_value(inst, "Normal", TEX[texkey]["nrm"])
    mel.set_material_instance_texture_parameter_value(inst, "ARM", TEX[texkey]["arm"])
    mel.set_material_instance_vector_parameter_value(inst, "Tint", unreal.LinearColor(*tint_rgb, 1.0))
    mel.set_material_instance_vector_parameter_value(inst, "DustColor", unreal.LinearColor(*dust_rgb, 1.0))
    mel.set_material_instance_scalar_parameter_value(inst, "DustAmount", dust)
    mel.set_material_instance_scalar_parameter_value(inst, "UVScale", uv_scale)
    mel.update_material_instance(inst)     # rebuild cached uniform expressions, else the render lags the asset
    eal.save_asset(path)
    return inst


# Tints multiply the scan, so they stay near 1 for the greys; the belongings keep their colour on purpose -
# Reference B's only colour is "scattered belongings among the grey". Dust amounts are lower than the first
# run's because with a correctly-linear dust colour the wash no longer has to be weak to avoid going white,
# and every point of dust is a point of the concrete's own grain lost.
# UVScale: gen_rubble.py writes UVs in TILES of TILE_M metres (concrete 2.0 m, masonry 1.0 m). Most slab and
# block faces are 1-3 m across, so at UVScale 1.0 each face mapped very nearly ONE whole copy of the 2K scan -
# and since every face got the same copy, every plate in the field wore the same concentric stain. It was
# plainly visible in the 45 m nadir (the survey altitude that matters) as a repeated rectangular motif on every
# slab. Scaling the UVs up puts several tiles across a face, so the scan reads as grain rather than as a motif.
SLOT = {
    "concrete": mi("MI_Rubble_Concrete", "concrete", (0.88, 0.89, 0.92), 0.38, uv_scale=2.5),
    "masonry": mi("MI_Rubble_Masonry", "masonry", (0.86, 0.86, 0.86), 0.36, uv_scale=2.0),
    "rebar": mi("MI_Rubble_Rebar", "rebar", (0.80, 0.52, 0.34), 0.15, uv_scale=1.0),
    "wood": mi("MI_Rubble_Wood", "wood", (0.92, 0.86, 0.78), 0.30, uv_scale=1.5),
    "metal": mi("MI_Rubble_Metal", "metal", (0.88, 0.88, 0.90), 0.26, uv_scale=1.5),
    "fabric": mi("MI_Rubble_Fabric_A", "fabric", (0.24, 0.40, 0.66), 0.22, uv_scale=1.0),
}
FABRIC_B = mi("MI_Rubble_Fabric_B", "fabric", (0.66, 0.26, 0.22), 0.22, uv_scale=1.0)
for _n, _m in list(SLOT.items()) + [("fabric_b", FABRIC_B)]:
    assert_compiles(_m)
print(f"all {len(SLOT) + 1} rubble materials compile")


# --- meshes -----------------------------------------------------------------------------------------------------
def import_file(path, dest, name):
    t = unreal.AssetImportTask()
    for k, v in (("filename", path), ("destination_path", dest), ("destination_name", name),
                 ("automated", True), ("replace_existing", True), ("save", True)):
        t.set_editor_property(k, v)
    tools.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.get_editor_property("imported_object_paths")]


meshes, tri_report, bad_tris, unknown = {}, {}, [], []
for name, info in plan["variants"].items():
    objs = import_file(fr"{REPO}\data\scene\rubble\{name}.obj", MESHDIR, f"SM_{name}")
    sm = next((o for o in objs if isinstance(o, unreal.StaticMesh)), None)
    if sm is None:
        raise RuntimeError(f"{name}.obj did not import as a StaticMesh (got {objs})")
    ns = sm.get_editor_property("nanite_settings")     # imports arrive with Nanite ON; the project runs GI=None
    ns.set_editor_property("enabled", False)
    sm.set_editor_property("nanite_settings", ns)
    if RUBBLE_COLLISION:
        bs = sm.get_editor_property("body_setup")
        bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        sm.set_editor_property("body_setup", bs)
    for i, sl in enumerate(sm.get_editor_property("static_materials")):
        nm = str(sl.get_editor_property("material_slot_name")).lower()
        key = next((k for k in SLOT if nm.startswith(k)), None)
        if key is None:
            unknown.append((name, nm))
            continue
        sm.set_material(i, SLOT[key])
    eal.save_asset(sm.get_path_name())
    got = sm.get_num_triangles(0)
    tri_report[name] = (got, info["tris"])
    # A Nanite-enabled mesh reports its FALLBACK here (347 triangles for a 524k asset), so this doubles as proof
    # that Nanite really is off. Anything else means the import lost geometry.
    if abs(got - info["tris"]) > max(8, 0.02 * info["tris"]):
        bad_tris.append((name, got, info["tris"]))
    meshes[name] = sm

if unknown:
    raise RuntimeError(f"unknown material slots (expected {sorted(SLOT)}): {unknown[:5]}")
if bad_tris:
    raise RuntimeError(
        "imported triangle counts do not match the generator (name, imported, expected): "
        f"{bad_tris[:5]} -- either Nanite is still on (the fallback mesh is reported) or the import lost "
        "geometry. Do NOT place this field.")
print(f"imported {len(meshes)} rubble meshes, Nanite off, triangle counts match the generator "
      f"(total {sum(v[0] for v in tri_report.values()):,})")


# --- place ----------------------------------------------------------------------------------------------------
# unreal.Transform's `rotation` field is a QUAT, not a Rotator (UE 5.8), and its constructor argument order is
# not the property order. Assign through set_editor_property, convert the rotator explicitly, and PROVE the
# builder once - a silently-identity rotation would lay 1,799 slabs perfectly flat and nothing would report it.
def _rot_to_quat(rot):
    try:
        return rot.quaternion()
    except Exception:                                               # noqa: BLE001
        return unreal.MathLibrary.conv_rotator_to_quaternion(rot)


def _quat_to_rot(q):
    try:
        return q.rotator()
    except Exception:                                               # noqa: BLE001
        return unreal.MathLibrary.conv_quaternion_to_rotator(q)


def xform(rec):
    """World transform for one layout record. yaw_deg is measured from east towards north (the gen_buildings
    convention) and the OBJ importer flips handedness, so the yaw is 90 - yaw_deg exactly as in
    build_buildings.py and in the actor-per-item version of this script."""
    t = unreal.Transform()
    t.set_editor_property("translation", unreal.Vector(rec["north_m"] * 100.0, rec["east_m"] * 100.0,
                                                       (rec["base_asl_m"] - BASE_Z) * 100.0))
    t.set_editor_property("rotation", _rot_to_quat(
        unreal.Rotator(rec["roll_deg"], rec["pitch_deg"], 90.0 - rec["yaw_deg"])))
    s = float(rec["scale"])
    t.set_editor_property("scale3d", unreal.Vector(s, s, s))
    return t


_p = {"north_m": 12.34, "east_m": -56.78, "base_asl_m": BASE_Z + 9.0,
      "roll_deg": 3.0, "pitch_deg": -11.0, "yaw_deg": 40.0, "scale": 1.25}
_t = xform(_p)
_tr = _t.get_editor_property("translation")
_rr = _quat_to_rot(_t.get_editor_property("rotation"))
_sc = _t.get_editor_property("scale3d")
if (abs(_tr.x - 1234.0) > 0.5 or abs(_tr.y + 5678.0) > 0.5 or abs(_tr.z - 900.0) > 0.5
        or abs(_sc.x - 1.25) > 1e-3 or abs(_rr.roll - 3.0) > 0.05 or abs(_rr.pitch + 11.0) > 0.05
        or abs(_rr.yaw - 50.0) > 0.05):                             # 90 - 40 = 50
    raise RuntimeError(f"transform builder is wrong: t=({_tr.x:.1f},{_tr.y:.1f},{_tr.z:.1f}) "
                       f"rot=(roll {_rr.roll:.2f}, pitch {_rr.pitch:.2f}, yaw {_rr.yaw:.2f}) scale {_sc.x:.3f}; "
                       f"expected (1234,-5678,900), roll 3 pitch -11 yaw 50, scale 1.25")
print(f"transform builder proved: (1234,-5678,900) cm, roll 3 pitch -11 yaw 50, scale 1.25")

removed = 0
for a in list(eas.get_all_level_actors()):
    if str(a.get_folder_path()) == FOLDER:
        eas.destroy_actor(a)
        removed += 1
if removed:
    unreal.SystemLibrary.collect_garbage()

# The actor-per-item version overrode the fabric slot on every odd-id belonging so the field carries two fabric
# colours. An instance cannot carry a material override, so the two colours become two components instead.
FABRIC_IDX = {n: [i for i, sl in enumerate(sm.get_editor_property("static_materials"))
                  if str(sl.get_editor_property("material_slot_name")).lower().startswith("fabric")]
              for n, sm in meshes.items()}

buckets = {}
for rec in plan["items"]:
    alt = bool(rec["family"] == "belonging" and rec["id"] % 2 and FABRIC_IDX[rec["variant"]])
    buckets.setdefault((rec["variant"], alt), []).append(rec)
if len(buckets) > MAX_ACTORS:
    raise RuntimeError(f"{len(buckets)} actors is over the {MAX_ACTORS} tripwire - this lane instances, it does "
                       f"not spawn one actor per item. NOTHING WAS PLACED.")
print(f"placing {len(plan['items']):,} instances across {len(buckets)} actors (tripwire {MAX_ACTORS})")

placed, comps = 0, 0
for (vname, alt), recs in sorted(buckets.items()):
    sm = meshes[vname]
    label = f"Rubble_{vname}" + ("_altfabric" if alt else "")
    act = eas.spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
    if act is None:
        raise RuntimeError(f"could not spawn the container actor for {label}")
    # UE 5.8's Python bindings do NOT expose Actor.add_component_by_class (checked live: it is absent from
    # dir(unreal.Actor), and the call raises AttributeError). SubobjectDataSubsystem is the path the Details
    # panel's "+ Add Component" takes and it produces an INSTANCE component that serialises with the actor.
    # Same route as build_vegetation.py, which verified persistence by saving, reloading the map from disk and
    # reading the instances back.
    handles = sds.k2_gather_subobject_data_for_instance(act)
    new_handle, why = sds.add_new_subobject(unreal.AddNewSubobjectParams(
        parent_handle=handles[0], new_class=unreal.HierarchicalInstancedStaticMeshComponent,
        blueprint_context=None, conform_transform_to_parent=True))
    if str(why):
        raise RuntimeError(f"{label}: add_new_subobject refused: {why}")
    sds.rename_subobject(new_handle, "Rubble")
    comp = act.get_component_by_class(unreal.HierarchicalInstancedStaticMeshComponent)
    if comp is None:
        raise RuntimeError(f"{label}: no HISM on the actor after add_new_subobject - the instances would "
                           f"silently not exist and the actor would be an empty shell")
    try:
        comp.set_static_mesh(sm)
    except Exception:                                               # noqa: BLE001
        comp.set_editor_property("static_mesh", sm)
    if comp.get_editor_property("static_mesh") != sm:
        raise RuntimeError(f"{label}: the HISM did not take the mesh {sm.get_name()}")
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    if RUBBLE_COLLISION:
        comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
    else:
        comp.set_collision_profile_name("NoCollision")
        comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    if alt:
        for i in FABRIC_IDX[vname]:
            comp.set_material(i, FABRIC_B)
    act.set_actor_label(label)                                      # label only - never rename()
    act.set_folder_path(FOLDER)
    act.tags = ["Rubble", recs[0]["family"], vname] + (["altfabric"] if alt else [])
    # The instance transforms are WORLD coordinates, so the container must sit at the identity. Assert it
    # rather than assume it: a stray root transform would offset the entire field.
    al, ar, asx = act.get_actor_location(), act.get_actor_rotation(), act.get_actor_scale3d()
    if (abs(al.x) + abs(al.y) + abs(al.z) > 1.0 or abs(ar.pitch) + abs(ar.yaw) + abs(ar.roll) > 0.01
            or abs(asx.x - 1) + abs(asx.y - 1) + abs(asx.z - 1) > 1e-3):
        raise RuntimeError(f"{label}: container actor is not at the identity")

    n = 0
    for i in range(0, len(recs), BATCH):
        chunk = [xform(r) for r in recs[i:i + BATCH]]
        comp.add_instances(chunk, False, True)       # world_space=True: the transforms are world coordinates
        n += len(chunk)
        del chunk
        unreal.SystemLibrary.collect_garbage()
        mem_guard(f"{label} after {n}/{len(recs)}")
    got = comp.get_instance_count()
    if got != len(recs):
        raise RuntimeError(f"{label}: added {n} instances but the component reports {got}")
    # Read one instance straight back out of the component in WORLD space and compare it to the record it came
    # from. get_instance_count() alone would pass on a field of instances heaped at (0,0,0).
    want = xform(recs[0]).get_editor_property("translation")
    gt = comp.get_instance_transform(0, True).get_editor_property("translation")
    if max(abs(gt.x - want.x), abs(gt.y - want.y), abs(gt.z - want.z)) > 1.0:
        raise RuntimeError(
            f"{label}: instance 0 reads back at ({gt.x:.1f}, {gt.y:.1f}, {gt.z:.1f}) cm but the layout says "
            f"({want.x:.1f}, {want.y:.1f}, {want.z:.1f}) cm - the whole field is misplaced.")
    placed += got
    comps += 1
    print(f"  {label:34s} {got:5d} instances  mesh={sm.get_name()}")

n_now = sum(1 for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER)
if n_now != comps:
    raise RuntimeError(f"spawned {comps} instancer actors but the {FOLDER} folder holds {n_now}")
if placed != len(plan["items"]):
    raise RuntimeError(f"placed {placed} instances, the layout has {len(plan['items'])}")

MEM1 = memline("after placing")
saved = les.save_current_level()
MEM2 = memline("after save")
c = plan["counts"]
print(f"removed {removed} old actors, placed {placed} rubble INSTANCES across {comps} HISM actors "
      f"(~{c['approx_triangles']:,} triangles, budget {c['triangle_budget']:,}) | "
      f"collision {'QUERY_ONLY' if RUBBLE_COLLISION else 'OFF'} | saved {saved}")
print(f"  total level actors now: {len(eas.get_all_level_actors())}")
print(f"  memory delta: AvailableVirtual {MEM0['AvailableVirtual']:.2f} -> {MEM2['AvailableVirtual']:.2f} GiB, "
      f"UsedVirtual {MEM0['UsedVirtual']:.2f} -> {MEM2['UsedVirtual']:.2f} GiB "
      f"(peak {MEM2['PeakUsedVirtual']:.2f} GiB, commit limit {MEM2['CommitLimit']:.2f} GiB)")
print(f"  by group : {c['by_group']}")
print(f"  by family: {c['by_family']}")
print("NOW LOOK: run tools/scene/qa_shots.py and tools/scene/qa_rubble.py and open the images. Check "
      "specifically that the rubble is NOT grey checker (material compile), that the slabs are metre-scale and "
      "not 100x small (OBJ units), and that the survivors on the fan sit in voids rather than inside a pile.")
