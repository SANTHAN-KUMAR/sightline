"""Measure, do not guess: sweep the Nanite levers that could bring distant crowns back (PIE OFF).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\sweep_nanite_foliage.py').read())"
    uv run python tools/scene/check_sky.py --suffix _mppe025 --metrics-only   # etc, host side

At the 450 m demo camera the 4,342 crowns render as brown skeletons. The mechanism is visible in the mesh
data rather than inferable from the picture: every tree has an OPAQUE branch/trunk material and a
**BLEND_MASKED** leaf material. Nanite simplification merges clusters and interpolates UVs, so the
alpha-tested leaf surfaces stop passing the mask test long before the opaque woody structure disappears.
What survives at 8 px is exactly what we see - the branches.

Two independent hypotheses predict the same picture, so they are separated by experiment:

  H1 screen-space LOD    `r.Nanite.MaxPixelsPerEdge` (default 1.0) gives a tree that is 8 px tall about
                         8 px worth of geometry. Lowering it buys detail everywhere, for GPU time.
  H2 streaming residency `r.Nanite.Streaming.StreamingPoolSize` is 512 MB and these five meshes carry
                         10.7 M unique Nanite triangles. At 45 m a handful of trees are on screen and get
                         full detail; at 450 m all 4,342 are, and an exhausted pool would serve every one
                         of them from its coarse root pages. That ALSO looks like "fine near, skeletal far".

If H1 is the cause, MaxPixelsPerEdge moves the number and pool size does not. If H2 is the cause, the
reverse. If neither moves it, the cause is the build-time simplification itself and the lever is
`shape_preservation` on the meshes (NaniteShapePreservation.VOXELIZE is documented as "simplify triangles
to voxels in the distance to preserve the perceived volume of the object. Useful for foliage that thins out
otherwise"), which costs a Nanite rebuild rather than a cvar.

Cost is measured, not assumed: each variant reports its capture wall time and the machine's free physical
and commit memory, because the budget here is an 8 GB laptop GPU and a machine that has already killed the
editor once at 27.49 GiB of commit. A variant that fixes the picture and doubles the frame time is a
finding, not a fix, and is reported as such.

Renders only `lod_far`, at exactly the qa_1_valley camera, one PNG per variant into _artifacts/sky_lane.
Restores every cvar it touched in a finally block.
"""

import ctypes
import time

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\sky_lane"
MIN_AVAIL_VIRTUAL_GIB = 0.60          # abort rather than take the editor down with us
MIN_AVAIL_PHYSICAL_GIB = 0.30

#: DO NOT ADD r.Nanite.Streaming.StreamingPoolSize 2048 BACK.
#: It is not slow, it is FATAL, and it killed the editor mid-sweep on 2026-09-11 01:42:
#:     Fatal error: [NaniteStreamingManager.cpp] [Line: 640]
#:     Streaming pool size (2048MB) must be smaller than the largest allocation supported by the
#:     graphics hardware (2048MB)
#: The engine requires strictly LESS THAN the GPU's maximum single allocation, and on this RTX 4060 Laptop
#: (8 GB) that maximum is exactly 2048 MB, so 2048 fails the assert. POOL_MAX_MB below is the guard.
POOL_MAX_MB = 1024

#: (suffix, {cvar: value}) - each applied on top of the restored baseline, never cumulatively
VARIANTS = [
    ("_base",     {}),
    ("_mppe050",  {"r.Nanite.MaxPixelsPerEdge": 0.5}),
    ("_mppe025",  {"r.Nanite.MaxPixelsPerEdge": 0.25}),
    ("_mppe010",  {"r.Nanite.MaxPixelsPerEdge": 0.1}),
    ("_pool1024", {"r.Nanite.Streaming.StreamingPoolSize": 1024}),
]
TOUCHED = ["r.Nanite.MaxPixelsPerEdge", "r.Nanite.Streaming.StreamingPoolSize"]

for _s, _c in VARIANTS:
    _p = _c.get("r.Nanite.Streaming.StreamingPoolSize")
    if _p is not None and _p > POOL_MAX_MB:
        raise RuntimeError(f"variant {_s} asks for a {_p} MB Nanite streaming pool, over the {POOL_MAX_MB} MB "
                           f"ceiling for this GPU. This is a hard engine assert, not a slowdown: it takes "
                           f"the editor down instantly and loses unsaved work.")


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


les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

sysl = unreal.SystemLibrary
BASE = {c: sysl.get_console_variable_float_value(c) for c in TOUCHED}
print("baseline cvars:", {k: round(v, 4) for k, v in BASE.items()})

rt = unreal.RenderingLibrary.create_render_target2d(world, 1600, 900,
                                                    unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("SWEEP_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)
comp.set_editor_property("fov_angle", 75.0)


def restore():
    for c, v in BASE.items():
        sysl.execute_console_command(world, f"{c} {v}")


rows = []
try:
    for suffix, cvars in VARIANTS:
        ap, av = mem()
        if av < MIN_AVAIL_VIRTUAL_GIB or ap < MIN_AVAIL_PHYSICAL_GIB:
            print(f"  ABORTING before {suffix}: AvailableVirtual {av:.2f} GiB / AvailablePhysical "
                  f"{ap:.2f} GiB is under the floor. Sweep stopped, cvars restored.")
            break
        restore()
        for c, v in cvars.items():
            sysl.execute_console_command(world, f"{c} {v}")
        got = {c: sysl.get_console_variable_float_value(c) for c in cvars}
        bad = {c: (v, got[c]) for c, v in cvars.items() if abs(got[c] - v) > 1e-6}
        if bad:
            raise RuntimeError(f"{suffix}: cvar did not take: {bad}. Measuring a variant that was never "
                               f"applied would be worse than not measuring at all.")
        cap.set_actor_location_and_rotation(unreal.Vector(-60000, -20000, 45000),
                                            unreal.Rotator(0, -22, 35), False, False)
        comp.capture_scene()                       # warm: let streaming settle before timing
        t0 = time.perf_counter()
        comp.capture_scene()
        dt = (time.perf_counter() - t0) * 1000.0
        unreal.RenderingLibrary.export_render_target(world, rt, OUT, f"lod_far{suffix}.png")
        ap2, av2 = mem()
        rows.append((suffix, cvars, dt, ap2, av2))
        print(f"  {suffix:10s} {str(cvars):58s} capture {dt:7.1f} ms   "
              f"availPhys {ap2:.2f} GiB  availVirt {av2:.2f} GiB")
finally:
    restore()
    eas.destroy_actor(cap)
    now = {c: sysl.get_console_variable_float_value(c) for c in TOUCHED}
    print("restored cvars:", {k: round(v, 4) for k, v in now.items()})
    if any(abs(now[c] - BASE[c]) > 1e-6 for c in TOUCHED):
        print("  !! a cvar did NOT restore - the level is left in a non-default render state")

print(f"\n{len(rows)} variants rendered to {OUT}. Measure them (host side, needs numpy).")
print("--lod-only because only lod_far is rendered per variant; the 45 m reference is per-BUILD and is")
print("taken from the unsuffixed lod_near.png, which a distance-LOD dial does not affect:")
for suffix, _c, _d, _p, _v in rows:
    print(f"  uv run python tools/scene/check_sky.py --suffix {suffix} --metrics-only --lod-only")
print("The capture times above are NOT a cost measurement: capture_scene() returns before the GPU is done.")
print("Then LOOK at the winner. A number that moved is not a crown that came back.")
