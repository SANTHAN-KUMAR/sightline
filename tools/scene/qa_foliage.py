"""Render the foliage QA views and dump the tree meshes' REAL material state (running editor, PIE OFF).

    FOLIAGE_TAG='before'; exec(open(r'D:\\Sightline\\tools\\scene\\qa_foliage.py').read())
    FOLIAGE_TAG='after';  exec(open(r'D:\\Sightline\\tools\\scene\\qa_foliage.py').read())
    uv run python tools/scene/check_foliage_materials.py after     # host side: the half that can say NO

Places nothing, moves nothing, saves nothing. It reads the level and writes PNGs plus one JSON.

WHAT IT PRODUCES, AND WHY EACH FRAME EXISTS
-------------------------------------------
The red team's evidence for the leaf-material defect was a comparison inside ONE frame: "sunlit grass brilliant
green, canopy beside it near-black at the same exposure". To turn that into a check that can fail, a pixel has
to be classifiable as canopy or as ground WITHOUT using its brightness -- otherwise the test is circular (it
would define canopy as "the dark bit" and then discover that the dark bit is dark).

So every view is captured FOUR times from the identical camera:

    <view>_beauty.png       SCS_FINAL_COLOR_LDR, everything visible          <- what is measured
    <view>_noveg.png        SCS_FINAL_COLOR_LDR, Vegetation folder hidden    <- the same scene, no trees
    <view>_nrm.png          SCS_NORMAL,          everything visible          <- geometry, not lighting
    <view>_nrm_noveg.png    SCS_NORMAL,          Vegetation folder hidden

`hide_actor_components()` on the capture component is what removes the canopy; it is given the 9 HISM container
actors, so all 5,508 instances go at once and nothing in the level is touched. (The `hidden_actors` PROPERTY
looks like the obvious route and is not usable here: the editor refuses it with "Property 'HiddenActors' for
attribute 'hidden_actors' on 'SceneCaptureComponent2D' cannot be edited on templates".)

The canopy mask is then `normal changed between nrm and nrm_noveg`. World normals depend on geometry alone, so
a leaf in front of ground reads differently from the ground behind it whatever its albedo or its lighting --
which is exactly the property a non-circular mask needs. The GROUND reference additionally requires the beauty
frames to agree, which throws away any ground pixel whose lighting changed when the trees were removed, i.e.
every tree SHADOW. Without that second condition the shadow of a crown would be scored as crown and the
measurement would be biased dark by the very thing it is trying to measure.

Two crown views, both low obliques, differing only in which side of the tree the camera is on:

    fol_2_crown_backlit     camera on the sun's side of the crown, looking back into it with the sun BEHIND it
    fol_3_crown_frontlit    camera with the sun behind IT, the same crown lit from the front

Backlit is the frame that matters. A single-sided lit leaf with no subsurface term has nothing to return to
the camera when the light is on the far side, so it goes black; MSM_TWO_SIDED_FOLIAGE plus a subsurface colour
is precisely the fix for that. The front-lit frame is the control: if both frames change, something global
changed; if only the backlit frame changes, the transmission term is doing what it claims.
"""

import ctypes
import json
import os
import time

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\foliage"
FOLDER = "Vegetation"
W, H = 1600, 900
TAG = globals().get("FOLIAGE_TAG", "current")

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: captures and material statistics are unreliable while PIE runs")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
os.makedirs(OUT, exist_ok=True)


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
    return {"AvailablePhysical": ms.ullAvailPhys / G, "AvailableVirtual": ms.ullAvailPageFile / G,
            "CommitLimit": ms.ullTotalPageFile / G}


def memline(tag):
    m = mem()
    print(f"  [mem] {tag:16s} AvailablePhysical {m['AvailablePhysical']:.2f} GiB  "
          f"AvailableVirtual {m['AvailableVirtual']:.2f} GiB  (commit limit {m['CommitLimit']:.2f} GiB)")
    return m


MEM0 = memline(f"{TAG} start")

with open(REPO + r"\data\scene\vegetation.json") as fh:
    plan = json.load(fh)
with open(REPO + r"\data\scene\foliage_qa_views.json") as fh:
    views = json.load(fh)
BASE_Z = plan["base_z_m"]
CAT = plan["catalogue"]
TREE_PIDS = sorted({it["pid"] for it in plan["items"]})


def ue(north_m, east_m, asl_m):
    return unreal.Vector(north_m * 100.0, east_m * 100.0, (asl_m - BASE_Z) * 100.0)


# --- 1. dump the material state of the five tree meshes -------------------------------------------------------
def bbox_volume(sm):
    b = sm.get_bounding_box()
    return (b.max.x - b.min.x) * (b.max.y - b.min.y) * (b.max.z - b.min.z)


def resolve_mesh(pid):
    """Same resolution build_vegetation.py used: hint first, then the biggest StaticMesh under the prop
    folder by bounding volume (get_num_triangles reports the Nanite fallback, so it cannot be used here)."""
    hint = CAT[pid].get("asset_hint")
    if hint and eal.does_asset_exist(hint):
        a = unreal.load_asset(hint)
        if isinstance(a, unreal.StaticMesh):
            return a
    folder = f"/Game/Sightline/Props/{pid}"
    found = [unreal.load_asset(p) for p in eal.list_assets(folder, recursive=True, include_folder=False)]
    found = [a for a in found if isinstance(a, unreal.StaticMesh)]
    if not found:
        raise RuntimeError(f"{pid}: no static mesh under {folder}")
    return max(found, key=bbox_volume)


def ename(v):
    """The bare name of a UE enum value. `str(v)` is '<BlendMode.BLEND_OPAQUE: 0>', so splitting on '.' alone
    leaves 'BLEND_OPAQUE: 0>' -- which then never compares equal to anything, and every check that reads it
    fails for a reason that has nothing to do with the scene."""
    n = getattr(v, "name", None)
    if n:
        return str(n)
    return str(v).strip("<>").split(":")[0].split(".")[-1]


def chain_of(mat):
    """Every material asset this slot depends on, instance first, root Material last. The defect is a project
    material instance whose PARENT lives in an engine plugin, so the leaf asset's own path proves nothing --
    the whole chain has to be reported."""
    out, seen, cur = [], set(), mat
    while cur is not None and cur.get_path_name() not in seen:
        seen.add(cur.get_path_name())
        out.append(cur.get_path_name())
        cur = cur.get_editor_property("parent") if isinstance(cur, unreal.MaterialInstanceConstant) else None
    return out


def blend_of(mat):
    """The instance's own blend mode AND its parent's. They can disagree: `build_vegetation.py` set an
    OPAQUE override on every leaf instance and the renderer still blends, so the override alone is not
    evidence of what gets drawn. Both are reported and the check looks at both."""
    parent_blend = "?"
    base = mat.get_base_material() if isinstance(mat, unreal.MaterialInstanceConstant) else mat
    if base is not None:
        parent_blend = ename(base.get_editor_property("blend_mode"))
    if isinstance(mat, unreal.MaterialInstanceConstant):
        ov = mat.get_editor_property("base_property_overrides")
        if ov.get_editor_property("override_blend_mode"):
            return ename(ov.get_editor_property("blend_mode")), True, parent_blend
        return parent_blend, False, parent_blend
    return ename(mat.get_editor_property("blend_mode")), False, parent_blend


def two_sided_of(mat):
    if isinstance(mat, unreal.MaterialInstanceConstant):
        ov = mat.get_editor_property("base_property_overrides")
        if ov.get_editor_property("override_two_sided"):
            return bool(ov.get_editor_property("two_sided")), True
        base = mat.get_base_material()
        return (bool(base.get_editor_property("two_sided")) if base else False), False
    return bool(mat.get_editor_property("two_sided")), False


def shading_of(mat):
    base = mat.get_base_material() if isinstance(mat, unreal.MaterialInstanceConstant) else mat
    if base is None:
        return "?"
    return ename(base.get_editor_property("shading_model"))


def textures_of(mat):
    """Which texture assets this slot actually resolves to, by parameter name."""
    got = {}
    if isinstance(mat, unreal.MaterialInstanceConstant):
        try:
            for pv in mat.get_editor_property("texture_parameter_values"):
                name = str(pv.get_editor_property("parameter_info").get_editor_property("name"))
                v = pv.get_editor_property("parameter_value")
                got[name] = v.get_path_name() if v else None
        except Exception as exc:                                    # noqa: BLE001 - editor API varies
            got["!error"] = str(exc)
    return got


def role_of(name):
    n = name.lower()
    return "leaf" if ("leaves" in n or "leaf" in n) else "bark"


state = {"tag": TAG, "when": time.time(), "when_local": time.strftime("%Y-%m-%d %H:%M:%S"),
         "level": world.get_name(), "meshes": []}
print(f"tree material state ({TAG}):")
for pid in TREE_PIDS:
    sm = resolve_mesh(pid)
    ns = sm.get_editor_property("nanite_settings")
    rec = {"pid": pid, "mesh_path": sm.get_path_name(),
           "nanite": bool(ns.get_editor_property("enabled")), "slots": []}
    for i, slot in enumerate(sm.get_editor_property("static_materials")):
        mi = slot.get_editor_property("material_interface")
        sname = str(slot.get_editor_property("material_slot_name"))
        if mi is None:
            rec["slots"].append({"index": i, "slot_name": sname, "material_path": None, "role": "?",
                                 "error": "EMPTY SLOT"})
            continue
        blend, blend_ov, parent_blend = blend_of(mi)
        ts, ts_ov = two_sided_of(mi)
        st = mel.get_statistics(mi)
        s = {"index": i, "slot_name": sname, "material_path": mi.get_path_name(),
             "material_name": mi.get_name(), "class": type(mi).__name__,
             "chain": chain_of(mi), "blend_mode": blend, "blend_overridden": blend_ov,
             "parent_blend_mode": parent_blend,
             "two_sided": ts, "two_sided_overridden": ts_ov, "shading_model": shading_of(mi),
             "instructions": int(st.num_pixel_shader_instructions),
             "texture_samples": int(st.num_pixel_texture_samples),
             "role": role_of(mi.get_name() + "|" + sname), "textures": textures_of(mi)}
        rec["slots"].append(s)
        print(f"  {pid:16s} [{i}] {s['role']:5s} {s['material_name']:32s} blend={blend:14s} "
              f"(parent {parent_blend:14s}) 2side={ts!s:5s} shade={s['shading_model']:24s} "
              f"{s['instructions']:4d} instr")
        for c in s["chain"][1:]:
            print(f"      parent -> {c}")
    state["meshes"].append(rec)

# --- 2. the capture rig ---------------------------------------------------------------------------------------
veg_actors = [a for a in eas.get_all_level_actors() if str(a.get_folder_path()) == FOLDER]
print(f"\nVegetation folder holds {len(veg_actors)} container actors "
      f"(these are what `hidden_actors` removes for the no-vegetation passes)")
if not veg_actors:
    raise RuntimeError(f"no actors in the {FOLDER} folder - the canopy is not in this level")

sun = None
for a in eas.get_all_level_actors():
    if isinstance(a, unreal.DirectionalLight):
        sun = a
        break
if sun is None:
    for a in eas.get_all_level_actors():
        if a.get_component_by_class(unreal.DirectionalLightComponent) is not None:
            sun = a
            break
if sun is None:
    raise RuntimeError("no DirectionalLight in the level: a backlit crown view cannot be aimed")
fwd = sun.get_actor_forward_vector()
print(f"sun: {sun.get_actor_label()} forward=({fwd.x:.3f}, {fwd.y:.3f}, {fwd.z:.3f}) "
      f"(the direction light TRAVELS; the backlit camera sits on the far side of the crown from here)")
state["sun_forward"] = [fwd.x, fwd.y, fwd.z]
state["sun_label"] = sun.get_actor_label()

rt_ldr = unreal.RenderingLibrary.create_render_target2d(world, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("FOLIAGE_QA_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt_ldr)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)

written = []


def shot(name, loc, rot, fov, source, hide):
    comp.set_editor_property("fov_angle", fov)
    comp.set_editor_property("capture_source", source)
    # NOT `hidden_actors`: that property is flagged non-editable on templates and the editor refuses it
    # ("Property 'HiddenActors' ... cannot be edited on templates"). hide_actor_components /
    # clear_hidden_components are BlueprintCallable and take the same effect on the live component.
    comp.clear_hidden_components()
    for _a in hide:
        comp.hide_actor_components(_a)
    cap.set_actor_location_and_rotation(loc, rot, False, False)
    comp.capture_scene()
    comp.capture_scene()                        # second pass: temporal effects settle
    unreal.RenderingLibrary.export_render_target(world, rt_ldr, OUT, name + ".png")
    written.append(name + ".png")
    return os.path.join(OUT, name + ".png")


def view(name, loc, rot, fov):
    """One view, four passes. See the module docstring for why all four are needed."""
    LDR = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
    NRM = unreal.SceneCaptureSource.SCS_NORMAL
    shot(f"{name}_beauty", loc, rot, fov, LDR, [])
    shot(f"{name}_noveg", loc, rot, fov, LDR, veg_actors)
    shot(f"{name}_nrm", loc, rot, fov, NRM, [])
    shot(f"{name}_nrm_noveg", loc, rot, fov, NRM, veg_actors)
    print(f"  {name}: 4 passes at ({loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f}) cm "
          f"rot(p={rot.pitch:.1f}, y={rot.yaw:.1f}) fov {fov:.0f}")
    return {"name": name, "loc_cm": [loc.x, loc.y, loc.z],
            "rot": [rot.roll, rot.pitch, rot.yaw], "fov_deg": fov}


try:
    nd = views["nadir"]
    cr = views["crown"]
    FOV = views["fov_deg"]
    shots_meta = []

    # view 1: nadir at survey altitude over hillslope canopy
    shots_meta.append(view("fol_1_nadir",
                           ue(nd["north_m"], nd["east_m"], nd["cam_asl_m"]),
                           unreal.Rotator(0, -90, 0), FOV))

    # views 2 and 3: low obliques into one crown, from opposite sides of the sun
    ctr = ue(cr["north_m"], cr["east_m"], cr["centre_asl_m"])
    hx, hy = fwd.x, fwd.y
    hn = max(1e-6, (hx * hx + hy * hy) ** 0.5)
    hx, hy = hx / hn, hy / hn
    d_cm = cr["radius_m"] * 100.0 * 1.7
    drop_cm = cr["radius_m"] * 100.0 * 0.35       # sit below crown centre: look UP into the foliage, not down on it
    for label, sgn in (("fol_2_crown_backlit", +1.0), ("fol_3_crown_frontlit", -1.0)):
        loc = unreal.Vector(ctr.x + hx * d_cm * sgn, ctr.y + hy * d_cm * sgn, ctr.z - drop_cm)
        rot = unreal.MathLibrary.find_look_at_rotation(loc, ctr)
        shots_meta.append(view(label, loc, rot, 50.0))
finally:
    eas.destroy_actor(cap)

state["views"] = shots_meta
state["view_plan"] = views
state["images"] = written
state["mem"] = {"before": MEM0, "after": memline(f"{TAG} end")}

with open(os.path.join(OUT, f"state_{TAG}.json"), "w") as fh:
    json.dump(state, fh, indent=1)
print(f"\n{len(written)} PNGs + state_{TAG}.json written to {OUT}")
print(f"NOW RUN: uv run python tools/scene/check_foliage_materials.py {TAG}")
print("AND LOOK at fol_1_nadir_beauty.png and fol_2_crown_backlit_beauty.png.")
