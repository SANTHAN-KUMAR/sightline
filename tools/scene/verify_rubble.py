"""Measure what the rubble + structural-damage lane actually put in the level, and write it down (editor side).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\verify_rubble.py').read())"
    uv run python tools/scene/check_rubble.py       # host side: judges this report, exits non-zero

This script DECIDES NOTHING. It walks the live level and dumps measurements to
`_artifacts/rubble/placement_report.json`; `tools/scene/check_rubble.py` compares that report against
`data/scene/rubble_layout.json` and `data/scene/damage.json` and fails if they disagree. Splitting it that way
is deliberate: the thing that measures runs where the truth is, and the thing that judges runs where it can
exit non-zero and be read in a diff.

What it measures, and why each one is a trap that has already bitten this project:
  * instance count per HISM component, and per mesh variant   - a `True` from add_instances() means nothing;
  * one instance transform per component, read back in WORLD space and compared with the layout - a component
    whose actor drifted off the origin gives a perfectly-sized field in the wrong place, silently;
  * `get_num_triangles(0)` per mesh - a Nanite-enabled import reports its FALLBACK here (347 triangles for a
    524k asset), so a wrong number means Nanite came back on;
  * material statistics for every material a rubble/debris component uses - a material that fails to compile
    renders as the grey WorldGridMaterial checker and reports NOTHING except zeroed statistics;
  * the static mesh actually bound to each damaged House_### actor - `set_static_mesh` returns None either way;
  * the total level actor count, which is the number that killed the previous attempt at ~5,500.
"""

import json
import os

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\rubble"
RUBBLE_FOLDER = "Rubble"
DAMAGE_FOLDER = "Damage"

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\flood_valley.json") as fh:
    BASE_Z = json.load(fh)["base_z_m"]

report = {"level": None, "pie": les.is_in_play_in_editor(), "actor_count": 0,
          "rubble": {}, "damage_debris": {}, "houses": {}, "materials": {}, "meshes": {}, "errors": []}

_all = eas.get_all_level_actors()
report["actor_count"] = len(_all)
report["level"] = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world().get_name()
report["actors_by_folder"] = {}
for a in _all:
    f = str(a.get_folder_path()) or "(root)"
    report["actors_by_folder"][f] = report["actors_by_folder"].get(f, 0) + 1


def _quat_to_rot(q):
    """FTransform.Rotation is a QUAT in UE 5.8, not a Rotator."""
    try:
        return q.rotator()
    except Exception:                                                   # noqa: BLE001
        return unreal.MathLibrary.conv_quaternion_to_rotator(q)


def instancers(folder):
    """Every HISM/ISM component on every actor in `folder`, with what it holds."""
    out = {}
    for a in _all:
        if str(a.get_folder_path()) != folder:
            continue
        comps = a.get_components_by_class(unreal.InstancedStaticMeshComponent)
        if not comps:
            out[a.get_actor_label()] = {"instances": 0, "mesh": None,
                                        "note": "actor has NO instanced component"}
            continue
        for ci, c in enumerate(comps):
            sm = c.get_editor_property("static_mesh")
            n = c.get_instance_count()
            rec = {
                "instances": n,
                "mesh": None if sm is None else sm.get_name(),
                "mesh_path": None if sm is None else sm.get_path_name(),
                "class": type(c).__name__,
                "mobility": str(c.get_editor_property("mobility")),
                "materials": [m.get_name() if m else None for m in c.get_materials()],
                "material_paths": [m.get_path_name() if m else None for m in c.get_materials()],
            }
            loc = a.get_actor_location()
            rec["actor_location_cm"] = [loc.x, loc.y, loc.z]
            if n:
                t0 = c.get_instance_transform(0, True)
                tl = c.get_instance_transform(n - 1, True)
                rec["instance0_world_cm"] = [t0.translation.x, t0.translation.y, t0.translation.z]
                _r0 = _quat_to_rot(t0.rotation)
                rec["instance0_rot_deg"] = [_r0.roll, _r0.pitch, _r0.yaw]
                rec["instance0_scale"] = t0.scale3d.x
                rec["instanceN_world_cm"] = [tl.translation.x, tl.translation.y, tl.translation.z]
                # the spread proves the field is a field and not a heap of coincident instances
                xs, ys, zs = [], [], []
                step = max(1, n // 24)
                for i in range(0, n, step):
                    t = c.get_instance_transform(i, True)
                    xs.append(t.translation.x)
                    ys.append(t.translation.y)
                    zs.append(t.translation.z)
                rec["sample_bbox_cm"] = [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]
            key = a.get_actor_label() if len(comps) == 1 else f"{a.get_actor_label()}#{ci}"
            out[key] = rec
    return out


report["rubble"] = instancers(RUBBLE_FOLDER)
report["damage_debris"] = instancers(DAMAGE_FOLDER)

# --- the meshes the rubble components point at: triangle counts + Nanite state ----------------------------
for pkg, tag in (("/Game/Sightline/Rubble", "rubble"), ("/Game/Sightline/Buildings/Damaged", "damaged")):
    for p in eal.list_assets(pkg, recursive=False, include_folder=False):
        a = unreal.load_asset(p)
        if not isinstance(a, unreal.StaticMesh):
            continue
        try:
            nanite = bool(a.get_editor_property("nanite_settings").get_editor_property("enabled"))
        except Exception as exc:                                        # noqa: BLE001
            nanite, _ = None, report["errors"].append(f"nanite read failed for {p}: {exc}")
        report["meshes"][a.get_name()] = {
            "group": tag, "triangles": a.get_num_triangles(0), "lods": a.get_num_lods(), "nanite": nanite,
            "slots": [str(s.get_editor_property("material_slot_name"))
                      for s in a.get_editor_property("static_materials")],
            "slot_materials": [(s.get_editor_property("material_interface").get_name()
                                if s.get_editor_property("material_interface") else None)
                               for s in a.get_editor_property("static_materials")],
        }

# --- material statistics ----------------------------------------------------------------------------------
# A material that fails to compile renders as the grey checker and reports zeroed statistics and nothing else.
wanted = set()
for pkg in ("/Game/Sightline/Rubble/Materials",):
    for p in eal.list_assets(pkg, recursive=True, include_folder=False):
        wanted.add(p.split(".")[0])
wanted.add("/Game/Sightline/Buildings/Materials/MI_Rubble")
for grp in (report["rubble"], report["damage_debris"]):
    for rec in grp.values():
        for mp in rec.get("material_paths") or []:
            if mp and mp.startswith("/Game/Sightline"):
                wanted.add(mp.split(".")[0])
for p in sorted(wanted):
    m = unreal.load_asset(p)
    if m is None:
        report["materials"][p] = {"loaded": False}
        continue
    try:
        s = mel.get_statistics(m)
        rec = {"loaded": True, "class": type(m).__name__,
               "instructions": s.num_pixel_shader_instructions,
               "texture_samples": s.num_pixel_texture_samples}
    except Exception as exc:                                            # noqa: BLE001
        report["materials"][p] = {"loaded": True, "error": str(exc)}
        continue
    # The palette parameters, read back from the asset. On the first run DustColor was authored as an sRGB tan
    # and consumed LINEARLY (linear 0.74 = sRGB 0.885), which washed every slab to featureless white. Nothing
    # in the level report could have caught that, so the numbers themselves are recorded and judged host-side.
    for pname in ("DustAmount", "UVScale"):
        try:
            rec[pname] = float(mel.get_material_instance_scalar_parameter_value(m, pname))
        except Exception:                                               # noqa: BLE001
            pass
    for pname in ("DustColor", "Tint"):
        try:
            c = mel.get_material_instance_vector_parameter_value(m, pname)
            rec[pname] = [c.r, c.g, c.b]
        except Exception:                                               # noqa: BLE001
            pass
    try:
        t = mel.get_material_instance_texture_parameter_value(m, "BaseColor")
        rec["BaseColorTexture"] = None if t is None else t.get_name()
    except Exception:                                                   # noqa: BLE001
        pass
    report["materials"][p] = rec

# --- damaged houses: what mesh is really bound ------------------------------------------------------------
for a in _all:
    lbl = a.get_actor_label()
    if not lbl.startswith("House_"):
        continue
    try:
        sm = a.static_mesh_component.get_editor_property("static_mesh")
    except Exception as exc:                                            # noqa: BLE001
        report["errors"].append(f"{lbl}: no static mesh component ({exc})")
        continue
    loc = a.get_actor_location()
    report["houses"][lbl] = {
        "mesh": None if sm is None else sm.get_name(),
        "tags": [str(t) for t in a.tags],
        "loc_cm": [round(loc.x, 1), round(loc.y, 1), round(loc.z, 1)],
        "base_z_m": BASE_Z,
    }

with open(os.path.join(OUT, "placement_report.json"), "w") as fh:
    json.dump(report, fh, indent=1)

_ri = sum(r["instances"] for r in report["rubble"].values())
_di = sum(r["instances"] for r in report["damage_debris"].values())
_dm = sum(1 for h in report["houses"].values() if "Damaged" in h["tags"])
print(f"level {report['level']}: {report['actor_count']} actors")
print(f"  rubble  : {_ri} instances in {len(report['rubble'])} components")
print(f"  debris  : {_di} instances in {len(report['damage_debris'])} components")
print(f"  houses  : {len(report['houses'])} total, {_dm} tagged Damaged")
print(f"  meshes  : {len(report['meshes'])}, materials probed: {len(report['materials'])}")
if report["errors"]:
    print(f"  ERRORS  : {report['errors'][:5]}")
print(f"wrote {OUT}\\placement_report.json -- now run: uv run python tools/scene/check_rubble.py")
