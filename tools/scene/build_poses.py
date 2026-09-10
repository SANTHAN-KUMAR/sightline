"""Author the survivor pose library on the Rocketbox skeletons (running editor, PIE off).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_poses.py').read())"

Why this exists: the 9 Microsoft Rocketbox characters ship a single bind-pose frame (an A-pose, arms 45 deg out).
A survivor lying prone and a survivor standing must differ in SILHOUETTE, because at the 40-60 m survey altitude
of SOLUTION_DOC 5.5c a person is only 20-60 px and the silhouette is essentially all the detector has. So the
postures of `sightline.schemas.POSTURES` are authored here as real one-key AnimSequences per character.

Measured facts this script depends on (probed 2026-09-10, Male_Adult_04, and asserted below):
  * Bone names are 3ds Max Biped style (`Bip01-L-UpperArm`); bone-local +X runs DOWN the limb toward the child.
  * `AnimationLibrary.get_animation_track_names()` returns names LOWERCASED, while `get_bone_name()` returns them
    cased. Comparing them with Python `==`/`in` silently matches nothing, which makes a pose come out as the bind
    pose with no error at all. Every lookup here is lowercased.
  * `AnimationDataController.add_bone_track` is deprecated AND silently no-ops; `add_bone_curve` is the working
    call. A pose built with the former reads back byte-identical to the bind pose.
  * Component space: +Z up, +X to the character's left, +Y forward.
  * Joint axes are NOT the same for arms and legs (their bind orientations differ):
        arm  : pitch = adduction (-44 deg puts the hand exactly at the side), yaw = elbow flexion, roll = twist
        leg  : yaw   = hip/knee flexion (negative = forward at the hip, positive = knee bend), pitch = abduction
        root : pitch -90 = face DOWN (prone), pitch +90 = face UP (supine). A roll of 180 does NOT
               flip the face: at pitch 90 the rotator is gimbal-locked, so roll 180 spins the body
               about the vertical and merely swaps which way the head points.
Ground contact is not hard-coded: each pose's lowest bone is found by forward kinematics and written to
`data/scene/poses.json` as `ground_offset_cm`, which the spawner subtracts so the body rests on the surface.
"""

import json

import unreal

A = unreal.AnimationLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: asset writes are unreliable while PIE runs")

ROOT = "/Game/Sightline/Characters/Rocketbox"
POSE_DIR = "/Game/Sightline/Characters/Poses"
REPO = r"D:\Sightline"

# --- the pose library -------------------------------------------------------------------------------------
# Deltas are (d_pitch, d_yaw, d_roll) degrees applied in the bone's LOCAL space on top of the bind pose.
#
# The left and right ARM bones have MIRRORED bind orientations (L-UpperArm bind pitch -31.16 / roll +10.56 vs
# R-UpperArm +31.16 / -10.56), so an identical delta swings one arm down and the other one UP. That is exactly
# what went wrong first time: every survivor stood with one arm at its side and one stuck straight out, and no
# API call reported anything (measured by FK: same delta puts the right hand 52 cm from its shoulder, negated
# pitch puts it 1.19 cm from it, mirroring the left exactly). LEG bones are NOT mirrored - the same delta gives
# an identical foot position on both sides, and negating it kicks the right leg backwards.
# `arm()` and `leg()` encode that asymmetry once so a pose definition cannot get it wrong.

def arm(dp, dy=0.0, dr=0.0):
    """Same articulation on both arms, with the right side's mirrored bind axes accounted for."""
    return {"Bip01-L-UpperArm": (dp, dy, dr), "Bip01-R-UpperArm": (-dp, dy, -dr)}


def forearm(dp, dy=0.0, dr=0.0):
    return {"Bip01-L-Forearm": (dp, dy, dr), "Bip01-R-Forearm": (-dp, dy, -dr)}


def leg(dp, dy=0.0, dr=0.0):
    return {"Bip01-L-Thigh": (dp, dy, dr), "Bip01-R-Thigh": (dp, dy, dr)}


def knee(dp, dy=0.0, dr=0.0):
    return {"Bip01-L-Calf": (dp, dy, dr), "Bip01-R-Calf": (dp, dy, dr)}


ARM_DOWN = {**arm(-44), **forearm(0, 14)}          # hands at the sides
ARM_LOOSE = {**arm(-34), **forearm(0, 26)}         # slack, slightly away from the body

POSES = {
    # upright, arms at the sides: the baseline "stranded but mobile" survivor
    "standing": dict(ARM_DOWN),
    # seated on a roof or slab, knees up: the commonest posture on a flooded roof (SOLUTION_DOC 2.3 row 2)
    "sitting": {**ARM_LOOSE, **leg(0, -78), **knee(0, 82), "Bip01-Spine1": (0, 6, 0)},
    # face down, limbs slack: injured/unconscious on the deposit fan
    "prone": {**ARM_LOOSE, "Bip01": (-90, 0, 0), **leg(0, -8)},
    # face up
    "supine": {**ARM_LOOSE, "Bip01": (90, 0, 0), **leg(0, -8)},
    # treading water / wading: arms out from the body, legs slightly flexed
    "half_submerged": {**arm(-14), **forearm(0, 48), **leg(0, -22), **knee(0, 30)},
    # curled under debris: the hard case for the detector, and R10's "cannot clear" evidence
    "trapped": {"Bip01": (-90, 0, 0), "Bip01-Spine1": (0, 24, 0), "Bip01-Spine2": (0, 18, 0),
                **leg(0, -92), **knee(0, 108), **arm(-30), **forearm(0, 78)},
    # both arms raised: signalling. A distinctive silhouette from above, and the demo's "found me" case.
    "waving": {**arm(58), **forearm(0, 34)},
}

FLESH_CM = 9.0   # half-thickness of a torso: keeps a lying body ON the surface, not sunk into it

CHARACTERS = ["Female_Adult_04", "Female_Adult_11", "Female_Adult_17", "Female_Child_02",
              "Male_Adult_04", "Male_Adult_08", "Male_Adult_13", "Male_Adult_15", "Male_Child_02"]


def bone_parents(mesh):
    """Lowercased bone -> parent map. Needs a live component; the reference skeleton is not exposed directly."""
    tmp = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 0, -300000))
    try:
        c = tmp.skeletal_mesh_component
        c.set_skeletal_mesh_asset(mesh)
        out = {}
        for i in range(c.get_num_bones()):
            b = str(c.get_bone_name(i))
            out[b.lower()] = str(c.get_parent_bone(b)).lower()
        return out
    finally:
        eas.destroy_actor(tmp)


def fk_all(local, parents, deltas):
    """Component-space transform of every bone, with local delta rotations applied along each chain."""
    deltas = {k.lower(): v for k, v in deltas.items()}
    out = {}

    def solve(b):
        if b in out:
            return out[b]
        t = local[b]
        q = t.rotation
        if b in deltas:
            dp, dy, dr = deltas[b]
            q = q.multiply(unreal.Rotator(dr, dp, dy).quaternion())
        step = unreal.Transform(location=t.translation, rotation=q.rotator(), scale=t.scale3d)
        p = parents.get(b, "")
        world = step if p not in local else unreal.MathLibrary.compose_transforms(step, solve(p))
        out[b] = world
        return world

    for b in local:
        solve(b)
    return out


def build(character):
    src = f"{ROOT}/{character}"
    base = unreal.load_asset(f"{src}/{character}_Anim")
    skel = unreal.load_asset(f"{src}/{character}_Skeleton")
    mesh = unreal.load_asset(f"{src}/{character}")
    if not (base and skel and mesh):
        raise RuntimeError(f"{character}: missing base anim / skeleton / mesh")
    tracks = [str(n) for n in A.get_animation_track_names(base)]           # already lowercase
    local = {t: A.get_bone_pose_for_frame(base, t, 0, False) for t in tracks}
    parents = bone_parents(mesh)

    # The two child characters are rigged with a `Bip02-` prefix instead of `Bip01-`; the bone structure is
    # otherwise identical (82 tracks, same names). Detect the prefix rather than hard-coding it.
    prefixes = sorted({t.split("-")[0] for t in tracks if t.startswith("bip")})
    if len(prefixes) != 1:
        raise RuntimeError(f"{character}: expected one biped prefix, found {prefixes}")
    prefix = prefixes[0]

    def remap(d):
        return {prefix + b.lower()[len("bip01"):]: v for b, v in d.items()}

    info = {}
    for pose, raw_deltas in POSES.items():
        deltas = remap(raw_deltas)
        unknown = [b for b in deltas if b.lower() not in local]
        if unknown:
            raise RuntimeError(f"{character}/{pose}: bones not on this skeleton: {unknown}")
        name = f"A_{character}_{pose}"
        path = f"{POSE_DIR}/{name}"
        if eal.does_asset_exist(path):
            eal.delete_asset(path)
        f = unreal.AnimSequenceFactory()
        f.set_editor_property("target_skeleton", skel)
        anim = AT.create_asset(name, POSE_DIR, unreal.AnimSequence, f)
        ctrl = anim.get_editor_property("controller")
        ctrl.open_bracket("build pose")
        ctrl.set_frame_rate(unreal.FrameRate(30, 1))
        ctrl.set_number_of_frames(unreal.FrameNumber(1))
        low = {k.lower(): v for k, v in deltas.items()}
        for b in tracks:
            t = local[b]
            q = t.rotation
            if b in low:
                dp, dy, dr = low[b]
                q = q.multiply(unreal.Rotator(dr, dp, dy).quaternion())
            if not ctrl.add_bone_curve(b):                       # add_bone_track is deprecated and no-ops
                raise RuntimeError(f"{name}: add_bone_curve failed for {b}")
            if not ctrl.set_bone_track_keys(b, [t.translation], [q], [t.scale3d]):
                raise RuntimeError(f"{name}: set_bone_track_keys failed for {b}")
        ctrl.close_bracket()
        eal.save_asset(path)

        # Geometry of the finished pose, by forward kinematics: ground offset and footprint.
        world = fk_all(local, parents, deltas)
        # `Bip01` (the root) and `Bip01-Footsteps` are helper bones pinned at z = 0 and are not part of the
        # body: including them made min(z) always 0, so every upright pose reported ground_offset 0 and a
        # seated character measured as tall as a standing one.
        body = {k: v for k, v in world.items() if k not in (prefix, prefix + "-footsteps")}
        zs = [w.translation.z for w in body.values()]
        xs = [w.translation.x for w in body.values()]
        ys = [w.translation.y for w in body.values()]
        head, nose = body.get(prefix + "-head"), body.get(prefix + "-mnose")
        # Left/right symmetry: every pose here is bilaterally symmetric, so the hands and feet must land at
        # mirrored x. The first pose library got this wrong (mirrored bind axes on the arms) and produced
        # one-arm-out survivors that nothing flagged. Component +X is the character's left.
        sym = 0.0
        for lb, rb in ((prefix + "-l-hand", prefix + "-r-hand"), (prefix + "-l-foot", prefix + "-r-foot")):
            lw, rw = body.get(lb), body.get(rb)
            if lw and rw:
                sym = max(sym, abs(lw.translation.x + rw.translation.x),
                          abs(lw.translation.y - rw.translation.y), abs(lw.translation.z - rw.translation.z))
        info[pose] = {
            "asset": path,
            # Bones run through the middle of the body, so drop by the lowest bone less a flesh
            # allowance, otherwise a lying figure sinks half its thickness into the ground.
            "ground_offset_cm": round(-min(zs) + FLESH_CM, 2),
            "bbox_cm": [round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1), round(max(zs) - min(zs), 1)],
            "height_cm": round(max(zs) - min(zs), 1),
            "head_z_cm": round(head.translation.z, 1) if head else None,
            "face_down": (None if not (head and nose) else bool(nose.translation.z < head.translation.z - 1.0)),
            "symmetry_err_cm": round(sym, 2),
        }
    return info


out = {"generated_by": "tools/scene/build_poses.py", "postures": sorted(POSES), "characters": {}}
for ch in CHARACTERS:
    out["characters"][ch] = build(ch)
    print(f"  {ch}: {len(POSES)} poses")

print("\nposture geometry (Male_Adult_04):")
ref = out["characters"]["Male_Adult_04"]
for p in sorted(POSES):
    r = ref[p]
    print(f"  {p:16s} height {r['height_cm']:6.1f} cm  bbox {str(r['bbox_cm']):22s} "
          f"ground_offset {r['ground_offset_cm']:7.2f}  sym_err {r['symmetry_err_cm']:5.2f} cm  "
          f"face_down={r['face_down']}")

# --- self-checks: a pose that silently fell back to the bind pose must not pass -----------------------------
errs = []
if not ref["prone"]["face_down"]:
    errs.append("prone is not face down")
if ref["supine"]["face_down"]:
    errs.append("supine is not face up")
# A lying body must be much shorter than a standing one and much longer horizontally.
if ref["prone"]["height_cm"] > 60:
    errs.append(f"prone height {ref['prone']['height_cm']} cm - body did not lie down")
if ref["standing"]["height_cm"] < 150:
    errs.append(f"standing height {ref['standing']['height_cm']} cm - unexpected")
if max(ref["prone"]["bbox_cm"][0], ref["prone"]["bbox_cm"][1]) < 150:
    errs.append("prone footprint too short to be a lying adult")
if ref["sitting"]["height_cm"] >= ref["standing"]["height_cm"] - 20:
    errs.append("sitting is not shorter than standing")
for _p, _r in ref.items():                       # every pose here is bilaterally symmetric
    if _r["symmetry_err_cm"] > 3.0:
        errs.append(f"{_p} is left/right ASYMMETRIC by {_r['symmetry_err_cm']} cm (mirrored bind axes?)")
if errs:
    raise RuntimeError("pose self-check FAILED: " + "; ".join(errs))

with open(REPO + r"\data\scene\poses.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(f"\n{len(CHARACTERS) * len(POSES)} pose assets written; self-checks passed; data/scene/poses.json updated")
