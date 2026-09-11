"""Render the views that show whether the rubble field and the structural damage are RIGHT (editor, PIE OFF).

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_rubble.py').read())"

`qa_shots.py` renders the standard scene views; none of them looks at the deposit fan, which is where this lane
put 1,799 items. These are the views that can show the specific ways this work fails, and each one exists to
answer a question that a return value cannot:

  qa_rubble_1_ground     eye height ON the fan, looking along it. This is the direct comparison with
                         SCENE_REFERENCE "Reference B": is the palette dust-covered, desaturated tan/grey with
                         very low colour contrast, are there broken plates at every angle with voids between
                         them - or is it a uniform gravel bed?
  qa_rubble_2_oblique    15 m up, 30 deg down: reads the structure of the field and its extent.
  qa_rubble_3_nadir45    45 m nadir over the densest cell - what the detector actually sees.
  qa_rubble_4_void       a survivor in a protected void (the `trapped` posture / occlusion slice).
  qa_rubble_5_burial     the floor slab laid over a 2.7-burial survivor: it must actually roof them.
  qa_rubble_6_scale      one pile at 6 m with a survivor in frame, to catch the OBJ-unit trap (a field
                         imported 100x too small looks like gravel and passes every count).
  qa_rubble_7_damage45   45 m over the damaged-house cluster: do they read as DAMAGED from survey altitude?
  qa_rubble_8_damage     one damaged house at 25 m, close enough to see the missing roof and the spill.

Grey checker anywhere in these frames means a material failed to compile and every count in the level report
is worthless.
"""

import json
import math
import os

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\editor_shots"
W, H = 1600, 900

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\rubble_layout.json") as fh:
    rub = json.load(fh)
with open(REPO + r"\data\scene\damage.json") as fh:
    dmg = json.load(fh)
with open(REPO + r"\data\scene\settlement.json") as fh:
    town = json.load(fh)
BASE_Z = rub["base_z_m"]
ITEMS = rub["items"]


def ue(north_m, east_m, asl_m):
    return unreal.Vector(north_m * 100.0, east_m * 100.0, (asl_m - BASE_Z) * 100.0)


def ground(north_m, east_m):
    """ASL of the nearest placed rubble item - the fan surface, without needing the terrain asset."""
    best, bz = 1e18, rub["water_level_m"]
    for it in ITEMS:
        d = (it["north_m"] - north_m) ** 2 + (it["east_m"] - east_m) ** 2
        if d < best:
            best, bz = d, it["base_asl_m"]
    return bz


def look_at(cam, target):
    """Rotator that points a camera at a world point (UE: +X forward, yaw about +Z, pitch positive is up)."""
    dx, dy, dz = target.x - cam.x, target.y - cam.y, target.z - cam.z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return (0.0, pitch, yaw)


# Where does the collapse field read strongest? NOT simply the densest 20 m cell: the fan apex is the densest
# and it sits on a wooded shoreline slope, so the first version of this script pointed three cameras into the
# vegetation lane's canopy and photographed leaves. Score cells by the STRUCTURAL load (piles and slabs are the
# unit of a collapse field; loose blocks are gravel) and penalise nearby trees, which are the thing actually in
# the way. This is a camera choice, not a scene change - the field itself is wherever gen_rubble.py put it.
STRUCTURAL = {"pile", "slab", "masonry", "belonging"}
try:
    with open(REPO + r"\data\scene\vegetation.json") as fh:
        _veg = json.load(fh)
    TREES = [(t["north_m"], t["east_m"]) for t in _veg.get("items", [])]
except Exception as exc:                                                # noqa: BLE001
    print(f"  ! vegetation.json unreadable ({exc}); scoring without the tree penalty")
    TREES = []

cells, cell_z = {}, {}
for it in ITEMS:
    k = (round(it["north_m"] / 20) * 20, round(it["east_m"] / 20) * 20)
    c = cells.setdefault(k, [0, 0])
    c[0] += 1
    c[1] += 1 if it["family"] in STRUCTURAL else 0
    cell_z.setdefault(k, []).append(it["base_asl_m"])
cell_z = {k: sum(v) / len(v) for k, v in cell_z.items()}
# The cell must stand clear of the flood, or the ground camera's eye sits 1.7 m above the waterline and half
# the frame is water - which is what the previous pick did (fan surface 1062.1 m, water 1061.7 m).
MIN_FREEBOARD_M = 4.0
DRY = [k for k in cells if cell_z[k] >= rub["water_level_m"] + MIN_FREEBOARD_M] or list(cells)

tree_cells = {}
for n, e in TREES:
    k = (round(n / 20) * 20, round(e / 20) * 20)
    tree_cells[k] = tree_cells.get(k, 0) + 1


def score(k):
    n, e = k
    near_trees = sum(tree_cells.get((n + dn, e + de), 0)
                     for dn in (-20, 0, 20) for de in (-20, 0, 20))
    return cells[k][1] * 3 + cells[k][0] - 4.0 * near_trees


DN, DE = max(DRY, key=score)
DZ = ground(DN, DE)
_near = sum(tree_cells.get((DN + dn, DE + de), 0) for dn in (-20, 0, 20) for de in (-20, 0, 20))
print(f"chosen cell: north {DN} east {DE} | {cells[(DN, DE)][0]} items "
      f"({cells[(DN, DE)][1]} structural) | {_near} trees within 30 m | fan surface {DZ:.1f} m ASL "
      f"({DZ - rub['water_level_m']:.1f} m above water) | {len(DRY)}/{len(cells)} cells had "
      f"{MIN_FREEBOARD_M:.0f} m freeboard")

rt = unreal.RenderingLibrary.create_render_target2d(world, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("QA_RUBBLE_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)
shots = []


def shot(name, loc, rot, fov=75.0):
    comp.set_editor_property("fov_angle", fov)
    cap.set_actor_location_and_rotation(loc, unreal.Rotator(*rot), False, False)
    comp.capture_scene()
    comp.capture_scene()                        # second pass: temporal effects settle
    unreal.RenderingLibrary.export_render_target(world, rt, OUT, name + ".png")
    shots.append(name)
    print(f"  {OUT}\\{name}.png")


try:
    # 1. Reference B comparison: standing ON the fan, 1.7 m eye height, looking along the field
    cam = ue(DN - 26, DE - 14, DZ + 1.7)
    shot("qa_rubble_1_ground", cam, look_at(cam, ue(DN + 30, DE + 10, DZ + 1.0)), fov=70.0)

    # 2. oblique, 15 m up
    cam = ue(DN - 55, DE - 30, DZ + 15.0)
    shot("qa_rubble_2_oblique", cam, look_at(cam, ue(DN + 20, DE + 5, DZ)), fov=75.0)

    # 3. nadir at survey altitude
    shot("qa_rubble_3_nadir45", ue(DN, DE, DZ + 45.0), (0, -90, 0), fov=74.0)

    # 4. a survivor in a protected void
    voids = [i for i in ITEMS if i["group"] == "void"]
    if voids:
        v = voids[0]
        tgt = ue(v["north_m"], v["east_m"], v["base_asl_m"] + 0.5)
        cam = unreal.Vector(tgt.x - 700, tgt.y - 380, tgt.z + 420)
        shot("qa_rubble_4_void", cam, look_at(cam, tgt), fov=45.0)

    # 5. the burial slab of 2.7 - it must be ON TOP of the survivor, not beside them
    burial = [i for i in ITEMS if i["group"] == "burial"]
    if burial:
        b = burial[0]
        tgt = ue(b["north_m"], b["east_m"], b["base_asl_m"] + 0.4)
        cam = unreal.Vector(tgt.x - 480, tgt.y - 260, tgt.z + 260)
        shot("qa_rubble_5_burial", cam, look_at(cam, tgt), fov=42.0)

    # 6. scale check: the biggest pile in the field, seen from 6 m
    big = max(ITEMS, key=lambda i: i["tris"])
    tgt = ue(big["north_m"], big["east_m"], big["base_asl_m"] + 1.2)
    cam = unreal.Vector(tgt.x - 600, tgt.y - 300, tgt.z + 200)
    shot("qa_rubble_6_scale", cam, look_at(cam, tgt), fov=50.0)

    # 7-8. structural damage
    by_id = {h["id"]: h for h in town["houses"]}
    picked = [h for h in dmg["houses"] if h["id"] in by_id]
    cn = sum(by_id[h["id"]]["north_m"] for h in picked) / len(picked)
    ce = sum(by_id[h["id"]]["east_m"] for h in picked) / len(picked)
    # the damaged house with the most damaged neighbours, so one frame carries several
    def near(h):
        return sum(1 for g in picked
                   if abs(by_id[g["id"]]["north_m"] - by_id[h["id"]]["north_m"]) < 90
                   and abs(by_id[g["id"]]["east_m"] - by_id[h["id"]]["east_m"]) < 90)
    hub = max(picked, key=near)
    hs = by_id[hub["id"]]
    print(f"damage hub: House_{hub['id']:03d} ({hub['variant']}), {near(hub)} damaged houses within 90 m; "
          f"cluster centre north {cn:.0f} east {ce:.0f}")
    shot("qa_rubble_7_damage45", ue(hs["north_m"], hs["east_m"], hs["base_asl_m"] + 45.0), (0, -90, 0), fov=74.0)
    # 30 m up and 35 m back, i.e. ABOVE the canopy looking down at ~40 deg. At 15 m the camera sat inside a
    # tree and photographed leaves.
    tgt = ue(hs["north_m"], hs["east_m"], hs["base_asl_m"] + 2.5)
    cam = unreal.Vector(tgt.x - 3500, tgt.y - 1400, tgt.z + 3000)
    shot("qa_rubble_8_damage", cam, look_at(cam, tgt), fov=50.0)
finally:
    eas.destroy_actor(cap)

for f in shots:
    p = os.path.join(OUT, f + ".png")
    if not os.path.exists(p) or os.path.getsize(p) < 20000:
        raise RuntimeError(f"{p} is missing or suspiciously small "
                           f"({os.path.getsize(p) if os.path.exists(p) else 'absent'} bytes) - the capture "
                           f"produced nothing to look at")
print(f"\n{len(shots)} rubble/damage QA shots written to {OUT}")
print("OPEN THEM. Compare qa_rubble_1_ground with SCENE_REFERENCE Reference B: dust-covered desaturated "
      "tan/grey, very low colour contrast, plates at every angle, voids between them. Grey checker anywhere "
      "means a material failed to compile.")
