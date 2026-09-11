"""Render the views that let a human judge the utility network, the boats, the foam ribbon and the water.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\qa_utilities_shots.py').read())"

`qa_shots.py` renders the standard scene views; it aims at survivors and at the valley, so a wire that hangs
straight, a boat that hovers 40 cm over the flood or a foam ribbon that z-fights the water plane are all
invisible in it. These shots aim at the lane's own features and are framed so that each specific failure mode
would be visible:

  util_1_pole_water   one pole from 18 m at water level - is the shaft VERTICAL, and does it enter the water
                      at the depth utilities.json claims (flood_depth_m) rather than floating or being buried?
  util_2_span_side    a whole span side-on, camera on the span's perpendicular bisector at conductor height.
                      A catenary sags; a straight-line span is a chord. Framed so the sag (~1-2 % of span) is
                      tens of pixels, not one.
  util_3_span_close   mid-span from 12 m: is the conductor a CONTINUOUS line or a dotted one at close range?
  util_4_boats        ONE boat in open water from 10 m at 6 deg above the surface - the grazing angle is the
                      only one that shows a hull intersecting the water or hovering above it. A cluster shot
                      framed on its centroid puts the boats behind a house, which is how the first attempt at
                      this view showed no boat at all.
  util_4b_boat_high   the same boat from 30 deg: the whole hull, so 'floats' can be judged as well as 'sits'.
  util_10_nadir_gsd   45 m nadir at the REAL survey GSD (utilities.json survey.gsd_cm_per_px). Wire continuity
                      MUST be judged here and not in util_7: util_7's wider FOV samples the 3.6 cm conductor
                      at ~0.4 px, where any line aliases into dashes no matter how it was built.
  util_5_foam_graze   the shoreline from 9 m at 4 deg: foam sits 3 cm above the water plane, so a grazing view
                      is where z-fighting (stippled/flickering banding) would show.
  util_6_foam_nadir   the same shoreline from 40 m nadir: does the ribbon read as a foam line ON the edge?
  util_7_nadir45      45 m nadir over the settlement - the survey view. Water colour, wire continuity.
  util_8_oblique45    45 m at 30 deg off nadir - Reference A's framing. Poles in water, wires across frame.
  util_9_shallow      a shallow margin from 12 m at 25 deg: can the drowned ground be SEEN through the water?
                      This is the single test for whether the extinction coefficients are too high.

Writes to _artifacts/editor_shots/util_*.png. Read-only with respect to the level: it spawns one
SceneCapture2D and destroys it in a finally.
"""

import json
import math
import os

import unreal

REPO = r"D:\Sightline"
OUT = REPO + r"\_artifacts\editor_shots"
W, H = 1920, 1080

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
os.makedirs(OUT, exist_ok=True)

with open(REPO + r"\data\scene\utilities.json") as fh:
    U = json.load(fh)
with open(REPO + r"\data\scene\foam.json") as fh:
    F = json.load(fh)
BASE_Z = U["base_z_m"]
WATER = U["water_level_m"]


def ue(north_m, east_m, asl_m):
    return unreal.Vector(north_m * 100.0, east_m * 100.0, (asl_m - BASE_Z) * 100.0)


rt = unreal.RenderingLibrary.create_render_target2d(world, W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
cap.set_actor_label("QA_UTIL_TEMP")
comp = cap.capture_component2d
comp.set_editor_property("texture_target", rt)
comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
comp.set_editor_property("capture_every_frame", False)
comp.set_editor_property("capture_on_movement", False)


def look_at(cam, tgt):
    """UE yaw is measured from +X toward +Y; ue() maps north->X and east->Y, so this is a compass bearing
    from north towards east. Getting it wrong points the camera at empty sky, which is how a 'render' can
    silently show nothing at all."""
    dx, dy, dz = tgt.x - cam.x, tgt.y - cam.y, tgt.z - cam.z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return (0.0, pitch, yaw)


def shot(name, cam, tgt, fov=60.0):
    comp.set_editor_property("fov_angle", fov)
    cap.set_actor_location_and_rotation(cam, unreal.Rotator(*look_at(cam, tgt)), False, False)
    comp.capture_scene()
    comp.capture_scene()                        # second pass: temporal effects settle
    unreal.RenderingLibrary.export_render_target(world, rt, OUT, name + ".png")
    d = math.dist((cam.x, cam.y, cam.z), (tgt.x, tgt.y, tgt.z)) / 100.0
    gsd = 2.0 * d * math.tan(math.radians(fov / 2.0)) / W * 100.0
    print(f"  {name}.png  dist {d:6.1f} m  fov {fov:4.1f}  gsd {gsd:5.2f} cm/px")


# ---- pick the features to aim at ---------------------------------------------------------------------------
poles = U["pole_list"]
chain_len = U["network"]["chain_lengths"]
# chains are contiguous slices of pole_list (28 + 26 + 6 = 60; 27 + 25 + 5 = 57 spans)
spans, i0 = [], 0
for n in chain_len:
    for j in range(i0, i0 + n - 1):
        a, b = poles[j], poles[j + 1]
        spans.append((a, b, math.dist((a["east_m"], a["north_m"]), (b["east_m"], b["north_m"]))))
    i0 += n
if len(spans) != U["counts"]["spans"]:
    raise RuntimeError(f"reconstructed {len(spans)} spans, utilities.json says {U['counts']['spans']} - "
                       f"the chain layout in pole_list is not what this script assumes")

# a pole standing in a decent depth of water, so 'enters the water at the right depth' is actually testable
pole = max(poles, key=lambda p: (1.2 < p["flood_depth_m"] < 3.0, p["height_m"]))
# the longest span whose two poles are both well in the water: most sag, easiest to judge
span = max(spans, key=lambda s: s[2] if min(s[0]["flood_depth_m"], s[1]["flood_depth_m"]) > 1.0 else -1)

boats = U["boats"]["items"]
# A boat in OPEN water: framing on a cluster centroid hides the boats behind the house they are moored to.
# settlement.json carries the house footprints; distance to the nearest house is the cheapest 'is it in the
# open' measure available without querying the level.
with open(REPO + r"\data\scene\settlement.json") as fh:
    SET = json.load(fh)
_houses = SET.get("houses") or SET.get("buildings") or []


def house_clear(b):
    if not _houses:
        return 0.0
    return min(math.dist((h["east_m"], h["north_m"]), (b["east_m"], b["north_m"])) for h in _houses)


bc = max(boats, key=house_clear)

# a shoreline probe: gen_foam recorded terrain ASL right at the waterline
probe = min(F["probe_points"], key=lambda p: abs(p["terrain_asl_m"] - F["water_level_m"]))

print(f"aiming at: pole east {pole['east_m']} north {pole['north_m']} depth {pole['flood_depth_m']} m, "
      f"h {pole['height_m']} m")
print(f"           span {span[2]:.1f} m between ({span[0]['east_m']},{span[0]['north_m']}) and "
      f"({span[1]['east_m']},{span[1]['north_m']})")
print(f"           boat {bc['name']} ({bc['variant']}) east {bc['east_m']} north {bc['north_m']}, "
      f"{house_clear(bc):.0f} m clear of the nearest house, base {bc['base_asl_m']:.3f} vs water "
      f"{WATER:.3f} m ASL (= {(bc['base_asl_m'] - WATER) * 100.0:+.1f} cm)")
print(f"           shoreline probe east {probe['east_m']} north {probe['north_m']} "
      f"terrain {probe['terrain_asl_m']:.2f} vs water {F['water_level_m']:.2f}")

try:
    # 1. one pole, from water level, 18 m out. Vertical? correct submersion?
    pw = (pole["north_m"], pole["east_m"])
    tgt = ue(pw[0], pw[1], WATER + pole["height_m"] * 0.35)
    cam = ue(pw[0] - 12.7, pw[1] - 12.7, WATER + 1.6)
    shot("util_1_pole_water", cam, tgt, fov=45.0)

    # 2. a whole span, side-on from the perpendicular bisector at conductor height.
    a, b, L = span
    mn, me = (a["north_m"] + b["north_m"]) / 2.0, (a["east_m"] + b["east_m"]) / 2.0
    dn, de = b["north_m"] - a["north_m"], b["east_m"] - a["east_m"]
    pn, pe = -de / L, dn / L                                   # unit perpendicular in the ground plane
    top = max(a["ground_asl_m"] + a["height_m"], b["ground_asl_m"] + b["height_m"])
    # frame the whole span plus a pole either side: need ~1.5 L of horizontal coverage
    dist = (1.5 * L) / (2.0 * math.tan(math.radians(30.0)))
    tgt2 = ue(mn, me, top - 1.2)
    cam2 = ue(mn + pn * dist, me + pe * dist, top - 0.5)
    shot("util_2_span_side", cam2, tgt2, fov=60.0)

    # 3. mid-span from 12 m: continuity of the conductor at close range
    cam3 = ue(mn + pn * 8.5, me + pe * 8.5, top - 1.0)
    shot("util_3_span_close", cam3, ue(mn, me, top - 1.4), fov=50.0)

    # 4. ONE boat in open water at a grazing 6 deg: does the hull sit ON the surface?
    cn, ce = bc["north_m"], bc["east_m"]
    d4 = 10.0
    shot("util_4_boats", ue(cn - d4 * 0.71, ce - d4 * 0.71, WATER + d4 * math.tan(math.radians(6.0))),
         ue(cn, ce, WATER + 0.25), fov=32.0)
    # 4b. the same boat from 30 deg, so the whole hull and its waterline are both in frame
    d4b = 11.0
    shot("util_4b_boat_high",
         ue(cn - d4b * 0.71, ce - d4b * 0.71, WATER + d4b * math.tan(math.radians(30.0))),
         ue(cn, ce, WATER + 0.25), fov=34.0)

    # 5. shoreline at a 4 deg graze - where foam/water z-fighting would show
    d5 = 11.0
    shot("util_5_foam_graze",
         ue(probe["north_m"] - d5 * 0.71, probe["east_m"] - d5 * 0.71,
            WATER + d5 * math.tan(math.radians(4.0))),
         ue(probe["north_m"], probe["east_m"], WATER), fov=55.0)

    # 6. the same shoreline from 40 m nadir
    shot("util_6_foam_nadir", ue(probe["north_m"], probe["east_m"], WATER + 40.0),
         ue(probe["north_m"], probe["east_m"], WATER), fov=50.0)

    # 7. survey view: 45 m nadir. Aim at the pole with the most NEIGHBOURING poles, not at the centroid of
    # all 60: the centroid of three separate chains lands in open water between them, which is how the first
    # attempt at this view produced a frame containing no wire at all.
    def crowd(p):
        return sum(1 for o in poles
                   if math.dist((o["east_m"], o["north_m"]), (p["east_m"], p["north_m"])) < 60)
    hub = max(poles, key=crowd)
    nn, nee = hub["north_m"], hub["east_m"]
    print(f"    nadir aimed at the pole hub east {nee} north {nn} ({crowd(hub)} poles within 60 m)")
    shot("util_7_nadir45", ue(nn, nee, WATER + 45.0), ue(nn, nee, WATER), fov=74.0)

    # 8. Reference A framing: 45 m up, 30 deg off nadir, looking across the network
    shot("util_8_oblique45", ue(nn - 78.0, nee - 78.0, WATER + 45.0), ue(nn, nee, WATER), fov=74.0)

    # 9. a shallow margin: is the drowned ground visible THROUGH the water?
    d9 = 13.0
    shot("util_9_shallow",
         ue(probe["north_m"] - d9 * 0.71, probe["east_m"] - d9 * 0.71,
            WATER + d9 * math.tan(math.radians(25.0))),
         ue(probe["north_m"] + 6.0, probe["east_m"] + 6.0, WATER - 0.25), fov=55.0)

    # 10. nadir at the REAL survey GSD. The conductor is 3.6 cm; utilities.json claims 2.04 px at
    # survey.gsd_cm_per_px. Any wider FOV samples it below 1 px, where EVERY line aliases into dashes, so
    # judging 'the wires are dotted' anywhere but here would condemn geometry that is actually fine.
    gsd = U["survey"]["gsd_cm_per_px"] / 100.0
    alt = 45.0
    fov10 = 2.0 * math.degrees(math.atan((W * gsd / 2.0) / alt))
    shot("util_10_nadir_gsd", ue(nn, nee, WATER + alt), ue(nn, nee, WATER), fov=fov10)
    print(f"    (util_10 fov {fov10:.1f} deg reproduces the survey GSD "
          f"{U['survey']['gsd_cm_per_px']} cm/px; conductor should be "
          f"{U['network']['wire_diameter_m'] * 100.0 / U['survey']['gsd_cm_per_px']:.2f} px)")
finally:
    eas.destroy_actor(cap)

print(f"\nwritten to {OUT}. OPEN THEM. The point of each frame is in this file's docstring.")
