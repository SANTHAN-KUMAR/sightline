"""Place any House_### that `settlement.json` declares but the level does not have (editor, PIE OFF). Idempotent.

    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\place_missing_houses.py').read())"

WHY THIS EXISTS. On 2026-09-11 the level held 73 House_### actors while `data/scene/settlement.json` declared
76: `gen_buildings.py` grew the settlement by three `zone=bank` houses (ids 73, 74, 75) and `build_buildings.py`
had not been re-run since, so the level was stale with respect to its own layout. `data/scene/damage.json` is
generated from the CURRENT settlement and plans damage for house 74, so `build_damage.py` failed - correctly -
with "1 planned houses have no actor, first: [74]".

WHY NOT JUST RE-RUN build_buildings.py. Because it would silently undo another lane's work.
`build_buildings.py` rebuilds `M_PBR_Master` from an empty graph, and `build_roof_materials.py` afterwards
CLEARS that same graph and rebuilds it with anti-tiling and the flood tide line (its lines 169-170). Running
build_buildings.py again would leave the settlement with a master material that has neither. That is the exact
"two scripts clearing and rebuilding the same graph silently undo each other" trap named in build_rubble.py's
docstring, so this script touches NO material graph at all.

WHAT IT DOES. Exactly `build_buildings.py`'s placement block, for the missing ids only: the archetype mesh, the
wall-tint material on the wall slot, static mobility, the Building/zone/archetype tags, the House_### label and
the Settlement folder. It creates nothing else and modifies no existing actor.

SELF-HEALING. `build_buildings.py` destroys every House_* and respawns all of them from settlement.json, so
whenever the buildings lane next runs it these actors are simply replaced. Nothing here needs unwinding.

NAMING. `set_actor_label` only, never `rename()` - see build_rubble.py's docstring for the Obj.cpp:383 crash.
"""

import json

import unreal

REPO = r"D:\Sightline"
MATDIR = "/Game/Sightline/Buildings/Materials"

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: spawns and saves are unreliable while PIE runs")

with open(REPO + r"\data\scene\settlement.json") as fh:
    town = json.load(fh)
with open(REPO + r"\data\scene\flood_valley.json") as fh:
    BASE = json.load(fh)["base_z_m"]

WALLS = [unreal.load_asset(f"{MATDIR}/MI_Wall_{n}") for n in ("Cream", "Yellow", "Mint", "Weathered")]
if any(w is None for w in WALLS):
    raise RuntimeError("MI_Wall_* are missing - run tools/scene/build_buildings.py first")

have = {int(a.get_actor_label().split("_")[1]) for a in eas.get_all_level_actors()
        if a.get_actor_label().startswith("House_")}
missing = [h for h in town["houses"] if h["id"] not in have]
print(f"level has {len(have)} houses, settlement.json declares {len(town['houses'])}; "
      f"missing {[h['id'] for h in missing]}")

made = 0
for hs in missing:
    arch = hs["archetype"]
    sm = unreal.load_asset(f"/Game/Sightline/Buildings/SM_{arch}")
    if sm is None:
        raise RuntimeError(f"SM_{arch} is missing - run tools/scene/build_buildings.py first")
    slots = [str(s.get_editor_property("material_slot_name")).lower()
             for s in sm.get_editor_property("static_materials")]
    wall_slot = next((i for i, n in enumerate(slots) if n.startswith("wall")), None)
    if wall_slot is None:
        raise RuntimeError(f"SM_{arch} has no wall slot: {slots}")
    loc = unreal.Vector(hs["north_m"] * 100.0, hs["east_m"] * 100.0, (hs["base_asl_m"] - BASE) * 100.0)
    act = eas.spawn_actor_from_object(sm, loc, unreal.Rotator(0, 0, 90.0 - hs["yaw_deg"]))
    if act is None:
        raise RuntimeError(f"spawn failed for house {hs['id']}")
    act.set_actor_label(f"House_{hs['id']:03d}")            # label only - never rename()
    act.set_folder_path("Settlement")
    comp = act.static_mesh_component
    comp.set_mobility(unreal.ComponentMobility.STATIC)
    comp.set_material(wall_slot, WALLS[hs["wall_tint"]])
    act.tags = ["Building", hs["zone"], arch]
    made += 1
    print(f"  placed House_{hs['id']:03d} {arch} zone={hs['zone']} north {hs['north_m']} east {hs['east_m']} "
          f"(wall slot {wall_slot}, tint {hs['wall_tint']})")

have = {int(a.get_actor_label().split("_")[1]) for a in eas.get_all_level_actors()
        if a.get_actor_label().startswith("House_")}
want = {h["id"] for h in town["houses"]}
if have != want:
    raise RuntimeError(f"still missing {sorted(want - have)}, extra {sorted(have - want)}")
print(f"placed {made}; the level now holds all {len(have)} houses settlement.json declares")
print(f"level actors: {len(eas.get_all_level_actors())} | saved {les.save_current_level()}")
