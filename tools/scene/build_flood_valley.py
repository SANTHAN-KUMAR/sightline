"""Build /Game/Sightline/Maps/FloodValley inside the RUNNING editor. Idempotent: re-running updates in place.

Run through the sightline MCP (editor up, PIE stopped):
    ue_python  code="exec(open(r'D:\\Sightline\\tools\\scene\\build_flood_valley.py').read())"
Prerequisite: `uv run python tools/scene/gen_terrain.py` (writes data/scene/*).

What it builds (every choice is from SOLUTION_DOC §4/§5.1 or a measured fact in docs/CONTEXT.md):
  - a non-World-Partition level with the AirSim game mode
  - Sun (movable DirectionalLight) wired into BP_Sky_Sphere's "Directional light actor": Cosys' simSetTimeOfDay
    looks the sun up through exactly that property
  - movable real-time SkyLight (ambient without Lumen), ExponentialHeightFog (clear baseline; weather via API)
  - unbound PostProcessVolume forcing GI = None and SSR, motion blur 0 (the Blocks map re-enabled Lumen)
  - terrain static mesh SM_FloodValley: Nanite OFF, complex-as-simple collision, actor yaw +90 deg (UE's OBJ
    importer flips handedness; +90 gives NED: UE X = north, Y = east), object name "Ground" (sim_fly treats
    contact with "Ground" as normal)
  - PlayerStart on the command-post pad; FloodWater: a MOVABLE single-layer-water plane at flood stage, tag
    "FloodWater", no collision, so the FloodLevel is settable at runtime via simSetObjectPose (tools/scene/flood_level.py)
Editor Python has no numpy: everything here reads flood_valley.json only.
"""

import json

import unreal

REPO = r"D:\Sightline"
MAP = "/Game/Sightline/Maps/FloodValley"
TERRAIN_OBJ = REPO + r"\data\scene\flood_valley.obj"
ZONES_PNG = REPO + r"\data\scene\flood_valley_zones.png"
META = REPO + r"\data\scene\flood_valley.json"

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
eal = unreal.EditorAssetLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()

if les.is_in_play_in_editor():
    raise RuntimeError("Stop PIE first: level edits and saves are refused while PIE runs")

with open(META) as f:
    meta = json.load(f)
base = meta["base_z_m"]
launch = meta["launch_site"]


def ue(east_m, north_m, asl_m=None):
    """Scene-local metres -> UE centimetres (NED-aligned: X = north, Y = east)."""
    return unreal.Vector(north_m * 100.0, east_m * 100.0, 0.0 if asl_m is None else (asl_m - base) * 100.0)


def import_asset(filename, dest_path, dest_name):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", filename)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("destination_name", dest_name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    tools.import_asset_tasks([task])
    return unreal.load_asset(f"{dest_path}/{dest_name}")


# --- level --------------------------------------------------------------------------------------
if eal.does_asset_exist(MAP):
    les.load_level(MAP)
else:
    les.new_level(MAP, False)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
actors = {a.get_actor_label(): a for a in eas.get_all_level_actors()}


def ensure(label, cls_or_obj, loc=unreal.Vector(0, 0, 0), rot=unreal.Rotator(0, 0, 0)):
    a = actors.get(label)
    if a is None:
        if isinstance(cls_or_obj, type):
            a = eas.spawn_actor_from_class(cls_or_obj, loc, rot)
        else:
            a = eas.spawn_actor_from_object(cls_or_obj, loc, rot)
        a.set_actor_label(label)
        actors[label] = a
    return a


# --- lighting -----------------------------------------------------------------------------------
sun = ensure("Sun", unreal.DirectionalLight, unreal.Vector(0, 0, 20000), unreal.Rotator(roll=0, pitch=-50, yaw=35))
lc = sun.get_component_by_class(unreal.DirectionalLightComponent)
lc.set_mobility(unreal.ComponentMobility.MOVABLE)
lc.set_editor_property("intensity", 8.0)
lc.set_editor_property("dynamic_shadow_distance_movable_light", 40000.0)
lc.set_editor_property("dynamic_shadow_cascades", 4)

sky = ensure("SkySphere", unreal.load_asset("/Engine/EngineSky/BP_Sky_Sphere.BP_Sky_Sphere"))
sky.set_editor_property("Directional light actor", sun)
sky.call_method("RefreshMaterial")

sl = ensure("SkyLight", unreal.SkyLight, unreal.Vector(0, 0, 5000))
slc = sl.get_component_by_class(unreal.SkyLightComponent)
slc.set_mobility(unreal.ComponentMobility.MOVABLE)
slc.set_editor_property("real_time_capture", True)

fog = ensure("HeightFog", unreal.ExponentialHeightFog)
fog.get_component_by_class(unreal.ExponentialHeightFogComponent).set_editor_property("fog_density", 0.004)

ppv = ensure("PPV_NoGI", unreal.PostProcessVolume)
ppv.set_editor_property("unbound", True)
s = ppv.get_editor_property("settings")
s.set_editor_property("override_dynamic_global_illumination_method", True)
s.set_editor_property("dynamic_global_illumination_method", unreal.DynamicGlobalIlluminationMethod.NONE)
s.set_editor_property("override_reflection_method", True)
s.set_editor_property("reflection_method", unreal.ReflectionMethod.SCREEN_SPACE)
s.set_editor_property("override_motion_blur_amount", True)
s.set_editor_property("motion_blur_amount", 0.0)
ppv.set_editor_property("settings", s)

world.get_world_settings().set_editor_property(
    "default_game_mode", unreal.load_class(None, "/Script/AirSim.AirSimGameMode"))

# --- terrain ------------------------------------------------------------------------------------
sm = import_asset(TERRAIN_OBJ, "/Game/Sightline/Terrain", "SM_FloodValley")
ns = sm.get_editor_property("nanite_settings")
ns.set_editor_property("enabled", False)
sm.set_editor_property("nanite_settings", ns)
bs = sm.get_editor_property("body_setup")
bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
sm.set_editor_property("body_setup", bs)
eal.save_asset(sm.get_path_name())
if sm.get_num_triangles(0) != meta["triangles"]:
    raise RuntimeError(f"terrain has {sm.get_num_triangles(0)} tris, expected {meta['triangles']} (Nanite fallback?)")

ground = actors.get("Ground") or actors.get("Terrain")
if ground is None:
    ground = eas.spawn_actor_from_object(sm, unreal.Vector(0, 0, 0))
    actors["Ground"] = ground
ground.static_mesh_component.set_static_mesh(sm)
ground.set_actor_location(unreal.Vector(0, 0, 0), False, False)
ground.set_actor_rotation(unreal.Rotator(roll=0, pitch=0, yaw=meta["ue_import"]["terrain_actor_yaw_deg"]), False)
ground.static_mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
if ground.get_name() != "Ground":
    ground.rename("Ground")
ground.set_actor_label("Ground")

zones = import_asset(ZONES_PNG, "/Game/Sightline/Terrain", "T_FloodValley_Zones")
zones.set_editor_property("srgb", False)
zones.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP)
zones.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS)
zones.set_editor_property("address_x", unreal.TextureAddress.TA_CLAMP)
zones.set_editor_property("address_y", unreal.TextureAddress.TA_CLAMP)
eal.save_asset(zones.get_path_name())

# --- verify georeferencing with line traces against the generator's own numbers ---------------------
def trace_z_cm(v):
    hit = unreal.SystemLibrary.line_trace_single(world, unreal.Vector(v.x, v.y, 100000), unreal.Vector(v.x, v.y, -100000),
                                                 unreal.TraceTypeQuery.ECC_VISIBILITY, True, [],
                                                 unreal.DrawDebugTrace.NONE, True)
    return hit.to_tuple()[4].z if hit else None


pad_expect = (launch["ground_asl_m"] - base) * 100.0
pad_got = trace_z_cm(ue(launch["east_m"], launch["north_m"]))
if pad_got is None or abs(pad_got - pad_expect) > 30.0:
    raise RuntimeError(f"pad trace {pad_got} cm != expected {pad_expect:.0f} cm: terrain transform is wrong")

# --- PlayerStart on the pad, flood water at flood stage -----------------------------------------------
ps = ensure("PlayerStart_Home", unreal.PlayerStart)
ps.set_actor_location(unreal.Vector(*meta["ue_import"]["player_start_cm"]), False, False)
ps.set_actor_rotation(unreal.Rotator(0, 0, 0), False)

water = ensure("FloodWater", unreal.load_asset("/Engine/BasicShapes/Plane"))
water.set_actor_location(unreal.Vector(0, 0, meta["ue_import"]["flood_water_z_cm"]), False, False)
water.set_actor_scale3d(unreal.Vector(meta["size_m"], meta["size_m"], 1.0))  # the plane is 1 m square
wc = water.static_mesh_component
wc.set_mobility(unreal.ComponentMobility.MOVABLE)
wc.set_material(0, unreal.load_asset("/Game/Sightline/Water/M_FloodWater"))
wc.set_collision_profile_name("NoCollision")
wc.set_editor_property("cast_shadow", False)
if "FloodWater" not in [str(t) for t in water.tags]:
    water.tags = list(water.tags) + ["FloodWater"]
if water.get_name() != "FloodWater":
    water.rename("FloodWater")

print("saved", les.save_current_level())
print({a.get_actor_label(): a.get_name() for a in eas.get_all_level_actors()})
print(f"pad trace {pad_got:.0f} cm (expected {pad_expect:.0f}); terrain tris {sm.get_num_triangles(0)}")
