"""Direct-RPC probe of Cosys-AirSim semantics the MCP tools depend on (no MCP layer):
  1. landed_state before/after takeoff and at altitude; landAsync timing and z trajectory
  2. simPause while the drone is MOVING: does the pose freeze? does the state timestamp freeze?
  3. runtime spawn / move / read-back / destroy of a movable object (vs static level actors)
Run: uv run python -u tools/day1/diag_semantics.py  (needs a running sim with sim/settings/default.json)
"""

import json
import time

import cosysairsim as airsim

V = "Drone"


def pos(c):
    p = c.getMultirotorState(vehicle_name=V).kinematics_estimated.position
    return [round(p.x_val, 3), round(p.y_val, 3), round(p.z_val, 3)]


def main() -> None:
    c = airsim.MultirotorClient(timeout_value=30)
    c.confirmConnection()
    c.reset()
    time.sleep(1)
    out = {}
    c.enableApiControl(True, vehicle_name=V)
    c.armDisarm(True, vehicle_name=V)
    s = c.getMultirotorState(vehicle_name=V)
    out["spawn"] = {"pos": pos(c), "landed_state": int(s.landed_state)}

    # 1. landed state + landing
    c.takeoffAsync(vehicle_name=V).join()
    out["after_takeoff"] = {"pos": pos(c), "landed_state": int(c.getMultirotorState(vehicle_name=V).landed_state)}
    c.moveToPositionAsync(0, 0, -15, 5, vehicle_name=V).join()
    time.sleep(2)
    out["at_15m"] = {"pos": pos(c), "landed_state": int(c.getMultirotorState(vehicle_name=V).landed_state)}
    t = time.time()
    c.landAsync(timeout_sec=60, vehicle_name=V).join()
    out["landAsync"] = {"seconds": round(time.time() - t, 2), "pos_after": pos(c),
                        "landed_state": int(c.getMultirotorState(vehicle_name=V).landed_state)}
    traj = []
    for _ in range(10):  # does it keep descending after landAsync returned?
        time.sleep(1)
        traj.append(pos(c)[2])
    out["z_after_landAsync_10s"] = traj

    # 2. pause while moving
    c.takeoffAsync(vehicle_name=V).join()
    c.moveToPositionAsync(0, 0, -10, 5, vehicle_name=V).join()
    c.moveToPositionAsync(60, 0, -10, 8, vehicle_name=V)  # not joined: moving
    time.sleep(2)
    c.simPause(True)
    s0 = c.getMultirotorState(vehicle_name=V)
    p0 = pos(c)
    time.sleep(1.5)
    s1 = c.getMultirotorState(vehicle_name=V)
    p1 = pos(c)
    out["pause_while_moving"] = {"is_pause": c.simIsPause(), "pos_t0": p0, "pos_t1.5": p1,
                                 "timestamp_delta_s": (s1.timestamp - s0.timestamp) / 1e9}
    c.simContinueForTime(0.5)
    time.sleep(1.5)
    p2 = pos(c)
    out["continue_for_0.5s"] = {"pos": p2, "moved_m": round(sum((a - b) ** 2 for a, b in zip(p1, p2, strict=True)) ** 0.5, 3),
                                "is_pause_after": c.simIsPause()}
    c.simPause(False)
    c.hoverAsync(vehicle_name=V).join()

    # 3. spawn / move / destroy
    assets = c.simListAssets()
    cube = next((a for a in assets if a.lower() in ("cube", "1m_cube", "templatecube_rounded")), None) or \
        next((a for a in assets if "cube" in a.lower()), None)
    out["assets_count"] = len(assets)
    out["cube_asset"] = cube
    if cube:
        pose = airsim.Pose(airsim.Vector3r(5, 5, -2), airsim.Quaternionr())
        name = c.simSpawnObject("sightline_probe", cube, pose, airsim.Vector3r(1, 1, 1), False)
        p_sp = c.simGetObjectPose(name).position
        ok = c.simSetObjectPose(name, airsim.Pose(airsim.Vector3r(7, 6, -2), airsim.Quaternionr()), True)
        p_mv = c.simGetObjectPose(name).position
        out["spawn_move"] = {"name": name, "spawned_at": [p_sp.x_val, p_sp.y_val, p_sp.z_val], "set_ok": ok,
                             "moved_to": [p_mv.x_val, p_mv.y_val, p_mv.z_val], "destroyed": c.simDestroyObject(name)}
    # static level actor for comparison
    ok = c.simSetObjectPose("TemplateCube_Rounded_1", airsim.Pose(airsim.Vector3r(0, 0, -50), airsim.Quaternionr()), True)
    p = c.simGetObjectPose("TemplateCube_Rounded_1").position
    out["static_actor_set_pose"] = {"returned": ok, "actual": [p.x_val, p.y_val, p.z_val]}
    c.landAsync(vehicle_name=V).join()
    c.reset()
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
