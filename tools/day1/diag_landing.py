"""Probe: hover settling and low-altitude landing behaviour of SimpleFlight (direct RPC, no MCP)."""
import json, time
import cosysairsim as airsim
V = "Drone"
c = airsim.MultirotorClient(timeout_value=30); c.confirmConnection()

def st():
    s = c.getMultirotorState(vehicle_name=V); k = s.kinematics_estimated
    col = c.simGetCollisionInfo(vehicle_name=V)
    return {"z": round(k.position.z_val, 3), "x": round(k.position.x_val, 3), "y": round(k.position.y_val, 3),
            "vz": round(k.linear_velocity.z_val, 3), "landed": int(s.landed_state),
            "col": col.has_collided, "col_obj": col.object_name}

out = {}
c.reset(); time.sleep(1)
c.enableApiControl(True, vehicle_name=V); c.armDisarm(True, vehicle_name=V)
out["home"] = st()
c.takeoffAsync(vehicle_name=V).join()
c.moveToPositionAsync(15, -10, -20, 6, vehicle_name=V).join()
c.hoverAsync(vehicle_name=V).join()
hov = []
for _ in range(20):
    s = st(); hov.append([s["x"], s["y"], s["z"]]); time.sleep(0.5)
out["hover_trace_xyz_0.5s"] = hov

# A: RTL path - home at altitude, moveToZ to 2 m above home ground, landAsync (not joined), trace
home_z = out["home"]["z"]
c.moveToPositionAsync(0, 0, -20, 6, vehicle_name=V).join()
c.moveToZAsync(home_z - 2.0, 3.0, vehicle_name=V).join()
out["A_after_moveToZ"] = st()
c.landAsync(timeout_sec=120, vehicle_name=V)
trace = []
for _ in range(60):
    trace.append(st()); time.sleep(0.5)
out["A_land_trace"] = [f"z={t['z']} vz={t['vz']} landed={t['landed']} col={t['col_obj']}" for t in trace[::4]]

# B: landAsync joined straight from 20 m
c.takeoffAsync(vehicle_name=V).join()
c.moveToPositionAsync(0, 0, -20, 6, vehicle_name=V).join()
t0 = time.time(); c.landAsync(timeout_sec=200, vehicle_name=V).join(); out["B_join_seconds"] = round(time.time() - t0, 1)
out["B_after_join"] = st()
trace = []
for _ in range(40):
    trace.append(st()); time.sleep(1)
out["B_trace_1s"] = [f"z={t['z']} vz={t['vz']} landed={t['landed']}" for t in trace[::4]]
c.reset()
print(json.dumps(out, indent=1))
