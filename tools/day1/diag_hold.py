"""Probe (direct RPC): idle-hold primitive, collision reporting, position-held landing, landed_state flip.
Run with Blocks + sim/settings/default.json. Prints JSON."""

import json
import math
import time

import cosysairsim as airsim

V = "Drone"
c = airsim.MultirotorClient(timeout_value=30)
c.confirmConnection()


def brief():
    s = c.getMultirotorState(vehicle_name=V)
    k = s.kinematics_estimated
    v = k.linear_velocity
    return {"pos": [round(k.position.x_val, 3), round(k.position.y_val, 3), round(k.position.z_val, 3)],
            "speed": round(math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2), 3), "vz": round(v.z_val, 3),
            "landed": int(s.landed_state)}


def col():
    ci = c.simGetCollisionInfo(vehicle_name=V)
    return {k: getattr(ci, k) for k in dir(ci) if not k.startswith("_") and not callable(getattr(ci, k))
            and k not in ("normal", "impact_point", "position")}


out = {}
c.reset()
time.sleep(1)
c.enableApiControl(True, vehicle_name=V)
c.armDisarm(True, vehicle_name=V)
out["collision_at_spawn"] = col()
c.takeoffAsync(vehicle_name=V).join()
c.moveToPositionAsync(0, 0, -25, 3, vehicle_name=V).join()          # climb vertically first
c.moveToPositionAsync(20, -15, -25, 6, vehicle_name=V).join()        # translate at altitude

# settle on the target by refreshing a position goal every 0.1 s
t0 = time.time()
while time.time() - t0 < 12:
    c.moveToPositionAsync(20, -15, -25, 2, timeout_sec=1, vehicle_name=V)
    b = brief()
    if b["speed"] < 0.2 and math.dist(b["pos"], [20, -15, -25]) < 0.3:
        break
    time.sleep(0.1)
out["settle"] = {"seconds": round(time.time() - t0, 1), **brief()}

# idle hold: one long server-side zero-velocity task at fixed altitude (keeps the 60 ms watchdog fed)
c.moveByVelocityZAsync(0, 0, -25, 1e6, vehicle_name=V)
p0 = brief()["pos"]
trace = []
for _ in range(15):
    time.sleep(1)
    trace.append(brief()["pos"])
out["idle_hold_15s"] = {"start": p0, "end": trace[-1], "max_drift_m": round(max(math.dist(p0, p) for p in trace), 3)}
# on-screen overlay is the only place the watchdog message appears (not in Blocks.log): capture it for inspection
from PIL import ImageGrab  # noqa: E402 - probe script, not the MCP server
ImageGrab.grab().save(r"D:\Sightline\_artifacts\verification\idle_hold_overlay.png")

# position-held descent at (20,-15): setpoint 1-2 m below current altitude, re-issued every 0.2 s
t0 = time.time()
stall = None
land_trace = []
while time.time() - t0 < 90:
    b = brief()
    land_trace.append(b)
    if b["landed"] == 0:
        break
    agl = 0.698 - b["pos"][2]
    v = 3.0 if agl > 6 else (1.0 if agl > 2 else 0.4)
    c.moveToPositionAsync(20, -15, b["pos"][2] + max(v, 0.5) * 1.5, v, timeout_sec=1, vehicle_name=V)
    if b["vz"] < 0.05 and time.time() - t0 > 1.5:
        stall = stall or time.time()
        if time.time() - stall > 2.0:
            break
    else:
        stall = None
    time.sleep(0.2)
out["descent"] = {"seconds": round(time.time() - t0, 1), "final": brief(), "xy_error_m":
                  round(math.dist(brief()["pos"][:2], [20, -15]), 3), "collision": col()}

# what flips landed_state after touchdown?
c.cancelLastTask(vehicle_name=V)
time.sleep(3)
out["after_cancel_3s"] = brief()
c.moveByVelocityAsync(0, 0, 0.3, 2.0, vehicle_name=V).join()
time.sleep(1)
out["after_push_down"] = brief()
c.armDisarm(False, vehicle_name=V)
time.sleep(2)
out["after_disarm"] = brief()
c.armDisarm(True, vehicle_name=V)
t0 = time.time()
c.takeoffAsync(vehicle_name=V).join()
out["takeoff_after_disarm_rearm"] = {"seconds": round(time.time() - t0, 1), **brief()}
c.reset()
print(json.dumps(out, indent=1), flush=True)
