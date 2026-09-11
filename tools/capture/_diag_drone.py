"""Why is a survey blocked before its first frame? Report the vehicle state the flight is waiting on."""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "scene"))

import cosysairsim as airsim  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    c = airsim.MultirotorClient()
    c.confirmConnection()

t0 = c.getMultirotorState().timestamp
time.sleep(1.0)
t1 = c.getMultirotorState().timestamp
print(f"sim clock: {'RUNNING' if t1 > t0 else 'FROZEN (paused)'}  ({t1 - t0} ns in 1 s wall)")

st = c.getMultirotorState()
k = c.simGetGroundTruthKinematics()
print(f"landed_state : {st.landed_state}  (0=Landed, 1=Flying)")
print(f"position NED : x={k.position.x_val:.2f} y={k.position.y_val:.2f} z={k.position.z_val:.2f}")
print(f"velocity     : {k.linear_velocity.x_val:.2f} {k.linear_velocity.y_val:.2f} "
      f"{k.linear_velocity.z_val:.2f}")
print(f"api control  : {c.isApiControlEnabled()}")

from sightline.mission.pattern import Scenario  # noqa: E402

scn = Scenario.load()
home = scn.home
asl = home["ground_asl_m"] - k.position.z_val
n = home["north_m"] + k.position.x_val
e = home["east_m"] + k.position.y_val
print(f"world        : east {e:.1f} north {n:.1f} asl {asl:.1f} m")
print(f"terrain here : {scn.terrain.surface_asl(e, n):.1f} m  -> AGL {asl - scn.terrain.surface_asl(e, n):.1f} m")

t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    imgs = c.simGetImages([airsim.ImageRequest("survey", airsim.ImageType.Scene, False, False)])
print(f"one Scene grab took {time.time() - t:.2f} s, {imgs[0].width}x{imgs[0].height}")
