"""Block until AirSim has a real vehicle and the sim is ticking. Exit 0 when the sim can actually be flown.

Started in Simulate mode (`editor_play_simulate`) the editor spawns no player pawn, AirSim reports
"There were no compatible vehicles created for current SimMode", and every flight call blocks forever at
0 % CPU with nothing in the viewport. PLAY mode (`editor_request_begin_play`) is the one that works. This
checks the thing that actually distinguishes them - a listed vehicle whose state can be read - rather than
just that the RPC port accepts a connection.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

DEADLINE_S = 420

t_start = time.time()
import cosysairsim as airsim  # noqa: E402

last = ""
while time.time() - t_start < DEADLINE_S:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c = airsim.MultirotorClient()
            c.confirmConnection()
            vehicles = c.listVehicles()
            st = c.getMultirotorState()
        t0 = st.timestamp
        time.sleep(1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            t1 = c.getMultirotorState().timestamp
        if not vehicles:
            last = "connected but NO VEHICLES - that is Simulate mode, not Play"
        elif t1 <= t0:
            last = f"vehicle {vehicles} present but the sim clock is FROZEN (paused)"
        else:
            k = c.simGetGroundTruthKinematics()
            print(f"READY after {time.time() - t_start:.0f}s: vehicles={vehicles}, clock ticking "
                  f"({t1 - t0} ns/s)")
            print(f"  landed_state={st.landed_state}  z={k.position.z_val:.2f}")
            sys.exit(0)
    except Exception as exc:                                    # noqa: BLE001
        last = f"{type(exc).__name__}"
    time.sleep(5)

print(f"FAIL after {DEADLINE_S}s: {last}")
sys.exit(2)
