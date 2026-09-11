"""Kill any hung revisit/probe process and make sure the simulator is not left paused.

A probe that sets `simPause(True)` and then dies leaves the whole simulator frozen, which looks exactly like
a hung editor. This puts it back.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PATTERNS = ("settlement_realpath", "refine_labels_with_depth", "validate_depth_gate", "mission.live")

killed = []
for pat in PATTERNS:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{pat}*' }} "
             f"| Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            pid = line.strip()
            if pid.isdigit():
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=30)
                killed.append(pid)
    except Exception as exc:                                     # noqa: BLE001
        print(f"  kill scan for {pat!r} failed: {exc}")
print(f"killed: {killed or 'none'}")

import cosysairsim as airsim  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    c = airsim.MultirotorClient()
    c.confirmConnection()
c.simPause(False)
with contextlib.suppress(Exception):
    c.armDisarm(False)
    c.enableApiControl(False)
t0 = c.getMultirotorState().timestamp
import time  # noqa: E402

time.sleep(1.0)
t1 = c.getMultirotorState().timestamp
print(f"sim clock {t0} -> {t1}  ({'RUNNING' if t1 > t0 else 'STILL FROZEN'})")
