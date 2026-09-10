"""Day-1 test #4 / feature F3: the gamepad THROUGH Cosys-AirSim, and the API <-> RC takeover handover.

Needs a running sim (editor PIE, -game, or the packaged build) started with sim/settings/default.json and a
gamepad plugged in. Reads `getMultirotorState().rc_data` with a direct cosysairsim client (high rate) and drives
the vehicle through the *sightline MCP server* (`sim_fly arm|release|takeoff|move_to|rtl`), exactly as Claude
Code will.

Sequence
  1. wait up to --wait s for the first non-zero stick/switch input (never blocks forever; records NOT OBSERVED)
  2. sample --sample s of rc_data: ranges per axis, switches seen, update rate
  3. handover, with the user HOLDING the left stick (throttle) up the whole time:
       API control ON  -> the vehicle must ignore the stick (station keeping)
       sim_fly release -> API control OFF, the stick must fly the vehicle   [latency API->RC]
       sim_fly arm     -> API control ON again, the tool must regain authority [latency RC->API]
       sim_fly move_to -> the vehicle must obey the tool again
  4. land + reset

Instructions are printed to the console AND drawn on the sim viewport (simPrintLogMessage), so the person at the
controller can follow them without watching this terminal.

Run: .venv/Scripts/python.exe -u tools/day1/gamepad_airsim.py [--wait 180] [--sample 20] [--alt 8]
Results: _artifacts/verification/gamepad_airsim_<ts>.json   Exit code = number of failed checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cosysairsim as airsim
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
VEHICLE = "Drone"
RESULTS: list[dict] = []


def record(name: str, ok: bool | None, detail: str) -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": str(detail)[:400]})
    tag = "PASS" if ok else ("FAIL" if ok is False else "N/OBS")
    print(f"[{tag}] {name}: {str(detail)[:220]}", flush=True)


def say(c, msg: str) -> None:
    print(f"\n>>> {msg}\n", flush=True)
    try:
        c.simPrintLogMessage("SIGHTLINE >>> ", msg)
    except Exception:  # noqa: BLE001 - HUD message is a convenience only
        pass


def rc_of(c):
    return c.getMultirotorState(vehicle_name=VEHICLE).rc_data


def stick_active(rc) -> bool:
    return bool(rc.switches) or abs(rc.throttle - 0.5) > 0.15 or max(abs(rc.pitch), abs(rc.roll), abs(rc.yaw)) > 0.15


def pos_speed(c):
    k = c.getMultirotorState(vehicle_name=VEHICLE).kinematics_estimated
    v = k.linear_velocity
    return ([k.position.x_val, k.position.y_val, k.position.z_val],
            math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2))


def server_params() -> StdioServerParameters:
    cfg = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["sightline"]
    return StdioServerParameters(command=cfg["command"], args=list(cfg["args"]),
                                 env={**os.environ, **cfg.get("env", {})}, cwd=str(REPO))


class McpBridge:
    """A sightline MCP stdio session driven from a background asyncio thread.

    The main thread must stay free of a running asyncio loop: the cosysairsim client owns a tornado IOLoop that
    refuses to start inside one ("This event loop is already running" / windows_events assertion). This is the
    same constraint that makes server.py run every AirSim call on its own thread (docs/CONTEXT.md section 7).
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop = None
        self._session: ClientSession | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-client")
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError("MCP session did not start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        async with stdio_client(server_params()) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            self._session = s
            self._ready.set()
            await self._stop.wait()

    def call(self, name: str, args: dict, timeout_s: float = 180.0) -> tuple[bool, str, float]:
        async def go():
            return await self._session.call_tool(name, args, read_timeout_seconds=timedelta(seconds=timeout_s))
        t = time.time()
        res = asyncio.run_coroutine_threadsafe(go(), self._loop).result(timeout_s + 30)
        txt = "\n".join(x.text for x in res.content if getattr(x, "type", "") == "text")
        return bool(res.isError), txt, time.time() - t

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=20)


def fly(m: McpBridge, args: dict, timeout_s: float = 180.0) -> tuple[bool, str, float]:
    return m.call("sim_fly", args, timeout_s)


def api_control_of(txt: str) -> bool | None:
    try:
        return json.loads(txt[txt.index("{"):])["api_control"]
    except Exception:  # noqa: BLE001
        return None


def main(wait_s: float, sample_s: float, alt: float) -> int:
    c = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=30)
    c.confirmConnection()

    settings = json.loads((REPO / "sim" / "settings" / "default.json").read_text())
    rcset = settings["Vehicles"][VEHICLE].get("RC", {})
    record("settings RC block", True, f"RemoteControlID={rcset.get('RemoteControlID')} "
           f"AllowAPIWhenDisconnected={rcset.get('AllowAPIWhenDisconnected')} "
           f"AllowAPIAlways={settings['Vehicles'][VEHICLE].get('AllowAPIAlways')}")

    rc = rc_of(c)
    record("rc_data reaches AirSim (RemoteControlID 0)", bool(rc.is_initialized and rc.is_valid),
           f"is_initialized={rc.is_initialized} is_valid={rc.is_valid} vendor_id={rc.vendor_id} "
           f"throttle={rc.throttle} pitch={rc.pitch} roll={rc.roll} yaw={rc.yaw} switches={rc.switches}")
    if not rc.is_initialized:
        record("live stick input", None, "no RC device initialised in the sim; nothing else can be observed")
        return finish()

    # ---- 1. wait for the first real input -------------------------------------------------------
    say(c, f"MOVE THE STICKS on the gamepad (you have {wait_s:.0f} s)")
    t0, first = time.time(), None
    while time.time() - t0 < wait_s:
        rc = rc_of(c)
        if stick_active(rc):
            first = time.time() - t0
            break
        time.sleep(0.05)
    if first is None:
        record("live stick input observed", None,
               f"NOT OBSERVED: sticks stayed neutral for {wait_s:.0f} s (throttle {rc.throttle}, "
               f"pitch/roll/yaw {rc.pitch}/{rc.roll}/{rc.yaw}). The device is enumerated and valid, "
               "but nobody moved it.")
        return finish()
    record("live stick input observed", True, f"first non-neutral input after {first:.1f} s")

    # ---- 2. sample -------------------------------------------------------------------------------
    say(c, f"Sweep BOTH sticks fully and press buttons for {sample_s:.0f} s")
    rng = {k: [1e9, -1e9] for k in ("throttle", "pitch", "roll", "yaw", "left_z", "right_z")}
    switches, n, t0 = set(), 0, time.time()
    stamps: list[float] = []
    while time.time() - t0 < sample_s:
        rc = rc_of(c)
        n += 1
        for k in rng:
            v = float(getattr(rc, k))
            rng[k] = [min(rng[k][0], v), max(rng[k][1], v)]
        if rc.switches:
            switches.add(int(rc.switches))
        stamps.append(time.time())
        time.sleep(0.02)
    hz = n / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
    travel = {k: round(v[1] - v[0], 3) for k, v in rng.items()}
    record("rc_data axes move with the sticks", travel["throttle"] > 0.5 and
           max(travel["pitch"], travel["roll"]) > 0.5,
           f"ranges={ {k: [round(a, 3), round(b, 3)] for k, (a, b) in rng.items()} } travel={travel} "
           f"switch bitmasks seen={sorted(switches)} polled at {hz:.0f} Hz over {n} samples")

    # ---- 3. handover -----------------------------------------------------------------------------
    m = McpBridge()
    try:
        fly(m, {"action": "reset", "vehicle": VEHICLE})
        err, txt, _ = fly(m, {"action": "takeoff", "vehicle": VEHICLE, "timeout_s": 40})
        fly(m, {"action": "move_to", "x": 0.0, "y": 0.0, "z": -alt, "velocity": 3.0,
                      "vehicle": VEHICLE, "timeout_s": 60})
        p, _ = pos_speed(c)
        record("hover under API control before handover", abs(p[2] + alt) < 1.5, f"pos={[round(x, 2) for x in p]}")

        say(c, "HOLD the LEFT stick FULLY UP (throttle) and KEEP HOLDING until told to let go")
        t0 = time.time()
        while time.time() - t0 < 60 and abs(rc_of(c).throttle - 0.5) < 0.3:
            time.sleep(0.05)
        held = abs(rc_of(c).throttle - 0.5) >= 0.3
        if not held:
            record("throttle held for handover", None, "NOT OBSERVED: the throttle stick was not held within 60 s")
            fly(m, {"action": "rtl", "z": -alt, "vehicle": VEHICLE, "timeout_s": 120})
            return finish()
        thr = rc_of(c).throttle
        record("throttle held for handover", True, f"rc throttle={thr:.2f} (neutral 0.5)")

        # 3a. API control ON: the stick must NOT move the vehicle
        p0, _ = pos_speed(c)
        time.sleep(4.0)
        p1, _ = pos_speed(c)
        drift = math.dist(p0, p1)
        record("API control ON ignores the held stick (AllowAPIAlways)", drift < 1.0,
               f"moved {drift:.2f} m in 4 s while the throttle stick was held at {thr:.2f}")

        # 3b. release -> RC flies it
        base, _ = pos_speed(c)
        t_issue = time.time()
        err, txt, dt_call = fly(m, {"action": "release", "vehicle": VEHICLE})
        t_ret = time.time()
        released = api_control_of(txt) is False
        t_motion = None
        while time.time() - t_issue < 15:
            p, v = pos_speed(c)
            if abs(p[2] - base[2]) > 0.5 or v > 0.8:
                t_motion = time.time()
                break
            time.sleep(0.01)
        p_rc, _ = pos_speed(c)
        lat_api_rc = (t_motion - t_issue) if t_motion else None
        record("sim_fly release -> API control OFF", released and not err,
               f"api_control={api_control_of(txt)}; call took {dt_call * 1000:.0f} ms")
        record("gamepad flies the drone after release", t_motion is not None,
               f"vehicle moved {math.dist(base, p_rc):.2f} m (z {base[2]:.2f} -> {p_rc[2]:.2f}) with the stick held; "
               f"handover latency API->RC = {lat_api_rc * 1000:.0f} ms from issuing sim_fly release "
               f"({(t_motion - t_ret) * 1000:.0f} ms after the tool returned)"
               if t_motion else "no motion within 15 s of release while the stick was held")

        # 3c. arm -> API control back
        t_issue2 = time.time()
        err2, txt2, dt_call2 = fly(m, {"action": "arm", "vehicle": VEHICLE})
        reacquired = api_control_of(txt2) is True
        fly(m, {"action": "hover", "vehicle": VEHICLE, "timeout_s": 30})
        t_stable = None
        while time.time() - t_issue2 < 20:
            _, v = pos_speed(c)
            if v < 0.5:
                t_stable = time.time()
                break
            time.sleep(0.01)
        lat_rc_api = (t_stable - t_issue2) if t_stable else None
        record("sim_fly arm -> API control ON again", reacquired and not err2,
               f"api_control={api_control_of(txt2)}; call took {dt_call2 * 1000:.0f} ms")
        record("API regains authority over a held stick", t_stable is not None,
               f"handover latency RC->API (issue sim_fly arm -> vehicle speed < 0.5 m/s) = "
               f"{lat_rc_api * 1000:.0f} ms" if t_stable else "vehicle never stabilised within 20 s")

        # 3d. the tools fly it again while the stick is still held
        p_before, _ = pos_speed(c)
        tgt = [p_before[0] + 10.0, p_before[1], -alt]
        err3, txt3, dt3 = fly(m, {"action": "move_to", "x": tgt[0], "y": tgt[1], "z": tgt[2],
                                        "velocity": 3.0, "vehicle": VEHICLE, "timeout_s": 60})
        p_after, _ = pos_speed(c)
        record("drone obeys tools again after re-acquire", not err3 and math.dist(p_after, tgt) < 2.0,
               f"move_to {[round(x, 1) for x in tgt]} -> {[round(x, 2) for x in p_after]} "
               f"(err {math.dist(p_after, tgt):.2f} m in {dt3:.1f} s), stick still held")

        say(c, "You can LET GO of the sticks now - returning home")
        time.sleep(2.0)
        err4, txt4, dt4 = fly(m, {"action": "rtl", "z": -alt, "vehicle": VEHICLE, "timeout_s": 120})
        record("rtl after handover test", not err4, txt4.splitlines()[0] if txt4 else "")
        fly(m, {"action": "reset", "vehicle": VEHICLE})
    finally:
        m.close()
    return finish()


def finish() -> int:
    out = REPO / "_artifacts" / "verification" / f"gamepad_airsim_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))
    fails = sum(1 for r in RESULTS if r["ok"] is False)
    nobs = sum(1 for r in RESULTS if r["ok"] is None)
    print(f"\n{sum(1 for r in RESULTS if r['ok'])} PASS / {fails} FAIL / {nobs} NOT OBSERVED -> {out}", flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=180.0)
    ap.add_argument("--sample", type=float, default=20.0)
    ap.add_argument("--alt", type=float, default=8.0)
    a = ap.parse_args()
    sys.exit(main(a.wait, a.sample, a.alt))
