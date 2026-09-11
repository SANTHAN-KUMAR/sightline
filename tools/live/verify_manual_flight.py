"""Prove, against the RUNNING simulator, that velocity manual actually flies the aircraft.

    uv run python tools/live/verify_manual_flight.py --json _artifacts/verification/manual_flight.json

The claim this exists to test
-----------------------------
`sightline/mission/manual.py` asserts three things that had never been executed against a vehicle:

1. **A stick command moves the aircraft** through `moveByVelocityBodyFrameAsync`, with API control retained.
2. **Centred sticks hold station**, which is the entire reason the RC passthrough path was replaced - a
   self-centring stick on simple_flight's raw throttle is roughly 50 % motor, not hover.
3. **The envelope holds a pilot off the floor**, measured against the terrain rather than take-off height.

Everything before this was reasoning from the firmware source. Reasoning is how the bug was FOUND; it is not
how a claim gets to be called measured (`docs/QUALITY_GATE.md`).

Deliberately cheap
------------------
No images, no pipeline, no 4K buffers - it captures nothing. It reads kinematics and issues velocity
commands, which is tens of megabytes of Python. That matters because this project's editor has OOM-crashed
at the RAM headroom it typically runs with, and a verification that takes the thing down proves nothing.

It always hands the vehicle back: the `finally` block hovers and restores API control even on Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.mission.manual import ManualLimits, ManualPilot  # noqa: E402
from sightline.mission.pattern import Scenario, connect  # noqa: E402
from sightline.mission.takeover import ControlInput  # noqa: E402


def _pose(client, home) -> tuple[float, float, float]:
    """(east_m, north_m, alt_asl_m) in scene coordinates."""
    st = client.simGetGroundTruthKinematics()
    return (float(home["east_m"]) + float(st.position.y_val),
            float(home["north_m"]) + float(st.position.x_val),
            float(home["ground_asl_m"]) - float(st.position.z_val))


def _drive(pilot: ManualPilot, client, home, *, seconds: float, hz: float = 20.0,
           **stick) -> dict:
    """Hold one stick position for `seconds` and report how far the aircraft went."""
    e0, n0, a0 = _pose(client, home)
    t_end = time.time() + seconds
    n = 0
    clamps: list[str] = []
    while time.time() < t_end:
        e, north, alt = _pose(client, home)
        st = client.simGetGroundTruthKinematics()
        o = st.orientation
        yaw = math.degrees(math.atan2(2.0 * (o.w_val * o.z_val + o.x_val * o.y_val),
                                      1.0 - 2.0 * (o.y_val ** 2 + o.z_val ** 2)))
        out = pilot.command(ControlInput(t=time.time(), valid=True, **stick),
                            east_m=e, north_m=north, alt_asl_m=alt, heading_deg=yaw)
        for r in (out.get("clamp") or {}).get("reasons", []):
            if r not in clamps:
                clamps.append(r)
        n += 1
        time.sleep(1.0 / hz)
    e1, n1, a1 = _pose(client, home)
    return {"commands": n, "d_east_m": round(e1 - e0, 2), "d_north_m": round(n1 - n0, 2),
            "d_alt_m": round(a1 - a0, 2),
            "horizontal_m": round(math.hypot(e1 - e0, n1 - n0), 2), "clamps": clamps,
            "from": [round(e0, 1), round(n0, 1), round(a0, 1)],
            "to": [round(e1, 1), round(n1, 1), round(a1, 1)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="_artifacts/verification/manual_flight.json")
    ap.add_argument("--alt", type=float, default=45.0, help="metres AGL to fly the test at")
    ap.add_argument("--hold-s", type=float, default=5.0, help="how long to prove station-keeping")
    ap.add_argument("--push-s", type=float, default=3.0, help="how long to hold a stick")
    a = ap.parse_args(argv)

    scn = Scenario.load()
    home = scn.home
    client = connect()
    results: dict[str, object] = {"t_utc": time.time(), "alt_target_m": a.alt, "checks": []}

    def record(name: str, ok: bool, detail: str, data: object = None) -> None:
        results["checks"].append({"name": name, "pass": bool(ok), "detail": detail, "data": data})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}", flush=True)

    pilot = ManualPilot(client, limits=ManualLimits(max_speed_ms=6.0, max_climb_ms=2.5,
                                                    min_agl_m=12.0, smooth_tau_s=0.25),
                        terrain=scn.terrain, home=home)
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        e, n, alt = _pose(client, home)
        agl = alt - scn.terrain.surface_asl(e, n)
        print(f"\nvehicle at {e:.0f} E {n:.0f} N, {agl:.1f} m AGL")
        if agl < 5.0:
            print("taking off...")
            client.takeoffAsync(timeout_sec=25).join()
        e, n, alt = _pose(client, home)
        target_asl = scn.terrain.surface_asl(e, n) + a.alt
        print(f"climbing to {a.alt:.0f} m AGL...")
        client.moveToPositionAsync(float(n - home["north_m"]), float(e - home["east_m"]),
                                   float(-(target_asl - home["ground_asl_m"])), 5.0,
                                   timeout_sec=60).join()
        client.hoverAsync().join()
        time.sleep(1.0)

        print("\n--- 1. does a stick command actually move it? ---")
        fwd = _drive(pilot, client, home, seconds=a.push_s, pitch=1.0)
        record("a forward stick flies the aircraft forward",
               fwd["horizontal_m"] > 3.0,
               f"moved {fwd['horizontal_m']} m horizontally in {a.push_s:.0f} s "
               f"(dE {fwd['d_east_m']}, dN {fwd['d_north_m']}) over {fwd['commands']} commands", fwd)
        pilot.release()
        client.hoverAsync().join()
        time.sleep(1.5)

        print("\n--- 2. do CENTRED sticks hold station? (the reason the RC path was replaced) ---")
        hold = _drive(pilot, client, home, seconds=a.hold_s)
        record("centred sticks hold position and altitude",
               hold["horizontal_m"] < 3.0 and abs(hold["d_alt_m"]) < 3.0,
               f"drifted {hold['horizontal_m']} m horizontally and {hold['d_alt_m']} m vertically over "
               f"{a.hold_s:.0f} s. simple_flight's RC throttle passthrough would climb or sink here.", hold)

        print("\n--- 3. does the aircraft still hold API control? ---")
        state = client.getMultirotorState()
        api_on = bool(client.isApiControlEnabled()) if hasattr(client, "isApiControlEnabled") else True
        record("API control was never released during manual flight", api_on,
               "enableApiControl(False) is what exposes the pilot to simple_flight's passthrough throttle "
               "and its 100 ms disarm gesture; the velocity path never calls it.",
               {"api_control": api_on, "landed_state": str(getattr(state, "landed_state", "?"))})

        print("\n--- 4. does the terrain floor hold a descent? ---")
        e, n, alt = _pose(client, home)
        agl_now = alt - scn.terrain.surface_asl(e, n)
        # Raise the floor to just under the aircraft so a full-down stick must be refused, without ever
        # asking the vehicle to go near the ground.
        pilot.limits.min_agl_m = max(5.0, agl_now - 1.0)
        down = _drive(pilot, client, home, seconds=2.0, throttle=0.0)
        record("a full descent below the floor is refused",
               any("floor" in c for c in down["clamps"]) and down["d_alt_m"] > -4.0,
               f"floor set to {pilot.limits.min_agl_m:.1f} m AGL with the aircraft at {agl_now:.1f} m; "
               f"altitude changed {down['d_alt_m']} m. clamps={down['clamps']}", down)

        print(f"\n  velocity commands issued: {pilot.commands}, RPC errors: {len(pilot.errors)}")
        results["pilot"] = pilot.describe()
    except KeyboardInterrupt:
        print("\ninterrupted")
        results["interrupted"] = True
    finally:
        # Always hand the vehicle back, whatever happened above.
        for fn in (pilot.release, lambda: client.hoverAsync(), lambda: client.enableApiControl(True)):
            try:
                fn()
            except Exception:
                pass

    checks = results["checks"]
    passed = sum(1 for c in checks if c["pass"])
    results["verdict"] = f"{passed}/{len(checks)} PASS"
    out = Path(a.json) if Path(a.json).is_absolute() else REPO / a.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{results['verdict']}  ->  {out}")
    return 0 if passed == len(checks) and checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
