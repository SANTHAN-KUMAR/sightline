"""ONE command: fly the drone in the simulator with the trained model and stream results to the dashboard.

    uv run python tools/live/demo.py                    # the demo
    uv run python tools/live/demo.py --check            # verify everything, start nothing
    uv run python tools/live/demo.py --pytorch          # use best.pt instead of the TensorRT engine
    uv run python tools/live/demo.py --alt 45 --speed 7

What it does, in order, refusing to continue if a step cannot be verified:

  1. the C2 backend            - starts `sightline.api.serve` if nothing answers on the port
  2. the simulator             - checks AirSim has a VEHICLE, not just an open port. PIE started in
                                 Simulate mode spawns no pawn, AirSim reports "no compatible vehicles",
                                 and every flight call then blocks forever at 0 % CPU with nothing moving.
                                 That cost an hour once; it is now a hard precondition.
  3. the detector              - prefers the TensorRT engine (125 ms/frame measured) over best.pt (159 ms)
  4. the flight                - `sightline.mission.live`, which runs the FULL pipeline per frame
                                 (detect -> geolocate -> track -> dedup -> triage) and POSTs each record
                                 to the C2 over the same WebSocket the dashboard listens on

Then it prints the dashboard URL and streams the per-frame line so the operator can watch detections,
tracks and records accumulate while the aircraft is still flying.

Nothing here re-implements the pipeline: the flight plan comes from `sightline.mission.pattern` (the same
module the dataset capture flew) and the stages from `sightline.pipeline.FramePipeline` (the same one the
offline replay calls). One lawnmower, one detect/geolocate/track/dedup/triage order.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

UV = r"D:\Tools\uv\uv.exe"
#: Scenario variations - each one a DIFFERENT DETECTION PROBLEM, not a different label.
#:
#: MEASURED 2026-09-11 by tools/live/_check_scenarios.py: `simSetTimeOfDay` and `simSetWeatherParameter`
#: DO NOTHING in this level. Five presets spanning 06:40 to 18:15 with rain up to 0.6 rendered frames that
#: are pixel-for-pixel the same lighting: luma spread 4.7 %, R/B spread 1.0 %, and the contact strip shows
#: five identical images. The level's sun is not the actor AirSim's time-of-day drives, and the weather FX
#: are not enabled in this world. Shipping those as "five slices" would have been five labels on one slice.
#:
#: The same measurement casts doubt on the CAPTURED DATASET: its passes are labelled `clear_morning` and
#: `clear_midday`, and dataset_gate.py measured luma 0.640 vs 0.649 for them - the same signature. Treat
#: that dataset as ONE lighting condition until the sun is wired.
#:
#: So these vary what this project demonstrably controls. Altitude is the single largest lever on recall
#: (SOLUTION_DOC 5.5: pixels-on-target), and speed sets how many shutter releases a survivor gets, which is
#: what the section 5.6 confirmation gate consumes. Both are real, measurable and under our control.
SCENARIOS: dict[str, dict] = {
    "nominal": {"alt": 45.0, "speed": 7.0, "shutter": 4.0,
                "why": "the ACCEPTANCE slice: 45 m, inside the 40-60 m nominal band (5.5c step 5). "
                       "Every headline number comes from here."},
    "low":     {"alt": 30.0, "speed": 6.0, "shutter": 3.0,
                "why": "30 m: BELOW the nominal band. Roughly 1.5x the pixels on target and a narrower "
                       "swath - the easiest detection and the slowest coverage. A named slice, not the "
                       "headline."},
    "high":    {"alt": 80.0, "speed": 9.0, "shutter": 6.0,
                "why": "80 m: ABOVE the band. Pixels-on-target roughly halve, so small and prone targets "
                       "fall toward the 20 px floor. A named HARD slice; never averaged with acceptance."},
    "slow":    {"alt": 45.0, "speed": 4.0, "shutter": 3.0,
                "why": "45 m at 4 m/s: the tracker's best case. A survivor gets ~7 shutter releases "
                       "instead of ~4, so the 3-hits-in-window gate closes far more often. Shows what "
                       "recall looks like when the loop is not starved."},
    "fast":    {"alt": 45.0, "speed": 12.0, "shutter": 8.0,
                "why": "45 m at 12 m/s: the survey speed the dataset was flown at. A target is in frame "
                       "for ~2 shutter releases, BELOW the 3 the gate needs - this is the configuration "
                       "that produced detections but no confirmed tracks."},
}


#: The acceptance slice. Derived from the table so it cannot drift out of `choices` again.
DEFAULT_SCENARIO = "nominal"


def apply_scenario(name: str) -> str:
    """Scenario state that the SIMULATOR must be told about. Currently none - see the note above.

    Kept as the seam: when the level's sun is wired to AirSim's time-of-day, the weather and lighting go
    here and `_check_scenarios.py` becomes the test that they actually took effect.
    """
    sc = SCENARIOS[name]
    return f"{name}: {sc['alt']:.0f} m AGL, {sc['speed']:.0f} m/s, shutter {sc['shutter']:.0f} m"


ENGINE = REPO / "models/detect/f8b_sim/weights/best.engine"
WEIGHTS = REPO / "models/detect/f8b_sim/weights/best.pt"


def port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:                                            # noqa: BLE001
        return False
    finally:
        s.close()


def wait_for(fn, timeout_s: float, what: str) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if fn():
            return True
        time.sleep(2)
    print(f"  TIMEOUT waiting for {what} ({timeout_s:.0f}s)")
    return False


def c2_up(port: int) -> bool:
    if not port_open(port):
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
            return bool(json.load(r).get("ok"))
    except Exception:                                            # noqa: BLE001
        return False


def sim_has_vehicle() -> tuple[bool, str]:
    """AirSim answering is NOT enough - it must have a vehicle, or the flight blocks forever."""
    if not port_open(41451):
        return False, "AirSim not listening on 41451 - is the editor open with PIE running?"
    try:
        import contextlib
        import io

        import cosysairsim as airsim                             # noqa: PLC0415

        with contextlib.redirect_stdout(io.StringIO()):
            c = airsim.MultirotorClient()
            c.confirmConnection()
            veh = list(c.listVehicles())
            t0 = c.getMultirotorState().timestamp
        time.sleep(1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            t1 = c.getMultirotorState().timestamp
        if not veh:
            return False, ("connected but NO VEHICLE. PIE is in Simulate mode; it must be PLAY "
                           "(LevelEditorSubsystem.editor_request_begin_play), which spawns the pawn.")
        if t1 <= t0:
            return False, f"vehicle {veh} present but the sim clock is FROZEN (paused)"
        return True, f"vehicles={veh}, clock ticking"
    except Exception as exc:                                     # noqa: BLE001
        return False, f"RPC failed: {type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO,
                    help="which world to fly: " + "; ".join(f"{k} = {v['why'].split(chr(46))[0]}"
                                                            for k, v in SCENARIOS.items()))
    ap.add_argument("--alt", type=float, default=0.0,
                    help="metres AGL; 0 = take the scenario's own altitude")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="7 m/s keeps a survivor in frame for ~4 shutter releases at the measured loop rate, "
                         "which is what the section 5.6 confirmation gate needs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--shutter-m", type=float, default=0.0,
                    help="metres between shutter releases; 0 = the scenario's own value")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--pytorch", action="store_true", help="use best.pt instead of the TensorRT engine")
    ap.add_argument("--engine", action="store_true",
                    help="force the TensorRT engine even with the editor running (it will likely die with "
                         "CUDA_ERROR_ILLEGAL_ADDRESS: 2.6 GB of context plus a 4K renderer exceeds 8 GB)")
    ap.add_argument("--check", action="store_true", help="verify everything and exit, starting nothing")
    ap.add_argument("--no-reset", action="store_true",
                    help="take off from wherever the aircraft is, instead of resetting it to the pad first. "
                         "The reset is what makes two runs of one scenario fly the same ground track.")
    ap.add_argument("--ignore-safety", action="store_true",
                    help="fly a plan the battery model rejects (the violation is stamped in the data card)")
    a = ap.parse_args()

    sc = SCENARIOS[a.scenario]
    if a.alt <= 0:
        a.alt = float(sc["alt"])
    if a.speed <= 0:
        a.speed = float(sc["speed"])
    if a.shutter_m <= 0:
        a.shutter_m = float(sc["shutter"])
    print("=" * 78)
    print("SIGHTLINE — LIVE DEMO")
    print("=" * 78)
    print(f"  scenario    {a.scenario}  -  {sc['why']}")

    # --- 1. detector --------------------------------------------------------------------------------
    # MEASURED 2026-09-11: the TensorRT engine is faster (125 ms vs 159 ms) but its execution context alone
    # allocates 2,642 MiB. With the Unreal editor rendering a 4K scene (~4.3 GB) that does not fit in this
    # 8 GB card, and TensorRT dies with CUDA_ERROR_ILLEGAL_ADDRESS inside Ultralytics' warmup. So the engine
    # is for UNCONTENDED inference - offline replay, or the Jetson the architecture actually targets - and a
    # live flight beside the editor uses PyTorch. Choosing automatically beats crashing mid-demo.
    editor_up = False
    try:
        import psutil                                            # noqa: PLC0415

        editor_up = any("UnrealEditor" in (pr.info.get("name") or "")
                        for pr in psutil.process_iter(["name"]))
    except Exception:                                            # noqa: BLE001
        pass
    if editor_up and not a.engine and ENGINE.exists():
        print("  note        editor is running: using PyTorch, not the engine "
              "(TensorRT's 2.6 GB context + the renderer does not fit in 8 GB)")
        a.pytorch = True

    if a.pytorch or not ENGINE.exists():
        if not WEIGHTS.exists():
            print(f"FATAL: no detector. {WEIGHTS.relative_to(REPO)} is missing - run the fine-tune first.")
            return 2
        det_path, det_note = WEIGHTS, "PyTorch fp16 (~159 ms/frame measured)"
        if not a.pytorch:
            det_note += "  [no TensorRT engine: build with tools/train/_build_engine.py]"
    else:
        det_path, det_note = ENGINE, "TensorRT fp16 (~125 ms/frame measured, 1.27x over PyTorch)"
    print(f"  detector    {det_path.name}  -  {det_note}")

    # --- 2. C2 backend ------------------------------------------------------------------------------
    if c2_up(a.port):
        print(f"  dashboard   already serving on :{a.port}")
    elif a.check:
        print(f"  dashboard   NOT RUNNING on :{a.port}")
    else:
        print(f"  dashboard   starting sightline.api.serve on :{a.port} ...")
        subprocess.Popen([UV, "run", "python", "-m", "sightline.api.serve", "--port", str(a.port)],
                         cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_for(lambda: c2_up(a.port), 90, "the C2 server"):
            return 2
        print(f"  dashboard   up on :{a.port}")

    # --- 3. simulator -------------------------------------------------------------------------------
    ok, why = sim_has_vehicle()
    print(f"  simulator   {'OK - ' + why if ok else 'NOT READY - ' + why}")
    if not ok:
        print("\nThe simulator is the one thing this script will not start for you: the editor takes minutes\n"
              "to open and PIE has to be in PLAY mode. Open the project, press Play, then re-run.")
        if not a.check:
            return 2

    if ok and not a.check:
        applied = apply_scenario(a.scenario)
        if applied:
            print(f"  world set   {applied}")

    url = f"http://127.0.0.1:{a.port}/app/map/index.html"
    print(f"\n  DASHBOARD   {url}")
    if a.check:
        print("\n--check: everything above was verified; nothing was started.")
        return 0 if ok else 2

    # --- 4. fly -------------------------------------------------------------------------------------
    out = REPO / "_artifacts/dataset/live_demo"
    cmd = [UV, "run", "python", "-u", "-m", "sightline.mission.live",
           "--alt", str(a.alt), "--speed", str(a.speed),
           "--detector", "rgb", "--weights", str(det_path), "--conf", str(a.conf),
           "--shutter-m", str(a.shutter_m), "--max-tilt-deg", "25",
           "--c2", f"http://127.0.0.1:{a.port}", "--out", str(out),
           "--max-minutes", str(a.minutes),
           # This demo is the AUTONOMOUS flight: detect -> geolocate -> track -> dedup -> triage -> map.
           # It needs no pilot, and constructing one currently raises in `pilot.describe()` (a slots
           # member_descriptor reaching json.dumps) while the gamepad work is in flight in another session.
           # `--control none` keeps the two demos independent rather than coupling this one to that fix.
           "--control", "none"]
    if not a.no_reset:
        # Same button, same flight. See live.py's fly() for why this is not a cosmetic nicety.
        cmd.append("--reset-world")
    if a.ignore_safety:
        cmd.append("--ignore-safety")

    print("\n" + "=" * 78)
    print("FLYING — open the dashboard above to watch records arrive")
    print("=" * 78 + "\n")
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
