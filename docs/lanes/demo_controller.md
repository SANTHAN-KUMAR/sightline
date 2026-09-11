# Lane: external-controller demo readiness (audit only — 2026-09-11 ~06:30)

**Read-only audit. I edited no code and no shared doc.** `sightline/mission/*`, `sim/**` and
`tools/capture|scene|train/*` are claimed by the orchestrator session in `COORDINATION.md`; every finding
below lands in those files, so nothing here is actionable by me without a handover.

Scope: the two demo modes the project wants to hand to judges.
* **Mode 1** — drone flies its own survey; a judge grabs the pad and it obeys; **after a while with no input
  it goes back to flying itself**; results keep landing on the dashboard throughout.
* **Mode 2** — free flight anywhere in the map, same live dashboard.

## State of the machine at audit time
Editor not running, no AirSim RPC, GPU idle, 5.5 GB RAM free (`sightline status`). Full suite
**928 passed, 2 skipped, 1 failed**; the failure
(`test_track.py::test_ultralytics_backend_is_reachable_but_not_exercised_here`) **passes in isolation** —
a test-order state leak, not a break. Xbox 360 pad **is attached right now**: 6 axes, 11 buttons, 1 hat,
axes 0-3 rest 0.000, axes 4-5 rest -1.000 (the trigger layout `takeover.py` already documents).

## What is genuinely built and works
`TakeoverMachine` / `ControlSource` / `VehicleAuthority` (`sightline/mission/takeover.py`, 622 lines) is
real, well-separated work: immediate stick takeover, gated hand-back, RTL from every state, every transition
timestamped into `mode_log.json` + the telemetry `mode` column + the map chip. The live loop
(`sightline/mission/live.py`) polls it at 50 Hz and the dashboard colours the track by mode
(`app/map/index.html:250`). `tests/test_live_mission.py` + `tests/test_api.py` = 89 green.
`_artifacts/live/map_live_manual_final.png` shows it working: `mode MANUAL · 45 m AGL`, 7 scored records,
`LIVE ws`, `outbox 0`, `errors 0`.

## Findings — what stands between this and handing a judge a controller

### 1. Mode 1's defining behaviour does not exist: there is no idle hand-back
`TakeoverMachine.poll` leaves MANUAL **only** on `ctl.resume` (`takeover.py:266-273`). There is no
inactivity timer anywhere. Measured: takeover, then 120 s of perfectly centred sticks and no buttons →
`mode = MANUAL, transitions = 1`. Pad unplugged mid-MANUAL (`valid=False`) → still MANUAL for ever; the
`not ctl.valid` branch deliberately holds the mode (`takeover.py:234-239`).

### 2. …and the button that *is* the only way back is unverified hardware
`XBOX_BUTTONS = {"resume": 0, "takeover": 1, "hold": 2, "rtl": 3}` (`takeover.py:54`) is a guess;
`buttons_verified_against_hardware: False` and the TRACKER says so. **(1) and (2) together mean there is
currently no demonstrated route from MANUAL back to AUTO with a physical pad.** The pad is attached —
this is a five-minute fix with a human pressing four buttons.

### 3. The drone's position on the dashboard freezes whenever the shutter doesn't fire
`push_pose` has exactly one call site: inside `handle_frame` (`live.py:440`). No frame → no pose → the
drone marker and its footprint stop moving. Pose is cheap and superseded every second; it should not be
coupled to the shutter.

### 4. Nothing is captured during a "transit" phase, and in MANUAL a transit never ends
`live.py:687` gates capture on `phase == "line"`. The phase loop breaks on arrival only when
`self.takeover.flying_itself` (`live.py:697`), so in MANUAL it runs to `--leg-timeout-s` (**default 400 s**).
A judge who takes over during a transit gets **up to 6.7 minutes of a completely dead dashboard** — no
records, and by (3) no drone movement either. This is the single most likely way the demo looks broken.

### 5. The 8° tilt gate throws away exactly the frames a judge generates
`_should_shoot` rejects any frame with roll or pitch above `--max-tilt-deg` (default 8) unconditionally,
MANUAL included (`live.py:717`). Anyone flying with authority banks past 8°, so the pins stop arriving
precisely while they are flying hardest. The camera is gimbal-stabilised, so the gate is not protecting
image quality. (The orchestrator session has independently queued raising this gate — it is discarding
~54 % of *survey* frames for the same reason.)

### 6. Coverage does not accumulate live
`FramePipeline` runs detect → geo → track → dedup → triage. The map's coverage/POD overlay is read from a
**pre-exported manifest** on disk (`sightline/api/coverage_feed.py:93,216`), not recomputed in flight, so
flying over new ground does not grow the searched-area layer. `live.py:711`'s docstring claims "Every frame
in MANUAL still runs detection, geolocation **and coverage**" — the first two are true, the third is not.
If "live results" for the judges includes the coverage layer, this is a build, not a wiring fix.

### 7. There is no free-flight mode at all (Mode 2)
No `--free` / manual-only entry point exists anywhere in `sightline/` or `tools/`. Mode 2 can only be
approximated by starting `live.py` and immediately grabbing the sticks, which drops straight into (4).

### 8. Handing this pad to an untrained judge is physically hazardous in-sim
In MANUAL, `enableApiControl(False)` gives the vehicle to simple_flight's RC path, and that path is not
a camera-drone flight mode:
* **Throttle is `Passthrough`** — `GoalMode()`'s default 4th axis
  (`Plugins/AirSim/.../firmware/interfaces/CommonStructs.hpp:287`), i.e. raw motor output, **not altitude
  hold**. A self-centred left stick is ~50 % motor, not hover: let go and the aircraft climbs or sinks.
  This is also what an idle judge in Mode 1 gets while (1) keeps it in MANUAL.
* **The disarm gesture is live**: yaw full-left + throttle ≤ 0.1 + roll ≥ 0.9, held **100 ms**
  (`firmware/RemoteControl.hpp:225-247`, `firmware/Params.hpp:36-40`). A judge yanking the left stick into
  the bottom corner cuts the motors mid-air.
A demo pad wants a velocity/position-hold mapping, not raw passthrough with a live disarm gesture.

### 9. None of the simulator half has ever run in PIE
`fly()`, `capture_live()`, `AirSimRcSource` and `VehicleAuthority` against a real client are **unexecuted**
against the running editor — TRACKER §"NOT done" states this plainly. Every demonstration so far is the
replay path driven by `tools/live/demo_takeover.py`'s scripted pilot, which the tool's own docstring is
honest about. The 394 ms API→RC handover number comes from `tools/day1/gamepad_airsim.py` (V8, 13 PASS),
not from `live.py`.

### 10. Minor
Shutdown while in MANUAL calls `landAsync` without re-enabling API control (`live.py:783-788`); it fails
into the bare `except` and the vehicle is left to the pad.

## Suggested order of work (if this lane is handed over)
1. (4) + (3) — decouple pose from the shutter; capture during transit; break the phase loop on a MANUAL
   idle. Cheapest fix, removes the "demo looks dead" failure.
2. (1) + (2) — an `--idle-resume-s` timer, and press the four buttons on the attached pad to settle the map.
3. (8) — a demo RC profile with altitude/velocity hold and the disarm gesture disabled.
4. (7) — a real `--free` mode that never issues a waypoint.
5. (5) — raise/skip the tilt gate in MANUAL (coordinate: the orchestrator is already editing that gate).
6. (6) — decide whether the demo needs live coverage; it is a build, not a wire.
7. (9) — one PIE flight, which is the only thing that converts all of the above from reasoned to measured.
