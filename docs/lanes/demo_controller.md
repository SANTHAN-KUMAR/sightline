# Lane: the judge controller demo (F3) — LANDED 2026-09-11

Hand a judge a gamepad. The aircraft is flying its own survey; they take it, fly it, mark what they see, and
when they stop the mission takes itself back and carries on. Or they go anywhere they like in the valley and
the whole pipeline runs on every frame regardless. Both on one screen, both live.

## Run it

```
uv run python tools/live/pad_calibrate.py                         # ONCE per controller, ~60 s
uv run python -m sightline.api.serve --port 8781 --db _artifacts/store/demo.db --fresh
uv run python tools/live/demo_controller.py --live --scenario nominal
#  -> http://127.0.0.1:8781/app/map/index.html
```

`--live --free` starts in free flight (mode 2); the FREE-FLY button moves between the two in the air.
`--scripted` rehearses the entire thing with a scripted judge over recorded frames and no simulator, which
is how it is demonstrated with the editor shut. `--scenario` is imported from `tools/live/demo.py`, so the
judge demo and the autonomous demo describe the same flight rather than two different 45 m.

| control | does |
|---|---|
| right stick | pitch / roll — fly |
| left stick | throttle / yaw — climb, descend, spin |
| **any stick** | takes control instantly. No button needed, no confirmation. |
| TAKEOVER / RESUME / HOLD / RTL | the §5.2 mode buttons. RTL from every state, always. |
| MARK | a gold pin on the map where the aircraft is. Only ever adds (R10). |
| BOOST | hold for 18 m/s instead of 9 |
| ORBIT | circle the last mark, hands free, nose pointed at it |
| FREE-FLY | survey ⇄ go anywhere |
| CAMERA | cycle rgb / segmentation / depth |
| *nothing, for 12 s* | **the mission takes itself back** and continues from where the aircraft is |

## What was wrong, and what each fix was

The audit that opened this lane is below under "The ten findings". Nine are closed; one is scoped out and
named. In order of how badly each would have broken a demo:

**1. Handing a judge the pad was physically unsafe.** §5.2 specifies the handover as
`enableApiControl(False)`, after which simple_flight obeys the RC channels. Reading the firmware this
project ships: the throttle channel is `GoalModeType::Passthrough`
(`firmware/interfaces/CommonStructs.hpp`) — **raw motor output, not altitude hold**, so a self-centring
stick is ~50 % motor and the aircraft climbs or sinks the moment anyone lets go. And the disarm gesture is
live — yaw full-left + throttle ≤ 0.1 + roll ≥ 0.9 held **100 ms** (`firmware/RemoteControl.hpp`,
`firmware/Params.hpp`) — so shoving the left stick into a corner, the single most common thing an
inexperienced person does with a gamepad, cuts the motors mid-air.

→ `sightline/mission/manual.py`. MANUAL now **keeps API control** and flies the pad as a velocity command.
Centred sticks are zero velocity, which is genuinely hold-station; there are no RC channels in the loop, so
neither the passthrough throttle nor the disarm gesture is reachable by the pilot; and the F2 envelope
(geofence, 120 m ceiling, terrain-relative floor) is enforced **on a human pilot** for the first time —
`sightline/mission/safety.py`'s own docstring is a complaint that the flight code enforced none of it.
`--manual-mode rc` still gives the literal §5.2 handover, and the data card records which one flew.

**2. There was no idle hand-back at all** — the behaviour demo mode 1 is entirely built on. `poll()` left
MANUAL only on a RESUME button. Measured before the fix: takeover, then 120 s of centred sticks → still
MANUAL; pad unplugged → still MANUAL for ever.

→ `TakeoverMachine.idle_resume_s` (default 12 s), `--idle-resume-s`, `0` restores the documented behaviour.
A pad that is **put down, unplugged or goes flat counts as idle from the first invalid poll**, because
"nobody is holding this" is the condition the timer exists to notice. RTL is never idle-resumed and the
constructor refuses to be configured otherwise.

**3. The only route back to AUTO was four unverified integers.** `XBOX_BUTTONS` was a guess and said so.
Press RESUME, watch the aircraft fly home, conclude the software is broken.

→ `sightline/mission/padmap.py`: the mapping is **data** with provenance
(`measured` / `partial` / `default-guess`), measured by `tools/live/pad_calibrate.py` into
`data/controller/<slug>.json`, refused if a "stick" rests like a trigger, refused if two controls share a
button, refused if the saved map is for a differently-shaped pad. `--require-verified-pad` (which the demo
runner sets) will not start on a guess. The HUD colours an unverified pad amber.

**4. The dashboard froze whenever the shutter didn't fire.** `push_pose` had one call site, inside
`handle_frame`. On a transit leg, or while a pilot hovered, or any time a frame was rejected, the drone
marker simply stopped — and a frozen marker reads as a crash.

→ `live.py::push_pose_now()`, rate-limited by `--pose-hz-s`, reusing kinematics the loop already sampled.

**5. A takeover during a transit leg produced nothing at all.** Capture was gated on `phase == "line"`, and
in MANUAL the phase loop broke only on `--leg-timeout-s` (**400 s**). Worst case: **6.7 minutes of a
completely dead dashboard**, and the pattern silently advancing several legs underneath the pilot.

→ transit legs shoot too; the phase budget is AUTO time only, so the pattern does not advance while a human
is flying.

**6. The tilt gate discarded exactly the frames a judge generates.** `_should_shoot` rejected roll/pitch
above `--max-tilt-deg` (8) unconditionally. Anyone flying with authority banks past that, so the map went
quiet in proportion to how hard they flew — and the camera is gimbal-stabilised, so it was not protecting
image quality.

→ `--manual-tilt-deg` (35) applies while a human is flying. `--max-tilt-deg` is untouched; the data card
records both.

**7. There was no free-flight mode** (demo mode 2). → `live.py::fly_free()`: a real loop that issues no
waypoint, shutters on time rather than along-track distance, holds station when nobody is flying, and runs
ORBIT. `--free`, or the FREE-FLY button in the air.

**8. Nothing showed the judge what they were doing.** → `sightline/api/control_feed.py`, two routes, a
`"control"` live message, and the HUD (flight mode, both sticks, every button, the idle countdown ring,
envelope warnings, an append-only event feed, gold pilot-mark pins). The WS snapshot is capped at 12 events
to stay inside §5.8's 5 KB frame; the HTTP route serves the whole log.

**9. None of the simulator half had ever run.** → **closed, see below.**

**10. Coverage still does not accumulate live.** `FramePipeline` runs detect → geo → track → dedup →
triage; the map's POD overlay is a pre-exported manifest. `live.py`'s docstring claimed "and coverage" and
now does not. **This is the one finding left open** — it is a build, not a wiring fix, and it is named here
rather than quietly dropped.

## LOOKED AT — and the defect that only appeared when it flew

`docs/QUALITY_GATE.md` is a hard rule, and it earned its keep twice in this lane.

**Flown against the live simulator** — `tools/live/verify_manual_flight.py`, **4/4 PASS**,
`_artifacts/verification/manual_flight.json`. 196 velocity commands, 0 RPC errors, no images, vehicle
handed back:

| check | measured |
|---|---|
| a forward stick flies it | **12.87 m in 3 s** through `moveByVelocityBodyFrameAsync` |
| centred sticks hold station | **1.87 m horizontal, 0.11 m vertical drift over 5 s** |
| API control never released | confirmed in flight |
| the terrain floor refuses a descent | held, and **says so** |

**Check 4 failed on the first flight, and it was a real defect.** The envelope bleeds a descent off over the
last 5 m, but only the *hard* stop set a flag — so with the floor 1 m below the aircraft the descent was
correctly reduced and `clamp.reasons` came back **empty**. The envelope was overriding the pilot while the
HUD said nothing: the "input silently ignored, therefore the controller is broken" failure that
`EnvelopeClamp` exists to prevent. Fixed in `_clamp_vertical` (the soft region now reports `easing off the
floor/ceiling`), two regression tests, re-flown, 4/4. **No amount of reading would have found it** — the
unit tests all passed, because they asserted the hard stop.

**The HUD, photographed** — `_artifacts/live/hud_ring.png` (real Edge, 0 external requests, 0 console
errors): amber MANUAL border, the **idle countdown ring at 28 s**, 2 marks, `70 s flown by hand`, the event
feed carrying `PILOT MARK #1/#2` and `AUTO -> MANUAL stick deflection`, gold `M1` pin on the map, records
still arriving. Also `_artifacts/live/hud_countdown.png` with ORBIT engaged.

Two things the pictures caught that nothing else did:
* the first screenshot showed a HUD that was rendering perfectly and was **completely invisible** — the
  capture lane's `#camera` panel is `z-index:6` in the same corner. Moved to bottom-right; both now visible.
* `headless_check.mjs --cdp-port` defaults to 9333 and both sessions use it, so a screenshot attached to the
  *other* session's browser and photographed the wrong machine while reporting my URL. Use a private port.

## Where it stands

* **Tests:** 44 new in `tests/test_pilot_demo.py`; `test_live_mission.py` + `test_api.py` 89 green. Three
  `VehicleAuthority` tests were pinned to `manual_mode="rc"` — they were written for the old default and
  still describe real behaviour.
* **R10:** scanned with the project's own scanner (`sightline/triage/guardrails.py`), not a token list, over
  every new module. Two hits were fixed by **removing the operation**, not by writing an allowance: a
  `ControlState.reset()` nobody called, and a dict mutation in the scripted judge. A mark has no unmark.
* **Not done, named:** live coverage accumulation (finding 10); the four demo buttons have still never been
  pressed on physical hardware (`pad_calibrate.py` exists and takes a minute, but it needs a human's thumb —
  `--require-verified-pad` refuses to start until someone has); `--detector rgb` unexercised here.

## Files

Mine: `sightline/mission/{padmap,manual}.py`, `sightline/api/control_feed.py`,
`tools/live/{pad_calibrate,demo_controller,verify_manual_flight}.py`, `tests/test_pilot_demo.py`,
`data/controller/`. Edited: `sightline/mission/takeover.py`, `sightline/api/{app,live}.py`,
`sightline/mission/live.py` (five surgical regions), `app/map/index.html` (**handed to the frontend lane
2026-09-11 07:4x — inventory in `COORDINATION.md`; I no longer touch it**).
