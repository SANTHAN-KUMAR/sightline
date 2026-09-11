"""The judge demo, end to end: hand someone a pad, watch the map, never touch the keyboard.

    # terminal 1 - the C2 and the map
    uv run python -m sightline.api.serve --port 8781 --db _artifacts/store/demo.db --fresh

    # terminal 2 - pick ONE
    uv run python tools/live/demo_controller.py --live                 # real simulator, real pad, mode 1
    uv run python tools/live/demo_controller.py --live --free          # real simulator, real pad, mode 2
    uv run python tools/live/demo_controller.py --scripted             # no simulator, scripted judge

    # then open  http://127.0.0.1:8781/app/map/index.html

What the two demo modes are
---------------------------
**Mode 1 - takeover.** The aircraft flies its own survey. A judge picks up the pad and moves a stick; the
mode flips to MANUAL on that poll and the aircraft obeys them. They fly, they press MARK on anything they
spot, and when they stop flying the mission takes itself back after `--idle-resume-s` seconds and carries on
from where the aircraft is - it does not restart the pattern. Records keep arriving on the map the whole
time, in AUTO and in MANUAL alike.

**Mode 2 - free flight.** The same aircraft, the same pipeline, the same map, but no pattern at all: the
pilot goes anywhere in the valley and every frame still runs detect -> geolocate -> track -> dedup -> triage.
`--free` starts there; the FREE-FLY button moves between the two in the air, so one demo is both.

`--scripted` runs the identical code with a scripted hand on the sticks and recorded frames instead of a
simulator, which is how this is rehearsed (and screenshotted) with the editor shut. It says so in its own
output, on the map, and in the data card: **no human moved a stick in a scripted run.**

Pre-flight
----------
`--live` refuses to start on a pad whose buttons nobody has ever pressed, because the one failure a demo
cannot survive is a judge pressing RESUME and the aircraft flying home. Fix it once:

    uv run python tools/live/pad_calibrate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.mission import live as livemod  # noqa: E402
from sightline.mission.padmap import load_pad_map  # noqa: E402
from sightline.mission.takeover import ControlInput, ControlSource  # noqa: E402


class ScriptedJudge(ControlSource):
    """A judge on a timetable, keyed to the frame index the mission has reached.

    Each entry is ``(frame_idx, kwargs)``; from that frame onward the source reports those inputs until the
    next entry. That is enough to express "grab the sticks", "fly", "press MARK", "let go and walk away".

    It reports `valid=True` throughout, because the pad in a real demo stays plugged in. A step whose kwargs
    are `{"valid": False}` simulates the pad being put down or going flat, which is the other way the idle
    hand-back is supposed to fire.
    """

    name = "scripted"
    verified_against_hardware = False

    def __init__(self, script: list[tuple[int, dict]], mission: "livemod.LiveMission"):
        self.script = sorted(script, key=lambda x: x[0])
        self.mission = mission
        self.reads = 0

    def _current(self) -> dict:
        k = self.mission.takeover.frame_idx
        cur: dict = {}
        for at, kw in self.script:
            if k >= at:
                cur = kw
        return cur

    def read(self, now: float) -> ControlInput:
        self.reads += 1
        cur = self._current()
        kw = {k: v for k, v in cur.items() if k != "valid"}
        return ControlInput(t=now, valid=bool(cur.get("valid", True)), source=self.name, **kw)

    def describe(self) -> dict:
        return {"source": self.name, "verified_against_hardware": False,
                "device": "SCRIPTED (no human)", "buttons_verified_against_hardware": False,
                "script": [{"from_frame": a, **kw} for a, kw in self.script],
                "note": "SCRIPTED JUDGE - no human moved a physical stick during this run. The state "
                        "machine, the manual pilot, the envelope, the telemetry and the map are real."}


def build_script(a: argparse.Namespace) -> list[tuple[int, dict]]:
    """The rehearsal: every F3 path a judge can take, in the order a demo would show them.

    The idle hand-back is deliberately demonstrated TWICE and in both of its forms - sticks centred, and the
    pad going away - because it is the behaviour the whole demo turns on and the one nobody can see happen
    unless they are shown it.
    """
    t = a.takeover_at
    return [
        (0, {}),
        # --- mode 1: the takeover -----------------------------------------------------------------
        (t, {"pitch": 0.85}),                       # grab the sticks -> MANUAL, immediately
        (t + 10, {"pitch": 0.6, "roll": -0.4}),     # fly it
        (t + 18, {"mark": True}),                   # "there's someone down there"
        (t + 19, {"roll": 0.5}),
        (t + 26, {}),                               # let go, and walk away -> IDLE HAND-BACK
        # --- mode 1 again, ending with the pad put down ---------------------------------------------
        (t + 60, {"yaw": 0.7, "throttle": 0.8}),    # a second judge takes it
        (t + 68, {"mark": True}),
        (t + 69, {"boost": True, "pitch": 0.9}),    # BOOST across the valley
        (t + 78, {"valid": False}),                 # pad put down / battery flat -> IDLE HAND-BACK
        # --- the explicit buttons --------------------------------------------------------------------
        (t + 110, {"hold": True}),                  # HOLD -> position hold
        (t + 111, {}),
        (t + 125, {"resume": True}),                # RESUME -> AUTO (sticks have been centred > 1 s)
        (t + 126, {}),
        # --- mode 2: free flight ---------------------------------------------------------------------
        (t + 140, {"freefly": True}),               # go anywhere
        (t + 141, {"pitch": 0.7}),
        (t + 150, {"mark": True}),
        (t + 151, {"orbit": True}),                 # circle the mark, hands free
        (t + 152, {}),
        (t + 170, {"freefly": True}),               # back to the survey
        (t + 171, {}),
        # --- always available --------------------------------------------------------------------------
        (a.rtl_at, {"rtl": True}),
        (a.rtl_at + 1, {}),
    ]


def preflight(a: argparse.Namespace) -> int:
    """Say out loud what is about to fly, and refuse the one configuration a demo cannot survive."""
    print("=" * 78)
    print("SIGHTLINE - judge controller demo")
    print("=" * 78)
    if a.scripted:
        print("  source      : SCRIPTED JUDGE over recorded frames. No human will touch a stick, and no\n"
              "                simulator is involved. Everything downstream of the sticks is the real thing.")
        return 0

    try:
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("  NO CONTROLLER ATTACHED.\n"
                  "  Plug the pad in and run this again, or use --scripted to rehearse without one.")
            return 2
        js = pygame.joystick.Joystick(0)
        js.init()
        m = load_pad_map(js.get_name(), n_axes=js.get_numaxes(), n_buttons=js.get_numbuttons(),
                         n_hats=js.get_numhats())
        print(f"  controller  : {js.get_name()}")
        print(f"  mapping     : {m.provenance}")
        if not m.verified and not a.allow_unverified:
            print("\n  REFUSING TO START.\n"
                  "  Nobody has ever pressed the buttons on this pad, so RESUME / HOLD / RTL / MARK are\n"
                  "  guesses. A judge pressing RESUME and watching the aircraft fly home is the one\n"
                  "  failure this demo cannot survive.\n\n"
                  "      uv run python tools/live/pad_calibrate.py\n\n"
                  "  takes about a minute. Pass --allow-unverified to fly the guess anyway.")
            return 3
        for name in ("takeover", "resume", "hold", "rtl"):
            print(f"    {name:<9}: button {m.button(name)}")
        for name in ("mark", "boost", "orbit", "freefly", "camera"):
            if m.has(name):
                print(f"    {name:<9}: button {m.button(name)}")
    except ImportError:
        print("  pygame is not installed; cannot check the pad.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="fly the real simulator with a real pad")
    mode.add_argument("--scripted", action="store_true",
                      help="rehearse the whole demo with a scripted judge and recorded frames")
    ap.add_argument("--replay", default="_artifacts/live/clip_seed23",
                    help="the recorded run --scripted drives the live loop from")
    ap.add_argument("--out", default="")
    ap.add_argument("--c2", default="http://127.0.0.1:8781")
    ap.add_argument("--alt", type=float, default=45.0)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--free", action="store_true", help="start in FREE FLIGHT (demo mode 2)")
    ap.add_argument("--idle-resume-s", type=float, default=12.0)
    ap.add_argument("--manual-mode", choices=("velocity", "rc"), default="velocity")
    ap.add_argument("--detector", choices=("truth", "rgb", "model"), default="truth")
    ap.add_argument("--weights", default="")
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="fly a pad whose buttons have never been pressed (not for a real demo)")
    ap.add_argument("--takeover-at", type=int, default=40, help="scripted: frame the first judge grabs it")
    ap.add_argument("--rtl-at", type=int, default=300, help="scripted: frame the demo ends")
    ap.add_argument("--images", action="store_true", help="write frames to disk (slower)")
    a = ap.parse_args(argv)

    rc = preflight(a)
    if rc:
        return rc

    out = a.out or f"_artifacts/live/demo_{time.strftime('%Y%m%d-%H%M%S')}"
    argv2 = ["--out", out, "--c2", a.c2, "--alt", str(a.alt), "--speed", str(a.speed),
             "--detector", a.detector, "--idle-resume-s", str(a.idle_resume_s),
             "--manual-mode", a.manual_mode, "--max-minutes", str(a.max_minutes),
             "--shoot-transit", "--log-every", "40"]
    if a.weights:
        argv2 += ["--weights", a.weights]
    if a.free:
        argv2 += ["--free"]
    if not a.images:
        argv2 += ["--no-images"]
    if a.scripted:
        argv2 += ["--replay", a.replay, "--pace", str(a.pace)]
    elif not a.allow_unverified:
        argv2 += ["--require-verified-pad"]

    args = livemod.build_parser().parse_args(argv2)
    mission = livemod.LiveMission(args)
    if a.scripted:
        mission.control_override = ScriptedJudge(build_script(a), mission)
        print("\n  SCRIPTED JUDGE: no human touches a stick in this run.\n")
    else:
        print("\n  Hand the pad over. The aircraft is flying its survey; move a stick to take it.\n"
              f"  It hands itself back {a.idle_resume_s:.0f} s after you stop.\n"
              f"  Watch it happen at {a.c2}/app/map/index.html\n")

    rc = mission.run()
    report(mission, out)
    return rc


def report(mission: "livemod.LiveMission", out: str) -> None:
    """What the demo actually did - the thing to read out while the map is still on screen."""
    d = Path(out) if Path(out).is_absolute() else REPO / out
    try:
        log = json.loads((d / "mode_log.json").read_text(encoding="utf-8"))
        card = json.loads((d / "data_card.json").read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\n(could not read the run's own outputs: {type(e).__name__}: {e})")
        return

    print("\n" + "=" * 78)
    print("F3 mode log - every change of who was flying")
    print("=" * 78)
    for row in log:
        print(f"  frame {row['frame_idx']:>5}  {row['from']:>6} -> {row['to']:<6} "
              f"{row['reason']:<18} applied={row['applied']} in {row['apply_ms']} ms")

    tk = card.get("takeover", {})
    idle = [r for r in log if r["reason"] == "idle hand-back"]
    print("\n  seconds by mode  :", json.dumps(tk.get("seconds_by_mode", {})))
    print(f"  idle hand-backs  : {len(idle)}  (threshold {tk.get('idle_resume_s')} s)")
    print(f"  pilot marks      : {len(tk.get('pilot_marks', []))}")
    print(f"  manual flight    : {tk.get('manual_mode')}")
    print(f"  refusals         : {json.dumps(tk.get('refusals', []))}")
    mp = tk.get("manual_pilot") or {}
    if mp:
        print(f"  velocity commands: {mp.get('commands', 0)}, envelope enforced: "
              f"{mp.get('envelope_enforced')}, terrain floor: {mp.get('terrain_floor')}")
    print(f"\n  outputs          : {d}")


if __name__ == "__main__":
    raise SystemExit(main())
