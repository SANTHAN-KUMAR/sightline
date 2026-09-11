"""F3 end to end on the live map: AUTO -> MANUAL -> AUTO -> HOLD -> AUTO -> RTL, while records arrive.

    uv run python -m sightline.api.serve --port 8781 --db _artifacts/store/demo.db --fresh
    uv run python tools/live/demo_takeover.py --replay _artifacts/live/clip_seed23 --c2 http://127.0.0.1:8781

Everything except the pilot is real: the same `TakeoverMachine`, the same `VehicleAuthority`, the same
telemetry, the same C2 push, the same map. What is scripted is the CONTROLLER INPUT - a `ScriptedSource`
stands in for a human hand on the sticks, deflecting at a chosen frame and pressing RESUME later.

**Stated plainly, because it matters:** no human moved a physical stick during this run. A real Xbox 360 pad
is attached and `PygameGamepadSource` reads it correctly (axes measured, resting values checked), but
nobody was there to press a button, so the physical-gamepad path is exercised only as far as reading it.
`tests/test_live_mission.py::test_an_untouched_pad_does_not_take_over` covers the hardware read; this covers
everything downstream of the stick.
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
from sightline.mission.takeover import ControlInput, ControlSource  # noqa: E402


class ScriptedSource(ControlSource):
    """A pilot on a timetable, keyed to the FRAME index the mission has reached.

    Each entry is ``(frame_idx, kwargs)``; from that frame onward the source reports those inputs until the
    next entry. That is enough to express "grab the sticks", "let go", "press RESUME a second later".
    """

    name = "scripted"
    verified_against_hardware = False

    def __init__(self, script: list[tuple[int, dict]], mission: "livemod.LiveMission"):
        self.script = sorted(script, key=lambda x: x[0])
        self.mission = mission
        self.applied: list[str] = []

    def _current(self) -> dict:
        k = self.mission.takeover.frame_idx
        cur: dict = {}
        for at, kw in self.script:
            if k >= at:
                cur = kw
        return cur

    def read(self, now: float) -> ControlInput:
        return ControlInput(t=now, valid=True, source=self.name, **self._current())

    def describe(self) -> dict:
        return {"source": self.name, "verified_against_hardware": False,
                "script": [{"from_frame": a, **kw} for a, kw in self.script],
                "note": "SCRIPTED PILOT - no human moved a physical stick during this run. The state "
                        "machine, the vehicle authority, the telemetry and the map are the real ones."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", default="_artifacts/live/clip_seed23")
    ap.add_argument("--out", default="_artifacts/live/run_takeover")
    ap.add_argument("--c2", default="http://127.0.0.1:8781")
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--takeover-at", type=int, default=90)
    ap.add_argument("--release-at", type=int, default=150)
    ap.add_argument("--hold-at", type=int, default=200)
    ap.add_argument("--hold-resume-at", type=int, default=240)
    ap.add_argument("--rtl-at", type=int, default=300)
    a = ap.parse_args()

    argv = ["--replay", a.replay, "--out", a.out, "--c2", a.c2, "--pace", str(a.pace),
            "--detector", "truth", "--no-images", "--log-every", "40", "--verify"]
    args = livemod.build_parser().parse_args(argv)
    mission = livemod.LiveMission(args)
    # The pilot's timetable. "sticks centred >= 1 s" is a real gate, so RESUME is pressed a few frames after
    # the sticks are released rather than in the same instant.
    script = [
        (0, {}),
        (a.takeover_at, {"roll": 0.85}),                       # grab the sticks -> MANUAL, immediately
        (a.release_at, {}),                                    # let go; the 1 s centred clock starts
        (a.release_at + 4, {"resume": True}),                  # RESUME -> AUTO, re-plan from here
        (a.release_at + 5, {}),
        (a.hold_at, {"hold": True}),                           # HOLD -> position hold
        (a.hold_at + 1, {}),
        (a.hold_resume_at, {"resume": True}),                  # back to AUTO
        (a.hold_resume_at + 1, {}),
        (a.rtl_at, {"rtl": True}),                             # RTL, always available
        (a.rtl_at + 1, {}),
    ]
    mission.control_override = ScriptedSource(script, mission)
    print("SCRIPTED PILOT: no human touched a stick in this run. State machine, authority, telemetry and "
          "map are real.\n")
    rc = mission.run()

    log = json.loads((mission.out / "mode_log.json").read_text())
    print("\n--- F3 mode log ---")
    for row in log:
        print(f"  frame {row['frame_idx']:>5}  {row['from']:>6} -> {row['to']:<6} "
              f"{row['reason']:<20} applied={row['applied']} in {row['apply_ms']} ms")
    card = json.loads((mission.out / "data_card.json").read_text())
    print("\nseconds by mode:", json.dumps(card["takeover"]["seconds_by_mode"]))
    print("refusals:", json.dumps(card["takeover"]["refusals"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
