"""Measure this machine's controller, once, so the demo never depends on a guessed button index.

    uv run python tools/live/pad_calibrate.py              # walk the mapping, write it, verify it
    uv run python tools/live/pad_calibrate.py --show       # just watch the pad live (no writes)
    uv run python tools/live/pad_calibrate.py --verify     # replay the saved map against the pad

Why this is a tool and not a constant
-------------------------------------
`sightline/mission/takeover.py` shipped with `XBOX_BUTTONS = {"resume": 0, "takeover": 1, ...}` and an honest
comment saying nobody had ever pressed them. That is a fine state for a state-machine test and a bad state
for a demo, because the ONLY way out of MANUAL was a button, so the entire hand-back path rested on four
unverified integers. Worse, the failure is silent and inverted: press RESUME, watch the aircraft fly home,
conclude the software is broken.

Axes are worse still. This project has already been bitten once: the "standard" SDL2 layout quoted in most
tutorials puts `pitch` on axis 4, which on this driver is a **trigger resting at -1.0**. Opening that source
reported full deflection with nobody touching the pad and took the aircraft into MANUAL immediately. So this
tool measures axes by asking the human to MOVE each one and watching which index changes, rather than by
trusting a table.

What it writes: `data/controller/<slug>.json`, loaded automatically by `PygameGamepadSource` for any pad
reporting that name, with `provenance: "measured"` and a timestamp. `sightline/mission/padmap.py` refuses a
saved map whose axis/button counts do not match the pad now attached, so a map cannot be applied to a
different device that happens to share a name.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.mission.padmap import (CONTROL_AXES, CONTROL_BUTTONS, OPTIONAL_BUTTONS,  # noqa: E402
                                      PadMap, load_pad_map, mark_measured, pad_slug, save_pad_map)

#: The order the human is walked through, and what to say. Each entry is
#: (axis name, instruction, the sign the reported value should end up having when held that way).
AXIS_STEPS: list[tuple[str, str, float]] = [
    ("throttle", "LEFT stick fully UP (climb)", +1.0),
    ("yaw", "LEFT stick fully RIGHT (spin right)", +1.0),
    ("pitch", "RIGHT stick fully UP (fly forward)", +1.0),
    ("roll", "RIGHT stick fully RIGHT (slide right)", +1.0),
]

#: Buttons, in the order they are asked for. The four mode buttons are required; the rest may be skipped
#: with ENTER, because a pad with six buttons is still a perfectly good demo pad.
BUTTON_STEPS: list[tuple[str, str, bool]] = [
    ("takeover", "TAKE CONTROL   (suggested: B / circle - the one you grab in a hurry)", True),
    ("resume", "GIVE IT BACK    (suggested: A / cross - resume the survey)", True),
    ("hold", "HOLD POSITION   (suggested: X / square)", True),
    ("rtl", "RETURN TO LAUNCH(suggested: Y / triangle)", True),
    ("mark", "MARK WHAT I SEE (suggested: right bumper) - adds a pin on the map", False),
    ("boost", "BOOST           (suggested: left bumper) - hold to fly faster", False),
    ("orbit", "ORBIT MY MARK   (suggested: right stick click) - circle the last mark", False),
    ("freefly", "FREE-FLY TOGGLE (suggested: Start) - survey <-> go anywhere", False),
    ("camera", "CYCLE CAMERA    (suggested: Back/Select) - rgb / segmentation / depth", False),
]

MOVE_THRESHOLD = 0.60          # how far an axis must travel to count as "the one they moved"
SETTLE_S = 0.35                # how long it must stay there, so a flick past centre is not a measurement


def _open_pygame():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.joystick.init()
    return pygame


def _pick_pad(pygame, index: int):
    n = pygame.joystick.get_count()
    if n == 0:
        print("No controller found.\n"
              "  * plug it in, then run this again\n"
              "  * on Windows an Xbox pad needs no driver; a DS4 usually wants DS4Windows or Steam Input\n"
              "  * a wireless pad that has gone to sleep enumerates as absent - press its home button")
        return None
    if n > 1 and index == 0:
        print(f"{n} controllers enumerated:")
        for i in range(n):
            j = pygame.joystick.Joystick(i)
            j.init()
            print(f"   [{i}] {j.get_name()}")
        print("Using [0]; pass --index N to pick another.\n")
    js = pygame.joystick.Joystick(index)
    js.init()
    return js


def _snapshot(pygame, js) -> tuple[list[float], list[int]]:
    pygame.event.pump()
    return ([float(js.get_axis(i)) for i in range(js.get_numaxes())],
            [int(js.get_button(i)) for i in range(js.get_numbuttons())])


def _wait_centred(pygame, js, rest: list[float], *, timeout_s: float = 10.0) -> None:
    """Block until every axis is back near its resting value, so consecutive steps cannot bleed together."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ax, btn = _snapshot(pygame, js)
        if all(abs(a - r) < 0.25 for a, r in zip(ax, rest)) and not any(btn):
            return
        time.sleep(0.02)


def _measure_axis(pygame, js, rest: list[float], label: str, want_sign: float) -> tuple[int, float] | None:
    """Return (axis index, sign) for the axis the human just moved, or None if they gave up.

    `sign` is what the raw reading must be MULTIPLIED by so that holding the control the way the instruction
    describes produces a POSITIVE value. That is the whole reason this is measured rather than assumed:
    SDL reports screen-style Y (down is +1) on most pads and joystick-style on some.
    """
    print(f"    hold {label} ... ", end="", flush=True)
    held_since: float | None = None
    winner: int | None = None
    t0 = time.time()
    while time.time() - t0 < 30.0:
        ax, _ = _snapshot(pygame, js)
        deltas = [abs(a - r) for a, r in zip(ax, rest)]
        if not deltas:
            break
        i = max(range(len(deltas)), key=lambda k: deltas[k])
        if deltas[i] >= MOVE_THRESHOLD:
            if winner != i:
                winner, held_since = i, time.time()
            elif held_since is not None and time.time() - held_since >= SETTLE_S:
                raw = ax[i]
                sign = want_sign if raw >= 0 else -want_sign
                print(f"axis {i}  (reads {raw:+.2f}, so sign {sign:+.0f})")
                _wait_centred(pygame, js, rest)
                return i, sign
        else:
            winner, held_since = None, None
        time.sleep(0.02)
    print("timed out - skipping")
    return None


def _measure_button(pygame, js, label: str, required: bool, taken: dict[int, str]) -> int | None:
    """Return the index of the button the human just pressed, or None when they skipped it."""
    hint = "" if required else "   (or press ENTER on the keyboard to skip)"
    print(f"    press {label}{hint} ... ", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 30.0:
        _, btn = _snapshot(pygame, js)
        pressed = [i for i, v in enumerate(btn) if v]
        if len(pressed) == 1:
            i = pressed[0]
            if i in taken:
                print(f"button {i} is already {taken[i].upper()} - press a different one")
                while any(_snapshot(pygame, js)[1]):
                    time.sleep(0.02)
                t0 = time.time()
                continue
            print(f"button {i}")
            while any(_snapshot(pygame, js)[1]):      # wait for release, or the next step catches it too
                time.sleep(0.02)
            return i
        if not required and _enter_pressed():
            print("skipped")
            return None
        time.sleep(0.02)
    print("timed out" + ("" if required else " - skipped"))
    return None


def _enter_pressed() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        return ch in (b"\r", b"\n")
    return False


def calibrate(index: int = 0, *, repo: Path | None = None) -> int:
    pygame = _open_pygame()
    js = _pick_pad(pygame, index)
    if js is None:
        return 2

    name = js.get_name()
    n_axes, n_buttons, n_hats = js.get_numaxes(), js.get_numbuttons(), js.get_numhats()
    print(f"\nCalibrating: {name}  ({n_axes} axes, {n_buttons} buttons, {n_hats} hat)\n")

    print("  Let go of everything for a moment...")
    time.sleep(0.8)
    rest, _ = _snapshot(pygame, js)
    print(f"  resting axes: {[round(a, 3) for a in rest]}")
    triggers = [i for i, a in enumerate(rest) if abs(a) > 0.5]
    if triggers:
        print(f"  axes {triggers} rest away from centre - those are TRIGGERS, and this tool will not let "
              f"one be mapped as a stick.")

    print("\n--- sticks -------------------------------------------------------------")
    axes: dict[str, int] = {}
    invert: list[str] = []
    for axis_name, label, want in AXIS_STEPS:
        got = _measure_axis(pygame, js, rest, label, want)
        if got is None:
            print(f"\nCould not measure {axis_name!r}. Nothing has been written. Run it again.")
            return 3
        i, sign = got
        if i in axes.values():
            clash = next(k for k, v in axes.items() if v == i)
            print(f"\naxis {i} is already {clash!r}. Two flight axes cannot share one stick direction - "
                  f"run it again and move the right one.")
            return 3
        axes[axis_name] = i
        if sign < 0:
            invert.append(axis_name)

    print("\n--- buttons ------------------------------------------------------------")
    buttons: dict[str, int] = {}
    taken: dict[int, str] = {}
    for btn_name, label, required in BUTTON_STEPS:
        i = _measure_button(pygame, js, label, required, taken)
        if i is None:
            if required:
                print(f"\n{btn_name!r} is required - there would be no way to use it. Nothing written.")
                return 3
            continue
        buttons[btn_name] = i
        taken[i] = btn_name

    m = PadMap(name=name, slug=pad_slug(name), axes=axes, buttons=buttons, invert=tuple(invert),
               rest_axes=tuple(round(a, 4) for a in rest), n_axes=n_axes, n_buttons=n_buttons, n_hats=n_hats)
    problems = m.validate()
    if problems:
        print("\nThis mapping is not safe to fly:")
        for p in problems:
            print(f"   - {p}")
        print("Nothing written.")
        return 4

    m = mark_measured(m)
    path = save_pad_map(m, repo=repo)
    print(f"\nWritten: {path}")
    print(f"  {m.summary()}")
    print("\nEvery flight from now on loads this automatically. Verifying live - move things, "
          "Ctrl+C when satisfied.\n")
    return show(index, repo=repo, saved=m)


def show(index: int = 0, *, repo: Path | None = None, saved: PadMap | None = None) -> int:
    """Live view of the pad through the map that will actually fly it."""
    pygame = _open_pygame()
    js = _pick_pad(pygame, index)
    if js is None:
        return 2
    m = saved or load_pad_map(js.get_name(), repo=repo, n_axes=js.get_numaxes(),
                              n_buttons=js.get_numbuttons(), n_hats=js.get_numhats())
    print(f"{m.summary()}\n")
    if not m.verified:
        print("  !! this mapping is NOT measured - the button names below are a guess\n")

    def bar(v: float, width: int = 21) -> str:
        mid = width // 2
        k = max(0, min(width - 1, int(round(mid + v * mid))))
        row = ["-"] * width
        row[mid] = "|"
        row[k] = "#"
        return "".join(row)

    try:
        while True:
            pygame.event.pump()
            vals = {}
            for a in CONTROL_AXES:
                i = m.axis(a)
                raw = float(js.get_axis(i)) if 0 <= i < js.get_numaxes() else 0.0
                vals[a] = m.apply_dead_zone(raw * m.sign(a))
            down = [n for n in CONTROL_BUTTONS + OPTIONAL_BUTTONS
                    if 0 <= m.button(n) < js.get_numbuttons() and js.get_button(m.button(n))]
            line = "  ".join(f"{a[:3]} {bar(vals[a])}" for a in CONTROL_AXES)
            print(f"\r{line}   [{' '.join(x.upper() for x in down) or '-':<28}]", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")
    return 0


def verify(index: int = 0, *, repo: Path | None = None) -> int:
    pygame = _open_pygame()
    js = _pick_pad(pygame, index)
    if js is None:
        return 2
    m = load_pad_map(js.get_name(), repo=repo, n_axes=js.get_numaxes(),
                     n_buttons=js.get_numbuttons(), n_hats=js.get_numhats())
    pygame.event.pump()
    rest = tuple(round(float(js.get_axis(i)), 4) for i in range(js.get_numaxes()))
    problems = m.validate(rest_axes=rest)
    print(m.summary())
    print(f"  provenance : {m.provenance}")
    print(f"  rest axes  : {list(rest)}")
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        return 1
    if not m.verified:
        print("  NOT MEASURED - the sticks are safe but the buttons are a guess. "
              "Run this tool with no arguments before a demo.")
        return 1
    print("  OK - measured, self-consistent, and safe to fly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0, help="which controller, when several are attached")
    ap.add_argument("--show", action="store_true", help="watch the pad live; write nothing")
    ap.add_argument("--verify", action="store_true", help="check the saved map against the attached pad")
    a = ap.parse_args(argv)
    if a.show:
        return show(a.index)
    if a.verify:
        return verify(a.index)
    return calibrate(a.index)


if __name__ == "__main__":
    raise SystemExit(main())
