"""Live gamepad readout: prove the pad is delivering input, and show what the takeover machine would see.

    uv run python tools/scene/_pad_probe.py [seconds]

The live mission enumerated the pad ("Xbox 360 Controller", axes verified) and yet a deflection produced no
takeover. That leaves three candidates, and this separates them:

  1. SDL is not delivering CHANGING values headless (axes frozen at rest)  -> values below will not move;
  2. values move but stay under the deadband                               -> values move, "TAKEOVER" stays no;
  3. values move and clear the deadband                                    -> the fault is downstream, in the
                                                                             mission loop rather than the pad.

It prints the raw axes, the four mapped control axes, any pressed button index, and whether the deflection
would cross `TakeoverMachine.DEADBAND` - so a button index that is wrong in the table shows up as "button N
pressed" with the wrong name beside it.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0

try:
    from sightline.mission.takeover import TakeoverMachine

    DEADBAND = float(TakeoverMachine.DEADBAND)
except Exception:                                            # noqa: BLE001
    DEADBAND = 0.15

import pygame

pygame.init()
pygame.joystick.init()
n = pygame.joystick.get_count()
print(f"joysticks enumerated: {n}")
if n == 0:
    print("FAIL: SDL sees no joystick from this process. If the mission is holding it exclusively, that is\n"
          "      itself the finding; otherwise the pad is not visible to SDL at all.")
    raise SystemExit(2)

js = pygame.joystick.Joystick(0)
js.init()
print(f"device: {js.get_name()}   axes {js.get_numaxes()}  buttons {js.get_numbuttons()}  hats {js.get_numhats()}")
print(f"deadband: {DEADBAND}")
print(f"\nMOVE THE STICKS AND PRESS BUTTONS for {SECONDS:.0f}s ...\n")

AX = {"yaw": 0, "throttle": 1, "roll": 2, "pitch": 3}
rest = None
moved = {k: 0.0 for k in AX}
moved_all = [0.0] * js.get_numaxes()
pressed: dict[int, int] = {}
hat_seen: dict[int, tuple] = {}
t0 = time.time()
last = 0.0

while time.time() - t0 < SECONDS:
    pygame.event.pump()
    axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
    if rest is None:
        rest = list(axes)
    btns = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
    for b in btns:
        pressed[b] = pressed.get(b, 0) + 1
    for i in range(len(axes)):
        moved_all[i] = max(moved_all[i], abs(axes[i] - rest[i]))
    for k, i in AX.items():
        if i < len(axes):
            moved[k] = max(moved[k], abs(axes[i] - rest[i]))
    hats = [js.get_hat(i) for i in range(js.get_numhats())]
    for i, h in enumerate(hats):
        if h != (0, 0):
            hat_seen[i] = h
    if time.time() - last > 0.5:
        last = time.time()
        # per-axis delta from rest, so a dead axis is obvious next to a live one
        d = [round(abs(axes[i] - rest[i]), 3) for i in range(len(axes))]
        defl = max((abs(axes[i] - rest[i]) for i in AX.values() if i < len(axes)), default=0.0)
        print(f"  axes {axes}  delta {d}  hats {hats}  buttons {btns}  "
              f"stick-deflection {defl:.3f} TAKEOVER={'YES' if defl > DEADBAND else 'no'}")
    time.sleep(0.03)

print("\n--- summary ---")
print(f"max deflection per control axis: { {k: round(v, 3) for k, v in moved.items()} }")
print(f"deadband {DEADBAND}  ->  "
      f"{'AT LEAST ONE AXIS WOULD TRIGGER TAKEOVER' if max(moved.values(), default=0) > DEADBAND else 'NO AXIS EVER CLEARED THE DEADBAND'}")
names = {0: "resume", 1: "takeover", 2: "hold", 3: "rtl"}
if pressed:
    print("buttons seen (index -> polls held, mapped name):")
    for b, c in sorted(pressed.items()):
        print(f"   button {b}: {c} polls   name in table: {names.get(b, '(unmapped)')}")
else:
    print("NO BUTTON PRESS SEEN AT ALL")
print(f"hats that moved: {hat_seen or 'none'}")
live = [i for i in range(js.get_numaxes()) if abs(moved_all[i]) > 0.2]
print(f"axes that actually moved: {live or 'NONE'}   (0-3 should be the two sticks)")
pygame.joystick.quit()
pygame.quit()
