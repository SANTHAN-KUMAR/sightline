"""Day-1 test #4 (part 1, no simulator): is a gamepad connected, and does live input arrive?
Reads Windows XInput directly (xinput1_4.dll via ctypes, no install) and pygame's joystick layer if installed.
Samples for --seconds and reports the range each axis covered and every button seen, so a human wiggling the
sticks proves the full input path. Run: uv run python -u tools/day1/gamepad_check.py [--seconds 25]
"""

import argparse
import ctypes
import json
import time
from ctypes import wintypes


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", wintypes.WORD), ("bLeftTrigger", ctypes.c_ubyte), ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short), ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short), ("sThumbRY", ctypes.c_short)]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", XINPUT_GAMEPAD)]


BUTTONS = {0x0001: "DPAD_UP", 0x0002: "DPAD_DOWN", 0x0004: "DPAD_LEFT", 0x0008: "DPAD_RIGHT", 0x0010: "START",
           0x0020: "BACK", 0x0040: "LEFT_THUMB", 0x0080: "RIGHT_THUMB", 0x0100: "LB", 0x0200: "RB",
           0x1000: "A", 0x2000: "B", 0x4000: "X", 0x8000: "Y"}


def main(seconds: float, wait_for_input: float = 0.0) -> None:
    out = {"xinput": {}, "pygame": {}}
    xi = ctypes.WinDLL("xinput1_4.dll")
    st = XINPUT_STATE()
    slots = [i for i in range(4) if xi.XInputGetState(i, ctypes.byref(st)) == 0]
    out["xinput"]["connected_slots"] = slots

    pg_js = None
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        out["pygame"]["count"] = n
        if n:
            pg_js = pygame.joystick.Joystick(0)
            pg_js.init()
            out["pygame"].update(name=pg_js.get_name(), guid=pg_js.get_guid(), axes=pg_js.get_numaxes(),
                                 buttons=pg_js.get_numbuttons(), hats=pg_js.get_numhats())
    except ImportError:
        out["pygame"]["error"] = "pygame not installed in this env"

    print(json.dumps(out, indent=1), flush=True)
    if not slots:
        print("NO XInput controller connected", flush=True)
        return
    slot = slots[0]
    if wait_for_input:
        print(f"WAITING up to {wait_for_input:.0f} s for the first stick/button/trigger input on slot {slot} ...", flush=True)
        t_w = time.time()
        while time.time() - t_w < wait_for_input:
            xi.XInputGetState(slot, ctypes.byref(st))
            g = st.Gamepad
            if g.wButtons or g.bLeftTrigger > 30 or g.bRightTrigger > 30 or \
                    max(abs(g.sThumbLX), abs(g.sThumbLY), abs(g.sThumbRX), abs(g.sThumbRY)) > 8000:
                print(f"first input after {time.time() - t_w:.1f} s", flush=True)
                break
            time.sleep(0.01)
        else:
            print(f"NO INPUT within {wait_for_input:.0f} s: the controller is enumerated but sends no state "
                  "(receiver without a paired pad, or pad idle/asleep)", flush=True)
            return
    rng = {k: [0, 0] for k in ("LX", "LY", "RX", "RY", "LT", "RT")}
    pg_rng, pg_buttons = {}, set()
    seen, packets, t0, last_print = set(), set(), time.time(), 0.0
    print(f"sampling slot {slot} for {seconds:.0f} s - move both sticks, press buttons, pull triggers", flush=True)
    while time.time() - t0 < seconds:
        if pg_js is not None:
            import pygame
            pygame.event.pump()
            for a in range(pg_js.get_numaxes()):
                v = round(pg_js.get_axis(a), 3)
                lo, hi = pg_rng.get(a, (v, v))
                pg_rng[a] = (min(lo, v), max(hi, v))
            pg_buttons |= {b for b in range(pg_js.get_numbuttons()) if pg_js.get_button(b)}
        if time.time() - last_print > 0.5:
            last_print = time.time()
            g = st.Gamepad
            print(f"t={time.time() - t0:4.1f}s raw={bytes(st).hex()} xinput LX={g.sThumbLX} LY={g.sThumbLY} "
                  f"RX={g.sThumbRX} RY={g.sThumbRY} LT={g.bLeftTrigger} RT={g.bRightTrigger} btn=0x{g.wButtons:04x} | "
                  f"pygame axes={[round(pg_js.get_axis(a), 2) for a in range(pg_js.get_numaxes())] if pg_js else None}",
                  flush=True)
        if xi.XInputGetState(slot, ctypes.byref(st)) != 0:
            print("controller disconnected during sampling", flush=True)
            break
        g = st.Gamepad
        packets.add(st.dwPacketNumber)
        for k, v in (("LX", g.sThumbLX), ("LY", g.sThumbLY), ("RX", g.sThumbRX), ("RY", g.sThumbRY),
                     ("LT", g.bLeftTrigger), ("RT", g.bRightTrigger)):
            rng[k] = [min(rng[k][0], v), max(rng[k][1], v)]
        seen |= {name for bit, name in BUTTONS.items() if g.wButtons & bit}
        time.sleep(0.01)
    full = {k: (v[0] < -30000 and v[1] > 30000) if k in ("LX", "LY", "RX", "RY") else v[1] > 250 for k, v in rng.items()}
    print(json.dumps({"slot": slot, "input_packets": len(packets), "axis_ranges": rng,
                      "axis_full_travel": full, "buttons_seen": sorted(seen),
                      "pygame_axis_ranges": {str(k): v for k, v in pg_rng.items()},
                      "pygame_buttons_seen": sorted(pg_buttons)}, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--wait-for-input", type=float, default=0.0,
                    help="block up to N seconds for the first input before sampling (removes prompt-timing races)")
    a = ap.parse_args()
    main(a.seconds, a.wait_for_input)
