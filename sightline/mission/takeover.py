"""F3: gamepad takeover and hand-back, with every mode switch logged (SOLUTION_DOC §5.2, §7 step 3).

The state machine of §5.2, verbatim::

              ┌──────────── stick deflection > deadband or TAKEOVER button ────────────┐
              ▼                                                                       │
       ┌─────────────┐   RESUME button (and sticks centred ≥ 1 s)   ┌───────────────┐ │
       │   MANUAL    │ ─────────────────────────────────────────────►│  AUTO-RESUME  │ │
       │ (gamepad)   │                                               │ re-plan from   │ │
       └─────────────┘                                               │ current pose;  │ │
              ▲                                                      │ continue the   │ │
              │ any time                                             │ pattern        │ │
       ┌─────────────┐   pattern complete / low battery / geofence   └──────┬────────┘ │
       │    AUTO     │ ◄───────────────────────────────────────────────────┘          │
       │ (mission)   │ ── HOLD button ──► HOVER (position hold) ── RESUME ──► AUTO     │
       └─────────────┘ ── RTL button  ──► RETURN-TO-LAUNCH (always available) ─────────┘

Three pieces, deliberately separate so each can be tested without the other two:

``TakeoverMachine``  pure logic. No simulator, no clock of its own, no I/O. Every transition it makes is
                     appended to :attr:`TakeoverMachine.transitions` with a timestamp, which is what §5.2
                     means by "every transition is written into the telemetry CSV" and what lets the
                     evaluation slice coverage by AUTO vs MANUAL.
``ControlSource``    where the pilot's intent comes from: the pad as Cosys-AirSim sees it (the path that
                     actually flies the vehicle), the pad as pygame sees it (named buttons), or the
                     keyboard (no pad attached).
``VehicleAuthority`` the two RPC calls that hand the vehicle over and take it back, isolated so a fake
                     client can prove the handover happens on the same poll that decides it.

AUTO-RESUME is a transient, not a mode: `Telemetry.mode` is the frozen four-value vocabulary
("AUTO", "MANUAL", "HOLD", "RTL"), and re-planning happens while the machine is already back in AUTO.
:attr:`TakeoverMachine.resume_pending` is True for exactly the moment the mission runner has to re-plan.

R10 note: nothing in this module deletes a record or marks anything cleared. A mode switch changes who is
flying, never what has been found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

__all__ = ["Mode", "MODES", "ControlInput", "ModeTransition", "TakeoverMachine", "ControlSource",
           "AirSimRcSource", "PygameGamepadSource", "KeyboardSource", "NullSource", "open_control_source",
           "VehicleAuthority", "XBOX_BUTTONS"]

Mode = Literal["AUTO", "MANUAL", "HOLD", "RTL"]

#: The frozen vocabulary from `sightline/schemas.py::FlightMode`. Kept in step by a test.
MODES: tuple[str, ...] = ("AUTO", "MANUAL", "HOLD", "RTL")

#: Xbox pad button indices as SDL2/XInput reports them through pygame. UNTESTED against a physical pad in
#: this project (see docs/TRACKER.md); override with `PygameGamepadSource(buttons=...)`.
XBOX_BUTTONS: dict[str, int] = {"resume": 0, "takeover": 1, "hold": 2, "rtl": 3}


# --- what the pilot is doing ------------------------------------------------------------------------------
@dataclass(slots=True)
class ControlInput:
    """One poll of the pilot's controller. Axes are -1..1; throttle is 0..1 and centres at 0.5."""

    t: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.5
    takeover: bool = False
    resume: bool = False
    hold: bool = False
    rtl: bool = False
    #: False when no controller is attached at all - the machine must never read intent out of that.
    valid: bool = False
    source: str = "none"

    def deflection(self) -> float:
        """How far the pilot has moved anything, on one 0..1 scale."""
        return max(abs(self.roll), abs(self.pitch), abs(self.yaw), abs(self.throttle - 0.5) * 2.0)

    def any_button(self) -> bool:
        return bool(self.takeover or self.resume or self.hold or self.rtl)


@dataclass(slots=True)
class ModeTransition:
    """One logged mode switch. This is the row the telemetry CSV carries and the map colours the track by."""

    t_utc: float
    from_mode: str
    to_mode: str
    reason: str
    frame_idx: int = -1
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"t_utc": self.t_utc, "from": self.from_mode, "to": self.to_mode, "reason": self.reason,
                "frame_idx": self.frame_idx, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.from_mode} -> {self.to_mode}  ({self.reason})"


# --- the state machine ------------------------------------------------------------------------------------
class TakeoverMachine:
    """Pure F3 logic. Feed it :class:`ControlInput`; it returns the transition it made, or None.

        m = TakeoverMachine()
        tr = m.poll(ControlInput(t=now, roll=0.9, valid=True))   # -> AUTO -> MANUAL
        ...
        m.poll(ControlInput(t=now, valid=True))                  # sticks centred; the 1 s clock starts
        tr = m.poll(ControlInput(t=now + 1.5, resume=True, valid=True))  # -> MANUAL -> AUTO, resume_pending

    Deliberate choices, each from the document rather than from taste:

    * **Takeover is immediate.** A single poll above the deadband switches the mode; there is no
      confirmation, no debounce and no hysteresis on the way IN. The pilot grabbing the sticks is the
      whole point of the feature.
    * **Hand-back is not immediate.** RESUME only lands when the sticks have been centred for
      ``centred_s`` (§5.2: "RESUME button (and sticks centred ≥ 1 s)"), so a pilot who presses RESUME with
      a thumb still on the stick does not get the mission flying into their input.
    * **RTL is available from every state, including RTL itself** (§7: "HOLD and RTL are always one button
      away"), and it is checked before anything else in the poll.
    * **RTL is not a trap.** The pilot can take the vehicle back out of RTL with the sticks, exactly as they
      can out of AUTO; otherwise a mis-press would be unrecoverable.
    """

    #: §5.2 "stick deadband tuning" was a day-1 unknown. 0.15 is what `tools/day1/gamepad_airsim.py`
    #: used to call a stick "active", and that probe measured full -1..1 travel on every axis.
    DEADBAND = 0.15
    CENTRED_S = 1.0

    def __init__(self, *, deadband: float = DEADBAND, centred_s: float = CENTRED_S,
                 initial: Mode = "AUTO", clock: Callable[[], float] = time.time):
        if not 0.0 < deadband < 1.0:
            raise ValueError(f"deadband must be in (0, 1), got {deadband}")
        if initial not in MODES:
            raise ValueError(f"unknown mode {initial!r}; expected one of {MODES}")
        self.deadband = float(deadband)
        self.centred_s = float(centred_s)
        self.mode: str = initial
        #: The mode the machine started in, and when the mission clock started. `seconds_in` needs both to
        #: attribute the time before the FIRST transition, which is most of a normal flight.
        self.initial_mode: str = initial
        self.t_start: float | None = None
        self.clock = clock
        self.transitions: list[ModeTransition] = []
        self.frame_idx = -1
        #: True for exactly the moment after a MANUAL -> AUTO hand-back: the mission must RE-PLAN from the
        #: current pose (§5.2 "AUTO-RESUME"), not restart the pattern. The runner clears it.
        self.resume_pending = False
        self._centred_since: float | None = None
        self._last_input: ControlInput | None = None
        #: RESUME presses that were refused because the sticks were not centred long enough. "I pressed
        #: RESUME and nothing happened" is the most confusing thing a takeover UI can do, so it is recorded.
        self._refusals: list[tuple[float, str]] = []
        self.polls = 0
        self.polls_valid = 0

    # -- reporting ---------------------------------------------------------------------------------
    @property
    def manual(self) -> bool:
        return self.mode == "MANUAL"

    @property
    def flying_itself(self) -> bool:
        """True when the mission script owns the vehicle (AUTO, HOLD and RTL are all API-flown)."""
        return self.mode != "MANUAL"

    def seconds_in(self, mode: str, now: float | None = None) -> float:
        """Total seconds spent in `mode`, so coverage can be attributed to AUTO vs MANUAL (§5.2).

        Counts from :attr:`t_start` — the first poll — not from the first transition. A flight that spends
        ten minutes in AUTO and then thirty seconds in MANUAL must not report ten minutes of nothing: the
        whole point of the number is to say what share of the coverage a human flew.
        """
        now = self.clock() if now is None else now
        t0 = self.t_start
        if t0 is None:
            return 0.0
        total = 0.0
        cur = self.initial_mode
        for tr in self.transitions:
            if cur == mode:
                total += max(0.0, tr.t_utc - t0)
            t0, cur = tr.t_utc, tr.to_mode
        if cur == mode:
            total += max(0.0, now - t0)
        return total

    def status(self) -> dict[str, Any]:
        li = self._last_input
        return {
            "mode": self.mode,
            "resume_pending": self.resume_pending,
            "deadband": self.deadband,
            "centred_s": self.centred_s,
            "transitions": len(self.transitions),
            "t_start": self.t_start,
            "initial_mode": self.initial_mode,
            "polls": self.polls,
            "polls_with_a_controller": self.polls_valid,
            "control_source": li.source if li else "none",
            "controller_present": bool(li and li.valid),
            "deflection": round(li.deflection(), 3) if li else 0.0,
            "last_transition": self.transitions[-1].as_dict() if self.transitions else None,
        }

    # -- transitions -------------------------------------------------------------------------------
    def _switch(self, to: str, reason: str, t: float, detail: str = "") -> ModeTransition | None:
        if to == self.mode:
            return None
        tr = ModeTransition(t_utc=t, from_mode=self.mode, to_mode=to, reason=reason,
                            frame_idx=self.frame_idx, detail=detail)
        self.mode = to
        self.transitions.append(tr)
        if to == "AUTO" and tr.from_mode in ("MANUAL", "HOLD", "RTL"):
            self.resume_pending = True
        return tr

    def force(self, to: Mode, reason: str, t: float | None = None, detail: str = "") -> ModeTransition | None:
        """A transition the MISSION commands rather than the pilot: pattern complete, low battery, geofence."""
        if to not in MODES:
            raise ValueError(f"unknown mode {to!r}; expected one of {MODES}")
        when = self.clock() if t is None else t
        if self.t_start is None:
            self.t_start = when
        return self._switch(to, reason, when, detail)

    def poll(self, ctl: ControlInput) -> ModeTransition | None:
        """One controller sample. Returns the transition it caused, or None."""
        self.polls += 1
        self._last_input = ctl
        if self.t_start is None:
            self.t_start = ctl.t
        if not ctl.valid:
            # No controller attached: hold whatever mode the mission is in and let `force` drive it. Reading
            # intent out of an absent device is how a demo takes itself into MANUAL and hovers for ever.
            self._centred_since = None
            return None
        self.polls_valid += 1

        defl = ctl.deflection()
        if defl <= self.deadband:
            if self._centred_since is None:
                self._centred_since = ctl.t
        else:
            self._centred_since = None

        # RTL first: always available, from every state (§7 step 3).
        if ctl.rtl:
            return self._switch("RTL", "RTL button", ctl.t, f"deflection={defl:.2f}")

        # Taking control is immediate and needs no button.
        if ctl.takeover:
            return self._switch("MANUAL", "TAKEOVER button", ctl.t, f"deflection={defl:.2f}")
        if defl > self.deadband and self.mode != "MANUAL":
            return self._switch("MANUAL", "stick deflection", ctl.t,
                                f"deflection={defl:.2f} > deadband={self.deadband:.2f}")

        if ctl.hold:
            return self._switch("HOLD", "HOLD button", ctl.t, f"deflection={defl:.2f}")

        if ctl.resume and self.mode != "AUTO":
            held = 0.0 if self._centred_since is None else ctl.t - self._centred_since
            if self._centred_since is None or held < self.centred_s:
                self._refusals.append((ctl.t, f"RESUME ignored: sticks centred for {held:.2f} s "
                                              f"< {self.centred_s:.1f} s"))
                return None
            return self._switch("AUTO", "RESUME button", ctl.t, f"sticks centred {held:.2f} s")
        return None

    @property
    def refusals(self) -> list[tuple[float, str]]:
        return list(self._refusals)


# --- where the pilot's intent comes from -------------------------------------------------------------------
class ControlSource:
    """Base: `read(now)` returns a :class:`ControlInput`. Never raises; an unreadable device is `valid=False`."""

    name = "none"
    #: True when this source has been exercised against a physical device in this project.
    verified_against_hardware = False

    def read(self, now: float) -> ControlInput:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "verified_against_hardware": self.verified_against_hardware}


class NullSource(ControlSource):
    """No controller. Every poll is `valid=False`, so the machine only ever moves on `force()`."""

    name = "none"

    def read(self, now: float) -> ControlInput:
        return ControlInput(t=now, valid=False, source=self.name)


class AirSimRcSource(ControlSource):
    """The pad as **Cosys-AirSim itself** sees it (`getMultirotorState().rc_data`).

    This is the authoritative source for *sticks*, because it is the same signal that actually flies the
    vehicle once `enableApiControl(False)` is called - `tools/day1/gamepad_airsim.py` measured full -1..1
    travel on throttle/pitch/roll/yaw and a 394 ms API->RC handover through exactly this path.

    It is NOT authoritative for *buttons*: `rc_data.switches` came back as the integer 0 in all three
    verification runs and nobody has ever pressed a button while it was being watched, so the bit meanings
    below are a guess and are labelled as one. Stick takeover does not depend on them.
    """

    name = "airsim_rc"
    verified_against_hardware = True  # axes only; see `switch_bits_verified`
    switch_bits_verified = False

    def __init__(self, client: Any, *, vehicle: str = "", switch_bits: dict[str, int] | None = None):
        self.client = client
        self.vehicle = vehicle
        self.switch_bits = switch_bits or {"takeover": 0, "resume": 1, "hold": 2, "rtl": 3}
        self.errors = 0

    def read(self, now: float) -> ControlInput:
        try:
            st = (self.client.getMultirotorState(vehicle_name=self.vehicle) if self.vehicle
                  else self.client.getMultirotorState())
            rc = st.rc_data
        except Exception:
            self.errors += 1
            return ControlInput(t=now, valid=False, source=self.name)
        if not bool(getattr(rc, "is_valid", False)):
            return ControlInput(t=now, valid=False, source=self.name)
        sw = int(getattr(rc, "switches", 0) or 0)
        return ControlInput(
            t=now, roll=float(rc.roll), pitch=float(rc.pitch), yaw=float(rc.yaw),
            throttle=float(rc.throttle), valid=True, source=self.name,
            takeover=bool(sw >> self.switch_bits["takeover"] & 1),
            resume=bool(sw >> self.switch_bits["resume"] & 1),
            hold=bool(sw >> self.switch_bits["hold"] & 1),
            rtl=bool(sw >> self.switch_bits["rtl"] & 1),
        )

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(axes_verified=True, switch_bits_verified=self.switch_bits_verified,
                 switch_bits=self.switch_bits, read_errors=self.errors,
                 note="axes measured over full travel on 2026-09-10 (13 PASS / 0 FAIL); rc_data.switches "
                      "has only ever been observed as 0, so the button bits are unverified")
        return d


#: MEASURED on the Xbox 360 pad attached to this machine (2026-09-11), SDL2 joystick API on Windows:
#: axes 0-3 rest at 0.0 and are the two sticks; axes 4 and 5 rest at **-1.0** and are the triggers.
#: The "standard" layout quoted in most tutorials (0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT) is NOT what this
#: driver reports, and believing it put `pitch` on a trigger - which rests at -1.0, reads as full stick
#: deflection, and took the aircraft into MANUAL the instant the source was opened, with nobody touching it.
XBOX_AXES: dict[str, int] = {"yaw": 0, "throttle": 1, "roll": 2, "pitch": 3}

#: An axis resting outside this is not a self-centring stick - it is a trigger, or a broken mapping.
REST_TOLERANCE = 0.30


class PygameGamepadSource(ControlSource):
    """The pad through SDL2/XInput, which is where NAMED BUTTONS come from.

    `pygame.joystick` reports buttons individually, so TAKEOVER / RESUME / HOLD / RTL are real buttons here
    rather than bits of an integer.

    **The resting state is checked at construction, and a mapped axis that is not centred is a hard error.**
    A control source that reports full deflection before anyone has touched it does not fail gracefully: it
    takes the aircraft off the mission and hovers, and every log line afterwards says MANUAL as though a
    pilot had asked for it. Refusing to open is the only safe behaviour, and the message names the axis and
    its resting value so the mapping can be fixed in one go.
    """

    name = "pygame"
    #: Axes have been read from a physical Xbox 360 pad on this machine. Buttons have NOT been pressed on
    #: one; `buttons_verified_against_hardware` says so separately rather than letting one flag imply both.
    verified_against_hardware = True
    buttons_verified_against_hardware = False

    def __init__(self, index: int = 0, *, buttons: dict[str, int] | None = None,
                 axes: dict[str, int] | None = None, check_rest: bool = True):
        import os                                          # noqa: PLC0415

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame                                      # noqa: PLC0415

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            raise RuntimeError(f"no gamepad at index {index} "
                               f"({pygame.joystick.get_count()} joysticks enumerated)")
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        self.buttons = dict(buttons or XBOX_BUTTONS)
        self.axes = dict(axes or XBOX_AXES)
        self.device = self.js.get_name()
        for _ in range(5):
            self.pygame.event.pump()
            time.sleep(0.01)
        self.rest_axes = [round(self._axis(i), 4) for i in range(self.js.get_numaxes())]
        if check_rest:
            self._assert_sticks_are_centred()

    def _assert_sticks_are_centred(self) -> None:
        bad = {name: self.rest_axes[i] for name, i in self.axes.items()
               if 0 <= i < len(self.rest_axes) and abs(self.rest_axes[i]) > REST_TOLERANCE}
        if bad:
            raise RuntimeError(
                f"{self.device}: axis mapping is wrong - "
                + ", ".join(f"{n} (axis {self.axes[n]}) rests at {v:+.2f}" for n, v in bad.items())
                + f". An axis resting near +-1 is a TRIGGER, not a stick, and would read as a permanent "
                  f"takeover. All axes at rest: {self.rest_axes}. "
                  f"Pass axes={{...}} with the right indices (this machine's pad: {XBOX_AXES}).")

    def _axis(self, i: int) -> float:
        try:
            return float(self.js.get_axis(i))
        except Exception:
            return 0.0

    def _button(self, key: str) -> bool:
        i = self.buttons.get(key, -1)
        if i < 0 or i >= self.js.get_numbuttons():
            return False
        try:
            return bool(self.js.get_button(i))
        except Exception:
            return False

    def read(self, now: float) -> ControlInput:
        try:
            self.pygame.event.pump()
            return ControlInput(
                t=now, roll=self._axis(self.axes["roll"]), pitch=-self._axis(self.axes["pitch"]),
                yaw=self._axis(self.axes["yaw"]),
                # SDL sticks are -1..1 and centre at 0; the schema's throttle is 0..1 centred at 0.5.
                throttle=0.5 - self._axis(self.axes["throttle"]) / 2.0,
                takeover=self._button("takeover"), resume=self._button("resume"),
                hold=self._button("hold"), rtl=self._button("rtl"),
                valid=True, source=self.name)
        except Exception:
            return ControlInput(t=now, valid=False, source=self.name)

    def close(self) -> None:
        try:
            self.js.quit()
            self.pygame.joystick.quit()
        except Exception:
            pass

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(device=self.device, buttons=self.buttons, axes=self.axes, rest_axes=self.rest_axes,
                 buttons_verified_against_hardware=self.buttons_verified_against_hardware,
                 note="axes READ from this physical pad and their resting values checked; the four button "
                      "indices have NOT been pressed on hardware")
        return d


class KeyboardSource(ControlSource):
    """No pad attached: drive the state machine from the console.

        T takeover   R resume   H hold   L return-to-launch
        W/S pitch    A/D roll   Q/E yaw   SPACE centre the sticks

    A key press latches the axis until SPACE, so a single keystroke is enough to cross the deadband and the
    "sticks centred ≥ 1 s" rule still means something. Windows only (`msvcrt`); on anything else it reports
    `valid=False` and says so, rather than pretending to be a controller.
    """

    name = "keyboard"
    verified_against_hardware = False
    KEYS = {"t": "takeover", "r": "resume", "h": "hold", "l": "rtl"}
    STEP = 0.6

    def __init__(self) -> None:
        try:
            import msvcrt                                   # noqa: PLC0415
        except ImportError:
            self.msvcrt = None
        else:
            self.msvcrt = msvcrt
        self.roll = self.pitch = self.yaw = 0.0
        self.throttle = 0.5
        self._pending: dict[str, bool] = {}
        self.keys_seen = 0

    @property
    def available(self) -> bool:
        return self.msvcrt is not None

    def _drain(self) -> None:
        if self.msvcrt is None:
            return
        while self.msvcrt.kbhit():
            try:
                ch = self.msvcrt.getch().decode("utf-8", "ignore").lower()
            except Exception:
                continue
            self.keys_seen += 1
            if ch in self.KEYS:
                self._pending[self.KEYS[ch]] = True
            elif ch == "w":
                self.pitch = self.STEP
            elif ch == "s":
                self.pitch = -self.STEP
            elif ch == "a":
                self.roll = -self.STEP
            elif ch == "d":
                self.roll = self.STEP
            elif ch == "q":
                self.yaw = -self.STEP
            elif ch == "e":
                self.yaw = self.STEP
            elif ch == " ":
                self.roll = self.pitch = self.yaw = 0.0
                self.throttle = 0.5

    def read(self, now: float) -> ControlInput:
        if self.msvcrt is None:
            return ControlInput(t=now, valid=False, source=self.name)
        self._drain()
        p, self._pending = self._pending, {}
        return ControlInput(t=now, roll=self.roll, pitch=self.pitch, yaw=self.yaw, throttle=self.throttle,
                            takeover=p.get("takeover", False), resume=p.get("resume", False),
                            hold=p.get("hold", False), rtl=p.get("rtl", False),
                            valid=True, source=self.name)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(keys_seen=self.keys_seen, available=self.available,
                 note="KEYBOARD FALLBACK - no physical gamepad was attached")
        return d


def open_control_source(kind: str = "auto", *, client: Any = None, index: int = 0) -> ControlSource:
    """Pick a control source. ``auto`` prefers a real pad, then the sim's own view of it, then the keyboard.

    pygame comes first because it is the only source with real buttons; the AirSim RC path is second because
    it is the one that actually flies the vehicle in MANUAL, and it still gives full stick control.
    """
    if kind not in ("auto", "pygame", "airsim", "keyboard", "none"):
        raise ValueError(f"unknown control source {kind!r}")
    if kind == "none":
        return NullSource()
    if kind in ("auto", "pygame"):
        try:
            return PygameGamepadSource(index)
        except Exception:
            if kind == "pygame":
                raise
    if kind in ("auto", "airsim") and client is not None:
        src = AirSimRcSource(client)
        probe = src.read(time.time())
        if probe.valid or kind == "airsim":
            return src
    if kind in ("auto", "keyboard"):
        kb = KeyboardSource()
        if kb.available or kind == "keyboard":
            return kb
    return NullSource()


# --- the two RPC calls that actually hand the vehicle over --------------------------------------------------
class VehicleAuthority:
    """Applies a mode to the vehicle. Isolated so a fake client can prove the handover is immediate.

    §5.2: "on deflection above the deadband it calls `enableApiControl(False)`, after which SimpleFlight
    obeys the RC channels directly; on RESUME it calls `enableApiControl(True)`". `AllowAPIAlways: true`
    stays set, which is why the vehicle ignores a held stick while the API owns it (measured: 0.01 m in 4 s).
    """

    def __init__(self, client: Any, *, vehicle: str = "", hover_fn: Callable[[], None] | None = None,
                 rtl_fn: Callable[[], None] | None = None):
        self.client = client
        self.vehicle = vehicle
        self.hover_fn = hover_fn
        self.rtl_fn = rtl_fn
        self.calls: list[tuple[float, str]] = []
        self.errors: list[str] = []

    def _call(self, what: str, fn: Callable[[], Any]) -> bool:
        try:
            fn()
        except Exception as e:
            self.errors.append(f"{what}: {type(e).__name__}: {e}")
            return False
        self.calls.append((time.time(), what))
        return True

    def _api(self, on: bool) -> bool:
        if self.vehicle:
            return self._call(f"enableApiControl({on})",
                              lambda: self.client.enableApiControl(on, self.vehicle))
        return self._call(f"enableApiControl({on})", lambda: self.client.enableApiControl(on))

    def apply(self, tr: ModeTransition) -> dict[str, Any]:
        """Make the vehicle obey `tr.to_mode`. Returns what it did, with the wall-clock cost."""
        t0 = time.perf_counter()
        did: list[str] = []
        if tr.to_mode == "MANUAL":
            # Release FIRST and ask questions later: the pilot is already moving the sticks.
            if self._api(False):
                did.append("api_control=False")
        else:
            if tr.from_mode == "MANUAL" and self._api(True):
                did.append("api_control=True")
            if tr.to_mode == "HOLD" and self.hover_fn is not None:
                if self._call("hover", self.hover_fn):
                    did.append("hover")
            elif tr.to_mode == "RTL" and self.rtl_fn is not None:
                if self._call("rtl", self.rtl_fn):
                    did.append("rtl")
        return {"transition": tr.as_dict(), "did": did, "ms": round((time.perf_counter() - t0) * 1e3, 2),
                "errors": list(self.errors[-3:])}
