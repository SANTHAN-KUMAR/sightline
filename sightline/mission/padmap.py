"""The controller mapping: which axis is a stick, which button is which, and who says so.

`sightline/mission/takeover.py` shipped with a hard-coded `XBOX_BUTTONS` whose four indices were a guess and
were labelled as one (`buttons_verified_against_hardware: False`). That is fine for a state-machine test and
useless the morning of a demo, because the only route out of MANUAL is a button and nobody had ever pressed
one. This module makes the mapping **data**: measured once by `tools/live/pad_calibrate.py`, written to
`data/controller/<slug>.json`, loaded by name at runtime, and carrying its own provenance so an unverified
default can never be mistaken for a measured one.

Three things it refuses to do, each because the alternative has already bitten this project once:

* **It will not map a trigger as a stick.** Axes 4 and 5 on this machine's pad rest at **-1.0**. The
  "standard" SDL2 layout quoted in most tutorials puts `pitch` on axis 4; believing it reported full
  deflection with nobody touching the pad and took the aircraft into MANUAL the instant the source opened.
  :meth:`PadMap.validate` fails on any mapped stick axis whose resting value is outside `REST_TOLERANCE`.
* **It will not silently invent a mapping for an unknown pad.** :func:`load_pad_map` returns a map whose
  `provenance` says `default-guess` and whose `verified` is False. Callers print it; the HUD shows it amber.
* **It will not let two controls share an index.** A calibration where RESUME and RTL are both button 0
  means the judge presses "go back to the mission" and the aircraft flies home instead.

R10: nothing here deletes a record or marks anything cleared. It maps hardware to intent.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["PadMap", "REST_TOLERANCE", "CONTROL_AXES", "CONTROL_BUTTONS", "OPTIONAL_BUTTONS",
           "XBOX360_SDL_WINDOWS", "DEFAULT_MAPS", "pad_slug", "map_dir", "load_pad_map", "save_pad_map",
           "describe_unknown_pad"]

#: An axis resting outside this is not a self-centring stick - it is a trigger, or a broken mapping.
#: Same number as `takeover.REST_TOLERANCE`; kept here because this module is the one that enforces it.
REST_TOLERANCE = 0.30

#: The four flight axes. Every map must bind all four.
CONTROL_AXES: tuple[str, ...] = ("roll", "pitch", "yaw", "throttle")

#: The four mode buttons the F3 state machine reads. Every map must bind all four.
CONTROL_BUTTONS: tuple[str, ...] = ("takeover", "resume", "hold", "rtl")

#: Demo buttons. A map may bind none of them; `-1` means "this pad has no such button".
#: `mark`    - the judge flags what they are looking at; it becomes a real record on the map (R10-safe: it
#:             only ever ADDS).
#: `boost`   - hold for a speed multiplier, so a judge can cross the valley without waiting.
#: `orbit`   - hold to circle the last thing the judge marked, hands-free, for the camera.
#: `freefly` - toggle between "resume the survey" and "go anywhere", i.e. demo mode 1 <-> mode 2.
#: `camera`  - cycle the feed the HUD shows (rgb -> segmentation -> depth).
OPTIONAL_BUTTONS: tuple[str, ...] = ("mark", "boost", "orbit", "freefly", "camera")


@dataclass(slots=True)
class PadMap:
    """One controller's mapping, with provenance attached to it rather than to a comment.

    `axes` and `buttons` are index maps. `invert` names the axes whose sign must be flipped so that
    **positive always means the intuitive direction**: stick forward = positive pitch (nose down/forward),
    stick right = positive roll, stick right = positive yaw, stick up = positive throttle. SDL reports
    screen-style Y (down is +1), so `pitch` and `throttle` are inverted on essentially every pad.
    """

    name: str = "unknown"
    slug: str = "unknown"
    axes: dict[str, int] = field(default_factory=dict)
    buttons: dict[str, int] = field(default_factory=dict)
    invert: tuple[str, ...] = ()
    #: Per-axis dead zone applied before anything else sees the value. Bigger than the state machine's
    #: takeover deadband on purpose: this one removes stick NOISE, that one decides INTENT.
    dead_zone: float = 0.08
    #: Resting value of every axis, as measured. Empty when the map was never seen against hardware.
    rest_axes: tuple[float, ...] = ()
    n_axes: int = 0
    n_buttons: int = 0
    n_hats: int = 0
    #: "measured" (a human pressed every button), "default-guess" (shipped mapping, nobody pressed anything),
    #: or "partial" (axes measured, buttons not).
    provenance: str = "default-guess"
    measured_utc: float = 0.0
    note: str = ""

    # -- queries ---------------------------------------------------------------------------------------
    @property
    def verified(self) -> bool:
        """True only when a human physically pressed every mode button on this pad."""
        return self.provenance == "measured"

    def axis(self, name: str) -> int:
        return int(self.axes.get(name, -1))

    def button(self, name: str) -> int:
        return int(self.buttons.get(name, -1))

    def has(self, name: str) -> bool:
        return self.button(name) >= 0

    def sign(self, name: str) -> float:
        return -1.0 if name in self.invert else 1.0

    def apply_dead_zone(self, v: float) -> float:
        """Dead zone, rescaled so the usable travel still reaches +-1 instead of jumping at the edge."""
        dz = self.dead_zone
        if dz <= 0.0:
            return max(-1.0, min(1.0, v))
        a = abs(v)
        if a <= dz:
            return 0.0
        return max(-1.0, min(1.0, (a - dz) / (1.0 - dz))) * (1.0 if v >= 0 else -1.0)

    # -- validation ------------------------------------------------------------------------------------
    def validate(self, *, rest_axes: tuple[float, ...] | None = None) -> list[str]:
        """Every reason this map must not be used. Empty list means it is safe to fly.

        Called at source construction, not at first deflection: a mapping that reads a trigger as a stick
        must fail while someone is still looking at a terminal, not thirty seconds into a judge's flight.
        """
        problems: list[str] = []
        for name in CONTROL_AXES:
            if self.axis(name) < 0:
                problems.append(f"axis {name!r} is not bound")
        for name in CONTROL_BUTTONS:
            if self.button(name) < 0:
                problems.append(f"button {name!r} is not bound - there would be no way to use it")

        bound_axes = [self.axis(n) for n in CONTROL_AXES if self.axis(n) >= 0]
        if len(set(bound_axes)) != len(bound_axes):
            problems.append(f"two flight axes share an index: {self.axes}")
        used = {n: self.button(n) for n in CONTROL_BUTTONS + OPTIONAL_BUTTONS if self.button(n) >= 0}
        seen: dict[int, str] = {}
        for name, idx in used.items():
            if idx in seen:
                problems.append(f"buttons {seen[idx]!r} and {name!r} are both index {idx} - "
                                f"one press would fire both")
            seen[idx] = name

        rest = tuple(rest_axes) if rest_axes is not None else self.rest_axes
        if rest:
            if self.n_axes and len(rest) != self.n_axes:
                problems.append(f"map is for a {self.n_axes}-axis pad; this one reports {len(rest)}")
            for name in CONTROL_AXES:
                i = self.axis(name)
                if 0 <= i < len(rest) and abs(rest[i]) > REST_TOLERANCE:
                    problems.append(
                        f"{name} is axis {i}, which rests at {rest[i]:+.2f}. An axis resting near +-1 is a "
                        f"TRIGGER, not a stick, and would read as a permanent takeover. "
                        f"All axes at rest: {list(rest)}")
            for name in CONTROL_BUTTONS + OPTIONAL_BUTTONS:
                i = self.button(name)
                if i >= 0 and self.n_buttons and i >= self.n_buttons:
                    problems.append(f"button {name!r} is index {i} but the pad has {self.n_buttons}")
        return problems

    # -- serialisation ---------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "slug": self.slug, "axes": dict(self.axes), "buttons": dict(self.buttons),
                "invert": list(self.invert), "dead_zone": self.dead_zone, "rest_axes": list(self.rest_axes),
                "n_axes": self.n_axes, "n_buttons": self.n_buttons, "n_hats": self.n_hats,
                "provenance": self.provenance, "measured_utc": self.measured_utc, "note": self.note,
                "verified": self.verified}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PadMap":
        return cls(
            name=str(d.get("name", "unknown")), slug=str(d.get("slug", "unknown")),
            axes={str(k): int(v) for k, v in (d.get("axes") or {}).items()},
            buttons={str(k): int(v) for k, v in (d.get("buttons") or {}).items()},
            invert=tuple(str(x) for x in (d.get("invert") or ())),
            dead_zone=float(d.get("dead_zone", 0.08)),
            rest_axes=tuple(float(x) for x in (d.get("rest_axes") or ())),
            n_axes=int(d.get("n_axes", 0)), n_buttons=int(d.get("n_buttons", 0)),
            n_hats=int(d.get("n_hats", 0)),
            provenance=str(d.get("provenance", "default-guess")),
            measured_utc=float(d.get("measured_utc", 0.0)), note=str(d.get("note", "")))

    def describe(self) -> dict[str, Any]:
        d = self.to_dict()
        d["problems"] = self.validate()
        return d

    def summary(self) -> str:
        flag = "MEASURED" if self.verified else self.provenance.upper()
        bound = ", ".join(f"{n}={self.button(n)}" for n in CONTROL_BUTTONS)
        extra = ", ".join(f"{n}={self.button(n)}" for n in OPTIONAL_BUTTONS if self.has(n))
        return (f"{self.name} [{flag}] axes {self.axes} invert {list(self.invert)} | {bound}"
                + (f" | {extra}" if extra else ""))


# --- the shipped defaults -----------------------------------------------------------------------------------
#: Xbox 360 pad, SDL2 on Windows, as MEASURED on this machine (2026-09-10/11): 6 axes, 11 buttons, 1 hat;
#: axes 0-3 rest at 0.0 and are the two sticks, axes 4-5 rest at -1.0 and are the triggers. The axis half of
#: this map is therefore measured. The BUTTON half is the SDL2 Xbox controller order
#: (0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB, 6 Back, 7 Start, 8 LS, 9 RS, 10 Guide) - consistent with the 11 buttons
#: the driver reports, but nobody has pressed one, which is exactly why `provenance` says so.
#:
#: Layout is RC "mode 2", which is what every drone pilot and every sim expects:
#:   LEFT stick  = throttle (up/down) + yaw (left/right)
#:   RIGHT stick = pitch (forward/back) + roll (left/right)
XBOX360_SDL_WINDOWS = PadMap(
    name="Xbox 360 Controller",
    slug="xbox-360-controller",
    axes={"yaw": 0, "throttle": 1, "roll": 2, "pitch": 3},
    buttons={"resume": 0, "takeover": 1, "hold": 2, "rtl": 3,
             "boost": 4, "mark": 5, "camera": 6, "freefly": 7, "orbit": 9},
    invert=("pitch", "throttle"),
    dead_zone=0.08,
    rest_axes=(0.0, 0.0, 0.0, 0.0, -1.0, -1.0),
    n_axes=6, n_buttons=11, n_hats=1,
    provenance="partial",
    note="axes MEASURED on this machine (0-3 sticks rest 0.0, 4-5 triggers rest -1.0); buttons are the "
         "SDL2 Xbox order and have NOT been pressed. Run tools/live/pad_calibrate.py to make this measured.",
)

DEFAULT_MAPS: dict[str, PadMap] = {XBOX360_SDL_WINDOWS.slug: XBOX360_SDL_WINDOWS}


def pad_slug(name: str) -> str:
    """A filesystem-safe key for a device name. `'Xbox 360 Controller'` -> `'xbox-360-controller'`."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return s or "unknown"


def map_dir(repo: Path | str | None = None) -> Path:
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    return root / "data" / "controller"


def load_pad_map(device_name: str, *, repo: Path | str | None = None,
                 n_axes: int = 0, n_buttons: int = 0, n_hats: int = 0) -> PadMap:
    """The best map available for this device: a calibrated file, then a shipped default, then nothing.

    A calibrated file whose axis/button COUNTS do not match the pad now attached is refused rather than
    used - it is a map for a different device that happens to report the same name, and using it would put
    RESUME on whatever button happens to sit at that index.
    """
    slug = pad_slug(device_name)
    p = map_dir(repo) / f"{slug}.json"
    if p.is_file():
        try:
            m = PadMap.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:                                    # a corrupt file must not fly the aircraft
            m = None
            bad = f"{p.name} could not be read ({type(e).__name__}); falling back to the shipped map"
        else:
            bad = ""
            if n_axes and m.n_axes and m.n_axes != n_axes:
                bad = f"{p.name} is for a {m.n_axes}-axis pad, this one has {n_axes}"
            elif n_buttons and m.n_buttons and m.n_buttons != n_buttons:
                bad = f"{p.name} is for a {m.n_buttons}-button pad, this one has {n_buttons}"
        if m is not None and not bad:
            return m
        if bad:
            fallback = DEFAULT_MAPS.get(slug)
            if fallback is not None:
                return replace(fallback, note=f"{bad}. {fallback.note}")

    shipped = DEFAULT_MAPS.get(slug)
    if shipped is not None:
        return replace(shipped, n_axes=n_axes or shipped.n_axes, n_buttons=n_buttons or shipped.n_buttons,
                       n_hats=n_hats or shipped.n_hats)
    return describe_unknown_pad(device_name, n_axes=n_axes, n_buttons=n_buttons, n_hats=n_hats)


def save_pad_map(m: PadMap, *, repo: Path | str | None = None) -> Path:
    d = map_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{m.slug}.json"
    p.write_text(json.dumps(m.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p


def describe_unknown_pad(device_name: str, *, n_axes: int = 0, n_buttons: int = 0,
                         n_hats: int = 0) -> PadMap:
    """A guess for a pad nobody has ever mapped: the first four axes, the first four buttons, said plainly.

    It is deliberately a map that :meth:`PadMap.validate` will reject the moment a trigger turns up in the
    first four axes, which is the common case on pads that are not this one.
    """
    return PadMap(
        name=str(device_name) or "unknown", slug=pad_slug(device_name),
        axes={"yaw": 0, "throttle": 1, "roll": 2, "pitch": 3},
        buttons={"resume": 0, "takeover": 1, "hold": 2, "rtl": 3},
        invert=("pitch", "throttle"), rest_axes=(), n_axes=n_axes, n_buttons=n_buttons, n_hats=n_hats,
        provenance="default-guess", measured_utc=0.0,
        note=f"NO MAP EXISTS for {device_name!r}. This is the first-four-axes / first-four-buttons guess. "
             f"Run `uv run python tools/live/pad_calibrate.py` before flying it in front of anyone.")


def mark_measured(m: PadMap, *, note: str = "") -> PadMap:
    return replace(m, provenance="measured", measured_utc=time.time(),
                   note=note or "every axis moved and every button pressed by a human during calibration")
