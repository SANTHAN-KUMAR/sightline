"""What the pilot is doing, pushed to the map so a room full of people can see the handover happen.

The map already showed `mode MANUAL` as a chip and coloured the MANUAL stretch of the track amber. That is
enough for a post-flight reading and not enough for a live demo: when a judge picks up the pad, the audience
watching the screen should see the takeover the instant it happens, see the sticks move, see the aircraft
being held off the geofence, and see the countdown that will hand the mission back when they stop flying.

This module is the transport for that. It is deliberately the same shape as
`sightline/api/mission_feed.py::MissionState` - a small thread-safe object the flight loop pushes into and
the WebSocket publishes out of - and deliberately NOT part of `Telemetry`. `Telemetry` is a frozen schema
that every recorded frame carries and every replay reads; stick positions are ephemeral presentation state
that is superseded 50 times a second and must never end up in a dataset.

Two rules it keeps:

* **Events are append-only.** `ControlState.event()` adds to a bounded deque and nothing removes an entry
  except the deque's own age limit. Guardrail R10 is about records rather than UI events, but a feed that
  quietly drops the awkward moments is the same failure in a smaller place.
* **The pad's provenance travels with it.** `pad.verified` says whether a human has actually pressed these
  buttons (`sightline/mission/padmap.py`), and the HUD colours the controller amber when it has not, so
  nobody demonstrates a guessed mapping believing it was measured.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

__all__ = ["ControlState", "EVENT_KINDS", "MAX_EVENTS"]

#: Bounded so a long demo cannot grow without limit. 200 is several minutes of a busy flight.
MAX_EVENTS = 200

#: The event vocabulary the HUD styles. Anything else renders as a plain line rather than being dropped.
EVENT_KINDS: tuple[str, ...] = (
    "mode",        # a flight-mode transition, from the TakeoverMachine
    "mark",        # the pilot flagged something they could see
    "envelope",    # the geofence / ceiling / floor held the aircraft back
    "record",      # a new survivor record reached the map
    "pad",         # a controller was attached, lost, or found to be unverified
    "note",        # anything else worth a line
)


class ControlState:
    """Live controller + pilot state. One writer (the flight loop), many readers (the WebSocket).

        state.update(mode="MANUAL", sticks={...}, idle_remaining_s=8.3, ...)
        state.event("mode", "AUTO -> MANUAL", detail="stick deflection")
        payload = state.snapshot()
    """

    def __init__(self, max_events: int = MAX_EVENTS):
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._seq = 0
        self.updated_utc = 0.0
        self.mode = "AUTO"
        self.sticks: dict[str, float] = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "throttle": 0.5}
        self.buttons: dict[str, bool] = {}
        self.pad: dict[str, Any] = {"attached": False, "device": "", "source": "none", "verified": False,
                                    "provenance": "none"}
        self.idle_remaining_s: float | None = None
        self.idle_resume_s: float = 0.0
        self.envelope: dict[str, Any] = {"any": False, "reasons": []}
        self.flight: dict[str, Any] = {"agl_m": None, "speed_ms": None, "boost": False,
                                       "heading_deg": None, "manual_mode": "velocity"}
        self.stats: dict[str, Any] = {"seconds_by_mode": {}, "marks": 0, "transitions": 0,
                                      "idle_handbacks": 0, "frames": 0, "records": 0}
        self.free_flight = False
        self.camera = "rgb"
        #: The pilot's own pins as a GeoJSON FeatureCollection. Kept apart from `records` on purpose: a
        #: record carries detection provenance and feeds the metrics; a mark is a human pointing at ground.
        self.marks: dict[str, Any] = {"type": "FeatureCollection", "features": []}

    # ---- producers -----------------------------------------------------------------------------------
    def update(self, **fields: Any) -> None:
        """Merge a partial update. Unknown keys are ignored rather than raising into a flight loop."""
        with self._lock:
            for k, v in fields.items():
                if k == "sticks" and isinstance(v, dict):
                    self.sticks = {**self.sticks, **{kk: float(vv) for kk, vv in v.items()}}
                elif k == "buttons" and isinstance(v, dict):
                    self.buttons = {kk: bool(vv) for kk, vv in v.items()}
                elif k == "marks" and isinstance(v, dict):
                    self.marks = v
                elif k in ("pad", "envelope", "flight", "stats") and isinstance(v, dict):
                    setattr(self, k, {**getattr(self, k), **v})
                elif hasattr(self, k) and not k.startswith("_"):
                    setattr(self, k, v)
            self.updated_utc = time.time()

    def event(self, kind: str, text: str, *, detail: str = "", t_utc: float | None = None,
              **extra: Any) -> dict[str, Any]:
        """Append one line to the feed. Returns the event, so a caller can log the same object it published."""
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t_utc": float(t_utc if t_utc is not None else time.time()),
                  "kind": str(kind), "text": str(text), "detail": str(detail), **extra}
            self._events.append(ev)
            self.updated_utc = ev["t_utc"]
            return dict(ev)

    def note_mark(self) -> int:
        with self._lock:
            self.stats = {**self.stats, "marks": int(self.stats.get("marks", 0)) + 1}
            return int(self.stats["marks"])

    # ---- consumer ------------------------------------------------------------------------------------
    def snapshot(self, *, since_seq: int = 0, max_events: int = 12) -> dict[str, Any]:
        """Everything the HUD draws. `since_seq` returns only events newer than one already rendered.

        `max_events` exists because `sightline/api/live.py::MAX_MESSAGE_BYTES` caps a live-feed frame at
        5 KB - §5.8's "small enough for a field radio", which a 200-entry backlog would blow through. The
        WebSocket therefore carries the tail; a client that wants the whole log asks
        `GET /api/mission/control?since_seq=N` for it, where no such budget applies.
        """
        with self._lock:
            events = [e for e in self._events if e["seq"] > since_seq]
            if max_events > 0:
                events = events[-max_events:]
            return {
                "updated_utc": self.updated_utc,
                "mode": self.mode,
                "sticks": dict(self.sticks),
                "buttons": dict(self.buttons),
                "pad": dict(self.pad),
                "idle_remaining_s": self.idle_remaining_s,
                "idle_resume_s": self.idle_resume_s,
                "envelope": dict(self.envelope),
                "flight": dict(self.flight),
                "stats": dict(self.stats),
                "free_flight": bool(self.free_flight),
                "camera": self.camera,
                "marks": dict(self.marks),
                "events": events,
                "event_seq": self._seq,
            }

    @property
    def events(self) -> list[dict[str, Any]]:
        """The whole log. There is deliberately no way to empty it from outside this object: the feed is
        append-only for the same reason the record store is (R10), and a fresh process starts a fresh one."""
        with self._lock:
            return [dict(e) for e in self._events]
