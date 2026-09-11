"""The live feed of §5.9: "a FastAPI backend pushing GeoJSON over a WebSocket at each record update".

Wire format — every frame is one JSON object with this envelope::

    {"type": <str>, "seq": <int>, "t_utc": <float>, "schema_version": "1.0.0", ...payload}

``seq`` is monotonic per server process, so a client can tell it missed a frame and re-fetch
``GET /api/records.geojson``. Message types:

======================  =========================================================================
``hello``               sent once on connect: server time, counts, capability flags
``snapshot``            ``records``: a full RFC 7946 FeatureCollection (also sent on connect)
``record``              ``op`` (always ``"upsert"``) + ``event`` + ``feature``: ONE record changed
``mission``             ``drone`` / ``footprint`` / ``track`` / ``plan`` — the live flight layers
``coverage``            the search-quality overlay sidecar changed; the client re-fetches the PNG
``outbox``              ``depth`` / ``online`` / ... — the §7 step 6 status bar
``pong``                reply to a client ``ping``
======================  =========================================================================

There is **no delete op** (guardrail R10). A record that the commander dismisses arrives as a ``record``
upsert whose ``status`` is ``"dismissed"``; the map moves it to its own layer and keeps it.

Thumbnails travel as URIs, never as bytes (§5.8: "thumbnails are file references in the live feed, keeps
messages under 5 KB").
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from sightline.schemas import SCHEMA_VERSION

__all__ = ["LiveHub", "envelope", "MESSAGE_TYPES", "MAX_MESSAGE_BYTES"]

#: "control" is the F3 pilot HUD feed (`sightline/api/control_feed.py`): flight mode, stick positions, the
#: idle-hand-back countdown and the pilot event log. It is a separate type from "mission" because it updates
#: at a different rate for a different reason - "mission" moves when the aircraft moves, "control" moves when
#: a HUMAN moves - and because a client that only wants the map should not have to parse stick positions.
MESSAGE_TYPES: tuple[str, ...] = ("hello", "snapshot", "record", "mission", "coverage", "outbox", "pong",
                                  "control")

#: §5.8 keeps live-feed messages small enough for a field radio. Asserted in tests/test_api.py.
MAX_MESSAGE_BYTES = 5120


def envelope(type_: str, seq: int, **payload: Any) -> dict[str, Any]:
    if type_ not in MESSAGE_TYPES:
        raise ValueError(f"unknown live message type {type_!r}; expected one of {MESSAGE_TYPES}")
    return {"type": type_, "seq": seq, "t_utc": time.time(), "schema_version": SCHEMA_VERSION, **payload}


class LiveHub:
    """Fan-out to every connected WebSocket. Publishing is safe from non-async threads.

    The store's change callback runs on whichever thread wrote the record (the pipeline, or the uploader),
    so :meth:`publish_threadsafe` hops onto the server's event loop with ``run_coroutine_threadsafe``.
    """

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seq = 0
        self._lock = asyncio.Lock()
        self.sent = 0
        self.dropped = 0
        self.max_seen_bytes = 0

    # ---- wiring --------------------------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def clients(self) -> int:
        return len(self._clients)

    @property
    def seq(self) -> int:
        return self._seq

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def register(self, ws: Any) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: Any) -> None:
        async with self._lock:
            self._clients.discard(ws)

    # ---- publishing ----------------------------------------------------------------------------------
    async def publish(self, type_: str, **payload: Any) -> dict[str, Any]:
        msg = envelope(type_, self.next_seq(), **payload)
        text = json.dumps(msg, separators=(",", ":"), default=str)
        self.max_seen_bytes = max(self.max_seen_bytes, len(text.encode("utf-8")))
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send_text(text)
                self.sent += 1
            except Exception:
                self.dropped += 1
                await self.unregister(ws)
        return msg

    async def send_to(self, ws: Any, type_: str, **payload: Any) -> dict[str, Any]:
        msg = envelope(type_, self.next_seq(), **payload)
        text = json.dumps(msg, separators=(",", ":"), default=str)
        self.max_seen_bytes = max(self.max_seen_bytes, len(text.encode("utf-8")))
        await ws.send_text(text)
        self.sent += 1
        return msg

    def publish_threadsafe(self, type_: str, **payload: Any) -> None:
        """Called from the record store's write thread. Never raises, never blocks the pipeline."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.publish(type_, **payload), loop)
        except RuntimeError:
            pass
