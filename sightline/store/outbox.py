"""F18 offline outbox (SOLUTION_DOC §5.10).

    "every record and thumbnail is written to the local SQLite log first, then an upload job keyed by
     (clip_id, record_id, version) is enqueued; an uploader thread retries with back-off and acknowledges on
     success; the server upserts idempotently. Delivery is at-least-once and replayable, the local pipeline
     never blocks on the network."

`persist-queue`'s :class:`SQLiteAckQueue` is the durable queue: a job that is fetched but not acknowledged goes
back to the ready set on ``nack`` (and on process restart, via ``auto_resume``), so a crash mid-upload replays
rather than loses. Nothing here deletes a record — a job that is abandoned is moved to the queue's *failed*
state, which keeps the row (guardrail R10 applies to the log; the queue simply never touches it).

The job key
-----------
``f"{kind}:{clip_id}:{record_id}:{version}"``. ``kind`` (``record`` / ``thumb``) namespaces the doc's
``(clip_id, record_id, version)`` because a record and its evidence thumbnail are two artifacts at the same
version. The receiving side upserts on that key, so a double delivery is a no-op.
"""

from __future__ import annotations

import base64
import hashlib
import os
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from persistqueue import SQLiteAckQueue
from persistqueue.exceptions import Empty

from sightline.schemas import Record

__all__ = ["Outbox", "Uploader", "HttpTransport", "Transport", "job_key", "clip_id_of"]


def clip_id_of(rec: Record) -> str:
    cid = str(rec.source.get("clip_id", "") or "")
    if not cid and rec.evidence:
        cid = rec.evidence[0].clip_id
    return cid


def job_key(kind: str, clip_id: str, record_id: str, version: int) -> str:
    return f"{kind}:{clip_id}:{record_id}:{version}"


class Transport(Protocol):
    """Ships one job. MUST raise on any failure (that is what triggers the retry)."""

    def __call__(self, job: dict[str, Any]) -> dict[str, Any]: ...


class Outbox:
    """Durable at-least-once queue of upload jobs. Enqueueing never blocks on the network."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.q = SQLiteAckQueue(str(self.path), multithreading=True, auto_commit=True, auto_resume=True)
        self._lock = threading.Lock()
        self._enqueued = 0

    # ---- producers -----------------------------------------------------------------------------------
    def enqueue_record(self, rec: Record) -> dict[str, Any]:
        cid = clip_id_of(rec)
        job = {
            "key": job_key("record", cid, rec.record_id, rec.version),
            "kind": "record",
            "clip_id": cid,
            "record_id": rec.record_id,
            "version": rec.version,
            "enqueued_utc": time.time(),
            "payload": {"feature": rec.to_feature()},
        }
        return self._put(job)

    def enqueue_thumb(self, record_id: str, version: int, clip_id: str, path: str | os.PathLike[str],
                      *, uri: str = "") -> dict[str, Any]:
        data = Path(path).read_bytes()
        job = {
            "key": job_key("thumb", clip_id, record_id, version),
            "kind": "thumb",
            "clip_id": clip_id,
            "record_id": record_id,
            "version": version,
            "enqueued_utc": time.time(),
            "payload": {
                "name": Path(path).name,
                "uri": uri or f"/api/thumbs/{Path(path).name}",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "b64": base64.b64encode(data).decode("ascii"),
            },
        }
        return self._put(job)

    def _put(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.q.put(job)
            self._enqueued += 1
        return job

    # ---- state ---------------------------------------------------------------------------------------
    def depth(self) -> int:
        """Jobs not yet delivered: ready + in flight. This is the number the status bar shows (§7 step 6)."""
        return int(self.q.active_size())

    def stats(self) -> dict[str, Any]:
        return {
            "depth": self.depth(),
            "ready": int(self.q.ready_count()),
            "unacked": int(self.q.unack_count()),
            "acked": int(self.q.acked_count()),
            "failed": int(self.q.ack_failed_count()),
            "enqueued": self._enqueued,
            "path": str(self.path),
        }

    def close(self) -> None:
        try:
            self.q.close()
        except Exception:
            pass


class Uploader(threading.Thread):
    """Retries with exponential back-off and acknowledges on success.

    The pipeline never waits on this thread: it only ever reads from the queue. ``online`` is derived —
    it flips false after the first failure and true again after the first success.
    """

    daemon = True

    def __init__(
        self,
        outbox: Outbox,
        transport: Transport,
        *,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        poll_s: float = 0.05,
        max_attempts: int | None = None,
        jitter: float = 0.1,
        on_change: Callable[[dict[str, Any]], None] | None = None,
        name: str = "sightline-uploader",
    ):
        super().__init__(name=name, daemon=True)
        self.outbox = outbox
        self.transport = transport
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self.poll_s = poll_s
        self.max_attempts = max_attempts
        self.jitter = jitter
        self.on_change = on_change
        # NOT `_stop`: `threading.Thread._stop` is a bound method CPython calls itself during
        # thread teardown (`_wait_for_tstate_lock`). Shadowing it with an Event made
        # `Uploader.stop()` raise `TypeError: 'Event' object is not callable` whenever the
        # thread had already exited - i.e. on every clean shutdown that joined a finished
        # uploader. Found by the live lane on 2026-09-11.
        self._stop_event = threading.Event()
        self._attempts: dict[str, int] = {}
        self._next_try = 0.0
        self.sent = 0
        self.failures = 0
        self.abandoned = 0
        self.online: bool | None = None  # None = nothing tried yet
        self.last_error = ""
        self.last_success_utc = 0.0

    # ---- one step, so tests can drive the loop deterministically -------------------------------------
    def step(self) -> str:
        """Attempt exactly one job. Returns 'idle' | 'sent' | 'retry' | 'abandoned' | 'backoff'."""
        if time.monotonic() < self._next_try:
            return "backoff"
        try:
            raw = self.outbox.q.get(raw=True, block=False)
        except Empty:
            return "idle"
        job = raw["data"]
        key = job.get("key", "")
        try:
            self.transport(job)
        except Exception as exc:  # any failure -> the job goes back on the queue
            self.failures += 1
            self.online = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            n = self._attempts[key] = self._attempts.get(key, 0) + 1
            if self.max_attempts is not None and n >= self.max_attempts:
                self.outbox.q.ack_failed(id=raw["pqid"])  # kept in the queue's failed table, not deleted
                self.abandoned += 1
                self._emit()
                return "abandoned"
            back = min(self.max_backoff_s, self.base_backoff_s * (2 ** (n - 1)))
            back *= 1.0 + random.uniform(0.0, self.jitter)
            self._next_try = time.monotonic() + back
            self.outbox.q.nack(id=raw["pqid"])
            self._emit()
            return "retry"
        self.outbox.q.ack(id=raw["pqid"])
        self.sent += 1
        self.online = True
        self.last_error = ""
        self.last_success_utc = time.time()
        self._attempts.pop(key, None)
        self._next_try = 0.0
        self._emit()
        return "sent"

    def _emit(self) -> None:
        if self.on_change:
            try:
                self.on_change(self.stats())
            except Exception:
                pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            state = self.step()
            if state in ("idle", "backoff"):
                self._stop_event.wait(self.poll_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    def drain(self, timeout_s: float = 10.0) -> bool:
        """Block until the queue is empty (test/shutdown helper). True if it drained."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.outbox.depth() == 0:
                return True
            if not self.is_alive():
                self.step()
            else:
                time.sleep(0.02)
        return self.outbox.depth() == 0

    def stats(self) -> dict[str, Any]:
        s = self.outbox.stats()
        s.update(
            sent=self.sent,
            failures=self.failures,
            abandoned=self.abandoned,
            online=self.online,
            last_error=self.last_error,
            last_success_utc=self.last_success_utc,
        )
        return s


class HttpTransport:
    """POSTs the job to the cloud sink's idempotent upsert endpoint. Any non-2xx raises."""

    def __init__(self, base_url: str, *, path: str = "/api/upload", timeout_s: float = 5.0,
                 client: Any | None = None):
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.timeout_s = timeout_s
        self._client = client
        self._own = client is None

    def _c(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def __call__(self, job: dict[str, Any]) -> dict[str, Any]:
        r = self._c().post(self.base_url + self.path, json=job, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        if self._own and self._client is not None:
            self._client.close()
            self._client = None
