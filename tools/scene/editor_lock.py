"""One writer at a time for the Unreal editor.

    uv run python tools/scene/editor_lock.py acquire <holder> [--wait-min 45]
    uv run python tools/scene/editor_lock.py release <holder>
    uv run python tools/scene/editor_lock.py status
    uv run python tools/scene/editor_lock.py break   # only for a lock whose holder is gone

There is exactly ONE editor process and every `ue_python` call runs on its game thread. Two lanes placing
actors at once interleave in ways that are silent and awful: `build_damage.py` resets every House_### to its
pristine mesh as its first step, so a rubble lane reading house positions mid-reset gets the wrong geometry,
and an actor destroyed by one lane while another holds a reference to it is the `Obj.cpp:383` rename crash
that killed the editor twice on 2026-09-10.

So: acquire before the FIRST `ue_python` call of a piece of editor work, release after the LAST one. While
you are waiting, do host-side work - generators, layout validation, checks - none of which needs the lock.

A lock older than --stale-min (default 60) is reported as stale so a crashed holder cannot block the project
for ever. Breaking someone else's live lock is a choice you have to make explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "_artifacts" / "editor.lock"


def read() -> dict | None:
    if not LOCK.exists():
        return None
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception:
        return {"holder": "(unreadable)", "t": 0.0}


def describe(d: dict, stale_min: float) -> str:
    age = (time.time() - float(d.get("t", 0))) / 60.0
    return (f"held by {d.get('holder')!r} for {age:.1f} min (pid {d.get('pid')})"
            + ("  [STALE]" if age > stale_min else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("acquire", "release", "status", "break"))
    ap.add_argument("holder", nargs="?", default="")
    ap.add_argument("--wait-min", type=float, default=45.0)
    ap.add_argument("--stale-min", type=float, default=60.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)

    if a.action == "status":
        d = read()
        print("free" if d is None else describe(d, a.stale_min))
        return 0

    if a.action == "break":
        d = read()
        if d is None:
            print("free")
            return 0
        LOCK.unlink(missing_ok=True)
        print(f"broke lock {describe(d, a.stale_min)}")
        return 0

    if not a.holder:
        print("a holder name is required, e.g. 'vegetation-lane'")
        return 2

    if a.action == "release":
        d = read()
        if d is None:
            print("already free")
            return 0
        if d.get("holder") != a.holder:
            print(f"REFUSING: lock is {describe(d, a.stale_min)}, not yours ({a.holder!r})")
            return 1
        LOCK.unlink(missing_ok=True)
        print(f"released by {a.holder!r}")
        return 0

    deadline = time.time() + a.wait_min * 60
    announced = False
    while True:
        d = read()
        if d is None:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue                     # someone won the race; go round again
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"holder": a.holder, "pid": os.getpid(), "t": time.time(),
                           "note": a.note}, fh)
            print(f"acquired by {a.holder!r}")
            return 0
        age = (time.time() - float(d.get("t", 0))) / 60.0
        if age > a.stale_min:
            print(f"lock is STALE ({describe(d, a.stale_min)}). If that holder is really gone, run: "
                  f"editor_lock.py break")
            return 1
        if not announced:
            print(f"waiting: {describe(d, a.stale_min)} - do host-side work meanwhile")
            announced = True
        if time.time() > deadline:
            print(f"gave up after {a.wait_min:.0f} min; lock still {describe(d, a.stale_min)}")
            return 1
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
