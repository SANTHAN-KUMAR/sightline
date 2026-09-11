"""Stop everything Sightline is running: every C2 server, every flight, every demo child.

    uv run python tools/live/stop_all.py            # stop them
    uv run python tools/live/stop_all.py --dry-run  # list what WOULD be stopped

The dashboard's "Stop everything" button stops flights, which is the urgent case: an aircraft with nobody
driving it. This is the other half - the one a person runs between demos, when several servers from several
sessions are holding ports and it is no longer obvious which dashboard is the live one.

That ambiguity is not cosmetic. A flight streams its records to a C2 chosen by a port number, so a stale
server squatting on 8781 silently collects the run while the operator watches an empty page. One server, one
port, one dashboard.

NOTHING HERE TOUCHES DATA. It terminates processes. Record stores are files on disk and are left exactly as
they are, which is also what guardrail R10 requires: no code path in this project deletes a record.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: Matched against the full command line. Deliberately narrow: every entry names a Sightline entry point,
#: so a developer's unrelated `python -m http.server` is never in scope.
PATTERNS = (
    "sightline.api.serve",
    "sightline.mission.live",
    "tools/live/demo.py",
    "tools\\live\\demo.py",
    "tools/live/demo_controller.py",
    "tools\\live\\demo_controller.py",
)

#: Only these binaries are ever killed, even on a command-line match. The match can appear in a shell
#: wrapper, an editor window title or this script's own arguments; the interpreter check is what keeps the
#: sweep from taking out the terminal it was typed into.
EXE_OK = ("python.exe", "python", "uv.exe", "uv")


def _procs() -> list[tuple[int, str, str]]:
    """(pid, exe name, command line) for candidate processes. Windows via CIM, POSIX via ps."""
    out: list[tuple[int, str, str]] = []
    if sys.platform == "win32":
        ps = ("Get-CimInstance Win32_Process | "
              "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
        raw = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90).stdout
        import json

        try:
            rows = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return out
        for r in rows if isinstance(rows, list) else [rows]:
            cmd = r.get("CommandLine") or ""
            out.append((int(r.get("ProcessId") or 0), r.get("Name") or "", cmd))
    else:
        raw = subprocess.run(["ps", "-eo", "pid=,comm=,args="],
                             capture_output=True, text=True, timeout=60).stdout
        for line in raw.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and parts[0].isdigit():
                out.append((int(parts[0]), parts[1], parts[2]))
    return out


def targets() -> list[tuple[int, str]]:
    me = subprocess.os.getpid()
    hits: list[tuple[int, str]] = []
    for pid, name, cmd in _procs():
        if pid in (0, me) or not cmd:
            continue
        if "stop_all" in cmd:                         # never this script, or a shell running it
            continue
        if not any(name.lower().endswith(e) for e in EXE_OK):
            continue
        if any(p in cmd for p in PATTERNS):
            hits.append((pid, cmd[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would be stopped, stop nothing")
    a = ap.parse_args()

    hits = targets()
    if not hits:
        print("nothing running - no Sightline server, flight or demo found")
        return 0

    print(f"{len(hits)} process(es):")
    for pid, cmd in hits:
        print(f"  {pid:>7}  {cmd}")
    if a.dry_run:
        print("\n--dry-run: nothing was stopped")
        return 0

    stopped = 0
    for pid, _cmd in hits:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=30)
            else:
                import signal

                subprocess.os.kill(pid, signal.SIGTERM)
            stopped += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"  {pid}: {type(exc).__name__}: {exc}")
    print(f"\nstopped {stopped} process tree(s). Record stores on disk are untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
