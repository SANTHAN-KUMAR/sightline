"""Start and stop the demos from the dashboard, so a judge never needs a terminal.

Mounted by `sightline.api.app`:

    GET  /api/demo/catalog     what can be run, and what each one demonstrates
    GET  /api/demo/status      what is running now, for how long, and its last output
    POST /api/demo/start       {"id": "<catalog id>"}  - start one
    POST /api/demo/stop        stop everything this module started

**Only ids in `CATALOG` can ever be launched.** The request names a catalog entry and nothing else: no
command, no arguments, no paths. A dashboard that could run an arbitrary string would be a remote shell on
whatever laptop is showing the demo, which is not a trade worth making for a convenience button.

One demo at a time, deliberately. Two flights would fight over one simulator and one 8 GB GPU, and the
resulting mess would look like a broken product rather than a misuse. `start` on a busy runner returns 409
rather than queueing, because a judge pressing a button twice should be told, not silently obeyed twice.

STOP IS THE IMPORTANT PART. It terminates the process group, not just the parent: `demo.py` spawns
`sightline.mission.live`, which spawns nothing but holds the simulator's API control. Killing only the
parent leaves an aircraft flying with nobody driving it, which is exactly the state this endpoint exists to
get out of.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
UV = os.environ.get("SIGHTLINE_UV", r"D:\Tools\uv\uv.exe")
TAIL_LINES = 220

#: The port THIS server is answering on. `sightline.api.serve` sets it at startup.
#:
#: It matters more than it looks. A flight streams its records to a C2 chosen by its own flag, which
#: defaulted to 8781 regardless of where this server was listening. Start the dashboard on any other port -
#: because 8781 was already taken, which is exactly when someone reaches for another one - and the button
#: launches a flight that streams into a DIFFERENT server: the page the operator is watching stays empty
#: while a stale one fills up. The C2 that launches a flight must be the C2 that receives it.
SERVE_PORT = 8781


def set_serve_port(port: int) -> None:
    global SERVE_PORT
    SERVE_PORT = int(port)


def _demo(scenario: str) -> list[str]:
    """A demo is a SCENARIO NAME and nothing else, so the button and the terminal run the same flight.

    Determinism does not come from a seed here: `mission.live` injects telemetry noise only when
    `--noise-seed` is given, and the demo never gives it, so the geolocation chain is already exact rather
    than sampled. What actually varies between two runs is the WORLD and the STORE, which `preflight()`
    checks and `reset_world()` puts back.

    `--ignore-safety` is required because the F2 battery model refuses this plan at leg 30/31 - the
    violation is stamped into the data card either way.
    """
    return [UV, "run", "python", "-u", "tools/live/demo.py",
            "--scenario", scenario, "--ignore-safety", "--minutes", "12"]


def _c2_args(entry: dict[str, Any]) -> list[str]:
    """Point a launcher at THIS server. `demo.py` takes a port; `demo_controller.py` takes a URL."""
    if entry.get("c2_flag") == "url":
        return ["--c2", f"http://127.0.0.1:{SERVE_PORT}"]
    return ["--port", str(SERVE_PORT)]


def _controller(free: bool) -> list[str]:
    cmd = [UV, "run", "python", "-u", "tools/live/demo_controller.py", "--live", "--ignore-safety"]
    return cmd + ["--free"] if free else list(cmd)


#: The ONLY things that can be launched. `needs_sim` drives the dashboard's warning, not a block: a judge
#: may legitimately start the page before the editor is up.
CATALOG: list[dict[str, Any]] = [
    {"id": "nominal", "group": "Autonomous survey", "title": "Nominal — 45 m",
     "sub": "The acceptance slice. 45 m, 7 m/s, inside the 40–60 m band every headline number comes from.",
     "cmd": _demo("nominal"), "needs_sim": True},
    {"id": "low", "group": "Autonomous survey", "title": "Low — 30 m",
     "sub": "Below the band. ~1.5× the pixels on target, a narrower swath: easiest detection, slowest coverage.",
     "cmd": _demo("low"), "needs_sim": True},
    {"id": "high", "group": "Autonomous survey", "title": "High — 80 m",
     "sub": "Above the band. Pixels on target roughly halve, so small and prone targets approach the 20 px floor.",
     "cmd": _demo("high"), "needs_sim": True},
    {"id": "slow", "group": "Autonomous survey", "title": "Slow — 4 m/s",
     "sub": "The tracker's best case: ~7 shutter releases per survivor instead of ~4, so the confirmation gate closes far more often.",
     "cmd": _demo("slow"), "needs_sim": True},
    {"id": "fast", "group": "Autonomous survey", "title": "Fast — 12 m/s",
     "sub": "The failure case, kept on purpose. ~2 releases per survivor, below the 3 the gate needs: detections appear, tracks do not confirm.",
     "cmd": _demo("fast"), "needs_sim": True},
    {"id": "pilot_handback", "group": "Operator takeover", "title": "Takeover & hand-back",
     "sub": "The mission flies its survey; take the pad and fly it yourself. After 12 s of no input the mission takes itself back and continues from where the aircraft is.",
     "cmd": _controller(False), "needs_sim": True, "needs_pad": True, "c2_flag": "url"},
    {"id": "pilot_free", "group": "Operator takeover", "title": "Free flight",
     "sub": "Fly anywhere you like. The same detection, geolocation, tracking and triage pipeline runs on every frame.",
     "cmd": _controller(True), "needs_sim": True, "needs_pad": True, "c2_flag": "url"},
]
BY_ID = {c["id"]: c for c in CATALOG}


def _vehicle_check() -> tuple[bool, str]:
    """Is the simulator actually flyable? Delegates to the demo's own check, never a second copy of it.

    `tools/live/demo.py` is not a package, so it is loaded by path. The alternative - reimplementing
    "is there a vehicle" here - is how the button and the terminal end up disagreeing about whether the
    simulator is ready, which is worse than the import gymnastics.
    """
    import importlib.util                                        # noqa: PLC0415

    try:
        spec = importlib.util.spec_from_file_location("_demo_mod", REPO / "tools/live/demo.py")
        mod = importlib.util.module_from_spec(spec)               # type: ignore[arg-type]
        spec.loader.exec_module(mod)                              # type: ignore[union-attr]
        return mod.sim_has_vehicle()
    except Exception as exc:                                      # noqa: BLE001
        return False, f"could not run the simulator check: {type(exc).__name__}: {exc}"


def preflight(demo_id: str) -> tuple[bool, list[dict[str, Any]]]:
    """Everything that must be true BEFORE a button starts a flight.

    The point is to fail in the dashboard, in words, in two seconds - instead of launching into a failure
    the judge watches unfold for a minute in a log pane. A check that cannot be run counts as a failure.
    """
    entry = BY_ID.get(demo_id) or {}
    checks: list[dict[str, Any]] = []

    weights = REPO / "models/detect/f8b_sim/weights/best.pt"
    engine = REPO / "models/detect/f8b_sim/weights/best.engine"
    if weights.exists():
        which = "best.pt" + (" (+ TensorRT engine available)" if engine.exists() else "")
        checks.append({"name": "Detector", "ok": True, "detail": which})
    else:
        checks.append({"name": "Detector", "ok": False,
                       "detail": "models/detect/f8b_sim/weights/best.pt is missing - the fine-tune has "
                                 "not been pulled down to this machine."})

    if entry.get("needs_sim"):
        ok, why = _vehicle_check()
        checks.append({"name": "Simulator", "ok": ok, "detail": why})

    # The pad is REPORTED, never blocking: the takeover demo is the other session's lane, and a judge is
    # entitled to start it and plug a pad in afterwards.
    if entry.get("needs_pad"):
        detail = "not checked here - this demo is hand-flown; plug the pad in before you take control"
        checks.append({"name": "Gamepad", "ok": True, "detail": detail, "advisory": True})

    return all(c["ok"] for c in checks), checks


class Runner:
    """Owns at most one demo process. Thread-safe; the API may be hit from several requests at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._id: str | None = None
        self._started: float = 0.0
        self._log: deque[str] = deque(maxlen=TAIL_LINES)
        self._pump: threading.Thread | None = None

    # -- state ---------------------------------------------------------------------------------------
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        with self._lock:
            alive = self.running()
            return {
                "running": alive,
                "id": self._id if alive else None,
                "title": (BY_ID.get(self._id or "") or {}).get("title") if alive else None,
                "seconds": round(time.time() - self._started, 1) if alive else 0.0,
                "exit_code": None if alive or self._proc is None else self._proc.poll(),
                "last_id": self._id,
                "log": list(self._log)[-60:],
            }

    # -- control -------------------------------------------------------------------------------------
    def start(self, demo_id: str) -> tuple[int, dict[str, Any]]:
        entry = BY_ID.get(demo_id)
        if entry is None:
            return 404, {"error": f"unknown demo {demo_id!r}"}
        ok, checks = preflight(demo_id)
        if not ok:
            bad = "; ".join(f"{c['name']}: {c['detail']}" for c in checks if not c["ok"])
            return 412, {"error": f"not ready - {bad}", "checks": checks}
        with self._lock:
            if self.running():
                return 409, {"error": f"{self._id!r} is already running", "id": self._id}
            self._log.clear()
            cmd = list(entry["cmd"]) + _c2_args(entry)
            self._log.append(f"$ {' '.join(cmd[3:])}")
            # A new process GROUP, so stop() can take the whole tree down. On Windows that is a job-like
            # group via CREATE_NEW_PROCESS_GROUP; on POSIX, setsid.
            kw: dict[str, Any] = {}
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kw["start_new_session"] = True
            self._proc = subprocess.Popen(
                cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, **kw)
            self._id, self._started = demo_id, time.time()
            self._pump = threading.Thread(target=self._drain, args=(self._proc,), daemon=True)
            self._pump.start()
            return 200, {"started": demo_id, "title": entry["title"]}

    def stop(self) -> dict[str, Any]:
        """Stop everything this module started, and anything still flying the aircraft.

        The child spawns `sightline.mission.live`, which holds AirSim API control. Killing only the parent
        leaves an aircraft in the air with nobody driving it, so the whole tree goes.
        """
        with self._lock:
            was, p = self._id, self._proc
            killed = 0
            if p is not None and p.poll() is None:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                       capture_output=True, timeout=30)
                    else:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    killed += 1
                except Exception as exc:                          # noqa: BLE001
                    self._log.append(f"[stop] {type(exc).__name__}: {exc}")
            self._proc = None
            # Belt and braces: a flight from a previous session, or one started from a terminal, is still
            # an aircraft nobody is driving. The point of this button is that afterwards nothing is flying.
            killed += _kill_stragglers(self._log)
            self._log.append(f"[stop] terminated {killed} process tree(s)")
            return {"stopped": was, "killed": killed}

    def _drain(self, p: subprocess.Popen) -> None:
        try:
            assert p.stdout is not None
            for line in p.stdout:
                line = line.rstrip()
                if line and "WARNING" not in line:
                    self._log.append(line)
        except Exception:                                        # noqa: BLE001
            pass


def _kill_stragglers(log: deque[str]) -> int:
    """Kill any mission process this runner did not start. Returns how many were found."""
    pats = ("mission.live", "demo_controller.py", "tools/live/demo.py", "tools\\live\\demo.py")
    n = 0
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
                 "'mission\\.live|demo_controller|live.demo\\.py' } | Select-Object -ExpandProperty ProcessId"],
                capture_output=True, text=True, timeout=45).stdout
            for pid in [x.strip() for x in out.splitlines() if x.strip().isdigit()]:
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True, timeout=20)
                n += 1
        else:
            for pat in pats:
                r = subprocess.run(["pkill", "-f", pat], capture_output=True, timeout=20)
                n += 1 if r.returncode == 0 else 0
    except Exception as exc:                                     # noqa: BLE001
        log.append(f"[stop] straggler sweep failed: {type(exc).__name__}")
    return n


RUNNER = Runner()
