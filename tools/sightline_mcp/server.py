"""Sightline MCP server: engine lifecycle, builds, editor Python, logs and Cosys-AirSim control.

Complements Epic's built-in Unreal MCP plugin (HTTP, runs *inside* the editor). This server runs
*outside* the editor, so it still works when the editor is closed, compiling, or crashed:

  * process control   - launch/close editor, launch standalone -game sim, status (RAM/VRAM/ports)
  * build             - UnrealBuildTool builds, project-file generation, BuildCookRun packaging (as jobs)
  * editor python     - run Python inside a running editor (remote execution) or headless (commandlet)
  * logs              - tail/grep project logs
  * simulation        - Cosys-AirSim RPC: state, flight, capture (RGB/seg/IR/depth), weather, time, objects

Run: uv run python tools/sightline_mcp/server.py   (stdio transport; registered in .mcp.json)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import functools
from concurrent.futures import ThreadPoolExecutor

import anyio
import psutil
from mcp.server.fastmcp import FastMCP, Image

# Native-extension modules MUST be imported here, at startup, before mcp.run() starts reading stdin.
# On Windows the stdio transport keeps a synchronous ReadFile pending on the stdin pipe; loading a DLL-backed
# module later (numpy's _multiarray_umath during C-runtime init) waits on that handle and deadlocks the call.
# Diagnosed with faulthandler 2026-09-10 (tools/day1/diag_mcp_hang.py). Never add lazy imports of
# numpy / cv2 / torch / PIL / cosysairsim inside tool functions.
import cosysairsim as airsim  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ue_remote import (  # noqa: E402
    MODE_EVAL_STATEMENT,
    MODE_EXEC_FILE,
    MODE_EXEC_STATEMENT,
    CommandConnectionClosed,
    UnrealRemote,
)

REPO = Path(__file__).resolve().parents[2]
UE_ROOT = Path(os.environ.get("SIGHTLINE_UE_ROOT", r"D:\UE_5.8"))
PROJECT = Path(os.environ.get("SIGHTLINE_UPROJECT", REPO / "sim" / "SightlineSim" / "SightlineSim.uproject"))
PROJECT_NAME = PROJECT.stem
EDITOR_EXE = UE_ROOT / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"
EDITOR_CMD_EXE = UE_ROOT / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
BUILD_BAT = UE_ROOT / "Engine" / "Build" / "BatchFiles" / "Build.bat"
RUNUAT_BAT = UE_ROOT / "Engine" / "Build" / "BatchFiles" / "RunUAT.bat"
UBT_DLL = UE_ROOT / "Engine" / "Binaries" / "DotNET" / "UnrealBuildTool" / "UnrealBuildTool.dll"
SETTINGS_DIR = REPO / "sim" / "settings"
ARTIFACTS = REPO / "_artifacts"
JOB_LOGS = REPO / "_logs" / "jobs"
AIRSIM_HOST = os.environ.get("SIGHTLINE_AIRSIM_HOST", "127.0.0.1")
AIRSIM_PORT = int(os.environ.get("SIGHTLINE_AIRSIM_PORT", "41451"))
UE_MCP_PORT = int(os.environ.get("SIGHTLINE_UE_MCP_PORT", "8000"))


def _merge_user_env() -> list[str]:
    """Copy persisted user-level env vars (HKCU\\Environment) that this process lacks into os.environ, so every
    child (editor, commandlets, UBT/UAT) sees them. Claude Code and other hosts often predate the setup that wrote
    them (docs/CONTEXT.md section 3); without this, `UE-LocalDataCachePath` is missing and the editor puts its DDC
    and Zen store on C:. Existing process values always win; PATH/TEMP/TMP are never touched."""
    import winreg  # stdlib builtin (no DLL load); runs at import time, before mcp.run()
    added = []
    # Some MCP hosts (the Python SDK's stdio_client) pass only a small env whitelist. Restore the machine-level
    # basics children need: nvidia-smi/NVML fails without ProgramFiles; UBT finds vswhere/Windows SDK via them.
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion") as k:
            for var, val_name in (("ProgramFiles", "ProgramFilesDir"), ("ProgramW6432", "ProgramW6432Dir"),
                                  ("ProgramFiles(x86)", "ProgramFilesDir (x86)"), ("CommonProgramFiles", "CommonFilesDir")):
                if var not in os.environ:
                    with contextlib.suppress(OSError):
                        os.environ[var] = winreg.QueryValueEx(k, val_name)[0]
                        added.append(var)
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            for var in ("ComSpec", "windir", "OS", "NUMBER_OF_PROCESSORS"):
                if var not in os.environ:
                    with contextlib.suppress(OSError):
                        os.environ[var] = winreg.ExpandEnvironmentStrings(winreg.QueryValueEx(k, var)[0])
                        added.append(var)
        if "ProgramData" not in os.environ:
            os.environ["ProgramData"] = os.path.join(os.environ.get("SYSTEMDRIVE", "C:") + "\\", "ProgramData")
            added.append("ProgramData")
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            i = 0
            while True:
                try:
                    name, value, typ = winreg.EnumValue(k, i)
                except OSError:
                    break
                i += 1
                if name.upper() in ("PATH", "TEMP", "TMP") or name in os.environ or not isinstance(value, str):
                    continue
                os.environ[name] = winreg.ExpandEnvironmentStrings(value) if typ == winreg.REG_EXPAND_SZ else value
                added.append(name)
    except OSError:
        pass
    return added


ENV_ADDED_FROM_REGISTRY = _merge_user_env()
# UnrealBuildAccelerator (UBT's local executor in 5.8) defaults its CAS store (capacity 40 GB) to
# %ProgramData%\Epic\UnrealBuildAccelerator on C:. UBAExecutor.cs honours UBA_ROOT, so builds started from here use D:.
os.environ.setdefault("UBA_ROOT", r"D:\UE_Cache\UBA")

mcp = FastMCP("sightline")


def threaded(fn):
    """Run a blocking tool in a worker thread so the MCP asyncio loop stays responsive (pings, cancellation,
    concurrent calls) while builds, flights or editor commands run. functools.wraps keeps the original signature,
    which FastMCP uses to build the tool's input schema.

    abandon_on_cancel=True: when the client cancels the request (notifications/cancelled) or disconnects, the
    request is released at once instead of waiting for the blocking call; the worker thread finishes in the
    background and its result is dropped. Without it a stdin EOF left the server alive until every in-flight
    flight/editor command returned (minutes with a wedged sim), and main() could not exit."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs), abandon_on_cancel=True)
    return wrapper

# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _ue_processes() -> list[dict]:
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "create_time"]):
        name = (p.info["name"] or "").lower()
        if name.startswith("unrealeditor") or name.startswith(PROJECT_NAME.lower()) or name == "shadercompileworker.exe":
            out.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "rss_gb": round((p.info["memory_info"].rss if p.info["memory_info"] else 0) / 2**30, 2),
                "age_s": int(time.time() - p.info["create_time"]),
                "cmdline": " ".join(p.info["cmdline"] or [])[:300],
            })
    return out


def _gpu() -> dict | None:
    try:
        # 15 s: an idle Optimus laptop GPU can take several seconds to wake for the first query.
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
        name, used, total, util, temp = [x.strip() for x in r.stdout.strip().split(",")]
        return {"name": name, "vram_used_mb": int(used), "vram_total_mb": int(total), "util_pct": int(util), "temp_c": int(temp)}
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _log_path() -> Path:
    return PROJECT.parent / "Saved" / "Logs" / f"{PROJECT_NAME}.log"


def _filter_build_output(text: str, max_lines: int = 120) -> str:
    keep = [ln for ln in text.splitlines()
            if re.search(r"\berror\b|\bwarning C|fatal|Result:|Total time|BUILD (SUCCESSFUL|FAILED)|Unable to", ln, re.I)]
    if not keep:
        keep = text.splitlines()[-40:]
    return "\n".join(keep[-max_lines:])


_remote_lock = threading.Lock()       # serialises editor commands: the protocol has ONE TCP command channel
_remote_init_lock = threading.Lock()  # guards creation of the shared UnrealRemote; held only briefly
_remote: UnrealRemote | None = None
# 0 = ephemeral loopback port per connect (two server instances never collide); set to pin a port if needed.
REMOTE_CMD_PORT = int(os.environ.get("SIGHTLINE_UE_REMOTE_CMD_PORT", "0"))


def _get_remote() -> UnrealRemote:
    """The server's single UnrealRemote. It is never replaced (a stale command channel is just disconnected), so
    status() can read its discovery results without taking _remote_lock, which a running editor command holds for
    up to timeout_s (previously status() blocked behind every long ue_python call)."""
    global _remote
    with _remote_init_lock:
        if _remote is None:
            _remote = UnrealRemote(command_port=REMOTE_CMD_PORT)
        return _remote


def _editor_run(code: str, mode: str = MODE_EXEC_FILE, timeout_s: float = 300.0) -> dict:
    # Fail fast with a clear message instead of waiting on multicast discovery (was 2 x 5 s) when no editor exists.
    if not any(p["name"].lower().startswith("unrealeditor") for p in _ue_processes()):
        raise RuntimeError("No Unreal Editor is running (no UnrealEditor process). Start it with editor_launch() "
                           "and wait for editor_wait_ready().")
    # Only one command can use the channel; never wait longer than this call's own timeout for the previous one.
    if not _remote_lock.acquire(timeout=max(timeout_s, 1)):
        raise TimeoutError(f"another editor command is still running after {timeout_s:g}s; nothing was sent")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            remote = _get_remote()
            try:
                return remote.run(code, mode=mode, timeout_s=timeout_s)
            except TimeoutError as e:
                # The command was delivered and is still running (or queued) on the editor's game thread. Re-sending
                # it would execute it a second time (duplicate spawns/edits), so report the timeout instead of retrying.
                raise TimeoutError(f"editor did not answer within {timeout_s:g}s; the command was delivered and may "
                                   "still be running in the editor (not re-sent). Raise timeout_s for long scripts.") from e
            except (ConnectionError, CommandConnectionClosed):
                # Stale command channel (editor restarted since the last call): the command never reached a live
                # editor, so reconnect once. Discovery failures ("no editor found") are not retried (was 2 x 5 s).
                remote.disconnect()
                return remote.run(code, mode=mode, timeout_s=timeout_s)
    finally:
        _remote_lock.release()


def _format_remote(res: dict) -> str:
    lines = [f"success: {res.get('success')}"]
    if res.get("result") not in (None, "", "None"):
        lines.append(f"result: {res.get('result')}")
    for o in res.get("output") or []:
        lines.append(f"[{o.get('type')}] {str(o.get('output', '')).rstrip()}")
    return "\n".join(lines)


# ---- background jobs (builds and packaging take minutes; never block a tool call on them) --------
_jobs: dict[str, dict] = {}


def _start_job(name: str, argv: list[str], cwd: Path | None = None) -> dict:
    JOB_LOGS.mkdir(parents=True, exist_ok=True)
    job_id = f"{name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    log = JOB_LOGS / f"{job_id}.log"
    fh = open(log, "w", encoding="utf-8", errors="replace")  # noqa: SIM115 - owned by the job
    fh.write(f"$ {subprocess.list2cmdline(argv)}\n\n")
    fh.flush()
    proc = subprocess.Popen(argv, cwd=str(cwd or REPO), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    _jobs[job_id] = {"proc": proc, "log": log, "fh": fh, "argv": argv, "started": time.time()}
    return {"job_id": job_id, "pid": proc.pid, "log": str(log)}


def _job_info(job_id: str, tail: int = 60) -> dict:
    j = _jobs.get(job_id)
    if not j:
        log = JOB_LOGS / f"{job_id}.log"
        if not log.exists():
            raise ValueError(f"unknown job {job_id}")
        text = log.read_text(encoding="utf-8", errors="replace")
        return {"job_id": job_id, "state": "unknown (server restarted)", "tail": "\n".join(text.splitlines()[-tail:])}
    rc = j["proc"].poll()
    if rc is not None and not j["fh"].closed:
        j["fh"].close()
    text = j["log"].read_text(encoding="utf-8", errors="replace")
    return {
        "job_id": job_id,
        "state": "running" if rc is None else ("succeeded" if rc == 0 else "failed"),
        "exit_code": rc,
        "elapsed_s": int(time.time() - j["started"]),
        "log": str(j["log"]),
        "summary": _filter_build_output(text) if rc is not None else None,
        "tail": "\n".join(text.splitlines()[-tail:]),
    }


# --------------------------------------------------------------------------------------------------
# status / process control
# --------------------------------------------------------------------------------------------------


@mcp.tool()
@threaded
def status() -> str:
    """Snapshot of the dev environment: UE processes (RSS), system RAM, GPU VRAM, and which endpoints are up
    (Epic MCP :8000, AirSim RPC :41451, editor Python remote execution). Call this first when unsure."""
    vm = psutil.virtual_memory()
    remote_nodes = []
    try:
        # discovery only; deliberately NOT under _remote_lock, so status() answers while ue_python runs
        remote_nodes = _get_remote().nodes(wait_s=1.5)
    except OSError as e:
        remote_nodes = [{"error": str(e)}]
    data = {
        "project": str(PROJECT),
        "project_exists": PROJECT.exists(),
        "ue_processes": _ue_processes(),
        "ram_used_gb": round(vm.used / 2**30, 1),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "gpu": _gpu(),
        "endpoints": {
            f"unreal_mcp_http:{UE_MCP_PORT}": _port_open(UE_MCP_PORT),
            f"airsim_rpc:{AIRSIM_PORT}": _port_open(AIRSIM_PORT, AIRSIM_HOST),
            "editor_python_remote_exec": [{k: n.get(k) for k in ("node_id", "project_name", "engine_version", "machine")}
                                          for n in remote_nodes],
        },
        "running_jobs": [k for k, j in _jobs.items() if j["proc"].poll() is None],
        "ddc_path_for_children": os.environ.get("UE-LocalDataCachePath"),
        "env_added_from_registry": ENV_ADDED_FROM_REGISTRY,
    }
    return json.dumps(data, indent=2)


def _settings_arg(settings_profile: str | None) -> list[str]:
    if not settings_profile:
        return []
    p = Path(settings_profile)
    if not p.is_absolute():
        p = SETTINGS_DIR / (settings_profile if settings_profile.endswith(".json") else f"{settings_profile}.json")
    if not p.exists():
        raise FileNotFoundError(f"AirSim settings profile not found: {p}")
    return [f"-settings={p}"]


@mcp.tool()
@threaded
def editor_launch(map_path: str | None = None, settings_profile: str | None = "default",
                  extra_args: list[str] | None = None, wait_ready_s: int = 0) -> str:
    """Launch the Unreal Editor on the Sightline project.

    map_path: optional level to open, e.g. "/Game/Sightline/Maps/FloodValley".
    settings_profile: AirSim settings JSON in sim/settings (name without .json) passed via -settings=.
    wait_ready_s: if > 0, block until editor Python remote execution answers (first launch compiles shaders:
      allow 600+ s). Otherwise returns immediately; poll with status() or editor_wait_ready()."""
    if any(p["name"].lower().startswith("unrealeditor") and "-run=" not in p["cmdline"] for p in _ue_processes()):
        return "An editor is already running; use status() or editor_close() first."
    argv = [str(EDITOR_EXE), str(PROJECT)]
    if map_path:
        argv.append(map_path)
    argv += _settings_arg(settings_profile) + (extra_args or [])
    # stdin/stdout must never be inherited: they are this server's MCP protocol pipes.
    proc = subprocess.Popen(argv, cwd=str(PROJECT.parent), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    msg = f"launched pid={proc.pid}: {subprocess.list2cmdline(argv)}"
    if wait_ready_s > 0:
        # tools are async wrappers (@threaded); call the sync implementation directly from inside a tool
        msg += "\n" + editor_wait_ready.__wrapped__(wait_ready_s)
    return msg


@mcp.tool()
@threaded
def editor_wait_ready(timeout_s: int = 600) -> str:
    """Block until the running editor answers Python remote execution (i.e. fully initialised)."""
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        if not any(p["name"].lower().startswith("unrealeditor") for p in _ue_processes()):
            return "Editor process is not running (crashed or closed). Check ue_log()."
        try:
            # never overrun the caller's deadline (was a fixed 20 s probe + 5 s sleep past timeout_s)
            probe_s = max(1.0, min(20.0, deadline - time.time()))
            res = _editor_run("import unreal; print(unreal.SystemLibrary.get_engine_version())", timeout_s=probe_s)
            return "editor ready\n" + _format_remote(res)
        except Exception as e:  # noqa: BLE001 - keep polling
            last_err = str(e)
            time.sleep(max(0.0, min(5.0, deadline - time.time())))
    return f"timed out after {timeout_s}s waiting for editor; last error: {last_err}"


@mcp.tool()
@threaded
def editor_close(force: bool = False, timeout_s: int = 60) -> str:
    """Close the editor. Graceful (quit_editor via Python) unless force=True, which kills UE processes.
    Unsaved changes are lost on force."""
    if not force:
        try:
            _editor_run("import unreal; unreal.SystemLibrary.quit_editor()", timeout_s=10)
        except Exception:  # noqa: BLE001 - the channel drops as the editor exits
            pass
        deadline = time.time() + timeout_s
        while time.time() < deadline and any(p["name"].lower().startswith("unrealeditor") and "-run=" not in p["cmdline"]
                                             for p in _ue_processes()):
            time.sleep(1)
    killed = []
    for p in _ue_processes():
        if force or p["name"].lower().startswith("unrealeditor"):
            try:
                psutil.Process(p["pid"]).kill()
                killed.append(p["pid"])
            except psutil.Error:
                pass
    return f"closed; force-killed pids: {killed}" if killed else "closed cleanly"


@mcp.tool()
@threaded
def sim_launch_game(map_path: str | None = None, settings_profile: str | None = "default", res_x: int = 1280,
                    res_y: int = 720, windowed: bool = True, render_offscreen: bool = False,
                    extra_args: list[str] | None = None) -> str:
    """Run the project as a standalone game (-game) - lighter than the editor and the right mode for data runs.
    render_offscreen=True runs without a window (captures still render)."""
    argv = [str(EDITOR_EXE), str(PROJECT)]
    if map_path:
        argv.append(map_path)
    argv += ["-game", f"-ResX={res_x}", f"-ResY={res_y}", "-windowed" if windowed else "-fullscreen", "-log"]
    if render_offscreen:
        argv.append("-RenderOffscreen")
    argv += _settings_arg(settings_profile) + (extra_args or [])
    proc = subprocess.Popen(argv, cwd=str(PROJECT.parent), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return f"launched pid={proc.pid}: {subprocess.list2cmdline(argv)}\nPoll sim_ping() until the RPC server is up."


# --------------------------------------------------------------------------------------------------
# build / package
# --------------------------------------------------------------------------------------------------


@mcp.tool()
@threaded
def ue_build(target: str | None = None, configuration: str = "Development", platform: str = "Win64",
             clean: bool = False, extra_args: list[str] | None = None) -> str:
    """Build a project target with UnrealBuildTool as a background job (returns job_id; poll job_status).
    target defaults to '<Project>Editor'. The editor must be closed (or use Live Coding) for editor targets.
    extra_args: passed to UBT verbatim, e.g. ["-MaxParallelActions=6"] to cap compiler RAM on this 16 GB machine."""
    target = target or f"{PROJECT_NAME}Editor"
    argv = [str(BUILD_BAT), target, platform, configuration, f"-Project={PROJECT}", "-WaitMutex", "-NoHotReloadFromIDE"]
    if clean:
        argv.append("-Clean")
    argv += extra_args or []
    return json.dumps(_start_job(f"build-{target}", argv), indent=2)


@mcp.tool()
@threaded
def ue_generate_project_files() -> str:
    """Regenerate the Visual Studio solution for the project (synchronous, ~30-90 s)."""
    # Installed (Launcher) engines ship no GenerateProjectFiles.bat. Use the same UBT mode the Explorer verb
    # "Generate Visual Studio project files" (UnrealVersionSelector) runs: Build.bat -projectfiles ... -rocket.
    argv = [str(BUILD_BAT), "-projectfiles", f"-project={PROJECT}", "-game", "-progress"]
    argv.append("-rocket" if (UE_ROOT / "Engine" / "Build" / "InstalledBuild.txt").exists() else "-engine")
    r = subprocess.run(argv, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL, errors="replace")
    return f"exit {r.returncode}\n" + _filter_build_output(r.stdout + r.stderr)


@mcp.tool()
@threaded
def ue_package(configuration: str = "Development", output_dir: str | None = None) -> str:
    """Cook + package a standalone Windows build (BuildCookRun) as a background job. Output defaults to
    D:/Sightline/_build/<config>. This is the build to use for data-capture runs (lower RAM than the editor)."""
    out = Path(output_dir) if output_dir else REPO / "_build" / configuration
    argv = [str(RUNUAT_BAT), "BuildCookRun", f"-project={PROJECT}", "-noP4", "-platform=Win64",
            f"-clientconfig={configuration}", "-build", "-cook", "-stage", "-pak", "-archive",
            f"-archivedirectory={out}", "-utf8output", "-unattended"]
    return json.dumps(_start_job("package", argv), indent=2)


@mcp.tool()
@threaded
def job_status(job_id: str, tail: int = 60) -> str:
    """State, exit code, filtered error summary and log tail of a background job."""
    if tail < 1:  # [-0:] would return the whole (possibly huge) job log
        raise ValueError(f"tail must be >= 1 (got {tail})")
    return json.dumps(_job_info(job_id, tail), indent=2)


@mcp.tool()
@threaded
def job_list() -> str:
    """All background jobs started by this server session, plus job logs on disk."""
    live = {k: ("running" if j["proc"].poll() is None else f"exit {j['proc'].returncode}") for k, j in _jobs.items()}
    on_disk = sorted(p.stem for p in JOB_LOGS.glob("*.log"))[-20:] if JOB_LOGS.exists() else []
    return json.dumps({"session_jobs": live, "recent_job_logs": on_disk}, indent=2)


@mcp.tool()
@threaded
def job_kill(job_id: str) -> str:
    """Terminate a running background job and its child processes."""
    j = _jobs.get(job_id)
    if not j or j["proc"].poll() is not None:
        return "not running"
    parent = psutil.Process(j["proc"].pid)
    for c in parent.children(recursive=True):
        c.kill()
    parent.kill()
    return "killed"


# --------------------------------------------------------------------------------------------------
# editor python / console / logs
# --------------------------------------------------------------------------------------------------


@mcp.tool()
@threaded
def ue_python(code: str, mode: str = "file", timeout_s: int = 300) -> str:
    """Execute Python inside the RUNNING editor (full `unreal` API: actors, assets, materials, levels, PIE...).
    mode: 'file' (multi-statement script, default), 'statement' (prints result), 'eval' (returns value).
    Returns success flag, result and captured log output. Use print() to return data."""
    modes = {"file": MODE_EXEC_FILE, "statement": MODE_EXEC_STATEMENT, "eval": MODE_EVAL_STATEMENT}
    if mode not in modes:  # validate before touching the editor (was a bare KeyError: "'bogus'")
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(modes)}")
    m = modes[mode]
    return _format_remote(_editor_run(code, mode=m, timeout_s=timeout_s))


@mcp.tool()
@threaded
def ue_python_headless(script: str, nullrhi: bool = True, timeout_s: int = 1800) -> str:
    """Run a Python script in a headless editor commandlet (no running editor needed): asset import, level
    generation, batch edits. `script` is a path to a .py file or inline code. Editor must NOT have the same
    assets open. Returns the exit code and Python/error lines from the log."""
    p = Path(script)
    if not (p.suffix == ".py" and p.exists()):
        tmp = ARTIFACTS / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        p = tmp / f"headless_{uuid.uuid4().hex[:8]}.py"
        p.write_text(script, encoding="utf-8")
    argv = [str(EDITOR_CMD_EXE), str(PROJECT), "-run=pythonscript", f"-script={p}", "-unattended", "-nopause",
            "-nosplash", "-stdout", "-FullStdOutLogOutput"]
    if nullrhi:
        argv.append("-nullrhi")
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, errors="replace",
                       stdin=subprocess.DEVNULL)
    lines = [ln for ln in r.stdout.splitlines() if re.search(r"LogPython|Error|Fatal|Warning: .*Python", ln)]
    return f"exit {r.returncode}\n" + "\n".join(lines[-200:])


@mcp.tool()
@threaded
def ue_console(command: str) -> str:
    """Run an Unreal console command in the running editor world (e.g. 'stat fps', 'r.ScreenPercentage 50')."""
    code = ("import unreal\n"
            "w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world() or "
            "unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()\n"
            f"unreal.SystemLibrary.execute_console_command(w, {command!r})\nprint('ok')")
    return _format_remote(_editor_run(code))


@mcp.tool()
@threaded
def ue_log(lines: int = 150, grep: str | None = None, log_file: str | None = None) -> str:
    """Tail the project log (Saved/Logs/<Project>.log) or another log path; optional regex filter."""
    if lines < 1:  # text[-0:] is the WHOLE file: lines=0 used to return a multi-hundred-MB log in one message
        raise ValueError(f"lines must be >= 1 (got {lines})")
    path = Path(log_file) if log_file else _log_path()
    if not path.exists():
        return f"log not found: {path}"
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().splitlines()
    if grep:
        rx = re.compile(grep, re.I)
        text = [ln for ln in text if rx.search(ln)]
    return f"{path}\n" + "\n".join(text[-lines:])


# --------------------------------------------------------------------------------------------------
# Cosys-AirSim
# --------------------------------------------------------------------------------------------------

_sim_lock = threading.Lock()
_sim_client = None
# The cosysairsim client (rpc-msgpack on tornado) owns an IOLoop that is thread-affine and cannot run on a thread
# that already runs an asyncio loop (it deadlocks). Every AirSim call therefore runs on this single dedicated
# thread, which creates the client and keeps it for the life of the server.
_SIM_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim-rpc")


def _airsim():
    return airsim  # imported at startup (see the import note at the top); the sim itself may be absent


def _client():
    global _sim_client
    airsim = _airsim()
    if _sim_client is None:
        if not _port_open(AIRSIM_PORT, AIRSIM_HOST):
            raise RuntimeError(f"AirSim RPC not listening on {AIRSIM_HOST}:{AIRSIM_PORT}. Start PIE/game first.")
        c = airsim.MultirotorClient(ip=AIRSIM_HOST, port=AIRSIM_PORT, timeout_value=60)
        # Handshake with a short RPC timeout. msgpackrpc copies the session timeout into each Future at call time,
        # so this bounds only the connect; later calls keep 60 s. A port that accepts but never answers (sim frozen,
        # paused in a debugger, still loading, foreign process) used to cost 60 s per call, serialised on the sim
        # thread (sim_ping 61 s, a queued sim_state 121 s). Regression test: test_wedged_airsim_fails_within_seconds.
        connect_s = int(os.environ.get("SIGHTLINE_AIRSIM_CONNECT_TIMEOUT_S", "5"))
        c.client._timeout = connect_s
        try:
            c.confirmConnection()
        except Exception as e:
            with contextlib.suppress(Exception):
                c.client.close()
            raise RuntimeError(f"AirSim RPC at {AIRSIM_HOST}:{AIRSIM_PORT} accepted the connection but did not answer "
                               f"within {connect_s}s ({e}). Sim frozen/paused/loading, or another process owns the "
                               "port.") from e
        c.client._timeout = 60
        _sim_client = c
        global _capture_warmed
        _capture_warmed = False  # a new connection may mean a new sim process: warm the capture pipeline again
    return _sim_client


def _sim(fn):
    """Run fn(client) under the lock, dropping the client on connection errors so the next call reconnects.

    stdout is redirected to stderr: cosysairsim print()s (e.g. "Connected!" in confirmConnection) and in a stdio
    MCP server stdout is the JSON-RPC channel, so any stray print corrupts the protocol stream."""
    def run():
        global _sim_client
        with _sim_lock, contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(_client())
            except Exception:
                _sim_client = None
                raise
    return _SIM_THREAD.submit(run).result()


def _vec(v) -> list[float]:
    return [round(v.x_val, 4), round(v.y_val, 4), round(v.z_val, 4)]


def _quat(q) -> list[float]:
    return [round(q.w_val, 5), round(q.x_val, 5), round(q.y_val, 5), round(q.z_val, 5)]


@mcp.tool()
@threaded
def sim_ping() -> str:
    """Connect to Cosys-AirSim, report server version, vehicles and sim clock."""
    def f(c):
        return {"ping": c.ping(), "server_version": c.getServerVersion(), "client_version": c.getClientVersion(),
                "vehicles": c.listVehicles(), "paused": c.simIsPause()}
    return json.dumps(_sim(f), indent=2)


@mcp.tool()
@threaded
def sim_state(vehicle: str = "") -> str:
    """Multirotor kinematics (NED, m), orientation quaternion (w,x,y,z), GPS, landed state, collision, API control."""
    airsim = _airsim()

    def f(c):
        s = c.getMultirotorState(vehicle_name=vehicle)
        k = s.kinematics_estimated
        col = c.simGetCollisionInfo(vehicle_name=vehicle)
        return {
            "timestamp_ns": s.timestamp,
            "position_ned": _vec(k.position),
            "orientation_wxyz": _quat(k.orientation),
            # cosysairsim 3.4.1 helper: quaternion_to_euler_angles(q) -> (roll, pitch, yaw) in radians
            "euler_roll_pitch_yaw_deg": [round(x * 57.29578, 2) for x in airsim.quaternion_to_euler_angles(k.orientation)],
            "linear_velocity": _vec(k.linear_velocity),
            "gps": {"lat": s.gps_location.latitude, "lon": s.gps_location.longitude, "alt": s.gps_location.altitude},
            "landed_state": int(s.landed_state),
            "api_control": c.isApiControlEnabled(vehicle_name=vehicle),
            "collision": {"has_collided": col.has_collided, "object": col.object_name},
        }
    return json.dumps(_sim(f), indent=2)


@contextlib.contextmanager
def _rpc_timeout(c, seconds: float):
    """msgpackrpc copies Session._timeout into each request when it is SENT (Future(loop, self._timeout)) and a
    *Async(...).join() waits on that request, so any flight longer than the client timeout would fail with
    'Request timed out' while the sim is fine. Raise it for the duration of a long command, then restore."""
    old = c.client._timeout
    c.client._timeout = max(old, seconds)
    try:
        yield
    finally:
        c.client._timeout = old


_FLY_ACTIONS = {"arm", "takeoff", "land", "hover", "move_to", "rtl", "release", "reset"}
_home: dict[str, list[float]] = {}  # per-vehicle home (NED), recorded at arm/reset while on the ground


def _vehicle_brief(c, vehicle: str) -> dict:
    s = c.getMultirotorState(vehicle_name=vehicle)
    k = s.kinematics_estimated
    v = k.linear_velocity
    return {"pos": [k.position.x_val, k.position.y_val, k.position.z_val], "vz": v.z_val,
            "speed": (v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2) ** 0.5, "landed": int(s.landed_state)}


def _settle(c, vehicle: str, timeout_s: float = 10.0, max_speed: float = 0.3) -> bool:
    """Wait, without commanding, until the vehicle is nearly still. After a move SimpleFlight's native position hold
    (the watchdog fallback) settles within ~5 cm; re-sending position goals (0.8 m error after 12 s) or a zero-velocity
    hold (1.2 m drift in 15 s) are both worse - measured 2026-09-10, tools/day1/diag_hold.py."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _vehicle_brief(c, vehicle)["speed"] < max_speed:
            return True
        time.sleep(0.2)
    return False


def _land_controller(c, vehicle: str, x: float, y: float, ground_z: float | None, timeout_s: float) -> dict:
    """Client-side, verified landing at (x, y). landAsync is NOT used: in Cosys-AirSim 3.4.1
    MultirotorApiBase::land() (MultirotorApiBase.cpp:59-76) counts ticks with `z_vel <= approx_zero_vel`, so a hovering
    vehicle satisfies it before descending and the call returns at altitude (measured: 0.2 s at 21 m,
    tools/day1/diag_landing.py); the 60 ms api_goal_timeout (Params.hpp:136) then hovers there.
    Descent HOLDS the horizontal position: moveToPositionAsync(x, y, z_below, v) re-issued every 0.2 s (measured xy
    error 0.001 m; a velocity-only descent drifted 3-4 m into obstacles). Speed is 3 / 1 / 0.4 m/s by height above a
    known ground (e.g. home), 1 m/s otherwise. Touchdown = descent stalled 2 s while commanding down; the vehicle is
    then DISARMED, which is what makes SimpleFlight report landed_state == Landed (OffboardApi::detectLanded needs
    motor output below armed throttle). `takeoff` re-arms."""
    t0, stalled_since, b, touched = time.time(), None, {}, False
    while time.time() - t0 < timeout_s:
        b = _vehicle_brief(c, vehicle)
        if b["landed"] == 0:
            touched = True
            break
        if ground_z is not None:
            agl = ground_z - b["pos"][2]  # NED: z grows downward
            v = 3.0 if agl > 6.0 else (1.0 if agl > 2.0 else 0.4)
        else:
            v = 1.0
        c.moveToPositionAsync(x, y, b["pos"][2] + max(v, 0.5) * 1.5, v, timeout_sec=1.0, vehicle_name=vehicle)
        if b["vz"] < 0.05 and time.time() - t0 > 1.5:  # commanded down but not descending: resting on something
            stalled_since = stalled_since or time.time()
            if time.time() - stalled_since > 2.0:
                touched = True
                break
        else:
            stalled_since = None
        time.sleep(0.2)
    if not touched:
        c.hoverAsync(vehicle_name=vehicle)
        return {"landed": False, "seconds": round(time.time() - t0, 1), "pos": b.get("pos")}
    c.cancelLastTask(vehicle_name=vehicle)
    c.armDisarm(False, vehicle_name=vehicle)
    for _ in range(15):
        b = _vehicle_brief(c, vehicle)
        if b["landed"] == 0:
            break
        time.sleep(0.2)
    return {"landed": b["landed"] == 0, "basis": "touchdown (descent stalled) + disarm -> landed_state Landed",
            "seconds": round(time.time() - t0, 1), "pos": b["pos"]}


@mcp.tool()
@threaded
def sim_fly(action: str, x: float = 0.0, y: float = 0.0, z: float = -10.0, velocity: float = 5.0,
            yaw_deg: float | None = None, timeout_s: float = 60.0, wait: bool = True, vehicle: str = "") -> str:
    """Flight command for a SimpleFlight multirotor (NED metres; z negative is up).
    action: 'arm' (enable API control + arm; records home if landed), 'takeoff' (arms if needed), 'hover',
    'move_to' (STRAIGHT line to x, y, z at `velocity` - climb vertically first when obstacles are near; wait=False
    returns immediately while it flies - poll sim_state; other sim tools such as sim_clock stay usable),
    'land' (settle, then land here), 'rtl' (climb vertically to altitude z if lower, fly to home at that altitude,
    settle, land at home), 'release' (disable API control -> gamepad/RC has the drone), 'reset' (back to spawn).
    'land'/'rtl' use a position-held landing controller (landAsync is broken upstream, see _land_controller), end
    DISARMED with landed_state Landed, and fail unless touchdown is confirmed. 'takeoff' and 'reset' wait for the
    vehicle to come to rest first (a falling vehicle makes takeoffAsync return without climbing) and 'takeoff'
    verifies it actually climbed. Any collision with a real object (not the world "Ground" floor) during the command
    makes the call fail with the object name.
    Between commands SimpleFlight's 60 ms API watchdog shows "API call was not received, entering hover mode for
    safety": that is its native position hold (~5 cm accuracy), the best hold available - expected, not an error."""
    airsim = _airsim()
    if action not in _FLY_ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(_FLY_ACTIONS)}")
    note = ""

    def f(c):
        if action == "arm":
            c.enableApiControl(True, vehicle_name=vehicle)
            c.armDisarm(True, vehicle_name=vehicle)
            b = _vehicle_brief(c, vehicle)
            if b["landed"] == 0:
                _home[vehicle] = b["pos"]
        elif action == "takeoff":
            # a vehicle that is still falling (e.g. right after reset) makes takeoffAsync return without climbing
            _settle(c, vehicle, timeout_s=5.0, max_speed=0.2)
            if vehicle not in _home:
                _home[vehicle] = _vehicle_brief(c, vehicle)["pos"]
            c.enableApiControl(True, vehicle_name=vehicle)
            c.armDisarm(True, vehicle_name=vehicle)  # landing ends disarmed
            z0 = _vehicle_brief(c, vehicle)["pos"][2]
            c.takeoffAsync(timeout_sec=timeout_s, vehicle_name=vehicle).join()
            if z0 - _vehicle_brief(c, vehicle)["pos"][2] < 0.5:  # did not climb: command the altitude explicitly
                c.moveToPositionAsync(_vehicle_brief(c, vehicle)["pos"][0], _vehicle_brief(c, vehicle)["pos"][1],
                                      z0 - 3.0, 1.5, timeout_sec=timeout_s, vehicle_name=vehicle).join()
        elif action == "hover":
            c.hoverAsync(vehicle_name=vehicle).join()
        elif action == "move_to":
            yaw = airsim.YawMode(False, yaw_deg) if yaw_deg is not None else airsim.YawMode()
            fut = c.moveToPositionAsync(x, y, z, velocity, timeout_sec=timeout_s, yaw_mode=yaw,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, vehicle_name=vehicle)
            if wait:
                fut.join()
        elif action == "land":
            _settle(c, vehicle)
            b = _vehicle_brief(c, vehicle)
            return _land_controller(c, vehicle, b["pos"][0], b["pos"][1], None, max(timeout_s, 180.0))
        elif action == "rtl":
            home = _home.get(vehicle, [0.0, 0.0, 0.0])
            b = _vehicle_brief(c, vehicle)
            alt = min(z, b["pos"][2])  # NED: the more negative z is the higher altitude
            if b["pos"][2] - alt > 0.5:  # climb vertically before any horizontal motion
                c.moveToPositionAsync(b["pos"][0], b["pos"][1], alt, 3.0, timeout_sec=timeout_s, vehicle_name=vehicle).join()
            c.moveToPositionAsync(home[0], home[1], alt, velocity, timeout_sec=timeout_s, vehicle_name=vehicle).join()
            _settle(c, vehicle)
            return _land_controller(c, vehicle, home[0], home[1], home[2], max(timeout_s, 180.0))
        elif action == "release":
            c.enableApiControl(False, vehicle_name=vehicle)
        elif action == "reset":
            c.reset()
            time.sleep(0.5)
            _settle(c, vehicle, timeout_s=5.0, max_speed=0.2)  # reset drops the vehicle; let it come to rest
            _home[vehicle] = _vehicle_brief(c, vehicle)["pos"]
        return _vehicle_brief(c, vehicle)

    def g(c):
        t_before = c.simGetCollisionInfo(vehicle_name=vehicle).time_stamp
        with _rpc_timeout(c, max(timeout_s, 180.0) + 60.0):
            out = f(c)
        ci = c.simGetCollisionInfo(vehicle_name=vehicle)
        hit = None
        # "Ground" is the world floor (the flat Blocks floor, or the FloodValley terrain mesh, which is named
        # "Ground" for this reason): contact with it is normal on the pad, during a climb and at touchdown. Other
        # objects (buildings, debris, level geometry) always fail. On terrain a valley wall is also "Ground", so a
        # contact whose normal is more than ~45 deg off vertical is a hillside strike and fails too.
        if ci.has_collided and ci.time_stamp != t_before and ci.object_name:
            steep = abs(ci.normal.z_val) < 0.7
            if ci.object_name != "Ground" or steep:
                hit = {"object": ci.object_name + (" (steep terrain)" if ci.object_name == "Ground" else ""),
                       "penetration_m": round(ci.penetration_depth, 3)}
        return out, hit

    res, hit = _sim(g)
    if action in ("land", "rtl"):
        if not res["landed"]:
            raise RuntimeError(f"{action}: touchdown NOT confirmed after {res['seconds']} s; last position {res['pos']}. "
                               "Vehicle is still airborne - check sim_state, then re-issue land/rtl or take over.")
        note = f" (touchdown confirmed via {res['basis']} in {res['seconds']} s)"
    if hit:
        raise RuntimeError(f"{action}: COLLISION with {hit['object']!r} (penetration {hit['penetration_m']} m) during "
                           f"the command.\n" + sim_state.__wrapped__(vehicle))
    return f"{action}: done{note}\n" + sim_state.__wrapped__(vehicle)


_IMAGE_TYPES = {"scene": "Scene", "depth": "DepthPerspective", "depth_planar": "DepthPlanar",
                "segmentation": "Segmentation", "infrared": "Infrared", "surface_normals": "SurfaceNormals",
                "annotation": "Annotation"}
_capture_warmed = False  # touched only on the single AirSim thread, under _sim_lock


def _capture_once(c, reqs, vehicle: str):
    """simGetImages, with a one-shot warm-up per sim connection.

    Measured 2026-09-10 (cold editor, first PIE, nadir camera at 20.5 m): the FIRST simGetImages of a fresh engine
    process returns the post-processed image types before their capture materials exist - float depth and
    depth_planar both came back as 0.0-0.6133 (an unconverted buffer, not metres) and surface_normals was a
    1 727 850-byte PNG instead of ~80 kB. There is no error and the sizes look right, so the caller cannot tell.
    The very next request (4 s later) was correct and stayed correct. Discarding one request costs ~1 s once per
    sim session and makes every capture a caller sees valid.
    Repro/regression: tools/day1/diag_capture_warmup.py; evidence: docs/verification/dev_workflow.md D1."""
    global _capture_warmed
    if not _capture_warmed:
        with contextlib.suppress(Exception):
            c.simGetImages(reqs, vehicle_name=vehicle)  # discarded: renderer/material warm-up
        time.sleep(0.25)
        _capture_warmed = True
    return c.simGetImages(reqs, vehicle_name=vehicle)


@mcp.tool()
@threaded
def sim_capture(camera: str = "0", image_types: list[str] | None = None, vehicle: str = "", save: bool = True,
                preview: str | None = "scene", preview_width: int = 960, external: bool = False) -> list:
    """Capture synchronized images from one camera: any of scene, segmentation, infrared, depth, depth_planar,
    surface_normals, annotation. Saves PNG (uint8) / NPY (float depth) + a JSON sidecar with camera pose to
    _artifacts/captures/<timestamp>/ and returns a downscaled preview image so results can be inspected."""
    airsim = _airsim()
    image_types = image_types or ["scene", "segmentation", "infrared", "depth"]
    unknown = [t for t in image_types if t not in _IMAGE_TYPES]
    if unknown:
        raise ValueError(f"unknown image type(s) {unknown}; valid: {sorted(_IMAGE_TYPES)}")
    reqs = []
    for t in image_types:
        enum = getattr(airsim.ImageType, _IMAGE_TYPES[t])
        is_float = t.startswith("depth")
        reqs.append(airsim.ImageRequest(camera, enum, pixels_as_float=is_float, compress=not is_float))

    # cosysairsim 3.4.1: simGetImages(requests, vehicle_name) - there is no `external` argument; external cameras
    # are addressed by camera name from the settings "ExternalCameras" block.
    responses = _sim(lambda c: _capture_once(c, reqs, vehicle))

    out_dir = ARTIFACTS / "captures" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
    meta, preview_png = [], None
    for t, r in zip(image_types, responses, strict=True):
        entry = {"type": t, "width": r.width, "height": r.height, "time_stamp": r.time_stamp,
                 "camera_position": _vec(r.camera_position), "camera_orientation_wxyz": _quat(r.camera_orientation)}
        if r.width == 0:
            entry["error"] = "empty image (camera/image type not configured in settings?)"
            meta.append(entry)
            continue
        if t.startswith("depth"):
            arr = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
            entry.update(min=float(arr.min()), max=float(arr.max()))
            if float(arr.max()) <= 1.0:
                # Cosys depth is in METRES; an all-below-1 m frame is almost always the unconverted buffer the
                # renderer returns before the capture materials exist (see _capture_once), not a real scene.
                entry["warning"] = ("depth max <= 1.0 m: this looks like an unconverted buffer, not metres "
                                    "(renderer not warm). Re-capture, or ignore only if the camera really is "
                                    "within 1 m of everything it sees.")
            if save:
                np.save(out_dir / f"{t}.npy", arr)
            vis = (255 * np.clip(arr / max(np.percentile(arr, 99), 1e-3), 0, 1)).astype(np.uint8)
            png_bytes = io.BytesIO()
            PILImage.fromarray(vis).save(png_bytes, "PNG")
            png = png_bytes.getvalue()
        else:
            png = bytes(r.image_data_uint8)
        if save:
            (out_dir / f"{t}.png").write_bytes(png)
            entry["file"] = str(out_dir / f"{t}.png")
        if t == preview:
            im = PILImage.open(io.BytesIO(png)).convert("RGB")
            if im.width > preview_width:
                im = im.resize((preview_width, int(im.height * preview_width / im.width)))
            b = io.BytesIO()
            im.save(b, "JPEG", quality=85)
            preview_png = b.getvalue()
        meta.append(entry)
    if save:
        (out_dir / "capture.json").write_text(json.dumps(meta, indent=2))
    result: list = [json.dumps({"dir": str(out_dir) if save else None, "images": meta}, indent=2)]
    if preview_png:
        result.append(Image(data=preview_png, format="jpeg"))
    return result


@mcp.tool()
@threaded
def sim_environment(weather_enabled: bool | None = None, rain: float | None = None, fog: float | None = None,
                    dust: float | None = None, road_wetness: float | None = None,
                    time_of_day: str | None = None, celestial_clock_speed: float = 1.0,
                    wind_ned_ms: list[float] | None = None) -> str:
    """Set weather (0..1 intensities), time of day ('YYYY-MM-DD HH:MM:SS', requires sky sphere + directional light
    in the level; 'off' restores the level's own lighting) and wind (NED m/s). Only provided fields change."""
    airsim = _airsim()

    def f(c):
        done = []
        if weather_enabled is not None:
            c.simEnableWeather(weather_enabled)
            done.append(f"weather={weather_enabled}")
        for name, val in (("Rain", rain), ("Fog", fog), ("Dust", dust), ("Roadwetness", road_wetness)):
            if val is not None:
                c.simSetWeatherParameter(getattr(airsim.WeatherParameter, name), float(val))
                done.append(f"{name}={val}")
        if time_of_day == "off":
            c.simSetTimeOfDay(False)
            done.append("time_of_day=off")
        elif time_of_day is not None:
            c.simSetTimeOfDay(True, start_datetime=time_of_day, is_start_datetime_dst=False,
                              celestial_clock_speed=celestial_clock_speed, update_interval_secs=1, move_sun=True)
            done.append(f"time={time_of_day}")
        if wind_ned_ms is not None:
            c.simSetWind(airsim.Vector3r(*wind_ned_ms))
            done.append(f"wind={wind_ned_ms}")
        return done
    return "applied: " + ", ".join(_sim(f) or ["nothing"])


@mcp.tool()
@threaded
def sim_clock(action: str, seconds: float = 0.0) -> str:
    """Simulation clock: 'pause', 'resume', 'step' (continue for `seconds` then pause), 'status'.
    Measured semantics (2026-09-10): pause freezes physics and vehicle pose; under ScalableClock the state
    `timestamp_ns` keeps following wall time, so judge a pause by pose, not timestamp. Use the SteppableClock
    settings profile for deterministic capture."""
    if action not in ("pause", "resume", "step", "status"):
        raise ValueError(f"unknown action {action!r}; expected pause|resume|step|status")

    def f(c):
        if action == "pause":
            c.simPause(True)
        elif action == "resume":
            c.simPause(False)
        elif action == "step":
            c.simContinueForTime(seconds)
        return {"paused": c.simIsPause()}
    return json.dumps(_sim(f))


@mcp.tool()
@threaded
def sim_objects(name_regex: str = ".*", with_pose: bool = False, limit: int = 200) -> str:
    """List scene objects (actor names) matching a regex; optionally with world pose (NED, m)."""
    def f(c):
        names = c.simListSceneObjects(name_regex)[:limit]
        if not with_pose:
            return names
        out = []
        for n in names:
            p = c.simGetObjectPose(n)
            out.append({"name": n, "position_ned": _vec(p.position), "orientation_wxyz": _quat(p.orientation)})
        return out
    return json.dumps(_sim(f), indent=2)


@mcp.tool()
@threaded
def sim_set_object_pose(name: str, x: float, y: float, z: float, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                        roll_deg: float = 0.0, teleport: bool = True) -> str:
    """Move a scene object (actor) to a NED pose in metres/degrees, verified by reading the pose back.
    Fails clearly for actors with Static mobility (most level geometry) or unknown names: runtime-movable objects
    should be created with sim_spawn_object."""
    airsim = _airsim()
    # cosysairsim 3.4.1 helper: euler_to_quaternion(roll, pitch, yaw) in radians
    pose = airsim.Pose(airsim.Vector3r(x, y, z),
                       airsim.euler_to_quaternion(roll_deg / 57.29578, pitch_deg / 57.29578, yaw_deg / 57.29578))

    def f(c):
        ok = c.simSetObjectPose(name, pose, teleport)
        return ok, _vec(c.simGetObjectPose(name).position)
    ok, actual = _sim(f)
    err = max(abs(a - b) for a, b in zip(actual, [x, y, z], strict=True)) if all(v == v for v in actual) else float("inf")
    if not ok or err > 0.05:
        raise RuntimeError(f"sim_set_object_pose({name!r}) not applied (returned {ok}, actual {actual}, requested "
                           f"{[x, y, z]}). The actor is Static (immovable at runtime) or the name does not exist; "
                           "spawn movable objects with sim_spawn_object.")
    return json.dumps({"name": name, "requested": [x, y, z], "actual": actual, "verified": True})


@mcp.tool()
@threaded
def sim_detections(camera: str = "0", mesh_name_filter: str = "Human*", radius_m: float = 200.0,
                   image_type: str = "scene", vehicle: str = "") -> str:
    """Cosys-AirSim detection API: amodal 2D boxes, 3D boxes and geo points of meshes matching a wildcard filter
    as seen by a camera (ground-truth labels)."""
    airsim = _airsim()
    if image_type not in _IMAGE_TYPES:  # was a bare KeyError: "'nope'"
        raise ValueError(f"unknown image_type {image_type!r}; valid: {sorted(_IMAGE_TYPES)}")
    enum = getattr(airsim.ImageType, _IMAGE_TYPES[image_type])

    def f(c):
        c.simSetDetectionFilterRadius(camera, enum, radius_m * 100, vehicle_name=vehicle)
        c.simAddDetectionFilterMeshName(camera, enum, mesh_name_filter, vehicle_name=vehicle)
        dets = c.simGetDetections(camera, enum, vehicle_name=vehicle)
        return [{"name": d.name,
                 "box2d": [d.box2D.min.x_val, d.box2D.min.y_val, d.box2D.max.x_val, d.box2D.max.y_val],
                 "geo": [d.geo_point.latitude, d.geo_point.longitude, d.geo_point.altitude]} for d in dets]
    return json.dumps(_sim(f), indent=2)


@mcp.tool()
@threaded
def sim_list_assets(name_regex: str = ".*", limit: int = 300) -> str:
    """List asset names that sim_spawn_object can spawn (static meshes / blueprints known to the sim)."""
    rx = re.compile(name_regex, re.I)
    names = sorted(a for a in _sim(lambda c: c.simListAssets()) if rx.search(a))
    return json.dumps({"count": len(names), "assets": names[:limit]}, indent=2)


@mcp.tool()
@threaded
def sim_spawn_object(name: str, asset: str, x: float, y: float, z: float, yaw_deg: float = 0.0, scale: float = 1.0,
                     physics_enabled: bool = False, is_blueprint: bool = False) -> str:
    """Spawn a runtime-movable object from `asset` (see sim_list_assets) at a NED pose (m, deg). Returns the actual
    actor name (the sim may rename it) and its read-back pose. Used for seeded placement of debris/actors."""
    airsim = _airsim()
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.euler_to_quaternion(0.0, 0.0, yaw_deg / 57.29578))

    def f(c):
        actual = c.simSpawnObject(name, asset, pose, airsim.Vector3r(scale, scale, scale), physics_enabled, is_blueprint)
        if not actual:
            raise RuntimeError(f"simSpawnObject returned no actor for asset {asset!r} (not in sim_list_assets?)")
        p = c.simGetObjectPose(actual)
        return {"name": actual, "position_ned": _vec(p.position), "orientation_wxyz": _quat(p.orientation)}
    return json.dumps(_sim(f), indent=2)


@mcp.tool()
@threaded
def sim_destroy_object(name: str) -> str:
    """Destroy a runtime-spawned object by actor name."""
    if not _sim(lambda c: c.simDestroyObject(name)):
        raise RuntimeError(f"simDestroyObject({name!r}) returned False (unknown actor name?)")
    return f"destroyed {name}"


def main() -> None:
    """Serve MCP over stdio with the JSON-RPC stream isolated from every other writer of stdout.

    fd 1 is duplicated to a private handle that only the transport writes; then fd 1 (C runtime / native code) and
    sys.stdout (Python print(), e.g. cosysairsim's "Connected!") are pointed at stderr, so no stray output can
    corrupt the protocol. Previously only AirSim/editor calls were wrapped in contextlib.redirect_stdout, which swaps
    the process-global sys.stdout and is not thread-safe: two overlapping tool threads could restore the real
    stdout while the other was still printing. Regression test: tests/test_mcp_protocol.py::test_stray_stdout_*."""
    import traceback

    from mcp.server.stdio import stdio_server  # pure Python, already imported by FastMCP

    sys.stdout.flush()
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    out = anyio.wrap_file(io.TextIOWrapper(proto, encoding="utf-8"))

    async def serve() -> None:
        async with stdio_server(stdout=out) as (read_stream, write_stream):
            await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())

    code = 0
    try:
        anyio.run(serve)
    except BaseException:  # noqa: BLE001 - report and exit non-zero
        sys.stderr.write(traceback.format_exc())
        code = 1
    finally:
        sys.stderr.flush()
        # stdin EOF means the client is gone. Do not wait for abandoned worker threads (wedged sim RPC, long editor
        # command) or the AirSim executor's atexit join: exit now so no orphaned server process lingers.
        os._exit(code)


if __name__ == "__main__":
    main()
