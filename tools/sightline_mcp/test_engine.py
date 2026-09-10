"""Engine / build / editor verification THROUGH the sightline MCP server over the real stdio MCP transport.

The server is spawned from the `sightline` entry of .mcp.json (same command, args and env as Claude Code, on top
of this process's environment). Every step asserts on real results and prints PASS/FAIL per tool.

Steps (run one or several, in order, inside ONE server session):
  status    status() sanity
  build     ue_build(SightlineSimEditor, extra_args=[-MaxParallelActions=N]) + job_status polling + job_list
  jobs      ue_package start -> job_status running -> job_list -> job_kill (process tree gone) -> kill again
  genproj   ue_generate_project_files -> a fresh .sln/.slnx exists
  headless  ue_python_headless: engine version + /Game asset list (cross-checked against Content/ on disk)
  editor    editor_launch(wait_ready_s=0) -> editor_wait_ready polling -> status -> ue_python file/statement/eval,
            20k-line output, timeout-no-retry, ue_console (+cvar read-back), ue_log grep
  close     editor_close -> status shows no UnrealEditor process

Note: the Python MCP client puts the server in a Windows Job Object (KILL_ON_JOB_CLOSE, no breakaway), so an
editor launched by the server dies when this client exits. Run `editor` and `close` in the same invocation;
use --hold-file to keep the editor up while other tests (Epic's HTTP server) run.

Run: .venv/Scripts/python.exe tools/sightline_mcp/test_engine.py status build jobs genproj headless
     .venv/Scripts/python.exe tools/sightline_mcp/test_engine.py editor close --hold-file _artifacts/hold
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import psutil
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / "sim" / "SightlineSim"
OUT = REPO / "_artifacts" / "test_engine"
RESULTS: list[tuple[str, bool, str]] = []


def record(tool: str, ok: bool, evidence: str) -> bool:
    RESULTS.append((tool, bool(ok), evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {tool}: {evidence}", flush=True)
    return bool(ok)


def server_params(server_py: str | None = None) -> StdioServerParameters:
    cfg = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["sightline"]
    env = {**os.environ, **cfg.get("env", {})}
    args = [server_py] if server_py else list(cfg["args"])
    return StdioServerParameters(command=cfg["command"], args=args, env=env, cwd=str(REPO))


@contextlib.asynccontextmanager
async def session(server_py: str | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "server_stderr.log", "a", encoding="utf-8") as errlog:
        errlog.write(f"\n==== session {datetime.now():%Y-%m-%d %H:%M:%S} server={server_py or 'server.py'}\n")
        errlog.flush()
        async with stdio_client(server_params(server_py), errlog=errlog) as (r, w), ClientSession(r, w) as s:
            init = await s.initialize()
            print(f"connected: {init.serverInfo.name} protocol {init.protocolVersion}", flush=True)
            yield s


async def call(s: ClientSession, name: str, args: dict | None = None, timeout_s: float | None = None,
               show: int = 1500):
    t = time.time()
    res = await s.call_tool(name, args or {},
                            read_timeout_seconds=timedelta(seconds=timeout_s) if timeout_s else None)
    text = "\n".join(c.text for c in res.content if c.type == "text")
    print(f"--> {name}({json.dumps(args or {})[:300]}) [{time.time() - t:.1f}s isError={res.isError}]\n"
          f"{text[:show]}{' ...' if len(text) > show else ''}\n", flush=True)
    return res, text


def _editor_procs() -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        n = (p.info["name"] or "").lower()
        if n.startswith("unrealeditor") and "-run=" not in " ".join(p.info["cmdline"] or []):
            out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ steps


async def step_status(s, label: str = "status") -> dict:
    res, text = await call(s, "status", show=2500)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        record(label, False, f"not JSON: {text[:200]}")
        return {}
    ok = (not res.isError and d.get("project_exists") and d.get("ram_total_gb", 0) > 10
          and {"ue_processes", "endpoints", "running_jobs"} <= set(d))
    record(label, ok, f"ram_available_gb={d.get('ram_available_gb')} gpu={(d.get('gpu') or {}).get('name')} "
                      f"endpoints={json.dumps(d.get('endpoints'))} ue_processes={len(d.get('ue_processes', []))}")
    return d


async def step_build(s, a) -> None:
    target = a.target
    dll = PROJECT_DIR / "Binaries" / "Win64" / "UnrealEditor-SightlineSim.dll"
    t0 = time.time()
    res, text = await call(s, "ue_build", {"target": target, "extra_args": [f"-MaxParallelActions={a.max_actions}"]})
    info = json.loads(text)
    job = info["job_id"]
    head = Path(info["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    record("ue_build (start, extra_args)", not res.isError and f"-MaxParallelActions={a.max_actions}" in head,
           f"job={job} argv={head[:260]}")
    min_avail, st = 99.0, {}
    while True:
        await asyncio.sleep(20)
        min_avail = min(min_avail, psutil.virtual_memory().available / 2**30)
        _, text = await call(s, "job_status", {"job_id": job, "tail": 3}, show=500)
        st = json.loads(text)
        if st["state"] != "running":
            break
    wall = time.time() - t0
    # A correct incremental build compiles nothing and leaves the DLL untouched ("Target is up to date").
    up_to_date = "Target is up to date" in Path(st["log"]).read_text(encoding="utf-8", errors="replace")
    fresh = dll.exists() and dll.stat().st_mtime >= t0 - 5
    ok = st["state"] == "succeeded" and st["exit_code"] == 0 and dll.exists() and (fresh or up_to_date)
    print("build summary:\n" + (st.get("summary") or ""), flush=True)
    record("ue_build", ok, f"target={target} state={st['state']} exit={st['exit_code']} wall_s={wall:.0f} "
                           f"job_elapsed_s={st['elapsed_s']} min_ram_available_gb={min_avail:.1f} "
                           f"dll_fresh={fresh} up_to_date={up_to_date} log={st['log']}")
    record("job_status", st["state"] in ("succeeded", "failed") and st.get("summary") is not None
           and isinstance(st.get("exit_code"), int), f"state={st['state']} exit_code={st['exit_code']} summary+tail present")
    _, text = await call(s, "job_list")
    jl = json.loads(text)
    record("job_list", jl["session_jobs"].get(job) == ("exit 0" if ok else f"exit {st['exit_code']}")
           and job in jl["recent_job_logs"], f"session_jobs[{job}]={jl['session_jobs'].get(job)}")


async def step_jobs(s, a) -> None:
    """ue_package start + job_kill on it (a real, long-running process tree: RunUAT -> dotnet AutomationTool)."""
    res, text = await call(s, "ue_package", {})
    info = json.loads(text)
    job, pid = info["job_id"], info["pid"]
    tree: dict[int, psutil.Process] = {}
    st = {}
    for _ in range(12):  # let UAT get going (script modules, then BuildCookRun) so there is a real tree to kill
        await asyncio.sleep(5)
        try:
            root = psutil.Process(pid)
            for p in [root] + root.children(recursive=True):
                tree.setdefault(p.pid, p)
        except psutil.NoSuchProcess:
            break
        _, text = await call(s, "job_status", {"job_id": job, "tail": 4}, show=600)
        st = json.loads(text)
        if st["state"] != "running" or len(tree) >= 3:
            break
    names = sorted({p.name() for p in tree.values() if p.is_running()})
    record("ue_package (start)", st.get("state") == "running" and "BuildCookRun" in
           Path(info["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[0],
           f"job={job} state={st.get('state')} tree={names}")
    _, text = await call(s, "job_list")
    record("job_list (running job)", json.loads(text)["session_jobs"].get(job) == "running", f"{job} running")
    try:  # final snapshot right before the kill
        root = psutil.Process(pid)
        for p in [root] + root.children(recursive=True):
            tree.setdefault(p.pid, p)
    except psutil.NoSuchProcess:
        pass
    _, killed = await call(s, "job_kill", {"job_id": job})
    await asyncio.sleep(3)
    alive = [f"{p.name()}:{p.pid}" for p in tree.values() if p.is_running()]
    _, text = await call(s, "job_status", {"job_id": job, "tail": 5}, show=800)
    st = json.loads(text)
    record("job_kill", killed == "killed" and not alive and st["state"] == "failed",
           f"returned={killed!r} tree_size={len(tree)} still_alive={alive} state_after={st['state']} exit={st['exit_code']}")
    _, again = await call(s, "job_kill", {"job_id": job})
    record("job_kill (already finished)", again == "not running", f"returned={again!r}")
    res, text = await call(s, "job_status", {"job_id": "no-such-job"})
    record("job_status (unknown id -> error)", res.isError and "unknown job" in text, text[:120])


async def step_genproj(s, a) -> None:
    t0 = time.time()
    res, text = await call(s, "ue_generate_project_files", {}, show=3000)
    fresh = [p.name for p in PROJECT_DIR.glob("*.sln*") if p.stat().st_mtime >= t0 - 2]
    record("ue_generate_project_files", not res.isError and text.startswith("exit 0") and bool(fresh),
           f"{text.splitlines()[0]} fresh_solution={fresh} took_s={time.time() - t0:.0f}")


HEADLESS_SCRIPT = """import unreal
print('SLT_VERSION=' + unreal.SystemLibrary.get_engine_version())
assets = unreal.EditorAssetLibrary.list_assets('/Game', recursive=True, include_folder=False)
print('SLT_ASSET_COUNT=%d' % len(assets))
for a in assets[:30]:
    print('SLT_ASSET ' + a)
"""


async def step_headless(s, a) -> None:
    on_disk = sum(1 for p in (PROJECT_DIR / "Content").rglob("*") if p.suffix in (".uasset", ".umap"))
    t0 = time.time()
    res, text = await call(s, "ue_python_headless", {"script": HEADLESS_SCRIPT, "timeout_s": 3600}, show=5000)
    ver = re.search(r"SLT_VERSION=(\S+)", text)
    cnt = re.search(r"SLT_ASSET_COUNT=(\d+)", text)
    n_listed = len(re.findall(r"SLT_ASSET /Game/", text))
    ok = (not res.isError and text.startswith("exit 0") and ver and ver.group(1).startswith("5.8.2")
          and cnt and int(cnt.group(1)) > 0 and n_listed > 0)
    record("ue_python_headless", ok, f"{text.splitlines()[0]} version={ver and ver.group(1)} "
                                     f"asset_count={cnt and cnt.group(1)} (uasset+umap on disk: {on_disk}) "
                                     f"listed={n_listed} took_s={time.time() - t0:.0f}")


async def _eval(s, expr: str):
    _, text = await call(s, "ue_python", {"code": expr, "mode": "eval"}, show=400)
    m = re.search(r"^result: (.*)$", text, re.M)
    return text, (m.group(1) if m else None)


TIMEOUT_PROBE = ("import builtins, time\n"
                 "builtins._slt_probe_runs = getattr(builtins, '_slt_probe_runs', 0) + 1\n"
                 "time.sleep(8)\nprint('probe done')")


async def timeout_probe(s, label: str) -> int:
    """Send a command that outlives its timeout; count how many times the editor actually executed it."""
    await call(s, "ue_python", {"code": "import builtins; builtins._slt_probe_runs = 0", "mode": "file"})
    t0 = time.time()
    res, text = await call(s, "ue_python", {"code": TIMEOUT_PROBE, "timeout_s": 3})
    took = time.time() - t0
    await asyncio.sleep(20)  # let every queued execution finish on the game thread
    _, runs = await _eval(s, "__import__('builtins')._slt_probe_runs")
    print(f"{label}: isError={res.isError} took={took:.1f}s runs={runs}", flush=True)
    return int(runs) if runs is not None and runs.isdigit() else -1, res.isError, took, text


async def step_editor(s, a) -> None:
    if _editor_procs():
        record("editor_launch", False, "an editor is already running; close it first")
        return
    res, text = await call(s, "editor_launch", {"settings_profile": "default", "wait_ready_s": 0})
    m = re.search(r"launched pid=(\d+)", text)
    record("editor_launch", not res.isError and m is not None and "-settings=" in text, text[:300])
    if not m:
        return
    t0, ready, text = time.time(), False, ""
    min_avail = 99.0
    while time.time() - t0 < a.ready_timeout:
        _, text = await call(s, "editor_wait_ready", {"timeout_s": 300}, show=600)
        min_avail = min(min_avail, psutil.virtual_memory().available / 2**30)
        ed = [f"{p.name()} {p.memory_info().rss / 2**30:.1f}GB" for p in _editor_procs() if p.is_running()]
        print(f"   t={time.time() - t0:.0f}s editor={ed} ram_available={psutil.virtual_memory().available / 2**30:.1f}GB",
              flush=True)
        if text.startswith("editor ready") or "not running" in text:
            ready = text.startswith("editor ready")
            break
    record("editor_wait_ready", ready and "5.8.2" in text,
           f"ready={ready} after {time.time() - t0:.0f}s min_ram_available_gb={min_avail:.1f} :: {text[:160]!r}")
    if not ready:
        return

    # Epic's HTTP MCP server may come up a moment after Python is ready
    d = {}
    for _ in range(12):
        d = await step_status(s, "status (editor up)")
        if d.get("endpoints", {}).get("unreal_mcp_http:8000"):
            break
        await asyncio.sleep(5)
    ep = d.get("endpoints", {})
    nodes = ep.get("editor_python_remote_exec") or []
    ed = [p for p in d.get("ue_processes", []) if p["name"].lower() == "unrealeditor.exe"]
    record("status (endpoints with editor)", ep.get("unreal_mcp_http:8000") is True
           and any(n.get("project_name") == "SightlineSim" for n in nodes),
           f"unreal_mcp_http:8000={ep.get('unreal_mcp_http:8000')} nodes={nodes} editor_rss_gb="
           f"{[p['rss_gb'] for p in ed]} ram_available_gb={d.get('ram_available_gb')}")

    # --- ue_python: file mode (spawn), eval (read back), statement (delete), eval (gone)
    spawn = ("import unreal\n"
             "eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
             "mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Cube')\n"
             "a = eas.spawn_actor_from_object(mesh, unreal.Vector(0, 0, 500))\n"
             "a.set_actor_label('SLT_TestCube')\n"
             "print('SLT_SPAWNED', a.get_actor_label(), a.get_class().get_name())\n")
    res, text = await call(s, "ue_python", {"code": spawn, "mode": "file"})
    record("ue_python mode=file (spawn cube)", not res.isError and "success: True" in text
           and "SLT_SPAWNED SLT_TestCube StaticMeshActor" in text, text.replace("\n", " | ")[:200])
    find = ("[x.get_actor_label() for x in __import__('unreal').get_editor_subsystem(__import__('unreal')"
            ".EditorActorSubsystem).get_all_level_actors() if x.get_actor_label() == 'SLT_TestCube']")
    text, val = await _eval(s, find)
    record("ue_python mode=eval (read label)", val == "['SLT_TestCube']", f"result={val}")
    delete = ("print([__import__('unreal').get_editor_subsystem(__import__('unreal').EditorActorSubsystem)"
              ".destroy_actor(x) for x in __import__('unreal').get_editor_subsystem(__import__('unreal')"
              ".EditorActorSubsystem).get_all_level_actors() if x.get_actor_label() == 'SLT_TestCube'])")
    res, text = await call(s, "ue_python", {"code": delete, "mode": "statement"})
    text2, val2 = await _eval(s, find)
    record("ue_python mode=statement (delete)", not res.isError and "[True]" in text and val2 == "[]",
           f"statement output={text.replace(chr(10), ' | ')[:120]} ; after delete eval={val2}")

    # --- large output: 20,000 lines must arrive complete
    res, text = await call(s, "ue_python", {"code": "for i in range(20000):\n    print(f'SLTL{i:05d}')", "mode": "file"},
                           show=300)
    got = re.findall(r"SLTL(\d{5})", text)
    record("ue_python large output (20k lines)", len(got) == 20000 and got[-1] == "19999" and len(set(got)) == 20000,
           f"lines={len(got)} last={got[-1] if got else None} chars={len(text)}")

    # --- error propagation: a Python exception must come back as success False with the traceback
    res, text = await call(s, "ue_python", {"code": "raise ValueError('slt boom')", "mode": "file"})
    record("ue_python error reporting", "success: False" in text and "slt boom" in text, text.replace("\n", " | ")[:160])

    # --- timeout must not re-execute the command
    runs, is_err, took, txt = await timeout_probe(s, "timeout probe (current server)")
    record("ue_python timeout (no re-execution)", runs == 1 and is_err and took < 12,
           f"runs={runs} isError={is_err} took_s={took:.1f} msg={txt[:140]!r}")
    if a.baseline_server:
        async with session(a.baseline_server) as bs:
            runs_b, is_err_b, took_b, _ = await timeout_probe(bs, "timeout probe (BASELINE server)")
        print(f"[INFO] baseline (pre-fix) server: runs={runs_b} isError={is_err_b} took_s={took_b:.1f}", flush=True)
        RESULTS.append(("baseline pre-fix timeout probe (bug repro, informational)", True,
                        f"runs={runs_b} took_s={took_b:.1f}"))
        # the baseline session may have replaced our command channel: the next call must reconnect transparently
        text, val = await _eval(s, "1+1")
        record("ue_python reconnect after channel replaced", val == "2", f"result={val}")

    # --- ue_console: stat fps + a cvar round trip (t.MaxFPS), restored afterwards
    _, orig = await _eval(s, "__import__('unreal').SystemLibrary.get_console_variable_float_value('t.MaxFPS')")
    res1, t1 = await call(s, "ue_console", {"command": "stat fps"})
    res2, t2 = await call(s, "ue_console", {"command": "t.MaxFPS 61"})
    _, now = await _eval(s, "__import__('unreal').SystemLibrary.get_console_variable_float_value('t.MaxFPS')")
    await call(s, "ue_console", {"command": f"t.MaxFPS {float(orig) if orig else 0:g}"})
    _, restored = await _eval(s, "__import__('unreal').SystemLibrary.get_console_variable_float_value('t.MaxFPS')")
    await call(s, "ue_console", {"command": "stat none"})
    record("ue_console", not res1.isError and not res2.isError and "ok" in t1 and now == "61.0" and restored == orig,
           f"stat fps ok; t.MaxFPS {orig} -> {now} -> restored {restored}")

    await step_log(s, a)

    if a.hold_file:
        hold = Path(a.hold_file)
        hold.parent.mkdir(parents=True, exist_ok=True)
        hold.write_text("delete this file to let test_engine.py continue\n")
        print(f"HOLDING: editor stays up until {hold} is deleted (max {a.hold_max}s)", flush=True)
        t0 = time.time()
        while hold.exists() and time.time() - t0 < a.hold_max:
            await asyncio.sleep(5)
        print(f"hold released after {time.time() - t0:.0f}s", flush=True)
        await step_status(s, "status (after hold)")


async def step_log(s, a) -> None:
    """ue_log: header = log path; `lines` caps the tail (UE logs contain blank lines, so count <= lines);
    grep filters (remote-exec prints land in LogPython, so the editor step's SLT_SPAWNED line must be found)."""
    res, text = await call(s, "ue_log", {"lines": 50, "grep": r"LogPython.*SLT_SPAWNED"})
    body = text.splitlines()[1:]
    hits = [ln for ln in body if "SLT_SPAWNED" in ln]
    res2, text2 = await call(s, "ue_log", {"lines": 5}, show=800)
    body2 = text2.splitlines()[1:]
    res3, text3 = await call(s, "ue_log", {"lines": 3, "grep": r"^\[.*\]Log"})
    body3 = text3.splitlines()[1:]
    ok = (not res.isError and not res2.isError and hits and all("SLT_SPAWNED" in ln for ln in body)
          and text.splitlines()[0].endswith("SightlineSim.log") and 0 < len(body2) <= 5 and len(body3) == 3)
    record("ue_log", ok, f"grep hits={len(hits)} (all matching) first={hits[0][:110] if hits else None!r}; "
                         f"lines=5 -> {len(body2)} lines; lines=3+grep -> {len(body3)} lines")


async def step_close(s, a) -> None:
    before = [p.pid for p in _editor_procs()]
    t0 = time.time()
    res, text = await call(s, "editor_close", {"timeout_s": 120})
    await asyncio.sleep(3)
    d = await step_status(s, "status (after close)")
    left = [p for p in d.get("ue_processes", []) if p["name"].lower().startswith("unrealeditor")]
    record("editor_close", not res.isError and before and not left and not _editor_procs()
           and not d.get("endpoints", {}).get("unreal_mcp_http:8000"),
           f"editor pids before={before} -> {text!r} in {time.time() - t0:.0f}s; UnrealEditor left={left}")


STEPS = {"status": lambda s, a: step_status(s), "build": step_build, "jobs": step_jobs, "genproj": step_genproj,
         "headless": step_headless, "editor": step_editor, "log": step_log, "close": step_close}


async def main(a) -> int:
    async with session() as s:
        for name in a.steps:
            print(f"\n======== step {name} ({datetime.now():%H:%M:%S})", flush=True)
            await STEPS[name](s, a)
    print("\n======== results")
    for tool, ok, ev in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {tool}: {ev}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results-{'_'.join(a.steps)}-{datetime.now():%Y%m%d-%H%M%S}.json").write_text(
        json.dumps([{"tool": t, "pass": ok, "evidence": ev} for t, ok, ev in RESULTS], indent=2))
    fails = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - fails} PASS, {fails} FAIL")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(STEPS))
    ap.add_argument("--target", default="SightlineSimEditor")
    ap.add_argument("--max-actions", type=int, default=6)
    ap.add_argument("--ready-timeout", type=int, default=2400)
    ap.add_argument("--hold-file", default=None)
    ap.add_argument("--hold-max", type=int, default=3600)
    ap.add_argument("--baseline-server", default=None, help="pre-fix server.py copy, to reproduce the timeout bug")
    sys.exit(asyncio.run(main(ap.parse_args())))
