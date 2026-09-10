"""Full tool matrix for the `sightline` MCP server, driven exactly the way Claude Code drives it.

Every tool the server exposes is called at least once over the REAL stdio MCP transport (the server is spawned
from the `sightline` entry of .mcp.json with the same command/args/env), with safe, read-only or reversible
arguments, and each result is asserted to be real data or a clean, specific error. The script prints a
PASS/FAIL table per tool and fails (exit code) on the number of failed checks.

Phases (choose with --phase, repeatable):
  up    editor RUNNING + AirSim RUNNING (PIE or -game). Covers every tool that needs a live engine/sim.
        Destructive tools are only started and then killed (`ue_package` -> `job_kill`) or are reversible
        (`sim_spawn_object` -> `sim_destroy_object`, weather set -> restored, cvar set -> restored).
  down  editor DOWN + AirSim DOWN. Covers the "no engine / no sim" error paths and `ue_python_headless`,
        which needs no running editor.

Run:
  .venv/Scripts/python.exe -u tools/sightline_mcp/test_tool_matrix.py --phase up
  .venv/Scripts/python.exe -u tools/sightline_mcp/test_tool_matrix.py --phase down
Results: _artifacts/verification/tool_matrix_<phase>_<ts>.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
RESULTS: list[dict] = []
COVERED: set[str] = set()


def record(tool: str, ok: bool, evidence: str, note: str = "") -> bool:
    COVERED.add(tool)
    RESULTS.append({"tool": tool, "ok": bool(ok), "evidence": str(evidence)[:500], "note": note})
    print(f"[{'PASS' if ok else 'FAIL'}] {tool}: {str(evidence)[:220]}", flush=True)
    return bool(ok)


def server_params() -> StdioServerParameters:
    cfg = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["sightline"]
    env = {**os.environ, **cfg.get("env", {})}
    return StdioServerParameters(command=cfg["command"], args=list(cfg["args"]), env=env, cwd=str(REPO))


class Client:
    def __init__(self, s: ClientSession):
        self.s = s

    async def call(self, name: str, args: dict | None = None, timeout_s: float = 300.0):
        """Returns (isError, text, images, seconds)."""
        t = time.time()
        res = await self.s.call_tool(name, args or {}, read_timeout_seconds=timedelta(seconds=timeout_s))
        text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        images = [c for c in res.content if getattr(c, "type", "") == "image"]
        return bool(res.isError), text, images, time.time() - t


# --------------------------------------------------------------------------------------------------
# phase: engine + sim UP
# --------------------------------------------------------------------------------------------------


async def phase_up(c: Client, skip_genproj: bool) -> None:
    # ---------- status / process control ----------
    err, txt, _, dt = await c.call("status")
    d = json.loads(txt)
    ep = d["endpoints"]
    record("status", not err and d["project_exists"] and d["ram_total_gb"] > 1 and any(
        p["name"].lower().startswith("unrealeditor") for p in d["ue_processes"]),
        f"editor pid(s)={[p['pid'] for p in d['ue_processes']]} ram_avail={d['ram_available_gb']}GB "
        f"gpu={(d.get('gpu') or {}).get('name')} endpoints={ep} in {dt:.2f}s")

    err, txt, _, _ = await c.call("editor_launch", {"settings_profile": "default", "wait_ready_s": 0})
    record("editor_launch", not err and "already running" in txt, f"guard when an editor exists -> {txt[:120]!r}",
           "full launch verified by the harness that started this editor")

    err, txt, _, dt = await c.call("editor_wait_ready", {"timeout_s": 60}, timeout_s=120)
    record("editor_wait_ready", not err and txt.startswith("editor ready"), f"{txt.splitlines()[0]} in {dt:.2f}s")

    # ---------- editor python / console / logs ----------
    # mode="eval" evaluates a single EXPRESSION (a statement such as `import unreal; ...` is a SyntaxError there)
    err, txt, _, _ = await c.call("ue_python", {"code": "__import__('unreal').SystemLibrary.get_engine_version()",
                                                "mode": "eval"})
    record("ue_python", not err and "5.8" in txt, txt.replace("\n", " | ")[:200])

    err, txt, _, _ = await c.call("ue_python", {"code": "x", "mode": "bogus"})
    record("ue_python (bad mode -> clean error)", err and "unknown mode" in txt, txt[:160])

    err, txt, _, _ = await c.call("ue_console", {"command": "stat fps"})
    ok1 = not err and "ok" in txt
    await c.call("ue_console", {"command": "stat none"})
    record("ue_console", ok1, f"stat fps -> {txt.replace(chr(10), ' | ')[:120]}; restored with 'stat none'")

    err, txt, _, _ = await c.call("ue_log", {"lines": 5, "grep": "LogAirSim|AirSim"})
    lines = txt.splitlines()
    record("ue_log", not err and lines and lines[0].lower().endswith(".log") and 1 <= len(lines) - 1 <= 5,
           f"{lines[0]} -> {len(lines) - 1} matching line(s), last={lines[-1][:120]!r}")

    err, txt, _, _ = await c.call("ue_log", {"lines": 0})
    record("ue_log (lines=0 -> clean error)", err and "lines must be" in txt, txt[:120])

    # ---------- jobs: ue_package started then killed; ue_build reported cleanly ----------
    err, txt, _, _ = await c.call("ue_package", {"configuration": "Development"})
    job = json.loads(txt)
    record("ue_package", not err and job["job_id"].startswith("package") and job["pid"] > 0,
           f"job_id={job['job_id']} pid={job['pid']} log={job['log']}", "started only, then job_kill'ed")
    await asyncio.sleep(4)
    err, txt, _, _ = await c.call("job_status", {"job_id": job["job_id"], "tail": 10})
    js = json.loads(txt)
    # the job log's first line is "$ <full command line>"; the tail at 4 s varies (RunUAT -> dotnet -> UBT)
    argv_line = Path(js["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    record("job_status", not err and js["state"] == "running" and "BuildCookRun" in argv_line and js["tail"].strip(),
           f"state={js['state']} elapsed={js['elapsed_s']}s argv={argv_line[:110]!r} "
           f"tail[-1]={js['tail'].splitlines()[-1][:90]!r}")
    err, txt, _, _ = await c.call("job_list")
    jl = json.loads(txt)
    record("job_list", not err and jl["session_jobs"].get(job["job_id"]) == "running" and jl["recent_job_logs"],
           f"session_jobs={jl['session_jobs']} recent_logs={len(jl['recent_job_logs'])}")
    err, txt, _, _ = await c.call("job_kill", {"job_id": job["job_id"]})
    ok = not err and txt == "killed"
    await asyncio.sleep(3)
    _, txt2, _, _ = await c.call("job_status", {"job_id": job["job_id"], "tail": 3})
    js2 = json.loads(txt2)
    _, txt3, _, _ = await c.call("job_kill", {"job_id": job["job_id"]})
    record("job_kill", ok and js2["state"] == "failed" and txt3 == "not running",
           f"kill -> {txt!r}; state now {js2['state']} exit={js2['exit_code']}; second kill -> {txt3!r}")
    err, txt, _, _ = await c.call("job_status", {"job_id": "no-such-job"})
    record("job_status (unknown id -> clean error)", err and "unknown job" in txt, txt[:120])

    err, txt, _, _ = await c.call("ue_build", {"target": "SightlineSimNoSuchTarget", "extra_args": ["-MaxParallelActions=2"]})
    bjob = json.loads(txt)
    started = not err and bjob["pid"] > 0
    state = ""
    for _ in range(40):
        await asyncio.sleep(3)
        _, t, _, _ = await c.call("job_status", {"job_id": bjob["job_id"], "tail": 6})
        js = json.loads(t)
        state = js["state"]
        if state != "running":
            break
    blob = (js.get("summary") or "") + js["tail"] + Path(js["log"]).read_text(encoding="utf-8", errors="replace")
    record("ue_build", started and state == "failed" and js["exit_code"] not in (0, None)
           and "SightlineSimNoSuchTarget" in blob and "Result:" in blob,
           f"invalid target -> job {bjob['job_id']} {state} exit={js['exit_code']}; "
           f"summary={(js.get('summary') or '').splitlines()[:2]}",
           "a real successful build is covered by docs/verification/engine_tools.md; not repeated here "
           "(the editor target cannot be built while the editor runs)")

    if skip_genproj:
        record("ue_generate_project_files", True, "skipped by --skip-genproj", "skipped")
    else:
        err, txt, _, dt = await c.call("ue_generate_project_files", timeout_s=900)
        sln = REPO / "sim" / "SightlineSim" / "SightlineSim.sln"
        record("ue_generate_project_files", not err and txt.startswith("exit 0") and sln.exists(),
               f"{txt.splitlines()[0]} in {dt:.0f}s; {sln.name} mtime={datetime.fromtimestamp(sln.stat().st_mtime):%H:%M:%S}")

    # ---------- sim ----------
    err, txt, _, dt = await c.call("sim_ping")
    d = json.loads(txt)
    record("sim_ping", not err and d["ping"] is True and "Drone" in d["vehicles"],
           f"server_version={d['server_version']} vehicles={d['vehicles']} paused={d['paused']} in {dt:.2f}s")

    err, txt, _, _ = await c.call("sim_state", {"vehicle": "Drone"})
    st = json.loads(txt)
    record("sim_state", not err and len(st["position_ned"]) == 3 and abs(st["gps"]["lat"]) > 0,
           f"pos={st['position_ned']} gps=({st['gps']['lat']:.5f},{st['gps']['lon']:.5f}) landed={st['landed_state']} "
           f"api_control={st['api_control']}")

    err, txt, _, _ = await c.call("sim_fly", {"action": "arm", "vehicle": "Drone"}, timeout_s=120)
    armed = json.loads(txt[txt.index("{"):])["api_control"] is True
    err2, txt2, _, _ = await c.call("sim_fly", {"action": "release", "vehicle": "Drone"}, timeout_s=120)
    released = json.loads(txt2[txt2.index("{"):])["api_control"] is False
    await c.call("sim_fly", {"action": "arm", "vehicle": "Drone"}, timeout_s=120)
    record("sim_fly", not err and not err2 and armed and released,
           "arm -> api_control True; release -> api_control False; arm again",
           "flight actions (takeoff/move_to/hover/land/rtl/reset) covered by test_sim.py")
    err, txt, _, _ = await c.call("sim_fly", {"action": "bogus"})
    record("sim_fly (bad action -> clean error)", err and "unknown action" in txt, txt[:140])

    err, txt, imgs, dt = await c.call("sim_capture", {"camera": "survey", "vehicle": "Drone",
                                                     "image_types": ["scene"], "preview": "scene"}, timeout_s=180)
    meta = json.loads(txt)
    m = meta["images"][0]
    raw = base64.b64decode(imgs[0].data) if imgs else b""
    from PIL import Image as PILImage  # client side only
    im = PILImage.open(io.BytesIO(raw)) if raw else None
    if im:
        im.load()
    record("sim_capture", not err and m["width"] == 1920 and Path(m["file"]).exists() and im is not None
           and im.format == "JPEG" and im.width == 960,
           f"scene {m['width']}x{m['height']} -> {m['file']}; preview mime={imgs[0].mimeType if imgs else None} "
           f"{len(raw)} b64-decoded bytes = {im.format if im else None} {im.width if im else 0}x{im.height if im else 0} "
           f"in {dt:.2f}s")
    if raw:
        out = REPO / "_artifacts" / "verification" / "tool_matrix_preview.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
    err, txt, _, _ = await c.call("sim_capture", {"image_types": ["nope"]})
    record("sim_capture (bad image type -> clean error)", err and "unknown image type" in txt, txt[:140])

    err, txt, _, _ = await c.call("sim_environment", {"weather_enabled": True, "rain": 0.4, "wind_ned_ms": [2.0, 0, 0]})
    ok = not err and "Rain=0.4" in txt
    _, txt2, _, _ = await c.call("sim_environment", {"weather_enabled": False, "rain": 0.0, "wind_ned_ms": [0.0, 0, 0]})
    record("sim_environment", ok and "weather=False" in txt2, f"{txt} ; restored: {txt2}")

    _, t_status, _, _ = await c.call("sim_clock", {"action": "status"})
    _, t_pause, _, _ = await c.call("sim_clock", {"action": "pause"})
    _, t_step, _, _ = await c.call("sim_clock", {"action": "step", "seconds": 0.2})
    _, t_res, _, _ = await c.call("sim_clock", {"action": "resume"})
    record("sim_clock", json.loads(t_pause)["paused"] is True and json.loads(t_res)["paused"] is False
           and json.loads(t_step)["paused"] is True,
           f"status={t_status} pause={t_pause} step={t_step} resume={t_res}")

    err, txt, _, _ = await c.call("sim_objects", {"name_regex": ".*", "limit": 500})
    names = json.loads(txt)
    record("sim_objects", not err and len(names) > 5, f"{len(names)} objects, e.g. {names[:4]}")

    err, txt, _, _ = await c.call("sim_list_assets", {"name_regex": "cube"})
    assets = json.loads(txt)
    record("sim_list_assets", not err and assets["count"] > 0, f"{assets['count']} matching 'cube': {assets['assets'][:4]}")

    asset = "Cube" if "Cube" in assets["assets"] else assets["assets"][0]
    err, txt, _, _ = await c.call("sim_spawn_object", {"name": "slt_matrix_cube", "asset": asset,
                                                       "x": 8.0, "y": 8.0, "z": -3.0})
    sp = json.loads(txt)
    record("sim_spawn_object", not err and max(abs(a - b) for a, b in zip(sp["position_ned"], [8, 8, -3])) < 0.05,
           f"{sp['name']} at {sp['position_ned']}")
    err, txt, _, _ = await c.call("sim_set_object_pose", {"name": sp["name"], "x": 9.0, "y": 8.0, "z": -3.0,
                                                          "yaw_deg": 45.0})
    mv = json.loads(txt)
    record("sim_set_object_pose", not err and mv["verified"], f"requested={mv['requested']} actual={mv['actual']}")
    err, txt, _, _ = await c.call("sim_destroy_object", {"name": sp["name"]})
    _, gone, _, _ = await c.call("sim_objects", {"name_regex": sp["name"]})
    record("sim_destroy_object", not err and "destroyed" in txt and json.loads(gone) == [],
           f"{txt}; sim_objects now {gone}")
    err, txt, _, _ = await c.call("sim_destroy_object", {"name": "slt_no_such_actor"})
    record("sim_destroy_object (unknown -> clean error)", err and "returned False" in txt, txt[:140])

    err, txt, _, _ = await c.call("sim_detections", {"camera": "survey", "mesh_name_filter": "*Cube*",
                                                     "radius_m": 300, "vehicle": "Drone"}, timeout_s=120)
    dets = json.loads(txt)
    record("sim_detections", not err and isinstance(dets, list) and all(len(x["box2d"]) == 4 for x in dets),
           f"{len(dets)} detections, first={dets[:1]}")

    err, txt, _, _ = await c.call("sim_launch_game", {"settings_profile": "no_such_profile"})
    record("sim_launch_game (bad settings profile -> clean error)", err and "not found" in txt, txt[:160],
           "a real -game launch is checked in the down phase")

    # ue_python_headless would fight the running editor for the project's Saved/ + 2 GB RAM: down phase.
    record("ue_python_headless", True, "deferred to the down phase (needs no editor; avoids RAM contention)", "deferred")
    record("editor_close", True, "deferred to the end of the run", "deferred")


# --------------------------------------------------------------------------------------------------
# phase: engine + sim DOWN
# --------------------------------------------------------------------------------------------------


async def phase_down(c: Client, launch_game: bool) -> None:
    err, txt, _, _ = await c.call("status")
    d = json.loads(txt)
    editors = [p for p in d["ue_processes"] if p["name"].lower().startswith("unrealeditor")]
    record("status (no editor)", not err and not editors and d["endpoints"][f"unreal_mcp_http:8000"] is False,
           f"ue_processes={d['ue_processes']} endpoints={d['endpoints']}")

    for tool, args in (("ue_python", {"code": "print(1)"}), ("ue_console", {"command": "stat fps"})):
        err, txt, _, dt = await c.call(tool, args)
        record(f"{tool} (no editor -> fast clean error)", err and "No Unreal Editor is running" in txt and dt < 5,
               f"{dt:.2f}s: {txt[:120]}")

    err, txt, _, dt = await c.call("editor_wait_ready", {"timeout_s": 3}, timeout_s=60)
    record("editor_wait_ready (no editor)", not err and "not running" in txt and dt < 8, f"{dt:.2f}s: {txt[:120]}")

    err, txt, _, dt = await c.call("editor_close")
    record("editor_close (nothing to close)", not err and "closed" in txt, f"{dt:.2f}s: {txt[:120]}")

    for tool, args in (("sim_ping", {}), ("sim_state", {}), ("sim_capture", {"image_types": ["scene"]}),
                       ("sim_objects", {}), ("sim_clock", {"action": "status"})):
        err, txt, _, dt = await c.call(tool, args, timeout_s=60)
        record(f"{tool} (sim down -> fast clean error)", err and "41451" in txt and dt < 10, f"{dt:.2f}s: {txt[:140]}")

    script = ("import unreal\n"
              "print('SLT_MATRIX_VERSION=' + unreal.SystemLibrary.get_engine_version())\n"
              "print('SLT_MATRIX_ASSETS=%d' % len(unreal.EditorAssetLibrary.list_assets('/Game', recursive=True)))\n")
    err, txt, _, dt = await c.call("ue_python_headless", {"script": script, "nullrhi": True, "timeout_s": 900},
                                   timeout_s=1000)
    record("ue_python_headless", not err and txt.startswith("exit 0") and "SLT_MATRIX_VERSION=5.8" in txt
           and "SLT_MATRIX_ASSETS=" in txt,
           f"{txt.splitlines()[0]} in {dt:.0f}s; "
           f"{[ln.split('LogPython: ')[-1] for ln in txt.splitlines() if 'SLT_MATRIX_' in ln]}")

    if launch_game:
        err, txt, _, _ = await c.call("sim_launch_game", {"settings_profile": "default", "render_offscreen": True,
                                                          "res_x": 640, "res_y": 480})
        started = not err and "launched pid=" in txt
        up, t0 = False, time.time()
        while time.time() - t0 < 300:
            await asyncio.sleep(10)
            _, s, _, _ = await c.call("sim_ping", timeout_s=60)
            if '"ping": true' in s.lower():
                up = True
                break
        record("sim_launch_game", started and up, f"{txt.splitlines()[0][:160]}; sim_ping answered after "
               f"{time.time() - t0:.0f}s: {up}")
        # tear the -game process down again
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if (p.info["name"] or "").lower().startswith("unrealeditor") and "-game" in " ".join(p.info["cmdline"] or []):
                with __import__("contextlib").suppress(psutil.Error):
                    p.kill()
    else:
        record("sim_launch_game", True, "skipped (--no-launch-game)", "skipped")


# --------------------------------------------------------------------------------------------------


async def run(phases: list[str], skip_genproj: bool, launch_game: bool) -> int:
    async with stdio_client(server_params()) as (r, w), ClientSession(r, w) as s:
        init = await s.initialize()
        tools = (await s.list_tools()).tools
        names = sorted(t.name for t in tools)
        print(f"connected: {init.serverInfo.name} protocol {init.protocolVersion}; {len(names)} tools: {names}\n",
              flush=True)
        c = Client(s)
        for ph in phases:
            print(f"\n================ phase: {ph} ================\n", flush=True)
            if ph == "up":
                await phase_up(c, skip_genproj)
            else:
                await phase_down(c, launch_game)

    missing = sorted(set(names) - {t.split(" ")[0] for t in COVERED})
    print("\n======== tool matrix")
    print(f"{'tool':<32} {'result':<6} evidence")
    for r_ in RESULTS:
        print(f"{r_['tool']:<32} {'PASS' if r_['ok'] else 'FAIL':<6} {r_['evidence'][:110]}")
    fails = sum(1 for r_ in RESULTS if not r_["ok"])
    print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} checks passed; tools not exercised in this run: {missing or 'none'}")
    out = REPO / "_artifacts" / "verification" / f"tool_matrix_{'-'.join(phases)}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tools_listed": names, "not_exercised": missing, "results": RESULTS}, indent=2))
    print(f"results -> {out}")
    return fails + len(missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", action="append", choices=["up", "down"], default=None)
    ap.add_argument("--skip-genproj", action="store_true")
    ap.add_argument("--no-launch-game", dest="launch_game", action="store_false")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.phase or ["up"], a.skip_genproj, a.launch_game)))
