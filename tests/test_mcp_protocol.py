"""MCP protocol-robustness suite for the `sightline` stdio server (tools/sightline_mcp/server.py).

Needs NO simulator and NO editor. Every protocol test spawns the server over the real stdio transport exactly as
.mcp.json does (same interpreter, script and env), with one safety override: SIGHTLINE_AIRSIM_PORT points at a
port nobody listens on (or at a local fake), so no test can ever send a command to a live AirSim instance. Tests
that would touch a live editor are skipped when an UnrealEditor process exists.

Run: D:\\Sightline\\.venv\\Scripts\\python.exe -m pytest tests -v
Report: docs/verification/mcp_protocol.md
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

import anyio
import jsonschema
import psutil
import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

REPO = Path(__file__).resolve().parents[1]
MCP_CFG = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["sightline"]
PYTHON = MCP_CFG["command"]
SERVER_ARGS = list(MCP_CFG["args"])
SERVER = Path(SERVER_ARGS[0])
UE_REMOTE = SERVER.parent / "ue_remote.py"
HARNESS = Path(__file__).with_name("_harness_server.py")
TMP = REPO / "_artifacts" / "tmp" / "mcp_protocol"
TMP.mkdir(parents=True, exist_ok=True)

# The 24 tools the server shipped with; later additions are allowed, removals/renames are not.
ORIGINAL_TOOLS = {
    "status", "editor_launch", "editor_wait_ready", "editor_close", "sim_launch_game", "ue_build",
    "ue_generate_project_files", "ue_package", "job_status", "job_list", "job_kill", "ue_python",
    "ue_python_headless", "ue_console", "ue_log", "sim_ping", "sim_state", "sim_fly", "sim_capture",
    "sim_environment", "sim_clock", "sim_objects", "sim_set_object_pose", "sim_detections",
}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


DEAD_PORT = _free_port()  # bound then released: nothing listens here


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def editor_running() -> bool:
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower().startswith("unrealeditor"):
            return True
    return False


def server_params(script_args: list[str] | None = None, airsim_port: int | None = DEAD_PORT,
                  extra_env: dict | None = None) -> StdioServerParameters:
    env = dict(MCP_CFG.get("env") or {})
    if airsim_port is not None:
        env["SIGHTLINE_AIRSIM_PORT"] = str(airsim_port)
    env.update(extra_env or {})
    return StdioServerParameters(command=PYTHON, args=script_args or SERVER_ARGS, cwd=str(REPO), env=env)


def harness_params(mode: str, **kw) -> StdioServerParameters:
    return server_params(script_args=[str(HARNESS), mode], **kw)


class Probe:
    """Collects everything abnormal the client transport receives (e.g. a non-JSON line on the server's stdout)."""

    def __init__(self):
        self.errors: list[Exception] = []

    async def handler(self, message) -> None:
        if isinstance(message, Exception):
            self.errors.append(message)


@contextlib.asynccontextmanager
async def session(params: StdioServerParameters | None = None, expect_clean_stdout: bool = True):
    params = params or server_params()
    errpath = TMP / f"server-{uuid.uuid4().hex[:8]}.stderr.log"
    probe = Probe()
    with open(errpath, "w", encoding="utf-8", errors="replace") as errlog:
        t0 = time.perf_counter()
        async with stdio_client(params, errlog=errlog) as (r, w):
            async with ClientSession(r, w, read_timeout_seconds=timedelta(seconds=180),
                                     message_handler=probe.handler) as s:
                with anyio.fail_after(20):
                    s.init_result = await s.initialize()
                s.startup_s = time.perf_counter() - t0
                s.probe = probe
                s.errpath = errpath
                yield s
    if expect_clean_stdout:
        assert not probe.errors, f"client received non-JSON-RPC data on the server's stdout: {probe.errors!r}"


async def call(s: ClientSession, name: str, args: dict | None = None, limit: float = 30.0):
    """call_tool with a hard wall-clock bound. Returns (result_or_McpError, seconds)."""
    t0 = time.perf_counter()
    with anyio.fail_after(limit):
        try:
            res = await s.call_tool(name, args or {})
        except McpError as e:
            res = e
    return res, time.perf_counter() - t0


def is_error(res) -> bool:
    return isinstance(res, McpError) or bool(res.isError)


def text(res) -> str:
    if isinstance(res, McpError):
        return f"McpError: {res.error.message}"
    return "\n".join(c.text for c in res.content if c.type == "text")


def server_processes() -> list[psutil.Process]:
    out = []
    for p in psutil.Process().children(recursive=True):
        with contextlib.suppress(psutil.Error):
            cmd = " ".join(p.cmdline())
            if "server.py" in cmd or "_harness_server.py" in cmd:
                out.append(p)
    return out


class WedgedEndpoint:
    """A TCP port that accepts connections but never answers: a frozen sim, a paused debugger, a sim still
    loading its map, or a foreign process owning the port."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.held: list[socket.socket] = []
        self._stop = False

    def _accept(self):
        self.sock.settimeout(0.2)
        while not self._stop:
            with contextlib.suppress(OSError):
                c, _ = self.sock.accept()
                self.held.append(c)

    def __enter__(self):
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        for c in self.held:
            c.close()
        self.sock.close()


class RawServer:
    """The server driven with hand-written JSON-RPC lines over plain pipes (no MCP client library, no job object):
    for malformed input and for proving the server exits by itself on stdin EOF."""

    def __init__(self, airsim_port: int = DEAD_PORT):
        env = os.environ.copy()
        env.update(MCP_CFG.get("env") or {})
        env["SIGHTLINE_AIRSIM_PORT"] = str(airsim_port)
        self.err = open(TMP / f"raw-{uuid.uuid4().hex[:8]}.stderr.log", "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen([PYTHON, *SERVER_ARGS], cwd=str(REPO), env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.err, creationflags=subprocess.CREATE_NO_WINDOW)
        self.q: queue.Queue = queue.Queue()
        self.raw_lines: list[str] = []
        self.responses: dict = {}
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for raw in self.p.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except ValueError:
                self.raw_lines.append(line)
        self.q.put(None)

    def send(self, msg) -> None:
        data = msg if isinstance(msg, bytes) else json.dumps(msg).encode("utf-8")
        self.p.stdin.write(data + b"\n")
        self.p.stdin.flush()

    def wait_for(self, rid, timeout: float = 15.0) -> dict:
        """Response with this id. Responses may arrive in any order (tool calls run concurrently), so others are
        kept for later wait_for calls."""
        deadline = time.monotonic() + timeout
        while rid not in self.responses:
            m = self.q.get(timeout=max(0.01, deadline - time.monotonic()))
            if m is None:
                raise EOFError("server closed stdout")
            if isinstance(m, dict) and "id" in m:
                self.responses[m["id"]] = m
        return self.responses.pop(rid)

    def initialize(self, version: str = "2025-11-25") -> dict:
        self.send({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                   "params": {"protocolVersion": version, "capabilities": {},
                              "clientInfo": {"name": "raw-protocol-test", "version": "0"}}})
        r = self.wait_for(0, 20)
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r

    def tree(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.p.pid)
            return [root, *root.children(recursive=True)]
        except psutil.Error:
            return []

    def kill(self):
        for p in self.tree():
            with contextlib.suppress(psutil.Error):
                p.kill()
        self.err.close()


def tool_defs_from_source() -> dict[str, ast.FunctionDef]:
    """Every function in server.py decorated with @mcp.tool(...) -> its AST (the source of truth for schemas)."""
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            if isinstance(target, ast.Attribute) and target.attr == "tool" and getattr(target.value, "id", "") == "mcp":
                name = node.name
                if isinstance(d, ast.Call):
                    for kw in d.keywords:
                        if kw.arg == "name":
                            name = ast.literal_eval(kw.value)
                out[name] = node
    return out


_SCALAR = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "None": "null",
           "dict": "object"}


def expected_json_types(annotation: str) -> set[str] | None:
    types_ = set()
    for part in (p.strip() for p in annotation.split("|")):
        if part in _SCALAR:
            types_.add(_SCALAR[part])
        elif part.startswith("list[") or part == "list":
            types_.add("array")
        else:
            return None  # Literal/Annotated/custom: not checked by this simple mapper
    return types_


def schema_json_types(prop: dict) -> set[str]:
    if "type" in prop:
        return {prop["type"]} if isinstance(prop["type"], str) else set(prop["type"])
    return {t for sub in prop.get("anyOf", []) for t in schema_json_types(sub)}


# --------------------------------------------------------------------------------------------------
# 1. handshake, startup, stdout hygiene
# --------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_handshake_startup_and_stderr_logging():
    async with session() as s:
        init = s.init_result
        assert s.startup_s < 10, f"server took {s.startup_s:.1f}s to answer initialize"
        assert init.protocolVersion == types.LATEST_PROTOCOL_VERSION
        assert init.serverInfo.name == "sightline"
        assert init.capabilities.tools is not None
        with anyio.fail_after(5):
            await s.send_ping()
        errpath = s.errpath
    log = errpath.read_text(encoding="utf-8", errors="replace")
    assert "Processing request of type" in log, "server logs must go to stderr"


def test_raw_protocol_version_negotiation_and_malformed_input():
    rs = RawServer()
    try:
        init = rs.initialize(version="2025-06-18")  # older clients must still be served
        assert init["result"]["protocolVersion"] == "2025-06-18"
        rs.send(b"this is not json at all")
        rs.send(b"[1, 2, 3]")
        rs.send(b"{" + b"x" * 1_000_000)  # 1 MB of broken JSON on one line
        rs.send({"jsonrpc": "2.0", "id": 10, "method": "no/such_method"})
        rs.send({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                 "params": {"name": "job_list", "arguments": "not-an-object"}})
        rs.send({"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "job_list"}})
        rs.send({"jsonrpc": "2.0", "id": 13, "method": "ping"})
        assert "error" in rs.wait_for(10), "unknown method must get a JSON-RPC error"
        assert "error" in rs.wait_for(11), "non-object arguments must get a JSON-RPC error"
        r12 = rs.wait_for(12)
        assert "result" in r12 and not r12["result"].get("isError"), r12
        assert rs.wait_for(13)["result"] == {}
        assert rs.p.poll() is None, "server died on malformed input"
        assert not rs.raw_lines, f"non-JSON on stdout: {rs.raw_lines[:3]}"
    finally:
        rs.kill()


@pytest.mark.anyio
async def test_stray_stdout_is_diverted_to_stderr():
    """Regression (main() stdout isolation): print(), a raw fd-1 write and a Win32 WriteFile(STD_OUTPUT_HANDLE)
    from inside a tool must land on stderr, never in the JSON-RPC stream."""
    async with session(harness_params("noise")) as s:
        for _ in range(3):
            res, _ = await call(s, "editor_wait_ready", {"timeout_s": 1})
            assert not is_error(res), text(res)
        assert len((await s.list_tools()).tools) >= len(ORIGINAL_TOOLS)
        errpath = s.errpath
    log = errpath.read_text(encoding="utf-8", errors="replace")
    for marker in ("STRAY-PRINT-MARKER", "STRAY-FD1-MARKER", "STRAY-WIN32-MARKER"):
        assert marker in log, f"{marker} should have been diverted to stderr"


@pytest.mark.anyio
async def test_stray_stdout_detector_negative_control():
    """Proves the Probe above really detects protocol corruption (garbage written before the protection)."""
    async with session(harness_params("garbage"), expect_clean_stdout=False) as s:
        res, _ = await call(s, "job_list")
        assert not is_error(res)
        probe = s.probe
    assert probe.errors, "the stray-output detector failed to notice a non-JSON line"


# --------------------------------------------------------------------------------------------------
# 2. tools/list and schemas
# --------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tools_list_schemas_match_python_signatures():
    source = tool_defs_from_source()
    async with session() as s:
        tools = {t.name: t for t in (await s.list_tools()).tools}
    assert set(tools) == set(source), f"listed vs @mcp.tool in source: {set(tools) ^ set(source)}"
    assert ORIGINAL_TOOLS <= set(tools), f"tools removed/renamed: {ORIGINAL_TOOLS - set(tools)}"
    problems = []
    for name, fn in source.items():
        t = tools[name]
        if not (t.description or "").strip():
            problems.append(f"{name}: empty description")
        try:
            jsonschema.Draft202012Validator.check_schema(t.inputSchema)
            if t.outputSchema:
                jsonschema.Draft202012Validator.check_schema(t.outputSchema)
        except jsonschema.SchemaError as e:
            problems.append(f"{name}: invalid JSON Schema: {e.message}")
            continue
        props = t.inputSchema.get("properties", {})
        args = fn.args.args + fn.args.kwonlyargs
        pos_defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
        defaults = dict(zip([a.arg for a in fn.args.args], pos_defaults, strict=True))
        defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)})
        if set(props) != {a.arg for a in args}:
            problems.append(f"{name}: schema params {sorted(props)} != signature {[a.arg for a in args]}")
            continue
        required = {a for a, d in defaults.items() if d is None}
        if set(t.inputSchema.get("required", [])) != required:
            problems.append(f"{name}: required {t.inputSchema.get('required')} != {sorted(required)}")
        for a in args:
            d = defaults[a.arg]
            if d is not None:
                want = ast.literal_eval(d)
                if props[a.arg].get("default", "<missing>") != want:
                    problems.append(f"{name}.{a.arg}: default {props[a.arg].get('default', '<missing>')!r} != {want!r}")
            if a.annotation is not None:
                exp = expected_json_types(ast.unparse(a.annotation))
                if exp is not None and schema_json_types(props[a.arg]) != exp:
                    problems.append(f"{name}.{a.arg}: types {schema_json_types(props[a.arg])} != {exp}")
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------------------------------
# 3. argument validation
# --------------------------------------------------------------------------------------------------

BAD_CALLS = [
    # (tool, arguments, substring the error must contain)
    ("sim_fly", {"action": "bogus"}, "bogus"),
    ("sim_clock", {"action": "bogus"}, "bogus"),
    ("ue_python", {"code": "print('noop')", "mode": "bogus"}, "bogus"),
    ("sim_capture", {"image_types": ["nope"]}, "nope"),
    ("sim_capture", {"image_types": "scene"}, "image_types"),
    ("sim_detections", {"image_type": "nope"}, "nope"),
    ("job_status", {}, "job_id"),
    ("job_status", {"job_id": "x", "tail": -1}, "tail"),
    ("ue_log", {"lines": "abc"}, "lines"),
    ("ue_log", {"lines": 0}, "lines"),
    ("sim_set_object_pose", {"name": "a", "x": "north", "y": 0, "z": 0}, "x"),
    ("editor_wait_ready", {"timeout_s": "soon"}, "timeout_s"),
    ("ue_console", {}, "command"),
    ("no_such_tool", {}, "no_such_tool"),
]


@pytest.mark.anyio
async def test_invalid_arguments_fail_fast_and_cleanly():
    async with session() as s:
        failures = []
        for name, args, needle in BAD_CALLS:
            res, dt = await call(s, name, args, limit=15)
            msg = text(res)
            if not is_error(res):
                failures.append(f"{name}{args}: accepted -> {msg[:200]!r}")
            elif needle not in msg:
                failures.append(f"{name}{args}: unclear error {msg[:200]!r}")
            elif "not listening" in msg:
                failures.append(f"{name}{args}: validated only after trying the sim: {msg[:200]!r}")
            elif dt > 5:
                failures.append(f"{name}{args}: took {dt:.1f}s")
        res, _ = await call(s, "status")  # still serving
        assert not is_error(res), text(res)
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------------------------------
# 4. endpoints down / wedged
# --------------------------------------------------------------------------------------------------

SIM_ARGS = {"sim_fly": {"action": "hover"}, "sim_clock": {"action": "status"},
            "sim_set_object_pose": {"name": "SightlineProtocolTestNoSuchActor", "x": 0, "y": 0, "z": 0}}
_SAMPLE = {"string": "SightlineProtocolTest", "number": 0, "integer": 0, "boolean": False, "array": [],
           "object": {}}


def _sample_args(tool: types.Tool) -> dict:
    if tool.name in SIM_ARGS:
        return SIM_ARGS[tool.name]
    props = tool.inputSchema.get("properties", {})
    return {p: _SAMPLE[sorted(schema_json_types(props[p]) - {"null"})[0]] for p in tool.inputSchema.get("required", [])}


@pytest.mark.anyio
async def test_every_sim_tool_errors_fast_when_airsim_is_down():
    async with session() as s:  # SIGHTLINE_AIRSIM_PORT -> DEAD_PORT
        tools = [t for t in (await s.list_tools()).tools
                 if t.name.startswith("sim_") and "launch" not in t.name]
        slow, unclear = [], []
        for t in tools:
            res, dt = await call(s, t.name, _sample_args(t), limit=20)
            if dt > 5:
                slow.append(f"{t.name}: {dt:.1f}s")
            if t.name in ORIGINAL_TOOLS and not (is_error(res) and str(DEAD_PORT) in text(res)):
                unclear.append(f"{t.name}: {text(res)[:160]!r}")
    assert len(tools) >= 10
    assert not slow, slow
    assert not unclear, unclear


@pytest.mark.anyio
async def test_sim_down_on_the_real_default_port():
    if port_open(41451):
        pytest.skip("AirSim RPC 41451 is open (another agent's sim is running): down-scenario not applicable")
    async with session(server_params(airsim_port=None)) as s:
        res, dt = await call(s, "sim_ping")
    assert is_error(res) and "41451" in text(res) and dt < 5, (dt, text(res))


@pytest.mark.anyio
async def test_wedged_airsim_fails_within_seconds_and_status_stays_live():
    """Regression (_client handshake timeout): a port that accepts but never answers used to block sim_ping for
    61 s and every queued sim call for another 60 s each (sim_state: 121 s)."""
    with WedgedEndpoint() as w:
        async with session(server_params(airsim_port=w.port)) as s:
            out = {}

            async def run(key, name):
                res, dt = await call(s, name, limit=150)
                out[key] = (res, dt, time.perf_counter())

            async with anyio.create_task_group() as tg:
                tg.start_soon(run, "ping", "sim_ping")
                await anyio.sleep(0.3)
                tg.start_soon(run, "state", "sim_state")
                tg.start_soon(run, "status", "status")
    ping, state, status_ = out["ping"], out["state"], out["status"]
    assert is_error(ping[0]) and "did not answer" in text(ping[0]), text(ping[0])
    assert ping[1] < 12, f"sim_ping took {ping[1]:.1f}s against a wedged endpoint"
    assert is_error(state[0]) and state[1] < 20, f"queued sim_state took {state[1]:.1f}s"
    assert not is_error(status_[0]) and status_[2] < ping[2], "status must not wait behind the sim thread"


@pytest.mark.anyio
async def test_editor_tools_fail_fast_without_editor():
    if editor_running() or port_open(8000):
        pytest.skip("an Unreal Editor is running (another agent): no-editor scenario not applicable")
    async with session() as s:
        res, dt = await call(s, "ue_python", {"code": "print('sightline protocol test noop')"})
        assert is_error(res) and "No Unreal Editor is running" in text(res) and dt < 5, (dt, text(res))
        res, dt = await call(s, "ue_console", {"command": "sightline_protocol_test_noop"})
        assert is_error(res) and "No Unreal Editor is running" in text(res) and dt < 5, (dt, text(res))
        res, dt = await call(s, "editor_wait_ready", {"timeout_s": 5})
        assert dt < 7 and "not running" in text(res), (dt, text(res))
        missing = TMP / "definitely-missing.log"
        res, dt = await call(s, "ue_log", {"log_file": str(missing)})
        assert "log not found" in text(res) and dt < 3
        res, dt = await call(s, "job_status", {"job_id": "no-such-job"})
        assert is_error(res) and "unknown job" in text(res) and dt < 3
        res, dt = await call(s, "job_kill", {"job_id": "no-such-job"})
        assert "not running" in text(res) and dt < 3


# --------------------------------------------------------------------------------------------------
# 5. concurrency  /  6. cancellation
# --------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_concurrent_calls_complete_and_are_not_crosswired():
    logs = {}
    for i in range(3):
        marker = f"MARKER-{i}-{uuid.uuid4().hex}"
        p = TMP / f"concurrent-{i}.log"
        p.write_text("\n".join(f"line {n} {marker}" for n in range(2000)), encoding="utf-8")
        logs[marker] = p
    calls = [("status", {}), ("status", {}), ("job_list", {}), ("job_list", {}), ("sim_ping", {}), ("sim_ping", {})]
    calls += [("ue_log", {"log_file": str(p), "lines": 5}) for p in logs.values()]
    if not editor_running():
        calls.append(("editor_wait_ready", {"timeout_s": 8}))
    else:
        calls.append(("job_list", {}))
    results: dict[int, tuple] = {}
    async with session() as s:
        t0 = time.perf_counter()
        async with anyio.create_task_group() as tg:
            for i, (name, args) in enumerate(calls):
                async def one(i=i, name=name, args=args):
                    results[i] = await call(s, name, args, limit=60)
                tg.start_soon(one)
        wall = time.perf_counter() - t0
        res, _ = await call(s, "job_list")
        assert not is_error(res)
    assert len(results) == len(calls) == 10
    assert wall < 30, f"10 concurrent calls took {wall:.1f}s"
    for i, (name, args) in enumerate(calls):
        res, _ = results[i]
        body = text(res)
        if name == "status":
            assert "endpoints" in json.loads(body)
        elif name == "job_list":
            assert "session_jobs" in json.loads(body)
        elif name == "sim_ping":
            assert is_error(res) and str(DEAD_PORT) in body
        elif name == "ue_log":
            mine = [m for m, p in logs.items() if str(p) == args["log_file"]][0]
            assert mine in body and all(m not in body for m in logs if m != mine), "ue_log results cross-wired"


@pytest.mark.anyio
async def test_slow_editor_call_does_not_block_status():
    if editor_running():
        pytest.skip("an Unreal Editor is running: editor_wait_ready would talk to it")
    async with session() as s:
        out = {}

        async def run(key, name, args):
            res, dt = await call(s, name, args, limit=60)
            out[key] = (res, dt)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run, "wait", "editor_wait_ready", {"timeout_s": 8})
            await anyio.sleep(0.2)
            tg.start_soon(run, "status", "status", {})
    assert not is_error(out["status"][0]) and out["status"][1] < 8
    assert out["wait"][1] < 12


@pytest.mark.anyio
async def test_fake_editor_long_command_lock_retry_semantics():
    """Regression (_editor_run / status locking): with a (fake) editor attached, a long ue_python used to hold
    _remote_lock and block status(); a timed-out command must not be re-sent; a dropped channel reconnects once."""
    async with session(harness_params("fake_editor")) as s:
        out = {}

        async def run(key, name, args):
            res, dt = await call(s, name, args, limit=60)
            out[key] = (res, dt, time.perf_counter())

        async with anyio.create_task_group() as tg:
            tg.start_soon(run, "py", "ue_python", {"code": "SLEEP:8"})
            await anyio.sleep(0.5)
            tg.start_soon(run, "status", "status", {})
        assert not is_error(out["py"][0]) and "attempt#1" in text(out["py"][0])
        assert not is_error(out["status"][0]) and out["status"][1] < 6
        assert out["status"][2] < out["py"][2], "status() waited for the running editor command"
        assert "fake-node" in text(out["status"][0])

        res, dt = await call(s, "ue_python", {"code": "TIMEOUT", "timeout_s": 2})
        assert is_error(res) and "not re-sent" in text(res) and dt < 4, (dt, text(res))
        res, _ = await call(s, "ue_python", {"code": "SLEEP:0"})
        assert "attempt#3" in text(res), f"the timed-out command was re-sent: {text(res)}"

        res, _ = await call(s, "ue_python", {"code": "DROP_ONCE"})
        assert not is_error(res) and "reconnected attempt#5" in text(res), text(res)


@pytest.mark.anyio
async def test_editor_wait_ready_honours_its_timeout():
    """Regression (editor_wait_ready deadline): the readiness probe used a fixed 20 s command timeout plus a 5 s
    sleep, so editor_wait_ready(timeout_s=3) against an editor that never answers returned after ~25 s."""
    async with session(harness_params("fake_editor")) as s:
        res, dt = await call(s, "editor_wait_ready", {"timeout_s": 3}, limit=60)
    assert "timed out after 3s" in text(res) and dt < 6, (dt, text(res))


@pytest.mark.anyio
async def test_cancellation_releases_request_and_server_stays_healthy():
    with WedgedEndpoint() as w:
        async with session(server_params(airsim_port=w.port)) as s:
            outcome = {}

            async def slow():
                try:
                    await s.call_tool("sim_ping", {})
                    outcome["how"] = "completed"
                except McpError as e:
                    outcome["how"] = f"McpError: {e.error.message}"
                outcome["t"] = time.perf_counter()

            async with anyio.create_task_group() as tg:
                rid = s._request_id  # the id the next request will use
                tg.start_soon(slow)
                await anyio.sleep(1.0)
                t_cancel = time.perf_counter()
                await s.send_notification(types.ClientNotification(types.CancelledNotification(
                    params=types.CancelledNotificationParams(requestId=rid, reason="protocol test"))))
            assert "cancel" in outcome["how"].lower(), outcome
            assert outcome["t"] - t_cancel < 2.0, "cancelled request was not released promptly"
            # cancelling an unknown / finished id is harmless
            await s.send_notification(types.ClientNotification(types.CancelledNotification(
                params=types.CancelledNotificationParams(requestId=987654, reason="unknown id"))))
            res, dt = await call(s, "status")
            assert not is_error(res) and dt < 8
            res, _ = await call(s, "sim_ping", limit=40)  # the sim queue drains; the tool still answers
            assert is_error(res) and "did not answer" in text(res)


# --------------------------------------------------------------------------------------------------
# 7. large payloads
# --------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_multi_megabyte_result_and_request_are_delivered_intact():
    lines = [f"{i:07d} LogSightline: payload héllo → 中文 ✓ 🚁 " + "x" * 70 for i in range(55_000)]
    body = "\n".join(lines)
    big = TMP / "large_synthetic.log"
    big.write_text(body, encoding="utf-8", newline="\n")
    assert big.stat().st_size > 5_000_000
    async with session() as s:
        res, dt = await call(s, "ue_log", {"log_file": str(big), "lines": 1_000_000}, limit=60)
        assert not is_error(res)
        got = text(res).split("\n", 1)[1]
        assert hashlib.sha256(got.encode()).hexdigest() == hashlib.sha256(body.encode()).hexdigest()
        res, dt = await call(s, "sim_objects", {"name_regex": "a" * 3_000_000}, limit=30)  # 3 MB request
        assert is_error(res) and str(DEAD_PORT) in text(res)
        res, _ = await call(s, "job_list")
        assert not is_error(res)


# --------------------------------------------------------------------------------------------------
# 8. lifecycle
# --------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_disconnect_leaves_no_orphaned_server_process():
    before = {p.pid for p in server_processes()}
    async with session() as s:
        await call(s, "status")
        mine = [p for p in server_processes() if p.pid not in before]
        assert mine, "could not find the spawned server process"
    gone, alive = psutil.wait_procs(mine, timeout=6)
    assert not alive, f"orphaned server processes: {[p.pid for p in alive]}"


def test_server_exits_on_stdin_eof_even_with_calls_in_flight():
    """Regression (threaded abandon_on_cancel + main() os._exit): without the MCP client's job-object kill, a closed
    stdin must end the server promptly even while tool threads are blocked (here: wedged sim RPC calls)."""
    with WedgedEndpoint() as w:
        rs = RawServer(airsim_port=w.port)
        try:
            rs.initialize()
            for i in range(3):
                rs.send({"jsonrpc": "2.0", "id": 100 + i, "method": "tools/call",
                         "params": {"name": "sim_ping", "arguments": {}}})
            time.sleep(1.0)
            procs = rs.tree()
            assert procs
            t0 = time.perf_counter()
            rs.p.stdin.close()
            gone, alive = psutil.wait_procs(procs, timeout=10)
            elapsed = time.perf_counter() - t0
            assert not alive, f"server still alive {elapsed:.1f}s after stdin EOF"
            assert elapsed < 5, f"server needed {elapsed:.1f}s to exit after stdin EOF"
        finally:
            rs.kill()


@pytest.mark.anyio
async def test_restart_and_two_simultaneous_instances():
    marker = f"TWO-INSTANCES-{uuid.uuid4().hex}"
    log = TMP / "two_instances.log"
    log.write_text(marker, encoding="utf-8")
    async with session() as a, session() as b:
        results = {}

        async def work(tag, s):
            for name, args in (("status", {}), ("job_list", {}), ("ue_log", {"log_file": str(log)})):
                res, _ = await call(s, name, args)
                results[(tag, name)] = res

        async with anyio.create_task_group() as tg:
            tg.start_soon(work, "a", a)
            tg.start_soon(work, "b", b)
        assert all(not is_error(r) for r in results.values())
        assert marker in text(results[("a", "ue_log")]) and marker in text(results[("b", "ue_log")])
    async with session() as c:  # restart after both closed
        res, _ = await call(c, "job_list")
        assert not is_error(res)


def test_remote_exec_command_listener_is_never_shared():
    """Two server instances each open their own remote-execution command listener. Formerly both bound the fixed
    127.0.0.1:6776 with SO_REUSEADDR (allowed twice on Windows), so an editor could connect to the wrong server."""
    sys.path.insert(0, str(UE_REMOTE.parent))
    import ue_remote

    a, b = ue_remote.UnrealRemote(), ue_remote.UnrealRemote()
    (la, pa), (lb, pb) = a.open_listener(), b.open_listener()
    try:
        assert pa and pb and pa != pb and 6776 not in (pa, pb)
    finally:
        la.close()
        lb.close()
    pinned = _free_port()
    lx, _ = ue_remote.UnrealRemote(command_port=pinned).open_listener()
    try:
        with pytest.raises(OSError):
            ue_remote.UnrealRemote(command_port=pinned).open_listener()
    finally:
        lx.close()
    assert "SIGHTLINE_UE_REMOTE_CMD_PORT" in SERVER.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# 9. static guards (Windows stdio deadlock, stdout hygiene, entry point)
# --------------------------------------------------------------------------------------------------

NATIVE = {"numpy", "cv2", "torch", "PIL", "cosysairsim"}


def _root(mod: str | None) -> str:
    return (mod or "").split(".")[0]


def test_native_modules_are_imported_at_startup_only():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {_root(a.name) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add(_root(node.module))
    assert {"numpy", "PIL", "cosysairsim"} <= top, f"missing startup imports: {top}"
    offenders = []
    for path in (SERVER, UE_REMOTE):
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            for n in ast.walk(fn):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    mods = [n.module]
                elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
                    f = n.func
                    fname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                    if fname in ("__import__", "import_module"):
                        mods = [str(n.args[0].value)]
                bad = [m for m in mods if _root(m) in NATIVE]
                if bad:
                    offenders.append(f"{path.name}:{n.lineno} lazy import of {bad} inside {getattr(fn, 'name', 'lambda')}")
    assert not offenders, "\n".join(offenders)


def test_no_bare_print_and_protected_entry_point():
    offenders = []
    for path in (SERVER, UE_REMOTE):
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" \
                    and not any(k.arg == "file" for k in n.keywords):
                offenders.append(f"{path.name}:{n.lineno}")
    assert not offenders, f"print() to stdout in a stdio MCP server: {offenders}"
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    runs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run" and getattr(n.func.value, "id", "") == "mcp"]
    assert not runs, "mcp.run() bypasses main()'s stdout isolation"
    mains = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert mains and "main()" in ast.unparse(mains[-1]), "__main__ must call main()"
