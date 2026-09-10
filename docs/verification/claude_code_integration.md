# Claude Code CLI <-> project MCP servers: client-side verification

Date: 2026-09-10. Claude Code 2.1.260 (`C:\Users\kiran\.local\bin\claude.exe`, not on PATH), Windows 11,
PowerShell, run from `D:\Sightline`. Scope: can the Claude Code *client* load, approve, discover and call the
two servers in `.mcp.json` (`unreal` = Epic HTTP server in the editor, `sightline` = our stdio server).
Server-side unit and smoke tests are covered elsewhere.

## Summary

| # | Check | Result |
|---|---|---|
| 1 | CLI help: MCP commands and flags read from `--help` | PASS |
| 2 | `.mcp.json` servers recognised (`claude mcp list` / `get`) | PASS (both listed, scope "Project config") |
| 2b | Approval for non-interactive use | PASS: `-p` connects them without approval (see below). `mcp list` still shows "Pending approval" |
| 3a | `sightline` connects and its 24 tools are discovered in a `-p` session | PASS (connected in 747-853 ms, 24 `mcp__sightline__*` tools in the init event) |
| 3b | Model-driven tool calls (`status`, `job_list`, `sim_ping`) via `claude -p` | **BLOCKED**: the CLI is not logged in (see Open issues) |
| 4a | `unreal` reported as failed while the editor is down, `sightline` unaffected | PASS |
| 4b | `unreal` connects and is discovered by Claude Code once the editor is up | PASS: connected over HTTP in 736 ms, protocol 2025-11-25; tools `mcp__unreal__list_toolsets`, `mcp__unreal__describe_toolset`, `mcp__unreal__call_tool` (editor was started by another agent at 18:52) |
| 4c | Model-driven `list_toolsets` / `call_tool(IsPIERunning)` via `claude -p` | **BLOCKED**: same login issue as 3b |
| 5 | Timeout limits found; project-level values set and confirmed to take effect | PASS (`MCP_TIMEOUT` confirmed live in the debug log) |

## 1. CLI surface (from `claude --help`, `claude mcp --help`)

- `claude mcp list|get <name>|add|add-json|remove|reset-project-choices|login|logout|serve`. Help text: "Unapproved
  .mcp.json servers are shown as ⏸ Pending approval and not connected to; approved servers are health-checked".
- Print mode: `-p`, `--output-format text|json|stream-json` (stream-json needs `--verbose`), `--max-turns`,
  `--max-budget-usd`, `--allowedTools` / `--disallowedTools`, `--tools` (built-in tools; `""` disables them),
  `--mcp-config`, `--strict-mcp-config`, `--settings`, `--setting-sources`, `--permission-mode`,
  `--no-session-persistence`, `--debug-file <path>`.
- `-p` help: "The workspace trust dialog is skipped when Claude is run in non-interactive mode".
- MCP tool names are `mcp__<server>__<tool>`. Allow-list patterns must be anchored to a server:
  `mcp__sightline__status` or `mcp__sightline__*` work, but a bare `mcp__*` is skipped with a warning
  (docs: code.claude.com/docs/en/permissions).

## 2. Recognition and approval

```
PS D:\Sightline> claude mcp list
unity-editor-mcp: C:\Users\kiran\AppData\Local\Unity\bin\unity.exe mcp - √ Connected   <- user-level, not ours
unreal: http://localhost:8000/mcp (HTTP) - ⏸ Pending approval (run `claude` to approve)
sightline: D:/Sightline/.venv/Scripts/python.exe D:/Sightline/tools/sightline_mcp/server.py - ⏸ Pending approval (run `claude` to approve)

PS D:\Sightline> claude mcp get sightline
  Scope: Project config (shared via .mcp.json)   Status: ⏸ Pending approval   Type: stdio
  Command/Args/Env: exactly as in .mcp.json (SIGHTLINE_UE_ROOT, SIGHTLINE_UPROJECT)
```

What I tried, and what happened:
- Added `"enabledMcpjsonServers": ["unreal", "sightline"]` to `D:\Sightline\.claude\settings.json` (new file).
  The docs list this key (along with `enableAllProjectMcpServers` / `disabledMcpjsonServers`) and say it is honoured in a
  *trusted* folder. The trust flag for `D:\Sightline` is already `true` in the user config (read, not modified).
  `claude mcp list` **still showed "Pending approval"**, both from this shell and from a clean environment.
  A temporary `settings.local.json` with the same key made no difference either, so I deleted it.
- A `--debug-file` run shows that `-p` sessions do load `D:\Sightline\.claude\settings.json` and
  `settings.local.json`, and they connect both `.mcp.json` servers **regardless of the approval state**. That
  holds both before and after the settings change (init event:
  `[{"name":"unreal","status":"failed"},{"name":"sightline","status":"connected"}]`).
- Conclusion: **no approval step is needed for `claude -p`**. For interactive `claude`, the one-time
  "new MCP servers found in .mcp.json" prompt (or `/mcp`) records the approval in the user's `~/.claude.json`.
  The user has to do that, because I did not touch user-level config. The project-level key is kept because
  it is the documented mechanism and it does no harm. Whether it suppresses the interactive prompt could not
  be tested without a TTY.

## 3. Discovery and calls through `claude -p`

Command used (the model call stops at the auth error, but the init event and debug log are produced before it):
```
cd D:\Sightline
claude -p "Call the tool mcp__sightline__status once and reply with only its raw output." `
  --output-format stream-json --verbose --allowedTools "mcp__sightline__status" `
  --max-turns 3 --no-session-persistence --model sonnet > pre_approval.jsonl
```
- The init event lists `mcp_servers` sightline=connected, unreal=failed, plus all 24 sightline tools:
  editor_close, editor_launch, editor_wait_ready, job_kill, job_list, job_status, sim_capture, sim_clock,
  sim_detections, sim_environment, sim_fly, sim_launch_game, sim_objects, sim_ping, sim_set_object_pose,
  sim_state, status, ue_build, ue_console, ue_generate_project_files, ue_log, ue_package, ue_python,
  ue_python_headless (each as `mcp__sightline__<name>`).
- Debug log: `MCP server "sightline": Successfully connected (transport: stdio) in 747ms`,
  `negotiatedProtocolVersion":"2025-11-25`, capabilities tools+prompts+resources.
- Result event: `"is_error":true, "terminal_reason":"api_error", "result":"Failed to authenticate: OAuth session
  expired and could not be refreshed"`. `claude auth status` gives `{"loggedIn": false, "authMethod": "none"}`, also
  in a clean environment with the inherited desktop-host variables removed. So **no tool call was made**, and
  `status` / `job_list` / `sim_ping` through Claude Code are **unverified**.

To finish check 3b after logging in, run these (read-only tools only, cost-capped):
```
claude -p "Call mcp__sightline__status, then mcp__sightline__job_list, then mcp__sightline__sim_ping. Reply DONE." `
  --output-format stream-json --verbose `
  --allowedTools "mcp__sightline__status,mcp__sightline__job_list,mcp__sightline__sim_ping" `
  --max-turns 6 --max-budget-usd 0.50 --no-session-persistence > calls.jsonl
# verify: calls.jsonl has "type":"user" tool_result blocks whose content contains "project_exists": true and
# "ram_total_gb" (status), "session_jobs" (job_list), and the sim_ping text (AirSim up or down).
Select-String calls.jsonl -Pattern 'project_exists|ram_total_gb|session_jobs|tool_use_id'
```

## 4. `unreal` server

Editor down (verified):
```
claude mcp list   -> unreal: http://localhost:8000/mcp (HTTP) - ⏸ Pending approval   (list does not probe unapproved servers)
-p init event     -> {"name":"unreal","status":"failed"}
debug log         -> MCP server "unreal": HTTP Connection failed after 65ms: Unable to connect ... (code: ConnectionRefused)
                     [MCP] Retry: 1 transiently-failed remote server(s) after 500ms backoff  -> fails again in 3-4 ms
```
The failure is fast (under 100 ms, one retry) and **does not affect `sightline`**, which connected in the same
session (PASS).

Editor up (from 18:53, started by another agent): Claude Code connected and discovered the three `unreal` tools.
See the addendum at the end of this file. To finish the model-driven check after logging in, run:
```
claude -p "Call the unreal MCP tool list_toolsets, then call_tool for IsPIERunning (read-only). Reply DONE." `
  --output-format stream-json --verbose --allowedTools "mcp__unreal__*" `
  --max-turns 6 --max-budget-usd 0.50 --no-session-persistence > unreal.jsonl
```
Confirm the exact tool names from the init event's `tools` array (expect `mcp__unreal__list_toolsets`,
`mcp__unreal__describe_toolset`, `mcp__unreal__call_tool`). Discovery alone, without the model, can be checked
with `claude -p noop --tools "" --output-format stream-json --verbose --debug-file dbg.log`.

## 5. Timeouts (read from docs and from the 2.1.260 binary)

| Limit | Variable / field | Default | Applies to |
|---|---|---|---|
| Server startup | `MCP_TIMEOUT` (ms) | 30000 | connect + initialize, per server |
| Tool call hard limit | `MCP_TOOL_TIMEOUT` (ms) or per-server `"timeout"` in `.mcp.json` | ~1e8 ms (~28 h) per docs | wall clock; progress does **not** extend it |
| Tool idle limit | `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` (ms, 0 disables) | **stdio 1800000 (30 min), http 300000 (5 min)** | aborts if the server sends no response **or progress notification** for that long |
| HTTP request | derived | max(tool timeout, 60000, MCP_TIMEOUT) | `unreal` |

The binary computes the idle window as `min(max(IDLE_ENV ?? default, server.timeout ?? 0, 1000), toolTimeout)`.
On abort the message is: "sent no response or progress for Ns; aborting. ... set a per-server "timeout"".

**Risk:** the sightline tools run in a worker thread (`@threaded`) and send **no progress notifications**.
With the defaults, `editor_wait_ready(timeout_s=2400)`, `editor_launch(wait_ready_s>1800)` and
`ue_python_headless(timeout_s=1800)` would be aborted by Claude Code at 30 minutes. `ue_build` / `ue_package`
are background jobs polled with `job_status`, so they are not at risk. `ue_generate_project_files` (max 900 s)
is fine too. Epic's `unreal` server streams progress only when a progress token is sent, so a silent call
longer than 5 minutes (for example, long automation tests) would hit the HTTP idle limit.

**Set (project level, `D:\Sightline\.claude\settings.json` `env`):**
```json
"env": {
  "MCP_TIMEOUT": "60000",                        // cold venv/psutil import headroom (measured 0.75-0.85 s)
  "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "3600000", // 1 h: covers editor_wait_ready(2400) and headless 1800 s
  "MCP_TOOL_TIMEOUT": "7200000"                  // 2 h hard cap (the idle window is clamped to this, so it must be >= idle)
}
```
Verified live: after the change, the debug log shows `Starting connection with timeout of 60000ms` for every
server (it was 30000 ms before), so project `env` does reach the MCP layer. An alternative that affects only
sightline would be `"timeout": 3600000` on the `sightline` entry in `.mcp.json`, which raises that server's idle
window. I did not apply it because `.mcp.json` is shared config. A longer-term fix is for long sightline tools
to send MCP progress notifications (`ctx.report_progress`) while they poll.

## Files changed

- **Created** `D:\Sightline\.claude\settings.json`: `enabledMcpjsonServers: ["unreal","sightline"]` (documented
  approval key; no measurable effect on `mcp list` here) plus the three timeout `env` values above.
- Created and then **deleted** `D:\Sightline\.claude\settings.local.json` (same key, no effect).
- Not touched: `.mcp.json`, anything under `C:\Users\kiran\.claude*` (only read the trust flag), TRACKER, CONTEXT.

## How to (re)connect after an editor restart

- Epic's server forgets sessions on restart (unknown `Mcp-Session-Id` -> 404). Claude Code connects to `unreal`
  **once at session start**. If the editor was down then, or has restarted since, the server stays failed or
  stale.
- Interactive session: type `/mcp`, select `unreal`, then Reconnect (or `/mcp reconnect unreal`). Otherwise quit
  and start `claude` again.
- `claude -p`: each invocation is a new session and reconnects by itself. Start it after `status` shows
  `unreal_mcp_http:8000: true`.
- Check order: `mcp__sightline__status` (port 8000 up?) -> `/mcp` reconnect -> `mcp__unreal__list_toolsets`.

## Open issues

1. **CLI login expired**. `claude auth status` gives `loggedIn:false` and the debug log says "OAuth refresh token is
   no longer valid; run /login". The user must run `claude auth login` (or `/login` inside `claude`). Signing in
   is not something the agent may do. After that, run the commands in §3 and §4 to close checks 3b and 4b.
2. `claude mcp list` keeps showing "Pending approval" despite the project key. Approve once interactively
   (startup prompt or `/mcp`). This is display-only for `-p`, which connects anyway.
3. The `unity-editor-mcp` server also loads in every Sightline session. It is not ours, and it stays connected even
   with `--strict-mcp-config --mcp-config D:\Sightline\.mcp.json`, so it is injected from outside the MCP config
   files. It is harmless, and with `--tools ""` the only MCP tools in that session were the 24 sightline ones.
   Disable it per project via `/mcp` if unwanted.
   (With `--allowedTools "mcp__sightline__*"` the debug log showed no unanchored-pattern warning, so the pattern
   is accepted.)
4. The sightline long-blocking tools should send progress notifications (see §5) so they don't rely on a raised
   idle timeout.
5. `D:\Sightline` is not a git repo yet. When it becomes one, commit `.claude/settings.json` and gitignore
   `.claude/settings.local.json`.

## Addendum: port 8000 polling and `unreal` with the editor up

- I polled `Test-NetConnection localhost -Port 8000` about every 60 s from 18:28 local. It was down until **18:53**; the
  editor process (started by another agent) began at 18:52:25.
- **Port open is not the same as server ready.** From 18:53 to about 18:57, while the editor was still loading and its
  process showed as not responding, a raw `initialize` POST timed out after 20 s. Claude Code logged
  `version negotiation probe timed out on the http transport; reconnecting pinned legacy within the remaining
  budget`. At 18:57:20 the raw probe returned HTTP 200 (`protocolVersion 2025-06-18`, `tools.listChanged:true`,
  `Mcp-Session-Id` set).
- Claude Code with the server ready (session kept open with `--input-format stream-json`):
  ```
  MCP server "unreal": Successfully connected (transport: http) in 736ms
  MCP server "unreal": Connection established with capabilities: {"hasTools":true,"hasResources":true,
      "negotiatedProtocolVersion":"2025-11-25", ...}
  init event: mcp_servers = unity-editor-mcp connected, unreal connected, sightline connected
  unreal tools: mcp__unreal__call_tool, mcp__unreal__describe_toolset, mcp__unreal__list_toolsets
  ```
  Command:
  `(sleep 6; echo '{"type":"user","message":{"role":"user","content":"noop"}}'; sleep 10) | claude -p
  --input-format stream-json --output-format stream-json --verbose --tools "" --max-turns 1 --no-session-persistence`
  (bash). The model turn then failed on auth, as expected. No `call_tool` was made.
- Timing caveat: HTTP servers connect **non-blocking**. A `-p` prompt given on the command line starts before the
  HTTP handshake finishes, so the init event showed `"unreal","status":"pending"` and the session had no `unreal`
  tools. When scripting `unreal` calls with `-p`, check first that `status` reports port 8000 and that a raw
  `initialize` answers. If `unreal` tools are missing, retry. In interactive sessions, use `/mcp` to see when it
  turns connected.
- The `unreal` HTTP transport logged `timeoutMs: 7200000`, so the HTTP request timeout follows `MCP_TOOL_TIMEOUT`
  from the project `env`.
- The sightline server now exposes **27** tools. Another agent added `sim_destroy_object`, `sim_list_assets` and
  `sim_spawn_object` during this session. All 27 appear in the init event, so rediscovery works.
