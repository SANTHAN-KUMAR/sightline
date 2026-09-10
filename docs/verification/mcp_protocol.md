# `sightline` MCP server: protocol-robustness verification (2026-09-10)

Suite: `tests/test_mcp_protocol.py` (+ fault-injection harness `tests/_harness_server.py`).
Run: `D:\Sightline\.venv\Scripts\python.exe -m pytest tests/test_mcp_protocol.py -v`, which takes about 80-100 s and needs
no editor or simulator. (`pytest tests` also collects another agent's `tests/test_stack.py`. Its failures
(torchcodec/FFmpeg, TensorRT, fastapi) are unrelated to MCP.)

Every protocol test spawns `server.py` over the real stdio transport exactly as `.mcp.json` does (same interpreter,
args and env), with one safety override: `SIGHTLINE_AIRSIM_PORT` points at a port nobody listens on, or at a local
"wedged" fake. As a result, no test can reach a live AirSim. Tests that would talk to a live editor skip themselves
when an `UnrealEditor` process exists. The fake-editor tests replace `UnrealRemote` in-process and never touch the
network.

## Results

Run 1: editor closed, 41451 and 8000 closed. All 21 tests that existed then passed.
Final run: another agent's editor was up (pid 23796, :8000 open). 20 passed and 2 skipped, both no-editor scenarios.

| # | Area | Test | Result | Measured |
|---|---|---|---|---|
| 1 | Handshake | `test_handshake_startup_and_stderr_logging` | pass | initialize < 1 s (limit 10 s); protocol `2025-11-25`; serverInfo `sightline`; logs on stderr |
| 1 | Old protocol + malformed input (raw pipes) | `test_raw_protocol_version_negotiation_and_malformed_input` | pass | `2025-06-18` negotiated; garbage, JSON array, 1 MB broken line, unknown method and non-object arguments all handled; ping still answered; stdout 100 % JSON |
| 1 | Stray stdout | `test_stray_stdout_is_diverted_to_stderr` | pass | `print()`, `os.write(1)`, `WriteFile(GetStdHandle(STD_OUTPUT))` all land on stderr; 0 client parse errors |
| 1 | Detector sanity | `test_stray_stdout_detector_negative_control` | pass | garbage written before protection is detected |
| 2 | tools/list | `test_tools_list_schemas_match_python_signatures` | pass | listed tools == `@mcp.tool` functions in source (now 24 original + main agent's additions); Draft 2020-12 metaschema; params, required, defaults and JSON types == Python signatures; descriptions non-empty |
| 3 | Bad arguments (14 cases) | `test_invalid_arguments_fail_fast_and_cleanly` | pass | every case isError with a clear message, each < 5 s, validated before contacting the sim; server keeps serving |
| 4 | Sim down (all sim_* tools, auto-discovered) | `test_every_sim_tool_errors_fast_when_airsim_is_down` | pass | each < 1 s, message names host:port |
| 4 | Sim down, real default port | `test_sim_down_on_the_real_default_port` | pass | skips itself if 41451 is open (it skipped in one intermediate run when Blocks was up) |
| 4 | Wedged sim (accepts, never answers) | `test_wedged_airsim_fails_within_seconds_and_status_stays_live` | pass | sim_ping ~6 s, queued sim_state ~12 s (were 61 s / 121 s); status unaffected |
| 4 | Editor absent | `test_editor_tools_fail_fast_without_editor` | pass (run 1) / skip | ue_python/ue_console < 1 s (were 10.1 s); editor_wait_ready(5) < 1 s; ue_log missing file, job_status/job_kill unknown id all clean |
| 5 | 10 concurrent calls | `test_concurrent_calls_complete_and_are_not_crosswired` | pass | all complete (~5 s wall); ue_log markers never cross-wired |
| 5 | Slow call vs status | `test_slow_editor_call_does_not_block_status` | pass (run 1) / skip | |
| 5 | Long editor command vs status; timeout not re-sent; reconnect once | `test_fake_editor_long_command_lock_retry_semantics` | pass | status 2-3 s while an 8 s command runs |
| 5 | editor_wait_ready deadline | `test_editor_wait_ready_honours_its_timeout` | pass | timeout_s=3 returns in ~3 s (was up to ~25 s) |
| 6 | Cancellation | `test_cancellation_releases_request_and_server_stays_healthy` | pass | `notifications/cancelled` answered < 2 s; unknown id harmless; next calls fine |
| 7 | Large payloads | `test_multi_megabyte_result_and_request_are_delivered_intact` | pass | 6.3 MB UTF-8 (non-ASCII) result sha256-identical; 3 MB request handled |
| 8 | Disconnect | `test_disconnect_leaves_no_orphaned_server_process` | pass | launcher + interpreter gone |
| 8 | stdin EOF with blocked tool threads (raw pipes, no job object) | `test_server_exits_on_stdin_eof_even_with_calls_in_flight` | pass | exit < 1 s with 3 wedged sim calls in flight |
| 8 | Restart + two simultaneous instances | `test_restart_and_two_simultaneous_instances` | pass | |
| 8 | Remote-exec command port | `test_remote_exec_command_listener_is_never_shared` | pass | two instances get distinct ephemeral ports; a pinned port is exclusive |
| 9 | Startup import guard (AST) | `test_native_modules_are_imported_at_startup_only` | pass | numpy/PIL/cosysairsim at top level; no lazy numpy/cv2/torch/PIL/cosysairsim import (incl. `__import__`/`import_module`) in any function of server.py or ue_remote.py |
| 9 | stdout hygiene + entry point (AST) | `test_no_bare_print_and_protected_entry_point` | pass | no `print()` without `file=`; no `mcp.run()`; `__main__` calls `main()` |

## Defects found and fixed

| ID | Defect (root cause) | Fix | Regression test |
|---|---|---|---|
| D1 | **Wedged AirSim = 60 s per call, serialised.** Against a port that accepts but never answers (sim frozen, paused, loading, foreign process), `sim_ping` took 61 s and a queued `sim_state` 121 s. The handshake ran with the client's 60 s RPC timeout on the single sim thread. | `server.py _client()`: the handshake runs with `c.client._timeout = SIGHTLINE_AIRSIM_CONNECT_TIMEOUT_S` (default 5), restored to 60 afterwards. msgpackrpc copies the timeout into each Future at call time, so only the connect is shortened. On failure the client is closed and a clear error is raised. | `test_wedged_airsim_fails_within_seconds_and_status_stays_live`, `test_cancellation_...` |
| D2 | **`status()` blocked behind every editor command.** `status()` took `_remote_lock` for discovery, and `_editor_run` held that lock for the whole command (up to `timeout_s` = 300 s). | `_remote_init_lock` + `_get_remote()` (one `UnrealRemote` that is never replaced). `status()` reads discovery without the command lock. `UnrealRemote.start()` is now thread-safe. | `test_fake_editor_long_command_lock_retry_semantics` |
| D3 | **No-editor ue_python/ue_console took 10 s, and the retry policy was wrong.** The catch-all `except Exception` retried discovery failures (5 s + 5 s) and replaced the `UnrealRemote` each time. A queued command could also wait forever on the lock. | `_editor_run`: fails fast when no `UnrealEditor` process exists. It retries only on `ConnectionError` / new `ue_remote.CommandConnectionClosed` (stale channel) and never on discovery failures. The lock wait is bounded by the call's `timeout_s`. (No re-send on `TimeoutError` was added concurrently by another agent; covered by the attempt counter in the test.) | `test_editor_tools_fail_fast_without_editor`, `test_fake_editor_long_command_lock_retry_semantics` |
| D4 | **`editor_wait_ready(timeout_s)` overran its deadline** by up to 25 s: a fixed 20 s probe plus a 5 s sleep. | Probe timeout and sleep are clamped to the remaining time. | `test_editor_wait_ready_honours_its_timeout` |
| D5 | **stdout protection was partial and racy.** Only AirSim/editor calls used `contextlib.redirect_stdout`, which swaps the process-global `sys.stdout`. Two overlapping tool threads could restore the real stdout while the other still printed, and native writes to fd 1 were never covered. | New `server.main()`: the transport writes to a private `os.dup` of fd 1; fd 1 is `dup2`'d to stderr (the UCRT also updates `STD_OUTPUT_HANDLE`); `sys.stdout = sys.stderr`. `__main__` calls `main()` instead of `mcp.run()`. | `test_stray_stdout_is_diverted_to_stderr` (+ negative control), `test_no_bare_print_and_protected_entry_point` |
| D6 | **Server could outlive its client.** `anyio.to_thread.run_sync` without `abandon_on_cancel` makes a cancelled request, or a transport close, wait for the blocking thread. On stdin EOF the process also joined the AirSim executor thread at exit. Found by reading the code; no pre-fix measurement (no git history). | `threaded()` uses `abandon_on_cancel=True`. `main()` calls `os._exit` after the stdio loop ends. | `test_server_exits_on_stdin_eof_even_with_calls_in_flight`, `test_cancellation_...` |
| D7 | **`ue_log(lines=0)` / `job_status(tail=0)` returned the whole file**, because `text[-0:]` is the full list. A multi-hundred-MB editor log would go out in a single message. | Both reject values < 1 with a clear error. | `test_invalid_arguments_fail_fast_and_cleanly` |
| D8 | **Unclear enum errors.** `ue_python(mode="bogus")` returned `'bogus'` and `sim_detections(image_type="nope")` returned `'nope'` (bare KeyErrors). | Explicit validation with the list of valid values. (`sim_fly`/`sim_clock`/`sim_capture` were fixed concurrently by the main agent; the same test covers them, including "validated before contacting the sim".) | `test_invalid_arguments_fail_fast_and_cleanly` |
| D9 | **Two server instances could cross-wire the editor's command channel.** The fixed `127.0.0.1:6776` listener used `SO_REUSEADDR`, which Windows allows twice, so an editor could connect to the other server. This only matters while an editor runs and both servers connect at the same moment. | `ue_remote.UnrealRemote(command_port=0)` by default: `open_listener()` binds an ephemeral port and advertises it in `open_connection`. The engine connects to whatever port is named (`PythonScriptRemoteExecution.cpp` l.394-401). `SIGHTLINE_UE_REMOTE_CMD_PORT` pins a port, with `SO_EXCLUSIVEADDRUSE`. | `test_remote_exec_command_listener_is_never_shared` |

Files changed: `tools/sightline_mcp/server.py` (`threaded`, `_remote_*`/`_get_remote`/`_editor_run`, `status` discovery,
`editor_wait_ready`, `ue_python` mode, `ue_log` lines, `job_status` tail, `_client` handshake, `sim_detections`
image_type, new `main()`), `tools/sightline_mcp/ue_remote.py` (`CommandConnectionClosed`, locked `start()`,
`open_listener()`, default `command_port=0`). No tool renamed and no parameter removed or retyped.

## Open issues (not fixed here)

1. **Flights longer than 60 s hit the RPC session timeout** (main agent's section). `sim_fly(..., timeout_s=90)` joins a
   Future created with the client's 60 s session timeout, so a long move or RTL should fail with "Request timed out"
   and drop the client. Suggested fix: set `c.client._timeout = int(timeout_s) + 10` before `*Async(...)` in `sim_fly`,
   then restore it. Inferred from the msgpackrpc source; not measured against a live sim.
2. **Live check of the ephemeral command port (D9) is still pending.** It is protocol-compatible per the engine source,
   but has not been exercised against a real editor, since this task must not send commands to the editor. After the
   `sightline` server restarts, one `ue_python("print(1)")` confirms it.
3. **Running Claude Code sessions still use the old server code** until `/mcp` reconnect or a restart.
4. **Python SDK client + Windows job object:** `mcp.client.stdio` puts the server in a Job with KILL_ON_JOB_CLOSE.
   Processes the server launches (editor, `-game`, UBT jobs) inherit the job and die when such a client session closes.
   This affects `smoke_test.py`/`test_sim.py`-style scripts that call `editor_launch`, not Claude Code as far as I can
   tell from libuv's job setup (SILENT_BREAKAWAY_OK); not measured.
5. **Two Claude sessions driving one editor at once:** each server opens its own command connection. How the editor
   arbitrates concurrent connections from two remote nodes is untested (needs a live editor).
6. FastMCP ignores unknown argument names (e.g. a typo in an optional parameter) without an error. This is SDK behaviour, left as is.
