# The real development workflow: editor + Play-in-Editor + AirSim + gamepad, through MCP (2026-09-10)

Scope: the loop the project will actually be built in — Claude Code drives the `sightline` stdio MCP server and
Epic's in-editor `unreal` HTTP MCP server to launch the editor, start **Play-in-Editor**, fly and capture through
Cosys-AirSim, hand the drone to a gamepad and back, and stop again. Everything below was exercised **through the
real MCP transports** (a stdio `ClientSession` spawned from the `sightline` entry of `.mcp.json`, and a Streamable
HTTP `ClientSession` to `http://localhost:8000/mcp`) plus real `claude -p` CLI sessions. Nothing was tested by
importing `server.py` and calling functions.

Previously verified elsewhere and **not** repeated here: engine/build/editor/log/job tools and the Epic toolset
inventory (`engine_tools.md`), protocol robustness (`mcp_protocol.md`), the AirSim suite against the **packaged**
Blocks build (39/39, `TRACKER.md` V4), CLI discovery (`claude_code_integration.md`).

Machine during the run: 15.7 GB RAM (3.1–6.8 GB free), RTX 4060. The packaged Blocks build was never run at the
same time as the editor.

## Summary

| # | Area | Result |
|---|---|---|
| 1 | Editor launched through MCP, our settings profile proven in use | PASS (4/4) |
| 2 | PIE started through Epic's `unreal` server, AirSim RPC up | PASS (3/3) |
| 3 | AirSim suite (`test_sim.py`) against PIE | 38/39 first run -> **defect D1 found and fixed** -> 39/39 |
| 4 | StopPIE / sim tools fail cleanly / second StartPIE | PASS (10/10) |
| 5 | `editor_close` while PIE is running | PASS (4/4, four separate editors) |
| 6 | Gamepad through AirSim (day-1 #4 / F3) | 2 PASS, **live stick input NOT OBSERVED** (two 180 s windows) |
| 7 | Full tool matrix, all 27 tools, editor/sim up and down | PASS 33/33 (up) + 12/12 (down) |
| 8 | `claude -p` CLI spot checks (editor tool, Epic toolset, image tool, error path) | PASS, with **defect D2** (intermittent) |
| 9 | Regression suites re-run after the changes | PASS: pytest 22/22, smoke_test 27 tools, doctor 0 FAIL |

Files changed: `tools/sightline_mcp/server.py` (defect D1 only — new `_capture_once()` + a depth `warning`
field; no tool renamed, no parameter added or retyped, 27 tools before and after). New:
`tools/sightline_mcp/test_tool_matrix.py`, `tools/day1/gamepad_airsim.py`, `tools/day1/diag_capture_warmup.py`,
this document. Not touched: `TRACKER.md`, `CONTEXT.md`, `.mcp.json`, `pyproject.toml`, `uv.lock`, `.venv`.
State at the end: editor closed, no `-game`/packaged sim running, ports 8000 and 41451 closed.

Test drivers added by this work:
| Script | What it does |
|---|---|
| `tools/sightline_mcp/test_tool_matrix.py --phase up\|down` | every sightline tool over stdio MCP, safe/reversible args, PASS/FAIL table, JSON in `_artifacts/verification/tool_matrix_*.json` |
| `tools/day1/gamepad_airsim.py --wait N --sample N --alt N` | `rc_data` through AirSim + API<->RC handover with latencies; never blocks longer than `--wait` |
| `tools/day1/diag_capture_warmup.py [--through-mcp]` | repro/regression for defect D1 (cold-start capture). `--raw` on a cold sim should show one bad first batch; `--through-mcp` must never show one |

---

## 1. Editor launched through MCP, with **our** settings file

`editor_launch(settings_profile="default", wait_ready_s=0)` then polling `editor_wait_ready(timeout_s=30)`:

```
launched pid=26728: D:\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe
        D:\Sightline\sim\SightlineSim\SightlineSim.uproject -settings=D:\Sightline\sim\settings\default.json
[ 31.7s] timed out after 30s ... (still loading)
[ 48.8s] editor ready / success: True / [Info] 5.8.2-56702186+++UE5+Release-5.8
```
Cold-ish launch **48.8 s**; two later launches in the same session took **18.2 s** and **26.6 s** (warm D: DDC).
No rebuild was needed at any point (`Target is up to date`).

### 1.1 Defect D0 (documentation, not code): "Loaded settings from ..." never reaches the project log

The task's expected evidence cannot exist. In Cosys-AirSim 5.8-v3.4.1 the message is emitted by
`UAirBlueprintLib::LogMessage` (`Plugins/AirSim/Source/AirBlueprintLib.cpp:361-400`), where **every `UE_LOG` line
is commented out**; the only output is `GEngine->AddOnScreenDebugMessage(...)`. So `Loaded settings from
D:\Sightline\sim\settings\default.json` appears **on the viewport HUD only** and never in
`Saved/Logs/SightlineSim.log`. Confirmed by grepping the whole log for `Loaded` / `AirSim` after PIE.

**Use this evidence instead** (all four collected live, PIE running):

| Evidence | Value |
|---|---|
| Command line actually used (log line 489) | `LogInit: Command Line: -settings=D:\Sightline\sim\settings\default.json` |
| `Documents\AirSim` fallback | **does not exist** on this machine (`%USERPROFILE%\Documents\AirSim` absent), so it cannot be the source |
| Vehicle name from RPC | `listVehicles() -> ['Drone']` — AirSim's built-in default is not `Drone`; it comes from our `Vehicles` block |
| Camera from RPC | `simGetImages('survey', Scene)` -> **1920x1080** — our `Cameras.survey` + `CameraDefaults.CaptureSettings` |
| `OriginGeopoint` from RPC | `getHomeGeoPoint() -> lat 11.48703, lon 76.14512, alt 1061.9 m` — matches `OriginGeopoint` in `sim/settings/default.json`, **including the `Altitude: 1060` value another agent wrote into the file a few minutes before PIE started** (it was 900 when the editor process was launched). That proves the file is read at PIE start, from our repo path, not cached and not from `Documents`. |

## 2. PIE started through Epic's `unreal` server

`call_tool` on the EditorApp toolset. The argument schema is strict — `options` is required and inside it
`bSimulate`, `playMode` and `warmupSeconds` are **all required** (no defaults), and `playMode` must be one of the
`EPlayModeType` enum names:

```python
call_tool(toolset_name="EditorToolset.EditorAppToolset", tool_name="StartPIE",
          arguments={"options": {"bSimulate": False, "playMode": "PlayMode_InViewPort", "warmupSeconds": 8}})
-> {"returnValue": null}
```
Timings: first StartPIE of a fresh editor **14.3 s**, later ones **10.4 s** and **~2 s** (the call returns after
`PostPIEStarted` + `warmupSeconds`). `IsPIERunning` -> `{"returnValue":true}`. AirSim RPC :41451 was already
listening when StartPIE returned (`open after 0.0s`). `LogPlayLevel: PIE: Play in editor total start time
2.662 seconds` / `LogLoad: Game class is 'AirSimGameMode'` / `AirSim Annotation [InstanceSegmentation]: Completed
full level instance segmentation RGB annotation` in the project log.

Client note: the Python MCP SDK prints `Session termination failed: 202` when it closes a session — Epic's server
answers `DELETE /mcp` with 202 instead of 200. Cosmetic; the session is terminated.

## 3. AirSim suite against PIE — defect D1

`.venv\Scripts\python.exe -u tools\sightline_mcp\test_sim.py --alt 20 --detect-filter "*Cube*"`

First run against a **cold editor's first PIE**: **38/39**, results
`_artifacts/verification/sim_tools_20260910-192832.json`. The one failure was the only difference from the
packaged-build run:

```
[FAIL] sim_capture depth range plausible :: min=0.0 max=0.61328125 (alt 20.0 m, nadir camera)
```

### D1: the FIRST `simGetImages` of a fresh engine process returns unconverted buffers (silently)

Reproduced deterministically (`tools/day1/diag_capture_warmup.py`, cold editor -> first PIE -> take off to
20.5 m -> six-type batch, five times, 4.5 s apart):

| batch | t since connect | depth (min,max) m | depth_planar (min,max) m | surface_normals PNG bytes |
|---|---|---|---|---|
| **0** | 20.8 s | **0.0, 0.6133** | **0.0, 0.6133** | **1 727 850** |
| 1 | 25.3 s | 20.9531, 28.3281 | 18.1875, 21.1875 | 80 878 |
| 2 | 29.8 s | 20.9531, 28.3438 | 18.1875, 21.1875 | 80 825 |
| 3 | 34.3 s | 20.9531, 28.3438 | 18.1875, 21.1875 | 80 825 |
| 4 | 38.5 s | 20.9531, 28.3281 | 18.1875, 21.1875 | 80 608 |

Only the **first** request in a fresh engine process is affected — it comes back before the capture components'
post-process materials exist, so float depth is a raw 0..~0.61 buffer instead of metres, and `surface_normals` is a
21x larger (wrong) PNG. `width`, `height` and the pixel count are all correct, there is no error and the RPC
succeeds, so nothing downstream can tell. It did **not** reproduce after `StopPIE`+`StartPIE` inside an
already-warm editor, nor on the second capture — which is why the packaged-build run never saw it and why the
naive "capture twice by hand" check passes.

**Fix (`tools/sightline_mcp/server.py`)** — new `_capture_once(c, reqs, vehicle)` used by `sim_capture`: the first
capture after each AirSim connection issues the request **twice** and returns the second result (a 0.25 s pause in
between); `_capture_warmed` is reset in `_client()` whenever a new client is created, so a new sim process is
warmed again. Cost: ~1 s, once per sim session. Plus a belt-and-braces guard: a float depth image whose `max` is
<= 1.0 m now carries a `warning` field in the returned JSON saying it looks like an unconverted buffer.

**Re-verified**: cold editor (launch 26.6 s), fresh PIE, `test_sim.py --alt 20 --detect-filter "*Cube*"` ->
**39/39** (`_artifacts/verification/sim_tools_20260910-200657.json`, log `_artifacts/test_sim_pie_after_fix.log`).
The formerly failing check now reads `sim_capture depth range plausible  min=21.0 max=28.40625 (alt 20.0 m,
nadir camera)` and `surface_normals.png` is back to 81 023 bytes (it was 1 727 850 in the bad frame). Every other check
behaved exactly as it did against the packaged build, including `sim_fly land`/`rtl` touchdown (`rtl` landed
0.2 mm from home), `sim_clock pause` freezing a moving vehicle (0.000 m in 1.5 s), spawn/move/destroy and the
detection API.

## 4. StopPIE, sim tools with the sim gone, second StartPIE (10/10 PASS)

`scratchpad pie_cycle.py`, results `_artifacts/verification/pie_cycle.json`:

| Check | Evidence |
|---|---|
| `IsPIERunning` before | `{"returnValue":true}` |
| `StopPIE` | `{"returnValue":null}` in **2.0 s** |
| AirSim RPC closes | :41451 closed **1.0 s** after StopPIE |
| `IsPIERunning` after | `{"returnValue":false}` |
| `sim_ping` with the sim gone | isError, **0.32 s**, `AirSim RPC not listening on 127.0.0.1:41451. Start PIE/game first.` |
| `sim_state` | isError, **0.31 s**, same message |
| `sim_capture` | isError, **0.31 s**, same message |
| `StopPIE` a second time | clean error `A play session is not currently running.` |
| second `StartPIE` | `{"returnValue":null}` in **10.4 s**, :41451 open again immediately |
| `sim_ping` after it | `{"ping": true, "server_version": 4, "vehicles": ["Drone"], "paused": false}` in **0.30 s** |

No hangs anywhere: the stale client is dropped by `_sim()` and the next call reconnects.

The fix was then re-checked in isolation on yet another cold editor with
`tools/day1/diag_capture_warmup.py --through-mcp --batches 3`: batch 0 already read
`depth [20.78, 27.80] m` (`_artifacts/verification/capture_warmup_mcp.json`), i.e. the first capture a caller
ever sees is now correct.

## 5. `editor_close` with PIE running (4/4, run four times)

`_artifacts/verification/editor_close_with_pie.json`:
```
[PASS] state before close: PIE RPC 41451=True epic 8000=True editor pids=[26728]
[PASS] editor_close with PIE running: 'closed cleanly' in 8.7s   (later runs: 6.8s, 7.9s, 8.9s)
[PASS] editor gone: editor pids=[] 8000=False 41451=False
[PASS] status after close: ue_processes=[] all endpoints False, ram_available 6.8 GB
```
The graceful path (`unreal.SystemLibrary.quit_editor()` through remote execution) works while PIE is live; no
"save changes?" dialog blocked it and no force-kill was needed.

## 6. Gamepad through AirSim (day-1 test #4, feature F3) — partly NOT OBSERVED

`tools/day1/gamepad_airsim.py` (new). Results
`_artifacts/verification/gamepad_airsim_20260910-193842.json` and `..._195738.json`.

| Check | Result | Evidence |
|---|---|---|
| settings RC block | PASS | `RemoteControlID=0`, `AllowAPIWhenDisconnected=True`, `AllowAPIAlways=True` from `sim/settings/default.json` |
| `rc_data` reaches AirSim | **PASS** | `is_initialized=True is_valid=True vendor_id='VID_045E' throttle=0.5 pitch=-0.0 roll=0.0 yaw=0.0 switches=0` — the pad is enumerated by the sim (VID_045E = Microsoft), and the resting state is correct (throttle 0.5 == centred left stick) |
| live stick input | **NOT OBSERVED** | two separate 180 s windows (19:35 and 19:57 local), each announced on the PIE HUD (`simPrintLogMessage`) *and* in a Windows message box. Every sample stayed exactly neutral (`throttle 0.5`, `pitch/roll/yaw 0.0`, `switches 0`). Nobody touched the controller during the windows. |
| axis travel / switches | not reached | needs live input |
| API-ON ignores a held stick | not reached | needs live input |
| `sim_fly release` -> gamepad flies it, latency | not reached | needs live input |
| `sim_fly arm` -> API re-acquires, latency | not reached | needs live input |

This matches `TRACKER.md` V8 ("live input NOT yet observed (all-zero for 55 s)") at the Windows/XInput level: the
device enumerates but nothing is moving it. **Nothing indicates a defect in our code or in AirSim** — the whole
chain up to "the sim reads the device" is proven; only the human input is missing.

To finish it, with PIE running and someone at the controller:
```
.venv\Scripts\python.exe -u tools\day1\gamepad_airsim.py --wait 180 --sample 20 --alt 8
```
It prints instructions to the console **and** draws them on the PIE viewport, waits at most `--wait` seconds for
the first non-neutral sample, samples the axes, then flies the handover sequence
(`takeoff` -> `move_to -8 m` -> hold throttle -> `release` -> `arm` -> `move_to` -> `rtl`) and reports both
handover latencies. It exits with NOT OBSERVED rather than blocking if nobody touches the pad.

### 6.1 `RemoteControlID` / `AllowAPIWhenDisconnected` / `AllowAPIAlways`, read from the plugin source

- `RemoteControlID: 0` selects joystick index 0 (`PawnSimApi::getRemoteControlID()` -> `rc.remote_control_id`;
  `-1` disables RC). Observed working: `rc_data.is_initialized/is_valid` are true with the pad on slot 0.
- Input is read through **DirectInput with `DISCL_NONEXCLUSIVE | DISCL_BACKGROUND`**
  (`Source/SimJoyStick/DirectInputJoyStick.cpp:308`), so the Unreal window does **not** need focus for the
  gamepad to fly the drone. That matters for a Claude-driven loop where the terminal has focus.
- Axis mapping (`Source/PawnSimApi.cpp:337-346`): `throttle = (left_y + 1) / 2` (so 0.5 is centred),
  `yaw = left_x`, `roll = right_x`, `pitch = -right_y`; `switches` is the button bitmask.
- `AllowAPIAlways: true` makes `allow_api_control_` unconditionally true
  (`simple_flight/firmware/RemoteControl.hpp:190-200`: `allow = rc.allow_api_always; allow |= isRcConnected() ?
  readChannel(allow_api_control_channel) > 0.1 : rc.allow_api_when_disconnected`). With it set,
  **`AllowAPIWhenDisconnected` is a no-op** in our profile (it only applies when no RC is connected) and the RC's
  "allow API" switch (switch 1) is not needed. That is the behaviour we want for F3: the API can always take the
  drone back, and `sim_fly release` is what hands it to the pad.
- Every API-side half of the handover **is** verified: `sim_fly release` -> `api_control False` and `sim_fly arm`
  -> `api_control True`, both in `test_sim.py` (39/39) and in the tool matrix. What is unverified is only the
  physical "stick moves the drone" half and the two latencies.

## 7. Full tool matrix (`tools/sightline_mcp/test_tool_matrix.py`)

All **27** tools the server exposes are called at least once over the real stdio transport, with safe arguments;
destructive ones are only started and then killed (`ue_package` -> `job_kill`) or reversed
(`sim_spawn_object` -> `sim_destroy_object`, weather set -> restored, `stat fps` -> `stat none`).
`ue_build` is exercised with an invalid target so nothing on disk changes (a real successful build is in
`engine_tools.md`, and the editor target cannot be built while the editor runs).

### 7.1 Phase `up` — editor running, PIE running: **33/33 PASS**
(`_artifacts/verification/tool_matrix_up_20260910-194315.json`)

| Tool / check | Result | Evidence |
|---|---|---|
| `status` | PASS | editor pid 26728, RAM avail 3.1 GB, GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, `unreal_mcp_http:8000 True`, `airsim_rpc:41451 True`, remote-exec node listed |
| `editor_launch` (guard) | PASS | `An editor is already running; use status() or editor_close() first.` |
| `editor_wait_ready` | PASS | `editor ready` in 2.34 s |
| `ue_python` | PASS | eval -> `'5.8.2-56702186+++UE5+Release-5.8'` (**works while PIE runs**) |
| `ue_python` bad mode | PASS | `unknown mode 'bogus'; expected one of ['eval', 'file', 'statement']` |
| `ue_console` | PASS | `stat fps` -> ok, restored with `stat none` (**works while PIE runs**) |
| `ue_log` | PASS | log path header + exactly 5 matching lines for the grep |
| `ue_log(lines=0)` | PASS | `lines must be >= 1 (got 0)` |
| `ue_package` | PASS | `job_id=package-...` pid 8736, real `RunUAT.bat BuildCookRun ...` argv |
| `job_status` | PASS | `running`, elapsed 4 s, argv line + live tail |
| `job_list` | PASS | the package job listed as `running`, 9 job logs on disk |
| `job_kill` | PASS | `killed` -> state `failed` exit 15 -> second kill `not running` |
| `job_status` unknown id | PASS | `unknown job no-such-job` |
| `ue_build` | PASS | invalid target -> job `failed` exit 8, summary `Result: Failed (RulesError)`, target name in the log |
| `ue_generate_project_files` | PASS | `exit 0` in 25 s, fresh `SightlineSim.sln` (skipped in the final re-run) |
| `sim_ping` | PASS | `server_version=4 vehicles=['Drone'] paused=False` in 0.05 s |
| `sim_state` | PASS | pos, GPS 11.48703/76.14512, `landed_state`, `api_control` |
| `sim_fly` | PASS | `arm` -> api_control True; `release` -> False; `arm` again |
| `sim_fly` bad action | PASS | `unknown action 'bogus'; expected one of [...]` |
| `sim_capture` | PASS | scene 1920x1080 saved; preview `image/jpeg`, 12 193 b64-decoded bytes, **decodes to a valid JPEG 960x540** |
| `sim_capture` bad type | PASS | `unknown image type(s) ['nope']; valid: [...]` |
| `sim_environment` | PASS | `weather=True, Rain=0.4, wind=[2.0,0,0]` then restored |
| `sim_clock` | PASS | status/pause/step/resume all report the expected `paused` flag |
| `sim_objects` | PASS | 212 objects |
| `sim_list_assets` | PASS | 18 assets matching `cube` |
| `sim_spawn_object` | PASS | spawned at [8,8,-3] (read back) |
| `sim_set_object_pose` | PASS | moved to [9,8,-3], verified |
| `sim_destroy_object` | PASS | destroyed; `sim_objects` now `[]` |
| `sim_destroy_object` unknown | PASS | `simDestroyObject('slt_no_such_actor') returned False` |
| `sim_detections` | PASS | valid list (0 boxes from that pose; 2 boxes with `*Cube*` from the `test_sim.py` pose) |
| `sim_launch_game` bad profile | PASS | `AirSim settings profile not found: ...no_such_profile.json` |
| `ue_python_headless`, `editor_close` | deferred | run in phase `down` / at the end |

### 7.2 Phase `down` — editor closed, sim gone: **12/12 PASS**
(`_artifacts/verification/tool_matrix_down_20260910-195929.json`)

| Tool / check | Result | Evidence |
|---|---|---|
| `status` | PASS | no UE processes, all endpoints false |
| `ue_python` / `ue_console` | PASS | isError in **0.64 s** each: `No Unreal Editor is running ...` |
| `editor_wait_ready` | PASS | 0.63 s: `Editor process is not running (crashed or closed). Check ue_log().` |
| `editor_close` (nothing to close) | PASS | 1.95 s, `closed cleanly` |
| `sim_ping`/`sim_state`/`sim_capture`/`sim_objects`/`sim_clock` | PASS | isError in **0.31–0.33 s** each, message names `127.0.0.1:41451` |
| `ue_python_headless` | PASS | `exit 0` in 12 s; `SLT_MATRIX_VERSION=5.8.2-56702186+++UE5+Release-5.8`, `SLT_MATRIX_ASSETS=9` |
| `sim_launch_game` | PASS | real `-game` process launched (`-ResX=640 -ResY=480 -windowed -log -RenderOffscreen`), **`sim_ping` answered 20 s later**, then torn down |

## 8. `claude -p` CLI spot checks

CLI 2.1.260, logged in (`claude auth status -> loggedIn: true`). Every run: `--output-format stream-json
--verbose --no-session-persistence --model sonnet --max-budget-usd <cap>`; raw transcripts in
`_artifacts/cli_calls_a.jsonl` … `_d.jsonl`. Init event in every run:
`mcp_servers = [unity-editor-mcp connected, unreal connected, sightline connected]`.

| Check | Result | Evidence |
|---|---|---|
| editor tool `mcp__sightline__ue_python` | PASS | model called it with `mode: "eval"`, got `success: True / result: '5.8.2-56702186+++UE5+Release-5.8'` |
| Epic toolset `mcp__unreal__call_tool` (`IsPIERunning`) | PASS **with D2** | 2 of 3 calls returned `{"returnValue":true}`; one failed with `The socket connection was closed unexpectedly` (see D2) |
| image tool `mcp__sightline__sim_capture` | PASS | tool_result carried an `image` block, `media_type image/jpeg`, **13 725 bytes that decode to a valid JPEG 960x540**, and the model described the actual picture ("flat pale grey field … a blurred light-grey diagonal edge cutting across the top-left corner" = the ground from 25 m with a rotor arm in frame). The JSON reported `scene 1920x1080` and `segmentation 1920x1080`. |
| error path `mcp__sightline__sim_fly(action="bogus")` | PASS | `is_error: true`, `Error executing tool sim_fly: unknown action 'bogus'; expected one of [...]`, session continued normally |
| a full flight through the CLI | PASS | `takeoff` -> `move_to (60,0,-20)` -> `rtl` all succeeded from the model, `rtl: done (touchdown confirmed ... in 20.8 s)` |

### D2: `mcp__unreal__*` calls fail intermittently with "socket connection was closed unexpectedly"

Root cause found in the engine source: **UE's HTTP server hard-codes its keep-alive idle timeout**
(`Engine/Source/Runtime/Online/HTTPServer/Private/HttpConnection.h:266`,
`static constexpr float ConnectionKeepAliveTimeout = 15.0f;`) and advertises it
(`Keep-Alive: timeout=15.000000`). There is **no cvar and no ini setting** for it. Measured with a raw HTTP/1.1
socket (`scratchpad epic_keepalive.py`):

```
initialize -> HTTP/1.1 200  keep-alive=timeout=15.000000
after   2s idle on the SAME socket -> HTTP/1.1 202
after  10s idle on the SAME socket -> HTTP/1.1 200 {"returnValue":true}
after  20s idle on the SAME socket -> DEAD: ConnectionAbortedError [WinError 10053]
```
Claude Code's HTTP client (Node/undici) pools keep-alive sockets and does **not** retry a POST whose socket dies,
so a `call_tool` that lands in the race window around the server's 15 s idle close fails once with undici's
`The socket connection was closed unexpectedly`. It is intermittent, not deterministic: a later run with a
deliberate ~25 s gap (a `sim_fly move_to` between the two `unreal` calls) succeeded, because by then undici had
already discarded the socket and opened a new one.

Not fixable in `server.py` (it is Epic's server plus Claude Code's HTTP client). **Mitigation: retry the same
`mcp__unreal__*` call once** — a retry opens a fresh connection and succeeds. The `sightline` stdio server is
completely unaffected. Verified from Python that the failure is not caused by anything we control: the same call
with and without a progress token, and after 5/20/35/65 s of idle, always succeeds through `httpx`
(`scratchpad epic_progress.py`, `epic_idle.py`) — only a *reused* dead socket fails.

## 9. Regression suites re-run at the end

Run with the editor closed and no sim, after every change in this session:

| Suite | Result |
|---|---|
| `.venv\Scripts\python.exe -m pytest tests/test_mcp_protocol.py -q` | **22 passed in 91.88 s** (previously 20 passed / 2 skipped — the two no-editor scenarios now run because no editor was up) |
| `.venv\Scripts\python.exe tools\sightline_mcp\smoke_test.py` | exit 0 — `sightline 1.30.0 protocol 2025-11-25`, **27 tools, missing: none**, `status` returned real RAM and the RTX 4060 |
| `.venv\Scripts\python.exe tools\doctor.py --live` | exit 0 — **0 FAIL, 4 WARN**. The WARNs are all expected with the editor down: `Epic MCP HTTP :8000`, `AirSim RPC :41451`, `RAM available >= 9 GB` (6.6 GB free), and the pre-existing `uv on PATH` (restart the terminal). |

## 10. Defects found in this work

| ID | Defect | Where | Fix |
|---|---|---|---|
| **D1** | The first `simGetImages` of a fresh engine process silently returns unconverted buffers: float depth 0.0–0.6133 instead of metres, `surface_normals` a 1.7 MB (wrong) PNG. No error, correct width/height. Broke `sim_capture depth range plausible` in the first PIE run of `test_sim.py`. | `tools/sightline_mcp/server.py` `sim_capture` | New `_capture_once()`: one discarded warm-up request per AirSim connection (`_capture_warmed`, reset in `_client()`), then the real request. Plus a `warning` field when a float depth image's max is <= 1.0 m. Re-verified 39/39 on a cold editor. |
| **D2** | `mcp__unreal__*` intermittently fails with `The socket connection was closed unexpectedly` (UE's hard-coded 15 s HTTP keep-alive idle close + Node keep-alive socket reuse). | Epic's `ModelContextProtocol` HTTP server + Claude Code's HTTP client | Not fixable by us. Retry the call once. Documented in §8. |
| **D0** | "Loaded settings from &lt;path&gt;" can never appear in the project log (all `UE_LOG` calls in `UAirBlueprintLib::LogMessage` are commented out in Cosys 3.4.1) — a verification method, not a bug in our code. | plugin source | Use the RPC evidence in §1.1 instead. |
| **H1** | `tools/day1/gamepad_airsim.py` first crashed with `RuntimeError: This event loop is already running` / `assert self._self_reading_future is None`. | my own new script | The cosysairsim client owns a tornado IOLoop that cannot start on a thread that already runs an asyncio loop — the exact reason `server.py` keeps `_SIM_THREAD` (CONTEXT §7). Rewrote the script to be synchronous and to run its MCP `ClientSession` in a background asyncio thread (`McpBridge`). Worth knowing for every future script that mixes MCP and cosysairsim. |
| **H2** | Harness-only: `test_tool_matrix.py` initially used `ue_python(mode="eval")` with a statement (SyntaxError — eval takes an expression), and asserted on volatile job-log tails. | new script | Fixed; final run 33/33 + 12/12. |

## 11. Open issues / things the next session must know

1. **Gamepad live input is still NOT OBSERVED** (day-1 #4, F3). Everything up to "the sim reads the device" is
   proven; run `tools/day1/gamepad_airsim.py` with someone holding the controller to close it and to get the two
   handover latencies.
2. **Retry `mcp__unreal__*` once** on `socket connection was closed unexpectedly` (D2). Consider preferring the
   `sightline` server's `ue_python` for anything that has an equivalent there, since it is immune.
3. **The first capture after any fresh sim process is now warmed automatically**, but only inside `sim_capture`.
   Any future code that calls `simGetImages` directly (dataset recorders, benchmarks) must discard its own first
   response — see `_capture_once()` in `server.py`.
4. Launching the editor from a **Python** MCP client kills it when that client exits (`stdio_client` puts the
   server in a Job Object with `KILL_ON_JOB_CLOSE`, no breakaway — `engine_tools.md` §7.1). This whole run
   therefore kept one long-lived "hold" session open while other scripts ran. Claude Code's own spawn path is not
   affected.
5. `sim_detections` returned 0 boxes from (0,0,-20) and 2 boxes from (15,-10,-20) with `*Cube*` — the filter and
   radius are fine, the pose matters. Not a defect, but tests should fly to a pose with known geometry.
6. Epic's server answers `DELETE /mcp` with **202**, so the Python SDK logs `Session termination failed: 202`.
   Cosmetic.
7. `ue_python` and `ue_console` work normally **while PIE is running** — Claude can edit the level, read cvars and
   run console commands mid-flight.
