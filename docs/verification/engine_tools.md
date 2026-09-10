# Engine-side MCP tooling verification (2026-09-10)

Scope: the engine / build / editor / log / job tools of the `sightline` stdio MCP server
(`tools/sightline_mcp/server.py`) and Epic's in-editor `unreal` MCP server (`http://localhost:8000/mcp`).
Everything below was exercised **through the real MCP protocol** - a stdio `ClientSession` spawning the server
exactly as `.mcp.json` declares it, and a Streamable HTTP `ClientSession` for Epic's server - never by importing
and calling Python functions. AirSim / `sim_*` tools are out of scope (owned by the main agent).

Machine state during the run: 15.7 GB RAM, pagefile-backed commit limit 28-31 GB, RTX 4060 8 GB. For part of the
session the main agent's `Blocks.exe` held 9.3 GB of commit; the editor test was run after it exited.

## 1. Test drivers (re-runnable)

| Script | What it does |
|---|---|
| `tools/sightline_mcp/test_engine.py <steps>` | steps: `status build jobs genproj headless editor log close`. Options `--target --max-actions --ready-timeout --hold-file --hold-max --baseline-server`. Exit code = FAIL count. Console log `_artifacts/test_engine/run_*.log`, JSON `_artifacts/test_engine/results-*.json`, server stderr `_artifacts/test_engine/server_stderr.log`. |
| `tools/sightline_mcp/test_unreal_mcp.py -v` | Epic server conformance: initialize, tools/list, list_toolsets, describe_toolset for every toolset, one real `call_tool`. |
| `tools/sightline_mcp/test_unreal_calls.py` | Real `call_tool` invocations across EditorApp / Logs / ConfigSettings toolsets + full inventory dump to `_artifacts/unreal_mcp/inventory.json`, raw schemas in `_artifacts/unreal_mcp/describe/`, viewport capture to `_artifacts/unreal_mcp/capture.png`. |
| `tools/sightline_mcp/repro_timeout_reexec.py <server.py> [--hitch]` | Reproduces the timed-out-command double-execution bug against a live editor (see §4.6). |

`test_engine.py` spawns the server with the `.mcp.json` command/args plus **the full process environment**, which
is what Claude Code does; the plain Python SDK default (`smoke_test.py`) passes only a 12-variable whitelist
(see §4.3).

## 2. Results - `sightline` server (stdio MCP)

| Tool | Result | Evidence |
|---|---|---|
| `status` | PASS | JSON with `project_exists`, RAM/GPU (RTX 4060 Laptop), endpoints, running jobs, and (new) `ddc_path_for_children`, `env_added_from_registry`. With the editor up: `unreal_mcp_http:8000 = true` and a remote-exec node `{project_name: SightlineSim, engine_version: 5.8.2-56702186+++UE5+Release-5.8, machine: LAPTOP-N07SN4K4}`. |
| `ue_build` (new `extra_args`) | PASS | Cold build of `SightlineSimEditor` with `["-MaxParallelActions=6"]`: 85 actions, UBT `Result: Succeeded`, `Total execution time: 868.69 s` (wall 881 s), fresh `Binaries/Win64/UnrealEditor-SightlineSim.dll`; flag visible in the job's argv line. Incremental re-run: exit 0 in 20 s, `Target is up to date`, 0 actions. |
| `job_status` | PASS | `running` -> `succeeded` with `exit_code`, filtered `summary`, `tail`; unknown id -> MCP error `unknown job no-such-job`. |
| `job_list` | PASS | Shows the running package job as `running`, finished build as `exit 0`, plus on-disk job logs. |
| `job_kill` | PASS | Killed the live `RunUAT BuildCookRun` tree (cmd.exe + conhost.exe + dotnet.exe): all three gone 3 s later, `job_status` -> `failed`, exit 15; second kill -> `not running`. |
| `ue_package` | PASS (start only, then killed by design) | Job started, argv is `RunUAT.bat BuildCookRun -project=... -build -cook -stage -pak -archive`; state `running` with a real UAT process tree. A full cook/package was deliberately not run. |
| `ue_generate_project_files` | PASS (after fix §4.1) | `exit 0`, `Result: Succeeded`, 14 s; fresh `SightlineSim.sln` + `SightlineSim.slnx` (+ `Automation_SightlineSim.*`). |
| `ue_python_headless` | PASS | `exit 0`, `SLT_VERSION=5.8.2-56702186+++UE5+Release-5.8`, `SLT_ASSET_COUNT=9` and 9 `/Game/...` asset paths - matches the 9 `.uasset`/`.umap` files on disk; 47 s with `-nullrhi`. |
| `editor_launch` | PASS | `launched pid=23796 ... SightlineSim.uproject -settings=D:\Sightline\sim\settings\default.json` (profile resolved from `sim/settings`). |
| `editor_wait_ready` | PASS | Ready after **44 s** (no long shader compile: the headless run had already warmed the D: DDC), returning the engine version through remote execution. |
| `ue_python` mode=file | PASS | Spawned a StaticMeshActor from `/Engine/BasicShapes/Cube`, set label: `SLT_SPAWNED SLT_TestCube StaticMeshActor`, `success: True`. |
| `ue_python` mode=eval | PASS | `result=['SLT_TestCube']` reading the actor back. |
| `ue_python` mode=statement | PASS | `destroy_actor` -> `[True]`; follow-up eval returns `[]` (actor really gone). |
| `ue_python` large output | PASS | 20 000 printed lines returned complete and in order (20 000 unique markers, last `SLTL19999`, 340 013 chars) - no truncation. |
| `ue_python` error reporting | PASS | `success: False` + full traceback for `raise ValueError('slt boom')`. |
| `ue_python` timeout | PASS (after fix §4.6) | Timed-out command executes **once**, tool returns a clear timeout error; pre-fix server executed it **twice** (§4.6). |
| `ue_console` | PASS | `stat fps` -> `ok`; `t.MaxFPS 61` verified by reading the cvar back (`0.0 -> 61.0`), then restored to `0.0`; `stat none` to clean up. |
| `ue_log` | PASS | Header is the log path `...\Saved\Logs\SightlineSim.log`; `grep=LogPython.*SLT_SPAWNED` returns only matching lines (the cube spawn from `ue_python`); `lines=5` and `lines=3`+grep honour the cap. (Reported FAIL inside the editor run: that was a bug in my assertion, not the tool - see §5.) |
| `editor_close` | PASS | `closed cleanly` in 18 s; `status` afterwards: no UnrealEditor process, `unreal_mcp_http:8000 = false`, no remote-exec node; independent `Get-Process` check also empty. |

Totals: `status build jobs genproj headless` = 8/8 PASS (plus 4/4 on the incremental build re-run),
`editor close` = 16 PASS / 1 FAIL (the FAIL being the harness assertion of §5, re-verified PASS afterwards),
`log` = PASS.

**Build time**: cold editor build 868.7 s UBT / 881 s wall with `-MaxParallelActions=6`; it rebuilt the whole
Cosys-AirSim plugin from source (85 actions) - the prebuilt `UnrealEditor-AirSim.dll` is not reused when the
plugin's source is present. UBT reported `[Adaptive Build] Excluded from AirSim unity file: <every .cpp>`, i.e.
adaptive unity treats all AirSim files as "modified" (they are writable and not under source control), so the
plugin compiles without unity files. Incremental builds are ~20 s.

**Editor RAM**: `status` reported `rss_gb 2.01` while the machine had only 0.9 GB free; measured directly the
editor held **5.15 GB private commit**, 1.67 GB working set, 3.56 GB peak working set (the low RSS is paging
pressure, not a small editor). System commit sat at 28.7/30.1 GB with the editor up.

## 3. Results - Epic's `unreal` server (Streamable HTTP, in-editor)

| Call | Result | Evidence |
|---|---|---|
| `initialize` | PASS | protocol `2025-11-25` negotiated, `Mcp-Session-Id` issued. `serverInfo.name`/`version` come back **empty**. |
| `tools/list` | PASS | Exactly 3 native tools with tool-search on: `list_toolsets`, `describe_toolset`, `call_tool`. |
| `list_toolsets` | PASS | 31 toolsets (see §6). |
| `describe_toolset` | PASS | 31/31 toolsets return valid JSON schemas, 392 tools total; unknown name -> error listing available toolsets. |
| `call_tool` EditorAppToolset.`IsPIERunning` | PASS | `{"returnValue":false}` |
| `call_tool` EditorAppToolset.`GetSelectedActors` | PASS | `[{"refPath":"/Game/FlyingCPP/Maps/FlyingExampleMap.FlyingExampleMap:PersistentLevel.Ground"}]` |
| `call_tool` EditorAppToolset.`SearchCVars` | PASS | `{"t.MaxFPS":{"help":"Caps FPS to the given value...","value":0}}` |
| `call_tool` EditorAppToolset.`GetCameraTransform` | PASS | Returned the live viewport pose, reused as the capture pose. |
| `call_tool` EditorAppToolset.`CaptureViewport` | PASS | base64 PNG in the JSON result: `image/png`, **1267x688**, 568 954 bytes -> `_artifacts/unreal_mcp/capture.png` (a real render of the Blocks level). Needs explicit `captureTransform` **and** `annotations` (§6.2). |
| `call_tool` LogsToolset.`GetLogEntries` | PASS | 5 entries with `category=""`; and with `pattern=SLT_SPAWNED` it returns the line printed by the *sightline* server's `ue_python` - cross-server proof both paths address the same editor. |
| `call_tool` LogsToolset.`GetLogCategories` | PASS | `["LogPython","LogPythonOnlineDocsCommandlet","LogPythonRemoteExecution","LogPythonScriptCommandlet"]` |
| `call_tool` ConfigSettingsToolset.`ListContainers` | PASS | `["Editor","InputBinding","Project"]` |
| `call_tool` ConfigSettingsToolset.`ListCategories` | PASS | 6 categories for `Project` |
| `call_tool` ConfigSettingsToolset.`ListSections` | PASS | `['Encryption','GameplayTags','General','HardwareTargeting','Maps','Movies','Packaging','SupportedPlatforms']` |
| `call_tool` ConfigSettingsToolset.`GetSectionPropertyValues` | PASS | `Project/Project/Maps` -> `{"EditorStartupMap":{"refPath":"/Game/FlyingCPP/Maps/FlyingExampleMap.FlyingExampleMap"}}`, matching `Config/DefaultEngine.ini`. Read-only; nothing was written. |
| `call_tool` unknown tool | PASS | error `Unknown tool NoSuchTool` |

`test_unreal_calls.py`: 17 PASS / 0 FAIL. `test_unreal_mcp.py -v`: exit 0, 31/31 toolsets described.

## 4. Bugs found and fixed

### 4.1 `ue_generate_project_files` ran a batch file that does not exist (server.py)
`Engine/Build/BatchFiles/GenerateProjectFiles.bat` ships only with source builds; this engine is an installed
(Launcher) build (`Engine/Build/InstalledBuild.txt` present). The tool could never have worked.
**Fix** (`server.py`, `ue_generate_project_files`): call the same UBT mode the Explorer verb uses -
`Build.bat -projectfiles -project=<uproject> -game -progress` plus `-rocket` for installed engines
(`-engine` otherwise). Verified: exit 0, fresh `.sln`/`.slnx`.

### 4.2 Child processes did not inherit the D:-cache environment -> DDC/Zen on C: (server.py)
The user-level env vars from `docs/CONTEXT.md` §3 (notably `UE-LocalDataCachePath=D:\UE_Cache\DDC`) were set
after the current hosts started, so they are absent from the server's environment: the **server Claude Code
itself spawned** (parent `claude.exe`) had 90 variables and no `UE-LocalDataCachePath`. Any editor, commandlet or
build it launches would therefore write its DDC **and the Zen store** under `%LOCALAPPDATA%` on C: (GB-scale),
violating the "everything on D:" rule.
**Fix** (`server.py`, new `_merge_user_env()` run at import): copy every value in `HKCU\Environment` that the
process lacks into `os.environ` (process values always win; `PATH`/`TEMP`/`TMP` untouched). `status` now reports
`ddc_path_for_children` and `env_added_from_registry`.
Verified live: after the editor ran, `D:\UE_Cache\DDC` = 258 MB (with a `Zen` subfolder) while
`%LOCALAPPDATA%\UnrealEngine\Common\DerivedDataCache` stayed at 1.1 MB; `status` shows
`ddc_path_for_children: D:\UE_Cache\DDC` and 11 restored variables.

### 4.3 Hosts that pass a minimal environment broke GPU query (and would break UBT) (server.py)
The Python MCP SDK's `stdio_client` passes only 12 whitelisted variables (`APPDATA, HOMEDRIVE, HOMEPATH,
LOCALAPPDATA, PATH, PATHEXT, PROCESSOR_ARCHITECTURE, SYSTEMDRIVE, SYSTEMROOT, TEMP, USERNAME, USERPROFILE`).
Under it `nvidia-smi` fails with `Failed to initialize NVML: Unknown Error`, which is why `status` used to report
`"gpu": null` in `smoke_test.py` / `doctor.py --live`. Bisecting the missing variables showed NVML needs
`ProgramFiles` (or `ProgramW6432`); UBT would be similarly exposed since it locates vswhere and the Windows SDK
under `ProgramFiles(x86)`.
**Fix** (`server.py`, in `_merge_user_env()`): backfill machine-level basics from HKLM when absent -
`ProgramFiles`, `ProgramW6432`, `ProgramFiles(x86)`, `CommonProgramFiles`, `ComSpec`, `windir`, `OS`,
`NUMBER_OF_PROCESSORS`, `ProgramData`. Verified: `smoke_test.py` now reports the real GPU.

### 4.4 UnrealBuildAccelerator wrote its 40 GB-capacity store to C: (server.py)
UE 5.8's default local executor is UBA; `UBAExecutor.cs` defaults `RootDir` to
`%ProgramData%\Epic\UnrealBuildAccelerator` (`UBA Storage capacity 40 GB` in the build log).
**Fix** (`server.py`, next to the env merge): `os.environ.setdefault("UBA_ROOT", r"D:\UE_Cache\UBA")` - the env
var `UBAExecutor.cs` honours. Verified: a subsequent build succeeded and created `D:\UE_Cache\UBA\{cas,castemp,sessions}`.
Note this only covers builds started through this server; a user-level `UBA_ROOT` would also cover VS/IDE builds.

### 4.5 `ue_build` had no way to cap compiler parallelism (server.py)
**Fix**: added `extra_args: list[str] | None` passed verbatim to UBT. Used as `["-MaxParallelActions=6"]`;
free RAM still dipped to 0.9 GB during the cold build, so do not raise it on this machine.

### 4.6 A timed-out `ue_python` command could be executed twice (server.py) - reproduced
`_editor_run` caught *every* exception and re-sent the same command once. A `socket.timeout` means the command
was already delivered and is still running on the game thread, so the retry executes it a second time (duplicate
actor spawns, duplicate asset edits).
**Fix** (`server.py`, `_editor_run`): catch `TimeoutError` separately and re-raise with a clear message
("the command was delivered and may still be running in the editor (not re-sent)"); retry only for a dead
channel. (The main agent has since extended the same function with a lock timeout and a
`ConnectionError`/`CommandConnectionClosed` retry; the timeout branch is preserved.)
**Reproduction** (`repro_timeout_reexec.py --hitch`, live editor): a one-shot slate post-tick callback stalls the
game thread for 6 s - the stand-in for a map load or Blueprint compile - and a command with `timeout_s=2` is sent
into that stall:
- pre-fix server: **executed 2x** (both runs), call returned success after ~6.7 s;
- fixed server: **executed 1x** (both runs), call returned the timeout error after 2.9 s.

Related finding (no code change): the editor accepts **one remote-execution command connection at a time**. When
a second sightline server instance connects, the first one's channel is silently dropped; the current code
recovers (reconnect on `CommandConnectionClosed`), the pre-fix code failed with "No Unreal Editor found".
Two Claude sessions driving `ue_python` at once will therefore interleave, not run in parallel.

### 4.7 `test_unreal_mcp.py` passed while testing almost nothing (test script)
It `json.loads`-ed the `list_toolsets` catalog (which is plain `- Name: description` text), so `toolsets` was
`None` and it described **0** toolsets; it called `describe_toolset` with `{"name": ...}` while the server
requires `toolset_name`; and it never checked `isError`, so failures were invisible. It also never made the
`call_tool` its docstring promised.
**Fix**: parse the catalog with a dotted-name regex, use `toolset_name`, assert on `isError` and on schema
validity, and make a real `IsPIERunning` call. Now: 31/31 described, exit 0.

## 5. Bugs in my own harness (fixed, listed for honesty)
- `ue_log` check asserted "5 lines requested => exactly 5 non-empty lines"; UE logs contain blank lines, so the
  tool was right and the assertion wrong. Rewritten as `0 < lines <= requested` + "all grep results match", and
  moved into its own `log` step (re-run: PASS).
- `ue_build` check required a freshly-written DLL, which a correct **incremental** build never produces. Now
  accepts a fresh DLL *or* `Target is up to date` in the job log (re-run: 4/4 PASS).
- `test_unreal_calls.py` initially parsed sub-bullets of toolset descriptions as toolsets, used the qualified
  tool names from `describe_toolset` for `call_tool` (which wants the bare name), and did not unwrap the
  `{"returnValue": ...}` envelope. All fixed; the script is now the reference for how to call Epic's tools.

## 6. Epic toolset inventory (31 toolsets, 392 tools)

Full machine-readable dump: `_artifacts/unreal_mcp/inventory.json` (names, one-line purpose, argument names);
raw per-toolset JSON schemas in `_artifacts/unreal_mcp/describe/`.

| Toolset | Tools | Purpose |
|---|---|---|
| `EditorToolset.EditorAppToolset` | 21 | Editor state: cvars, viewport capture/camera, actor+asset selection, content browser, PIE control |
| `EditorToolset.LogsToolset` | 4 | Read the output log, list log categories, get/set verbosity |
| `ConfigSettingsToolset.ConfigSettingsToolset` | 8 | List/inspect/edit project & editor settings sections |
| `ToolsetRegistry.AgentSkillToolset` | 4 | List/read/create/update AgentSkills |
| `AutomationTestToolset.AutomationTestToolset` | 7 | Automation test discovery and execution |
| `LiveCodingToolset.LiveCodingToolset` | 1 | Live Coding compile |
| `PluginToolset.PluginToolset` | 17 | Create, edit, enable, query plugins |
| `PhysicsToolsets.PhysicsAssetToolset` | 17 | Create and manage Physics Assets |
| `PCGToolset.PCGToolset` / `.PCGSpatialToolset` | 30 / 1 | Build and modify PCG graphs; spatial helpers |
| `NiagaraToolsets.NiagaraToolset_System` | 46 | Niagara system/emitter/module creation and modification |
| `NiagaraToolsets.NiagaraToolset_Assets` / `_Component` / `_Blueprint` / `_Info` | 3 / 4 / 2 / 1 | Niagara asset discovery, components, BP integration, guidance |
| `editor_toolset.toolsets.blueprint.BlueprintTools` | 53 | Blueprint graphs, nodes, variables, DSL |
| `editor_toolset.toolsets.asset.AssetTools` | 21 | Assets in the project and files on disk |
| `editor_toolset.toolsets.actor.ActorTools` | 17 | Inspect/modify actors and components |
| `editor_toolset.toolsets.scene.SceneTools` | 20 | Current level: spawn, query, organise |
| `editor_toolset.toolsets.material.MaterialTools` / `.material_instance.MaterialInstanceTools` | 22 / 13 | Materials, material functions, material instances |
| `editor_toolset.toolsets.static_mesh.StaticMeshTools` / `.skeletal_mesh.SkeletalMeshTools` | 16 / 22 | Mesh inspection and modification |
| `editor_toolset.toolsets.data_table.DataTableTools` / `.curve_table.CurveTableTools` / `.string_table.StringTableTools` / `.data_asset.DataAssetTools` | 10 / 9 / 8 / 1 | Table and data-asset authoring |
| `editor_toolset.toolsets.texture.TextureTools` / `.primitive.PrimitiveTools` / `.object.ObjectTools` | 2 / 4 / 6 | Textures, primitive components, generic UObject property access |
| `editor_toolset.toolsets.programmatic.ProgrammaticToolset` | 2 | Batches calls to other tools through a small sandboxed Python interpreter |

`EditorToolset.EditorAppToolset` tools (the ones this project will use most):
`CaptureViewport`, `CaptureAssetImage`, `GetCameraTransform`, `SetCameraTransform`, `FocusOnActors`,
`GetSelectedActors`, `SelectActors`, `GetVisibleActors`, `GetSelectedAssets`, `SelectAssets`, `GetOpenAssets`,
`OpenEditorForAsset`, `GetContentBrowserPath`, `SetContentBrowserPath`, `SearchCVars`, `StartPIE`, `StopPIE`,
`IsPIERunning`, `WorldPosToScreenCoords`, `ScreenCoordsToWorld`.
`EditorToolset.LogsToolset`: `GetLogEntries`, `GetLogCategories`, `GetVerbosity`, `SetVerbosity`.
`ConfigSettingsToolset`: `ListContainers`, `ListCategories`, `ListSections`, `GetSectionSchema`,
`GetSectionPropertyValues`, `SetSectionProperties`, `SaveSection`, `ResetSectionToDefaults`.

### 6.1 How to call (verified)
`call_tool(toolset_name=<dotted toolset>, tool_name=<bare name>, arguments={...})`. `describe_toolset` takes
`toolset_name` and reports **qualified** tool names (`EditorToolset.EditorAppToolset.IsPIERunning`) - strip the
prefix for `tool_name`. Argument names in the schemas are camelCase (`captureTransform`, `maxEntries`).
Every result is text JSON shaped `{"returnValue": ...}`.

### 6.2 Quirks future sessions must know
- `list_toolsets` output is text, and toolset **descriptions contain their own `- ` bullets**. Only dotted names
  (`Plugin.Toolset`) are real toolsets; naive line parsing invents 15 fake ones.
- `GetLogEntries`: `pattern` is required, and the schema's default for `category` is literally `"LogsToolset"`,
  so omitting `category` fails with `Log category 'LogsToolset' not found`. Always pass `category: ""`.
- `CaptureViewport`: although `captureTransform` and `annotations` are optional in C++, calling with `{}` fails
  ("input param `captureTransform` needs a default value", then the same for `annotations`). Pass an explicit
  transform (e.g. from `GetCameraTransform`) and an annotations object; `{"gridSpacing":0,"gridExtent":0,
  "gridHeight":0,"maxLabelDistance":0,"classFilter":{"refPath":"/Script/Engine.Actor"},"maxLabels":0}` disables
  the overlays.
- `serverInfo.name`/`version` are empty strings in `initialize`.
- The server exists only while the editor runs; after an editor restart Claude Code must reconnect the `unreal`
  server (`/mcp`).

## 7. Open issues / recommendations
1. **An editor launched by the server dies with a Python-SDK client.** `stdio_client` puts the server in a
   Windows Job Object with `KILL_ON_JOB_CLOSE` and no breakaway, so the editor (a grandchild) is killed when the
   test client exits. That is why `test_engine.py` runs `editor` and `close` in one session and offers
   `--hold-file`. Claude Code's spawn path is not affected in the same way (its editor survived every step here),
   but if an editor ever disappears when a session ends, this is the reason.
2. **Memory is the binding constraint.** Cold build with 6 parallel actions dipped to 0.9 GB free; editor commit
   is ~5.2 GB and system commit reached 28.7/30.1 GB. Do not run the editor and a packaged sim at once, and keep
   `-MaxParallelActions=6` (or lower). A system-managed pagefile on D: is still the recommended user action.
3. **Set `UBA_ROOT` (and the D: cache vars) at user level** so IDE/VS builds also stay off C:. The server now
   compensates for its own children only.
4. **AirSim rebuilds from source and without unity files** (adaptive unity excludes every file because they are
   writable/not in source control). Cold editor builds will always cost ~15 min. If that becomes painful,
   marking the plugin read-only or configuring `bUseAdaptiveUnityBuild=false` would restore unity builds.
5. **UBA listens on `0.0.0.0:1345`** during builds (local executor). Harmless on a laptop, worth knowing.
6. **`ue_package` was never run to completion** (killed by design after proving it starts). A real cook/package
   still needs a first full run - budget disk and time.
7. The server Claude Code currently has running was started **without** the `.mcp.json` `env` block
   (`SIGHTLINE_UPROJECT` absent). The defaults resolve to the same paths, so it works, but if `.mcp.json` env
   entries are ever changed, that session must be restarted to pick them up.
8. `sim_launch_game` (AirSim section, not modified here) benefits automatically from the env fixes since they are
   applied to `os.environ` at startup.
