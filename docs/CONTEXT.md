# Project context (living document)

What a new session needs to know that is **not** obvious from the code. Keep it factual; date every change.
The build plan itself lives in `SOLUTION_DOC.md`; progress lives in `TRACKER.md`.

**Phase status (end of session 2, 2026-09-10):** environment and both MCP servers are validated end to end (see
§4a and `docs/verification/`). Development has started: the FloodValley scene foundation (F1) is built and
verified in PIE (terrain, zones, water + FloodLevel API, lighting, georeferencing), materials are built but their
final look is not yet verified. Scene facts are in §7 "FloodValley scene facts" and "Material scripting".
The build document was updated to a **simulation-first** version (§5 decisions). New sessions: read
`docs/HANDBOOK.md`, then `docs/TRACKER.md` "Handoff state".

## 1. The project in one paragraph
PS2 "Real-time vision system for identifying survivors in flood, landslide and tsunami zones". We simulate a
debris-flow-fed monsoon flash flood in a hill-valley settlement (Wayanad/Chaliyar type, three zones: deposit
fan, flooded settlement, channel) in Unreal Engine 5.8 + Cosys-AirSim. A SimpleFlight drone flies coverage
patterns autonomously with gamepad takeover and streams RGB + thermal + labels + telemetry into a vision pipeline
(tiled YOLO26 RGB + thermal late fusion, BoT-SORT, geolocation, geo-dedup, triage scoring, search-quality/POD
map, offline outbox, MapLibre C2 map, evaluation harness). Requirements R1-R12 + A13/A14 are in SOLUTION_DOC §1.1;
features F1-F21 in §8; day plan in §9; day-1 unknowns in §10.

## 2. Machine (verified 2026-09-10)
| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU | Intel i7-14650HX, 16C/24T |
| RAM | 15.7 GB. **Binding constraint.** Pagefile: 9 GB on C: -> commit limit ~24.7 GB |
| GPU | RTX 4060 Laptop, 8 GB VRAM, driver 592.82 (Optimus: GPU sleeps when idle) |
| Disks | C: ~46 GB free (keep it that way), D: ~190 GB free (everything goes here) |
| Gamepad | none detected at setup time (needed for F3 takeover tests) |

## 3. Paths and toolchain
| Component | Version | Location | Notes |
|---|---|---|---|
| Unreal Engine | 5.8.2 (CL 56702186) | `D:\UE_5.8` | Launcher install |
| Visual Studio | 2022 Community 17.14 (MSVC 14.44) | `D:\VS\2022\Community`; cache `D:\VS\Cache`; shared `D:\VS\Shared` | UE 5.8 accepts MSVC 14.44.35211+ or 14.50.35723+ (`Engine/Config/Windows/Windows_SDK.json`). Cosys 5.8 binaries were built with VS 2022 17.14.23 (solution doc said VS 2026: superseded) |
| Windows SDK | 10.0.22621 + 10.0.26100 | `C:\Program Files (x86)\Windows Kits\10` (pre-existing) | UE MainVersion 22621 |
| Cosys-AirSim | 5.8-v3.4.1 (17 Jul 2026) | plugin -> `sim/SightlineSim/Plugins/AirSim`; zips in `_downloads/` | re-fetch with `tools/setup/fetch_airsim.ps1` |
| uv | 0.12.12 | `D:\Tools\uv\uv.exe` | |
| Python (project) | 3.11.16 (uv-managed) | `D:\Tools\uv-python`; venv `D:\Sightline\.venv` | cosysairsim declares <=3.11 |
| MCP Python SDK | 1.30.0 (pinned) | venv | last 1.x; 2.x exists but its API differs from FastMCP 1.x |
| Git | system | repo at `D:\Sightline` (branch `main`, no remote yet) | |

**Cache redirection (user-level env vars, set 2026-09-10):** `UV_CACHE_DIR=D:\Tools\cache\uv`,
`UV_PYTHON_INSTALL_DIR=D:\Tools\uv-python`, `UV_PYTHON_BIN_DIR=D:\Tools\uv-python\bin`,
`UV_TOOL_DIR=D:\Tools\uv-tools`, `UV_TOOL_BIN_DIR=D:\Tools\uv-tools\bin`, `PIP_CACHE_DIR=D:\Tools\cache\pip`,
`HF_HOME=D:\Tools\cache\hf`, `TORCH_HOME=D:\Tools\cache\torch`, `YOLO_CONFIG_DIR=D:\Tools\cache\ultralytics`,
`npm_config_cache=D:\Tools\cache\npm`, **`UE-LocalDataCachePath=D:\UE_Cache\DDC`** (redirects both the local DDC
and the Zen store: `BaseEngine.ini [Zen.AutoLaunch] LocalDataCachePathEnvOverride`). New shells pick these up;
already-open terminals/apps (including Claude Code itself) must be restarted to see them.

## 4. MCP architecture (how Claude drives Unreal)
- **`unreal` (Epic, built into UE 5.8)**: plugin `Engine/Plugins/Experimental/ModelContextProtocol` ("Unreal MCP",
  experimental). Streamable HTTP server `http://localhost:8000/mcp`, bound to loopback (HTTPServer default
  `BindAddress=localhost`, restated in `DefaultEngine.ini`). Auto-start via
  `Config/DefaultEditorPerProjectUserSettings.ini [/Script/ModelContextProtocolEngine.ModelContextProtocolSettings] bAutoStartServer=True`.
  Console: `ModelContextProtocol.StartServer [port]`, `.StopServer`, `.RefreshTools`. With `bEnableToolSearch=True`
  the server exposes `list_toolsets`, `describe_toolset`, `call_tool`; the tools come from the Toolset Registry
  plugins enabled in the `.uproject` (EditorToolset, ConfigSettings, Plugin, AutomationTest, LiveCoding,
  Physics, Niagara, PCG). EditorToolset includes CaptureViewport, Start/StopPIE, Get/SetCameraTransform,
  SelectActors, GetLogEntries, SearchCVars and more. It has **no Python-execution tool**.
- **`sightline` (ours, `tools/sightline_mcp/server.py`, stdio)**: fills the gaps. Editor Python goes through
  Epic's remote-execution protocol (UDP multicast 239.0.0.1:6766 discovery, TCP command channel on
  127.0.0.1:6776), re-implemented in `ue_remote.py` because the engine's reference client truncates large
  results. Requires `bRemoteExecution=True` in `DefaultEngine.ini` (set). AirSim via `cosysairsim` RPC on 41451.
- Both are registered in `D:\Sightline\.mcp.json`. After changing `.mcp.json`, restart Claude Code.
- Epic server protocol facts (read from source, `ModelContextProtocol.h` / `ModelContextProtocolServer.cpp`):
  supported protocol versions `2025-11-25`, `2025-06-18`, `2024-11-05` (negotiated per session); sessions via
  the `Mcp-Session-Id` header (missing = 400, unknown = 404 "client should reinitialize", which happens after
  an editor restart); the `Mcp-Protocol-Version` header must match the negotiated version; `Origin` must be
  absent or localhost (DNS-rebinding guard, else 403); `GET /mcp` returns 405 (no standalone SSE stream,
  allowed by the spec); tool calls with a progress token stream `text/event-stream`, others return one
  `application/json` response. **After an editor restart Claude Code must reconnect the `unreal` server**
  (run `/mcp` in the terminal UI, or restart the session).
- Conformance test with the same client transport as Claude Code: `tools/sightline_mcp/test_unreal_mcp.py`.
- Our client library `mcp==1.30.0` negotiates `2025-11-25` (seen in the smoke test).

## 4a. Verified capability inventory (2026-09-10)
- **Epic `unreal` server: 31 toolsets, 392 tools** (full inventory in `docs/verification/engine_tools.md`). Verified
  real calls: IsPIERunning, GetSelectedActors, SearchCVars, GetCameraTransform, CaptureViewport (1267x688 PNG),
  GetLogEntries/GetLogCategories, ConfigSettings reads. Gotchas: `GetLogEntries` needs an explicit `category: ""`;
  `CaptureViewport` rejects `{}` (needs `captureTransform` + `annotations`); toolset catalog is TEXT, not JSON.
- **sightline server: 27 tools**, all verified. Engine side in `engine_tools.md`, AirSim side 39/39 in
  `_artifacts/verification/sim_tools_*.json`, protocol robustness in `mcp_protocol.md` (20 pass / 2 skip).
- **Claude Code CLI drives both** (`claude_code_integration.md`): real `claude -p` calls returned live tool data;
  `unreal` fails fast and cleanly when the editor is down without affecting `sightline`.
- **Timings:** cold editor build 881 s (rebuilds all of Cosys-AirSim), incremental 20 s, editor ready ~44 s,
  editor ~5.15 GB private commit. Packaged Blocks: RPC up ~6 s after launch, ~1.6 GB.
- **Gamepad (wired Xbox 360, XInput slot 0):** full stick travel, both triggers, 9 buttons seen via XInput and
  pygame. AirSim-side `rc_data` + takeover handover: see `dev_workflow.md`.
- **Landing/RTL:** own position-held controller + disarm; RTL touchdown measured 0.2 mm from home.

## 5. Decisions and deviations from the solution doc
| Date | Decision | Why |
|---|---|---|
| 2026-09-10 | VS 2022 17.14 instead of VS 2026 | Cosys-AirSim 5.8-v3.4.1 release notes: built with VS 2022 17.14.23; UE 5.8 lists 14.44 as preferred |
| 2026-09-10 | Project `SightlineSim` created from the Cosys Blocks editor project (module renamed, redirects in DefaultEngine.ini) | doc §5.1 step 1 ("start from Blocks") with our own module name |
| 2026-09-10 | Low-memory renderer profile in `DefaultEngine.ini`: no Lumen/HWRT/VSM, SSR, TSR, Streaming pool 2500 | doc §4 |
| 2026-09-10 | AirSim settings passed with `-settings=<repo path>`, never `Documents\AirSim` | reproducible, versioned per scenario (doc §5.1 "a settings.json per scenario") |
| 2026-09-10 | Python 3.11 single env; `opencv-python` only | cosysairsim classifiers stop at 3.11; `opencv-contrib-python` (listed in Cosys requirements.txt, not in the wheel's deps) collides with Ultralytics' opencv-python |
| 2026-09-10 | Epic's MCP plugin plus our own server, instead of third-party UE MCP plugins | first-party, ships prebuilt with 5.8.2, no extra C++ to maintain; ours covers lifecycle, builds and AirSim |
| 2026-09-10 | `OriginGeopoint` 11.4870 N, 76.1450 E, **1060 m** (approximate Chooralmala/Mundakkai valley) | doc §5.1 "set OriginGeopoint to the Wayanad valley". Altitude corrected from the guessed 900 m to the Copernicus GLO-30 value at that point, measured with dem-stitcher (docs/verification/python_stack.md §7). **Superseded below** |
| 2026-09-10 | **`OriginGeopoint` = 11.4870 N, 76.1450 E, 1046.007 m** (map centre at the ASL height of UE world Z = 0) | **Cosys anchors OriginGeopoint at the UE world origin, not the PlayerStart** (measured: vehicle GPS = OriginGeopoint + PlayerStart offset, to 0.3 m). The FloodValley terrain is synthetic; its lowest point (`base_z_m` in `data/scene/flood_valley.json`) is world Z = 0. Verified: drone on the pad reads 11.4883295 N / 76.1495101 E = the pad's computed geopoint |
| 2026-09-10 | Simulation-first (doc §5.5c, new in the updated SOLUTION_DOC): the renderer is the deployment domain; demo model F8b trained on sim frames only, randomisation OFF by default, nominal slice 40-60 m / daylight / occlusion < 50 %, splits by scenario seed; every number carries a `sim`/`real` domain column | user's updated build document (root `PS2-Survivor-Vision-Research-and-Build-Document.md`, copied to `docs/SOLUTION_DOC.md`) |
| 2026-09-10 (session 3) | **The simulator is the only deployment domain: optimise the model FOR the renderer, not for real imagery.** F8b (sim-only fine-tune, randomisation OFF, splits by scenario seed) is the model. **F8c (transferable real+synthetic model) is dropped from the critical path** and no real datasets are pulled. Training is deliberately minimal — the model only has to generalise across seeds/altitudes/times of day inside this renderer, so a short fine-tune is the correct amount of work, not a compromise | user instruction, session 3: "you are supposed to optimize for your simulated environment including the model, not for the real images or real scenario, so with very minimal training it can work". Consistent with doc §5.5c |

## 6. Verified facts about Cosys-AirSim 5.8-v3.4.1 (from repo docs, 2026-09-10)
- Settings search order: `-settings="abs path"` or `-settings={json}` > exe dir > launch dir > `Documents\AirSim`.
- ImageType: Scene 0, DepthPlanar 1, DepthPerspective 2, DepthVis 3, DisparityNormalized 4, Segmentation 5,
  SurfaceNormals 6, Infrared 7, OpticalFlow 8, OpticalFlowVis 9, Lighting 10, Annotation 11.
- **Infrared is a per-object ID -> grey 0-255 map** (object ID 42 renders (42,42,42)); IDs set with
  `simSetSegmentationObjectID(mesh, id, is_regex)`. Realistic thermal (per-material T and emissivity, diurnal
  state, immersion cooling) therefore needs the post-process thermal capture of doc §5.1 step 7.
- Instance segmentation APIs: `simListInstanceSegmentationObjects()`, `simListInstanceSegmentationPoses(ned, only_visible)`,
  `simGetSegmentationColorMap()`.
- CaptureSettings carries Lumen toggles (`LumenGIEnable`, `LumenReflectionEnable`); keep them false (doc §4).
- Python client wheel deps: `numpy`, `rpc-msgpack` only.

## 7. Known pitfalls
- **Memory**: at setup, free RAM was ~2.3-2.8 GB with browsers, Claude and the VS installer open, and commit
  23.5/24.7 GB. The editor needs 8-11 GB. Close browsers before editor sessions, and never train while the
  engine runs (doc §4). **Recommended user action:** add a system-managed pagefile on D: (System > Advanced >
  Performance > Virtual memory). This is a system setting, so Claude must not change it.
- The first editor launch compiles shaders for a long time (budget 20-40 min). Use `editor_launch(wait_ready_s=...)`
  or poll `status()`.
- UBT cannot build the editor target while that editor is running with Live Coding. Close it first.
- Never let MCP-server child processes inherit stdin/stdout (they are the protocol pipes); `server.py` uses `DEVNULL`.
- **MCP stdio servers on Windows: import every native-extension module (numpy, cv2, torch, PIL, cosysairsim) at
  startup, never lazily inside a tool.** The stdio transport holds a synchronous read on the stdin pipe, and a
  later DLL load (numpy's C-runtime init) waits on that handle, so the tool call hangs forever. Diagnosed
  2026-09-10 with `tools/day1/diag_mcp_hang.py` (faulthandler stack dump). This applies to every future MCP tool.
- Never `print()` to stdout inside a stdio MCP server (stdout is the JSON-RPC channel). cosysairsim prints on
  connect, so `server.py` redirects stdout to stderr around all AirSim and editor calls.
- The cosysairsim client is used from one dedicated thread (`_SIM_THREAD`); tools are async wrappers (`@threaded`)
  so long flights and builds never block the MCP event loop.
- **SimpleFlight API watchdog = 60 ms** (`Plugins/AirSim/.../simple_flight/firmware/Params.hpp:136 api_goal_timeout`).
  Airborne under API control with no new setpoint for 60 ms, it logs "API call was not received, entering hover mode
  for safety" and holds position. It fires after every completed command, so it is expected and harmless. Mission code
  that needs continuous motion must stream setpoints (re-issue velocity/position commands) or accept a hold between
  commands.
- **`landAsync` is broken in Cosys-AirSim 3.4.1**: `MultirotorApiBase::land()` treats `z_vel <= approx_zero_vel` as
  "landed" for 10 ticks, so it returns at altitude before descending (measured: returned in 0.2 s at 21 m) and the
  watchdog then hovers there. `landed_state` only becomes Landed on the ground with motors below armed throttle
  (`OffboardApi.hpp:213-228`). **Never use landAsync.** Use the sightline `sim_fly land/rtl` controller
  (`_land_controller` in server.py: streamed descent setpoints plus verified touchdown) or reuse that function.
- `simPause` freezes physics and pose; under ScalableClock the state timestamp keeps following wall time.
  `simContinueForTime(t)` advances and re-pauses. Static level actors cannot be moved (`simSetObjectPose` -> False);
  spawn movable objects with `simSpawnObject` (asset names from `simListAssets`).
- SimpleFlight braking from 6 m/s overshoots ~2 m and settles in ~7-8 s. Allow settling before precise captures.
- winget may treat a failed VS install as installed; re-run the VS bootstrapper directly (see `docs/SETUP.md`).

**FloodValley scene facts (measured 2026-09-10, session 2):**
- **UE's OBJ importer flips handedness**: an OBJ with x = east, y = north lands as UE +X = east, UE -Y = north
  (checked with line traces against the height grid at three asymmetric points). The terrain actor carries
  **yaw +90 deg**, which gives AirSim's NED convention exactly: UE X = north, UE Y = east. Spawners must use
  `UE (X, Y, Z) cm = (north*100, east*100, (asl - base_z)*100)` (also in `data/scene/flood_valley.json` `ue_import`).
- **The OBJ must be written in centimetres**: UE does not rescale OBJ units (a metre OBJ imported 100x too small).
- **Imported meshes come in with Nanite ON**; `get_num_triangles` then reports the fallback mesh (347 tris for a
  524k-tri terrain). Turn Nanite off on import (doc §4) and set `CTF_USE_COMPLEX_AS_SIMPLE` for terrain collision.
- **The terrain actor's object name is `Ground`**: `sim_fly` treats contact with "Ground" as normal (pad, takeoff,
  touchdown). Any other name makes every takeoff from the terrain fail as a collision. From the next sightline
  server start, a `Ground` contact with a normal > ~45 deg off vertical (a hillside strike) fails as well.
- **Cosys uses the actor object name** (`GetName()`), not the editor label, for `simListSceneObjects` / poses /
  collision reports. Rename actors (`actor.rename(label)`) when a stable API name matters (`Ground`, `FloodWater`).
- **Editor Python has no numpy.** Do grid maths on the host (`uv run`) and pass numbers in.
- `LevelEditorSubsystem.editor_request_end_play()` is asynchronous: in the same script, `get_all_level_actors()`
  still returns the PIE world and `save_current_level()` returns False. End PIE in one call, edit in the next.
- AirSim settings files are re-read on every PIE start (a changed OriginGeopoint took effect without an editor
  restart).
- **"Use Less CPU when in Background" lives in `UEditorPerformanceSettings`**, config section
  `[/Script/UnrealEd.EditorPerformanceSettings]` in `Config/DefaultEditorSettings.ini` (source:
  `Editor/UnrealEd/Classes/Editor/EditorPerformanceSettings.h:74`). The old override in
  `DefaultEditorPerProjectUserSettings.ini [/Script/UnrealEd.EditorPerProjectUserSettings]` was silently ignored.
  Fixed 2026-09-10; takes effect at the next editor start.
- **Red viewport text "A sky light with real-time capture enabled ... requires at least a SkyAtmosphere"**: the
  legacy BP_Sky_Sphere dome is not an `IsSky` mesh, so a real-time SkyLight captures black. FloodValley now has a
  SkyAtmosphere (sun flagged `atmosphere_sun_light`) as the visible sky; BP_Sky_Sphere stays only because Cosys'
  `simSetTimeOfDay` finds the sun through its "Directional light actor" property, and is hidden.
- **GPU is used, not the iGPU** (diagnosed 2026-09-10 on a user report of CPU 91 % / NVIDIA 2 % / Intel 20 %):
  UE's RHI picks the RTX 4060 (log: CsvProfiler gpu="NVIDIA GeForce RTX 4060 Laptop GPU", texture pool from its
  6.5 GB); `nvidia-smi` lists UnrealEditor.exe as a C+G client. The Intel iGPU drives the display (Optimus), so
  desktop compositing and frame copies show there. The CPU spike was imports/texture compression/asset unzipping,
  all CPU-bound. Under PIE with the bare scene the 4060 ran 75-82 % busy at **P5, ~750 MHz, 8-14 W of 40 W**, throttle
  reason `Idle` only (AC power, Windows "Best performance" overlay): the load is too light to boost, not a cap.
  Re-measure once the scene is dressed; `nvidia-smi -q -d PERFORMANCE` gives the throttle reasons.

**Material scripting through Python (measured 2026-09-10):**
- Deleting a material that a level actor references fails (`EnsureFailed ... ForceDeleteObjects` callstack in the
  log) and `create_asset` then returns None. Reuse the asset and clear its graph instead.
- `MaterialEditingLibrary.delete_all_material_expressions` does **not** clear everything (a terrain graph grew
  64 -> 95 -> 111 nodes over three reruns). Delete each node from `get_material_expressions()` and assert the count
  is 0; `delete_unused_expressions` strips leftovers from an existing asset.
- `BlendAngleCorrectedNormals` is a material *function*, not a `MaterialExpression*` class; a script that dies
  mid-graph leaves a material that fails to compile (log: "Failed to compile Material ... Default Material will be
  used") and renders as the grey checker. Sum + Normalize is the cheap substitute.
- Check a built material with `get_material_property_input_node(mat, MP_*)` and the used-texture count: a
  compile failure shows as 0 used textures even though the nodes exist.
- Textures, saves and material recompiles are unreliable while PIE runs (imports come back as 32 px
  placeholders, saves return False). Build with PIE off.
- The survey camera sits at NED (0, 0, 0.30) below the body: at (0.30, 0, 0.15) part of the airframe showed as a
  blurred dark blob in a nadir frame corner.
