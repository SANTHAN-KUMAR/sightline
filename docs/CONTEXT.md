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
- **`unreal.Transform.rotation` is an FQuat, not an FRotator** (UE 5.8). Assigning a `Rotator` to it raises;
  worse, a mock that models it as a Rotator passes offline and the editor then rejects it. Build the transform
  with `set_editor_property("rotation", rot.quaternion())` and read it back with `q.rotator()`. Also note the
  CONSTRUCTOR order is `(rotation, translation, scale)` while the property names are
  `rotation / translation / scale3d`. Every script that builds instance transforms proves the builder
  round-trips one known value before using it (see `build_rubble.py`, `build_vegetation.py`).
- **`unreal.LinearColor` is LINEAR, and a colour picked as if it were sRGB will be far too bright.** Linear 0.74
  is sRGB 0.885 - near white. This washed the entire rubble field to featureless white paper on 2026-09-11
  while every programmatic check passed; only a render showed it. Convert explicitly:
  `linear = ((srgb + 0.055) / 1.055) ** 2.4` for srgb > 0.04045. When judging a material's albedo, measure the
  source scan's own linear mean from the file in `_downloads` rather than guessing.
- **A metre-true UV tile the same size as the mesh face makes every instance identical.** `gen_rubble.py` writes
  UVs in tiles of `TILE_M` (concrete 2.0 m); most slab faces are 1-3 m, so at `UVScale = 1.0` each face mapped
  one whole copy of the 2K scan and every plate in the field wore the same stain, plainly visible at the 45 m
  survey altitude. Put several tiles across a face so the scan reads as grain, not as a motif.
- **`gen_rubble.py` computes UVs per polygon from that polygon's own first edge**, so a fan-triangulated face
  gets a different texture orientation per triangle - a quilt of wedges visible at close range. Not fixed; it
  needs a change to the UV construction and a regeneration of all 40 OBJs.
- **The level can be stale with respect to its own layout JSON.** On 2026-09-11 the level held 73 `House_###`
  actors while `settlement.json` declared 76, because `gen_buildings.py` was re-run and `build_buildings.py`
  was not. Anything generated from the CURRENT settlement (here `damage.json`) then references actors that do
  not exist. **Do not fix this by re-running `build_buildings.py`**: it rebuilds `M_PBR_Master` from an empty
  graph and `build_roof_materials.py` clears and rebuilds that same graph to add anti-tiling and the tide line,
  so a re-run silently drops both. `tools/scene/place_missing_houses.py` places only the missing ids and
  touches no material graph.
- **`str()` of a UE enum is `'<BlendMode.BLEND_OPAQUE: 0>'`, so `str(v).split(".")[-1]` yields
  `'BLEND_OPAQUE: 0>'`** — which then compares equal to nothing and silently fails every check that reads it.
  This produced false FAILs on blend mode and would have produced a false FAIL on the shading model *after* a
  correct fix. Use `v.name`, with `str(v).strip("<>").split(":")[0].split(".")[-1]` as the fallback
  (`ename()` in `build_foliage_materials.py` / `qa_foliage.py`).
- **`SceneCaptureComponent2D.hidden_actors` cannot be set from Python**: the editor refuses it with
  *"Property 'HiddenActors' ... cannot be edited on templates"*, even though `capture_source`, `fov_angle` and
  `texture_target` all set fine on the same component. Use the BlueprintCallable functions instead —
  `comp.hide_actor_components(actor)` and `comp.clear_hidden_components()`. This is the mechanism
  `qa_foliage.py` uses to render a matched no-vegetation pass for masking.
- **A material instance's `base_property_overrides` can say one thing while the renderer does another.**
  `build_vegetation.py` set `override_blend_mode=True, blend_mode=BLEND_OPAQUE` on all 5 leaf instances; the
  dump read `BLEND_OPAQUE` back; and the canopy still rendered blended, with sky visible through the alpha
  cut-outs and a white haze of accumulated card layers over every crown. **Do not treat an override as
  evidence of what is drawn** — look at the pixels. Note also that `get_base_material()` returns the ROOT
  `UMaterial`, skipping any intermediate instance, so a chain like
  `leaf -> MI_Default_Blend_DS -> M_Default` reports the ROOT's opaque blend and hides the translucent link
  in the middle. Walk `get_editor_property("parent")` to see the whole chain.
- **The tree meshes in the project are `<pid>_2k`, not the `<pid>_2k_lite` that `vegetation.json`'s
  `asset_hint` points at.** The hint misses and `resolve_mesh()`'s folder scan silently picks the right asset,
  so everything works — but any script that trusts the hint, or that resolves differently, would author onto
  an asset nothing draws. `build_foliage_materials.py` cross-checks the resolved mesh against the
  `static_mesh` each HISM component actually holds before touching anything.
- **The Interchange glTF importer renames textures unpredictably.** Poly Haven's `<mat>_arm_2k.jpg` arrives as
  `<mat>_rough` (named for its glTF metallicRoughness role, not its filename), and a leaf base colour arrives
  as `<mat>_diff-<mat>_alpha` because the importer composes an alpha channel in. Matching on `_arm` alone
  finds nothing; match `_rough` too, and never assume the source filename survives import.
- **Re-prefixing a name is not idempotent.** `f"MI_{slot_material_name}"` is fine on the first run and creates
  `MI_MI_<name>` on the second, because the slot now holds the material the first run made — a full duplicate
  set of 15 materials with the previous set orphaned. Strip the prefix before adding it
  (`canonical()` in `build_foliage_materials.py`, which also sweeps strays out of the folder afterwards).
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

**Utilities / water / QA facts (measured 2026-09-11, session 4, utilities lane):**
- **`ue_python` mode `file` is `RunFile`, not `exec`.** It takes the FIRST WHITESPACE-DELIMITED TOKEN of the
  string as a filename, so any multi-line or space-containing snippet fails with "Could not load Python file
  'D:/UE_5.8/Engine/Binaries/Win64/<your code>'". Write the script to a file and send the single expression
  `exec(open(r'...').read())` - note it must contain **no spaces**. To pass a flag, write a two-line wrapper
  file that sets it and then execs the target.
- **`ue_python` exec()s into ONE persistent namespace that survives between calls.** A module-level flag set
  by one call (e.g. `SELFTEST=True`) is still set for the next one, so a script that reads
  `globals().get("FLAG")` silently keeps the previous run's mode. Consume such flags with `globals().pop(...)`.
- **`ProceduralMeshLibrary.get_section_from_static_mesh(mesh, lod, section)` returns plain `unreal.Vector`
  positions**, not vertex structs - `v.position` raises `AttributeError`. It is the way to verify placement
  from the actual geometry: an actor BOUNDS check (which `build_utilities.py` does) is satisfied by a mesh
  that is mirrored, rotated 180 deg, or scaled about its centre.
- **`ctypes.windll.kernel32.GetCurrentProcess()` truncates on 64-bit** unless you set
  `restype = wintypes.HANDLE`; the -1 pseudo-handle becomes 0xFFFFFFFF and `GetProcessMemoryInfo` then
  silently reports **0.0 for every field** rather than failing. Any memory number that comes back as exactly
  0.0 GiB is this bug, not an idle process.
- **Judge wire/thin-line continuity only at the real survey GSD** (1.7656 cm/px). The 3.6 cm conductors are
  2.04 px there and render as continuous lines; in a wider frame they fall below 1 px and alias into dashes.
  The dashed lines visible on the water in oblique frames are the wires' SHADOWS, not the wires.
- **`POLE_BURY_M = 1.2`** (`gen_utilities.py:81`): pole geometry deliberately continues 1.2 m below ground so
  a pole on a slope never shows a gap. A placement check that expects shafts to start at ground level fails
  all 60 poles by exactly 120 cm.
- **Water: the single-layer-water volume is what makes depth readable, and the coefficient that matters is
  the DEPTH SCALE, not the hue.** Scattering and Absorption are per-cm; dividing both by the same factor
  leaves the single-scattering albedo `w = S/(S+A)` - and therefore the colour - completely unchanged while
  moving how far you can see into the water. `tune_water.py` now carries `TURBIDITY_DIVISOR` for exactly
  this. Sanity target from `K_d ~= 1.44 / z_SD`: Secchi ~1 m gives ~1.4 /m in the green, which reads as
  green-teal water with the drowned ground visible in the shallows and opaque beyond ~3 m.
- **A build script's printed summary can be a hard-coded literal.** `tune_water.py` reported "extinction /m:
  clear R 6.2 G 4.1 B 5.0" about a material it had just written with 2.07/1.37/1.67, because those two lines
  were f-strings with the numbers typed in. Derive report lines from the values actually written.
- **An inverted self-test is worth writing.** `check_utilities_placed.py`'s `SELFTEST=True` injects a 50 cm
  pole shift and expects failure; it immediately proved the check's pole assertion was insensitive to a 50 cm
  error, because "geometry exists within 60 cm" is not a position test. Measuring the shaft-axis centroid is.

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
- **Anything placed in the thousands MUST be instanced, not spawned** (measured 2026-09-11, vegetation lane).
  One `StaticMeshActor` per item died inside `EditorActorSubsystem.SpawnActorFromObject` at 5,508 items with
  `UsedVirtual 27.48 GiB` / `AvailableVirtual 0.00 GiB` - **commit** exhaustion, not RAM (this machine has
  15.7 GB of RAM). The same 5,508 placed as instances on 9
  `HierarchicalInstancedStaticMeshComponent`s cost **no measurable commit at all** (`UsedVirtual` 7.49 GiB
  before the first instance and 7.49 GiB after the last) and took under a second. Instances are safe for
  anything that does not need its own instance-segmentation ID - trees, ferns, debris - and unsafe only for
  `Human_*` / `Animal_*`, which must stay one actor each.
- **UE 5.8's Python bindings do NOT expose `Actor.add_component_by_class`** (it is absent from
  `dir(unreal.Actor)`, despite being a Blueprint node). To add a component to a level actor from Python:
  `sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)`,
  `handles = sds.k2_gather_subobject_data_for_instance(actor)`, then
  `sds.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=handles[0], new_class=..., blueprint_context=None))`,
  which returns `(handle, fail_reason)` - the fail reason is an EMPTY STRING on success. That is the Details
  panel's "+ Add Component" path; the component is an INSTANCE component and **does serialise into the .umap**
  (verified by writing 3 instances, saving, reloading the map from disk and reading all 3 back). A bare
  `unreal.Actor` spawned via `spawn_actor_from_class` already has a `DefaultSceneRoot`, so `handles[0]` (the
  actor) is the right parent.
- **ISM/HISM Python API shapes** (5.8): `add_instances(transforms, should_return_indices, world_space=False,
  update_navigation=True)` and `get_instance_transform(index, world_space=False) -> Transform or None` - a bare
  Transform, **not** a `(bool, Transform)` tuple. Pass `world_space=True` explicitly when the transforms are
  world coordinates. A Rotator -> Quat -> Rotator round trip normalises yaw into [-180, 180], so a planned yaw
  of 300 reads back as -60: compare yaw modulo 360, never by subtraction.
- **`ctypes` memory readings silently come back as ZERO** unless `kernel32.GetCurrentProcess.restype` is set to
  `c_void_p` first. Without it the (HANDLE)-1 pseudo-handle is truncated to 32 bits and `GetProcessMemoryInfo`
  fails, filling every field with 0 - which looks like a real reading. Set the restype and the argtypes, and
  check the return code. The fields then map exactly onto UE's own `FWindowsPlatformMemory::GetStats()`:
  `AvailablePhysical = ullAvailPhys`, `AvailableVirtual = ullAvailPageFile`, `UsedPhysical = WorkingSetSize`,
  `UsedVirtual = PagefileUsage`, so they are directly comparable with a fatal-error report.
- **Poly Haven tree leaf materials: force Opaque, never Masked.** The glTFs declare `alphaMode: BLEND` on every
  leaf slot, but every leaf texture is a **JPEG**, and JPEG has no alpha channel - there is no opacity source
  anywhere in these assets. The leaves are real scanned geometry, not alpha-cut cards (confirmed in a close-up
  render: individual serrated compound leaves with veining). Translucent would break Nanite, the segmentation
  mask and overdraw; masked would have nothing to sample. Note `unreal.BlendMode.BM_OPAQUE` does not exist in
  UE 5.8 - it is `BLEND_OPAQUE`.
- **A leaning instance's direction can be measured instead of argued**: transform a local point up the trunk
  with `unreal.MathLibrary.transform_location(instance_transform, unreal.Vector(0, 0, bbox.max.z))` and take
  the azimuth of the world XY offset. That uses the engine's own rotation convention rather than reasoning
  about UE's left-handed pitch/yaw order. FloodValley's drowned trees measure 178.9 deg (south, downstream).
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

**Cosys-AirSim capture facts (measured 2026-09-10, session 3). Each of these fails SILENTLY.**
- **`simGetImages(...).image_data_uint8` is RGB, not BGR.** Proven two ways: the silt-brown flood surface gives
  channel means 199/182/154, and reversing the segmentation palette attributes the centre of a 45 m frame to a
  survivor 283 m away while not reversing attributes it to a house 2 m away. OpenCV is BGR, so a raw buffer
  handed to `cv2.imwrite` saves with red and blue swapped, and a PNG read with `cv2.imread` must be reversed
  before being matched against the palette. `tools/capture/labels.mask_to_rgb(source=...)` is the one place
  that conversion is expressed.
- **`simGetCameraInfo(cam).fov` is NOT the rendered FOV.** It reported 89.904 deg for the survey camera whose
  true HFOV is **73.98 deg** (f_px 2548.7). Trusting it puts a 21 % error into every GSD, footprint and
  geolocation number. Calibrate instead: `tools/capture/calibrate_camera.py` least-squares fits f_px against
  known survivor positions, residual RMS 5.6 px over 48 observations - which doubles as a validation of the
  whole nadir projection chain. Result is `data/scene/camera_survey.json`; the capture runner refuses to start
  without it.
- **`simListInstanceSegmentationObjects()` does not return actor names.** It returns `<Actor>_<uid>` for
  skeletal actors and `<Mesh>_<n>_<Actor>_<uid>` for static ones, so an exact match on "Human_007" finds
  nothing. Recover the actor by regex. The palette from `simGetSegmentationColorMap()` is indexed by position
  in that list.
- **A pose set while the sim is PAUSED is ignored entirely** - neither `simSetVehiclePose` nor
  `simSetKinematics` moves the vehicle, and every frame comes out identical (8 waypoints produced 8 byte-alike
  frames taken from the ground). Unpaused, the pose applies **asynchronously, one call late**, so capturing
  immediately photographs the previous waypoint. Poll `simGetVehiclePose()` until it matches before capturing.
- **`simSetVehiclePose` does not clear the body's motion.** Teleporting every frame under a live solver spun the
  airframe to **2.2e8 rad/s** and a 17 m/s descent. The survey camera is mounted 30 cm BELOW the body, so a
  tumbling airframe swings into shot and fills frames with propellers - it silently destroyed a 731-frame
  dataset (42 boxes recovered of ~110 expected) while every JSON check passed. Use `simSetKinematics` with all
  four motion fields zeroed, disarm SimpleFlight (`enableApiControl(False)` + `armDisarm(False)`) so its 60 ms
  hover watchdog stops fighting the teleport, and **park the drone on exit** or it tumbles until PIE stops.
- Always render a contact sheet (`tools/capture/contact_sheet.py`) and LOOK at a dataset before using it.

**Unreal Python facts for posed skeletal actors (measured 2026-09-10, session 3):**
- **`SkeletalMeshComponent.set_animation()` does not serialise.** It sets transient state only;
  `animation_data.anim_to_play` stays None, so the pose is lost on level save and every character reverts to
  its bind (A/T) pose at runtime while the editor still looks right. Set the `SingleAnimationPlayData` struct.
- **`AnimationDataController.add_bone_track` is deprecated AND silently no-ops.** A pose built with it reads
  back byte-identical to the bind pose. `add_bone_curve` works.
- **`AnimationLibrary.get_animation_track_names()` returns names LOWERCASED** while `get_bone_name()` returns
  them cased, so matching bone names with Python `==`/`in` silently applies no deltas.
- **Arm bones have mirrored bind orientations, leg bones do not.** The same local delta swings the left arm down
  and the right arm up (measured: identical delta puts the right hand 52.5 cm from its shoulder, negated pitch
  1.19 cm). Legs take the same delta on both sides; negating kicks the right leg backwards.
- **Root pitch -90 is face-down, +90 face-up.** A roll of 180 does NOT flip the face: at pitch 90 the rotator is
  gimbal-locked, so it spins the body about the vertical and only swaps which end the head is at.
- **`Bip01` and `Bip01-Footsteps` are helper bones pinned at z = 0**, so including them in pose geometry makes
  min(z) always 0 and every upright pose reports a ground offset of 0.
- The two Rocketbox children are rigged with a **`Bip02-` prefix**; detect it rather than hard-coding.
- **The Rocketbox FBX import binds NO textures.** It creates `FBXLegacyPhongSurfaceMaterial` instances that
  compile cleanly (418 instructions) and report no error, yet every character renders as a white mannequin. Bind
  the TGAs to `DiffuseColorMap`/`NormalMap`/`SpecularColorMap` **and set the matching `*MapWeight` scalars** - a
  bound texture with weight 0 still renders flat.
- **Renaming an actor onto a name another actor still holds is a FATAL engine error**, not an exception:
  `Renaming an object (...) on top of an existing object`, Obj.cpp:383. Destroying an actor does not free its
  name until garbage collection, so a destroy-then-respawn rebuild crashes the editor; renaming the doomed actor
  to `DEAD_*` first just moves the collision (a crashed run saves those names into the level). **Update actors
  in place** and only ever rename a genuinely new one (`tools/scene/build_actors.py`), or avoid renaming
  entirely and clear by outliner folder (`tools/scene/build_props.py`).
- **A material parameter's DEFAULT texture must match its sampler type** or the whole material fails to compile
  and renders as the grey WorldGridMaterial checker: the engine's sRGB `DefaultDiffuse` under
  `SAMPLERTYPE_MASKS` did exactly that to every building. Assert compilation with
  `MaterialEditingLibrary.get_statistics()` - a failed compile reports zero instructions and nothing else.
- **Pick a material mask channel by MEASURING it.** The terrain's leaf-litter mask sampled a roughness channel
  whose 10th percentile was already 0.878, so `saturate((G-0.55)*4)` was 1.0 everywhere and litter replaced
  100 % of the grass layer; the grass tint provably had no effect. Measure mean/std on the host first.
- **A generated terrain OBJ needs vertex normals.** Without `vn` and smoothing groups UE flat-shades a regular
  grid, which reads as a corduroy ridge pattern at survey altitude.
- **The editor dies of the Windows COMMIT LIMIT, not of RAM, at a few thousand actors.** Placing 5,508 trees
  with `EditorActorSubsystem.spawn_actor_from_object` killed it at
  `PeakUsedVirtual 27.49 GiB` against a commit limit of 26.7 GB (15.7 GB RAM + an 11 GB auto pagefile on C:),
  while `UsedPhysical` was only 5.15 GiB and `AvailableVirtual` had reached 0.00 GiB. The fatal line is
  `Ran out of memory allocating 1082560 bytes ... The paging file is too small for this operation to complete`,
  with a one-frame script stack pointing straight at `SpawnActorFromObject`. One UObject actor + component +
  transform per item is the wrong data structure at this count: bulk scenery must go in an
  `(Hierarchical)InstancedStaticMeshComponent`, one actor per mesh variant holding thousands of instances.
  Only `Human_*` / `Animal_*` need unique instance-segmentation IDs, so sharing an ID across a tree or rubble
  HISM costs nothing - nothing labels scenery, it only has to occlude.
- **`unreal.BlendMode.BM_OPAQUE` does not exist in UE 5.8; the member is `BLEND_OPAQUE`.** The offline mock in
  `tools/scene/check_realism.py` answered any attribute with a lambda, so the typo passed 217 checks and then
  raised `AttributeError` on the real call and aborted the build. UE enums in that mock are now `Enum(...)`
  objects carrying the real member names, and an unknown member raises there too.
- **Clear scenery away from survivors by the CROWN, not the trunk.** `gen_vegetation.py` rejected a tree only
  if its trunk came within 3 m of a survivor while jacaranda crowns are 8-16 m in radius (p50 10.6 m), so 22
  of 71 survivors recorded `occlusion: 0` stood under a canopy - including the whole roof-top group
  Human_000..003 under one 13.5 m crown. The clearance test now runs AFTER the species and scale are drawn,
  against the actual crown radius, and the generator re-measures the finished list and refuses to write a
  layout that contradicts `actors.json` (cost: 35 trees of 4,377).
- **`actors.json` occlusion is an INTENT, and five generators contradict it.** `gen_rubble.py` deliberately
  leans slabs over the fan survivors ("the plate roofs the void without filling it") and `gen_props.py` caps
  the buried ones, but nothing ever told `actors.json`: 25 of 71 survivors had a level the finished geometry
  disagreed with, in both directions. `tools/scene/reconcile_occlusion.py` measures coverage from the final
  layouts, and `tools/capture/measure_occlusion.py` measures it per observation from `visible_px` and the
  frame's own GSD, which is the only number that is actually observed rather than assumed.
- **A full-area lawnmower spends its frames on empty ground.** The 5-95 % quantile box over the survivor
  positions is 607 x 960 m for 69 survivors: the 2026-09-11 run returned 56 boxes from 285 frames, and the
  yield matched the geometric prediction exactly, so the capture path was right and the flight plan was
  wrong. `survey.py --plan patches` single-links the survivors and flies a small lawnmower over each cluster
  instead; `--time-of-day` / `--rain` / `--fog` / `--condition` add appearance variety at no extra flying,
  which buys far more than another few hundred frames of the same light.
- **The capture cadence is set by the TRACKER, not by taste, and getting it wrong silently kills the whole
  chain after geolocation.** `sightline/track/config.py` encodes SOLUTION_DOC 5.6 rule 3 as `min_hits=3`
  inside `min_hits_window_s=2.0`. Run the spine on the 2026-09-11 survey and it reports
  `frames 285 -> detections 56 -> located 56 -> tracks 0 -> records 0 -> ranked 0`: the shutter was 26 m at
  11 m/s, so frames were 2.4 s apart and each survivor was seen about 1.5 times. Three hits could never fall
  inside a 2 s window, so no track was ever confirmed and there were no records, no triage and no map pins.
  The tracker was right; the flight plan was not. Shutter spacing must therefore satisfy BOTH
      s <= speed * window_s / (min_hits - 1)      (the hits fall inside the window)
      s <= frame_height_m / min_hits              (the target is still in frame for all three)
  and `tools/capture/campaign.py:shutter_m` derives it from exactly those two, which at 12 m/s gives 9.9 m at
  35 m AGL and 12.0 m at 55 and 80 m. Detector TRAINING does not care about cadence; everything downstream
  of detection does.
- **RunPod: use SECURE cloud, not COMMUNITY, and the reason is SSH.** Measured 2026-09-11. A community-cloud
  RTX 4090 provisions fine but exposes **no ssh port at all** - `runtime.ports` carries only an internal
  http entry with `isIpPublic: false` - so the per-pod `PUBLIC_KEY` has nothing to authenticate against.
  RunPod's other route, the `<podHostId>@ssh.runpod.io` proxy, authenticates against the **account's**
  registered public key rather than the pod's, so a key generated by our tooling is rejected
  (`Permission denied (publickey)`) and fixing that would mean editing the user's account settings.
  A SECURE-cloud pod exposes a real public TCP port and the injected key works: verified end to end with
  RTX 4090 24 GB, torch 2.8.0+cu128, CUDA available, 64 vCPU, 60 GB disk. Cold start also differs sharply -
  community took **11 minutes** to pull the ~20 GB PyTorch image with `desiredStatus: RUNNING` and
  `runtime: null` the whole time (a short timeout reports failure on a pod that is merely downloading, and
  leaves it billing), while secure was reachable in under a minute.
- **A default `Python-urllib` User-Agent is refused by Cloudflare in front of api.runpod.io** with HTTP 403
  "error code: 1010", even with a valid key. Set a real User-Agent.
- **`r.Nanite.Streaming.StreamingPoolSize=2048` instantly kills the editor on this GPU.** Not an OOM - a hard
  engine assert, `NaniteStreamingManager.cpp:640`:
  `Streaming pool size (2048MB) must be smaller than the largest allocation supported by the graphics
  hardware (2048MB)`. The RTX 4060 Laptop's largest single allocation is exactly 2048 MB and the check is a
  strict `<`, so 2048 is the one value that is fatal rather than merely expensive. A sweep of
  512 -> 1024 -> 2048 therefore dies on its last step with everything before it fine. Stay at or below
  1024 on this machine. `r.Nanite.MaxPixelsPerEdge` 1 -> 0.25 -> 0.1 was safe (it is a quality/cost dial,
  not an allocation), and was measured at availPhys 0.54-0.68 GiB with no crash.

### Sky and distant-foliage LOD (sky lane, 2026-09-11 01:50)

- **The "flat navy sky" was the ExponentialHeightFog, not the SkyAtmosphere.** This cost a wrong first fix and
  is not guessable from the picture. `Material.is_sky` (the flag that makes an unlit opaque material behave as
  a sky dome) removes **aerial perspective** but, in the engine's own words, *"Height and Volumetric fog
  effects will still be applied"*. This level's fog has `fog_inscattering_luminance` (0,0,0) with
  `sky_atmosphere_ambient_contribution_color_scale` 1.0, so it takes its colour **from the SkyAtmosphere** —
  navy — and at a 40 km sky dome's depth the fog is fully saturated, so it repainted the dome completely
  except for a thin strip near the zenith. Isolated by experiment in one frame, band y60-190 (domain=sim):

  | configuration | mean RGB | B/R |
  |---|---|---|
  | dome + fog + atmosphere | (0.021, 0.108, 0.268) | 12.81 |
  | dome + atmosphere, **fog off** | (0.511, 0.510, 0.505) | **0.99** |
  | dome, fog off, atmosphere off | identical to the row above | 0.99 |
  | dome + fog, **atmosphere off** | (0.000, 0.000, 0.000) | — |

  The third row proves the dome was doing the work all along; the second isolates the fog. Fixed with
  `fog_cutoff_distance = 2_000_000` cm (20 km): every piece of real geometry is within ~3 km and the dome is
  the only thing beyond, so the terrain's aerial haze is untouched and only the sky loses its fog.
- **Pattern for an HDRI sky that matches the light**: unlit + `BLEND_OPAQUE` + two-sided + `is_sky`, emissive =
  `TextureSampleParameterCube(-CameraVectorWS) * SkyBrightness`, on a full sphere at 40 km. Sampling the
  SkyLight's OWN TextureCube makes the background literally the light probe. Keep the brightness control a
  **scalar** — it then cannot shift chromaticity, so no amount of tuning can break sky-matches-light.
  `HDRIBackdrop` does not exist in this project (plugin not enabled); there is no `SkySphere` Python class.
- **`MaterialExpressionTextureSampleParameterCube`'s UV pin is `UVs`, not `Coordinates`** ("Coordinates" is the
  C++ member name and reads perfectly plausibly). `connect_material_expressions` just returns **False** — no
  exception — and an unwired sky dome renders solid black. Resolve pin names with
  `MaterialEditingLibrary.get_material_expression_input_names(expr)` and assert every connection.
- **`NaniteShapePreservation.PRESERVE_AREA` fixes foliage that thins out at distance; `VOXELIZE` does nothing
  in 5.8.2 here.** VOXELIZE's docstring describes this exact defect, and it applied and persisted on the
  asset, but the render did not change by even 0.001 — every `r.Nanite.Voxel*` cvar reads 0, so the voxel path
  appears inactive in this build. PRESERVE_AREA ("legacy foliage technique") is the one that works, and it is
  build-time data with **no runtime cost**. Rebuild is ~2 min for 10.7 M Nanite triangles across 5 meshes.
- **Distant-crown quality is screen-space LOD, not streaming residency.** Measured on the 450 m demo frame,
  darkest-quartile G/R in a fixed canopy crop (bark is red-dominant, leaves green-dominant), domain=sim:
  `r.Nanite.Streaming.StreamingPoolSize` 512 -> 1024 moved it 0.723 -> 0.722, i.e. nothing.
  `r.Nanite.MaxPixelsPerEdge` 1.0 / 0.5 / 0.25 / 0.1 gave 0.723 / 0.835 / 0.918 / 0.949. With PRESERVE_AREA
  on the meshes the same dial gives 0.879 / 0.980 / 0.978 / 0.957 — it **plateaus at 0.5**, so spending 6.25x
  the geometry to reach 0.1 buys nothing.
- **`SceneCaptureComponent2D.capture_scene()` returns before the GPU finishes**, so wrapping it in
  `time.perf_counter()` measures ~0.1 ms and is NOT a cost measurement. Do not report those numbers as frame
  cost; they are meaningless.
- **The campaign's four "conditions" share ONE sky, and the names overstate that.** `simSetTimeOfDay` moves
  the directional light only; the SkyLight's ambient comes from a specified cubemap
  (`overcast_soil_puresky_2k`) that does not change with it. So `clear_morning`, `clear_midday`,
  `hazy_afternoon` and `rain_overcast` differ by SUN ANGLE, fog and rain over a single overcast sky - four
  lighting conditions, not four sky types. That is still a real diversity axis (sun elevation and azimuth
  change shadow direction and length in every frame, and fog/rain change contrast and lens appearance), but
  a reader who takes "clear_morning" to mean a blue-sky scene would be misled. Swapping the cubemap per pass
  would give genuine sky variety and costs one import each; it is not done today because the SkyLight
  intensity was solved against THIS cubemap and each new sky would need its own re-measurement through
  `tools/scene/measure_lighting.py`.
- **A truncated capture pass drops the SAME survivors every time, because patch order is deterministic.**
  `survey.py --plan patches` orders patches nearest-first from the launch site and that order is identical on
  every run, so when `--max-minutes` cuts a pass short it always cuts the FARTHEST patches. Measured on
  2026-09-11: the 35 m pass ran at 2.8 s/frame including transits (48 % of elapsed time was patch-to-patch
  transit, in gaps of 36-68 s where the shutter correctly holds), projecting ~35 min against a 30 min cap.
  The failure mode to watch for is therefore not "one pass is short" but "the same distant survivors are
  missing from EVERY pass", which looks like a dataset that simply never saw them. Check coverage across the
  whole campaign with `quality_report.py`'s never-seen breakdown, and fix a systematic hole with a short
  supplementary pass over the missing patches rather than re-flying everything.
- **The shutter gate rejects about half the planned shots, so frame estimates from track length are 2x
  optimistic.** Measured on the 2026-09-11 35 m pass: 7.4 km of track at a 9.8 m shutter predicts ~755
  frames; the pass flew all 29 legs in 21.6 min (inside its 30 min cap, NOT truncated) and produced **347
  frames, 110 boxes, 44 of 69 survivors**. The missing ~54 % are shots the gate refused because the airframe
  was banked beyond `--max-tilt-deg 8` or off the commanded AGL by more than `--alt-tol-m 8` - measured AGL
  ranged 33.0-43.0 m against a commanded 35, so 43 sat exactly at the tolerance edge. This is the gate doing
  its job (`validate.py` fails any frame beyond +-12 deg tilt, so those frames would have been thrown away
  later anyway), but it means `est_frames = track_m / shutter_m` in `SurveyPlan` is an UPPER BOUND, not an
  estimate, and any dataset-size plan built on it will be out by a factor of two. Note the tilt gate at 8 deg
  is stricter than the validator's 12 deg; relaxing it toward 10-12 deg would recover frames that would still
  pass validation.

## Live integration lane, 2026-09-11 — facts that cost hours to find

**1. `Telemetry.q_gimbal` means two different things in this repo, and the tracker loses.**
`sightline.geo` reads it as camera-FRD -> NED (`ChainConfig.gimbal_frame="frd"`, its default), so a nadir
camera is `(0.7071, 0, -0.7071, 0)`. The FROZEN schema (`schemas.py:71-73`), `ingest/spec.py`,
`track/geometry.py` and `coverage/footprint.py` read it as optical -> NED, so a nadir camera is the
**IDENTITY** quaternion — and `Telemetry.gimbal_pitch_deg()` returns -90 for identity and **-180** for the
geo literal. Symptom when a producer writes the geo literal: `track.geometry.telemetry_affine` warps the
image 42 px for a 9.8 m step that geometrically demands 555 px, nothing associates, and a survey yields
detections, geolocations and **zero tracks**. Build the quaternion with
`sightline.pipeline.nadir_gimbal_quat()` (which calls `spec.gimbal_quat_from_euler`) and convert at the geo
boundary with `sightline.pipeline.to_geo_gimbal()`. The two conventions differ by exactly
`spec.Q_FRD_FROM_CAM`. **Unresolved at the contract level** — see docs/TRACKER.md.

**2. Camera-motion compensation is mandatory for a survey, not a refinement.** At 45 m AGL with
f = 2548.7 px, a 9.8 m shutter step moves the whole image 555 px while a survivor is ~60 px across:
consecutive frames share no pixels. `cmc_enabled=False` gives 0 tracks; `True` gives one track per survivor.
`FramePipeline` defaults it True.

**3. `rank_records(records)` with no `TriageContext` does not score.** It sorts and fills `priority_rank`,
leaving `score = 0.0` and every `ScoreComponents` field at its default. A map full of `P(living) 0.00` cards
is what that looks like. Always pass a context; pin `now_utc` to the FRAME clock, or a live run and a later
replay of the same frames disagree.

**4. `sightline.store.Uploader` used to shadow `threading.Thread._stop`** with an `Event`, so
`Uploader.stop()` raised `TypeError: 'Event' object is not callable` on any clean shutdown that joined an
already-finished thread. Fixed (`_stop_event`). Do not name an attribute `_stop` on a `Thread` subclass.

**5. The C2's idempotent upsert rejects a same-version re-write with 409, and the uploader retries a failed
job for ever** — one 409 blocks the whole outbox queue behind it (measured: 4 of 1019 records delivered).
So a live pusher must never enqueue the same `(record_id, version)` twice with different content. In
practice that means the "has this record changed?" test must not include anything that varies with the
clock: the triage score decays continuously and `Deduplicator` only bumps `version` when it re-clusters.

**6. This machine's Xbox 360 pad, measured (SDL2 joystick API through pygame).** Axes 0-3 are the two sticks
and rest at 0.0; **axes 4 and 5 are the triggers and rest at -1.0**. The widely-quoted "0 LX, 1 LY, 2 LT,
3 RX, 4 RY, 5 RT" layout is NOT what this driver reports, and mapping a stick onto axis 4 makes the control
source report full deflection before anyone touches it. `PygameGamepadSource` reads the resting state at
construction and refuses such a mapping by name. 11 buttons are enumerated; none has been pressed under
observation, so the button indices remain unverified.

**7. The Claude Browser pane cannot render the C2 map.** MapLibre's module web worker does not load there, so
`map.on("load")` never fires and the page sits at "starting…" with `records 0` — while the API is serving
records perfectly. Use `node app/map/headless_check.mjs --url ... --out ...` (real Edge over CDP, verifies
zero external requests and zero console errors) for any screenshot of the map.

**8. `sightline.coverage.footprint.ground_footprint` is 90 deg rotated** from both the geo and the track
lanes on the same telemetry (measured at gimbal yaw 0 and 90). The live map draws its camera footprint from
`sightline.geo.footprint_ned` instead — the same chain that geolocates the pins inside it — which measures
67.8 m east x 38.1 m north at 45 m AGL, the across-track x along-track orientation `survey.py`'s line
spacing assumes.
- **The camera convention is SETTLED BY MEASUREMENT, not by reading settings.** The live lane found `geo` and
  `track` reading `q_gimbal` 90 deg apart from `coverage.footprint.ground_footprint`, patched it at one named
  boundary (`pipeline.to_geo_gimbal`) and correctly called that a labelled workaround rather than a
  resolution, noting the tie-breaker was unknowable because `telemetry.csv` has no `gimbal_yaw_deg` column.
  It is knowable from the captured data. Using 110 boxes whose survivors have known world positions, predict
  each survivor's pixel position under each candidate convention and keep the one that puts them where they
  actually are:

      camera yaw WORLD-FIXED north      median residual    95.2 px
      camera yaw FOLLOWS the airframe   median residual  1087.4 px      (11x worse)

      x=+E/gsd, y=-N/gsd  (east-right, north-up)     95.2 px = 1.37 m   <- TRUE
      x=+E/gsd, y=+N/gsd                           1096.5 px = 15.77 m
      x=+N/gsd, y=-E/gsd  (rotated 90)             1472.0 px = 21.17 m
      x=-E/gsd, y=-N/gsd                           2365.1 px = 34.02 m

  So: the gimbal is stabilised in ALL THREE axes at pitch -90, roll 0, yaw 0 (north), exactly as
  `sim/settings/dataset.json` declares with `Gimbal.Stabilization = 1.0`; the image is always north-up
  regardless of the airframe, which flies alternate legs backwards under `DrivetrainType.ForwardOnly`; and
  **image-right is due EAST**. `sightline.geo` and `sightline.track` have it right; `coverage.footprint` is
  the one that is rotated. The residual 1.37 m is accounted for by the deliberately noise-injected telemetry,
  the box centre being the torso rather than the ground point, and the GSD approximation - the next-best
  hypothesis is 11x worse, so there is no ambiguity to argue about.
  **Actions:** fix `coverage.footprint.ground_footprint` to match, and add a `gimbal_yaw_deg` column to
  `telemetry.csv` so no future reader has to re-derive this.

### GPU contention, not inference cost — the live-loop latency finding (2026-09-11 06:30)

**A diagnosis recorded earlier in this session was wrong, and the correction matters more than the original
claim.** The live loop measured 1165-3205 ms per frame of detection and I attributed it to inference being
"10-25x over the section 5.11 budget". Measured again with the Unreal editor CLOSED, the identical model on
the identical 15-tile 1024 px grid runs at:

    pytorch fp16   median 166.9 ms   min 153.7   max 170.2      (6 real 4K frames, RTX 4060 Laptop)

against section 5.11's ~113 ms budget. So uncontended inference is within a factor of 1.5 of budget, not 10x
over it. The 1.2-3.2 s figure was the Unreal editor rendering a 4K scene and the detector competing for the
same 8 GB card.

**Why this changes the conclusion.** The bottleneck is deployment topology, not model cost or code:

* on this laptop one GPU does two jobs, and that is the constraint;
* in the architecture section 5.11 actually describes, the detector runs on a Jetson Orin with no renderer
  on the same device, so the contention does not exist;
* it also explains the tracking failure. At 167 ms the loop runs near 1.2 FPS rather than 0.6, which halves
  the inter-frame camera translation (~450 px at 0.6 FPS) and makes frame-to-frame association far more
  tractable. The tracker was being starved, not malfunctioning.

**Consequence for any future measurement:** never benchmark the detector while the editor is up, and never
quote a live-loop latency without saying whether the renderer was running. Both numbers are true; they
measure different things.

**TensorRT export, same session.** Building at the full 15-tile batch fails in the FP16 AutoCast step, which
runs a CPU reference pass to validate numerics: at 15 x 3 x 1024 x 1024 the attention intermediates exhaust
host RAM and onnxruntime raises `bad allocation` on a MatMul node. Build at batch 6 (which is what section
5.11 specifies anyway) with `dynamic=True` so the 15-tile grid can still be sent as chunks.
