# Tracker

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done and verified · `[!]` blocked · `[-]` dropped (say why).
"Verified" means it was run and observed working, not just written. Update this file at the end of every chunk
of work and append to the Session log.

## Next actions (start here) — DEVELOPMENT PHASE
The environment and both MCP servers are validated (see the verification table below); feature work starts now.
Read `docs/HANDBOOK.md` §6 before touching the scene — it lists the traps that would silently corrupt the dataset.

1. **F1 flood-valley scene — FOUNDATION DONE (2026-09-10, session 2), dressing in progress.**
   `/Game/Sightline/Maps/FloodValley` exists and is the project's startup + game map. Rebuild it any time with
   `tools/scene/gen_terrain.py` then `ue_python code="exec(open(r'D:\Sightline\tools\scene\build_flood_valley.py').read())"`
   (idempotent; self-checks the terrain transform with a line trace). FloodLevel at runtime:
   `uv run python tools/scene/flood_level.py <asl_m>`. **Remaining for F1:** terrain material (zone mask
   `T_FloodValley_Zones` + CC0 PBR textures), water normals/foam, settlement buildings, debris, vegetation,
   then day-1 #2 with a real half-submerged actor (water already writes depth and has its own instance colour).
   The generator also needs zone polygons/spawn masks for the spawners (next item).
2. **Actor + debris spawners** (seeded, reproducible): pose/submersion classes per §2.3 row 2-3 and §6.2, tagged
   `Human_<id>` / `Animal_<id>`, movable actors only (static ones cannot be posed at runtime).
3. **Thermal**: object-ID temperature table first (day-1 test #3), then the §5.1 step 7 post-process material.
4. **Capture pipeline (F5)**: waypoint capture with `simPause` + `simGetImages` (SteppableClock profile), auto-labels
   from instance masks, per-frame telemetry CSV/JSON; then the day-1 unknowns below.
5. Assets: Mixamo needs an Adobe login (ask the user) — UE mannequin is the fallback for pose variety.

## Phase 0: environment and MCP (day 1 morning)
- [x] Located UE 5.8.2 at `D:\UE_5.8`; confirmed Epic's ModelContextProtocol plugin ships in 5.8.2
- [x] uv 0.12.12 on D:, caches redirected to D: (see CONTEXT §3)
- [x] Python 3.11 env + `uv.lock` (cosysairsim 3.4.1, mcp 1.30.0, numpy 2.2.6, opencv 4.14)
- [x] Cosys-AirSim 5.8-v3.4.1 plugin + Blocks editor project downloaded (size-verified); plugin installed in project
- [x] Blocks packaged build (reference sim, used for AirSim verification without the editor)
- [x] Visual Studio 2022 17.14 on D: (MSVC 14.44.35228 — accepted by UE 5.8; the folder name 14.44.35207 is
  irrelevant, UBT reads cl.exe's file version)
- [x] `SightlineSim` project scaffolded (module, targets, low-memory renderer, Python remote exec, MCP auto-start)
- [x] sightline MCP server written; stdio smoke test passes (24 tools)
- [x] `SightlineSimEditor` compiles: Result Succeeded in 869 s (2026-09-10 18:47), MSVC 14.44.35228 (VS 2022 on D:),
  Windows SDK 10.0.22621, via UBA (cache redirected to D:\UE_Cache\UBA). Log: `_logs/jobs/build-SightlineSimEditor-*.log`
- [x] Editor opened the project (FlyingExampleMap, AirSim plugin loaded) at 18:52; engine-tools agent verifying MCP live
- [ ] Editor opens the project with AirSim loaded; editor RAM measured
- [x] `unreal` MCP (HTTP :8000) reachable from Claude Code; 31 toolsets / 392 tools listed and called
- [x] `ue_python` remote execution verified against the live editor (file/statement/eval, 20k-line output intact)
- [x] PIE + `sim_ping` + takeoff + `sim_capture` verified in-editor (39/39, same as the packaged build)
- [x] `tools/doctor.py --live` passes (0 FAIL)
- [x] Blocks packaged exe runs; the Python client connects (API cross-check)
- [x] **Phase 0 complete (2026-09-10).** Environment, both MCP servers and all 27 tools validated; see the
  verification table below and `docs/verification/`.

## Verification workstreams (2026-09-10): every tool in the doc, end to end
Reports land in `docs/verification/`. A tool is not "ready" until its report says PASS with evidence.
| Stream | Scope | Report | Status |
|---|---|---|---|
| V1 engine MCP | sightline engine/build/editor/log/job tools + Epic `unreal` MCP live (toolset inventory) | `engine_tools.md` | [x] all sightline engine tools PASS; Epic: 31 toolsets / 392 tools, 17 real calls incl. CaptureViewport PNG. Fixed: `ue_generate_project_files` (batch file absent in installed engine), missing user env -> editor DDC would go to C:, UBA store -> D:, double-execution after a timed-out editor command |
| V9 dev workflow | AirSim inside the editor (PIE), gamepad via AirSim + takeover, full 27-tool matrix, CLI spot-checks | `dev_workflow.md` | [x] PIE loop PASS (launch 49 s cold, StopPIE -> tools error in 0.31 s, editor closes cleanly 4/4); AirSim **39/39 in-editor**; tool matrix **33/33 up / 12/12 down, all 27 tools**; CLI drove a real flight + capture (image survived MCP as valid JPEG). Fixed D1: first `simGetImages` per engine process returned **unconverted** depth/normals (silently wrong) -> warm-up frame discarded + implausible-depth warning |
| V7 GPU latency | day-1 test #8: C1/C2/C3 strategies, TensorRT FP16, RTX 4060 | `gpu_latency.md` | [x] all six configs **inside 300 ms**. Design pass (C2, yolo26s, 6×1280×1088 batched) = **65.0 ms median / 68.3 p90** vs the doc's ~60 ms estimate; C1-s 16.7 ms, C3-s 29.2 ms. FP16 only; INT8 and Orin (F20) still open |
| V2 MCP protocol | sightline server schemas, errors, concurrency, cancellation, lifecycle (`tests/test_mcp_protocol.py`) | `mcp_protocol.md` | [x] 20 pass / 2 skip (no-editor cases skip while an editor runs; passed with it down). 9 defects fixed |
| V3 Claude Code integration | CLI loads `.mcp.json`, real tool calls via `claude -p`, timeouts, reconnect | `claude_code_integration.md` | [!] servers load + tools discovered (27) + unreal connects; real `claude -p` calls BLOCKED: CLI not logged in (user: `claude auth login`). Project `.claude/settings.json` sets MCP timeouts |
| V4 AirSim tools | every sim_* tool through MCP (`tools/sightline_mcp/test_sim.py`) | results JSON in `_artifacts/verification/` | [x] **39/39** vs packaged Blocks (2026-09-10 19:19). Defects found+fixed: broken `landAsync` (own position-held landing + disarm; RTL lands 0.2 mm from home), velocity-only descent drifting into obstacles, static-actor pose silently "succeeding", `to_eularian_angles`/`to_quaternion`/`simGetImages(external=)` wrong API names, 60 s RPC timeout on long flights, falling-after-reset breaking takeoff, over-strict Ground collision rule |
| V5 Python stack | all Python libs in doc: install pinned + functional tests (`tests/test_stack.py`) | `python_stack.md` | [~] |
| V6 external tools | Cesium, X-AnyLabeling, PMTiles+MapLibre offline, Fields2Cover, PX4/QGC, DroneModels, TAK, gamepad | `external_tools.md` | [x] X-AnyLabeling, PMTiles, MapLibre offline, DroneModels PASS; Cesium STAGED (`_staging/plugins`, BuildId match); Fields2Cover REJECT on Windows (own boustrophedon); PX4/QGC/TAK documented |
| V8 gamepad | XInput + pygame detection, live input, AirSim `rc_data`, API release/re-acquire handover | `_artifacts/verification/gamepad_airsim_*.json` | [x] **13 PASS / 0 FAIL** (2026-09-10 20:31). AirSim sees the pad (VID_045E, is_valid); axes full travel; under API control a held stick moves the drone 0.01 m in 4 s; `release` -> API off in 23 ms, vehicle follows the stick after **394 ms**; `arm` -> authority back, stabilised in **2.96 s**, then `move_to` obeyed to 0.22 m with the stick still held; rtl landed. F3's takeover semantics are proven |
| V7 GPU performance | day-1 test #8: C1/C2/C3 latency on the 4060 (after V5) | `gpu_latency.md` | [ ] |

MCP defects already found and fixed (2026-09-10): stdout pollution by cosysairsim prints; Windows stdio deadlock on lazy
numpy import (now eager imports); blocking tools moved off the event loop (`@threaded`); tool-to-tool calls via
`__wrapped__`; wrong cosysairsim API names (`to_eularian_angles`, `to_quaternion`, `simGetImages(external=)`).
Also: `materials.csv` missing -> Cosys skips material stencil init (installed via `tools/setup/install_materials.ps1`).

## Open items carried into development
- ~~Gamepad handover unmeasured~~ **CLOSED 2026-09-10**: 13 PASS / 0 FAIL, latencies API->RC 394 ms and
  RC->API 2.96 s (`tools/day1/gamepad_airsim.py`). The Unreal window does NOT need focus (DirectInput uses
  `DISCL_BACKGROUND`); with `AllowAPIAlways:true`, `AllowAPIWhenDisconnected` is a no-op. F3 must log every
  mode switch into the telemetry CSV (§5.2) so coverage can be attributed to AUTO vs MANUAL.
- **Epic `unreal` MCP drops idle HTTP sockets after 15 s** (`HttpConnection.h:266 ConnectionKeepAliveTimeout`,
  hard-coded, no cvar). Symptom: "The socket connection was closed unexpectedly". **Retry the call once.**
- **Cosys 3.4.1 logs nothing to the UE log** (`UAirBlueprintLib::LogMessage` has every `UE_LOG` commented out), so
  "Loaded settings from ..." can only be seen on the on-screen HUD. Prove settings use via `listVehicles()`,
  camera resolution and the home geopoint instead.

## Known issues to fix when the flood level is authored
- **Lumen is still active in PIE** despite `r.DynamicGlobalIlluminationMethod=0` in DefaultEngine.ini: the Blocks
  sample map (`FlyingExampleMap`) carries a PostProcessVolume that overrides the project default, so the editor logs
  "Lumen ... has no ray tracing data and won't operate correctly" and pays for GI we do not want on an 8 GB GPU.
  Author `Sightline/Maps/FloodValley` with GI = None (and no PPV override) rather than patching Epic's sample map.
- **Never run GPU/ML work while the editor + PIE are up** on this machine: 16 GB RAM / 8 GB VRAM triggers Windows
  memory-pressure warnings and makes both slow (observed 2026-09-10 20:05 with a TensorRT export alongside PIE).

## Day-1 unknowns (SOLUTION_DOC §10), with results
| # | Test | Status | Result |
|---|---|---|---|
| 1 | Cosys 3.4.1 loads on UE 5.8.2; editor RAM with project open | [ ] | |
| 2 | Instance seg hides submerged pixels (single-layer water / translucent plane); IgnoreMarked camera | [ ] | |
| 3 | Infrared image type after ID remap; capture FPS 4K raw vs PNG | [ ] | |
| 4 | Xbox RemoteControlID, AllowAPIAlways, handover latency, moveByRC need | [!] | no gamepad connected yet |
| 5 | Weather visible in Scene, absent in Segmentation; time-of-day needs sky sphere | [ ] | |
| 6 | Detection-API box flicker on prone / 70 % submerged skeletal meshes | [ ] | |
| 8 | End-to-end ms on the 4060 for C1/C2/C3 | [ ] | |

## Features (SOLUTION_DOC §8)
| # | Feature | MVP? | Status |
|---|---|---|---|
| F1 | Flood-valley scenario, 3 zones, weather/time/flood level, actor spawners | MVP | [ ] |
| F2 | Coverage patterns, orbit-on-detect, revisit queue, battery/geofence | MVP | [ ] |
| F2b | Decision planner (Koopman + greedy) | Stretch | [ ] |
| F3 | Gamepad takeover/hand-back, HOLD/RTL, logged mode switches | MVP | [ ] |
| F4 | PX4 SITL path | Stretch | [ ] |
| F5 | Synthetic dataset with visible/amodal boxes, attributes, telemetry | MVP | [ ] |
| F6 | Annotation guideline + X-AnyLabeling loop | MVP | [ ] |
| F7 | Ingest (sim export, DJI SRT; MAVLink/ULog stretch) | MVP | [ ] |
| F8 | Tiled YOLO26 RGB detector, TensorRT, frozen threshold | MVP | [ ] |
| F9 | Thermal YOLO26n + WBF/ProbEn late fusion, RGB-only fallback | MVP | [ ] |
| F9b | Radiometric thermal (sim + stills) | MVP in sim | [ ] |
| F10 | Crop verifier: is_real + posture + submersion + occlusion | MVP | [ ] |
| F11 | BoT-SORT/TrackTrack with CMC, 3-hit confirm | MVP | [ ] |
| F12 | Geo-dedup (DBSCAN 2xCE90), count, motion, stale | MVP | [ ] |
| F13 | Geolocation chain + error budget | MVP | [ ] |
| F14 | Triage score + GeoJSON/KML/KMZ | MVP | [ ] |
| F15 | MapLibre offline map, WebSocket, record cards | MVP | [ ] |
| F16 | Search-quality raster (POD), burial polygons | MVP | [ ] |
| F16b | Per-presentation coverage layers | MVP (2 layers) | [ ] |
| F17 | CoT/TAK export | Stretch | [ ] |
| F18 | SQLite log + persist-queue outbox + cloud upsert | MVP | [ ] |
| F19 | Evaluation script + slices + FiftyOne | MVP | [ ] |
| F20 | Jetson Orin engines + latency table | Stretch | [ ] |
| F21 | Guardrails (no auto-close, no delete, dismiss-with-reason) | MVP | [ ] |

## Session log
- **2026-09-10 (session 1, part 2)**: Validation phase closed. Six verification streams run in parallel by
  subagents (engine MCP, protocol, Claude Code integration, external tools, Python stack, dev workflow) plus the
  AirSim suite by the main session. Results: 27/27 sightline tools, Epic 31 toolsets/392 tools, AirSim 39/39
  (packaged and in-editor), protocol 22/22, stack 37/0, doctor 0 FAIL. Defects found and fixed are listed per
  stream above; the most dangerous were the Windows stdio deadlock, the broken `landAsync`, and the unconverted
  first capture. Gamepad verified at the Windows level; AirSim-side handover still unmeasured. Docs finalised
  (HANDBOOK/CONTEXT/TRACKER), two commits, memory files written. **Development phase not started**; the only
  dev artefact is the unvalidated `tools/scene/gen_terrain.py` draft.
- **2026-09-10 (session 1, part 1)**: Environment setup. Found UE 5.8.2 plus the built-in Epic MCP plugin. Chose VS 2022 17.14
  (Cosys build toolchain). Installed uv and the Python 3.11 env on D:, redirected all caches to D:, downloaded
  Cosys-AirSim 5.8-v3.4.1, scaffolded SightlineSim, wrote the sightline MCP server (smoke test OK), and wrote
  CLAUDE.md/CONTEXT/TRACKER. A network drop interrupted the VS install, which was restarted via the bootstrapper.
  RAM headroom is low (see CONTEXT §7).
