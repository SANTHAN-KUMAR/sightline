# Tracker

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done and verified · `[!]` blocked · `[-]` dropped (say why).
"Verified" means it was run and observed working, not just written. Update this file at the end of every chunk
of work and append to the Session log.

## Next actions (start here)
1. Wait for or verify the VS 2022 install (`D:\VS\2022\Community\VC\Tools\MSVC\14.44.*`). If it is missing, re-run the
   bootstrapper (docs/SETUP.md §2).
2. Build `SightlineSimEditor` (sightline MCP `ue_build`), then launch the editor (`editor_launch(wait_ready_s=2400)`;
   the first launch compiles shaders).
3. Verify both MCP servers live: `unreal` (list_toolsets, CaptureViewport, StartPIE) and `sightline`
   (`ue_python`, `sim_ping`, `sim_fly`, `sim_capture`).
4. Day-1 tests from SOLUTION_DOC §10 (the table below).

## Phase 0: environment and MCP (day 1 morning)
- [x] Located UE 5.8.2 at `D:\UE_5.8`; confirmed Epic's ModelContextProtocol plugin ships in 5.8.2
- [x] uv 0.12.12 on D:, caches redirected to D: (see CONTEXT §3)
- [x] Python 3.11 env + `uv.lock` (cosysairsim 3.4.1, mcp 1.30.0, numpy 2.2.6, opencv 4.14)
- [x] Cosys-AirSim 5.8-v3.4.1 plugin + Blocks editor project downloaded (size-verified); plugin installed in project
- [~] Blocks packaged build download (optional reference)
- [~] Visual Studio 2022 17.14 on D: (first attempt killed by a network drop; re-running)
- [x] `SightlineSim` project scaffolded (module, targets, low-memory renderer, Python remote exec, MCP auto-start)
- [x] sightline MCP server written; stdio smoke test passes (24 tools)
- [x] `SightlineSimEditor` compiles: Result Succeeded in 869 s (2026-09-10 18:47), MSVC 14.44.35228 (VS 2022 on D:),
  Windows SDK 10.0.22621, via UBA (cache redirected to D:\UE_Cache\UBA). Log: `_logs/jobs/build-SightlineSimEditor-*.log`
- [x] Editor opened the project (FlyingExampleMap, AirSim plugin loaded) at 18:52; engine-tools agent verifying MCP live
- [ ] Editor opens the project with AirSim loaded; editor RAM measured
- [ ] `unreal` MCP (HTTP :8000) reachable from Claude Code, toolsets listed
- [ ] `ue_python` remote execution verified against the live editor
- [ ] PIE + `sim_ping` + takeoff + `sim_capture` (scene/seg/IR/depth) verified
- [ ] `tools/doctor.py` passes end to end
- [ ] Blocks packaged exe runs; the Python client connects (API cross-check)

## Verification workstreams (2026-09-10): every tool in the doc, end to end
Reports land in `docs/verification/`. A tool is not "ready" until its report says PASS with evidence.
| Stream | Scope | Report | Status |
|---|---|---|---|
| V1 engine MCP | sightline engine/build/editor/log/job tools + Epic `unreal` MCP live (toolset inventory) | `engine_tools.md` | [x] all sightline engine tools PASS; Epic: 31 toolsets / 392 tools, 17 real calls incl. CaptureViewport PNG. Fixed: `ue_generate_project_files` (batch file absent in installed engine), missing user env -> editor DDC would go to C:, UBA store -> D:, double-execution after a timed-out editor command |
| V9 dev workflow | AirSim inside the editor (PIE), gamepad via AirSim + takeover, full 27-tool matrix, CLI spot-checks | `dev_workflow.md` | [~] agent running |
| V2 MCP protocol | sightline server schemas, errors, concurrency, cancellation, lifecycle (`tests/test_mcp_protocol.py`) | `mcp_protocol.md` | [x] 20 pass / 2 skip (no-editor cases skip while an editor runs; passed with it down). 9 defects fixed |
| V3 Claude Code integration | CLI loads `.mcp.json`, real tool calls via `claude -p`, timeouts, reconnect | `claude_code_integration.md` | [!] servers load + tools discovered (27) + unreal connects; real `claude -p` calls BLOCKED: CLI not logged in (user: `claude auth login`). Project `.claude/settings.json` sets MCP timeouts |
| V4 AirSim tools | every sim_* tool through MCP (`tools/sightline_mcp/test_sim.py`) | results JSON in `_artifacts/verification/` | [x] **39/39** vs packaged Blocks (2026-09-10 19:19). Defects found+fixed: broken `landAsync` (own position-held landing + disarm; RTL lands 0.2 mm from home), velocity-only descent drifting into obstacles, static-actor pose silently "succeeding", `to_eularian_angles`/`to_quaternion`/`simGetImages(external=)` wrong API names, 60 s RPC timeout on long flights, falling-after-reset breaking takeoff, over-strict Ground collision rule |
| V5 Python stack | all Python libs in doc: install pinned + functional tests (`tests/test_stack.py`) | `python_stack.md` | [~] |
| V6 external tools | Cesium, X-AnyLabeling, PMTiles+MapLibre offline, Fields2Cover, PX4/QGC, DroneModels, TAK, gamepad | `external_tools.md` | [x] X-AnyLabeling, PMTiles, MapLibre offline, DroneModels PASS; Cesium STAGED (`_staging/plugins`, BuildId match); Fields2Cover REJECT on Windows (own boustrophedon); PX4/QGC/TAK documented |
| V8 gamepad | XInput + pygame detection, live input, AirSim `rc_data`, API release/re-acquire handover | (this file) | [~] detected (XInput slot 0, pygame "Xbox 360 Controller", 6 axes/11 buttons); live input NOT yet observed (all-zero for 55 s) - re-test with `tools/day1/gamepad_check.py --wait-for-input 120` |
| V7 GPU performance | day-1 test #8: C1/C2/C3 latency on the 4060 (after V5) | `gpu_latency.md` | [ ] |

MCP defects already found and fixed (2026-09-10): stdout pollution by cosysairsim prints; Windows stdio deadlock on lazy
numpy import (now eager imports); blocking tools moved off the event loop (`@threaded`); tool-to-tool calls via
`__wrapped__`; wrong cosysairsim API names (`to_eularian_angles`, `to_quaternion`, `simGetImages(external=)`).
Also: `materials.csv` missing -> Cosys skips material stencil init (installed via `tools/setup/install_materials.ps1`).

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
- **2026-09-10 (session 1)**: Environment setup. Found UE 5.8.2 plus the built-in Epic MCP plugin. Chose VS 2022 17.14
  (Cosys build toolchain). Installed uv and the Python 3.11 env on D:, redirected all caches to D:, downloaded
  Cosys-AirSim 5.8-v3.4.1, scaffolded SightlineSim, wrote the sightline MCP server (smoke test OK), and wrote
  CLAUDE.md/CONTEXT/TRACKER. A network drop interrupted the VS install, which was restarted via the bootstrapper.
  RAM headroom is low (see CONTEXT §7).
