# Tracker

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done and verified · `[!]` blocked · `[-]` dropped (say why).
"Verified" means it was run and observed working, not just written. Update this file at the end of every chunk
of work and append to the Session log.

## Handoff state (end of session 2, 2026-09-10) — READ FIRST
- **Stopped mid-verification by the user.** Everything below is saved to disk and committed; nothing is half-written.
- **Editor may still be running** (pid 25948, launched with `-settings=sim/settings/default.json`) with a PIE
  session requested at the moment of the stop. Run `status` first; stop PIE (`LevelEditorSubsystem.editor_request_end_play()`)
  or `editor_close` before building anything. The `unreal` (Epic) MCP server did not connect at session start
  because the editor was down: reconnect it (`/mcp`) once the editor runs.
- **Session 2 resumed (2026-09-10 21:27):** items 1-3 below checked in PIE. Water renders (silt-brown, ripples,
  glints; capture `20260910-212806-083610`), the airframe blob is gone, and `sim_environment(time_of_day=
  "2024-07-30 06:30:00")` gives warm low dawn light with long terrain shadows (capture `20260910-212927-215281`), so
  the hidden BP_Sky_Sphere still drives the sun. Still wrong: terrain normals/ARM tile visibly at 45 m and the grass
  reads dry-yellow -> `build_materials.py` reworked (bigger tiles, near/far blend on all three maps weighted by a
  smooth macro noise, `GrassTint`, silt band 1.2 m, water `NormalStrength` 0.35); re-check after the rebuild.
- **Written but NOT yet verified** (do these first, in this order):
  1. `M_FloodValleyTerrain` with anti-tiling (84 nodes / 16 textures) and the rebuilt `M_FloodWater` (21 nodes,
     1 texture, compiles) have **not been seen in a capture yet**. The last capture (`_artifacts/captures/20260910-211540-830858`)
     predates both fixes: it showed visible 4-5 m tiling and the water as the grey default material.
  2. Survey camera moved to NED (0, 0, 0.30) in `sim/settings/default.json` + `capture_4k.json` to remove an
     airframe blob from the nadir frame corner. Not yet checked in a capture.
  3. SkyAtmosphere added and BP_Sky_Sphere hidden: check the red "sky light real-time capture" viewport warning is
     gone, **and that `sim_environment(time_of_day=...)` still moves the sun** (Cosys finds the sun through the
     hidden BP_Sky_Sphere's "Directional light actor" property; unverified since hiding it).
  4. `sim/SightlineSim/Config/DefaultEditorSettings.ini` sets `bThrottleCPUWhenNotForeground=False` in the correct
     section: takes effect at the **next editor launch**; confirm the editor keeps full frame rate in the background.
  5. `tools/sightline_mcp/server.py`: `sim_fly` now fails a `Ground` contact whose normal is > ~45 deg off vertical
     (hillside strike). Active only after the sightline MCP server restarts. **Re-run `tools/sightline_mcp/test_sim.py`
     and `test_tool_matrix.py`** after that restart; they were not re-run after this change.
- Verification flight recipe: PIE -> `sim_fly takeoff` (the first sim call right after PIE start often times out
  while PIE boots: retry once) -> `move_to` (0, -60, -40) = ~45 m above the flood surface over the east hillslope
  and silt margin -> `sim_capture camera=survey image_types=[scene, segmentation]` -> `rtl`.
- `PS2-Survivor-Vision-Research-and-Build-Document.md` in the repo root is the user's copy of the updated doc; it is
  identical to `docs/SOLUTION_DOC.md` and deliberately left untracked.

## Handoff state (session 3, 2026-09-10 23:40) — READ THIS FIRST, it supersedes session 2's handoff

**Read `docs/QUALITY_GATE.md` before touching anything.** Seven separate pieces of work this session passed every
programmatic check and were still wrong; all seven were caught by looking at a picture. Eyeball verification is a
hard rule now, not a nicety. `docs/SCENE_REFERENCE.md` holds the user's two reference photographs and is the
visual target.

### Verified working (seen in a render, not inferred)
- **Scene**: 1072 actors — terrain (Ground, vertex normals, no corduroy), FloodWater, 73 Kerala houses,
  919 debris items placed by flood transport physics, 71 posed survivors, full lighting rig (GI off).
- **Survivor poses (F1)**: 7 postures authored as real one-key AnimSequences on all 9 Rocketbox skeletons
  (63 assets), left/right symmetric to 0.00 cm, characters fully textured. Standing 44.5 cm wide with arms down,
  waving 141.8, prone 167.2 long — the silhouettes actually differ, which is the point at 20-95 px.
- **Ground truth**: survivor positions match the sim to **0.00 m**. Two "buried" survivors are now genuinely
  buried under a debris cap — verified 0 visible pixels, 0 labels emitted.
- **Camera**: calibrated `f_px = 2548.72`, HFOV 73.98 deg, residual RMS 5.6 px over 48 observations (which is
  also a validation of the nadir projection chain). `simGetCameraInfo().fov` reports 89.9 and is WRONG.
- **Capture (F5) by FLYING (F2)**: `sightline/mission/survey.py` flies a boustrophedon and shoots on the move,
  terrain-following, with a tilt gate. A 60-frame test validated clean; survivors measured 51-93 px.
- **Pipeline lanes**: ~20k lines across ingest/geo/track/dedup/triage/export/coverage/plan/store/api/eval.
  **541 tests pass** (144 geo/triage/export/store + 298 ingest/track/dedup + 101 coverage/plan/eval).
  10 real bugs found and fixed by those tests, listed in `docs/lanes/*.md`.

### NOT done / not verified
- **No usable training dataset yet.** Two runs were discarded (teleport artefacts). The full flown run is the
  next thing to finish and validate.
- **No model trained.** F8b is minimal by decision (see CONTEXT decisions) but has not run.
- **Thermal (F9b)**: `tools/capture/thermal_ids.py` written (object-ID temperature table, decodable to
  absolute C), **never run**.
- **Weather scripting** never exercised in this scene; **time of day** not re-verified since the sky rebuild.
- **Scene vs `docs/SCENE_REFERENCE.md`**: no trees (the single biggest visual gap), no rubble field on the fan,
  no poles/wires, no boats, houses all pristine, water reads tan rather than green-teal.

### Run order for the scene (all idempotent, PIE OFF)
```
uv run python tools/scene/gen_terrain.py            # host: OBJ + zone masks (now writes vertex normals)
ue_python exec build_flood_valley.py                # level, lighting, water
ue_python exec build_materials.py                   # terrain + water materials
ue_python exec build_buildings.py                   # 73 houses (asserts every material compiles)
uv run python tools/scene/gen_actors.py             # host: 71 survivors, seeded
ue_python exec build_poses.py                       # 63 pose assets (asserts symmetry + face direction)
ue_python exec build_characters.py                  # binds the Rocketbox textures (the FBX import binds none)
ue_python exec build_actors.py                      # spawns/updates survivors IN PLACE (never renames)
uv run python tools/scene/gen_props.py              # host: 919 debris + burial caps
ue_python exec build_props.py                       # places debris
ue_python exec qa_shots.py                          # ALWAYS: render and LOOK
```
Capture, validate, then look:
```
uv run python -m sightline.mission.survey --alt 45 --speed 11 --out _artifacts/dataset/<name>
uv run python tools/capture/validate.py _artifacts/dataset/<name>      # exits non-zero if unclean
uv run python tools/capture/contact_sheet.py _artifacts/dataset/<name> # then OPEN the sheet
```

## Next actions (start here) — DEVELOPMENT PHASE
Read `docs/HANDBOOK.md` §5-§6 before touching the sim or the scene. The doc is now **simulation-first** (§5.5c):
the renderer is the deployment domain, randomisation OFF by default, every number carries a `sim`/`real` column.

0. **Close the five unverified items in "Handoff state" above**, commit.
1. **F1 flood-valley scene — foundation done, dressing in progress.**
   `/Game/Sightline/Maps/FloodValley` is the startup + game map. Rebuild from scratch, PIE off:
   `uv run python tools/scene/gen_terrain.py`, then
   `ue_python code="exec(open(r'D:\Sightline\tools\scene\build_flood_valley.py').read())"` (idempotent; aborts if the
   terrain transform is wrong), then `...build_materials.py` the same way. FloodLevel at runtime:
   `uv run python tools/scene/flood_level.py <asl_m>` (read-back verified). **Remaining for F1:**
   - settlement buildings on the terrace (zone G). **No free login-free realistic house asset exists** (see Assets);
     either build modular houses by script from the downloaded plaster / clay-tile / corrugated-sheet textures, or
     ask the user to sign in to Fab. Kerala type: 1-2 storey, flat concrete or clay-tile / corrugated roofs.
   - debris on the fan and channel from the Poly Haven models (rocks, logs, stumps, barrels, crates, jerrycans,
     tyres, trash bags, covered car) — glTF under `_downloads/assets/polyhaven/<id>/`, not yet imported.
   - waterline foam (ambientCG Foam001/002 downloaded), vegetation (palms/areca blocked, fern_02 available).
   - zone polygons / spawn masks exported by `gen_terrain.py` for the spawners (it only writes zone counts today).
2. **Actor + debris spawners** (seeded, reproducible): pose/submersion classes per §2.3 rows 2-3 and §6.2, tagged
   `Human_<id>` / `Animal_<id>`, MOVABLE actors only (static ones cannot be posed at runtime). Humans: 9 Microsoft
   Rocketbox rigged FBX (MIT) in `_downloads/assets/rocketbox/`, not yet imported; the UE5 mannequin
   (`D:\UE_5.8\Templates\TemplateResources\High\Characters`, 125 MB) has death anims usable for lying poses. Then
   day-1 #2 with a half-submerged actor + the `IgnoreMarked` camera, and day-1 #6 (detection-box flicker).
3. **Thermal**: object-ID temperature table first (day-1 test #3), then the §5.1 step 7 post-process material.
4. **Capture pipeline (F5)**: waypoint capture with `simPause` + `simGetImages` (SteppableClock profile), auto-labels
   from instance masks, per-frame telemetry CSV/JSON. Simulation-first: nominal slice at 40-60 m, split by seed.
5. **Assets blocked on logins** (ask the user): houses, palm/areca/coconut trees, rigged cow/goat/dog, uncovered car,
   photoscanned humans — Fab / Sketchfab / Mixamo / MetaHuman all need an account. Full list in
   `_downloads/assets/MANIFEST.md` "Blockers".

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
- [x] Editor opens the project with AirSim loaded; editor RAM measured: 3.4 GB RSS idle, 4.1-4.5 GB in PIE on
  FloodValley (2026-09-10, session 2)
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
| V4 AirSim tools | every sim_* tool through MCP (`tools/sightline_mcp/test_sim.py`) | results JSON in `_artifacts/verification/` | [x] **39/39** vs packaged Blocks (2026-09-10 19:19). Defects found+fixed: broken `landAsync` (own position-held landing + disarm; RTL lands 0.2 mm from home), velocity-only descent drifting into obstacles, static-actor pose silently "succeeding", `to_eularian_angles`/`to_quaternion`/`simGetImages(external=)` wrong API names, 60 s RPC timeout on long flights, falling-after-reset breaking takeoff, over-strict Ground collision rule. **Session 2 changed the Ground rule (steep contact fails): re-run after the next server restart** |
| V5 Python stack | all Python libs in doc: install pinned + functional tests (`tests/test_stack.py`) | `python_stack.md` | [~] |
| V6 external tools | Cesium, X-AnyLabeling, PMTiles+MapLibre offline, Fields2Cover, PX4/QGC, DroneModels, TAK, gamepad | `external_tools.md` | [x] X-AnyLabeling, PMTiles, MapLibre offline, DroneModels PASS; Cesium STAGED (`_staging/plugins`, BuildId match); Fields2Cover REJECT on Windows (own boustrophedon); PX4/QGC/TAK documented |
| V8 gamepad | XInput + pygame detection, live input, AirSim `rc_data`, API release/re-acquire handover | `_artifacts/verification/gamepad_airsim_*.json` | [x] **13 PASS / 0 FAIL** (2026-09-10 20:31). AirSim sees the pad (VID_045E, is_valid); axes full travel; under API control a held stick moves the drone 0.01 m in 4 s; `release` -> API off in 23 ms, vehicle follows the stick after **394 ms**; `arm` -> authority back, stabilised in **2.96 s**, then `move_to` obeyed to 0.22 m with the stick still held; rtl landed. F3's takeover semantics are proven |

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

## Known issues
- ~~Lumen active in PIE via Epic's FlyingExampleMap PPV~~ **RESOLVED 2026-09-10 (session 2)**: FloodValley is the
  startup and game-default map; its unbound `PPV_NoGI` forces GI = None and SSR.
- **Never run GPU/ML work while the editor + PIE are up** on this machine: 16 GB RAM / 8 GB VRAM triggers Windows
  memory-pressure warnings and makes both slow (observed 2026-09-10 20:05 with a TensorRT export alongside PIE).
- Free RAM with editor + PIE on FloodValley was 1.9-2.7 GB (2026-09-10). Close browsers before editor sessions.

## Day-1 unknowns (SOLUTION_DOC §10), with results
| # | Test | Status | Result |
|---|---|---|---|
| 1 | Cosys 3.4.1 loads on UE 5.8.2; editor RAM with project open | [x] | loads; editor RSS 3.4 GB idle / 4.1-4.5 GB in PIE on FloodValley (2 km terrain + water), 2026-09-10 |
| 2 | Instance seg hides submerged pixels (single-layer water / translucent plane); IgnoreMarked camera | [~] | single-layer-water plane **writes depth** (depth 45.75 m at nadir = drone height above water) and gets **its own instance colour** separate from the terrain, shoreline edge clean; the half-submerged ACTOR check and IgnoreMarked are still to do (needs a human actor) |
| 3 | Infrared image type after ID remap; capture FPS 4K raw vs PNG | [ ] | IR renders (all-black before any ID/temperature table: expected) |
| 4 | Xbox RemoteControlID, AllowAPIAlways, handover latency, moveByRC need | [x] | 13 PASS / 0 FAIL; API->RC 394 ms, RC->API 2.96 s (V8 above) |
| 5 | Weather visible in Scene, absent in Segmentation; time-of-day needs sky sphere | [ ] | FloodValley has BP_Sky_Sphere (hidden) wired to the sun + SkyAtmosphere; TOD not yet exercised |
| 6 | Detection-API box flicker on prone / 70 % submerged skeletal meshes | [ ] | |
| 8 | End-to-end ms on the 4060 for C1/C2/C3 | [x] | all six configs < 300 ms; C2/yolo26s 65 ms (V7 above) |

## Features (SOLUTION_DOC §8)
| # | Feature | MVP? | Status |
|---|---|---|---|
| F1 | Flood-valley scenario, 3 zones, weather/time/flood level, actor spawners | MVP | [~] terrain, zones, water + FloodLevel API, lighting, materials built; buildings/debris/actors/spawners to do |
| F2 | Coverage patterns, orbit-on-detect, revisit queue, battery/geofence | MVP | [ ] |
| F2b | Decision planner (Koopman + greedy) | Stretch | [ ] |
| F3 | Gamepad takeover/hand-back, HOLD/RTL, logged mode switches | MVP | [ ] (handover mechanics proven in V8) |
| F4 | PX4 SITL path | Stretch | [ ] |
| F5 | Synthetic dataset with visible/amodal boxes, attributes, telemetry | MVP | [ ] |
| F6 | Annotation guideline + X-AnyLabeling loop | MVP | [ ] |
| F7 | Ingest (sim export, DJI SRT; MAVLink/ULog stretch) | MVP | [ ] |
| F8 | Tiled YOLO26 RGB detector, TensorRT, frozen threshold | MVP | [ ] |
| F8b | **Demo model**: sim-only fine-tune, held-out sim scenes, randomisation off (§5.5c) | MVP | [ ] |
| F8c | Transferable model: real + synthetic, real-clip eval, domain gap | Stretch | [ ] |
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
- **2026-09-10 (session 2)**: Development phase started; stopped by the user for handoff mid-verification.
  Adopted the user's updated build doc (simulation-first §5.5c, F8b/F8c) as `docs/SOLUTION_DOC.md`. Built the
  FloodValley foundation entirely through MCP: terrain generator reworked (4 m grid, downstream fan lobe, blended
  terrace, command-post pad, OBJ in cm), terrain imported (Nanite off, complex collision, yaw +90 for NED, object
  name `Ground`), sun + BP_Sky_Sphere (hidden) + SkyAtmosphere + real-time sky light + fog + PPV (GI None),
  PlayerStart on the pad, movable single-layer-water `FloodWater`; FloodValley made the startup/game map.
  Verified in PIE: GPS = pad geopoint (after finding Cosys anchors OriginGeopoint at the UE world origin),
  takeoff/flight/RTL on the terrain, water writes depth and has its own instance colour, FloodLevel settable via the
  API (`flood_level.py`, read-back verified), `build_flood_valley.py` idempotent. 79 photoreal CC0/MIT assets
  downloaded by a subagent (`_downloads/assets/MANIFEST.md`, 1.74 GB); houses, palms, animals, uncovered car blocked
  on logins. Terrain (zone/slope/silt-line, 5 Poly Haven sets, anti-tiling) and water (panning normals) materials
  built by `build_materials.py`; last visual check pending (see Handoff state). Diagnosed a user report of high
  CPU / idle NVIDIA: UE is on the 4060; spike was imports; fixed the background-throttle setting's config section.
  Commits: b0213ca (foundation), then the handoff commit.
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

## Scene polish backlog (session 3) — what stands between now and a demo-grade disaster scene
Ordered by how much each changes what the camera sees.
1. **Debris and wreckage** — nothing floats or piles anywhere yet. A real debris-flow flood is defined by its
   wrack line: rafted timber, drums, sheeting and vehicles jammed against upstream walls; boulders and trunks
   dropped high on the fan; light plastics circling in eddies. 79 CC0 props are downloaded and imported
   (`data/scene/props.json`) but none are placed. Biggest single visual gap.
2. **Sitting pose sits on an invisible chair** — hips and knees both at ~80 deg, so a survivor on a flat roof
   has their feet dangling below the slab. On a roof people sit with legs out or crossed: flex the hips ~85 deg
   and keep the knees near straight so the pelvis rests ON the surface.
3. **Water reads as flat card at survey altitude** — the albedo is right but there is no large-scale surface
   variation, so a 45 m nadir frame is a uniform tan field. Needs a low-frequency normal/roughness break-up and
   some suspended-sediment streaking, plus foam at the waterline (ambientCG Foam001/002 downloaded).
4. **Roof textures tile visibly** from above; the buildings need the same anti-tiling treatment the terrain got.
5. **No vegetation** — fern_02 is imported; palms/areca are blocked on a Fab login (see Assets blockers).
6. **Damage state** — every house is pristine. Flood-damaged walls, missing sheets and collapsed sections would
   sell the scenario and add the occlusion cases the detector should be tested against.
