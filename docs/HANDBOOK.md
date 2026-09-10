# Sightline handbook for session agents

Written 2026-09-10 at the start of the development phase; **updated at the end of session 2 (2026-09-10)**.
Read `CLAUDE.md` first (rules and commands), then this file, then `TRACKER.md` (start at "Handoff state" and
"Next actions") and `CONTEXT.md` (environment facts). `SOLUTION_DOC.md` is the specification; read it by section,
it is ~80k tokens.

---

## 1. What exists right now

**The whole project is developed through MCP.** Claude drives Unreal and the simulator; there is no manual
click-through workflow. Two servers:

| Server | Transport | Alive when | What it does |
|---|---|---|---|
| `unreal` | HTTP `localhost:8000/mcp` | only while the editor runs | Epic's in-editor MCP plugin: **31 toolsets, 392 tools** (actors, assets, viewport capture, PIE, logs, config, PCG, Niagara, physics, automation). Discovery via `list_toolsets` -> `describe_toolset` -> `call_tool` |
| `sightline` | stdio (`tools/sightline_mcp/server.py`) | always | **27 tools** we own: editor launch/close, builds and packaging as background jobs, editor Python (live + headless), logs, and all Cosys-AirSim control (flight, capture, weather, clock, objects, detections) |

Verified working (evidence in `docs/verification/`, one report per stream):

| Stream | Result |
|---|---|
| Engine/build/editor/log/job tools | all PASS; Epic side 17 real calls incl. a viewport PNG (`engine_tools.md`) |
| MCP protocol robustness | `tests/test_mcp_protocol.py` **22/22** (`mcp_protocol.md`) |
| AirSim tools | **39/39** against the packaged build *and* in-editor PIE (`dev_workflow.md`) |
| Whole tool matrix | **33/33** editor+sim up, **12/12** both down, all 27 tools (`test_tool_matrix.py`) |
| Claude Code CLI -> MCP | real flight + capture driven by the model; image survives the MCP path |
| Python/ML stack | **37 pass / 0 fail**; torch 2.14+cu130, TensorRT 11.3 (`python_stack.md`) |
| External tools | offline PMTiles+MapLibre map, X-AnyLabeling, DroneModels PASS; Cesium staged (`external_tools.md`) |
| Environment doctor | `tools/doctor.py --live` 0 FAIL |

**The project builds and runs:** `SightlineSimEditor` cold build 881 s, incremental 20 s, editor ready ~45 s,
PIE gives a working AirSim on port 41451.

---

## 2. Development state (end of session 2)

**F1 (the flood-valley scene) has a working foundation; everything after F1 is untouched.**

The map `/Game/Sightline/Maps/FloodValley` is the project's startup and game-default map. It is produced by three
scripts, all idempotent and all runnable through MCP:

| Step | Command | Produces |
|---|---|---|
| 1. terrain data | `uv run python tools/scene/gen_terrain.py` (seed 7, 2048 m, 4 m cells) | `data/scene/flood_valley.{obj,json,_height.npy,_zones.png,_preview.png}` (obj/npy gitignored) |
| 2. level | `ue_python code="exec(open(r'D:\Sightline\tools\scene\build_flood_valley.py').read())"` (PIE off) | the level: terrain `Ground`, `FloodWater`, sun, hidden BP_Sky_Sphere, SkyAtmosphere, sky light, fog, `PPV_NoGI`, `PlayerStart_Home`, AirSim game mode. Aborts if a line trace at the pad disagrees with the generator |
| 3. materials | `ue_python code="exec(open(r'D:\Sightline\tools\scene\build_materials.py').read())"` (PIE off) | `M_FloodValleyTerrain` (zone mask + slope + flood silt line, 5 Poly Haven 2K sets, anti-tiling) and `M_FloodWater` (single-layer water, silt turbidity parameters, panning normals) |
| runtime | `uv run python tools/scene/flood_level.py [asl_m]` | reads / sets the flood level through `simSetObjectPose`, read-back verified |

Scene layout (local metres, x = east, y = north, origin = map centre = 11.4870 N 76.1450 E): deposit fan upstream
(north), meandering channel, flooded settlement terrace on the east bank (0.5-3.7 m under flood stage 1061.68 m
ASL), command-post pad at east 492 / north 148, 4.9 m above flood stage = the drone's home.

**Verified in PIE:** GPS at the pad equals the pad's computed geopoint; takeoff, flight, RTL on the terrain; water
writes depth and has its own instance-segmentation colour; FloodLevel changes through the API; the level script
rebuilds cleanly. **Written but not yet verified** — see TRACKER "Handoff state": the final materials in a capture,
the lowered survey camera, time of day with the hidden sky sphere, the background-throttle fix, and the new
steep-contact rule in `sim_fly`.

**Assets:** `_downloads/assets/` (gitignored, 1.74 GB, `MANIFEST.md`) holds 79 photoreal assets: Poly Haven/ambientCG
CC0 textures and models (rocks, logs, stumps, barrels, crates, plastics, tyre, covered car, fern) and 9 Microsoft
Rocketbox rigged humans (MIT). Only the five terrain texture sets are imported so far (`/Game/Sightline/Textures/`).
**Blocked on logins (ask the user):** houses, palms/areca/coconut, rigged animals, uncovered car, photoscanned
humans. The user's bar: **photoreal only, no stylised or low-poly assets**, chosen for realism and runtime cost.

---

## 3. Hard rules (these are not negotiable)

1. **No shallow proxies.** Build what `SOLUTION_DOC.md` specifies. If something must be stubbed (e.g. the FMCW
   radar of Appendix C), label it a stub in code *and* in `TRACKER.md`.
2. **Never weaken a test to make it pass.** Every check in this repo was written because something real broke.
   If a suite fails, fix the cause or record the failure honestly.
3. **Everything installs to D:.** See `CONTEXT.md` §3 for the redirected caches. C: has ~40 GB free and is the
   binding constraint after RAM.
4. **Guardrail R10:** no code path may delete a record or mark a search segment "cleared".
5. **Every accuracy number states its slice** (zone x altitude x band x time-of-day x occlusion x posture) **and its
   domain** (`sim` or `real`, in the same sentence). Never average a sim number with a real one (§5.5c).
6. **Pinned dependencies only**: `uv add`, never `pip install` into the venv; commit `uv.lock`.
7. **Simulation-first (§5.5c).** The renderer is the deployment domain. The demo model (F8b) is trained on sim
   frames only, split by scenario seed (never by frame), with **domain randomisation OFF by default**; do not turn
   randomisation on midway and compare numbers across the switch.
8. **Keep `TRACKER.md` and `CONTEXT.md` current as you go**, not only at the end (the user asked for this).

---

## 4. Machine limits — the most common cause of "weird" failures

16 GB RAM, RTX 4060 with 8 GB VRAM, ~24.7 GB commit limit. Consequences, all observed in practice:

- **Never run GPU/ML work while the editor or PIE is running.** A TensorRT export alongside PIE triggered a
  Windows memory-pressure warning and made both crawl (2026-09-10 20:05).
- **Never train while the engine runs** (`SOLUTION_DOC.md` §4). Capture first, close the engine, then train.
- **One simulator at a time.** The packaged Blocks build and the editor's PIE both bind AirSim's port 41451.
  Check with `status` before launching anything.
- Close browsers before an editor session; the editor wants 5+ GB. With editor + PIE on FloodValley, free RAM was
  1.9-2.7 GB.
- Use a **packaged build** for long data-capture runs; the editor costs several GB more.
- **UE renders on the RTX 4060** (verified). The Intel iGPU drives the display (Optimus), so it shows activity too.
  CPU spikes during imports / texture compression / shader compiles are expected. With the bare scene the 4060 sat
  at P5 (~750 MHz) under PIE with throttle reason `Idle` only: the load was light, not capped.

---

## 5. Simulator behaviour you must know (all source-verified or measured)

These cost hours to rediscover. Details and line references in `CONTEXT.md` §7.

- **`landAsync` never lands.** Upstream `MultirotorApiBase::land()` returns while still airborne (measured: 0.2 s
  at 21 m). Use `sim_fly land` / `rtl`, which stream a position-held descent and confirm touchdown, then disarm.
  **Only a disarm makes SimpleFlight report `Landed`.**
- **The 60 ms API watchdog** (`api_goal_timeout`) prints "API call was not received, entering hover mode for
  safety" after *every* command. It is SimpleFlight's own position hold (~5 cm accuracy) and is **not an error**.
  Continuous motion needs streamed setpoints.
- **The first `simGetImages` of a fresh engine process returns unconverted buffers** - depth in 0..0.6 instead of
  metres, oversized normals - with correct dimensions and no error. `sim_capture` discards a warm-up frame per
  connection. If you write your own capture path, do the same or your dataset will be silently wrong.
- **`simPause` freezes physics and pose**, but under `ScalableClock` the state timestamp keeps following wall
  time: judge a pause by pose, not timestamp. Use the `capture_4k` profile (SteppableClock) for deterministic runs.
- **Static level actors cannot be moved** (`simSetObjectPose` returns False); spawn movable ones with
  `sim_spawn_object` (asset names from `sim_list_assets`) or give level actors Movable mobility (`FloodWater`).
- **Cosys 3.4.1 logs nothing to the UE log** (every `UE_LOG` in `UAirBlueprintLib::LogMessage` is commented out).
  Prove settings are in use via `listVehicles()`, camera resolution and the home geopoint - not by grepping the log.
- **`OriginGeopoint` is anchored at the UE WORLD ORIGIN, not the PlayerStart** (measured: vehicle GPS =
  OriginGeopoint + PlayerStart offset, to 0.3 m). Settings carry the map centre at `base_z_m` (1046.007 m).
- **AirSim settings are re-read on every PIE start**; no editor restart is needed after editing `sim/settings/*.json`.
- **Cosys names objects by `GetName()`, not the editor label.** Rename actors whose API name matters
  (`actor.rename("FloodWater")`).
- **`sim_fly` ignores contact with an object named `Ground`**, so the terrain actor must be named `Ground`. From the
  next sightline-server start, a `Ground` contact steeper than ~45 deg fails as a hillside strike.
- **The first sim tool call right after PIE starts often times out** while PIE boots; `sim_state` then shows
  whether the command ran. Retry once.
- **`materials.csv` must sit next to the executable** or material stencil initialisation is skipped
  (`tools/setup/install_materials.ps1` installs it; re-run after packaging).
- **Braking from 6 m/s overshoots ~2 m and settles in 7-8 s.** Let the vehicle settle before precise captures.
- **Epic's MCP server drops idle HTTP sockets after 15 s** (hard-coded). On "socket connection was closed
  unexpectedly", **retry the call once**.
- **After an editor restart, reconnect the `unreal` server** (`/mcp` -> Reconnect, or a new session).
  Port 8000 opens before the server answers; poll `status` first.

---

## 6. Traps specific to the work that comes next

**Scene building (F1)** — the ones marked ✔ were hit and solved in session 2; keep them solved.
- Use **static-mesh terrain, not Landscape**, and avoid foliage actors for anything that must be labelled:
  Cosys gives Landscape/foliage a single default instance-segmentation colour, which corrupts auto-labels (§5.1).
- ✔ **GI = None**: FloodValley's `PPV_NoGI` forces it. Never use Epic's FlyingExampleMap (its PPV re-enables Lumen).
- ✔ **UE's OBJ importer flips handedness** (x = east -> UE +X, y = north -> UE -Y). The terrain actor carries yaw
  +90 so UE X = north, Y = east = NED. **Every spawner must use** `UE (X, Y, Z) cm = (north*100, east*100,
  (asl - base_z)*100)`, also in `flood_valley.json` `ue_import`.
- ✔ **OBJ must be in centimetres** (UE does not rescale) and **imported meshes arrive with Nanite ON** (the triangle
  count then reports the fallback mesh). `build_flood_valley.py` handles both and asserts the triangle count.
- ✔ **Editor Python has no numpy.** Do grid maths on the host and read `flood_valley.json` in the editor.
- ✔ **Builds, imports and saves fail silently while PIE runs** (32 px texture placeholders, `save` returns False,
  `editor_request_end_play()` is asynchronous). End PIE in one call, edit in the next.
- ✔ **Material scripting:** deleting a referenced material fails (ensure + callstack in the log); reuse and clear
  instead. `delete_all_material_expressions` leaves nodes behind; delete each and assert zero.
  `BlendAngleCorrectedNormals` is a function, not an expression. Check a material by its used-texture count: 0 means
  it failed to compile and renders as the grey checker.
- ✔ **Real-time sky light needs a SkyAtmosphere** (red viewport warning otherwise). BP_Sky_Sphere stays in the level
  only for Cosys' time-of-day sun lookup, hidden.
- The **water surface must write depth/stencil** so submerged body parts are hidden in the label passes: the
  single-layer-water plane does write depth (measured); the half-submerged *actor* test is still to do.
- A single flat water plane is only physical over a gentle gradient; the valley falls 8 m over 2 km on purpose.
- Keep the DEM/geopoint consistent: `OriginGeopoint` altitude = `base_z_m` from `flood_valley.json`. If the
  generator changes, regenerate and update both settings files.

**Synthetic data (F5)**
- Thermal in Cosys is a **per-object ID -> grey map**, not radiometry. The doc's realistic thermal (per-material
  temperature/emissivity, diurnal state, immersion cooling) is the §5.1 step 7 post-process upgrade, and it is
  real work, not a setting.
- Auto-labels come from **instance segmentation masks**, and the visible-extent box is the training box (§6.3).
- Measure 4K capture throughput before committing to it (day-1 test #3, `tools/day1/capture_benchmark.py`).
- Simulation-first: fly the nominal slice at 40-60 m and tile the 4K frame at native resolution (§5.5c).

**Pipeline (F7-F19)**
- `dji-log-parser` has **no Python bindings** (the doc is wrong); use the Rust CLI via subprocess.
- `torchcodec` ships no FFmpeg on Windows and borrowing PyAV's crashed the process; use PyAV + PyNvVideoCodec.
- Fields2Cover does not build on Windows; write the boustrophedon generator (§5.3a says ~60 lines).
- Only one of `opencv-python` / `-headless` / `-contrib` may be installed. FiftyOne lives in `envs/fiftyone`.

---

## 7. How to work a session

1. `uv run python tools/doctor.py --live` - catches a broken environment before you waste an hour.
   (`uv` is at `D:\Tools\uv\uv.exe`; it may not be on PATH inside the agent's PowerShell.)
2. Read `TRACKER.md` "Handoff state" and "Next actions"; work the top item.
3. Drive Unreal through MCP. Long jobs (`ue_build`, `ue_package`) return a `job_id`; poll `job_status`.
   Batch editor work into scripts under `tools/scene/` and run them with `exec(open(...).read())` — fewer round
   trips, and the scene stays reproducible. The user asked for speed without dropping quality.
4. Re-run the relevant suite after changes:
   `tests/test_mcp_protocol.py`, `tools/sightline_mcp/test_tool_matrix.py`, `test_sim.py`, `tests/test_stack.py`.
5. **Close the editor and any sim when you finish.**
6. Update `TRACKER.md` (status + session log) and `CONTEXT.md` (new facts/decisions), then commit.

**If you spawn subagents:** give each an exclusive resource. Only one agent may own the editor, one the sim port,
one the Python environment (`pyproject.toml`/`uv.lock`). Tell them to re-read a shared file immediately before
editing and to use small unique-string edits - several agents edit `server.py`. `SendMessage` is disabled in these
sessions, so a running subagent cannot be redirected: brief it fully up front (e.g. the asset quality bar).

---

## 8. Open items

| Item | State | Who can close it |
|---|---|---|
| ~~Gamepad handover (F3, day-1 #4)~~ | **CLOSED**: 13 PASS / 0 FAIL; API→RC 394 ms, RC→API 2.96 s | — (re-run `tools/day1/gamepad_airsim.py` after changes to `sim_fly`) |
| ~~GPU latency table (day-1 #8)~~ | **CLOSED**: all six configs < 300 ms; design pass C2/yolo26s = 65 ms (`gpu_latency.md`) | INT8 and the Orin table (F20) remain open |
| Session-2 unverified items | final materials, camera mount, TOD with hidden sky sphere, throttle fix, steep-contact rule | next agent (TRACKER "Handoff state") |
| Houses, palms, rigged animals, uncovered car, photoscanned humans | blocked: Fab / Sketchfab / Mixamo / MetaHuman need a login | the user (sign-in) — or build houses by script from the downloaded textures |
| Rocketbox humans + Poly Haven models | downloaded, not imported | next agent (spawners) |
| Cesium for Unreal | staged in `_staging/plugins`, BuildId matches | integration steps in `external_tools.md`; needs the user's Cesium ion sign-in |
| X-AnyLabeling on GPU | runs on CPU; cuDNN 9 missing | unpack cuDNN 9 into `D:\Tools\cudnn` |
| Real datasets (§6.1) | none downloaded; optional for the demo model (§5.5c), needed for F8c | licences differ per set; record each in the data card |
| PX4 SITL / QGC (F4, stretch) | not installed; WSL has no distro | steps in `external_tools.md` |

---

## 9. Where things live

```
CLAUDE.md              rules, commands, MCP overview (read first)
docs/HANDBOOK.md       this file
docs/TRACKER.md        handoff state, next actions, feature table, session log
docs/CONTEXT.md        machine, paths, decisions, verified facts, pitfalls (§7 has the FloodValley facts)
docs/SETUP.md          rebuild from scratch
docs/SOLUTION_DOC.md   the specification (simulation-first version, 2026-09-10)
docs/verification/     one report per verification stream (the evidence)
tools/sightline_mcp/   the MCP server + its test suites
tools/day1/            day-1 probes and benchmarks (diag_*, gamepad, latency, capture)
tools/scene/           gen_terrain.py, build_flood_valley.py, build_materials.py, flood_level.py
tools/setup/           reproducible install/restore scripts
data/scene/            generator outputs (json + pngs tracked; obj/npy regenerated)
sim/SightlineSim/      the UE 5.8 project (AirSim plugin not committed; fetch_airsim.ps1 restores it)
  Content/Sightline/   Maps/FloodValley, Terrain/, Water/, Textures/
sim/settings/          AirSim settings profiles, passed with -settings=
tests/                 pytest suites (protocol, stack)
_downloads/assets/     CC0/MIT scene assets + MANIFEST.md (gitignored)
_artifacts/captures/   every sim_capture (gitignored)
_artifacts/ _logs/ _downloads/ _build/ _staging/   local only, gitignored
```
