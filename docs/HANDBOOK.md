# Sightline handbook for session agents

Written 2026-09-10, at the end of the environment/validation phase and the start of the development phase.
Read `CLAUDE.md` first (rules and commands), then this file, then `TRACKER.md` (what to do next) and
`CONTEXT.md` (environment facts). `SOLUTION_DOC.md` is the specification; read it by section, it is ~77k tokens.

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

## 2. The development phase has NOT started

Nothing from the feature list (F1-F21) is implemented. The scene is still Epic's Blocks sample map.
`tools/scene/gen_terrain.py` is a **draft, never imported into Unreal and never validated** - treat it as a
starting sketch, not a deliverable, and regenerate its outputs before trusting anything in `data/scene/`.

Start from `TRACKER.md` -> "Next actions".

---

## 3. Hard rules (these are not negotiable)

1. **No shallow proxies.** Build what `SOLUTION_DOC.md` specifies. If something must be stubbed (e.g. the FMCW
   radar of Appendix C), label it a stub in code *and* in `TRACKER.md`.
2. **Never weaken a test to make it pass.** Every check in this repo was written because something real broke.
   If a suite fails, fix the cause or record the failure honestly.
3. **Everything installs to D:.** See `CONTEXT.md` §3 for the redirected caches. C: has ~40 GB free and is the
   binding constraint after RAM.
4. **Guardrail R10:** no code path may delete a record or mark a search segment "cleared".
5. **Every accuracy number states its slice** (zone x altitude x band x time-of-day x occlusion x posture) and
   says whether it came from simulation or real footage.
6. **Pinned dependencies only**: `uv add`, never `pip install` into the venv; commit `uv.lock`.

---

## 4. Machine limits — the most common cause of "weird" failures

16 GB RAM, RTX 4060 with 8 GB VRAM, ~24.7 GB commit limit. Consequences, all observed in practice:

- **Never run GPU/ML work while the editor or PIE is running.** A TensorRT export alongside PIE triggered a
  Windows memory-pressure warning and made both crawl (2026-09-10 20:05).
- **Never train while the engine runs** (`SOLUTION_DOC.md` §4). Capture first, close the engine, then train.
- **One simulator at a time.** The packaged Blocks build and the editor's PIE both bind AirSim's port 41451.
  Check with `status` before launching anything.
- Close browsers before an editor session; the editor wants 5+ GB.
- Use a **packaged build** for long data-capture runs; the editor costs several GB more.

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
  `sim_spawn_object` (asset names from `sim_list_assets`).
- **Cosys 3.4.1 logs nothing to the UE log** (every `UE_LOG` in `UAirBlueprintLib::LogMessage` is commented out).
  Prove settings are in use via `listVehicles()`, camera resolution and the home geopoint - not by grepping the log.
- **`materials.csv` must sit next to the executable** or material stencil initialisation is skipped
  (`tools/setup/install_materials.ps1` installs it; re-run after packaging).
- **Braking from 6 m/s overshoots ~2 m and settles in 7-8 s.** Let the vehicle settle before precise captures.
- **Epic's MCP server drops idle HTTP sockets after 15 s** (hard-coded). On "socket connection was closed
  unexpectedly", **retry the call once**.
- **After an editor restart, reconnect the `unreal` server** (`/mcp` -> Reconnect, or a new session).
  Port 8000 opens before the server answers; poll `status` first.

---

## 6. Traps specific to the work that comes next

**Scene building (F1)**
- Use **static-mesh terrain, not Landscape**, and avoid foliage actors for anything that must be labelled:
  Cosys gives Landscape/foliage a single default instance-segmentation colour, which corrupts auto-labels (§5.1).
- Author the level with **GI = None**. Epic's sample map re-enables Lumen through its PostProcessVolume, which is
  why the editor logs Lumen ray-tracing warnings; do not inherit that.
- The **water surface must write depth/stencil** or submerged body parts will still appear in the label passes
  (§5.1 step 6). Verify with an actual capture, not by assumption.
- A single flat water plane is only physical over a gentle gradient. Keep the valley fall small or model the
  water surface per reach.
- The DEM says **1060 m** at the origin geopoint; `sim/settings/*.json` now matches. Keep them consistent.

**Synthetic data (F5)**
- Thermal in Cosys is a **per-object ID -> grey map**, not radiometry. The doc's realistic thermal (per-material
  temperature/emissivity, diurnal state, immersion cooling) is the §5.1 step 7 post-process upgrade, and it is
  real work, not a setting.
- Auto-labels come from **instance segmentation masks**, and the visible-extent box is the training box (§6.3).
- Measure 4K capture throughput before committing to it (day-1 test #3, `tools/day1/capture_benchmark.py`).

**Pipeline (F7-F19)**
- `dji-log-parser` has **no Python bindings** (the doc is wrong); use the Rust CLI via subprocess.
- `torchcodec` ships no FFmpeg on Windows and borrowing PyAV's crashed the process; use PyAV + PyNvVideoCodec.
- Fields2Cover does not build on Windows; write the boustrophedon generator (§5.3a says ~60 lines).
- Only one of `opencv-python` / `-headless` / `-contrib` may be installed. FiftyOne lives in `envs/fiftyone`.

---

## 7. How to work a session

1. `uv run python tools/doctor.py --live` - catches a broken environment before you waste an hour.
2. Read `TRACKER.md` "Next actions"; work the top item.
3. Drive Unreal through MCP. Long jobs (`ue_build`, `ue_package`) return a `job_id`; poll `job_status`.
4. Re-run the relevant suite after changes:
   `tests/test_mcp_protocol.py`, `tools/sightline_mcp/test_tool_matrix.py`, `test_sim.py`, `tests/test_stack.py`.
5. **Close the editor and any sim when you finish.**
6. Update `TRACKER.md` (status + session log) and `CONTEXT.md` (new facts/decisions), then commit.

**If you spawn subagents:** give each an exclusive resource. Only one agent may own the editor, one the sim port,
one the Python environment (`pyproject.toml`/`uv.lock`). Tell them to re-read a shared file immediately before
editing and to use small unique-string edits - several agents edit `server.py`.

---

## 8. Open items inherited by the development phase

| Item | State | Who can close it |
|---|---|---|
| ~~Gamepad handover (F3, day-1 #4)~~ | **CLOSED**: 13 PASS / 0 FAIL; API→RC 394 ms, RC→API 2.96 s | — (re-run `tools/day1/gamepad_airsim.py` after changes to `sim_fly`) |
| Cesium for Unreal | staged in `_staging/plugins`, BuildId matches | integration steps in `external_tools.md`; needs the user's Cesium ion sign-in |
| X-AnyLabeling on GPU | runs on CPU; cuDNN 9 missing | unpack cuDNN 9 into `D:\Tools\cudnn` |
| Human/animal assets | none acquired | Mixamo needs an Adobe login (user); UE mannequin is the fallback |
| ~~GPU latency table (day-1 #8)~~ | **CLOSED**: all six configs < 300 ms; design pass C2/yolo26s = 65 ms (`gpu_latency.md`) | INT8 and the Orin table (F20) remain open |
| Real datasets (§6.1) | none downloaded | licences differ per set; record each in the data card |
| PX4 SITL / QGC (F4, stretch) | not installed; WSL has no distro | steps in `external_tools.md` |

---

## 9. Where things live

```
CLAUDE.md              rules, commands, MCP overview (read first)
docs/HANDBOOK.md       this file
docs/TRACKER.md        status, next actions, feature table, session log
docs/CONTEXT.md        machine, paths, decisions, verified facts, pitfalls
docs/SETUP.md          rebuild from scratch
docs/SOLUTION_DOC.md   the specification
docs/verification/     one report per verification stream (the evidence)
tools/sightline_mcp/   the MCP server + its test suites
tools/day1/            day-1 probes and benchmarks (diag_*, gamepad, latency, capture)
tools/scene/           scene generation (draft)
tools/setup/           reproducible install/restore scripts
sim/SightlineSim/      the UE 5.8 project (AirSim plugin not committed; fetch_airsim.ps1 restores it)
sim/settings/          AirSim settings profiles, passed with -settings=
tests/                 pytest suites (protocol, stack)
_artifacts/ _logs/ _downloads/ _build/ _staging/   local only, gitignored
```
