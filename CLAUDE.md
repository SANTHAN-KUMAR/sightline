# Sightline — Aerial Survivor Triage System (PS2)

Read this first in every session. It is the entry point for continuing work in a new session or account.

## Read order for a fresh session
1. `CLAUDE.md` (this file): rules, commands, tooling.
2. `docs/TRACKER.md`: what is done, in progress and next. **Start from its "Next actions" list.**
3. `docs/CONTEXT.md`: environment facts, decisions, deviations from the solution doc, known pitfalls.
4. `docs/SOLUTION_DOC.md`: the full research/build document (source of truth for *what* to build). Read by
   section on demand; it is long (~77k tokens).

## Non-negotiable project rules
- **No shallow proxies.** Build the real thing the solution doc specifies (real Cosys-AirSim sensors, real
  geolocation chain, real metrics). If something must be stubbed (e.g. FMCW radar, Appendix C), label it as a
  stub in code *and* in `docs/TRACKER.md`.
- **Everything installs to D:.** Do not install tools, caches, datasets or models on C:. Cache env vars are set
  at user level (see `docs/CONTEXT.md` §Paths). If a new tool defaults to C:, redirect it and record how.
- **Pinned dependencies.** Python env is managed by `uv` with `uv.lock` committed. Add packages with
  `uv add <pkg>`; never `pip install` into the venv. Python is 3.11 (cosysairsim constraint).
- **Simulation numbers are labelled as simulation.** Every metric states its slice (doc §5.12).
- **Guardrail R10:** no code path may delete a record or mark a segment "cleared".
- **Update the tracker** (`docs/TRACKER.md`) at the end of every meaningful chunk of work, including a
  session-log line, so the next session can resume. Record new facts/decisions in `docs/CONTEXT.md`.

## Tooling: two MCP servers (configured in `.mcp.json`)
| Server | Transport | Lives | Use it for |
|---|---|---|---|
| `unreal` | HTTP `http://localhost:8000/mcp` | inside the editor (Epic's experimental ModelContextProtocol plugin, auto-started by project config) | editor toolsets: actors, viewport capture, PIE start/stop, logs, config, PCG, physics, automation tests. Tools are discovered via `list_toolsets` / `describe_toolset` / `call_tool`. Only up while the editor runs. |
| `sightline` | stdio (`tools/sightline_mcp/server.py`) | outside the editor | launch/close editor, standalone `-game` runs, UBT builds and packaging (background jobs), Python inside the running editor (`ue_python`), headless editor Python, logs, and all Cosys-AirSim control (fly, capture RGB/seg/IR/depth, weather, time, detections). Works when the editor is closed or crashed. |

Typical loop: `status` -> `ue_build` + `job_status` -> `editor_launch(wait_ready_s=900)` -> edit via `ue_python` /
`unreal` toolsets -> PIE start -> `sim_ping` -> `sim_fly` / `sim_capture` (returns a preview image) -> `ue_log`.

## Commands
```
uv sync                                        # restore the exact Python env from uv.lock
uv run python tools/doctor.py                  # verify the whole toolchain (run at session start)
uv run python tools/sightline_mcp/smoke_test.py  # verify the sightline MCP server end to end
```
Unreal: `D:\UE_5.8` (5.8.2). Project: `sim/SightlineSim/SightlineSim.uproject`. Visual Studio 2022 17.14 at
`D:\VS\2022\Community`.

## Layout
```
docs/            CONTEXT.md, TRACKER.md, SOLUTION_DOC.md, SETUP.md
sim/SightlineSim UE 5.8 C++ project (Cosys-AirSim plugin in Plugins/AirSim, not committed)
sim/settings/    AirSim settings profiles, passed with -settings=<path> (never Documents\AirSim)
tools/           doctor.py, sightline_mcp/ (MCP server), setup/ (reproducible install scripts)
vendor/wheels/   cosysairsim-3.4.1 wheel from the Cosys release
_downloads/ _logs/ _artifacts/ _build/   local only (gitignored)
```
