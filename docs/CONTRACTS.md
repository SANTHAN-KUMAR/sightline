# Pipeline contracts and module ownership

Written 2026-09-10 (session 3) by the orchestrator, when the build fanned out to parallel agents.
**Read this before writing any code under `sightline/`.**

## 1. The one rule that makes parallel work possible

`sightline/schemas.py` is a **frozen contract** owned by the orchestrator. Every module exchanges those types and
nothing else. You may **not** rename a field, change its units, or change its meaning. If something is genuinely
missing, add an **optional** field with a default, and add a line to §5 below saying what you added and why.

Units: SI, suffixes `_m` / `_s` / `_deg` / `_px` / `_utc`. Angles in degrees. Timestamps are UTC POSIX floats.
Quaternions are `(w, x, y, z)`. Geographic order is `lat, lon` everywhere **except** inside GeoJSON, which is
`[lon, lat, alt]` by RFC 7946.

## 2. Module ownership — one owner per directory, no cross-writes

| Dir | Feature | Owner lane | Consumes | Produces |
|---|---|---|---|---|
| `sightline/schemas.py` | — | **orchestrator** | — | every type below |
| `sightline/common/` | shared helpers (geodesy, time, io) | orchestrator | — | pure functions |
| `sightline/ingest/` | F7 | B1 | files / sim export | `FrameBundle` stream |
| `sightline/detect/` | F8, F8b, F9, F10 | C (ML) | `FrameBundle` | `list[Detection]` |
| `sightline/geo/` | F13 | B2 | `Detection` + `Telemetry` + `Intrinsics` | `GeoFix` |
| `sightline/track/` | F11 | B3 | `list[Detection]` per frame | `Track` |
| `sightline/dedup/` | F12 | B3 | `list[Track]` | `list[Record]` |
| `sightline/triage/` | F14 (score) | B4 | `list[Record]` | ranked `list[Record]` |
| `sightline/export/` | F14 (formats), F17 | B4 | `list[Record]` | GeoJSON / KML / KMZ / CoT |
| `sightline/coverage/` | F16, F16b | B6 | `Telemetry` + `Intrinsics` | `CoverageGrid` |
| `sightline/plan/` | F2, F2b | B6 | `CoverageGrid` | waypoint lists |
| `sightline/store/` | F18 | B5 | `Record` | SQLite log + outbox |
| `sightline/api/` | F15, F18 | B5 | `Record` | FastAPI + WebSocket |
| `app/map/` | F15 | B5 | WebSocket GeoJSON | MapLibre page |
| `sightline/eval/` | F19 | B7 | records + ground truth | `MetricRow` list |
| `sightline/mission/` | F2, F3 | **orchestrator (sim lane)** | sim | flight + telemetry |
| `tools/scene/`, `tools/capture/` | F1, F5 | **orchestrator (sim lane)** | Unreal | scene + dataset |
| `docs/annotation_guideline.md` | F6 | B8 | — | the guideline |

**Never edit another lane's directory.** If you need a change there, write it in your report and the orchestrator
will route it. `pyproject.toml` / `uv.lock` are orchestrator-only: if a package is missing, report it, do not add it.

## 3. Resource rules (this machine is small: 16 GB RAM, 8 GB VRAM)

1. **Do not touch the Unreal editor or AirSim.** No `mcp__sightline__*` sim/editor tools, no PIE. The orchestrator
   owns them. If your module needs sim data, use the fixtures in `tests/fixtures/` or synthesise arrays.
2. **Do not import `torch`, `ultralytics`, `tensorrt`, `rfdetr` or run anything on the GPU** unless you are the
   ML lane and the orchestrator has told you the editor is closed. Everything else must unit-test on numpy/CPU.
3. `pytest` runs must be scoped to your own test file (`uv run pytest tests/test_<yours>.py -q`), never the whole
   suite — other agents are running.
4. `uv` is at `D:\Tools\uv\uv.exe` and may not be on PATH. Everything installs to D:.

## 4. Definition of done for a lane

* The module imports cleanly and its public functions have type hints matching `schemas.py`.
* A `tests/test_<lane>.py` exists, is **runnable offline**, and passes. Tests assert real behaviour with numbers
  from the doc — never weaken a test to make it pass (hard rule 2 in `docs/HANDBOOK.md`).
* Numbers that leave your module carry a `SliceKey` with an explicit `domain` (`sim` or `real`).
* No code path deletes a record or marks a segment cleared (guardrail R10).
* Anything stubbed is labelled a stub in the code **and** reported so it reaches `docs/TRACKER.md`.
* You wrote a short report: what works, what is verified, what is left, what you need from other lanes.

## 5. Contract amendments (append here; do not rewrite history)

| Date | Lane | Change |
|---|---|---|
| 2026-09-10 | orchestrator | Initial freeze at `SCHEMA_VERSION = 1.0.0`. |
| 2026-09-10 | orchestrator | **1.1.0**: added `"waving"` to `Posture` / `POSTURES`. It is a real authored posture (`tools/scene/build_poses.py`), 12 of the 71 survivors use it, and it has a distinctive silhouette from above (measured 26x25 px where standing is 17x30). Additive only: nothing was renamed or repurposed. |
| 2026-09-10 | orchestrator | `common/geodesy.quat_to_euler` gained an explicit gimbal-lock branch. At \|pitch\| = 90 - exactly where a nadir survey camera sits - the naive atan2 form split the rotation arbitrarily between roll and yaw, so a quat->euler->quat round trip could rotate the azimuth by up to 180 deg. Worst round-trip error over a pitch/yaw grid: 180 deg -> 2.4e-06 deg. Found by the ingest lane in its own copy of the same formula. |
