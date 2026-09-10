# Lane B5 — record store, API and C2 map (F15, F18)

Session 3, 2026-09-10/11. Owner directories: `sightline/store/`, `sightline/api/`, `app/map/`,
`tests/test_store.py`, `tests/test_api.py`. Nothing outside those was edited (`git status` confirms
`schemas.py`, `common/`, `coverage/`, `plan/`, `pyproject.toml` and `uv.lock` are untouched).

**Status: the map, the API, the live feed and the offline queue all run and have been looked at.**
54 tests pass (21 store + 33 API). The headless render passes 11 checks including "zero external requests".
One real bug was found by looking at the picture (§6).

---

## 1. How to run it

```
D:\Tools\uv\uv.exe run python -m sightline.api.serve --port 8781 --demo --fresh
#  -> http://127.0.0.1:8781/app/map/index.html
```

One process serves the map page, the vendored MapLibre/PMTiles stack, the basemap byte-ranges, the REST API and
the WebSocket. `--demo` loads the FloodValley scenario (§3). Add `?selfcheck=1` to the URL to make the page
publish its own verdict (`document.title` -> `MAP_OK` / `MAP_FAIL`, plus `window.__sightlineReport`).

Never port 8000 — that is the Unreal MCP server. The server refuses any non-loopback bind address.

Headless proof, which is also the regression test for the page:

```
node app/map/headless_check.mjs --url "http://127.0.0.1:8781/app/map/index.html?selfcheck=1" \
     --out _artifacts/verification/c2_map.png --shots _artifacts/verification/c2 --interact \
     --json _artifacts/verification/c2_headless.json
```

Offline queue demo (two processes, edge + cloud):

```
D:\Tools\uv\uv.exe run python -m sightline.api.serve --port 8781 --demo --fresh --outbox \
     --outbox-dir _artifacts/verification/outbox_demo --cloud-url http://127.0.0.1:8782   # link is down
D:\Tools\uv\uv.exe run python -m sightline.api.serve --port 8782 --fresh --db _artifacts/.../cloud.db
```

Python API tests: `D:\Tools\uv\uv.exe run pytest tests/test_api.py -q` (33 passed, 4 s).

---

## 2. The screenshots, and what is in them

All under `D:\Sightline\_artifacts\verification\`. **I read every one of these images.**

| File | What is actually visible |
|---|---|
| `c2_map.png` | The whole C2 surface at 1600x1000. Dark Protomaps basemap of the real AO with its real labels (Puthumala, Chooralmala, Ambedkar Colony, Mundakai, the Iruvazhinji river), scale bar 300 m, "Protomaps © OpenStreetMap". Over it: the POD raster as a dark-violet square (unswept ground) with an orange band hugging the flown lines; the flown track in green with one **orange MANUAL** segment inside lane 2; the unflown remainder of the boustrophedon as blue dashed lines; the drone as a yellow dot at the north end of line 5 with its camera footprint drawn around it; two **red hatched** "aerial search cannot clear" rectangles, one inside the swept band (the raster is transparent there, so the hatch shows through) and one east of it; seven numbered triage markers, biggest at rank 1, red for human / blue for animal, each inside a dashed `h_acc_m` ring; a small hollow grey ring for the dismissed record. Right panel: chips (`LIVE ws`, `8 records`, `6 confirmed`, `1 candidate`, `1 dismissed`, `outbox 0`, `POD effective mean 0.05 · ≥0.5 in 643 cells`, `k n/a · MODELLED`, `2 cannot-clear areas`, `mode AUTO · 55 m AGL`, `SIM simulated data`) and the ranked record list with a per-record bar chart of the score terms. |
| `c2/evidence_popup.png` | The evidence pop-up: the SIM evidence crop with its red detection box, then `SIM-001` + status pill, then the score table row by row — `P(living\|record) 0.910`, `w_class(t) t+2.0 h × 0.860`, `urgency_class × 2.400`, `1 + 0.1·count 1 (1–2) × 1.100`, **`priority score 2.066`** — then the formula and the thermal/motion boosts. The number never appears without its parts. |
| `c2/dismiss_refused.png` | Same pop-up scrolled to the action row: `Confirm` / `Mark stale` / reason box / `Dismiss`, and under it **in red: "R10: a dismissal needs a reason."** after clicking Dismiss with a blank reason. |
| `c2/after_dismiss.png` | The same record after dismissing it *with* a reason: the row is still in the list, still ranked 1, greyed with a `dismissed` pill; the chip count moves to `2 dismissed`; the map marker becomes a hollow grey ring. Nothing disappeared. |
| `c2/live_update.png` | `SIM-002` shown as `stale` after an HTTP change made from **outside the browser** (Node's fetch). The page never re-fetched `/api/records.geojson` — the WebSocket did it. |
| `c2_offline_queued.png` | The chip row reading `outbox 4 · link down` in red while the cloud sink is not running. |
| `c2_offline_drained.png` | The same chip row reading `outbox 0` after the sink came up and the queue drained. |
| `map_offline.png` | The pre-existing `verify_offline.html` basemap check, still passing (regression). |

---

## 3. What the demo scenario is, honestly

`sightline/api/demo_seed.py` has two halves and they are **not the same kind of thing**:

* **The 8 records are synthetic** — hand-authored cases with hand-chosen score components, so the triage UI can
  be exercised before the detection/geolocation lanes land. Every one carries `source.domain = "sim"`,
  `source.synthetic = true`, and a `notes` string beginning "SIMULATED demo record (not a measurement)". The
  page shows a `SIM simulated data` chip whenever it sees that domain.
* **The flight, the pattern and the coverage raster are computed by the real lanes.** The pattern is
  `sightline.plan.boustrophedon_route` over a 900 x 700 m box with the simulator's own camera (`SIM_RGB_4K`) at
  55 m AGL: 11 lines, 22 waypoints, 10 514 m, sweep width 85.2 m, spacing 61.5 m — all derived by B6, none
  hand-drawn. 46 % of it has been flown; that produces 160 telemetry samples at 30 m spacing (140 AUTO +
  20 MANUAL), which are fed to `sightline.coverage.CoverageMap` as two passes and exported by
  **B6's own `export_coverage`**. So the map is reading the production format, not a mock of it.

Everything the POD layer prints is `domain: "sim"` **and model-driven**: B6 reports `k_is_measured = false`
until F19 fits it, so the page prints a `k n/a · MODELLED` chip and, in the Layers pane, "modelled, NOT
measured" with B6's `k_basis` string underneath. Numbers from the current export, all `sim`:
`body` mean POD 0.063, `limb_only` 0.013, `effective` 0.054; 158 `cannot_clear` cells; 2 burial polygons.

---

## 4. What I consume from the coverage/plan lane, exactly as documented

Read from `docs/lanes/coverage_plan_eval.md` §5.1 and §5.2 and implemented in `sightline/api/coverage_feed.py`
and `sightline/api/mission_feed.py`. **No file in `sightline/coverage/` or `sightline/plan/` was edited.**

### 4.1 Coverage — `CONTRACT: sightline.coverage`

`read_manifest(dir)` loads `coverage.json`, refuses anything whose `product` is not `sightline.coverage` and
refuses a manifest with no layers. `read_coverage(dir)` normalises it into one sidecar
(`sightline.api.coverage_sidecar/1.1`) whose per-layer `image_url` / `geojson_url` are **API** URLs, so the page
never learns the directory layout:

```
GET /api/coverage/overlay.json           -> {contract, source, domain, grid, bounds, coordinates, ramp, bands,
                                             passes, cannot_clear_label, legend_note, default_layer,
                                             layers:[{layer, image_url, geojson_url, k, k_is_measured,
                                                      k_basis, stats}]}
GET /api/coverage/raster.png?layer=body  -> coverage_body.png, byte-identical to what B6 wrote
GET /api/coverage/bands.geojson?layer=…  -> that layer's POD band polygons
GET /api/coverage/cannot_clear.geojson   -> only the features with properties.cannot_clear == true, with
                                            label and domain preserved
GET /api/coverage/pod.png                -> back-compatible alias of raster.png (pre-B6 name)
```

I depend on three properties of B6's export and each is asserted in `tests/test_api.py`: image **row 0 is
north**; `coordinates` starts at the NW corner; and **alpha is 0 exactly on the `cannot_clear` cells** (asserted
pixel-for-pixel against `stats.cannot_clear_cells` — 158 == 158), because the map's hatch is drawn *through*
that hole. The Layers pane has a radio group for `body` / `limb_only` / `effective` and builds its POD legend
from the manifest's own magma `ramp`, not from a local copy.

**Nothing is missing from the contract.** Two small notes for B6, neither blocking:
1. `coverage.json` has no `generated_utc`, so the map cannot cache-bust the PNG by content. I fall back to
   `Cache-Control: no-cache` on the raster route. A `generated_utc` (or an ETag-able `run_id`) would be better.
2. `stats.mean_coverage` is `null` for the `effective` layer (correct — it is a mixture), and `k` is `null`
   there too. The page prints `k n/a` for it; that is deliberate, not a formatting bug.

### 4.2 Plan — `product: sightline.plan.route`

`MissionState.set_route(route.to_dict())` adopts the route JSON whole. Every waypoint field the contract lists
(`seq`, `action`, `alt_asl_m`, `agl_m`, `speed_ms`, `gimbal_pitch_deg`, `yaw_deg`, `dwell_s`,
`orbit_radius_m`, `segment_id`, `pass_id`, `reason`) is carried onto the GeoJSON waypoint Feature, and clicking
a waypoint on the map shows its `reason` verbatim — the planner explains itself to the operator instead of
being a blue line. `pattern`, `params` and `totals` ride on the plan FeatureCollection.
`GET /api/plan/route.json` returns the route byte-for-byte; `POST /api/plan/route` accepts one from the mission
lane out of process and refuses a payload whose `product` is wrong (422) rather than silently drawing nothing.

---

## 5. Layers, and the §5.9 rules they encode

| §5.9 requirement | How |
|---|---|
| **size = rank** | `circle-radius` interpolates on `priority_rank` (13 px at rank 1 -> 4.5 px at 25, 5 px for unranked). The rank number is printed in the marker. |
| **colour = class** | `match` on `cls`: human `#e63946`, animal `#3d87c9`; dismissed goes to a hollow grey ring on its own layer. |
| **ring = `h_acc_m`** | A real 64-gon computed in WGS-84 metres-per-degree (same formula as `common/geodesy.py`), so the ring is metre-true at every zoom, not a fixed pixel radius. |
| **evidence pop-up** | Thumbnail + the four score terms + the total + the printed formula + thermal/motion boosts. §5.8's rule that the number is never shown alone is enforced by a test that fails if `score != p_living*w_class*urgency*count_bonus`. |
| **POD raster** | B6's per-layer PNG as a MapLibre `image` source at the manifest's `coordinates`. |
| **"aerial search cannot clear" hatched** | A canvas-generated 45° hatch `fill-pattern` over a tinted fill and an outline; clicking one says POD is held at 0 inside it and that clearing it needs ground search, canine or radar. |
| **drone + camera footprint** | Live pose from `MissionState`; the footprint is the **real projected quadrilateral** from `sightline.coverage.footprint.ground_footprint` — the same function that decides which cells get credit — so the drawn shape and the swept cells agree by construction. |
| **planned pattern / flight track** | From the route JSON and the pose history; the track is split on `mode`, so the MANUAL takeover is a different colour (§7 step 3). |

---

## 6. The bug the picture found (and the check that now catches it)

The first render looked plausible and every return value said success: the manifest had layers, POD statistics,
158 burial cells; the page reported `MAP_OK`. **The picture showed the swept band nowhere near the flight
lines** — it ran off the bottom of the track and was far too wide.

Cause: `demo_seed` built the nadir gimbal quaternion by hand as "rotate −90° about Y". In this schema's
convention `Telemetry.gimbal_pitch_deg()` reads that back as **−180°**, so every footprint was oblique and
enormous. Measured overhang of the raster past the telemetry bounding box: **south 413.8 m, west 176.6 m, east
186.7 m, north −74.0 m** against an expected half-swath of 42.6 m. Fixed by using the coverage lane's own
`gimbal_quat(-90, gimbal_yaw_for_heading(heading))`, which is defined to round-trip. After the fix: south 54.0,
north 46.0, west 56.6, east 26.7 m — half a swath plus one 20 m cell, as it should be.

Three tests now fail if it comes back. Re-introducing the bug on purpose produces:

```
E  AssertionError: raster overhangs the track by 413.8 m to the south (tol 82.6 m)
E  AssertionError: across-track 358.1 m
E    assert 358.1205575938402 == 85.17110006259746 ± 1.70342
E  AssertionError: the survey camera is nadir; if this reads -180 the gimbal quaternion convention is wrong
E    assert -180.0 == -90.0 ± 0.5
FAILED tests/test_api.py::test_the_coverage_raster_is_geo_aligned_with_the_flown_track
FAILED tests/test_api.py::test_the_drawn_camera_footprint_matches_the_projected_one
```

Two UI defects were also found by looking, not by a return value: the evidence pop-up ran off the bottom of the
screen (the score total and the dismiss control were unreachable — the pop-up is now scroll-capped, the detail
list is behind a `<details>`, and `flyTo` lifts the marker so the pop-up fits), and the drone's halo was fatter
than the camera footprint so the footprint was invisible. `headless_check.mjs` now asserts
`dismiss_control_visible` from the button's own `getBoundingClientRect()`.

---

## 7. Checks that can fail, and their output

### 7.1 `tests/test_api.py` — 33 tests

```
D:\Tools\uv\uv.exe run pytest tests/test_api.py -q
.................................                                        [100%]
33 passed, 5 warnings in 3.36s
```

They cover: the B6 manifest read (and its refusal of a wrong `product`, an empty layer list, and a
`../../` path in an `image_url`); the transparent-alpha/`cannot_clear` correspondence; geo-alignment; the route
round-trip; RFC 7946 `[lon, lat]` order everywhere; rank ordering; `score == its parts`; `ce90 = 2.146 · h_acc`;
every thumbnail actually downloading as a JPEG; thumbnail path traversal; the WebSocket envelope, hello,
snapshot, per-update frame, monotonic `seq`, the 5 KB budget, `ping`/`pong` and `resync`; the R10 routes; and
the whole offline story. `tests/test_store.py` still passes (21).

### 7.2 R10, greped over this lane's own source

`test_no_delete_path_in_this_lanes_python_or_web_source` scans all of `sightline/store/*.py`,
`sightline/api/*.py`, `app/map/*.html` and `app/map/*.mjs` for `DELETE FROM`, `DROP TABLE|INDEX|TRIGGER|VIEW`,
`TRUNCATE`, `os.remove(`, `os.unlink(`, `shutil.rmtree(`, `.unlink(`, `@app.delete` and
`method: "DELETE"`. `test_this_lane_never_says_a_segment_is_cleared` bans the *words* `cleared`, `all clear`,
`search complete`, `area complete` unless a negation governs them within 60 characters. Both are live: the
wording check failed on two of my own docstrings when first written, and I reworded them rather than weakening
the check. `test_the_app_exposes_no_delete_route` walks the FastAPI route table.

### 7.3 Headless render — 11 checks

```
node app/map/headless_check.mjs --url "http://127.0.0.1:8781/app/map/index.html?selfcheck=1" --interact
PASS  ok:page_verdict  ok:no_external_requests  ok:no_console_errors  ok:layers_rendered
      ok:coverage_raster  ok:score_components  ok:evidence_popup  ok:dismiss_control_visible
      ok:r10_reason_required  ok:r10_record_kept  ok:live_websocket_push
```

with, from the page's own report:

```
"basemap_features": 117, "source_layers": ["buildings","earth","landuse","places","roads","water"],
"marker_features": 15, "ring_features": 14, "track_features": 3, "plan_features": 23,
"cannot_clear_features": 6, "drone_features": 2, "footprint_features": 2,
"coverage_raster_loaded": true, "coverage_source": "sightline.coverage", "coverage_layer": "effective",
"coverage_pod_mean": 0.0535, "coverage_k_is_measured": false, "cannot_clear_polygons": 2,
"plan_waypoints": 22, "track_modes": ["AUTO","MANUAL"], "records_loaded": 8,
"score_components_present": true, "thumbnails_present": 8, "ws_state": "live",
"resource_requests": 30, "external_requests": [], "errors": []
```

**Proof that the offline assertion can fail.** I added `<img src="https://tiles.example.com/x.png">` to the
page, re-ran, and got:

```
FAIL  FAIL:page_verdict  FAIL:no_external_requests  ok:no_console_errors  ok:layers_rendered ...
"external": [ "https://tiles.example.com/x.png" ]
```

then reverted. The browser is launched with `--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1`, so an
external host cannot even resolve; the check records every request from the page **and its workers** (MapLibre's
tile worker is auto-attached) and fails on any non-loopback URL. The page independently reports its own
`performance.getEntriesByType("resource")` list.

**Proof the live feed is really the WebSocket.** The check changes a record over HTTP *from Node*, then reads
the page's state:

```
before: {"id":"SIM-002","version":2,"status":"confirmed","seq":18,"ws":"live","fetches":5}
after : {"version":3,"status":"stale","seq":19,"fetches":5,"in_list":true}
```

one frame, no re-fetch (`fetches` unchanged at 5).

### 7.4 The offline story, run end to end against two live servers

Edge on 8781 with `--outbox --cloud-url http://127.0.0.1:8782` and nothing listening on 8782:

```
3 x POST /api/records/{id}/note  +  1 x POST /api/records/SIM-008/dismiss
depth 4  online False  failures 3  err "ConnectError: [WinError 10061] No connection could be made..."
  -> map chip: "outbox 4 · link down"      (_artifacts/verification/c2_offline_queued.png)
```

then the cloud sink is started on 8782 and the queue drains by itself:

```
FINAL depth 0  online True  sent 4  failures 6  acked 4
cloud records 4  by_status {'confirmed': 3, 'dismissed': 1}
  -> map chip: "outbox 0"                  (_artifacts/verification/c2_offline_drained.png)
```

Idempotency of the server-side upsert, against the running sink:

```
cloud before : {'records': 4, 'versions': 4, 'audit': 4}
  replay 1: (200, {'ok': True, 'applied': False, 'reason': 'duplicate', 'key': 'record:SIM_FLOODVALLEY_001:SIM-001:2', 'version': 2})
  replay 2: (200, {'ok': True, 'applied': False, 'reason': 'duplicate', 'key': 'record:SIM_FLOODVALLEY_001:SIM-001:2', 'version': 2})
cloud after  : {'records': 4, 'versions': 4, 'audit': 4}
  stale/altered: (409, '{"detail":"stale version 2 for SIM-001"}')
```

A replay changes nothing; a *different* record at an already-stored version is a 409, not a silent overwrite.
The in-process version of the same story is `test_the_queue_rises_offline_drains_on_reconnect_and_the_upsert_is_idempotent`
(16 jobs queued offline, +1 from an operator dismissal, 17 delivered on reconnect, thumbnails included).

---

## 8. The WebSocket message format

One JSON object per frame, `sightline/api/live.py`:

```json
{"type": "...", "seq": 42, "t_utc": 1789064123.5, "schema_version": "1.1.0", ...payload}
```

`seq` is monotonic per server process, so a client that sees a gap can `GET /api/records.geojson` or send
`{"type":"resync"}`. Types (`MESSAGE_TYPES`, and `envelope()` raises on anything else):

| type | payload | when |
|---|---|---|
| `hello` | `server`, `records`, `contracts` | once, on connect |
| `snapshot` | `records`: a full RFC 7946 FeatureCollection | on connect and on `resync` |
| `record` | `op` (always `"upsert"`), `event` (`create`/`update`/`status`/`dismiss`/`note`), `feature` | every store write |
| `mission` | `drone`, `footprint`, `track`, `plan`, `track_points`, `updated_utc` | on connect and on every pose/route push |
| `coverage` | — (the client re-fetches the sidecar) | when the coverage export changes |
| `outbox` | `depth`, `online`, `sent`, `failures`, … | uploader state changes |
| `pong` | `echo` | reply to a client `{"type":"ping","echo":…}` |

**There is no delete op** (R10). A dismissal arrives as a `record` upsert whose `status` is `"dismissed"`; the
map moves it to its own layer and keeps it. Thumbnails travel as URIs, never bytes: `MAX_MESSAGE_BYTES = 5120`
and a test asserts a real update frame stays under it.

---

## 9. What is stubbed or model-driven (labelled, per project rule 1)

1. **`POST /detect` uses `StubDetector`** — it loads no model and returns an empty box list; every response
   carries `"is_stub": true` and `/health` repeats it. This lane owns the *route*, the request/response shape
   and the merge-by-frame-index logic (which is tested and idempotent); the ML lane (C) owns the engine and
   swaps it in with `create_app(store, detector=…)`. No torch/ultralytics/tensorrt is imported anywhere here.
2. **The 8 demo records are synthetic**, per §3 above. They are not a measurement of anything.
3. **Every POD number on the map is model-driven**, per B6: `k` is derived, not calibrated, and `R_slice` is an
   analytic model. The page prints `k n/a · MODELLED` and "modelled, NOT measured" rather than hiding it. When
   F19 produces measured recalls, `k_is_measured` flips to true in the manifest and the page follows with no
   change here.
4. **`coverage_feed.grid_to_overlay()` is legacy** — it renders a bare `CoverageGrid` into a single-layer
   `overlay.json` + `pod.png`. It predates B6's exporter and is kept only so a caller holding a raw grid can
   still light the map up; `read_coverage()` prefers `coverage.json` whenever it exists. Anything owning a
   `CoverageMap` must use `sightline.coverage.export_coverage`.
5. **`sightline/api/wire.py::feature_to_record` is this lane's inverse of the frozen
   `Record.to_feature()`.** It lives here rather than in `schemas.py` so the frozen contract stays untouched.
   It is lossy exactly as the wire format is: `to_feature()` rounds coordinates to 6 dp (~0.11 m here), so the
   cloud copy is a *transport* copy and the edge log is the authority. If the orchestrator wants it promoted
   into `schemas.py`, it is 40 lines and fully tested.

---

## 10. What I need from other lanes

1. **Coverage lane (B6):** a `generated_utc` (or a stable content hash) in `coverage.json`, so the map can
   cache-bust a re-exported PNG properly. Not blocking.
2. **Mission lane (F2/F3):** call `POST /api/mission/pose` with `Telemetry` fields (and optionally a
   `footprint` ring) and `POST /api/plan/route` with `Route.to_dict()`, or hold the `MissionState` object
   directly. Both push a `mission` frame to every connected client. Nothing else is needed for the live layers.
3. **Pipeline (B1–B4):** point it at a `RecordStore` and every write appears on the map with no further
   wiring — `store.subscribe()` is what drives the feed. `python -m sightline.pipeline … --out …` writes
   `records.geojson`; when a dataset run exists, load it with `RecordStore.put()` per record (or hand me a
   loader task) and drop `--demo`.
4. **Orchestrator:** `docs/TRACKER.md` was deliberately not edited (other lanes are writing concurrently).
   Lines for it: *F15 map — done, headless-verified, 11 checks; F18 store + outbox — done, 54 tests; both
   consume the B6 coverage manifest and route JSON directly; the cloud detector is a labelled stub.*

---

## 11. File map

```
sightline/store/db.py            record log, WAL, R10 enforced by BEFORE DELETE triggers
sightline/store/outbox.py        SQLiteAckQueue outbox + retrying uploader (at-least-once, replayable)
sightline/api/app.py             every route; the cloud sink; static mounts with Range support
sightline/api/live.py            LiveHub + the envelope (§8 above)
sightline/api/coverage_feed.py   reads B6's coverage.json -> the map's sidecar; extracts cannot_clear
sightline/api/mission_feed.py    drone/footprint/track/plan; adopts B6's route JSON whole
sightline/api/detect_fallback.py POST /detect route + merge-by-frame-index (STUB detector)
sightline/api/wire.py            GeoJSON Feature -> Record (the cloud sink's decoder)
sightline/api/demo_seed.py       the FloodValley scenario (synthetic records + real plan/coverage)
sightline/api/serve.py           the launcher
app/map/index.html               the C2 page
app/map/headless_check.mjs       the render check (11 assertions, screenshots, offline enforcement)
app/map/serve.mjs                dependency-free static server (same URLs, no API)
app/map/verify_offline.html      the basemap-only smoke page from session 2 (still passing)
tests/test_store.py              21 tests   tests/test_api.py  33 tests
```
