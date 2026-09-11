# Quickstart — running Sightline

Everything runs from the repository root in **PowerShell**:

```powershell
cd D:\Sightline
```

All commands below are `D:\Tools\uv\uv.exe run python ...`. `uv` restores the exact pinned environment, so
there is no virtualenv to activate.

> **PowerShell note:** `&&` is not a statement separator in Windows PowerShell 5.1. Use `;` to chain, or run
> the commands on separate lines.

---

## 1. The dashboard (backend + frontend in one process)

**There is no separate dev server.** `sightline.api.serve` is the backend API *and* the web server that
serves the frontend. One command, one port:

```powershell
D:\Tools\uv\uv.exe run python -m sightline.api.serve --port 8781 --demo --fresh
```

| flag | what it does |
|---|---|
| `--demo` | seeds the FloodValley scenario so the map has something in it immediately |
| `--fresh` | starts from an empty database (a new file; nothing is ever deleted) |
| *(omit both)* | an empty log, ready for a live flight to fill |

**Expected output** — it stays in the foreground and logs requests:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:8781 (Press CTRL+C to quit)
```

**Then open the frontend:**

```
http://127.0.0.1:8781/app/map/index.html
```

You should see `SIGHTLINE — TRIAGE`, a dark map of the Wayanad valley, a `LIVE ws` chip, record cards with
score bars down the right-hand side, and an orange probability-of-detection raster over the search area.

If `--fresh` is omitted and the database already holds the demo records, seeding stops with
`StaleVersionError: SIM-001: incoming version 1 <= stored 1` — that is the store refusing to silently
overwrite an existing record, not a crash. Add `--fresh`, or point `--db` somewhere new.

---

## 2. The simulator

This is the one piece that is not scripted, because the editor takes minutes to open.

1. Open `sim/SightlineSim/SightlineSim.uproject` in Unreal Engine 5.8
2. Press **Play** (not *Simulate*)

**Simulate mode spawns no vehicle.** AirSim then reports *"There were no compatible vehicles created for
current SimMode"*, and every flight call blocks forever at 0 % CPU with nothing moving — indistinguishable
from a hung editor. It must be **Play**.

Verify from PowerShell:

```powershell
D:\Tools\uv\uv.exe run python tools\capture\_wait_vehicle.py
```

**Expected output:**

```
READY after 1s: vehicles=['Drone'], clock ticking (1001616000 ns/s)
  landed_state=0  z=0.30
```

---

## 3. The live demo

One command. It checks the backend, verifies the simulator actually has a vehicle, picks the detector, flies
the pattern with the trained model, and streams records to the dashboard:

```powershell
D:\Tools\uv\uv.exe run python tools\live\demo.py
```

**Expected output:**

```
==============================================================================
SIGHTLINE - LIVE DEMO
==============================================================================
  scenario    nominal  -  the ACCEPTANCE slice: 45 m, inside the 40-60 m nominal band ...
  note        editor is running: using PyTorch, not the engine (TensorRT's 2.6 GB ...)
  detector    best.pt  -  PyTorch fp16 (~159 ms/frame measured)
  dashboard   already serving on :8781
  simulator   OK - vehicles=['Drone'], clock ticking

  DASHBOARD   http://127.0.0.1:8781/app/map/index.html

==============================================================================
FLYING - open the dashboard above to watch records arrive
==============================================================================

taking off...
  leg 1/31 east   116.4  north 39 -> 138
  f00009 AUTO   agl  45.7 m  det  1  trk  0  rec  0  1504.9 ms (detect)
  f00019 AUTO   agl  45.8 m  det  4  trk  1  rec  1   751.0 ms (detect)
```

Read the per-frame line as: **det** = detections this frame, **trk** = confirmed tracks so far,
**rec** = deduplicated records created. `rec` climbing is the whole system working.

Useful flags:

| flag | effect |
|---|---|
| `--check` | verify everything and start nothing |
| `--port 8790` | use a different dashboard port |
| `--scenario high` | fly a different slice (see below) |
| `--ignore-safety` | fly a plan the battery model rejects (the violation is stamped in the data card) |
| `--pytorch` / `--engine` | force the detector instead of choosing automatically |

---

## 4. Watching it work

**The map** — `http://127.0.0.1:<port>/app/map/index.html`

* header chips: `LIVE ws`, record counts, `mode AUTO`, live AGL, and an amber `SIM simulated data` badge
* green bands are coverage actually flown; blue dashed is the planned route ahead
* record cards show the triage score **decomposed** — `P(living) · w_class · urgency · count×`
* clicking a marker opens the evidence popup with the full scoring formula and a dismiss box that
  **refuses an empty reason**

**The camera panel** — bottom-left of the map, appears automatically while a flight is running. This is what
the drone is looking at *right now* with the model's boxes drawn on it, colour-coded by score
(red weak → green strong), and a status strip reading `frame N | k detection(s) | AGL | mode | ms detect`.

Its data is also available directly:

```powershell
curl http://127.0.0.1:8781/api/live/view.json     # detections as data
curl -o frame.jpg http://127.0.0.1:8781/api/live/view.jpg   # the annotated frame
```

---

## 5. Scenario variations

```powershell
D:\Tools\uv\uv.exe run python tools\live\demo.py --scenario nominal   # 45 m, 7 m/s - ACCEPTANCE
D:\Tools\uv\uv.exe run python tools\live\demo.py --scenario low       # 30 m - ~1.5x pixels on target
D:\Tools\uv\uv.exe run python tools\live\demo.py --scenario high      # 80 m - pixels halve, HARD slice
D:\Tools\uv\uv.exe run python tools\live\demo.py --scenario slow      # 4 m/s - tracker's best case
D:\Tools\uv\uv.exe run python tools\live\demo.py --scenario fast      # 12 m/s - below what the gate needs
```

These vary **altitude and speed**, which are the levers this project demonstrably controls. Weather and
time-of-day presets were built, measured, and **removed**: `simSetTimeOfDay` and `simSetWeatherParameter` do
nothing in this level, and five presets spanning 06:40 to 18:15 with rain up to 0.6 rendered pixel-identical
frames (luma spread 4.7 %). `tools\live\_check_scenarios.py` is the measurement, and it exits non-zero if
the scenarios ever stop differing.

---

## 6. Checking the system is ready

```powershell
D:\Tools\uv\uv.exe run python tools\live\demo_ready.py
```

**Expected output** — one verdict, and a check that cannot run counts as a failure, never a pass:

```
  PASS  trained weights         models\detect\f8b_sim\weights\best.pt (19.5 MB)
  PASS  training metrics        final epoch mAP50 0.805, precision 0.903, recall 0.724
  PASS  tensorrt engine         23 MB
  PASS  engine benchmark        pytorch 158.9 ms -> tensorrt 125.3 ms (1.27x)
  PASS  C2 server               port 8781, 8 records, schema 1.3.0
  PASS  offline basemap         1.7 MB
  PASS  held-out split          4 runs, seeds [23, 47], 449 boxes
  PASS  test suite              929 passed, 2 skipped
==============================================================================
READY
```

---

## 7. No simulator? Replay a recorded flight

Same code path, same pipeline, no renderer competing for the GPU — which also lets the TensorRT engine run:

```powershell
D:\Tools\uv\uv.exe run python -m sightline.mission.live --replay _artifacts\dataset\seed47_alt55 --pace 4 --c2 http://127.0.0.1:8781
```

Or score a recorded run offline and write ranked records:

```powershell
D:\Tools\uv\uv.exe run python -m sightline.pipeline _artifacts\dataset\seed47_alt55 --detector truth --out _artifacts\run1
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Dashboard loads but the map is blank and the status stays `starting…` | MapLibre needs WebGL. Use a normal browser window; some embedded/sandboxed browsers never reach the map's `idle` event, which gates the app's boot. |
| `REFUSING TO FLY: the plan breaks an F2 constraint` | The battery model rejects the plan at that speed. Fly faster, shorten it, or pass `--ignore-safety` (the violation is stamped into the data card either way). |
| Flight sits at 0 frames, nothing moves in the viewport | PIE is in **Simulate**, not **Play**. No vehicle exists. |
| `CUDA_ERROR_ILLEGAL_ADDRESS` during a flight | The TensorRT engine (2.6 GB of context) and the editor do not both fit in 8 GB. `demo.py` avoids this automatically; only `--engine` forces it. |
| Detections appear but `rec` stays 0 | A survivor is not getting enough shutter releases inside the confirmation window. Try `--scenario slow`. |
