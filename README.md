# Sightline

**Aerial survivor triage for flood disasters — a simulation-first computer-vision system.**

A drone flies an autonomous search pattern over a flooded valley, detects people from 4K nadir imagery,
geolocates each one, deduplicates them into a single record per person, ranks them by urgency, and pushes
them live to an offline-capable command map — with guardrails that make it structurally impossible for the
software to declare an area "cleared".

Everything runs against a purpose-built Unreal Engine 5.8 simulation: the renderer is the deployment domain,
not a training aid.

---

## Why this exists

Search-and-rescue after a flood is a coverage problem under time pressure. A helicopter crew can see a lot
but cannot be everywhere; a drone can be systematic but only if something is looking at every frame. The hard
parts are not "run a detector" — they are:

* a person 40–60 m below the camera is **25–80 pixels**, so resolution and tiling matter more than
  architecture;
* survivors are **static** while the camera moves, which is the opposite of what pedestrian trackers assume;
* one survivor seen on four passes must be **one record**, not four markers;
* and a system that tells a commander an area is clear when it is not is worse than no system at all.

---

## What is built

```
        ┌── Unreal Engine 5.8 + Cosys-AirSim ──────────────────────────────┐
        │  flood valley · 1,183 actors · 71 survivors · scripted weather   │
        └───────────────┬──────────────────────────────────────────────────┘
                        │  4K RGB + instance mask + depth + telemetry
                        ▼
   ingest ─► detect ─► geolocate ─► track ─► dedup ─► triage ─► store ─► export
             (YOLO26s    (pixel→       (BoT-SORT   (DBSCAN   (urgency   (SQLite  (GeoJSON
              tiled at    WGS84,        + camera-   at 2×     score,     + offline KML/KMZ
              1024)       ±2.7 m)       motion      CE90)     visible    outbox)   /CoT)
                                        comp.)                components)
                        │
                        ▼
        ┌── C2 map (MapLibre + PMTiles, fully offline) ────────────────────┐
        │  live WebSocket · ranked record cards · probability-of-detection │
        │  raster · burial polygons · dismiss-with-reason (never delete)   │
        └──────────────────────────────────────────────────────────────────┘
```

**14 independent lane packages** under `sightline/`, ~56k lines of Python, **931 tests**.

---

## Results

All figures are **in simulation**, on a **held-out scenario seed** — a different actor layout, different
occluders, never seen in training. Every number in this project carries the domain it was measured in.

| Metric | Value |
|---|---|
| Detector | YOLO26s fine-tuned from COCO on simulator frames only |
| **mAP@0.5** | **0.807** |
| Precision / Recall | 0.888 / 0.721 *(at max-F1 confidence, not the operating point)* |
| Training | 100 epochs, 9.1 minutes on an H100 80GB, **$0.75** |
| Geolocation accuracy | **±2.7 m** (1σ), CE90 5.7 m |
| Dataset | 1,330 tiles / 841 boxes from 2 scenario seeds |

The training set is deliberately small. The design goal was *coverage of scenarios, failure conditions and
edge cases* — 10 pose/submersion combinations including half-submerged, head-only, trapped under rubble and
occluded-by-canopy — rather than volume.

---

## The parts worth reading

### Auto-labelling that refuses to lie

Labels come from the simulator's instance-segmentation mask, so boxes are exact by construction. That turned
out to be dangerous in a way worth documenting: Cosys-AirSim renders its annotation pass with the
`InstancedFoliage` and `InstancedGrass` show flags **off**, and every plant in this scene is a HISM instance.
The mask therefore sees *straight through the canopy* — a survivor under a fern is reported whole and
unoccluded, and the auto-labeller writes a confident box over pure leaf texture.

`tools/capture/labels.py:apply_depth_visibility` fixes it at source using the **depth buffer**, which does
render the canopy. Because depth is planar, a nadir frame gives `depth = camera_altitude − surface_altitude`
at every pixel, so per-pixel visibility is exact:

> a mask pixel claiming actor A is genuinely visible ⟺ `depth(pixel)` is not closer than A's own body

It also yields a *measured* visibility fraction instead of an estimated one, because the foliage-blind mask
happens to be the amodal silhouette.

### Guardrails that are enforced, not documented

`R10: no code path may delete a record or mark a segment "cleared".`

A static scanner (`sightline/triage/guardrails.py`) walks all 14 lane directories for deletion verbs and
"cleared" vocabulary, with a self-test that plants 16 real violations and requires all 16 to be caught. Three
independent layers enforce dismiss-with-reason: the guardrail function, a SQLite trigger, and an API 422.
Uncovered ground is reported as **"UNSEARCHED, not clear"** — including by the flight planner when battery
reserve runs out.

### Evaluation that cannot flatter itself

* Train/test split is **by scenario seed, never by frame** — `assert_no_seed_leak` refuses to build a dataset
  from one seed, which it did during development.
* `DomainMixError` and `SliceRoleError` make it structurally impossible to average a simulation number with a
  real one, or to pool a hard slice into the headline figure.
* `MetricRow` validates in `__post_init__` and is frozen: the "we found nothing" case emits an explicit
  `n=0` row with a reason rather than a `NaN` that would serialise as invalid JSON.

---

## Quick start

```bash
uv sync                                    # restore the pinned environment (Python 3.11)
uv run python tools/doctor.py              # verify the toolchain
uv run pytest tests/ -q                    # 931 tests
```

Run the command map against a demo scenario, no simulator required:

```bash
uv run python -m sightline.api.serve --port 8781 --demo
# then open http://127.0.0.1:8781/app/map/index.html
```

Replay a recorded flight through the full pipeline:

```bash
uv run python -m sightline.pipeline _artifacts/dataset/<run> --detector truth --out _artifacts/run1
```

Fly it live in the simulator with the trained detector:

```bash
uv run python -m sightline.mission.live --alt 45 --speed 7 \
    --detector rgb --weights models/detect/f8b_sim/weights/best.pt --c2 http://127.0.0.1:8781
```

---

## Layout

| Path | What |
|---|---|
| `sightline/` | the 14 lane packages — ingest, detect, geo, track, dedup, triage, export, coverage, plan, store, api, eval, mission |
| `tools/scene/` | procedural scene generation and the environment quality gate |
| `tools/capture/` | dataset capture, auto-labelling, and the dataset quality gate |
| `tools/train/` | RunPod GPU rental, training, TensorRT export |
| `app/map/` | the offline C2 map (MapLibre + PMTiles, all vendored) |
| `sim/SightlineSim/` | the Unreal Engine 5.8 project |
| `docs/` | solution document, contracts, quality gate, tracker |

---

## Known limitations

Stated plainly, because a system that hides these is not trustworthy:

| Limitation | Status |
|---|---|
| **Inference latency** — 1.2–3.2 s/frame for 15 tiles of YOLO26s on an RTX 4060 sharing VRAM with the editor, against the ~113 ms the design budgets. The loop runs at ~0.6 FPS, which stretches every downstream timing assumption. | TensorRT export is written and is the intended fix |
| **Track confirmation** — §5.6's "3 hits in 2 seconds" assumes ~5 FPS. At 0.6 FPS the window is derived from the *measured* rate instead, but association across ~450 px of inter-frame camera motion is fragile, so confirmations are well below detections. | Partially mitigated |
| **Thermal** — the radiometric annotation layer is implemented but has never rendered a verified frame. | Open |
| **Gamepad takeover** — the 4-mode state machine is complete and tested; the attached pad delivers trigger axes to SDL but not stick axes or buttons. Hardware/driver, not code. | Blocked externally |
| **Hard-negative classes** — no mannequins, clothing-only decoys or limb-like debris exist in the scene, so the detector has never been shown the distractors it is expected to reject. | Open |

Training data is 100 % synthetic by design. This model has learned *this renderer*; a real-footage claim
would require the transferable-model recipe, which is specified but not built.

---

## Licence and attribution

Unreal assets are CC0 from [Poly Haven](https://polyhaven.com). Detection uses
[Ultralytics](https://github.com/ultralytics/ultralytics) (**AGPL-3.0**) — fine for an open repository; a
closed commercial deployment would need their Enterprise licence. Simulation uses
[Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) (MIT). Basemap tiles are
[Protomaps](https://protomaps.com) / OpenStreetMap (ODbL).
