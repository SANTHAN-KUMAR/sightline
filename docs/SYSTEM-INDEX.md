# SightLine — System Index

**The whole depth of the project, in one navigable file.** Every module, contract, artefact
and number, with where each one is defined and where each one is verified.

[`README.md`](../README.md) is the argument. This is the map.

---

## 0. How to read this repository

Documents, in the order they are useful:

| document | what it is for |
|---|---|
| [`README.md`](../README.md) | the argument: what the system is, what it measured, why it is built this way |
| **this file** | the index: where everything lives and what it is responsible for |
| [`QUALITY_GATE.md`](QUALITY_GATE.md) | **mandatory.** Eyeball verification is a hard rule — read before calling anything done |
| [`CONTRACTS.md`](CONTRACTS.md) | the frozen schema and module ownership. **Read before writing any code under `sightline/`** |
| [`HANDBOOK.md`](HANDBOOK.md) | current state, hard rules, machine limits, simulator behaviour that costs hours to rediscover |
| [`TRACKER.md`](TRACKER.md) | what is done, in progress, next. Start from its "Next actions" |
| [`CONTEXT.md`](CONTEXT.md) | environment facts, decisions, deviations, known pitfalls |
| [`SOLUTION_DOC.md`](SOLUTION_DOC.md) | the full research and build document — source of truth for *what* to build. Long; read by section |
| [`SCENE_REFERENCE.md`](SCENE_REFERENCE.md) | the visual target the scene is built to |
| [`ANNOTATION_GUIDELINE.md`](ANNOTATION_GUIDELINE.md) | how partially visible subjects are boxed |
| [`FINETUNE_PLAN.md`](FINETUNE_PLAN.md) | the training recipe and its rationale |

**The single most important habit in this project:** *look at the thing before calling it
done.* Seven pieces of work passed every programmatic check and were still wrong — T-posed
survivors, one-armed poses, untextured characters, a dataset full of the drone's own
propellers, a material that silently failed to compile, a terrain layer completely replaced by
another, and "unfindable" survivors that were plainly visible. **Every one was caught by
looking at a picture; none by reading a return value.**

---

## 1. The system in one diagram

```
┌──────────────────────── SIMULATION ────────────────────────────────────────────┐
│  Unreal Engine 5.8 + Cosys-AirSim   ·   flood valley, 1,183 actors, 71 posed    │
│  survivors, 5,508 vegetation instances, scriptable weather and flood level      │
│  Terrain: real Copernicus GLO-30 DEM (N11_00_E076_00)                           │
│                                                                                 │
│  Cameras: 4K RGB · LWIR thermal · instance segmentation (labels only) · depth   │
└──────────────┬──────────────────────────────────────────────────────────────────┘
               │   ══ THE SEAM ══  only four artefacts cross, and truth is not one
               │   frames/*.jpg · DJI .SRT telemetry · PX4 ULog · radar .npz
               ▼
┌──────────────────────── PIPELINE (one FramePipeline object) ───────────────────┐
│                                                                                 │
│  INGEST ─▶ DETECT ─▶ FUSE ─▶ VERIFY ─▶ TRACK ─▶ GEOLOCATE ─▶ ACCUMULATE        │
│                                                        │                        │
│                                          DEDUPLICATE ◀─┘                        │
│                                                │                                │
│                                          SCORE & RANK                           │
└──────────────┬──────────────────────────────────┬───────────────────────────────┘
               │                                  │
     triage list (records)              coverage raster (POD)
               │                                  │
               ▼                                  ▼
┌──────────────────────── DELIVERY ──────────────────────────────────────────────┐
│  Append-only SQLite log  ·  outbox (at-least-once, idempotent)                  │
│  FastAPI + WebSocket  ─▶  offline MapLibre C2 map                               │
│  Exports: GeoJSON (RFC 7946) · KML/KMZ · Cursor-on-Target (ATAK/WinTAK)         │
└─────────────────────────────────────────────────────────────────────────────────┘
                              │
                    EVAL  ◀───┴───▶  the only package allowed both sides of the seam
```

---

## 2. The seam — the structural rule everything else rests on

```
   GENERATOR SIDE                 │  SEAM  │            PIPELINE SIDE
   ───────────────────────────────┼────────┼────────────────────────────────
   world, terrain, actors, truth  │        │  frames, telemetry, detections
   knows where everybody is       │   ✗    │  knows only what a drone writes
```

| property | how it is enforced |
|---|---|
| a pipeline module may not import a generator module | **AST test on every commit** — fails the build |
| truth files are never opened downstream | the generator writes `truth.json`; no pipeline module names it in executable code |
| `eval/` may touch both sides | it is the one job that legitimately needs truth: scoring |
| determinism | same seed → **same bytes**, gated by sha256 |

**Why it matters:** without it, every recall number in this repository would be measuring the
renderer's bookkeeping rather than a detection system.

### What crosses, exactly

| artefact | format | carries |
|---|---|---|
| video | MP4 / JPEG frames | RGB pixels (+ LWIR when carried) |
| telemetry | **DJI-format `.SRT`** | lat, lon, altitude, gimbal roll/pitch/yaw, timestamp — per frame |
| flight log | **PX4 ULog** | the autopilot's own record |
| radar cube *(on dispatch only)* | `.npz`, **complex64** | raw FMCW beat signal, I/Q phase preserved |

**Nothing else.** No actor list, no truth file, no "which pixel is a person". The `.SRT` is the
one that carries geometry, so it has to be exactly right — the pipeline reconstructs **every**
lat/lon from it.

---

## 3. Module index

One owner per directory, **no cross-writes.** If you need a change in another lane, route it
rather than reaching in. Full ownership table: [`CONTRACTS.md`](CONTRACTS.md) §2.

### 3.1 `sightline/schemas.py` — the frozen contract

Every module exchanges these types and nothing else. **You may not rename a field, change its
units, or change its meaning.** If something is genuinely missing, add an *optional* field with
a default and append a line to the amendment table.

| type | what it carries |
|---|---|
| `Telemetry` | pose, attitude, gimbal, time — per frame |
| `Intrinsics` | the camera model |
| `FrameBundle` | what ingest hands to detection: one processed frame + optional thermal + pose |
| `Detection` | one box on one frame, with band and tile index |
| `GeoFix` | a projected lat/lon with its own `h_acc_m` |
| `Track` | a confirmed persistence across frames |
| `Record` | **one living being** — the thing that reaches the operator |
| `MetricRow` | a number *plus its slice* — frozen, validates on construction |
| `SliceKey` | the domain and axes a number was measured on |

Units are SI with suffixes `_m` / `_s` / `_deg` / `_px` / `_utc`. Angles in degrees.
Quaternions are `(w, x, y, z)`. Geographic order is `lat, lon` **everywhere except** inside
GeoJSON, which is `[lon, lat, alt]` by RFC 7946.

### 3.2 The lanes

| package | consumes | produces | the decision that defines it |
|---|---|---|---|
| **`ingest/`** | files / sim export | `FrameBundle` stream | decode, telemetry parse, SLERP pose interpolation, decimation to the processing cadence. Sources: simulator export, DJI `.SRT`, MAVLink, PX4 ULog |
| **`detect/`** | `FrameBundle` | `list[Detection]` | **tiled** inference at native resolution — resolution management beats architecture for small targets. Cross-tile NMS at IoU 0.55 |
| **`detect/fusion.py`** | RGB + LWIR boxes | fused boxes | **late, box-level** WBF + ProbEn. With thermal absent this is the **identity function**, not a degraded mode |
| **`detect/verifier.py`** | candidate crops | `is_real` + posture + occlusion + submersion | frozen DINOv2 backbone; only four heads are fitted. Runs **before** tracking |
| **`geo/`** | `Detection` + `Telemetry` + `Intrinsics` | `GeoFix` | exact, reversible projection chain; DEM **march**, not plane fit; along-track and cross-track error computed **separately** |
| **`track/`** | detections per frame | `Track` | association gate sized **in metres**, not IoU — see §7.2. Confirms on 3 hits in a 2 s window |
| **`dedup/`** | `list[Track]` | `list[Record]` | DBSCAN on haversine at `eps = 2 × CE90`, then **split on simultaneity** — see §7.1 |
| **`triage/`** | `list[Record]` | ranked `list[Record]` | four visible score components; posture **promotes, never demotes** |
| **`triage/guardrails.py`** | the source tree | pass / fail | the **R10 static scanner** — see §8 |
| **`coverage/`** | `Telemetry` + `Intrinsics` | `CoverageGrid` | effort credited **per presentation class** at rate `r_slice`; `POD = 1 − e^(−kC)` capped at 0.99 |
| **`plan/`** | `CoverageGrid` | waypoint lists | maximises probability of success; **refuses** bands that add no POD rather than scoring them |
| **`store/`** | `Record` | SQLite log + outbox | **no DELETE and no UPDATE.** Dismissal adds a row |
| **`api/`** | `Record` | FastAPI + WebSocket | the live feed the map consumes |
| **`eval/`** | records + truth | `MetricRow` list | the only package allowed both sides of the seam |
| **`mission/`** | sim | flight + telemetry | autonomous survey, four-mode takeover state machine |
| **`common/`** | — | pure functions | geodesy, time, io — orchestrator-owned, no lane writes here |

### 3.3 Tooling

| path | what it does |
|---|---|
| `tools/scene/` | procedural scene generation — terrain, actors, rubble, vegetation, materials — and the **environment quality gate** (`qa_shots.py` stamps a manifest; `assert_qa_fresh.py` exits non-zero if the scene changed after the last render) |
| `tools/capture/` | dataset capture, **depth-buffer auto-labelling**, the dataset quality gate, contact sheets |
| `tools/train/` | GPU rental, training, export |
| `tools/sightline_mcp/` | the MCP server that drives the editor and the simulator from outside it |
| `app/map/` | the offline C2 map — MapLibre + PMTiles, **every dependency vendored**, zero external requests at runtime |
| `sim/SightlineSim/` | the Unreal Engine 5.8 project |

---

## 4. The simulation

### 4.1 The scene

| | |
|:--|:--|
| engine | Unreal Engine 5.8 · Cosys-AirSim v3.4.1 (MIT) |
| terrain | real **Copernicus GLO-30** DSM, tile `N11_00_E076_00` (ESA/Airbus, open licence) |
| actors | **1,183** — ground, flood water, 73 Kerala houses, 919 debris items placed by flood-transport physics, 18 damaged houses, 19 boats |
| rubble | 1,799 instances on 41 actors + 67 roof-debris pieces on 15 actors |
| vegetation | **5,508 instances on 9 actors** — 4,342 trees (5 Poly Haven species) + 1,166 understorey ferns |
| survivors | **71**, posed; 10 pose/submersion combinations including half-submerged, head-only, trapped under rubble, occluded by canopy |
| ground truth | survivor positions match the sim to **0.00 m** |
| flight | SimpleFlight, gamepad takeover with a logged four-mode state machine (AUTO / MANUAL / HOLD / RTL) |

*Scene preview figure intentionally omitted in this PR scope.*

**Memory is the binding constraint, not VRAM.** The editor dies of the Windows commit limit at
a few thousand actors, so every generator carries a `MAX_ACTORS = 400` tripwire and a memory
guard. Instancing is why 5,508 plants cost 9 actors rather than 5,508.

### 4.2 The capture stack

| pass | resolution | note |
|---|---|---|
| RGB | 4K (1920×1080 in gates, same focal length so GSD is identical) | HFOV 71.2° |
| LWIR thermal | **640 × 512** | one third the linear resolution of RGB |
| instance segmentation | — | **labels only** — never crosses the seam as pixels |
| depth | — | planar; the auto-labeller's visibility oracle |

**Processing cadence is a constraint, not a shortcut.** `processed_fps = 5.0` with
`frame_stride = 2` gives one processed frame every 0.4 s — roughly what a Jetson-class board
manages on a 4K frame with tiled inference. At a slower cadence **the tracker sees each person
twice and confirms almost nobody**, so `execute_run` now *refuses* such a run rather than
producing an empty triage list with a healthy-looking map:

```
RuntimeError: processing cadence starves the tracker: 3.2 looks per target
(6 m along-track at 5.0 m/s, one frame every 0.40 s) but confirmation needs 3.
```

Looks per target, measured: **13.42 at 60 m · 5.75 at 20 m.** Descending shrinks the footprint,
so each person is seen *fewer* times — one of the real costs of flying low, and it is in the
manifest.

### 4.3 Auto-labelling, and the trap it walked into

Labels come from the instance mask, so boxes are exact by construction — which turned out to be
dangerous:

> Cosys-AirSim renders its annotation pass with `InstancedFoliage` and `InstancedGrass`
> **off**, and every plant in this scene is a HISM instance. The mask sees **straight through
> the canopy**: a survivor under a fern is reported whole and unoccluded, and the auto-labeller
> writes a confident box over pure leaf texture.

**Fixed at source with the depth buffer**, which does render the canopy. Depth is planar, so a
nadir frame gives `depth = camera_altitude − surface_altitude` at every pixel:

> a mask pixel claiming actor A is genuinely visible ⟺ `depth(pixel)` is not closer than A's own body

It also yields a **measured** visibility fraction rather than an estimated one, because the
foliage-blind mask happens to be the amodal silhouette. Implementation:
`tools/capture/labels.py:apply_depth_visibility`.

### 4.4 The flight dynamics are not a waypoint lerp

> A multirotor cannot accelerate without tilting. **tan(tilt) = |a_horizontal| / g**

So the airframe pitches into every acceleration, the stabilised gimbal counter-rotates to hold
the optical axis, and the gimbal angles written to the `.SRT` **move even on a perfectly steady
nadir pass** — exactly as they do in real DJI logs. Downstream, the footprint wobbles at the
ends of every swath where the aircraft is turning, so **coverage there is genuinely worse than
in the middle.** That is an emergent property of the physics; a waypoint lerp would have quietly
produced a uniform footprint and a coverage map that overstates the turns.

Where PX4 is available, **SITL/SIH runs the real flight code** — the same firmware that flies
the aircraft, with simulated physics, so the control loops, attitude estimator and logging are
all the real ones. Where it is not, an acceleration-limited model with the tilt coupling above
keeps the property that matters.

---

## 5. Geolocation

### 5.1 The chain

```
detection centre (u,v)
   │  intrinsics + Brown-Conrady distortion         ① unproject
ray in OPTICAL frame       (x right, y down, z forward)
   │  R_OPTICAL_TO_BODY                             ② optical → body
ray in BODY frame          (x forward, y right, z down)
   │  gimbal quaternion                             ③ gimbal
   │  airframe attitude quaternion                  ④ attitude
ray in NED frame           (north, east, down)
   │  march against the DEM until it meets ground   ⑤ intersect
lat / lon / height + θ (off-nadir angle)
   │  the error budget, evaluated at this geometry  ⑥ budget
lat, lon, h_acc_m, CE90
```

**The gate that justifies this being its own module:** `project` and `unproject` are exact
inverses to **well under a tenth of a pixel** anywhere in the frame, corners under full
distortion included. The compositor runs the chain *forwards* to place an actor at a known
lat/lon; the pipeline runs **the same code backwards** to recover a lat/lon from a detection.
If those disagree, every geolocation number downstream is fiction.

**Terrain intersection is a march, not a plane fit.** Flat terrain is solved analytically then
refined geodetically — a local tangent plane and the ellipsoid disagree by ~0.8 mm over a 100 m
footprint, below our tolerance but not below our arithmetic. A real DEM needs marching,
**because a ray can graze a ridge and re-enter.**

### 5.2 The error budget

Two things here are deliberately more careful than the architecture document specified:

**① Along-track and cross-track are computed separately.** The doc lists the sensitivities as
one scalar, which quietly assumes isotropy. They are not isotropic: a **heading error is purely
cross-track**, an **altitude error is purely along-track.** At nadir the distinction vanishes
(which is where we fly); at 45° it is a factor of two.

**② DEM height uncertainty is its own term.** The doc's budget has no surface term. Copernicus
GLO-30 carries 2–4 m of vertical error and it enters through `tan θ` exactly as barometric
altitude does — so at 45° a 4 m DEM error contributes 4 m of horizontal error, **larger than
everything else combined.** This is the strongest quantitative argument for the nadir-biased
flight policy.

| term | along-track | cross-track |
|---|---|---|
| pointing (attitude ⊕ boresight) | `h·sec²θ·α` | `h·tanθ·α` |
| pixel | `h·sec²θ·(px/f)` | `h·tanθ·(px/f)` |
| heading (yaw) | — | `h·tanθ·ψ` |
| altitude | `tanθ·σ_alt` | — |
| DEM height | `tanθ·σ_dem` | — |
| GNSS | `σ_gnss` | `σ_gnss` |
| time sync | `v·σ_t/√2` | `v·σ_t/√2` |

**Random and bias are tracked separately, and it matters.** Averaging N observations shrinks the
random terms by √N and does **nothing** to the biases. Yaw and boresight are biases. So a
single-pass record still carries the full yaw term, and stacking observations can never talk
CE90 below the bias floor:

```
acc = √( median(σ)² / n_tracks  +  (0.55 × median(σ))² )
```

Getting this wrong makes CE90 come out too small → mistunes the dedup radius → **turns one
survivor into three records on the second pass.**

### 5.3 The telemetry noise actually applied

| source | model |
|---|---|
| GNSS | 2.0 m constant bias + 1.6 m Ornstein-Uhlenbeck walk, τ = 120 s |
| barometer | 0.8 m bias + 0.6 m walk (τ = 90 s) + 0.15 m white |
| yaw | **2.0° per-flight bias** + 0.10° white |
| attitude | 0.20° white, per frame |
| boresight | 0.10° fixed |
| time | 20 ms constant skew + 3 ms jitter |

Deterministic from the seed. A representative realisation: yaw bias **−2.24°**, GNSS bias
**(−3.19, −0.15, −0.44) m NED**, time skew **−21.2 ms**.

### 5.4 Frame-to-frame ≠ absolute — the distinction that fixed tracking

The tracker asks a **different question** from the record and gets a much smaller answer.
Absolute accuracy is dominated by terms **identical in both frames** — yaw bias, boresight, DEM,
GNSS common-mode — and every one **cancels** when you ask "did this box move?".

| AGL | frame-to-frame σ | absolute CE90 | ratio |
|---|---:|---:|---:|
| 60 m | 0.370 m | 5.60 m | **15×** |
| 20 m | 0.236 m | 5.58 m | **24×** |
| 8 m | 0.217 m | 5.58 m | **26×** |

---

## 6. The models

### 6.1 Detector

| | in-domain model | real-footage model |
|---|---|---|
| architecture | YOLO26s, COCO init | YOLO26s, COCO init |
| trained on | simulator frames only | real maritime rescue footage, 1024 px tiles |
| partition | **by scenario seed** | **by source video**, digest `8c4dce70710f7a7b` |
| hardware / cost | H100, 9.1 min, **$0.75** | A100 SXM 80 GB, ~6 min/epoch, **$3.90** |
| stopping | — | `patience=5` against the **best** epoch; stopped at 17, best at 12 |
| held-out result | **mAP@0.5 0.807** | **recall 0.821** on people, mAP50 0.783 |
| domain sentence carried in `provenance.json` | simulated valley | **"maritime rescue exercise — NOT flood/debris"** |

*Fine-tune PR figure intentionally omitted in this PR scope.*

**The domain gap is measured, not assumed:** the real-footage model scores **0.821 on maritime
swimmers** and **0.028 on rendered flood debris** — one of thirty-six, with four false
positives. Note the imagery there is *rendered*, so that number bounds nothing about real
landslide footage either. What it establishes is that **maritime performance must never be
presented as carrying over.**

**Runtime:** 2.2 ms per 1024 px tile on the A100. CoreML export **fails** on YOLO26 (an `int`
op at `10/m/0/attn/523`, identically on coremltools 9.0 and 8.3.0) — it is the architecture
meeting the converter, not a version to pin around. Runtime is PyTorch/MPS and the parity gate
moved to MPS-vs-CPU: **111/111 detections matched, agreement 1.000, max score delta 0.0.**

### 6.2 Verifier

| | |
|:--|:--|
| backbone | `dinov2_vits14`, 22.06 M params, **frozen**, Apache 2.0, pinned by sha256 |
| heads | 4 — `is_real`, `posture`, `occlusion`, `submersion` |
| crops | **98 px** — exactly 7 × 7 patches at patch size 14 |
| data | 7,157 train / 1,699 val / 1,293 test |
| fit time | **15.3 s** — heads only; the backbone never moves |

**Frozen is the load-bearing word.** We do not have enough real labelled survivor crops to
fine-tune a backbone honestly — so we don't. We borrow one trained **without labels** on 142 M
images, and fit only the part the data can support.

**The number that matters** is precision recovery on *real* photographs (526 annotated people,
200 frames, clips the model never saw): at a precision of 0.40, thresholding alone costs 33
points of recall and the verifier gives back **29**.

**The caveat travels inside `provenance.json`, not in a slide:** three of four heads are fitted
on procedurally rendered actors, so their accuracy is a statement about **rendered silhouettes**
and must never be averaged with a real-footage figure. Posture labels are *derived by rule* from
compositor attributes, not human-annotated. `occlusion` scores **0.4674** — close to guessing
from a 98 px rendered crop, and reported rather than hidden.

### 6.3 Radar — 24 GHz FMCW, for the buried case

| | |
|:--|:--|
| what it is | a **second sortie**: return to the record, hover 1.6–2.0 m, stare for a minute |
| data | raw beat cubes, **complex64** (I/Q phase preserved) |
| held-out | **frame 87.5% · dwell vote 88.9%** over 90 fresh dwells |
| dispatch | `POST /api/ops/runs/{id}/records/{rid}/radar` |

**The gate that protects the map:** dispatching radar **cannot move a survey layer.** All seven
coverage arrays must be byte-identical across a dispatch, and `POD_buried` stays exactly zero.

> Every radar number carries, **inside the payload rather than drawn by the view**, the sentence
> that it is agreement between our synthesiser and the reference discriminant — **not field
> performance.** Nothing here has met a real radar or real rubble.

---

## 7. The measurement apparatus

### 7.1 The recall curve — the most load-bearing artefact

```
altitude → GSD → pixels on target → r_slice (THE CURVE) → effort C → POD → the map
```

Measured on **400 held-out frames, 1,385 annotated people**, conf 0.05, match IoU 0.30.
Overall **0.751**. Full table in [`README.md`](../README.md#the-recall-curve--the-most-load-bearing-artefact-in-the-system).

The x-axis is **pixels**, not metres, because that is what the compositor computes as
`critical_px` for a presentation class. *A curve indexed on one quantity and consumed on another
is worse than no curve.*

**It also parameterises the synthetic detector**, which fires with probability `r_slice(size)`.
That is what lets geometry, tracking, dedup, coverage and scoring be gated **before a model
exists** — while still being parameterised by something measured on real footage rather than
invented. Every run that uses it is stamped **`TAINTED`** in its manifest and the evaluator
carries `detector_tainted: true` into every derived metric.

### 7.2 POD and the coverage scale `k`

```
POD = 1 − exp(−k · C)        capped at pod_ceiling = 0.99
```

`C` is accumulated **effort**, credited per presentation class at the rate `r_slice` — so five
passes at 60 m credit a prone body a lot and a limb almost nothing, which is the physical truth.

`k` is **fitted by maximum likelihood over (effort, found) pairs**, never assumed. It is fitted
**within one detector only**, because it converts effort at the rate `r_slice` credits, and that
rate describes one detector on one domain.

**A layer is stamped `CALIBRATED` only at ≥ 40 observations.** Below that it reads
**`PROVISIONAL`** on the badge, and the current fit is provisional — which means the calibration
figures are **partly in-sample** and should not be quoted as out-of-sample until the corpus is
restored. That limitation is carried on the artefact, not remembered by a person.

`pod_ceiling = 0.99` is the mechanism that stops an area ever being closed. **There is no setter
and no code path that raises it.**

### 7.3 The three recalls

Conflating them is the easiest way to mislead with this system.

| KPI | denominator | when to use it |
|---|---|---|
| `recall_overflown` | beings the aircraft **actually flew over** | **comparing sorties** |
| `recall_camera_visible` | every being a camera could ever have seen | "how much of this incident did we find" |
| `recall_buried` | beings under debris | **always 0**, reported so it is visible |

### 7.4 CE90 containment — the honesty check

Every record claims *"this person is within `h_acc_m` metres."* **`within_claimed_ce90` is the
fraction that actually were.** Target 0.90; measured **0.94**.

**Not circular:** association between record and truth uses a radius **3× wider** than the
claimed CE90, so containment is measured, not assumed.

### 7.5 ECE — is the map telling the truth?

The map says "this cell has POD 0.7". Over many cells, did 70% of the people in POD-0.7 cells
actually get found? **ECE is the average gap between the promise and the outcome**, binned and
effort-weighted. Lower is better; 0 means promises exactly kept.

| state | ECE |
|---|---:|
| uncalibrated (`k` assumed) | 0.697 |
| calibrated (`k` fitted) | 0.11 – 0.17 |
| after both defect fixes, 20 m | **0.104** |
| after both defect fixes, 60 m | **0.043** |

### 7.6 False positives, and why precision is not the headline

**FP = a record that matched no living being.** Matching is greedy by record score, by **ground
distance**, tolerance 12 m. Once an actor is matched it is consumed, so **a second record for
the same person counts as a false positive** — duplicates and hallucinations are the same
failure from the operator's chair.

Greedy rather than Hungarian, deliberately: greedy by score is the order a team actually works
the list in, so it is the order in which a wrong association actually costs someone. A global
optimum would flatter the system with knowledge the operator does not have.

**Dedup accuracy is exactly the FP column**, and it is **zero across every sortie**.

---

## 8. Guardrails — where each one is enforced

| rule | mechanism | file |
|---|---|---|
| the pipeline cannot see truth | AST seam test on every commit | `tests/` seam suite |
| **no area is ever closed** | `POD` capped at 0.99; no setter exists | `coverage/` |
| no record is ever deleted | **no DELETE, no UPDATE** in the store; dismissal adds a row | `store/db.py` |
| no priority reaches zero | `w_class(t)` floored at 0.05, property-tested to t + 1 year | `triage/curves.py` |
| posture promotes, never demotes | score computed both ways, larger wins, refusal recorded | `triage/score.py` |
| burial cannot be cleared | `r_slice(buried) ≡ 0` → `C ≡ 0` → `POD ≡ 0` | `coverage/quality.py` |
| identities never leak across splits | partition frozen and sha256-sealed before scene one | `tools/capture/` |
| every number carries its domain | `TAINTED` / `PROVISIONAL` / the domain sentence | manifests, `provenance.json` |

### The R10 scanner

`sightline/triage/guardrails.py` is not documentation — it is a **static analyser** that walks
every lane directory for delete-shaped operations on records and "cleared" vocabulary, with an
`Allowance` table of reviewed exceptions and a `self_test()` that **plants 16 real violations
and requires all 16 to be caught** before the gate goes green.

It even does a wording pass — `_blank_strings_and_comments()`, `_text_spans()`,
`_sentence_before()` — to distinguish a string literal that is a *state value* meaning
"finished" from prose *about* one. As one docstring puts it: this is not a grep.

Three independent layers enforce dismiss-with-reason: the guardrail function, a SQLite trigger,
and an API 422. Uncovered ground reads **"UNSEARCHED, not clear"** — including when the flight
planner runs out of battery reserve.

---

## 9. Evaluation that cannot flatter itself

| mechanism | what it prevents |
|---|---|
| splits **by scenario seed, never by frame**; `assert_no_seed_leak` | a held-out set that shares actors and occluders with training |
| `DomainMixError` | averaging a simulation number with a real one |
| `SliceRoleError` | pooling a hard slice into the headline figure |
| `MetricRow` frozen + validating in `__post_init__` | a `NaN` serialising as invalid JSON; a row being relabelled `real` in place |
| `MetricRow.undefined()` | the "we found nothing" case emitting an explicit `n=0` row **with a reason** rather than silence |
| cross-surface consistency check | any number disagreeing between manifest, store, GeoJSON, KML, CoT, the API, the raster and the evaluator |

**Current cross-surface status: ALL CONSISTENT.**

### The gate suite

Gates are the definition of done, and they are written to **fail on the old behaviour**:

| gate | what it pins |
|---|---|
| small-target tracking under realistic pose noise | **fails at 12 px and 30 px, passes at 112 px** — the defect's exact signature. Verified by reverting the fix and re-running |
| six unit gates on the dedup split | two people 9 m apart are two records; one person's two passes are one record with `count_estimate == 1`, **both tracks retained as evidence** |
| the two-pass gate | one pass 5 records for 7 beings; two passes **7 records, 7 beings, 0 FP** |
| the POD ceiling gate | compares at float32 resolution — `float32(0.99)` is 0.99000000954, and a 1e-9 tolerance failed correctly-capped cells by eight nanometres |
| the radar dispatch gate | all seven coverage arrays byte-identical across a dispatch |
| the planner refusal gate | a **refusal** counts as a band losing, because the planner refuses rather than scoring |

---

## 10. Interfaces

### The command map

Fully offline by construction — MapLibre GL + PMTiles basemap and every dependency
**vendored**. A headless check runs the page with every non-loopback name
unresolvable and **fails if a single external request is made.**

Layers: triage markers (size = rank, colour = class, ring = `h_acc_m`), the POD raster, hatched
*aerial search cannot clear* polygons, the planned pattern, the flight track coloured by mode,
the drone and its camera footprint.

### Exports

| format | for |
|---|---|
| **GeoJSON** (RFC 7946) | primary |
| **KML / KMZ** with embedded evidence | Google Earth |
| **Cursor-on-Target** XML, `ce` = the claimed CE90 | ATAK / WinTAK — the tablets SAR teams already run |
| **GeoPackage** | GIS hand-off |

All produced per run and **checked against each other.**

### Resilient delivery

> *A search does not stop because the radio does.*

| property | how |
|---|---|
| durable | SQLite outbox on disk |
| **at-least-once** | the only thing a lossy link can honestly promise |
| idempotent | the receiver deduplicates on `(record_id, content_hash)`, so a replay is a no-op rather than a duplicate survivor |
| retry-aware | `attempts` incremented per try; `delivered_at` set only on success |

This pairs with the append-only store: the queue can always be rebuilt from the record log, and
**no link failure can lose a survivor.**

---

## 11. Open issues, carried rather than hidden

| # | issue | status |
|---|---|---|
| **K25** | The **120+ px bin is a measurement artefact.** The curve says recall 0.465 above 120 px, but **98.7% of those predictions land their centre inside the person** — the model boxes head and shoulders while the annotation spans the whole body. This system geolocates from a **centre** and never reads extent, so the curve has been scoring a capability the pipeline does not consume. It penalises every low-altitude arm and starves the planner | **open, blocks the altitude ladder** |
| **K1** | **No flood-domain real imagery exists in any open corpus.** The domain-gap experiment ran the maritime model over rendered flood debris: recall 0.028, four false positives | **accepted, with a number.** The imagery is rendered, so it bounds nothing about real landslide footage. What it establishes is that maritime performance must never be presented as carrying over |
| **K20** | Per-class `k` spans 55×, which it should not — class differences are supposed to live in `C` via `r_slice` | root cause was the tracking defect; needs re-measuring after the curve is fixed |
| **K22** | Ultralytics NMS has a wall-clock limit and **truncates silently** when it fires | open, and named because it is the shape of failure this project refuses: *a busy machine finding fewer people, with only a log warning to say so* |
| **K21** | CoreML export fails on YOLO26 | **accepted** — runtime is PyTorch/MPS, parity gate moved to MPS-vs-CPU |

---

## 12. Glossary

| term | expansion | what it means here |
|---|---|---|
| **AGL** | above ground level | altitude over the terrain, not sea level |
| **C** | accumulated effort | credited per presentation class at rate `r_slice` |
| **CE90** | circular error 90% | the radius a record **claims** the person is within. Rayleigh quantile 2.146 × per-axis σ |
| **CoT** | Cursor-on-Target | the XML format SAR and military tablets already read |
| **ECE** | expected calibration error | the gap between what the map promised and what happened |
| **FP** | false positive | a **record** that matched no living being. Includes duplicates |
| **GSD** | ground sample distance | metres of ground per pixel — `AGL / fx` |
| **in-CE90** | containment | of records claiming "within X m", the fraction that actually were |
| **loc p90** | 90th-percentile localisation error | reported instead of the mean, because the mean hides the tail — and the tail is where a team walks to the wrong place |
| **LWIR** | long-wave infrared | 8–14 µm. A *band*; a "thermal camera" is the hardware that images it. They are not alternatives |
| **NMS** | non-maximum suppression | collapses overlapping detections of one object |
| **overflown** | — | beings the aircraft actually flew over — the correct denominator for comparing sorties |
| **POD** | probability of detection | `1 − exp(−k·C)`, capped at 0.99 |
| **r_slice** | — | recall for this presentation class under these conditions. What turns a look into effort |
| **SAHI** | slicing-aided hyper inference | tiled inference so small targets survive downscaling |
| **SITL / SIH** | software-in-the-loop / simulation-in-hardware | PX4 running the real flight code with simulated physics |
| **WBF** | weighted boxes fusion | box-level multi-band fusion |

---

## 13. Where every headline number comes from

| number | what it is | source |
|---|---|---|
| **0 false positives** | across the flown corpus (synthetic detector, stamped `TAINTED`) | run manifests + evaluator |
| **0.720** | `recall_overflown` at 20 m after both fixes | `trk-20m-42` |
| **0.94** | CE90 containment — claimed vs actual | evaluator, 3×-wider association radius |
| **0.043** | ECE at 60 m after both fixes | `trk-60m-42` |
| **42 : 1** | suppression ratio, raw detections → records | 758 → 18, `trk-20m-42` |
| **0.807 mAP@0.5** | in-domain detector, held-out scenario seed | `models/detect/f8b_sim/` |
| **0.821 recall** | real-footage detector, held-out by source video | `provenance.json`, digest `8c4dce70710f7a7b` |
| **×7.6** | recall gain over the COCO baseline | same held-out split |
| **$3.90 / $0.75** | end-to-end fine-tune cost | pod billing, both terminated |
| **+29 pts** | recall the verifier recovers at precision 0.40 | 526 annotated people, real footage |
| **+31%** | what carrying the thermal band is worth | valley sortie, 0.342 → 0.447 |
| **±2.7 m** | geolocation, 1σ nadir | error budget at the design altitude |
| **0.000** | recall below 12 px — a measured correction to our own design doc | recall curve, 8–12 px band |
| **0** | `POD_buried`, at any altitude, forever | by construction, no code path can raise it |

---

<div align="center">

**Every number in this index carries the domain it was measured in.**
**If one does not, that is a defect.**

</div>
