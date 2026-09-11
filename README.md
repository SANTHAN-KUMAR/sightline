<div align="center">

# SightLine

**Aerial survivor triage for flood and landslide disasters.**

A drone flies itself over a flooded valley, finds every living being from 4K nadir imagery,
and hands the incident commander two things — a ranked list of who to rescue,
and an honest map of where nobody has looked well enough yet.

<p>
<img alt="simulation" src="https://img.shields.io/badge/simulation-Unreal_Engine_5.8_%2B_Cosys--AirSim-0b8f76?style=flat-square">
<img alt="detector" src="https://img.shields.io/badge/detector-YOLO26s_fine--tuned-1570ef?style=flat-square">
<img alt="verifier" src="https://img.shields.io/badge/verifier-DINOv2_ViT--S%2F14_frozen-6941c6?style=flat-square">
<img alt="false positives" src="https://img.shields.io/badge/false_positives-0_in_the_flown_corpus-067647?style=flat-square">
<img alt="offline" src="https://img.shields.io/badge/C2_map-fully_offline-b54708?style=flat-square">
</p>

</div>

---

> ### The sentence this whole system exists to make true
>
> **“We searched there and found nobody”** should carry **a number**, not a feeling.
>
> A triage list on its own invites the fatal inference *“nothing on the list means nobody is
> there.”* So SightLine ships the **negative** as a first-class product: a calibrated
> probability-of-detection map, layered by how a person is presenting to the camera.
>
> **Nothing in this system closes a search area. The commander does.** That is enforced in
> code, not policy — see [Guardrails](#guardrails-enforced-in-code-not-policy).

---

<div align="center">

[**The two products**](#the-two-products) · [**Why this is hard**](#why-this-is-hard) · [**Full simulation**](#full-simulation) · [**The pipeline**](#the-pipeline-one-spine)

[**Fine-tuning**](#fine-tuning-the-detector) · [**The verifier**](#the-verifier-precision-you-can-buy-back) · [**Improvements**](#the-improvements-two-defects-that-were-holding-recall-at-zero) · [**Results**](#measured-results) · [**Guardrails**](#guardrails-enforced-in-code-not-policy)

**14 independent lane packages** under `sightline/`, ~56k lines of Python, **998 tests**.

**[docs/SYSTEM-INDEX.md](docs/SYSTEM-INDEX.md)** — the full depth: every module, contract, artefact and number

</div>

---

## The two products

*Figure intentionally omitted in this PR (docs-only scope: README + SYSTEM-INDEX text updates).*

**1 · The triage list.** One record per living being, ranked — with the four terms that made
the ranking shown *separately and never collapsed into one opaque number*, plus the position
and the circle that position is actually good to.

**2 · The search-quality map.** Not "we flew over there", but *how well we looked*, per
presentation class, converted into probability of detection. **No cell ever reads as cleared.**

They only work together. That is the whole argument.

---

## Why this is hard

A person is **not one detection problem.** How they present to a nadir camera changes the
critical dimension by almost **ten times** — the same camera, the same frame:

*Presentation ladder figure intentionally omitted in this PR scope.*

*Rows are presentation classes, columns are altitude. The bottom row is `buried` — and it is
empty at every altitude, because it must be.*

| class | critical dim | px @ 60 m | px @ 20 m |
|---|---:|---:|---:|
| `prone` — lying flat | 1.70 m | 59 | 148 |
| `cluster` — several together | 1.50 m | 52 | 130 |
| `upright` — you see shoulders, not height | 0.45 m | **16** | 39 |
| `wading` — standing in water | 0.45 m | 16 | 39 |
| `head_shoulders` — head barely above water | 0.25 m | 9 | 22 |
| `limb` — an arm out of debris | 0.18 m | **6** | 16 |
| `buried` | 0.00 m | **0** | **0** |

**This table is why a single "recall" number is misleading**, and why coverage is computed
per class rather than per cell.

**`buried` is zero by construction.** A camera cannot see through a metre of debris, so
`r_slice(buried) ≡ 0` → `C ≡ 0` → `POD ≡ 0`, at any altitude, forever. There is no tuned
value and no code path that can raise it.

---

## Full simulation

The simulation is not a training aid — it is **the test bench**, and the renderer is the
deployment domain.

*Flood-valley simulation figure intentionally omitted in this PR scope.*

*Unreal Engine 5.8, the flood valley under scripted weather. The three insets are the capture
stack the drone actually writes: RGB, instance segmentation (labels only), and LWIR.*

<table>
<tr><td width="50%">

**The scene**

| | |
|:--|:--|
| engine | **Unreal Engine 5.8 + Cosys-AirSim v3.4.1** |
| terrain | real **Copernicus GLO-30** DEM, tile `N11_00_E076_00` |
| actors | **1,183** — terrain, flood water, 73 Kerala houses, 919 debris items placed by flood-transport physics |
| survivors | **71**, posed across 10 pose/submersion combinations |
| vegetation | 5,508 instances — 4,342 trees (5 Poly Haven species) + 1,166 ferns |
| weather | scriptable rain, time of day, flood level |
| flight | SimpleFlight, autonomous survey **and** gamepad takeover at any moment |

</td><td width="50%">

**What crosses the seam**

Exactly four artefacts — *what a real drone writes, and nothing else*:

| artefact | carries |
|---|---|
| video | RGB frames (+ LWIR when carried) |
| telemetry | **DJI-format `.SRT`** — lat, lon, altitude, gimbal, timestamp |
| flight log | **PX4 ULog** |
| radar cube | `.npz`, **complex64**, I/Q phase preserved (on dispatch only) |

**No actor list. No truth file. No "which pixel is a person".**

</td></tr>
</table>

### The seam is the most important structural rule in the project

```
   GENERATOR SIDE                 │  SEAM  │            PIPELINE SIDE
   ───────────────────────────────┼────────┼────────────────────────────────
   world, terrain, actors, truth  │        │  frames, telemetry, detections
   knows where everybody is       │   ✗    │  knows only what a drone writes
                                  │        │
   writes: frames/*.jpg           │  ───▶  │  opens: frames/*.jpg
           flight.json            │  ───▶  │          flight.json
           truth.json             │   ✗    │          (never opened)
```

An **AST test runs on every commit and fails the build** if a pipeline module imports a
generator module. `eval/` is the only package allowed to touch both sides, because scoring is
the one job that legitimately needs truth.

*Without this, every recall number in this repository would be measuring the renderer's
bookkeeping rather than a detection system.*

### Two simulation paths, one pipeline

| path | what it gives | what it is for |
|---|---|---|
| **Unreal + Cosys-AirSim** | photoreal flood valley, real foliage occlusion, thermal and segmentation passes | scene realism, occlusion truth, the demo |
| **Compositor over the real DEM** | deterministic frames from the same Copernicus terrain, seeded and reproducible | the measured sortie corpus — same bytes from the same seed |

Both write the same four artefacts, so **one pipeline consumes both** and no result depends on
which produced it.

### Auto-labelling that refuses to lie

Labels come from the simulator's instance-segmentation mask, so boxes are exact by
construction. That turned out to be dangerous in a way worth documenting.

> Cosys-AirSim renders its annotation pass with `InstancedFoliage` and `InstancedGrass`
> **off**, and every plant in this scene is a HISM instance. The mask therefore sees
> **straight through the canopy** — a survivor under a fern is reported whole and unoccluded,
> and the auto-labeller writes a confident box over pure leaf texture.

Fixed at source with the **depth buffer**, which does render the canopy. Depth is planar, so a
nadir frame gives `depth = camera_altitude − surface_altitude` at every pixel:

> a mask pixel claiming actor A is genuinely visible ⟺ `depth(pixel)` is not closer than A's own body

It also yields a **measured** visibility fraction rather than an estimated one, because the
foliage-blind mask happens to be the amodal silhouette.

---

## The pipeline: one spine

Every frame from either source runs through the same object. Nothing bypasses it.

```
  frame + pose + intrinsics
        │
   ① DETECT       YOLO26s + SAHI 1024 px tiles, cross-tile NMS at IoU 0.55
        │
   ② FUSE         weighted box fusion across RGB + LWIR, ProbEn score rule
        │         with LWIR off this is the IDENTITY — nothing silently added
   ③ VERIFY       DINOv2 ViT-S/14 frozen, 4 heads — runs BEFORE tracking, so a
        │         rejected candidate never starts a track
   ④ TRACK        IoU + camera-motion compensation, ground-sized association gate
        │         confirms on 3 hits inside a 2 s window
   ⑤ GEOLOCATE    pixel → ray → terrain intersection → lat/lon, + per-observation CE90
        │
   ⑥ ACCUMULATE   coverage effort into the cell, per presentation class
        │
   ⑦ DEDUPLICATE  DBSCAN on haversine, eps = 2 × CE90, then split on simultaneity
        │
   ⑧ SCORE & RANK P(living) × w_class(t) × urgency × (1 + 0.1 × count)
        │
   triage list + coverage raster → API → UI → GeoJSON / KML / CoT
```

### What "detections → records" means

*Nadir detection figure intentionally omitted in this PR scope.*

The suppression funnel is the entire argument for tracking and dedup existing:

| stage | counts | 20 m run |
|---|---|---:|
| **raw detections** | every box, every frame, every band | **758** |
| **fused** | after cross-band weighted box fusion | **651** |
| **confirmed tracks** | survived 3 hits in 2 s | **26** |
| **records** | after geo-space dedup — one per living being | **18** |

**"Suppressed" is not a rejection pile.** It is the same person seen thirty times collapsing
into one row. The ratio — **42 : 1** here — measures how much noise the spine removes.

A raw detection list is unusable by a rescue team. An 18-row ranked list is.

<br clear="right">

### Leaving the image domain is what makes tracker ID switches stop mattering

A survivor tracked as ID 7 on the first pass and ID 41 on the second is **one record**,
because both land in the same place on the ground.

---

## Fine-tuning the detector

### The baseline that forced it

COCO-pretrained YOLO26s, out of the box, on held-out footage:

| | precision | **recall** | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| COCO YOLO26s, no fine-tuning | 0.865 | **0.108** | 0.062 | 0.027 |

**It finds one person in ten.** The high precision is an artefact of barely firing. For a
system whose entire purpose is not missing people, that is unusable — so the question was
never *should we fine-tune*, but *does it actually fix it, and how cheaply.*

### De-risked on a laptop first

One epoch, batch of four, whole frames — before spending anything on a GPU:

| | precision | **recall** | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| COCO baseline | 0.865 | 0.108 | 0.062 | 0.027 |
| **after 1 epoch, batch 4** | 0.692 | **0.461** | **0.461** | 0.233 |
| change | −20% | **×4.3** | **×7.4** | ×8.6 |

Two questions answered at once. *Does fine-tuning show real value?* — decisively; there is no
reading of ×4.3 recall where it is marginal. *Does a tiny batch work?* — yes, and the
mechanism matters: Ultralytics' `nbs=64` accumulates gradients over 16 steps at batch 4, so
the optimiser and BatchNorm behave as if the batch were 64. Losses fell **monotonically**,
which is the failure mode to fear at batch 4.

### The production run

*Fine-tuning curves figure intentionally omitted in this PR scope.*

*The real trace. The epoch-3 dip is visible in every panel; the peak is epoch 12.*

<table>
<tr><td width="55%">

| setting | value |
|---|---|
| hardware | **A100 SXM 80 GB** (rented) |
| data | **39,932 train / 10,464 val** native 1024 px tiles, 10% overlap |
| partition | **by source video** — held-out clips contain people never seen in training. Digest `8c4dce70710f7a7b` |
| batch / imgsz | 32 / 1024, `cache=ram`, `amp=true` |
| freeze | **none — full fine-tune**, every layer trainable |
| augmentation | mosaic 1.0 (closed last 10), scale 0.5, fliplr 0.5, **flipud 0.0** |
| stopping | `patience=5`, measured **against the best epoch, not the last** |
| speed | ~6 min/epoch · **early-stopped at 17, best at 12** |
| **cost** | **$3.90 end to end.** Pod terminated |

</td><td width="45%">

**Final, held out**

| | precision | **recall** | mAP50 |
|---|---:|---:|---:|
| all classes | 0.820 | **0.758** | 0.749 |
| **person only** | 0.757 | **0.821** | 0.783 |

### ×7.6 recall on people
over the COCO baseline, on the same held-out split, over 7,257 held-out instances — **for $3.90.**

</td></tr>
</table>

<details>
<summary><b>Four observations worth keeping from the training run</b></summary>

<br>

**① Epoch 1 already beat the whole laptop run.** Tiled at batch 32: R 0.661 vs 0.461.
**Tiling was worth more than 16× the batch size** — a statement about *small objects*, not
about optimisation. This is the single most useful thing learned.

**② The epoch-3 dip is not a bug.** P 0.768 → 0.622 and mAP50-95 0.417 → 0.319 in one epoch
looks alarming. It is ordinary mid-training volatility under mosaic augmentation, and it is
exactly why `patience` measures against the **best** epoch rather than the last. Had we
stopped on a two-epoch plateau we would have shipped epoch 2 and lost 5 points of mAP50-95.

**③ Classification loss fell monotonically while detection metrics oscillated.**
1.143 → 0.907 with no reversals, while P swung 0.62–0.82. The model was steadily getting
better at *what* while still arguing about *where* — what you expect when box regression is
the harder half.

**④ `flipud = 0.0` is deliberate.** Horizontal flips are free on aerial imagery; vertical
flips are not, because water surface, sun angle and body orientation all have a definite "up".
Flipping vertically would have taught the model that an upside-down wake is normal.

</details>

### Two fine-tunes, two domains — and the gap between them is measured

Because a number is only worth what its domain says it is:

| model | trained on | held-out result | what it is for |
|---|---|---|---|
| **in-domain** | simulator frames only, 10 pose/submersion combinations | **mAP@0.5 0.807**, $0.75, 9.1 min | flying the simulated valley |
| **real-footage** | real maritime rescue footage, tiled at 1024 | **recall 0.821** on people, $3.90 | proving fine-tuning transfers to photographs |

**The domain gap, measured rather than assumed:** the real-footage model scores **0.821 on
maritime swimmers** and **0.028 on rendered flood debris** — one of thirty-six, with four
false positives. That number is carried in `provenance.json` and reaches every run manifest
automatically:

> **Domain: maritime rescue exercise — NOT flood/debris.**

*Real-vs-procedural figure intentionally omitted in this PR scope.*

*Left: real photographs. Right: our procedural actors. Every number measured on the right
carries the label `PROCEDURAL` and the sentence "not photography; do not report detector
generality from it."*

### The recall curve — the most load-bearing artefact in the system

Measured on **400 held-out frames, 1,385 annotated people**. It is what turns a look into
effort, so **change the curve and every coverage number moves**:

```
altitude → GSD → pixels on target → r_slice (THE CURVE) → effort C → POD → the map
```

| size band | COCO baseline | **fine-tuned** | gain |
|---|---:|---:|---:|
| 8–12 px | — | **0.000** | — |
| 12–15 px | — | 0.611 | ×3.5 |
| 15–20 px | 0.203 | 0.617 | ×3.0 |
| 20–25 px | 0.189 | 0.475 | ×2.5 |
| 25–30 px | 0.277 | 0.604 | ×2.2 |
| 30–40 px | 0.492 | 0.756 | ×1.5 |
| 40–55 px | 0.609 | 0.874 | ×1.4 |
| 55–80 px | 0.526 | 0.911 | ×1.7 |
| 80–120 px | 0.378 | 0.885 | ×2.3 |

**The 8–12 px floor is real and it corrected our own architecture document.** Below about
twelve pixels this detector finds **nothing**, where the design had promised ~0.2. That is why
upright people (16 px at 60 m) are invisible at altitude — and it is a *measured* correction,
not a tuning failure.

---

## The verifier: precision you can buy back

`DINOv2 ViT-S/14`, **frozen**. Not one gradient step reaches the backbone — we run it once per
crop for a 384-dim embedding and train **only four small heads** on top.

| consequence | why it matters here |
|---|---|
| **fit time 15.3 s** | the whole verifier retrains in the time it takes to make tea |
| **cannot overfit the backbone** | our rendered actors cannot corrupt a representation learned from 142 M real images |
| **deterministic, auditable** | backbone pinned by sha256; the same crop always yields the same embedding |
| **honest with a tiny dataset** | 7,157 crops is far too few to fine-tune a transformer, and plenty to fit four heads |

**This is the entire reason a second stage is affordable at all.** We do not have enough real
labelled survivor crops to fine-tune a backbone honestly — so we don't. We borrow one trained
without labels, and fit only the part the data can support.

### What it buys, measured on real photographs

The detector is deliberately tuned toward recall, so it fires a lot — **7.7 false positives
per frame** at conf 0.05. You could raise the threshold, but that throws away real people too,
and **recall is not recoverable downstream while precision is.**

| target precision | recall **without** | recall **with verifier** | **Δ** |
|---:|---:|---:|---:|
| 0.05 | 0.8403 | 0.8403 | 0.0 |
| 0.10 | 0.8403 | 0.8403 | 0.0 |
| **0.20** | 0.7357 | **0.8365** | **+0.1008** |
| **0.40** | 0.5133 | **0.8080** | **+0.2947** |

> **At a precision of 0.40, thresholding alone costs 33 points of recall. The verifier gives
> back 29 of them.** That is the whole argument for a second stage — and it is measured on
> **real photographs of real people**, on clips the model has never seen.

At low precision targets the verifier correctly does nothing: when you accept everything,
there is nothing to recover. The value appears exactly where an operator would want to sit.

<details>
<summary><b>The caveat that travels inside the artefact, not in a slide</b></summary>

<br>

Three of the four heads are fitted on **procedurally rendered** actors
(`is_training_grade = False`). Their accuracy is a statement about **rendered silhouettes** and
must never be averaged with, or quoted as, a real-footage figure.

| head | accuracy | balanced acc | n |
|---|---:|---:|---:|
| `is_real` | **0.9567** | 0.9567 | 1,293 |
| `posture` | 0.7904 | 0.8188 | 644 |
| `submersion` | 0.9239 | 0.5795 | 644 |
| `occlusion` | **0.4674** | 0.5443 | 644 |

**Occlusion is the weakest head, and it is reported rather than hidden.** A four-way occlusion
judgement from a 98 px crop of a rendered silhouette is close to guessing, and the number says so.

</details>

---

## The thermal band, and why it is carried this way

*RGB/LWIR pair figure intentionally omitted in this PR scope.*

Fusion is **late and box-level** — Weighted Boxes Fusion with the ProbEn score rule — because
the property being protected is **the fallback**:

> If thermal is absent, or its registration residual is too large, the fusion step is skipped
> and the output is **exactly** the RGB detector's output. Not a degraded mode. **The identity
> function.**

Tightly-coupled mid-fusion cannot promise that: a missing modality is out of distribution for a
network trained on both. Box-level fusion has nothing to be out of distribution about.

ProbEn's rule `p = Π pᵢ / (Π pᵢ + Π(1 − pᵢ))` returns a single band's own score **unchanged**,
so a thermal-only hot spot survives at night and an RGB-only detection survives over midday
water — the two cases the whole thermal argument turns on. A naive average would halve both.

**Worth, measured:** valley sortie recall **0.342 → 0.447 (+31%)** when LWIR is carried;
1,262 raw detections fused to 1,218, so **44 cross-band pairs merged**.

The band is dropped when it is worthless, **and that decision is measured, not clocked** — in
the afternoon the ground is hotter than a human body and the contrast inverts. The diurnal
model is a tabulated asymmetric curve rather than a cosine, because a cosine forces the trough
exactly twelve hours from the peak (17:00), while ground thermal inertia puts it near 14:00.

---

## The improvements: two defects that were holding recall at zero

This is the cleanest engineering evidence in the project. **Same scene, same seed, same box,
same 315 frames, and identical raw detections (758).** Every difference is downstream of the
detector:

| run | fixes applied | raw dets | tracks | records | **recall** | **upright** |
|---|---|---:|---:|---:|---:|---:|
| baseline | none | 758 | 22 | 12 | 0.480 | **0/7** |
| + dedup split | ① | 758 | 22 | 14 | 0.560 | 0/7 |
| **+ tracker gate** | ① + ② | 758 | **26** | **18** | **0.720** | **4/7** |

Read across: the dedup split changed **records without changing tracks** — it was un-merging
people who had always been tracked. The tracker gate changed **tracks**, and those 4 new tracks
became exactly the 4 upright people. Nothing else moved. **False positives stayed at zero throughout.**

### ① Geo-dedup was merging distinct people

At 20 m, three separate records each had **three real people** inside their radius, and each
said `count = 1`.

**Mechanism:** `eps = 2 × CE90`, and **CE90 does not shrink when you descend** — the budget is
GNSS- and heading-bias dominated, and neither has an altitude lever. So the dedup radius is
13 m at 60 m AGL and *still 13 m at 15 m*. Descending buys pixels, then merges away the people
those pixels found.

**Fix:** split a cluster on the one constraint position cannot express —

> **Two tracks with an observation at the same timestamp are two beings**, whatever the
> distance between them.

The converse is **not** true and must not be assumed: one person's two overflights share no
timestamp and stay one record. So the conflict graph is coloured position-aware.

**Result:** 60 m 0.469 → 0.594 (prone **10/13 → 13/13**); 20 m 0.480 → 0.560.

### ② Association was IoU-only, so small targets were untrackable

Upright people fired **38 detections** and produced **2 tracks**. Detection was fine at 0.613
per look. The entire loss was *between detection and track*.

**Mechanism:** the frame-to-frame pose residual is ~0.32 m and **does not scale with the
target**. At 20 m that is 28 px:

| class | box | residual | associates |
|---|---:|---:|---:|
| prone | 112 px | 29 px | **82%** |
| cluster | 97 px | 31 px | 67% |
| upright | **30 px** | 70 px | **10%** |
| limb | **12 px** | 50 px | **0%** |

A 12 px box cannot overlap a prediction that is 50 px off, **no matter how correct the
association is.** IoU isn't a weak signal there; it is *undefined as a similarity.*

**The control that proved it:** feed the same pipeline the **true** pose and the residual
collapses to **2 px and every class associates 100%** — including the 12 px limb. The geometry
was right; the gate was wrong.

**Why descending could never have fixed it:** halving the altitude halves the GSD, which
shrinks the box and magnifies the residual in pixels *at the same rate*. Upright was 0/9 at
60 m, 0/7 at 20 m, 0/7 at 15 m while prone climbed with every metre — that shape is the
defect's signature.

**Fix:** gate on **3σ of the frame-to-frame residual, sized in metres**, converted through the
GSD. This is a different and much smaller quantity than the CE90 a record carries — yaw bias,
boresight, DEM and GNSS common-mode are identical in both frames and **cancel exactly**:

| AGL | frame-to-frame σ | absolute CE90 | ratio |
|---|---:|---:|---:|
| 60 m | 0.370 m | 5.60 m | **15×** |
| 20 m | 0.236 m | 5.58 m | **24×** |
| 8 m | 0.217 m | 5.58 m | **26×** |

Sizing the gate from CE90 would have gated at 2.6 m — **wide enough to associate two different
people.** Sized on the ground it is obviously safe: two people 0.7 m apart are one cluster, not
two records.

**Result:** 20 m recall **0.560 → 0.720**, upright **0/7 → 4/7**, FP still **0**,
calibration error **0.254 → 0.104**.

### ③ A mechanism that was built, measured, and deleted

An ego-motion refinement — one global (dx, dy) per frame, since the pose error is common to the
whole frame. Principled, and it raised upright tracks 2 → 5 in the diagnostic. Then measured
properly: it **fired on 7.3% of frames** and left the residual at 25.7 px against 28.2 px.

**Deleted.** A mechanism whose measured effect is nothing, and whose failure mode is global, is
not worth the lines.

---

## Measured results

Every sortie, full telemetry noise model applied, real DEM terrain:

| run | AGL | frames | records | **recall** | **FP** | loc p90 | CE90 claimed | **in-CE90** | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trk-60m-42` | 60 m | 127 | 19 | **0.594** | **0** | 4.27 m | 5.49 m | **0.941** | **0.043** |
| `trk-20m-42` | 20 m | 315 | 18 | **0.720** | **0** | 5.94 m | 6.42 m | **0.938** | **0.104** |
| `valley-baseline-42` | 60 m | 680 | 19 | 0.500 | **0** | 5.52 m | 5.10 m | 0.833 | 0.089 |
| `deposit-fan-limbs-42` | 45 m | 680 | 22 | 0.500 | **0** | 4.91 m | 5.18 m | **0.950** | 0.338 |
| `valley-t1-42` *(real model, wrong domain)* | 60 m | 209 | 5 | 0.028 | 4 | 0.79 m | 4.65 m | 1.000 | 0.360 |

**That last row is the honest one, and it is in the table on purpose.** Every run above it was
flown with the **synthetic detector** — which fires with probability `r_slice(size)` from the
*measured* recall curve, so the geometry, tracking, dedup, coverage and scoring chain can be
gated before a model exists. Every such run is stamped **`TAINTED`** in its manifest and the
evaluator carries `detector_tainted: true` into every derived metric.

`valley-t1-42` is the maritime model pointed at rendered flood debris: **precision collapses
along with recall.** That is the domain gap, and it is why no maritime number is ever presented
as carrying over.

### The three numbers that matter most

<table>
<tr>
<td align="center" width="33%">

### 0
**false positives**

across the flown corpus, from
400–1,400 raw detections per flight

</td>
<td align="center" width="33%">

### 0.94
**CE90 containment**

of records claiming "within X m",
the fraction that **actually were**
*(target 0.90)*

</td>
<td align="center" width="33%">

### 0.043
**calibration error**

the gap between what the map
**promised** and what happened
*(0 = promises exactly kept)*

</td>
</tr>
</table>

**CE90 containment is the most important honesty check in the system.** Every record claims
*"this person is within `h_acc_m` metres"* — this is the fraction that were. **It is not
circular:** association between record and truth uses a radius **3× wider** than the claimed
CE90, so containment is *measured*, not assumed.

**ECE asks: the map says "POD 0.7" — did 70% of the people in POD-0.7 cells actually get
found?** Uncalibrated it was 0.697; with `k` fitted by maximum likelihood over (effort, found)
pairs it is 0.043–0.104. The two defect fixes improved it as a side effect, which makes sense:
the map had been promising finds the tracker was throwing away.

**Precision is deliberately not the headline.** In search and rescue a false positive costs a
team a walk; a false negative costs a life. And precision is recoverable downstream where
recall is not.

<details>
<summary><b>Which recall? Three denominators, and conflating them is the easiest way to mislead</b></summary>

<br>

| KPI | denominator | when to use it |
|---|---|---|
| `recall_overflown` | beings the aircraft **actually flew over** | **comparing sorties.** A descent surveys less ground per battery; scoring against everyone in the scene would measure the box, not the altitude |
| `recall_camera_visible` | every being a camera could ever have seen | "how much of this incident did we find" |
| `recall_buried` | beings under debris | **always 0.** Reported so it is visible, never folded into the others as if flying more could move it |

Worked example, `trk-20m-42`: 40 beings total, 2 buried, **38 camera-visible**, but only
**25 overflown** in this box. 18 records → `recall_overflown` **0.720**,
`recall_camera_visible` 0.474. Both are true; they answer different questions.

</details>

---

## The triage score — four components, never collapsed

```
score = P(living) × w_class(t) × urgency × (1 + 0.1 × count_estimate)
```

A commander needs to know **why** a row is at the top, so all four are shown separately.

| component | range | what moves it |
|---|---|---|
| **P(living)** | 0 → 0.999 | fused confidence across independent tracks; **raised** by thermal-positive in the first 12 h and by observed motion. **Never lowered by a thermal negative** — a cold person is still a person |
| **w_class(t)** | **0.05** → 1.0 | survival decay: plateau, then half-life. `head_only` (0 h, 2 h) · `immersed` (1 h, 6 h) · `trapped` (48 h, 72 h) · `stranded` (24 h, 96 h). **Floored at 0.05 — nobody's priority ever reaches zero**, property-tested to t + 1 year |
| **urgency** | 0.85 → 1.30 | `head_only` 1.30 > `immersed` 1.20 > `trapped` 1.00 > `stranded` 0.85 |
| **count** | 1.1 → | a cluster of four outranks a single, but only mildly |

**Posture promotes, never demotes.** The score is computed both with and without the predicted
posture and **the larger wins**; when the prediction would have demoted the record, the refusal
is written onto the record itself: `"assumed-base (prediction refused: would demote)"`.

**Motion is decided by geodesy, not pixels.** A record is "moving" only when it has moved
further than the system's own uncertainty — `disp > 3 × CE90` over at least 5 s, which at
CE90 6.4 m is **19 m of real ground displacement.** Deliberately conservative, because
"moving" *raises* P(living) and must be earned. Consequence, stated plainly: a conscious person
shifting position under debris reads as **static**. The system under-claims motion rather than
over-claims life.

---

## Guardrails, enforced in code not policy

| rule | enforcement |
|---|---|
| The pipeline cannot see ground truth | **AST seam test on every commit**; `eval/` is the only package allowed both sides |
| **No area is ever closed** | `POD = 1 − exp(−kC)` capped at **0.99**. There is **no setter and no code path** that raises it |
| No record is ever deleted | the store has **no DELETE and no UPDATE**. Dismissal demands a reason and **adds a row** |
| No priority reaches zero | `w_class(t)` floor of 0.05, property-tested to t + 1 year |
| Posture promotes, never demotes | score computed both ways, larger wins, refusal recorded on the record |
| Burial polygons cannot be cleared | `r_slice(buried) ≡ 0` → `C ≡ 0` → `POD ≡ 0`, at any altitude |
| Actor identities never leak across splits | partition frozen and **sha256-sealed** before scene one; the digest travels to the training machine |
| Every number carries its domain | `TAINTED` in the manifest, `PROVISIONAL` on the coverage badge, the domain sentence in every model's provenance |

A **static scanner** walks every lane directory for deletion verbs and "cleared" vocabulary,
with a self-test that **plants 16 real violations and requires all 16 to be caught** before the
gate goes green. Uncovered ground reads **"UNSEARCHED, not clear."**

> These are not aspirations written in a design document. Each row above is a test that fails
> the build.

---

## Evaluation that cannot flatter itself

- **Splits are by scenario seed, never by frame.** `assert_no_seed_leak` refuses to build a
  dataset from a single seed — held-out means a different actor layout with different
  occluders, never seen in training.
- **`DomainMixError` and `SliceRoleError`** make it structurally impossible to average a
  simulation number with a real one, or to pool a hard slice into the headline.
- **`MetricRow` is frozen and validates on construction** — the "we found nothing" case emits
  an explicit `n=0` row with a reason rather than a `NaN` that would serialise as invalid JSON.
- **Cross-surface consistency** checks every number against every other place it is stored or
  served — manifest, store, GeoJSON, KML, Cursor-on-Target, the live API, the raster and the
  evaluator. Current status: **ALL CONSISTENT**.

---

## Repository

```
sightline/          the lane packages — one owner per directory, no cross-writes
  ingest/           decode, telemetry parse, SLERP pose, decimation
  detect/           tiled YOLO26s, WBF fusion, DINOv2 verifier
  geo/              pixel → ray → NED → WGS-84, DEM march, error budget
  track/            BoT-SORT + camera-motion compensation
  dedup/            DBSCAN in geo space at 2 × CE90, simultaneity split
  triage/           survival-decay score, ranking, R10 guardrail scanner
  coverage/         per-cell quality → POD, burial polygons
  plan/             search planner over the probability-of-success surface
  store/  api/      append-only SQLite log, outbox, FastAPI + WebSocket
  eval/             metrics, k calibration — the only package allowed both sides
tools/scene/        procedural scene generation and the environment quality gate
tools/capture/      dataset capture, depth-buffer auto-labelling, dataset gate
tools/train/        GPU rental, training, export
app/map/            the offline C2 map — MapLibre + PMTiles, all deps vendored
sim/SightlineSim/   the Unreal Engine 5.8 project
docs/               solution document, contracts, quality gate, system index
```

**→ [`docs/SYSTEM-INDEX.md`](docs/SYSTEM-INDEX.md)** — every module, contract, artefact and
number, with where each one is verified.

### Run it

**→ [QUICKSTART.md](QUICKSTART.md) is the runbook**: how to start the dashboard, bring up the simulator, fly
the live demo, and what each command should print.


```bash
uv sync                                    # restore the pinned environment (Python 3.11)
uv run python tools/doctor.py              # verify the toolchain
uv run pytest tests/ -q                    # 998 tests, the gates
```

The command map against a demo scenario, **no simulator required**:

```bash
uv run python -m sightline.api.serve --port 8781 --demo
# then open http://127.0.0.1:8781/app/map/index.html
```

Fly it live in the simulator with the trained detector:

```bash
uv run python -m sightline.mission.live --alt 45 --speed 7 \
    --detector rgb --weights models/detect/f8b_sim/weights/best.pt --c2 http://127.0.0.1:8781
```

---

## What we learned that is worth carrying forward

1. **A surprising result is a claim about your apparatus.** A recall drop on *large* targets
   looked like a model weakness for weeks. It was an annotation convention meeting a metric
   the pipeline does not consume — the system geolocates from a detection's **centre** and
   never reads its extent.

2. **Three defects were found by rendering a picture and looking at it** — salt-and-pepper
   masquerading as occlusion, square blobs in the thermal band, and actor records silently
   overwriting each other. **Green tests caught none of them.** Eyeball verification is a hard
   rule here for that reason.

3. **Two hypotheses were killed by their own data.** *"Upright is a detection problem, fix it
   by descending"* — refuted; upright fired on 61% of looks, and descending makes it worse.
   *"The dedup merge is the whole story"* — refuted by the fix itself, which moved upright not
   at all. That null result is what forced the diagnostic that found the real cause.

4. **Tiling was worth more than 16× the batch size.** The single most useful thing learned
   from the fine-tune, and it is a statement about small objects rather than optimisation.

---

## Licence and attribution

Unreal assets are CC0 from [Poly Haven](https://polyhaven.com). Detection uses
[Ultralytics](https://github.com/ultralytics/ultralytics) (**AGPL-3.0**) — fine for an open
repository; a closed commercial deployment would need their Enterprise licence. The verifier
backbone is DINOv2 (Apache-2.0). Simulation uses
[Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) (MIT). Terrain is
**Copernicus GLO-30** (ESA/Airbus, open licence). Basemap tiles are
[Protomaps](https://protomaps.com) / OpenStreetMap (ODbL) — vendored, so the command map makes
**zero external requests** at runtime.

<div align="center">
<br>

**Every other system tells you what it found.**
**SightLine also tells you where nobody has looked yet.**

</div>
