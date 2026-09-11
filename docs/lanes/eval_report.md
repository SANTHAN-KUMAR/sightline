# Lane: evaluation report (F19) — session 5, second session

Owner: the second Claude Code session, started 2026-09-11 02:33 while the orchestrator session was flying the
capture campaign. Coordination and lane split: `docs/lanes/COORDINATION.md`.

**Nothing in this lane touched the editor, PIE, AirSim, the GPU, or git.** No writes to `tools/capture/*`,
`tools/scene/*`, `tools/train/*`, `sightline/mission/*`, `sightline/pipeline.py`, `sim/**` or `data/scene/*`.

---

## 1. What was structurally impossible, and now is not

SOLUTION_DOC 5.12 requires **"FP/min per terrain type (water, debris, vegetation, roof)"**, at detection level
and again at record level after dedup. No amount of model training could have produced that number:

| | before |
|---|---|
| `slicing.AXIS_VALUES` | seven axes, **no `context`** — no `MetricRow` could hold a terrain type |
| `detection.false_positives()` | returned `frame_idx / bbox / score / size_px` — nothing about what was fired on |
| `MatchResult.counts_at` | its own docstring: *"a false positive has no ground-truth box and so has no occlusion, posture or pixel-size bin"* — so box-level axes were skipped for FP entirely |
| `GtBox.context` | existed with the right vocabulary, but was set by `detect.dataset._context_for(zone, submersion)` — the survivor's **zone label**, an intent, not an observation |
| `GtFrame` | carried no camera position, so a prediction could not be projected to the ground at all |

`context` is the one box-level axis a false positive *can* carry, because terrain is a property of **where the
box landed**, which is measurable with no ground truth behind it.

This is the same defect, and the same remedy, as occlusion: `actors.json` occlusion was an intent that five
generators contradicted in both directions, which is why `tools/capture/measure_occlusion.py` exists.

## 2. Delivered

| file | what |
|---|---|
| `sightline/eval/context.py` (new, ~440 lines) | ray-traced terrain measurement against the placed scene; `context_rows()` = 5.12's FP/min per terrain type; `measure_dataset_contexts()` re-labels ground truth from measurement |
| `tests/test_eval_context.py` (new) | **20 tests, offline**, incl. two that reproduce defects that were real in this module |
| `sightline/schemas.py` | `SliceKey.context` (optional, `"all"`); `SCHEMA_VERSION` 1.1.0 → **1.2.0** |
| `sightline/eval/slicing.py` | `context` registered in `AXIS_VALUES` and `BOX_AXES` |
| `sightline/eval/detection.py` | `_box_axis_label` learns `context` |
| `sightline/eval/groundtruth.py` | `GtFrame.camera_east_m / _north_m / _asl_m` (optional, `None`), `has_camera_pose` |
| `docs/CONTRACTS.md` §5 | the 1.2.0 amendment, per §1's additive-field rule |
| `tests/test_eval.py` | the slice-grid guard extended to 8 axes, kept strict (exact set equality) |
| `sightline/eval/campaign.py` (new, D2) | pass roles fixed before measuring; `SliceRoleError`; the model-free census |
| `tests/test_eval_campaign.py` (new) | **22 tests**, mostly about what the runner REFUSES to do |

`uv run pytest` over the 11 affected files → **605 passed**.

## 3. Measured, not asserted

### 3.1 The axis convention reproduces, independently

The coverage-geometry fix measured image-right = EAST on 110 boxes at 35 m. Re-measured here on **150 boxes at
55 m**, different pass, different code:

    x=+E, y=-N  (east-right, north-up)     median  2.75 m   <- TRUE
    x=+E, y=+N                             median 27.17 m
    x=+N, y=+E  (rotated 90)               median 43.49 m

At 35 m this lane gets **1.38 m** against the coverage lane's 1.37 m.

### 3.2 Projecting onto the terrain is wrong, and by how much

260 boxes across both passes, fix vs the survivor's known position in `actors.json`:

| scale taken from | median | p90 | max |
|---|---|---|---|
| the frame's GSD (terrain-referenced) | 2.03 m | 9.31 m | **77.59 m** |
| the target's own elevation | 0.77 m | 3.13 m | 5.01 m |

Every 77 m outlier was a **roof survivor imaged far off-axis** — pure parallax. The shipped classifier reaches
**median 1.32 m, p90 3.31 m, max 5.54 m** while solving for the surface height it does not know in advance.

### 3.3 The obvious implementation oscillates

Project onto the terrain → look up what is there → re-project at that surface's top → repeat. It does not
converge: the ray sweeps sideways as it descends (~15 m over a 20 m tree at a corner pixel), so each lookup
lands on a different object. Measured: **37 of 260 unconverged, residual p90 10.84 m**. Replaced by tracing —
each candidate is tested *at its own elevation* and the highest hit wins. Now **311 of 313 converge**.

### 3.4 Crowns are occlusion, not something to stand on

Letting vegetation drive the position solve dragged **26 of 140** roof survivors onto neighbouring canopy and
moved fixes by up to 14 m — against `tools/scene/check_vegetation.py`, which independently asserts no
unoccluded survivor sits under a crown at all. Vegetation is excluded from the position solve and still
reported as the context once the position is settled, so a target genuinely under canopy still reads
`vegetation`.

### 3.5 Two checks with teeth, on the flown data

`actors.json` records two things the classifier never reads:

* **who is on a roof** → **130 of 140** roof observations measure `structure` (**92.9 %**). The 10 misses are
  the tail of the projection error (3.4–4.5 m) pushing a fix off a small roof; 2 of 10 are flagged `ambiguous`.
* **who is in the water** → **45 of 45** submerged survivors land over **flooded terrain** (100 %), against a
  **53.2 %** control for dry survivors.

  Note the first version of this check asserted submerged survivors measure `water`, and it was a bad check:
  `submersion` describes the *person*, not the ground. Someone can be half-submerged **and** standing in a
  debris field — which is what 15 of them are. The test was fixed, not the code.

### 3.6 The heightfield axis order is decided by measurement

A silent transpose is exactly the defect that left `coverage/footprint.py` 90° out. `SceneGeometry.load()`
tries both orders against the known ground-level survivor elevations and keeps the better (**0.423 m** median
error); a test asserts the wrong order is at least 5× worse, so the check cannot pass vacuously.

## 4. The finding

**The measured terrain disagrees with the assumed one for 41.5 % of boxes** (183 of 313 agree), in both
directions:

    assumed structure -> measured open_ground   n=29
    assumed debris    -> measured open_ground   n=22
    assumed structure -> measured water         n=15
    assumed water     -> measured debris        n=15
    assumed water     -> measured vegetation    n=14
    assumed structure -> measured debris        n=13

`measure_dataset_contexts()` re-labels ground truth from measurement, so recall-per-terrain and
FP/min-per-terrain are the same axis rather than two different things sharing a name.

## 5. Owed to the orchestrator lane — one line in `survey.py`

`sightline/mission/survey.py:161` records **only the 71 actors'** palette entries. The full 1,252-object
name→RGB map exists at that moment (`names`, `_pal_rgb`) and is thrown away, so the dataset is
self-describing for survivors and **not** for terrain — a mask cannot be read for what a pixel *is*.

That is why this lane measures terrain from the placed scene rather than from the rendered mask. Persisting
the full map (~40 KB) would let a future capture read terrain type straight off the mask and cross-check the
geometric answer. `data/scene/thermal_table.json` is not a substitute: it holds 1,162 objects against the
campaign's 1,252, so the palette indices no longer line up.

**Not applied** — `survey.py` is the orchestrator's file and the campaign is mid-flight.

## 6. D2 — the acceptance runner (`sightline/eval/campaign.py`)

### 6.1 The trap the slice grid does not catch

`SliceKey` has an `altitude_band` axis, and it does **not** separate the campaign:

    altitude_band(45.0) == "45-60"      <- alt45rain, the RAIN pass
    altitude_band(55.0) == "45-60"      <- alt55, the ACCEPTANCE pass

There is no weather or condition axis, so any aggregation over the grid folds a rain pass into the headline
figure and nothing objects. `SliceRoleError` is the sibling of `DomainMixError`: passes with different roles
are never pooled, structurally. A test asserts the two bands are equal, so if `ALTITUDE_BANDS` is ever re-cut
the guard is reconsidered rather than quietly becoming redundant.

`role_of()` derives each role from the pass's own data card by the rules in `NOMINAL_SLICE` — below/above
40-60 m is a named HARD slice, in-band with weather is reported SEPARATELY, in-band and clear is THE
acceptance figure — so a pass flown later gets its role automatically and no slice can be promoted after the
numbers are seen. Each role carries a recorded `why`.

### 6.2 It immediately caught a real ambiguity

`_artifacts/dataset/` holds `train_seed23`, the abandoned 45 m run, whose data card is indistinguishable from
a campaign pass. It scores `acceptance` on its own card, so `acceptance_spec()` found **two** and refused
rather than picking one. Which runs constitute a campaign is now a declaration — `load_campaign(include=[...])`
— and naming a pass that is not there is an error, not a silent skip.

### 6.3 The census, which needs no model

> **SUPERSEDED, 2026-09-11 ~03:25.** The orchestrator session found that Cosys-AirSim's annotation renderer
> sets `show_flags.SetInstancedFoliage(false)` / `SetInstancedGrass(false)`
> (`ObjectAnnotator.cpp:SetViewForAnnotationRender`). Every plant here is a HISM instance and every rubble
> slab an ISM instance — both forced by the Windows commit limit — so **the instance mask renders terrain
> straight through canopy and slabs**, and a survivor under a fern appears in it whole and unoccluded. Median
> distinct mask colours over 29 sampled 4K frames: three. 18 of 313 boxes had no subject visible in RGB at
> all and are now `ignore: true`. The corrected counts are **alt55 192 scored (not 200), alt35 103 (not 110),
> campaign 295 (not 313)**. The table below predates that flagging.
>
> `census()` needs no change — it already excludes `uncertain` / `ignore` / group boxes per §6.3, so it will
> report the corrected numbers on the next run. What is stale is the numbers published here, and the
> "41.5 % of boxes" disagreement figure in §4, which was measured over 313.

Measured on the two passes flown so far:

| | `seed23_alt35` (hard) | `seed23_alt55` (**acceptance**) |
|---|---|---|
| frames | 347 | 293 |
| boxes scored | 110 | **200** (3 excluded as uncertain/group per 6.3) |
| minutes | 21.6 | 19.6 |
| unique survivors | 44 | 44 |
| one box is worth | 0.91 pts of recall | **0.50 pts** |
| terrain | roof 49, debris 23, open 19, veg 13, water 6 | roof 90, debris 40, open 31, water 21, veg 18 |
| pixel height | 80+ 57, 40-80 28, 20-40 22, <20 3 | 40-80 125, 20-40 47, **<20 22**, 80+ 6 |

**`docs/TRACKER.md` estimated the acceptance slice at "~72 boxes — too thin to claim >= 90 % recall on". It is
200.** Worth correcting before that estimate hardens into a decision.

The pixel-height column is the more useful warning: at 55 m only **6 boxes reach 80 px** and **22 fall under
20 px**, against 57 and 3 at 35 m. Whatever recall the acceptance slice returns will be dominated by the
40-80 px band, and the sub-20 px boxes sit at or below the operating floor 5.12 asks to be identified.

`recall_resolution` is reported beside every count: a 90 % claim over 200 boxes moves half a point if one box
flips, and that belongs next to the figure rather than in a footnote.

### 6.4 A guard in the eval lane caught me

`census` first emitted `recall_resolution = inf` for an empty pass. `metric_row` refuses any non-finite value
outright — *"report an explicit n=0 row with a note instead of a NaN or an infinity"* — so it now emits n=0
with a note saying the resolution is undefined. The guard was right and my test expecting `inf` was wrong.

## 7. Left in this lane

1. Wire predictions into the runner — `context_rows` and the census both take them unchanged; there is no
   model yet.
2. Record-level FP/min per terrain (post-dedup), the second half of 5.12's requirement.
3. Populate `GtFrame.camera_*` inside `sightline/detect/dataset.py` rather than from
   `campaign.camera_track()`. Works either way; `detect/dataset.py` is contested, so it was left alone.
4. **E**: the R10 source scanner covers 2 of 13 lane directories (AUDIT S10).
5. **F**: a NaN can leave the metric API at the package boundary (AUDIT S8).

## 8. Honest limits

* The classifier is **geometric**, against the layouts the scene was built from — not read from rendered
  pixels. It inherits any error in those layouts. Its own accuracy against independent ground truth is
  92.9 % (§3.5) and belongs in the report next to any number it enables.
* Prop footprints are the only assumed dimensions: `props_layout.json` records a placement but no bound, so
  `_PROP_SIZE` carries a nominal radius and height per role. Everything else is read from the layouts.
* `ambiguous` fires when the fix's uncertainty disc is mixed; it flagged 2 of the 10 roof misses, not all 10.
  A 3.3 m p90 against ~8–10 m houses cannot do better without per-box refinement.
