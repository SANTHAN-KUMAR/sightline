# Independent audit — 2026-09-11 00:00–00:30 IST

Auditor lane. Brief: disprove the claims in `docs/lanes/*.md`, `docs/TRACKER.md` "Handoff state" and the
scene/dataset JSON. Read-only on the project; no `mcp__sightline__*` / `mcp__unreal__*` calls; no GPU, no torch.

**Bottom line.** The Python pipeline is in much better shape than the session's history suggests: 664/664 tests
pass, and **43 of 45 deliberate code mutations were caught by the existing tests**. Every load-bearing number I
re-derived from `SOLUTION_DOC.md` reproduced. The failures are all on the other side of the line — in the
**scene and capture data**, where nobody has looked at a picture since 22:46. The dataset being written right
now is unusable, and the check that was supposed to catch it prints the evidence and passes.

Times are IST on 2026-09-10/11. Several lanes were editing files while I ran; where that matters I say so.

**Scope note.** I audited `coverage_plan_eval.md`, `ingest_track_dedup.md` and `scene_polish.md` in full — they
were the only lane reports present when I started. `detect.md` (23:51), `scene_realism.md` (23:59),
`env_assets.md` (00:03) and `store_api_map.md` (00:04) landed mid-audit and got a spot-check only: their stated
test counts are real (88 detect / 33 api / 21 store, all reproduced), `detect.md` labels `CropVerifier` an
honest stub, and `scene_realism.md` §A.6 is the direct subject of S3 below. They have **not** been audited to
the depth of the first three.

---

## Findings, by severity

### S1 — CRITICAL. The capture flight running right now is producing garbage ground truth

**Claim** (`docs/TRACKER.md` handoff, "Verified working"): *"Capture (F5) by FLYING (F2) … A 60-frame test
validated clean; survivors measured 51-93 px."* The run in progress is `_artifacts/dataset/train_seed23`.

**What I did.** Read the run's own `telemetry.csv` and `labels/*.json` (no sim calls), then drew the boxes on
the frames and looked.

**Evidence.** At 335 frames written:

```
boxes total 940; implausibly large 737 (78.4 %)
worst (size_px, plausible_ceiling_px, frame, actor, pose, agl):
   (3840, 358, 93, 'Human_028', 'sitting', 35.6)
   (3840, 353, 94, 'Human_028', 'sitting', 36.1)
   (3840, 351, 97, 'Human_028', 'sitting', 36.3)
frames containing a FULL-FRAME box: 238 of 335
actors that got a full-frame box: [('Human_019', 178), ('Human_028', 62), ('Human_068', 1)]
```

("plausible ceiling" above is generous: 2.5 m of body at that frame's own AGL with f_px = 2548.72, doubled.)

Frame 93, label file verbatim:

```
Human_028    pose=sitting         bbox=[0, 0, 3839, 2159] visible_px=8292369 size_px=3840 occl=1
Human_032    pose=waving          bbox=[3713, 1688, 3815, 1744] visible_px=1729 size_px=103 occl=0
Human_040    pose=half_submerged  bbox=[3163, 1540, 3164, 1547] visible_px=8    size_px=8   occl=1
Human_044    pose=half_submerged  bbox=[3826, 557, 3839, 599]   visible_px=294  size_px=43  occl=1
```

`visible_px = 8,292,369` out of a 3840×2160 = 8,294,400 px frame. **99.98 % of the frame is labelled as one
sitting survivor.** The mask itself says why:

```
mask shape (2160, 3840, 3);  distinct instance colours in the mask: 4
   colour [255, 127, 159] px 8292369     <- resolved as Human_028
   colour [255, 191, 159] px 1729
   colour [255, 159, 159] px 294
   colour [159, 255, 159] px 8
```

**What is actually true.** The terrain/background instance colour is being resolved to a `Human_###`. The
mechanism is in `D:\Sightline\tools\capture\labels.py`:

```python
def actor_index(names: Iterable[str]) -> dict[int, tuple[int, str, str]]:
    """palette index -> (actor_id, canonical name, class) for every Human_/Animal_ instance."""
    for i, n in enumerate(names):          # <-- i is the POSITION IN THE LIST
```

`labels_from_mask` then takes that `i` as an index into `palette_rgb(simGetSegmentationColorMap())`. Position in
the name list is only the palette index if `names` is the complete, identically-ordered instance list. It is
not, so three humans have been handed colours belonging to other objects — one of which is the terrain. A
nadir frame over terrain + houses + water containing **four** distinct mask colours is itself the tell.

I could not run the editor to confirm which object owns `[255,127,159]`, but the arithmetic is not in doubt.

**And the validator does not catch it.** `tools/capture/validate.py` on this run:

```
F    boxes: 948, size px  min 3 p50 1915 p90 3840 max 3840
WARN  6 boxes under 12 px (below the section 5.5 >=20 px rule; keep or drop knowingly)
FAIL  BURIED survivors appeared in labels: [62] (section 2.7 says aerial search cannot find them)
1 check(s) FAILED - do not train on this dataset
```

A **median human box of 1915 px** is printed as a statistic and passed. Check F has no plausibility bound
against the frame's own AGL. The run fails for an unrelated reason (S6), which is luck, not the check working.
Check C ("label-image agreement: 194 boxes over 56 frames, 0 mismatched") passes precisely *because* the mask
really does contain that colour over the whole frame — it verifies label↔mask consistency, not plausibility.

**Also in the same run:** only **13 of 69** detectable survivors have been seen; 56 never seen, including 18
sitting / 11 standing / 9 waving in the settlement.

**Action.** Kill or discard this run. Fix `actor_index` to use the real per-actor segmentation/instance id, not
the list position. Add to `validate.py` check F a hard failure when `size_px > critical_dim_m * f_px / agl_m`
with a small margin, per frame.

---

### S2 — HIGH. `data/scene/actors.json` is stale against `data/scene/poses.json`; 23 survivors are seated at the wrong height

**Claim** (`docs/TRACKER.md`): *"Ground truth: survivor positions match the sim to 0.00 m."* And the F1 run
order presents `gen_actors.py` → `build_poses.py` → `build_actors.py` as consistent.

**What I did.** `gen_actors.build(23)` is a pure function. I called it twice in-process (no writes) and diffed
against the file on disk.

**Evidence.**

```
gen_actors  run1==run2: True   matches on-disk: False
actors differing: 71 / 71

pose               n  ground_offset_cm on disk -> regenerated (delta)   height_cm on disk -> regen
sitting           21     -24.23 ->   -66.44  (d= -42.21)                109.0 ->   76.4
supine             2     -68.46 ->   -76.20  (d=  -7.74)                 46.0 ->   29.3
prone              6     -58.35 ->   -58.35  (d=  +0.00)                 42.2 ->   32.2
standing/waving/half_submerged/trapped: offsets unchanged
position / character / zone / submersion / occlusion / aerially_detectable changes: 0
```

File times: `actors.json` 22:12:37, `poses.json` 22:45:30. `poses.json` was rewritten by `build_poses.py` after
commit `e58f83e "…fix the sitting pose…"`; `actors.json` was never regenerated.

**What is actually true.** `tools/scene/build_actors.py:62` places each actor at
`(rec["base_asl_m"] - BASE_Z) * 100 + rec["ground_offset_cm"]`. With the stale value, **21 sitting survivors are
spawned 42.2 cm above their seat and 2 supine survivors 7.7 cm above the ground.** From nadir that is nearly
invisible; from any oblique QA view they float. Separately, the `height_cm` / `footprint_cm` that the F5
auto-labels and the eval slices treat as ground truth are wrong for 29 actors (sitting, supine, prone).

The positions themselves are fine — the "0.00 m" claim is about lat/lon and it holds. It is the vertical and
dimensional truth that is stale.

**Action.** `uv run python tools/scene/gen_actors.py`, then re-run `build_actors.py`. One command, but it must
happen before any dataset is kept.

---

### S3 — HIGH. The tree generator's survivor clearance does not implement the rule its own docstring states

**Claim** (`tools/scene/gen_vegetation.py`, module docstring, "CLEARANCES (all enforced, all reported)"):

> `survivor + 3.0 m` — **a crown over a survivor marked `aerially_detectable` would contradict the ground
> truth**, exactly like the foam quads in `gen_foam.py`

and, restated in `docs/lanes/scene_realism.md` §A.6 (landed 23:59, after I started):

> house footprint + 2.0 m (50 rejected) · survivor + 3.0 m (1 rejected) … no trunk stands in a wall and
> **no crown centre sits over a survivor**, for the same reason `gen_foam.py` keeps foam off them: it would
> contradict the `aerially_detectable` ground truth.

**What I did.** Intersected `data/scene/vegetation.json` (4362 trees, each with `crown_r_m` / `crown_r_minor_m`
and species height) against `data/scene/actors.json` (71 survivors).

**Evidence.**

```
survivors under a crown whose TOP clears their head by >1 m: 32 / 71
  ... of which within HALF the crown radius: 0   (nearest is d/r = 0.52)

actor      pose            occl detect      d_m  r_maj  r_min   d/r top-head_m
Human_005  sitting         0    True       5.91   10.6   8.32  0.56       11.4
Human_066  standing        0    True       7.66  14.82  11.62  0.52       23.8
Human_022  sitting         1    True       7.47  12.25   9.61  0.61       11.2
Human_043  half_submerged  1    True       7.48  12.37   9.71  0.61       17.9
… 28 more
```

**What is actually true.** The check is

```python
def clear_of_survivors(e, nn, r=CLEAR_SURVIVOR_M):        # CLEAR_SURVIVOR_M = 3.0
    return not bool(np.any(np.hypot(sv[:, 0] - e, sv[:, 1] - nn) < r))
```

— a **trunk**-to-survivor distance, tested *before* the species and scale are drawn, so the crown radius is not
even known yet. Jacaranda crowns in this layout run 8–15 m in radius. 3 m of trunk clearance does not keep a
12 m crown off a survivor. 32 of 71 survivors (45 %) sit inside a crown footprint whose canopy top is 8.6–23.8 m
above their head; 21 of those are recorded in `actors.json` with `occlusion: 0`.

This is the QUALITY_GATE's row 7 with the sign flipped: the ground truth says "findable, unoccluded"; the
geometry says "under a tree". I cannot render, so I cannot say how many pixels survive — but the clearance rule
as written cannot be the reason to believe they do.

"No crown **centre** sits over a survivor" is literally true and is not the property that matters: a crown
centre does not occlude, a crown does. The lane knowingly allows overhang for roofs ("that is realistic and is
also the partial-occlusion case the detector should see") but presents the survivor rule as protecting the
ground truth, and it does not. The lane's 217-check verifier asserts "every clearance (house / survivor /
launch pad)" — so it re-verifies the 3 m trunk rule and passes while the ground truth is wrong. That is a check
that cannot fail in the way that matters.

**Action.** Clear against `crown_r_minor_m * scale`, not 3.0 m; or re-derive `occlusion` from a rendered mask
after placement. Also re-check the 1 rejection reported as `rejected_survivor_clearance` — that number is
measuring the wrong thing.

---

### S4 — HIGH. `gsd_cm_px` in the F5 telemetry is a constant, and the flight is not at the altitude the data card will claim

**What I did.** Read `_artifacts/dataset/train_seed23/telemetry.csv` and the writer.

**Evidence.**

```
agl p0/p10/p50/p90/p100: [18.7, 37.3, 46.0, 75.6, 121.4]
frames above the 120 m legal ceiling: 1
frames below 25 m: 8
unique gsd_cm_px values: ['1.766']        <- one value, 335 rows
```

`sightline/mission/survey.py:203` writes the per-frame row as

```python
round(asl - surface_asl(cur_e, cur_n), 2),   # agl_m: measured, correct
...
round(a.alt / f_px * 100.0, 3),              # gsd_cm_px: a.alt is the CLI --alt, i.e. 45, always
```

and `:232` puts the same constant plus `"altitude_m_agl": a.alt` into `data_card.json`.

**What is actually true.** 45 / 2548.72 × 100 = 1.766 cm/px, which is the nominal figure and is wrong for every
frame whose AGL is not 45 m. At the run's own p90 (75.6 m) the true GSD is 2.97 cm/px; at the max (121.4 m) it
is 4.76 cm/px — **understated by 2.7×**. The ingest lane declares `gsd_cm_px` "carried, not used for geometry",
so the geolocation chain is safe; anything that slices by pixel size or quotes a GSD is not.

The altitude spread is a finding in its own right. §5.3b puts the `upright` ceiling at 60 m; the 90th percentile
of this flight is 75.6 m, so a large fraction of it is below the ≥20 px rule for upright and wading survivors
while still emitting ground-truth boxes for them. One frame is above the 120 m legal cap.

**Action.** Two-line fix: use the per-frame AGL for `gsd_cm_px`, and write the measured AGL distribution
(min/p50/p90/max) into the data card instead of the setpoint. Separately, the terrain-follow is not holding
45 m — worth understanding before the run is repeated.

---

### S5 — MODERATE. The scene QA gate has never passed; nothing generated after 22:46 has been looked at

`docs/QUALITY_GATE.md` §0 (added 23:32) makes `tools/scene/assert_qa_fresh.py` mandatory. Run at 00:05:

```
REAL EXIT=1
FAIL  no QA render has ever been taken (D:\Sightline\_artifacts\editor_shots\qa_manifest.json missing).
      Run:  ue_python exec tools/scene/qa_shots.py   then OPEN the images.
```

The gate itself works (real exit code 1 — I checked without a pipe, because `| tail` swallows it). The last QA
render is `qa_1_valley.png` / `qa_2_settlement.png` / `qa_4_nadir45.png` at **22:46**. Since then:
`damage.json` 23:15, `foam.json` 23:15, `vegetation.json` 23:42 (4362 trees), `rubble_layout.json` 23:45
(1799 items), `utilities.json` 23:47 (60 poles, 19 boats). **None of it has been seen.**

Clean bill on one adjacent worry: all 25 PNGs in `_artifacts/editor_shots/` are byte-distinct (I hashed them).
Several share a file size, which is a fixed-resolution artefact, not a repeated capture.

---

### S6 — MODERATE. A "buried, aerial search cannot find them" survivor is visible again, at 30 m

`validate.py` fails the live run with `BURIED survivors appeared in labels: [62]`. Detail:

```
frame 48  Human_062  size_px 115  visible_px 2534  agl 29.9
frame 49  Human_062  size_px 118  visible_px 2511  agl 29.5
```

TRACKER claims *"Two 'buried' survivors are now genuinely buried under a debris cap — verified 0 visible pixels,
0 labels emitted."* That verification evidently held at the altitude it was checked at and not at 30 m.

**Clean bill on the caps themselves.** Both buried survivors do have real caps in both layout files — this is
not a missing-cap bug, it is an insufficient-cap bug:

```
Human_055 nearest: 0.21 m Rubble_burial_1794 S_slab_4 "collapsed floor slab over Human_055: 2.7 burial boundary"
                   0.34 m Debris_rock_moss_set_01_0642 "burial cap over Human_055 (2.7: aerial search cannot clear)"
Human_062 nearest: 0.17 m Rubble_burial_1796 S_slab_3 …
                   0.34 m Debris_rock_moss_set_02_0651 …
```

---

### S7 — MODERATE. Two tests do not fail when the code is broken

Full mutation results are in the table below; these are the two survivors.

**(a) `tests/test_plan.py::test_revisit_queue_is_ordered_by_residual_pos_and_skips_burial_polygons`.**
`docs/lanes/coverage_plan_eval.md` §2.2 claims the revisit queue is *"Ordered by residual POS = POA × (1 − POD)"*.
Reversing the queue's pop order leaves all 31 plan tests green:

```
*** GREEN - NOT CAUGHT ***   M26 revisit queue order reversed
   sightline/plan/revisit.py:  "return self.items.pop(0) if self.items else None"
                            -> "return self.items.pop()  if self.items else None"
   31 passed in 0.94s
```

The test does assert `q.pop().priority == max(it.priority for it in items)` — but I ran the fixture and the
queue it builds has **one element**:

```
n items: 1
priorities: [0.5]
all equal? True   max == min: True
```

With one item, `pop(0)` and `pop()` are the same call, `max == min`, and `[x] == sorted([x], reverse=True)` is
free. Three assertions, zero discriminating power. The flat POA gives every low-POD cell the same residual, so
even a larger fixture would not order anything — the fixture needs a *non-flat* prior.

**(b) `tests/test_store.py` does not cover the R10 dismissal trigger on INSERT.**

```
*** GREEN - NOT CAUGHT ***   M22 R10: dismissal-needs-reason (INSERT) disabled
   sightline/store/db.py: r10_dismiss_needs_reason_ins WHEN clause replaced with WHEN 0
   21 passed in 1.51s
```

`test_the_database_refuses_a_reasonless_dismissal` exercises the **UPDATE** trigger by raw SQL, and
`test_dismissal_requires_a_reason_and_keeps_the_row` exercises the Python guard in `Store.upsert`. Nothing
exercises the INSERT trigger, which is the path a record arriving already-dismissed from the outbox or a direct
DB write would take. Low live risk (two other guards stand in front of it), but the trigger was written for a
reason and is unproven.

---

### S8 — MODERATE. A NaN can leave the metric API; `sightline/eval/`'s guarantee stops at its own package boundary

**Claim** (`docs/lanes/coverage_plan_eval.md` §2.3): *"The API makes a bare float impossible … `make_slice()`
without a domain is a `TypeError`; a NaN metric raises."*

**What I did.** Five probes against the frozen schema and the eval guards, then a realistic dedup call.

**Evidence — the domain half of the claim is genuinely sound:**

```
SliceKey()                       -> TypeError: missing 1 required positional argument: 'domain'
SliceKey(domain="banana")        -> accepted by schemas.py, but eval.metric_row and MetricSet.add both reject it
metric_row("x", nan, sim)        -> ValueError
combine_rows([sim row, real row])-> DomainMixError
```

**Evidence — the NaN half does not hold outside `sightline/eval/`.** `sightline/dedup/metrics.py:140-141` and
`sightline/detect/threshold.py:288-294` build `MetricRow(...)` directly from `schemas.py`, bypassing
`eval.metric_row()`'s finiteness check. The "we found nothing" case — the one you most need to report honestly
— produces:

```
dedup.record_precision      value=0.0   n=0
dedup.record_recall         value=0.0   n=1
dedup.mean_position_error_m value=nan   n=0   <-- NON-FINITE
dedup.ce90_m                value=nan   n=0   <-- NON-FINITE

json.dumps produced: *** bare NaN in the JSON (RFC 8259 invalid) ***
strict JSON parse FAILS: non-finite NaN
```

`sightline/export/geojson.py` already rejects non-finite coordinates for exactly this reason. Pooling is safe
(`weighted_mean` with n = 0 raises), so this is a serialisation and display hazard, not a wrong average.
Minor related note: `MetricRow` is a mutable `@dataclass(slots=True)`, so `row.slice = SliceKey(domain="real")`
relabels a sim number in place. Requires intent; worth `frozen=True` on the row if it is cheap.

---

### S9 — MODERATE. The capture-format contract test watches a writer that did not write the dataset

**Claim** (`docs/lanes/ingest_track_dedup.md` §2.1): *"`test_capture_run_columns_match_the_capture_writer_source`
reads the header literal **out of `tools/capture/run.py` itself** and asserts it equals `CAPTURE_RUN_COLUMNS`.
If the sim lane adds, renames or reorders a column, that test fails and names the difference."*

**What is actually true.** There are two capture writers.

```
tools/capture/run.py:188          21 columns, ending  … "flood_level_asl_m", "n_labels"
sightline/mission/survey.py:138   22 columns, ending  … "flood_level_asl_m", "n_labels", "speed_ms"
```

`survey.py` is the one that produced `_artifacts/dataset/train_seed23` (TRACKER: *"Capture (F5) by FLYING (F2):
`sightline/mission/survey.py`"*). The contract test pins `run.py` only. `speed_ms` is benign — the reader reads
by name and carries unknown columns — but the guarantee the report states is weaker than advertised: the writer
that matters is unwatched.

---

### S10 — LOW. The shipped R10 source scanner covers 2 of 13 lane directories

`sightline/triage/guardrails.py` is the best of the R10 checks — it blanks strings and comments before matching
code patterns, and matches SQL/literals in a separate raw pass. But:

```
LANE_SOURCE_DIRS = ("sightline/triage", "sightline/export")
```

The dedup lane has its own grep over `ingest`/`track`/`dedup`; the coverage lane has its own `tokenize` scan
over `coverage`/`plan`. Nothing covers `geo`, `store`, `eval`, `api`, `detect`, `common`, `mission`,
`pipeline.py`. I ran the scanner over the whole tree — **32 hits, and I read every one. None is a record
deletion.** The list, for the record:

* benign container work: `eval/slicing.py` (set pops), `schemas.py` (deserialisation), `api/wire.py` (pops from
  a *copy* and puts the data back), `geo/noise.py`, `store/db.py:294` (unsubscribe), `store/outbox.py:205`
  (attempt counters on success — the outbox has **no** eviction and no max depth, which is correct),
  `detect/__main__.py` (`del img` to free memory), `plan/revisit.py`, `api/live.py` (websocket client set).
* `track/tracker.py:256-257` prunes never-confirmed tracks. Labelled in the docstring, counted in
  `stats.tracks_pruned_unconfirmed`, and tracks are not records. Fine.
* `api/detect_fallback.py:135,146` — `OrderedDict.popitem(last=False)`, i.e. LRU eviction. I read it: bounded at
  `maxlen=300`, a cloud reply for an evicted frame is kept and counted as `late`, and this is a merge buffer,
  not the record log. **Not an R10 issue** — this was the "unbounded cache eviction" the brief asked me to hunt
  for and it is bounded and observable.
* the remaining hits are the `"cleared"` text pattern firing on prose that says the *opposite*
  (`coverage/__init__.py:8`, `coverage/accumulate.py:20`, `coverage/export.py:24`, `eval/calibration.py:15`,
  `api/coverage_feed.py:31,37`). That is a design flaw in the wording check, and it is what made
  `tests/test_api.py::test_this_lane_never_says_a_segment_is_cleared` fail at 23:57 (see "Transients").

**The store's R10 is real and structural** — SQLite triggers, not a convention:
`r10_records_no_delete`, `r10_versions_no_delete`, `r10_evidence_no_delete`, `r10_audit_no_delete`,
`r10_dismiss_needs_reason_ins/upd`, `r10_version_monotonic`, `r10_id_immutable`. Dropping the no-delete trigger
turns `tests/test_store.py` red immediately (M21).

---

### S11 — LOW. The terrain still tiles visibly at 45 m in the live capture

I looked at `flight_seed23_alt45_00183` (AGL 45.71 m, GSD 1.79 cm/px). The mud/silt material shows an obvious
repeating grid at roughly 8–9 m period across the whole frame. The flood water reads as a nearly featureless
cream plane with faint straight seams and one small patch of ripples in the corner — it does not read as water
at this altitude. `docs/TRACKER.md` already lists *"water reads tan rather than green-teal"* as open; the
tiling was supposedly addressed. Annotated preview:
`…\scratchpad\mid45.png` (I can regenerate; the source is `_artifacts/dataset/train_seed23/images/`).

Also: `_artifacts/dataset/early_check.png` (23:10, a discarded `sim_seed23_alt45` run) shows **both** sampled
frames as a flat featureless blue-grey with `[0]` labels — the "pure sky" failure mode from the QUALITY_GATE
table, in a file nobody flagged.

### S12 — LOW. House count disagrees between the tracker and the data

`docs/TRACKER.md` "Verified working (seen in a render)" says **73 Kerala houses**. `data/scene/settlement.json`
has **76**, and `damage.json` / `docs/lanes/scene_polish.md` both use 76. One of the two is wrong; the tracker's
is the one flagged as eyeball-verified.

### S13 — LOW. Two independent implementations of the §5.5 operating-threshold rule

`sightline/eval/detection.py::freeze_operating_threshold` and
`sightline/detect/threshold.py::choose_operating_threshold` each implement *"highest confidence at which recall
≥ 0.92"*, each with its own `TARGET_RECALL` and its own `OperatingPoint` class. Both are correct today (I
mutated both — M9/M10 and M29/M30 all went red), and their unreachable-target tie-breaks differ slightly
(`argmax(recall)` vs `max(recall, conf)`). Flagging the duplication before they drift.

### S14 — LOW, two nits

* `tests/test_detect.py:304`: `assert fake  # keep the helper referenced` — a function object is always truthy.
  Harmless and self-documented, but it is literally a test line that cannot fail.
* `docs/lanes/coverage_plan_eval.md` §2.1 says the ceiling table *"reproduces §5.3b exactly"* and then quotes
  **151** px for a prone body at 30 m where the doc's table says **150**. The code's 151.3 is right and the doc
  is rounded; the geo lane flagged its two doc mismatches explicitly (`KNOWN_DOC_DISCREPANCIES`) and the
  coverage lane did not. Worth one line in the module.

---

## Mutation testing: do the tests actually fail when the code is broken?

Method: byte-exact textual mutation of one source file, scoped `pytest` run, unconditional byte-exact restore
with a post-restore SHA-256 assertion. Harness:
`…\scratchpad\mutate.py`. **45 mutations, 43 caught (95.6 %).**

| # | Mutation | File | Test target | Result |
|---|---|---|---|---|
| M1 | `POD_MAX = 0.99` → `1.0` | coverage/calibrate.py | test_coverage | RED |
| M2 | `R_MAX = 0.97` → `0.90` | coverage/quality.py | test_coverage | RED |
| M3 | `K_MIN, K_MAX = 1.0, 3.0` → `0.1, 30.0` | coverage/calibrate.py | test_coverage | RED |
| M4 | ΔPOD `exp(-k·C)` → `exp(-2k·C)` | coverage/accumulate.py | test_coverage | RED |
| M5 | `eps_ce90_multiple 2.0` → `1.0` | dedup/cluster.py | test_dedup | RED |
| M6 | `TRAPPED_HALFLIFE_H 72` → `36` | triage/curves.py | test_triage | RED |
| M7 | `SURVIVAL_FLOOR 0.05` → `0.0` | triage/curves.py | test_triage | RED |
| M8 | `STRANDED_PLATEAU_H 24` → `2` | triage/curves.py | test_triage | RED |
| M9 | `TARGET_RECALL 0.92` → `0.80` | eval/detection.py | test_eval | RED |
| M10 | threshold rule: highest conf → lowest conf | eval/detection.py | test_eval | RED |
| M11 | domain-mix guard `>1` → `>99` | eval/slicing.py | test_eval | RED |
| M12 | `track_buffer_s 30` → `5` | track/config.py | test_track | RED |
| M13 | `min_hits 3` → `1` | track/config.py | test_track | RED |
| M14 | GNSS term dropped from the budget | geo/budget.py | test_geo | RED |
| M15 | spacing `(1 − overlap)` → `(1 + overlap)` | plan/patterns.py | test_plan | RED |
| M16 | revert bug 3.5 (zone prior per cell) | coverage/prior.py | test_coverage | RED |
| M17 | revert bug 3.1 (`PrCurve.at` bisection) | eval/detection.py | test_eval | RED |
| M18 | revert bug 1.2 (still-window = 0 s) | dedup/cluster.py | test_dedup | RED |
| M19 | `min_dz 0.1` → `0.0` (near-horizon rays) | geo/chain.py | test_geo | RED |
| M20 | remove geodesy gimbal-lock branch | common/geodesy.py | geo+ingest+track | RED |
| M21 | drop SQL `r10_records_no_delete` trigger | store/db.py | test_store | RED |
| **M22** | **disable R10 dismissal-reason INSERT trigger** | **store/db.py** | **test_store** | **GREEN — NOT CAUGHT** |
| M23 | `CE90_FACTOR 2.1460` → `2.5` | schemas.py | geo+dedup+eval | RED |
| M24 | recall-vs-pixels anchor `PX_HALF` ×2 | coverage/quality.py | test_coverage | RED |
| M25 | `ANIMAL_FACTOR 0.3` → `1.0` | triage/curves.py | test_triage | RED |
| **M26** | **revisit queue pop order reversed** | **plan/revisit.py** | **test_plan** | **GREEN — NOT CAUGHT** |
| M27 | altitude band edge `45` → `44` | eval/slicing.py | test_eval | RED |
| M28 | pixel-size bin edge `20` → `18` | eval/slicing.py | test_eval | RED |
| M29 | detect lane: highest conf → lowest conf | detect/threshold.py | test_detect | RED |
| M30 | detect lane `TARGET_RECALL` → `0.75` | detect/threshold.py | test_detect | RED |
| M31 | revert bug 1.1 (ingest gimbal lock) | ingest/spec.py | test_ingest | RED |
| M32 | revert bug 3.2 (empty pruned route not aborted) | plan/constraints.py | test_plan | RED |
| M33 | revert bug 3.6 (`uncovered_fraction` → 0) | plan/patterns.py | test_plan | RED |
| M34 | revert bug 3.4 (pre-dawn thermal inverted) | plan/segments.py | test_plan | RED |
| M35 | GeoJSON coords → `[lat, lon]` | schemas.py | test_export | RED |
| M36 | sim noise GNSS σ `2.5` → `5.0` | ingest/sim.py | test_ingest | RED |
| M37 | sim noise yaw **bias** removed | ingest/sim.py | test_ingest | RED |
| M38 | tile overlap `0.2` → `0.05` | detect/tiler.py | test_detect | RED |
| M39 | tile size `1024` → `640` | detect/tiler.py | test_detect | RED |
| M40 | track confidence `max` → `min` | dedup/cluster.py | test_dedup | RED |
| M41 | POD clamp bypassed on write | coverage/accumulate.py | test_coverage | RED |
| M42 | `dedup_radius_m` 2×CE90 → 1×CE90 | geo/budget.py | geo+dedup | RED |
| M43 | `weighted_mean` → unweighted macro mean | eval/slicing.py | test_eval | RED |
| M44 | `TRAPPED_PLATEAU_H 48` → `480` | triage/curves.py | test_triage | RED |
| M45 | confirmation window `2 s` → `60 s` | track/config.py | test_track | RED |

Every mutation was reverted. See "Tree state" below.

---

## Numeric claims I re-derived, and they hold

These are clean bills. Do not re-check them.

**Geolocation error budget vs §5.7 (`sightline/geo/budget.py`).** 19 of 21 doc cells reproduce at 1-decimal
rounding; the two that do not are named in `KNOWN_DOC_DISCREPANCIES` with the derivation — the lane reported
its own mismatches instead of tuning them away, which is exactly right. I re-derived the design cell by hand:
nadir, 60 m, consumer → √(0.6455² + 0.1172² + 2.5² + 0.2²) = **2.592 m**, doc says 2.6. Dominant-term reading
matches the doc's prose (nadir = GNSS-dominated; 45° at 100 m = heading-dominated).

**POD closed form (§5.3).** `k = −ln(1−q)/q` satisfies `1 − exp(−k·q) = q` and `1 − exp(−k·n·q) = 1 − (1−q)ⁿ`
to 1e-12 for q ∈ {0.10, 0.45, 0.9044}, n ∈ {1,2,3,7}. The `DEFAULT_K` table in the lane report re-derives
exactly: body 0.9044→2.5957, cluster 0.8923→2.4975, upright 0.6452→1.6061, head_only 0.4438→1.3218,
limb_only 0.2310→1.1371. Every entry carries `measured=False`.

**CE90 and the dedup radius (§5.6/§5.7).** `CE90_FACTOR = 2.1460` vs `√(−2 ln 0.1) = 2.145966`. 2 × CE90 at
h_acc 2.6 m = **11.159 m** (report claims 11.16). The chain's own 60 m nadir h_acc of 2.592 m gives 11.13 m.

**§5.3b altitude ceilings, verbatim from the doc table.** `altitude_ceiling_table(M3T_WIDE_4K)` =
120 / 120 / 60 / 60 / 33.3 / 24.0 / 0 m; `M3T_THERMAL_640` = 64.5 / 56.9 / 17.1 / 17.1 / 9.5 / 6.8 m.

**Recall-vs-pixels anchors (§2.5 / Appendix E).** `recall_from_px`: 20 → 0.700, 60 → 0.900, 10 → 0.471,
150 → 0.951, ∞ → `R_MAX` 0.97.

**Survival decay (§5.8).** `survival_weight("trapped", t)` = 1.0 at t < 48 h, 0.5 at 120 h, 0.25 at 192 h,
0.05 at 2000 h; `"stranded"` = 1.0 at t < 24 h, 0.5 at 120 h, 0.05 floor. Matches
`max(0.05, 1 if t<plateau else 0.5**((t−plateau)/halflife))` computed by hand at every point. Animal floor
0.05 × 0.3 = 0.015, never zero.

**Operating-threshold rule (§5.5).** Both implementations pick the highest swept confidence whose recall clears
0.92, and both report `achieved=False` with the recall actually reached rather than lowering the bar.

**Scene JSON internal consistency.** `actors.json` counts (71 total / 69 detectable / 2 buried / by_pose /
by_zone / by_submersion / by_occlusion / 7 groups) all recompute exactly from the `actors` array.
`props_layout.json` 130+273+260+30+226 = 919 = total, floating 884 + grounded 35 = 919.
`rubble_layout.json` by_group and by_family both sum to 1799 = items; 2,407,920 tris ≤ 2,500,000 budget.
`damage.json` by_variant = 18, by_archetype 3+3+4+5+3 = 18 over 10+13+13+24+16 = 76, by_zone 17+1 = 18.

**Generator determinism.** `gen_actors.build(23)`, `gen_props.build(31)` and
`gen_vegetation.build(67, 4600, 110.0, True)` each produce identical output on two consecutive in-process runs.
`gen_props` and `gen_vegetation` also match their on-disk JSON byte-for-byte (`gen_vegetation` after dropping
the `_cover_png` ndarray that `write_preview` consumes before the dump). Only `gen_actors` disagrees with disk,
and that is S2 — staleness, not nondeterminism.

**torch / ultralytics discipline.** No module-level import of `torch`, `ultralytics` or `tensorrt` anywhere in
`sightline/`. Every use is function-local (`detect/export.py:88`, `detect/rgb.py:168`, `detect/train.py:186`,
`detect/verifier.py:248`, `track/backends.py:247`), and `tests/test_track.py` asserts `torch` never enters
`sys.modules`.

**Stubs are labelled where they exist.** `StubDetector` reports `is_stub: True` in every `/detect` response and
in `/healthz`, and `tests/test_api.py:431` asserts it. `coverage/calibrate.py` stamps
`"PLACEHOLDER until F19 fits it"` on every `DEFAULT_K` basis string and carries `k_is_measured: false` into the
export manifest. `ingest/decode.py:185` labels `NvdecReader` as written-from-docs-never-executed and
`open_video()` will not select it without an opt-in. `track/backends.py:277` raises `NotImplementedError` with
the AGPL/torch reason. I found **no stub presented as working**.

**Full suite.** `664 passed, 9 warnings in 13.22s` at 00:03 (`--ignore=test_stack.py --ignore=test_mcp_protocol.py`).

---

## Transients — the suite was red 6 minutes before it was green

At 23:57 `tests/test_api.py` had **3 failures**; at 00:02 the same file passed 33/33. The API lane was editing
while I ran. For the record, in case they recur:

* `test_this_lane_never_says_a_segment_is_cleared` — the lane's own wording scan fired on prose that says the
  opposite (`sightline/api/coverage_feed.py:37: R10: nothing here can mark a cell "cleared"` and
  `app/map/index.html:21`). Same design flaw as S10.
* `test_records_geojson_is_rfc7946_ranked_…` — `assert 6.22 == 6.2234 ± 6.2e-04`: `ce90_m` was rounded to 2 dp
  on the wire while the test asserted `2.1460 × h_acc` at `rel=1e-4`.
* `test_the_queue_rises_offline_drains_on_reconnect_…` — `assert 9 == 17` on applied outbox jobs.

I also lost ~35 minutes to a whole-suite run that produced no output and had to be abandoned; the same command
completed in 13 s later. Probably file contention with the lanes editing `sightline/api/` and `sightline/detect/`.

---

## Claims I could NOT verify, and what it would take

| Claim | Why not | What is needed |
|---|---|---|
| Everything in `docs/lanes/scene_polish.md` §A/§B/§C (anti-tiling material, tide line, damage variants, foam) | All three are editor scripts; the lane itself says they were only dry-run against a mock `unreal` module. Nothing has been rendered since 22:46 (S5). | `ue_python exec build_roof_materials.py / build_damage.py / build_foam.py`, then `qa_shots.py` and **open the PNGs** |
| Whether the 32 canopy-covered survivors (S3) are actually occluded in pixels | Needs a render / segmentation mask | one nadir capture over the settlement + `labels_from_mask` (after S1 is fixed) |
| Whether `build_actors.py` has been re-run since the stale `actors.json` (S2) | Needs the live editor's actor transforms | `ue_python` query of a `Human_###` Z against `poses.json` |
| The vegetation/rubble/utilities triangle and VRAM budgets | Host-side JSON only; the meshes are not imported yet | after `build_vegetation.py` etc., `stat unit` / Nanite stats in PIE |
| Thermal (F9b) | `tools/capture/thermal_ids.py` never run; TRACKER already records the Infrared-pass-all-grey finding | the Annotation-layer investigation already in the tracker |
| `_artifacts/dataset/train_seed23/data_card.json` | Not written yet — `survey.py` writes it at the end and the flight is in progress | wait for the run (but see S1: discard it) |
| `sightline/coverage/zone_raster_from_scene()` | Untested by design, and it reads `flood_valley_zones.png` which the scene lane is regenerating | pin the zone raster, then test |
| Any `mcp__sightline__*` / `mcp__unreal__*` behaviour | Forbidden to this lane | orchestrator |

---

## Tree state

`git status --short` and `git diff` show **no modification attributable to this audit**. Every mutated file was
restored byte-for-byte and re-checked against `HEAD`:

```
OK  coverage/calibrate.py  coverage/quality.py  coverage/accumulate.py  coverage/prior.py
OK  dedup/cluster.py  triage/curves.py  eval/detection.py  eval/slicing.py  track/config.py
OK  geo/budget.py  geo/chain.py  plan/patterns.py  plan/constraints.py  plan/segments.py
OK  plan/revisit.py  schemas.py  store/db.py  common/geodesy.py  ingest/spec.py  ingest/sim.py
OK  export/geojson.py
```

One process note worth passing on: my first harness used `Path.read_text()`/`write_text()`, which silently
converted `sightline/coverage/calibrate.py` from LF to CRLF (7881 → 8056 bytes). `core.autocrlf=true` means
`git diff` showed **nothing**, so the corruption was invisible to git. I caught it with a SHA-256 assertion,
restored the file to the exact `HEAD` blob bytes, and switched the harness to `read_bytes`/`write_bytes`.
**Any agent rewriting a source file on this repo should use bytes, not text** — `git status` will not warn you.

The only files modified in the working tree are other lanes' live work: `sightline/api/*`, `sightline/detect/*`,
`tools/capture/validate.py`, `tools/scene/import_assets.py`, `app/map/*`, plus the untracked scene data.

---

## What I would do first, in order

1. **Stop the capture.** Fix `tools/capture/labels.py::actor_index` (S1), add the per-frame size bound to
   `validate.py` check F, and re-fly. Nothing downstream is worth anything until the boxes are real.
2. **`uv run python tools/scene/gen_actors.py` + `build_actors.py`** (S2). One command, 23 survivors.
3. **Fix `gsd_cm_px` and the data card altitude** in `sightline/mission/survey.py` (S4) — two lines, and do it
   before the re-fly so the new run is honest.
4. **Render and look** (S5). `qa_shots.py` then `assert_qa_fresh.py` green, and actually open the images —
   4362 trees, 1799 rubble items and 60 poles have never been seen.
5. **Re-derive `occlusion` / `aerially_detectable` from a mask** rather than from a 3 m trunk clearance (S3),
   and raise the burial caps until Human_062 is invisible at 25 m (S6).
6. The two weak tests (S7) and the NaN row (S8) are cheap and can wait for a quiet moment.
