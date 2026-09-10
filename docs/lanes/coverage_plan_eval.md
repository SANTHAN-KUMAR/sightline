# Lane report — coverage (F16/F16b), plan (F2/F2b), eval (F19)

Written 2026-09-10 by the B6+B7 verification agent. Scope: `sightline/coverage/`, `sightline/plan/`,
`sightline/eval/`, plus `tests/test_coverage.py`, `tests/test_plan.py`, `tests/test_eval.py`.

The three modules already existed and had **no tests**. This pass did not rewrite them: it read every module,
wrote tests that could fail, ran them, and fixed the module wherever a test exposed a real defect. Six genuine
bugs were found and fixed. Nothing was stubbed to make a test pass, and no test was weakened to make the code
pass; the places where the code is honestly a model rather than a measurement are listed in §6.

---

## 1. Test result

```
D:\Tools\uv\uv.exe run pytest tests/test_coverage.py tests/test_plan.py tests/test_eval.py -q
...
101 passed, 4 warnings in 2.55s
```

Per file: `test_coverage.py` **36 passed**, `test_plan.py` **31 passed**, `test_eval.py` **34 passed**.
(The four warnings are a `PendingDeprecationWarning` from rasterio's own `transform.from_origin`, not ours.)

All three files are offline: no editor, no AirSim, no GPU, no `torch` / `ultralytics` / `tensorrt`, no
`fiftyone`. The largest grid any test builds is 64 x 64 cells. `ruff check` is clean on all three.

The end-to-end demo still runs and writes a full report from synthetic data:

```
D:\Tools\uv\uv.exe run python -m sightline.eval --demo
{ "report": "D:\\Sightline\\_artifacts\\eval\\demo_synthetic.md", "rows": 170,
  "operating_conf": 0.0532, "operating_recall": 0.7767, "target_met": false, ... }
```

---

## 2. What is now verified

### 2.1 Coverage (F16 / F16b) — 36 tests

| Property (doc reference) | How it is checked |
|---|---|
| Nadir footprint is Appendix B | A camera with fx = fy = 1200 px at 60 m AGL must give a 60 x 30 m footprint on exact 5 m cell boundaries; area, GSD = h/f, off-nadir 0, horizon clipping and the "camera below ground" reject are all asserted. `gimbal_quat` round-trips through the frozen `Telemetry.gimbal_pitch_deg` to 1e-9. |
| **One nadir pass raises POD in exactly the covered cells and nowhere else** | That footprint lands on rows 14..25 x cols 17..22 = **72 cells**. `coverage > 0` and `pod > 0` are asserted to be *exactly* that boolean mask; outside it both are 0.0. A second test re-derives the mask by independently rasterising the projected footprint polygon (including at gimbal yaw 35°) and requires the two to be identical. |
| **POD = 1 − exp(−k·C) composes over repeated passes** | Six identical passes at a low-quality look (RGB at night, so the clamp is never reached): `C` is asserted to be exactly `n·q` and POD exactly `1 − exp(−k·n·q)` at each step, plus strictly increasing with a strictly shrinking increment. |
| The closed form is the doc's, by hand | `k = −ln(1−q)/q` is checked to satisfy both `1 − exp(−k·q) = q` and `1 − exp(−k·n·q) = 1 − (1−q)ⁿ` for q ∈ {0.10, 0.45, 0.9044}, n ∈ {1,2,3,7} — i.e. n passes read as n independent looks. |
| **Saturates towards 1 without reaching it** | 200 passes over one cell: `pod.max() == POD_MAX == 0.99` exactly, `(pod < 1.0).all()`, while `coverage` keeps growing past 50 — only the display saturates. |
| Frame rate cannot inflate coverage | Ten frames of the same ground **in one pass** give bit-identical `C` to one frame; ten separate **passes** give exactly 10x. |
| **A burial cell keeps its floor POD under unlimited effort (§2.7, R10)** | A 16-cell burial polygon under 50 passes: `pod[buried].max()` is 0.0 after *every* pass, `coverage[buried].max() == 0.0`, `pod_effective` 0.0, visibility 0.0, and even a consumer that calls `CoverageGrid.recompute_pod()` itself off the frozen schema type still sees 0 (because C is 0, not because of a display rule). Meanwhile the surrounding cells the same frames covered rose above 0.5, so the mask is doing the work. |
| Marking is retroactive | A polygon drawn *after* the flight retracts the effort already claimed there (`pod` and `coverage` go back to 0). |
| The `buried` layer is zero by construction | Its `V(cell, j)` is 0 everywhere and `set_visibility` cannot switch it back on; `analytic_recall("buried", …) == 0.0`. |
| **There is no "cleared" state in the lane** | `tokenize` scan of all 16 `.py` files in `sightline/coverage` + `sightline/plan`: no identifier anywhere contains `cleared`, and every identifier containing `clear` is checked against an explicit allowlist (`cannot_clear`, `clearable_cells`, `clip_polygon`, …). |
| **Per-presentation layers are independent** | After one 60 m pass with the sim 4K camera, `body` coverage is > 3x `limb_only`, the two arrays do not share memory, each carries its own `k`, and zeroing one layer's visibility stops that layer accruing while the other keeps rising. |
| The doc's own tables come back out | `altitude_ceiling_table(M3T_WIDE_4K)` reproduces §5.3b exactly: **120 / 120 / 60 / 60 / 33 / 24 / 0 m**, geometric body ceiling **227 m**; the 640x512 thermal column gives 64 / 17 / 7 m; pixels-on-target for a prone body at 30/60/90/120 m are **151 / 76 / 50 / 38 px** and for a limb **16 / 8 / 5 / 4 px**. |
| The recall model sits on its published anchors | `recall_from_px(20) == 0.70`, `(60) == 0.90`, `(10) ≈ 0.47`, `(150) ≈ 0.95`, saturating at `R_MAX = 0.97`; the array form is identical to the scalar form to 1e-12; the sub-8 px taper is real. Every value is flagged `measured=False` with "NOT a measured slice" in its basis. |
| Condition factors move the right way | night RGB < 0.2x day RGB; night thermal > 5x night RGB; the 06:50 crossover window costs the thermal band more than half; fog < heavy rain < dry; a 1/60 s dusk exposure loses > 40 % to motion blur. |
| A measured slice overrides the model only when it is thick enough | n = 100 overrides (per-cell array rescaled to the measured level); n = 5 (below `min_n = 30`) falls back to the labelled model; a measured recall of 1.0 is still clamped to `R_MAX`. |
| Mixture and `POD_eff` | Every zone mixture sums to 1 (including a two-layer MVP renormalisation); `pod_effective` equals the hand-computed `Σ w_j POD_j` and lies strictly between the two layers everywhere; the Dirichlet update re-weights a fan towards limb-only without mutating `ZONE_MIX`. |
| ΔPOD (the planner's currency) | Matches `e^(−kC) − e^(−k(C+q))` exactly; after saturating a cell the increment is < 1 % of the first pass's; it is 0 inside burial cells for both the single-layer and mixture forms. |
| k calibration | `calibrate_k_from_outcomes` recovers a planted k = 2.0 from 6000 synthetic outcomes to ±0.12, and clamps to `K_MIN`/`K_MAX` when the data say "found nothing" / "found everything". The reliability diagram gives ECE < 0.05 on an honest map and > 0.15 on one that finds half of what it promises. |
| Prior + Bayes (Appendix B) | The prior sums to 1; an LKP contributes exactly its stated mass and dominates its neighbourhood; flooded buildings are weighted by roof area x storeys x depth and dry ones are excluded; a channel bend outranks a straight reach; burial cells sit at the floor; after a pass with POD 0.99 the searched cell's probability drops by exactly 1/(1−POD) relative to its neighbours and **never reaches zero**, even at POD = 1. |
| Export contract | See §5 below — file set, manifest keys, `[lon, lat]` order, north-up row order, alpha 0 on burial cells, `domain: "sim"` on every statistic. |

### 2.2 Plan (F2 / F2b) — 31 tests

| Property | How it is checked |
|---|---|
| **Line spacing is derived, not hard-coded** | For five altitudes the chain `swath = 2h·tan(HFOV/2)` → `GSD = swath / W_px` → `spacing = sweep x (1 − overlap)` → `v ≤ blur·GSD/t_exp` is re-derived from the camera and compared. Doubling the altitude doubles the spacing; a different camera at the same altitude gives a different spacing; the value is asserted not to be any round constant. |
| Sweep width knows what it is searching for | At 60 m the sim 4K camera's sweep for a body is the full swath and for `limb_only` / `head_only` it is **0.0** (§5.3b: nearly blind); descending to 18 m makes the limb sweep positive but narrower than the raw swath. |
| **Covers the polygon with the intended overlap and no gaps** | For 4 headings x 3 (altitude, overlap) pairs: realised spacing ≤ nominal and realised overlap ≥ requested (never widened to a rounder number); the swath strips are shown to tile the across-track extent with `max(diff(centres)) ≤ W`; and an area-sampling gap check (`uncovered_fraction`, new) confirms no sampled point of the polygon is further than W/2 from a flown **line segment**. Flown along the polygon's own long axis — the router's default — the uncovered fraction is exactly 0.0. |
| The gap check is falsifiable | On a concave "plus" segment at a 200 m sweep it reports 9 % uncovered with a 10 m+ worst gap and `covers_polygon = False`, and the route carries a "measured gap" note; at a 120 m sweep the same polygon is clean. |
| Expanding square is correct | Leg lengths are exactly `s, s, 2s, 2s, 3s, 3s, 4s, 4s` with headings `0, 90, 180, 270, …`, where `s` is the same derived line spacing the lawnmower uses; `max_radius_m` really stops it and says so. |
| **Battery constraint can abort or shorten** | A 300 s battery truncates an 8-waypoint pattern, sets `aborted = True` with the arithmetic in `abort_reason`, appends an RTL leg, and the kept prefix is verified affordable (`elapsed + time_home ≤ usable`). `apply()` does not mutate its input. |
| **Geofence / no-go constraints can prune and abort** | A partial fence drops the outside waypoints (every survivor is inside the fence) and notes it; a fence containing nothing now sets `aborted` with "nothing left to fly" (this was a bug — §3.2); an operator no-go area prunes what falls inside it, and `Constraints()` creates no automatic buffer. `clip_polygon` shortens the pattern instead of shredding it. |
| Ceiling and floor | Altitudes are clamped into [15 m, 120 m], noted per route and per waypoint, and the ASL height moves with the clamp so AGL/ASL stay consistent. |
| Koopman allocation is the water-filling solution | `c_i = max(ln p_i − λ, 0)` is re-derived from the returned λ and matched to 1e-4; the budget is met; cells below `e^λ` get exactly 0; more prior never means less effort; a **flat prior gives an exactly flat allocation**; burial cells get nothing. |
| **The planner degrades to the plain pattern when the prior is flat** | With a flat POA, all four auto-segments carry identical mass, so the tie is broken by transit and the nearest segment wins. The chosen route is then compared **waypoint by waypoint** against `patterns.boustrophedon_route` built independently — same coordinates, same AGL, same per-waypoint `reason` string, same `geometry` dict. The only difference is the RTL leg the constraints append. There is no fallback branch to test because there isn't one. |
| …and follows the prior as soon as it is not flat | With mass concentrated on the *farthest* segment, the planner abandons the nearest one for it. |
| Every candidate is explainable | Survey / orbit / RTL candidates all present; every candidate has a finite value, a finite gain and a non-empty `reason`, or an `infeasible_reason`; the feasible set is sorted descending; the chosen action's reason is copied into `route.params["decision_reason"]`; `Decision.as_dict()` is JSON-serialisable (this was a bug — §3.3). |
| It declines to re-fly | Saturating the coverage drops the best candidate's value from > 1e-4 to < 1e-6 and the explanation says "every candidate scores about zero, so re-flying adds nothing measurable" — while still proposing an action, never closing the segment. |
| It is never paid for a burial polygon | Survey candidates over a fully-buried segment have gain exactly 0.0 while the others are positive. |
| Infeasible actions are pruned before scoring | Above-ceiling candidates carry "above the 120 m ceiling"; with a 60 s battery every survey is infeasible and RTL is chosen. |
| Loiter-until-dawn and descend-to-confirm | The loiter candidate waits exactly 3.0 h to reach 05:00 and explains itself; the confirmation orbit is at 25 m and buys > 5x the limb-only quality of a 90 m survey (at which altitude no survey line would even be planned). |
| Segments and assignment records | Auto-segments rank by prior mass and export as closed `[lon, lat]` rings with `domain: "sim"`; the ICS-204 record produces the §5.3b sentence with a **derived** altitude (22 m, this camera's limb-only ceiling); a burial segment says "aerial search cannot clear it" and points at radar/canine; `recommend()` is checked against a banned-word list (`cleared`, `complete`, `finished`, `done`, `no further`) across three POD regimes. |
| Revisit queue | Ordered by residual `POS = POA x (1 − POD)`, excludes burial cells (verified by mapping every queued item back to its grid cell), queues stale records, and **neither flies nor deletes a dismissed record** — which stays in the log with its reason (R10). |

### 2.3 Eval (F19) — 34 tests

| Property | How it is checked |
|---|---|
| **A perfect prediction set scores recall 1.0** | 12 exactly-coincident boxes at IoU 0.5 and 0.25: tp = 12, fp = 0, fn = 0, recall = precision = F1 = 1.0, FP/min = 0, every matched IoU = 1.0. |
| **Constructed misses score the exact recall at IoU 0.5 AND 0.25** | Four targets; predictions at IoU 1.00, 0.35, 0.10 and none. The constructed IoUs are first verified against `iou_matrix` to 1e-9. Result: `(tp, fn, fp) = (1, 3, 2)` and **recall 0.25** at IoU 0.5; `(2, 2, 1)` and **recall 0.50** at IoU 0.25. |
| Matching follows §5.12 and §6.3 | Greedy in descending score (swapping two scores moves the true positive); `uncertain` / `ignore` / `is_group` predictions are dropped rather than counted as FP; a group is one recall target satisfied at IoD ≥ 0.5 and is an FN when nothing lands on it. |
| FP/min is the documented ratio | 10 frames at 5 fps with 3 unmatched predictions gives exactly **90 FP/min** = 3 / (10/5/60). A zero `fps_processed` raises rather than dividing. |
| **The sweep picks the highest confidence at which recall ≥ 0.92 and freezes it** | A PR curve constructed to fall by exactly 1/25 per 0.01 of confidence. 0.92 x 25 = 23 targets, so the answer is the 23rd-highest score, **0.77**, with recall exactly 0.92 — and one notch higher (0.78) is verified to miss the target. `freeze_on_validation` records `frozen_on = "pr[val]"`, and the four `operating_*` rows carry it plus "measured on the split the threshold was frozen on, not the test clip". |
| The threshold cannot be tuned on test by accident | `freeze_on_validation` raises on a non-`val` split unless the caller opts in, and records the opt-in. |
| An unreachable target is a finding | With only 20 of 25 findable, `achieved = False`, `target_recall` stays 0.92, and the note says "Fly lower or retrain … rather than moving the target". |
| max-F1 is reported but is not the operating point | The max-F1 confidence (0.75) is asserted to differ from the frozen one. |
| **Slicing partitions without double counting** | For all seven axes: no bin label repeats, `Σ n(bins) + slice_unbinned_gt == n(all scored gt boxes)` exactly, and the unbinned row carries its own count. Box axes (occlusion / posture / pixel size) emit recall only; frame axes also emit precision and FP/min. Pooling one axis's bins with `combine_rows` reproduces the whole-clip recall to 1e-9. |
| **Combining a `sim` row with a `real` row RAISES** | `combine_rows` (all three aggregations), `require_single_domain`, `ratio_row` and `MetricSet.aggregate` all raise `DomainMixError`; the two rows may still sit in one `MetricSet` and be split by `by_domain()`. Two `sim` rows pool correctly by sample size (0.94 @ n=500 with 0.50 @ n=500 → 0.72 @ n=1000). |
| The API makes a bare float impossible | Nine public entry points (`headline_rows`, `slice_rows`, `evaluate_records`, `r10_rows`, `evaluate_geolocation`, `evaluate_pod_calibration`, `attribute_accuracy_rows`, `rank_correlation_rows`, `promotion_audit_rows`) are asserted to return a non-empty `MetricSet` of `MetricRow`, each with a valid `SliceKey.domain` and a finite value. `make_slice()` without a domain is a `TypeError`; a NaN metric raises. |
| **Duplicate rate is right on a constructed duplicate** | 3 survivors + 3 correct records + 1 duplicate 3.6 m from its source: record recall 1.0, **precision 0.75**, **duplicate rate 1/3**, record FP/min 1.0, count MAE 0. Two duplicates → 2/3. Adding a record 7 km away → the literal §5.6 rate goes to 3/3 while `record_duplicate_rate_near` stays at 2/3, which is why both are reported. |
| Buried survivors are excluded and reported | Recall's denominator is the findable survivors only; `buried_survivors_excluded` carries the count and the R10 guardrail text. |
| R10 audit | Retained-record count and dismissals-without-a-reason are both emitted as rows. |
| Hungarian | Matches `scipy.optimize.linear_sum_assignment` in total cost on five shapes including rectangular ones, refuses `inf`, and the gate drops far pairs. |
| Geolocation | Records placed at exactly 1/3/5/11 m north of the truth give median 4.0 m and max 11.0 m; published CE90 = 2.146 x h_acc; containment is exactly 0.75 (three of four inside a 6.0 m circle). Reporting a sim number with `noise_injected=False` carries "not a claim"; nothing matched is reported as n = 0, never as zero error. |
| POD reliability (§5.12) | An honest generator gives ECE < 0.05 and |bias| < 0.05; a map that finds half of what it promises gives ECE > 0.15 and a **positive** bias with "PROMISED more detection than it delivered". `pod_bins_on_diagonal` is averaged over eight seeds (each bin is a 95 % interval, so a perfect map still misses one occasionally) and must be ≥ 0.8, and strictly above the dishonest map's on every seed. |
| Tracking | The module docstring's hand computation is now a test: two tracks over four frames with one ID switch gives **DetA = 1, AssA = 0.75, HOTA = √0.75 = 0.8660254, ID switches = 1** from `motmetrics`. |
| Posture head (§5.5a) | A head that orders records perfectly against a flat baseline gives Spearman delta 1.0 and the verdict "keep the head on"; an inverted head gives the verdict "SWITCH THE HEAD OFF". The promotion audit catches the one record scored lower with the head than without and names it. Attribute accuracy is sliced by pixel-size bin and by class, and the bins partition the samples. |
| Harness order of operations | The threshold frozen on `val` is the one measured on `test` (recall exactly 0.92 at conf 0.77); a caller-supplied threshold with no validation curve is re-measured and says so. The whole-clip recall is re-derived by hand from the match result. |
| The report | Every metric table starts with a `domain` column and every data row under it starts with `| sim |` or `| real |`; a two-domain report says "Do not compute a combined figure from them"; every headline bullet carries "in simulation" or "on real footage". The JSON sidecar carries `domain` on every row. |
| FiftyOne bridge | The manifest is plain JSON (no fiftyone import), boxes are relative, `has_fp` / `has_fn` tags match the per-sample counts, and failure clusters are ordered by miss count. |
| Ground truth | Round-trips through JSON; §6.5 split-leakage guard catches the same seed group in two splits; `GtBox` rejects an inverted box, an out-of-vocabulary posture and an illegal occlusion bin. |
| The synthetic fixture itself | Deterministic for a seed; buried actors never appear as a visible box; every confirmed track has ≥ 3 observations (§5.6 rule 3). |

---

## 3. Bugs found and fixed

### 3.1 `PrCurve.at()` over-reported recall at any threshold not in the swept set — `sightline/eval/detection.py`

`at()` bisected to the **largest swept confidence ≤ conf**, which silently credits every prediction scoring
between that swept value and the requested threshold. Measured before the fix, on the constructed 25-target
curve:

```
at(0.775) -> recall 0.92     counts_at(0.775) -> recall 0.88     # over-reported by one target
at(1.5)   -> recall 0.04     counts_at(1.5)   -> recall 0.00     # above every score, nothing is kept
```

This matters because `harness.run_detection_eval` calls `curve.at(operating_conf)` to build the `OperatingPoint`
whenever the caller supplies its own threshold, so the `operating_recall` row could disagree with the
`recall@iou0.5` row in the same report. Fixed to take the **smallest swept confidence ≥ conf** (the set
`score >= conf` is identical to `score >= that value`) and to return `(0, 0, 0)` above the top of the curve.
`test_curve_lookup_agrees_with_counting_at_the_same_threshold` now checks 11 thresholds, on and between the
swept values, against `counts_at`.

### 3.2 A geofence that pruned every waypoint returned an empty route marked "not aborted" — `sightline/plan/constraints.py`

`Constraints.apply()` drops waypoints outside the geofence or inside an operator no-go area and notes the count,
but only battery truncation set `aborted` / `abort_reason`. A route with zero waypoints and `aborted = False`
reads to a caller as "flown, nothing to do". Fixed: when the constraints prune everything, the route is marked
aborted with `"nothing left to fly: all N waypoints were outside the geofence or inside an operator no-go
area"`, and the RTL leg is appended as it is for a battery abort. Partial pruning still reports through `notes`.

### 3.3 `Decision.as_dict()` could not be serialised — `sightline/plan/planner.py`

The decision timeline is a *logged* product ("the timeline shows: chose segment B at 0.031 expected finds per
minute over continuing swath 7 at 0.004"), but `Candidate.as_dict()` only filtered out numpy arrays, leaving
`Segment`, `PatternSpec`, `Route` and `PendingRecord` objects in `params`, so `json.dumps(decision.as_dict())`
raised `TypeError`. Added `_jsonable()`: segments reduce to their `seg_id`, pattern specs to a small dict of the
numbers that produced them, routes to `{pattern, n_waypoints}`, numpy scalars to Python floats, arrays are
dropped. Asserted by `test_every_candidate_carries_a_number_and_a_reason` and
`test_candidate_serialisation_drops_arrays_but_keeps_the_reason`.

### 3.4 The pre-dawn thermal recommendation was suppressed all day — `sightline/plan/segments.py`

```python
if local_hour is None or local_hour < window[0] or local_hour > CROSSOVER_WINDOWS_LOCAL_H[1][1]:
```

This emitted "a pre-dawn thermal pass … beats the crossover window" only *outside* 06:24–18:36 — i.e. at night,
and never during the working day, which is exactly when a commander plans the next pre-dawn pass. The guard was
clearly meant to suppress the advice while it is stale (inside a crossover window). Fixed to suppress it only
when `local_hour` is actually inside one of the two crossover windows.

### 3.5 The probability-of-area prior changed meaning with the raster resolution — `sightline/coverage/prior.py`

Buildings, channel bends and last-known positions are Gaussian bumps of a fixed total **mass**, so they are
resolution-invariant. `ZONE_BASE_WEIGHT` was applied as a flat weight **per cell**, so its share of the prior
scaled with the cell count: moving from the 10 m cells to the 5 m cells that §5.3 also allows quadrupled the
zone layer's weight and diluted every piece of actual evidence by 4x. Fixed by making the zone layer a weight
per unit *area* (`ZONE_WEIGHT_REFERENCE_CELL_M = 10.0`, scaled by `(cell_m / 10)²`), which leaves the documented
10 m numbers unchanged. `test_prior_does_not_change_with_the_raster_resolution` asserts the LKP's share of the
raw prior is the same at 10 m and 5 m to within 2 %.

### 3.6 `PatternGeometry.covers_polygon` was hard-coded `True` — `sightline/plan/patterns.py`

The field asserted a coverage property that was never checked, on a lane whose whole point is not to claim
search that did not happen. Added `uncovered_fraction(poly, lines, sweep_width)`: it samples the polygon on a
lattice at least 8 samples across the swath and measures the distance from each interior sample to the nearest
flown **line segment** (not the infinite line, so a clipped line gets no credit for ground it never overflew).
`covers_polygon` is now that fraction being zero, and `PatternGeometry` also reports `uncovered_fraction`,
`worst_gap_m` and `gap_sample_step_m`. `boustrophedon_route` appends a "measured gap" note when it is not zero.
The check is falsifiable: on a concave "plus" segment at a 200 m sweep it reports 9.1 % uncovered with a > 10 m
worst gap.

**Known residual, deliberately not designed away.** Lines are clipped to the polygon, so flying a rectangle
along a heading that is not one of its edges can leave a corner sliver beyond the last line's endpoint —
measured at **0.02 % of the area and 0.59 m past the swath edge** on a 400 x 300 m segment at heading 30°. The
standard fix is a headland overrun (fly past the boundary before turning), but that would put every line end
outside a geofence-clipped segment, where `Constraints.apply` drops it and shreds the route. The sliver is
therefore measured and reported rather than hidden, the tests bound it (< 0.1 % of area, < 1 m), and flying the
polygon's long axis — the router's default — has no gap at all. The coverage raster never claims the sliver
either, because it is built from real footprints, not from the pattern.

### Non-bugs worth recording

* `offset_ne` (uses `cos lat1`) and `ne_between` (uses `cos mean lat`) are different approximations, so a
  grid round-trip is off by ~0.1 mm per 130 m of easting. That is well inside `common/geodesy.py`'s own stated
  "< 1 cm" claim, but it means a point sitting *exactly* on a cell boundary can round either way. Tests use cell
  midpoints and assert the 1 cm bound. Nothing to fix in the lane; `common/` is orchestrator-owned.
* `Segment.mask` caches on shape only, so reusing one `Segment` against two different maps of the same shape but
  different origins would return a stale mask. Not reachable in the current flow; noted for the orchestrator.

---

## 4. How `k` is calibrated

**Today it is an honestly-labelled placeholder, and the code says so in three places.** The real fitting path is
implemented and tested; it simply has no data yet, because the detector (F8) does not exist.

* **Real path — `calibrate_k_from_outcomes(coverage, found)`.** Maximum likelihood for `POD = 1 − exp(−k·C)`
  over per-target `(accumulated coverage, was it found)` pairs from the held-out clip, by ternary search on a
  provably concave log-likelihood, bounded to `[K_MIN, K_MAX] = [1.0, 3.0]`. The floor is the random-search
  lower bound of §5.3 — the map may never claim *less* than random search. Verified: it recovers a planted
  k = 2.0 from 6000 synthetic outcomes to ±0.12, and saturates to the bounds on degenerate data.
  `reliability_diagram()` + `expected_calibration_error()` are the §5.12 check on the fit, one per presentation
  class, with Wilson intervals on the thin layers (§5.3b).
* **Shipped default — `DEFAULT_K`, derived, `measured = False`.** `k = −ln(1−q)/q` evaluated at the interim
  recall model's value `q` for the nominal mission slice (sim 4K camera, 60 m AGL, day, RGB, dry, 1/500 s). That
  choice makes one pass read back the detector's own recall and n passes compose as n independent looks. Every
  entry carries `PLACEHOLDER until F19 fits it` in its `basis`, `k_is_measured: false` travels into the export
  manifest, and `CoverageMap.summary()` repeats it per layer.

| presentation | q_ref (model) | k | measured |
|---|---|---|---|
| body / prone | 0.9044 | 2.5957 | no |
| cluster | 0.8923 | 2.4975 | no |
| upright / wading | 0.6452 | 1.6061 | no |
| head_only | 0.4438 | 1.3218 | no |
| limb_only | 0.2310 | 1.1371 | no |
| buried | 0.0 | 1.0 (irrelevant: the layer is zero) | no |

`POD_MAX = 0.99` is enforced on every write, so a saturated cell still carries a 1 % chance that someone is
there and was missed — the numeric form of "the system recommends, never closes".

---

## 5. The two output formats

### 5.1 Coverage raster export — `export_coverage(cmap, out_dir, stem="coverage")`

Writes, per presentation layer **plus** the mixture view `effective`:

```
coverage_<layer>.png        RGBA, one pixel per cell, IMAGE ROW 0 = NORTH (the array's row 0 is SOUTH).
                            Magma ramp over POD; alpha 0 wherever the cell is `cannot_clear`, so the map lane
                            draws its own hatch through the hole.
coverage_<layer>.geojson    FeatureCollection of POD band polygons (properties.band / pod_min / pod_max /
                            label / domain) plus one Feature per burial region with
                            properties.cannot_clear = true and label "aerial search cannot clear".
                            Coordinates are [lon, lat] per RFC 7946.
coverage.json               ONE manifest describing every layer.
```

The manifest (verified field by field in `test_export_contract`):

```json
{ "schema_version": "1.0.0", "product": "sightline.coverage", "domain": "sim",
  "grid":   {"origin_lat": 11.486096, "origin_lon": 76.144083, "cell_m": 10.0,
             "n_north": 20, "n_east": 20,
             "row_order": "image row 0 is NORTH; array row 0 is SOUTH"},
  "bounds": {"south": 11.486096, "west": 76.144083, "north": 11.487904, "east": 76.145917},
  "coordinates": [[west,north],[east,north],[east,south],[west,south]],   // MapLibre image source
  "ramp":   [{"pod": 0.0, "rgb": [13,8,60]}, ... {"pod": 0.95, "rgb": [252,222,125]}],
  "bands":  [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
  "cannot_clear_label": "aerial search cannot clear",
  "passes": [{"pass_id": 0, "frames": 1, "cells": 72, "mode": "AUTO"}],
  "layers": [{"layer": "body", "image_url": "coverage_body.png",
              "geojson_url": "coverage_body.geojson",
              "k": 2.5957, "k_is_measured": false, "k_basis": "derived: ... PLACEHOLDER until F19 fits it",
              "stats": {"mean_pod": ..., "max_pod": ..., "cells_pod_ge_0.5": ...,
                        "cells_pod_lt_0.2": ..., "clearable_cells": ..., "cannot_clear_cells": ...,
                        "mean_coverage": ..., "domain": "sim"}},
             {"layer": "limb_only", ...}, {"layer": "effective", "k": null, ...}],
  "legend_note": "POD is a probability of detection, never a cleared flag; it is clamped at 0.99 ..." }
```

`overlay_payload(cmap)` returns the identical manifest with each PNG inlined as a `data:image/png;base64,…` URI
for the WebSocket path. `domain: "sim"` appears on the manifest and again inside every layer's `stats` and every
GeoJSON feature, so a copied-out row still carries it.

### 5.2 Waypoint output — `Route.to_dict()` / `Route.to_json()` / `Route.to_geojson()`

```json
{ "schema_version": "1.0.0", "product": "sightline.plan.route", "domain": "sim",
  "pattern": "boustrophedon",            // boustrophedon | expanding_square | orbit | revisit | rtl | composite
  "params": {"camera": "sim RGB 4K", "agl_m": 55.0, "presentation": "body", "min_px": 20.0,
             "side_overlap": 0.25, "heading_deg": 0.0, "speed_ms": 8.0, "exposure_s": 0.002,
             "segment_id": "S03",
             "geometry": {"sweep_width_m": 85.171, "nominal_spacing_m": 63.878,
                          "actual_spacing_m": 32.414, "n_lines": 3,
                          "side_overlap_nominal": 0.25, "side_overlap_actual": 0.619,
                          "heading_deg": 0.0, "speed_limit_ms": 11.09, "gsd_m": 0.022,
                          "covers_polygon": true, "uncovered_fraction": 0.0,
                          "gap_sample_step_m": 5.323, "worst_gap_m": 0.0}},
  "frame":  {"centre_lat": 11.487, "centre_lon": 76.145, "base_z_m": 1046.007, "size_m": 2048.0,
             "axes": "north = UE +X, east = UE +Y",
             "ue_from_local": "X_cm = north_m*100, Y_cm = east_m*100, Z_cm = (alt_asl_m - base_z_m)*100"},
  "aborted": false, "abort_reason": "",
  "notes":  ["3 lines at 32.4 m (62 % overlap, requested 25 %), sweep width 85.2 m for 'body' at 55 m, ..."],
  "totals": {"n_waypoints": 6, "length_m": 664.83, "duration_s": 83.1},
  "waypoints": [
    { "seq": 0, "action": "goto",              // goto | orbit | loiter | hold | rtl | land
      "north_m": -100.0, "east_m": -32.41445,  // SCENE frame: metres north/east of the map centre
      "alt_asl_m": 1101.007, "agl_m": 55.0,
      "lat": 11.486096, "lon": 76.144703,      // WGS-84, lat/lon order
      "ue_cm": [-10000.0, -3241.445, 5500.0],  // the simulator's own units, via flood_valley.json ue_import
      "speed_ms": 8.0, "gimbal_pitch_deg": -90.0, "yaw_deg": null,   // null = face the direction of travel
      "dwell_s": 0.0, "orbit_radius_m": 0.0,
      "segment_id": "S03", "pass_id": 2,
      "reason": "boustrophedon line 1/3 start, sweep 85.2 m, spacing 32.4 m, gimbal yaw 90 deg" }, ... ] }
```

Verified in `test_waypoint_output_format`: the key set is exactly the contract, `seq` is dense, `alt_asl_m`
equals terrain + `agl_m`, `(lat, lon)` equals `scene.to_latlon(north_m, east_m)`, and `ue_cm` equals
`[north*100, east*100, (alt_asl − base_z)*100]` for **every** waypoint, not just the first. `to_geojson()` emits
a LineString of the track plus one Point per waypoint, in `[lon, lat, alt]` order.

Gimbal yaw is deliberately **not** a waypoint field: `patterns.gimbal_yaw_for_heading(heading) = heading + 90`
puts the wide axis of the frame across track, and it is stated in each waypoint's `reason` instead, because the
vehicle yaw and the gimbal yaw are different actuators.

---

## 6. What is still a model, a placeholder or a stub

Nothing in these three lanes is a shallow proxy for something that exists; these are the places where the real
input does not exist yet, all labelled in code:

1. **`R_slice` is an analytic model, not a measurement** (`coverage/quality.py`). The detector (F8) does not
   exist, so every `q_pass` today comes from `analytic_recall()` — a pixels-on-target logistic anchored on §2.5
   and Appendix E, times documented condition factors. Every value carries `measured=False` and the basis string
   "analytic model (§2.5 anchors) — NOT a measured slice". `SliceTable` is the real path and is tested: as soon
   as F19 emits measured recalls with n ≥ 30, they override the model per slice. **Consequence: every POD number
   the map can print today is model-driven and must be described that way.**
2. **`k` is derived, not calibrated** — §4 above. The MLE fitter and the reliability diagram are implemented and
   tested and are waiting for held-out outcomes.
3. **`ZONE_MIX` (the presentation mixture per zone) is a proposed modelling choice**, apportioned from §6.2's
   placement distribution as the §5.3b prose describes. `update_mix_from_observations()` is the path by which
   the verifier head (F10) replaces it at run time.
4. **The prior weights are proposed inputs, not measurements** (`coverage/prior.py`): zone base weights,
   `BUILDING_WEIGHT_PER_M2`, `HIGH_GROUND_WEIGHT`, `LKP_WEIGHT`, `BEND_WEIGHT`. They are anchored on §2.4's
   evidence and are meant to be commander-overridable.
5. **`V(cell, j)`, the visibility factor, defaults to 1.0 everywhere** except burial polygons and the `buried`
   layer. A real canopy/structure occlusion raster would come from the land-cover layer; `set_visibility()` is
   the entry point and is tested.
6. **`zone_raster_from_scene()` is untested here by design.** It resamples `data/scene/flood_valley_zones.png`,
   which the scene lane is regenerating during this session; a test against it would be a test of another lane's
   moving output. The prior and mixture paths it feeds are tested with synthetic zone rasters instead.
7. **Fields2Cover is not used** (it does not build on Windows — HANDBOOK §6); the doc's own fallback
   ("the boustrophedon generator is ~60 lines and is written by the team") is what `plan/patterns.py` is. Turn
   planning and headland generation are therefore absent: turns are implicit between line ends, and the corner
   sliver of §3.6 is the visible consequence.
8. **`footprint.py` deliberately duplicates the ray-plane step** rather than importing `sightline/geo/`
   (cross-lane imports are forbidden by `docs/CONTRACTS.md`). Both sides use
   `common.geodesy.quat_to_rot`, so they cannot drift on the quaternion convention. If the geo lane later
   exports a `project_pixel()` with a matching signature, these functions should collapse onto it.
9. **The planner's `_gain_over` applies one `q` uniformly over a whole segment** rather than integrating the
   actual per-frame footprints the route would fly. It is the §5.3a formula as written, and it is the right
   approximation for ranking ~20 candidate actions in milliseconds, but it is an approximation and the coverage
   raster (which does integrate real footprints) is the authority afterwards.
10. **No real detector, tracker or record stream has been through the eval harness.** Every number in
    `tests/test_eval.py` and in `_artifacts/eval/demo_synthetic.md` comes from `eval/synthetic.py`. The harness
    is proven correct; it has not yet been proven *useful*, which needs F8 and the captured dataset.

## 7. What the other lanes should know

* The eval lane's public API returns `MetricSet`/`MetricRow` only. If your lane needs a number, take
  `row.value` **with** `row.slice` — `MetricSet.get(name, domain)` is the accessor, and `overall(domain)` first
  when a name also has per-slice rows.
* `sightline.coverage` needs `PIL` (PNG), `rasterio` (band polygons) and optionally `shapely` (line clipping)
  and `scipy` (connected components). All are already pinned; the rasterio/shapely/scipy paths have documented
  pure-python fallbacks that are not exercised.
* No package was added, `pyproject.toml` / `uv.lock` / `schemas.py` / `common/` / `tools/**` were not touched,
  and no state-changing git command was run.
