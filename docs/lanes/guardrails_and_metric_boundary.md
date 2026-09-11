# Lane: F21 guardrails + the metric boundary — session 5, second session

Owner: the second Claude Code session, 2026-09-11, running as orchestrator over a code writer and a skeptical
red-team verifier per item. Lane split and rules: `docs/lanes/COORDINATION.md`. Sibling lane:
`docs/lanes/eval_report.md` (F19).

**Nothing here touched the editor, PIE, AirSim, the GPU, or git.** No writes to `tools/capture/*`,
`tools/scene/*`, `tools/train/*`, `sightline/mission/*`, `sightline/pipeline.py`, `sim/**`, `data/scene/*`, or
`tests/test_plan.py` (the orchestrator session edited that at 02:34 for its coverage-geometry fix).

---

## 1. Why these two, and not other features

The `F1`–`F21` table in `docs/TRACKER.md` is badly stale — `F7`, `F11`, `F12`, `F13`, `F14`, `F15`, `F18` and
`F19` all read `[ ]` and all exist with passing tests. Checking the tree instead of the checkboxes, and
removing everything the orchestrator session claimed, everything needing PIE/AirSim/GPU, and everything
already built, leaves three genuinely open items:

| | status |
|---|---|
| **F21 guardrails** | the R10 source scanner guarded **2 of 16 paths**. Taken. |
| **AUDIT S8** | non-finite metrics escaped the API and made the JSON RFC-8259 invalid. Taken (bundled: same promise — no number may lie). |
| **F6 X-AnyLabeling round-trip** | genuinely open; the orchestrator session rates it low value because the demo model is 100 % simulator data by design (5.5c) and this is real-footage tooling. Agreed, ranked last, not started. |

Declined with reasons: **F4** (PX4 SITL — needs the sim), **F8b** (needs the GPU), **F9/F9b** (the other
session's, in progress), **F20** (no Jetson hardware), **F10** (real and unclaimed, but its only missing piece
is DINOv2 weights, i.e. the GPU).

**Owed to the orchestrator session, not done here: AUDIT S7(a).**
`tests/test_plan.py::test_revisit_queue_is_ordered_by_residual_pos_and_skips_burial_polygons` does not fail
when the queue order is reversed — mutation M26 flipped `revisit.py`'s `items.pop(0)` to `items.pop()` and all
31 plan tests stayed green, so `coverage_plan_eval.md` §2.2's "ordered by residual POS" claim is currently
unverified by anything. Flagged rather than fixed: that file is theirs.

---

## 2. F21 — the R10 source guard now covers the system

R10: **no code path may delete a record or mark a segment "cleared".** `sightline/triage/guardrails.py` had
the best scanner in the repo — it blanks strings and comments before matching code patterns, and matches
SQL/literals in a separate raw pass — pointed at two directories:

```python
LANE_SOURCE_DIRS = ("sightline/triage", "sightline/export")
```

Now **16 paths**: all 14 `sightline/` lane directories in `docs/CONTRACTS.md` §2, plus `pipeline.py` and
`schemas.py`. (AUDIT S10 says "13 lane directories". There are 14; it missed one.)

    R10 source guard: 104 file(s) over 16 path(s); 25 raw hit(s), 25 allowed by review, 0 unallowed.

**32 raw hits → 25.** Minus 8 false positives on prose that *denies* clearing, plus 1 new hit the widened
vocabulary reaches. **No real R10 violation was found in any of the 14 lanes** — every one of the 32 original
sites was read individually rather than taken from the audit's summary, the reading agrees with S10 on all 32,
and an independent red-team sweep for ten patterns the scanner has no rule for also came back clean. Read that
as "nothing was found by two readings and several probes", **not** as "the scanner proved it" — §6 says why the
difference matters.

### 2.1 It has teeth — but read §6 before quoting this table

Six real R10 defects planted in a byte-for-byte copy of the tree. **Old scope caught 0/6; new scope 6/6** —
with the caveat red-team review then established: those six defects are six rows of the scanner's own pattern
table, so the result is close to tautological, and "0/6" depends on placement (putting the `p.unlink()` in
`export/bundle.py`, a real and equally plausible location inside the OLD scope, makes it 1/6).

| planted defect | file |
|---|---|
| `@app.delete("/api/records/{rid}")` | `api/app.py` |
| `conn.execute("DELETE FROM records ...")` | `store/db.py` |
| legend string → `f"Segment {stem}: search complete."` | `coverage/export.py` |
| `seg.status = "cleared"` | `plan/segments.py` |
| `records.remove(r)` on low scores | `pipeline.py` |
| `p.unlink()` on evidence paths | `eval/records.py` |

### 2.2 The wording check was the defect, not just the coverage

The `"cleared"` pattern was a raw grep, so it fired on sentences promising never to clear a segment —
`coverage/__init__.py:8`, `coverage/accumulate.py:20,341`, `coverage/export.py:24`, `eval/calibration.py:15`,
`api/coverage_feed.py:31,37`, `dedup/cluster.py:14`. That is what turned
`test_api.py::test_this_lane_never_says_a_segment_is_cleared` red at 23:57.

It is now token-aware: each mention is classified as prose, a state *value*, a message string or a bare
identifier, and negation/modality is applied **scoped to the sentence** (lookback stops at `.!?;:`, a blank
line, the end of the token, 3 lines, 400 chars). A state value is flagged *before* the denial test, so
`status = "cleared"  # never` cannot launder itself. A mutation test takes the real text of
`coverage/__init__.py`, deletes the single word `Nothing`, and asserts the check then fires — so the fix reads
sentences rather than muting the rule.

### 2.3 The best finding was against its own work

**`def mark_cleared(seg)` scanned completely clean.** `_` is a word character, so `\b` cannot see the
vocabulary *inside an identifier* — a real R10 violation named `mark_cleared` would have walked through the
guard built to catch it. Found by an edge probe after the work looked finished. Fixed with an
underscore-tolerant boundary in **code context only**, which is why prose keeps the strict rule. (The original
note here said the looser boundary "adds exactly 1 line across 104 files". That was wrong — see §6.)

Two shortcuts were rejected rather than taken: an allowance reason came in at 14 words against a 15-word bar
and was rewritten rather than the bar lowered; and an over-broad-allowance test planted lines that did not
match its snippet, so the plant was fixed rather than the assertion.

### 2.4 Allowances are reasoned, and rot is a failure

23 entries covering 25 sites, matched by `(path, kind, whitespace-normalised snippet)` — deliberately **not**
by line number — each carrying a `why`. `audit_allowances()` fails on a **stale** entry (matches nothing, so
the code moved and the reasoning is no longer verified) and on an **over-broad** one (matches more lines than
its `sites` count). Categories: copy-then-pop with the value re-inserted; guarded one-element-set unwrap;
display trimming; `del` of a local 4K frame; scratch algorithm state; subscriber/attempt bookkeeping; a work
queue dequeue where the record stays; track pruning (tracks are not records, and it is counted in
`stats.tracks_pruned_unconfirmed`); a bounded LRU merge buffer at `maxlen=300` whose evictions are still
counted `late`; and one piece of counterfactual prose.

Gate: `uv run python -m sightline.triage.guardrails` (exit non-zero), plus `--show-allowed` and `--self-test`.

### 2.5 Coupling the orchestrator session must know about

`tests/test_triage.py` calls `scan_lane_sources(REPO)` with no arguments, so it picks up the widened default
and **now scans `sightline/mission/` and `sightline/pipeline.py`** — their files. A benign `.pop()` there will
turn that test red until someone writes a reasoned allowance. That is the gate working, but it will look like
this lane broke theirs. Both paths are clean today. They have been told.

---

## 3. The reporting half of AUDIT S8 — an unmeasurable number must not print as zero

*(The `schemas.py` / `dedup` / `threshold` half was red-teamed separately; its verdict was **keep with
repairs**, and all six of its findings are fixed — see §7.)*

`MetricRow.undefined` pins `value = 0.0` so an unmeasurable row still survives
`json.dumps(..., allow_nan=False)`. That neutral zero is a **serialisation device** — and `report.py` rendered
it straight through three separate paths (`slice_table`, `_headline`, the pixel-height table), so a slice in
which nothing was measured would have printed:

    - **0.0 % nominal-slice recall at IoU 0.5, in simulation** (n = 0 ground-truth boxes).

"The system found nobody" and "the system was never asked" are opposite claims, and that is the more damaging
direction to be wrong in. Fixed in one place (`_value_str`), because a rule spelled in three places is a rule
that will be spelled in two. It now reads:

    - **nominal-slice recall at IoU 0.5 is undefined, in simulation** — no ground-truth boxes in this slice
      (n = 0 ground-truth boxes).

The domain word stays **inside the bold span in both branches**: 5.5c requires every recall figure to carry
its domain in the same sentence, and an undefined figure is still a figure being reported. Two tests, in both
directions — an undefined row must not render as `0.0 %`, and a genuine measured `0.0` over n=57 must still
render as a measured zero, so the guard cannot swallow a real detector failure.

---

## 4. Integration, which neither agent could check

The two items were written concurrently by separate agents, and they interact: the S8 agent edited
`schemas.py`, `dedup/metrics.py` and `detect/threshold.py` — three files F21 newly scans. Neither agent saw
the other's final state.

With both in place: gate `104 files / 16 paths / 25 raw / 25 allowed / 0 unallowed`, exit 0;
**693 passed** across 14 test files. This also explains the S8 agent's one intermittent
`test_triage.py` failure — a genuine race against the sibling's concurrent write, as it suspected, not a
regression.

---

## 6. Red team: the guard did not hold, and what was repaired

A skeptical verifier was run against the finished F21 work with a brief to disprove it. It did.

**It planted eight real R10 violations into a sandbox copy of the tree — a lowercase `delete from records`
SQL statement in `store/db.py`, an HTTP endpoint deleting a record, a comprehension dropping records in
`pipeline.py`, and an operator legend reading "this segment sweep is complete, stand down" — and the gate's
output was byte-identical to the clean tree: `25 raw hits, 25 allowed, 0 unallowed`, exit 0.**

Root causes and repairs, all applied:

| # | Hole | Repair |
|---|---|---|
| E1 | `_TEXT_PATTERNS` compiled without `re.IGNORECASE`, so only ALL-CAPS SQL matched. Lowercase SQL is ordinary style. Statements spanning lines also escaped, because the raw pass matched per line. | `re.IGNORECASE \| re.DOTALL` over the JOINED source, line number recovered from the match offset |
| E3 | Only the `@app.delete` **decorator** was known; `store.delete(rid)`, `session.delete(rec)` and `methods=["DELETE"]` routes were invisible | added `\.delete\s*\(` and a `methods=[...DELETE...]` rule |
| E8 | `(?<![\w.])del\s+` needed a space, so `del(segments[i])` — valid Python — matched nothing | added a `del\s*\(` rule |
| — | `__delitem__`, bare `remove(`/`unlink(` after `from os import remove`, and `records[:] = []` all unmatched | added; the slice rule fires only on assignment of an EMPTY container, since `rgb[:rows] = (...)` is an ordinary write |
| E4 | **Allowance laundering by comment.** `_matches` tested the RAW line, so `self._records.discard(rid)  # mirrors self._clients.discard(ws)` was granted an allowance *by its own comment* | the comment is stripped before matching; string literals are kept, because honest allowances quote them |
| E6 | **Allowance laundering by prefix.** `in` was a substring test, so the blessed `del img` also covered `del img_store.records[fr.stem]` | the snippet must now end on a token boundary |

**Two claims in the original report were false and are corrected here rather than dropped:**

* *"the looser word boundary adds exactly one line across 104 files."* Re-measured three ways: **4** lines match
  loose-but-not-strict, **3** would be newly reported by a loose-in-prose rule, and **0** are added to the
  actual output. None is 1. The rule is still worth having — it is what catches `def mark_cleared` — but its
  value is prospective, not something it currently finds.
* *"zero real R10 violations in the 14 lanes"* is **not disproven**, and no violation was found by any probe.
  But its basis was a scanner blind to the most natural deletion shapes, so what actually holds is: *a human
  read 32 sites once, and the store's eight SQLite `RAISE(ABORT)` triggers are real.* The triggers are the
  only enforcement here that cannot be talked past.

**Known and now documented rather than papered over:** the scan cannot see semantic record loss.
`records = [r for r in records if r.score >= t]` deletes records with no delete keyword at all, and is the
most likely real-world accident. Eight statements of that shape already exist in the tree — all benign — and
the scan is silent on every one. Detecting them means dataflow analysis, not another regex. The module now
says so in its own docstring.

**Still open, not repaired here:** the guard is a **test, not a gate** — there is no CI, no pre-commit hook,
and `tools/doctor.py` / `tools/scene/gate.py` mention R10 zero times, so it runs only when someone types the
command or runs `tests/test_triage.py`. `app/map/index.html` — the actual operator map, the one artefact where
a human reads a legend and stands a search down — is out of scope entirely (its text denies clearing today,
so there is no live violation, but nothing guards it). `sightline/__init__.py` is scanned by nothing.

## 5. Owed elsewhere

- `docs/TRACKER.md` needs a line for F21 (2 → 16 paths, 23 allowances, S10 closed, gate command) and for the
  S8 contract change. Neither agent may edit it and nor will this lane while the orchestrator session is
  appending to it every ~10 minutes.
- `sightline/triage/__init__.py` does not re-export `ALLOWANCES`, `Allowance`, `audit_allowances`,
  `self_test` or `TRIAGE_LANE_DIRS`. `LANE_SOURCE_DIRS`, `scan_source`, `scan_lane_sources` and
  `assert_no_record_deletion` kept their names and signatures precisely so that file needed no edit.
- `sightline/eval/records.py:116` uses an older, weaker spelling of "undefined" (a `note=` string with a
  non-zero `n`). Worth converging on `MetricRow.undefined`.

---

## 7. Red team: the metric boundary — keep with repairs, all applied

The S8 work was red-teamed against 12 numbered claims. Nine confirmed exactly (including both mutation
counts, reproduced independently). Six defects found, **all now fixed**:

| # | Finding | Severity | Fix |
|---|---|---|---|
| **D1** | **A REGRESSION, not a gap.** `combine_rows(how="mean")` pooled the undefined placeholder's neutral `0.0`: a real 4.0 m CE90 plus one undefined row returned **2.0 m**. Before undefined rows existed, the same input produced a NaN that `metric_row()` rejected with a `ValueError` — a loud failure turned into a quotable falsehood. | blocking | `mean`/`sum` now refuse undefined rows outright. `weighted_mean` is genuinely safe (an undefined row has `n=0`, so it weighs nothing) and correctly returns 4.0. |
| **D3** | `frozen=True` on `MetricRow` did **not** close the hole it was applied for: `SliceKey` stayed mutable, so `row.slice.domain = "real"` relabelled a simulation number in one character. `schemas.py` and `CONTRACTS.md` §5 both claimed it was shut. | high | `SliceKey` frozen too; verified shut in all three directions; the test that missed it now asserts the one-character variant. **720 tests passed with the freeze — it was cheap.** |
| **D4** | The shallow freeze was a laundering channel: `del row.detail["undefined_reason"]` then `to_dicts` → `from_dicts` returned a **measured zero**. Separately, `__post_init__` mutated the *caller's* dict. | high | `undefined_reason` is now a top-level serialised field, not only a `detail` mirror; the detail dict is copied, never adopted. |
| **D5** | `duplicate_rate` with zero records is a deliberate, test-pinned **measured `0.0`** ("no records is a miss, not an infinity of duplicates") and was being reported as *undefined* with a reason that was false — there were no records to match. Property and reported row contradicted each other. | moderate | only undefined when genuinely unbounded (records exist, none matched). |
| **D2** | `record_precision` with zero records is 0/0 printed as a measured `0.0` — reads as "every record it produced was wrong" when it produced none. The defence written for *recall* (a real `n=1` denominator, honest) had been silently extended to it. | moderate | undefined when `n_records == 0`; recall's carve-out stays, because it is correct. |
| **D7/D8** | The `_value_str` guard reached one of three renderers; `report.py`'s payload write had no `allow_nan=False`. | moderate | `campaign.py` fixed; `allow_nan=False` added, matching `export/geojson.py`. |

**Two tests were changed, and both had encoded the defect** — stated plainly because "fix the test" is the
move this project rightly distrusts. `test_records_that_match_nothing_no_longer_ship_an_infinity` pinned the
old, less accurate reason string; it now asserts the reason names *this* case and says how many records.
`test_an_empty_result_is_reported_not_dropped_and_not_invented` asserted `duplicate_rate` should be undefined
with no records, which is D5; it now asserts the measured `0.0` that `tests/test_dedup.py` independently pins.
Both were strengthened, not loosened, and neither was touched to make code pass.

**Still open from that review, low severity:** `detect/evaluate.py:203` renders a raw value (latent — its rows
all come through `metric_row()`); `dataclasses.replace(undefined_row, n=5)` raises rather than being handled;
`eval/records.py:116` uses an older, weaker "undefined" idiom. Three mutations survived the S8 suite and are
named in the review: the `value=0.0` pin on the direct-construction path is untested, two of the five
converted dedup rows are asserted present but never asserted undefined, and only one reason string's text is
checked.
