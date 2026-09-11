"""The metric API boundary: no number leaves this project as a NaN, an infinity, or a relabelled domain.

This file exists because of AUDIT finding **S8 — MODERATE**. `docs/lanes/coverage_plan_eval.md` §2.3 advertised
*"The API makes a bare float impossible ... `make_slice()` without a domain is a `TypeError`; a NaN metric
raises."* The domain half was true. The NaN half stopped at the `sightline/eval/` package boundary, because
`sightline.eval.slicing.metric_row()` is only one of three constructors: `sightline/dedup/metrics.py` and
`sightline/detect/threshold.py` build `MetricRow` straight from `sightline/schemas.py`. The audit measured it on
the "we found nothing" dedup case — the one you most need to report honestly — and got, verbatim:

    dedup.mean_position_error_m value=nan   n=0   <-- NON-FINITE
    dedup.ce90_m                value=nan   n=0   <-- NON-FINITE
    json.dumps produced: *** bare NaN in the JSON (RFC 8259 invalid) ***
    strict JSON parse FAILS: non-finite NaN

Reproducing that here measured a third leak the audit did not list: with records that match nothing,
`dedup.duplicate_rate` is `+inf`, and `json.dumps` writes a bare `Infinity`, equally invalid.

The fix is enforcement in `MetricRow.__post_init__` — the TYPE, not another helper — because the defect *was* a
producer bypassing the shared helper. `test_the_guard_lives_on_the_type_so_every_producer_inherits_it` is the
test that pins that choice: it constructs the raw dataclass with no helper anywhere in sight.

What is asserted here, and why each one can fail:

1. the audit's exact reproduction now survives `json.dumps(..., allow_nan=False)` and a strict re-parse;
2. an empty result is still REPORTED — the row is present, `n = 0`, and it says why it has no value. Dropping
   the row would hide the case `docs/QUALITY_GATE.md` most wants on the record, and inventing a number would be
   worse;
3. the honest 0.0 placeholder cannot contaminate a pooled number;
4. `frozen=True` blocks `row.slice = SliceKey(domain="real")`, which silently relabelled a simulation number as
   a real-world one — hard rule 5, the project's hardest reporting rule.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from sightline.dedup.metrics import GroundTruthSurvivor, dedup_accuracy
from sightline.detect.threshold import OperatingPoint, choose_operating_threshold, metric_rows
from sightline.eval.slicing import DomainMixError, MetricSet, combine_rows, make_slice, metric_row
from sightline.schemas import MetricRow, Record, SliceKey

SIM = SliceKey(domain="sim")

#: The survivor the audit's reproduction is scored against, at the flood-valley origin.
GT_LAT, GT_LON = 12.9, 77.6


def _at(dn_m: float, de_m: float) -> tuple[float, float]:
    """(lat, lon) `dn_m` north and `de_m` east of the ground-truth survivor. Flat-earth is exact enough at 5 km."""
    return GT_LAT + dn_m / 111_320.0, GT_LON + de_m / (111_320.0 * math.cos(math.radians(GT_LAT)))


def _truth() -> list[GroundTruthSurvivor]:
    return [GroundTruthSurvivor("A", GT_LAT, GT_LON)]


def _record_at(dn_m: float, de_m: float, record_id: str = "r1") -> Record:
    r = Record(record_id=record_id)
    r.lat, r.lon = _at(dn_m, de_m)
    return r


def _strict_json(rows: list[MetricRow]) -> str:
    """Serialise rows the way the harness does, refusing every RFC 8259 violation on the way out AND back.

    `allow_nan=False` is the writer's half; `parse_constant` is the reader's — `json.loads` accepts `NaN` by
    default, so a blob written by some other tool would sail back in without it. `sightline/export/geojson.py`
    already writes with `allow_nan=False` for exactly this reason; metrics get the same treatment.
    """
    def _reject(token: str) -> float:
        raise ValueError(f"non-finite {token} survived into the metrics JSON")

    text = json.dumps(MetricSet(list(rows)).to_dicts(), allow_nan=False)
    json.loads(text, parse_constant=_reject)
    return text


# ---------------------------------------------------------------------------------------------------------
# 1. the audit's reproduction, both halves
# ---------------------------------------------------------------------------------------------------------
def test_the_we_found_nothing_dedup_case_serialises_as_strict_json():
    """AUDIT S8's exact reproduction: 0 records against 1 survivor. This is the failure that had to be fixed.

    Before: `dedup.mean_position_error_m` and `dedup.ce90_m` were `nan`, `json.dumps` wrote a bare `NaN`, and a
    strict parse raised. After: every row is finite, the two undefined ones say so, and the blob round-trips.
    """
    acc = dedup_accuracy([], _truth(), match_radius_m=6.0)
    rows = acc.rows(SIM)

    assert all(math.isfinite(r.value) for r in rows), [str(r) for r in rows if not math.isfinite(r.value)]
    text = _strict_json(rows)                      # raises if any value is NaN or Infinity
    assert "NaN" not in text and "Infinity" not in text

    back = json.loads(text)
    assert {d["name"] for d in back} >= {"dedup.mean_position_error_m", "dedup.ce90_m"}
    assert all(d["domain"] == "sim" for d in back), "hard rule 5: every number keeps its domain through JSON"


def test_records_that_match_nothing_no_longer_ship_an_infinity():
    """The variant the audit did not measure: `duplicate_rate` is (records - matched) / matched = 1/0 = +inf.

    `DedupAccuracy.duplicate_rate` still returns `inf` — `tests/test_dedup.py` pins `math.isinf(...)` and a
    caller doing arithmetic deserves the honest non-number. The honesty is applied where it is REPORTED.
    """
    acc = dedup_accuracy([_record_at(5_000.0, 0.0)], _truth(), match_radius_m=6.0)
    assert math.isinf(acc.duplicate_rate), "the property is unchanged; only the reported row is"

    row = next(r for r in acc.rows(SIM) if r.name == "dedup.duplicate_rate")
    assert not row.is_defined and row.n == 0 and row.value == 0.0
    # The reason must name THIS case — records exist and none of them matched, so the rate is unbounded — and
    # not the generic "no sample" text, which is the wrong explanation when there ARE records. An operator
    # reads this string; a reason that describes a different failure is a quieter version of a wrong number.
    assert "unbounded" in row.undefined_reason
    assert "1 record" in row.undefined_reason, f"the reason must say how many records: {row.undefined_reason}"
    assert "Infinity" not in _strict_json(acc.rows(SIM))


def test_an_empty_result_is_reported_not_dropped_and_not_invented():
    """The whole point of the fix. An empty mission still produces every row, and says WHY each has no value.

    Recall is deliberately NOT in the undefined group: with one survivor and no records, recall really is 0.0
    over n = 1. Calling that "undefined" would flatter the system, which is the opposite of the intent.
    """
    rows = {r.name: r for r in dedup_accuracy([], _truth(), match_radius_m=6.0).rows(SIM)}
    assert {"dedup.record_precision", "dedup.record_recall", "dedup.duplicate_rate", "dedup.count_mae",
            "dedup.count_bias", "dedup.mean_position_error_m", "dedup.ce90_m"} <= set(rows)

    recall = rows["dedup.record_recall"]
    assert recall.is_defined and recall.value == 0.0 and recall.n == 1, "0 of 1 found is a measured 0, not a gap"

    # `duplicate_rate` belongs with recall, NOT with the undefined group — and this assertion originally had
    # it in the wrong one. `DedupAccuracy.duplicate_rate` deliberately returns a MEASURED 0.0 when there are
    # no records ("no records is a miss, not an infinity of duplicates"; the property says so and
    # `tests/test_dedup.py` pins it). Reporting that real zero as undefined, with the reason "no record
    # matched a ground-truth survivor", was false — there were no records to match — and left the property
    # and the reported row contradicting each other. Found by red-team review, not by this suite.
    dup = rows["dedup.duplicate_rate"]
    assert dup.is_defined and dup.value == 0.0, (
        "with zero records the duplicate rate is a measured 0.0, not an absence of measurement")

    for name in ("dedup.mean_position_error_m", "dedup.ce90_m"):
        row = rows[name]
        assert not row.is_defined and row.n == 0
        assert "no sample" in row.undefined_reason
        # The VALUE position must not hold a number a reader could quote out of the log.
        assert str(row).startswith(f"{name}=undefined ("), f"undefined row prints a value: {row}"


# ---------------------------------------------------------------------------------------------------------
# 2. the guard itself: on the type, with teeth
# ---------------------------------------------------------------------------------------------------------
def test_the_guard_lives_on_the_type_so_every_producer_inherits_it():
    """No helper is imported here on purpose. This is the raw frozen-contract dataclass, the way the two
    producers in AUDIT S8 called it — the call that used to succeed and ship a NaN into the metrics JSON."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="nan|inf|NaN"):
            MetricRow("dedup.ce90_m", bad, SIM, 0)

    ok = MetricRow("dedup.ce90_m", 4.25, SIM, 12)
    assert ok.value == 4.25 and ok.is_defined


def test_the_error_message_names_the_honest_alternative():
    """A guard that only says "no" teaches nobody. `eval.slicing.metric_row` already stated the contract —
    "report an explicit n=0 row with a note instead of a NaN or an infinity" — so the type says the same thing
    and names the two constructors that do it."""
    with pytest.raises(ValueError) as exc:
        MetricRow("track.motp", float("nan"), SIM, 0)
    message = str(exc.value)
    assert "n=0" in message and "MetricRow.undefined" in message and "finite_or_undefined" in message
    assert "8259" in message, "say why it matters: RFC 8259 has no NaN, so the row cannot be serialised"


def test_an_undefined_row_must_say_why_and_must_have_no_sample():
    """Two invariants that keep the escape hatch from becoming a laundry.

    `n == 0`: the 0.0 placeholder must never be pooled. A non-finite value over a NON-empty sample is arithmetic
    gone wrong, and gets an exception rather than a tidy "undefined" label.
    """
    with pytest.raises(ValueError, match="WHY"):
        MetricRow.undefined("x", SIM, "")

    with pytest.raises(ValueError, match="n=7"):
        MetricRow(name="x", value=0.0, slice=SIM, n=7, undefined_reason="nothing matched")

    with pytest.raises(ValueError, match="measured value"):
        MetricRow(name="x", value=0.93, slice=SIM, n=0, undefined_reason="nothing matched")

    # ... and the legitimate path: non-finite with no sample becomes the honest row.
    row = MetricRow.finite_or_undefined("x", float("nan"), SIM, 0, reason="no matched pairs")
    assert not row.is_defined and row.value == 0.0 and row.n == 0
    # ... while non-finite WITH a sample is a bug, and stays loud.
    with pytest.raises(ValueError, match="n=7"):
        MetricRow.finite_or_undefined("x", float("nan"), SIM, 7, reason="no matched pairs")


def test_the_undefined_placeholder_cannot_contaminate_a_pooled_number():
    """`combine_rows(how="weighted_mean")` weights by `n`, so an n=0 row contributes exactly nothing.

    This is why `undefined_reason` requires `n == 0` rather than merely recommending it: the placeholder is a
    0.0, and a 0.0 with weight would drag a pooled accuracy down and nobody would see why.
    """
    measured = MetricRow("dedup.ce90_m", 4.0, SIM, 5)
    empty = MetricRow.undefined("dedup.ce90_m", SIM, "no matched pairs in this slice")
    pooled = combine_rows([measured, empty], "dedup.ce90_m", how="weighted_mean")
    assert pooled.value == pytest.approx(4.0) and pooled.n == 5


# ---------------------------------------------------------------------------------------------------------
# 3. the second half of S8: a row is a reported number, so it may not be relabelled
# ---------------------------------------------------------------------------------------------------------
def test_a_row_cannot_be_relabelled_from_sim_to_real_in_place():
    """AUDIT S8's minor note. `MetricRow` was a mutable dataclass, so `row.slice = SliceKey(domain="real")`
    turned a simulation number into a real-world one with no trace — defeating `DomainMixError`, which is the
    enforcement behind hard rule 5 ("never average a simulation number with a real number")."""
    row = metric_row("recall@iou0.5", 0.94, make_slice("sim"), 500)

    with pytest.raises(dataclasses.FrozenInstanceError):
        row.slice = make_slice("real")          # type: ignore[misc]

    # Freezing the ROW alone did not close this. `row.slice.domain = "real"` reached through the frozen row
    # into a still-mutable `SliceKey` and relabelled the number anyway, in one character -- a lock on the door
    # of an open window. Caught by red-team review; `SliceKey` is frozen too now. Without this assertion the
    # test above passes while hard rule 5 stays wide open.
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.slice.domain = "real"               # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.value = 0.99                        # type: ignore[misc]
    assert row.slice.domain == "sim" and row.value == pytest.approx(0.94)

    # The domain guard is what the freeze protects, so prove it still fires on the relabelled COPY.
    with pytest.raises(DomainMixError):
        combine_rows([row, dataclasses.replace(row, slice=make_slice("real"))], "recall@iou0.5")


# ---------------------------------------------------------------------------------------------------------
# 4. the flag survives the paths a report actually takes
# ---------------------------------------------------------------------------------------------------------
def test_the_undefined_flag_survives_the_eval_lanes_json_round_trip():
    """`MetricSet.to_dicts()` carries `detail` but has no column for a new field, so the reason is mirrored into
    `detail` and read back out of it. Without the mirror an undefined row would return from `from_dicts()` as a
    measured 0.0 — a NaN laundered into a number, which is worse than the NaN."""
    rows = dedup_accuracy([], _truth(), match_radius_m=6.0).rows(SIM)
    restored = MetricSet.from_dicts(json.loads(_strict_json(rows)))

    before = {r.name: r.undefined_reason for r in rows}
    after = {r.name: r.undefined_reason for r in restored}
    assert after == before and any(after.values()), "the reasons must come back, not just the numbers"
    assert all("undefined" in str(r.detail) for r in restored if not r.is_defined)


def test_the_detector_lane_reports_an_empty_split_instead_of_a_bare_zero():
    """The other producer AUDIT S8 named. `sweep()` returns 0.0 for an empty denominator so the curve stays
    plottable; at the reporting boundary "recall = 0.0 over 0 boxes" reads as total failure when the truth is
    that there was nothing to find."""
    op, _points, _tally = choose_operating_threshold([])
    assert op.n_gt == 0 and op.n_frames == 0

    rows = {r.name: r for r in metric_rows(op, SIM)}
    assert set(rows) >= {"recall@IoU0.5", "precision@IoU0.5", "fp_per_frame"}
    for row in rows.values():
        assert not row.is_defined and row.n == 0 and row.undefined_reason
    assert "no ground-truth boxes" in rows["recall@IoU0.5"].undefined_reason
    _strict_json(list(rows.values()))


def test_a_broken_detector_number_raises_instead_of_being_laundered():
    """A non-finite recall over 40 real ground-truth boxes is not an empty result, it is broken arithmetic.
    The undefined row is for the empty case only; this one has to be loud."""
    op = OperatingPoint(
        conf=0.5, recall=float("nan"), precision=0.9, recall_secondary_iou=0.95, target_recall=0.92,
        achieved=False, tp=36, fp=4, fn=4, n_gt=40, n_frames=10, iou_thr=0.5, secondary_iou_thr=0.25,
        fp_per_frame=0.4, fp_per_minute=None, fps_processed=None, cls="human",
    )
    with pytest.raises(ValueError, match="n=40"):
        metric_rows(op, SIM)


def test_no_producer_in_the_project_can_emit_a_non_finite_row():
    """The property S8 actually claimed, checked over every metric producer this repo has, on the degenerate
    inputs that used to break them. `sightline/export/geojson.py` already refuses non-finite coordinates; after
    this change the metric API refuses non-finite values, and the claim in the lane report is true."""
    produced: list[MetricRow] = []
    produced += dedup_accuracy([], _truth(), match_radius_m=6.0).rows(SIM)                 # nothing found
    produced += dedup_accuracy([_record_at(5_000.0, 0.0)], _truth(), match_radius_m=6.0).rows(SIM)  # all wrong
    produced += dedup_accuracy([_record_at(0.4, 0.0)], _truth(), match_radius_m=6.0,
                               duration_s=600.0).rows(SIM)                                 # the happy path
    produced += metric_rows(choose_operating_threshold([])[0], SIM)                        # empty split
    produced += [metric_row("recall@iou0.5", 0.94, make_slice("sim"), 500)]                # the eval lane

    assert len(produced) > 20
    for row in produced:
        assert math.isfinite(row.value), f"{row.name} leaked {row.value!r}"
        assert row.is_defined or row.n == 0
    _strict_json(produced)
