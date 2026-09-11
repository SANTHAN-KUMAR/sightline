"""F19 — the evaluation harness, verified against SOLUTION_DOC §5.5, §5.6, §5.7, §5.12, §6.3 and §6.5.

Everything here is driven by ground truth and predictions **constructed by hand**, so the correct answer is
known before the code runs: a perfect set must score 1.0, a set with two deliberate misses must score exactly
the right fraction, a curve built to cross 0.92 at one known confidence must freeze there. No model, no GPU, no
simulator (docs/CONTRACTS.md §3).

    D:\\Tools\\uv\\uv.exe run pytest tests/test_eval.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sightline.common.geodesy import offset_ne
from sightline.eval import (
    IOU_PRIMARY,
    IOU_RELAXED,
    TARGET_RECALL,
    AttributeSample,
    DomainMixError,
    EvalConfig,
    EvalDataset,
    GtBox,
    GtFrame,
    GtSurvivor,
    MetricSet,
    RankingComparison,
    SliceError,
    add_record_eval,
    attribute_accuracy_rows,
    build_manifest,
    check_split_disjoint,
    combine_rows,
    dedup_radius_m,
    evaluate_geolocation,
    evaluate_pod_calibration,
    evaluate_records,
    evaluate_tracking,
    failure_clusters,
    freeze_on_validation,
    freeze_operating_threshold,
    gated_assignment,
    headline_rows,
    hungarian,
    iou_matrix,
    make_slice,
    match_dataset,
    match_frame,
    metric_row,
    narrow,
    partition,
    pixel_size_bin,
    promotion_audit_rows,
    r10_rows,
    rank_correlation_rows,
    ratio_row,
    reliability_bins,
    render_markdown,
    require_single_domain,
    run_detection_eval,
    samples_from_records,
    slice_rows,
    sweep_confidence,
    time_of_day_bin,
    tracking_gain_row,
    write_report,
)
from sightline.eval.detection import NEVER
from sightline.eval.groundtruth import URGENCY_ORDER, dataset_from_json, dataset_to_json
from sightline.eval.slicing import AXIS_VALUES, BOX_AXES, FRAME_AXES, altitude_band
from sightline.eval.synthetic import ORIGIN_LAT, ORIGIN_LON, make_scenario, perfect_clip, shifted_boxes
from sightline.eval.tracking import TrackFrame
from sightline.schemas import CE90_FACTOR, Detection, MetricRow, Record, SliceKey

SIM = make_slice("sim")
REAL = make_slice("real")


def _clip(boxes_per_frame: list[list[tuple[float, float, float, float]]], *, domain="sim", split="test",
          fps=5.0, clip_id="hand", agl_m=50.0, tod="day", zone="settlement") -> EvalDataset:
    ds = EvalDataset(domain=domain, fps_processed=fps, clip_id=clip_id, split=split)  # type: ignore[arg-type]
    gid = 0
    for fi, boxes in enumerate(boxes_per_frame):
        f = GtFrame(frame_idx=fi, clip_id=clip_id, zone=zone, agl_m=agl_m, time_of_day=tod)  # type: ignore[arg-type]
        for b in boxes:
            f.boxes.append(GtBox(bbox_px=b, gt_id=gid, frame_idx=fi, occlusion=0, posture="prone",
                                 submersion="dry"))
            gid += 1
        ds.frames.append(f)
    return ds


# ---------------------------------------------------------------------------------------------------------
# 1. recall: the number the whole project is judged on
# ---------------------------------------------------------------------------------------------------------
def test_a_perfect_prediction_set_scores_recall_one():
    ds, dets = perfect_clip(n_frames=4, n_boxes=3, score=0.9)
    for iou in (IOU_PRIMARY, IOU_RELAXED):
        res = match_dataset(ds, dets, iou_thr=iou)
        c = res.counts_at(0.0)
        assert res.n_targets == 12 and c.n_gt == 12
        assert c.tp == 12 and c.fp == 0 and c.fn == 0
        assert c.recall() == 1.0 and c.precision() == 1.0 and c.f1() == 1.0
        assert c.fp_per_min() == 0.0
        np.testing.assert_allclose(res.gt_match_iou, 1.0)
    rows = headline_rows(match_dataset(ds, dets), 0.5, SIM)
    assert rows.get("recall@iou0.5", "sim").value == 1.0
    assert all(isinstance(r, MetricRow) and r.slice.domain == "sim" for r in rows)


def test_constructed_misses_score_the_exact_recall_at_iou_0_5_and_0_25():
    """Four targets; one exact hit, one at IoU 0.35, one at IoU 0.10, one with no prediction at all."""
    boxes = [(0.0, 0.0, 100.0, 100.0), (200.0, 0.0, 300.0, 100.0),
             (400.0, 0.0, 500.0, 100.0), (600.0, 0.0, 700.0, 100.0)]
    ds = _clip([boxes])
    preds = [
        Detection(bbox_px=boxes[0], score=0.90, frame_idx=0),
        Detection(bbox_px=shifted_boxes(boxes[1], 0.35), score=0.80, frame_idx=0),
        Detection(bbox_px=shifted_boxes(boxes[2], 0.10), score=0.70, frame_idx=0),
    ]
    # the constructed IoUs really are what the names say
    assert float(iou_matrix([preds[1].bbox_px], [boxes[1]])[0, 0]) == pytest.approx(0.35, abs=1e-9)
    assert float(iou_matrix([preds[2].bbox_px], [boxes[2]])[0, 0]) == pytest.approx(0.10, abs=1e-9)

    strict = match_dataset(ds, preds, iou_thr=0.5).counts_at(0.0)
    assert (strict.tp, strict.fn, strict.fp, strict.n_gt) == (1, 3, 2, 4)
    assert strict.recall() == pytest.approx(0.25)
    assert strict.precision() == pytest.approx(1 / 3)

    relaxed = match_dataset(ds, preds, iou_thr=0.25).counts_at(0.0)
    assert (relaxed.tp, relaxed.fn, relaxed.fp, relaxed.n_gt) == (2, 2, 1, 4)
    assert relaxed.recall() == pytest.approx(0.50), "§5.5: the relaxed threshold is 'found the person'"
    assert IOU_RELAXED == 0.25 and IOU_PRIMARY == 0.5


def test_a_ground_truth_box_nobody_found_is_a_miss_at_every_threshold():
    ds = _clip([[(0.0, 0.0, 50.0, 50.0)]])
    res = match_dataset(ds, [], iou_thr=0.5)
    assert res.gt_match_score.tolist() == [NEVER]
    assert not res.gt_found(0.0).any() and not res.gt_found(-1e9).any()
    assert res.counts_at(0.0).recall() == 0.0


def test_greedy_matching_is_score_ordered_and_one_to_one():
    """§5.12: greedy in descending score. The best prediction takes the target; the next is a false positive."""
    gt = [GtBox(bbox_px=(0.0, 0.0, 100.0, 100.0), gt_id=0, frame_idx=0)]
    preds = [(0.0, 0.0, 100.0, 100.0), (5.0, 5.0, 105.0, 105.0)]
    m = match_frame(preds, [0.4, 0.9], gt, 0.5)
    assert m.tp == [(1, 0)] and m.fp == [0], "the 0.9 prediction wins the target"
    assert m.n_tp == 1 and m.n_fp == 1 and m.n_fn == 0
    m2 = match_frame(preds, [0.9, 0.4], gt, 0.5)
    assert m2.tp == [(0, 0)] and m2.fp == [1]


def test_uncertain_ignore_and_group_boxes_follow_the_annotation_guideline():
    """§6.3: `uncertain`/`ignore` predictions are dropped, a group is one recall target at IoD >= 0.5."""
    boxes = [
        GtBox(bbox_px=(0.0, 0.0, 100.0, 100.0), gt_id=0, frame_idx=0),
        GtBox(bbox_px=(200.0, 0.0, 300.0, 100.0), gt_id=1, frame_idx=0, uncertain=True),
        GtBox(bbox_px=(400.0, 0.0, 500.0, 100.0), gt_id=2, frame_idx=0, ignore=True),
        GtBox(bbox_px=(600.0, 0.0, 800.0, 100.0), gt_id=3, frame_idx=0, is_group=True, group_count=4),
    ]
    preds = [(0.0, 0.0, 100.0, 100.0), (210.0, 10.0, 290.0, 90.0), (410.0, 10.0, 490.0, 90.0),
             (620.0, 20.0, 660.0, 80.0), (2000.0, 0.0, 2050.0, 50.0)]
    m = match_frame(preds, [0.9, 0.8, 0.7, 0.6, 0.5], boxes, 0.5)
    assert m.tp == [(0, 0)]
    assert sorted(m.ignored_pred) == [1, 2, 3], "uncertain, ignore and group predictions are neither TP nor FP"
    assert m.fp == [4]
    assert m.group_hit == [3] and m.group_miss == []
    assert m.n_tp == 2 and m.n_fn == 0 and m.n_fp == 1
    assert [b.scored for b in boxes] == [True, False, False, True]

    missed_group = match_frame([(0.0, 0.0, 100.0, 100.0)], [0.9], boxes, 0.5)
    assert missed_group.group_miss == [3] and missed_group.n_fn == 1


def test_fp_per_min_is_the_documented_ratio():
    """§5.12: FP/min = FP_total / (frames_processed / fps_processed / 60)."""
    ds = _clip([[(0.0, 0.0, 40.0, 40.0)] for _ in range(10)], fps=5.0)
    fps_boxes = [Detection(bbox_px=(1000.0, 1000.0, 1040.0, 1040.0), score=0.6, frame_idx=i) for i in range(3)]
    res = match_dataset(ds, fps_boxes, iou_thr=0.5)
    assert ds.minutes_processed == pytest.approx(10 / 5.0 / 60.0)
    c = res.counts_at(0.0)
    assert c.fp == 3
    assert c.fp_per_min() == pytest.approx(3.0 / (10 / 5.0 / 60.0)) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        _ = EvalDataset(domain="sim", fps_processed=0.0).minutes_processed
    with pytest.raises(ValueError):
        EvalDataset(domain="synthetic").__post_init__()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------
# 2. the frozen operating threshold (§5.5)
# ---------------------------------------------------------------------------------------------------------
def _pr_clip(n=25, hits=25, split="val"):
    """n targets, one per frame; the `hits` highest-scoring ones are found, scores 0.99, 0.98, ... descending."""
    boxes = [[(10.0 * i, 0.0, 10.0 * i + 40.0, 60.0)] for i in range(n)]
    ds = _clip(boxes, split=split, clip_id="pr")
    preds = [Detection(bbox_px=boxes[i][0], score=round(0.99 - 0.01 * i, 4), frame_idx=i) for i in range(hits)]
    return ds, preds


def test_the_sweep_picks_the_highest_confidence_at_which_recall_reaches_0_92_and_freezes_it():
    ds, preds = _pr_clip(n=25, hits=25)
    res = match_dataset(ds, preds, iou_thr=0.5)
    curve = sweep_confidence(res, "sim")

    # the constructed curve: recall falls by 1/25 = 0.04 for every 0.01 of confidence
    assert curve.conf[0] == pytest.approx(0.75) and curve.conf[-1] == pytest.approx(0.99)
    np.testing.assert_allclose(curve.recall, np.arange(25, 0, -1) / 25.0)
    # 0.92 x 25 = 23 targets, so the 23rd-highest score, 0.99 - 0.22 = 0.77, is the last threshold that works
    op = freeze_operating_threshold(curve, TARGET_RECALL)
    assert TARGET_RECALL == 0.92
    assert op.conf == pytest.approx(0.77)
    assert op.recall == pytest.approx(0.92) and op.achieved
    assert res.counts_at(0.78).recall() < TARGET_RECALL, "one notch higher and the target is missed"
    assert res.counts_at(op.conf).recall() >= TARGET_RECALL
    # the threshold is frozen with the clip it was frozen on, so a reader cannot mistake the population
    frozen = freeze_on_validation(ds, preds)
    assert frozen.conf == pytest.approx(0.77)
    assert frozen.frozen_on_split == "val" and frozen.frozen_on_clip == "pr"
    assert frozen.frozen_on == "pr[val]"
    rows = frozen.rows(SIM)
    assert rows.get("operating_conf", "sim").value == pytest.approx(0.77)
    assert rows.get("operating_recall", "sim").value == pytest.approx(0.92)
    assert all(r.detail["frozen_on"] == "pr[val]" for r in rows)


def test_the_threshold_may_not_be_frozen_on_the_test_split_by_accident():
    ds, preds = _pr_clip(split="test")
    with pytest.raises(ValueError, match="validation split"):
        freeze_on_validation(ds, preds)
    op = freeze_on_validation(ds, preds, allow_non_val_split=True)
    assert op.frozen_on_split == "test", "opting in is allowed, but it is recorded"


def test_an_unreachable_recall_target_is_reported_as_a_finding_not_papered_over():
    ds, preds = _pr_clip(n=25, hits=20)  # only 20 of 25 can ever be found
    curve = sweep_confidence(match_dataset(ds, preds, iou_thr=0.5), "sim")
    op = freeze_operating_threshold(curve, TARGET_RECALL)
    assert not op.achieved
    assert op.recall == pytest.approx(0.80) == pytest.approx(curve.recall.max())
    assert op.conf == pytest.approx(curve.conf[0])
    assert "never reached" in op.note and "rather than moving the target" in op.note
    assert op.target_recall == 0.92, "the target is never lowered to make the report look good"


def test_the_max_f1_confidence_is_reported_but_is_not_the_operating_point():
    ds, preds = _pr_clip(n=25, hits=25)
    curve = sweep_confidence(match_dataset(ds, preds, iou_thr=0.5), "sim")
    assert curve.max_f1_conf() == pytest.approx(0.75), "max F1 is where precision and recall are both 1"
    assert freeze_operating_threshold(curve, TARGET_RECALL).conf != curve.max_f1_conf()


def test_curve_lookup_agrees_with_counting_at_the_same_threshold():
    """`PrCurve.at(c)` must be exactly `counts_at(c)`: it is what `run_detection_eval` reports as the
    operating recall when the caller supplies a threshold of its own."""
    ds, preds = _pr_clip(n=25, hits=25)
    res = match_dataset(ds, preds, iou_thr=0.5)
    curve = sweep_confidence(res, "sim")
    for conf in (0.0, 0.7495, 0.75, 0.7501, 0.769, 0.77, 0.775, 0.9899, 0.99, 0.995, 1.5):
        r, p, f = curve.at(conf)
        c = res.counts_at(conf)
        assert r == pytest.approx(c.recall()), conf
        assert p == pytest.approx(c.precision()), conf
        assert f == pytest.approx(c.fp_per_min()), conf


# ---------------------------------------------------------------------------------------------------------
# 3. slicing: every number carries its slice, and the bins partition (hard rule 5)
# ---------------------------------------------------------------------------------------------------------
def test_the_slice_grid_is_the_documented_one():
    """§5.12: zone x altitude x time of day x occlusion x posture x pixel size x modality x terrain type.

    `context` is the terrain the box sits on. §5.12 asks for it in the same breath as the rest of the grid --
    "FP/min per terrain type (water, debris, vegetation, roof)" -- but it was missing from both the table and
    this assertion until 2026-09-11, so no `MetricRow` could carry one. It is measured by
    `sightline.eval.context` from the placed scene, not derived from the survivor's zone label.
    """
    assert set(AXIS_VALUES) == {"zone", "altitude_band", "time_of_day", "occlusion", "posture",
                               "pixel_size", "modality", "context"}
    assert set(BOX_AXES) | set(FRAME_AXES) == set(AXIS_VALUES)
    assert not set(BOX_AXES) & set(FRAME_AXES), "an axis is a property of the box or of the frame"


def test_a_metric_cannot_exist_without_a_valid_domain():
    with pytest.raises(SliceError):
        make_slice("synthetic")
    with pytest.raises(TypeError):
        make_slice()  # type: ignore[call-arg]
    with pytest.raises(SliceError):
        make_slice("sim", zone="ocean")
    with pytest.raises(SliceError):
        make_slice("sim", nonsense="x")
    with pytest.raises(SliceError):
        narrow(SIM, domain="real")
    with pytest.raises(ValueError, match="NaN|nan|infinit"):
        metric_row("r", float("nan"), SIM)
    with pytest.raises(TypeError):
        MetricSet().add(0.5)  # type: ignore[arg-type]
    row = metric_row("recall@iou0.5", 0.9, narrow(SIM, zone="fan", altitude_band="45-60"), 12)
    assert "domain=sim" in str(row) and "zone=fan" in str(row) and "n=12" in str(row)


def test_binning_functions_use_the_documented_edges():
    assert [altitude_band(h) for h in (25.0, 30.0, 44.9, 45.0, 59.9, 60.0, 89.9, 90.0, 200.0)] == \
        ["<30", "30-45", "30-45", "45-60", "45-60", "60-90", "60-90", "90+", "90+"]
    assert [pixel_size_bin(p) for p in (0.0, 19.9, 20.0, 39.9, 40.0, 79.9, 80.0, 500.0)] == \
        ["<20", "<20", "20-40", "20-40", "40-80", "40-80", "80+", "80+"]
    assert [time_of_day_bin(v) for v in ("dawn", 6, 12, 18, 23, "2026-09-10T05:40:00", "18:05")] == \
        ["dawn", "dawn", "day", "dusk", "night", "dawn", "dusk"]
    with pytest.raises(SliceError):
        altitude_band(float("nan"))
    with pytest.raises(SliceError):
        pixel_size_bin(float("inf"))


def test_slices_partition_the_ground_truth_without_double_counting():
    sc = make_scenario(seed=3, n_survivors=6, n_frames=24)
    res = match_dataset(sc.dataset, sc.detections, iou_thr=0.5)
    rows = slice_rows(sc.dataset, res, 0.3, SIM, axes=BOX_AXES + FRAME_AXES)
    total = len(res.gt_boxes)
    assert total > 50

    for axis in BOX_AXES + FRAME_AXES:
        recalls = [r for r in rows if r.name.startswith("recall@") and r.detail.get("axis") == axis]
        assert recalls, axis
        labels = [getattr(r.slice, axis) for r in recalls]
        assert len(labels) == len(set(labels)), f"{axis}: a bin appeared twice"
        binned = sum(r.n for r in recalls)
        leftover = [r for r in rows if r.name == "slice_unbinned_gt" and r.detail.get("axis") == axis]
        assert len(leftover) == 1, axis
        assert binned + int(leftover[0].value) == total, f"{axis}: bins + unbinned != n"
        assert leftover[0].detail["binned"] == binned and leftover[0].n == total
        # no ground-truth box is counted in two bins of the same axis
        tp = sum(int(r.detail["tp"]) for r in recalls)
        assert tp <= total

    # box axes carry recall only: a false positive has no ground-truth box, so precision there is undefined
    for axis in BOX_AXES:
        assert not [r for r in rows if r.name.startswith("precision") and r.detail.get("axis") == axis]
    for axis in FRAME_AXES:
        assert [r for r in rows if r.name.startswith("precision") and r.detail.get("axis") == axis]

    # and the whole-clip recall is the sample-weighted pooling of one axis's bins
    whole = headline_rows(res, 0.3, SIM).overall("sim").get("recall@iou0.5", "sim")
    for axis in ("occlusion", "posture", "pixel_size"):
        recalls = [r for r in rows if r.name.startswith("recall@") and r.detail.get("axis") == axis]
        if sum(r.n for r in recalls) == total:
            pooled = combine_rows(recalls, "recall@iou0.5")
            assert pooled.value == pytest.approx(whole.value, abs=1e-9), axis


def test_partition_never_loses_or_duplicates_an_item():
    items = list(range(37))
    groups = partition(items, lambda i: str(i % 5))
    assert sum(len(v) for v in groups.values()) == 37
    assert sorted(x for v in groups.values() for x in v) == items


# ---------------------------------------------------------------------------------------------------------
# 4. hard rule 5: a sim row and a real row may never be collapsed
# ---------------------------------------------------------------------------------------------------------
def test_combining_a_sim_row_with_a_real_row_raises():
    s = metric_row("recall@iou0.5", 0.94, SIM, 500)
    r = metric_row("recall@iou0.5", 0.61, REAL, 120)
    for how in ("weighted_mean", "mean", "sum"):
        with pytest.raises(DomainMixError):
            combine_rows([s, r], "recall@iou0.5", how=how)
    with pytest.raises(DomainMixError):
        require_single_domain([s, r])
    with pytest.raises(DomainMixError):
        ratio_row("ratio", s, r)
    with pytest.raises(DomainMixError):
        MetricSet([s, r]).aggregate("recall@iou0.5")
    with pytest.raises(DomainMixError):
        combine_rows([], "recall@iou0.5")

    # the two may still sit side by side in one table; only collapsing them is refused
    both = MetricSet([s, r])
    assert both.domains() == ["real", "sim"]
    assert set(both.by_domain()) == {"sim", "real"}
    assert both.aggregate("recall@iou0.5", domain="sim").value == pytest.approx(0.94)
    assert both.filter(domain="real").rows == [r]
    # and the honest average of two sim rows is the sample-weighted one
    s2 = metric_row("recall@iou0.5", 0.50, SIM, 500)
    assert combine_rows([s, s2], "recall@iou0.5").value == pytest.approx(0.72)
    assert combine_rows([s, s2], "recall@iou0.5").n == 1000


def test_every_public_evaluation_function_returns_metric_rows_not_floats():
    sc = make_scenario(seed=5, n_survivors=5, n_frames=20)
    res = match_dataset(sc.dataset, sc.detections, iou_thr=0.5)
    sets = {
        "headline": headline_rows(res, 0.3, SIM),
        "slices": slice_rows(sc.dataset, res, 0.3, SIM),
        "records": evaluate_records(sc.records, sc.dataset.survivors, SIM, radius_m=6.0, minutes=1.0),
        "r10": r10_rows(sc.records, SIM),
        "geo": evaluate_geolocation(samples_from_records(sc.records), sc.dataset.survivors, SIM),
        "pod": evaluate_pod_calibration(sc.pod_predicted, sc.pod_found, SIM),
        "posture": attribute_accuracy_rows(sc.posture_samples, SIM, attribute="posture"),
        "ranking": rank_correlation_rows(sc.ranking, SIM),
        "promotion": promotion_audit_rows(sc.ranking, SIM),
    }
    for name, ms in sets.items():
        assert isinstance(ms, MetricSet) and len(ms) > 0, name
        for row in ms:
            assert isinstance(row, MetricRow), name
            assert isinstance(row.slice, SliceKey) and row.slice.domain in ("sim", "real"), name
            assert math.isfinite(row.value), f"{name}: {row.name}"
    dicts = sets["records"].to_dicts()
    assert all("domain" in d for d in dicts)
    assert len(MetricSet.from_dicts(dicts)) == len(dicts)


# ---------------------------------------------------------------------------------------------------------
# 5. record level after tracking and dedup (§5.6)
# ---------------------------------------------------------------------------------------------------------
def _survivors(n=3, buried=0):
    out = []
    for i in range(n + buried):
        lat, lon = offset_ne(ORIGIN_LAT, ORIGIN_LON, 100.0 * i, 0.0)
        out.append(GtSurvivor(gt_id=i, lat=lat, lon=lon, count=1, buried=i >= n))
    return out


def test_duplicate_rate_is_right_on_a_constructed_duplicate():
    """§5.6: duplicate rate = (records - unique matched GT) / unique matched GT."""
    survivors = _survivors(3)
    recs = [Record(lat=s.lat, lon=s.lon, cls="human", count_estimate=1) for s in survivors]
    dup_lat, dup_lon = offset_ne(survivors[0].lat, survivors[0].lon, 3.0, 2.0)  # 3.6 m from its own source
    recs.append(Record(lat=dup_lat, lon=dup_lon, cls="human", count_estimate=1,
                       notes="the dedup step failed to merge this cluster"))

    ms = evaluate_records(recs, survivors, SIM, radius_m=6.0, minutes=1.0)
    assert ms.get("record_recall", "sim").value == pytest.approx(1.0)
    assert ms.get("record_precision", "sim").value == pytest.approx(0.75), "the duplicate costs precision"
    assert ms.get("record_duplicate_rate", "sim").value == pytest.approx(1.0 / 3.0)
    assert ms.get("record_duplicate_rate_near", "sim").value == pytest.approx(1.0 / 3.0)
    assert ms.get("fp_per_min@record", "sim").value == pytest.approx(1.0)
    assert ms.get("count_mae", "sim").value == 0.0 and ms.get("count_exact_rate", "sim").value == 1.0
    assert ms.get("record_localisation_median_m", "sim").value < 0.01

    # two duplicates on the same survivor: 5 records, 3 matched -> 2/3
    recs.append(Record(lat=dup_lat, lon=dup_lon, cls="human", count_estimate=1))
    ms2 = evaluate_records(recs, survivors, SIM, radius_m=6.0, minutes=1.0)
    assert ms2.get("record_duplicate_rate", "sim").value == pytest.approx(2.0 / 3.0)

    # a record far from everything is a false record, not a duplicate: the "near" variant separates them
    far_lat, far_lon = offset_ne(ORIGIN_LAT, ORIGIN_LON, 5000.0, 5000.0)
    recs.append(Record(lat=far_lat, lon=far_lon, cls="human"))
    ms3 = evaluate_records(recs, survivors, SIM, radius_m=6.0, minutes=1.0)
    assert ms3.get("record_duplicate_rate", "sim").value == pytest.approx(3.0 / 3.0)
    assert ms3.get("record_duplicate_rate_near", "sim").value == pytest.approx(2.0 / 3.0)
    assert dedup_radius_m(2.8) == pytest.approx(2.0 * CE90_FACTOR * 2.8)


def test_buried_survivors_are_excluded_from_recall_and_reported_on_their_own_row():
    """§2.7 / §6.2: counting them as misses understates recall; hiding them would look like a clearance."""
    survivors = _survivors(3, buried=2)
    assert len([s for s in survivors if not s.findable]) == 2
    recs = [Record(lat=s.lat, lon=s.lon, cls="human") for s in survivors if s.findable]
    ms = evaluate_records(recs, survivors, SIM, radius_m=6.0, minutes=1.0)
    assert ms.get("record_recall", "sim").value == pytest.approx(1.0)
    assert ms.get("record_recall", "sim").n == 3, "the denominator is the findable survivors only"
    buried_row = ms.get("buried_survivors_excluded", "sim")
    assert buried_row.value == 2.0
    assert "R10" in buried_row.detail["guardrail"]
    assert "not visible" in buried_row.detail["basis"]


def test_r10_audit_counts_retained_records_and_unreasoned_dismissals():
    recs = [Record(status="confirmed"), Record(status="dismissed", dismissed_reason="operator: roofing sheet"),
            Record(status="dismissed")]
    ms = r10_rows(recs, SIM)
    assert ms.get("r10_records_retained", "sim").value == 3.0, "no code path deletes a record"
    assert ms.get("r10_dismissals_without_reason", "sim").value == 1.0
    assert "anything above 0 is a defect" in ms.get("r10_dismissals_without_reason", "sim").detail["basis"]


def test_tracking_gain_is_a_ratio_of_two_measured_rows_in_one_domain():
    det = metric_row("fp_per_min@iou0.5", 30.0, SIM, 100)
    rec = metric_row("fp_per_min@record", 2.0, SIM, 10)
    row = tracking_gain_row(det, rec)
    assert row.value == pytest.approx(15.0)
    assert row.detail["expectation"].endswith("10x")
    with pytest.raises(ZeroDivisionError):
        ratio_row("x", det, metric_row("fp_per_min@record", 0.0, SIM, 0))


def test_hungarian_matches_scipy_and_the_gate_drops_far_pairs():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    for shape in ((4, 4), (3, 6), (7, 3), (1, 1), (6, 6)):
        C = rng.uniform(0.0, 10.0, size=shape)
        r, c = hungarian(C)
        r2, c2 = scipy.linear_sum_assignment(C)
        assert C[r, c].sum() == pytest.approx(C[r2, c2].sum())
        assert len(set(r.tolist())) == len(r) and len(set(c.tolist())) == len(c)
    assert hungarian(np.zeros((0, 3)))[0].size == 0
    with pytest.raises(ValueError):
        hungarian(np.array([[np.inf, 1.0], [1.0, 1.0]]))
    gated = gated_assignment(np.array([[1.0, 50.0], [50.0, 2.0]]), gate=5.0)
    assert gated == [(0, 0), (1, 1)]
    assert gated_assignment(np.array([[50.0, 60.0]]), gate=5.0) == []


# ---------------------------------------------------------------------------------------------------------
# 6. geolocation (§5.7) and POD calibration (§5.3, §5.12)
# ---------------------------------------------------------------------------------------------------------
def test_geolocation_error_and_ce90_containment_are_measured_against_the_published_radius():
    survivors = _survivors(4)
    offsets = [1.0, 3.0, 5.0, 11.0]  # metres north of the truth; median 4, p90 ~ 9.2
    recs = []
    for s, d in zip(survivors, offsets):
        lat, lon = offset_ne(s.lat, s.lon, d, 0.0)
        recs.append(Record(lat=lat, lon=lon, h_acc_m=2.8, off_nadir_deg=5.0, agl_m=60.0))
    ms = evaluate_geolocation(samples_from_records(recs), survivors, SIM, match_radius_m=25.0,
                              noise_injected=True)
    assert ms.get("geo_error_median_m", "sim").value == pytest.approx(4.0, abs=0.05)
    assert ms.get("geo_error_max_m", "sim").value == pytest.approx(11.0, abs=0.05)
    assert ms.get("geo_published_ce90_mean_m", "sim").value == pytest.approx(CE90_FACTOR * 2.8)
    # CE90 = 6.0 m, so three of the four fixes fall inside their own published circle
    assert ms.get("geo_ce90_containment", "sim").value == pytest.approx(0.75)
    assert ms.get("geo_ce90_containment", "sim").detail["expected"] == 0.90

    # a sim number measured without the §5.7 noise model must carry the warning, not stand as a claim
    noisy_free = evaluate_geolocation(samples_from_records(recs), survivors, SIM, noise_injected=False)
    assert "not a claim" in noisy_free.get("geo_error_median_m", "sim").detail["warning"]
    # nothing matched is reported as nothing measured, never as zero error
    empty = evaluate_geolocation([], survivors, SIM)
    assert empty.get("geo_error_median_m", "sim").n == 0
    assert "not a zero-error result" in empty.get("geo_error_median_m", "sim").detail["note"]


def test_pod_reliability_diagram_calls_an_honest_map_honest_and_a_dishonest_one_dishonest():
    rng = np.random.default_rng(19)
    p = rng.uniform(0.05, 0.95, size=3000)
    honest = evaluate_pod_calibration(p.tolist(), (rng.random(3000) < p).tolist(), SIM,
                                      presentation="body")
    assert honest.get("pod_ece", "sim").value < 0.05
    assert abs(honest.get("pod_bias", "sim").value) < 0.05
    # each bin is a 95 % interval, so even a perfect map misses one now and then (this seed does);
    # the statistically honest statement is the average over seeds, asserted below.
    assert honest.get("pod_bins_on_diagonal", "sim").value >= 0.6

    liar = evaluate_pod_calibration(p.tolist(), (rng.random(3000) < p * 0.5).tolist(), SIM,
                                    presentation="limb_only")
    assert liar.get("pod_ece", "sim").value > 0.15
    assert liar.get("pod_bias", "sim").value > 0.1, "a positive bias means the map promised more than it found"
    assert "PROMISED more detection than it delivered" in liar.get("pod_bias", "sim").detail["basis"]
    assert all(r.detail["presentation"] == "limb_only" for r in liar)

    on_diagonal = []
    for seed in range(1, 9):
        r = np.random.default_rng(seed)
        q = r.uniform(0.05, 0.95, size=2000)
        ms = evaluate_pod_calibration(q.tolist(), (r.random(2000) < q).tolist(), SIM)
        on_diagonal.append(ms.get("pod_bins_on_diagonal", "sim").value)
    assert float(np.mean(on_diagonal)) >= 0.8, on_diagonal
    assert liar.get("pod_bins_on_diagonal", "sim").value < min(on_diagonal)

    bins = reliability_bins([0.1, 0.15, 0.9, 0.95], [False, False, True, True])
    assert sum(b.n for b in bins) == 4, "the bins partition the samples"
    assert [b.n_found for b in bins] == [0, 2]
    assert all(b.ci_lo <= b.observed <= b.ci_hi for b in bins)
    with pytest.raises(ValueError):
        reliability_bins([1.4], [True])
    empty = evaluate_pod_calibration([], [], SIM)
    assert empty.get("pod_ece", "sim").n == 0
    assert "unmeasured, not perfect" in empty.get("pod_ece", "sim").detail["note"]


# ---------------------------------------------------------------------------------------------------------
# 7. tracking (§5.6) and the posture head (§5.5a)
# ---------------------------------------------------------------------------------------------------------
def test_hota_matches_the_hand_computation_for_one_id_switch():
    """Two tracks over four frames, one ID switch: DetA = 1, AssA = (4 + 2x0.5 + 2x0.5)/8 = 0.75,
    HOTA = sqrt(0.75) = 0.8660254. That is the check that motmetrics is being driven correctly."""
    pytest.importorskip("motmetrics")
    frames = []
    for fi in range(4):
        frames.append(TrackFrame(
            frame_idx=fi,
            gt_ids=[1, 2], gt_boxes=[(0, 0, 10, 10), (100, 100, 110, 110)],
            hyp_ids=[11, 12 if fi < 2 else 13], hyp_boxes=[(0, 0, 10, 10), (100, 100, 110, 110)]))
    ms = evaluate_tracking(frames, SIM)
    assert ms.get("deta", "sim").value == pytest.approx(1.0)
    assert ms.get("assa", "sim").value == pytest.approx(0.75)
    assert ms.get("hota", "sim").value == pytest.approx(math.sqrt(0.75), abs=1e-6)
    assert ms.get("id_switches", "sim").value == 1.0
    assert ms.get("track_gt_detections", "sim").value == 8.0
    assert ms.get("track_gt_identities", "sim").value == 2.0
    assert all(r.slice.domain == "sim" for r in ms)


def test_posture_head_verdict_and_the_promotion_safety_rule():
    """§5.5a: the head stays on only if it improves the ORDERING, and it may never demote a record."""
    urgency = ["stranded", "trapped", "immersed", "stranded", "immersed"]
    truth = [URGENCY_ORDER[u] for u in urgency]
    good = RankingComparison.from_urgency_classes(
        score_with=[float(v) for v in truth],            # perfectly ordered
        score_without=[1.0, 1.0, 1.0, 1.0, 1.0],         # no ordering at all
        urgency_classes=urgency, record_ids=[f"r{i}" for i in range(5)])
    rows = rank_correlation_rows(good, SIM)
    assert rows.get("rank_corr_spearman_with_head", "sim").value == pytest.approx(1.0)
    assert rows.get("rank_corr_spearman_without_head", "sim").value == 0.0
    assert rows.get("rank_corr_spearman_delta", "sim").value == pytest.approx(1.0)
    assert rows.get("posture_head_improves_ranking", "sim").value == 1.0
    assert "keep the head on" in rows.get("posture_head_improves_ranking", "sim").detail["verdict"]

    bad = RankingComparison.from_urgency_classes(
        score_with=[float(-v) for v in truth], score_without=[float(v) for v in truth],
        urgency_classes=urgency, record_ids=[f"r{i}" for i in range(5)])
    verdict = rank_correlation_rows(bad, SIM).get("posture_head_improves_ranking", "sim")
    assert verdict.value == 0.0 and "SWITCH THE HEAD OFF" in verdict.detail["verdict"]

    audit = promotion_audit_rows(
        RankingComparison(score_with_head=[3.0, 1.0, 2.0], score_without_head=[1.0, 1.0, 5.0],
                          truth_urgency=[2.0, 2.0, 2.0], record_ids=["a", "b", "c"]), SIM)
    assert audit.get("posture_demotions", "sim").value == 1.0, "record c was scored LOWER with the head"
    assert audit.get("posture_demotions", "sim").detail["offending"] == ["c"]
    assert audit.get("posture_promotions", "sim").value == 1.0


def test_attribute_accuracy_is_sliced_by_pixel_size_and_by_class():
    samples = [AttributeSample(gt="prone", pred="prone", size_px=100.0),
               AttributeSample(gt="prone", pred="standing", size_px=15.0),
               AttributeSample(gt="standing", pred="standing", size_px=15.0),
               AttributeSample(gt="standing", pred="standing", size_px=50.0)]
    ms = attribute_accuracy_rows(samples, SIM, attribute="posture")
    overall = next(r for r in ms if r.detail.get("axis") is None)
    assert overall.value == pytest.approx(0.75) and overall.n == 4
    by_px = {r.slice.pixel_size: r for r in ms if r.detail.get("axis") == "pixel_size"}
    assert by_px["<20"].value == pytest.approx(0.5) and by_px["<20"].n == 2
    assert by_px["80+"].value == 1.0 and by_px["40-80"].value == 1.0
    assert sum(r.n for r in ms if r.detail.get("axis") == "pixel_size") == 4, "bins partition the samples"
    per_class = {r.detail["gt_class"]: r for r in ms if r.detail.get("axis") == "class"}
    assert per_class["prone"].value == pytest.approx(0.5) and per_class["standing"].value == 1.0
    assert per_class["prone"].slice.posture == "prone"
    assert attribute_accuracy_rows([], SIM).rows[0].n == 0


# ---------------------------------------------------------------------------------------------------------
# 8. the harness end to end, and the report
# ---------------------------------------------------------------------------------------------------------
def test_harness_measures_at_the_frozen_threshold_and_labels_where_it_was_frozen():
    val_ds, val_preds = _pr_clip(n=25, hits=25, split="val")
    op = freeze_on_validation(val_ds, val_preds)

    test_ds, test_preds = _pr_clip(n=25, hits=25, split="test")
    test_ds.clip_id = "held-out"
    res = run_detection_eval(test_ds, test_preds, op.conf, operating=op)
    assert res.domain == "sim"
    headline = res.metrics.overall("sim").get("recall@iou0.5", "sim")
    assert headline.value == pytest.approx(0.92), "measured at the frozen threshold, not re-tuned"
    assert headline.detail["at_conf"] == pytest.approx(0.77)
    assert res.metrics.get("operating_conf", "sim").value == pytest.approx(0.77)
    assert res.metrics.get("operating_recall", "sim").detail["frozen_on"] == "pr[val]"
    assert res.metrics.get("frames_processed", "sim").value == 25.0
    assert res.metrics.get("domain_randomisation_on", "sim").value == 0.0
    # the operating rows say out loud that they are not the test clip's numbers
    assert "not the test clip" in res.metrics.get("operating_recall", "sim").detail["measured_on"]

    # a threshold supplied with no validation curve is re-measured here AND says so
    solo = run_detection_eval(test_ds, test_preds, 0.775)
    assert solo.operating.recall == pytest.approx(
        solo.metrics.overall("sim").get("recall@iou0.5", "sim").value)
    assert "re-measured on THIS clip" in solo.operating.note


def test_harness_record_stage_and_failure_clusters():
    sc = make_scenario(seed=9, n_survivors=8, n_frames=40)
    res = run_detection_eval(sc.dataset, sc.detections, 0.3, EvalConfig())
    res = add_record_eval(res, sc.records, radius_m=8.0)
    names = set(res.metrics.names())
    assert {"recall@iou0.5", "recall@iou0.25", "record_precision", "record_recall",
            "record_duplicate_rate", "buried_survivors_excluded", "r10_records_retained"} <= names
    assert res.metrics.get("buried_survivors_excluded", "sim").value == float(sc.truth["n_buried"])
    assert len(res.failure_clusters) > 0
    assert set(res.failure_clusters[0]) == {"submersion", "zone", "altitude", "n_missed", "median_size_px"}
    assert sum(c["n_missed"] for c in res.failure_clusters) <= len(res.matches["0.5"].gt_boxes)
    # the whole-clip recall must equal a hand count over the match result
    r = res.matches["0.5"]
    assert res.metrics.overall("sim").get("recall@iou0.5", "sim").value == pytest.approx(
        float(r.gt_found(0.3).sum() + r.group_found(0.3).sum()) / r.n_targets)


def test_report_carries_the_domain_on_every_table_and_refuses_to_pool(tmp_path):
    sim = make_scenario(seed=2, n_survivors=5, n_frames=20, clip_id="sim-clip")
    real = make_scenario(seed=2, n_survivors=5, n_frames=20, domain="real", clip_id="real-clip")
    a = add_record_eval(run_detection_eval(sim.dataset, sim.detections, 0.3), sim.records)
    b = add_record_eval(run_detection_eval(real.dataset, real.detections, 0.3), real.records)

    md = render_markdown([a, b])
    assert "## Domain: sim (in simulation)" in md
    assert "## Domain: real (on real footage)" in md
    assert "Do not compute a combined figure from them" in md
    # every metric table starts with a domain column, and every data row under one starts with a domain
    lines = md.splitlines()
    tables = 0
    for i, line in enumerate(lines):
        if not line.startswith("| domain | metric |"):
            continue
        tables += 1
        for row in lines[i + 2:]:
            if not row.startswith("|"):
                break
            assert row.startswith(("| sim |", "| real |")), row
    assert tables >= 4, "the report is more than one table"
    assert "in simulation" in md and "on real footage" in md
    # the acceptance sentence cannot be quoted without the domain word attached to it
    for head in [x for x in lines if x.startswith("- **")]:
        assert "in simulation" in head or "on real footage" in head, head

    path = write_report([a, b], out_dir=tmp_path, stem="unit")
    assert path.exists() and (tmp_path / "unit.json").exists()
    payload = json.loads((tmp_path / "unit.json").read_text(encoding="utf-8"))
    assert {r["domain"] for r in payload["rows"]} == {"sim", "real"}
    assert all("domain" in r for r in payload["rows"])
    assert [run["domain"] for run in payload["runs"]] == ["sim", "real"]


def test_fiftyone_manifest_is_plain_json_and_tags_the_errors():
    sc = make_scenario(seed=4, n_survivors=5, n_frames=16)
    res = match_dataset(sc.dataset, sc.detections, iou_thr=0.5)
    man = build_manifest(sc.dataset, res, 0.3, name="unit")
    assert man["schema"] == "sightline-fiftyone-manifest/1" and man["domain"] == "sim"
    assert len(man["samples"]) == sc.dataset.n_frames
    assert json.dumps(man), "the bridge to the fiftyone env is a file, so it must be plain JSON"
    for s in man["samples"]:
        assert s["n_fp"] == sum(1 for p in s["predictions"] if p["eval"] == "fp")
        assert s["n_fn"] == sum(1 for g in s["ground_truth"] if g["eval"] == "fn")
        assert ("has_fp" in s["tags"]) == (s["n_fp"] > 0)
        for g in s["ground_truth"]:
            x, y, w, h = g["bounding_box"]
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0 and h > 0, "FiftyOne wants relative boxes"
    clusters = failure_clusters(res, 0.3)
    assert clusters == sorted(clusters, key=lambda c: -c["n_missed"])


def test_ground_truth_serialises_and_the_split_guard_catches_leakage():
    sc = make_scenario(seed=6, n_survivors=4, n_frames=8)
    round_trip = dataset_from_json(json.loads(json.dumps(dataset_to_json(sc.dataset))))
    assert round_trip.n_frames == sc.dataset.n_frames
    assert len(round_trip.scored_boxes()) == len(sc.dataset.scored_boxes())
    assert [s.gt_id for s in round_trip.findable_survivors()] == \
        [s.gt_id for s in sc.dataset.findable_survivors()]
    assert len(round_trip.buried_survivors()) == sc.truth["n_buried"]

    train = EvalDataset(domain="sim", seed_group="seed-7", split="train")
    test = EvalDataset(domain="sim", seed_group="seed-7", split="test")
    assert check_split_disjoint([train, test]) == {"seed-7": ["test", "train"]}, "§6.5: split by seed, not frame"
    assert check_split_disjoint([train, EvalDataset(domain="sim", seed_group="seed-8", split="test")]) == {}

    with pytest.raises(ValueError):
        GtBox(bbox_px=(10.0, 10.0, 5.0, 5.0))
    with pytest.raises(ValueError):
        GtBox(bbox_px=(0.0, 0.0, 5.0, 5.0), posture="floating")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GtBox(bbox_px=(0.0, 0.0, 5.0, 5.0), occlusion=7)


def test_the_synthetic_scenario_is_self_consistent():
    """The fixture the rest of this file leans on has to be what it claims to be."""
    sc = make_scenario(seed=7, n_survivors=10, n_frames=40, n_buried=2, duplicate_records=1)
    assert len(sc.dataset.buried_survivors()) == 2
    assert all(s.findable for s in sc.dataset.findable_survivors())
    assert not any(b.gt_id in {s.gt_id for s in sc.dataset.buried_survivors()}
                   for f in sc.dataset.frames for b in f.boxes), "a buried actor is never drawn as visible"
    assert sc.truth["n_gt_boxes"] == len(sc.dataset.scored_boxes())
    assert len(sc.detections) > 0 and all(d.frame_idx >= 0 for d in sc.detections)
    assert all(len(t.observations) >= 3 for t in sc.tracks), "§5.6 rule 3: >= 3 hits to confirm"
    assert make_scenario(seed=7).truth == make_scenario(seed=7).truth, "the fixture is deterministic"
    a, b = make_scenario(seed=7), make_scenario(seed=7)
    assert [d.bbox_px for d in a.detections] == [d.bbox_px for d in b.detections]


# ---------------------------------------------------------------------------------------------------------
# An unmeasurable metric must not be rendered as a measured zero (AUDIT S8, reporting half)
# ---------------------------------------------------------------------------------------------------------
def test_an_undefined_metric_never_renders_as_a_measured_zero():
    """`MetricRow.undefined` pins `value = 0.0` so the row survives `json.dumps(allow_nan=False)`.

    That neutral zero is a serialisation device. If it reaches a reader unguarded, the report says
    "**0.0 % nominal-slice recall, in simulation**" about a slice in which nothing was measured — which is a
    statement that the system found nobody, not that it was never asked. That is the more damaging direction
    to be wrong in, and it is the failure this test exists to prevent.
    """
    from sightline.eval.report import _headline, _value_str
    from sightline.schemas import MetricRow

    key = make_slice("sim")
    undefined = MetricRow.undefined("recall@nominal", key, "no ground-truth boxes in this slice")
    ms = MetricSet()
    ms.add(undefined)

    line = _headline(ms, "sim")[0]
    assert "0.0 %" not in line, f"an unmeasurable slice was rendered as a measured zero: {line}"
    assert "undefined" in line
    assert "no ground-truth boxes in this slice" in line, "the reason must travel with the absence"
    # 5.5c: every recall figure carries its domain in the same sentence — including a missing one.
    assert "in simulation" in line

    assert _value_str(undefined).startswith("undefined (")
    assert _value_str(MetricRow(name="recall@nominal", value=0.94, slice=key, n=200)) == "0.94"


def test_a_measured_zero_is_still_reported_as_a_measured_zero():
    """The guard must not swallow a real 0.0 — a detector that genuinely found nothing has to say so."""
    from sightline.eval.report import _headline, _value_str
    from sightline.schemas import MetricRow

    key = make_slice("sim")
    measured = MetricRow(name="recall@nominal", value=0.0, slice=key, n=57)
    ms = MetricSet()
    ms.add(measured)

    line = _headline(ms, "sim")[0]
    assert "0.0 % nominal-slice recall at IoU 0.5, in simulation" in line
    assert "undefined" not in line
    assert "n = 57" in line
    assert _value_str(measured) == "0"
