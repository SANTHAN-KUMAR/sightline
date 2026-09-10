"""The evaluation harness: one call that produces every §5.12 number as sliced `MetricRow`s.

Order of operations, and why:

1. **Freeze the operating threshold on the VALIDATION split** (§5.5). `run_detection_eval` takes an explicit
   `operating_conf`; `freeze_on_validation()` is the only thing allowed to choose one, and it refuses to do so
   on a split that is not `"val"` unless the caller says so out loud. Tuning the threshold on the test clip and
   then reporting recall on the same clip is the most common way an evaluation lies to itself.
2. **Measure at that threshold** on the test clip: recall at IoU 0.5 and 0.25, precision, FP/min.
3. **Slice everything** (hard rule 5). The nominal slice of §5.5c is computed explicitly and separately from
   the hard slices, because the acceptance figure is a claim about a stated slice, not about the whole clip.
4. **Record level after tracking and dedup**, with FP/min reported a second time so the ratio is visible.

Nothing here imports another lane's module: the harness consumes `Detection`, `Track` and `Record` — the frozen
`schemas.py` types — so it runs against whatever produced them, including hand-built synthetic predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sightline.eval.detection import (
    IOU_PRIMARY,
    IOU_RELAXED,
    TARGET_RECALL,
    MatchResult,
    OperatingPoint,
    PrCurve,
    false_positives,
    freeze_operating_threshold,
    headline_rows,
    match_dataset,
    missed_detections,
    recall_vs_pixel_height,
    slice_rows,
    sweep_confidence,
)
from sightline.eval.fiftyone_export import failure_clusters
from sightline.eval.groundtruth import EvalDataset
from sightline.eval.records import evaluate_records, r10_rows, tracking_gain_row
from sightline.eval.slicing import (
    BOX_AXES,
    FRAME_AXES,
    MetricSet,
    make_slice,
    metric_row,
)
from sightline.schemas import Detection, Record, SliceKey

#: §5.5c step 5: "the headline claim is recall on a stated slice: 40-60 m, daylight, occlusion below 50 %,
#: non-submerged presentations". `occlusion <= 1` is used because `schemas.OCCLUSION_BINS` puts bin 1 at
#: 25-75 % hidden, so the nominal slice here is slightly MORE permissive than the doc's wording — a
#: conservative direction for the claim, and stated in the report rather than left implicit.
NOMINAL = {
    "agl_m": (40.0, 60.0),
    "time_of_day": ("day",),
    "occlusion_max": 1,
    "submersion": ("dry", "wet", "unknown"),
}


@dataclass(slots=True)
class EvalConfig:
    iou_primary: float = IOU_PRIMARY
    iou_relaxed: float = IOU_RELAXED
    target_recall: float = TARGET_RECALL
    cls: str = "human"
    record_radius_m: float = 6.0  # §5.7: CE90 ~ 6 m at 60 m nadir with consumer GNSS
    geo_match_radius_m: float = 25.0
    slice_axes: tuple[str, ...] = BOX_AXES + FRAME_AXES
    pixel_height_edges: tuple[float, ...] = (0, 8, 12, 16, 20, 30, 40, 60, 80, 120, 1e9)


@dataclass(slots=True)
class EvalResult:
    """Everything one clip produced. `metrics` is the only thing a report is allowed to print numbers from."""

    dataset: EvalDataset
    config: EvalConfig
    metrics: MetricSet = field(default_factory=MetricSet)
    matches: dict[str, MatchResult] = field(default_factory=dict)
    curves: dict[str, PrCurve] = field(default_factory=dict)
    operating: OperatingPoint | None = None
    failure_clusters: list[dict[str, Any]] = field(default_factory=list)
    missed: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def domain(self) -> str:
        return self.dataset.domain

    def base_key(self) -> SliceKey:
        return make_slice(self.dataset.domain)


def freeze_on_validation(
    val: EvalDataset,
    detections: Sequence[Detection],
    cfg: EvalConfig | None = None,
    *,
    allow_non_val_split: bool = False,
) -> OperatingPoint:
    """§5.5: sweep confidence on the validation split, take the highest confidence with recall >= target.

    Refuses a non-`val` split unless the caller opts in, so a threshold cannot be silently tuned on test data.
    """
    cfg = cfg or EvalConfig()
    if val.split != "val" and not allow_non_val_split:
        raise ValueError(
            f"the operating threshold must be frozen on the validation split (SOLUTION_DOC 5.5); this dataset "
            f"is split={val.split!r}. Pass allow_non_val_split=True only if you mean it and will say so in the "
            "report."
        )
    res = match_dataset(val, detections, iou_thr=cfg.iou_primary, cls=cfg.cls)
    curve = sweep_confidence(res, val.domain)
    op = freeze_operating_threshold(curve, cfg.target_recall)
    op.frozen_on_split = val.split
    op.frozen_on_clip = val.clip_id
    return op


def nominal_masks(ds: EvalDataset, res: MatchResult) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(gt_mask, pred_mask, description) for the §5.5c nominal slice."""
    lo, hi = NOMINAL["agl_m"]
    gt = np.asarray([
        (lo <= f.agl_m <= hi)
        and (str(f.time_of_day).lower() in NOMINAL["time_of_day"] or _tod(f.time_of_day) in NOMINAL["time_of_day"])
        and (b.occlusion is None or b.occlusion <= NOMINAL["occlusion_max"])
        and (str(b.submersion) in NOMINAL["submersion"])
        for b, f in zip(res.gt_boxes, res.gt_frames)
    ], dtype=bool) if res.gt_boxes else np.zeros(0, dtype=bool)
    frame_ok = {
        f.frame_idx: (lo <= f.agl_m <= hi) and _tod(f.time_of_day) in NOMINAL["time_of_day"]
        for f in ds.frames
    }
    pred = np.asarray([frame_ok.get(int(i), False) for i in res.pred_frame], dtype=bool)
    n_frames = sum(1 for f in ds.frames if frame_ok.get(f.frame_idx, False))
    return gt, pred, {
        "definition": "40-60 m AGL, daylight, occlusion bin <= 1, non-submerged presentations",
        "source": "SOLUTION_DOC 5.5c step 5",
        "note": "schemas.OCCLUSION_BINS bin 1 is 25-75 % hidden, so this is slightly more permissive than "
                "the doc's 'occlusion below 50 %'",
        "n_frames": n_frames,
    }


def _tod(value: str) -> str:
    from sightline.eval.slicing import time_of_day_bin

    try:
        return time_of_day_bin(value)
    except Exception:
        return "day"


def run_detection_eval(
    ds: EvalDataset,
    detections: Sequence[Detection],
    operating_conf: float,
    cfg: EvalConfig | None = None,
    *,
    operating: OperatingPoint | None = None,
) -> EvalResult:
    """Detection-level §5.12 numbers at a FROZEN threshold, sliced. `operating_conf` comes from validation."""
    cfg = cfg or EvalConfig()
    out = EvalResult(dataset=ds, config=cfg)
    key = out.base_key()

    for iou in (cfg.iou_primary, cfg.iou_relaxed):
        res = match_dataset(ds, detections, iou_thr=iou, cls=cfg.cls)
        out.matches[f"{iou:g}"] = res
        out.curves[f"{iou:g}"] = sweep_confidence(res, ds.domain)
        out.metrics.extend(headline_rows(res, operating_conf, key))

    primary = out.matches[f"{cfg.iou_primary:g}"]
    curve = out.curves[f"{cfg.iou_primary:g}"]

    if operating is None:
        r, p, f = curve.at(operating_conf)
        operating = OperatingPoint(conf=operating_conf, recall=r, precision=p, fp_per_min=f,
                                   target_recall=cfg.target_recall, achieved=r >= cfg.target_recall,
                                   iou_thr=cfg.iou_primary, domain=ds.domain,
                                   max_f1_conf=curve.max_f1_conf(), n_gt=curve.n_gt,
                                   frozen_on_split=ds.split, frozen_on_clip=ds.clip_id,
                                   note="threshold supplied by the caller; the operating_* rows below were "
                                        "re-measured on THIS clip because no validation curve was passed in")
    out.operating = operating
    out.metrics.extend(operating.rows(key))

    # Show the number a library summary would have quoted, so the difference is on the record, not hidden.
    mf1 = curve.max_f1_conf()
    r_f1, p_f1, fp_f1 = curve.at(mf1)
    out.metrics.add(metric_row("recall@max_f1_conf", r_f1, key, curve.n_gt, at_conf=mf1,
                               warning="NOT the operating point; Ultralytics quotes P/R here (SOLUTION_DOC 5.5)"))
    out.metrics.add(metric_row("precision@max_f1_conf", p_f1, key, curve.n_gt, at_conf=mf1))
    out.metrics.add(metric_row("fp_per_min@max_f1_conf", fp_f1, key, curve.n_gt, at_conf=mf1))

    out.metrics.extend(slice_rows(ds, primary, operating_conf, key, cfg.slice_axes))
    out.metrics.extend(recall_vs_pixel_height(primary, operating_conf, key, cfg.pixel_height_edges))

    # the §5.5c nominal slice: the acceptance figure, stated as a claim about a stated slice
    gt_mask, pred_mask, desc = nominal_masks(ds, primary)
    if gt_mask.any():
        c = primary.counts_at(operating_conf, gt_mask=gt_mask, pred_mask=pred_mask)
        c.minutes = desc["n_frames"] / ds.fps_processed / 60.0 if desc["n_frames"] else 0.0
        out.metrics.add(metric_row("recall@nominal", c.recall(), key, c.n_gt, at_conf=operating_conf,
                                   iou_thr=cfg.iou_primary, tp=c.tp, fn=c.fn, **desc))
        out.metrics.add(metric_row("precision@nominal", c.precision(), key, c.tp + c.fp,
                                   at_conf=operating_conf, iou_thr=cfg.iou_primary, **desc))
        if c.minutes > 0:
            out.metrics.add(metric_row("fp_per_min@nominal", c.fp_per_min(), key, c.fp,
                                       at_conf=operating_conf, level="detection", **desc))
        relaxed = out.matches[f"{cfg.iou_relaxed:g}"]
        gt_mask_r, pred_mask_r, _ = nominal_masks(ds, relaxed)
        cr = relaxed.counts_at(operating_conf, gt_mask=gt_mask_r, pred_mask=pred_mask_r)
        out.metrics.add(metric_row("recall@nominal@iou0.25", cr.recall(), key, cr.n_gt,
                                   at_conf=operating_conf, iou_thr=cfg.iou_relaxed, **desc))
    else:
        out.metrics.add(metric_row("recall@nominal", 0.0, key, 0,
                                   note="no ground truth falls in the nominal slice; the acceptance figure "
                                        "cannot be claimed from this clip", **desc))

    # §5.5c hard rule 7: which randomisation setting produced this number
    out.metrics.add(metric_row("domain_randomisation_on", 1.0 if ds.randomisation else 0.0, key, ds.n_frames,
                               basis="SOLUTION_DOC 5.5c: never compare numbers across this switch"))
    out.metrics.add(metric_row("frames_processed", float(ds.n_frames), key, ds.n_frames,
                               fps_processed=ds.fps_processed, minutes=ds.minutes_processed,
                               split=ds.split, seed_group=ds.seed_group, clip_id=ds.clip_id))

    out.failure_clusters = failure_clusters(primary, operating_conf)
    out.missed = missed_detections(primary, operating_conf)
    out.false_positives = false_positives(primary, operating_conf)
    return out


def add_record_eval(
    result: EvalResult,
    records: Sequence[Record],
    *,
    radius_m: float | None = None,
) -> EvalResult:
    """Record-level metrics plus the FP/min ratio that measures what tracking + dedup bought (§5.6)."""
    cfg = result.config
    ds = result.dataset
    key = result.base_key()
    radius = cfg.record_radius_m if radius_m is None else radius_m
    result.metrics.extend(evaluate_records(records, ds.survivors, key, radius_m=radius,
                                           minutes=ds.minutes_processed, cls=cfg.cls))
    result.metrics.extend(r10_rows(records, key))
    try:
        det = result.metrics.get(f"fp_per_min@iou{cfg.iou_primary:g}", ds.domain)
        rec = result.metrics.get("fp_per_min@record", ds.domain)
        if rec.value > 0:
            result.metrics.add(tracking_gain_row(det, rec))
        else:
            # The ratio is undefined with a zero denominator; report what was actually removed instead of inf.
            result.metrics.add(metric_row("fp_per_min_eliminated", det.value, key, det.n,
                                          basis="raw detection FP/min minus record FP/min",
                                          note="record-level FP/min is 0: tracking + dedup removed every false "
                                               "positive on this clip, so the ratio is undefined"))
    except KeyError:
        pass
    return result


__all__ = ["NOMINAL", "EvalConfig", "EvalResult", "add_record_eval", "freeze_on_validation", "nominal_masks",
           "run_detection_eval"]
