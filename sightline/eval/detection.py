"""Detection-level metrics: recall @ IoU 0.5 and 0.25, precision, FP/min, and the frozen operating threshold.

The operating-threshold rule is §5.5, quoted so it is not paraphrased away:

    "Sweep confidence on the validation split, pick the highest confidence at which human recall >= 0.92 (a
    2-point margin over the 0.90 target), freeze it, and report precision and FP/min *at that threshold*.
    Ultralytics' reported per-class P/R are taken at the max-F1 confidence, not at your operating point."

So the sweep is implemented here, from raw matches, and never read off a library summary. `MatchResult` below
holds one score per ground-truth box (the score of the prediction that found it) and one outcome per
prediction; every number in this module is a count over those two arrays at a threshold, which is what makes
the per-slice recalls partition exactly.

FP/min is the §5.12 definition: `FP_total / (frames_processed / fps_processed / 60)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from sightline.eval.groundtruth import EvalDataset, GtBox, GtFrame
from sightline.eval.matching import match_frame
from sightline.eval.slicing import (
    AXIS_DEFAULT,
    BOX_AXES,
    FRAME_AXES,
    MetricSet,
    altitude_band,
    make_slice,
    metric_row,
    narrow,
    occlusion_label,
    pixel_size_bin,
    posture_label,
    time_of_day_bin,
)
from sightline.schemas import Detection, MetricRow, SliceKey

IOU_PRIMARY = 0.5
IOU_RELAXED = 0.25  # §5.5: the "found the person" metric — on a limb-only target the box extent is ill-defined
TARGET_RECALL = 0.92  # §5.5: 2-point margin over the 0.90 requirement

NEVER = -math.inf  # the "match score" of a ground-truth box no prediction ever found


@dataclass(slots=True)
class DetCounts:
    """RAW COUNTS at one threshold. Never report these directly — they carry no slice. `*_rows()` does."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_gt: int = 0
    n_pred: int = 0
    ignored: int = 0
    minutes: float = 0.0

    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def fp_per_min(self) -> float:
        if self.minutes <= 0:
            raise ValueError("FP/min needs a positive processed duration (frames / fps / 60)")
        return self.fp / self.minutes


@dataclass(slots=True)
class MatchResult:
    """One matching pass over a whole clip at one IoU threshold, from which any threshold can be scored."""

    iou_thr: float
    cls: str
    minutes: float
    n_frames: int
    # scored, non-group ground-truth boxes
    gt_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gt_local: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    gt_match_score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gt_match_iou: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gt_boxes: list[GtBox] = field(default_factory=list)
    gt_frames: list[GtFrame] = field(default_factory=list)
    # §6.3 group boxes: one recall target each, satisfied by any prediction at IoD >= 0.5
    group_hit_score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    group_boxes: list[GtBox] = field(default_factory=list)
    group_frames: list[GtFrame] = field(default_factory=list)
    # predictions
    pred_score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pred_outcome: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))
    pred_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    pred_gt: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # index into gt_* or -1
    detections: list[Detection] = field(default_factory=list)

    # --- counting -------------------------------------------------------------------------------------
    @property
    def n_targets(self) -> int:
        return len(self.gt_boxes) + len(self.group_boxes)

    def gt_found(self, conf: float) -> np.ndarray:
        return self.gt_match_score >= conf

    def group_found(self, conf: float) -> np.ndarray:
        return self.group_hit_score >= conf

    def counts_at(self, conf: float, gt_mask: np.ndarray | None = None,
                  pred_mask: np.ndarray | None = None) -> DetCounts:
        """Counts at threshold `conf`. `gt_mask`/`pred_mask` restrict to a slice.

        A box-level slice restricts the ground truth only: a false positive has no ground-truth box and so has
        no occlusion, posture or pixel-size bin. `slice_rows()` therefore emits precision/FP-per-min for
        frame-level axes only, which is the honest thing to do (§5.12).
        """
        gm = np.ones(len(self.gt_boxes), dtype=bool) if gt_mask is None else np.asarray(gt_mask, dtype=bool)
        pm = np.ones(len(self.pred_score), dtype=bool) if pred_mask is None else np.asarray(pred_mask, dtype=bool)
        keep = (self.pred_score >= conf) & pm
        found = self.gt_found(conf) & gm
        tp_boxes = int(found.sum())
        n_gt = int(gm.sum())
        if gt_mask is None:
            grp = self.group_found(conf)
            tp_boxes += int(grp.sum())
            n_gt += len(self.group_hit_score)
        fp = int((keep & (self.pred_outcome == "fp")).sum())
        return DetCounts(
            tp=tp_boxes,
            fp=fp,
            fn=n_gt - tp_boxes,
            n_gt=n_gt,
            n_pred=int(keep.sum()),
            ignored=int((keep & (self.pred_outcome == "ignored")).sum()),
            minutes=self.minutes,
        )

    def candidate_thresholds(self) -> np.ndarray:
        """Every confidence at which the counts can change, descending. Recall is a step function on these."""
        if not len(self.pred_score):
            return np.asarray([1.0])
        return np.unique(self.pred_score)[::-1]


def _frame_key(ds: EvalDataset, frame: GtFrame, base: SliceKey | None = None) -> SliceKey:
    key = base or make_slice(ds.domain)
    return narrow(
        key,
        zone=str(frame.zone),
        altitude_band=altitude_band(frame.agl_m),
        time_of_day=time_of_day_bin(frame.time_of_day),
        modality=str(frame.modality),
    )


def match_dataset(
    ds: EvalDataset,
    detections: Iterable[Detection],
    *,
    iou_thr: float = IOU_PRIMARY,
    cls: str = "human",
) -> MatchResult:
    """Match every prediction in the clip once, at one IoU threshold. Threshold-independent by construction."""
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        if cls is not None and d.cls != cls:
            continue
        by_frame.setdefault(d.frame_idx, []).append(d)

    res = MatchResult(iou_thr=float(iou_thr), cls=cls, minutes=ds.minutes_processed, n_frames=ds.n_frames)
    gt_frame, gt_local, gt_score, gt_iou = [], [], [], []
    grp_score: list[float] = []
    pred_score, pred_outcome, pred_frame, pred_gt = [], [], [], []

    for frame in ds.frames:
        preds = by_frame.get(frame.frame_idx, [])
        boxes = [b for b in frame.boxes if cls is None or b.cls == cls or not b.scored]
        local_scored = [i for i, b in enumerate(boxes) if b.scored and not b.is_group]
        local_group = [i for i, b in enumerate(boxes) if b.scored and b.is_group]
        base_gt = len(res.gt_boxes)
        base_grp = len(res.group_boxes)
        for i in local_scored:
            res.gt_boxes.append(boxes[i])
            res.gt_frames.append(frame)
            gt_frame.append(frame.frame_idx)
            gt_local.append(i)
            gt_score.append(NEVER)
            gt_iou.append(0.0)
        for i in local_group:
            res.group_boxes.append(boxes[i])
            res.group_frames.append(frame)
            grp_score.append(NEVER)

        m = match_frame([p.bbox_px for p in preds], [p.score for p in preds], boxes, iou_thr)
        scored_pos = {orig: k for k, orig in enumerate(local_scored)}
        group_pos = {orig: k for k, orig in enumerate(local_group)}
        outcome = ["fp"] * len(preds)
        gt_of_pred = [-1] * len(preds)
        for (pi, gj), iou in zip(m.tp, m.tp_iou):
            k = base_gt + scored_pos[gj]
            outcome[pi] = "tp"
            gt_of_pred[pi] = k
            gt_score[k] = float(preds[pi].score)
            gt_iou[k] = float(iou)
        for pi in m.ignored_pred:
            outcome[pi] = "ignored"
        # a group is satisfied by the highest-scoring prediction that landed on it
        if local_group:
            from sightline.eval.matching import GROUP_IOD_THRESHOLD, iod_matrix

            iod = iod_matrix([p.bbox_px for p in preds], [boxes[i].bbox_px for i in local_group])
            for pi in m.ignored_pred:
                if iod.shape[1] and float(iod[pi].max()) >= GROUP_IOD_THRESHOLD:
                    g = base_grp + group_pos[local_group[int(np.argmax(iod[pi]))]]
                    grp_score[g] = max(grp_score[g], float(preds[pi].score))
        for pi, p in enumerate(preds):
            res.detections.append(p)
            pred_score.append(float(p.score))
            pred_outcome.append(outcome[pi])
            pred_frame.append(frame.frame_idx)
            pred_gt.append(gt_of_pred[pi])

    res.gt_frame = np.asarray(gt_frame, dtype=int)
    res.gt_local = np.asarray(gt_local, dtype=int)
    res.gt_match_score = np.asarray(gt_score, dtype=float)
    res.gt_match_iou = np.asarray(gt_iou, dtype=float)
    res.group_hit_score = np.asarray(grp_score, dtype=float)
    res.pred_score = np.asarray(pred_score, dtype=float)
    res.pred_outcome = np.asarray(pred_outcome, dtype="<U8")
    res.pred_frame = np.asarray(pred_frame, dtype=int)
    res.pred_gt = np.asarray(pred_gt, dtype=int)
    return res


# --- the confidence sweep ----------------------------------------------------------------------------------
@dataclass(slots=True)
class PrCurve:
    """Recall / precision / FP-per-min as a function of the confidence threshold, at one IoU."""

    iou_thr: float
    domain: str
    conf: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    fp_per_min: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    n_gt: int

    def at(self, conf: float) -> tuple[float, float, float]:
        """(recall, precision, fp_per_min) at the highest swept confidence <= `conf`."""
        idx = int(np.searchsorted(self.conf, conf, side="right")) - 1
        idx = max(0, min(idx, len(self.conf) - 1))
        return float(self.recall[idx]), float(self.precision[idx]), float(self.fp_per_min[idx])

    def max_f1_conf(self) -> float:
        """Where Ultralytics would quote P and R. Reported only to show it is NOT the operating point."""
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                (self.precision + self.recall) > 0,
                2 * self.precision * self.recall / np.where((self.precision + self.recall) > 0,
                                                            self.precision + self.recall, 1.0),
                0.0,
            )
        return float(self.conf[int(np.argmax(f1))])


def sweep_confidence(res: MatchResult, domain: str) -> PrCurve:
    """Evaluate every candidate threshold. Ascending in `conf` so `at()` can bisect."""
    thresholds = np.sort(res.candidate_thresholds())
    recall, precision, fppm, tps, fps = [], [], [], [], []
    for c in thresholds:
        counts = res.counts_at(float(c))
        recall.append(counts.recall())
        precision.append(counts.precision())
        fppm.append(counts.fp_per_min())
        tps.append(counts.tp)
        fps.append(counts.fp)
    return PrCurve(
        iou_thr=res.iou_thr,
        domain=domain,
        conf=thresholds,
        recall=np.asarray(recall),
        precision=np.asarray(precision),
        fp_per_min=np.asarray(fppm),
        tp=np.asarray(tps, dtype=int),
        fp=np.asarray(fps, dtype=int),
        n_gt=res.n_targets,
    )


@dataclass(slots=True)
class OperatingPoint:
    """The frozen §5.5 operating threshold and the numbers measured AT it."""

    conf: float
    recall: float
    precision: float
    fp_per_min: float
    target_recall: float
    achieved: bool
    iou_thr: float
    domain: str
    max_f1_conf: float = 0.0
    n_gt: int = 0
    note: str = ""
    # which clip the threshold was frozen on: the `operating_*` numbers are measured THERE, not on the test
    # clip, and a report that does not say so invites the reader to compare two different populations.
    frozen_on_split: str = ""
    frozen_on_clip: str = ""

    @property
    def frozen_on(self) -> str:
        return f"{self.frozen_on_clip or '?'}[{self.frozen_on_split or '?'}]"

    def rows(self, key: SliceKey) -> MetricSet:
        common = {"iou_thr": self.iou_thr, "frozen_on": self.frozen_on,
                  "measured_on": "the split the threshold was frozen on, not the test clip"}
        ms = MetricSet()
        ms.add(metric_row("operating_conf", self.conf, key, self.n_gt, target_recall=self.target_recall,
                          achieved=self.achieved, max_f1_conf=self.max_f1_conf, note=self.note, **common))
        ms.add(metric_row("operating_recall", self.recall, key, self.n_gt, at_conf=self.conf,
                          achieved=self.achieved, **common))
        ms.add(metric_row("operating_precision", self.precision, key, self.n_gt, at_conf=self.conf, **common))
        ms.add(metric_row("operating_fp_per_min", self.fp_per_min, key, self.n_gt, at_conf=self.conf, **common))
        return ms


def freeze_operating_threshold(curve: PrCurve, target_recall: float = TARGET_RECALL) -> OperatingPoint:
    """Pick the HIGHEST confidence at which recall >= `target_recall` (§5.5) and freeze it.

    Recall is non-increasing in the threshold, so the admissible set is an interval ending at one of the swept
    confidences. If no threshold reaches the target, the lowest swept confidence is returned with
    `achieved=False` and a note: an unreachable target is a finding, not a reason to lower the bar.
    """
    ok = curve.recall >= target_recall
    if not ok.any():
        best = int(np.argmax(curve.recall))
        return OperatingPoint(
            conf=float(curve.conf[best]),
            recall=float(curve.recall[best]),
            precision=float(curve.precision[best]),
            fp_per_min=float(curve.fp_per_min[best]),
            target_recall=target_recall,
            achieved=False,
            iou_thr=curve.iou_thr,
            domain=curve.domain,
            max_f1_conf=curve.max_f1_conf(),
            n_gt=curve.n_gt,
            note=f"recall never reached {target_recall:.2f}; best is {curve.recall[best]:.4f} at the lowest "
                 "swept confidence. Fly lower or retrain (SOLUTION_DOC 5.5c step 1) rather than moving the target.",
        )
    i = int(np.max(np.nonzero(ok)[0]))  # conf is ascending: the last index that still meets the target
    return OperatingPoint(
        conf=float(curve.conf[i]),
        recall=float(curve.recall[i]),
        precision=float(curve.precision[i]),
        fp_per_min=float(curve.fp_per_min[i]),
        target_recall=target_recall,
        achieved=True,
        iou_thr=curve.iou_thr,
        domain=curve.domain,
        max_f1_conf=curve.max_f1_conf(),
        n_gt=curve.n_gt,
    )


# --- rows --------------------------------------------------------------------------------------------------
def headline_rows(res: MatchResult, conf: float, key: SliceKey, suffix: str = "") -> MetricSet:
    """recall / precision / FP-per-min / F1 at one threshold, one IoU, one slice."""
    c = res.counts_at(conf)
    tag = f"@iou{res.iou_thr:g}{suffix}"
    ms = MetricSet()
    ms.add(metric_row(f"recall{tag}", c.recall(), key, c.n_gt, tp=c.tp, fn=c.fn, at_conf=conf))
    ms.add(metric_row(f"precision{tag}", c.precision(), key, c.tp + c.fp, tp=c.tp, fp=c.fp, at_conf=conf))
    ms.add(metric_row(f"fp_per_min{tag}", c.fp_per_min(), key, c.fp, minutes=c.minutes, at_conf=conf,
                      level="detection"))
    ms.add(metric_row(f"f1{tag}", c.f1(), key, c.n_gt, at_conf=conf))
    return ms


def _box_axis_label(box: GtBox, frame: GtFrame, axis: str) -> str:
    if axis == "occlusion":
        return occlusion_label(box.occlusion)
    if axis == "posture":
        return posture_label(str(box.posture))
    if axis == "pixel_size":
        return pixel_size_bin(box.size_px)
    raise ValueError(f"{axis} is not a box axis")


def _frame_axis_label(frame: GtFrame, axis: str) -> str:
    if axis == "zone":
        return str(frame.zone)
    if axis == "altitude_band":
        return altitude_band(frame.agl_m)
    if axis == "time_of_day":
        return time_of_day_bin(frame.time_of_day)
    if axis == "modality":
        return str(frame.modality)
    raise ValueError(f"{axis} is not a frame axis")


def slice_rows(
    ds: EvalDataset,
    res: MatchResult,
    conf: float,
    base: SliceKey,
    axes: Sequence[str] = BOX_AXES + FRAME_AXES,
) -> MetricSet:
    """One row per bin of each axis. Bins partition the ground truth exactly (no box in two bins, none lost).

    Box axes (occlusion, posture, pixel_size) get **recall only**; frame axes also get precision and FP/min,
    because a false positive belongs to a frame but to no ground-truth box.
    """
    ms = MetricSet()
    tag = f"@iou{res.iou_thr:g}"
    frame_of_pred = {f.frame_idx: f for f in ds.frames}
    n_total = len(res.gt_boxes)
    for axis in axes:
        unpinned = AXIS_DEFAULT[axis]
        if axis in BOX_AXES:
            labels = np.asarray([_box_axis_label(b, f, axis) for b, f in zip(res.gt_boxes, res.gt_frames)])
        elif axis in FRAME_AXES:
            labels = np.asarray([_frame_axis_label(f, axis) for f in res.gt_frames])
        else:
            raise ValueError(f"unknown slice axis {axis!r}")
        binned = 0
        for label in sorted(set(labels.tolist())):
            if label == unpinned:
                continue  # the attribute is missing on those boxes; counted below, never silently dropped
            gm = labels == label
            binned += int(gm.sum())
            sk = narrow(base, **{axis: label})
            if axis in BOX_AXES:
                c = res.counts_at(conf, gt_mask=gm)
                ms.add(metric_row(f"recall{tag}", c.recall(), sk, c.n_gt, tp=c.tp, fn=c.fn,
                                  at_conf=conf, axis=axis))
                continue
            pred_labels = np.asarray([_frame_axis_label(frame_of_pred[int(i)], axis) for i in res.pred_frame])
            frame_labels = [_frame_axis_label(f, axis) for f in ds.frames]
            n_frames = sum(1 for x in frame_labels if x == label)
            c = res.counts_at(conf, gt_mask=gm, pred_mask=pred_labels == label)
            c.minutes = n_frames / ds.fps_processed / 60.0 if n_frames else 0.0
            ms.add(metric_row(f"recall{tag}", c.recall(), sk, c.n_gt, tp=c.tp, fn=c.fn, at_conf=conf, axis=axis))
            ms.add(metric_row(f"precision{tag}", c.precision(), sk, c.tp + c.fp, tp=c.tp, fp=c.fp,
                              at_conf=conf, axis=axis))
            if c.minutes > 0:
                ms.add(metric_row(f"fp_per_min{tag}", c.fp_per_min(), sk, c.fp, minutes=c.minutes,
                                  at_conf=conf, axis=axis, level="detection"))
        # Completeness guard: the bins of one axis must account for every scored ground-truth box. Whatever
        # they cannot bin (occlusion not annotated, zone unknown) is reported, not dropped.
        ms.add(metric_row("slice_unbinned_gt", float(n_total - binned), base, n_total, axis=axis,
                          binned=binned,
                          basis="ground-truth boxes this axis could not label; bins + this must equal n"))
    return ms


def recall_vs_pixel_height(res: MatchResult, conf: float, base: SliceKey, edges: Sequence[float] = (0, 8, 12, 16, 20, 30, 40, 60, 80, 120, 1e9)) -> MetricSet:
    """The §5.12 "missed-detection analysis: recall vs pixel-height histogram, to find the operating floor"."""
    ms = MetricSet()
    sizes = np.asarray([b.size_px for b in res.gt_boxes], dtype=float)
    found = res.gt_found(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if not n:
            continue
        label = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        ms.add(metric_row("recall_by_pixel_height", float(found[mask].sum()) / n,
                          narrow(base, pixel_size=pixel_size_bin(float(np.median(sizes[mask])))), n,
                          px_bin=label, at_conf=conf, iou_thr=res.iou_thr))
    return ms


def missed_detections(res: MatchResult, conf: float, limit: int = 50) -> list[dict]:
    """The ground-truth boxes nobody found at the operating point — the FiftyOne "false negatives" view."""
    out = []
    found = res.gt_found(conf)
    for i, ok in enumerate(found):
        if ok:
            continue
        b, f = res.gt_boxes[i], res.gt_frames[i]
        out.append({"frame_idx": f.frame_idx, "clip_id": f.clip_id, "bbox_px": list(b.bbox_px),
                    "gt_id": b.gt_id, "size_px": b.size_px, "occlusion": b.occlusion, "posture": str(b.posture),
                    "submersion": str(b.submersion), "zone": str(f.zone), "agl_m": f.agl_m})
        if len(out) >= limit:
            break
    return out


def false_positives(res: MatchResult, conf: float, limit: int = 50) -> list[dict]:
    """The unmatched predictions at the operating point — the FiftyOne "glint / roofing sheet" view."""
    out = []
    keep = (res.pred_score >= conf) & (res.pred_outcome == "fp")
    order = np.argsort(-res.pred_score)
    for i in order:
        if not keep[i]:
            continue
        d = res.detections[int(i)]
        out.append({"frame_idx": int(res.pred_frame[i]), "bbox_px": list(d.bbox_px), "score": float(d.score),
                    "size_px": d.size_px, "modality": str(d.modality), "tile_idx": d.tile_idx})
        if len(out) >= limit:
            break
    return out


__all__ = [
    "IOU_PRIMARY", "IOU_RELAXED", "NEVER", "TARGET_RECALL", "DetCounts", "MatchResult", "OperatingPoint",
    "PrCurve", "false_positives", "freeze_operating_threshold", "headline_rows", "match_dataset",
    "missed_detections", "recall_vs_pixel_height", "slice_rows", "sweep_confidence",
]
