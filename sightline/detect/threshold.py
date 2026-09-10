"""The operating-threshold rule (SOLUTION_DOC §5.5) and the detector-level metrics of §5.12.

The rule, verbatim from §5.5: *"Sweep confidence on the validation split, pick the highest confidence at which
human recall >= 0.92 (a 2-point margin over the 0.90 target), freeze it, and report precision and FP/min at that
threshold."*

Why this module exists at all rather than reading a number off Ultralytics: **`model.val()` reports per-class
precision and recall at the max-F1 confidence**, which is a different operating point and usually a much higher
confidence than the one that holds recall at 0.92. Quoting it would overstate precision and understate recall at
the threshold the pipeline actually runs. So the sweep is done here, on this project's own matcher.

Matching is COCO-style and is computed **once**, at confidence 0: predictions are visited in descending score and
each takes the unmatched ground truth it overlaps most (IoU >= `iou_thr`). Dropping low-score predictions later
cannot change the assignment of the higher-scoring ones, so a single pass supports every threshold in the sweep --
this is the same argument that makes the standard PR curve valid.

Everything is numpy and stdlib. It is exact, and `tests/test_detect.py` checks it against hand-computed counts.

Guardrail R10 note: nothing here deletes or clears anything. A frozen threshold is an append-only JSON record
under `models/<version>/operating_point.json`; re-freezing writes a new file, it does not overwrite history.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sightline.detect.tiler import iou_xyxy
from sightline.schemas import SCHEMA_VERSION, ClassName, Detection, MetricRow, SliceKey

__all__ = [
    "PRIMARY_IOU",
    "SECONDARY_IOU",
    "TARGET_RECALL",
    "FrameEval",
    "GroundTruthBox",
    "MatchTally",
    "OperatingPoint",
    "SweepPoint",
    "choose_operating_threshold",
    "fp_per_minute",
    "freeze_operating_point",
    "load_operating_point",
    "match_frame",
    "metric_rows",
    "sweep",
    "tally",
]

#: §5.5: 0.92, a 2-point margin over the 0.90 requirement (R2), so ordinary run-to-run variation cannot cross it.
TARGET_RECALL = 0.92
PRIMARY_IOU = 0.5
#: §5.5 secondary "found the person" metric (TinyPerson practice): on a limb-only target the box extent is
#: ill-defined, so IoU 0.5 punishes a detection that a human would call correct.
SECONDARY_IOU = 0.25


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """One labelled box plus the attributes §5.12 slices on. Visible extent is the training/eval box (§6.3)."""

    bbox_px: tuple[float, float, float, float]
    cls: ClassName = "human"
    occlusion: int | None = None
    posture: str = "unknown"
    submersion: str = "unknown"
    ignore: bool = False  # a labelled region that must neither count as a miss nor as a false positive
    track_id: int = -1


@dataclass(slots=True)
class FrameEval:
    """One validation frame: what the detector said, what the truth is, and which scenario seed it came from."""

    predictions: list[Detection]
    ground_truth: list[GroundTruthBox]
    frame_idx: int = -1
    clip_id: str = ""
    seed: str = ""
    altitude_band: str = "all"
    time_of_day: str = "all"
    zone: str = "unknown"


@dataclass(frozen=True, slots=True)
class _FrameMatch:
    gt_scores: np.ndarray       # per GT: score of the prediction matched to it, -1.0 when never matched
    gt_ignore: np.ndarray       # per GT: bool
    pred_scores: np.ndarray     # per prediction
    pred_is_tp: np.ndarray      # per prediction: matched a non-ignore GT
    pred_ignored: np.ndarray    # per prediction: matched an `ignore` GT -> counts as neither TP nor FP


def match_frame(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruthBox],
    iou_thr: float = PRIMARY_IOU,
    cls: ClassName | None = "human",
) -> _FrameMatch:
    """Greedy score-ordered matching of one frame, restricted to `cls` (None = every class)."""
    preds = [p for p in predictions if cls is None or p.cls == cls]
    gts = [g for g in ground_truth if cls is None or g.cls == cls]
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)

    gt_scores = np.full(len(gts), -1.0, dtype=np.float64)
    gt_taken = np.zeros(len(gts), dtype=bool)
    gt_ignore = np.array([g.ignore for g in gts], dtype=bool)
    pred_scores = np.zeros(len(preds), dtype=np.float64)
    pred_is_tp = np.zeros(len(preds), dtype=bool)
    pred_ignored = np.zeros(len(preds), dtype=bool)

    for slot, i in enumerate(order):
        p = preds[i]
        pred_scores[slot] = p.score
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if gt_taken[j]:
                continue
            v = iou_xyxy(p.bbox_px, g.bbox_px)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            gt_taken[best_j] = True
            if gt_ignore[best_j]:
                pred_ignored[slot] = True
            else:
                gt_scores[best_j] = p.score
                pred_is_tp[slot] = True
    return _FrameMatch(gt_scores, gt_ignore, pred_scores, pred_is_tp, pred_ignored)


@dataclass(slots=True)
class MatchTally:
    """Matching results accumulated over a whole split; the sweep runs off this, not off the frames."""

    gt_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pred_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pred_is_tp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    n_frames: int = 0
    iou_thr: float = PRIMARY_IOU
    cls: ClassName | None = "human"

    @property
    def n_gt(self) -> int:
        return int(self.gt_scores.size)

    def counts_at(self, conf: float) -> tuple[int, int, int]:
        """(tp, fp, fn) with the operating rule "keep predictions with score >= conf"."""
        tp = int(np.count_nonzero(self.gt_scores >= conf))
        keep = self.pred_scores >= conf
        kept_tp = int(np.count_nonzero(keep & self.pred_is_tp))
        fp = int(np.count_nonzero(keep)) - kept_tp
        return tp, fp, self.n_gt - tp


def tally(frames: Iterable[FrameEval], iou_thr: float = PRIMARY_IOU, cls: ClassName | None = "human") -> MatchTally:
    gt, ps, tp = [], [], []
    n = 0
    for fr in frames:
        m = match_frame(fr.predictions, fr.ground_truth, iou_thr, cls)
        gt.append(m.gt_scores[~m.gt_ignore])
        ps.append(m.pred_scores[~m.pred_ignored])
        tp.append(m.pred_is_tp[~m.pred_ignored])
        n += 1
    return MatchTally(
        np.concatenate(gt) if gt else np.zeros(0),
        np.concatenate(ps) if ps else np.zeros(0),
        np.concatenate(tp) if tp else np.zeros(0, dtype=bool),
        n, iou_thr, cls,
    )


@dataclass(frozen=True, slots=True)
class SweepPoint:
    conf: float
    recall: float
    precision: float
    f1: float
    tp: int
    fp: int
    fn: int


def _candidate_thresholds(t: MatchTally, grid_step: float = 0.005) -> np.ndarray:
    """Every score that can change the counts, plus a fine grid so the reported curve is readable."""
    grid = np.arange(0.0, 1.0 + grid_step, grid_step)
    obs = t.pred_scores[t.pred_scores > 0] if t.pred_scores.size else np.zeros(0)
    return np.unique(np.round(np.concatenate([grid, obs, np.zeros(1)]), 6))


def sweep(t: MatchTally, thresholds: Sequence[float] | None = None) -> list[SweepPoint]:
    ths = _candidate_thresholds(t) if thresholds is None else np.asarray(thresholds, dtype=float)
    out: list[SweepPoint] = []
    for c in ths:
        tp, fp, fn = t.counts_at(float(c))
        recall = tp / t.n_gt if t.n_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(SweepPoint(float(c), recall, precision, f1, tp, fp, fn))
    return out


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """The frozen threshold and every number that must be quoted with it (§5.5, §5.12)."""

    conf: float
    recall: float
    precision: float
    recall_secondary_iou: float
    target_recall: float
    achieved: bool
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_frames: int
    iou_thr: float
    secondary_iou_thr: float
    fp_per_frame: float
    fp_per_minute: float | None
    fps_processed: float | None
    cls: str
    model_version: str = ""
    domain: str = "sim"
    note: str = ""

    def label(self) -> str:
        d = "achieved" if self.achieved else "NOT ACHIEVED"
        return (f"conf={self.conf:.3f} recall={self.recall:.4f} (target {self.target_recall}, {d}) "
                f"precision={self.precision:.4f} in {self.domain}, n_gt={self.n_gt}")


def fp_per_minute(fp_total: int, n_frames: int, fps_processed: float) -> float:
    """§5.12: FP/min = FP_total / (frames_processed / fps_processed / 60)."""
    if n_frames <= 0 or fps_processed <= 0:
        raise ValueError("n_frames and fps_processed must be positive")
    return fp_total / (n_frames / fps_processed / 60.0)


def choose_operating_threshold(
    frames: Iterable[FrameEval],
    *,
    target_recall: float = TARGET_RECALL,
    iou_thr: float = PRIMARY_IOU,
    secondary_iou_thr: float = SECONDARY_IOU,
    cls: ClassName | None = "human",
    fps_processed: float | None = None,
    model_version: str = "",
    domain: str = "sim",
) -> tuple[OperatingPoint, list[SweepPoint], MatchTally]:
    """Pick the **highest** confidence whose recall still clears `target_recall`. Never guesses.

    If even conf = 0 misses the target the point is returned with ``achieved=False`` and the recall that was
    actually reached, so the caller reports the honest number instead of a silently lowered bar (hard rule 2).
    """
    frames = list(frames)
    t = tally(frames, iou_thr, cls)
    t2 = tally(frames, secondary_iou_thr, cls)
    points = sweep(t)
    ok = [p for p in points if p.recall >= target_recall]
    achieved = bool(ok)
    best = max(ok, key=lambda p: p.conf) if ok else max(points, key=lambda p: (p.recall, p.conf))

    r2 = sweep(t2, [best.conf])[0].recall
    fpf = best.fp / t.n_frames if t.n_frames else 0.0
    op = OperatingPoint(
        conf=best.conf, recall=best.recall, precision=best.precision, recall_secondary_iou=r2,
        target_recall=target_recall, achieved=achieved, tp=best.tp, fp=best.fp, fn=best.fn,
        n_gt=t.n_gt, n_frames=t.n_frames, iou_thr=iou_thr, secondary_iou_thr=secondary_iou_thr,
        fp_per_frame=fpf,
        fp_per_minute=(fp_per_minute(best.fp, t.n_frames, fps_processed) if fps_processed else None),
        fps_processed=fps_processed, cls=str(cls), model_version=model_version, domain=domain,
        note="" if achieved else f"no confidence reaches recall {target_recall}; best recall {best.recall:.4f}",
    )
    return op, points, t


def metric_rows(op: OperatingPoint, slice_key: SliceKey) -> list[MetricRow]:
    """The §5.12 numbers as `MetricRow`s, each carrying its slice. A number without its slice is a bug."""
    rows = [
        MetricRow("recall@IoU0.5", op.recall, slice_key, op.n_gt, {"conf": op.conf, "model": op.model_version}),
        MetricRow("recall@IoU0.25", op.recall_secondary_iou, slice_key, op.n_gt, {"conf": op.conf}),
        MetricRow("precision@IoU0.5", op.precision, slice_key, op.tp + op.fp, {"conf": op.conf}),
        MetricRow("fp_per_frame", op.fp_per_frame, slice_key, op.n_frames, {"conf": op.conf}),
    ]
    if op.fp_per_minute is not None:
        rows.append(MetricRow("fp_per_minute", op.fp_per_minute, slice_key, op.n_frames,
                              {"conf": op.conf, "fps_processed": op.fps_processed}))
    return rows


def sliced_operating_points(
    frames: Iterable[FrameEval],
    conf: float,
    *,
    iou_thr: float = PRIMARY_IOU,
    cls: ClassName | None = "human",
    domain: str = "sim",
) -> list[MetricRow]:
    """Recall at the **frozen** conf, reported per altitude band, time of day, zone and occlusion bin (§5.12).

    The threshold is never re-chosen per slice: one model version has one operating point, and the slices explain
    where it works and where it does not.
    """
    frames = list(frames)
    rows: list[MetricRow] = []

    def _recall(sel: list[FrameEval], key: SliceKey, gt_filter=None) -> None:
        gt_scores = []
        n_fr = 0
        for fr in sel:
            m = match_frame(fr.predictions, fr.ground_truth, iou_thr, cls)
            gts = [g for g in fr.ground_truth if cls is None or g.cls == cls]
            keep = np.array([(not g.ignore) and (gt_filter is None or gt_filter(g)) for g in gts], dtype=bool)
            if keep.size:
                gt_scores.append(m.gt_scores[keep])
            n_fr += 1
        s = np.concatenate(gt_scores) if gt_scores else np.zeros(0)
        n = int(s.size)
        rows.append(MetricRow("recall@IoU0.5", float(np.count_nonzero(s >= conf) / n) if n else 0.0, key, n,
                              {"conf": conf, "frames": n_fr}))

    _recall(frames, SliceKey(domain=domain))  # the nominal, whole-split number
    for band in sorted({f.altitude_band for f in frames} - {"all"}):
        _recall([f for f in frames if f.altitude_band == band], SliceKey(domain=domain, altitude_band=band))
    for tod in sorted({f.time_of_day for f in frames} - {"all", ""}):
        _recall([f for f in frames if f.time_of_day == tod], SliceKey(domain=domain, time_of_day=tod))
    for zone in sorted({f.zone for f in frames} - {"unknown", ""}):
        _recall([f for f in frames if f.zone == zone], SliceKey(domain=domain, zone=zone))  # type: ignore[arg-type]
    for occ in (0, 1, 2):
        _recall(frames, SliceKey(domain=domain, occlusion=str(occ)), lambda g, o=occ: g.occlusion == o)
    for posture in sorted({g.posture for f in frames for g in f.ground_truth} - {"unknown", ""}):
        _recall(frames, SliceKey(domain=domain, posture=posture), lambda g, p=posture: g.posture == p)
    return [r for r in rows if r.n > 0]


# --- freezing ------------------------------------------------------------------------------------------------
def freeze_operating_point(
    op: OperatingPoint,
    out_path: str | Path,
    *,
    points: Sequence[SweepPoint] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the threshold beside its model version. The record is the contract the pipeline reads at runtime."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frozen_utc": time.time(),
        "rule": "SOLUTION_DOC 5.5: highest confidence at which human recall >= target_recall",
        "operating_point": asdict(op),
    }
    if points is not None:
        payload["sweep"] = [asdict(sp) for sp in points]
    if extra:
        payload["extra"] = extra
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_operating_point(path: str | Path) -> OperatingPoint:
    d = json.loads(Path(path).read_text(encoding="utf-8"))["operating_point"]
    return OperatingPoint(**d)
