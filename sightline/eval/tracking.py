"""Track-level metrics: HOTA, IDF1 and ID switches via `motmetrics` (§5.6, §5.12).

`motmetrics` is installed from the maintained develop branch, not the PyPI 1.4.0 (which calls `np.asfarray`,
removed in NumPy 2 — see `pyproject.toml`). That branch adds `hota_alpha` / `deta_alpha` / `assa_alpha`
alongside the classic CLEAR/ID metrics, so HOTA does not have to be re-implemented here.

Verified by hand in `tests/test_eval.py`: two tracks over four frames with one ID switch on one of them gives
DetA = 1, AssA = (4x1 + 2x0.5 + 2x0.5) / 8 = 0.75, HOTA = sqrt(0.75) = 0.8660254 — which is exactly what
motmetrics returns. That is the check that the library is being driven correctly.

Object and hypothesis IDs must be **numeric** for motmetrics' event dataframe; string ids are mapped here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from sightline.eval.groundtruth import EvalDataset
from sightline.eval.slicing import MetricSet, metric_row
from sightline.schemas import SliceKey, Track

MOT_IOU_MAX_DISTANCE = 0.5  # motmetrics' `max_iou`: pairs with IoU < 0.5 are not associable


@dataclass(slots=True)
class TrackFrame:
    """One frame of the tracking comparison, in xyxy pixels."""

    frame_idx: int
    gt_ids: list[int] = field(default_factory=list)
    gt_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    hyp_ids: list[int] = field(default_factory=list)
    hyp_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def frames_from_tracks(ds: EvalDataset, tracks: Sequence[Track], *, cls: str = "human") -> list[TrackFrame]:
    """Build the per-frame comparison from ground truth and the tracker's confirmed `Track` objects."""
    by_frame: dict[int, TrackFrame] = {}
    for f in ds.frames:
        tf = by_frame.setdefault(f.frame_idx, TrackFrame(frame_idx=f.frame_idx))
        for b in f.boxes:
            if not b.scored or b.is_group or (cls is not None and b.cls != cls):
                continue
            tf.gt_ids.append(int(b.gt_id))
            tf.gt_boxes.append(tuple(b.bbox_px))
    for t in tracks:
        if cls is not None and t.cls != cls:
            continue
        for o in t.observations:
            tf = by_frame.setdefault(o.frame_idx, TrackFrame(frame_idx=o.frame_idx))
            tf.hyp_ids.append(int(t.track_id))
            tf.hyp_boxes.append(tuple(o.det.bbox_px))
    return [by_frame[k] for k in sorted(by_frame)]


def _xyxy_to_xywh(boxes: Sequence[tuple[float, float, float, float]]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=float)
    a = np.asarray(boxes, dtype=float)
    return np.stack([a[:, 0], a[:, 1], a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], axis=1)


def accumulate(frames: Iterable[TrackFrame], *, max_iou: float = MOT_IOU_MAX_DISTANCE):
    """Fill a `motmetrics.MOTAccumulator`. IDs are remapped to dense ints (motmetrics needs numeric ids)."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for tf in frames:
        d = mm.distances.iou_matrix(_xyxy_to_xywh(tf.gt_boxes), _xyxy_to_xywh(tf.hyp_boxes), max_iou=max_iou)
        acc.update([int(i) for i in tf.gt_ids], [int(i) for i in tf.hyp_ids], d)
    return acc


def evaluate_tracking(frames: Sequence[TrackFrame], key: SliceKey, *, max_iou: float = MOT_IOU_MAX_DISTANCE,
                      name: str = "sightline") -> MetricSet:
    """HOTA / DetA / AssA / IDF1 / ID switches / MOTA / MOTP as `MetricRow`s."""
    import motmetrics as mm

    acc = accumulate(frames, max_iou=max_iou)
    host = mm.metrics.create()
    wanted = ["hota_alpha", "deta_alpha", "assa_alpha", "idf1", "idp", "idr", "num_switches",
              "num_fragmentations", "mota", "motp", "num_objects", "num_unique_objects",
              "mostly_tracked", "mostly_lost"]
    have = [m for m in wanted if m in host.names]
    summary = host.compute(acc, metrics=have, name=name)
    n_gt = int(summary["num_objects"].values[0]) if "num_objects" in summary else 0

    def val(metric: str) -> float:
        v = summary[metric].values[0]
        return float(np.mean(v))

    ms = MetricSet()
    label = {"hota_alpha": "hota", "deta_alpha": "deta", "assa_alpha": "assa", "num_switches": "id_switches",
             "num_fragmentations": "fragmentations", "mostly_tracked": "mostly_tracked",
             "mostly_lost": "mostly_lost"}
    for metric in have:
        if metric in ("num_objects", "num_unique_objects"):
            continue
        ms.add(metric_row(label.get(metric, metric), val(metric), key, n_gt,
                          source="motmetrics(develop)", max_iou=max_iou))
    ms.add(metric_row("track_gt_detections", float(n_gt), key, n_gt, basis="ground-truth boxes compared"))
    if "num_unique_objects" in summary:
        ms.add(metric_row("track_gt_identities", float(summary["num_unique_objects"].values[0]), key, n_gt))
    return ms


def motmetrics_available() -> bool:
    try:
        import motmetrics  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken env
        return False
    return True


__all__ = [
    "MOT_IOU_MAX_DISTANCE", "TrackFrame", "accumulate", "evaluate_tracking", "frames_from_tracks",
    "motmetrics_available",
]
