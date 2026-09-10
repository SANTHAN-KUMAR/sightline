"""Box matching, ignore/group handling, and a numpy Hungarian solver (§5.12, §6.3).

Matching rule (§5.12): "per processed frame, greedy-match predictions to ground truth at IoU >= t, count
unmatched predictions". Greedy in *descending score order* is what makes the confidence sweep of
`detection.py` correct: the assignment for the top-k predictions is exactly the assignment you would get by
running the matcher on only those k, so one pass gives the whole precision-recall curve.

§6.3 changes what "unmatched" means for three kinds of ground truth:
  * `uncertain` / `ignore` boxes: a prediction that lands on one is **dropped** — neither TP nor FP — and
    missing one is not a miss. Matched by intersection-over-detection (IoD), as CityPersons/TinyPerson do.
  * `is_group` boxes (COCO `iscrowd = 1`): predictions inside are neither TP nor FP; the group counts as ONE
    recall target and is satisfied if any prediction reaches IoD >= 0.5 with it.
  * everything else is a normal scored target.

Nothing here returns a metric; `detection.py` turns these counts into `MetricRow`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sightline.eval.groundtruth import GtBox

GROUP_IOD_THRESHOLD = 0.5  # §6.3: "the group counts as one recall target if any detection overlaps at IoD >= 0.5"


def _as_array(boxes: Sequence) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=float)
    return np.asarray([tuple(b) for b in boxes], dtype=float)


def iou_matrix(a: Sequence, b: Sequence) -> np.ndarray:
    """Pairwise IoU of two xyxy box sets. Shape (len(a), len(b))."""
    A, B = _as_array(a), _as_array(b)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    area_b = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def iod_matrix(preds: Sequence, regions: Sequence) -> np.ndarray:
    """Intersection over DETECTION area — the §6.3 rule for ignore regions and crowd/group boxes."""
    P, R = _as_array(preds), _as_array(regions)
    if P.size == 0 or R.size == 0:
        return np.zeros((P.shape[0], R.shape[0]), dtype=float)
    x1 = np.maximum(P[:, None, 0], R[None, :, 0])
    y1 = np.maximum(P[:, None, 1], R[None, :, 1])
    x2 = np.minimum(P[:, None, 2], R[None, :, 2])
    y2 = np.minimum(P[:, None, 3], R[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = np.clip(P[:, 2] - P[:, 0], 0, None) * np.clip(P[:, 3] - P[:, 1], 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(area_p[:, None] > 0, inter / area_p[:, None], 0.0)
    return out


@dataclass(slots=True)
class FrameMatch:
    """The outcome of matching one frame. Indices refer to the input lists, in their original order."""

    tp: list[tuple[int, int]] = field(default_factory=list)  # (pred_idx, gt_idx) into the SCORED gt list
    fp: list[int] = field(default_factory=list)  # prediction indices with no target
    fn: list[int] = field(default_factory=list)  # scored gt indices nobody found
    ignored_pred: list[int] = field(default_factory=list)  # dropped by an ignore / uncertain / group box
    tp_iou: list[float] = field(default_factory=list)  # IoU of each tp pair, same order as `tp`
    group_hit: list[int] = field(default_factory=list)  # group gt indices satisfied by some prediction
    group_miss: list[int] = field(default_factory=list)

    @property
    def n_tp(self) -> int:
        return len(self.tp) + len(self.group_hit)

    @property
    def n_fn(self) -> int:
        return len(self.fn) + len(self.group_miss)

    @property
    def n_fp(self) -> int:
        return len(self.fp)


def match_frame(
    pred_boxes: Sequence,
    pred_scores: Sequence[float],
    gt_boxes: Sequence[GtBox],
    iou_thr: float,
    *,
    group_iod_thr: float = GROUP_IOD_THRESHOLD,
) -> FrameMatch:
    """Greedy score-ordered matching of one frame's predictions to one frame's ground truth.

    `gt_boxes` may mix scored, `uncertain`, `ignore` and `is_group` boxes; the returned indices are into the
    **scored** subset for `tp`/`fn` and into the **group** subset for `group_hit`/`group_miss`.
    """
    scored_idx = [i for i, g in enumerate(gt_boxes) if g.scored and not g.is_group]
    group_idx = [i for i, g in enumerate(gt_boxes) if g.scored and g.is_group]
    drop_idx = [i for i, g in enumerate(gt_boxes) if not g.scored]

    scored = [gt_boxes[i].bbox_px for i in scored_idx]
    groups = [gt_boxes[i].bbox_px for i in group_idx]
    drops = [gt_boxes[i].bbox_px for i in drop_idx]

    order = list(np.argsort(-np.asarray(list(pred_scores), dtype=float), kind="stable")) if len(pred_scores) else []
    ious = iou_matrix(pred_boxes, scored)
    iod_groups = iod_matrix(pred_boxes, groups)
    iod_drops = iod_matrix(pred_boxes, drops)

    m = FrameMatch()
    taken = np.zeros(len(scored), dtype=bool)
    group_found = np.zeros(len(groups), dtype=bool)
    for p in order:
        p = int(p)
        if len(scored):
            row = np.where(taken, -1.0, ious[p])
            j = int(np.argmax(row))
            if row[j] >= iou_thr and row[j] > 0.0:
                taken[j] = True
                m.tp.append((p, scored_idx[j]))
                m.tp_iou.append(float(row[j]))
                continue
        # not a true positive: is it excused by a group, ignore or uncertain box?
        if len(groups) and float(iod_groups[p].max()) >= group_iod_thr:
            group_found[int(np.argmax(iod_groups[p]))] = True
            m.ignored_pred.append(p)
            continue
        if len(drops) and float(iod_drops[p].max()) >= group_iod_thr:
            m.ignored_pred.append(p)
            continue
        m.fp.append(p)
    m.fn = [scored_idx[j] for j in range(len(scored)) if not taken[j]]
    m.group_hit = [group_idx[j] for j in range(len(groups)) if group_found[j]]
    m.group_miss = [group_idx[j] for j in range(len(groups)) if not group_found[j]]
    m.fp.sort()
    m.ignored_pred.sort()
    return m


# --- Hungarian (linear sum assignment) ---------------------------------------------------------------------
def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment for a rectangular cost matrix (Jonker-Volgenant style, O(n^2 m)).

    Written in numpy rather than taken from scipy so `sightline.eval` keeps the `schemas.py` dependency rule
    (stdlib + numpy). `tests/test_eval.py` cross-checks it against `scipy.optimize.linear_sum_assignment`.
    Use a large finite cost, not `inf`, for forbidden pairs.
    """
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2:
        raise ValueError("cost must be 2-D")
    if not np.all(np.isfinite(C)):
        raise ValueError("cost must be finite; use a large sentinel value for forbidden pairs")
    if C.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    big = float(np.abs(C).max()) * (n + m) + 1.0
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, big)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = C[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = float(cand[j1 - 1])
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    r = np.asarray(rows, dtype=int)
    c = np.asarray(cols, dtype=int)
    order = np.argsort(r if not transposed else c, kind="stable")
    r, c = r[order], c[order]
    return (c, r) if transposed else (r, c)


def gated_assignment(cost: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """Hungarian assignment, then drop every pair whose cost exceeds `gate` (§5.6: "Hungarian on distance,
    matched by distance <= r"). Returns (row, col) pairs sorted by row."""
    C = np.asarray(cost, dtype=float)
    if C.size == 0:
        return []
    big = (float(np.nanmax(C[np.isfinite(C)])) if np.any(np.isfinite(C)) else gate) + gate * 10.0 + 1.0
    padded = np.where(np.isfinite(C) & (C <= gate), C, big)
    r, c = hungarian(padded)
    return [(int(i), int(j)) for i, j in zip(r, c) if np.isfinite(C[i, j]) and C[i, j] <= gate]


__all__ = [
    "GROUP_IOD_THRESHOLD", "FrameMatch", "gated_assignment", "hungarian", "iod_matrix", "iou_matrix",
    "match_frame",
]
