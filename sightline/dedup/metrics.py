""""Deduplication accuracy", defined so the evaluation script can compute it (SOLUTION_DOC §5.6).

    "Match output records to ground-truth survivor IDs by distance <= r (Hungarian on distance). Report record
     precision (duplicates count as false positives), record recall, duplicate rate = (records - unique matched
     GT) / unique matched GT, count error per cluster, and track-level HOTA / IDF1 / ID switches. Also report
     FP/min twice: at the raw-detection level and at the record level after tracking and dedup. The ratio is the
     value the tracker adds."

Two things worth stating because they are easy to get wrong.

**Duplicates are false positives, by construction.** The Hungarian assignment is one-to-one, so a second record
on the same survivor cannot be matched and lands in the false-positive count. That is the intended arithmetic:
the operator sees two markers where there is one person, and the metric must say so.

**Every number carries its slice.** `SliceKey.domain` is not optional (§5.12 and hard rule 5): a simulation
number and a real number may not be averaged. All functions here take the slice and return `MetricRow`s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from sightline.common import geodesy
from sightline.schemas import MetricRow, Record, SliceKey

#: What §5.6 calls "r": the distance under which a record is accepted as the same being as a ground-truth
#: survivor. Defaults to CE90 at the design geometry (60 m nadir, consumer GNSS: 2.6 m 1-sigma -> ~6 m CE90).
DEFAULT_MATCH_RADIUS_M = 6.0


@dataclass(slots=True)
class GroundTruthSurvivor:
    """One real being in the scenario. `count` is how many people are at this position (a group on a roof)."""

    gt_id: str
    lat: float
    lon: float
    count: int = 1
    cls: str = "human"


@dataclass(slots=True)
class DedupMatch:
    gt_id: str
    record_id: str
    distance_m: float
    count_error: int


@dataclass(slots=True)
class DedupAccuracy:
    """The §5.6 record-level result. `rows()` turns it into sliced `MetricRow`s for the evaluation harness."""

    n_records: int
    n_gt: int
    matches: list[DedupMatch] = field(default_factory=list)
    unmatched_record_ids: list[str] = field(default_factory=list)
    unmatched_gt_ids: list[str] = field(default_factory=list)
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M
    duration_s: float = 0.0

    @property
    def n_matched(self) -> int:
        """Unique ground-truth survivors matched. One-to-one assignment, so this is also the matched-record count."""
        return len(self.matches)

    @property
    def precision(self) -> float:
        """Matched records / all records. An extra record on a survivor is a false positive, by construction."""
        return self.n_matched / self.n_records if self.n_records else 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else 0.0

    @property
    def duplicate_rate(self) -> float:
        """§5.6: (records - unique matched GT) / unique matched GT.

        Note this counts *every* unmatched record, so a record on a false positive that is nowhere near a
        survivor inflates it exactly as a genuine duplicate does. That is the doc's definition; `precision`
        separates the two only when combined with the false-positive positions.
        """
        if self.n_matched == 0:
            return float("inf") if self.n_records else 0.0
        return (self.n_records - self.n_matched) / self.n_matched

    @property
    def count_mae(self) -> float:
        """Mean absolute error of `count_estimate` over matched clusters (§5.6 "count error per cluster")."""
        if not self.matches:
            return 0.0
        return float(np.mean([abs(m.count_error) for m in self.matches]))

    @property
    def count_bias(self) -> float:
        """Signed mean count error: negative means the system under-counts groups."""
        if not self.matches:
            return 0.0
        return float(np.mean([m.count_error for m in self.matches]))

    @property
    def mean_distance_m(self) -> float:
        return float(np.mean([m.distance_m for m in self.matches])) if self.matches else float("nan")

    @property
    def ce90_of_matches_m(self) -> float:
        """The 90th percentile of matched-record position error — the empirical CE90 of the output."""
        if not self.matches:
            return float("nan")
        return float(np.percentile([m.distance_m for m in self.matches], 90))

    @property
    def record_fp_per_min(self) -> float:
        """§5.6: the record-level half of the FP/min pair. The raw-detection half comes from the eval lane."""
        if self.duration_s <= 0:
            return float("nan")
        return len(self.unmatched_record_ids) / (self.duration_s / 60.0)

    def rows(self, slice_key: SliceKey) -> list[MetricRow]:
        """The §5.6 numbers as sliced `MetricRow`s.

        **The "we found nothing" case is real and stays reported.** It is also the case this used to get wrong.
        With no matched pairs `mean_distance_m` and `ce90_of_matches_m` are NaN, and with records but no matches
        `duplicate_rate` is +inf. Those went into `MetricRow` unchecked and came out as a bare `NaN` / `Infinity`
        in JSON that no strict parser reads back — AUDIT S8, reproduced exactly:

            dedup.mean_position_error_m value=nan n=0   ->   json.dumps(allow_nan=False) raises

        Five of these numbers are statistics over `self.matches`; when there are none, each is *undefined*, not
        zero, and `over_matches()` emits the explicit n=0 row that says so. Precision and recall are NOT in that
        group: with one survivor and no records, recall really is 0.0 over n=1, and calling that "undefined"
        would flatter the system.

        The PROPERTIES keep returning NaN / inf — `tests/test_dedup.py` pins `math.isinf(duplicate_rate)`, and a
        caller doing arithmetic wants the honest non-number. The honesty is applied where the number is
        *reported*, which is here.
        """
        detail: dict[str, Any] = {
            "match_radius_m": self.match_radius_m,
            "n_records": self.n_records,
            "n_gt": self.n_gt,
            "n_matched": self.n_matched,
            "unmatched_records": list(self.unmatched_record_ids),
            "unmatched_gt": list(self.unmatched_gt_ids),
        }
        no_match = (f"no record matched a ground-truth survivor within {self.match_radius_m:g} m "
                    f"(records={self.n_records}, survivors={self.n_gt}), so this statistic has no sample")

        def over_matches(name: str, value: float) -> MetricRow:
            """A statistic over matched pairs. Undefined — never 0, never NaN — when nothing matched."""
            if self.n_matched == 0:
                return MetricRow.undefined(name, slice_key, no_match, dict(detail))
            return MetricRow(name, float(value), slice_key, self.n_matched, dict(detail))

        rows = [
            # Precision is matched/records. With ZERO records that is 0/0 — the system made no claims, so
            # there is nothing to be precise about — and the property returns a bare 0.0, which reads as
            # "every record it produced was wrong". This is the fix's own target pattern sitting two lines
            # above the rows it did fix, and it was walked past because the defence written for RECALL was
            # silently extended to it. Recall's defence is sound and stays: with one survivor and no records,
            # recall really is 0 of 1, a real denominator. Precision's denominator is empty. Found by
            # red-team review.
            (MetricRow.undefined("dedup.record_precision", slice_key,
                                 f"no records were produced (survivors={self.n_gt}), so there is nothing "
                                 f"whose precision could be measured", dict(detail))
             if self.n_records == 0 else
             MetricRow("dedup.record_precision", self.precision, slice_key, self.n_records, dict(detail))),
            MetricRow("dedup.record_recall", self.recall, slice_key, self.n_gt, dict(detail)),
            # NOT `over_matches`. With no records at all, `duplicate_rate` deliberately returns a MEASURED
            # 0.0 -- "no records is a miss, not an infinity of duplicates" (see the property, pinned by
            # `tests/test_dedup.py:498`). Routing it through `over_matches` reported that real zero as
            # undefined with the reason "no record matched a ground-truth survivor", which is false when
            # there were no records to match, and left the property and the reported row disagreeing.
            # It is only genuinely undefined in the case the property calls infinite: records exist and none
            # of them matched anything.
            (MetricRow.undefined("dedup.duplicate_rate", slice_key,
                                 f"{self.n_records} record(s) matched no survivor within "
                                 f"{self.match_radius_m:g} m, so the duplicate rate is unbounded",
                                 dict(detail))
             if self.n_records and self.n_matched == 0 else
             MetricRow("dedup.duplicate_rate", self.duplicate_rate, slice_key, self.n_matched, dict(detail))),
            over_matches("dedup.count_mae", self.count_mae),
            over_matches("dedup.count_bias", self.count_bias),
            over_matches("dedup.mean_position_error_m", self.mean_distance_m),
            over_matches("dedup.ce90_m", self.ce90_of_matches_m),
        ]
        if self.duration_s > 0:
            rows.append(
                MetricRow("dedup.record_fp_per_min", self.record_fp_per_min, slice_key, self.n_records, dict(detail))
            )
        return rows


def dedup_accuracy(
    records: Sequence[Record],
    truth: Sequence[GroundTruthSurvivor],
    *,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    duration_s: float = 0.0,
    include_dismissed: bool = False,
) -> DedupAccuracy:
    """Match records to ground truth with the Hungarian algorithm on distance, then score (§5.6).

    Args:
        records: the output records. Pass `Deduplicator.active_records()` — a record dismissed as a duplicate is
            not shown to anyone, so counting it as a false positive would slander the system. Set
            `include_dismissed=True` to score the raw registry instead.
        truth: the ground-truth survivors.
        match_radius_m: §5.6's `r`. Pairs further apart than this are never matched.
        duration_s: mission duration, for the record-level FP/min.
    """
    from scipy.optimize import linear_sum_assignment

    kept = [r for r in records if include_dismissed or r.status != "dismissed"]
    if not kept or not truth:
        return DedupAccuracy(
            n_records=len(kept),
            n_gt=len(truth),
            unmatched_record_ids=[r.record_id for r in kept],
            unmatched_gt_ids=[g.gt_id for g in truth],
            match_radius_m=match_radius_m,
            duration_s=duration_s,
        )

    dist = np.array(
        [[geodesy.haversine_m(g.lat, g.lon, r.lat, r.lon) for r in kept] for g in truth],
        dtype=float,
    )
    # Infeasible pairs get a cost far above any feasible one, so the assignment prefers every real match first
    # and only then pads. The padded pairs are filtered out immediately afterwards.
    big = float(dist.max()) + 10.0 * match_radius_m + 1.0
    cost = np.where(dist <= match_radius_m, dist, big)
    rows, cols = linear_sum_assignment(cost)

    matches: list[DedupMatch] = []
    matched_gt: set[int] = set()
    matched_rec: set[int] = set()
    for gi, ri in zip(rows, cols, strict=True):
        if dist[gi, ri] > match_radius_m:
            continue
        matches.append(
            DedupMatch(
                gt_id=truth[gi].gt_id,
                record_id=kept[ri].record_id,
                distance_m=float(dist[gi, ri]),
                count_error=int(kept[ri].count_estimate) - int(truth[gi].count),
            )
        )
        matched_gt.add(gi)
        matched_rec.add(ri)

    return DedupAccuracy(
        n_records=len(kept),
        n_gt=len(truth),
        matches=matches,
        unmatched_record_ids=[r.record_id for i, r in enumerate(kept) if i not in matched_rec],
        unmatched_gt_ids=[g.gt_id for i, g in enumerate(truth) if i not in matched_gt],
        match_radius_m=match_radius_m,
        duration_s=duration_s,
    )


def fp_per_min(n_false_positives: int, duration_s: float) -> float:
    """§5.6 asks for this at two levels; the ratio of the two is what the tracker is worth."""
    if duration_s <= 0:
        return float("nan")
    return n_false_positives / (duration_s / 60.0)


# --- track-level metrics (py-motmetrics) ------------------------------------------------------------------
#: The MOT metric names this project reports. `hota_alpha` / `deta_alpha` / `assa_alpha` come from the
#: maintained develop branch of py-motmetrics (see pyproject: PyPI 1.4.0 predates HOTA and crashes on NumPy 2).
MOT_METRICS: tuple[str, ...] = (
    "idf1",
    "idp",
    "idr",
    "mota",
    "motp",
    "num_switches",
    "num_fragmentations",
    "num_false_positives",
    "num_misses",
    "num_unique_objects",
    "num_detections",
    "mostly_tracked",
    "mostly_lost",
    "hota_alpha",
    "deta_alpha",
    "assa_alpha",
)

DistanceKind = Literal["iou", "euclidean_m"]


@dataclass(slots=True)
class FrameAssociations:
    """One frame of ground truth and hypotheses, in whichever space the metric is being computed in.

    For `distance="iou"` the arrays are boxes in xyxy pixels; for `"euclidean_m"` they are (lat, lon) pairs and
    the distance is metres on the ellipsoid — the space §5.6 says identity actually lives in.
    """

    frame_idx: int
    gt_ids: Sequence[Any]
    gt_data: np.ndarray
    hyp_ids: Sequence[Any]
    hyp_data: np.ndarray


def track_metrics(
    frames: Sequence[FrameAssociations],
    slice_key: SliceKey,
    *,
    distance: DistanceKind = "iou",
    max_iou: float = 0.5,
    max_distance_m: float = DEFAULT_MATCH_RADIUS_M,
    name: str = "sightline",
) -> list[MetricRow]:
    """HOTA / IDF1 / ID switches from `py-motmetrics` (§5.6 "Metrics" row).

    py-motmetrics casts object ids to float, so string ids are mapped to integers here rather than crashing
    inside pandas with a message that says nothing about ids.
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    gt_index: dict[Any, int] = {}
    hyp_index: dict[Any, int] = {}

    for f in frames:
        gt = [gt_index.setdefault(i, len(gt_index)) for i in f.gt_ids]
        hyp = [hyp_index.setdefault(i, len(hyp_index)) for i in f.hyp_ids]
        if distance == "iou":
            # py-motmetrics takes (x, y, w, h); this project's boxes are xyxy (`Detection.bbox_px`). Feeding it
            # xyxy silently computes an IoU against a rectangle of the wrong size — wrong numbers, no error.
            d = mm.distances.iou_matrix(
                _xyxy_to_xywh(f.gt_data, len(gt)),
                _xyxy_to_xywh(f.hyp_data, len(hyp)),
                max_iou=max_iou,
            )
        else:
            d = _geo_distance_matrix(f.gt_data, f.hyp_data, max_distance_m)
        acc.update(gt, hyp, d, frameid=int(f.frame_idx))

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=list(MOT_METRICS), name=name)
    detail = {
        "distance": distance,
        "max_iou": max_iou,
        "max_distance_m": max_distance_m,
        "n_frames": len(frames),
    }
    n = int(summary["num_detections"][name])
    # py-motmetrics returns NaN for the ratio metrics (motp, idf1, hota_alpha, ...) when nothing matched: they
    # are 0/0, not 0. Reporting the NaN put a value RFC 8259 cannot encode into the metrics blob (AUDIT S8);
    # reporting 0.0 would claim perfect localisation error on a run that localised nothing. Say undefined.
    no_match = (f"py-motmetrics scored no matched pairs over {len(frames)} frames, so this ratio is 0/0; "
                "the count metrics beside it are still exact")
    rows = []
    for metric in MOT_METRICS:
        value = summary[metric][name]
        rows.append(MetricRow.finite_or_undefined(f"track.{metric}", float(np.mean(value)), slice_key, n,
                                                  reason=no_match, detail=dict(detail)))
    return rows


def _xyxy_to_xywh(boxes: np.ndarray, n: int) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(n, 4)
    if n == 0:
        return b
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def _geo_distance_matrix(gt_latlon: np.ndarray, hyp_latlon: np.ndarray, max_distance_m: float) -> np.ndarray:
    """(n_gt, n_hyp) metres, with pairs beyond `max_distance_m` set to NaN as py-motmetrics expects."""
    gt = np.asarray(gt_latlon, dtype=float).reshape(-1, 2)
    hyp = np.asarray(hyp_latlon, dtype=float).reshape(-1, 2)
    if len(gt) == 0 or len(hyp) == 0:
        return np.empty((len(gt), len(hyp)))
    d = np.array(
        [[geodesy.haversine_m(a[0], a[1], b[0], b[1]) for b in hyp] for a in gt],
        dtype=float,
    )
    return np.where(d <= max_distance_m, d, np.nan)


def frames_from_tracks(
    tracks: Sequence[Any],
    truth_by_frame: dict[int, dict[Any, Any]],
    *,
    distance: DistanceKind = "iou",
) -> list[FrameAssociations]:
    """Turn `list[Track]` + `{frame_idx: {gt_id: box_or_latlon}}` into the per-frame input `track_metrics` wants."""
    hyp: dict[int, list[tuple[int, Any]]] = {}
    for track in tracks:
        for o in track.observations:
            payload = o.det.bbox_px if distance == "iou" else (o.fix.lat, o.fix.lon)
            hyp.setdefault(o.frame_idx, []).append((track.track_id, payload))

    width = 4 if distance == "iou" else 2
    out: list[FrameAssociations] = []
    for frame_idx in sorted(set(truth_by_frame) | set(hyp)):
        gt_items = list(truth_by_frame.get(frame_idx, {}).items())
        hyp_items = hyp.get(frame_idx, [])
        out.append(
            FrameAssociations(
                frame_idx=frame_idx,
                gt_ids=[k for k, _ in gt_items],
                gt_data=np.array([v for _, v in gt_items], dtype=float).reshape(len(gt_items), width),
                hyp_ids=[k for k, _ in hyp_items],
                hyp_data=np.array([v for _, v in hyp_items], dtype=float).reshape(len(hyp_items), width),
            )
        )
    return out


__all__ = [
    "DEFAULT_MATCH_RADIUS_M",
    "MOT_METRICS",
    "DedupAccuracy",
    "DedupMatch",
    "DistanceKind",
    "FrameAssociations",
    "GroundTruthSurvivor",
    "dedup_accuracy",
    "fp_per_min",
    "frames_from_tracks",
    "track_metrics",
]
