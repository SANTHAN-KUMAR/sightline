"""Record-level metrics after tracking and deduplication (§5.6 "deduplication accuracy", defined for a script).

    "Match output records to ground-truth survivor IDs by distance <= r (Hungarian on distance). Report record
    precision (duplicates count as false positives), record recall, duplicate rate = (records - unique matched
    GT) / unique matched GT, count error per cluster, and track-level HOTA / IDF1 / ID switches. Also report
    FP/min twice: at the raw-detection level and at the record level after tracking and dedup. The ratio is the
    value the tracker adds."

Buried survivors (§6.2) are excluded from the recall denominator and reported on their own row: counting them
as misses understates a recall the air asset could never achieve, and hiding them would look like a clearance,
which guardrail R10 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sightline.common.geodesy import haversine_m
from sightline.eval.groundtruth import GtSurvivor
from sightline.eval.matching import gated_assignment
from sightline.eval.slicing import MetricSet, metric_row, ratio_row
from sightline.schemas import CE90_FACTOR, MetricRow, Record, SliceKey


@dataclass(slots=True)
class RecordMatch:
    """One-to-one assignment of records to survivors, gated at `radius_m`."""

    radius_m: float
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (record index, survivor index)
    distances_m: list[float] = field(default_factory=list)
    unmatched_records: list[int] = field(default_factory=list)
    unmatched_survivors: list[int] = field(default_factory=list)
    near_records: list[int] = field(default_factory=list)  # records within radius of SOME survivor
    n_records: int = 0
    n_survivors: int = 0

    @property
    def n_matched(self) -> int:
        return len(self.pairs)


def distance_matrix_m(records: Sequence[Record], survivors: Sequence[GtSurvivor]) -> np.ndarray:
    if not records or not survivors:
        return np.zeros((len(records), len(survivors)), dtype=float)
    return np.asarray(
        [[haversine_m(r.lat, r.lon, s.lat, s.lon) for s in survivors] for r in records], dtype=float
    )


def match_records(
    records: Sequence[Record],
    survivors: Sequence[GtSurvivor],
    radius_m: float,
) -> RecordMatch:
    """Hungarian on great-circle distance, gated at `radius_m` (§5.6). Extra records near a matched survivor
    stay unmatched and therefore count as false positives — that is what makes a duplicate cost precision."""
    d = distance_matrix_m(records, survivors)
    pairs = gated_assignment(d, radius_m) if d.size else []
    matched_r = {i for i, _ in pairs}
    matched_s = {j for _, j in pairs}
    near = [i for i in range(len(records)) if d.shape[1] and float(d[i].min()) <= radius_m]
    return RecordMatch(
        radius_m=float(radius_m),
        pairs=[(int(i), int(j)) for i, j in pairs],
        distances_m=[float(d[i, j]) for i, j in pairs],
        unmatched_records=[i for i in range(len(records)) if i not in matched_r],
        unmatched_survivors=[j for j in range(len(survivors)) if j not in matched_s],
        near_records=near,
        n_records=len(records),
        n_survivors=len(survivors),
    )


def dedup_radius_m(h_acc_m: float) -> float:
    """§5.6: cluster at 2 x CE90. The default match gate for record-level evaluation."""
    return 2.0 * CE90_FACTOR * float(h_acc_m)


def evaluate_records(
    records: Sequence[Record],
    survivors: Sequence[GtSurvivor],
    key: SliceKey,
    *,
    radius_m: float,
    minutes: float,
    cls: str = "human",
) -> MetricSet:
    """Record precision/recall, duplicate rate, count error and record-level FP/min, all as `MetricRow`s."""
    recs = [r for r in records if cls is None or r.cls == cls]
    findable = [s for s in survivors if s.findable and (cls is None or s.cls == cls)]
    buried = [s for s in survivors if not s.findable and (cls is None or s.cls == cls)]
    m = match_records(recs, findable, radius_m)

    ms = MetricSet()
    n_matched = m.n_matched
    precision = n_matched / len(recs) if recs else 0.0
    recall = n_matched / len(findable) if findable else 0.0
    ms.add(metric_row("record_precision", precision, key, len(recs), matched=n_matched,
                      false_records=len(m.unmatched_records), radius_m=radius_m))
    ms.add(metric_row("record_recall", recall, key, len(findable), matched=n_matched,
                      missed=len(m.unmatched_survivors), radius_m=radius_m))

    # §5.6 literal formula. Pure false positives inflate it, so the "near" variant is reported beside it.
    if n_matched:
        ms.add(metric_row("record_duplicate_rate", (len(recs) - n_matched) / n_matched, key, len(recs),
                          basis="(records - unique matched GT) / unique matched GT", matched=n_matched))
        ms.add(metric_row("record_duplicate_rate_near", (len(m.near_records) - n_matched) / n_matched, key,
                          len(m.near_records), basis="records within the match radius of a survivor only",
                          matched=n_matched))
    else:
        ms.add(metric_row("record_duplicate_rate", 0.0, key, len(recs), matched=0,
                          note="no record matched a survivor; the duplicate rate is undefined and reported as 0"))

    if minutes > 0:
        ms.add(metric_row("fp_per_min@record", len(m.unmatched_records) / minutes, key,
                          len(m.unmatched_records), minutes=minutes, level="record", radius_m=radius_m))

    # count error per matched cluster (§5.6 "count error per cluster")
    if m.pairs:
        errs = np.asarray([recs[i].count_estimate - findable[j].count for i, j in m.pairs], dtype=float)
        ms.add(metric_row("count_mae", float(np.abs(errs).mean()), key, len(errs),
                          basis="mean |count_estimate - gt count| over matched records"))
        ms.add(metric_row("count_bias", float(errs.mean()), key, len(errs),
                          basis="mean signed count error; positive = over-counting"))
        ms.add(metric_row("count_exact_rate", float((errs == 0).mean()), key, len(errs)))

    # R10 honesty rows: what could never be found from the air, stated rather than hidden
    ms.add(metric_row("buried_survivors_excluded", float(len(buried)), key, len(buried),
                      basis="present in truth as not visible (SOLUTION_DOC 6.2); excluded from record_recall",
                      guardrail="R10: aerial search cannot clear these cells"))
    ms.add(metric_row("record_localisation_median_m",
                      float(np.median(m.distances_m)) if m.distances_m else 0.0, key, len(m.distances_m),
                      basis="distance from a matched record to its survivor"))
    return ms


def tracking_gain_row(det_fp_per_min: MetricRow, rec_fp_per_min: MetricRow) -> MetricRow:
    """"The ratio is the value the tracker adds" (§5.6). Raises `DomainMixError` across domains."""
    return ratio_row("fp_per_min_reduction", det_fp_per_min, rec_fp_per_min,
                     basis="raw detection FP/min divided by record FP/min after tracking + dedup",
                     expectation="SOLUTION_DOC 5.12 expects >= 10x")


def r10_rows(records: Sequence[Record], key: SliceKey) -> MetricSet:
    """R10 audit as sliced rows: dismissed records still present, and every dismissal carrying a reason."""
    dismissed = [r for r in records if r.status == "dismissed"]
    unreasoned = [r for r in dismissed if not r.dismissed_reason]
    ms = MetricSet()
    ms.add(metric_row("r10_records_retained", float(len(records)), key, len(records),
                      basis="no code path deletes a record; every record in the log is counted here"))
    ms.add(metric_row("r10_dismissals_without_reason", float(len(unreasoned)), key, len(dismissed),
                      basis="R10 requires a reason on every dismissal; anything above 0 is a defect"))
    return ms


__all__ = [
    "RecordMatch", "dedup_radius_m", "distance_matrix_m", "evaluate_records", "match_records", "r10_rows",
    "tracking_gain_row",
]
