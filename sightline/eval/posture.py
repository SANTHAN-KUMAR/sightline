"""Posture / submersion head evaluation (§5.5a) — accuracy by pixel size, and the metric that actually decides.

    "Report posture accuracy per class binned by target pixel size, because at 20-40 px posture is genuinely
    hard and the honest report must say where it works. The metric that actually matters is not raw accuracy
    but whether the ordering improves: measure rank correlation between the system's ranked list and the
    ground-truth urgency ordering, with and without the head. If the correlation does not improve, the head is
    decoration and should be switched off."  -- SOLUTION_DOC 5.5a

`rank_correlation_rows()` therefore emits a verdict row (`posture_head_improves_ranking`, 1 or 0) next to the
two correlations and their difference. Spearman rho and Kendall tau-b are implemented here in numpy, with
proper tie handling, because the ground-truth urgency ordering is full of ties (many "stranded" survivors) and
a tie-blind correlation would flatter the head.

§5.5a also states the safety rule the harness must not silently break: **posture may raise urgency, never lower
it.** `promotion_audit_rows()` checks that in the output: no record may rank lower with the head than without.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sightline.eval.groundtruth import URGENCY_ORDER
from sightline.eval.slicing import MetricSet, metric_row, narrow, partition, pixel_size_bin
from sightline.schemas import POSTURES, SUBMERSIONS, SliceKey


# --- rank statistics ---------------------------------------------------------------------------------------
def average_ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged (the definition Spearman's rho needs)."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation of the average ranks. Returns 0.0 when either side is entirely tied."""
    rx, ry = average_ranks(x), average_ranks(y)
    if len(rx) < 2:
        return 0.0
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b: concordant minus discordant, normalised for ties on both sides."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = len(a)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    pairs_tied_x = pairs_tied_y = 0  # n1 and n2 of the tau-b definition: pairs tied on that side, any other
    for i in range(n - 1):
        dx = a[i + 1:] - a[i]
        dy = b[i + 1:] - b[i]
        s = np.sign(dx) * np.sign(dy)
        concordant += int((s > 0).sum())
        discordant += int((s < 0).sum())
        pairs_tied_x += int((dx == 0).sum())
        pairs_tied_y += int((dy == 0).sum())
    n0 = n * (n - 1) / 2
    denom = np.sqrt(max(n0 - pairs_tied_x, 0) * max(n0 - pairs_tied_y, 0))
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


# --- classification accuracy -------------------------------------------------------------------------------
@dataclass(slots=True)
class AttributeSample:
    """One matched detection: what the head said, what the truth was, and how big the target was."""

    gt: str
    pred: str
    size_px: float
    conf: float = 0.0


def _accuracy(samples: Sequence[AttributeSample]) -> tuple[float, int]:
    if not samples:
        return 0.0, 0
    hits = sum(1 for s in samples if s.gt == s.pred)
    return hits / len(samples), len(samples)


def confusion(samples: Sequence[AttributeSample], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    m = {g: {p: 0 for p in labels} for g in labels}
    for s in samples:
        m.setdefault(s.gt, {p: 0 for p in labels}).setdefault(s.pred, 0)
        m[s.gt][s.pred] += 1
    return m


def attribute_accuracy_rows(
    samples: Sequence[AttributeSample],
    key: SliceKey,
    *,
    attribute: str = "posture",
    labels: Sequence[str] | None = None,
    slice_on_posture_axis: bool = True,
) -> MetricSet:
    """`<attribute>_accuracy` overall, per pixel-size bin, per class, and per class x pixel-size bin.

    The `posture` axis of `SliceKey` is used for per-class posture rows; `submersion` has no axis, so its class
    lives in `detail["gt_class"]` and the row is still sliced by pixel size. Nothing is ever emitted without a
    slice.
    """
    labels = list(labels or (POSTURES if attribute == "posture" else SUBMERSIONS))
    name = f"{attribute}_accuracy"
    ms = MetricSet()
    acc, n = _accuracy(samples)
    ms.add(metric_row(name, acc, key, n, attribute=attribute,
                      confusion=confusion(samples, labels) if samples else {}))
    if not samples:
        return ms

    for bin_label, group in sorted(partition(samples, lambda s: pixel_size_bin(s.size_px)).items()):
        a, m = _accuracy(group)
        ms.add(metric_row(name, a, narrow(key, pixel_size=bin_label), m, attribute=attribute, axis="pixel_size"))

    for cls_label, group in sorted(partition(samples, lambda s: s.gt).items()):
        a, m = _accuracy(group)
        sk = narrow(key, posture=cls_label) if (slice_on_posture_axis and cls_label in POSTURES) else key
        ms.add(metric_row(name, a, sk, m, attribute=attribute, gt_class=cls_label, axis="class"))
        for bin_label, sub in sorted(partition(group, lambda s: pixel_size_bin(s.size_px)).items()):
            a2, m2 = _accuracy(sub)
            sk2 = narrow(sk, pixel_size=bin_label)
            ms.add(metric_row(name, a2, sk2, m2, attribute=attribute, gt_class=cls_label,
                              axis="class x pixel_size"))
    return ms


# --- the metric that decides whether the head stays on -----------------------------------------------------
@dataclass(slots=True)
class RankingComparison:
    """The system's ranked list with and without the head, against the ground-truth urgency ordering.

    `truth` is the urgency VALUE per record (`URGENCY_ORDER`: immersed 4 > trapped 3 > stranded 2 > animal 1),
    not a rank, so ties are preserved and the tie-aware correlations see them.
    """

    score_with_head: list[float] = field(default_factory=list)
    score_without_head: list[float] = field(default_factory=list)
    truth_urgency: list[float] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.truth_urgency)
        if not (len(self.score_with_head) == len(self.score_without_head) == n):
            raise ValueError("with-head, without-head and truth lists must be the same length")

    @classmethod
    def from_urgency_classes(cls, score_with: Sequence[float], score_without: Sequence[float],
                             urgency_classes: Sequence[str], record_ids: Sequence[str] = ()) -> "RankingComparison":
        return cls(list(score_with), list(score_without),
                   [float(URGENCY_ORDER.get(str(u), 0)) for u in urgency_classes], list(record_ids))


def rank_correlation_rows(cmp: RankingComparison, key: SliceKey) -> MetricSet:
    """Spearman and Kendall, with and without the head, plus the delta and the on/off verdict (§5.5a)."""
    n = len(cmp.truth_urgency)
    ms = MetricSet()
    if n < 2:
        ms.add(metric_row("rank_corr_spearman_with_head", 0.0, key, n,
                          note="fewer than 2 matched records; the ordering cannot be scored"))
        return ms
    rho_on = spearman_rho(cmp.score_with_head, cmp.truth_urgency)
    rho_off = spearman_rho(cmp.score_without_head, cmp.truth_urgency)
    tau_on = kendall_tau_b(cmp.score_with_head, cmp.truth_urgency)
    tau_off = kendall_tau_b(cmp.score_without_head, cmp.truth_urgency)
    ms.add(metric_row("rank_corr_spearman_with_head", rho_on, key, n, head="on"))
    ms.add(metric_row("rank_corr_spearman_without_head", rho_off, key, n, head="off"))
    ms.add(metric_row("rank_corr_spearman_delta", rho_on - rho_off, key, n,
                      basis="with head minus without head; <= 0 means the head is decoration (5.5a)"))
    ms.add(metric_row("rank_corr_kendall_with_head", tau_on, key, n, head="on"))
    ms.add(metric_row("rank_corr_kendall_without_head", tau_off, key, n, head="off"))
    ms.add(metric_row("rank_corr_kendall_delta", tau_on - tau_off, key, n))
    improves = (rho_on > rho_off) and (tau_on >= tau_off)
    ms.add(metric_row("posture_head_improves_ranking", 1.0 if improves else 0.0, key, n,
                      verdict="keep the head on" if improves else
                              "SWITCH THE HEAD OFF: it does not improve the ordering (SOLUTION_DOC 5.5a)",
                      spearman_delta=rho_on - rho_off, kendall_delta=tau_on - tau_off))
    return ms


def promotion_audit_rows(cmp: RankingComparison, key: SliceKey) -> MetricSet:
    """§5.5a safety rule: posture may RAISE a record's urgency, never lower it. Any demotion is a defect."""
    n = len(cmp.truth_urgency)
    if n == 0:
        return MetricSet()
    with_head = np.asarray(cmp.score_with_head, dtype=float)
    without = np.asarray(cmp.score_without_head, dtype=float)
    demoted = int((with_head < without - 1e-9).sum())
    ms = MetricSet()
    ms.add(metric_row("posture_demotions", float(demoted), key, n,
                      basis="records the head scored LOWER than the base score; must be 0 (5.5a rule 2)",
                      offending=[cmp.record_ids[i] for i in np.nonzero(with_head < without - 1e-9)[0]][:20]))
    ms.add(metric_row("posture_promotions", float(int((with_head > without + 1e-9).sum())), key, n,
                      basis="records the head raised"))
    return ms


__all__ = [
    "AttributeSample", "RankingComparison", "attribute_accuracy_rows", "average_ranks", "confusion",
    "kendall_tau_b", "promotion_audit_rows", "rank_correlation_rows", "spearman_rho",
]
