"""POD calibration: the reliability diagram that decides whether the coverage map is honest (§5.3, §5.12).

    "Search-quality map calibration: for cells with a ground-truth survivor, the fraction detected, binned by
    the cell's predicted probability of detection (a reliability diagram; the map is honest if the bins lie
    near the diagonal)."  -- SOLUTION_DOC 5.12

and, for the per-presentation layers of §5.3b:

    "Expect the limb-only and head-only layers to be the poorly calibrated ones at first, because they have the
    fewest positives; report their confidence intervals rather than a bare number."

So every bin carries a Wilson score interval, and the summary rows are the expected calibration error (ECE),
the maximum calibration error (MCE) and the signed bias (predicted minus observed). A positive bias means the
map claims more search effort than it delivered, which is the direction that gets someone killed: it makes a
cell look searched when it is not. R10 is the same idea stated as a rule — the map shows POD, never "cleared".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sightline.eval.slicing import MetricSet, metric_row, narrow
from sightline.schemas import SliceKey

DEFAULT_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default). Correct at small n, unlike normal."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(slots=True)
class PodBin:
    """One bin of the reliability diagram."""

    lo: float
    hi: float
    n: int
    n_found: int
    predicted_mean: float
    observed: float
    ci_lo: float
    ci_hi: float

    @property
    def label(self) -> str:
        return f"{self.lo:.1f}-{self.hi:.1f}"

    @property
    def on_diagonal(self) -> bool:
        """Honest bin: the predicted value lies inside the observed rate's confidence interval."""
        return self.ci_lo <= self.predicted_mean <= self.ci_hi


def reliability_bins(pod_pred: Sequence[float], found: Sequence[bool],
                     bins: Sequence[float] = DEFAULT_BINS) -> list[PodBin]:
    """Bin predicted POD against the observed find rate. Bins are half-open, the last one closed at 1.0."""
    p = np.asarray(pod_pred, dtype=float)
    f = np.asarray(found, dtype=bool)
    if p.shape != f.shape:
        raise ValueError(f"pod_pred and found must be the same length, got {p.shape} and {f.shape}")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("predicted POD must be in [0, 1]")
    out: list[PodBin] = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        last = i == len(bins) - 2
        mask = (p >= lo) & ((p <= hi) if last else (p < hi))
        n = int(mask.sum())
        if not n:
            continue
        k = int(f[mask].sum())
        ci_lo, ci_hi = wilson_interval(k, n)
        out.append(PodBin(lo=float(lo), hi=float(hi), n=n, n_found=k, predicted_mean=float(p[mask].mean()),
                          observed=k / n, ci_lo=ci_lo, ci_hi=ci_hi))
    return out


def evaluate_pod_calibration(
    pod_pred: Sequence[float],
    found: Sequence[bool],
    key: SliceKey,
    *,
    bins: Sequence[float] = DEFAULT_BINS,
    presentation: str = "body",
) -> MetricSet:
    """ECE / MCE / bias plus one row per bin. Every bin row carries its Wilson 95 % interval in `detail`."""
    diag = reliability_bins(pod_pred, found, bins)
    ms = MetricSet()
    n = sum(b.n for b in diag)
    if not n:
        ms.add(metric_row("pod_ece", 0.0, key, 0, presentation=presentation,
                          note="no cells with a ground-truth survivor; calibration is unmeasured, not perfect"))
        return ms
    gaps = np.asarray([abs(b.predicted_mean - b.observed) for b in diag])
    weights = np.asarray([b.n for b in diag], dtype=float)
    signed = np.asarray([b.predicted_mean - b.observed for b in diag])
    ms.add(metric_row("pod_ece", float((gaps * weights).sum() / weights.sum()), key, n,
                      presentation=presentation, basis="sample-weighted mean |predicted - observed| over bins",
                      n_bins=len(diag)))
    ms.add(metric_row("pod_mce", float(gaps.max()), key, n, presentation=presentation,
                      basis="largest bin gap"))
    ms.add(metric_row("pod_bias", float((signed * weights).sum() / weights.sum()), key, n,
                      presentation=presentation,
                      basis="signed: positive means the map PROMISED more detection than it delivered"))
    ms.add(metric_row("pod_bins_on_diagonal", float(sum(b.on_diagonal for b in diag)) / len(diag), key,
                      len(diag), presentation=presentation,
                      basis="fraction of bins whose predicted POD lies inside the observed 95 % interval"))
    for b in diag:
        ms.add(metric_row("pod_observed_find_rate", b.observed, narrow(key), b.n, presentation=presentation,
                          pod_bin=b.label, predicted=b.predicted_mean, ci95=[b.ci_lo, b.ci_hi],
                          n_found=b.n_found, on_diagonal=b.on_diagonal))
    return ms


def survivors_found_by_cell(
    survivor_cells: Sequence[tuple[int, int]],
    pod_grid: np.ndarray,
    found_flags: Sequence[bool],
) -> tuple[list[float], list[bool]]:
    """Look each ground-truth survivor's cell up in the coverage grid's POD raster (`CoverageGrid.pod`).

    Returns (predicted POD per survivor, found per survivor) — the two inputs to the reliability diagram.
    """
    if len(survivor_cells) != len(found_flags):
        raise ValueError("survivor_cells and found_flags must be the same length")
    pod, ok = [], []
    n_north, n_east = pod_grid.shape
    for (i, j), f in zip(survivor_cells, found_flags):
        if not (0 <= i < n_north and 0 <= j < n_east):
            raise IndexError(f"survivor cell ({i}, {j}) is outside the {n_north}x{n_east} coverage grid")
        pod.append(float(pod_grid[i, j]))
        ok.append(bool(f))
    return pod, ok


__all__ = [
    "DEFAULT_BINS", "PodBin", "evaluate_pod_calibration", "reliability_bins", "survivors_found_by_cell",
    "wilson_interval",
]
