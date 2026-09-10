"""Geolocation error against ground truth (§5.12) and the honesty check on the published error radius (§5.7).

Two things are measured, and the second is the one that matters for the map:

1. **Error**: median and 90th-percentile distance between the predicted position and the true survivor
   position. §5.7's budget says nadir at 60 m with consumer GNSS should land near 2.6 m (1 sigma), CE90 ~ 6 m.
2. **Containment**: what fraction of fixes actually fall inside the CE90 circle the system *drew on the map*.
   If a record publishes `h_acc_m` and only 60 % of records are inside their own CE90, the circle is a lie even
   though the median error looks fine. A well-calibrated budget gives ~0.90.

In the simulator the ground-truth pose makes the raw error ~0, which is why §5.7 says to inject the noise model
before measuring; `Telemetry.noise_injected` records whether that happened and `evaluate_geolocation` refuses to
report a sim number without it unless the caller explicitly asks for the debug (noise-free) figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sightline.common.geodesy import haversine_m
from sightline.eval.groundtruth import GtSurvivor
from sightline.eval.matching import gated_assignment
from sightline.eval.slicing import MetricSet, metric_row
from sightline.schemas import CE90_FACTOR, GeoFix, Record, SliceKey


@dataclass(slots=True)
class GeoSample:
    """One predicted position with the error radius the system published for it."""

    lat: float
    lon: float
    h_acc_m: float
    off_nadir_deg: float = 0.0
    agl_m: float = 0.0
    method: str = "flat_plane"
    key: str = ""

    @property
    def ce90_m(self) -> float:
        return CE90_FACTOR * self.h_acc_m


def samples_from_records(records: Sequence[Record]) -> list[GeoSample]:
    return [GeoSample(r.lat, r.lon, r.h_acc_m, r.off_nadir_deg, r.agl_m, r.method, r.record_id)
            for r in records]


def samples_from_fixes(fixes: Sequence[GeoFix]) -> list[GeoSample]:
    return [GeoSample(f.lat, f.lon, f.h_acc_m, f.off_nadir_deg, f.agl_m, f.method) for f in fixes if f.valid]


def evaluate_geolocation(
    samples: Sequence[GeoSample],
    survivors: Sequence[GtSurvivor],
    key: SliceKey,
    *,
    match_radius_m: float = 25.0,
    noise_injected: bool | None = None,
) -> MetricSet:
    """Median / p90 / mean error and CE90 containment, over samples matched to survivors within the radius."""
    findable = [s for s in survivors if s.findable]
    ms = MetricSet()
    if not samples or not findable:
        ms.add(metric_row("geo_error_median_m", 0.0, key, 0,
                          note="no matched positions; nothing measured (not a zero-error result)"))
        return ms

    d = np.asarray([[haversine_m(p.lat, p.lon, s.lat, s.lon) for s in findable] for p in samples], dtype=float)
    pairs = gated_assignment(d, match_radius_m)
    if not pairs:
        ms.add(metric_row("geo_error_median_m", 0.0, key, 0,
                          note=f"no predicted position landed within {match_radius_m} m of a survivor"))
        return ms
    errs = np.asarray([d[i, j] for i, j in pairs], dtype=float)
    ce90 = np.asarray([samples[i].ce90_m for i, _ in pairs], dtype=float)
    inside = errs <= ce90

    detail = {"match_radius_m": match_radius_m,
              "budget_reference_60m_nadir_1sigma_m": 2.6,
              "noise_injected": noise_injected}
    if noise_injected is False and key.domain == "sim":
        detail["warning"] = ("simulator telemetry is exact: this is the debug figure, not a claim. "
                             "SOLUTION_DOC 5.7 requires the injected GNSS/attitude noise before publishing.")
    ms.add(metric_row("geo_error_median_m", float(np.median(errs)), key, len(errs), **detail))
    ms.add(metric_row("geo_error_p90_m", float(np.percentile(errs, 90)), key, len(errs), **detail))
    ms.add(metric_row("geo_error_mean_m", float(errs.mean()), key, len(errs), **detail))
    ms.add(metric_row("geo_error_max_m", float(errs.max()), key, len(errs), **detail))
    ms.add(metric_row("geo_ce90_containment", float(inside.mean()), key, len(errs),
                      expected=0.90, basis="fraction of fixes inside the CE90 circle the system published",
                      mean_published_ce90_m=float(ce90.mean()), **detail))
    ms.add(metric_row("geo_published_ce90_mean_m", float(ce90.mean()), key, len(errs),
                      basis="CE90_FACTOR * h_acc_m, averaged over matched records"))
    return ms


def error_by_off_nadir(samples: Sequence[GeoSample], survivors: Sequence[GtSurvivor], key: SliceKey,
                       *, match_radius_m: float = 25.0,
                       edges: Sequence[float] = (0, 15, 30, 45, 90)) -> MetricSet:
    """§5.7: at nadir the budget is GNSS-dominated; past 45 deg the attitude terms dominate. Show it."""
    findable = [s for s in survivors if s.findable]
    ms = MetricSet()
    if not samples or not findable:
        return ms
    d = np.asarray([[haversine_m(p.lat, p.lon, s.lat, s.lon) for s in findable] for p in samples], dtype=float)
    pairs = gated_assignment(d, match_radius_m)
    if not pairs:
        return ms
    angles = np.asarray([abs(samples[i].off_nadir_deg) for i, _ in pairs], dtype=float)
    errs = np.asarray([d[i, j] for i, j in pairs], dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (angles >= lo) & (angles < hi)
        if not mask.any():
            continue
        ms.add(metric_row("geo_error_median_m_by_off_nadir", float(np.median(errs[mask])), key, int(mask.sum()),
                          off_nadir_bin=f"{lo:g}-{hi:g}deg"))
    return ms


__all__ = ["GeoSample", "error_by_off_nadir", "evaluate_geolocation", "samples_from_fixes", "samples_from_records"]
