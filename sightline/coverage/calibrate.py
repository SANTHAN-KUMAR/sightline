"""Calibrating `k` in POD = 1 - exp(-k C), per presentation (SOLUTION_DOC §5.3, §5.3b, §5.12).

Two things live here, and the difference between them is the honest part of this lane:

  * `calibrate_k_from_outcomes()` / `reliability_diagram()` — **the real calibration**, the §5.12 procedure:
    fit `k` to (coverage, was-the-target-found) pairs from the held-out clip by maximum likelihood, then check the
    fit with a reliability diagram, one per presentation class. This is implemented and tested; it simply has no
    data to eat yet, because the detector (F8) does not exist.

  * `DEFAULT_K` — **a labelled placeholder**, derived rather than measured. See `k_for_reference_quality()`.

The shipped `DEFAULT_K` values are therefore marked `measured=False` and must be described as derived, never as
calibrated, anywhere they are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sightline.coverage.presentation import CRITICAL_DIM_M, SIM_RGB_4K, CameraModel
from sightline.coverage.quality import Conditions, analytic_recall

#: POD is clamped below 1 (§5.3: "clamped so POD never reaches 1"). A saturated cell still carries a 1 % chance
#: that a person is there and was missed, which is the numeric form of "the system recommends, never closes".
POD_MAX = 0.99
#: `k` is clamped into this range. The floor is the random-search lower bound POD = 1 - e^(-C) of §5.3: the map
#: may never claim *less* than random search. The ceiling stops a near-perfect reference quality from producing an
#: absurdly saturating map.
K_MIN, K_MAX = 1.0, 3.0

#: The nominal mission slice of §5.5c, which is the operating point the shipped `k` is derived at.
NOMINAL_AGL_M = 60.0
NOMINAL_CONDITIONS = Conditions(band="rgb", time_of_day="day", weather="dry", speed_ms=6.0, exposure_s=1.0 / 500.0)


def k_for_reference_quality(q_ref: float, repeat_correlation: float = 0.0) -> float:
    """`k` such that ONE pass of quality `q_ref` reports POD exactly `q_ref`, and n passes compose as n
    independent looks: 1 - exp(-k * n * q_ref) = 1 - (1 - q_ref)^n.

    Derivation: setting 1 - exp(-k q) = q gives k = -ln(1 - q) / q. At the operating point the map therefore reads
    back the detector's own recall after one pass ("POD 0.41 after two passes at 60 m" means what it says), and
    away from it the exponential is an approximation — which is precisely what the reliability diagram of §5.12
    exists to correct.

    `repeat_correlation` (rho) shrinks `k` by (1 - rho) for the view that repeat passes in identical conditions are
    correlated rather than independent. It defaults to 0 because the coverage model of §5.3 has no other place to
    put correlation, and because under-reporting the first pass is a worse failure than over-crediting the third.
    """
    q = float(min(max(q_ref, 1e-6), 0.999))
    k = -math.log(1.0 - q) / q
    return float(min(K_MAX, max(K_MIN, k * (1.0 - float(repeat_correlation)))))


@dataclass(frozen=True, slots=True)
class KValue:
    """A `k` with its provenance. `measured=False` means derived from the model, not fitted to outcomes."""

    k: float
    measured: bool
    q_ref: float = 0.0
    n: int = 0
    basis: str = ""


def derive_default_k(camera: CameraModel = SIM_RGB_4K, agl_m: float = NOMINAL_AGL_M,
                     cond: Conditions = NOMINAL_CONDITIONS, repeat_correlation: float = 0.0
                     ) -> dict[str, KValue]:
    """Derive one `k` per presentation from the interim recall model at the nominal slice. NOT a measurement."""
    gsd = camera.gsd_m(agl_m)
    out: dict[str, KValue] = {}
    for p in CRITICAL_DIM_M:
        if CRITICAL_DIM_M[p] <= 0.0:
            out[p] = KValue(K_MIN, False, 0.0, 0, "zero-layer presentation: k is irrelevant, the layer is zero")
            continue
        q = analytic_recall(p, gsd, cond, 0.0).value
        out[p] = KValue(
            k_for_reference_quality(q, repeat_correlation), False, q, 0,
            f"derived: k = -ln(1-q)/q at q={q:.3f} ({camera.name}, {agl_m:.0f} m AGL, "
            f"{cond.time_of_day}/{cond.band}/{cond.weather}) — PLACEHOLDER until F19 fits it",
        )
    return out


#: Shipped defaults. Derived, not calibrated — see the module docstring.
DEFAULT_K: dict[str, KValue] = derive_default_k()


def k_for(presentation: str) -> float:
    return DEFAULT_K[presentation].k if presentation in DEFAULT_K else K_MIN


# --- the real calibration path ----------------------------------------------------------------------------
def _log_likelihood(k: float, coverage: np.ndarray, found: np.ndarray) -> float:
    kc = np.clip(k * coverage, 0.0, 700.0)
    miss = np.exp(-kc)
    pod = np.clip(1.0 - miss, 1e-12, 1.0 - 1e-12)
    return float(np.sum(np.where(found, np.log(pod), -kc)))


def calibrate_k_from_outcomes(coverage: np.ndarray, found: np.ndarray, k_lo: float = K_MIN, k_hi: float = K_MAX,
                              iters: int = 80) -> KValue:
    """Maximum-likelihood `k` for POD = 1 - exp(-k C) given per-target accumulated coverage and find/miss flags.

    `coverage[i]` is the coverage accumulated in the cell holding ground-truth target i at the moment it was (or
    was not) found; `found[i]` is 1/0. Ternary search on a unimodal log-likelihood, bounded by [K_MIN, K_MAX].
    """
    c = np.asarray(coverage, dtype=float).ravel()
    y = np.asarray(found).ravel().astype(bool)
    if c.size == 0 or c.size != y.size:
        raise ValueError("coverage and found must be the same non-empty length")
    lo, hi = float(k_lo), float(k_hi)
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if _log_likelihood(m1, c, y) < _log_likelihood(m2, c, y):
            lo = m1
        else:
            hi = m2
    k = 0.5 * (lo + hi)
    return KValue(float(k), True, float(y.mean()), int(c.size),
                  f"MLE on {c.size} held-out targets, mean find rate {y.mean():.3f}")


def wilson_interval(k_found: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — §5.3b asks for confidence intervals on the thin (limb-only, head-only) layers."""
    if n <= 0:
        return (0.0, 1.0)
    p = k_found / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(slots=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    predicted: float
    observed: float
    ci_lo: float
    ci_hi: float


def reliability_diagram(predicted_pod: np.ndarray, found: np.ndarray, n_bins: int = 5) -> list[ReliabilityBin]:
    """§5.12's check: do cells predicting 0.4 really surface about 40 % of the ground truth in them?

    One diagram per presentation class (§5.3b). Bins are equal-width in predicted POD; empty bins are dropped.
    """
    p = np.asarray(predicted_pod, dtype=float).ravel()
    y = np.asarray(found).ravel().astype(bool)
    if p.size != y.size:
        raise ValueError("predicted_pod and found must be the same length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        hits = int(y[sel].sum())
        ci = wilson_interval(hits, n)
        out.append(ReliabilityBin(float(lo), float(hi), n, float(p[sel].mean()), hits / n, ci[0], ci[1]))
    return out


def expected_calibration_error(bins: list[ReliabilityBin]) -> float:
    """Sample-weighted mean |predicted - observed| over the reliability bins."""
    n = sum(b.n for b in bins)
    if n == 0:
        return 0.0
    return float(sum(b.n * abs(b.predicted - b.observed) for b in bins) / n)
