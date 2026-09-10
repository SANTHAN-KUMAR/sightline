"""`R_slice` and `q_pass`: how well one look at one cell searched it (SOLUTION_DOC §5.3, §5.3b, Appendix B).

    q_pass(cell, j) = R_slice(j ; GSD, band, time, blur, view) x V(cell, j)

`R_slice` is *meant* to be the detector recall measured on the evaluation slice matching those conditions (§5.12).
The detector does not exist yet, so this module ships two things:

  1. `SliceTable` — the real path. It is a lookup of measured recalls keyed by the §5.12 slice grid, populated from
     the evaluation lane's `MetricRow`s. When a slice is present with enough samples it is used verbatim.
  2. `analytic_recall()` — the labelled interim model used when a slice is missing. It is a pixels-on-target
     logistic anchored on the published numbers in §2.5 and Appendix E, multiplied by documented condition factors.
     **It is a model, not a measurement**, and every value it produces is flagged `measured=False` so a caller can
     tell the difference and the map legend can say so.

Nothing here ever returns 1.0: recall saturates at `R_MAX` (0.97), so a single pass can never claim a cell is
completely searched. That is the numeric half of the "the system recommends, never closes" guardrail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sightline.coverage.presentation import CRITICAL_DIM_M, PX_CUE_FLOOR, ZERO_LAYER_PRESENTATIONS

# --- 1. the pixels-on-target curve ------------------------------------------------------------------------
# Anchors, all from the solution document, fitted as a logistic in log10(pixels across the critical dimension):
#   * 60 px  -> 0.90   HERIDAL/AIR: ~60 px persons at 2 cm GSD gave 92.9 % relaxed recall, 86.1 % at IoU 0.5
#                      (§2.5, Appendix E). 0.90 sits between the two, deliberately on the conservative side.
#   * 20 px  -> 0.70   the §2.5 design floor: ">= 20 px for 90 % recall plausible" is the point where recall
#                      becomes usable, not the point where it is already 0.90; C2A found sub-20 px objects
#                      "detected with less frequency and lower confidence".
#   * R_MAX  =  0.97   a detector never recalls everything; also enforces POD < 1 per pass.
# Sanity checks the fit reproduces without being told: 10 px -> 0.47 against AIResQ's mAP50 0.55 at ~10 px
# (conservative, and AIResQ is real thermal), 150 px (prone body at 30 m, 4K) -> 0.95.
R_MAX = 0.97
_ANCHORS = ((20.0, 0.70), (60.0, 0.90))


def _fit_logistic() -> tuple[float, float]:
    (p1, r1), (p2, r2) = _ANCHORS
    z1 = math.log((r1 / R_MAX) / (1.0 - r1 / R_MAX))
    z2 = math.log((r2 / R_MAX) / (1.0 - r2 / R_MAX))
    b = (z2 - z1) / (math.log10(p2) - math.log10(p1))
    log10_p50 = math.log10(p1) - z1 / b
    return b, 10.0**log10_p50


LOGIT_SLOPE, PX_HALF = _fit_logistic()  # ~3.356 and ~10.4 px


def recall_from_px(px: float) -> float:
    """Detector recall as a function of pixels across the critical dimension. Interim model — see module docstring.

    Below `PX_CUE_FLOOR` (8 px, "< 8 px is a blob" in §2.5) a quadratic taper is applied on top of the logistic,
    because the logistic alone still credits ~0.25 recall at 5 px, which the evidence does not support.
    """
    if px <= 0.0 or not math.isfinite(px):
        return 0.0
    r = R_MAX / (1.0 + math.exp(-LOGIT_SLOPE * (math.log10(px) - math.log10(PX_HALF))))
    if px < PX_CUE_FLOOR:
        r *= (px / PX_CUE_FLOOR) ** 2
    return float(min(r, R_MAX))


# --- 2. condition factors ---------------------------------------------------------------------------------
# Multiplicative factors on the pixel-limited recall. All *proposed* engineering choices anchored on §2.3's
# attribute table; each one is a slice the evaluation harness will eventually measure and overwrite.
TIME_OF_DAY_FACTOR: dict[tuple[str, str], float] = {
    ("day", "rgb"): 1.00,
    ("day", "thermal"): 0.55,      # §2.3 rows 20-23: midday solar loading on mud and roofing sheets kills contrast
    ("dawn", "rgb"): 0.75,
    ("dawn", "thermal"): 1.00,     # §2.7: pre-dawn is thermal's strongest window
    ("dusk", "rgb"): 0.70,
    ("dusk", "thermal"): 0.95,
    ("night", "rgb"): 0.12,        # RGB is very nearly blind; kept non-zero because lights and flares exist
    ("night", "thermal"): 1.00,
}
#: Thermal crossover: the ground and the person reach the same temperature and a person on a roof disappears for
#: about an hour (§2.5, Appendix E: crossover ~06:50 and ~18:05). Applied to the thermal band only.
CROSSOVER_WINDOWS_LOCAL_H: tuple[tuple[float, float], ...] = ((6.4, 7.4), (17.6, 18.6))
CROSSOVER_THERMAL_FACTOR = 0.35

WEATHER_FACTOR: dict[str, float] = {"dry": 1.00, "light_rain": 0.90, "heavy_rain": 0.70, "fog": 0.55}
#: Oblique views reveal people under eaves and trees (§2.3 row 13) but the GSD penalty is already handled
#: per-cell by the slant range, so this factor carries only the occlusion-reveal gain, and it is small.
VIEW_GAIN_MAX = 0.10


@dataclass(slots=True)
class Conditions:
    """Everything about one look that changes recall. Mirrors the §5.12 slice axes the map has to honour."""

    band: str = "rgb"  # "rgb" | "thermal"
    time_of_day: str = "day"  # "day" | "dawn" | "dusk" | "night"
    local_hour: float | None = None  # decimal local hour, only needed for the thermal crossover window
    weather: str = "dry"  # key of WEATHER_FACTOR
    speed_ms: float = 6.0
    exposure_s: float = 1.0 / 500.0
    zone: str = "unknown"

    def blur_px(self, gsd_m: float) -> float:
        """Appendix B: blur_px = v * t_exp / GSD."""
        return self.speed_ms * self.exposure_s / gsd_m if gsd_m > 0 else float("inf")

    def time_factor(self) -> float:
        f = TIME_OF_DAY_FACTOR.get((self.time_of_day, self.band), 1.0)
        if self.band == "thermal" and self.local_hour is not None:
            for lo, hi in CROSSOVER_WINDOWS_LOCAL_H:
                if lo <= self.local_hour <= hi:
                    f *= CROSSOVER_THERMAL_FACTOR
                    break
        return f

    def weather_factor(self) -> float:
        return WEATHER_FACTOR.get(self.weather, 1.0)


def effective_px(length_m: float, gsd_m: float, blur_px: float) -> float:
    """Pixels on target after motion blur has smeared them (§2.3 row 10: "blur > 1 px smears 10-20 px targets").

    Blur is folded into the *effective resolution* rather than applied as a separate recall factor, because that is
    physically what it does: px_eff = px / sqrt(1 + blur_px^2), which is a no-op at the 0.4 px of a 1/500 s
    exposure at 60 m and a factor of ~3.8 at the 3.7 px of a 1/60 s dusk exposure.
    """
    if gsd_m <= 0.0:
        return 0.0
    return (length_m / gsd_m) / math.sqrt(1.0 + max(0.0, blur_px) ** 2)


@dataclass(slots=True)
class RecallEstimate:
    """One `R_slice` value plus the provenance the map legend and the report have to state."""

    value: float
    measured: bool
    n: int = 0
    px: float = 0.0
    basis: str = ""

    def __float__(self) -> float:
        return self.value


def analytic_recall(presentation: str, gsd_m: float, cond: Conditions, off_nadir_deg: float = 0.0
                    ) -> RecallEstimate:
    """The interim `R_slice`: pixel-limited recall times the documented condition factors. `measured=False`."""
    length_m = CRITICAL_DIM_M[presentation]
    if presentation in ZERO_LAYER_PRESENTATIONS or length_m <= 0.0:
        return RecallEstimate(0.0, False, 0, 0.0, "zero-layer presentation (§5.3b): no aerial coverage accrues")
    px = effective_px(length_m, gsd_m, cond.blur_px(gsd_m))
    r = recall_from_px(px)
    r *= cond.time_factor() * cond.weather_factor()
    r *= 1.0 + VIEW_GAIN_MAX * math.sin(math.radians(min(abs(off_nadir_deg), 60.0)))
    return RecallEstimate(float(min(max(r, 0.0), R_MAX)), False, 0, px,
                          "analytic model (§2.5 anchors) — NOT a measured slice")


# --- 3. the measured path ---------------------------------------------------------------------------------
def altitude_band(agl_m: float) -> str:
    """The §5.12 altitude bands. Used as a slice key so measured recalls can be looked up by flight condition."""
    if agl_m < 45.0:
        return "30-45"
    if agl_m < 60.0:
        return "45-60"
    if agl_m < 90.0:
        return "60-90"
    return "90+"


@dataclass(slots=True)
class SliceTable:
    """Measured recall per (presentation, altitude band, band, time of day, zone), from the F19 evaluation.

    `min_n` is the sample floor below which a measured cell is distrusted and the analytic model is used instead;
    §5.3b warns that the limb-only and head-only layers will be the thin ones, so the floor matters.
    """

    rows: dict[tuple[str, str, str, str, str], tuple[float, int]] = field(default_factory=dict)
    min_n: int = 30
    source: str = ""

    @staticmethod
    def key(presentation: str, agl_m: float, band: str, time_of_day: str, zone: str = "all"
            ) -> tuple[str, str, str, str, str]:
        return (presentation, altitude_band(agl_m), band, time_of_day, zone)

    def put(self, presentation: str, agl_m: float, band: str, time_of_day: str, recall: float, n: int,
            zone: str = "all") -> None:
        self.rows[self.key(presentation, agl_m, band, time_of_day, zone)] = (float(recall), int(n))

    def lookup(self, presentation: str, agl_m: float, cond: Conditions, gsd_m: float, off_nadir_deg: float = 0.0
               ) -> RecallEstimate:
        """Measured slice if one exists with `n >= min_n` (zone-specific first, then zone-agnostic), else analytic."""
        for zone in (cond.zone, "all"):
            hit = self.rows.get(self.key(presentation, agl_m, cond.band, cond.time_of_day, zone))
            if hit is not None and hit[1] >= self.min_n:
                return RecallEstimate(min(hit[0], R_MAX), True, hit[1], CRITICAL_DIM_M[presentation] / max(gsd_m, 1e-9),
                                      f"measured slice zone={zone} {self.source}".strip())
        return analytic_recall(presentation, gsd_m, cond, off_nadir_deg)


#: The default table is empty: every value is analytic until the evaluation lane fills it.
DEFAULT_SLICE_TABLE = SliceTable(source="empty — no detector evaluated yet")


# --- 4. vectorised forms (one value per GRID CELL, because GSD varies across an oblique frame) --------------
def recall_from_px_array(px: np.ndarray) -> np.ndarray:
    """`recall_from_px` over an array. Identical arithmetic, including the sub-8-px blob taper."""
    p = np.asarray(px, dtype=float)
    safe = np.where(p > 0.0, p, 1e-9)
    r = R_MAX / (1.0 + np.exp(-LOGIT_SLOPE * (np.log10(safe) - math.log10(PX_HALF))))
    r = np.where(p < PX_CUE_FLOOR, r * (np.clip(p, 0.0, None) / PX_CUE_FLOOR) ** 2, r)
    return np.clip(np.where(p > 0.0, r, 0.0), 0.0, R_MAX)


def analytic_recall_array(presentation: str, gsd_m: np.ndarray, cond: Conditions,
                          off_nadir_deg: np.ndarray | float = 0.0) -> np.ndarray:
    """Per-cell interim `R_slice`. `gsd_m` is the per-cell ground sample distance from the frame's geometry."""
    length_m = CRITICAL_DIM_M[presentation]
    g = np.asarray(gsd_m, dtype=float)
    if presentation in ZERO_LAYER_PRESENTATIONS or length_m <= 0.0:
        return np.zeros_like(g)
    blur = np.where(g > 0.0, cond.speed_ms * cond.exposure_s / np.where(g > 0.0, g, 1.0), np.inf)
    px = np.where(g > 0.0, length_m / np.where(g > 0.0, g, 1.0), 0.0) / np.sqrt(1.0 + np.clip(blur, 0.0, 1e6) ** 2)
    r = recall_from_px_array(px)
    r = r * cond.time_factor() * cond.weather_factor()
    theta = np.clip(np.abs(np.asarray(off_nadir_deg, dtype=float)), 0.0, 60.0)
    r = r * (1.0 + VIEW_GAIN_MAX * np.sin(np.radians(theta)))
    return np.clip(r, 0.0, R_MAX)


def slice_recall_array(table: SliceTable, presentation: str, agl_m: float, cond: Conditions, gsd_m: np.ndarray,
                       off_nadir_deg: np.ndarray | float = 0.0) -> tuple[np.ndarray, RecallEstimate]:
    """Per-cell `R_slice` honouring a measured slice when one exists.

    A measured slice is a single number for a whole (presentation, altitude band, band, time) cell of the §5.12
    grid, so it cannot describe how recall falls off across an oblique frame. It is therefore applied as a *level*:
    the analytic per-cell shape is rescaled so its value at the frame's best GSD equals the measured recall.
    """
    g = np.asarray(gsd_m, dtype=float)
    shape = analytic_recall_array(presentation, g, cond, off_nadir_deg)
    best_gsd = float(np.min(g)) if g.size else 0.0
    frame_est = table.lookup(presentation, agl_m, cond, max(best_gsd, 1e-9), 0.0)
    if not frame_est.measured or shape.size == 0:
        return shape, frame_est
    ref = float(analytic_recall(presentation, max(best_gsd, 1e-9), cond, 0.0).value)
    scale = (frame_est.value / ref) if ref > 1e-9 else 0.0
    return np.clip(shape * scale, 0.0, R_MAX), frame_est
