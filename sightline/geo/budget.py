"""The §5.7 geolocation error budget as executable code, not a comment.

Every number a record publishes as `h_acc_m` comes from here, evaluated at **that pixel's** off-nadir angle.
The dedup radius (§5.6: DBSCAN at 2 x CE90) and the map's uncertainty circle are both downstream of it.

Model
-----
θ is the angle between the ray and the vertical (0 = nadir). With camera height *h* above the intersected
surface, the doc's sensitivities are

============  ==========================  ===============================================================
term          sensitivity                 driven by
============  ==========================  ===============================================================
pointing      ``h · sec²θ · δθ``          vehicle pitch/roll ⊕ gimbal angle report ⊕ boresight alignment
heading       ``h · tanθ · δψ``           magnetometer yaw
altitude      ``tanθ · δh``               barometric AGL, or the DEM's vertical error
pixel         ``h · sec²θ · (δu/f)``      detection-box localisation in pixels
GNSS          ``δp``                      receiver horizontal accuracy
time-sync     ``v · δt``                  image-to-telemetry timestamp offset at ground speed v
============  ==========================  ===============================================================

The terms are treated as independent and combined in quadrature. `h_acc_m` is a **per-axis 1σ**, which is the
convention `schemas.CE90_FACTOR` assumes (CE90 = 2.1460σ is the 90th percentile of a 2-D circular Rayleigh with
equal per-axis σ).

Verification against the doc's table lives in `DOC_TABLE` / `compare_to_doc()` and is asserted in
`tests/test_geo.py`. 19 of the doc's 21 cells reproduce exactly at 1-decimal rounding; the two that do not are
recorded in `KNOWN_DOC_DISCREPANCIES` with the derivation, never fudged away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from sightline.schemas import CE90_FACTOR

__all__ = [
    "BudgetInputs",
    "CONSUMER",
    "RTK",
    "PESSIMISTIC",
    "CONSUMER_DEM_AGL",
    "PRESETS",
    "error_terms",
    "h_acc_m",
    "ce90_m",
    "dedup_radius_m",
    "DOC_TABLE",
    "KNOWN_DOC_DISCREPANCIES",
    "compare_to_doc",
    "format_doc_comparison",
    "MAX_OFF_NADIR_DEG",
]

#: `t = h / d_z` with the doc's `d_z <= 0.1` rejection => the largest usable ray-to-vertical angle.
MAX_OFF_NADIR_DEG = math.degrees(math.acos(0.1))  # 84.26°

#: 4K frame at HFOV 73.74° -> f = (3840/2)/tan(36.87°) = 2560 px. Used only when no real `Intrinsics.fx` is
#: available; the chain always passes the camera's actual fx, so the published `h_acc_m` is camera-specific.
DEFAULT_FOCAL_PX = 2560.0


@dataclass(frozen=True, slots=True)
class BudgetInputs:
    """One row of §5.7's "Inputs used". Angles in degrees, everything else SI.

    `name` travels onto `GeoFix.h_acc_basis` so a published radius says which assumptions produced it.
    """

    name: str = "consumer"
    gnss_h_m: float = 2.5  # consumer GNSS 2.5 m (RTK 0.05 m)
    attitude_deg: float = 0.5  # vehicle pitch/roll (pessimistic 1.7°)
    yaw_deg: float = 1.5  # magnetometer heading (pessimistic 3.0°)
    gimbal_deg: float = 0.2  # gimbal angle report
    boresight_deg: float = 0.3  # camera-to-gimbal boresight alignment
    agl_sigma_m: float = 1.0  # barometric AGL (4 m when the AGL comes from a DEM)
    pixel_sigma_px: float = 5.0  # detection localisation
    focal_px: float = DEFAULT_FOCAL_PX
    speed_ms: float = 5.0  # ground speed for the time-sync term (doc: 5-10 m/s)
    sync_s: float = 0.040  # image-to-telemetry sync (doc: 40-50 ms)

    @property
    def pointing_sigma_deg(self) -> float:
        """Vehicle attitude, gimbal report and boresight, in quadrature — the doc's `δθ`."""
        return math.sqrt(self.attitude_deg**2 + self.gimbal_deg**2 + self.boresight_deg**2)

    @property
    def pixel_sigma_rad(self) -> float:
        return self.pixel_sigma_px / self.focal_px

    def with_(self, **kw: float | str) -> "BudgetInputs":
        """A copy with fields overridden (the chain uses this to inject the real fx, GNSS σ and speed)."""
        return replace(self, **kw)  # type: ignore[arg-type]


#: §5.7 "Inputs used": the four rows the doc's table is computed from.
CONSUMER = BudgetInputs()
RTK = BudgetInputs(name="rtk", gnss_h_m=0.05)
PESSIMISTIC = BudgetInputs(name="pessimistic", attitude_deg=1.7, yaw_deg=3.0)
CONSUMER_DEM_AGL = BudgetInputs(name="consumer_dem_agl", agl_sigma_m=4.0)
PRESETS: dict[str, BudgetInputs] = {p.name: p for p in (CONSUMER, RTK, PESSIMISTIC, CONSUMER_DEM_AGL)}


# --- the model ---------------------------------------------------------------------------------------------
def error_terms(h_agl_m: float, off_nadir_deg: float, inp: BudgetInputs = CONSUMER) -> dict[str, float]:
    """Every term of the budget in metres, so a UI can show *why* a circle is the size it is.

    `h_agl_m` is the camera's height above the surface the ray hit (not its altitude MSL). `off_nadir_deg` is
    θ, the ray-to-vertical angle of **this pixel**, not the gimbal's nominal pitch.
    """
    th = math.radians(min(abs(float(off_nadir_deg)), MAX_OFF_NADIR_DEG))
    sec2 = 1.0 / (math.cos(th) ** 2)
    tan = math.tan(th)
    h = float(h_agl_m)
    return {
        "pointing": h * sec2 * math.radians(inp.pointing_sigma_deg),
        "heading": h * tan * math.radians(inp.yaw_deg),
        "altitude": tan * inp.agl_sigma_m,
        "pixel": h * sec2 * inp.pixel_sigma_rad,
        "gnss": float(inp.gnss_h_m),
        "sync": float(inp.speed_ms) * float(inp.sync_s),
    }


def h_acc_m(h_agl_m: float, off_nadir_deg: float, inp: BudgetInputs = CONSUMER) -> float:
    """Horizontal 1σ in metres for this geometry — the number published on `GeoFix.h_acc_m`."""
    return math.sqrt(sum(v * v for v in error_terms(h_agl_m, off_nadir_deg, inp).values()))


def ce90_m(h_acc: float) -> float:
    """90th-percentile circular error. Same factor as `schemas.ce90_m`; re-exported for budget-only callers."""
    return CE90_FACTOR * h_acc


def dedup_radius_m(h_acc: float) -> float:
    """§5.6: DBSCAN clusters tracks at 2 x CE90. Published here so geo and dedup cannot drift apart."""
    return 2.0 * ce90_m(h_acc)


def dominant_term(h_agl_m: float, off_nadir_deg: float, inp: BudgetInputs = CONSUMER) -> str:
    """Which term carries the budget — the doc's reading ("nadir is GNSS-dominated, 45° is attitude-dominated")."""
    t = error_terms(h_agl_m, off_nadir_deg, inp)
    return max(t, key=lambda k: t[k])


# --- reproduction of the doc's table -----------------------------------------------------------------------
#: SOLUTION_DOC §5.7 "Error budget". (label, θ in degrees, preset, {AGL: doc value in metres}).
DOC_TABLE: tuple[tuple[str, float, BudgetInputs, dict[int, float]], ...] = (
    ("Nadir, image centre, consumer GNSS (1σ)", 0.0, CONSUMER, {30: 2.5, 60: 2.6, 100: 2.7}),
    ("Nadir, image centre, RTK", 0.0, RTK, {30: 0.4, 60: 0.7, 100: 1.1}),
    ("Nadir, frame edge (θ=36°), consumer, AGL ±1 m", 36.0, CONSUMER, {30: 2.7, 60: 3.0, 100: 3.6}),
    ("Nadir, frame edge, AGL from DEM (±4 m)", 36.0, CONSUMER_DEM_AGL, {30: 3.9, 60: 4.1, 100: 4.6}),
    ("Oblique -45°, image centre, consumer", 45.0, CONSUMER, {30: 2.9, 60: 3.4, 100: 4.4}),
    ("Oblique -45°, RTK", 45.0, RTK, {30: 1.5, 60: 2.3, 100: 3.6}),
    ("Oblique -45°, pessimistic attitude (1.7°/3°), consumer", 45.0, PESSIMISTIC, {30: 3.9, 60: 5.5, 100: 8.5}),
)

#: Cells where this implementation disagrees with the doc's printed value. Both are in the 30 m column of an
#: oblique row; every other cell reproduces exactly at 1-decimal rounding, which is why these are reported
#: rather than tuned away. Values are (row label, AGL, doc value, value derived here).
#:
#: 1. "Oblique -45°, pessimistic, 30 m": doc 3.9, derived 3.617. The same row's 60 m (5.52 -> 5.5) and 100 m
#:    (8.46 -> 8.5) cells match, so the model is right and the printed 30 m cell is not: 3.9 would need
#:    h ≈ 35 m, or δθ ≈ 2.23° instead of 1.74°. 3.9 is also the value printed directly above it in the
#:    "AGL from DEM" row's 30 m cell, which is what a copy/paste error looks like.
#: 2. "Oblique -45°, RTK, 30 m": doc 1.5, derived 1.446 (rounds to 1.4). A 0.054 m gap — the doc's value is
#:    recovered exactly if that one cell used the fast end of the sync term (10 m/s x 50 ms = 0.5 m instead of
#:    5 m/s x 40 ms = 0.2 m), but using 0.5 m everywhere breaks the nadir-centre row (2.5 -> 2.6), so the
#:    table is not internally consistent on that term. This lane keeps 0.2 m, which reproduces 19 of 21 cells.
KNOWN_DOC_DISCREPANCIES: dict[tuple[str, int], tuple[float, float]] = {
    ("Oblique -45°, pessimistic attitude (1.7°/3°), consumer", 30): (3.9, 3.617),
    ("Oblique -45°, RTK", 30): (1.5, 1.446),
}


def compare_to_doc() -> list[dict[str, object]]:
    """One row per (geometry, altitude) cell: the doc's value, this model's value and whether they round equal."""
    rows: list[dict[str, object]] = []
    for label, theta, inp, cells in DOC_TABLE:
        for agl, doc_v in cells.items():
            got = h_acc_m(float(agl), theta, inp)
            rows.append(
                {
                    "label": label,
                    "off_nadir_deg": theta,
                    "preset": inp.name,
                    "agl_m": agl,
                    "doc_m": doc_v,
                    "computed_m": got,
                    "rounded_m": round(got, 1),
                    "matches": round(got, 1) == round(doc_v, 1),
                    "delta_m": got - doc_v,
                    "dominant": dominant_term(float(agl), theta, inp),
                }
            )
    return rows


def format_doc_comparison() -> str:
    """A printable side-by-side of the doc's table and this implementation (used in `docs/lanes/geo.md`)."""
    out = [f"{'geometry':<56}{'AGL':>5}{'doc':>7}{'ours':>8}{'diff':>8}  {'dominant term'}"]
    out.append("-" * 100)
    for r in compare_to_doc():
        flag = "" if r["matches"] else "  <-- MISMATCH"
        out.append(
            f"{str(r['label'])[:55]:<56}{r['agl_m']:>4} m{r['doc_m']:>7.1f}"
            f"{r['computed_m']:>8.3f}{r['delta_m']:>+8.3f}  {r['dominant']}{flag}"
        )
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover - `uv run python -m sightline.geo.budget`
    print(format_doc_comparison())
