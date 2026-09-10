"""Presentation classes, their critical dimensions, the cameras, and the zone mixes (SOLUTION_DOC §5.3b, §2.5).

The map must say *searched for what*. A cell flown once at 60 m is thoroughly searched for a body lying on a roof
and barely searched at all for a hand protruding from mud, so the coverage raster is a short stack — one layer per
presentation class — and the default view is a zone-weighted mixture of the layers.

Everything here is arithmetic on numbers that are printed in the solution document, so `tests/test_coverage.py`
re-derives the document's own tables from this module and fails if they drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sightline.schemas import PRESENTATIONS, Intrinsics

# --- 1. presentation classes ------------------------------------------------------------------------------
#: Critical dimension in metres, nadir view (§5.3b table). "body" is the MVP name for the prone/supine class and
#: is a synonym of "prone"; both are kept because `PRESENTATIONS` (frozen in schemas.py) contains both.
CRITICAL_DIM_M: dict[str, float] = {
    "body": 1.70,       # prone or supine body: deposit fan, rooftops, mud margins
    "prone": 1.70,      # synonym of "body"
    "cluster": 1.50,    # three or more together: rooftops, elevated roads
    "upright": 0.45,    # upright, sitting or crouched: rooftops, upper floors, high ground
    "wading": 0.45,     # wading or clinging, torso above water: channel, flooded streets
    "head_only": 0.25,  # head and shoulders: fast channel, deep water
    "limb_only": 0.18,  # hand, foot, forearm: deposit fan, debris dams, collapsed structures
    "buried": 0.0,      # fully buried: no visible part in any band (§2.7) — this layer is identically zero
}
assert set(CRITICAL_DIM_M) == set(PRESENTATIONS), "presentation set must match schemas.PRESENTATIONS"

#: The MVP pair the tracker asks for; the machinery below works for the full tuple.
MVP_PRESENTATIONS: tuple[str, ...] = ("body", "limb_only")

#: The one class for which no aerial coverage ever accumulates (§5.3b / §2.7). Not a special case in the code:
#: its visibility factor V(cell, j) is zero everywhere, so its layer stays at zero by construction.
ZERO_LAYER_PRESENTATIONS: frozenset[str] = frozenset({"buried"})

#: §2.5 design rule: >= 20 px along the critical dimension for "90 % recall plausible"; 8-20 px is cue-only.
MIN_PX_FOR_RECALL = 20.0
PX_CUE_FLOOR = 8.0
#: Drone Rules 2021 green-zone ceiling (§2.6). Also the ceiling the planner will never exceed.
LEGAL_CEILING_M = 120.0


# --- 2. cameras -------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CameraModel:
    """A camera as the coverage model needs it: pixel width, horizontal FOV, band, and a human-readable source."""

    name: str
    width_px: int
    height_px: int
    hfov_deg: float
    band: str  # "rgb" | "thermal"
    source: str = ""

    @property
    def fx_px(self) -> float:
        """Focal length in pixels; GSD at nadir is simply agl / fx."""
        return (self.width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def intrinsics(self) -> Intrinsics:
        return Intrinsics.from_hfov(self.width_px, self.height_px, self.hfov_deg, source="dronemodels")

    def gsd_m(self, agl_m: float) -> float:
        """Appendix B: GSD = 2 h tan(HFOV/2) / W_px."""
        return 2.0 * agl_m * math.tan(math.radians(self.hfov_deg) / 2.0) / self.width_px

    def swath_m(self, agl_m: float) -> float:
        """Appendix B footprint width at nadir = 2 h tan(HFOV/2)."""
        return 2.0 * agl_m * math.tan(math.radians(self.hfov_deg) / 2.0)

    def px_on_target(self, length_m: float, agl_m: float) -> float:
        return length_m / self.gsd_m(agl_m) if agl_m > 0 else float("inf")

    def ceiling_m(self, presentation: str, min_px: float = MIN_PX_FOR_RECALL, legal_cap: float = LEGAL_CEILING_M
                  ) -> float:
        """Highest AGL at which this presentation still spans `min_px`, capped by the legal ceiling (§5.3b)."""
        length_m = CRITICAL_DIM_M[presentation]
        if length_m <= 0.0:
            return 0.0  # fully buried: never
        return min(legal_cap, self.geometric_ceiling_m(presentation, min_px))

    def geometric_ceiling_m(self, presentation: str, min_px: float = MIN_PX_FOR_RECALL) -> float:
        """The same ceiling with no legal cap — the doc quotes this for the prone body ("geometric ~227 m")."""
        length_m = CRITICAL_DIM_M[presentation]
        if length_m <= 0.0:
            return 0.0
        return length_m * self.fx_px / min_px


def _camera_from_gsd(name: str, w: int, h: int, gsd_cm_at_60m: float, band: str, source: str) -> CameraModel:
    """Build a camera from the §2.5 GSD table entry, which is how every number below is anchored."""
    half = (gsd_cm_at_60m / 100.0) * w / (2.0 * 60.0)
    return CameraModel(name, w, h, math.degrees(2.0 * math.atan(half)), band, source)


#: The §2.5 cameras, each reconstructed from its published GSD at 60 m so the swaths in that table come back out.
M3T_WIDE_4K = _camera_from_gsd("M3T/M30T/H30T wide 4K", 3840, 2160, 2.25, "rgb", "SOLUTION_DOC §2.5 table")
M3T_THERMAL_640 = _camera_from_gsd("M3T/M30T thermal 640x512", 640, 512, 7.91, "thermal", "SOLUTION_DOC §2.5")
H20T_THERMAL_640 = _camera_from_gsd("H20T thermal 640x512", 640, 512, 5.33, "thermal", "SOLUTION_DOC §2.5")
H30T_THERMAL_1280 = _camera_from_gsd("H30T thermal 1280x1024", 1280, 1024, 3.05, "thermal", "SOLUTION_DOC §2.5")
#: What the simulator actually renders (sim/settings/capture_4k.json: 3840x2160 at FOV_Degrees 75.5).
SIM_RGB_4K = CameraModel("sim RGB 4K", 3840, 2160, 75.5, "rgb", "sim/settings/capture_4k.json")
SIM_RGB_1080 = CameraModel("sim RGB 1080p", 1920, 1080, 75.5, "rgb", "sim/settings/default.json")

CAMERAS: dict[str, CameraModel] = {
    c.name: c for c in (M3T_WIDE_4K, M3T_THERMAL_640, H20T_THERMAL_640, H30T_THERMAL_1280, SIM_RGB_4K, SIM_RGB_1080)
}


# --- 3. the zone mixture w_j(zone) ------------------------------------------------------------------------
# §5.3b: "the mix w_j(zone) comes straight from the placement distribution already specified in §2.3 row 2 and
# §6.2: rooftop and upright presentations dominate the flooded settlement, head-only and clinging dominate the
# channel, prone and limb-only dominate the deposit fan."
#
# STATUS: *proposed*, derived from §6.2's placement distribution (50 % roof or upper floor, 20 % tree/road/vehicle
# roof, 20 % wading or clinging, 10 % head-only, plus buried actors on the fan) re-apportioned per zone as the
# §5.3b prose states. It is a modelling choice, not a measurement. `ZONE_MIX` is replaced wholesale once the
# verifier head (F10) reports observed presentations — see `update_mix_from_observations`.
ZONE_MIX: dict[str, dict[str, float]] = {
    "settlement": {"body": 0.20, "cluster": 0.20, "upright": 0.40, "wading": 0.05,
                   "head_only": 0.02, "limb_only": 0.08, "buried": 0.05},
    "channel": {"body": 0.15, "cluster": 0.02, "upright": 0.10, "wading": 0.35,
                "head_only": 0.30, "limb_only": 0.05, "buried": 0.03},
    "fan": {"body": 0.35, "cluster": 0.05, "upright": 0.10, "wading": 0.03,
            "head_only": 0.02, "limb_only": 0.25, "buried": 0.20},
    "hillslope": {"body": 0.25, "cluster": 0.10, "upright": 0.45, "wading": 0.02,
                  "head_only": 0.03, "limb_only": 0.10, "buried": 0.05},
}
#: Zone-agnostic fallback: the §6.2 placement distribution with no zone information at all.
ZONE_MIX["unknown"] = {"body": 0.25, "cluster": 0.10, "upright": 0.30, "wading": 0.15,
                       "head_only": 0.10, "limb_only": 0.06, "buried": 0.04}


def mix_for_zone(zone: str, presentations: tuple[str, ...] | None = None) -> dict[str, float]:
    """The presentation mixture for a zone, renormalised over `presentations` (default: the keys in ZONE_MIX).

    The mixture is renormalised rather than truncated so that a two-layer MVP map still reports a proper weighted
    average (a mixture that does not sum to 1 would silently depress every POD it displays).
    """
    base = ZONE_MIX.get(zone) or ZONE_MIX["unknown"]
    keys = tuple(base) if presentations is None else tuple(presentations)
    w = {k: float(base.get(k, 0.0)) for k in keys}
    total = sum(w.values())
    if total <= 0.0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in w.items()}


def update_mix_from_observations(zone: str, counts: dict[str, int], prior_strength: float = 20.0
                                 ) -> dict[str, float]:
    """§5.3b: "the observed presentations from the verifier head can update the assumed mix as the flight proceeds".

    Dirichlet update with the tabulated mix as the prior (`prior_strength` pseudo-counts). A fan that turns out to
    be producing limb-only detections re-weights its own map. Returns a new mixture; `ZONE_MIX` is not mutated.
    """
    prior = mix_for_zone(zone)
    post = {k: prior_strength * v + float(counts.get(k, 0)) for k, v in prior.items()}
    for k, n in counts.items():  # a presentation observed but absent from the prior still gets its mass
        if k not in post:
            post[k] = float(n)
    total = sum(post.values())
    return {k: v / total for k, v in post.items()}


def altitude_ceiling_table(camera: CameraModel, min_px: float = MIN_PX_FOR_RECALL) -> dict[str, float]:
    """The §5.3b ceiling table for one camera: presentation -> highest AGL still >= `min_px` (legal cap applied)."""
    return {p: camera.ceiling_m(p, min_px) for p in ("body", "cluster", "upright", "wading",
                                                     "head_only", "limb_only", "buried")}
