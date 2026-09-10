"""The probability-of-area raster: where to look first (SOLUTION_DOC §2.4, §5.3, Appendix B).

    "A raster prior over the area of operations ... rooftops and upper floors of buildings inside the flood
     footprint (weighted by roof area and storeys), trees and terrain above flood level within ~200 m of
     inundated homes, stranded vehicles, the channel downstream with local maxima at bends, bridges, debris dams
     and recirculation zones, and last-known-position points from phones or reports."  (§5.3)

The prior is a probability *mass* over cells: it sums to 1 over the area of operations, so `POA x POD` is a
probability of success and the Bayesian update after an unsuccessful pass is `POA' proportional to POA (1 - POD)`
(Appendix B). Because POD is clamped below 1, no cell's POA ever reaches zero — that is the formal basis for
"the system recommends, never closes" (§5.3), and `bayesian_update` asserts it.

The weights below are *proposed*, anchored on §2.4's evidence (Kerala 2018 rooftop waits; Wayanad recoveries along
a 40 km river stretch at bends and debris dams). They are inputs a commander can override, not measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sightline.coverage.accumulate import ZONE_CODE, CoverageMap
from sightline.coverage.grid import SceneFrame, cell_centres_m, latlon_to_grid_ne

#: Base weight per zone before any feature is added (§2.4: three zones, three survivor populations).
ZONE_BASE_WEIGHT: dict[str, float] = {
    "settlement": 1.0,   # roofs and upper floors: Kerala 2018, people waited days on roofs
    "channel": 0.8,      # where bodies and clinging survivors accumulate (Wayanad, 40 km of Chaliyar)
    "fan": 0.5,          # deposit fan: many are buried here, and burial polygons remove those cells anyway
    "hillslope": 0.15,   # above flood level, low occupancy, but people do climb to it
    "unknown": 0.10,
}
#: A flooded house contributes roof_area_m2 * storeys * this, spread over a Gaussian of `BUILDING_SIGMA_M`.
BUILDING_WEIGHT_PER_M2 = 0.02
BUILDING_SIGMA_M = 12.0
#: §5.3: "trees and terrain above flood level within ~200 m of inundated homes".
HIGH_GROUND_HALO_M = 200.0
HIGH_GROUND_WEIGHT = 0.35
#: A last-known position from a phone or a report. Sigma is the reported accuracy, defaulted to a cell-tower-ish
#: 150 m; the weight is deliberately large because an LKP is the single strongest piece of evidence in land SAR.
LKP_SIGMA_M = 150.0
LKP_WEIGHT = 40.0
#: Channel bends, debris dams and recirculation zones: local maxima along the centreline (§2.4).
BEND_WEIGHT = 6.0
BEND_SIGMA_M = 60.0
#: Floor so no cell is ever exactly zero: the prior must stay a proper probability under repeated updates.
PRIOR_FLOOR = 1e-9


@dataclass(slots=True)
class LastKnownPosition:
    lat: float
    lon: float
    sigma_m: float = LKP_SIGMA_M
    weight: float = LKP_WEIGHT
    label: str = ""
    t_utc: float = 0.0


@dataclass(slots=True)
class PriorLayers:
    """Every contribution kept separately, so the UI can say *why* a cell is hot (§5.3a explainability)."""

    zone: np.ndarray
    buildings: np.ndarray
    high_ground: np.ndarray
    channel: np.ndarray
    lkp: np.ndarray
    total: np.ndarray
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"zone": self.zone, "buildings": self.buildings, "high_ground": self.high_ground,
                "channel": self.channel, "lkp": self.lkp, "total": self.total}


def _gaussian_bump(shape: tuple[int, int], cell_m: float, i: float, j: float, sigma_m: float,
                   weight: float, out: np.ndarray) -> None:
    """Add a Gaussian of total mass `weight` centred on fractional cell (i, j). Truncated at 3 sigma."""
    s = max(sigma_m / cell_m, 0.5)
    rad = int(math.ceil(3.0 * s))
    i0, i1 = max(0, int(i) - rad), min(shape[0], int(i) + rad + 1)
    j0, j1 = max(0, int(j) - rad), min(shape[1], int(j) + rad + 1)
    if i1 <= i0 or j1 <= j0:
        return
    di = (np.arange(i0, i1) - i)[:, None]
    dj = (np.arange(j0, j1) - j)[None, :]
    g = np.exp(-0.5 * (di * di + dj * dj) / (s * s))
    tot = float(g.sum())
    if tot > 0:
        out[i0:i1, j0:j1] += (weight * g / tot).astype(out.dtype)


def build_prior(cmap: CoverageMap, settlement_json: str | Path | None = None,
                lkps: list[LastKnownPosition] | None = None, channel_centreline: list[dict] | None = None,
                scene: SceneFrame | None = None) -> PriorLayers:
    """Assemble the §5.3 prior over `cmap`'s grid. Every layer is returned as well as the normalised total."""
    shape = cmap.shape
    grid = cmap.any_grid
    cell = grid.cell_m
    scene = scene or cmap.scene
    notes: list[str] = []

    zone = np.zeros(shape, dtype=np.float32)
    for name, code in ZONE_CODE.items():
        zone[cmap.zone_codes == code] = ZONE_BASE_WEIGHT.get(name, 0.1)

    buildings = np.zeros(shape, dtype=np.float32)
    high_ground = np.zeros(shape, dtype=np.float32)
    if settlement_json is not None and scene is not None:
        d = json.loads(Path(settlement_json).read_text(encoding="utf-8"))
        arch = d.get("archetypes", {})
        flooded = 0
        for hse in d.get("houses", []):
            a = arch.get(hse.get("archetype"), {})
            area = float(a.get("length_m", 9.0)) * float(a.get("width_m", 7.0))
            storeys = int(a.get("storeys", 1))
            depth = float(hse.get("flood_depth_m", 0.0))
            if depth <= 0.0:
                continue  # §5.3 counts buildings INSIDE the flood footprint
            flooded += 1
            lat, lon = scene.to_latlon(float(hse["north_m"]), float(hse["east_m"]))
            n, e = latlon_to_grid_ne(grid, lat, lon)
            w = BUILDING_WEIGHT_PER_M2 * area * storeys * (1.0 + min(depth, 4.0) / 4.0)
            _gaussian_bump(shape, cell, n / cell - 0.5, e / cell - 0.5, BUILDING_SIGMA_M, w, buildings)
            _gaussian_bump(shape, cell, n / cell - 0.5, e / cell - 0.5, HIGH_GROUND_HALO_M,
                           HIGH_GROUND_WEIGHT * storeys, high_ground)
        notes.append(f"{flooded} flooded buildings from {Path(settlement_json).name}")
        high_ground *= (cmap.zone_codes == ZONE_CODE["hillslope"]) | (cmap.zone_codes == ZONE_CODE["settlement"])

    channel = np.zeros(shape, dtype=np.float32)
    if channel_centreline and scene is not None:
        pts = np.array([[float(p["y_m"]), float(p["x_m"])] for p in channel_centreline], dtype=float)
        if len(pts) >= 3:
            # local curvature: bends, and by proxy the debris dams and recirculation zones that form at them
            d1 = np.gradient(pts, axis=0)
            d2 = np.gradient(d1, axis=0)
            speed = np.hypot(d1[:, 0], d1[:, 1]) + 1e-9
            curv = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3
            if curv.max() > 0:
                curv = curv / curv.max()
            downstream = np.linspace(0.4, 1.0, len(pts))  # §2.4: recoveries concentrate downstream
            for (sn, se), c, ds in zip(pts, curv, downstream):
                lat, lon = scene.to_latlon(sn, se)
                n, e = latlon_to_grid_ne(grid, lat, lon)
                _gaussian_bump(shape, cell, n / cell - 0.5, e / cell - 0.5, BEND_SIGMA_M,
                               BEND_WEIGHT * ds * (0.3 + 0.7 * float(c)), channel)
            notes.append(f"{len(pts)} channel centreline nodes, curvature-weighted")

    lkp = np.zeros(shape, dtype=np.float32)
    for p in lkps or []:
        n, e = latlon_to_grid_ne(grid, p.lat, p.lon)
        _gaussian_bump(shape, cell, n / cell - 0.5, e / cell - 0.5, p.sigma_m, p.weight, lkp)
    if lkps:
        notes.append(f"{len(lkps)} last-known-position pins")

    total = zone + buildings + high_ground + channel + lkp
    total = np.maximum(total, PRIOR_FLOOR)
    total[cmap.cannot_clear] = PRIOR_FLOOR  # §2.7: effort here cannot pay off from the air, so it never attracts it
    total = (total / total.sum()).astype(np.float32)
    return PriorLayers(zone, buildings, high_ground, channel, lkp, total, notes)


def bayesian_update(poa: np.ndarray, pod: np.ndarray) -> np.ndarray:
    """Appendix B: after an unsuccessful pass, POA' proportional to POA (1 - POD), renormalised.

    Because POD is clamped below 1 (§5.3), (1 - POD) is strictly positive, so no cell's probability ever reaches
    zero: "no segment ever reaches zero; that is the formal basis for 'the system recommends, never closes'".
    """
    p = np.asarray(poa, dtype=np.float64) * (1.0 - np.clip(np.asarray(pod, dtype=np.float64), 0.0, 1.0 - 1e-9))
    p = np.maximum(p, PRIOR_FLOOR)
    return (p / p.sum()).astype(np.float32)


def probability_of_success(poa: np.ndarray, pod: np.ndarray) -> np.ndarray:
    """Appendix B: POS = POA x POD. The surface the planner's greedy scorer maximises the increment of."""
    return (np.asarray(poa, dtype=np.float32) * np.asarray(pod, dtype=np.float32)).astype(np.float32)
