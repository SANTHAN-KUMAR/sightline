"""The search-quality map itself: effort accumulation, POD, and the per-presentation stack (§5.3, §5.3b, §2.7).

    q_pass(cell, j) = R_slice(j ; GSD, band, time, blur, view) x V(cell, j)
    C_j(cell)       = SUM over passes of q_pass(cell, j)
    POD_j(cell)     = 1 - exp(-k_j * C_j(cell))        clamped below 1
    POD_eff(cell)   = SUM over j of w_j(zone) * POD_j(cell)

**Accumulation is per PASS, not per frame.** The doc's model is `C = sum over passes of q_pass`, and that is what
this implements: within a pass, a cell takes the best look it got (`max` over the frames of that pass of
`cell area fraction x q`), and only at `end_pass()` is that added to `C`. The consequence is the important one —
**the answer does not depend on the frame rate**. Accumulating per frame would let a 30 fps flight claim ten times
the coverage of a 3 fps flight over identical ground, which is the classic way a coverage map becomes a lie.

**Burial polygons (§2.7, guardrail R10).** Cells marked `cannot_clear` are never written: no effort is added to
them, so `C` stays 0 and POD stays at its floor no matter how many passes are flown. This is enforced at the point
of accumulation rather than at display time, so even a consumer that calls `CoverageGrid.recompute_pod()` itself
sees zero. Separately, and by an independent mechanism, the `buried` presentation layer has V(cell, "buried") = 0
everywhere, so it is identically zero across the whole map (§5.3b). The two agree; neither depends on the other.

There is **no "cleared" state** in this module. POD is a probability, it is clamped below 1, and nothing here ever
sets a flag that means "done".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from sightline.coverage import calibrate as cal
from sightline.coverage.footprint import Footprint, ground_footprint
from sightline.coverage.grid import (
    SceneFrame,
    cell_centres_m,
    latlon_to_grid_ne,
    make_grid,
    polygon_cell_weights,
    rasterise_polygons,
)
from sightline.coverage.presentation import MVP_PRESENTATIONS, ZERO_LAYER_PRESENTATIONS, mix_for_zone
from sightline.coverage.quality import Conditions, RecallEstimate, SliceTable, slice_recall_array
from sightline.schemas import CoverageGrid, Intrinsics, Telemetry, Zone

ZONE_NAMES: tuple[str, ...] = ("unknown", "fan", "settlement", "channel", "hillslope")
ZONE_CODE: dict[str, int] = {z: i for i, z in enumerate(ZONE_NAMES)}


# --- conditions from telemetry ----------------------------------------------------------------------------
def local_hour(tel: Telemetry) -> float | None:
    if not tel.time_of_day:
        return None
    try:
        dt = datetime.fromisoformat(tel.time_of_day.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def time_of_day_label(hour: float | None) -> str:
    """Coarse label matching the §5.12 `time_of_day` slice."""
    if hour is None:
        return "day"
    if 5.0 <= hour < 8.0:
        return "dawn"
    if 8.0 <= hour < 17.0:
        return "day"
    if 17.0 <= hour < 19.5:
        return "dusk"
    return "night"


def conditions_from_telemetry(tel: Telemetry, band: str = "rgb", exposure_s: float = 1.0 / 500.0,
                              zone: str = "unknown") -> Conditions:
    """Build the `Conditions` for a frame from the pose sample the ingest lane produced."""
    h = local_hour(tel)
    rain = float(tel.weather.get("rain", 0.0))
    fog = float(tel.weather.get("fog", 0.0))
    weather = "dry"
    if rain >= 0.05:
        weather = "heavy_rain" if rain >= 0.5 else "light_rain"
    if fog >= 0.10:
        weather = "fog"
    speed = float(math.hypot(tel.vel_ned_ms[0], tel.vel_ned_ms[1]))
    return Conditions(band=band, time_of_day=time_of_day_label(h), local_hour=h, weather=weather,
                      speed_ms=speed, exposure_s=exposure_s, zone=zone)


# --- per-frame result -------------------------------------------------------------------------------------
@dataclass(slots=True)
class FrameCoverage:
    """What one frame contributed, kept so the UI can explain a pass and the tests can assert on it."""

    frame_idx: int
    pass_id: int
    footprint: Footprint
    cells_touched: int
    best_gsd_m: float
    worst_gsd_m: float
    q_max: dict[str, float] = field(default_factory=dict)
    recall_basis: dict[str, RecallEstimate] = field(default_factory=dict)
    skipped_reason: str = ""


@dataclass(slots=True)
class PassSummary:
    pass_id: int
    frames: int
    cells: int
    mean_q: dict[str, float] = field(default_factory=dict)
    mode: str = "AUTO"


# --- the map ----------------------------------------------------------------------------------------------
@dataclass(slots=True)
class CoverageMap:
    """A stack of `CoverageGrid`s, one per presentation, sharing one geometry and one `cannot_clear` mask."""

    grids: dict[str, CoverageGrid]
    cannot_clear: np.ndarray
    visibility: dict[str, np.ndarray]
    zone_codes: np.ndarray
    slice_table: SliceTable = field(default_factory=SliceTable)
    scene: SceneFrame | None = None
    supersample: int = 3
    _pass_buf: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _pass_id: int | None = field(default=None, repr=False)
    _pass_frames: int = field(default=0, repr=False)
    passes: list[PassSummary] = field(default_factory=list)

    # -- construction --------------------------------------------------------------------------------------
    @classmethod
    def create(cls, sw_lat: float, sw_lon: float, cell_m: float, n_north: int, n_east: int,
               presentations: Iterable[str] = MVP_PRESENTATIONS, slice_table: SliceTable | None = None,
               scene: SceneFrame | None = None) -> "CoverageMap":
        grids: dict[str, CoverageGrid] = {}
        vis: dict[str, np.ndarray] = {}
        for p in presentations:
            grids[p] = make_grid(sw_lat, sw_lon, cell_m, n_north, n_east, presentation=p, k=cal.k_for(p))
            v = np.ones((n_north, n_east), dtype=np.float32)
            if p in ZERO_LAYER_PRESENTATIONS:
                v[:] = 0.0  # §5.3b: the fully-buried layer is identically zero, by construction not by exception
            vis[p] = v
        return cls(grids=grids, cannot_clear=np.zeros((n_north, n_east), dtype=bool), visibility=vis,
                   zone_codes=np.zeros((n_north, n_east), dtype=np.uint8),
                   slice_table=slice_table or SliceTable(), scene=scene)

    @classmethod
    def for_scene(cls, scene: SceneFrame, cell_m: float = 10.0, presentations: Iterable[str] = MVP_PRESENTATIONS,
                  slice_table: SliceTable | None = None, size_m: float | None = None) -> "CoverageMap":
        side = scene.size_m if size_m is None else size_m
        n = int(round(side / cell_m))
        lat, lon = scene.to_latlon(-side / 2.0, -side / 2.0)
        return cls.create(lat, lon, cell_m, n, n, presentations, slice_table, scene)

    # -- geometry ------------------------------------------------------------------------------------------
    @property
    def any_grid(self) -> CoverageGrid:
        return next(iter(self.grids.values()))

    @property
    def shape(self) -> tuple[int, int]:
        g = self.any_grid
        return (g.n_north, g.n_east)

    @property
    def presentations(self) -> tuple[str, ...]:
        return tuple(self.grids)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_north, n_east) meshgrids of grid-NE centre coordinates."""
        n1, e1 = cell_centres_m(self.any_grid)
        return np.meshgrid(n1, e1, indexing="ij")

    # -- masks ---------------------------------------------------------------------------------------------
    def set_cannot_clear(self, mask: np.ndarray) -> None:
        """§2.7 rule 1: mark, never clear. Effort is never accumulated into these cells."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != self.shape:
            raise ValueError(f"cannot_clear mask {m.shape} does not match grid {self.shape}")
        self.cannot_clear |= m
        for p, g in self.grids.items():
            g.cannot_clear = self.cannot_clear
            g.coverage[self.cannot_clear] = 0.0  # nothing accumulated before the polygon was drawn survives it
            self.visibility[p] = np.where(self.cannot_clear, 0.0, self.visibility[p]).astype(np.float32)
        self._recompute()

    def add_burial_polygons(self, polys_ne: list[np.ndarray]) -> None:
        self.set_cannot_clear(rasterise_polygons(self.any_grid, polys_ne))

    def set_visibility(self, presentation: str, v: np.ndarray) -> None:
        """`V(cell, j)`: 1.0 open ground, lower under canopy or structures, 0 where nothing can be seen (§5.3)."""
        arr = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
        if arr.shape != self.shape:
            raise ValueError("visibility raster shape mismatch")
        if presentation in ZERO_LAYER_PRESENTATIONS:
            arr = np.zeros_like(arr)
        self.visibility[presentation] = np.where(self.cannot_clear, 0.0, arr).astype(np.float32)

    def set_zones(self, zone_codes: np.ndarray) -> None:
        z = np.asarray(zone_codes, dtype=np.uint8)
        if z.shape != self.shape:
            raise ValueError("zone raster shape mismatch")
        self.zone_codes = z

    def zone_at(self, i: int, j: int) -> str:
        return ZONE_NAMES[int(self.zone_codes[i, j])]

    # -- accumulation --------------------------------------------------------------------------------------
    def begin_pass(self, pass_id: int) -> None:
        if self._pass_id is not None and self._pass_id != pass_id:
            self.end_pass()
        if self._pass_id is None:
            self._pass_id = pass_id
            self._pass_frames = 0
            self._pass_buf = {p: np.zeros(self.shape, dtype=np.float32) for p in self.grids}

    def add_frame(self, tel: Telemetry, intr: Intrinsics, cond: Conditions | None = None, pass_id: int = 0,
                  ground_asl_m: float | None = None) -> FrameCoverage:
        """Project one frame's footprint and record the best look it gave each cell, for the current pass."""
        self.begin_pass(pass_id)
        self._pass_frames += 1
        grid = self.any_grid
        cond = cond or conditions_from_telemetry(tel)
        fp = ground_footprint(tel, intr, ground_asl_m)
        blank = FrameCoverage(tel.frame_idx, pass_id, fp, 0, 0.0, 0.0)
        if not fp.valid:
            blank.skipped_reason = fp.reject_reason
            return blank
        cam_n, cam_e = latlon_to_grid_ne(grid, tel.lat, tel.lon)
        poly = fp.poly_ne_m + np.array([cam_n, cam_e])
        win, w = polygon_cell_weights(grid, poly, supersample=self.supersample)
        if win.empty or not np.any(w > 0.0):
            blank.skipped_reason = "footprint outside the grid"
            return blank
        n1, e1 = cell_centres_m(grid)
        dn = n1[win.i0:win.i1][:, None] - cam_n
        de = e1[win.j0:win.j1][None, :] - cam_e
        gsd = fp.gsd_at(np.broadcast_to(dn, w.shape), np.broadcast_to(de, w.shape))
        off_nadir = np.degrees(np.arctan2(np.hypot(dn, de), max(fp.agl_m, 1e-6)))
        out = FrameCoverage(tel.frame_idx, pass_id, fp, int((w > 0).sum()),
                            float(gsd[w > 0].min()), float(gsd[w > 0].max()))
        sl = win.slice()
        live = ~self.cannot_clear[sl]  # R10: buried cells never receive effort
        for p in self.grids:
            r, basis = slice_recall_array(self.slice_table, p, fp.agl_m, cond, gsd,
                                          np.broadcast_to(off_nadir, w.shape))
            q = (w * r * self.visibility[p][sl] * live).astype(np.float32)
            np.maximum(self._pass_buf[p][sl], q, out=self._pass_buf[p][sl])
            out.q_max[p] = float(q.max()) if q.size else 0.0
            out.recall_basis[p] = basis
        return out

    def end_pass(self, mode: str = "AUTO") -> PassSummary | None:
        """Commit the current pass: `C += best look this pass`, then recompute POD. Idempotent when no pass is open."""
        if self._pass_id is None:
            return None
        cells = 0
        mean_q: dict[str, float] = {}
        for p, g in self.grids.items():
            buf = self._pass_buf[p]
            buf[self.cannot_clear] = 0.0
            g.coverage = (g.coverage + buf).astype(np.float32)
            touched = buf > 0.0
            cells = max(cells, int(touched.sum()))
            mean_q[p] = float(buf[touched].mean()) if touched.any() else 0.0
        summary = PassSummary(self._pass_id, self._pass_frames, cells, mean_q, mode)
        self.passes.append(summary)
        self._pass_id, self._pass_frames, self._pass_buf = None, 0, {}
        self._recompute()
        return summary

    def add_pass(self, frames: Iterable[tuple[Telemetry, Intrinsics]], cond: Conditions | None = None,
                 pass_id: int | None = None, mode: str = "AUTO") -> PassSummary | None:
        """Convenience: accumulate a whole pass from an iterable of (telemetry, intrinsics)."""
        pid = len(self.passes) if pass_id is None else pass_id
        for tel, intr in frames:
            self.add_frame(tel, intr, cond, pid)
        return self.end_pass(mode)

    # -- products ------------------------------------------------------------------------------------------
    def _recompute(self) -> None:
        for g in self.grids.values():
            g.recompute_pod()
            np.clip(g.pod, 0.0, cal.POD_MAX, out=g.pod)
            g.pod[self.cannot_clear] = 0.0

    def coverage(self, presentation: str) -> np.ndarray:
        return self.grids[presentation].coverage

    def pod(self, presentation: str) -> np.ndarray:
        """POD for one presentation layer: clamped below 1, zero inside every `cannot_clear` cell."""
        return self.grids[presentation].pod

    def mixture_weights(self) -> dict[str, np.ndarray]:
        """`w_j(zone)` as a full-grid array per presentation, renormalised over the layers this map actually has."""
        out = {p: np.zeros(self.shape, dtype=np.float32) for p in self.grids}
        for name, code in ZONE_CODE.items():
            sel = self.zone_codes == code
            if not sel.any():
                continue
            mix = mix_for_zone(name, self.presentations)
            for p in self.grids:
                out[p][sel] = mix[p]
        return out

    def pod_effective(self) -> np.ndarray:
        """§5.3b's default view: POD_eff(cell) = sum_j w_j(zone) POD_j(cell). Never shown without a layer selector."""
        w = self.mixture_weights()
        acc = np.zeros(self.shape, dtype=np.float32)
        for p in self.grids:
            acc += w[p] * self.grids[p].pod
        acc[self.cannot_clear] = 0.0
        return np.clip(acc, 0.0, cal.POD_MAX)

    def delta_pod(self, presentation: str, q: np.ndarray | float) -> np.ndarray:
        """§5.3a: dPOD = e^(-k C_before) - e^(-k C_after), the diminishing-returns increment the planner scores.

        Zero inside `cannot_clear` by construction, so the planner "will never waste effort pretending to clear
        them" (§5.3a) without a special case anywhere in the planner.
        """
        g = self.grids[presentation]
        add = np.asarray(q, dtype=np.float32) * self.visibility[presentation]
        before = np.exp(-g.k * g.coverage)
        after = np.exp(-g.k * (g.coverage + add))
        d = np.clip(before - after, 0.0, 1.0).astype(np.float32)
        d[self.cannot_clear] = 0.0
        return d

    def delta_pod_mixture(self, q: dict[str, np.ndarray | float]) -> np.ndarray:
        """§5.3b: the mixture-weighted increment sum_j w_j dPOD_j. "Nothing else changes" in the planner's score."""
        w = self.mixture_weights()
        acc = np.zeros(self.shape, dtype=np.float32)
        for p in self.grids:
            acc += w[p] * self.delta_pod(p, q.get(p, 0.0))
        return acc

    def searched_fraction(self, presentation: str, threshold: float = 0.5) -> float:
        """Fraction of clearable cells whose POD is above a threshold. NOT a "cleared" figure — a POD statistic."""
        live = ~self.cannot_clear
        n = int(live.sum())
        return float((self.pod(presentation)[live] >= threshold).sum() / n) if n else 0.0

    def statement(self, presentation_a: str = "body", presentation_b: str = "limb_only",
                  mask: np.ndarray | None = None) -> str:
        """The §5.3b sentence the map exists to be able to say, generated from the numbers."""
        sel = np.ones(self.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        sel = sel & ~self.cannot_clear
        if not sel.any():
            return "no clearable cells in this selection"
        a = float(self.pod(presentation_a)[sel].mean()) if presentation_a in self.grids else float("nan")
        b = float(self.pod(presentation_b)[sel].mean()) if presentation_b in self.grids else float("nan")
        s = f"{a:.2f} searched for {presentation_a} and {b:.2f} for {presentation_b} (simulation)"
        if b < 0.3 <= a:
            s += "; a low pass is recommended before it is treated as searched for partly buried casualties"
        return s

    def summary(self) -> dict[str, Any]:
        live = ~self.cannot_clear
        n_live = int(live.sum())
        return {
            "shape": self.shape,
            "cell_m": self.any_grid.cell_m,
            "passes": len(self.passes),
            "cannot_clear_cells": int(self.cannot_clear.sum()),
            "clearable_cells": n_live,
            "layers": {
                p: {
                    "k": g.k,
                    "k_is_measured": cal.DEFAULT_K.get(p, cal.KValue(g.k, False)).measured,
                    "mean_coverage": float(g.coverage[live].mean()) if n_live else 0.0,
                    "mean_pod": float(g.pod[live].mean()) if n_live else 0.0,
                    "max_pod": float(g.pod.max()) if g.pod.size else 0.0,
                    "pod_ge_0.5_fraction": self.searched_fraction(p, 0.5),
                }
                for p, g in self.grids.items()
            },
            "domain": "sim",
        }


def zone_raster_from_scene(map_: CoverageMap, zones_png: str, height_npy: str | None = None,
                           water_level_m: float | None = None) -> np.ndarray:
    """Resample `flood_valley_zones.png` (R = fan, G = settlement terrace, B = channel + banks) onto the grid.

    The PNG is written flipped by `tools/scene/gen_terrain.py`, so it is flipped back here; the scene raster is
    indexed [north, east] with [0, 0] at the SW corner, which matches the `CoverageGrid` layout.
    """
    from PIL import Image

    arr = np.flipud(np.asarray(Image.open(zones_png).convert("RGB")))
    n_src = arr.shape[0]
    scene = map_.scene or SceneFrame(map_.any_grid.origin_lat, map_.any_grid.origin_lon, 0.0)
    g = map_.any_grid
    n1, e1 = cell_centres_m(g)
    # grid-NE (from the SW corner) -> scene-NE (from the map centre) -> source pixel index
    o_n, o_e = scene.to_scene_ne(g.origin_lat, g.origin_lon)
    half, span = scene.size_m / 2.0, max(scene.size_m, 1e-6)
    src_n = np.clip((((n1 + o_n) + half) / span * (n_src - 1)).round(), 0, n_src - 1).astype(int)
    src_e = np.clip((((e1 + o_e) + half) / span * (n_src - 1)).round(), 0, n_src - 1).astype(int)
    sub = arr[np.ix_(src_n, src_e)]
    codes = np.full(map_.shape, ZONE_CODE["hillslope"], dtype=np.uint8)
    codes[sub[:, :, 1] > 127] = ZONE_CODE["settlement"]
    codes[sub[:, :, 0] > 127] = ZONE_CODE["fan"]
    codes[sub[:, :, 2] > 127] = ZONE_CODE["channel"]
    if height_npy is not None and water_level_m is not None:
        h = np.load(height_npy)
        hh = h[np.ix_(src_n, src_e)]
        codes[(hh < water_level_m) & (codes == ZONE_CODE["hillslope"])] = ZONE_CODE["channel"]
    map_.set_zones(codes)
    return codes


def zone_name_of(code: int) -> Zone:
    return ZONE_NAMES[int(code)]  # type: ignore[return-value]
