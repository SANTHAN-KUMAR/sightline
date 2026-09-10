"""Grid geometry, frames and polygon rasterisation for the search-quality raster (SOLUTION_DOC §5.3).

`CoverageGrid` (frozen in `schemas.py`) stores the arrays; this module supplies everything around them: where a
cell is on the Earth, how a polygon lands on the grid, and how the scenario's local frame relates to both.

Frames used here, named so they cannot be confused:
  * **scene NE** — metres north/east of the map centre in `data/scene/flood_valley.json`
    (11.4870 N, 76.1450 E). This is the simulator's frame: UE X = north, UE Y = east, UE Z = 0 at `base_z_m` ASL.
  * **grid NE** — metres north/east of the grid's SW corner, which is what `CoverageGrid.origin_lat/lon` names.
  * **WGS-84 lat/lon** — the only frame that crosses a module boundary, always ordered lat, lon (except in GeoJSON).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sightline.common.geodesy import ne_between, offset_ne
from sightline.schemas import CoverageGrid

DEFAULT_CELL_M = 10.0  # §5.3 "raster resolution (5-10 m cells)"


# --- 1. the scenario frame --------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SceneFrame:
    """The scenario's georeferencing, read from `data/scene/flood_valley.json`.

    `base_z_m` is the ASL height of UE world Z = 0, so `ue_cm()` reproduces the `ue_import.ue_from_local` rule
    that every spawner in `tools/scene/` already uses: X_cm = north*100, Y_cm = east*100, Z_cm = (asl-base_z)*100.
    """

    centre_lat: float
    centre_lon: float
    base_z_m: float
    size_m: float = 2048.0
    water_level_m: float = 0.0
    name: str = "flood_valley"

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneFrame":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        gp = d["map_centre_geopoint"]
        return cls(float(gp["lat"]), float(gp["lon"]), float(d["base_z_m"]), float(d.get("size_m", 2048.0)),
                   float(d.get("water_level_m", 0.0)), Path(path).stem)

    def to_latlon(self, north_m: float, east_m: float) -> tuple[float, float]:
        return offset_ne(self.centre_lat, self.centre_lon, north_m, east_m)

    def to_scene_ne(self, lat: float, lon: float) -> tuple[float, float]:
        return ne_between(self.centre_lat, self.centre_lon, lat, lon)

    def ue_cm(self, north_m: float, east_m: float, asl_m: float) -> tuple[float, float, float]:
        """The simulator's own coordinates, for the mission lane (`ue_import` in flood_valley.json)."""
        return (north_m * 100.0, east_m * 100.0, (asl_m - self.base_z_m) * 100.0)

    def sw_corner_latlon(self) -> tuple[float, float]:
        return self.to_latlon(-self.size_m / 2.0, -self.size_m / 2.0)


# --- 2. grid construction and cell geometry ---------------------------------------------------------------
def make_grid(sw_lat: float, sw_lon: float, cell_m: float, n_north: int, n_east: int, presentation: str = "body",
              k: float = 1.0) -> CoverageGrid:
    return CoverageGrid.empty(sw_lat, sw_lon, cell_m, n_north, n_east, presentation=presentation, k=k)


def make_grid_for_scene(frame: SceneFrame, cell_m: float = DEFAULT_CELL_M, presentation: str = "body",
                        k: float = 1.0, size_m: float | None = None) -> CoverageGrid:
    """A grid covering the whole scenario square, origin at its SW corner."""
    side = frame.size_m if size_m is None else size_m
    n = int(round(side / cell_m))
    lat, lon = frame.to_latlon(-side / 2.0, -side / 2.0)
    return make_grid(lat, lon, cell_m, n, n, presentation=presentation, k=k)


def cell_edges_m(grid: CoverageGrid) -> tuple[np.ndarray, np.ndarray]:
    """Grid-NE cell boundaries: (north edges of length n_north+1, east edges of length n_east+1)."""
    return (np.arange(grid.n_north + 1) * grid.cell_m, np.arange(grid.n_east + 1) * grid.cell_m)


def cell_centres_m(grid: CoverageGrid) -> tuple[np.ndarray, np.ndarray]:
    """Grid-NE centres as 1-D arrays (north for each row i, east for each column j)."""
    return ((np.arange(grid.n_north) + 0.5) * grid.cell_m, (np.arange(grid.n_east) + 0.5) * grid.cell_m)


def cell_area_m2(grid: CoverageGrid) -> float:
    return float(grid.cell_m * grid.cell_m)


def grid_ne_to_latlon(grid: CoverageGrid, north_m: float, east_m: float) -> tuple[float, float]:
    return offset_ne(grid.origin_lat, grid.origin_lon, north_m, east_m)


def latlon_to_grid_ne(grid: CoverageGrid, lat: float, lon: float) -> tuple[float, float]:
    return ne_between(grid.origin_lat, grid.origin_lon, lat, lon)


def cell_of_latlon(grid: CoverageGrid, lat: float, lon: float) -> tuple[int, int] | None:
    n, e = latlon_to_grid_ne(grid, lat, lon)
    i, j = int(math.floor(n / grid.cell_m)), int(math.floor(e / grid.cell_m))
    if 0 <= i < grid.n_north and 0 <= j < grid.n_east:
        return i, j
    return None


def bounds_latlon(grid: CoverageGrid) -> dict[str, float]:
    """South/west/north/east bounds. The map lane needs these to place an image overlay."""
    ne_lat, ne_lon = grid_ne_to_latlon(grid, grid.n_north * grid.cell_m, grid.n_east * grid.cell_m)
    return {"south": grid.origin_lat, "west": grid.origin_lon, "north": ne_lat, "east": ne_lon}


# --- 3. polygon rasterisation -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Window:
    """A half-open cell window [i0, i1) x [j0, j1) into the grid arrays."""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def empty(self) -> bool:
        return self.i1 <= self.i0 or self.j1 <= self.j0

    @property
    def shape(self) -> tuple[int, int]:
        return (max(0, self.i1 - self.i0), max(0, self.j1 - self.j0))

    def slice(self) -> tuple[slice, slice]:
        return (slice(self.i0, self.i1), slice(self.j0, self.j1))


def points_in_polygon(poly_ne: np.ndarray, pts_ne: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray crossing. Works for any simple polygon (convex or not), no shapely needed.

    `poly_ne` is (M, 2) north/east vertices (implicitly closed), `pts_ne` is (N, 2). Returns (N,) bool.
    """
    poly = np.asarray(poly_ne, dtype=float)
    pts = np.asarray(pts_ne, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, np.nan, y2 - y1) + x1)
    return (np.nan_to_num(crosses.astype(np.int8)).sum(axis=1) % 2).astype(bool)


def polygon_window(grid: CoverageGrid, poly_ne: np.ndarray, pad_cells: int = 1) -> Window:
    poly = np.asarray(poly_ne, dtype=float)
    i0 = int(math.floor(poly[:, 0].min() / grid.cell_m)) - pad_cells
    i1 = int(math.ceil(poly[:, 0].max() / grid.cell_m)) + pad_cells
    j0 = int(math.floor(poly[:, 1].min() / grid.cell_m)) - pad_cells
    j1 = int(math.ceil(poly[:, 1].max() / grid.cell_m)) + pad_cells
    return Window(max(0, i0), min(grid.n_north, i1), max(0, j0), min(grid.n_east, j1))


def polygon_cell_weights(grid: CoverageGrid, poly_ne: np.ndarray, supersample: int = 3
                         ) -> tuple[Window, np.ndarray]:
    """Fraction of each cell's area inside the polygon, over the polygon's bounding window.

    `supersample = s` estimates the fraction from an s x s sample lattice inside every cell, so an edge cell gets a
    weight in {0, 1/s^2, ..., 1} and a fully interior cell gets exactly 1.0. Effort is a swept *area*, so the
    fractional weight matters: a cell half inside the footprint was half searched.
    """
    win = polygon_window(grid, poly_ne)
    if win.empty:
        return win, np.zeros((0, 0), dtype=np.float32)
    s = max(1, int(supersample))
    off = (np.arange(s) + 0.5) / s * grid.cell_m
    ni = (np.arange(win.i0, win.i1) * grid.cell_m)[:, None] + off[None, :]
    ej = (np.arange(win.j0, win.j1) * grid.cell_m)[:, None] + off[None, :]
    nn = np.repeat(ni.ravel(), ej.size)
    ee = np.tile(ej.ravel(), ni.size)
    inside = points_in_polygon(poly_ne, np.column_stack([nn, ee]))
    hits = inside.reshape(win.shape[0], s, win.shape[1], s).sum(axis=(1, 3))
    return win, (hits / float(s * s)).astype(np.float32)


def rasterise_polygons(grid: CoverageGrid, polys_ne: list[np.ndarray], supersample: int = 2) -> np.ndarray:
    """A full-grid bool mask that is True where any polygon covers more than half a cell.

    Used for the burial polygons (§2.7) and for segment membership, where a hard in/out answer is what is wanted.
    """
    mask = np.zeros((grid.n_north, grid.n_east), dtype=bool)
    for poly in polys_ne:
        win, w = polygon_cell_weights(grid, poly, supersample=supersample)
        if not win.empty:
            mask[win.slice()] |= w > 0.5
    return mask


def polygons_from_latlon(grid: CoverageGrid, polys_latlon: list[list[tuple[float, float]]]) -> list[np.ndarray]:
    """Convert lat/lon rings (lat, lon order) to grid-NE arrays."""
    out = []
    for ring in polys_latlon:
        out.append(np.array([latlon_to_grid_ne(grid, lat, lon) for lat, lon in ring], dtype=float))
    return out


def scene_poly_to_grid(grid: CoverageGrid, scene: SceneFrame, poly_scene_ne: np.ndarray) -> np.ndarray:
    """Scene-NE (metres from the map centre) -> grid-NE (metres from the grid's SW corner), via lat/lon."""
    p = np.asarray(poly_scene_ne, dtype=float).reshape(-1, 2)
    return np.array([latlon_to_grid_ne(grid, *scene.to_latlon(float(n), float(e))) for n, e in p], dtype=float)


def grid_poly_to_scene(grid: CoverageGrid, scene: SceneFrame, poly_grid_ne: np.ndarray) -> np.ndarray:
    """The inverse of `scene_poly_to_grid`."""
    p = np.asarray(poly_grid_ne, dtype=float).reshape(-1, 2)
    return np.array([scene.to_scene_ne(*grid_ne_to_latlon(grid, float(n), float(e))) for n, e in p], dtype=float)


def grid_to_dict(grid: CoverageGrid) -> dict[str, Any]:
    """Metadata only (no arrays) — the header every export product repeats."""
    return {
        "origin_lat": grid.origin_lat,
        "origin_lon": grid.origin_lon,
        "cell_m": grid.cell_m,
        "n_north": grid.n_north,
        "n_east": grid.n_east,
        "presentation": grid.presentation,
        "k": grid.k,
        "bounds": bounds_latlon(grid),
    }
