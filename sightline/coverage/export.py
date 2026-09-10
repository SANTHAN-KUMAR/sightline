"""Export contract for the map lane (B5): a PNG overlay plus bounds, and a banded GeoJSON vector layer.

**THE CONTRACT** — see `docs/lanes/coverage_plan.md` for the same thing in prose.

`export_coverage(cmap, out_dir, stem)` writes, for each presentation layer plus the mixture view `effective`:

    <stem>_<layer>.png            RGBA image, row 0 = NORTH, one pixel per cell, magma ramp over POD.
                                  Alpha 0 wherever the cell is `cannot_clear` (so a hatch shows through).
    <stem>_<layer>.geojson        FeatureCollection: POD band polygons (properties.pod_min/pod_max/band) plus one
                                  `cannot_clear` Feature per burial region (properties.cannot_clear = true).
    <stem>.json                   ONE manifest describing every layer: bounds, MapLibre image `coordinates`,
                                  the colour ramp stops, per-layer statistics, k and its provenance, and the
                                  `domain` of every number ("sim").

`overlay_payload(cmap)` returns the same manifest with each PNG inlined as a `data:image/png;base64,...` URI, for
pushing over the WebSocket without touching the filesystem.

Reading it in MapLibre:

    map.addSource('pod', {type: 'image', url: layer.image_url, coordinates: layer.coordinates});
    map.addLayer({id: 'pod', type: 'raster', source: 'pod', paint: {'raster-opacity': 0.75}});
    map.addSource('pod_v', {type: 'geojson', data: layer.geojson_url});   // for hatching + click-to-inspect

Guardrail R10: nothing exported here says "cleared". The legend text is generated from the numbers and always
states the domain, and `cannot_clear` regions carry the label *aerial search cannot clear* (§2.7).
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sightline.common.geodesy import meridian_radius_m, normal_radius_m
from sightline.coverage import calibrate as cal
from sightline.coverage.accumulate import CoverageMap
from sightline.coverage.grid import bounds_latlon, grid_ne_to_latlon
from sightline.schemas import SCHEMA_VERSION, CoverageGrid

#: Magma-like stops, (pod, r, g, b). Low POD is dark: "we have not looked there" must be the loudest reading.
POD_RAMP: tuple[tuple[float, int, int, int], ...] = (
    (0.00, 13, 8, 60),
    (0.20, 86, 25, 105),
    (0.40, 155, 44, 96),
    (0.60, 215, 82, 62),
    (0.80, 246, 148, 58),
    (0.95, 252, 222, 125),
)
POD_BANDS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)
OVERLAY_ALPHA = 200  # 0-255; the map lane can still scale it with raster-opacity
EFFECTIVE_LAYER = "effective"
CANNOT_CLEAR_LABEL = "aerial search cannot clear"


def ramp_rgb(pod: np.ndarray) -> np.ndarray:
    """Map POD in [0, 1] to uint8 RGB by linear interpolation between `POD_RAMP` stops."""
    p = np.clip(np.asarray(pod, dtype=float), 0.0, 1.0)
    stops = np.array([s[0] for s in POD_RAMP], dtype=float)
    cols = np.array([[s[1], s[2], s[3]] for s in POD_RAMP], dtype=float)
    out = np.empty(p.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(p, stops, cols[:, c]).round().astype(np.uint8)
    return out


def overlay_rgba(pod: np.ndarray, cannot_clear: np.ndarray | None = None, alpha: int = OVERLAY_ALPHA) -> np.ndarray:
    """(n_north, n_east) POD -> (rows, cols, 4) uint8 with row 0 = NORTH, ready to save as a PNG."""
    rgb = ramp_rgb(pod)
    a = np.full(pod.shape, alpha, dtype=np.uint8)
    if cannot_clear is not None:
        a = np.where(cannot_clear, 0, a).astype(np.uint8)  # transparent: B5 draws the hatch underneath
    return np.flipud(np.dstack([rgb, a]))


def png_bytes(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_coordinates(grid: CoverageGrid) -> list[list[float]]:
    """MapLibre image-source `coordinates`: [top-left, top-right, bottom-right, bottom-left] as [lon, lat]."""
    b = bounds_latlon(grid)
    return [[b["west"], b["north"]], [b["east"], b["north"]], [b["east"], b["south"]], [b["west"], b["south"]]]


def _degree_transform(grid: CoverageGrid):
    """Affine (col, row) -> (lon, lat) with row 0 at the north edge. Local metres-per-degree at the grid centre."""
    from rasterio.transform import from_origin

    b = bounds_latlon(grid)
    mid = 0.5 * (b["north"] + b["south"])
    dlat = math.degrees(grid.cell_m / meridian_radius_m(mid))
    dlon = math.degrees(grid.cell_m / (normal_radius_m(mid) * math.cos(math.radians(mid))))
    return from_origin(b["west"], b["north"], dlon, dlat)


def banded_geojson(grid: CoverageGrid, pod: np.ndarray, cannot_clear: np.ndarray | None = None,
                   bands: tuple[float, ...] = POD_BANDS, layer: str = "body") -> dict[str, Any]:
    """POD contour bands as polygons, plus the burial regions, as one RFC 7946 FeatureCollection.

    Coordinates are [lon, lat] per RFC 7946. Polygons are traced with `rasterio.features.shapes`; when rasterio is
    unavailable the fallback emits one rectangle per cell, which is correct but larger.
    """
    feats: list[dict[str, Any]] = []
    cc = np.zeros(pod.shape, dtype=bool) if cannot_clear is None else np.asarray(cannot_clear, dtype=bool)
    band_idx = np.digitize(np.asarray(pod, dtype=float), np.array(bands[1:-1]), right=False).astype(np.int32) + 1
    band_idx[cc] = 0
    try:
        from rasterio.features import shapes as rio_shapes

        tr = _degree_transform(grid)
        src = np.flipud(band_idx)
        for geom, val in rio_shapes(src, mask=np.flipud(~cc), transform=tr, connectivity=4):
            b = int(val)
            if b < 1:
                continue
            feats.append({"type": "Feature", "geometry": geom, "properties": {
                "layer": layer, "band": b, "pod_min": bands[b - 1], "pod_max": bands[b],
                "label": f"POD {bands[b - 1]:.2f}-{bands[b]:.2f}", "cannot_clear": False, "domain": "sim"}})
        for geom, val in rio_shapes(np.flipud(cc).astype(np.uint8), mask=np.flipud(cc), transform=tr,
                                    connectivity=4):
            if int(val) == 1:
                feats.append({"type": "Feature", "geometry": geom, "properties": {
                    "layer": layer, "cannot_clear": True, "label": CANNOT_CLEAR_LABEL, "pod_min": 0.0,
                    "pod_max": 0.0, "domain": "sim"}})
    except ImportError:  # pragma: no cover - rasterio is a pinned dependency
        for i in range(grid.n_north):
            for j in range(grid.n_east):
                sw = grid_ne_to_latlon(grid, i * grid.cell_m, j * grid.cell_m)
                ne = grid_ne_to_latlon(grid, (i + 1) * grid.cell_m, (j + 1) * grid.cell_m)
                ring = [[sw[1], sw[0]], [ne[1], sw[0]], [ne[1], ne[0]], [sw[1], ne[0]], [sw[1], sw[0]]]
                feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                              "properties": {"layer": layer, "pod": float(pod[i, j]),
                                             "cannot_clear": bool(cc[i, j]), "domain": "sim"}})
    return {"type": "FeatureCollection", "schema_version": SCHEMA_VERSION, "layer": layer, "features": feats}


@dataclass(slots=True)
class LayerExport:
    name: str
    pod: np.ndarray
    coverage: np.ndarray | None
    k: float
    k_measured: bool
    k_basis: str


def _layers(cmap: CoverageMap, include_effective: bool = True) -> list[LayerExport]:
    out = []
    for p, g in cmap.grids.items():
        kv = cal.DEFAULT_K.get(p)
        out.append(LayerExport(p, g.pod, g.coverage, g.k, bool(kv and kv.measured), kv.basis if kv else ""))
    if include_effective and len(cmap.grids) > 1:
        out.append(LayerExport(EFFECTIVE_LAYER, cmap.pod_effective(), None, float("nan"), False,
                               "zone-weighted mixture of the layers (§5.3b)"))
    return out


def layer_stats(cmap: CoverageMap, lx: LayerExport) -> dict[str, Any]:
    live = ~cmap.cannot_clear
    n = int(live.sum())
    return {
        "mean_pod": float(lx.pod[live].mean()) if n else 0.0,
        "max_pod": float(lx.pod.max()) if lx.pod.size else 0.0,
        "cells_pod_ge_0.5": int((lx.pod[live] >= 0.5).sum()) if n else 0,
        "cells_pod_lt_0.2": int((lx.pod[live] < 0.2).sum()) if n else 0,
        "clearable_cells": n,
        "cannot_clear_cells": int(cmap.cannot_clear.sum()),
        "mean_coverage": float(lx.coverage[live].mean()) if (lx.coverage is not None and n) else None,
        "domain": "sim",
    }


def manifest(cmap: CoverageMap, stem: str = "coverage", include_effective: bool = True,
             urls: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """The single JSON header B5 reads. `urls` supplies image/geojson locations (file names or data URIs)."""
    grid = cmap.any_grid
    lay = []
    for lx in _layers(cmap, include_effective):
        u = (urls or {}).get(lx.name, {})
        lay.append({
            "layer": lx.name,
            "image_url": u.get("image", f"{stem}_{lx.name}.png"),
            "geojson_url": u.get("geojson", f"{stem}_{lx.name}.geojson"),
            "k": None if math.isnan(lx.k) else lx.k,
            "k_is_measured": lx.k_measured,
            "k_basis": lx.k_basis,
            "stats": layer_stats(cmap, lx),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "product": "sightline.coverage",
        "domain": "sim",
        "grid": {
            "origin_lat": grid.origin_lat, "origin_lon": grid.origin_lon, "cell_m": grid.cell_m,
            "n_north": grid.n_north, "n_east": grid.n_east,
            "row_order": "image row 0 is NORTH; array row 0 is SOUTH",
        },
        "bounds": bounds_latlon(grid),
        "coordinates": image_coordinates(grid),
        "ramp": [{"pod": s[0], "rgb": [s[1], s[2], s[3]]} for s in POD_RAMP],
        "bands": list(POD_BANDS),
        "cannot_clear_label": CANNOT_CLEAR_LABEL,
        "passes": [{"pass_id": p.pass_id, "frames": p.frames, "cells": p.cells, "mode": p.mode} for p in cmap.passes],
        "layers": lay,
        "legend_note": ("POD is a probability of detection, never a cleared flag; it is clamped at "
                        f"{cal.POD_MAX} so no cell can read as fully searched. All numbers are simulation."),
    }


def export_coverage(cmap: CoverageMap, out_dir: str | Path, stem: str = "coverage",
                    include_effective: bool = True, write_geojson: bool = True) -> dict[str, Any]:
    """Write the PNG + GeoJSON + manifest set into `out_dir`. Returns the manifest (also written to disk)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = cmap.any_grid
    for lx in _layers(cmap, include_effective):
        (out / f"{stem}_{lx.name}.png").write_bytes(png_bytes(overlay_rgba(lx.pod, cmap.cannot_clear)))
        if write_geojson:
            gj = banded_geojson(grid, lx.pod, cmap.cannot_clear, layer=lx.name)
            (out / f"{stem}_{lx.name}.geojson").write_text(json.dumps(gj), encoding="utf-8")
    man = manifest(cmap, stem, include_effective)
    (out / f"{stem}.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def overlay_payload(cmap: CoverageMap, include_effective: bool = True, include_geojson: bool = False
                    ) -> dict[str, Any]:
    """The same manifest with the PNGs inlined as data URIs — what the WebSocket pushes to the map."""
    urls: dict[str, dict[str, str]] = {}
    geo: dict[str, Any] = {}
    for lx in _layers(cmap, include_effective):
        b64 = base64.b64encode(png_bytes(overlay_rgba(lx.pod, cmap.cannot_clear))).decode("ascii")
        urls[lx.name] = {"image": f"data:image/png;base64,{b64}", "geojson": ""}
        if include_geojson:
            geo[lx.name] = banded_geojson(cmap.any_grid, lx.pod, cmap.cannot_clear, layer=lx.name)
    man = manifest(cmap, "coverage", include_effective, urls)
    if include_geojson:
        man["geojson"] = geo
    return man
