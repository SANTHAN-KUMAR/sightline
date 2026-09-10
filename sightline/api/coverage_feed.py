"""The coverage-raster contract between the coverage lane (B6) and the map (B5).

`CONTRACT: sightline.coverage.overlay/1.0`
-----------------------------------------
B6 owns :class:`~sightline.schemas.CoverageGrid`. The map cannot consume a numpy array over HTTP, so the two
lanes exchange **three files in one directory**:

===========================  ============================================================================
``pod.png``                  RGBA PNG, one pixel per grid cell, ``n_east`` wide and ``n_north`` tall.
                             **Row 0 of the PNG is the NORTHERNMOST row** (image top = north), i.e. the
                             ``CoverageGrid.pod`` array flipped vertically, because ``CoverageGrid`` puts
                             its origin at the SW corner and row 0 in the SOUTH. Alpha is 0 where
                             ``coverage == 0`` (never swept), so unsearched ground shows the basemap.
``cannot_clear.geojson``     RFC 7946 FeatureCollection of Polygons in ``[lon, lat]`` covering the
                             ``CoverageGrid.cannot_clear`` mask. Rendered **hatched** (§5.9). R10: these
                             areas are never "cleared", only marked as unclearable from the air.
``overlay.json``             the sidecar below; the map reads this first and everything else from it.
===========================  ============================================================================

``overlay.json``::

    {
      "contract": "sightline.coverage.overlay/1.0",
      "run_id": "flight_003", "generated_utc": 1789..., "domain": "sim",
      "presentation": "body", "k": 1.0,
      "origin_lat": 11.4870, "origin_lon": 76.1450,      # SW corner, as in CoverageGrid
      "cell_m": 20.0, "n_north": 60, "n_east": 80,
      "bounds": {"west":.., "south":.., "east":.., "north":..},
      "coordinates": [[w,n],[e,n],[e,s],[w,s]],           # MapLibre image source order: TL,TR,BR,BL
      "png": "pod.png", "cannot_clear": "cannot_clear.geojson",
      "pod_min":.., "pod_max":.., "pod_mean":.., "cells_swept": .., "cells_total": ..,
      "cells_cannot_clear": ..
    }

**How B6 produces it:** call :func:`grid_to_overlay(grid, out_dir, run_id=..., domain=...)` — it does the
flip, the colour ramp, the polygonisation and the sidecar. If B6 would rather write the files itself, the
three names, the north-up orientation and the sidecar keys above are the whole contract.

**How the map consumes it:** ``GET /api/coverage/overlay.json`` -> the sidecar, then ``GET
/api/coverage/pod.png`` as a MapLibre ``image`` source at ``coordinates``, and
``GET /api/coverage/cannot_clear.geojson`` as a hatched fill.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sightline.common.geodesy import offset_ne
from sightline.schemas import CoverageGrid

__all__ = [
    "OVERLAY_CONTRACT",
    "grid_to_overlay",
    "read_overlay",
    "mask_to_polygons",
    "pod_to_rgba",
    "POD_RAMP",
]

OVERLAY_CONTRACT = "sightline.coverage.overlay/1.0"

#: Viridis anchors (perceptually uniform, colour-blind safe). POD 0 -> dark violet, POD 1 -> yellow.
POD_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def pod_to_rgba(pod: np.ndarray, swept: np.ndarray, *, alpha: int = 185) -> np.ndarray:
    """(H, W) POD in [0, 1] -> (H, W, 4) uint8. Cells with no coverage at all are fully transparent."""
    p = np.clip(np.asarray(pod, dtype=np.float32), 0.0, 1.0)
    stops = np.array([s for s, _ in POD_RAMP], dtype=np.float32)
    cols = np.array([c for _, c in POD_RAMP], dtype=np.float32)
    out = np.zeros(p.shape + (4,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(p, stops, cols[:, ch]).astype(np.uint8)
    out[..., 3] = np.where(np.asarray(swept) > 0, alpha, 0).astype(np.uint8)
    return out


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a 1-D bool row."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for j, v in enumerate(row):
        if v and start is None:
            start = j
        elif not v and start is not None:
            out.append((start, j))
            start = None
    if start is not None:
        out.append((start, len(row)))
    return out


def mask_to_polygons(
    mask: np.ndarray, origin_lat: float, origin_lon: float, cell_m: float, *, reason: str = "burial_polygon"
) -> dict[str, Any]:
    """Bool (n_north, n_east) mask -> RFC 7946 FeatureCollection of axis-aligned rectangles.

    Greedy decomposition: horizontal runs per row, merged downward while identical. Exact (no smoothing),
    and it keeps the polygon count low enough for MapLibre to fill with a hatch pattern.
    """
    m = np.asarray(mask, dtype=bool)
    n_north = m.shape[0]
    open_rects: dict[tuple[int, int], int] = {}  # (j0, j1) -> i_start
    closed: list[tuple[int, int, int, int]] = []  # (i0, i1, j0, j1) half-open
    for i in range(n_north):
        rows = set(_runs(m[i]))
        for key in list(open_rects):
            if key not in rows:
                closed.append((open_rects.pop(key), i, key[0], key[1]))
        for key in rows:
            open_rects.setdefault(key, i)
    for key, i0 in open_rects.items():
        closed.append((i0, n_north, key[0], key[1]))

    feats = []
    for i0, i1, j0, j1 in sorted(closed):
        s_lat, w_lon = offset_ne(origin_lat, origin_lon, i0 * cell_m, j0 * cell_m)
        n_lat, e_lon = offset_ne(origin_lat, origin_lon, i1 * cell_m, j1 * cell_m)
        ring = [[w_lon, s_lat], [e_lon, s_lat], [e_lon, n_lat], [w_lon, n_lat], [w_lon, s_lat]]
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "reason": reason,
                    "cells": int((i1 - i0) * (j1 - j0)),
                    "area_m2": float((i1 - i0) * (j1 - j0) * cell_m * cell_m),
                    "note": "aerial search cannot clear this area (R10)",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def grid_to_overlay(
    grid: CoverageGrid,
    out_dir: str | Path,
    *,
    run_id: str = "live",
    domain: str = "sim",
    alpha: int = 185,
) -> dict[str, Any]:
    """Write ``pod.png`` + ``cannot_clear.geojson`` + ``overlay.json`` for the map. Returns the sidecar."""
    from PIL import Image  # Pillow is already in the env; imported lazily so the module stays light

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pod = np.asarray(grid.pod, dtype=np.float32)
    cov = np.asarray(grid.coverage, dtype=np.float32)
    # CoverageGrid row 0 is the SOUTHERNMOST; a PNG's row 0 is the top of the image, i.e. NORTH.
    rgba = pod_to_rgba(np.flipud(pod), np.flipud(cov), alpha=alpha)
    Image.fromarray(rgba, mode="RGBA").save(out / "pod.png", optimize=True)

    cc = grid.cannot_clear
    cc_fc = (
        mask_to_polygons(cc, grid.origin_lat, grid.origin_lon, grid.cell_m)
        if cc is not None
        else {"type": "FeatureCollection", "features": []}
    )
    (out / "cannot_clear.geojson").write_text(json.dumps(cc_fc), encoding="utf-8")

    south, west = grid.origin_lat, grid.origin_lon
    north, east = offset_ne(south, west, grid.n_north * grid.cell_m, grid.n_east * grid.cell_m)
    sidecar = {
        "contract": OVERLAY_CONTRACT,
        "run_id": run_id,
        "generated_utc": time.time(),
        "domain": domain,
        "presentation": grid.presentation,
        "k": float(grid.k),
        "origin_lat": float(south),
        "origin_lon": float(west),
        "cell_m": float(grid.cell_m),
        "n_north": int(grid.n_north),
        "n_east": int(grid.n_east),
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "coordinates": [[west, north], [east, north], [east, south], [west, south]],
        "png": "pod.png",
        "cannot_clear": "cannot_clear.geojson",
        "pod_min": float(pod.min()) if pod.size else 0.0,
        "pod_max": float(pod.max()) if pod.size else 0.0,
        "pod_mean": float(pod.mean()) if pod.size else 0.0,
        "cells_swept": int((cov > 0).sum()),
        "cells_total": int(pod.size),
        "cells_cannot_clear": int(np.asarray(cc).sum()) if cc is not None else 0,
    }
    (out / "overlay.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def read_overlay(out_dir: str | Path) -> dict[str, Any] | None:
    p = Path(out_dir) / "overlay.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("contract") != OVERLAY_CONTRACT:
        raise ValueError(f"{p}: expected contract {OVERLAY_CONTRACT}, got {d.get('contract')!r}")
    return d
