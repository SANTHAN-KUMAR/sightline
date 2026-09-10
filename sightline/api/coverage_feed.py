"""The coverage-raster contract between the coverage lane (B6) and the map (B5).

**The contract is B6's, not ours.** `sightline.coverage.export.export_coverage(cmap, out_dir)` writes the file
set below (documented in `docs/lanes/coverage_plan_eval.md` §5.1); this module only *reads* it and re-shapes it
into the URLs the map fetches. Nothing here writes into `sightline/coverage/`.

`CONTRACT: sightline.coverage` (B6, product key in the manifest)
---------------------------------------------------------------
===============================  =======================================================================
``coverage.json``                ONE manifest: ``grid``, ``bounds``, MapLibre image ``coordinates``,
                                 the magma ``ramp``, the POD ``bands``, ``passes``, ``cannot_clear_label``,
                                 ``legend_note``, and one entry per layer (``body``, ``limb_only``,
                                 ``effective``) carrying ``image_url`` / ``geojson_url`` / ``k`` /
                                 ``k_is_measured`` / ``k_basis`` / ``stats`` (every stat has ``domain``).
``coverage_<layer>.png``         RGBA, one pixel per cell, **image row 0 = NORTH** (the CoverageGrid array's
                                 row 0 is SOUTH). Alpha is 0 wherever the cell is ``cannot_clear``, so the
                                 map's hatch shows through the hole rather than being painted over.
``coverage_<layer>.geojson``     POD band polygons (``properties.band`` / ``pod_min`` / ``pod_max`` /
                                 ``label`` / ``domain``) **plus** one Feature per burial region with
                                 ``properties.cannot_clear = true`` and the label "aerial search cannot
                                 clear". RFC 7946, ``[lon, lat]``.
===============================  =======================================================================

What this module adds
---------------------
* :func:`read_coverage` finds whichever format is on disk and returns ONE normalised sidecar (see
  :data:`SIDECAR_CONTRACT`) whose ``layers[i].image_url`` / ``geojson_url`` are **API URLs**, so the page never
  has to know the directory layout: ``/api/coverage/raster.png?layer=body``.
* :func:`cannot_clear_geojson` pulls the ``cannot_clear`` Features out of a layer's GeoJSON, because §5.9 draws
  them as their own hatched layer. Nothing is renamed and no polygon is dropped: R10 means such an area is
  never "cleared" by an overflight, only ever marked unclearable from the air.
* :func:`grid_to_overlay` is the **legacy** path: it renders a bare :class:`~sightline.schemas.CoverageGrid`
  (no ``CoverageMap``, no k provenance) into ``pod.png`` + ``cannot_clear.geojson`` + ``overlay.json``. It
  predates B6's exporter and is kept only so a lane holding a raw grid can still light the map up; anything
  that owns a ``CoverageMap`` must use ``sightline.coverage.export_coverage``.

R10: nothing here can mark a cell "cleared". POD is transported as a probability and the ``cannot_clear``
polygons travel with their label attached.
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
    "COVERAGE_PRODUCT",
    "OVERLAY_CONTRACT",
    "SIDECAR_CONTRACT",
    "CANNOT_CLEAR_LABEL",
    "read_coverage",
    "read_overlay",
    "read_manifest",
    "normalise_manifest",
    "layer_names",
    "layer_file",
    "cannot_clear_geojson",
    "grid_to_overlay",
    "mask_to_polygons",
    "pod_to_rgba",
    "POD_RAMP",
]

#: B6's manifest ``product`` key (`sightline/coverage/export.py`).
COVERAGE_PRODUCT = "sightline.coverage"
#: This lane's legacy single-grid sidecar (see `grid_to_overlay`).
OVERLAY_CONTRACT = "sightline.coverage.overlay/1.0"
#: What `read_coverage()` returns and what `app/map/index.html` consumes.
SIDECAR_CONTRACT = "sightline.api.coverage_sidecar/1.1"
CANNOT_CLEAR_LABEL = "aerial search cannot clear"

#: Preference order when the manifest does not say which layer to show first.
_LAYER_PREFERENCE = ("effective", "body", "limb_only")

#: Viridis anchors, used only by the legacy `grid_to_overlay` path. B6 ships its own magma ramp in the manifest.
POD_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


# --- reading B6's export ----------------------------------------------------------------------------------
def read_manifest(cov_dir: str | Path, stem: str = "coverage") -> dict[str, Any] | None:
    """The raw B6 manifest, or None when B6 has not exported into this directory."""
    p = Path(cov_dir) / f"{stem}.json"
    if not p.is_file():
        return None
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("product") != COVERAGE_PRODUCT:
        raise ValueError(f"{p}: expected product {COVERAGE_PRODUCT!r}, got {man.get('product')!r}")
    if not man.get("layers"):
        raise ValueError(f"{p}: manifest has no layers")
    return man


def layer_names(man: dict[str, Any]) -> list[str]:
    return [str(x.get("layer")) for x in man.get("layers", [])]


def default_layer(man: dict[str, Any]) -> str:
    names = layer_names(man)
    for want in _LAYER_PREFERENCE:
        if want in names:
            return want
    return names[0] if names else ""


def layer_file(cov_dir: str | Path, man: dict[str, Any], layer: str, kind: str) -> Path | None:
    """Resolve ``kind`` in {'image', 'geojson'} for ``layer`` to a file inside ``cov_dir``.

    The manifest's URLs are file names written by B6; a data: URI (``overlay_payload``) has no file and
    returns None. Path traversal out of ``cov_dir`` is refused.
    """
    key = "image_url" if kind == "image" else "geojson_url"
    entry = next((x for x in man.get("layers", []) if x.get("layer") == layer), None)
    if entry is None:
        return None
    name = str(entry.get(key) or "")
    if not name or name.startswith("data:") or "://" in name:
        return None
    root = Path(cov_dir).resolve()
    p = (root / name).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


def normalise_manifest(man: dict[str, Any], *, base_url: str = "/api/coverage") -> dict[str, Any]:
    """B6 manifest -> the sidecar the map reads. Layer URLs become API URLs; nothing else is renamed."""
    cache_bust = int(man.get("generated_utc") or 0) or None
    layers = []
    for x in man.get("layers", []):
        name = str(x.get("layer"))
        q = f"?layer={name}" + (f"&t={cache_bust}" if cache_bust else "")
        layers.append({
            "layer": name,
            "image_url": f"{base_url}/raster.png{q}",
            "geojson_url": f"{base_url}/bands.geojson{q}",
            "k": x.get("k"),
            "k_is_measured": bool(x.get("k_is_measured", False)),
            "k_basis": x.get("k_basis", ""),
            "stats": x.get("stats", {}),
        })
    return {
        "contract": SIDECAR_CONTRACT,
        "source": COVERAGE_PRODUCT,
        "available": True,
        "schema_version": man.get("schema_version", ""),
        "domain": man.get("domain", "sim"),
        "grid": man.get("grid", {}),
        "bounds": man.get("bounds", {}),
        "coordinates": man.get("coordinates", []),
        "ramp": man.get("ramp", []),
        "bands": man.get("bands", []),
        "passes": man.get("passes", []),
        "cannot_clear_label": man.get("cannot_clear_label", CANNOT_CLEAR_LABEL),
        "legend_note": man.get("legend_note", ""),
        "layers": layers,
        "default_layer": default_layer(man),
        "cannot_clear_url": f"{base_url}/cannot_clear.geojson",
        "generated_utc": man.get("generated_utc"),
    }


def _normalise_legacy(side: dict[str, Any], *, base_url: str = "/api/coverage") -> dict[str, Any]:
    """The legacy single-grid `overlay.json` presented through the same sidecar shape."""
    t = int(side.get("generated_utc") or 0)
    layer = str(side.get("presentation") or "body")
    return {
        "contract": SIDECAR_CONTRACT,
        "source": OVERLAY_CONTRACT,
        "available": True,
        "schema_version": side.get("schema_version", ""),
        "domain": side.get("domain", "sim"),
        "grid": {"origin_lat": side.get("origin_lat"), "origin_lon": side.get("origin_lon"),
                 "cell_m": side.get("cell_m"), "n_north": side.get("n_north"),
                 "n_east": side.get("n_east"),
                 "row_order": "image row 0 is NORTH; array row 0 is SOUTH"},
        "bounds": side.get("bounds", {}),
        "coordinates": side.get("coordinates", []),
        "ramp": [{"pod": s, "rgb": list(c)} for s, c in POD_RAMP],
        "bands": [0.0, 0.25, 0.5, 0.75, 1.0],
        "passes": [],
        "cannot_clear_label": CANNOT_CLEAR_LABEL,
        "legend_note": "POD is a probability of detection, never a cleared flag.",
        "layers": [{
            "layer": layer,
            "image_url": f"{base_url}/raster.png?layer={layer}" + (f"&t={t}" if t else ""),
            "geojson_url": None,
            "k": side.get("k"),
            "k_is_measured": False,
            "k_basis": "legacy single-grid overlay; k carried through unvalidated",
            "stats": {"mean_pod": side.get("pod_mean"), "max_pod": side.get("pod_max"),
                      "cells_swept": side.get("cells_swept"), "cells_total": side.get("cells_total"),
                      "cannot_clear_cells": side.get("cells_cannot_clear"),
                      "domain": side.get("domain", "sim")},
        }],
        "default_layer": layer,
        "cannot_clear_url": f"{base_url}/cannot_clear.geojson",
        "generated_utc": side.get("generated_utc"),
    }


def read_coverage(cov_dir: str | Path, *, base_url: str = "/api/coverage") -> dict[str, Any] | None:
    """The normalised sidecar for whatever is in ``cov_dir``, or None when nothing has been exported."""
    man = read_manifest(cov_dir)
    if man is not None:
        return normalise_manifest(man, base_url=base_url)
    p = Path(cov_dir) / "overlay.json"
    if not p.is_file():
        return None
    side = json.loads(p.read_text(encoding="utf-8"))
    if side.get("contract") != OVERLAY_CONTRACT:
        raise ValueError(f"{p}: expected contract {OVERLAY_CONTRACT}, got {side.get('contract')!r}")
    return _normalise_legacy(side, base_url=base_url)


#: Back-compatible alias (the API route and older callers used this name).
read_overlay = read_coverage


def cannot_clear_geojson(cov_dir: str | Path, layer: str | None = None) -> dict[str, Any]:
    """Every "aerial search cannot clear" polygon, as its own FeatureCollection.

    B6 ships them inside each layer's banded GeoJSON tagged ``properties.cannot_clear = true``; the legacy
    path ships a whole file. Either way the label travels with the geometry and nothing is dropped.
    """
    root = Path(cov_dir)
    man = read_manifest(root)
    if man is not None:
        lay = layer or default_layer(man)
        p = layer_file(root, man, lay, "geojson")
        feats: list[dict[str, Any]] = []
        if p is not None:
            fc = json.loads(p.read_text(encoding="utf-8"))
            for f in fc.get("features", []):
                if (f.get("properties") or {}).get("cannot_clear"):
                    props = dict(f["properties"])
                    props.setdefault("label", man.get("cannot_clear_label", CANNOT_CLEAR_LABEL))
                    props.setdefault("domain", man.get("domain", "sim"))
                    feats.append({**f, "properties": props})
        return {"type": "FeatureCollection", "source": COVERAGE_PRODUCT, "layer": lay, "features": feats}
    legacy = root / "cannot_clear.geojson"
    if legacy.is_file():
        fc = json.loads(legacy.read_text(encoding="utf-8"))
        fc["source"] = OVERLAY_CONTRACT
        return fc
    return {"type": "FeatureCollection", "features": []}


# --- legacy: render a bare CoverageGrid ------------------------------------------------------------------
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
                    "cannot_clear": True,
                    "label": CANNOT_CLEAR_LABEL,
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
    """LEGACY. Render a bare `CoverageGrid` into ``pod.png`` + ``cannot_clear.geojson`` + ``overlay.json``.

    Superseded by ``sightline.coverage.export_coverage``, which carries the presentation layers and the
    provenance of ``k``. Kept for a caller that holds only a grid.
    """
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
