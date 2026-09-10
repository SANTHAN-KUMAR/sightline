"""F14 (formats) - GeoPackage for GIS hand-off (SOLUTION_DOC §5.8).

    GeoPackage export via geopandas/pyogrio for GIS hand-off. ... SpatiaLite on Windows needs a DLL and is
    skipped; PostGIS needs a server and is rejected for a laptop.  -- §5.8

One file, two layers: ``triage`` and ``dismissed``. GeoPackage columns are scalar, so the nested parts of a
record (the score components, the evidence list, the source dict, the pass list) are written as JSON strings
in their own columns rather than being thrown away - a record must survive every export path intact (R10).

geopandas and pyogrio are imported inside the functions: they are the heaviest dependency in this lane and the
GeoJSON/KML/CoT paths must not pay for them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from sightline.export.geojson import LAYER_DISMISSED, LAYER_TRIAGE, layer_of, split_layers
from sightline.schemas import Record, ce90_m

#: Scalar columns, in the order a GIS user will meet them. Everything else goes into the JSON columns below.
SCALAR_COLUMNS: tuple[str, ...] = (
    "record_id",
    "layer",
    "priority_rank",
    "score",
    "p_living",
    "w_class",
    "urgency",
    "count_bonus",
    "urgency_class",
    "elapsed_h",
    "thermal_boost",
    "motion_boost",
    "posture_promoted",
    "status",
    "cls",
    "cluster_id",
    "confidence",
    "confidence_max_det",
    "lat",
    "lon",
    "alt_msl_m",
    "h_acc_m",
    "ce90_m",
    "h_acc_basis",
    "method",
    "dem_source",
    "agl_m",
    "off_nadir_deg",
    "n_observations",
    "n_tracks_merged",
    "first_seen_utc",
    "last_seen_utc",
    "motion_state",
    "motion_displacement_m",
    "motion_window_s",
    "count_estimate",
    "count_min",
    "count_max",
    "count_basis",
    "posture",
    "posture_conf",
    "submersion",
    "submersion_conf",
    "occlusion",
    "modality",
    "thermal_hot",
    "thermal_c",
    "pixel_size_px",
    "gsd_cm_px",
    "zone",
    "notes",
    "dismissed_reason",
    "dismissed_by",
    "dismissed_utc",
    "schema_version",
    "version",
    "domain",
)

#: Nested members kept as JSON text so nothing is lost on the way to a GIS.
JSON_COLUMNS: tuple[str, ...] = ("score_components", "evidence", "source", "seen_in_passes", "n_evidence")


def record_to_row(record: Record, domain: Literal["sim", "real"] = "sim") -> dict[str, Any]:
    """Flatten one record into GeoPackage-safe scalars plus JSON text columns."""
    c = record.components
    row: dict[str, Any] = {
        "record_id": str(record.record_id),
        "layer": layer_of(record),
        "priority_rank": int(record.priority_rank),
        "score": float(record.score),
        "p_living": float(c.p_living),
        "w_class": float(c.w_class),
        "urgency": float(c.urgency),
        "count_bonus": float(c.count_bonus),
        "urgency_class": str(c.urgency_class),
        "elapsed_h": float(c.elapsed_h),
        "thermal_boost": float(c.thermal_boost),
        "motion_boost": float(c.motion_boost),
        "posture_promoted": bool(c.posture_promoted),
        "status": str(record.status),
        "cls": str(record.cls),
        "cluster_id": int(record.cluster_id),
        "confidence": float(record.confidence),
        "confidence_max_det": float(record.confidence_max_det),
        "lat": round(float(record.lat), 6),
        "lon": round(float(record.lon), 6),
        "alt_msl_m": round(float(record.alt_msl_m), 2),
        "h_acc_m": float(record.h_acc_m),
        "ce90_m": float(ce90_m(record.h_acc_m)),
        "h_acc_basis": str(record.h_acc_basis),
        "method": str(record.method),
        "dem_source": str(record.dem_source),
        "agl_m": float(record.agl_m),
        "off_nadir_deg": float(record.off_nadir_deg),
        "n_observations": int(record.n_observations),
        "n_tracks_merged": int(record.n_tracks_merged),
        "first_seen_utc": float(record.first_seen_utc),
        "last_seen_utc": float(record.last_seen_utc),
        "motion_state": str(record.motion_state),
        "motion_displacement_m": float(record.motion_displacement_m),
        "motion_window_s": float(record.motion_window_s),
        "count_estimate": int(record.count_estimate),
        "count_min": int(record.count_min),
        "count_max": int(record.count_max),
        "count_basis": str(record.count_basis),
        "posture": str(record.posture),
        "posture_conf": float(record.posture_conf),
        "submersion": str(record.submersion),
        "submersion_conf": float(record.submersion_conf),
        "occlusion": -1 if record.occlusion is None else int(record.occlusion),
        "modality": str(record.modality),
        "thermal_hot": bool(record.thermal_hot),
        "thermal_c": float("nan") if record.thermal_c is None else float(record.thermal_c),
        "pixel_size_px": float(record.pixel_size_px),
        "gsd_cm_px": float(record.gsd_cm_px),
        "zone": str(record.zone),
        "notes": str(record.notes),
        "dismissed_reason": str(record.dismissed_reason),
        "dismissed_by": str(record.dismissed_by),
        "dismissed_utc": float(record.dismissed_utc),
        "schema_version": str(record.schema_version),
        "version": int(record.version),
        "domain": str(domain),
    }
    feature = record.to_feature()
    props = feature["properties"]
    row["score_components"] = json.dumps(props.get("score_components", {}), default=str)
    row["evidence"] = json.dumps(props.get("evidence", []), default=str)
    row["source"] = json.dumps(props.get("source", {}), default=str)
    row["seen_in_passes"] = json.dumps(list(record.seen_in_passes))
    row["n_evidence"] = len(record.evidence)
    return row


def records_to_geodataframe(records: Iterable[Record], *, domain: Literal["sim", "real"] = "sim"):
    """A WGS-84 (EPSG:4326) point GeoDataFrame with one row per record."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    items = list(records)
    rows = [record_to_row(r, domain) for r in items]
    columns = list(SCALAR_COLUMNS) + list(JSON_COLUMNS)
    frame = pd.DataFrame(rows, columns=columns)
    geometry = [Point(round(float(r.lon), 6), round(float(r.lat), 6), round(float(r.alt_msl_m), 2)) for r in items]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")


def write_geopackage(
    path: Path | str,
    records: Iterable[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
) -> dict[str, int]:
    """Write ``triage`` and ``dismissed`` layers into one ``.gpkg``. Returns ``{layer: row_count}``.

    Both layers are always written, even when empty, so an absent layer can never be mistaken for a deletion.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    by_layer = split_layers(records)
    counts: dict[str, int] = {}
    for layer in (LAYER_TRIAGE, LAYER_DISMISSED):
        frame = records_to_geodataframe(by_layer[layer], domain=domain)
        frame.to_file(str(target), layer=layer, driver="GPKG", engine="pyogrio")
        counts[layer] = len(frame)
    return counts


def read_layer_ids(path: Path | str, layer: str) -> list[str]:
    """Record ids in one GeoPackage layer - the retention check's read side."""
    import geopandas as gpd

    frame = gpd.read_file(str(Path(path)), layer=layer, engine="pyogrio")
    if "record_id" not in frame.columns:
        return []
    return [str(v) for v in frame["record_id"].tolist()]


def list_layers(path: Path | str) -> list[str]:
    import pyogrio

    return [str(row[0]) for row in pyogrio.list_layers(str(Path(path)))]
