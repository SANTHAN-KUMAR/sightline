"""F14 (formats) + F17 (TAK): GeoJSON, KML/KMZ, GeoPackage and Cursor-on-Target (SOLUTION_DOC §5.8, §5.9).

GeoJSON is primary. Dismissed records are exported in their own layer by every writer here and are never
dropped (guardrail R10). Evidence thumbnails stay *file references* in the live feed, which is what keeps a
record message under 5 KB, and are embedded only inside the KMZ bundle.

    from sightline.export import export_bundle
    result = export_bundle("_artifacts/triage", ranked, domain="sim", thumb_root="_artifacts/thumbs")
"""

from sightline.export.bundle import ALL_FORMATS, BundleResult, export_bundle
from sightline.export.cot import (
    COT_TIME_FORMAT,
    COT_TYPE_ANIMAL,
    COT_TYPE_BY_STATUS,
    DEFAULT_HOW,
    DEFAULT_STALE_S,
    build_cot_document,
    callsign_for,
    cot_type_for,
    read_uids,
    record_to_cot_element,
    record_to_cot_xml,
    records_to_cot_xml,
    validate_cot_event,
    validate_cot_xml,
    write_cot,
    write_cot_layers,
)
from sightline.export.geojson import (
    COORD_DECIMALS,
    LAYER_DISMISSED,
    LAYER_TRIAGE,
    LAYERS,
    build_feature,
    build_feature_collection,
    feature_summaries,
    layer_of,
    read_ids,
    split_layers,
    validate_feature,
    validate_feature_collection,
    validate_file,
    write_geojson,
    write_layers,
)
from sightline.export.kml import (
    build_kml,
    collect_thumbnails,
    kmz_contents,
    resolve_thumb,
    write_kml,
    write_kmz,
)

__all__ = [
    "ALL_FORMATS",
    "COORD_DECIMALS",
    "COT_TIME_FORMAT",
    "COT_TYPE_ANIMAL",
    "COT_TYPE_BY_STATUS",
    "DEFAULT_HOW",
    "DEFAULT_STALE_S",
    "LAYERS",
    "LAYER_DISMISSED",
    "LAYER_TRIAGE",
    "BundleResult",
    "build_cot_document",
    "build_feature",
    "build_feature_collection",
    "build_kml",
    "callsign_for",
    "collect_thumbnails",
    "cot_type_for",
    "export_bundle",
    "feature_summaries",
    "kmz_contents",
    "layer_of",
    "read_ids",
    "read_uids",
    "record_to_cot_element",
    "record_to_cot_xml",
    "records_to_cot_xml",
    "resolve_thumb",
    "split_layers",
    "validate_cot_event",
    "validate_cot_xml",
    "validate_feature",
    "validate_feature_collection",
    "validate_file",
    "write_cot",
    "write_cot_layers",
    "write_geojson",
    "write_kml",
    "write_kmz",
    "write_layers",
]


def write_geopackage(*args, **kwargs):
    """GeoPackage writer (lazy re-export: importing geopandas costs ~300 MB, so it is deferred)."""
    from sightline.export.gpkg import write_geopackage as _impl

    return _impl(*args, **kwargs)


def records_to_geodataframe(*args, **kwargs):
    """GeoDataFrame builder (lazy re-export, see :func:`write_geopackage`)."""
    from sightline.export.gpkg import records_to_geodataframe as _impl

    return _impl(*args, **kwargs)


__all__ += ["records_to_geodataframe", "write_geopackage"]
