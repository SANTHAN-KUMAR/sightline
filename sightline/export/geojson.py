"""F14 (formats) - GeoJSON, the primary output (SOLUTION_DOC §5.8, §1.3 "Product 1").

    **Record schema (GeoJSON Feature, RFC 7946, coordinates ``[lon, lat, alt]``, 6 decimals).**
    ... GeoJSON is primary (Python ``geojson`` or plain ``json``).  -- §5.8

``Record.to_feature()`` and ``feature_collection()`` in the frozen schema already produce the Feature and the
FeatureCollection. This module is the file-level layer around them: the dismissed/active split that R10
requires, the ``domain`` label that hard rule 5 requires on anything numeric leaving a module, and an RFC 7946
validator so a malformed export fails in the test suite instead of in a responder's map.

**Layers.** GeoJSON has no native layer concept, so both spellings are provided and both are used by the
bundle: every Feature carries ``properties.layer``, and :func:`write_layers` additionally writes one file per
layer. Dismissed records are never dropped from either (§5.8 R10).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from sightline.schemas import SCHEMA_VERSION, Record

LAYER_TRIAGE = "triage"
LAYER_DISMISSED = "dismissed"
LAYERS: tuple[str, str] = (LAYER_TRIAGE, LAYER_DISMISSED)

#: RFC 7946 §11.2 recommends 6 decimals; §5.8 fixes it at 6. The schema rounds; this is here for the validator.
COORD_DECIMALS = 6


def layer_of(record: Record) -> str:
    """Which export layer a record belongs to. Dismissal moves a record between layers, never out of them."""
    return LAYER_DISMISSED if record.status == "dismissed" else LAYER_TRIAGE


def split_layers(records: Iterable[Record]) -> dict[str, list[Record]]:
    """Partition into the two layers. Every input record lands in exactly one of them."""
    out: dict[str, list[Record]] = {LAYER_TRIAGE: [], LAYER_DISMISSED: []}
    for rec in records:
        out[layer_of(rec)].append(rec)
    return out


def build_feature(record: Record, layer: str | None = None) -> dict[str, Any]:
    """``Record.to_feature()`` plus the layer tag. The Record itself is not modified."""
    feature = record.to_feature()
    props = dict(feature["properties"])
    props["layer"] = layer or layer_of(record)
    feature["properties"] = props
    return feature


def build_feature_collection(
    records: Iterable[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    layer: str | None = None,
    generated_utc: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An RFC 7946 FeatureCollection with the provenance fields this project requires on every artefact.

    ``domain`` is mandatory in spirit (hard rule 5 in ``docs/HANDBOOK.md``): a file of simulator records must
    say so on its face, because a sim number may never be averaged with a real one.
    """
    items = list(records)
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.time() if generated_utc is None else float(generated_utc),
        "domain": domain,
        "layer": layer if layer is not None else "all",
        "n_features": len(items),
        "features": [build_feature(r, layer) for r in items],
    }
    if extra:
        for key, value in extra.items():
            if key not in ("type", "features"):
                fc[key] = value
    return fc


def write_geojson(
    path: Path | str,
    records: Iterable[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    layer: str | None = None,
    generated_utc: float | None = None,
    indent: int | None = 2,
) -> Path:
    """Write one FeatureCollection. ``allow_nan=False`` - RFC 7946 has no NaN, so a bad number fails loudly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fc = build_feature_collection(records, domain=domain, layer=layer, generated_utc=generated_utc)
    problems = validate_feature_collection(fc)
    if problems:
        raise ValueError(f"refusing to write invalid GeoJSON to {target}: " + "; ".join(problems[:5]))
    target.write_text(json.dumps(fc, indent=indent, allow_nan=False, default=str), encoding="utf-8")
    return target


def write_layers(
    directory: Path | str,
    records: Iterable[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    stem: str = "records",
    generated_utc: float | None = None,
) -> dict[str, Path]:
    """Write one file per layer: ``<stem>_triage.geojson`` and ``<stem>_dismissed.geojson``.

    Both files are always written, even when a layer is empty, so a consumer can tell "no dismissals" from
    "the dismissed layer was not exported" (R10: a missing file must never be read as a deletion).
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer = split_layers(records)
    paths: dict[str, Path] = {}
    for layer in LAYERS:
        paths[layer] = write_geojson(
            out_dir / f"{stem}_{layer}.geojson",
            by_layer[layer],
            domain=domain,
            layer=layer,
            generated_utc=generated_utc,
        )
    return paths


# --- validation ---------------------------------------------------------------------------------------------
def validate_feature_collection(fc: dict[str, Any]) -> list[str]:
    """Return a list of RFC 7946 problems; empty means valid for the shape this project emits.

    Checked: the FeatureCollection / Feature / Point type strings, ``features`` being an array, ``geometry``
    and ``properties`` being present on every Feature, positions being ``[lon, lat]`` or ``[lon, lat, alt]``
    with the longitude FIRST (RFC 7946 §3.1.1) and inside the valid ranges, all numbers finite, ``id`` being a
    string or number (§3.2), and coordinates rounded to :data:`COORD_DECIMALS` as §5.8 requires.
    """
    problems: list[str] = []
    if not isinstance(fc, dict):
        return ["not a JSON object"]
    if fc.get("type") != "FeatureCollection":
        problems.append(f'type must be "FeatureCollection", got {fc.get("type")!r}')
    features = fc.get("features")
    if not isinstance(features, list):
        return problems + ['"features" must be an array']
    for i, feature in enumerate(features):
        problems.extend(f"features[{i}]: {p}" for p in validate_feature(feature))
    return problems


def validate_feature(feature: Any) -> list[str]:
    """RFC 7946 checks for a single Point Feature."""
    problems: list[str] = []
    if not isinstance(feature, dict):
        return ["not a JSON object"]
    if feature.get("type") != "Feature":
        problems.append(f'type must be "Feature", got {feature.get("type")!r}')
    if "geometry" not in feature:
        problems.append('missing "geometry" (RFC 7946 §3.2 requires the member, even when null)')
    if "properties" not in feature:
        problems.append('missing "properties" (RFC 7946 §3.2)')
    if "id" in feature and not isinstance(feature["id"], (str, int, float)):
        problems.append('"id" must be a string or a number (RFC 7946 §3.2)')

    geom = feature.get("geometry")
    if geom is None:
        return problems
    if not isinstance(geom, dict):
        return problems + ["geometry is not a JSON object"]
    if geom.get("type") != "Point":
        problems.append(f'geometry.type must be "Point" for a record, got {geom.get("type")!r}')
    coords = geom.get("coordinates")
    if not isinstance(coords, (list, tuple)):
        return problems + ["geometry.coordinates must be an array"]
    if len(coords) not in (2, 3):
        problems.append(f"a position is [lon, lat] or [lon, lat, alt]; got {len(coords)} values")
        return problems
    for j, value in enumerate(coords):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"coordinates[{j}] is not a number")
            return problems
        if not math.isfinite(float(value)):
            problems.append(f"coordinates[{j}] is not finite; RFC 7946 §3.1 forbids NaN and Infinity")
            return problems
    lon, lat = float(coords[0]), float(coords[1])
    if not -180.0 <= lon <= 180.0:
        problems.append(f"longitude {lon} outside [-180, 180] - is the order [lat, lon] by mistake?")
    if not -90.0 <= lat <= 90.0:
        problems.append(f"latitude {lat} outside [-90, 90] - is the order [lat, lon] by mistake?")
    for j, value in enumerate(coords[:2]):
        if round(float(value), COORD_DECIMALS) != float(value):
            problems.append(f"coordinates[{j}] is not rounded to {COORD_DECIMALS} decimals (§5.8)")

    props = feature.get("properties")
    if isinstance(props, dict):
        if props.get("layer") not in LAYERS:
            problems.append(f"properties.layer must be one of {LAYERS}, got {props.get('layer')!r}")
        if not props.get("score_components"):
            problems.append("properties.score_components missing - §5.8 forbids showing the score alone")
    return problems


def validate_file(path: Path | str) -> list[str]:
    """Parse a written file and validate it. Also proves the file is JSON at all."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        fc = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"not valid JSON: {exc}"]
    if "NaN" in text or "Infinity" in text:
        return ["contains NaN/Infinity, which RFC 7946 §3.1 forbids"]
    return validate_feature_collection(fc)


def read_ids(path: Path | str) -> list[str]:
    """Record ids present in a written GeoJSON file - the retention check's read side."""
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    ids: list[str] = []
    for feature in fc.get("features", []):
        rid = feature.get("id")
        if rid is None:
            rid = (feature.get("properties") or {}).get("record_id")
        if rid is not None:
            ids.append(str(rid))
    return ids


def feature_summaries(records: Sequence[Record]) -> list[dict[str, Any]]:
    """A compact live-feed payload: the map marker fields only, thumbnails kept as URIs.

    §5.8: "Thumbnails are file references in the live feed (keeps messages under 5 KB) and are embedded only in
    exported bundles". This is the shape the WebSocket feed (F15, lane B5) can send without breaking that.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        out.append(
            {
                "record_id": rec.record_id,
                "layer": layer_of(rec),
                "priority_rank": rec.priority_rank,
                "score": rec.score,
                "score_components": {
                    "p_living": rec.components.p_living,
                    "w_class": rec.components.w_class,
                    "urgency": rec.components.urgency,
                    "count_bonus": rec.components.count_bonus,
                    "urgency_class": rec.components.urgency_class,
                    "elapsed_h": rec.components.elapsed_h,
                    "posture_promoted": rec.components.posture_promoted,
                },
                "cls": rec.cls,
                "status": rec.status,
                "lat": round(rec.lat, COORD_DECIMALS),
                "lon": round(rec.lon, COORD_DECIMALS),
                "h_acc_m": rec.h_acc_m,
                "count_estimate": rec.count_estimate,
                "posture": rec.posture,
                "posture_conf": rec.posture_conf,
                "submersion": rec.submersion,
                "submersion_conf": rec.submersion_conf,
                "thumb_uri": rec.evidence[0].thumb_uri if rec.evidence else "",
            }
        )
    return out
