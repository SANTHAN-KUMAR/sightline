"""Decoder for the §5.8 wire format: GeoJSON Feature -> `Record`.

`schemas.Record.to_feature()` is the encoder and is frozen; the cloud sink needs the inverse to upsert what
the outbox posts. It lives here (this lane) rather than in `schemas.py` so the frozen contract stays untouched
— see `docs/lanes/store_api_map.md` if the orchestrator wants it promoted.

**Lossy by design:** `to_feature()` rounds coordinates to 6 decimals (~0.11 m at this latitude) and altitude to
2. A decoded record therefore matches the original to 1e-6 deg, not bit-for-bit. That is the doc's wire format,
not a bug — but it means the cloud copy is a *transport* copy; the edge log is the authority.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from sightline.schemas import Evidence, Record, ScoreComponents

__all__ = ["feature_to_record", "RECORD_FIELDS"]

RECORD_FIELDS = {f.name for f in fields(Record)}
_COMPONENT_FIELDS = {f.name for f in fields(ScoreComponents)}
_EVIDENCE_FIELDS = {f.name for f in fields(Evidence)}


def feature_to_record(feature: dict[str, Any]) -> Record:
    if feature.get("type") != "Feature":
        raise ValueError("not a GeoJSON Feature")
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Point":
        raise ValueError("record geometry must be a Point")
    coords = list(geom.get("coordinates") or [])
    if len(coords) < 2:
        raise ValueError("record geometry needs [lon, lat(, alt)]")
    lon, lat = float(coords[0]), float(coords[1])
    alt = float(coords[2]) if len(coords) > 2 else 0.0

    props = dict(feature.get("properties") or {})
    comps = props.pop("score_components", None) or {}
    ev = props.pop("evidence", None) or []
    kw = {k: v for k, v in props.items() if k in RECORD_FIELDS}
    kw["lat"], kw["lon"], kw["alt_msl_m"] = lat, lon, alt
    kw["components"] = ScoreComponents(**{k: v for k, v in comps.items() if k in _COMPONENT_FIELDS})
    kw["evidence"] = [
        Evidence(**{**{k: v for k, v in e.items() if k in _EVIDENCE_FIELDS},
                    "bbox_px": tuple(e.get("bbox_px", (0.0, 0.0, 0.0, 0.0)))})
        for e in ev
    ]
    kw["seen_in_passes"] = list(kw.get("seen_in_passes") or [])
    kw["source"] = dict(kw.get("source") or {})
    rid = feature.get("id")
    if rid and "record_id" not in kw:
        kw["record_id"] = str(rid)
    return Record(**kw)
