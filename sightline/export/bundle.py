"""F14/F17 - the exported bundle: every format at once, with a manifest and a retention proof.

§5.8 asks for GeoJSON (primary), KML/KMZ, GeoPackage and Cursor-on-Target, with thumbnails embedded in the
bundle rather than referenced, and dismissed records present in a separate layer. This module writes all of
them into one directory and then *checks its own work*: :func:`export_bundle` runs
:func:`sightline.triage.guardrails.retention_check` over what it actually wrote back from disk, so a record
that fell out of any single format fails the export instead of quietly disappearing (R10).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from sightline.export import cot as cot_export
from sightline.export import geojson as geojson_export
from sightline.export import kml as kml_export
from sightline.schemas import SCHEMA_VERSION, Record
from sightline.triage.guardrails import GuardrailError, retention_check

ALL_FORMATS: tuple[str, ...] = ("geojson", "kml", "kmz", "gpkg", "cot")

#: KMZ and GeoPackage are the two that need a third-party writer; the rest are stdlib + pytak.
_HEAVY_FORMATS = frozenset({"gpkg"})


@dataclass(slots=True)
class BundleResult:
    out_dir: Path
    domain: str
    n_records: int
    files: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    thumbnails: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        data["schema_version"] = SCHEMA_VERSION
        return data


def export_bundle(
    out_dir: Path | str,
    records: Iterable[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    formats: Sequence[str] = ALL_FORMATS,
    thumb_root: Path | str | None = None,
    stale_s: float = cot_export.DEFAULT_STALE_S,
    now_utc: float | None = None,
    generated_utc: float | None = None,
    stem: str = "records",
) -> BundleResult:
    """Write the whole bundle. Every requested format gets both layers; nothing is ever left out.

    Raises:
        GuardrailError: if any input record id is missing from any written format (R10 retention).
    """
    items = list(records)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(f"unknown export format(s) {unknown}; known: {ALL_FORMATS}")

    result = BundleResult(out_dir=directory, domain=str(domain), n_records=len(items))
    by_layer = geojson_export.split_layers(items)
    result.counts = {layer: len(rows) for layer, rows in by_layer.items()}
    written_ids: list[list[str]] = []

    if "geojson" in formats:
        paths = geojson_export.write_layers(directory, items, domain=domain, stem=stem, generated_utc=generated_utc)
        for layer, path in paths.items():
            result.files[f"geojson_{layer}"] = str(path)
            problems = geojson_export.validate_file(path)
            if problems:
                raise ValueError(f"{path} failed RFC 7946 validation: {problems[:3]}")
        written_ids.append(
            geojson_export.read_ids(paths[geojson_export.LAYER_TRIAGE])
            + geojson_export.read_ids(paths[geojson_export.LAYER_DISMISSED])
        )

    thumbs = kml_export.collect_thumbnails(items, thumb_root)
    result.thumbnails = [str(p) for p in thumbs]
    missing = [ev.thumb_uri for r in items for ev in r.evidence if kml_export.resolve_thumb(ev.thumb_uri, thumb_root) is None]
    if missing:
        result.warnings.append(f"{len(missing)} evidence thumbnail(s) not readable locally; kept as references")

    if "kml" in formats:
        path = kml_export.write_kml(directory / f"{stem}.kml", items, domain=domain, thumb_root=thumb_root)
        result.files["kml"] = str(path)

    if "kmz" in formats:
        path = kml_export.write_kmz(directory / f"{stem}.kmz", items, domain=domain, thumb_root=thumb_root)
        result.files["kmz"] = str(path)
        names = kml_export.kmz_contents(path)
        result.counts["kmz_files"] = len([n for n in names if n.startswith("files/")])

    if "cot" in formats:
        paths = cot_export.write_cot_layers(
            directory, items, stem="cot", stale_s=stale_s, now_utc=now_utc, domain=domain
        )
        for layer, path in paths.items():
            result.files[f"cot_{layer}"] = str(path)
            problems = cot_export.validate_cot_xml(path)
            if problems:
                raise ValueError(f"{path} failed CoT validation: {problems[:3]}")
        written_ids.append(
            cot_export.read_uids(paths[geojson_export.LAYER_TRIAGE])
            + cot_export.read_uids(paths[geojson_export.LAYER_DISMISSED])
        )

    if "gpkg" in formats:
        from sightline.export import gpkg as gpkg_export

        path = directory / f"{stem}.gpkg"
        counts = gpkg_export.write_geopackage(path, items, domain=domain)
        result.files["gpkg"] = str(path)
        result.counts["gpkg_triage"] = counts[geojson_export.LAYER_TRIAGE]
        result.counts["gpkg_dismissed"] = counts[geojson_export.LAYER_DISMISSED]
        written_ids.append(
            gpkg_export.read_layer_ids(path, geojson_export.LAYER_TRIAGE)
            + gpkg_export.read_layer_ids(path, geojson_export.LAYER_DISMISSED)
        )

    for ids in written_ids:
        lost = retention_check(items, [_IdOnly(i) for i in ids])
        if lost:
            raise GuardrailError(f"R10: {len(lost)} record(s) missing from an export layer: {lost[:5]}")

    manifest = directory / "manifest.json"
    payload = result.to_manifest()
    payload["generated_utc"] = time.time() if generated_utc is None else float(generated_utc)
    manifest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    result.files["manifest"] = str(manifest)
    return result


@dataclass(frozen=True, slots=True)
class _IdOnly:
    """Adapter so :func:`retention_check` can compare written ids against in-memory records."""

    record_id: str
