"""F14 (formats) - KML and KMZ via ``simplekml`` (SOLUTION_DOC §5.8).

    KML/KMZ via ``simplekml`` (stale but complete; embed thumbnails with ``addfile()``), for Google Earth
    users. ... Thumbnails are file references in the live feed (keeps messages under 5 KB) and are embedded
    only in exported bundles (KMZ ``files/`` or a single-file HTML report).  -- §5.8

So: :func:`write_kml` emits plain KML whose thumbnails stay external references, and :func:`write_kmz` packs
the same document plus the actual image bytes under ``files/`` inside the archive. The two layers of §5.8's R10
paragraph become two KML Folders, so a Google Earth user sees the dismissed records greyed out in their own
folder rather than not at all.
"""

from __future__ import annotations

import html
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from sightline.export.geojson import LAYER_DISMISSED, LAYER_TRIAGE, split_layers
from sightline.schemas import SCHEMA_VERSION, Record

FOLDER_TITLES: dict[str, str] = {
    LAYER_TRIAGE: "Triage list (ranked)",
    LAYER_DISMISSED: "Dismissed - retained (R10)",
}

#: Google Earth marker colours, aabbgrr. Immersed/head-only red, trapped orange, stranded yellow, animal blue,
#: dismissed grey. Colour carries the urgency class; scale carries the rank (§5.9's "size = rank" convention).
_URGENCY_COLOUR: dict[str, str] = {
    "immersed": "ff0000ff",
    "trapped": "ff0080ff",
    "stranded": "ff00d7ff",
    "animal": "ffff8000",
    "unknown": "ffffffff",
}
_DISMISSED_COLOUR = "ff9e9e9e"

_ICON_HREF = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"


def resolve_thumb(uri: str, root: Path | str | None = None) -> Path | None:
    """Turn an ``Evidence.thumb_uri`` into a local file path, or ``None`` when it is not a readable local file.

    Accepts ``file:///D:/x.jpg``, ``D:\\x.jpg`` and ``thumbs/x.jpg`` (relative to ``root``). Remote ``http(s)``
    URIs are left alone: the live feed references them and the bundle simply keeps the reference.
    """
    if not uri:
        return None
    text = str(uri)
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme in ("http", "https"):
        return None
    if parsed.scheme == "file":
        text = urllib.request.url2pathname(parsed.path)
    candidate = Path(text)
    if not candidate.is_absolute() and root is not None:
        candidate = Path(root) / candidate
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def collect_thumbnails(records: Iterable[Record], root: Path | str | None = None) -> list[Path]:
    """Every distinct local evidence thumbnail referenced by these records, in a stable order."""
    seen: dict[str, Path] = {}
    for rec in records:
        for ev in rec.evidence:
            path = resolve_thumb(ev.thumb_uri, root)
            if path is not None:
                seen.setdefault(str(path.resolve()).lower(), path)
    return [seen[key] for key in sorted(seen)]


def _colour_for(record: Record) -> str:
    if record.status == "dismissed":
        return _DISMISSED_COLOUR
    return _URGENCY_COLOUR.get(str(record.components.urgency_class), _URGENCY_COLOUR["unknown"])


def _scale_for(record: Record, n_total: int) -> float:
    """Rank -> marker scale, 1.4 at the top of the list down to 0.7 at the bottom (§5.9 "size = rank")."""
    if record.priority_rank < 0 or n_total <= 1:
        return 1.0
    fraction = min(1.0, max(0.0, record.priority_rank / float(n_total - 1)))
    return round(1.4 - 0.7 * fraction, 3)


def _description_html(record: Record, thumb_href: str | None) -> str:
    """The balloon: the evidence thumbnail plus every score component, never the number alone (§5.8)."""
    c = record.components
    rows: list[tuple[str, str]] = [
        ("score", f"{record.score:.4f}"),
        ("rank", "unranked" if record.priority_rank < 0 else f"#{record.priority_rank + 1}"),
        ("p_living", f"{c.p_living:.3f}"),
        ("w_class(t)", f"{c.w_class:.3f}"),
        ("urgency", f"{c.urgency:.2f} ({c.urgency_class})"),
        ("count bonus", f"{c.count_bonus:.2f} (n={record.count_estimate})"),
        ("elapsed", f"{c.elapsed_h:.1f} h since t0"),
        ("thermal boost", f"x{c.thermal_boost:.2f}" + (" (thermal-positive)" if record.thermal_hot else "")),
        ("motion boost", f"x{c.motion_boost:.2f} ({record.motion_state})"),
        ("posture", f"{record.posture} ({record.posture_conf:.2f})" + (" [promoted]" if c.posture_promoted else "")),
        ("submersion", f"{record.submersion} ({record.submersion_conf:.2f})"),
        ("class", record.cls),
        ("status", record.status),
        ("confidence", f"{record.confidence:.3f} (best det {record.confidence_max_det:.3f})"),
        ("position", f"{record.lat:.6f}, {record.lon:.6f} +/- {record.h_acc_m:.1f} m ({record.h_acc_basis})"),
        ("observations", f"{record.n_observations} in {record.n_tracks_merged} track(s), passes {record.seen_in_passes}"),
        ("zone", record.zone),
    ]
    if record.status == "dismissed":
        rows.append(("dismissed", f"{record.dismissed_reason} (by {record.dismissed_by}) - record retained (R10)"))
    body = "".join(
        f"<tr><td><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    img = f'<p><img src="{html.escape(thumb_href, quote=True)}" width="320"/></p>' if thumb_href else ""
    return f"<![CDATA[{img}<table>{body}</table>]]>"


def build_kml(
    records: Sequence[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    embed_thumbnails: bool = False,
    thumb_root: Path | str | None = None,
    document_name: str = "Sightline triage list",
) -> tuple[Any, list[str]]:
    """Build the ``simplekml.Kml`` document. Returns ``(kml, embedded_paths_inside_the_archive)``.

    ``embed_thumbnails`` is only meaningful for a KMZ - it calls ``Kml.addfile()``, which stages the bytes for
    the archive and returns the ``files/<name>`` href to reference from the balloon.
    """
    import simplekml  # local: keeps the triage lane importable without it

    kml = simplekml.Kml(name=document_name)
    kml.document.description = (
        f"Sightline triage export - schema {SCHEMA_VERSION} - domain={domain} - "
        f"{len(records)} record(s). Dismissed records are retained in their own folder (guardrail R10)."
    )

    embedded: dict[str, str] = {}
    if embed_thumbnails:
        for path in collect_thumbnails(records, thumb_root):
            embedded[str(path.resolve()).lower()] = kml.addfile(str(path))

    by_layer = split_layers(records)
    n_total = len(records)
    for layer in (LAYER_TRIAGE, LAYER_DISMISSED):
        folder = kml.newfolder(name=FOLDER_TITLES[layer])
        folder.description = f"{len(by_layer[layer])} record(s)"
        for rec in by_layer[layer]:
            thumb_href: str | None = None
            if rec.evidence:
                local = resolve_thumb(rec.evidence[0].thumb_uri, thumb_root)
                if local is not None:
                    thumb_href = embedded.get(str(local.resolve()).lower(), local.as_uri())
                elif rec.evidence[0].thumb_uri:
                    thumb_href = str(rec.evidence[0].thumb_uri)

            rank_label = "--" if rec.priority_rank < 0 else f"#{rec.priority_rank + 1}"
            point = folder.newpoint(
                name=f"{rank_label} {rec.cls} {rec.components.urgency_class} {rec.score:.3f}",
                coords=[(round(rec.lon, 6), round(rec.lat, 6), round(rec.alt_msl_m, 2))],
            )
            point.description = _description_html(rec, thumb_href)
            point.altitudemode = simplekml.AltitudeMode.clamptoground
            point.style.iconstyle.icon.href = _ICON_HREF
            point.style.iconstyle.color = _colour_for(rec)
            point.style.iconstyle.scale = _scale_for(rec, n_total)
            point.style.labelstyle.scale = 0.8

            data = point.extendeddata
            data.newdata("record_id", rec.record_id)
            data.newdata("layer", layer)
            data.newdata("status", rec.status)
            data.newdata("priority_rank", rec.priority_rank)
            data.newdata("score", f"{rec.score:.6f}")
            data.newdata("p_living", f"{rec.components.p_living:.6f}")
            data.newdata("w_class", f"{rec.components.w_class:.6f}")
            data.newdata("urgency", f"{rec.components.urgency:.6f}")
            data.newdata("count_bonus", f"{rec.components.count_bonus:.6f}")
            data.newdata("urgency_class", str(rec.components.urgency_class))
            data.newdata("elapsed_h", f"{rec.components.elapsed_h:.4f}")
            data.newdata("posture", f"{rec.posture} {rec.posture_conf:.2f}")
            data.newdata("submersion", f"{rec.submersion} {rec.submersion_conf:.2f}")
            data.newdata("h_acc_m", f"{rec.h_acc_m:.2f}")
            data.newdata("domain", domain)
            if rec.status == "dismissed":
                data.newdata("dismissed_reason", rec.dismissed_reason)
                data.newdata("dismissed_by", rec.dismissed_by)

    return kml, sorted(set(embedded.values()))


def write_kml(
    path: Path | str,
    records: Sequence[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    thumb_root: Path | str | None = None,
) -> Path:
    """Write a plain ``.kml``. Thumbnails stay external references (§5.8's live-feed rule)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kml, _ = build_kml(records, domain=domain, embed_thumbnails=False, thumb_root=thumb_root)
    kml.save(str(target))
    return target


def write_kmz(
    path: Path | str,
    records: Sequence[Record],
    *,
    domain: Literal["sim", "real"] = "sim",
    thumb_root: Path | str | None = None,
    embed_thumbnails: bool = True,
) -> Path:
    """Write a ``.kmz`` with the evidence thumbnails embedded under ``files/`` (§5.8's exported-bundle rule)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kml, _ = build_kml(records, domain=domain, embed_thumbnails=embed_thumbnails, thumb_root=thumb_root)
    kml.savekmz(str(target))
    return target


def kmz_contents(path: Path | str) -> list[str]:
    """Names inside a written KMZ - what the tests assert the thumbnails are actually there by."""
    import zipfile

    with zipfile.ZipFile(Path(path)) as archive:
        return archive.namelist()
