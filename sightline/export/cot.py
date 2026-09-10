"""F17 - Cursor-on-Target for TAK, built with ``pytak`` (SOLUTION_DOC §5.9).

    **Responder integration: TAK.** Real teams increasingly run ATAK/WinTAK. The detections are emitted as
    Cursor-on-Target (CoT) events using ``pytak`` so a WinTAK/ATAK client or a TAK server shows the same
    markers. This is the same integration OpenAthena uses to push target coordinates into ATAK.  -- §5.9

The document names the format and the library but not the CoT type strings, so the mapping below is ADOPTED and
written out. Two decisions are worth arguing with:

* **Type by status, not by class.** A record the pipeline has not yet corroborated is ``a-p-G`` - *pending*
  affiliation, ground - which is what OpenAthena publishes for a calculated target and is the honest thing to
  put on a responder's screen. Only a ``confirmed`` record becomes ``a-f-G`` (friendly ground), and animals are
  ``a-n-G`` (neutral) so they cannot be mistaken for people.
* **A long stale time.** CoT markers vanish from ATAK when they go stale, and a survivor marker silently
  expiring is the map-level version of the thing R10 forbids. The default is 24 h, not pytak's usual minutes.

Dismissed records are emitted too, in their own document, with the dismissal reason in ``<remarks>`` - §5.8's
"separate layer", expressed in the only way a stream of CoT events can express one.
"""

from __future__ import annotations

import datetime as _dt
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from sightline.export.geojson import LAYER_DISMISSED, LAYER_TRIAGE, layer_of, split_layers
from sightline.schemas import SCHEMA_VERSION, Record, ce90_m

with warnings.catch_warnings():  # pytak warns about the optional aiohttp extra we do not use
    warnings.simplefilter("ignore")
    import pytak

#: CoT timestamp format, from ``pytak.ISO_8601_UTC``.
COT_TIME_FORMAT = pytak.ISO_8601_UTC

#: ADOPTED type mapping, see the module docstring.
COT_TYPE_BY_STATUS: dict[str, str] = {
    "candidate": "a-p-G",  # pending affiliation, ground - an uncorroborated calculated target
    "confirmed": "a-f-G",  # friendly ground - a person to be rescued
    "stale": "a-u-G",  # unknown affiliation, ground
    "dismissed": "a-u-G",
}
COT_TYPE_ANIMAL = "a-n-G"  # neutral ground

#: 24 h. A survivor marker must not silently expire off the map (§1.4, R10).
DEFAULT_STALE_S = 86400.0

#: ``how`` = machine, fused: the position is a fusion of GNSS, attitude and the ray-ground intersection (§5.7).
DEFAULT_HOW = "m-f"

#: CoT's "unknown" sentinel for a field we cannot supply.
COT_UNKNOWN = 9999999.0

_TYPE_RX = re.compile(r"^[a-z](?:-[A-Za-z0-9]+)+$")


def cot_type_for(record: Record) -> str:
    if record.cls == "animal":
        return COT_TYPE_ANIMAL
    return COT_TYPE_BY_STATUS.get(str(record.status), "a-u-G")


def callsign_for(record: Record, prefix: str = "SL") -> str:
    """``SL-007 IMMERSED`` - rank first, because that is what a responder picks off the map."""
    rank = "NEW" if record.priority_rank < 0 else f"{record.priority_rank + 1:03d}"
    return f"{prefix}-{rank} {str(record.components.urgency_class).upper()}"


def cot_timestamp(t_utc: float) -> str:
    return _dt.datetime.fromtimestamp(float(t_utc), tz=_dt.UTC).strftime(COT_TIME_FORMAT)


def remarks_for(record: Record) -> str:
    """The human-readable payload. §5.8 forbids showing the score without its components, TAK included."""
    c = record.components
    parts = [
        f"Sightline {record.cls} rank {'--' if record.priority_rank < 0 else record.priority_rank + 1}",
        (
            f"score {record.score:.4f} = p_living {c.p_living:.3f} x w_class {c.w_class:.3f}"
            f" x urgency {c.urgency:.2f} ({c.urgency_class}) x count {c.count_bonus:.2f}"
        ),
        f"t+{c.elapsed_h:.1f} h; thermal x{c.thermal_boost:.2f}; motion x{c.motion_boost:.2f} ({record.motion_state})",
        f"posture {record.posture} {record.posture_conf:.2f}; submersion {record.submersion} {record.submersion_conf:.2f}"
        + (" [posture promoted]" if c.posture_promoted else ""),
        f"count {record.count_estimate} ({record.count_min}-{record.count_max}); obs {record.n_observations}",
        f"h_acc {record.h_acc_m:.1f} m 1-sigma, CE90 {ce90_m(record.h_acc_m):.1f} m ({record.method})",
        f"status {record.status}",
    ]
    if record.status == "dismissed":
        parts.append(f"DISMISSED by {record.dismissed_by}: {record.dismissed_reason} (record retained, R10)")
    if record.evidence:
        parts.append(f"evidence {record.evidence[0].thumb_uri}")
    return " | ".join(parts)


def record_to_cot_element(
    record: Record,
    *,
    stale_s: float = DEFAULT_STALE_S,
    how: str = DEFAULT_HOW,
    now_utc: float | None = None,
    domain: Literal["sim", "real"] = "sim",
    callsign_prefix: str = "SL",
) -> ET.Element:
    """One CoT ``<event>`` for one record.

    ``time``/``start`` come from the record's ``last_seen_utc`` (falling back to ``now_utc``), so replaying an
    export does not rewrite history. ``point/ce`` is CE90 in metres from the record's own 1-sigma ``h_acc_m``
    via :func:`sightline.schemas.ce90_m`; ``le`` is CoT's unknown sentinel because the vertical budget is not
    carried on the record.
    """
    now = _dt.datetime.now(tz=_dt.UTC).timestamp() if now_utc is None else float(now_utc)
    t_event = float(record.last_seen_utc) if record.last_seen_utc else now

    point = pytak.cot_point(
        lat=round(float(record.lat), 6),
        lon=round(float(record.lon), 6),
        hae=round(float(record.alt_msl_m), 2),
        ce=round(ce90_m(float(record.h_acc_m)), 2),
        le=COT_UNKNOWN,
        precision=6,
    )

    contact = ET.Element("contact", {"callsign": callsign_for(record, callsign_prefix)})
    remarks = ET.Element("remarks")
    remarks.text = remarks_for(record)
    precision = ET.Element("precisionlocation", {"geopointsrc": "CALC", "altsrc": "CALC"})
    status = ET.Element("status", {"readiness": "true"})
    sightline_el = ET.Element(
        "sightline",
        {
            "schema_version": SCHEMA_VERSION,
            "domain": domain,
            "layer": layer_of(record),
            "record_id": str(record.record_id),
            "status": str(record.status),
            "cls": str(record.cls),
            "priority_rank": str(record.priority_rank),
            "score": f"{record.score:.6f}",
            "p_living": f"{record.components.p_living:.6f}",
            "w_class": f"{record.components.w_class:.6f}",
            "urgency": f"{record.components.urgency:.6f}",
            "count_bonus": f"{record.components.count_bonus:.6f}",
            "urgency_class": str(record.components.urgency_class),
            "elapsed_h": f"{record.components.elapsed_h:.4f}",
            "posture": str(record.posture),
            "posture_conf": f"{record.posture_conf:.3f}",
            "submersion": str(record.submersion),
            "submersion_conf": f"{record.submersion_conf:.3f}",
            "posture_promoted": "true" if record.components.posture_promoted else "false",
            "count_estimate": str(record.count_estimate),
            "h_acc_m": f"{record.h_acc_m:.2f}",
            "dismissed_reason": str(record.dismissed_reason),
        },
    )

    detail = pytak.cot_detail(contact, remarks, precision, status, sightline_el)
    event = pytak.cot_event(
        uid=str(record.record_id),
        cot_type=cot_type_for(record),
        how=how,
        point=point,
        detail=detail,
    )
    # pytak stamps time/start/stale from wall clock; anchor them on the record so exports are reproducible.
    event.set("time", cot_timestamp(t_event))
    event.set("start", cot_timestamp(t_event))
    event.set("stale", cot_timestamp(t_event + float(stale_s)))
    return event


def record_to_cot_xml(record: Record, **kwargs) -> bytes:
    """One CoT message, serialised - the unit a TAK client or ``pytak`` worker consumes."""
    return pytak.serialize_cot(record_to_cot_element(record, **kwargs))


def records_to_cot_xml(records: Iterable[Record], **kwargs) -> list[bytes]:
    return [record_to_cot_xml(r, **kwargs) for r in records]


def build_cot_document(records: Sequence[Record], **kwargs) -> ET.Element:
    """An ``<events>`` wrapper holding one ``<event>`` per record - the file form, not the wire form."""
    root = ET.Element("events", {"schema_version": SCHEMA_VERSION, "count": str(len(records))})
    for rec in records:
        root.append(record_to_cot_element(rec, **kwargs))
    return root


def write_cot(path: Path | str, records: Sequence[Record], **kwargs) -> Path:
    """Write one ``<events>`` document. Use :func:`write_cot_layers` to get §5.8's separate dismissed layer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = build_cot_document(list(records), **kwargs)
    ET.ElementTree(root).write(str(target), encoding="utf-8", xml_declaration=True)
    return target


def write_cot_layers(directory: Path | str, records: Iterable[Record], *, stem: str = "cot", **kwargs) -> dict[str, Path]:
    """``cot_triage.xml`` and ``cot_dismissed.xml``. Both are always written, even when empty (R10)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer = split_layers(records)
    return {
        layer: write_cot(out_dir / f"{stem}_{layer}.xml", by_layer[layer], **kwargs)
        for layer in (LAYER_TRIAGE, LAYER_DISMISSED)
    }


# --- validation ---------------------------------------------------------------------------------------------
_REQUIRED_EVENT_ATTRS = ("version", "uid", "type", "how", "time", "start", "stale")


def validate_cot_xml(data: bytes | str | Path) -> list[str]:
    """Return CoT problems; empty means valid. Accepts one ``<event>``, an ``<events>`` document, or a path."""
    if isinstance(data, Path):
        text = data.read_text(encoding="utf-8")
    elif isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    if root.tag == "events":
        events = list(root.findall("event"))
        if len(events) != len(list(root)):
            return ["<events> may only contain <event> children"]
    elif root.tag == "event":
        events = [root]
    else:
        return [f"root element must be <event> or <events>, got <{root.tag}>"]
    problems: list[str] = []
    for i, event in enumerate(events):
        problems.extend(f"event[{i}]: {p}" for p in validate_cot_event(event))
    return problems


def validate_cot_event(event: ET.Element) -> list[str]:
    """Structural checks against the CoT event schema."""
    problems: list[str] = []
    for attr in _REQUIRED_EVENT_ATTRS:
        if not event.get(attr):
            problems.append(f"missing required attribute {attr!r}")
    if event.get("version") not in (None, "2.0"):
        problems.append(f'version must be "2.0", got {event.get("version")!r}')
    cot_type = event.get("type") or ""
    if cot_type and not _TYPE_RX.match(cot_type):
        problems.append(f"type {cot_type!r} is not a CoT type string")

    times: dict[str, _dt.datetime] = {}
    for attr in ("time", "start", "stale"):
        raw = event.get(attr)
        if not raw:
            continue
        try:
            times[attr] = _dt.datetime.strptime(raw, COT_TIME_FORMAT).replace(tzinfo=_dt.UTC)
        except ValueError:
            problems.append(f"{attr}={raw!r} is not CoT ISO-8601 UTC ({COT_TIME_FORMAT})")
    if "stale" in times and "start" in times and times["stale"] <= times["start"]:
        problems.append("stale must be after start, or the marker is dead on arrival")

    points = event.findall("point")
    if len(points) != 1:
        problems.append(f"expected exactly one <point>, found {len(points)}")
        return problems
    point = points[0]
    values: dict[str, float] = {}
    for attr in ("lat", "lon", "hae", "ce", "le"):
        raw = point.get(attr)
        if raw is None:
            problems.append(f"point is missing {attr!r}")
            continue
        try:
            values[attr] = float(raw)
        except ValueError:
            problems.append(f"point/{attr}={raw!r} is not a number")
    if "lat" in values and not -90.0 <= values["lat"] <= 90.0:
        problems.append(f"point/lat {values['lat']} outside [-90, 90]")
    if "lon" in values and not -180.0 <= values["lon"] <= 180.0:
        problems.append(f"point/lon {values['lon']} outside [-180, 180]")
    if "ce" in values and values["ce"] < 0:
        problems.append("point/ce must not be negative")

    detail = event.find("detail")
    if detail is None:
        problems.append("missing <detail>")
        return problems
    contact = detail.find("contact")
    if contact is None or not contact.get("callsign"):
        problems.append("<detail> must carry a <contact callsign=...> for the marker label")
    return problems


def read_uids(path: Path | str) -> list[str]:
    """The record ids inside a written CoT document - the retention check's read side."""
    root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
    events = [root] if root.tag == "event" else list(root.findall("event"))
    return [str(e.get("uid")) for e in events if e.get("uid")]
