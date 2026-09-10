"""Lane B4 - export formats (F14) and Cursor-on-Target for TAK (F17).

Everything here writes into `_artifacts/lanes/triage_export/` (gitignored, on D:) and reads the files back:
a format is only verified when it has been through the disk. Offline, CPU only, no simulator. Run with::

    D:\\Tools\\uv\\uv.exe run pytest tests/test_export.py -q
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from sightline.export import (
    COT_TYPE_ANIMAL,
    COT_TYPE_BY_STATUS,
    DEFAULT_STALE_S,
    LAYER_DISMISSED,
    LAYER_TRIAGE,
    build_feature,
    build_feature_collection,
    callsign_for,
    collect_thumbnails,
    cot_type_for,
    export_bundle,
    feature_summaries,
    kmz_contents,
    layer_of,
    read_ids,
    read_uids,
    record_to_cot_element,
    record_to_cot_xml,
    split_layers,
    validate_cot_event,
    validate_cot_xml,
    validate_feature,
    validate_feature_collection,
    validate_file,
    write_geojson,
    write_kml,
    write_kmz,
    write_layers,
)
from sightline.export.cot import COT_TIME_FORMAT, remarks_for, write_cot_layers
from sightline.export.gpkg import list_layers, read_layer_ids, records_to_geodataframe, write_geopackage
from sightline.schemas import SCHEMA_VERSION, Evidence, Record, ce90_m
from sightline.triage import TriageContext, dismiss, rank_records, retention_check, scan_lane_sources

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "_artifacts" / "lanes" / "triage_export"

T0 = 1789000000.0
HOUR = 3600.0
CTX = TriageContext(incident_t0_utc=T0, now_utc=T0 + 6 * HOUR, water_temp_c=24.0, domain="sim")


@pytest.fixture(scope="module")
def workdir() -> Path:
    d = SCRATCH / "export"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def thumbs(workdir: Path) -> Path:
    """Two small fake JPEGs standing in for evidence crops; the KMZ has to carry these exact bytes."""
    d = workdir / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ev_head_only.jpg").write_bytes(b"\xff\xd8\xff\xe0HEAD-ONLY-CROP")
    (d / "ev_roof.jpg").write_bytes(b"\xff\xd8\xff\xe0ROOF-CROP")
    return d


def make_record(rid: str, **kw) -> Record:
    defaults = dict(
        record_id=rid,
        cluster_id=1,
        status="confirmed",
        cls="human",
        lat=11.5231234567,
        lon=76.1329876543,
        alt_msl_m=762.345,
        h_acc_m=2.6,
        confidence=0.82,
        confidence_max_det=0.91,
        n_observations=7,
        n_tracks_merged=2,
        seen_in_passes=[0, 1],
        first_seen_utc=T0 + HOUR,
        last_seen_utc=T0 + 2 * HOUR,
        count_estimate=1,
        zone="settlement",
        source={"platform": "sim-M3T", "telemetry": "airsim", "sim": True, "aoi_id": "wayanad-demo"},
    )
    defaults.update(kw)
    return Record(**defaults)


@pytest.fixture(scope="module")
def records(thumbs: Path) -> list[Record]:
    """A ranked, realistic list: head-only, immersed, trapped, stranded, animal, and one dismissed."""
    head = make_record(
        "head-only-001",
        submersion="head_only",
        submersion_conf=0.72,
        posture="half_submerged",
        posture_conf=0.61,
        thermal_hot=True,
        motion_state="moving",
        count_estimate=1,
    )
    head.evidence.append(
        Evidence(
            thumb_uri=str(thumbs / "ev_head_only.jpg"),
            clip_id="clip-01",
            frame_idx=1420,
            frame_time_utc=T0 + 2 * HOUR,
            bbox_px=(1201.0, 640.0, 1233.0, 668.0),
            det_conf=0.72,
            camera="thermal",
        )
    )
    immersed = make_record("immersed-002", lat=11.5241, lon=76.1341, submersion="half", submersion_conf=0.66)
    trapped = make_record("trapped-003", lat=11.5251, lon=76.1351, posture="trapped", posture_conf=0.7)
    stranded = make_record(
        "stranded-004", lat=11.5261, lon=76.1361, posture="sitting", posture_conf=0.8, count_estimate=4
    )
    stranded.evidence.append(
        Evidence(
            thumb_uri=str(thumbs / "ev_roof.jpg"),
            clip_id="clip-01",
            frame_idx=980,
            frame_time_utc=T0 + 2 * HOUR,
            bbox_px=(300.0, 210.0, 352.0, 268.0),
            det_conf=0.83,
        )
    )
    animal = make_record("animal-005", cls="animal", lat=11.5271, lon=76.1371, confidence=0.9)
    dismissed = make_record("dismissed-006", lat=11.5281, lon=76.1381, confidence=0.95, status="candidate")
    dismiss(dismissed, "operator inspected the crop: blue tarpaulin", "IC-1", t_utc=T0 + 5 * HOUR)
    missing_thumb = make_record("no-thumb-007", lat=11.5291, lon=76.1391)
    missing_thumb.evidence.append(
        Evidence(
            thumb_uri=str(thumbs / "does_not_exist.jpg"),
            clip_id="clip-02",
            frame_idx=5,
            frame_time_utc=T0 + 3 * HOUR,
            bbox_px=(0.0, 0.0, 10.0, 10.0),
            det_conf=0.5,
        )
    )
    return rank_records([head, immersed, trapped, stranded, animal, dismissed, missing_thumb], CTX)


# ==============================================================================================================
# 1. GeoJSON, RFC 7946 (§5.8)
# ==============================================================================================================
def test_feature_is_rfc7946_with_lon_lat_alt_at_six_decimals(records: list[Record]):
    rec = records[0]
    feature = build_feature(rec)
    assert validate_feature(feature) == []
    assert feature["type"] == "Feature"
    assert feature["id"] == rec.record_id
    assert feature["geometry"]["type"] == "Point"
    coords = feature["geometry"]["coordinates"]
    assert len(coords) == 3
    assert coords[0] == round(rec.lon, 6) != round(rec.lat, 6)  # longitude FIRST (RFC 7946 §3.1.1)
    assert coords[1] == round(rec.lat, 6)
    assert coords[2] == round(rec.alt_msl_m, 2)
    assert abs(coords[0] - rec.lon) < 1e-6 and coords[0] != rec.lon  # actually rounded, not merely equal


def test_feature_collection_carries_schema_domain_and_score_components(records: list[Record]):
    fc = build_feature_collection(records, domain="sim", generated_utc=T0)
    assert validate_feature_collection(fc) == []
    assert fc["type"] == "FeatureCollection"
    assert fc["schema_version"] == SCHEMA_VERSION
    assert fc["domain"] == "sim"  # hard rule 5: a sim artefact says so on its face
    assert fc["n_features"] == len(records) == len(fc["features"])
    for feature in fc["features"]:
        comp = feature["properties"]["score_components"]
        assert set(comp) >= {"p_living", "w_class", "urgency", "count_bonus", "urgency_class", "elapsed_h"}
        product = comp["p_living"] * comp["w_class"] * comp["urgency"] * comp["count_bonus"]
        assert product == pytest.approx(feature["properties"]["score"], rel=1e-12)


def test_validator_rejects_a_swapped_coordinate_pair():
    """The single most likely GeoJSON bug in a lat/lon codebase, caught rather than shipped."""
    good = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [176.132988, 11.523123]},
        "properties": {"layer": LAYER_TRIAGE, "score_components": {"p_living": 1.0}},
    }
    assert validate_feature(good) == []
    swapped = json.loads(json.dumps(good))
    swapped["geometry"]["coordinates"] = [11.523123, 176.132988]
    problems = validate_feature(swapped)
    assert problems and "latitude" in problems[0]


def test_validator_rejects_nan_and_unrounded_coordinates():
    base = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [76.132988, 11.523123]},
        "properties": {"layer": LAYER_TRIAGE, "score_components": {"p_living": 1.0}},
    }
    nan_feature = json.loads(json.dumps(base))
    nan_feature["geometry"]["coordinates"] = [float("nan"), 11.523123]
    assert any("finite" in p for p in validate_feature(nan_feature))
    unrounded = json.loads(json.dumps(base))
    unrounded["geometry"]["coordinates"] = [76.13298765432, 11.523123]
    assert any("6 decimals" in p for p in validate_feature(unrounded))
    missing_components = json.loads(json.dumps(base))
    missing_components["properties"] = {"layer": LAYER_TRIAGE}
    assert any("score_components" in p for p in validate_feature(missing_components))


def test_write_geojson_refuses_to_emit_nan(workdir: Path):
    bad = make_record("nan-lat", lat=float("nan"))
    with pytest.raises(ValueError, match="RFC 7946|finite|invalid"):
        write_geojson(workdir / "must_not_exist.geojson", [bad], domain="sim")
    bad_prop = make_record("nan-prop", h_acc_m=float("nan"))
    with pytest.raises(ValueError):
        write_geojson(workdir / "must_not_exist2.geojson", [bad_prop], domain="sim")


def test_written_geojson_validates_from_disk(workdir: Path, records: list[Record]):
    path = write_geojson(workdir / "all.geojson", records, domain="sim", generated_utc=T0)
    assert validate_file(path) == []
    assert set(read_ids(path)) == {r.record_id for r in records}
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_dismissed_records_get_their_own_geojson_layer_and_are_never_dropped(workdir: Path, records: list[Record]):
    """§5.8 R10: dismissed records stay in the exported bundle under a separate layer."""
    paths = write_layers(workdir, records, domain="sim", stem="records", generated_utc=T0)
    assert set(paths) == {LAYER_TRIAGE, LAYER_DISMISSED}
    triage_ids = read_ids(paths[LAYER_TRIAGE])
    dismissed_ids = read_ids(paths[LAYER_DISMISSED])
    assert dismissed_ids == ["dismissed-006"]
    assert "dismissed-006" not in triage_ids
    assert retention_check(records, [_Id(i) for i in triage_ids + dismissed_ids]) == []
    for path, layer in ((paths[LAYER_TRIAGE], LAYER_TRIAGE), (paths[LAYER_DISMISSED], LAYER_DISMISSED)):
        fc = json.loads(path.read_text(encoding="utf-8"))
        assert fc["layer"] == layer
        assert all(f["properties"]["layer"] == layer for f in fc["features"])
    # the dismissal reason travels with the record
    dismissed_fc = json.loads(paths[LAYER_DISMISSED].read_text(encoding="utf-8"))
    props = dismissed_fc["features"][0]["properties"]
    assert props["dismissed_reason"] == "operator inspected the crop: blue tarpaulin"
    assert props["dismissed_by"] == "IC-1"
    assert props["score"] > 0.0  # §1.4: priority is never zeroed by a dismissal


def test_empty_layers_are_still_written(workdir: Path):
    """A missing file must never be readable as a deletion."""
    paths = write_layers(workdir / "empty", [make_record("only-active")], domain="sim", generated_utc=T0)
    assert paths[LAYER_DISMISSED].is_file()
    assert json.loads(paths[LAYER_DISMISSED].read_text(encoding="utf-8"))["features"] == []


def test_live_feed_summary_stays_under_5kb_and_keeps_thumbnails_as_references(records: list[Record]):
    """§5.8: "Thumbnails are file references in the live feed (keeps messages under 5 KB)"."""
    summaries = feature_summaries(records)
    assert len(summaries) == len(records)
    for summary in summaries:
        payload = json.dumps(summary).encode("utf-8")
        assert len(payload) < 5120, (summary["record_id"], len(payload))
        assert b"\xff\xd8" not in payload  # no embedded JPEG bytes
        assert set(summary["score_components"]) >= {"p_living", "w_class", "urgency", "count_bonus"}
    with_thumb = next(s for s in summaries if s["record_id"] == "head-only-001")
    assert with_thumb["thumb_uri"].endswith("ev_head_only.jpg")


# ==============================================================================================================
# 2. KML / KMZ (§5.8)
# ==============================================================================================================
def test_kml_has_both_layers_as_folders_and_every_record(workdir: Path, records: list[Record]):
    path = write_kml(workdir / "records.kml", records, domain="sim", thumb_root=workdir)
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    folders = root.findall(".//k:Folder", ns)
    assert len(folders) == 2
    names = [f.findtext("k:name", default="", namespaces=ns) for f in folders]
    assert any("Triage" in n for n in names) and any("Dismissed" in n for n in names)
    placemarks = root.findall(".//k:Placemark", ns)
    assert len(placemarks) == len(records)
    text = path.read_text(encoding="utf-8")
    for rec in records:
        assert rec.record_id in text
    assert 'src="files/' not in text  # plain KML keeps thumbnails as external references (§5.8)
    assert "ev_head_only.jpg" in text


def test_kml_balloon_shows_every_score_component(workdir: Path, records: list[Record]):
    """§5.8: the number is never shown alone."""
    path = workdir / "records.kml"
    text = path.read_text(encoding="utf-8")
    for token in ("p_living", "w_class(t)", "urgency", "count bonus", "posture", "submersion", "thermal boost"):
        assert token in text, token
    assert "record retained (R10)" in text


def test_kmz_actually_contains_the_thumbnail_bytes(workdir: Path, thumbs: Path, records: list[Record]):
    """§5.8: thumbnails "are embedded only in exported bundles (KMZ files/ ...)"."""
    path = write_kmz(workdir / "records.kmz", records, domain="sim", thumb_root=workdir)
    names = kmz_contents(path)
    assert "doc.kml" in names
    embedded = sorted(n for n in names if n.startswith("files/"))
    assert embedded == ["files/ev_head_only.jpg", "files/ev_roof.jpg"]
    with zipfile.ZipFile(path) as archive:
        assert archive.read("files/ev_head_only.jpg") == (thumbs / "ev_head_only.jpg").read_bytes()
        assert archive.read("files/ev_roof.jpg") == (thumbs / "ev_roof.jpg").read_bytes()
        doc = archive.read("doc.kml").decode("utf-8")
    assert 'src="files/ev_head_only.jpg"' in doc
    for rec in records:
        assert rec.record_id in doc


def test_missing_thumbnails_are_skipped_not_fatal(workdir: Path, records: list[Record]):
    found = collect_thumbnails(records, workdir)
    assert len(found) == 2 and all(p.is_file() for p in found)
    assert all("does_not_exist" not in str(p) for p in found)


# ==============================================================================================================
# 3. Cursor-on-Target for TAK (§5.9, F17)
# ==============================================================================================================
def test_cot_event_parses_and_carries_the_right_fields(records: list[Record]):
    rec = records[0]
    payload = record_to_cot_xml(rec, stale_s=DEFAULT_STALE_S, now_utc=T0 + 6 * HOUR, domain="sim")
    assert validate_cot_xml(payload) == []
    event = ET.fromstring(payload.decode("utf-8"))
    assert event.tag == "event"
    assert event.get("version") == "2.0"
    assert event.get("uid") == rec.record_id
    assert event.get("type") == COT_TYPE_BY_STATUS["confirmed"] == "a-f-G"
    assert event.get("how") == "m-f"

    point = event.find("point")
    assert float(point.get("lat")) == pytest.approx(round(rec.lat, 6), abs=1e-9)
    assert float(point.get("lon")) == pytest.approx(round(rec.lon, 6), abs=1e-9)
    assert float(point.get("hae")) == pytest.approx(round(rec.alt_msl_m, 2), abs=1e-9)
    assert float(point.get("ce")) == pytest.approx(round(ce90_m(rec.h_acc_m), 2), abs=1e-9)
    assert float(point.get("le")) == 9999999.0

    detail = event.find("detail")
    assert detail.find("contact").get("callsign") == callsign_for(rec)
    assert detail.find("contact").get("callsign").startswith("SL-001")
    remarks = detail.findtext("remarks") or ""
    for token in ("p_living", "w_class", "urgency", "count", "posture", "CE90"):
        assert token in remarks, token
    sl = detail.find("sightline")
    assert sl.get("record_id") == rec.record_id
    assert sl.get("domain") == "sim" and sl.get("layer") == LAYER_TRIAGE
    assert float(sl.get("score")) == pytest.approx(rec.score, rel=1e-6)
    assert sl.get("posture") == rec.posture and float(sl.get("posture_conf")) == pytest.approx(rec.posture_conf)
    assert sl.get("posture_promoted") == "true"


def test_cot_timestamps_are_anchored_on_the_record_and_stale_late(records: list[Record]):
    """A survivor marker that silently expires off the ATAK map is the map-level version of a deletion."""
    rec = records[0]
    event = record_to_cot_element(rec, stale_s=DEFAULT_STALE_S, now_utc=T0 + 99 * HOUR)
    import datetime as dt

    parse = lambda s: dt.datetime.strptime(s, COT_TIME_FORMAT).replace(tzinfo=dt.UTC)
    start, stale = parse(event.get("start")), parse(event.get("stale"))
    assert event.get("time") == event.get("start")
    assert start.timestamp() == pytest.approx(rec.last_seen_utc, abs=1e-3)  # not wall clock
    assert (stale - start).total_seconds() == pytest.approx(DEFAULT_STALE_S, abs=1e-3)
    assert DEFAULT_STALE_S == 86400.0


def test_cot_types_map_by_status_and_class(records: list[Record]):
    assert cot_type_for(make_record("a", status="candidate")) == "a-p-G"
    assert cot_type_for(make_record("b", status="confirmed")) == "a-f-G"
    assert cot_type_for(make_record("c", status="stale")) == "a-u-G"
    assert cot_type_for(make_record("d", status="dismissed")) == "a-u-G"
    assert cot_type_for(make_record("e", cls="animal", status="confirmed")) == COT_TYPE_ANIMAL == "a-n-G"


def test_cot_layers_written_and_every_record_survives(workdir: Path, records: list[Record]):
    paths = write_cot_layers(workdir, records, stale_s=DEFAULT_STALE_S, now_utc=T0 + 6 * HOUR, domain="sim")
    assert set(paths) == {LAYER_TRIAGE, LAYER_DISMISSED}
    for path in paths.values():
        assert validate_cot_xml(path) == []
    uids = read_uids(paths[LAYER_TRIAGE]) + read_uids(paths[LAYER_DISMISSED])
    assert set(uids) == {r.record_id for r in records}
    assert read_uids(paths[LAYER_DISMISSED]) == ["dismissed-006"]
    dismissed_event = ET.fromstring(paths[LAYER_DISMISSED].read_text(encoding="utf-8")).find("event")
    remarks = dismissed_event.find("detail").findtext("remarks") or ""
    assert "DISMISSED by IC-1" in remarks and "blue tarpaulin" in remarks and "record retained" in remarks


def test_cot_validator_catches_broken_events(records: list[Record]):
    """A validator that never fires proves nothing."""
    good = record_to_cot_element(records[1], now_utc=T0 + 6 * HOUR)
    assert validate_cot_event(good) == []

    no_point = record_to_cot_element(records[1], now_utc=T0 + 6 * HOUR)
    no_point.remove(no_point.find("point"))
    assert any("point" in p for p in validate_cot_event(no_point))

    bad_lat = record_to_cot_element(records[1], now_utc=T0 + 6 * HOUR)
    bad_lat.find("point").set("lat", "300.0")
    assert any("lat" in p for p in validate_cot_event(bad_lat))

    bad_time = record_to_cot_element(records[1], now_utc=T0 + 6 * HOUR)
    bad_time.set("stale", "not-a-timestamp")
    assert any("ISO-8601" in p for p in validate_cot_event(bad_time))

    dead_on_arrival = record_to_cot_element(records[1], stale_s=-10.0, now_utc=T0 + 6 * HOUR)
    assert any("stale must be after start" in p for p in validate_cot_event(dead_on_arrival))

    no_uid = record_to_cot_element(records[1], now_utc=T0 + 6 * HOUR)
    del no_uid.attrib["uid"]
    assert any("uid" in p for p in validate_cot_event(no_uid))

    assert validate_cot_xml(b"<not-cot/>") == ["root element must be <event> or <events>, got <not-cot>"]
    assert any("well-formed" in p for p in validate_cot_xml(b"<event"))


def test_cot_remarks_never_show_the_score_alone(records: list[Record]):
    for rec in records:
        text = remarks_for(rec)
        assert f"{rec.score:.4f}" in text
        assert "p_living" in text and "w_class" in text and "urgency" in text


# ==============================================================================================================
# 4. GeoPackage (§5.8)
# ==============================================================================================================
def test_geopackage_has_both_layers_and_round_trips_every_record(workdir: Path, records: list[Record]):
    path = workdir / "records.gpkg"
    counts = write_geopackage(path, records, domain="sim")
    assert counts == {LAYER_TRIAGE: len(records) - 1, LAYER_DISMISSED: 1}
    assert set(list_layers(path)) == {LAYER_TRIAGE, LAYER_DISMISSED}
    triage_ids = read_layer_ids(path, LAYER_TRIAGE)
    dismissed_ids = read_layer_ids(path, LAYER_DISMISSED)
    assert dismissed_ids == ["dismissed-006"]
    assert set(triage_ids + dismissed_ids) == {r.record_id for r in records}


def test_geopackage_keeps_the_score_components_and_evidence(workdir: Path, records: list[Record]):
    frame = records_to_geodataframe(records, domain="sim")
    assert len(frame) == len(records)
    assert frame.crs is not None and frame.crs.to_epsg() == 4326
    row = frame[frame["record_id"] == "head-only-001"].iloc[0]
    assert row["urgency_class"] == "immersed" and bool(row["posture_promoted"]) is True
    assert row["domain"] == "sim"
    assert row["ce90_m"] == pytest.approx(ce90_m(2.6), rel=1e-9)
    components = json.loads(row["score_components"])
    product = components["p_living"] * components["w_class"] * components["urgency"] * components["count_bonus"]
    assert product == pytest.approx(row["score"], rel=1e-9)
    evidence = json.loads(row["evidence"])
    assert len(evidence) == 1 and evidence[0]["thumb_uri"].endswith("ev_head_only.jpg")
    # geometry is lon/lat, matching GeoJSON
    assert row.geometry.x == pytest.approx(round(records[0].lon, 6), abs=1e-9)
    assert row.geometry.y == pytest.approx(round(records[0].lat, 6), abs=1e-9)


# ==============================================================================================================
# 5. The bundle, and R10 across every path at once
# ==============================================================================================================
def test_bundle_writes_every_format_and_retains_every_record(workdir: Path, records: list[Record]):
    out = workdir / "bundle"
    result = export_bundle(
        out, records, domain="sim", thumb_root=workdir, now_utc=T0 + 6 * HOUR, generated_utc=T0 + 6 * HOUR
    )
    assert result.n_records == len(records)
    for key in ("geojson_triage", "geojson_dismissed", "kml", "kmz", "cot_triage", "cot_dismissed", "gpkg", "manifest"):
        assert Path(result.files[key]).is_file(), key
    assert result.counts[LAYER_DISMISSED] == 1
    assert result.counts["kmz_files"] == 2
    assert result.counts["gpkg_triage"] == len(records) - 1
    assert result.warnings and "thumbnail" in result.warnings[0]  # the deliberately missing crop

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["domain"] == "sim" and manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_records"] == len(records)


def test_the_dismissed_record_survives_every_single_export_path(workdir: Path, records: list[Record]):
    """The one assertion R10 lives or dies on."""
    out = workdir / "bundle"
    export_bundle(out, records, domain="sim", thumb_root=workdir, now_utc=T0 + 6 * HOUR, generated_utc=T0 + 6 * HOUR)
    rid = "dismissed-006"
    assert rid in read_ids(out / "records_dismissed.geojson")
    assert rid in read_uids(out / "cot_dismissed.xml")
    assert rid in read_layer_ids(out / "records.gpkg", LAYER_DISMISSED)
    assert rid in (out / "records.kml").read_text(encoding="utf-8")
    with zipfile.ZipFile(out / "records.kmz") as archive:
        assert rid in archive.read("doc.kml").decode("utf-8")
    # and it kept its score, its reason and its author everywhere
    feature = next(
        f for f in json.loads((out / "records_dismissed.geojson").read_text(encoding="utf-8"))["features"]
        if f["id"] == rid
    )
    assert feature["properties"]["score"] > 0.0
    assert feature["properties"]["dismissed_reason"] and feature["properties"]["dismissed_by"] == "IC-1"


def test_bundle_rejects_an_unknown_format(workdir: Path, records: list[Record]):
    with pytest.raises(ValueError, match="unknown export format"):
        export_bundle(workdir / "bad", records, formats=("geojson", "shapefile"))


def test_layer_helpers_partition_without_loss(records: list[Record]):
    by_layer = split_layers(records)
    assert sum(len(v) for v in by_layer.values()) == len(records)
    assert all(layer_of(r) == LAYER_DISMISSED for r in by_layer[LAYER_DISMISSED])
    assert retention_check(records, by_layer[LAYER_TRIAGE], by_layer[LAYER_DISMISSED]) == []


def test_no_delete_shaped_code_in_the_export_lane():
    violations = scan_lane_sources(REPO, dirs=("sightline/export",))
    assert violations == [], "\n".join(str(v) for v in violations)


class _Id:
    """Minimal stand-in so retention_check can consume ids read back from a file."""

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id


def test_no_module_in_this_lane_imports_torch_or_touches_the_sim():
    """Lane constraint: pure CPU, no GPU, no simulator control."""
    forbidden = ("torch", "ultralytics", "tensorrt", "rfdetr", "cosysairsim", "mcp__sightline")
    for py in sorted((REPO / "sightline" / "triage").rglob("*.py")) + sorted((REPO / "sightline" / "export").rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in text and f"from {name}" not in text, (py, name)
    assert math.isfinite(1.0)
