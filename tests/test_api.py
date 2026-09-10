"""F15 + F18 backend, the map page and the contracts this lane consumes (lane B5). Offline, no GPU, no sim.

    D:\\Tools\\uv\\uv.exe run pytest tests/test_api.py -q

Every test here is written so it CAN fail:

* the coverage tests build a **real** `sightline.coverage` export (B6's own `export_coverage`) and assert this
  lane reads exactly that file set — a wrong product key, a missing layer or a renamed URL fails;
* `test_the_coverage_raster_is_geo_aligned_with_the_flown_track` re-derives the swept bounding box from the
  telemetry and fails if the raster drifts from the track. It caught a real bug: a hand-rolled "nadir"
  quaternion that `Telemetry.gimbal_pitch_deg()` read back as -180 deg, which made every footprint oblique
  and pushed the POD raster 414 m south of the flight;
* the R10 tests grep this lane's own source (Python **and** the map page) for a delete path or the word
  "cleared", and drive the HTTP dismiss route end to end;
* the offline test runs the §5.10 story through the real ASGI app: enqueue while the link is down, watch
  `/api/outbox` rise, reconnect, watch it drain, then replay the same job and require `applied = false`.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sightline.api import create_app
from sightline.api.coverage_feed import (
    COVERAGE_PRODUCT,
    SIDECAR_CONTRACT,
    cannot_clear_geojson,
    layer_file,
    read_coverage,
    read_manifest,
)
from sightline.api.demo_seed import BURIAL_RINGS_LATLON, SCENE_JSON, build_route, route_frames, seed_store
from sightline.api.live import MAX_MESSAGE_BYTES, MESSAGE_TYPES, envelope
from sightline.api.mission_feed import ROUTE_PRODUCT, MissionState, route_to_plan_geojson
from sightline.api.wire import feature_to_record
from sightline.schemas import SCHEMA_VERSION, Record
from sightline.store import Outbox, RecordStore, Uploader

REPO = Path(__file__).resolve().parents[1]
LANE_PY = sorted((REPO / "sightline" / "store").glob("*.py")) + sorted((REPO / "sightline" / "api").glob("*.py"))
LANE_WEB = sorted((REPO / "app" / "map").glob("*.html")) + sorted((REPO / "app" / "map").glob("*.mjs"))


# ==========================================================================================================
# fixtures
# ==========================================================================================================
@pytest.fixture(scope="module")
def flight():
    """The real plan-lane route and the telemetry of the part of it that has been flown."""
    route = build_route()
    return route, route_frames(route)


@pytest.fixture(scope="module")
def coverage_dir(tmp_path_factory, flight):
    """A REAL B6 export: `sightline.coverage.export_coverage` writing its own contract, no mock."""
    from sightline.coverage import CoverageMap, export_coverage
    from sightline.coverage.grid import SceneFrame, polygons_from_latlon
    from sightline.coverage.presentation import SIM_RGB_4K

    _route, frames = flight
    out = tmp_path_factory.mktemp("coverage")
    scene = SceneFrame.from_json(SCENE_JSON)
    cmap = CoverageMap.for_scene(scene, cell_m=20.0, presentations=("body", "limb_only"))
    cmap.add_burial_polygons(polygons_from_latlon(cmap.any_grid, [list(r) for r in BURIAL_RINGS_LATLON]))
    intr = SIM_RGB_4K.intrinsics()
    cmap.add_pass(((t, intr) for t in frames), pass_id=0, mode="AUTO")
    export_coverage(cmap, out, stem="coverage")
    return out


@pytest.fixture()
def store(tmp_path):
    with RecordStore(tmp_path / "records.db") as s:
        yield s


@pytest.fixture()
def seeded(store):
    seed_store(store)
    return store


@pytest.fixture()
def app_client(seeded, coverage_dir, flight):
    route, frames = flight
    mission = MissionState()
    from sightline.coverage.presentation import SIM_RGB_4K

    mission.intrinsics = SIM_RGB_4K.intrinsics()
    mission.set_route(route)
    for t in frames[::10]:
        mission.update_pose(t)
    app = create_app(seeded, mission=mission, coverage_dir=coverage_dir, serve_static=False)
    with TestClient(app) as c:
        yield c


# ==========================================================================================================
# 1. the contracts this lane CONSUMES (owned by the coverage / plan lane, B6)
# ==========================================================================================================
def test_the_b6_coverage_manifest_is_read_as_written(coverage_dir):
    man = read_manifest(coverage_dir)
    assert man is not None, "export_coverage wrote nothing this lane can find"
    assert man["product"] == COVERAGE_PRODUCT
    names = [x["layer"] for x in man["layers"]]
    assert names == ["body", "limb_only", "effective"], names
    for x in man["layers"]:
        assert (coverage_dir / x["image_url"]).is_file(), x["image_url"]
        assert (coverage_dir / x["geojson_url"]).is_file(), x["geojson_url"]
        assert x["stats"]["domain"] == "sim", "a coverage statistic reached the map with no domain"
    # the row order matters: MapLibre draws the PNG top-down, CoverageGrid stores it bottom-up
    assert "row 0 is NORTH" in man["grid"]["row_order"]
    assert man["coordinates"][0][1] > man["coordinates"][3][1], "image coordinates must start at the NW corner"


def test_a_manifest_with_the_wrong_product_is_refused(tmp_path):
    (tmp_path / "coverage.json").write_text(json.dumps({"product": "something.else", "layers": [{}]}))
    with pytest.raises(ValueError, match="expected product"):
        read_manifest(tmp_path)


def test_a_manifest_with_no_layers_is_refused(tmp_path):
    (tmp_path / "coverage.json").write_text(json.dumps({"product": COVERAGE_PRODUCT, "layers": []}))
    with pytest.raises(ValueError, match="no layers"):
        read_manifest(tmp_path)


def test_the_sidecar_turns_layer_files_into_api_urls(coverage_dir):
    side = read_coverage(coverage_dir)
    assert side["contract"] == SIDECAR_CONTRACT and side["source"] == COVERAGE_PRODUCT
    assert side["default_layer"] == "effective"
    for lay in side["layers"]:
        assert lay["image_url"].startswith("/api/coverage/raster.png?layer=")
        assert lay["geojson_url"].startswith("/api/coverage/bands.geojson?layer=")
        assert lay["k_is_measured"] is False, "k is derived, not calibrated — the map must not claim otherwise"
    assert side["ramp"] and side["bands"], "the legend needs B6's own ramp and bands, not a local guess"
    assert "never a cleared flag" in side["legend_note"]


def test_no_coverage_directory_reads_as_unavailable_not_as_empty(tmp_path):
    assert read_coverage(tmp_path) is None


def test_layer_file_refuses_a_path_outside_the_coverage_directory(coverage_dir, tmp_path):
    man = dict(read_manifest(coverage_dir))
    man["layers"] = [{"layer": "evil", "image_url": "../../../windows/win.ini"}]
    assert layer_file(coverage_dir, man, "evil", "image") is None


def test_the_cannot_clear_polygons_survive_extraction_with_their_label(coverage_dir):
    fc = cannot_clear_geojson(coverage_dir, "body")
    assert fc["features"], "the burial polygons vanished between B6 and the map"
    assert len(fc["features"]) == len(BURIAL_RINGS_LATLON)
    for f in fc["features"]:
        assert f["properties"]["cannot_clear"] is True
        assert f["properties"]["label"] == "aerial search cannot clear"
        assert f["properties"]["domain"] == "sim"
        ring = f["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "polygon ring is not closed"
        lons = [c[0] for c in ring]
        assert 76.0 < min(lons) < 76.3, f"[lon, lat] order broken: {ring[0]}"


def test_the_coverage_png_is_transparent_exactly_where_search_cannot_clear(coverage_dir):
    """B6 punches alpha 0 through the burial cells so the map's hatch shows through. If that ever became
    opaque, the hatch would be invisible and an operator could read the area as searched."""
    from PIL import Image

    a = np.asarray(Image.open(coverage_dir / "coverage_body.png").convert("RGBA"))[..., 3]
    assert (a == 0).any(), "no transparent cells at all — the burial mask never reached the PNG"
    assert (a > 0).any(), "the whole raster is transparent"
    man = read_manifest(coverage_dir)
    cells = man["layers"][0]["stats"]["cannot_clear_cells"]
    assert int((a == 0).sum()) == cells, f"{(a == 0).sum()} transparent px vs {cells} cannot_clear cells"


def test_the_coverage_raster_is_geo_aligned_with_the_flown_track(flight, coverage_dir):
    """The swept extent must be the track plus half a footprint — no more, no less.

    This is the check that caught the oblique-gimbal bug: with a hand-rolled q_gimbal the raster ran 414 m
    south and 177 m west of the track and every return value still said "success".
    """
    from sightline.coverage.grid import bounds_latlon
    from sightline.coverage.presentation import SIM_RGB_4K

    _route, frames = flight
    man = read_manifest(coverage_dir)
    g = man["grid"]
    cell_m = g["cell_m"]
    from PIL import Image

    rgba = np.asarray(Image.open(coverage_dir / "coverage_body.png").convert("RGBA"))
    # POD 0 is the ramp's darkest stop; "swept" is anything above it, opaque.
    dark = np.array([s["rgb"] for s in man["ramp"]][0], dtype=int)
    swept = (np.abs(rgba[..., :3].astype(int) - dark).sum(axis=2) > 12) & (rgba[..., 3] > 0)
    assert swept.any(), "nothing at all was swept"
    rows, cols = np.nonzero(swept)  # image row 0 = north
    b = man["bounds"]
    lat_per_row = (b["north"] - b["south"]) / rgba.shape[0]
    lon_per_col = (b["east"] - b["west"]) / rgba.shape[1]
    north = b["north"] - rows.min() * lat_per_row
    south = b["north"] - (rows.max() + 1) * lat_per_row
    west = b["west"] + cols.min() * lon_per_col
    east = b["west"] + (cols.max() + 1) * lon_per_col

    lats = [t.lat for t in frames]
    lons = [t.lon for t in frames]
    m_lat = 110574.0
    m_lon = 111320.0 * math.cos(math.radians(sum(lats) / len(lats)))
    half_swath = SIM_RGB_4K.swath_m(frames[0].agl_m) / 2.0            # 42.6 m at 55 m AGL
    tol = half_swath + 2 * cell_m                                     # + grid quantisation
    over = {
        "south": (min(lats) - south) * m_lat,
        "north": (north - max(lats)) * m_lat,
        "west": (min(lons) - west) * m_lon,
        "east": (east - max(lons)) * m_lon,
    }
    for side, m in over.items():
        assert -cell_m <= m <= tol, f"raster overhangs the track by {m:.1f} m to the {side} (tol {tol:.1f} m)"


def test_the_route_json_becomes_a_plan_layer_without_losing_a_waypoint(flight):
    route, _frames = flight
    fc = route_to_plan_geojson(route)
    assert fc["product"] == ROUTE_PRODUCT and fc["pattern"] == "boustrophedon"
    wps = [f for f in fc["features"] if f["properties"]["kind"] == "waypoint"]
    assert len(wps) == len(route["waypoints"]) == route["totals"]["n_waypoints"]
    for f, w in zip(wps, route["waypoints"]):
        assert f["geometry"]["coordinates"] == [w["lon"], w["lat"]], "GeoJSON must be [lon, lat] (RFC 7946)"
        assert f["properties"]["seq"] == w["seq"]
        assert f["properties"]["reason"] == w["reason"], "the reason a leg exists must reach the operator"
    line = [f for f in fc["features"] if f["properties"]["kind"] == "plan"][0]
    assert len(line["geometry"]["coordinates"]) == len(route["waypoints"])


@pytest.mark.parametrize("bad", [
    {"product": "not.a.route", "waypoints": [{"lat": 1, "lon": 2}]},
    {"product": ROUTE_PRODUCT, "waypoints": []},
])
def test_a_payload_that_is_not_a_route_is_refused_not_silently_dropped(bad):
    with pytest.raises(ValueError):
        route_to_plan_geojson(bad)


def test_the_drawn_camera_footprint_matches_the_projected_one(flight):
    """The map draws the same quadrilateral the coverage raster credits, not a nadir approximation."""
    from sightline.api.mission_feed import camera_footprint, nadir_footprint
    from sightline.coverage.presentation import SIM_RGB_4K

    _route, frames = flight
    intr = SIM_RGB_4K.intrinsics()
    t = frames[5]
    ring = camera_footprint(t, intr)
    assert ring and len(ring) == 5 and ring[0] == ring[-1]
    m_lat, m_lon = 110574.0, 111320.0 * math.cos(math.radians(t.lat))
    n_m = [(c[1] - t.lat) * m_lat for c in ring[:4]]
    e_m = [(c[0] - t.lon) * m_lon for c in ring[:4]]
    across = max(e_m) - min(e_m)
    along = max(n_m) - min(n_m)
    # gimbal yaw = heading + 90 puts the WIDE axis across track; the leg flies north, so across = east-west
    assert across == pytest.approx(SIM_RGB_4K.swath_m(t.agl_m), rel=0.02), f"across-track {across:.1f} m"
    assert along == pytest.approx(SIM_RGB_4K.swath_m(t.agl_m) * intr.height_px / intr.width_px, rel=0.02)
    # the nadir fallback ignores the gimbal yaw and is therefore 90 deg out — documented, not hidden
    flat = nadir_footprint(t.lat, t.lon, t.agl_m, intr, 0.0)
    flat_across = (max(c[0] for c in flat) - min(c[0] for c in flat)) * m_lon
    assert flat_across == pytest.approx(SIM_RGB_4K.swath_m(t.agl_m), rel=0.02)


# ==========================================================================================================
# 2. GUARDRAIL R10 — no delete, no "cleared", anywhere in this lane
# ==========================================================================================================
def test_no_delete_path_in_this_lanes_python_or_web_source():
    banned = re.compile(
        r"\bDELETE\s+FROM\b|\bDROP\s+(TABLE|INDEX|TRIGGER|VIEW|DATABASE)\b|\bTRUNCATE\b"
        r"|\bos\.remove\s*\(|\bos\.unlink\s*\(|\bshutil\.rmtree\s*\(|\.unlink\s*\("
        r"|@app\.delete\b|method\s*:\s*[\"']DELETE[\"']",
        re.IGNORECASE,
    )
    assert LANE_PY and LANE_WEB, "the lane source scan found no files — the globs are wrong"
    offenders = []
    for p in LANE_PY + LANE_WEB:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "R10 violation in lane B5 source:\n" + "\n".join(offenders)


def test_this_lane_never_says_a_segment_is_cleared():
    """R10 in words as well as in code: no UI string may tell an operator an area is cleared/complete."""
    banned = re.compile(r"\b(cleared|all\s+clear|search\s+complete|area\s+complete)\b", re.IGNORECASE)
    # The word may appear only under a negation ("never cleared", "cannot clear"), and the negation has to be
    # close enough to actually govern it — a paragraph that says "no" somewhere does not license the claim.
    negated = re.compile(r"\b(never|cannot|can't|not|no|nothing|without)\b", re.IGNORECASE)
    offenders = []
    for p in LANE_PY + LANE_WEB:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in banned.finditer(line):
                if not negated.search(line[max(0, m.start() - 60):m.start()]):
                    offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "R10 wording violation:\n" + "\n".join(offenders)


def test_the_app_exposes_no_delete_route(app_client):
    routes = app_client.app.routes
    for r in routes:
        assert "DELETE" not in getattr(r, "methods", set()) or (), f"{r.path} accepts DELETE"


def test_dismiss_requires_a_reason_and_the_record_survives_it(app_client, seeded):
    rid = seeded.records()[0].record_id
    before = seeded.get(rid)
    r = app_client.post(f"/api/records/{rid}/dismiss", json={"reason": "   "})
    assert r.status_code == 422 and "R10" in r.json()["detail"]
    assert seeded.get(rid).status == before.status, "a refused dismissal must change nothing"

    r = app_client.post(f"/api/records/{rid}/dismiss",
                        json={"reason": "reviewed on the ground: it is a mannequin", "by": "commander"})
    assert r.status_code == 200
    f = r.json()["feature"]
    assert f["properties"]["status"] == "dismissed"
    assert f["properties"]["dismissed_reason"] == "reviewed on the ground: it is a mannequin"
    assert f["properties"]["version"] == before.version + 1

    # the row, its history and its audit trail are all still there
    assert app_client.get(f"/api/records/{rid}").status_code == 200
    hist = app_client.get(f"/api/records/{rid}/history").json()["versions"]
    assert len(hist) >= 2 and hist[0]["properties"]["status"] != "dismissed"
    audit = app_client.get(f"/api/records/{rid}/audit").json()["audit"]
    assert any(a["action"] == "dismiss" and a["reason"] for a in audit)
    ids = [x["id"] for x in app_client.get("/api/records.geojson").json()["features"]]
    assert rid in ids, "a dismissed record must still be served, not filtered out of existence"


def test_dismissed_records_can_be_excluded_from_a_query_without_being_lost(app_client, seeded):
    rid = seeded.records()[0].record_id
    app_client.post(f"/api/records/{rid}/dismiss", json={"reason": "duplicate"})
    visible = app_client.get("/api/records.geojson", params={"include_dismissed": False}).json()["features"]
    assert rid not in [f["id"] for f in visible]
    allf = app_client.get("/api/records.geojson").json()["features"]
    assert rid in [f["id"] for f in allf]


# ==========================================================================================================
# 3. what the map fetches
# ==========================================================================================================
def test_records_geojson_is_rfc7946_ranked_and_carries_every_score_component(app_client):
    fc = app_client.get("/api/records.geojson").json()
    assert fc["type"] == "FeatureCollection" and fc["schema_version"] == SCHEMA_VERSION
    assert len(fc["features"]) == 8
    ranks = [f["properties"]["priority_rank"] for f in fc["features"]]
    assert ranks == sorted(ranks), f"the list is not in rank order: {ranks}"
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        assert 76.0 < lon < 76.3 and 11.3 < lat < 11.7, f"[lon, lat] order broken: {lon},{lat}"
        c = f["properties"]["score_components"]
        for k in ("p_living", "w_class", "urgency", "count_bonus", "urgency_class"):
            assert k in c, f"{k} missing: the score would be shown as a bare number (§5.8 forbids it)"
        assert f["properties"]["score"] == pytest.approx(
            c["p_living"] * c["w_class"] * c["urgency"] * c["count_bonus"], rel=1e-6), "score != its parts"
        # CE90 = 2.1460 * h_acc (schemas.CE90_FACTOR), served rounded to centimetres
        assert f["properties"]["ce90_m"] == pytest.approx(2.1460 * f["properties"]["h_acc_m"], abs=0.005)
        assert f["properties"]["source"]["domain"] == "sim", "an unlabelled number reached the map"
        assert f["properties"]["evidence"][0]["thumb_uri"].startswith("/api/thumbs/")


def test_every_evidence_thumbnail_actually_downloads(app_client):
    fc = app_client.get("/api/records.geojson").json()
    n = 0
    for f in fc["features"]:
        for ev in f["properties"]["evidence"]:
            r = app_client.get(ev["thumb_uri"])
            assert r.status_code == 200 and r.content[:2] == b"\xff\xd8", f"{ev['thumb_uri']} is not a JPEG"
            n += 1
    assert n == 8


def test_a_thumbnail_name_cannot_escape_the_thumbs_directory(app_client):
    for name in ("..%2F..%2Fsecrets.txt", "..", ".env"):
        assert app_client.get(f"/api/thumbs/{name}").status_code in (404, 400)


def test_the_coverage_routes_serve_the_layer_that_was_asked_for(app_client, coverage_dir):
    side = app_client.get("/api/coverage/overlay.json").json()
    assert side["available"] is True and side["source"] == COVERAGE_PRODUCT
    for lay in side["layers"]:
        img = app_client.get(f"/api/coverage/raster.png?layer={lay['layer']}")
        assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert img.content == (coverage_dir / f"coverage_{lay['layer']}.png").read_bytes()
        gj = app_client.get(f"/api/coverage/bands.geojson?layer={lay['layer']}").json()
        assert gj["type"] == "FeatureCollection" and gj["features"]
    assert app_client.get("/api/coverage/raster.png?layer=nonesuch").status_code == 404


def test_the_mission_feed_carries_drone_footprint_track_and_plan(app_client, flight):
    route, _ = flight
    m = app_client.get("/api/mission").json()
    assert m["drone"]["geometry"]["type"] == "Point"
    assert m["drone"]["properties"]["mode"] in ("AUTO", "MANUAL", "HOLD", "RTL")
    assert m["drone"]["properties"]["gimbal_pitch_deg"] == pytest.approx(-90.0, abs=0.5), \
        "the survey camera is nadir; if this reads -180 the gimbal quaternion convention is wrong"
    assert m["footprint"]["geometry"]["type"] == "Polygon"
    modes = {f["properties"]["mode"] for f in m["track"]["features"]}
    assert "MANUAL" in modes, "the logged operator takeover is not distinguishable on the track (§7 step 3)"
    assert len([f for f in m["plan"]["features"] if f["properties"]["kind"] == "waypoint"]) == \
        len(route["waypoints"])


def test_the_plan_route_is_served_verbatim(app_client, flight):
    route, _ = flight
    assert app_client.get("/api/plan/route.json").json() == route


def test_a_route_can_be_pushed_in_and_a_bad_one_is_refused(app_client, flight):
    route, _ = flight
    assert app_client.post("/api/plan/route", json=route).json()["waypoints"] == len(route["waypoints"])
    assert app_client.post("/api/plan/route", json={"product": "nope"}).status_code == 422


def test_health_names_every_contract_it_speaks(app_client):
    h = app_client.get("/health").json()
    assert h["ok"] and h["schema_version"] == SCHEMA_VERSION
    assert h["contracts"]["coverage"] == COVERAGE_PRODUCT
    assert h["contracts"]["route"] == ROUTE_PRODUCT
    assert h["detector"]["is_stub"] is True, "the cloud detector is a STUB and must say so"


# ==========================================================================================================
# 4. the live feed
# ==========================================================================================================
def test_the_envelope_is_the_documented_shape():
    m = envelope("record", 7, feature={})
    assert set(m) >= {"type", "seq", "t_utc", "schema_version"}
    assert m["schema_version"] == SCHEMA_VERSION and m["seq"] == 7
    with pytest.raises(ValueError):
        envelope("delete", 1)          # R10: there is no delete op on the wire
    assert "delete" not in MESSAGE_TYPES


def test_a_client_gets_hello_then_a_snapshot_then_one_frame_per_record_update(app_client, seeded):
    with app_client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["records"] == 8
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and len(snap["records"]["features"]) == 8
        mission = ws.receive_json()
        assert mission["type"] == "mission"

        rid = seeded.records()[0].record_id
        app_client.post(f"/api/records/{rid}/note", json={"note": "dispatched to Team Bravo"})
        frame = ws.receive_json()
        assert frame["type"] == "record" and frame["op"] == "upsert" and frame["event"] == "note"
        assert frame["feature"]["id"] == rid
        assert "Team Bravo" in frame["feature"]["properties"]["notes"]
        assert frame["seq"] > snap["seq"], "seq must be monotonic so a client can detect a gap"
        assert len(json.dumps(frame).encode()) < MAX_MESSAGE_BYTES, "a live frame outgrew the field radio budget"

        ws.send_json({"type": "ping", "echo": 42})
        pong = ws.receive_json()
        assert pong["type"] == "pong" and pong["echo"] == 42
    # a dismissal arrives as an upsert, never as a delete
    with app_client.websocket_connect("/ws") as ws:
        ws.receive_json(), ws.receive_json(), ws.receive_json()
        app_client.post(f"/api/records/{seeded.records()[0].record_id}/dismiss", json={"reason": "duplicate"})
        frame = ws.receive_json()
        assert frame["type"] == "record" and frame["op"] == "upsert"
        assert frame["feature"]["properties"]["status"] == "dismissed"


def test_resync_replays_the_whole_collection(app_client):
    with app_client.websocket_connect("/ws") as ws:
        ws.receive_json(), ws.receive_json(), ws.receive_json()
        ws.send_json({"type": "resync"})
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and len(snap["records"]["features"]) == 8


# ==========================================================================================================
# 5. the offline story, end to end through the real app (§5.10)
# ==========================================================================================================
class _Link:
    """A transport that posts into the ASGI app, with a switch for "the link is down"."""

    def __init__(self, client: TestClient):
        self.client = client
        self.online = False
        self.calls: list[str] = []
        self.responses: list[dict] = []

    def __call__(self, job):
        self.calls.append(job["key"])
        if not self.online:
            raise ConnectionError("link down")
        r = self.client.post("/api/upload", json=job)
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        self.responses.append(r.json())
        return r.json()


def test_the_queue_rises_offline_drains_on_reconnect_and_the_upsert_is_idempotent(tmp_path, coverage_dir):
    # separate directories on purpose: sharing one thumbs/ would make the cloud "already have" every crop
    edge = RecordStore(tmp_path / "edge" / "records.db")
    cloud = RecordStore(tmp_path / "cloud" / "records.db")
    outbox = Outbox(tmp_path / "outbox")
    app = create_app(edge, outbox=outbox, upload_store=cloud, coverage_dir=coverage_dir, serve_static=False)
    try:
        with TestClient(app) as client:
            link = _Link(client)
            up = Uploader(outbox, link, base_backoff_s=0.001, max_backoff_s=0.004, jitter=0.0)

            # --- LINK DOWN: the pipeline writes locally and enqueues; nothing blocks ------------------
            recs = seed_store(edge)
            for rec in recs:
                outbox.enqueue_record(rec)
                if rec.evidence:
                    p = edge.thumbs_dir / Path(rec.evidence[0].thumb_uri).name
                    outbox.enqueue_thumb(rec.record_id, rec.version, "SIM_FLOODVALLEY_001", p)
            depth_after_seed = client.get("/api/outbox").json()["depth"]
            assert depth_after_seed == 16, f"8 records + 8 thumbs should be queued, got {depth_after_seed}"

            for _ in range(20):
                up.step()
                time.sleep(0.005)
            assert up.online is False and up.sent == 0
            assert client.get("/api/outbox").json()["depth"] == 16, "a failed upload must not lose the job"
            assert cloud.stats()["records"] == 0, "nothing reached the cloud while the link was down"

            # an operator action while offline also queues, and the local log is authoritative
            rid = recs[0].record_id
            client.post(f"/api/records/{rid}/dismiss", json={"reason": "offline review: duplicate"})
            assert client.get("/api/outbox").json()["depth"] == 17
            assert edge.get(rid).status == "dismissed"

            # --- LINK UP ------------------------------------------------------------------------------
            link.online = True
            assert up.drain(timeout_s=20.0), f"the queue never drained: {outbox.stats()}"
            assert client.get("/api/outbox").json()["depth"] == 0
            assert up.online is True and up.sent == 17
            assert cloud.stats()["records"] == 8, cloud.stats()
            assert cloud.get(rid).status == "dismissed"
            assert cloud.get(rid).dismissed_reason == "offline review: duplicate"
            assert (cloud.thumbs_dir / f"{recs[1].record_id}_v1.jpg").is_file(), "the evidence crop never landed"

            # --- AT-LEAST-ONCE: the same job delivered twice must change nothing ------------------------
            applied = [r for r in link.responses if r.get("applied")]
            assert len(applied) == 17
            job = {"key": f"record:SIM_FLOODVALLEY_001:{recs[1].record_id}:1", "kind": "record",
                   "payload": {"feature": cloud.get(recs[1].record_id).to_feature()}}
            versions_before = cloud.stats()["versions"]
            again = client.post("/api/upload", json=job).json()
            assert again["applied"] is False and again["reason"] == "duplicate"
            assert cloud.stats()["versions"] == versions_before, "a replay created a new version"

            # a job whose payload is not a record is refused, not half-applied
            assert client.post("/api/upload", json={"key": "x", "kind": "record",
                                                    "payload": {"feature": {}}}).status_code == 422
            assert client.post("/api/upload", json={"key": "x", "kind": "wat"}).status_code == 422
            up.stop()
    finally:
        outbox.close()
        edge.close()
        cloud.close()


def test_the_wire_round_trip_keeps_every_field_a_commander_reads(seeded):
    for rec in seeded.records():
        back = feature_to_record(rec.to_feature())
        assert isinstance(back, Record)
        assert back.record_id == rec.record_id and back.status == rec.status
        assert back.components == rec.components
        assert back.evidence == rec.evidence
        assert back.lat == pytest.approx(rec.lat, abs=1e-6)   # to_feature() rounds to 6 dp by contract
        assert back.lon == pytest.approx(rec.lon, abs=1e-6)
        assert back.dismissed_reason == rec.dismissed_reason


# ==========================================================================================================
# 6. the map page itself (the parts a headless render cannot assert)
# ==========================================================================================================
def test_the_map_page_loads_nothing_from_the_network():
    """Static scan; `app/map/headless_check.mjs` is the dynamic proof (it fails on one external request)."""
    bad = re.compile(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+", re.IGNORECASE)
    for p in LANE_WEB:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in bad.finditer(line):
                url = m.group(0)
                # attribution links are hrefs a human clicks, never fetched by the page
                assert "href=" in line or "protomaps.com" in url or "openstreetmap.org" in url \
                    or line.strip().startswith("//"), f"{p.name}:{i} fetches {url}"


def test_the_map_page_encodes_the_section_5_9_layer_rules():
    html = (REPO / "app" / "map" / "index.html").read_text(encoding="utf-8")
    assert "priority_rank" in html and "circle-radius" in html, "marker size must be driven by rank"
    assert '"cls"' in html and "CLASS_COLOR" in html, "marker colour must be driven by class"
    assert "h_acc_m" in html and "circlePolygon" in html, "the accuracy ring must be metre-true"
    assert "fill-pattern" in html and "hatch" in html, "cannot-clear must be hatched, not just tinted"
    assert "score_components" in html and "score = P(living)" in html, \
        "the popup must show the score's parts and the formula, never the number alone"
    assert "R10" in html
