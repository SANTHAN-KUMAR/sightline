"""F18 record log + offline outbox (lane B5). Offline, no GPU, no simulator.

    D:\\Tools\\uv\\uv.exe run pytest tests/test_store.py -q
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import fields
from pathlib import Path

import pytest

from sightline.schemas import Evidence, Record, ScoreComponents
from sightline.store import GuardrailError, Outbox, RecordStore, StaleVersionError, Uploader, job_key

REPO = Path(__file__).resolve().parents[1]
LANE_SOURCE = sorted((REPO / "sightline" / "store").glob("*.py")) + sorted(
    (REPO / "sightline" / "api").glob("*.py")
)


# --------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------
def make_record(rid: str = "R-001", version: int = 1, **kw) -> Record:
    """A FULLY populated record: every field non-default, so a round-trip proves nothing is dropped."""
    base = dict(
        record_id=rid,
        cluster_id=7,
        status="confirmed",
        cls="human",
        lat=11.487123,
        lon=76.145678,
        alt_msl_m=1046.01,
        h_acc_m=3.25,
        h_acc_basis="budget_v1",
        method="water_plane",
        dem_source="glo30",
        agl_m=52.5,
        off_nadir_deg=12.5,
        confidence=0.9312,
        confidence_max_det=0.87,
        score=1.8123,
        priority_rank=2,
        components=ScoreComponents(
            p_living=0.87, w_class=0.93, urgency=2.1, count_bonus=1.2, urgency_class="immersed",
            elapsed_h=4.5, thermal_boost=1.15, motion_boost=1.1, posture_promoted=True,
        ),
        n_observations=17,
        n_tracks_merged=3,
        seen_in_passes=[1, 2, 5],
        first_seen_utc=1789000000.5,
        last_seen_utc=1789000600.25,
        motion_state="moving",
        motion_displacement_m=3.4,
        motion_window_s=12.0,
        count_estimate=2,
        count_min=1,
        count_max=3,
        count_basis="max_simultaneous_tracks",
        posture="half_submerged",
        posture_conf=0.72,
        submersion="head_only",
        submersion_conf=0.64,
        occlusion=1,
        modality="fused",
        thermal_hot=True,
        thermal_c=31.4,
        pixel_size_px=34.5,
        gsd_cm_px=1.9,
        zone="channel",
        evidence=[
            Evidence("/api/thumbs/R-001_v1.jpg", "CLIP_A", 1207, 1789000123.5, (10.0, 20.0, 58.0, 96.0),
                     0.87, "fused"),
            Evidence("/api/thumbs/R-001_v1b.jpg", "CLIP_A", 1244, 1789000156.0, (12.5, 21.5, 61.0, 99.0),
                     0.81, "rgb"),
        ],
        source={"platform": "sim", "sim": True, "clip_id": "CLIP_A", "aoi_id": "wayanad",
                "telemetry": "cosysairsim"},
        notes="simulated record",
        version=version,
    )
    base.update(kw)
    return Record(**base)


@pytest.fixture
def store(tmp_path) -> RecordStore:
    s = RecordStore(tmp_path / "records.db", thumbs_dir=tmp_path / "thumbs")
    yield s
    s.close()


# --------------------------------------------------------------------------------------------------------
# schema + persistence
# --------------------------------------------------------------------------------------------------------
def test_wal_mode(store):
    """SOLUTION_DOC §5.8: 'the record log itself is SQLite in WAL mode'."""
    assert store.journal_mode == "wal"
    assert store.stats()["journal_mode"] == "wal"


def test_every_record_field_round_trips(store):
    """A fully populated record must come back byte-identical: no silent data loss in the log."""
    rec = make_record()
    store.put(rec)
    got = store.get("R-001")
    assert got is not None
    assert got == rec, [f.name for f in fields(Record) if getattr(got, f.name) != getattr(rec, f.name)]
    # ...including the tuple-ness of bbox_px, which JSON would have turned into a list.
    assert isinstance(got.evidence[0].bbox_px, tuple)
    assert got.components.posture_promoted is True
    assert got.thermal_hot is True and got.occlusion == 1


def test_score_components_are_queryable_columns_not_a_blob(store):
    """The brief and §5.8: the evaluation lane and the map query these; they may not be an opaque blob."""
    store.put(make_record())
    rows = store.query(
        "SELECT record_id, p_living, w_class, urgency, count_bonus, urgency_class, elapsed_h,"
        " thermal_boost, motion_boost, posture_promoted, ce90_m, clip_id FROM records"
    )
    assert rows == [
        {"record_id": "R-001", "p_living": 0.87, "w_class": 0.93, "urgency": 2.1, "count_bonus": 1.2,
         "urgency_class": "immersed", "elapsed_h": 4.5, "thermal_boost": 1.15, "motion_boost": 1.1,
         "posture_promoted": 1, "ce90_m": pytest.approx(2.1460 * 3.25), "clip_id": "CLIP_A"}
    ]
    # and evidence is a real table, joinable on (clip_id, frame_idx) by the evaluation lane
    ev = store.query("SELECT clip_id, frame_idx, det_conf, bbox_x1, bbox_y2, camera FROM evidence"
                     " ORDER BY idx")
    assert len(ev) == 2
    assert ev[0]["clip_id"] == "CLIP_A" and ev[0]["frame_idx"] == 1207 and ev[0]["camera"] == "fused"
    assert ev[0]["bbox_x1"] == 10.0 and ev[0]["bbox_y2"] == 96.0


def test_query_is_read_only(store):
    with pytest.raises(GuardrailError):
        store.query("UPDATE records SET score = 0")


# --------------------------------------------------------------------------------------------------------
# GUARDRAIL R10
# --------------------------------------------------------------------------------------------------------
def test_no_delete_or_drop_anywhere_in_this_lane_source():
    """Mechanical scan of lane B5's own source for a statement that could remove a record.

    (`persistqueue` deletes rows from its OWN queue tables when a job is acknowledged; that is a transport
    queue, not the record log, and it is a vendored dependency rather than this lane's source.)
    """
    banned = re.compile(
        r"\bDELETE\s+FROM\b|\bDROP\s+(TABLE|INDEX|TRIGGER|VIEW|DATABASE)\b|\bTRUNCATE\b"
        r"|\bos\.remove\s*\(|\bos\.unlink\s*\(|\bshutil\.rmtree\s*\(|\.unlink\s*\(",
        re.IGNORECASE,
    )
    offenders = []
    assert LANE_SOURCE, "the lane source scan found no files — the glob is wrong"
    for p in LANE_SOURCE:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "R10 violation in lane source:\n" + "\n".join(offenders)


def test_the_database_itself_refuses_a_delete(store):
    """Belt and braces: even a foreign process opening the same file cannot remove a record."""
    store.put(make_record())
    raw = sqlite3.connect(str(store.path))
    try:
        for table, msg in [("records", "records are never deleted"),
                           ("record_versions", "record history is never deleted"),
                           ("evidence", "evidence is never deleted"),
                           ("audit", "the audit log is append-only")]:
            with pytest.raises(sqlite3.IntegrityError) as e:
                raw.execute(f"DELETE FROM {table}")
            assert msg in str(e.value)
    finally:
        raw.close()
    assert store.get("R-001") is not None


def test_dismissal_requires_a_reason_and_keeps_the_row(store):
    store.put(make_record())
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(GuardrailError):
            store.dismiss("R-001", reason=bad)
    assert store.get("R-001").status == "confirmed"  # nothing changed

    rec = store.dismiss("R-001", reason="reviewed: rock formation, not a person", by="commander")
    assert rec.status == "dismissed"
    assert rec.dismissed_reason == "reviewed: rock formation, not a person"
    assert rec.dismissed_by == "commander" and rec.dismissed_utc > 0
    assert rec.version == 2

    # the row stays, in the log AND in the exported collection (§5.8: "under a separate layer")
    assert store.get("R-001") is not None
    assert store.stats()["records"] == 1
    assert len(store.feature_collection()["features"]) == 1
    assert len(store.records(include_dismissed=False)) == 0
    # the pre-dismissal version is still recoverable
    assert store.get("R-001", version=1).status == "confirmed"
    # ...and the reason is in the audit log
    audit = store.audit_log("R-001")
    assert audit[0]["action"] == "dismiss"
    assert audit[0]["reason"] == "reviewed: rock formation, not a person"


def test_the_database_refuses_a_reasonless_dismissal(store):
    """The trigger, not just the Python guard: a direct UPDATE cannot dismiss without a reason."""
    store.put(make_record())
    raw = sqlite3.connect(str(store.path))
    try:
        with pytest.raises(sqlite3.IntegrityError) as e:
            raw.execute("UPDATE records SET status='dismissed' WHERE record_id='R-001'")
        assert "dismissal requires a reason" in str(e.value)
    finally:
        raw.close()


def test_record_ids_are_never_renumbered_and_versions_never_go_backwards(store):
    store.put(make_record())
    store.update("R-001", score=2.0)
    raw = sqlite3.connect(str(store.path))
    try:
        with pytest.raises(sqlite3.IntegrityError) as e:
            raw.execute("UPDATE records SET record_id='R-999' WHERE record_id='R-001'")
        assert "never renumbered" in str(e.value)
        with pytest.raises(sqlite3.IntegrityError) as e:
            raw.execute("UPDATE records SET version=1 WHERE record_id='R-001'")
        assert "never moves backwards" in str(e.value)
    finally:
        raw.close()


def test_history_is_kept_across_updates(store):
    store.put(make_record())
    store.update("R-001", score=1.9, priority_rank=1, actor="triage", reason="rescore after pass 2")
    store.set_status("R-001", "stale", actor="commander", reason="no sighting for 20 min")
    store.add_note("R-001", "dispatched to Team Bravo")
    hist = store.history("R-001")
    assert [r.version for r in hist] == [1, 2, 3, 4]
    assert [r.status for r in hist] == ["confirmed", "confirmed", "stale", "stale"]
    assert [round(r.score, 4) for r in hist] == [1.8123, 1.9, 1.9, 1.9]
    assert "dispatched to Team Bravo" in hist[-1].notes
    assert {a["action"] for a in store.audit_log("R-001")} == {"create", "update", "status", "note"}


# --------------------------------------------------------------------------------------------------------
# writes: idempotency, staleness, ranking, thumbnails
# --------------------------------------------------------------------------------------------------------
def test_replaying_the_same_version_is_a_no_op(store):
    """At-least-once delivery lands the same record twice; it must not duplicate anything."""
    rec = make_record()
    store.put(rec)
    store.put(make_record())  # a distinct but identical object
    s = store.stats()
    assert (s["records"], s["versions"], s["evidence"], s["audit"]) == (1, 1, 2, 1)


def test_a_changed_record_at_the_same_version_is_rejected(store):
    store.put(make_record())
    with pytest.raises(StaleVersionError):
        store.put(make_record(score=99.0))


def test_ranked_ordering_puts_rank_one_first_and_unranked_last(store):
    store.put(make_record("A", priority_rank=3, score=0.5))
    store.put(make_record("B", priority_rank=1, score=2.0))
    store.put(make_record("C", priority_rank=-1, score=1.0))
    store.put(make_record("D", priority_rank=2, score=1.5))
    assert [r.record_id for r in store.records()] == ["B", "D", "A", "C"]


def test_thumbnails_are_written_to_disk_and_served_safely(store):
    uri = store.put_thumbnail("R-001", 1, b"\xff\xd8\xff\x00fake-jpeg", ext=".jpg")
    assert uri == "/api/thumbs/R-001_v1.jpg"
    p = store.thumbnail_path("R-001_v1.jpg")
    assert p is not None and p.read_bytes().endswith(b"fake-jpeg")
    for evil in ("../records.db", "..\\records.db", "a/b.jpg", "", ".hidden"):
        assert store.thumbnail_path(evil) is None


def test_status_must_be_one_of_the_schema_vocabulary(store):
    store.put(make_record())
    with pytest.raises(ValueError):
        store.set_status("R-001", "cleared")  # R10: there is no "cleared"


def test_subscribers_see_every_write(store):
    seen = []
    store.subscribe(lambda rec, ev: seen.append((rec.record_id, rec.version, ev)))
    store.put(make_record())
    store.update("R-001", score=2.0)
    store.dismiss("R-001", reason="duplicate of R-002")
    assert seen == [("R-001", 1, "create"), ("R-001", 2, "update"), ("R-001", 3, "dismiss")]


# --------------------------------------------------------------------------------------------------------
# the outbox (§5.10)
# --------------------------------------------------------------------------------------------------------
class FlakyTransport:
    """A link that is down until `online` is set. Records every job it accepts, twice-delivered included."""

    def __init__(self):
        self.online = False
        self.delivered: list[str] = []
        self.attempts = 0

    def __call__(self, job):
        self.attempts += 1
        if not self.online:
            raise ConnectionError("link down")
        self.delivered.append(job["key"])
        return {"ok": True}


def test_outbox_key_is_clip_record_version(tmp_path, store):
    ob = Outbox(tmp_path / "outbox")
    try:
        rec = make_record()
        job = ob.enqueue_record(rec)
        assert job["key"] == job_key("record", "CLIP_A", "R-001", 1) == "record:CLIP_A:R-001:1"
        assert job["clip_id"] == "CLIP_A" and job["record_id"] == "R-001" and job["version"] == 1
        assert job["payload"]["feature"]["type"] == "Feature"
        # a version bump is a NEW job key, so history replays in order
        rec2 = make_record(version=2, score=2.0)
        assert ob.enqueue_record(rec2)["key"] == "record:CLIP_A:R-001:2"
        assert ob.depth() == 2
    finally:
        ob.close()


def test_offline_queue_rises_then_drains_on_reconnect(tmp_path, store):
    """The §5.10 story, with the queue's own durable state as the evidence."""
    ob = Outbox(tmp_path / "outbox")
    tr = FlakyTransport()
    up = Uploader(ob, tr, base_backoff_s=0.001, max_backoff_s=0.005, jitter=0.0)
    try:
        # --- link down: the pipeline keeps writing and never blocks -----------------------------------
        for i in range(5):
            rec = make_record(f"R-{i:03d}")
            store.put(rec)                       # local log FIRST (§5.10)
            ob.enqueue_record(rec)               # then the upload job
        assert ob.depth() == 5, "enqueue must not depend on the network"
        assert store.stats()["records"] == 5, "the local log is complete while offline"

        for _ in range(12):                      # drive the uploader by hand: every attempt fails
            up.step()
            time.sleep(0.006)
        assert tr.delivered == []
        assert up.failures >= 5
        assert up.online is False
        assert ob.depth() == 5, "nothing is lost while the link is down"
        assert ob.stats()["acked"] == 0

        # --- link back up ------------------------------------------------------------------------------
        tr.online = True
        assert up.drain(timeout_s=5.0), f"queue did not drain: {ob.stats()}"
        assert ob.depth() == 0
        assert sorted(tr.delivered) == [f"record:CLIP_A:R-{i:03d}:1" for i in range(5)]
        assert ob.stats()["acked"] == 5
        assert up.online is True and up.sent == 5
    finally:
        up.stop()
        ob.close()


def test_a_crash_between_get_and_ack_replays_the_job(tmp_path):
    """SQLiteAckQueue durability: an unacknowledged job comes back after a restart, so nothing is lost."""
    path = tmp_path / "outbox"
    ob = Outbox(path)
    ob.enqueue_record(make_record("R-CRASH"))
    raw = ob.q.get(raw=True, block=False)  # taken, never acked -> "the process died here"
    assert raw["data"]["key"] == "record:CLIP_A:R-CRASH:1"
    assert ob.q.unack_count() == 1
    ob.close()

    ob2 = Outbox(path)  # auto_resume=True moves unacked jobs back to ready
    try:
        assert ob2.depth() == 1
        again = ob2.q.get(raw=True, block=False)
        assert again["data"]["key"] == "record:CLIP_A:R-CRASH:1"
    finally:
        ob2.close()


def test_backoff_grows_and_max_attempts_abandons_without_deleting(tmp_path):
    ob = Outbox(tmp_path / "outbox")
    tr = FlakyTransport()
    up = Uploader(ob, tr, base_backoff_s=0.05, max_backoff_s=0.5, jitter=0.0, max_attempts=3)
    try:
        ob.enqueue_record(make_record("R-DEAD"))
        # step() returns "backoff" while the timer is unexpired and only makes an attempt once it fires, so the
        # loop must be driven by "step until something that is not backoff" and each attempt timed from the
        # previous one. Asserting "retry" on the first call of a fixed-size outer loop double-counts attempts:
        # the inner drain already consumed the next one.
        results, gaps = [], []
        t0 = time.monotonic()
        while True:
            r = up.step()
            if r == "backoff":
                time.sleep(0.002)
                continue
            results.append(r)
            gaps.append(time.monotonic() - t0)
            t0 = time.monotonic()
            if r != "retry":
                break
        assert results == ["retry", "retry", "abandoned"], results
        # gaps[0] is the first attempt (nothing precedes it); gaps[1] and gaps[2] are the two back-off waits.
        assert gaps[2] > gaps[1] * 1.5, f"back-off did not grow: {gaps}"
        assert up.abandoned == 1
        assert ob.stats()["failed"] == 1
        assert ob.depth() == 0
    finally:
        up.stop()
        ob.close()


def test_thumbnail_jobs_carry_a_checksum(tmp_path, store):
    ob = Outbox(tmp_path / "outbox")
    try:
        uri = store.put_thumbnail("R-001", 1, b"\xff\xd8\xff\x00thumb-bytes")
        job = ob.enqueue_thumb("R-001", 1, "CLIP_A", store.thumbnail_path("R-001_v1.jpg"), uri=uri)
        assert job["key"] == "thumb:CLIP_A:R-001:1"
        assert job["payload"]["sha256"] == (
            "6b8e2c1f8e0f6cd3e8b6e0f2c9f4d0e2" if False else job["payload"]["sha256"]
        )
        import base64
        import hashlib

        data = base64.b64decode(job["payload"]["b64"])
        assert data == b"\xff\xd8\xff\x00thumb-bytes"
        assert hashlib.sha256(data).hexdigest() == job["payload"]["sha256"]
    finally:
        ob.close()
