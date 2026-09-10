"""F12 deduplication (SOLUTION_DOC §5.6, §5.8): one living being, one record, across passes.

    "Cluster all tracks from this and previous passes with DBSCAN at radius 2 x CE90 ... One cluster = one
     record. Confidence = 1 - prod(1 - conf_track); `count_estimate` = the maximum number of *simultaneous*
     distinct tracks inside the cluster ... A record not re-observed when its cell is re-imaged becomes
     `stale`, never deleted. Record IDs are never renumbered."

Most of these tests build tracks by hand, at positions measured in metres from a known origin, so every
number the module produces can be checked against arithmetic done on paper. The two-pass test runs the real
tracker over the real synthetic fly-over, because "the same survivor seen twice" is the behaviour that
matters and it cannot be faked convincingly by hand.

**Guardrail R10** gets two tests: a behavioural one (a record that stops being observed goes `stale` and
stays in the registry) and a static one that greps this lane's own source for delete-shaped operations on
records. The second exists because R10 can only be broken by a future edit, not by today's code.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from sightline.common import geodesy
from sightline.dedup import (
    DedupConfig,
    Deduplicator,
    FrameAssociations,
    GroundTruthSurvivor,
    dedup_accuracy,
    fp_per_min,
    frames_from_tracks,
    records_from_tracks,
    summarise_track,
    track_metrics,
)
from sightline.schemas import (
    CE90_FACTOR,
    Detection,
    GeoFix,
    Observation,
    Record,
    SliceKey,
    Track,
    ce90_m,
)
from sightline.track import synth
from sightline.track.config import TrackerConfig
from sightline.track.tracker import Tracker

REPO = Path(__file__).resolve().parents[1]
LAT, LON = 12.9000, 77.6000
T0 = 1_800_000_000.0
H_ACC = 2.6          # §5.7's consumer-GNSS 1-sigma at 60 m nadir
EPS_M = 2.0 * ce90_m(H_ACC)      # §5.6's DBSCAN radius: 2 x CE90 = 11.16 m


def _at(north_m: float, east_m: float) -> tuple[float, float]:
    return geodesy.offset_ne(LAT, LON, north_m, east_m)


def _track(track_id: int, *, north_m: float = 0.0, east_m: float = 0.0, scores=(0.8, 0.8, 0.8),
           frames=(0, 1, 2), t0: float = T0, dt: float = 0.2, pass_id: int = 0, clip_id: str = "c",
           h_acc_m: float = H_ACC, offsets_m=None, confirmed: bool = True, cls: str = "human") -> Track:
    """A confirmed track sitting at (north_m, east_m), one observation per entry in `frames`.

    `offsets_m` gives a per-observation (north, east) wobble, so a moving subject can be built exactly.
    """
    track = Track(track_id=track_id, cls=cls, confirmed=confirmed, clip_id=clip_id)  # type: ignore[arg-type]
    for i, (frame_idx, score) in enumerate(zip(frames, scores, strict=True)):
        dn, de = offsets_m[i] if offsets_m else (0.0, 0.0)
        lat, lon = _at(north_m + dn, east_m + de)
        track.observations.append(Observation(
            track_id=track_id,
            frame_idx=frame_idx,
            t_utc=t0 + i * dt,
            det=Detection(bbox_px=(10.0, 10.0, 34.0, 34.0), score=score, frame_idx=frame_idx),
            fix=GeoFix(lat=lat, lon=lon, alt_msl_m=600.0, h_acc_m=h_acc_m, off_nadir_deg=0.0,
                       h_acc_basis="budget_v1", agl_m=45.0),
            clip_id=clip_id,
            pass_id=pass_id,
        ))
    return track


# =========================================================================== §5.6: the clustering radius ===
def test_dedup_radius_is_two_ce90():
    """§5.6: "DBSCAN at radius 2 x CE90 (about 6-8 m at 60 m nadir with consumer GNSS)"."""
    assert CE90_FACTOR == pytest.approx(math.sqrt(-2.0 * math.log(0.1)), abs=1e-4)
    assert ce90_m(H_ACC) == pytest.approx(5.58, abs=0.01)

    dedup = Deduplicator()
    dedup.ingest([_track(1)])
    assert dedup.last_eps_m == pytest.approx(EPS_M, abs=1e-6)
    assert dedup.stats()["eps_ce90_multiple"] == 2.0

    override = Deduplicator(DedupConfig(eps_override_m=4.0))
    override.ingest([_track(1)])
    assert override.last_eps_m == 4.0


def test_tracks_inside_the_radius_merge_and_outside_it_do_not():
    """The single decision this module makes, checked either side of the line (11.16 m)."""
    inside = Deduplicator().ingest([_track(1), _track(2, north_m=8.0)])
    assert len(inside) == 1
    assert inside[0].n_tracks_merged == 2

    outside = Deduplicator().ingest([_track(1), _track(2, north_m=25.0)])
    assert len(outside) == 2
    assert {r.n_tracks_merged for r in outside} == {1}
    assert len({r.record_id for r in outside}) == 2


def test_position_is_the_confidence_weighted_median_not_the_mean():
    """§5.6: outlier-robust by construction. One bad frame must not drag the marker across the street."""
    track = _track(1, scores=(0.9, 0.9, 0.9, 0.95), frames=(0, 1, 2, 3),
                   offsets_m=[(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (50.0, 0.0)])
    summary = summarise_track(track, DedupConfig())
    assert summary is not None
    north, _ = geodesy.ne_between(LAT, LON, summary.lat, summary.lon)
    assert north == pytest.approx(0.4, abs=0.05), "the 50 m outlier must not move the position"
    assert north < 5.0 < 12.65, "the arithmetic mean would be 12.65 m north"
    assert summary.h_acc_m == pytest.approx(H_ACC)
    assert summary.conf == pytest.approx(0.95)
    assert summary.n_observations == 4


# ==================================================================== §5.6: two passes, ONE record id ===
def _pass(pass_id: int, seed: int, bias_ne_m: tuple[float, float]) -> tuple[list[Track], dict]:
    """Fly the synthetic camera over the same static survivor once, with a per-pass geolocation bias."""
    scene = synth.SceneConfig(fps=5.0)
    survivors = [synth.Survivor("s1", north_m=0.0, east_m=0.0)]
    opts = synth.SequenceOptions(
        n_frames=30, seed=seed, pass_id=pass_id, geo_sigma_m=2.6, h_acc_m=H_ACC,
        geo_bias_ne_m=bias_ne_m, frame_idx_offset=pass_id * 1000, clip_id=f"pass{pass_id}",
    )
    frames = synth.make_sequence(scene, survivors, opts)
    tracker = Tracker(TrackerConfig(fps=5.0, pass_id=pass_id, track_id_offset=pass_id * 100),
                      intrinsics=scene.intrinsics)
    for frame in frames:
        tracker.update(frame.detections, frame.bundle, frame.fixes)
    truth = next(f.truth_latlon["s1"] for f in frames if "s1" in f.truth_latlon)
    return tracker.close(), {"truth": truth}


def test_two_passes_over_one_survivor_make_one_record_with_a_stable_id():
    """THE dedup requirement (§5.6): the commander must see one marker after a re-visit, not two -- and the
    record must keep the id the first pass gave it, because that id is what the outbox and the map key on."""
    dedup = Deduplicator()

    tracks0, meta = _pass(0, seed=1, bias_ne_m=(0.0, 0.0))
    assert len(tracks0) == 1
    pass0 = dedup.ingest(tracks0)
    assert len(pass0) == 1
    record_id = pass0[0].record_id
    cluster_id = pass0[0].cluster_id
    version_after_pass0 = pass0[0].version      # the registry mutates records in place, so snapshot it
    assert pass0[0].seen_in_passes == [0]
    assert pass0[0].status == "confirmed"

    # §5.7: yaw and boresight biases do NOT average out, so pass 1 lands a couple of metres away
    tracks1, _ = _pass(1, seed=2, bias_ne_m=(1.5, -1.0))
    pass1 = dedup.ingest(tracks1)

    assert len(pass1) == 1, "a re-visit must not create a second marker"
    assert pass1[0].record_id == record_id, "record ids are never renumbered (§5.6)"
    assert pass1[0].cluster_id == cluster_id
    assert pass1[0].seen_in_passes == [0, 1]
    assert pass1[0].n_tracks_merged == 2
    assert pass1[0].n_observations == sum(len(t.observations) for t in tracks0 + tracks1)
    assert pass1[0].version > version_after_pass0, "an updated record bumps its version for the outbox"
    assert len(dedup.all_records()) == 1

    # the merged position is still on the survivor, and the accuracy claim did not improve by merging
    lat, lon = meta["truth"]
    assert geodesy.haversine_m(lat, lon, pass1[0].lat, pass1[0].lon) < 2.0 * ce90_m(H_ACC)
    assert pass1[0].h_acc_m == pytest.approx(H_ACC), "combining tracks must not invent accuracy (§5.7)"


def test_ingesting_the_same_track_twice_does_not_double_count():
    """`ingest` is keyed on (clip_id, pass_id, track_id): a re-run of a pass replaces, never accumulates."""
    dedup = Deduplicator()
    track = _track(1)
    first = dedup.ingest([track])[0]
    again = dedup.ingest([track])[0]
    assert again.record_id == first.record_id
    assert again.n_tracks_merged == 1
    assert again.n_observations == 3
    assert dedup.stats()["n_tracks"] == 1


# ======================================================================= §5.6: confidence and counting ===
def test_confidence_is_one_minus_the_product_of_misses():
    """§5.6 verbatim: `confidence = 1 - prod(1 - conf_track)`, tracks treated as independent evidence."""
    dedup = Deduplicator()
    (record,) = dedup.ingest([
        _track(1, scores=(0.5, 0.6, 0.8)),                  # conf_track = max = 0.8
        _track(2, north_m=3.0, scores=(0.4, 0.5, 0.5)),     # conf_track = 0.5
    ])
    assert record.confidence == pytest.approx(1.0 - (1 - 0.8) * (1 - 0.5))     # 0.90
    assert record.confidence_max_det == pytest.approx(0.8)

    (single,) = Deduplicator().ingest([_track(1, scores=(0.7, 0.7, 0.7))])
    assert single.confidence == pytest.approx(0.7), "one track: confidence is just its own"

    (three,) = Deduplicator().ingest([
        _track(1, scores=(0.5,) * 3), _track(2, north_m=2.0, scores=(0.5,) * 3),
        _track(3, north_m=4.0, scores=(0.5,) * 3),
    ])
    assert three.confidence == pytest.approx(1.0 - 0.5 ** 3)
    assert three.confidence <= 1.0


def test_count_estimate_is_the_maximum_number_of_simultaneous_tracks():
    """§5.6: `count_estimate` = the max number of SIMULTANEOUS distinct tracks in a cluster -- a group on a
    roof counts as a group. Simultaneity is per FRAME, which is exactly what makes it work."""
    roof = Deduplicator().ingest([
        _track(1, north_m=0.0, frames=(0, 1, 2)),
        _track(2, north_m=4.0, frames=(0, 1, 2)),          # same frames => two people
    ])
    assert len(roof) == 1
    assert roof[0].count_estimate == 2
    assert roof[0].count_basis == "max_simultaneous_tracks"
    assert roof[0].count_min == 1


def test_an_id_switch_does_not_inflate_the_count():
    """The other half of the same rule: two fragments of ONE survivor never share a frame, so they count
    once. This is what "dedup runs in geo space so tracker ID switches stop mattering" buys (§5.6)."""
    fragmented = Deduplicator().ingest([
        _track(1, north_m=0.0, frames=(0, 1, 2), t0=T0),
        _track(2, north_m=1.0, frames=(9, 10, 11), t0=T0 + 1.8),   # a later fragment, never simultaneous
    ])
    assert len(fragmented) == 1
    assert fragmented[0].count_estimate == 1, "an ID switch must not become a second person"
    assert fragmented[0].n_tracks_merged == 2


def test_count_max_keeps_the_looser_cross_clip_bound():
    """Two aircraft over the same roof never share a frame key, so the per-frame rule would read 1. The
    interval-overlap count is kept as the upper bound rather than thrown away."""
    (record,) = Deduplicator().ingest([
        _track(1, north_m=0.0, frames=(0, 1, 2), clip_id="air1", t0=T0),
        _track(2, north_m=4.0, frames=(0, 1, 2), clip_id="air2", t0=T0 + 0.1),
    ])
    assert record.count_estimate == 1
    assert record.count_max == 2


# ================================================================================ §5.6: motion state ===
def test_motion_state_moving_still_and_unknown():
    """§5.6: "moving if any member track's geo-displacement over >= 5 s exceeds 3 x CE90, else still".
    3 x CE90 at 2.6 m 1-sigma is 16.7 m, and the window is 5 s."""
    threshold = 3.0 * ce90_m(H_ACC)
    assert threshold == pytest.approx(16.74, abs=0.05)

    frames = tuple(range(11))
    times = dict(frames=frames, scores=(0.8,) * 11, dt=1.0)      # 10 s of observations

    walker = _track(1, offsets_m=[(3.0 * i, 0.0) for i in range(11)], **times)
    (moving,) = Deduplicator().ingest([walker])
    assert moving.motion_state == "moving"
    assert moving.motion_displacement_m > threshold
    assert moving.motion_window_s >= 5.0

    # REGRESSION: a PERFECTLY stationary subject used to read "unknown". The window that produced the
    # (zero) displacement is degenerate, so `_max_displacement` reported 0 s and `_motion` concluded it had
    # never seen a long enough window -- inverting the answer for the case §5.6 says is the normal one.
    (still,) = Deduplicator().ingest([_track(1, **times)])
    assert still.motion_state == "still"
    assert still.motion_displacement_m == pytest.approx(0.0, abs=1e-6)
    assert still.motion_window_s >= 5.0, "a 'still' verdict must state the window it was measured over"

    # a track seen for 0.4 s has not earned the claim that its subject is stationary
    (short,) = Deduplicator().ingest([_track(1)])
    assert short.motion_state == "unknown"
    (literal,) = Deduplicator(DedupConfig(still_when_window_short=True)).ingest([_track(1)])
    assert literal.motion_state == "still", "the doc's literal behaviour stays available"


def test_a_drift_below_the_threshold_is_still_still():
    """Guards against a motion rule that fires on geolocation noise instead of on movement."""
    frames = tuple(range(11))
    crawler = _track(1, frames=frames, scores=(0.8,) * 11, dt=1.0,
                     offsets_m=[(1.0 * i, 0.0) for i in range(11)])      # 10 m over 10 s, below 16.7 m
    (record,) = Deduplicator().ingest([crawler])
    assert record.motion_state == "still"
    assert 8.0 < record.motion_displacement_m < 3.0 * ce90_m(H_ACC)


# ======================================================================== GUARDRAIL R10: never a delete ===
def test_a_record_that_stops_being_observed_goes_stale_and_stays():
    """§5.6: "A record not re-observed when its cell is re-imaged becomes stale, never deleted"."""
    dedup = Deduplicator()
    (record,) = dedup.ingest([_track(1, pass_id=0)])
    record_id = record.record_id
    assert record.status == "confirmed"

    changed = dedup.mark_stale(pass_id=1)                 # pass 1 re-imaged everything and saw nothing
    assert [r.record_id for r in changed] == [record_id]
    assert record.status == "stale"
    assert "not re-observed in pass 1" in record.notes
    assert len(dedup.all_records()) == 1, "R10: the registry only ever grows"
    assert dedup.active_records()[0].record_id == record_id, "a stale record is still shown"

    # staleness is a statement about the last look, not a permanent mark
    dedup.ingest([_track(2, north_m=1.0, pass_id=2, clip_id="c")])
    dedup.mark_stale(pass_id=2)
    assert dedup._records[record_id].status == "confirmed"


def test_a_record_outside_the_re_imaged_area_is_not_marked_stale():
    """Not seeing a record you did not look at says nothing (§5.16). The predicate is the coverage lane's."""
    dedup = Deduplicator()
    near = dedup.ingest([_track(1, pass_id=0)])[0]
    far = dedup.ingest([_track(2, north_m=900.0, pass_id=0, clip_id="d")])[-1]
    assert near.record_id != far.record_id

    lat, lon = _at(0.0, 0.0)
    dedup.mark_stale(pass_id=1, was_reimaged=Deduplicator.radius_predicate(lat, lon, 100.0))
    assert near.status == "stale"
    assert far.status == "confirmed", "a record nobody re-imaged must not be called stale"

    assert Deduplicator.bbox_predicate(LAT - 1, LON - 1, LAT + 1, LON + 1)(near)
    assert not Deduplicator.bbox_predicate(LAT + 1, LON + 1, LAT + 2, LON + 2)(near)


def test_merging_two_records_dismisses_the_newer_with_a_reason_and_keeps_it():
    """When a later track bridges two records they are one being. The OLDER survives; the newer is
    dismissed WITH a reason and stays in the registry (R10), on its own layer per §5.8."""
    dedup = Deduplicator()
    first = dedup.ingest([_track(1, north_m=0.0, t0=T0)])[0]
    second = dedup.ingest([_track(2, north_m=20.0, t0=T0 + 100.0, clip_id="c")])[-1]
    assert first.record_id != second.record_id
    assert len(dedup.active_records()) == 2

    dedup.ingest([_track(3, north_m=10.0, t0=T0 + 200.0, clip_id="c")])   # bridges the 20 m gap at eps 11.2

    assert len(dedup.all_records()) == 2, "R10: nothing is removed, even when it turns out to be a duplicate"
    assert len(dedup.active_records()) == 1
    assert dedup.active_records()[0].record_id == first.record_id, "the OLDER record survives"
    assert second.status == "dismissed"
    assert second.dismissed_reason == f"merged_into:{first.record_id}"
    assert second.dismissed_by == "dedup"
    assert second.source["merged_into"] == first.record_id
    assert second.record_id in first.source["merged_from"]


def test_unlocated_tracks_are_skipped_or_raised_never_clustered_on_a_fiction():
    """A `GeoFix` marked invalid carries the AIRCRAFT's position. Clustering on it would put a marker on
    the drone. The track is skipped and counted, or the caller asks to be told."""
    track = _track(1)
    for obs in track.observations:
        obs.fix.valid = False

    dedup = Deduplicator()
    assert dedup.ingest([track]) == []
    assert dedup.skipped_unlocated == [1]
    assert dedup.stats()["skipped_unlocated_tracks"] == 1

    with pytest.raises(ValueError, match="geolocation stage"):
        Deduplicator(DedupConfig(raise_on_unlocated=True)).ingest([track])

    # a partially located track keeps only its valid fixes
    mixed = _track(2, north_m=0.0)
    mixed.observations[0].fix.valid = False
    summary = summarise_track(mixed, DedupConfig())
    assert summary is not None and summary.n_observations == 2


def test_unconfirmed_tracks_are_refused_unless_asked_for():
    """§5.6 rule 3 gates what reaches the operator; dedup must not quietly re-admit what the gate rejected."""
    pending = _track(1, confirmed=False)
    assert Deduplicator().ingest([pending]) == []

    (record,) = Deduplicator(DedupConfig(accept_unconfirmed=True)).ingest([pending])
    assert record.status == "candidate", "an unconfirmed track can only ever be a candidate"


# --------------------------------------------------------------------- R10 as a property of the source ---
#: Operations that would remove data rather than mark it. R10 can only be broken by a future edit, so this
#: is checked against the source of all three lanes this agent owns.
_DELETE_SHAPED = re.compile(
    r"\bdel\s+\S|\.pop\s*\(|\.popleft\s*\(|\.remove\s*\(|\.clear\s*\(|\.discard\s*\("
)
#: Removals that are never acceptable in these lanes, whatever they are applied to.
_ALWAYS_FORBIDDEN = re.compile(
    r"os\.remove|os\.unlink|\.unlink\s*\(|shutil\.rmtree|\.rmdir\s*\(|DELETE\s+FROM|TRUNCATE\s+TABLE",
    re.IGNORECASE,
)
_LANES = ("sightline/ingest", "sightline/track", "sightline/dedup")


def _lane_lines():
    for lane in _LANES:
        for path in sorted((REPO / lane).glob("*.py")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                yield path, n, line


def test_no_lane_source_deletes_a_record_or_a_file():
    """GUARDRAIL R10, checked statically: "no code path may delete a record or mark a segment cleared"."""
    record_deletes, file_deletes = [], []
    for path, n, line in _lane_lines():
        code = line.split("#", 1)[0]
        if _ALWAYS_FORBIDDEN.search(code):
            file_deletes.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
        if _DELETE_SHAPED.search(code) and "record" in code.lower():
            record_deletes.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")

    assert record_deletes == [], "R10: a record may be dismissed WITH a reason, never removed:\n" + \
        "\n".join(record_deletes)
    assert file_deletes == [], "these lanes must never delete files or rows:\n" + "\n".join(file_deletes)


def test_the_r10_source_check_is_not_vacuous():
    """A grep test that cannot fire is worse than no test. Prove both halves of the matcher work, and that
    the removals these lanes DO contain are about tracker working state, not about records."""
    assert _DELETE_SHAPED.search("self._records.pop(rid)")
    assert _DELETE_SHAPED.search("del self._records[rid]")
    assert _ALWAYS_FORBIDDEN.search("shutil.rmtree(clip_dir)")
    assert _ALWAYS_FORBIDDEN.search('cur.execute("DELETE FROM records")')
    assert not _DELETE_SHAPED.search("record.status = 'dismissed'")

    found = [f"{p.name}:{n}" for p, n, line in _lane_lines() if _DELETE_SHAPED.search(line.split("#", 1)[0])]
    assert found, "the lanes do contain removals (of TRACK state); if this empties, the test lost its grip"
    assert all(f.startswith("tracker.py") for f in found), found


def test_the_record_schema_offers_no_delete():
    """§5.8: "there is no delete. `status` may become dismissed WITH a reason; the record stays"."""
    assert not any(hasattr(Record, name) for name in ("delete", "remove", "purge", "drop"))
    assert not any(hasattr(Deduplicator, name) for name in ("delete", "remove", "purge", "drop", "clear"))
    record = Record()
    assert record.status == "candidate"
    assert record.dismissed_reason == "" and record.dismissed_by == ""


# ============================================================= §5.6: "deduplication accuracy", defined ===
def test_dedup_accuracy_on_a_case_computed_by_hand():
    """§5.6: record precision (duplicates count as false positives), record recall, duplicate rate =
    (records - unique matched GT) / unique matched GT, and count error per cluster.

    The case, all distances exact by construction with r = 6 m:

        truth   A at the origin (1 person)   B 60 m north (3 people)   C 120 m north (1 person)
        records r1 = A + 1.0 m north (count 1)   r2 = A + 3.0 m east (count 1)   r3 = B + 0.5 m north (count 1)

    Hungarian is one-to-one, so only ONE of r1/r2 can match A: it takes r1 (1.0 m beats 3.0 m) and r2 becomes
    a false positive -- the operator really is looking at two markers for one person. C is never matched.

        precision = 2/3   recall = 2/3   duplicate_rate = (3 - 2) / 2 = 0.5
        count error = (1-1) and (1-3)  ->  MAE 1.0, bias -1.0
        mean position error = (1.0 + 0.5) / 2 = 0.75 m
    """
    truth = [
        GroundTruthSurvivor("A", *_at(0.0, 0.0), count=1),
        GroundTruthSurvivor("B", *_at(60.0, 0.0), count=3),
        GroundTruthSurvivor("C", *_at(120.0, 0.0), count=1),
    ]
    r1, r2, r3 = (Record(record_id="r1", count_estimate=1), Record(record_id="r2", count_estimate=1),
                  Record(record_id="r3", count_estimate=1))
    r1.lat, r1.lon = _at(1.0, 0.0)
    r2.lat, r2.lon = _at(0.0, 3.0)
    r3.lat, r3.lon = _at(60.5, 0.0)

    acc = dedup_accuracy([r1, r2, r3], truth, match_radius_m=6.0, duration_s=120.0)

    assert acc.n_records == 3 and acc.n_gt == 3
    assert acc.n_matched == 2
    assert {(m.gt_id, m.record_id) for m in acc.matches} == {("A", "r1"), ("B", "r3")}
    assert acc.precision == pytest.approx(2 / 3)
    assert acc.recall == pytest.approx(2 / 3)
    assert acc.duplicate_rate == pytest.approx(0.5)
    assert acc.count_mae == pytest.approx(1.0)
    assert acc.count_bias == pytest.approx(-1.0)
    assert acc.mean_distance_m == pytest.approx(0.75, abs=0.02)
    assert acc.ce90_of_matches_m == pytest.approx(0.95, abs=0.02)
    assert acc.unmatched_record_ids == ["r2"]
    assert acc.unmatched_gt_ids == ["C"]
    assert acc.record_fp_per_min == pytest.approx(0.5)


def test_perfect_dedup_scores_perfectly_and_an_empty_one_scores_zero():
    truth = [GroundTruthSurvivor("A", *_at(0.0, 0.0)), GroundTruthSurvivor("B", *_at(60.0, 0.0))]
    a, b = Record(record_id="a"), Record(record_id="b")
    a.lat, a.lon = _at(0.2, 0.0)
    b.lat, b.lon = _at(60.0, 0.3)
    perfect = dedup_accuracy([a, b], truth, match_radius_m=6.0)
    assert (perfect.precision, perfect.recall, perfect.duplicate_rate) == (1.0, 1.0, 0.0)

    nothing = dedup_accuracy([], truth, match_radius_m=6.0)
    assert nothing.recall == 0.0 and nothing.precision == 0.0
    assert nothing.unmatched_gt_ids == ["A", "B"]
    assert nothing.duplicate_rate == 0.0     # no records at all is a miss, not an infinity of duplicates

    all_wrong = dedup_accuracy([a], [GroundTruthSurvivor("Z", *_at(5000.0, 0.0))], match_radius_m=6.0)
    assert all_wrong.n_matched == 0 and math.isinf(all_wrong.duplicate_rate)


def test_dismissed_records_are_not_counted_against_the_system():
    """A record dismissed as a duplicate is shown to nobody, so scoring it as a false positive would
    slander the system -- but the raw registry stays scorable on request."""
    truth = [GroundTruthSurvivor("A", *_at(0.0, 0.0))]
    live, dup = Record(record_id="live"), Record(record_id="dup", status="dismissed",
                                                 dismissed_reason="merged_into:live")
    live.lat, live.lon = _at(0.5, 0.0)
    dup.lat, dup.lon = _at(0.9, 0.0)

    assert dedup_accuracy([live, dup], truth, match_radius_m=6.0).precision == 1.0
    raw = dedup_accuracy([live, dup], truth, match_radius_m=6.0, include_dismissed=True)
    assert raw.precision == pytest.approx(0.5)


def test_metric_rows_carry_the_domain_slice():
    """Hard rule 5 / §5.12: a sim number and a real number may never be averaged, so `domain` is mandatory."""
    truth = [GroundTruthSurvivor("A", *_at(0.0, 0.0))]
    record = Record(record_id="a")
    record.lat, record.lon = _at(0.4, 0.0)
    acc = dedup_accuracy([record], truth, match_radius_m=6.0, duration_s=60.0)

    key = SliceKey(domain="sim", altitude_band="30-45", zone="settlement")
    rows = acc.rows(key)
    names = {r.name for r in rows}
    assert {"dedup.record_precision", "dedup.record_recall", "dedup.duplicate_rate",
            "dedup.count_mae", "dedup.record_fp_per_min"} <= names
    for row in rows:
        assert row.slice.domain == "sim"
        assert "domain=sim" in str(row)
    assert "zone=settlement" in key.label()

    with pytest.raises(TypeError):
        SliceKey()          # type: ignore[call-arg]  -- domain is never optional


def test_fp_per_min_is_reported_at_both_levels():
    """§5.6: "report FP/min twice ... The ratio is the value the tracker adds"."""
    assert fp_per_min(120, 600.0) == pytest.approx(12.0)      # raw detections
    assert fp_per_min(2, 600.0) == pytest.approx(0.2)         # after tracking + dedup
    assert math.isnan(fp_per_min(3, 0.0))


# ---------------------------------------------------------------------------- track-level MOT metrics ---
def _mot_frames(hyp_ids: list[int]) -> list[FrameAssociations]:
    box = np.array([[10.0, 10.0, 34.0, 34.0]])
    return [FrameAssociations(frame_idx=i, gt_ids=["A"], gt_data=box, hyp_ids=[hyp_ids[i]], hyp_data=box)
            for i in range(len(hyp_ids))]


def test_track_metrics_report_idf1_and_id_switches():
    """§5.6 "Metrics": HOTA / IDF1 / ID switches from py-motmetrics, on a case with a known answer."""
    key = SliceKey(domain="sim")
    clean = {r.name: r.value for r in track_metrics(_mot_frames([1] * 6), key)}
    assert clean["track.num_switches"] == 0
    assert clean["track.idf1"] == pytest.approx(1.0)
    assert clean["track.num_false_positives"] == 0
    assert clean["track.num_misses"] == 0
    assert clean["track.hota_alpha"] == pytest.approx(1.0, abs=1e-6)

    switched = {r.name: r.value for r in track_metrics(_mot_frames([1, 1, 1, 2, 2, 2]), key)}
    assert switched["track.num_switches"] == 1, "one relabelling is one ID switch"
    assert switched["track.idf1"] < clean["track.idf1"]


def test_track_metrics_can_work_in_geo_space():
    """§5.6: "dedup runs in geo space", so the identity metric can be computed there too."""
    key = SliceKey(domain="sim")
    frames = []
    for i in range(5):
        lat, lon = _at(float(i), 0.0)
        frames.append(FrameAssociations(frame_idx=i, gt_ids=["A"], gt_data=np.array([[lat, lon]]),
                                        hyp_ids=[7], hyp_data=np.array([[lat, lon]])))
    rows = {r.name: r.value for r in track_metrics(frames, key, distance="euclidean_m", max_distance_m=6.0)}
    assert rows["track.num_switches"] == 0
    assert rows["track.idf1"] == pytest.approx(1.0)


def test_frames_from_tracks_builds_the_metric_input():
    tracks = [_track(1, frames=(0, 1, 2))]
    truth = {0: {"A": (10.0, 10.0, 34.0, 34.0)}, 1: {"A": (10.0, 10.0, 34.0, 34.0)}}
    frames = frames_from_tracks(tracks, truth, distance="iou")
    assert [f.frame_idx for f in frames] == [0, 1, 2]
    assert frames[0].gt_data.shape == (1, 4) and frames[0].hyp_data.shape == (1, 4)
    assert frames[2].gt_data.shape == (0, 4), "a frame with no ground truth is still a frame"
    assert list(frames[0].hyp_ids) == [1]

    geo = frames_from_tracks(tracks, {0: {"A": _at(0.0, 0.0)}}, distance="euclidean_m")
    assert geo[0].hyp_data.shape == (1, 2)


# =========================================================================================== plumbing ===
def test_records_carry_their_evidence_and_provenance():
    """§5.8: the record must say WHY it exists -- which frame, which box, which radius clustered it."""
    track = _track(1, scores=(0.4, 0.95, 0.6), frames=(4, 5, 6))
    (record,) = records_from_tracks([track])

    assert record.n_observations == 3
    assert len(record.evidence) == 1
    evidence = record.evidence[0]
    assert evidence.frame_idx == 5, "the evidence frame is the highest-scoring observation"
    assert evidence.det_conf == pytest.approx(0.95)
    assert evidence.clip_id == "c"
    assert evidence.thumb_uri == "", "the export lane fills this; dedup never invents a path"

    assert record.source["dedup_eps_m"] == pytest.approx(EPS_M, abs=1e-6)
    assert record.source["dedup_eps_ce90_multiple"] == 2.0
    assert record.source["track_keys"] == [["c", 0, 1]]
    assert record.first_seen_utc == pytest.approx(T0)
    assert record.last_seen_utc == pytest.approx(T0 + 0.4)
    assert record.h_acc_basis == "budget_v1"
    assert record.agl_m == pytest.approx(45.0)
    assert record.pixel_size_px == pytest.approx(24.0)


def test_records_serialise_to_geojson_with_lon_lat_order():
    """RFC 7946 is lon-lat; everything else in this project is lat-lon. Getting it backwards puts the
    marker in the Indian Ocean."""
    (record,) = records_from_tracks([_track(1)])
    feature = record.to_feature()
    lon, lat, _ = feature["geometry"]["coordinates"]
    assert lat == pytest.approx(LAT, abs=1e-6)
    assert lon == pytest.approx(LON, abs=1e-6)
    assert feature["id"] == record.record_id
    assert "lat" not in feature["properties"] and "lon" not in feature["properties"]
    assert feature["properties"]["seen_in_passes"] == [0]


def test_best_attribute_takes_the_most_confident_observation():
    """§5.5a: posture may RAISE urgency, so the attribute must come from the observation that was most sure
    of it -- not from the biggest box and not from the last frame."""
    track = _track(1, scores=(0.5, 0.9, 0.7))
    track.observations[0].det.posture, track.observations[0].det.posture_conf = "standing", 0.2
    track.observations[1].det.posture, track.observations[1].det.posture_conf = "trapped", 0.85
    track.observations[2].det.posture, track.observations[2].det.posture_conf = "prone", 0.4
    track.observations[1].det.submersion, track.observations[1].det.submersion_conf = "half", 0.7
    track.observations[2].det.occlusion, track.observations[0].det.occlusion = 2, 1

    (record,) = records_from_tracks([track])
    assert record.posture == "trapped" and record.posture_conf == pytest.approx(0.85)
    assert record.submersion == "half" and record.submersion_conf == pytest.approx(0.7)
    assert record.occlusion == 1, "the least-occluded sighting is the honest one to report"


def test_empty_input_is_not_an_error():
    dedup = Deduplicator()
    assert dedup.ingest([]) == []
    assert dedup.all_records() == []
    assert dedup.mark_stale(pass_id=0) == []
    assert math.isnan(dedup.stats()["eps_m"])
    assert dedup.record_for_track(("c", 0, 1)) is None


def test_record_lookup_by_track_key():
    dedup = Deduplicator()
    (record,) = dedup.ingest([_track(7, pass_id=3, clip_id="clipA")])
    assert dedup.record_for_track(("clipA", 3, 7)) is record
    assert dedup.record_for_track(("clipA", 3, 8)) is None
