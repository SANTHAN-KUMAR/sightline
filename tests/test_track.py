"""F11 tracking (SOLUTION_DOC §5.6): a STATIC survivor under a MOVING camera.

    "Survivors are mostly *static* and the *camera* moves, which is the opposite of the pedestrian-tracking
     assumption most trackers are tuned for."

Every test here is driven by `sightline.track.synth`, which projects a survivor pinned to a real lat/lon
through a camera flying over it. The right answer is therefore known by construction: one being, one track,
for as long as it is in frame.

The four things §5.6 says will break, each with a test that would actually catch it:

1. **buffers in the wrong unit** — "5 FPS x 30 s = 150, not the default 30", and `roboflow/trackers` restates
   it again in 30-FPS frames. Both conversions are pinned, including what the library resolves them to.
2. **identity across a dropout** — three missed frames must not mint a new record, and the contrast case
   (a buffer too short) must mint one, so the first test cannot pass for the wrong reason.
3. **the confirmation gate** — 3 hits within 2 s, timed in seconds because a decimated stream has no fixed
   frame period.
4. **camera-motion compensation over water** — rippling water gives optical flow a confident WRONG answer,
   so the telemetry homography has to take over, and which source was used has to be observable.

No GPU, no torch, no ultralytics (CONTRACTS.md §3 rule 2): the shipped backend is `roboflow/trackers`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sightline.common import geodesy
from sightline.schemas import Detection, FrameBundle, Intrinsics, Telemetry
from sightline.track import synth
from sightline.track.backends import BACKENDS, RoboflowBoTSORT, UltralyticsBoTSORT
from sightline.track.cmc import NO_FRAME, SightlineCMC
from sightline.track.config import TrackerConfig
from sightline.track.geometry import (
    affine_disagreement_px,
    affine_from_homography,
    ground_plane_homography,
    project_ground_point,
    rotation_optical_to_ned,
    telemetry_affine,
)
from sightline.track.tracker import Tracker, unlocated_fix

SURVIVOR = synth.Survivor("s1", north_m=0.0, east_m=0.0)


def _run(scene: synth.SceneConfig, opts: synth.SequenceOptions, cfg: TrackerConfig | None = None,
         survivors: list[synth.Survivor] | None = None) -> tuple[Tracker, list[synth.SyntheticFrame]]:
    frames = synth.make_sequence(scene, survivors or [SURVIVOR], opts)
    tracker = Tracker(cfg or TrackerConfig(fps=scene.fps), intrinsics=scene.intrinsics)
    for frame in frames:
        tracker.update(frame.detections, frame.bundle, frame.fixes)
    return tracker, frames


# =================================================================================== §5.6 rule 1: units ===
def test_track_buffer_is_stated_in_processed_frames():
    """§5.6 rule 1, the number the doc spells out: 5 FPS x 30 s = 150 processed frames, not 30."""
    cfg = TrackerConfig(fps=5.0, track_buffer_s=30.0)
    assert cfg.track_buffer_frames == 150
    assert cfg.min_hits_window_frames == 10           # 2 s at 5 FPS

    # seconds are the unit that survives a change of frame rate
    assert TrackerConfig(fps=10.0, track_buffer_s=30.0).track_buffer_frames == 300
    assert TrackerConfig(fps=2.0, track_buffer_s=30.0).track_buffer_frames == 60


def test_roboflow_buffer_unit_trap_is_converted_not_assumed():
    """`roboflow/trackers` states `lost_track_buffer` in 30-FPS frames and rescales it by `frame_rate`.

    Handing it the doc's 150 processed frames would silently give a 25-frame / 5-second buffer -- a fifth of
    the intended one. This test pins BOTH halves: what the library does with the naive number, and what it
    resolves our converted number to.
    """
    from trackers import BoTSORTTracker

    cfg = TrackerConfig(fps=5.0, track_buffer_s=30.0)
    assert cfg.roboflow_lost_track_buffer() == 900

    naive = BoTSORTTracker(lost_track_buffer=cfg.track_buffer_frames, frame_rate=cfg.fps)
    assert naive.maximum_frames_without_update == 25
    assert naive.maximum_time_without_update == pytest.approx(5.0)

    backend = RoboflowBoTSORT(cfg)
    assert backend.lost_track_buffer_frames == 150, "assert on what the LIBRARY resolved, not on the config"
    assert backend.lost_track_buffer_s == pytest.approx(30.0)


def test_match_threshold_convention_is_converted_for_the_similarity_backend():
    """BoT-SORT's `match_thresh` is a COST (1 - IoU); `roboflow/trackers` takes a minimum SIMILARITY."""
    cfg = TrackerConfig(match_thresh=0.8, second_match_thresh=0.5, unconfirmed_match_thresh=0.7)
    assert cfg.similarity_first_assoc == pytest.approx(0.2)
    assert cfg.similarity_second_assoc == pytest.approx(0.5)
    assert cfg.similarity_unconfirmed_assoc == pytest.approx(0.3)
    # §5.6 rule 5: lowering match_thresh 0.8 -> 0.6 DEMANDS a higher IoU, i.e. is stricter
    assert TrackerConfig(match_thresh=0.6).similarity_first_assoc == pytest.approx(0.4)


def test_cmc_downscale_lands_on_the_640px_target():
    """§5.6 rule 4: "a 640-px-wide downscale; ORB/ECC on 4K is far too slow"."""
    cfg = TrackerConfig(cmc_target_width_px=640)
    assert cfg.cmc_downscale(3840) == 6            # 4K -> 640
    assert cfg.cmc_downscale(1920) == 3            # 1080p -> 640
    assert cfg.cmc_downscale(640) == 1
    assert cfg.cmc_downscale(320) == 1             # never upscaled


def test_config_rejects_an_unreachable_confirmation_gate():
    """3 hits inside 2 s is impossible below 1 FPS. Failing loudly beats never emitting a record."""
    with pytest.raises(ValueError, match="unreachable confirmation gate"):
        TrackerConfig(fps=1.0, min_hits=3, min_hits_window_s=1.0)
    with pytest.raises(ValueError, match="track_low_thresh"):
        TrackerConfig(track_low_thresh=0.9, track_high_thresh=0.5)
    with pytest.raises(ValueError, match="fps"):
        TrackerConfig(fps=0.0)
    TrackerConfig(fps=1.0, min_hits=3, min_hits_window_s=2.0)   # exactly reachable: allowed


def test_roboflow_backend_refuses_a_config_it_cannot_honour():
    """The library hardcodes `track_low_thresh` at 0.1 and always fuses the score. A config that asks for
    something else must be rejected, not silently ignored."""
    with pytest.raises(ValueError, match="track_low_thresh"):
        RoboflowBoTSORT(TrackerConfig(track_low_thresh=0.3))
    with pytest.raises(ValueError, match="fuse_score"):
        RoboflowBoTSORT(TrackerConfig(fuse_score=False))
    with pytest.raises(ValueError, match="ReID"):
        RoboflowBoTSORT(TrackerConfig(with_reid=True))


# ======================================================== §5.6: a static target under a moving camera ===
def test_static_survivor_under_a_moving_camera_keeps_one_track_id():
    """THE case §5.6 exists for. The survivor never moves; the camera crosses over it at 3 m/s, which is
    ~24 px of image motion per processed frame -- the target moves its own width every frame."""
    scene = synth.SceneConfig(speed_ms=3.0, fps=5.0)
    tracker, frames = _run(scene, synth.SequenceOptions(n_frames=30, seed=1))

    tracks = tracker.close()
    assert len(tracks) == 1, f"one survivor became {len(tracks)} tracks"
    track = tracks[0]
    assert track.confirmed
    assert track.cls == "human"

    visible = [f for f in frames if "s1" in f.truth_det_index]
    assert len(visible) >= 15, "the synthetic fly-over must actually show the survivor"
    assert len(track.observations) == len(visible), "every visible frame must reach the one track"
    assert {o.track_id for o in track.observations} == {track.track_id}
    assert not tracker.pending_tracks()
    assert tracker.stats.tracks_started == 1

    # the boxes really did move across the frame -- the test is not passing on a stationary image
    xs = [o.det.centre_px()[1] for o in track.observations]
    assert max(xs) - min(xs) > 4 * track.observations[0].det.height_px


def test_moving_camera_really_moves_the_target_in_the_image():
    """Guards the guard: if the synthetic camera stopped moving, the test above would be vacuous."""
    scene = synth.SceneConfig(speed_ms=3.0, fps=5.0)
    frames = synth.make_sequence(scene, [SURVIVOR], synth.SequenceOptions(n_frames=10, seed=0))
    seen = [f.truth_px["s1"] for f in frames if "s1" in f.truth_px]
    steps = [abs(seen[i + 1][1] - seen[i][1]) for i in range(len(seen) - 1)]
    assert min(steps) > 20.0, f"expected ~24 px/frame of image motion, got {min(steps):.1f}"


def test_three_frame_dropout_does_not_break_identity():
    """§5.6 rule 1's payoff: a 0.6 s gap is nothing against a 30 s buffer."""
    scene = synth.SceneConfig(fps=5.0)
    opts = synth.SequenceOptions(n_frames=30, dropout_frames=(10, 11, 12), seed=2)
    tracker, frames = _run(scene, opts)

    tracks = tracker.close()
    assert len(tracks) == 1
    track = tracks[0]
    frame_ids = [o.frame_idx for o in track.observations]
    assert set(range(10, 13)).isdisjoint(frame_ids), "the dropout frames really were empty"
    assert max(frame_ids) > 12, "the track must survive PAST the dropout, not merely start before it"
    assert len(track.observations) == len([f for f in frames if "s1" in f.truth_det_index])


def test_a_dropout_longer_than_the_buffer_does_start_a_new_track():
    """The contrast case. Without it, the dropout test above could pass for the wrong reason (e.g. because
    the tracker never forgets anything), and the 150-frame buffer would be doing nothing."""
    scene = synth.SceneConfig(fps=5.0)
    opts = synth.SequenceOptions(n_frames=30, dropout_frames=tuple(range(10, 16)), seed=2)
    short = TrackerConfig(fps=5.0, track_buffer_s=0.4)      # 2 processed frames
    tracker, _ = _run(scene, opts, cfg=short)
    assert len(tracker.close()) == 2, "a 1.2 s gap must outlive a 0.4 s buffer"

    long_buffer, _ = _run(scene, opts, cfg=TrackerConfig(fps=5.0, track_buffer_s=30.0))
    assert len(long_buffer.close()) == 1, "the doc's 30 s buffer must bridge the same gap"


def test_confidence_dips_into_the_bytetrack_second_stage_keep_the_track():
    """§5.6 rule 2: `track_low_thresh` 0.1 with `fuse_score` on is what rescues a flickering 15-px blob."""
    scene = synth.SceneConfig(fps=5.0)
    dips = {k: 0.15 for k in (8, 9, 10, 11)}                # below track_high_thresh (0.5), above 0.1
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=30, conf_dips=dips, seed=4))
    tracks = tracker.close()
    assert len(tracks) == 1
    scores = {o.frame_idx: o.det.score for o in tracks[0].observations}
    assert all(scores.get(k) == pytest.approx(0.15) for k in dips), "the dipped frames must still be tracked"


def test_box_jitter_does_not_split_the_track():
    scene = synth.SceneConfig(fps=5.0)
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=30, box_jitter_px=3.0, seed=5))
    assert len(tracker.close()) == 1


# ============================================================== §5.6 rule 3: 3 hits within 2 seconds ===
def _bundle(frame_idx: int, t_utc: float, intr: Intrinsics) -> FrameBundle:
    tel = Telemetry(t_utc=t_utc, lat=12.9, lon=77.6, alt_msl_m=645.0, agl_m=45.0, frame_idx=frame_idx)
    return FrameBundle(frame_idx=frame_idx, t_utc=t_utc, telemetry=tel, intrinsics=intr, clip_id="gate")


def _feed(tracker: Tracker, intr: Intrinsics, hits: list[int], fps: float, box=(100.0, 100.0, 124.0, 124.0),
          n_frames: int | None = None) -> None:
    """Feed `n_frames` frames at `fps`, with a detection only on the frame indices in `hits`."""
    total = n_frames if n_frames is not None else (max(hits) + 1)
    for i in range(total):
        dets = [Detection(bbox_px=box, score=0.9, frame_idx=i)] if i in hits else []
        tracker.update(dets, _bundle(i, 1_800_000_000.0 + i / fps, intr))


def test_confirmation_needs_three_hits_and_gets_them_on_the_third_frame():
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=intr)
    box = (100.0, 100.0, 124.0, 124.0)

    for i in range(3):
        tracker.update([Detection(bbox_px=box, score=0.9, frame_idx=i)],
                       _bundle(i, 1_800_000_000.0 + i / 5.0, intr))
        if i < 2:
            assert tracker.confirmed_tracks() == [], f"confirmed after only {i + 1} hit(s)"
            assert len(tracker.pending_tracks()) == 1

    assert len(tracker.confirmed_tracks()) == 1, "3 hits inside 0.4 s must confirm"
    assert len(tracker.confirmed_tracks()[0].observations) == 3


def test_confirmation_refuses_three_hits_spread_beyond_two_seconds():
    """The half of §5.6 rule 3 no backend can express: the window is in SECONDS, and a decimated stream has
    no fixed frame period, so hit COUNT alone is not the gate."""
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0, unconfirmed_prune_s=None), intrinsics=intr)
    _feed(tracker, intr, [0, 6, 12, 18], fps=5.0)           # 1.2 s apart: any 3 span 2.4 s
    assert tracker.confirmed_tracks() == []
    pending = tracker.pending_tracks()
    assert len(pending) == 1 and len(pending[0].observations) == 4, "the hits happened; only the gate refused"


def test_confirmation_accepts_three_hits_exactly_on_the_two_second_boundary():
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0, unconfirmed_prune_s=None), intrinsics=intr)
    _feed(tracker, intr, [0, 5, 10], fps=5.0)               # 1.0 s apart: span exactly 2.0 s
    assert len(tracker.confirmed_tracks()) == 1


def test_single_frame_false_positives_never_become_tracks():
    """§5.5c runs a deliberately low detector threshold and "recovers precision downstream". This gate is
    the first half of that recovery; geo-dedup (F12) is the second."""
    scene = synth.SceneConfig(fps=5.0)
    opts = synth.SequenceOptions(n_frames=30, false_positives_per_frame=3, fp_score=0.55, seed=6)
    tracker, _ = _run(scene, opts)

    tracks = tracker.close()
    assert len(tracks) == 1, f"{len(tracks)} tracks from 1 survivor + 90 scattered false positives"
    assert tracker.stats.detections_in > 60
    assert tracker.stats.tracks_pruned_unconfirmed > 10, "the ghosts must be pruned, not left in the matrix"


# ================================================== §5.6 rule 4: camera-motion compensation over water ===
def _cmc_scene() -> synth.SceneConfig:
    """Small enough to render 10 frames inside a couple of hundred MB (HANDBOOK §4: RAM is the constraint)."""
    return synth.SceneConfig(intrinsics=synth.tile_intrinsics(480, 270, 30.0), fps=5.0, speed_ms=3.0)


def test_cmc_uses_optical_flow_over_textured_dry_ground():
    scene = _cmc_scene()
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=10, render=True, seed=3))
    counts = tracker.cmc_source_counts()
    # the backend skips compensation on a frame with neither tracks nor detections, so the log is one
    # shorter than the sequence: the survivor is not yet inside the frame on frame 0.
    assert sum(counts.values()) == len(tracker.cmc_log) >= 9
    assert counts["optical_flow"] >= 7, counts
    assert counts["identity"] == 0
    # the one telemetry frame is the bootstrap: optical flow has no previous frame yet, and says so
    first = tracker.cmc_log[0]
    assert first.source == "telemetry_homography" and "bootstrap" in first.reason

    agreeing = [r for r in tracker.cmc_log if r.source == "optical_flow"]
    assert max(r.disagreement_px for r in agreeing) < 2.0, "on real ground the two estimates must agree"
    assert min(r.motion_px for r in agreeing) > 5.0, "and both must report the real camera motion"


def test_cmc_falls_back_to_telemetry_over_rippling_water():
    """§5.6 rule 4 verbatim: "Rippling water breaks optical-flow compensation, so fall back to a
    telemetry-predicted homography"."""
    scene = _cmc_scene()
    water = synth.WaterConfig(mode="ripple", coverage=1.0, ripple_strength=1.0)
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=10, render=True, water=water, seed=3))

    counts = tracker.cmc_source_counts()
    assert counts["optical_flow"] == 0, counts
    assert counts["telemetry_homography"] == len(tracker.cmc_log) >= 9, counts
    assert len(tracker.close()) == 1, "the fallback must keep the survivor on one track"

    disagreements = [r.disagreement_px for r in tracker.cmc_log if math.isfinite(r.disagreement_px)]
    assert disagreements and min(disagreements) > TrackerConfig().cmc_max_disagreement_px


def test_cmc_catches_coherent_water_drift_that_an_inlier_count_cannot():
    """The failure RANSAC cannot see: a drifting surface gives optical flow plenty of correspondences and a
    confident WRONG answer. Only the telemetry cross-check catches it."""
    scene = _cmc_scene()
    water = synth.WaterConfig(mode="drift", drift_ne_ms=(6.0, 0.0), coverage=1.0)
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=10, render=True, water=water, seed=3))

    counts = tracker.cmc_source_counts()
    assert counts["telemetry_homography"] == len(tracker.cmc_log) >= 9, counts
    flagged = [r for r in tracker.cmc_log if "disagrees with telemetry" in r.reason]
    assert len(flagged) >= 8
    assert all(not r.optflow_failed for r in flagged), "optical flow SUCCEEDED here -- it was simply wrong"


def test_cmc_source_is_recorded_per_frame_and_summarised():
    """§5.6 rule 4: "make the switch between CMC sources observable" -- and TUNING_ORDER step 1 depends on it."""
    scene = _cmc_scene()
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=10, render=True, seed=3))
    log = tracker.cmc_log
    assert len(log) >= 9
    indices = [r.frame_idx for r in log]
    assert indices == list(range(indices[0], indices[0] + len(log))), "one entry per compensated frame, in order"
    assert all(r.reason for r in log), "every decision must carry a human-readable reason"
    assert all(r.affine.shape == (2, 3) for r in log)
    assert sum(tracker.cmc_source_counts().values()) == len(log)


def test_cmc_without_frames_or_telemetry_degrades_to_identity_and_says_so():
    cfg = TrackerConfig(fps=5.0, cmc_telemetry_enabled=False)
    cmc = SightlineCMC(cfg, synth.tile_intrinsics())
    cmc.set_context(0, None, None)
    affine = cmc.estimate(NO_FRAME)
    assert np.allclose(affine, np.eye(2, 3))
    assert cmc.last.source == "identity"
    assert "no telemetry" in cmc.last.reason
    assert cmc.source_counts() == {"optical_flow": 0, "telemetry_homography": 0, "identity": 1}


def test_cmc_disabled_by_config_is_still_logged():
    cmc = SightlineCMC(TrackerConfig(cmc_enabled=False), synth.tile_intrinsics())
    cmc.set_context(3, None, None)
    assert np.allclose(cmc.estimate(None), np.eye(2, 3))
    assert cmc.last.source == "identity" and "disabled" in cmc.last.reason


def test_tracker_with_cmc_disabled_exposes_no_sources():
    tracker = Tracker(TrackerConfig(fps=5.0, cmc_enabled=False), intrinsics=synth.tile_intrinsics())
    assert tracker.cmc is None
    assert tracker.cmc_source_counts() == {}
    assert tracker.cmc_log == []


# ============================================================ the telemetry-predicted homography (F13') ===
def test_ground_plane_homography_agrees_with_a_direct_projection():
    """`geometry.py` is checked against an independent projection of the same ground points, not against
    itself: the homography is only worth falling back to if it is actually the right warp."""
    intr = synth.tile_intrinsics(960, 540, 30.0)
    scene = synth.SceneConfig(intrinsics=intr, fps=5.0, speed_ms=3.0)
    prev, _ = synth._telemetry(scene, 4, np.random.default_rng(0), synth.SequenceOptions())
    cur, _ = synth._telemetry(scene, 5, np.random.default_rng(0), synth.SequenceOptions())

    h = ground_plane_homography(prev, cur, intr)
    assert h is not None

    worst = 0.0
    for dn in (-4.0, 0.0, 4.0):
        for de in (-4.0, 0.0, 4.0):
            rel_prev = (dn, de, -prev.ned_m[2])
            p_prev = project_ground_point(prev, intr, *rel_prev)
            p_cur = project_ground_point(cur, intr, dn - (cur.ned_m[0] - prev.ned_m[0]),
                                         de - (cur.ned_m[1] - prev.ned_m[1]), -cur.ned_m[2])
            assert p_prev is not None and p_cur is not None
            mapped = h @ np.array([p_prev[0], p_prev[1], 1.0])
            mapped = mapped[:2] / mapped[2]
            worst = max(worst, float(np.hypot(*(mapped - np.array(p_cur)))))
    assert worst < 1e-6, f"homography and projection disagree by {worst} px"


def test_telemetry_affine_is_exact_at_nadir_and_degrades_when_oblique():
    """BoT-SORT warps Kalman states with a 2x3 affine, so the homography has to be reduced. The reduction is
    exact for a nadir camera and loses accuracy as the view goes oblique -- `residual_px` says how much, so
    a caller can refuse the fallback rather than trust it silently."""
    intr = synth.tile_intrinsics(960, 540, 30.0)

    nadir = synth.SceneConfig(intrinsics=intr, gimbal_pitch_deg=-90.0, fps=5.0)
    prev, _ = synth._telemetry(nadir, 0, np.random.default_rng(0), synth.SequenceOptions())
    cur, _ = synth._telemetry(nadir, 1, np.random.default_rng(0), synth.SequenceOptions())
    flat = telemetry_affine(prev, cur, intr)
    assert flat is not None and flat.residual_px < 1e-6

    oblique = synth.SceneConfig(intrinsics=intr, gimbal_pitch_deg=-45.0, agl_m=45.0, fps=5.0)
    prev, _ = synth._telemetry(oblique, 0, np.random.default_rng(0), synth.SequenceOptions())
    cur, _ = synth._telemetry(oblique, 1, np.random.default_rng(0), synth.SequenceOptions())
    tilted = telemetry_affine(prev, cur, intr)
    assert tilted is not None and tilted.residual_px > flat.residual_px


def test_ground_plane_homography_refuses_impossible_geometry():
    intr = synth.tile_intrinsics()
    scene = synth.SceneConfig(intrinsics=intr, fps=5.0)
    prev, _ = synth._telemetry(scene, 0, np.random.default_rng(0), synth.SequenceOptions())
    cur, _ = synth._telemetry(scene, 1, np.random.default_rng(0), synth.SequenceOptions())
    assert ground_plane_homography(prev, cur, intr, agl_m=0.0) is None
    assert ground_plane_homography(prev, cur, intr, agl_m=float("nan")) is None
    assert telemetry_affine(prev, cur, intr, agl_m=0.0) is None


def test_rotation_optical_to_ned_puts_the_nadir_view_axis_down():
    """The one place the camera-frame convention lives. If this flips, every warp and every fix flips."""
    tel = Telemetry(t_utc=0.0, lat=12.9, lon=77.6, alt_msl_m=645.0, agl_m=45.0,
                    q_gimbal=synth.geodesy.euler_to_quat(0.0, 0.0, 0.0))
    r = rotation_optical_to_ned(tel)
    assert tel.gimbal_pitch_deg() == pytest.approx(-90.0)
    assert np.allclose(r @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)   # view axis -> down
    assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)   # image right -> east
    assert np.allclose(r @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-12)  # image down -> south


def test_affine_disagreement_is_measured_in_pixels():
    identity = np.eye(2, 3, dtype=np.float32)
    shifted = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert affine_disagreement_px(identity, shifted, 640, 360) == pytest.approx(7.0)
    assert affine_disagreement_px(identity, identity, 640, 360) == pytest.approx(0.0)

    fit = affine_from_homography(np.eye(3), 640, 360)
    assert fit.residual_px == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fit.affine, np.eye(2, 3), atol=1e-9)


# ================================================================================ pipeline plumbing ===
def test_observations_carry_the_supplied_fix_pass_id_and_clip():
    scene = synth.SceneConfig(fps=5.0)
    cfg = TrackerConfig(fps=5.0, pass_id=2, track_id_offset=500)
    tracker, frames = _run(scene, synth.SequenceOptions(n_frames=20, clip_id="pass2", seed=8), cfg=cfg)

    (track,) = tracker.close()
    assert track.track_id >= 500, "each pass gets its own id block so dedup never sees a collision"
    assert track.clip_id == "pass2"
    for obs in track.observations:
        assert obs.pass_id == 2
        assert obs.clip_id == "pass2"
        assert obs.fix.valid and obs.fix.h_acc_m > 0.0
        assert obs.det.frame_idx == obs.frame_idx

    # the fixes really are the ones synth handed in, not invented
    by_frame = {f.bundle.frame_idx: f for f in frames}
    for obs in track.observations[:5]:
        src = by_frame[obs.frame_idx]
        assert obs.fix.lat == pytest.approx(src.fixes[src.truth_det_index["s1"]].lat)


def test_without_a_geolocator_every_fix_is_marked_invalid():
    """A track with no geolocation must be refused by dedup, never clustered on a fiction."""
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=intr)
    _feed(tracker, intr, [0, 1, 2], fps=5.0)
    (track,) = tracker.close()
    assert all(not o.fix.valid for o in track.observations)
    assert all(o.fix.reject_reason == "not_geolocated" for o in track.observations)
    assert all(math.isinf(o.fix.h_acc_m) for o in track.observations)

    tel = Telemetry(t_utc=0.0, lat=1.0, lon=2.0, alt_msl_m=100.0, agl_m=45.0)
    fix = unlocated_fix(tel)
    assert not fix.valid and fix.lat == 1.0 and fix.alt_msl_m == pytest.approx(55.0)


def test_a_geolocate_callback_is_used_when_no_fixes_are_supplied():
    intr = synth.tile_intrinsics()
    calls: list[int] = []

    def geolocate(det, bundle):
        calls.append(bundle.frame_idx)
        return synth.GeoFix(lat=12.9, lon=77.6, alt_msl_m=600.0, h_acc_m=2.6, off_nadir_deg=0.0)

    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=intr, geolocate=geolocate)
    _feed(tracker, intr, [0, 1, 2], fps=5.0)
    (track,) = tracker.close()
    assert calls == [0, 1, 2]
    assert all(o.fix.valid and o.fix.h_acc_m == 2.6 for o in track.observations)


def test_fixes_must_be_parallel_to_detections():
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=intr)
    det = Detection(bbox_px=(1.0, 1.0, 20.0, 20.0), score=0.9)
    with pytest.raises(ValueError, match="parallel"):
        tracker.update([det], _bundle(0, 1_800_000_000.0, intr), fixes=[])


def test_describe_records_everything_needed_to_reproduce_a_run():
    """§5.12: a number without the configuration that produced it cannot be reproduced."""
    scene = synth.SceneConfig(fps=5.0)
    tracker, _ = _run(scene, synth.SequenceOptions(n_frames=12, render=False, seed=9))
    d = tracker.describe()
    assert d["track_buffer_frames"] == 150
    assert d["backend_lost_buffer_frames"] == 150
    assert d["backend_lost_buffer_s"] == pytest.approx(30.0)
    assert d["min_hits"] == 3 and d["min_hits_window_s"] == 2.0
    assert d["with_reid"] is False
    assert d["backend_name"] == "trackers.BoTSORTTracker"
    assert d["backend_licence"] == "Apache-2.0"
    assert d["cmc_sources"]["telemetry_homography"] > 0
    assert d["stats"]["tracks_confirmed"] == 1


def test_empty_frames_are_tolerated():
    intr = synth.tile_intrinsics()
    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=intr)
    for i in range(5):
        assert tracker.update([], _bundle(i, 1_800_000_000.0 + i / 5.0, intr)) == []
    assert tracker.close() == []


def test_ultralytics_backend_is_reachable_but_not_exercised_here():
    """§5.6 lists Ultralytics' BoT-SORT first, but it is AGPL-3.0 and importing it pulls torch, which
    CONTRACTS.md §3 rule 2 forbids in this lane. It stays wired for the A/B and refuses to pretend."""
    assert BACKENDS["ultralytics"] is UltralyticsBoTSORT
    assert "AGPL" in UltralyticsBoTSORT.licence

    stub = object.__new__(UltralyticsBoTSORT)               # no __init__: nothing is imported
    with pytest.raises(NotImplementedError, match="torch"):
        stub.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), None, None)

    # CONTRACTS.md section 3 rule 2 is about what THIS LANE imports, so ask that question in a clean
    # interpreter. Asserting on the running session's `sys.modules` instead made the check order-dependent:
    # `conftest.py`'s CUDA fixture imports torch, so any GPU test scheduled earlier in the same session made
    # this fail while the lane itself was innocent. A subprocess tests the actual property and cannot be
    # polluted by a neighbour.
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    probe = subprocess.run(
        [_sys.executable, "-c",
         "import sightline.track, sightline.track.backends, sys; "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, timeout=120,
        cwd=str(_Path(__file__).resolve().parents[1]))
    assert probe.returncode == 0, f"probe failed: {probe.stderr[-400:]}"
    assert probe.stdout.strip() == "False", "importing this lane pulled torch in"


def test_unknown_backend_is_rejected_by_name():
    from sightline.track.backends import make_backend

    with pytest.raises(ValueError, match="unknown tracker backend"):
        make_backend(TrackerConfig(backend="botsort2"))     # type: ignore[arg-type]


# ==================================================================== the synthetic generator itself ===
def test_synthetic_survivor_projects_to_its_own_latlon():
    """If `synth` were wrong, every test above would be measuring the wrong thing."""
    scene = synth.SceneConfig(fps=5.0)
    frames = synth.make_sequence(scene, [SURVIVOR], synth.SequenceOptions(n_frames=6, geo_sigma_m=0.0, seed=0))
    for frame in frames:
        if "s1" not in frame.truth_det_index:
            continue
        lat, lon = frame.truth_latlon["s1"]
        fix = frame.fixes[frame.truth_det_index["s1"]]
        assert geodesy.haversine_m(lat, lon, fix.lat, fix.lon) < 1e-6

        # the box is the projected ground footprint: at 45 m with fx=1791 px, 0.6 m is ~24 px
        det = frame.detections[frame.truth_det_index["s1"]]
        expected = 0.6 / scene.agl_m * scene.intrinsics.fx
        assert det.width_px == pytest.approx(expected, rel=0.02)


def test_synthetic_geo_bias_is_a_per_pass_constant():
    """§5.7: yaw and boresight biases do NOT average out over frames, which is what dedup must absorb."""
    scene = synth.SceneConfig(fps=5.0)
    opts = synth.SequenceOptions(n_frames=8, geo_sigma_m=0.0, geo_bias_ne_m=(3.0, -2.0), seed=0)
    for frame in synth.make_sequence(scene, [SURVIVOR], opts):
        if "s1" not in frame.truth_det_index:
            continue
        lat, lon = frame.truth_latlon["s1"]
        fix = frame.fixes[frame.truth_det_index["s1"]]
        north, east = geodesy.ne_between(lat, lon, fix.lat, fix.lon)
        assert (north, east) == pytest.approx((3.0, -2.0), abs=1e-3)
