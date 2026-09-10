"""Frame-to-frame tracker backends behind one interface, so §5.6's A/B is a config change (F11).

`roboflow/trackers` BoT-SORT (Apache-2.0) is the **shipped** path. §5.6 lists Ultralytics' BoT-SORT first, but
Ultralytics is AGPL-3.0 and this product would inherit that licence, while the doc itself names
`roboflow/trackers` as the "licence-clean alternative ... same behaviour". The Ultralytics backend stays
reachable for the A/B the doc asks for and is imported lazily so nothing here drags in torch.

Both backends expose the same three calls:

    ids = backend.update(boxes_xyxy, scores, frame, timestamp)   # -1 where the backend has not named the track
    backend.prune_unconfirmed(keep_ids, max_age_s)
    backend.reset()

`update` returns one id per input detection, in input order.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from sightline.schemas import Intrinsics
from sightline.track.config import TrackerConfig


@runtime_checkable
class TrackerBackend(Protocol):
    """The surface `Tracker` needs. Anything else about a backend is its own business."""

    name: str

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        frame: np.ndarray | None,
        timestamp: float | None,
    ) -> np.ndarray: ...

    def track_age_frames(self, track_id: int) -> int | None: ...

    def track_hits(self, track_id: int) -> int | None: ...

    def prune_unconfirmed(self, keep_ids: set[int], max_age_s: float) -> int: ...

    def reset(self) -> None: ...


class RoboflowBoTSORT:
    """`trackers.BoTSORTTracker` (Apache-2.0) wired to `TrackerConfig`.

    Two conversions happen here and both are traps documented in `config.py`:

    * `lost_track_buffer` is stated by the library **in 30-FPS frames** and rescaled by `frame_rate`. Handing it
      the doc's 150 processed frames would yield a 25-frame / 5-second buffer. `roboflow_lost_track_buffer()`
      converts.
    * the library's `minimum_iou_threshold_*` are **similarities**, BoT-SORT's `match_thresh` is a **cost**.

    The library's `track_low_thresh` is fixed at 0.1 in its source (`low_mask = confidences > 0.1`), which is
    exactly §5.6 rule 2's value; a config that asks for something else is rejected rather than silently ignored.
    `fuse_score` is likewise always on in its first and third association stages.
    """

    #: What `trackers` hardcodes as its second-stage lower bound. Kept as a constant so the check below is honest.
    LIBRARY_LOW_THRESH = 0.1

    name = "trackers.BoTSORTTracker"
    licence = "Apache-2.0"

    def __init__(self, cfg: TrackerConfig, intrinsics: Intrinsics | None = None, cmc: Any | None = None) -> None:
        from trackers import BoTSORTTracker  # local import: keeps `sightline.track` importable without it

        if abs(cfg.track_low_thresh - self.LIBRARY_LOW_THRESH) > 1e-9:
            raise ValueError(
                f"{self.name} hardcodes track_low_thresh={self.LIBRARY_LOW_THRESH} (§5.6 rule 2). "
                f"TrackerConfig asks for {cfg.track_low_thresh}; change the backend or the config, do not "
                "assume the value took effect."
            )
        if not cfg.fuse_score:
            raise ValueError(f"{self.name} always fuses IoU with detection score; fuse_score=False is a lie")
        if cfg.with_reid:
            raise ValueError(f"{self.name} has no ReID branch; §5.6 rejects ReID at 20-60 px anyway")

        self.cfg = cfg
        self._cmc = cmc
        self.tracker = BoTSORTTracker(
            lost_track_buffer=cfg.roboflow_lost_track_buffer(),
            frame_rate=cfg.fps,
            track_activation_threshold=cfg.new_track_thresh,
            minimum_consecutive_frames=cfg.backend_min_consecutive_frames,
            minimum_iou_threshold_first_assoc=cfg.similarity_first_assoc,
            minimum_iou_threshold_second_assoc=cfg.similarity_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=cfg.similarity_unconfirmed_assoc,
            high_conf_det_threshold=cfg.track_high_thresh,
            enable_cmc=cfg.cmc_enabled,
            cmc_method=cfg.cmc_method,
            cmc_downscale=cfg.cmc_downscale(intrinsics.width_px if intrinsics else cfg.cmc_target_width_px),
            instant_first_frame_activation=cfg.backend_min_consecutive_frames <= 1,
        )
        if cmc is not None:
            # Replacing the estimator (rather than pre-warping) keeps BoT-SORT's ordering intact: the warp is
            # applied after Kalman predict and before association, which is where it belongs.
            self.tracker.cmc = cmc
            self.tracker.enable_cmc = True

    @property
    def lost_track_buffer_frames(self) -> int:
        """What the library actually resolved the buffer to, in processed frames. Assert on this, not on config."""
        return int(self.tracker.maximum_frames_without_update)

    @property
    def lost_track_buffer_s(self) -> float:
        return float(self.tracker.maximum_time_without_update)

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        frame: np.ndarray | None,
        timestamp: float | None,
    ) -> np.ndarray:
        import supervision as sv

        from sightline.track.cmc import NO_FRAME

        n = len(boxes_xyxy)
        if n == 0:
            dets = sv.Detections.empty()
            dets.confidence = np.zeros(0, dtype=np.float32)
        else:
            dets = sv.Detections(
                xyxy=np.asarray(boxes_xyxy, dtype=np.float32).reshape(n, 4),
                confidence=np.asarray(scores, dtype=np.float32).reshape(n),
                class_id=np.zeros(n, dtype=int),
            )
        if frame is None and self._cmc is not None:
            # See cmc.NO_FRAME: the library gates compensation on `frame is not None`, so telemetry-only CMC
            # would be silently skipped on a metadata replay. The sentinel keeps the gate open.
            frame = NO_FRAME
        out = self.tracker.update(dets, frame=frame, timestamp=timestamp)

        # `update` returns detections re-ordered by association stage, so map back to input order by box match.
        src = np.asarray(boxes_xyxy, dtype=np.float32).reshape(n, 4)
        ids = np.full(n, -1, dtype=int)
        taken = np.zeros(n, dtype=bool)
        if n and len(out) and out.tracker_id is not None:
            for box, tid in zip(np.asarray(out.xyxy, dtype=np.float32), np.asarray(out.tracker_id), strict=False):
                d = np.abs(src - box).max(axis=1)
                d[taken] = np.inf
                j = int(np.argmin(d))
                if np.isfinite(d[j]) and d[j] < 1e-3:
                    ids[j] = int(tid)
                    # `out` also carries unassociated detections with id -1 (including the ones that just
                    # spawned a track). Those must stay claimable by `_name_tracks_spawned_this_frame`.
                    taken[j] = int(tid) >= 0
        if n:
            self._name_tracks_spawned_this_frame(src, ids, taken)
        return ids

    def _name_tracks_spawned_this_frame(self, src: np.ndarray, ids: np.ndarray, taken: np.ndarray) -> None:
        """Give a brand-new tracklet its id in the frame that created it, not the frame after.

        `trackers` allocates an id only on a track's **second** successful update (except on the very first
        frame of a clip). The wrapper owns §5.6 rule 3's "3 hits within 2 s", so a hit it cannot see is a hit
        that does not count: a survivor would need four detections to clear a three-hit gate, and the first
        detection would be missing from the track's observations and from its evidence. Both are silent
        pessimism, so the id is allocated here instead.

        A tracklet spawned during this update has `age == 0` — `predict()` ran on every pre-existing track
        before spawning — and its state still holds the exact box it was created from, which is how it is
        matched back to the input detection.
        """
        for track in getattr(self.tracker, "tracks", []):
            if track.tracker_id != -1 or getattr(track, "age", 1) != 0:
                continue
            box = np.asarray(track.get_state_bbox(), dtype=np.float32).reshape(4)
            d = np.abs(src - box).max(axis=1)
            d[taken] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]) or d[j] > 0.5:
                continue  # cannot say which detection spawned it; leave it unnamed rather than guess
            track.tracker_id = self.tracker._allocate_tracker_id()
            ids[j] = int(track.tracker_id)
            taken[j] = True

    def track_age_frames(self, track_id: int) -> int | None:
        """Frames since the tracklet was created, or None if the backend cannot say.

        The wrapper needs this to time §5.6 rule 3's window honestly. `trackers` spawns a tracklet with
        `tracker_id = -1` and only names it on its **second** successful update, so the first hit is invisible
        from the outside; `age` (incremented by every `predict()`) recovers the frame it was created on.
        """
        for t in getattr(self.tracker, "tracks", []):
            if t.tracker_id == track_id:
                return int(getattr(t, "age", 0))
        return None

    def track_hits(self, track_id: int) -> int | None:
        """Total successful updates the backend has recorded for this track, or None."""
        for t in getattr(self.tracker, "tracks", []):
            if t.tracker_id == track_id:
                return int(getattr(t, "number_of_successful_updates", 0))
        return None

    def prune_unconfirmed(self, keep_ids: set[int], max_age_s: float) -> int:
        """Drop backend tracklets that never passed the wrapper's confirmation gate and have gone quiet.

        With `backend_min_consecutive_frames = 1` every detection immediately becomes a named track so the
        wrapper can time §5.6 rule 3's 2-second window. Without this prune a low detector threshold would leave
        every one-frame false positive in the association matrix for the whole 30-second buffer.
        """
        tracks = getattr(self.tracker, "tracks", None)
        if tracks is None:
            return 0
        before = len(tracks)
        self.tracker.tracks = [
            t
            for t in tracks
            if t.tracker_id in keep_ids
            or t.tracker_id == -1
            or getattr(t, "time_since_update_seconds", 0.0) <= max_age_s
        ]
        return before - len(self.tracker.tracks)

    def reset(self) -> None:
        self.tracker.tracks = []
        self.tracker.frame_id = 0
        self.tracker._reset_id_allocator()
        self.tracker._init_timestamp_state(self.cfg.fps)


class UltralyticsBoTSORT:
    """§5.6's first-listed tracker, kept reachable for the A/B. **Not exercised by this lane's tests.**

    Ultralytics is **AGPL-3.0**: using it in the shipped product licenses the product under AGPL. Treat this as
    an evaluation-only backend. Importing it pulls torch, so the import is deferred to `__init__` and the
    tracking lane's test suite never constructs one (CONTRACTS.md §3 rule 2 forbids importing torch here).
    """

    name = "ultralytics.BOTSORT"
    licence = "AGPL-3.0 (evaluation only)"

    def __init__(self, cfg: TrackerConfig, intrinsics: Intrinsics | None = None, cmc: Any | None = None) -> None:
        from types import SimpleNamespace

        from ultralytics.trackers.bot_sort import BOTSORT  # noqa: F401  (pulls torch)

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer_frames,  # Ultralytics takes this in PROCESSED frames directly
            match_thresh=cfg.match_thresh,  # and in the cost convention, so no conversion here
            fuse_score=cfg.fuse_score,
            gmc_method=cfg.cmc_method,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=cfg.with_reid,
            model="auto",
        )
        self.cfg = cfg
        self.args = args
        self.tracker = BOTSORT(args, frame_rate=int(round(cfg.fps)))
        self._cmc = cmc
        if cmc is not None:
            self.tracker.gmc = cmc

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        frame: np.ndarray | None,
        timestamp: float | None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "UltralyticsBoTSORT is wired for the §5.6 A/B but is not exercised by the tracking lane "
            "(importing ultralytics pulls torch, which CONTRACTS.md §3 rule 2 forbids in this lane). "
            "Finish and verify this adapter in the ML lane's environment before quoting numbers from it."
        )

    def track_age_frames(self, track_id: int) -> int | None:
        return None

    def track_hits(self, track_id: int) -> int | None:
        return None

    def prune_unconfirmed(self, keep_ids: set[int], max_age_s: float) -> int:
        return 0

    def reset(self) -> None:
        self.tracker.reset()


BACKENDS = {"trackers": RoboflowBoTSORT, "ultralytics": UltralyticsBoTSORT}


def make_backend(cfg: TrackerConfig, intrinsics: Intrinsics | None = None, cmc: Any | None = None):
    try:
        cls = BACKENDS[cfg.backend]
    except KeyError:
        raise ValueError(f"unknown tracker backend {cfg.backend!r}; choose one of {sorted(BACKENDS)}") from None
    return cls(cfg, intrinsics, cmc)
