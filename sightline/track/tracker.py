"""F11: the frame-to-frame tracker, tuned for a static survivor under a moving camera (SOLUTION_DOC §5.6).

    "Survivors are mostly *static* and the *camera* moves, which is the opposite of the pedestrian-tracking
     assumption most trackers are tuned for."

`Tracker` is a thin wrapper, and thin on purpose — the association itself is BoT-SORT's, tuned through
`TrackerConfig`. What the wrapper owns is the part no backend can express:

* **the confirmation gate** (§5.6 rule 3): 3 hits inside 2 s before a track is emitted. Every backend can count
  hits; none of them can time the window, because a decimated stream has no fixed frame period. So the backend is
  told to name tracks immediately and the wrapper does the timing.
* **camera-motion compensation with a fallback** (§5.6 rule 4): see `sightline/track/cmc.py`.
* **the pipeline types**: in come `Detection` + `FrameBundle` (+ the geolocation lane's `GeoFix`), out go the
  frozen `Track` / `Observation` of `schemas.py` that `sightline/dedup/` consumes.

Public API::

    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=bundle.intrinsics)
    for bundle, detections, fixes in stream:
        live = tracker.update(detections, bundle, fixes)     # confirmed tracks touched by THIS frame
    tracks = tracker.close()                                 # every confirmed track of the pass, for dedup
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sightline.schemas import Detection, FrameBundle, GeoFix, Intrinsics, Observation, Telemetry, Track
from sightline.track.backends import make_backend
from sightline.track.cmc import CmcResult, SightlineCMC
from sightline.track.config import TrackerConfig


def unlocated_fix(tel: Telemetry, reason: str = "not_geolocated") -> GeoFix:
    """A placeholder for "this detection has not been through F13 yet".

    It is marked `valid = False` and carries the aircraft's own position, never a made-up target position: the
    dedup stage drops invalid fixes rather than clustering a fiction. `Tracker` emits these only when no
    geolocation source was supplied, so a pure identity test can run without the geo lane.
    """
    return GeoFix(
        lat=tel.lat,
        lon=tel.lon,
        alt_msl_m=tel.alt_msl_m - tel.agl_m,
        h_acc_m=float("inf"),
        off_nadir_deg=0.0,
        agl_m=tel.agl_m,
        valid=False,
        reject_reason=reason,
    )


@dataclass(slots=True)
class _TrackState:
    """Wrapper-side bookkeeping for one backend track id."""

    global_id: int
    hits: deque[float] = field(default_factory=deque)
    last_t: float = 0.0
    confirmed_at: float | None = None
    n_hits: int = 0


@dataclass(slots=True)
class TrackerStats:
    frames: int = 0
    detections_in: int = 0
    detections_associated: int = 0
    tracks_started: int = 0
    tracks_confirmed: int = 0
    tracks_pruned_unconfirmed: int = 0


class Tracker:
    """BoT-SORT with the §5.6 static-survivor tuning, a timed confirmation gate and observable CMC."""

    def __init__(
        self,
        cfg: TrackerConfig | None = None,
        intrinsics: Intrinsics | None = None,
        *,
        geolocate=None,
        clip_id: str = "",
    ) -> None:
        """
        Args:
            cfg: the §5.6 tuning. Defaults are the doc's numbers.
            intrinsics: used to size the CMC downscale and the telemetry homography. Per-frame intrinsics from
                the `FrameBundle` override it.
            geolocate: optional `(Detection, FrameBundle) -> GeoFix` from the F13 lane. Without it every
                observation carries an invalid `GeoFix` and dedup will refuse the track.
            clip_id: stamped on every track and observation when the bundle does not carry one.
        """
        self.cfg = cfg or TrackerConfig()
        self.intrinsics = intrinsics
        self.clip_id = clip_id
        self._geolocate = geolocate

        self.cmc = SightlineCMC(self.cfg, intrinsics) if self.cfg.cmc_enabled else None
        self.backend = make_backend(self.cfg, intrinsics, self.cmc)

        self._tracks: dict[int, Track] = {}
        self._state: dict[int, _TrackState] = {}  # backend id -> state
        self._next_global_id = self.cfg.track_id_offset
        self._prev_tel: Telemetry | None = None
        #: frame_idx -> t_utc for the recent past, so a track's creation frame can be given a real timestamp.
        self._frame_times: dict[int, float] = {}
        self._frame_times_keep = self.cfg.track_buffer_frames + 8
        self.stats = TrackerStats()

    # --- main loop ----------------------------------------------------------------------------------------
    def update(
        self,
        detections: list[Detection],
        bundle: FrameBundle,
        fixes: list[GeoFix] | None = None,
    ) -> list[Track]:
        """Feed one processed frame. Returns the confirmed tracks that were updated by this frame.

        `fixes`, when given, must be parallel to `detections` (one `GeoFix` per box, same order).
        """
        if fixes is not None and len(fixes) != len(detections):
            raise ValueError(f"fixes must be parallel to detections ({len(fixes)} vs {len(detections)})")

        intr = bundle.intrinsics or self.intrinsics
        tel = bundle.telemetry
        if self.cmc is not None:
            self.cmc.set_context(bundle.frame_idx, self._prev_tel, tel, intr)

        boxes = np.array([d.bbox_px for d in detections], dtype=np.float32).reshape(len(detections), 4)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        ids = self.backend.update(boxes, scores, bundle.rgb, bundle.t_utc)

        self.stats.frames += 1
        self.stats.detections_in += len(detections)
        self._prev_tel = tel
        self._remember_frame_time(bundle.frame_idx, bundle.t_utc)

        touched: list[Track] = []
        clip = bundle.clip_id or self.clip_id
        for i, det in enumerate(detections):
            backend_id = int(ids[i])
            if backend_id < 0:
                continue
            self.stats.detections_associated += 1
            state = self._state.get(backend_id)
            if state is None:
                state = _TrackState(global_id=self._new_global_id())
                self._seed_hidden_hits(state, backend_id, bundle)
                self._state[backend_id] = state
                self._tracks[state.global_id] = Track(track_id=state.global_id, cls=det.cls, clip_id=clip)
                self.stats.tracks_started += 1

            track = self._tracks[state.global_id]
            was_confirmed = track.confirmed
            self._register_hit(state, bundle.t_utc)
            track.confirmed = state.confirmed_at is not None
            if track.confirmed and not was_confirmed:
                self.stats.tracks_confirmed += 1

            fix = self._fix_for(det, bundle, fixes[i] if fixes is not None else None)
            stamped = det if det.frame_idx >= 0 else dataclasses.replace(det, frame_idx=bundle.frame_idx)
            track.observations.append(
                Observation(
                    track_id=state.global_id,
                    frame_idx=bundle.frame_idx,
                    t_utc=bundle.t_utc,
                    det=stamped,
                    fix=fix,
                    clip_id=clip,
                    pass_id=self.cfg.pass_id,
                )
            )
            if track.confirmed:
                touched.append(track)

        self._prune(bundle.t_utc)
        return touched

    # --- confirmation gate (§5.6 rule 3) ------------------------------------------------------------------
    def _register_hit(self, state: _TrackState, t_utc: float) -> None:
        """3 hits inside 2 s. Sticky: a confirmed track does not become unconfirmed when it later goes quiet."""
        state.hits.append(t_utc)
        state.n_hits += 1
        state.last_t = t_utc
        while len(state.hits) > self.cfg.min_hits:
            state.hits.popleft()
        if state.confirmed_at is None and len(state.hits) >= self.cfg.min_hits:
            span = state.hits[-1] - state.hits[0]
            if span <= self.cfg.min_hits_window_s:
                state.confirmed_at = t_utc

    def _seed_hidden_hits(self, state: _TrackState, backend_id: int, bundle: FrameBundle) -> None:
        """Recover the hits the backend made before it named the track, so the 2-second window is real.

        `trackers` spawns a tracklet unnamed and allocates its id on the **next** match, so exactly one hit —
        the one that created the track — happens before the wrapper can see anything. That hit is not optional
        bookkeeping: without it a track would need FOUR detections to clear a THREE-hit gate, and every recall
        number downstream would be quietly pessimistic.

        `age` counts `predict()` calls since creation, so the creation frame is `frame_idx - age` and its
        timestamp is in `_frame_times`. A backend that cannot report age gets no seed and the gate simply
        becomes one hit stricter — stated here rather than hidden.
        """
        age = getattr(self.backend, "track_age_frames", lambda _id: None)(backend_id)
        hits = getattr(self.backend, "track_hits", lambda _id: None)(backend_id)
        if not age or age <= 0:
            return
        n_hidden = 1 if hits is None else max(0, int(hits) - 1)
        if n_hidden <= 0:
            return
        created_idx = bundle.frame_idx - int(age)
        t_created = self._frame_times.get(created_idx, bundle.t_utc - age / self.cfg.fps)
        for _ in range(min(n_hidden, self.cfg.min_hits)):
            state.hits.append(t_created)
            state.n_hits += 1

    def _remember_frame_time(self, frame_idx: int, t_utc: float) -> None:
        self._frame_times[frame_idx] = t_utc
        if len(self._frame_times) > self._frame_times_keep:
            for k in sorted(self._frame_times)[: len(self._frame_times) - self._frame_times_keep]:
                del self._frame_times[k]

    # --- helpers ------------------------------------------------------------------------------------------
    def _new_global_id(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def _fix_for(self, det: Detection, bundle: FrameBundle, supplied: GeoFix | None) -> GeoFix:
        if supplied is not None:
            return supplied
        if self._geolocate is not None:
            return self._geolocate(det, bundle)
        return unlocated_fix(bundle.telemetry)

    def _prune(self, t_utc: float) -> None:
        """Drop never-confirmed tracks that have gone quiet, in the wrapper and in the backend.

        This removes *tracks*, which are working state. Guardrail R10 is about **records** (`sightline/dedup/`),
        and nothing here touches one.
        """
        max_age = self.cfg.unconfirmed_prune_s
        if max_age is None:
            return
        drop = [
            backend_id
            for backend_id, st in self._state.items()
            if st.confirmed_at is None and (t_utc - st.last_t) > max_age
        ]
        for backend_id in drop:
            st = self._state.pop(backend_id)
            self._tracks.pop(st.global_id, None)
            self.stats.tracks_pruned_unconfirmed += 1
        keep = {bid for bid, st in self._state.items() if st.confirmed_at is not None}
        self.backend.prune_unconfirmed(keep, max_age)

    # --- output -------------------------------------------------------------------------------------------
    def confirmed_tracks(self) -> list[Track]:
        """Every track that has passed the §5.6 rule 3 gate so far, in id order. This is dedup's input."""
        return [t for _, t in sorted(self._tracks.items()) if t.confirmed]

    def pending_tracks(self) -> list[Track]:
        """Tracks that exist but have not yet made 3 hits in 2 s. Never emitted; useful when debugging recall."""
        return [t for _, t in sorted(self._tracks.items()) if not t.confirmed]

    def close(self) -> list[Track]:
        """Finish the pass and hand the confirmed tracks over."""
        return self.confirmed_tracks()

    # --- observability ------------------------------------------------------------------------------------
    @property
    def cmc_log(self) -> list[CmcResult]:
        return self.cmc.history if self.cmc is not None else []

    def cmc_source_counts(self) -> dict[str, int]:
        """§5.6 rule 4: "make the switch between CMC sources observable"."""
        return self.cmc.source_counts() if self.cmc is not None else {}

    def describe(self) -> dict[str, object]:
        d = dict(self.cfg.describe())
        d["backend_name"] = self.backend.name
        d["backend_licence"] = getattr(self.backend, "licence", "unknown")
        buf = getattr(self.backend, "lost_track_buffer_frames", None)
        if buf is not None:
            d["backend_lost_buffer_frames"] = buf
            d["backend_lost_buffer_s"] = self.backend.lost_track_buffer_s
        d["cmc_sources"] = self.cmc_source_counts()
        d["stats"] = dataclasses.asdict(self.stats)
        return d
