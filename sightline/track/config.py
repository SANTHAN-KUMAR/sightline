"""SOLUTION_DOC §5.6 "static-survivor tuning", written as named parameters instead of magic numbers.

The whole point of §5.6 is that **survivors are static and the camera moves** — the opposite of the pedestrian
assumption every off-the-shelf tracker ships tuned for. Three consequences drive every default in this file:

1. A still target under a moving camera only looks still *after* camera-motion compensation, so CMC is not an
   optional extra here — it is the feature. See `sightline/track/cmc.py`.
2. Frames arrive **decimated** (2-10 FPS), so every buffer expressed "in frames" has to be restated in processed
   frames or it silently means the wrong number of seconds. See `TrackerConfig.track_buffer_frames`.
3. A 20-60 px blob flickers. The second-stage low-confidence association (`track_low_thresh` = 0.1, `fuse_score`
   on) is what keeps it alive; the confirmation gate (3 hits within 2 s) is what stops the resulting false
   positives from reaching the operator.

Threshold conventions differ between implementations and this is the single easiest thing to get wrong:

* BoT-SORT / ByteTrack `match_thresh` is a **maximum linear-assignment cost**, cost = 1 - fused_IoU. Lowering it
  (0.8 -> 0.6) demands a *higher* IoU (0.2 -> 0.4), i.e. a **stricter** match.
* `roboflow/trackers` takes the same knob as a **minimum similarity** (`minimum_iou_threshold_*`).

`TrackerConfig` stores the BoT-SORT cost convention (so the doc's numbers can be read straight off the page) and
converts on the way into whichever backend is used. `similarity_first_assoc` etc. do the conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

CmcMethod = Literal["sparseOptFlow", "orb", "sift", "ecc"]
BackendName = Literal["trackers", "ultralytics"]

#: The order §5.6 rule 5 says to tune in when identities still flip. Read top to bottom; stop at the first fix.
TUNING_ORDER: tuple[str, ...] = (
    "1. verify CMC is actually running and which source it used (Tracker.cmc_source_counts()) — a silent "
    "fallback to identity looks exactly like a tracker problem",
    "2. lower match_thresh 0.8 -> 0.6 (stricter IoU: 0.2 -> 0.4) so a track cannot latch onto a neighbour",
    "3. only then raise track_buffer_s / relax the confirmation window",
    "4. detector quality dominates: on VisDrone-MOT/UAVDT tracker choice moves HOTA ~1 point (§5.6 rule 6). "
    "Gate confirmation rather than tuning association endlessly.",
)


@dataclass(slots=True)
class TrackerConfig:
    """Every §5.6 tuning knob, with the doc's value as the default and the reason in the comment.

    Nothing here is a magic number: each field cites the rule it comes from.
    """

    # --- frame rate and buffers (§5.6 rule 1) ---------------------------------------------------------------
    #: Frames per second the tracker actually sees, i.e. AFTER decimation. Not the video's native rate.
    fps: float = 5.0
    #: How long a survivor may be missed before the track is dropped, in SECONDS. The doc states the buffer as
    #: "5 FPS x 30 s = 150" processed frames; seconds are the unit that survives a change of `fps`.
    track_buffer_s: float = 30.0

    # --- association thresholds (§5.6 rules 2 and 5; BoT-SORT COST convention, see module docstring) --------
    #: Detections at or above this confidence start the first association stage and may spawn new tracks.
    track_high_thresh: float = 0.50
    #: §5.6 rule 2. Detections between `track_low_thresh` and `track_high_thresh` get the ByteTrack second-stage
    #: match, which is what rescues a survivor whose confidence dipped for a few frames.
    track_low_thresh: float = 0.10
    #: Minimum confidence to spawn a NEW track. Deliberately low (§5.5c step 4: "run a low confidence threshold
    #: and recover precision downstream") — the confirmation gate and geo-dedup pay for the false positives.
    new_track_thresh: float = 0.25
    #: §5.6 rule 5. Max assignment cost in the first stage; similarity required = 1 - match_thresh.
    match_thresh: float = 0.80
    #: Second (low-confidence) stage. BoT-SORT's own default; cost and similarity coincide at 0.5.
    second_match_thresh: float = 0.50
    #: Unconfirmed-track stage. ByteTrack hardcodes cost 0.7 (similarity 0.3).
    unconfirmed_match_thresh: float = 0.70
    #: §5.6 rule 2. Multiply IoU by detection score before assignment. Both shipped backends do this always;
    #: the flag exists so a backend that cannot is rejected loudly rather than silently behaving differently.
    fuse_score: bool = True
    #: §5.6 "Rejected": at 20-60 px an appearance embedding is mostly noise. Never turn this on without an A/B.
    with_reid: bool = False

    # --- confirmation gate (§5.6 rule 3) --------------------------------------------------------------------
    #: A track is emitted only after this many hits ...
    min_hits: int = 3
    #: ... falling inside this many seconds. The time window is the half of the rule no backend can express, so
    #: `Tracker` owns it (see `backend_min_consecutive_frames`).
    min_hits_window_s: float = 2.0
    #: What the BACKEND is told. 1 means "give every track an id immediately" so the wrapper can see the first
    #: hits and time the window itself. Raising it hides the early hits and makes the window unmeasurable.
    backend_min_consecutive_frames: int = 1
    #: Backend tracklets that failed the confirmation window are dropped after this long without an update, so a
    #: low detector threshold cannot fill the association matrix with ghosts. `None` disables the prune.
    unconfirmed_prune_s: float | None = 2.0

    # --- camera motion compensation (§5.6 rule 4) -----------------------------------------------------------
    cmc_enabled: bool = True
    cmc_method: CmcMethod = "sparseOptFlow"
    #: §5.6 rule 4: "run sparseOptFlow on a 640-px-wide downscale; ORB/ECC on 4K is far too slow."
    cmc_target_width_px: int = 640
    #: §5.6 rule 4 fallback. Rippling water produces a coherent but WRONG flow field, so an inlier count alone
    #: will not catch it: when the flow-estimated transform disagrees with the telemetry-predicted one by more
    #: than this many pixels (measured at the frame corners), the telemetry homography wins.
    cmc_max_disagreement_px: float = 12.0
    #: Below this the two agree closely enough that optical flow (which sees real parallax) is preferred.
    cmc_telemetry_enabled: bool = True

    # --- bookkeeping ------------------------------------------------------------------------------------
    #: Which coverage pass these tracks belong to; copied onto every `Observation` and accumulated by dedup.
    pass_id: int = 0
    #: First track id this instance hands out. Give each pass its own block so ids never collide in dedup.
    track_id_offset: int = 0
    backend: BackendName = "trackers"
    #: Keep the per-frame CMC decision log (§5.6 rule 4: "make the switch observable"). Bounded by `cmc_log_max`.
    cmc_log: bool = True
    cmc_log_max: int = 20000

    # populated by __post_init__; do not set by hand
    _checked: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive: the buffer is expressed in seconds and converted with it")
        if not 0.0 <= self.track_low_thresh < self.track_high_thresh <= 1.0:
            raise ValueError("need 0 <= track_low_thresh < track_high_thresh <= 1")
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if self.min_hits_window_s <= 0:
            raise ValueError("min_hits_window_s must be positive")
        if self.min_hits > 1 and (self.min_hits - 1) / self.fps > self.min_hits_window_s:
            raise ValueError(
                f"unreachable confirmation gate: {self.min_hits} hits cannot fall inside "
                f"{self.min_hits_window_s} s at {self.fps} FPS (needs >= {(self.min_hits - 1) / self.fps:.2f} s)"
            )
        self._checked = True

    # --- derived quantities -----------------------------------------------------------------------------
    @property
    def track_buffer_frames(self) -> int:
        """§5.6 rule 1: the buffer in PROCESSED frames. 5 FPS x 30 s = 150, not the default 30."""
        return max(1, int(round(self.fps * self.track_buffer_s)))

    @property
    def min_hits_window_frames(self) -> int:
        """The confirmation window in processed frames (2 s at 5 FPS = 10)."""
        return max(1, int(math.ceil(self.fps * self.min_hits_window_s)))

    def roboflow_lost_track_buffer(self) -> int:
        """`roboflow/trackers` states `lost_track_buffer` **in 30-FPS frames** and rescales it by `frame_rate`.

        It computes `max(1, ceil(frame_rate / 30 * lost_track_buffer))` internally, so handing it our 150
        processed frames would give `ceil(5/30 * 150)` = 25 frames = 5 s — a fifth of the intended buffer. This
        is exactly the unit trap §5.6 rule 1 warns about, one level further down. Convert instead.
        """
        return max(1, int(round(self.track_buffer_frames * 30.0 / self.fps)))

    @property
    def similarity_first_assoc(self) -> float:
        """`minimum_iou_threshold_first_assoc` for a similarity-convention backend."""
        return 1.0 - self.match_thresh

    @property
    def similarity_second_assoc(self) -> float:
        return 1.0 - self.second_match_thresh

    @property
    def similarity_unconfirmed_assoc(self) -> float:
        return 1.0 - self.unconfirmed_match_thresh

    def cmc_downscale(self, frame_width_px: int) -> int:
        """Integer downscale factor that lands closest to `cmc_target_width_px` without going below it.

        4K (3840) -> 6 -> 640 px. 1080p (1920) -> 3 -> 640 px. 640 -> 1. A frame already narrower than the
        target is never upscaled.
        """
        if frame_width_px <= self.cmc_target_width_px:
            return 1
        return max(1, int(frame_width_px // self.cmc_target_width_px))

    def describe(self) -> dict[str, object]:
        """Flat dict of everything a run should record so a result can be reproduced (§5.12)."""
        return {
            "fps": self.fps,
            "track_buffer_s": self.track_buffer_s,
            "track_buffer_frames": self.track_buffer_frames,
            "roboflow_lost_track_buffer": self.roboflow_lost_track_buffer(),
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "match_thresh_cost": self.match_thresh,
            "match_similarity": self.similarity_first_assoc,
            "fuse_score": self.fuse_score,
            "with_reid": self.with_reid,
            "min_hits": self.min_hits,
            "min_hits_window_s": self.min_hits_window_s,
            "cmc_enabled": self.cmc_enabled,
            "cmc_method": self.cmc_method,
            "cmc_target_width_px": self.cmc_target_width_px,
            "cmc_max_disagreement_px": self.cmc_max_disagreement_px,
            "backend": self.backend,
            "pass_id": self.pass_id,
        }
