"""F11 tracking: BoT-SORT tuned for a static survivor under a moving camera (SOLUTION_DOC §5.6).

    from sightline.track import Tracker, TrackerConfig
    tracker = Tracker(TrackerConfig(fps=5.0), intrinsics=bundle.intrinsics)
    live = tracker.update(detections, bundle, fixes)   # -> list[Track], confirmed only
    tracks = tracker.close()                           # -> the pass's tracks, for sightline.dedup

Owned by lane B3 (see docs/CONTRACTS.md). Consumes `Detection` + `FrameBundle` + `GeoFix`, produces `Track`.
"""

from sightline.track.backends import RoboflowBoTSORT, TrackerBackend, UltralyticsBoTSORT, make_backend
from sightline.track.cmc import CmcResult, CmcSource, SightlineCMC
from sightline.track.config import TUNING_ORDER, TrackerConfig
from sightline.track.geometry import (
    affine_disagreement_px,
    affine_from_homography,
    ground_plane_homography,
    rotation_optical_to_ned,
    telemetry_affine,
)
from sightline.track.tracker import Tracker, TrackerStats, unlocated_fix

__all__ = [
    "TUNING_ORDER",
    "CmcResult",
    "CmcSource",
    "RoboflowBoTSORT",
    "SightlineCMC",
    "Tracker",
    "TrackerBackend",
    "TrackerConfig",
    "TrackerStats",
    "UltralyticsBoTSORT",
    "affine_disagreement_px",
    "affine_from_homography",
    "ground_plane_homography",
    "make_backend",
    "rotation_optical_to_ned",
    "telemetry_affine",
    "unlocated_fix",
]
