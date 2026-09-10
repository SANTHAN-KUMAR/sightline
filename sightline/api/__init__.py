"""F15/F18 backend: FastAPI + WebSocket over the F18 record log (SOLUTION_DOC §5.9, §5.10).

Guardrail R10: there is no delete route. ``POST /api/records/{id}/dismiss`` requires a reason and keeps the
row; every version stays reachable at ``/api/records/{id}/history``.
"""

from sightline.api.app import REPO, create_app
from sightline.api.coverage_feed import OVERLAY_CONTRACT, grid_to_overlay, read_overlay
from sightline.api.detect_fallback import DETECT_CONTRACT, DetectorPlugin, FrameMerger, StubDetector
from sightline.api.live import MAX_MESSAGE_BYTES, MESSAGE_TYPES, LiveHub, envelope
from sightline.api.mission_feed import MissionState, nadir_footprint
from sightline.api.wire import feature_to_record

__all__ = [
    "create_app",
    "REPO",
    "LiveHub",
    "envelope",
    "MESSAGE_TYPES",
    "MAX_MESSAGE_BYTES",
    "MissionState",
    "nadir_footprint",
    "StubDetector",
    "DetectorPlugin",
    "FrameMerger",
    "DETECT_CONTRACT",
    "OVERLAY_CONTRACT",
    "grid_to_overlay",
    "read_overlay",
    "feature_to_record",
]
