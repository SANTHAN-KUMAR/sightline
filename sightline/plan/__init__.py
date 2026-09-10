"""F2 / F2b — search patterns, constraints and the decision planner.

Owned by lane B6 (`docs/CONTRACTS.md`). Consumes `CoverageGrid` (via `sightline.coverage.CoverageMap`) and a
probability-of-area raster; produces waypoint lists in the scenario's local frame **and** lat/lon (see
`sightline/plan/waypoints.py` for the exact route JSON the mission lane flies).

    F2   patterns.boustrophedon_route / expanding_square_route / orbit_route, constraints.Constraints,
         revisit.RevisitQueue
    F2b  planner.koopman_allocation (Layer 1) and planner.DecisionPlanner (Layer 2)

The planner degrades to the plain pattern under a flat prior — that is a tested property, not a fallback branch.
"""

from sightline.plan.constraints import Battery, Constraints
from sightline.plan.patterns import (
    DEFAULT_SIDE_OVERLAP,
    PatternGeometry,
    PatternSpec,
    altitude_for_min_px,
    bearing_deg,
    boustrophedon_lines,
    boustrophedon_route,
    expanding_square_route,
    gimbal_yaw_for_heading,
    line_spacing_m,
    orbit_route,
    polygon_area_m2,
    polygon_centroid_ne,
    principal_axis_heading,
    rect_polygon,
    speed_limit_ms,
    sweep_width_m,
    swath_m,
)
from sightline.plan.planner import (
    ALTITUDE_BANDS_M,
    CONFIRM_AGL_M,
    Candidate,
    Decision,
    DecisionPlanner,
    PendingRecord,
    PlannerState,
    allocate_segment_minutes,
    koopman_allocation,
)
from sightline.plan.revisit import RevisitItem, RevisitQueue, from_coverage, from_records
from sightline.plan.segments import (
    AssignmentRecord,
    Segment,
    assignment_record,
    auto_segments,
    nearest_segment,
    recommend,
    segment_of_point,
    segments_geojson,
)
from sightline.plan.waypoints import Route, TerrainSampler, Waypoint, make_waypoint

__all__ = [
    "Route", "Waypoint", "make_waypoint", "TerrainSampler",
    "boustrophedon_route", "boustrophedon_lines", "expanding_square_route", "orbit_route",
    "sweep_width_m", "swath_m", "line_spacing_m", "speed_limit_ms", "altitude_for_min_px",
    "gimbal_yaw_for_heading", "principal_axis_heading", "bearing_deg", "rect_polygon", "polygon_area_m2",
    "polygon_centroid_ne", "PatternSpec", "PatternGeometry", "DEFAULT_SIDE_OVERLAP",
    "Constraints", "Battery",
    "Segment", "auto_segments", "assignment_record", "AssignmentRecord", "recommend", "segments_geojson",
    "segment_of_point", "nearest_segment",
    "koopman_allocation", "allocate_segment_minutes", "DecisionPlanner", "PlannerState", "Decision",
    "Candidate", "PendingRecord", "ALTITUDE_BANDS_M", "CONFIRM_AGL_M",
    "RevisitQueue", "RevisitItem", "from_coverage", "from_records",
]
