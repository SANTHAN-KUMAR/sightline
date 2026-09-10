"""F16 / F16b — the search-quality map: per-cell accumulated effort and probability of detection.

    C_j(cell)   = sum over passes of  R_slice(j; GSD, band, time, blur, view) * V(cell, j)
    POD_j(cell) = 1 - exp(-k_j * C_j(cell)),  clamped below 1, zero inside every `cannot_clear` cell
    POD_eff     = sum over j of w_j(zone) * POD_j          (the default view, §5.3b)

Owned by lane B6 (`docs/CONTRACTS.md`). Consumes `Telemetry` + `Intrinsics`, produces `CoverageGrid`.
Nothing in this package has a "cleared" state (guardrail R10), and no amount of effort raises the POD of a cell
inside a burial polygon (§2.7).

Typical use::

    scene = SceneFrame.from_json("data/scene/flood_valley.json")
    cmap  = CoverageMap.for_scene(scene, cell_m=10.0, presentations=("body", "limb_only"))
    cmap.add_burial_polygons([deposit_fan_poly_ne])
    for tel, intr in pass_frames:
        cmap.add_frame(tel, intr, pass_id=0)
    cmap.end_pass()
    export_coverage(cmap, "_artifacts/coverage")
"""

from sightline.coverage.accumulate import (
    ZONE_CODE,
    ZONE_NAMES,
    CoverageMap,
    FrameCoverage,
    PassSummary,
    conditions_from_telemetry,
    zone_raster_from_scene,
)
from sightline.coverage.calibrate import (
    DEFAULT_K,
    K_MAX,
    K_MIN,
    POD_MAX,
    KValue,
    calibrate_k_from_outcomes,
    expected_calibration_error,
    k_for,
    k_for_reference_quality,
    reliability_diagram,
    wilson_interval,
)
from sightline.coverage.export import (
    POD_BANDS,
    POD_RAMP,
    banded_geojson,
    export_coverage,
    manifest,
    overlay_payload,
    overlay_rgba,
)
from sightline.coverage.footprint import Footprint, gimbal_quat, ground_footprint, swath_m
from sightline.coverage.grid import (
    SceneFrame,
    Window,
    bounds_latlon,
    cell_centres_m,
    cell_of_latlon,
    grid_ne_to_latlon,
    latlon_to_grid_ne,
    make_grid,
    make_grid_for_scene,
    points_in_polygon,
    polygon_cell_weights,
    rasterise_polygons,
)
from sightline.coverage.presentation import (
    CRITICAL_DIM_M,
    LEGAL_CEILING_M,
    MIN_PX_FOR_RECALL,
    MVP_PRESENTATIONS,
    SIM_RGB_4K,
    ZONE_MIX,
    CameraModel,
    altitude_ceiling_table,
    mix_for_zone,
    update_mix_from_observations,
)
from sightline.coverage.prior import (
    LastKnownPosition,
    PriorLayers,
    bayesian_update,
    build_prior,
    probability_of_success,
)
from sightline.coverage.quality import (
    Conditions,
    RecallEstimate,
    SliceTable,
    altitude_band,
    analytic_recall,
    recall_from_px,
)

__all__ = [
    "CoverageMap", "FrameCoverage", "PassSummary", "conditions_from_telemetry", "zone_raster_from_scene",
    "ZONE_CODE", "ZONE_NAMES",
    "DEFAULT_K", "KValue", "POD_MAX", "K_MIN", "K_MAX", "k_for", "k_for_reference_quality",
    "calibrate_k_from_outcomes", "reliability_diagram", "expected_calibration_error", "wilson_interval",
    "export_coverage", "overlay_payload", "manifest", "banded_geojson", "overlay_rgba", "POD_RAMP", "POD_BANDS",
    "Footprint", "ground_footprint", "gimbal_quat", "swath_m",
    "SceneFrame", "Window", "make_grid", "make_grid_for_scene", "cell_centres_m", "bounds_latlon",
    "grid_ne_to_latlon", "latlon_to_grid_ne", "cell_of_latlon", "points_in_polygon", "polygon_cell_weights",
    "rasterise_polygons",
    "CameraModel", "SIM_RGB_4K", "CRITICAL_DIM_M", "MVP_PRESENTATIONS", "MIN_PX_FOR_RECALL", "LEGAL_CEILING_M",
    "ZONE_MIX", "mix_for_zone", "update_mix_from_observations", "altitude_ceiling_table",
    "build_prior", "bayesian_update", "probability_of_success", "LastKnownPosition", "PriorLayers",
    "Conditions", "SliceTable", "RecallEstimate", "analytic_recall", "recall_from_px", "altitude_band",
]
