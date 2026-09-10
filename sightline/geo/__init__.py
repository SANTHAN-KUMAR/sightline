"""F13 geolocation (SOLUTION_DOC §5.7): a detected pixel -> lat/lon with a per-pixel error radius.

Public API other lanes call
---------------------------
    from sightline.geo import ChainConfig, project_detection, project_pixel, h_acc_m, dedup_radius_m

    cfg = ChainConfig(method="flat_plane")                 # or "dem" / "water_plane"
    fix = project_detection(det, telemetry, intrinsics, cfg)   # -> schemas.GeoFix
    if fix.valid:
        radius_m = dedup_radius_m(fix.h_acc_m)             # §5.6 DBSCAN eps = 2 x CE90

Simulator runs must inject the §5.7 noise model first, or the measured error is ~0 and the dedup radius is
tuned wrong for real footage::

    from sightline.geo import NoiseConfig, TelemetryNoise
    noise = TelemetryNoise(NoiseConfig(seed=7))            # NoiseConfig.off() for debugging
    noisy = noise.apply(truth_telemetry)                   # truth kept on noise.samples
"""

from sightline.geo.budget import (  # noqa: F401
    CONSUMER,
    CONSUMER_DEM_AGL,
    DOC_TABLE,
    KNOWN_DOC_DISCREPANCIES,
    MAX_OFF_NADIR_DEG,
    PESSIMISTIC,
    PRESETS,
    RTK,
    BudgetInputs,
    ce90_m,
    compare_to_doc,
    dedup_radius_m,
    dominant_term,
    error_terms,
    format_doc_comparison,
    h_acc_m,
)
from sightline.geo.chain import (  # noqa: F401
    AIRSIM_EARTH_RADIUS_M,
    FLOODVALLEY_ORIGIN,
    REJECTED_H_ACC_M,
    ChainConfig,
    OriginGeopoint,
    airsim_ned_to_geodetic,
    boresight_ned,
    effective_hfov_deg,
    error_breakdown,
    footprint_ned,
    geodetic_from_ned,
    gsd_cm_px,
    hfov_ambiguity_m,
    hfov_from_dfov_deg,
    intrinsics_from_fov,
    load_scene_origin,
    ned_from_geodetic,
    off_nadir_deg,
    optical_to_frd,
    pixel_ray_ned,
    project_detection,
    project_detections,
    project_pixel,
    project_pixel_ne,
    ray_optical,
)
from sightline.geo.dem import DEFAULT_DEM_PATH, DemSampler, load_default_dem  # noqa: F401
from sightline.geo.noise import NoiseConfig, NoisySample, TelemetryNoise, inject  # noqa: F401

__all__ = [
    # chain
    "ChainConfig",
    "project_pixel",
    "project_pixel_ne",
    "project_detection",
    "project_detections",
    "pixel_ray_ned",
    "ray_optical",
    "optical_to_frd",
    "boresight_ned",
    "off_nadir_deg",
    "footprint_ned",
    "gsd_cm_px",
    "error_breakdown",
    "REJECTED_H_ACC_M",
    # intrinsics / FOV
    "intrinsics_from_fov",
    "hfov_from_dfov_deg",
    "effective_hfov_deg",
    "hfov_ambiguity_m",
    # scenario anchoring
    "OriginGeopoint",
    "FLOODVALLEY_ORIGIN",
    "AIRSIM_EARTH_RADIUS_M",
    "geodetic_from_ned",
    "ned_from_geodetic",
    "airsim_ned_to_geodetic",
    "load_scene_origin",
    # budget
    "BudgetInputs",
    "CONSUMER",
    "RTK",
    "PESSIMISTIC",
    "CONSUMER_DEM_AGL",
    "PRESETS",
    "error_terms",
    "h_acc_m",
    "ce90_m",
    "dedup_radius_m",
    "dominant_term",
    "MAX_OFF_NADIR_DEG",
    "DOC_TABLE",
    "KNOWN_DOC_DISCREPANCIES",
    "compare_to_doc",
    "format_doc_comparison",
    # DEM
    "DemSampler",
    "DEFAULT_DEM_PATH",
    "load_default_dem",
    # sim noise
    "NoiseConfig",
    "NoisySample",
    "TelemetryNoise",
    "inject",
]
