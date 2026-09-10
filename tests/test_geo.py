"""F13 geolocation tests (SOLUTION_DOC §5.7). Offline, CPU/numpy only, no editor and no GPU.

Every assertion is either an analytic identity (a nadir camera must see the point directly below it) or a
number from the doc. Nothing here is weakened to make it pass: where this implementation disagrees with the
doc's printed error-budget table, the test asserts the *derived* value and points at
`budget.KNOWN_DOC_DISCREPANCIES`, which carries the derivation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sightline.common.geodesy import euler_to_quat, ne_between, quat_to_euler
from sightline.geo import (
    CONSUMER,
    DOC_TABLE,
    KNOWN_DOC_DISCREPANCIES,
    PESSIMISTIC,
    REJECTED_H_ACC_M,
    RTK,
    ChainConfig,
    DemSampler,
    NoiseConfig,
    OriginGeopoint,
    TelemetryNoise,
    airsim_ned_to_geodetic,
    boresight_ned,
    ce90_m,
    compare_to_doc,
    dedup_radius_m,
    dominant_term,
    effective_hfov_deg,
    error_terms,
    footprint_ned,
    gsd_cm_px,
    h_acc_m,
    hfov_ambiguity_m,
    hfov_from_dfov_deg,
    intrinsics_from_fov,
    load_default_dem,
    load_scene_origin,
    off_nadir_deg,
    optical_to_frd,
    pixel_ray_ned,
    project_detection,
    project_pixel,
    project_pixel_ne,
    ray_optical,
)
from sightline.geo.budget import MAX_OFF_NADIR_DEG
from sightline.schemas import CE90_FACTOR, Detection, GeoFix, Intrinsics, Telemetry

# --- fixtures ------------------------------------------------------------------------------------------------
LAT0, LON0 = 11.4870, 76.1450  # FloodValley OriginGeopoint (docs/CONTEXT.md §5)
GROUND_ASL = 1046.0069580078125


def cam(hfov_deg: float = 73.7398, w: int = 3840, h: int = 2160, dist: tuple[float, ...] = ()) -> Intrinsics:
    """A 4K camera whose HFOV makes fx exactly 2560 px — the budget's reference focal length."""
    return intrinsics_from_fov(w, h, hfov_deg, dist=dist)


def tele(
    *,
    agl: float = 60.0,
    pitch: float = -90.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    lat: float = LAT0,
    lon: float = LON0,
    ground_asl: float = GROUND_ASL,
    **kw,
) -> Telemetry:
    """Ground-truth telemetry for a gimbal at (roll, pitch, yaw); pitch -90 = nadir."""
    return Telemetry(
        t_utc=1_780_000_000.0,
        lat=lat,
        lon=lon,
        alt_msl_m=ground_asl + agl,
        agl_m=agl,
        q_body=euler_to_quat(0.0, 0.0, yaw),
        q_gimbal=euler_to_quat(roll, pitch, yaw),
        gimbal_is_earth_referenced=True,
        **kw,
    )


def ne_of(tel: Telemetry, fix: GeoFix) -> tuple[float, float]:
    return ne_between(tel.lat, tel.lon, fix.lat, fix.lon)


# =============================================================================================================
# 1. Intrinsics and the 4K-crop ambiguity (§5.7 step 1)
# =============================================================================================================
def test_intrinsics_from_explicit_hfov_matches_the_schema_helper():
    intr = cam()
    assert intr.fx == pytest.approx(2560.0, abs=1e-3)
    assert intr.fx == pytest.approx(Intrinsics.from_hfov(3840, 2160, 73.7398).fx, rel=1e-12)
    assert intr.hfov_deg() == pytest.approx(73.7398, abs=1e-9)
    assert (intr.cx, intr.cy) == (1920.0, 1080.0)


def test_intrinsics_requires_an_explicit_effective_hfov():
    with pytest.raises(ValueError):
        intrinsics_from_fov(3840, 2160, 0.0)
    with pytest.raises(ValueError):
        intrinsics_from_fov(3840, 2160, 181.0)


def test_dfov_to_hfov_and_the_4k_crop_ambiguity():
    """DJI publishes a DIAGONAL FOV for the 4:3 sensor. Whether the 16:9 video keeps the full sensor width
    changes the effective HFOV, and the doc warns about exactly this. Quantify the cost of guessing."""
    dfov = 84.0  # a typical DJI wide lens
    hfov_43 = hfov_from_dfov_deg(dfov, 4.0, 3.0)
    hfov_169_native = hfov_from_dfov_deg(dfov, 16.0, 9.0)
    assert hfov_43 == pytest.approx(71.532, abs=1e-3)
    assert hfov_169_native == pytest.approx(76.248, abs=1e-3)

    # a full-width (vertical) crop of the 4:3 sensor keeps the width, so the HFOV is the 4:3 one
    assert effective_hfov_deg(dfov, mode="full_width") == pytest.approx(hfov_43, abs=1e-9)
    # "native" means the DFOV already describes the 16:9 frame
    assert effective_hfov_deg(dfov, mode="native") == pytest.approx(hfov_169_native, abs=1e-9)
    # a 2x digital crop halves the retained width
    assert effective_hfov_deg(dfov, mode="full_width", zoom=2.0) == pytest.approx(39.614, abs=1e-3)
    # cropping the sides only makes sense when the video is NARROWER than the sensor
    assert effective_hfov_deg(
        dfov, native_aspect=(16.0, 9.0), video_aspect=(4.0, 3.0), mode="full_height"
    ) == pytest.approx(60.960, abs=1e-3)
    with pytest.raises(ValueError):  # a crop can never widen the lens
        effective_hfov_deg(dfov, mode="full_height")
    with pytest.raises(ValueError):
        effective_hfov_deg(dfov, mode="guess")

    # cost of guessing wrong, at the mid-right frame edge from 60 m nadir
    err = hfov_ambiguity_m(3840, 2160, hfov_43, hfov_169_native, 60.0)
    assert err == pytest.approx(3.87, abs=0.05)  # bigger than the whole GNSS budget: hence the doc's day-1 test
    assert hfov_ambiguity_m(3840, 2160, hfov_43, hfov_43, 60.0) == pytest.approx(0.0)


# =============================================================================================================
# 2. The ray (§5.7 steps 2-4)
# =============================================================================================================
def test_optical_ray_and_frd_reorder():
    intr = cam()
    d = ray_optical(intr.cx, intr.cy, intr)
    assert np.allclose(d, [0.0, 0.0, 1.0])
    assert np.allclose(optical_to_frd(np.array([1.0, 2.0, 3.0])), [3.0, 1.0, 2.0])
    # a pixel to the right is +x optical -> +y (right) in FRD
    d2 = optical_to_frd(ray_optical(intr.cx + 500, intr.cy, intr))
    assert d2[1] > 0 and d2[2] == pytest.approx(0.0)


def test_nadir_boresight_points_straight_down_for_every_heading():
    for yaw in (0.0, 37.0, 90.0, 180.0, -120.0):
        b = boresight_ned(tele(yaw=yaw))
        assert np.allclose(b, [0.0, 0.0, 1.0], atol=1e-12)
        assert off_nadir_deg(b) == pytest.approx(0.0, abs=1e-9)


def test_optical_gimbal_convention_is_supported_too():
    """`schemas.Telemetry` documents q_gimbal as optical->NED; `euler_to_quat` produces FRD->NED. Both work."""
    intr = cam()
    frd = tele(pitch=-90.0)
    cfg_frd = ChainConfig(gimbal_frame="frd")
    # identity quaternion in the OPTICAL convention is also nadir (optical +z = forward -> NED down)
    optical = Telemetry(t_utc=0.0, lat=LAT0, lon=LON0, alt_msl_m=GROUND_ASL + 60, agl_m=60.0)
    cfg_opt = ChainConfig(gimbal_frame="optical")
    a = project_pixel(intr.cx, intr.cy, frd, intr, cfg_frd)
    b = project_pixel(intr.cx, intr.cy, optical, intr, cfg_opt)
    assert a.off_nadir_deg == pytest.approx(0.0, abs=1e-9)
    assert b.off_nadir_deg == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(*ne_between(a.lat, a.lon, b.lat, b.lon)) < 1e-6


def test_body_composition_when_the_gimbal_is_not_earth_referenced():
    """MAVLink can report the gimbal relative to the body; then body attitude must be composed in (step 4)."""
    intr = cam()
    earth = tele(pitch=-45.0, yaw=90.0)  # gimbal already earth-referenced, looking 45 deg down toward east
    body = Telemetry(
        t_utc=0.0,
        lat=LAT0,
        lon=LON0,
        alt_msl_m=GROUND_ASL + 60,
        agl_m=60.0,
        q_body=euler_to_quat(0.0, 0.0, 90.0),  # airframe heading east
        q_gimbal=euler_to_quat(0.0, -45.0, 0.0),  # gimbal 45 deg down relative to the airframe
        gimbal_is_earth_referenced=False,
    )
    a, an, ae = project_pixel_ne(intr.cx, intr.cy, earth, intr)
    b, bn, be = project_pixel_ne(intr.cx, intr.cy, body, intr)
    assert (an, ae) == pytest.approx((bn, be), abs=1e-9)
    assert a.off_nadir_deg == pytest.approx(b.off_nadir_deg, abs=1e-12)
    # and ignoring the body attitude would be wrong: the fix would land north instead of east
    body.gimbal_is_earth_referenced = True  # pretend the flag said "earth-referenced" when it did not
    _, wn, we = project_pixel_ne(intr.cx, intr.cy, body, intr)
    assert (wn, we) == pytest.approx((60.0, 0.0), abs=1e-9)  # 90 deg of heading thrown away
    assert math.hypot(wn - bn, we - be) == pytest.approx(60.0 * math.sqrt(2.0), abs=1e-6)


# =============================================================================================================
# 3. Ground intersection: the analytic cases (§5.7 step 5)
# =============================================================================================================
@pytest.mark.parametrize("agl", [30.0, 60.0, 100.0])
@pytest.mark.parametrize("yaw", [0.0, 47.0, 180.0, -95.0])
def test_nadir_principal_point_projects_to_the_point_directly_below(agl, yaw):
    """The sharpest identity in the whole chain: a nadir camera's centre pixel is the camera's own position."""
    intr = cam()
    tel = tele(agl=agl, yaw=yaw)
    fix, north, east = project_pixel_ne(intr.cx, intr.cy, tel, intr)
    assert fix.valid and fix.method == "flat_plane"
    assert (north, east) == pytest.approx((0.0, 0.0), abs=1e-12)
    n, e = ne_of(tel, fix)
    assert math.hypot(n, e) < 1e-4  # sub-millimetre, let alone sub-centimetre
    assert fix.off_nadir_deg == pytest.approx(0.0, abs=1e-9)
    assert fix.alt_msl_m == pytest.approx(GROUND_ASL, abs=1e-9)
    assert fix.slant_range_m == pytest.approx(agl, abs=1e-12)
    assert fix.agl_m == pytest.approx(agl)


@pytest.mark.parametrize("du", [200.0, 1000.0, 1900.0])
def test_off_nadir_pixel_lands_at_the_analytic_ground_offset(du):
    """For a nadir camera, a pixel du to the right lands exactly h * du / fx to the east (yaw 0). No
    approximation is involved: the ray is (0, du/f, 1) after the FRD re-order, so t = h * |d| and east = h*du/f."""
    intr, agl = cam(), 60.0
    tel = tele(agl=agl)
    fix, north, east = project_pixel_ne(intr.cx + du, intr.cy, tel, intr)
    expect_east = agl * du / intr.fx
    assert east == pytest.approx(expect_east, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)
    assert fix.off_nadir_deg == pytest.approx(math.degrees(math.atan(du / intr.fx)), abs=1e-9)
    assert fix.slant_range_m == pytest.approx(math.hypot(agl, expect_east), abs=1e-9)
    # and the geodesic step preserves it: recovering the offset from lat/lon agrees to well under a centimetre
    n, e = ne_of(tel, fix)
    assert e == pytest.approx(expect_east, abs=1e-3)
    assert abs(n) < 1e-3


def test_off_nadir_pixel_follows_the_camera_heading():
    """Rotating the gimbal's yaw must rotate the ground offset by the same angle, nothing else."""
    intr, agl, du = cam(), 60.0, 1000.0
    r0 = math.hypot(*project_pixel_ne(intr.cx + du, intr.cy, tele(agl=agl, yaw=0.0), intr)[1:])
    for yaw in (0.0, 30.0, 90.0, 217.0):
        _, n, e = project_pixel_ne(intr.cx + du, intr.cy, tele(agl=agl, yaw=yaw), intr)
        assert math.hypot(n, e) == pytest.approx(r0, abs=1e-9)
        assert math.degrees(math.atan2(e, n)) % 360.0 == pytest.approx((yaw + 90.0) % 360.0, abs=1e-9)


def test_oblique_45_principal_ray_lands_one_altitude_away():
    """A -45 deg gimbal puts the centre pixel exactly h metres away along the camera azimuth, at slant h*sqrt(2)."""
    intr, agl = cam(), 60.0
    for yaw, want in ((0.0, (60.0, 0.0)), (90.0, (0.0, 60.0)), (180.0, (-60.0, 0.0))):
        tel = tele(agl=agl, pitch=-45.0, yaw=yaw)
        fix, n, e = project_pixel_ne(intr.cx, intr.cy, tel, intr)
        assert (n, e) == pytest.approx(want, abs=1e-9)
        assert fix.off_nadir_deg == pytest.approx(45.0, abs=1e-9)
        assert fix.slant_range_m == pytest.approx(agl * math.sqrt(2.0), abs=1e-9)


def test_near_horizon_rays_are_rejected_at_dz_0_1():
    """Doc step 5: reject rays with d_z <= 0.1. sin(5.739 deg) = 0.1, so -5 deg rejects and -10 deg does not."""
    intr = cam()
    bad = project_pixel(intr.cx, intr.cy, tele(pitch=-5.0), intr)
    assert not bad.valid and bad.reject_reason == "near_horizon"
    assert bad.h_acc_m == REJECTED_H_ACC_M  # huge and finite: a consumer that ignores `valid` degrades safely
    assert (bad.lat, bad.lon) == (LAT0, LON0)  # falls back to the camera's own position, never NaN
    assert bad.off_nadir_deg == pytest.approx(85.0, abs=1e-9)

    good = project_pixel(intr.cx, intr.cy, tele(pitch=-10.0), intr)
    assert good.valid and good.reject_reason == ""
    assert good.off_nadir_deg == pytest.approx(80.0, abs=1e-9)

    # exactly on the boundary the ray is rejected (the doc's condition is `<=`)
    edge = math.degrees(math.asin(0.1))
    assert not project_pixel(intr.cx, intr.cy, tele(pitch=-edge), intr).valid
    assert project_pixel(intr.cx, intr.cy, tele(pitch=-(edge + 0.01)), intr).valid
    assert MAX_OFF_NADIR_DEG == pytest.approx(90.0 - edge, abs=1e-9)


def test_upward_ray_and_zero_agl_are_rejected_not_crashed():
    intr = cam()
    up = project_pixel(intr.cx, intr.cy, tele(pitch=+45.0), intr)
    assert not up.valid and up.reject_reason == "near_horizon"
    landed = project_pixel(intr.cx, intr.cy, tele(agl=0.0), intr)
    assert not landed.valid and landed.reject_reason == "no_agl"


def test_undistortion_matters_at_the_frame_edge():
    """Doc step 2: skipping Brown-Conrady costs "up to ~1 deg at the frame edge of a wide lens"."""
    dist = (-0.08, 0.015, 0.0, 0.0, 0.0)  # a moderate wide-lens k1,k2,p1,p2,k3
    intr = cam(dist=dist)
    tel = tele(agl=60.0)
    a = pixel_ray_ned(intr.width_px - 1.0, intr.cy, intr, tel, ChainConfig(undistort=True))
    b = pixel_ray_ned(intr.width_px - 1.0, intr.cy, intr, tel, ChainConfig(undistort=False))
    ang = math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
    assert ang == pytest.approx(1.229, abs=0.01), f"edge undistortion angle {ang:.3f} deg"  # the doc's "~1 deg"

    fa = project_pixel(intr.width_px - 1.0, intr.cy, tel, intr, ChainConfig(undistort=True))
    fb = project_pixel(intr.width_px - 1.0, intr.cy, tel, intr, ChainConfig(undistort=False))
    ground = math.hypot(*ne_between(fa.lat, fa.lon, fb.lat, fb.lon))
    assert ground == pytest.approx(2.044, abs=0.02), f"skipping undistortion moves the fix {ground:.2f} m"
    # a stronger lens costs proportionally more: k1 = -0.15 moves the same pixel 4.4 m
    strong = cam(dist=(-0.15, 0.03, 0.0005, -0.0005, 0.0))
    sa = project_pixel(strong.width_px - 1.0, strong.cy, tel, strong, ChainConfig(undistort=True))
    sb = project_pixel(strong.width_px - 1.0, strong.cy, tel, strong, ChainConfig(undistort=False))
    assert math.hypot(*ne_between(sa.lat, sa.lon, sb.lat, sb.lon)) == pytest.approx(4.40, abs=0.05)
    # the centre pixel is unaffected (no radial distortion on the principal point)
    assert project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(undistort=True)).lat == pytest.approx(
        project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(undistort=False)).lat, abs=1e-12
    )


def test_detection_anchor_is_the_box_foot_by_default():
    intr = cam()
    tel = tele(agl=60.0)
    det = Detection(bbox_px=(1900.0, 1000.0, 1940.0, 1160.0), score=0.9)
    foot = project_detection(det, tel, intr, ChainConfig(anchor="foot"))
    centre = project_detection(det, tel, intr, ChainConfig(anchor="centre"))
    assert det.foot_px() == (1920.0, 1160.0) and det.centre_px() == (1920.0, 1080.0)
    assert foot.lat != centre.lat
    # the default anchor is the foot, and it agrees with projecting that pixel directly
    assert project_detection(det, tel, intr).lat == foot.lat
    direct = project_pixel(*det.foot_px(), tel, intr)
    assert ne_of(tel, foot) == pytest.approx(ne_of(tel, direct), abs=1e-9)
    # the box is 80 px below centre, so the foot is 80 * h / fx = 1.875 m further north-ish on the ground
    assert math.hypot(*ne_between(centre.lat, centre.lon, foot.lat, foot.lon)) == pytest.approx(
        80.0 * 60.0 / intr.fx, abs=1e-3
    )


# =============================================================================================================
# 4. DEM and water plane (§5.7 step 5)
# =============================================================================================================
@pytest.fixture(scope="module")
def dem() -> DemSampler:
    return load_default_dem()


def test_dem_sampler_bilinear_and_bounds(dem):
    import rasterio

    assert dem.shape == (289, 289)
    assert dem.contains(LAT0, LON0)
    assert not dem.contains(LAT0, 80.0)
    assert math.isnan(dem.elevation(LAT0, 80.0))  # never extrapolates, never silently returns 0

    with rasterio.open(dem.path) as ds:
        t = ds.transform
    a = dem.array
    lon_a, lat_a = t @ (10 + 0.5, 20 + 0.5)  # centre of pixel (row 20, col 10)
    lon_b, lat_b = t @ (11 + 0.5, 20 + 0.5)
    assert dem.elevation(lat_a, lon_a) == pytest.approx(float(a[20, 10]), abs=1e-6)
    assert dem.elevation((lat_a + lat_b) / 2, (lon_a + lon_b) / 2) == pytest.approx(
        float(a[20, 10] + a[20, 11]) / 2, abs=1e-6
    )
    assert 25.0 < dem.res_m < 35.0  # GLO-30 is 1 arcsec


def test_dem_ray_march_lands_on_the_terrain_surface(dem):
    """The marched intersection must lie ON the DEM surface to within the bisection tolerance, and over real
    17-degree terrain it must differ materially from the flat-plane answer."""
    intr = cam()
    ground = dem.elevation(LAT0, LON0)
    tel = tele(agl=100.0, pitch=-45.0, yaw=0.0, ground_asl=ground)
    cfg = ChainConfig(method="dem", dem=dem, dem_step_m=3.0, dem_bisect_m=0.05)

    fix, north, east = project_pixel_ne(intr.cx, intr.cy, tel, intr, cfg)
    assert fix.valid and fix.method == "dem" and fix.dem_source == dem.name
    ray_alt = tel.alt_msl_m - fix.slant_range_m * math.cos(math.radians(fix.off_nadir_deg))
    assert dem.elevation(fix.lat, fix.lon) == pytest.approx(ray_alt, abs=0.10)
    assert fix.alt_msl_m == pytest.approx(ray_alt, abs=1e-9)
    assert fix.agl_m == pytest.approx(tel.alt_msl_m - fix.alt_msl_m, abs=1e-9)

    flat = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(method="flat_plane"))
    sep = math.hypot(*ne_between(flat.lat, flat.lon, fix.lat, fix.lon))
    assert sep > 5.0, sep  # measured 14.9 m on this slope: terrain relief is not a rounding error
    # the DEM's larger vertical uncertainty shows up in the published radius
    assert fix.h_acc_m > flat.h_acc_m


def test_dem_nadir_agrees_with_the_flat_plane(dem):
    """Straight down, a DEM march and a flat plane at the true AGL must give the same point."""
    intr = cam()
    ground = dem.elevation(LAT0, LON0)
    tel = tele(agl=80.0, ground_asl=ground)
    a = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(method="dem", dem=dem))
    b = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(method="flat_plane"))
    assert math.hypot(*ne_between(a.lat, a.lon, b.lat, b.lon)) < 0.05
    assert a.alt_msl_m == pytest.approx(ground, abs=0.10)


def test_dem_miss_falls_back_or_rejects(dem):
    """A ray that leaves the tile before hitting terrain must be reported, not silently mis-located."""
    intr = cam()
    ground = dem.elevation(LAT0, LON0)
    tel = tele(agl=100.0, pitch=-10.0, yaw=180.0, ground_asl=ground)  # shallow ray, long run to the tile edge
    fb = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(method="dem", dem=dem, dem_max_range_m=200.0))
    assert fb.valid and fb.method == "flat_plane" and "fallback_flat_plane" in fb.dem_source
    strict = project_pixel(
        intr.cx, intr.cy, tel, intr, ChainConfig(method="dem", dem=dem, dem_max_range_m=200.0, dem_fallback=False)
    )
    assert not strict.valid and strict.reject_reason == "dem_miss"
    with pytest.raises(ValueError):
        project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(method="dem", dem=None))


def test_water_plane_uses_the_flood_surface_and_floors_the_vertical_error():
    """Over flood water the "ground" is the water surface, and DEM error is at least the flood depth (§5.7)."""
    intr = cam()
    water_asl = 1061.6814259297319  # data/scene/flood_valley.json water_level_m
    tel = tele(agl=60.0, pitch=-45.0, ground_asl=GROUND_ASL)  # camera at 1106.007 m ASL
    cfg = ChainConfig(method="water_plane", water_level_asl_m=water_asl, flood_depth_m=15.67)
    fix, north, _ = project_pixel_ne(intr.cx, intr.cy, tel, intr, cfg)
    assert fix.method == "water_plane"
    assert fix.alt_msl_m == pytest.approx(water_asl)
    assert fix.agl_m == pytest.approx(tel.alt_msl_m - water_asl, abs=1e-9)
    assert north == pytest.approx(fix.agl_m, abs=1e-9)  # 45 deg: ground offset == height above the water

    # the flood depth raises the altitude term of the budget over the 1 m barometric default
    shallow = ChainConfig(method="water_plane", water_level_asl_m=water_asl, flood_depth_m=0.0)
    assert fix.h_acc_m > project_pixel(intr.cx, intr.cy, tel, intr, shallow).h_acc_m
    terms = error_terms(fix.agl_m, fix.off_nadir_deg, CONSUMER.with_(agl_sigma_m=15.67))
    assert terms["altitude"] == pytest.approx(15.67, abs=1e-6)  # tan(45) * depth

    # `Telemetry.flood_level_asl_m` is used when the config does not override it
    tel2 = tele(agl=60.0, pitch=-45.0, flood_level_asl_m=water_asl)
    auto = project_pixel(intr.cx, intr.cy, tel2, intr, ChainConfig(method="water_plane"))
    assert auto.alt_msl_m == pytest.approx(water_asl)
    # and with neither, it falls back to barometric AGL plus the manual offset
    manual = project_pixel(
        intr.cx, intr.cy, tele(agl=60.0, pitch=-45.0), intr, ChainConfig(method="water_plane", water_offset_m=-5.0)
    )
    assert manual.agl_m == pytest.approx(55.0)


# =============================================================================================================
# 5. The error budget (§5.7 "Error budget")
# =============================================================================================================
def test_budget_reproduces_the_doc_table():
    """19 of the doc's 21 cells must reproduce exactly at 1-decimal rounding."""
    rows = compare_to_doc()
    assert len(rows) == 21
    mismatches = [(r["label"], r["agl_m"]) for r in rows if not r["matches"]]
    assert set(mismatches) == set(KNOWN_DOC_DISCREPANCIES), mismatches
    for r in rows:
        key = (r["label"], r["agl_m"])
        if key in KNOWN_DOC_DISCREPANCIES:
            continue
        assert round(float(r["computed_m"]), 1) == pytest.approx(r["doc_m"]), r


def test_the_two_documented_doc_discrepancies_are_asserted_not_fudged():
    """Where this lane disagrees with the doc, it asserts its OWN derivation and says so (see
    `budget.KNOWN_DOC_DISCREPANCIES` for the reasoning). Both are in the 30 m column of an oblique row."""
    assert h_acc_m(30.0, 45.0, PESSIMISTIC) == pytest.approx(3.617, abs=0.001)  # doc prints 3.9
    assert h_acc_m(60.0, 45.0, PESSIMISTIC) == pytest.approx(5.519, abs=0.001)  # doc 5.5 -> matches
    assert h_acc_m(100.0, 45.0, PESSIMISTIC) == pytest.approx(8.465, abs=0.001)  # doc 8.5 -> matches
    assert h_acc_m(30.0, 45.0, RTK) == pytest.approx(1.446, abs=0.001)  # doc prints 1.5

    # 3.9 at 30 m would need either h = 35 m or a pointing sigma of 2.23 deg, neither of which the doc states
    assert h_acc_m(35.08, 45.0, PESSIMISTIC) == pytest.approx(3.9, abs=0.01)
    assert PESSIMISTIC.pointing_sigma_deg == pytest.approx(1.7378, abs=1e-4)


def test_budget_headline_numbers_from_the_doc_prose():
    """§5.7 "The stated error budget for the assumed altitude": 60 m nadir consumer ~2.6 m centre, ~3.0 m edge,
    CE90 ~6 m; RTK 0.7-1.7 m."""
    assert h_acc_m(60.0, 0.0, CONSUMER) == pytest.approx(2.6, abs=0.05)
    assert h_acc_m(60.0, 36.0, CONSUMER) == pytest.approx(3.0, abs=0.05)
    assert ce90_m(h_acc_m(60.0, 0.0, CONSUMER)) == pytest.approx(6.0, abs=0.5)
    # "with RTK ~ 0.7-1.7 m": 0.688 and 1.696, i.e. the doc's interval to its stated precision
    assert round(h_acc_m(60.0, 0.0, RTK), 1) == 0.7
    assert round(h_acc_m(60.0, 36.0, RTK), 1) == 1.7
    assert CE90_FACTOR == pytest.approx(math.sqrt(-2.0 * math.log(0.1)), abs=1e-4)
    assert dedup_radius_m(2.6) == pytest.approx(2.0 * CE90_FACTOR * 2.6)


def test_budget_sensitivities_are_the_doc_formulas():
    h, th = 60.0, 36.0
    t = error_terms(h, th, CONSUMER)
    sec2, tan = 1.0 / math.cos(math.radians(th)) ** 2, math.tan(math.radians(th))
    assert t["pointing"] == pytest.approx(h * sec2 * math.radians(CONSUMER.pointing_sigma_deg))
    assert t["heading"] == pytest.approx(h * tan * math.radians(CONSUMER.yaw_deg))
    assert t["altitude"] == pytest.approx(tan * CONSUMER.agl_sigma_m)
    assert t["pixel"] == pytest.approx(h * sec2 * CONSUMER.pixel_sigma_px / CONSUMER.focal_px)
    assert t["gnss"] == CONSUMER.gnss_h_m
    assert t["sync"] == pytest.approx(CONSUMER.speed_ms * CONSUMER.sync_s)

    # at nadir the heading and altitude terms vanish exactly; that is why yaw error does not move a nadir fix
    n = error_terms(h, 0.0, CONSUMER)
    assert n["heading"] == 0.0 and n["altitude"] == 0.0
    # altitude error is independent of height; pointing and pixel scale linearly with it
    assert error_terms(2 * h, th, CONSUMER)["altitude"] == pytest.approx(t["altitude"])
    assert error_terms(2 * h, th, CONSUMER)["pointing"] == pytest.approx(2 * t["pointing"])
    # the doc's reading: nadir is GNSS-dominated, oblique-and-high is attitude/heading-dominated
    assert dominant_term(60.0, 0.0, CONSUMER) == "gnss"
    assert dominant_term(100.0, 45.0, PESSIMISTIC) == "pointing"
    assert dominant_term(100.0, 45.0, RTK) == "heading"
    # and h_acc grows monotonically with off-nadir angle
    accs = [h_acc_m(60.0, a, CONSUMER) for a in (0, 10, 20, 36, 45, 60, 75)]
    assert accs == sorted(accs)


def test_h_acc_is_per_pixel_not_per_frame():
    """§5.7 step 7: "Emit h_acc_m for THIS pixel's geometry", so the dedup radius is per record."""
    intr = cam()
    tel = tele(agl=60.0)
    centre = project_pixel(intr.cx, intr.cy, tel, intr)
    corner = project_pixel(intr.width_px - 1.0, intr.height_px - 1.0, tel, intr)
    assert corner.off_nadir_deg > 20.0 > centre.off_nadir_deg
    assert corner.h_acc_m > centre.h_acc_m + 0.05
    assert centre.h_acc_basis == "budget_v1/consumer"
    # the chain uses the camera's real focal length and the receiver's reported accuracy, not the preset
    tel_rtk = tele(agl=60.0)
    tel_rtk.h_acc_m = 0.05
    # with an RTK-grade receiver reported on the telemetry, the fix drops to the budget's RTK row
    assert project_pixel(intr.cx, intr.cy, tel_rtk, intr).h_acc_m == pytest.approx(
        h_acc_m(60.0, 0.0, RTK.with_(speed_ms=0.0, focal_px=intr.fx)), rel=1e-9
    )
    assert project_pixel(intr.cx, intr.cy, tel_rtk, intr).h_acc_m < 0.8
    # and the time-sync term uses the reported ground speed (a hovering drone has no sync error)
    tel_fast = tele(agl=60.0)
    tel_fast.vel_ned_ms = (10.0, 0.0, 0.0)
    assert project_pixel(intr.cx, intr.cy, tel_fast, intr).h_acc_m > centre.h_acc_m


def test_gsd_and_footprint_helpers():
    intr = cam()
    assert gsd_cm_px(60.0, intr) == pytest.approx(100.0 * 60.0 / intr.fx, abs=1e-12)
    assert gsd_cm_px(60.0, intr) == pytest.approx(2.344, abs=0.002)  # 2.34 cm/px at 60 m nadir
    assert gsd_cm_px(60.0, intr, 45.0) == pytest.approx(2.0 * gsd_cm_px(60.0, intr), rel=1e-9)
    fp = footprint_ned(intr, tele(agl=60.0))
    assert len(fp) == 4
    width = abs(fp[1][1] - fp[0][1])
    assert width == pytest.approx(2 * 60.0 * math.tan(math.radians(73.7398 / 2)), rel=1e-3)
    assert footprint_ned(intr, tele(agl=60.0, pitch=-2.0)) == []  # horizon corner -> no footprint


# =============================================================================================================
# 6. Simulator anchoring and the AirSim datum (docs/CONTEXT.md §5)
# =============================================================================================================
def test_scene_origin_matches_the_recorded_geopoint():
    o = load_scene_origin()
    assert (o.lat, o.lon) == pytest.approx((11.4870, 76.1450))
    assert o.alt_m == pytest.approx(1046.007, abs=0.001)


def test_airsim_datum_reproduces_the_recorded_launch_site_geopoint():
    """`data/scene/flood_valley.json` records the pad at NED (148 N, 492 E) reading 11.4883295 / 76.1495101.
    That is only reproducible with AirSim's SPHERICAL projection; a WGS-84 geodesic is 0.93 m north of it."""
    o = load_scene_origin()
    lat, lon, alt = airsim_ned_to_geodetic((148.0, 492.0, -(1066.5859375 - o.alt_m)), o)
    assert (lat, lon) == pytest.approx((11.4883295, 76.1495101), abs=2e-7)
    assert alt == pytest.approx(1066.586, abs=0.01)

    from sightline.common.geodesy import offset_ne_geodesic

    glat, glon = offset_ne_geodesic(o.lat, o.lon, 148.0, 492.0)
    dn, de = ne_between(lat, lon, glat, glon)
    assert dn == pytest.approx(0.93, abs=0.05) and abs(de) < 0.1  # the 0.633 % meridian mismatch, measured


def test_airsim_datum_costs_063_percent_of_the_northward_ray_offset():
    intr = cam()
    o = load_scene_origin()
    tel = tele(agl=60.0, pitch=-45.0, yaw=0.0, ned_m=(0.0, 0.0, -60.0))
    a = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig())  # wgs84
    b = project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(datum="airsim_sphere", origin=o))
    dn, de = ne_between(a.lat, a.lon, b.lat, b.lon)
    assert dn == pytest.approx(-0.633 / 100.0 * 60.0, abs=0.02) and abs(de) < 0.01
    # the datum-free local offset is identical either way: this is why sim evaluation should use it
    assert project_pixel_ne(intr.cx, intr.cy, tel, intr, ChainConfig())[1:] == pytest.approx(
        project_pixel_ne(intr.cx, intr.cy, tel, intr, ChainConfig(datum="airsim_sphere", origin=o))[1:], abs=1e-12
    )
    with pytest.raises(ValueError):
        project_pixel(intr.cx, intr.cy, tel, intr, ChainConfig(datum="epsg4326"))


def test_ned_round_trip_through_the_origin():
    o = OriginGeopoint(LAT0, LON0, GROUND_ASL)
    for ned in ((0.0, 0.0, 0.0), (250.0, -400.0, -60.0), (-1000.0, 1000.0, 12.0)):
        back = o.to_ned(*o.to_geodetic(ned))
        assert back == pytest.approx(ned, abs=0.02)  # ellipsoid vs geodesic, sub-decimetre over 1.4 km


# =============================================================================================================
# 7. The simulator noise injector (§5.7 "In the simulator")
# =============================================================================================================
def poses(n: int, hz: float = 5.0) -> list[Telemetry]:
    """A hovering nadir camera at 60 m AGL, heading 030, sampled at `hz`. Ground truth, no noise."""
    return [
        Telemetry(
            t_utc=1_780_000_000.0 + i / hz,
            lat=LAT0,
            lon=LON0,
            alt_msl_m=GROUND_ASL + 60.0,
            agl_m=60.0,
            q_body=euler_to_quat(0.0, 0.0, 30.0),
            q_gimbal=euler_to_quat(0.0, -90.0, 30.0),
            ned_m=(0.0, 0.0, -60.0),
            vel_ned_ms=(5.0, 0.0, 0.0),
            frame_idx=i,
        )
        for i in range(n)
    ]


def test_noise_is_reproducible_under_a_fixed_seed():
    a = TelemetryNoise(NoiseConfig(seed=7)).apply_stream(poses(40))
    b = TelemetryNoise(NoiseConfig(seed=7)).apply_stream(poses(40))
    c = TelemetryNoise(NoiseConfig(seed=8)).apply_stream(poses(40))
    assert a == b
    assert a[0] != c[0]
    # reset() restores the same stream from the same instance
    inj = TelemetryNoise(NoiseConfig(seed=7))
    first = inj.apply_stream(poses(40))
    inj.reset()
    assert inj.apply_stream(poses(40)) == first


def test_noise_marks_the_telemetry_and_reports_the_gnss_accuracy():
    n = TelemetryNoise(NoiseConfig(seed=1)).apply(poses(1)[0])
    assert n.noise_injected is True
    assert n.h_acc_m == 2.5 and n.v_acc_m == 1.0


def test_noise_switch_turns_it_off_for_debugging():
    truth = poses(3)
    off = TelemetryNoise(NoiseConfig.off()).apply_stream(truth)
    assert off == truth
    assert all(t.noise_injected is False for t in off)
    import os

    os.environ["SIGHTLINE_GEO_NOISE"] = "0"
    try:
        assert NoiseConfig.from_env().enabled is False
    finally:
        os.environ.pop("SIGHTLINE_GEO_NOISE")
    assert NoiseConfig.from_env(seed=3).enabled is True


def test_noise_has_the_standard_deviations_the_doc_specifies():
    """GNSS 2.5 m (per horizontal axis), 1.5 deg yaw BIAS, 0.5 deg pitch/roll, 1 m barometric.

    GNSS and baro are first-order Gauss-Markov walks, so consecutive samples inside one run are correlated and
    their within-run sample std is not an estimator of the marginal sigma. The marginal sigma is measured
    across independent runs; the white-noise attitude terms are measured within one run.
    """
    n_runs = 2000
    p0 = poses(1)[0]
    dn, de, dh, dy = [], [], [], []
    for s in range(n_runs):
        smp = TelemetryNoise(NoiseConfig(seed=s)).apply_pair(p0)
        dn.append(smp.d_north_m)
        de.append(smp.d_east_m)
        dh.append(smp.d_alt_m)
        dy.append(smp.d_yaw_deg)
    tol = 4.0 / math.sqrt(2 * n_runs)  # 4 sigma on the std estimator, as a relative tolerance
    assert float(np.std(dn)) == pytest.approx(2.5, rel=tol)
    assert float(np.std(de)) == pytest.approx(2.5, rel=tol)
    assert float(np.std(dh)) == pytest.approx(1.0, rel=tol)
    assert float(np.std(dy)) == pytest.approx(1.5, rel=tol)
    assert abs(float(np.mean(dn))) < 4 * 2.5 / math.sqrt(n_runs)  # unbiased

    inj = TelemetryNoise(NoiseConfig(seed=3))
    inj.apply_stream(poses(3000))
    et = inj.error_table()
    tol2 = 4.0 / math.sqrt(2 * 3000)
    assert float(et["d_pitch_deg"].std()) == pytest.approx(0.5, rel=tol2)
    assert float(et["d_roll_deg"].std()) == pytest.approx(0.5, rel=tol2)
    assert float(et["d_yaw_deg"].std()) == 0.0  # a BIAS: it does not average away over a pass (doc §5.7)
    assert et["d_yaw_deg"][0] == pytest.approx(inj.yaw_bias_deg)


def test_noise_gnss_walk_is_correlated_but_stationary():
    """A free random walk would grow without bound and make the measured error depend on clip length."""
    inj = TelemetryNoise(NoiseConfig(seed=5))
    inj.apply_stream(poses(1500, hz=5.0))  # 300 s at 5 Hz
    e = inj.error_table()["d_north_m"]
    assert abs(float(np.corrcoef(e[:-1], e[1:])[0, 1])) > 0.9  # 0.2 s steps, tau = 30 s -> rho = 0.993
    first, last = e[:500], e[-500:]
    assert 0.3 < float(np.std(last)) / max(float(np.std(first)), 1e-9) < 3.0  # no drift blow-up


def test_noise_preserves_ground_truth_alongside():
    truth = poses(20)
    inj = TelemetryNoise(NoiseConfig(seed=11))
    noisy = inj.apply_stream(truth)
    assert inj.truth_log() == truth  # the originals, unmutated (R10: nothing is discarded)
    assert all(t.lat == LAT0 and not t.noise_injected for t in truth)
    assert any(t.lat != LAT0 for t in noisy)
    s = inj.samples[0]
    assert inj.horizontal_offset_m(s) == pytest.approx(s.d_horizontal_m, abs=0.01)
    assert set(inj.error_table()) >= {"d_north_m", "d_east_m", "d_alt_m", "d_yaw_deg", "d_pitch_deg", "d_roll_deg"}
    # the NED copy moves with the geodetic one, so both stay consistent
    assert noisy[0].ned_m[0] == pytest.approx(s.d_north_m, abs=1e-9)
    assert noisy[0].ned_m[2] == pytest.approx(-60.0 - s.d_alt_m, abs=1e-9)
    assert noisy[0].agl_m == pytest.approx(60.0 + s.d_alt_m, abs=1e-9)


def test_noise_does_not_hit_gimbal_lock_on_a_nadir_camera():
    """Regression. `quat_to_euler` cannot separate roll from yaw at pitch = -90, so perturbing a nadir gimbal
    by euler round-trip silently rotates the camera's azimuth by its whole heading. The injector perturbs by
    quaternion multiplication instead."""
    q = euler_to_quat(0.0, -90.0, 30.0)
    roll, pitch, yaw = quat_to_euler(q)
    assert pitch == pytest.approx(-90.0, abs=1e-6)
    # `common/geodesy.quat_to_euler` used to split the rotation arbitrarily between roll and yaw here and lose
    # the heading (it returned roll=180 / yaw=180 for a 30 deg azimuth). It now resolves the lock explicitly -
    # roll = 0, all the rotation into yaw - so the azimuth survives the round trip. The injector still perturbs
    # by quaternion multiplication rather than through euler, which is what the rest of this test checks.
    assert roll == pytest.approx(0.0, abs=1e-6)
    assert yaw % 360.0 == pytest.approx(30.0, abs=1e-6)

    intr = cam()
    truth = poses(1)[0]  # gimbal yaw 30 deg, nadir
    cfg = NoiseConfig(seed=4, gnss_sigma_m=0.0, baro_sigma_m=0.0, attitude_sigma_deg=0.0, yaw_bias_sigma_deg=1.5)
    smp = TelemetryNoise(cfg).apply_pair(truth)
    a, an, ae = project_pixel_ne(intr.cx + 1000.0, intr.cy, truth, intr)
    b, bn, be = project_pixel_ne(intr.cx + 1000.0, intr.cy, smp.noisy, intr)
    az_a, az_b = math.degrees(math.atan2(ae, an)), math.degrees(math.atan2(be, bn))
    assert az_a == pytest.approx(120.0, abs=1e-6)  # camera yaw 30 + a +u pixel's 90 deg bearing
    assert (az_b - az_a) == pytest.approx(smp.d_yaw_deg, abs=1e-3)  # shifts by the bias, and by nothing else
    assert math.hypot(bn, be) == pytest.approx(math.hypot(an, ae), abs=1e-6)


def test_noise_reproduces_the_budget_sensitivities():
    """The injected errors must move the fix the way the budget says they do (§5.7 sensitivities)."""
    intr = cam()
    truth = poses(1)[0]
    base = project_pixel_ne(intr.cx, intr.cy, truth, intr)[1:]

    # a pure yaw bias leaves a NADIR fix where it was (heading term = h*tan(0)*dpsi = 0)
    only_yaw = NoiseConfig(seed=2, gnss_sigma_m=0.0, baro_sigma_m=0.0, attitude_sigma_deg=0.0)
    ny = TelemetryNoise(only_yaw).apply(truth)
    assert project_pixel_ne(intr.cx, intr.cy, ny, intr)[1:] == pytest.approx(base, abs=1e-9)

    # a pure tilt moves a nadir fix by exactly h * hypot(dpitch, droll) (pointing term at theta = 0)
    only_tilt = NoiseConfig(seed=1, gnss_sigma_m=0.0, baro_sigma_m=0.0, yaw_bias_sigma_deg=0.0)
    inj = TelemetryNoise(only_tilt)
    smp = inj.apply_pair(truth)
    _, tn, te = project_pixel_ne(intr.cx, intr.cy, smp.noisy, intr)
    moved = math.hypot(tn - base[0], te - base[1])
    assert moved == pytest.approx(60.0 * math.radians(math.hypot(smp.d_pitch_deg, smp.d_roll_deg)), rel=1e-3)


def test_injected_error_matches_the_published_h_acc_end_to_end():
    """Monte Carlo: project a nadir pixel through many noisy telemetries and check that the spread of the
    resulting fixes matches the radius the chain publishes. The budget's pixel and time-sync terms describe
    detector and ingest error, which this injector does not model, so they are zeroed for the comparison."""
    intr = cam()
    truth = poses(1)[0]
    n_runs = 900
    dn = np.empty(n_runs)
    de = np.empty(n_runs)
    for s in range(n_runs):
        noisy = TelemetryNoise(NoiseConfig(seed=10_000 + s)).apply(truth)
        fix = project_pixel(intr.cx, intr.cy, noisy, intr)
        # the fix's absolute position error, relative to the truth camera position directly below
        dn[s], de[s] = ne_between(LAT0, LON0, fix.lat, fix.lon)

    predicted = h_acc_m(60.0, 0.0, CONSUMER.with_(pixel_sigma_px=0.0, sync_s=0.0))
    tol = 5.0 / math.sqrt(2 * n_runs)
    assert float(dn.std()) == pytest.approx(predicted, rel=tol), (float(dn.std()), predicted)
    assert float(de.std()) == pytest.approx(predicted, rel=tol), (float(de.std()), predicted)
    # 90 % of the fixes fall inside the published CE90
    r = np.hypot(dn, de)
    assert 0.85 <= float((r <= ce90_m(predicted)).mean()) <= 0.95


# =============================================================================================================
# 8. Guardrails and contract compliance
# =============================================================================================================
def test_the_chain_never_mutates_its_inputs():
    intr = cam()
    tel = tele(agl=60.0)
    before = (tel.lat, tel.lon, tel.alt_msl_m, tel.agl_m, tel.q_gimbal, tel.noise_injected)
    project_pixel(intr.cx, intr.cy, tel, intr)
    assert (tel.lat, tel.lon, tel.alt_msl_m, tel.agl_m, tel.q_gimbal, tel.noise_injected) == before


def test_every_fix_is_a_schema_geofix_with_a_stated_basis():
    intr = cam()
    for cfg in (ChainConfig(), ChainConfig(method="water_plane", water_level_asl_m=GROUND_ASL + 10)):
        fix = project_pixel(intr.cx + 400, intr.cy, tele(agl=60.0, pitch=-60.0), intr, cfg)
        assert isinstance(fix, GeoFix)
        assert fix.method in ("flat_plane", "dem", "water_plane")
        assert fix.h_acc_basis.startswith("budget_v1/")
        assert math.isfinite(fix.h_acc_m) and fix.h_acc_m > 0.0
        assert math.isfinite(fix.lat) and math.isfinite(fix.lon)
