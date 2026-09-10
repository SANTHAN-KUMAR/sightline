"""F7 ingest (SOLUTION_DOC §5.4): parsers, alignment, decimation, and the simulator export contract.

Everything here runs **offline on synthetic data plus the two sample logs already in `data/samples/`**. No
Unreal, no AirSim, no GPU, no network. The synthetic inputs are constructed so the right answer is known by
hand, which is the only way an alignment test can fail honestly.

What each group pins down, and why it exists:

* **DJI SRT** — both documented families (`[key: value]` and the newer `<font>`-wrapped form) plus the
  Matrice function-call form, including the unit trap `fnum 170 = f/1.7` that §5.4 calls out by name.
* **The gimbal quaternion** — a regression test for a real bug: the 3-2-1 readback was gimbal-locked at
  exactly nadir, which is the simulator's default camera pose.
* **Alignment** — a known attitude and position are recovered from an interpolated series to a stated
  tolerance, and a known injected `t_offset_s` is recovered by the barometric cross-correlation.
* **Decimation** — at k = 3 and k = 6 the right frames are processed AND every decoded frame is still
  reachable, because §5.4 requires the skipped ones for evidence thumbnails.
* **The simulator export** — a `sightline-sim-capture` clip written by `spec.SimCaptureWriter` reads back
  exactly, and the `tools/capture/run.py` dataset format reads back exactly, including its header.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from sightline.common import geodesy
from sightline.ingest import detect_source, open_clip
from sightline.ingest.align import BootClock, TelemetrySeries, cross_correlate_lag, estimate_t_offset
from sightline.ingest.decimate import FrameIndex, FrameRef, decimate, decimation_for_fps
from sightline.ingest.dji_srt import detect_srt_format, parse_srt_text, srt_to_telemetry
from sightline.ingest.sim import (
    CAPTURE_RUN_COLUMNS,
    CaptureRunReader,
    NoiseModel,
    SimExportReader,
    inject_noise,
    is_capture_run,
)
from sightline.ingest.spec import (
    CameraSpec,
    CaptureFormatError,
    CaptureManifest,
    SimCaptureWriter,
    gimbal_euler_from_quat,
    gimbal_quat_from_euler,
    validate_capture,
)
from sightline.schemas import Telemetry

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"

ORIGIN_LAT, ORIGIN_LON = 12.9000, 77.6000
T0 = 1_800_000_000.0


# =============================================================================================== DJI SRT ===
#: Mini 3 Pro / Mavic Air 2 family: bare `[key: value]` groups, no `<font>`, LEGACY integer units.
SRT_OLD = """1
00:00:00,000 --> 00:00:00,033
[iso : 100] [shutter : 1/1000.0] [fnum : 170] [ev : 0] [ct : 5562] [color_md : default] \
[focal_len : 240] [latitude: 12.900000] [longitude: 77.600000] [rel_alt: 45.000 abs_alt: 645.000]

2
00:00:00,033 --> 00:00:00,066
[iso : 200] [shutter : 1/500.0] [fnum : 170] [ev : 0] [ct : 5562] [color_md : default] \
[focal_len : 240] [latitude: 12.900100] [longitude: 77.600100] [rel_alt: 45.500 abs_alt: 645.500]
"""

#: Mavic 3 / Air 3 (format3, legacy units) then Mini 5 Pro / Mavic 4 Pro (format3b, decimal units + gimbal).
SRT_HTML = """1
00:00:00,000 --> 00:00:00,033
<font size="28">SrtCnt : 1, DiffTime : 33ms
2026-09-10 12:00:00,000,000
[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] [ct : 5562] [color_md : default] \
[focal_len : 240] [latitude: 12.900000] [longitude: 77.600000] [rel_alt: 45.000 abs_alt: 645.000] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt : 2, DiffTime : 33ms
2026-09-10 12:00:00,500,000
[iso : 100] [shutter : 1/1000.0] [fnum : 1.8] [ev : 0] [ct : 5562] [color_md : default] \
[focal_len : 24.00] [latitude: 12.900100] [longitude: 77.600100] [rel_alt: 45.500 abs_alt: 645.500] \
[gb_yaw: 33.5 gb_pitch: -60.0 gb_roll: 0.0] </font>
"""

#: Matrice 30 / 350 lineage: function-call payload, GPS written LON-first, F.PRY + G.PRY present.
SRT_M30 = """1
00:00:00,000 --> 00:00:00,033
<font size="36">F/2.8, SS 400, ISO 100, EV 0, DZOOM 1.0, GPS (77.600000, 12.900000, 21), D 80.50m, \
H 45.00m, H.S 3.20m/s, V.S 0.00m/s, F.PRY (1.2, -0.5, 88.0), G.PRY (-60.0, 0.0, 90.0)</font>
"""


def test_srt_old_bracket_form_parses_to_the_right_values():
    """§5.4 row 2: the `[key: value]` generation, including `fnum 170` = f/1.7."""
    entries = parse_srt_text(SRT_OLD)
    assert len(entries) == 2
    a, b = entries

    assert a.srt_format == "format1"
    assert detect_srt_format("[latitude: 1] [longitude: 2]", had_font_tag=False) == "format1"
    assert a.frame_idx == 0 and b.frame_idx == 1
    assert a.lat == pytest.approx(12.9) and a.lon == pytest.approx(77.6)
    assert b.lat == pytest.approx(12.9001) and b.lon == pytest.approx(77.6001)
    assert a.rel_alt_m == pytest.approx(45.0) and a.abs_alt_m == pytest.approx(645.0)
    assert b.rel_alt_m == pytest.approx(45.5)

    # the §5.4 unit trap, with the rule that fired recorded on the entry
    assert a.fnum == pytest.approx(1.7)
    assert a.unit_rule["fnum"] == "legacy_x100"
    assert a.focal_len_mm == pytest.approx(24.0)
    assert a.unit_rule["focal_len"] == "legacy_x10"
    assert a.shutter_s == pytest.approx(1 / 1000.0)
    assert b.shutter_s == pytest.approx(1 / 500.0)
    assert a.iso == 100.0 and b.iso == 200.0
    assert a.color_md == "default"

    # consumer Mavic-class SRT carries no gimbal and no date line
    assert not a.has_gimbal and a.gimbal_pitch_deg is None
    assert a.t_utc is None
    # subtitle timecodes are still parsed, which is what clip_start_utc hangs on
    assert a.start_s == pytest.approx(0.0) and b.start_s == pytest.approx(0.033)


def test_srt_html_bracket_form_parses_to_the_right_values():
    """The newer `<font ...>SrtCnt/FrameCnt` form: date line, decimal units, optional `gb_*` gimbal."""
    entries = parse_srt_text(SRT_HTML)
    assert len(entries) == 2
    a, b = entries

    assert a.srt_format == "format3"       # legacy integer fnum
    assert b.srt_format == "format3b"      # decimal fnum + FrameCnt
    assert a.frame_idx == 0 and b.frame_idx == 1   # SrtCnt/FrameCnt are 1-based

    # the same field written both ways in the same file must land on the same physical quantity
    assert a.fnum == pytest.approx(2.8) and a.unit_rule["fnum"] == "legacy_x100"
    assert b.fnum == pytest.approx(1.8) and b.unit_rule["fnum"] == "literal_decimal"
    assert a.focal_len_mm == pytest.approx(24.0) and a.unit_rule["focal_len"] == "legacy_x10"
    assert b.focal_len_mm == pytest.approx(24.0) and b.unit_rule["focal_len"] == "literal_decimal"

    # the date line becomes t_utc; the ,millis,micros tail is real precision, not decoration
    assert a.t_utc == pytest.approx(1789041600.0)            # 2026-09-10T12:00:00Z
    assert b.t_utc - a.t_utc == pytest.approx(0.5, abs=1e-6)  # ",500,000" = +0.5 s

    assert a.diff_time_ms == 33.0
    assert not a.has_gimbal
    assert b.has_gimbal
    assert b.gimbal_pitch_deg == pytest.approx(-60.0)
    assert b.gimbal_yaw_deg == pytest.approx(33.5)
    assert b.gimbal_roll_deg == pytest.approx(0.0)
    assert b.lat == pytest.approx(12.9001) and b.lon == pytest.approx(77.6001)


def test_srt_matrice_function_form_resolves_gps_order_and_attitude():
    """format2c writes `GPS(lon, lat, ...)`; |lat| <= 90 is the only reliable discriminator (§5.4 gotcha)."""
    (e,) = parse_srt_text(SRT_M30)
    assert e.srt_format == "format2c"
    assert e.lat == pytest.approx(12.9) and e.lon == pytest.approx(77.6)
    assert e.gimbal_pitch_deg == pytest.approx(-60.0)
    assert e.gimbal_yaw_deg == pytest.approx(90.0)
    assert e.flight_pitch_deg == pytest.approx(1.2)
    assert e.flight_roll_deg == pytest.approx(-0.5)
    assert e.flight_yaw_deg == pytest.approx(88.0)
    assert e.height_home_m == pytest.approx(45.0)
    assert e.rel_alt_m == pytest.approx(45.0)      # H stands in for rel_alt when no [rel_alt] group exists
    assert e.dist_home_m == pytest.approx(80.5)
    assert e.h_speed_ms == pytest.approx(3.2)
    assert e.v_speed_ms == pytest.approx(0.0)
    assert e.fnum == pytest.approx(2.8) and e.unit_rule["fnum"] == "f_stop_token"
    assert e.shutter_s == pytest.approx(1 / 400.0)
    assert e.iso == 100.0


@pytest.mark.parametrize(
    ("token", "value", "rule"),
    [("170", 1.7, "legacy_x100"), ("280", 2.8, "legacy_x100"), ("1.8", 1.8, "literal_decimal"),
     ("2.80", 2.8, "literal_decimal"), ("22", 22.0, "literal_small_integer")],
)
def test_srt_fnum_unit_rule(token, value, rule):
    """A decimal point means literal; a big bare integer means the legacy x100 encoding (§5.4)."""
    (e,) = parse_srt_text(f"1\n00:00:00,000 --> 00:00:00,033\n[fnum : {token}] [latitude: 1.0] [longitude: 2.0]\n")
    assert e.fnum == pytest.approx(value)
    assert e.unit_rule["fnum"] == rule


def test_srt_tz_offset_is_applied_and_recorded():
    """DJI writes aircraft-LOCAL time with no zone. The offset must move the clock and reach the report."""
    utc = parse_srt_text(SRT_HTML)[0].t_utc
    ist = parse_srt_text(SRT_HTML, tz_offset_h=5.5)[0].t_utc
    assert utc - ist == pytest.approx(5.5 * 3600.0)

    _, report = srt_to_telemetry(parse_srt_text(SRT_HTML, tz_offset_h=5.5), tz_offset_h=5.5)
    assert report.tz_offset_h == 5.5, "the provenance report must state the offset that produced its timestamps"


def test_srt_to_telemetry_marks_the_nadir_assumption():
    """A consumer SRT has no gimbal. The -90 substitution is allowed, but never silently (§5.4)."""
    entries = parse_srt_text(SRT_OLD)
    tel, report = srt_to_telemetry(entries, clip_id="c", clip_start_utc=T0, takeoff_alt_msl_m=600.0)
    assert len(tel) == 2
    assert report.n_with_gimbal == 0
    assert report.assumed_nadir is True
    assert any("nadir" in w for w in report.warnings)
    assert tel[0].gimbal_pitch_deg() == pytest.approx(-90.0)

    # takeoff_alt_msl_m supplied => altitude is takeoff + rel_alt, NOT the SRT's abs_alt (§5.7 "Factors")
    assert report.alt_basis == "takeoff_msl + rel_alt"
    assert tel[0].alt_msl_m == pytest.approx(645.0)
    assert tel[1].alt_msl_m == pytest.approx(645.5)
    assert tel[0].agl_m == pytest.approx(45.0)
    assert tel[0].t_utc == pytest.approx(T0)              # no date line => clip_start_utc + subtitle time
    assert tel[1].t_utc == pytest.approx(T0 + 0.033)

    off, off_report = srt_to_telemetry(entries, clip_start_utc=T0, assume_nadir=False)
    assert off_report.assumed_nadir is False
    assert off[0].gimbal_pitch_deg() == pytest.approx(0.0), "assume_nadir=False must leave the gimbal level"


def test_srt_without_takeoff_altitude_warns_about_the_datum():
    _, report = srt_to_telemetry(parse_srt_text(SRT_OLD), clip_start_utc=T0)
    assert "abs_alt" in report.alt_basis
    assert any("datum" in w for w in report.warnings)


# ================================================================================== gimbal quaternion ===
@pytest.mark.parametrize("yaw", [-179.0, -90.0, -33.0, 0.0, 35.0, 120.0, 179.0])
@pytest.mark.parametrize("pitch", [-90.0, -89.999, -89.5, -60.0, -30.0, 0.0, 45.0, 90.0])
@pytest.mark.parametrize("roll", [0.0, 3.0, -7.5])
def test_gimbal_euler_quaternion_round_trip(roll, pitch, yaw):
    """REGRESSION. `gimbal_euler_from_quat` used to be gimbal-locked at exactly -90 -- the simulator's
    default camera pose -- and returned (180, -90, 180) for EVERY yaw. `sim.inject_noise` performs exactly
    this round trip on every frame, so a nadir capture with a non-zero heading had its camera azimuth
    silently rotated by up to 180 deg before geolocation ever saw it."""
    q = gimbal_quat_from_euler(roll, pitch, yaw)
    r2, p2, y2 = gimbal_euler_from_quat(q)
    back = gimbal_quat_from_euler(r2, p2, y2)

    m1, m2 = geodesy.quat_to_rot(q), geodesy.quat_to_rot(back)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(m1.T @ m2)) - 1.0) / 2.0))))
    assert angle < 1e-4, f"round trip moved the camera by {angle} deg"

    if abs(pitch) < 89.9:            # away from lock, the angles themselves must come back
        assert (r2, p2, y2) == pytest.approx((roll, pitch, yaw), abs=1e-6)


def test_gimbal_lock_reports_azimuth_as_yaw_with_zero_roll():
    """At nadir, roll and yaw are one freedom. All of it is reported as yaw, so 'yaw' keeps meaning
    'which way north points in the image' -- the only useful reading for a nadir camera."""
    for yaw in (0.0, 35.0, -120.0):
        r, p, y = gimbal_euler_from_quat(gimbal_quat_from_euler(0.0, -90.0, yaw))
        assert r == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(-90.0, abs=1e-6)
        assert y == pytest.approx(yaw, abs=1e-6)


def test_gimbal_pitch_helper_agrees_with_the_frozen_schema():
    """`spec` and `schemas.Telemetry.gimbal_pitch_deg()` must agree, or -90 means two different things."""
    for pitch in (-90.0, -75.0, -45.0, 0.0):
        t = Telemetry(t_utc=T0, lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_msl_m=645.0, agl_m=45.0,
                      q_gimbal=gimbal_quat_from_euler(0.0, pitch, 40.0))
        assert t.gimbal_pitch_deg() == pytest.approx(pitch, abs=1e-6)
        assert gimbal_euler_from_quat(t.q_gimbal)[1] == pytest.approx(pitch, abs=1e-6)


# ========================================================================================== alignment ===
def _pose_series(n=11, dt=1.0, yaw_rate=10.0, pitch=-60.0):
    samples = []
    for i in range(n):
        lat, lon = geodesy.offset_ne(ORIGIN_LAT, ORIGIN_LON, 3.0 * i * dt, 4.0 * i * dt)
        samples.append(Telemetry(
            t_utc=T0 + i * dt, lat=lat, lon=lon, alt_msl_m=600.0 + 2.0 * i * dt, agl_m=45.0,
            q_body=geodesy.euler_to_quat(0.0, 0.0, yaw_rate * i * dt),
            q_gimbal=gimbal_quat_from_euler(0.0, pitch, yaw_rate * i * dt),
            ned_m=(3.0 * i * dt, 4.0 * i * dt, -45.0),
            mode="AUTO" if i < 5 else "MANUAL",
            vel_ned_ms=(3.0, 4.0, 0.0), h_acc_m=2.5, v_acc_m=1.0, clip_id="c",
        ))
    return TelemetrySeries.from_telemetry(samples, clip_id="c")


def test_telemetry_series_recovers_a_known_pose_between_samples():
    """§5.4: positions and scalars by np.interp, attitude by SLERP -- never linear Euler."""
    s = _pose_series()
    assert len(s) == 11 and s.duration_s == pytest.approx(10.0)

    for frac in (0.0, 0.25, 3.5, 7.25, 10.0):
        t = s.at(T0 + frac)
        assert t.alt_msl_m == pytest.approx(600.0 + 2.0 * frac, abs=1e-9)
        north, east = geodesy.ne_between(ORIGIN_LAT, ORIGIN_LON, t.lat, t.lon)
        assert north == pytest.approx(3.0 * frac, abs=1e-3)
        assert east == pytest.approx(4.0 * frac, abs=1e-3)
        assert t.ned_m == pytest.approx((3.0 * frac, 4.0 * frac, -45.0), abs=1e-9)

        roll, pitch, yaw = gimbal_euler_from_quat(t.q_gimbal)
        assert pitch == pytest.approx(-60.0, abs=1e-6)
        assert yaw == pytest.approx(10.0 * frac, abs=1e-6), "SLERP of a constant yaw rate is exact"
        assert roll == pytest.approx(0.0, abs=1e-6)
        assert geodesy.quat_to_euler(t.q_body)[2] == pytest.approx(10.0 * frac, abs=1e-6)
        assert np.linalg.norm(t.q_gimbal) == pytest.approx(1.0, abs=1e-12)


def test_telemetry_series_slerp_beats_linear_quaternion_blending():
    """The point of SLERP: a straight lerp of the endpoints is measurably off mid-arc."""
    a = gimbal_quat_from_euler(0.0, -60.0, 0.0)
    b = gimbal_quat_from_euler(0.0, -60.0, 120.0)
    s = TelemetrySeries.from_telemetry([
        Telemetry(t_utc=T0, lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_msl_m=645.0, agl_m=45.0, q_gimbal=a),
        Telemetry(t_utc=T0 + 1.0, lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_msl_m=645.0, agl_m=45.0, q_gimbal=b),
    ])
    # at the midpoint the two agree by symmetry, so compare a quarter of the way along the arc
    slerped = np.asarray(s.at(T0 + 0.25).q_gimbal)
    lerp = np.asarray(a) + 0.25 * (np.asarray(b) - np.asarray(a))
    lerp = lerp / np.linalg.norm(lerp)

    assert gimbal_euler_from_quat(tuple(slerped))[2] == pytest.approx(30.0, abs=1e-6)
    assert abs(gimbal_euler_from_quat(tuple(lerp))[2] - 30.0) > 1.0, "normalised lerp is not angle-uniform"


def test_telemetry_series_holds_discrete_fields_and_can_refuse_to_extrapolate():
    """Interpolating a flight mode is nonsense: it is held nearest-previous."""
    s = _pose_series()
    assert s.at(T0 + 4.9).mode == "AUTO"
    assert s.at(T0 + 5.1).mode == "MANUAL"

    assert s.covers(T0 + 5.0) and not s.covers(T0 + 50.0)
    assert s.at(T0 + 50.0).alt_msl_m == pytest.approx(620.0)      # clamped to the last sample
    with pytest.raises(ValueError):
        s.at(T0 + 50.0, clamp=False)


def test_telemetry_series_applies_the_r12_offset():
    """R12 convention, used everywhere in this package: `t_telemetry = t_frame + t_offset_s`."""
    s = _pose_series()
    s.t_offset_s = 2.0
    assert s.at(T0 + 1.0).alt_msl_m == pytest.approx(606.0)       # sampled at T0+3.0
    assert s.at(T0 + 1.0).t_utc == pytest.approx(T0 + 3.0)


def test_telemetry_series_sorts_and_deduplicates():
    rows = [
        Telemetry(t_utc=T0 + 2.0, lat=1.0, lon=2.0, alt_msl_m=10.0, agl_m=1.0),
        Telemetry(t_utc=T0 + 0.0, lat=1.0, lon=2.0, alt_msl_m=0.0, agl_m=1.0),
        Telemetry(t_utc=T0 + 2.0, lat=1.0, lon=2.0, alt_msl_m=99.0, agl_m=1.0),   # supersedes the first
    ]
    s = TelemetrySeries.from_telemetry(rows)
    assert len(s) == 2
    assert list(s.t) == [T0, T0 + 2.0]
    assert s.alt_msl_m[-1] == 99.0
    with pytest.raises(ValueError):
        TelemetrySeries.from_telemetry([])


def test_boot_clock_maps_a_monotonic_autopilot_clock_to_utc():
    bc = BootClock.from_pairs([10.0, 110.0, 210.0], [T0 + 10.0, T0 + 110.0, T0 + 210.0])
    assert bc.n_pairs == 3
    assert bc.skew == pytest.approx(1.0, abs=1e-6)
    assert float(bc.to_utc(60.0)) == pytest.approx(T0 + 60.0, abs=1e-3)
    assert float(bc.utc_of_ms(60_000.0)) == pytest.approx(T0 + 60.0, abs=1e-3)

    single = BootClock.from_pairs([12.0], [T0 + 12.0])
    assert single.skew == 1.0 and single.n_pairs == 1
    with pytest.raises(ValueError):
        BootClock.from_pairs([1.0, 2.0], [0.0, 0.0])   # time_unix_usec == 0 means "no GNSS time yet"


# ---------------------------------------------------------------------- the R12 per-clip video offset ---
def _takeoff_profile(t: np.ndarray, t0: float) -> np.ndarray:
    """Sit on the ground for 10 s, climb at 6 m/s to 60 m, then hold. The 'visible take-off' of §5.4."""
    return np.minimum(60.0, np.maximum(0.0, (t - t0 - 10.0) * 6.0))


@pytest.mark.parametrize("true_offset", [-2.5, -0.4, 0.0, 1.37, 3.2])
def test_estimate_t_offset_recovers_an_injected_offset(true_offset):
    """§5.4: cross-correlate barometric altitude against the video's height proxy. The Roboflow DJI
    georeferencing project lists exactly this as its open problem, so it gets an exact-answer test."""
    rng = np.random.default_rng(7)
    t_tel = T0 + np.arange(0.0, 60.0, 0.1)
    alt = _takeoff_profile(t_tel, T0) + rng.normal(0.0, 0.05, t_tel.size)

    # frame clock runs `true_offset` behind the telemetry clock: t_telemetry = t_frame + true_offset
    t_frames = T0 - true_offset + np.arange(0.0, 55.0, 1 / 30.0)
    proxy = 0.017 * np.interp(t_frames + true_offset, t_tel, alt)   # arbitrary monotone scaling, as in video

    offset, peak = estimate_t_offset(t_frames, proxy, t_tel, alt, max_lag_s=6.0)
    assert offset == pytest.approx(true_offset, abs=0.05)
    assert peak > 0.9


def test_estimate_t_offset_refuses_to_guess_on_an_uncorrelated_signal():
    """A bad alignment silently applied is worse than none: every geolocation downstream inherits it."""
    rng = np.random.default_rng(0)
    t_tel = T0 + np.arange(0.0, 60.0, 0.1)
    alt = _takeoff_profile(t_tel, T0)
    t_frames = T0 + np.arange(0.0, 55.0, 1 / 30.0)
    noise = rng.normal(0.0, 1.0, t_frames.size)
    with pytest.raises(ValueError, match="refusing to guess"):
        estimate_t_offset(t_frames, noise, t_tel, alt, max_lag_s=6.0, min_correlation=0.5)


def test_cross_correlate_lag_sign_convention():
    """`y_a(t + lag) ~= y_b(t)`; the docstring's sign is load bearing for `estimate_t_offset`."""
    t = np.arange(0.0, 30.0, 0.05)
    y = _takeoff_profile(t, 0.0)
    lag, peak = cross_correlate_lag(t, np.interp(t + 1.0, t, y), t, y, max_lag_s=5.0)
    assert lag == pytest.approx(-1.0, abs=0.05)
    assert peak > 0.9
    with pytest.raises(ValueError):
        cross_correlate_lag([0.0, 1.0], [0.0, 1.0], t, y)


# ========================================================================================= decimation ===
def test_decimation_for_fps_matches_the_doc():
    assert decimation_for_fps(30.0, 10.0) == 3        # §5.4: k = 3 -> 10 FPS
    assert decimation_for_fps(30.0, 5.0) == 6         # §5.4: k = 6 -> 5 FPS
    assert decimation_for_fps(30.0, 30.0) == 1
    assert decimation_for_fps(0.0, 5.0) == 1


def test_decimate_yields_every_kth_item():
    assert list(decimate(range(10), 3)) == [0, 3, 6, 9]
    assert list(decimate(range(10), 6)) == [0, 6]
    assert list(decimate(range(10), 3, offset=1)) == [1, 4, 7]
    assert list(decimate(range(10), 1)) == list(range(10))
    with pytest.raises(ValueError):
        list(decimate(range(3), 0))


def test_frame_index_only_grows_and_finds_the_nearest_frame():
    """R10 in the small: nothing in the index is ever removed, `mark_processed` only annotates."""
    idx = FrameIndex("c")
    for i in range(5):
        idx.add(FrameRef(frame_idx=i, t_utc=T0 + i / 30.0, path=f"f{i}.png"))
    idx.mark_processed(0)
    idx.mark_processed(3)
    assert len(idx) == 5
    assert idx.processed_indices() == [0, 3]
    assert idx.nearest(T0 + 2.4 / 30.0).frame_idx == 2
    assert idx.get(99) is None
    assert not hasattr(idx, "remove") and not hasattr(idx, "delete")


# =============================================================== the sightline-sim-capture round trip ===
def _write_sim_capture(root: Path, n: int = 13, hfov_deg: float = 60.0, w: int = 64, h: int = 36,
                       fps: float = 30.0, write_images: bool = True) -> Path:
    """A minimal but SPEC-COMPLETE `sightline-sim-capture` clip. Frame `i` is filled with the value `i`,
    so a test can prove `load_frame(i)` returned frame `i` and not merely 'some image'.

    `write_images=False` skips the PNGs for the long, metadata-only clips (the noise-model tests).
    """
    import cv2

    clip = root / "clip0"
    manifest = CaptureManifest(
        clip_id="clip0", domain="sim", capture_start_utc=T0,
        origin_geopoint={"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "alt_msl_m": 600.0},
        camera=CameraSpec(name="survey", width_px=w, height_px=h, hfov_deg=hfov_deg),
        flood_level_asl_m=600.0,
    )
    writer = SimCaptureWriter(clip, manifest).open()
    for i in range(n):
        north, east = 2.0 * i, 0.5 * i
        lat, lon = geodesy.offset_ne(ORIGIN_LAT, ORIGIN_LON, north, east)
        q = gimbal_quat_from_euler(0.0, -90.0, 0.0)
        rel = f"frames/rgb_{i:07d}.png"
        if write_images:
            cv2.imwrite(str(clip / rel), np.full((h, w, 3), i % 256, np.uint8))
        writer.append(
            frame_idx=i, t_utc=T0 + i / fps, t_sim_s=i / fps, rgb_path=rel,
            ned_n_m=north, ned_e_m=east, ned_d_m=-45.0,
            q_body_w=1.0, q_body_x=0.0, q_body_y=0.0, q_body_z=0.0,
            q_gimbal_w=q[0], q_gimbal_x=q[1], q_gimbal_y=q[2], q_gimbal_z=q[3],
            lat=lat, lon=lon, alt_msl_m=645.0, agl_m=45.0,
            img_w_px=w, img_h_px=h, hfov_deg=hfov_deg, mode="AUTO",
            rain=0.0, fog=0.0, wind_ms=3.0, cloud=0.2, flood_level_asl_m=600.0,
        )
    writer.close()
    return clip


def test_sim_capture_round_trips_through_the_reader(tmp_path):
    clip = _write_sim_capture(tmp_path)
    assert validate_capture(clip) == []
    assert detect_source(clip) == "sim"

    reader = SimExportReader(clip)
    assert reader.clip_id == "clip0"
    assert len(reader.rows) == 13

    intr = reader.intrinsics(reader.rows[0])
    assert (intr.width_px, intr.height_px) == (64, 36)
    assert intr.hfov_deg() == pytest.approx(60.0, abs=1e-9)
    assert intr.fx == pytest.approx((64 / 2) / math.tan(math.radians(60.0) / 2))

    tel = reader.telemetry(reader.rows[3])
    assert tel.t_utc == pytest.approx(T0 + 3 / 30.0)
    assert tel.agl_m == pytest.approx(45.0)
    assert tel.ned_m == pytest.approx((6.0, 1.5, -45.0))
    assert tel.gimbal_pitch_deg() == pytest.approx(-90.0)
    assert tel.gimbal_is_earth_referenced is True
    assert tel.weather == {"rain": 0.0, "fog": 0.0, "wind_ms": 3.0, "cloud": 0.2}
    assert tel.flood_level_asl_m == pytest.approx(600.0)
    north, east = geodesy.ne_between(ORIGIN_LAT, ORIGIN_LON, tel.lat, tel.lon)
    assert (north, east) == pytest.approx((6.0, 1.5), abs=1e-3)


def test_sim_capture_rejects_a_missing_required_column(tmp_path):
    clip = _write_sim_capture(tmp_path, n=3)
    frames = clip / "frames.csv"
    text = frames.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    drop = header.index("agl_m")
    frames.write_text("\n".join(",".join(p for j, p in enumerate(line.split(",")) if j != drop)
                                for line in text), encoding="utf-8")
    assert any("agl_m" in p for p in validate_capture(clip))
    with pytest.raises(CaptureFormatError, match="agl_m"):
        SimExportReader(clip)


def test_writer_rejects_an_unknown_column_and_a_bad_quaternion(tmp_path):
    manifest = CaptureManifest(clip_id="c", origin_geopoint={"lat": 1.0, "lon": 2.0, "alt_msl_m": 3.0})
    writer = SimCaptureWriter(tmp_path / "c", manifest).open()
    with pytest.raises(CaptureFormatError, match="unknown column"):
        writer.append(frame_idx=0, t_utc=T0, altitude_metres=45.0)
    with pytest.raises(CaptureFormatError, match="quaternion norm"):
        writer.append(frame_idx=0, t_utc=T0, rgb_path="frames/a.png", ned_n_m=0.0, ned_e_m=0.0, ned_d_m=-45.0,
                      q_body_w=2.0, q_body_x=0.0, q_body_y=0.0, q_body_z=0.0,
                      q_gimbal_w=1.0, q_gimbal_x=0.0, q_gimbal_y=0.0, q_gimbal_z=0.0,
                      lat=1.0, lon=2.0, alt_msl_m=48.0, agl_m=45.0,
                      img_w_px=64, img_h_px=36, hfov_deg=60.0)
    writer.close()


@pytest.mark.parametrize("k", [1, 3, 6])
def test_decimation_keeps_every_decoded_frame_reachable(tmp_path, k):
    """§5.4's decimation guarantee, stated as a test: processing every k-th frame must not make the OTHER
    frames unreachable -- `Track.best()` needs any of them for `Evidence.thumb_uri` (§5.6, §5.8)."""
    n = 13
    clip = _write_sim_capture(tmp_path, n=n)
    with open_clip(clip, decimate_k=k) as c:
        processed = [b.frame_idx for b in c]
        assert processed == list(range(0, n, k))
        assert len(c) == len(processed)
        assert len(c.frame_index) == n, "decimation must index every DECODED frame, not only processed ones"
        assert c.frame_index.processed_indices() == processed

        for i in range(n):
            img = c.load_frame(i)
            assert img is not None, f"frame {i} became unreachable at k={k}"
            # frame i was written filled with the value i: this proves the right frame came back
            assert int(img.mean()) == i


def test_target_fps_resolves_k_from_the_source_rate(tmp_path):
    clip = _write_sim_capture(tmp_path, n=13, fps=30.0)
    for target, expect in ((10.0, 3), (5.0, 6)):
        with open_clip(clip, target_fps=target) as c:   # source rate inferred from the frame times
            assert c.k == expect


def test_noise_injection_is_deterministic_and_keeps_the_clean_truth(tmp_path):
    """§5.7 in the simulator: perturb the telemetry, keep the truth beside it for evaluation."""
    clip = _write_sim_capture(tmp_path, n=40, write_images=False)
    with open_clip(clip, noise=NoiseModel(seed=11)) as a, open_clip(clip, noise=NoiseModel(seed=11)) as b:
        assert np.allclose(a.telemetry.lat, b.telemetry.lat)
        assert np.allclose(a.telemetry.q_gimbal, b.telemetry.q_gimbal)
        assert a.telemetry.noise_injected and not a.telemetry_truth.noise_injected

        # the truth series is untouched and the noised one is genuinely different.
        # Compare in METRES: at 12.9 deg latitude a 3 m offset is a 1e-6 relative change, which np.allclose's
        # default rtol would happily call "equal".
        assert np.array_equal(a.telemetry_truth.lat, SimExportReader(clip).telemetry_series().lat)
        assert _max_offset_m(a.telemetry_truth, a.telemetry) > 0.5

        # what the perturbed telemetry REPORTS is what a real receiver would publish (§5.7)
        assert np.allclose(a.telemetry.h_acc_m, 2.5)
        assert np.allclose(a.telemetry.v_acc_m, 1.0)
        assert np.allclose(a.telemetry_truth.h_acc_m, 0.0)

    with open_clip(clip, noise=NoiseModel(seed=12)) as c:
        assert _max_offset_m(c.telemetry, a.telemetry) > 0.5, "a different seed must give a different draw"


def _offsets_m(truth, noised) -> np.ndarray:
    """Per-sample horizontal deviation in metres between two series of the same length."""
    return np.array([math.hypot(*geodesy.ne_between(truth.lat[i], truth.lon[i], noised.lat[i], noised.lon[i]))
                     for i in range(len(truth))], dtype=float)


def _max_offset_m(truth, noised) -> float:
    return float(_offsets_m(truth, noised).max())


def test_noise_injection_error_magnitude_matches_the_model(tmp_path):
    """The §5.7 budget is a claim about metres and degrees; check the numbers, not just that it ran.

    Measured over an ENSEMBLE of seeds, not along one clip: half the variance is a per-clip constant bias
    (§5.7: "averaging N frames shrinks the random terms but not the biases") and the wander has a 30 s
    correlation time, so the spread *within* a short clip is legitimately far below the stated sigma. That
    distinction is the model's whole point, so the test has to respect it.
    """
    clip = _write_sim_capture(tmp_path, n=100, fps=10.0, write_images=False)
    truth = SimExportReader(clip).telemetry_series()

    radial: list[float] = []
    d_alt: list[float] = []
    for seed in range(25):
        noised = inject_noise(truth, NoiseModel(seed=seed))
        radial.extend(_offsets_m(truth, noised))
        d_alt.extend(noised.alt_msl_m - truth.alt_msl_m)

    # 2D Gaussian with radial sigma 2.5 m => RMS radius = 2.5 m; baro RMS = 1.0 m.
    assert float(np.sqrt(np.mean(np.square(radial)))) == pytest.approx(2.5, rel=0.35)
    assert float(np.sqrt(np.mean(np.square(d_alt)))) == pytest.approx(1.0, rel=0.35)

    # §5.7's other half: the error is CORRELATED in time, not independent per frame.
    one = inject_noise(truth, NoiseModel(seed=3))
    step = np.diff(_offsets_m(truth, one))
    assert float(np.std(step)) < 0.2 * 2.5, "a per-frame-independent model would jump by ~sigma every frame"


def _nadir_series(n: int = 20, yaw_deg: float = 0.0) -> TelemetrySeries:
    return TelemetrySeries.from_telemetry([
        Telemetry(t_utc=T0 + i, lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_msl_m=645.0, agl_m=45.0,
                  q_body=geodesy.euler_to_quat(0.0, 0.0, yaw_deg),
                  q_gimbal=gimbal_quat_from_euler(0.0, -90.0, yaw_deg))
        for i in range(n)
    ], clip_id="nadir")


def test_noise_injection_attitude_error_matches_the_model():
    """§5.7: pitch/roll 0.5 deg per sample, magnetometer yaw 1.5 deg as a per-clip BIAS. The pooled rotation
    away from truth is therefore sqrt(1.5^2 + 2 * 0.5^2) = 1.66 deg.

    Measured on the QUATERNION, not on Euler angles: the 3-2-1 readback of a near-nadir attitude is not
    unique (a pitch that crosses -90 comes back as its mirror with roll and yaw flipped by 180 deg), so an
    Euler-space assertion here would be testing the representation, not the model.
    """
    truth = _nadir_series()
    angles = [
        _quat_angle_deg(truth.q_gimbal[i], inject_noise(truth, NoiseModel(seed=s)).q_gimbal[i])
        for s in range(120) for i in (0, 7, 15)
    ]
    assert float(np.sqrt(np.mean(np.square(angles)))) == pytest.approx(
        math.sqrt(1.5 ** 2 + 2 * 0.5 ** 2), rel=0.25)

    only_attitude = [
        _quat_angle_deg(truth.q_gimbal[i],
                        inject_noise(truth, NoiseModel(seed=s, yaw_bias_sigma_deg=0.0)).q_gimbal[i])
        for s in range(120) for i in (0, 7, 15)
    ]
    assert float(np.sqrt(np.mean(np.square(only_attitude)))) == pytest.approx(math.sqrt(2) * 0.5, rel=0.25)


def test_noise_model_bias_term_does_not_average_out(tmp_path):
    """§5.7's conclusion — "averaging N frames shrinks the random terms by sqrt(N) but not the biases" — is
    only reproduced if a per-clip constant bias really exists. `bias_fraction` is that knob, so pin it."""
    clip = _write_sim_capture(tmp_path, n=100, fps=10.0, write_images=False)
    truth = SimExportReader(clip).telemetry_series()

    pure_bias = inject_noise(truth, NoiseModel(seed=5, bias_fraction=1.0))
    offsets = _offsets_m(truth, pure_bias)
    assert float(np.ptp(offsets)) < 1e-6, "a pure bias must be the SAME offset on every frame"
    assert float(np.ptp(pure_bias.alt_msl_m - truth.alt_msl_m)) < 1e-9
    assert float(offsets.mean()) > 0.1

    pure_wander = inject_noise(truth, NoiseModel(seed=5, bias_fraction=0.0))
    assert float(np.ptp(_offsets_m(truth, pure_wander))) > 0.1, "pure wander must vary along the clip"


def test_noise_injection_preserves_a_non_zero_nadir_heading(tmp_path):
    """REGRESSION for the gimbal-lock bug, end to end. `inject_noise` round-trips the gimbal through Euler
    angles; with the locked readback a nadir camera flying a 120 deg heading came out pointing 120 deg away
    in azimuth, which silently mislocates every off-centre detection."""
    truth = _nadir_series(yaw_deg=120.0)
    noised = inject_noise(truth, NoiseModel(seed=1))
    worst = max(_quat_angle_deg(truth.q_gimbal[i], noised.q_gimbal[i]) for i in range(len(truth)))
    assert worst < 6.0, f"the noise model moved the camera by {worst:.1f} deg; the budget is ~1.6 deg"


def _quat_angle_deg(qa, qb) -> float:
    """Rotation angle between two quaternions, in degrees. Sign-agnostic (q and -q are one rotation)."""
    a = np.asarray(qa, dtype=float) / np.linalg.norm(qa)
    b = np.asarray(qb, dtype=float) / np.linalg.norm(qb)
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


# ================================================== the tools/capture/run.py dataset format (F5 -> F7) ===
CAPTURE_RUN_HEADER_RE = re.compile(r"tw\.writerow\(\[(.*?)\]\)", re.DOTALL)


def test_capture_run_columns_match_the_capture_writer_source():
    """CONTRACT TEST. `sightline/ingest/sim.py` claims to read what `tools/capture/run.py` writes. This
    reads the header straight out of that script, so the claim cannot rot silently: if the sim lane adds,
    renames or reorders a column, this fails and names it."""
    source = (REPO / "tools" / "capture" / "run.py").read_text(encoding="utf-8")
    match = CAPTURE_RUN_HEADER_RE.search(source)
    assert match, "could not find the telemetry.csv header literal in tools/capture/run.py"
    written = tuple(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', match.group(1)))
    assert written == CAPTURE_RUN_COLUMNS, (
        "tools/capture/run.py's telemetry.csv header no longer matches sim.CAPTURE_RUN_COLUMNS.\n"
        f"  writer: {written}\n  reader: {CAPTURE_RUN_COLUMNS}"
    )


def _write_capture_run(root: Path, n: int = 6, w: int = 64, h: int = 36, hfov: float = 89.904,
                       clip_id: str = "sim_seed23_alt45", water: float = 1061.681, alt: float = 45.0,
                       yaw_deg: float = 0.0) -> Path:
    """A clip in EXACTLY the layout `tools/capture/run.py` produces (see CAPTURE_RUN_COLUMNS)."""
    import csv

    import cv2

    clip = root / "run0"
    for sub in ("images", "labels", "masks"):
        (clip / sub).mkdir(parents=True, exist_ok=True)
    with (clip / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(CAPTURE_RUN_COLUMNS))
        for i in range(n):
            east, north = -344.0, -281.1 + 40.43 * i
            lat, lon = geodesy.offset_ne(ORIGIN_LAT, ORIGIN_LON, north, east)
            q = geodesy.euler_to_quat(0.0, 0.0, yaw_deg)
            stem = f"{clip_id}_{i:05d}"
            cv2.imwrite(str(clip / "images" / f"{stem}.png"), np.full((h, w, 3), i, np.uint8))
            cv2.imwrite(str(clip / "masks" / f"{stem}.png"), np.zeros((h, w, 3), np.uint8))
            (clip / "labels" / f"{stem}.json").write_text(json.dumps([
                {"actor_id": 3, "name": "Human_003", "cls": "human", "bbox_px": [10, 12, 19, 25],
                 "visible_px": 90, "size_px": 14, "pose": "waving", "submersion": "half",
                 "occlusion": 1, "zone": "settlement", "group": "roof23", "aerially_detectable": True},
            ]), encoding="utf-8")
            writer.writerow([i, T0 + i * 0.31, clip_id, round(east, 2), round(north, 2),
                             round(water + alt, 2), round(alt, 2), lat, lon,
                             round(q[0], 6), round(q[1], 6), round(q[2], 6), round(q[3], 6),
                             -90.0, round(hfov, 3), w, h, 4.68, "AUTO", round(water, 3), 1])
    (clip / "data_card.json").write_text(json.dumps(
        {"clip_id": clip_id, "scenario_seed": 23, "altitude_m": alt, "domain": "sim",
         "width": w, "height": h, "hfov_deg": hfov, "gsd_cm_px": 4.68}), encoding="utf-8")
    return clip


def test_capture_run_clip_is_detected_and_read(tmp_path):
    clip = _write_capture_run(tmp_path)
    assert is_capture_run(clip)
    assert detect_source(clip) == "capture_run"

    reader = CaptureRunReader(clip)
    assert reader.clip_id == "sim_seed23_alt45"
    assert reader.domain == "sim"
    assert reader.validate() == []
    assert len(reader.rows) == 6

    intr = reader.intrinsics()
    assert (intr.width_px, intr.height_px) == (64, 36)
    assert intr.hfov_deg() == pytest.approx(89.904, abs=1e-6)

    tel = reader.telemetry(reader.rows[2])
    assert tel.t_utc == pytest.approx(T0 + 0.62)
    assert tel.alt_msl_m == pytest.approx(1106.68)
    assert tel.agl_m == pytest.approx(45.0)
    assert tel.flood_level_asl_m == pytest.approx(1061.681)
    assert tel.gimbal_pitch_deg() == pytest.approx(-90.0)
    assert tel.gimbal_is_earth_referenced is True
    assert tel.h_acc_m == 0.0, "clean simulator truth; inject_noise() writes the §5.7 numbers"
    # east_m/north_m are SCENE ENU; the ned_m down datum is the flood surface, so ned_d ~ -agl_m
    assert tel.ned_m == pytest.approx((-281.1 + 40.43 * 2, -344.0, -45.0), abs=1e-2)


def test_capture_run_gimbal_yaw_follows_the_airframe(tmp_path):
    """`telemetry.csv` writes a gimbal PITCH only. Which way north points in the image comes from the
    vehicle quaternion, and that assumption is switchable and listed."""
    clip = _write_capture_run(tmp_path, n=2, yaw_deg=37.0)
    follow = CaptureRunReader(clip)
    assert gimbal_euler_from_quat(follow.telemetry(follow.rows[0]).q_gimbal)[2] == pytest.approx(37.0, abs=1e-4)
    assert any("yaw is taken from the vehicle" in a for a in follow.assumptions)

    north = CaptureRunReader(clip, gimbal_yaw_follows_vehicle=False)
    assert gimbal_euler_from_quat(north.telemetry(north.rows[0]).q_gimbal)[2] == pytest.approx(0.0, abs=1e-4)


def test_capture_run_truth_labels_become_schema_detections(tmp_path):
    clip = _write_capture_run(tmp_path, n=2)
    reader = CaptureRunReader(clip)
    raw = reader.truth_labels(1)
    assert raw and raw[0]["pose"] == "waving"

    (det,) = reader.truth_detections(1)
    # the mask box is inclusive of both edges; Detection.bbox_px is half-open
    assert det.bbox_px == (10.0, 12.0, 20.0, 26.0)
    assert det.width_px == 10.0 and det.height_px == 14.0
    assert det.size_px == raw[0]["size_px"]
    assert det.cls == "human"
    assert det.occlusion == 1
    assert det.submersion == "half" and det.submersion_conf == 1.0
    # "waving" is not in the frozen POSTURES vocabulary: it must become "unknown", never a neighbour
    # "waving" is a real authored posture, added to the frozen POSTURES vocabulary in SCHEMA_VERSION
    # 1.1.0 (docs/CONTRACTS.md section 5). The converter used to coerce it to "unknown"; it now passes
    # through with full confidence like any other known posture.
    assert det.posture == "waving" and det.posture_conf == 1.0


def test_capture_run_opens_as_a_clip_and_indexes_every_frame(tmp_path):
    clip = _write_capture_run(tmp_path, n=6)
    with open_clip(clip, decimate_k=3) as c:
        assert c.source_type == "capture_run" and c.domain == "sim"
        bundles = list(c)
        assert [b.frame_idx for b in bundles] == [0, 3]
        assert bundles[0].rgb is not None and bundles[0].rgb.shape == (36, 64, 3)
        assert bundles[0].intrinsics.hfov_deg() == pytest.approx(89.904, abs=1e-6)
        assert len(c.frame_index) == 6
        for i in range(6):
            assert int(c.load_frame(i).mean()) == i
        assert c.frame_index[2].aux["labels"].endswith("_00002.json")
        assert c.reports["data_card"]["scenario_seed"] == 23


def test_capture_run_frame_index_survives_frame_gaps(tmp_path):
    """`run.py` skips empty frames, so `frame_idx` has gaps. A gap is a gap, never a renumber."""
    clip = _write_capture_run(tmp_path, n=4)
    path = clip / "telemetry.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0]] + [lines[1], lines[3]]) + "\n", encoding="utf-8")
    reader = CaptureRunReader(clip)
    assert [int(r["frame_idx"]) for r in reader.rows] == [0, 2]
    assert reader.validate() == []
    assert sorted(r.frame_idx for r in reader.frame_index()) == [0, 2]


def test_capture_run_rejects_a_non_numeric_cell(tmp_path):
    """A malformed number must be loud: a silent 0.0 would poison every geolocation on that row."""
    clip = _write_capture_run(tmp_path, n=2)
    path = clip / "telemetry.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("1106.68", "n/a"), encoding="utf-8")
    with pytest.raises(CaptureFormatError, match="not a number"):
        CaptureRunReader(clip).rows


# ================================================================================ real autopilot logs ===
def test_ulog_sample_reads_into_a_utc_series():
    """`data/samples/px4_sample_log_small.ulg` is a real PX4 log committed for exactly this test."""
    path = SAMPLES / "px4_sample_log_small.ulg"
    if not path.is_file():
        pytest.skip(f"{path} is not present")
    from sightline.ingest.ulog import read_ulog

    series, report = read_ulog(path, clip_id="px4")
    assert len(series) > 5
    assert np.all(np.diff(series.t) >= 0.0)
    assert series.t[0] > 1.0e9, "timestamps must be mapped off the boot clock onto UTC"
    assert -90.0 <= float(series.lat[0]) <= 90.0
    assert -180.0 <= float(series.lon[0]) <= 180.0
    assert np.all(np.isfinite(series.alt_msl_m))
    assert np.allclose(np.linalg.norm(series.q_body, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(series.q_gimbal, axis=1), 1.0, atol=1e-6)
    # every fallback the reader took is stated rather than hidden
    assert report.warnings
    assert report.boot_clock is not None and report.boot_clock.n_pairs > 0
    assert detect_source(path) == "ulog"


def _synthetic_tlog(path: Path, n: int = 20) -> Path:
    """A real `.tlog`: 8-byte big-endian microsecond stamp + the raw MAVLink 2 frame, per message."""
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink.dialects.v20 import common as mav

    class _Sink:
        def __init__(self):
            self.buf = bytearray()

        def write(self, data):
            self.buf += data

    sink = _Sink()
    link = mav.MAVLink(sink, srcSystem=1, srcComponent=1)
    out = bytearray()

    def emit(msg, t_us: int) -> None:
        sink.buf.clear()
        link.send(msg)
        out.extend(struct.pack(">Q", t_us) + bytes(sink.buf))

    epoch = int(T0)
    for i in range(n):
        boot_ms = 1000 + i * 100
        t_us = (epoch + i) * 1_000_000
        emit(mav.MAVLink_system_time_message(time_unix_usec=epoch * 1_000_000 + boot_ms * 1000,
                                             time_boot_ms=boot_ms), t_us)
        emit(mav.MAVLink_global_position_int_message(
            time_boot_ms=boot_ms, lat=int(ORIGIN_LAT * 1e7) + i * 100, lon=int(ORIGIN_LON * 1e7),
            alt=645_000, relative_alt=45_000, vx=300, vy=0, vz=0, hdg=0), t_us)
        emit(mav.MAVLink_attitude_message(time_boot_ms=boot_ms, roll=0.0, pitch=0.0, yaw=0.5,
                                          rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0), t_us)
        emit(mav.MAVLink_gps_raw_int_message(
            time_usec=t_us, fix_type=3, lat=int(ORIGIN_LAT * 1e7), lon=int(ORIGIN_LON * 1e7),
            alt=645_000, eph=150, epv=200, vel=300, cog=0, satellites_visible=12), t_us)
    path.write_bytes(bytes(out))
    return path


def test_mavlink_tlog_reads_the_messages_the_doc_names(tmp_path):
    """§5.4 row 4. The log is synthesised here because no real flight log ships with this repo -- which is
    also stated in `mavlink.py`, so the test does not pretend to be more than it is."""
    path = _synthetic_tlog(tmp_path / "flight.tlog")
    assert detect_source(path) == "mavlink"
    from sightline.ingest.mavlink import read_mavlink

    series, report = read_mavlink(path, clip_id="m")
    assert len(series) == 20
    assert report.counts["GLOBAL_POSITION_INT"] == 20
    assert report.counts["SYSTEM_TIME"] == 20

    assert float(series.lat[0]) == pytest.approx(ORIGIN_LAT, abs=1e-6)
    assert float(series.alt_msl_m[0]) == pytest.approx(645.0)      # mm -> m
    assert float(series.agl_m[0]) == pytest.approx(45.0)           # relative_alt, mm -> m
    assert float(series.h_acc_m[0]) == pytest.approx(1.5)          # eph 150 cm -> 1.5 m
    assert float(series.v_acc_m[0]) == pytest.approx(2.0)
    assert np.allclose(series.vel_ned_ms[0], (3.0, 0.0, 0.0))      # cm/s -> m/s
    assert series.t[0] > 1.0e9 and abs(series.t[0] - (T0 + 1.0)) < 0.01
    assert report.boot_clock.n_pairs == 20
    # no GIMBAL_DEVICE_ATTITUDE_STATUS in this log: the nadir substitution must be declared
    assert any("nadir" in w for w in report.warnings)
    assert gimbal_euler_from_quat(tuple(series.q_gimbal[0]))[1] == pytest.approx(-90.0, abs=1e-6)


# ====================================================================================== source router ===
def test_detect_source_routes_every_documented_source(tmp_path):
    (tmp_path / "a.srt").write_text(SRT_OLD, encoding="utf-8")
    assert detect_source(tmp_path / "a.srt") == "dji_srt"
    assert detect_source(tmp_path) == "dji_srt"          # a directory holding one

    (tmp_path / "b.ulg").write_bytes(b"\x00")
    assert detect_source(tmp_path / "b.ulg") == "ulog"
    (tmp_path / "c.tlog").write_bytes(b"\x00")
    assert detect_source(tmp_path / "c.tlog") == "mavlink"

    (tmp_path / "d.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(CaptureFormatError):
        detect_source(tmp_path / "d.txt")
    with pytest.raises(FileNotFoundError):
        detect_source(tmp_path / "missing")


def test_a_bare_video_is_refused_with_an_actionable_message(tmp_path):
    video = tmp_path / "DJI_0001.MP4"
    video.write_bytes(b"\x00" * 16)
    assert detect_source(video) == "video"
    with pytest.raises(CaptureFormatError, match="no telemetry"):
        open_clip(video)


# ============================================================================== the remaining readers ===
def test_thermal_frames_are_read_as_centi_kelvin_not_truncated_to_8_bit(tmp_path):
    """§5.5b: a radiometric thermal frame is a 16-bit PNG of centi-kelvin. `IMREAD_GRAYSCALE` silently
    truncates it to 8 bits, which would turn 34.0 C into noise -- so the reader must use IMREAD_UNCHANGED
    and must refuse a frame that is declared radiometric but stored as 8-bit."""
    import cv2

    from sightline.ingest.decode import read_thermal

    kelvin_centi = np.full((8, 12), int(round((34.0 + 273.15) * 100)), dtype=np.uint16)
    radiometric = tmp_path / "t16.png"
    cv2.imwrite(str(radiometric), kelvin_centi)

    raw = read_thermal(radiometric, radiometric=True)
    assert raw.dtype == np.uint16 and raw.shape == (8, 12)
    assert float(raw[0, 0]) == pytest.approx(30715.0)
    assert (raw.astype(np.float32) / 100.0 - 273.15)[0, 0] == pytest.approx(34.0, abs=0.01)

    agc = tmp_path / "t8.png"
    cv2.imwrite(str(agc), np.full((8, 12), 200, dtype=np.uint8))
    assert read_thermal(agc, radiometric=False).dtype == np.uint8
    with pytest.raises(ValueError, match="uint16"):
        read_thermal(agc, radiometric=True)
    with pytest.raises(FileNotFoundError):
        read_thermal(tmp_path / "missing.png", radiometric=False)


def test_thermal_partner_frames_reach_the_bundle(tmp_path):
    """The thermal half of a `FrameBundle` (§5.5b), end to end through the capture spec."""
    import cv2

    clip = tmp_path / "thermal"
    (clip / "frames").mkdir(parents=True)
    (clip / "thermal").mkdir(parents=True)
    manifest = CaptureManifest(
        clip_id="th", domain="sim", capture_start_utc=T0,
        origin_geopoint={"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "alt_msl_m": 600.0},
        camera=CameraSpec(name="survey", width_px=32, height_px=18, hfov_deg=60.0),
        thermal_camera=CameraSpec(name="ir", width_px=16, height_px=12, hfov_deg=45.0,
                                  radiometric=True, units="centi_kelvin"),
    )
    writer = SimCaptureWriter(clip, manifest).open()
    q = gimbal_quat_from_euler(0.0, -90.0, 0.0)
    for i in range(3):
        cv2.imwrite(str(clip / f"frames/rgb_{i:07d}.png"), np.full((18, 32, 3), 5, np.uint8))
        cv2.imwrite(str(clip / f"thermal/ir_{i:07d}.png"),
                    np.full((12, 16), int(round((36.5 + 273.15) * 100)), np.uint16))
        writer.append(
            frame_idx=i, t_utc=T0 + i / 30.0, rgb_path=f"frames/rgb_{i:07d}.png",
            thermal_path=f"thermal/ir_{i:07d}.png", thermal_radiometric=1,
            ned_n_m=0.0, ned_e_m=0.0, ned_d_m=-45.0,
            q_body_w=1.0, q_body_x=0.0, q_body_y=0.0, q_body_z=0.0,
            q_gimbal_w=q[0], q_gimbal_x=q[1], q_gimbal_y=q[2], q_gimbal_z=q[3],
            lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_msl_m=645.0, agl_m=45.0,
            img_w_px=32, img_h_px=18, hfov_deg=60.0)
    writer.close()

    assert validate_capture(clip) == []
    with open_clip(clip) as c:
        bundle = next(iter(c))
        assert bundle.thermal_is_radiometric is True
        assert bundle.thermal is not None and bundle.thermal.dtype == np.uint16
        assert bundle.thermal.shape == (12, 16)
        assert bundle.thermal_intrinsics is not None
        assert bundle.thermal_intrinsics.width_px == 16
        assert bundle.thermal_intrinsics.hfov_deg() == pytest.approx(45.0, abs=1e-6)
        celsius = bundle.thermal_celsius()
        assert celsius is not None and float(celsius[0, 0]) == pytest.approx(36.5, abs=0.01)


def test_frames_jsonl_is_an_equivalent_telemetry_table(tmp_path):
    """§5.4 / spec.py: `frames.jsonl` is accepted instead of `frames.csv`, and a truncated final line --
    what a killed editor leaves behind -- costs one row, not the clip."""
    clip = _write_sim_capture(tmp_path, n=5)
    rows = list(SimExportReader(clip).rows)
    lines = [json.dumps({k: v for k, v in row.values.items() if v is not None}) for row in rows]
    (clip / "frames.jsonl").write_text("\n".join(lines) + "\n" + lines[0][:20], encoding="utf-8")
    (clip / "frames.csv").unlink()

    reader = SimExportReader(clip)
    assert len(reader.rows) == 5, "the truncated tail line must be skipped, not fatal"
    assert reader.telemetry(reader.rows[2]).ned_m == pytest.approx((4.0, 1.0, -45.0))


def test_airsim_rec_is_read_with_its_assumptions_declared(tmp_path):
    """`airsim_rec.txt` carries no GPS, no intrinsics and no camera pose (verified against
    `RecordingFile.cpp`). Everything supplied from outside is listed as an assumption."""
    from sightline.ingest.sim import AirSimRecReader, is_airsim_rec

    clip = tmp_path / "rec"
    (clip / "images").mkdir(parents=True)
    header = "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile"
    lines = [header]
    for i in range(4):
        name = f"img_Drone_survey_0_{1000 + i}.png"
        seg = f"img_Drone_survey_5_{1000 + i}.png"
        (clip / "images" / name).write_bytes(b"\x00")
        # TimeStamp is milliseconds on the AIRSIM clock (SteppableClock), not wall time
        lines.append(f"Drone\t{100000 + i * 200}\t{2.0 * i}\t{0.5 * i}\t-45.0\t1.0\t0.0\t0.0\t0.0\t{name};{seg}")
    (clip / "airsim_rec.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert is_airsim_rec(clip)
    assert detect_source(clip) == "airsim_rec"

    reader = AirSimRecReader(clip, origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON, origin_alt_msl_m=600.0,
                             hfov_deg=75.5, width_px=3840, height_px=2160, capture_start_utc=T0,
                             ground_alt_msl_m=600.0)
    rows = reader.rows()
    assert rows is reader.rows(), "rows are cached: telemetry() needs t0 per row and re-reading was O(n^2)"
    assert len(rows) == 4
    assert rows[1].t_sim_s == pytest.approx(100.2)          # ms -> s
    assert rows[0].images["scene"].endswith("_0_1000.png")
    assert rows[0].images["segmentation"].endswith("_5_1000.png")

    tel = reader.telemetry(rows[2])
    assert tel.t_utc == pytest.approx(T0 + 0.4)             # clock rebased on capture_start_utc
    assert tel.alt_msl_m == pytest.approx(645.0)            # origin_alt - POS_Z
    assert tel.agl_m == pytest.approx(45.0)
    assert tel.gimbal_pitch_deg() == pytest.approx(-90.0)   # the documented -90 ASSUMPTION
    north, east = geodesy.ne_between(ORIGIN_LAT, ORIGIN_LON, tel.lat, tel.lon)
    assert (north, east) == pytest.approx((4.0, 1.0), abs=1e-3)

    assert any("no camera orientation" in a for a in reader.assumptions)
    assert any("no GPS" in a for a in reader.assumptions)
    assert any("no intrinsics" in a for a in reader.assumptions)
    index = reader.frame_index()
    assert len(index) == 4 and index[0].path.endswith("_0_1000.png")


def test_airsim_rec_without_a_utc_anchor_says_the_clock_is_not_wall_time(tmp_path):
    """HANDBOOK §5 / spec.py's clock rule: under SteppableClock the AirSim timestamp is not wall time."""
    from sightline.ingest.sim import AirSimRecReader

    clip = tmp_path / "rec2"
    clip.mkdir()
    (clip / "airsim_rec.txt").write_text(
        "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile\n"
        "Drone\t1000\t0\t0\t-45\t1\t0\t0\t0\t\n", encoding="utf-8")
    reader = AirSimRecReader(clip, origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON, origin_alt_msl_m=600.0,
                             hfov_deg=75.5, width_px=1920, height_px=1080)
    assert any("NOT wall time" in a for a in reader.assumptions)


def test_airsim_rec_needs_its_missing_parameters_supplied(tmp_path):
    clip = tmp_path / "rec3"
    clip.mkdir()
    (clip / "airsim_rec.txt").write_text(
        "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile\n"
        "Drone\t1000\t0\t0\t-45\t1\t0\t0\t0\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="OriginGeopoint"):
        open_clip(clip)


def test_a_bare_srt_opens_as_a_metadata_only_clip(tmp_path):
    """§5.12's evaluation harness replays metadata with no pixels; ingest must support that."""
    srt = tmp_path / "DJI_0002.SRT"
    srt.write_text(SRT_HTML, encoding="utf-8")
    with open_clip(srt, takeoff_alt_msl_m=600.0) as c:
        assert c.source_type == "dji_srt" and c.domain == "real"
        bundles = list(c)
        assert [b.frame_idx for b in bundles] == [0, 1]
        assert all(b.rgb is None for b in bundles)
        assert bundles[0].telemetry.alt_msl_m == pytest.approx(645.0)
        assert bundles[1].telemetry.gimbal_pitch_deg() == pytest.approx(-60.0, abs=1e-6)
        assert c.reports["srt"].formats == {"format3": 1, "format3b": 1}


def test_srt_frame_index_stays_aligned_when_an_entry_has_no_position(tmp_path):
    """REGRESSION: entries with no lat/lon are dropped from the telemetry samples. Zipping the two lists
    positionally shifted every later frame's index and PTS by one."""
    text = SRT_HTML.replace("[latitude: 12.900000] [longitude: 77.600000] ", "", 1)
    srt = tmp_path / "gap.SRT"
    srt.write_text(text, encoding="utf-8")
    with open_clip(srt, takeoff_alt_msl_m=600.0) as c:
        refs = list(c.frame_index)
        assert len(refs) == 1, "the position-less entry cannot become a frame"
        assert refs[0].frame_idx == 1, "the surviving frame must keep ITS index, not inherit index 0"
        assert refs[0].t_utc == pytest.approx(1789041600.5)
        assert refs[0].pts_s == pytest.approx(0.033)
