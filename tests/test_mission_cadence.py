"""The survey cadence must be able to confirm a track, and survey.py must refuse to fly when it cannot.

This is the regression test for the 2026-09-11 defect: a 285-frame survey produced 56 detections, 56
geolocations and then **0 tracks, 0 records, 0 ranked** - every stage after geolocation was empty. Nothing was
broken; the shutter was 26 m at 11 m/s, so frames were 2.36 s apart and a survivor was in frame for 1.5 of
them, and SOLUTION_DOC 5.6 rule 3 needs 3 hits inside 2 s before a track is emitted. The flight plan made the
second half of the system impossible and no check said so.
"""

from __future__ import annotations

import math

import pytest

from sightline.mission import survey
from sightline.track.config import TrackerConfig
from tools.capture.campaign import shutter_m

HFOV = 73.983


def frame_height_m(alt: float, hfov_deg: float = HFOV, w: int = 3840, h: int = 2160) -> float:
    return 2.0 * alt * math.tan(math.radians(hfov_deg) / 2.0) * h / w


def test_survey_constants_match_the_tracker() -> None:
    """survey.py copies the gate rather than importing the tracker stack; the copy must not drift."""
    cfg = TrackerConfig()
    assert survey.MIN_HITS == cfg.min_hits
    assert survey.MIN_HITS_WINDOW_S == cfg.min_hits_window_s


@pytest.mark.parametrize("alt,speed", [(35.0, 12.0), (55.0, 12.0), (80.0, 12.0), (45.0, 10.0)])
def test_derived_shutter_can_confirm_a_track(alt: float, speed: float) -> None:
    """Whatever `campaign.shutter_m` returns must satisfy BOTH halves of the confirmation gate."""
    s = shutter_m(alt, speed, HFOV)
    hm = frame_height_m(alt)
    span = (survey.MIN_HITS - 1) * s / speed
    hits = hm / s
    assert span <= survey.MIN_HITS_WINDOW_S + 1e-9, (
        f"{survey.MIN_HITS} shots span {span:.2f} s at {alt} m / {speed} m/s, "
        f"outside the {survey.MIN_HITS_WINDOW_S} s window")
    assert hits >= survey.MIN_HITS - 1e-9, (
        f"a target is in frame for only {hits:.2f} shots at {alt} m with a {s} m shutter")


def test_the_cadence_that_produced_zero_tracks_is_rejected() -> None:
    """The exact settings of the 2026-09-11 run must fail both halves of the gate.

    If this ever passes, the gate has been loosened and the pipeline will silently go empty again.
    """
    alt, speed, shutter = 45.0, 11.0, 26.0
    hm = frame_height_m(alt)
    span = (survey.MIN_HITS - 1) * shutter / speed
    hits = hm / shutter
    assert span > survey.MIN_HITS_WINDOW_S, f"expected the 2026-09-11 cadence to bust the window, got {span:.2f} s"
    assert hits < survey.MIN_HITS, f"expected fewer than {survey.MIN_HITS} hits, got {hits:.2f}"
    # and the fix the guard suggests must itself pass
    need = min(speed * survey.MIN_HITS_WINDOW_S / (survey.MIN_HITS - 1), hm / survey.MIN_HITS)
    assert (survey.MIN_HITS - 1) * need / speed <= survey.MIN_HITS_WINDOW_S + 1e-9
    assert hm / need >= survey.MIN_HITS - 1e-9
