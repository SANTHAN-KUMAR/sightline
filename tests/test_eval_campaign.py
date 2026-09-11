"""Tests for `sightline.eval.campaign` — the acceptance report's structure (SOLUTION_DOC 5.5c / 5.12).

Runnable offline: every test builds its own data card or dataset. The point of this module is that the
acceptance slice is fixed BEFORE measuring and cannot be widened afterwards, so most of these tests are about
what it REFUSES to do.
"""

from __future__ import annotations

import json

import pytest

from sightline.eval.campaign import (
    NOMINAL_SLICE,
    PassSpec,
    SliceRoleError,
    acceptance_spec,
    camera_track,
    census,
    refuse_mixed_roles,
    render_census,
    role_of,
)
from sightline.eval.groundtruth import EvalDataset, GtBox, GtFrame
from sightline.eval.slicing import altitude_band


def card(agl: float, condition: str = "clear_midday", **weather: float) -> dict:
    return {"altitude_m_agl": agl, "condition": condition, "frames": 100, "minutes": 10.0,
            "total_boxes": 50, "weather": weather or {"rain": None, "fog": None}}


# --------------------------------------------------------------------------- roles, fixed in advance

@pytest.mark.parametrize("agl,expect", [(35.0, "hard"), (39.9, "hard"), (40.0, "acceptance"),
                                        (55.0, "acceptance"), (60.0, "acceptance"), (80.0, "hard")])
def test_the_role_follows_the_nominal_band_and_not_a_list_of_pass_names(agl, expect):
    assert role_of(card(agl)).role == expect


def test_weather_inside_the_band_is_reported_separately_never_folded_in():
    s = role_of(card(45.0, "rain_overcast", rain=0.55, fog=0.25), "alt45rain")
    assert s.role == "separate"
    assert "rain 0.55" in s.why and "SEPARATELY" in s.why


def test_the_reason_is_recorded_with_the_role():
    """5.5c step 5: define it before you measure it. A role with no stated reason is a choice made later."""
    for spec in (role_of(card(35.0)), role_of(card(55.0)), role_of(card(45.0, rain=0.5))):
        assert spec.why and len(spec.why) > 30


def test_the_altitude_band_axis_cannot_separate_the_rain_pass_from_the_acceptance_pass():
    """The trap this module exists to close, asserted rather than described.

    If `ALTITUDE_BANDS` is ever re-cut so that 45 m and 55 m fall in different bands, this test fails and the
    `SliceRoleError` machinery can be reconsidered — it should not quietly become redundant.
    """
    assert altitude_band(45.0) == altitude_band(55.0) == "45-60"
    rain = role_of(card(45.0, "rain_overcast", rain=0.55), "alt45rain")
    clear = role_of(card(55.0), "alt55")
    assert rain.role != clear.role, "only the ROLE separates them; the slice grid does not"


def test_the_nominal_slice_is_the_one_the_doc_states():
    assert NOMINAL_SLICE["agl_m"] == (40.0, 60.0)
    assert NOMINAL_SLICE["occlusion_below"] == 0.50
    assert "5.5c" in NOMINAL_SLICE["source"]


# --------------------------------------------------------------------------- what it refuses

def test_pooling_passes_of_different_roles_raises():
    specs = [role_of(card(35.0), "alt35"), role_of(card(55.0), "alt55")]
    with pytest.raises(SliceRoleError, match="different roles"):
        refuse_mixed_roles(specs)


def test_pooling_passes_of_the_same_role_is_allowed():
    refuse_mixed_roles([role_of(card(35.0), "alt35"), role_of(card(80.0), "alt80")])  # both hard


def test_the_headline_needs_exactly_one_acceptance_pass():
    one = [role_of(card(55.0), "alt55"), role_of(card(35.0), "alt35")]
    assert acceptance_spec(one).name == "alt55"

    with pytest.raises(SliceRoleError, match="found 0"):
        acceptance_spec([role_of(card(35.0), "alt35")])

    # The real case: `_artifacts/dataset/train_seed23` is an abandoned 45 m pass whose card is
    # indistinguishable from a campaign pass, so it scores `acceptance` too.
    with pytest.raises(SliceRoleError, match="found 2"):
        acceptance_spec([role_of(card(55.0), "alt55"), role_of(card(45.0), "train_seed23")])


def test_naming_a_pass_that_is_not_there_is_an_error_not_a_silent_skip(tmp_path):
    from sightline.eval.campaign import load_campaign

    (tmp_path / "alt55").mkdir()
    (tmp_path / "alt55" / "data_card.json").write_text(json.dumps(card(55.0)), encoding="utf-8")
    with pytest.raises(SliceRoleError, match="not found"):
        load_campaign(tmp_path, include=["alt55", "alt99"])


# --------------------------------------------------------------------------- the census

def _dataset(*boxes: GtBox) -> EvalDataset:
    return EvalDataset(domain="sim", frames=[GtFrame(frame_idx=0, boxes=list(boxes), agl_m=55.0)])


def test_the_census_counts_scored_boxes_and_says_what_it_excluded():
    ds = _dataset(
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=1, context="structure", posture="standing", occlusion=0),
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=2, context="water", posture="prone", occlusion=1),
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=3, context="water", uncertain=True),   # 6.3: not scored
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=4, context="debris", is_group=True),   # 6.3: not scored
    )
    ms = census(ds, role_of(card(55.0), "alt55"))
    got = {r.name: r.value for r in ms.rows if r.slice.label() == "domain=sim"}
    assert got["boxes_scored"] == 2
    assert got["boxes_excluded"] == 2
    assert got["unique_gt_ids"] == 2


def test_recall_resolution_says_what_one_box_is_worth():
    """A 90 % claim over 40 boxes moves 2.5 points if one flips. That belongs next to the count."""
    ds = _dataset(*[GtBox(bbox_px=(0, 0, 10, 20), gt_id=i, context="water") for i in range(40)])
    ms = census(ds, role_of(card(55.0), "alt55"))
    res = next(r for r in ms.rows if r.name == "recall_resolution")
    assert res.value == pytest.approx(1 / 40)


def test_an_empty_pass_reports_n_zero_rather_than_an_infinity():
    """`metric_row` refuses non-finite values outright — an infinity would be a number nobody can read."""
    ms = census(_dataset(), role_of(card(55.0), "alt55"))
    res = next(r for r in ms.rows if r.name == "recall_resolution")
    assert res.n == 0 and res.value == 0.0
    assert "undefined" in str(res.detail)


def test_the_census_breaks_boxes_down_on_the_terrain_axis():
    ds = _dataset(
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=1, context="structure"),
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=2, context="structure"),
        GtBox(bbox_px=(0, 0, 10, 20), gt_id=3, context="water"),
    )
    ms = census(ds, role_of(card(55.0), "alt55"))
    by_ctx = {r.slice.context: r.value for r in ms.rows
              if r.name == "boxes_scored" and r.slice.context != "all"}
    assert by_ctx == {"structure": 2.0, "water": 1.0}


# --------------------------------------------------------------------------- telemetry

def test_camera_track_reads_the_columns_the_capture_reader_drops(tmp_path):
    """`detect.dataset.load_run` keeps alt_msl_m/lat/lon but not east_m/north_m."""
    (tmp_path / "telemetry.csv").write_text(
        "frame_idx,east_m,north_m,alt_msl_m,agl_m\n0,10.5,-20.25,1100.0,45.0\n1,11.0,-21.0,1101.0,46.0\n",
        encoding="utf-8")
    track = camera_track(tmp_path)
    assert track[0] == (10.5, -20.25, 1100.0)
    assert track[1] == (11.0, -21.0, 1101.0)


def test_a_missing_telemetry_file_gives_no_poses_rather_than_zeros(tmp_path):
    """(0, 0, 0) would project every box to the map origin and look like data."""
    assert camera_track(tmp_path) == {}


# --------------------------------------------------------------------------- the rendered report

def test_the_report_leads_with_roles_and_names_the_single_acceptance_pass():
    rows = []
    for name, c in (("alt35", card(35.0, "clear_morning")), ("alt55", card(55.0)),
                    ("alt45rain", card(45.0, "rain_overcast", rain=0.55))):
        spec = role_of(c, name)
        ds = _dataset(GtBox(bbox_px=(0, 0, 10, 20), gt_id=1, context="water"))
        rows.append((ds, spec, census(ds, spec)))
    md = render_census(rows)
    assert "Acceptance figure comes from `alt55` alone" in md
    assert "**hard**" in md and "**separate**" in md
    assert "roof / structure" in md or "5.12" in md


def test_the_report_says_so_when_there_is_no_single_acceptance_pass():
    rows = []
    for name, c in (("alt55", card(55.0)), ("train_seed23", card(45.0))):
        spec = role_of(c, name)
        ds = _dataset(GtBox(bbox_px=(0, 0, 10, 20), gt_id=1, context="water"))
        rows.append((ds, spec, census(ds, spec)))
    md = render_census(rows)
    assert "No single acceptance pass" in md and "found 2" in md
