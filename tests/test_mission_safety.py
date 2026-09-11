"""The flown plan must be checkable against F2's geofence, ceiling and battery reserve.

These exist because the constraints in `sightline/plan/constraints.py` are fully implemented and fully
tested and are NOT REFERENCED by either flight module. `sightline/mission/survey.py` and
`sightline/mission/live.py` between them contain no mention of `Constraints`, `geofence` or `Battery`, so the
drone that flew the 2026-09-11 capture campaign enforced none of them. A feature that is done in a library
and absent from the thing that runs is not done.
"""

from __future__ import annotations

import numpy as np
import pytest

from sightline.mission.pattern import Leg, SurveyPlan
from sightline.mission.safety import check_plan
from sightline.plan.constraints import Battery, Constraints


def plan_of(legs: list[Leg], alt_m: float = 45.0, speed_ms: float = 12.0) -> SurveyPlan:
    return SurveyPlan(legs=legs, plan="patches", alt_m=alt_m, speed_ms=speed_ms, line_spacing_m=50.0,
                      frame_w_m=68.0, frame_h_m=38.0, shutter_m=12.0, survivors_in_plan=10,
                      survivors_total=10)


#: a 400 m box centred on the origin, as (north, east) pairs
FENCE = np.array([[-200.0, -200.0], [-200.0, 200.0], [200.0, 200.0], [200.0, -200.0]])


def test_a_plan_inside_every_constraint_passes() -> None:
    p = plan_of([Leg(east_m=0.0, north_start_m=-100.0, north_end_m=100.0)])
    v = check_plan(p, Constraints(home_ne=(0.0, 0.0), geofence_ne=FENCE,
                                  battery=Battery(endurance_s=3600.0)))
    assert v.ok, v.summary()
    assert v.battery_truncates_at is None
    assert v.flyable_legs == v.total_legs == 1


def test_a_leg_outside_the_geofence_is_caught() -> None:
    p = plan_of([Leg(east_m=0.0, north_start_m=-100.0, north_end_m=100.0),
                 Leg(east_m=900.0, north_start_m=100.0, north_end_m=-100.0)])   # far outside
    v = check_plan(p, Constraints(home_ne=(0.0, 0.0), geofence_ne=FENCE,
                                  battery=Battery(endurance_s=36000.0)))
    assert not v.ok
    kinds = {x.kind for x in v.violations}
    assert "geofence" in kinds, v.summary()
    assert any(x.leg_index == 1 for x in v.violations if x.kind == "geofence")


def test_the_ceiling_and_the_floor_are_both_enforced() -> None:
    legs = [Leg(east_m=0.0, north_start_m=-50.0, north_end_m=50.0)]
    high = check_plan(plan_of(legs, alt_m=150.0), Constraints(home_ne=(0.0, 0.0)))
    assert any(x.kind == "ceiling" for x in high.violations), high.summary()
    low = check_plan(plan_of(legs, alt_m=1.0), Constraints(home_ne=(0.0, 0.0)))
    assert any(x.kind == "min_agl" for x in low.violations), low.summary()


def test_the_battery_reserve_truncates_and_says_the_rest_is_UNSEARCHED() -> None:
    """The wording matters: R10 forbids the system ever implying an area was cleared."""
    legs = [Leg(east_m=e, north_start_m=-150.0, north_end_m=150.0) for e in (0.0, 50.0, 100.0, 150.0)]
    v = check_plan(plan_of(legs, speed_ms=10.0),
                   Constraints(home_ne=(0.0, 0.0), geofence_ne=FENCE,
                               battery=Battery(endurance_s=120.0, reserve_frac=0.2, rtl_speed_ms=12.0)))
    assert not v.ok
    assert v.battery_truncates_at is not None
    assert v.flyable_legs < v.total_legs
    msg = " ".join(x.detail for x in v.violations if x.kind == "battery")
    assert "UNSEARCHED" in msg and "not clear" in msg, msg


def test_transits_between_legs_are_counted_not_ignored() -> None:
    """Two legs far apart must cost more battery than the same two legs adjacent.

    Ignoring transit is how a plan that looks affordable on paper runs out of battery in the air: on the
    2026-09-11 campaign, patch-to-patch transits were 48 % of elapsed time.
    """
    near = [Leg(east_m=0.0, north_start_m=0.0, north_end_m=100.0),
            Leg(east_m=10.0, north_start_m=100.0, north_end_m=0.0)]
    far = [Leg(east_m=0.0, north_start_m=0.0, north_end_m=100.0),
           Leg(east_m=1500.0, north_start_m=100.0, north_end_m=0.0)]
    c = Constraints(home_ne=(0.0, 0.0), battery=Battery(endurance_s=100000.0))
    assert check_plan(plan_of(far), c).est_flight_s > check_plan(plan_of(near), c).est_flight_s + 100.0


def test_an_operator_no_go_area_is_honoured_only_when_set() -> None:
    """SOLUTION_DOC 5.2: a no-fly buffer around buried polygons exists ONLY if the commander sets one."""
    legs = [Leg(east_m=0.0, north_start_m=-50.0, north_end_m=50.0)]
    plain = Constraints(home_ne=(0.0, 0.0), geofence_ne=FENCE, battery=Battery(endurance_s=36000.0))
    assert check_plan(plan_of(legs), plain).ok

    box = np.array([[-60.0, -60.0], [-60.0, 60.0], [60.0, 60.0], [60.0, -60.0]])
    with_nogo = Constraints(home_ne=(0.0, 0.0), geofence_ne=FENCE, no_go_ne=[box],
                            battery=Battery(endurance_s=36000.0))
    v = check_plan(plan_of(legs), with_nogo)
    assert any(x.kind == "no_go" for x in v.violations), v.summary()


@pytest.mark.parametrize("alt", [35.0, 45.0, 55.0, 80.0])
def test_every_altitude_the_campaign_flies_is_legal(alt: float) -> None:
    """The four capture passes must all sit inside the 120 m ceiling and above the floor."""
    v = check_plan(plan_of([Leg(east_m=0.0, north_start_m=-50.0, north_end_m=50.0)], alt_m=alt),
                   Constraints(home_ne=(0.0, 0.0), battery=Battery(endurance_s=36000.0)))
    assert not any(x.kind in ("ceiling", "min_agl") for x in v.violations), v.summary()
