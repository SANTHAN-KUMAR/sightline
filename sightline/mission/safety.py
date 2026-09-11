"""Check a flown survey plan against the F2 constraints the flight code does not currently enforce.

    from sightline.mission.safety import check_plan
    verdict = check_plan(plan, constraints)      # plan: mission.pattern.SurveyPlan

There are two planners in this project and only one of them knows about safety.

* `sightline/plan/` is the doc-complete one - boustrophedon, expanding square, orbit-on-detection, a revisit
  queue, and `constraints.Constraints` with a battery RTL reserve, a geofence, the 120 m ceiling and operator
  no-go areas. It has 31 tests, including one that asserts a short battery truncates a route and appends an
  RTL leg whose reason says "battery reserve".
* `sightline/mission/pattern.py` is the one that actually FLIES, via `survey.py` and `live.py`.

Nothing connects them. Grep either flight module for `Constraints`, `geofence` or `Battery` and you get
nothing: **the drone that flies the capture campaign enforces no geofence, no battery reserve and no
ceiling.** The constraints are written, tested, and out of the loop - which is the failure mode where a
feature is "done" on paper and absent in the thing that runs.

This module is the missing join, kept deliberately thin and separate:

* it evaluates a `SurveyPlan` against `Constraints` WITHOUT importing anything from the flight path, so it
  can be added to `survey.py` and `live.py` later without a circular import and without touching a script
  that is mid-flight;
* it walks the legs in order, accumulating flight time exactly as the vehicle would, and reports the first
  leg at which the battery could no longer reach home inside its reserve - the same rule `constraints.apply`
  uses, so the two cannot disagree about when to turn back;
* it REPORTS rather than mutates. A planner that silently shortens a route hides the fact that the area was
  never searched, and `docs/SOLUTION_DOC.md` R10 is explicit that the system must never quietly conclude a
  search. The caller decides what to do; the verdict says what is true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sightline.mission.pattern import SurveyPlan
from sightline.plan.constraints import Constraints


@dataclass(slots=True)
class Violation:
    """One constraint the plan breaks, named so a reader knows what to change."""

    kind: str                 # "geofence" | "no_go" | "ceiling" | "min_agl" | "battery"
    leg_index: int            # -1 when the violation is a property of the whole plan
    detail: str


@dataclass(slots=True)
class SafetyVerdict:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    #: index of the first leg the battery cannot afford, or None when the whole plan fits inside the reserve
    battery_truncates_at: int | None = None
    flyable_legs: int = 0
    total_legs: int = 0
    est_flight_s: float = 0.0
    est_usable_s: float = 0.0

    def summary(self) -> str:
        if self.ok:
            return (f"plan is inside all constraints: {self.total_legs} legs, "
                    f"{self.est_flight_s / 60:.1f} min of {self.est_usable_s / 60:.1f} min usable")
        head = f"{len(self.violations)} constraint violation(s) over {self.total_legs} legs"
        if self.battery_truncates_at is not None:
            head += (f"; battery forces RTL at leg {self.battery_truncates_at} "
                     f"({self.flyable_legs}/{self.total_legs} legs flyable)")
        return head + "\n  " + "\n  ".join(f"{v.kind}[leg {v.leg_index}]: {v.detail}" for v in
                                           self.violations[:10])


def check_plan(plan: SurveyPlan, c: Constraints, *, transit_speed_ms: float | None = None) -> SafetyVerdict:
    """Walk `plan` leg by leg and report every constraint it breaks. Nothing is mutated.

    Time is accumulated the way the vehicle actually spends it: the transit from the previous leg's end to
    this leg's start, then the leg itself. Ignoring transits is how a plan that looks affordable on paper
    runs out of battery in the air - on the 2026-09-11 campaign, patch-to-patch transits were 48 % of
    elapsed time.
    """
    v: list[Violation] = []
    alt = plan.alt_m
    if alt > c.ceiling_m:
        v.append(Violation("ceiling", -1,
                           f"commanded {alt:.0f} m AGL is above the {c.ceiling_m:.0f} m ceiling"))
    if alt < c.min_agl_m:
        v.append(Violation("min_agl", -1,
                           f"commanded {alt:.0f} m AGL is below the {c.min_agl_m:.0f} m floor"))

    transit = transit_speed_ms if transit_speed_ms is not None else plan.speed_ms
    elapsed = 0.0
    here = tuple(c.home_ne)
    truncate_at: int | None = None
    flyable = 0

    for i, leg in enumerate(plan.legs):
        start = (leg.north_start_m, leg.east_m)
        end = (leg.north_end_m, leg.east_m)
        for name, pt in (("start", start), ("end", end)):
            if not c.inside_geofence(pt):
                v.append(Violation("geofence", i,
                                   f"leg {name} N{pt[0]:.0f} E{pt[1]:.0f} is outside the geofence"))
            if c.no_go_ne and c.inside_no_go(pt):
                v.append(Violation("no_go", i,
                                   f"leg {name} N{pt[0]:.0f} E{pt[1]:.0f} is inside an operator no-go area"))

        elapsed += math.dist(here, start) / max(transit, 0.1)      # transit onto the line
        elapsed += leg.length_m / max(plan.speed_ms, 0.1)          # the line itself
        here = end
        if truncate_at is None and not c.battery.can_continue(elapsed, here, c.home_ne):
            truncate_at = i
            v.append(Violation("battery", i,
                               f"after this leg the vehicle needs "
                               f"{c.battery.time_home_s(here, c.home_ne):.0f} s to reach home and would have "
                               f"spent {elapsed:.0f} s of {c.battery.usable_s:.0f} s usable - the reserve is "
                               f"gone. Everything after this leg is UNSEARCHED, not clear."))
        if truncate_at is None:
            flyable = i + 1

    return SafetyVerdict(ok=not v, violations=v, battery_truncates_at=truncate_at,
                         flyable_legs=flyable if truncate_at is not None else len(plan.legs),
                         total_legs=len(plan.legs), est_flight_s=elapsed,
                         est_usable_s=c.battery.usable_s)
