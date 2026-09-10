"""Survival-decay curves and urgency weights for the §5.8 priority score.

Every constant in this module is traceable to a line of `docs/SOLUTION_DOC.md`. Where the document states an
ordering but not a number, the number is marked ADOPTED and the reasoning is written out, so a reviewer can
disagree with the value without having to reverse-engineer the intent.

Source lines (SOLUTION_DOC.md):

* §5.8 line 836 - ``w_class(t)`` is the survival-decay curve for the inferred entrapment class: immersed
  (rank key = time-to-exhaustion at the water temperature), trapped-in-structure (1.0 for 48 h, then half-life
  ~= 3 days, floor 0.05), stranded (1.0 for 24 h, then half-life ~= 4 days), animal (0.3 x human curve).
  **The floor is never zero.**
* §5.8 line 837 - ``urgency_class`` orders immersed > trapped > stranded, with head-only detections at the top.
* §2.5 line 209 - Immersed in water: 21-27 C water: exhaustion 3-12 h, survival 3 h-indefinite; 15-21 C:
  exhaustion 2-7 h.  Highest urgency: w = 1.0 with time-to-exhaustion as the rank key.
* §2.5 line 208 - Trapped with air: w(t) = 1.0 for t < 48 h, then exponential with half-life ~= 3 days,
  floor 0.05 (never zero).
* §2.5 line 210 - Stranded: w(t) = 1.0 for t < 24 h, then slow decay (half-life ~4 days).
* §2.5 line 211 - Animals: separate list; w = 0.3 x human curve.
* §1.4 - the system never lowers a record's priority to zero.
* §2.5 line 1217 (sources table) - "cold-water survival charts (USCG)" is the cited origin of the immersion
  numbers, and the two bands quoted in §2.5 are rows 5 and 6 of that chart, so the whole chart is used.
"""

from __future__ import annotations

import math
from typing import Literal

#: The three entrapment situations the score actually branches on. ``Record.cls`` carries human vs animal
#: separately, and ``UrgencyClass`` in the frozen schema adds "animal"/"unknown" for display.
Situation = Literal["immersed", "trapped", "stranded"]

SITUATIONS: tuple[Situation, ...] = ("immersed", "trapped", "stranded")

#: Ordering used to take the maximum of the base and posture-informed situations (§5.5a rule 2).
SITUATION_RANK: dict[str, int] = {"stranded": 0, "trapped": 1, "immersed": 2}

# --- survival decay ---------------------------------------------------------------------------------------
#: §2.5 line 208 gives floor 0.05 for the trapped curve; §5.8 line 836 says "The floor is never zero" for all
#: of them. ADOPTED: the one published floor is used for every human curve, because the document supplies no
#: second number and a non-zero floor is mandatory (§1.4).
SURVIVAL_FLOOR = 0.05

#: §2.5 line 211: "w = 0.3 x human curve". Applied to the whole curve, floor included, so the animal floor is
#: 0.015 - small, but never zero.
ANIMAL_FACTOR = 0.3

TRAPPED_PLATEAU_H = 48.0  # §2.5 line 208: "1.0 for t < 48 h"
TRAPPED_HALFLIFE_H = 72.0  # §2.5 line 208: "half-life ~= 3 days"
STRANDED_PLATEAU_H = 24.0  # §2.5 line 210: "1.0 for t < 24 h"
STRANDED_HALFLIFE_H = 96.0  # §2.5 line 210: "half-life ~4 days"

#: USCG cold-water chart, ``(upper_water_c, exhaustion_h, expected_survival_h)`` ascending in temperature.
#: Rows 5 and 6 are quoted verbatim in §2.5 line 209 ("15-21 C: exhaustion 2-7 h" and "21-27 C: exhaustion
#: 3-12 h, survival 3 h-indefinite"), which is what identifies the chart; the remaining rows come from the same
#: chart. ``exhaustion_h`` is the LOWER bound of the published band - the conservative planning number, because
#: a triage list must assume the survivor is at the fast end of the range. ``expected_survival_h`` is the upper
#: bound of the published survival band.
#:
#: ADOPTED, and the only two invented numbers here: the chart says "indefinite" twice, and a rank key has to be
#: finite. The 21.1-26.7 C row takes 48 h of expected survival, and the > 26.7 C row takes 12 h to exhaustion
#: and 60 h of survival. Both were chosen so the table stays monotone in temperature (asserted in the tests).
IMMERSION_TABLE: tuple[tuple[float, float, float], ...] = (
    (0.3, 0.25, 0.75),  # <= 0.3 C: exhaustion < 15 min, survival 15-45 min
    (4.4, 0.25, 1.5),  # 0.3-4.4 C: 15-30 min, 30-90 min
    (10.0, 0.5, 3.0),  # 4.4-10 C: 30-60 min, 1-3 h
    (15.6, 1.0, 6.0),  # 10-15.6 C: 1-2 h, 1-6 h
    (21.1, 2.0, 40.0),  # 15.6-21.1 C: 2-7 h  <- quoted in §2.5 line 209
    (26.7, 3.0, 48.0),  # 21.1-26.7 C: 3-12 h, survival 3 h-"indefinite"  <- quoted in §2.5 line 209
    (math.inf, 12.0, 60.0),  # > 26.7 C: "indefinite"
)

#: Kerala monsoon flood water, mid-band of the 21-27 C row §2.5 quotes first (§2.3 row 19 gives "Water 15-30 C").
DEFAULT_WATER_TEMP_C = 24.0

# --- urgency ----------------------------------------------------------------------------------------------
#: §5.8 line 837 fixes the ORDER (immersed > trapped > stranded, head-only at the top) but not the numbers.
#: ADOPTED: a geometric ladder, chosen so the bands cannot overlap once the immersion deadline gain below is
#: applied - the ordering is then a structural property of the constants, not an accident of the inputs.
URGENCY_BASE: dict[str, float] = {"stranded": 1.0, "trapped": 2.0, "immersed": 4.0}

#: §5.8 line 837: "head-only detections go to the top of the list", and §2.5 line 209 repeats it. Head-only is
#: an immersed sub-case, so it is the immersed band lifted clear of it.
HEAD_ONLY_URGENCY = 8.0

#: §2.5 line 209: "w = 1.0 with time-to-exhaustion as the rank key". Survival probability stays on its plateau
#: while the person is still above water, but the DEADLINE gets closer, and that is what has to order the
#: immersed block. ADOPTED gain 0.5: an immersed record climbs from 4.0 at t = 0 to 6.0 once it has reached the
#: time-to-exhaustion for its water temperature. Colder water shortens that clock, so a cold-water record
#: overtakes a warm-water one - which is exactly "time-to-exhaustion as the rank key".
IMMERSION_DEADLINE_GAIN = 0.5

# The band separation the ordering guarantee rests on; violating it silently would break triage, so it is
# checked at import time as well as in the tests.
assert URGENCY_BASE["stranded"] < URGENCY_BASE["trapped"] < URGENCY_BASE["immersed"] < HEAD_ONLY_URGENCY
assert URGENCY_BASE["immersed"] * (1.0 + IMMERSION_DEADLINE_GAIN) <= HEAD_ONLY_URGENCY


def clamp01(x: float) -> float:
    """Clamp to [0, 1] and coerce to float (NaN falls through to 0.0)."""
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0.0 else (min(v, 1.0))


def _plateau_then_halflife(elapsed_h: float, plateau_h: float, halflife_h: float, floor: float) -> float:
    """``1.0`` until ``plateau_h``, then exponential decay by half-lives, never below ``floor``."""
    t = max(0.0, float(elapsed_h))
    if t <= plateau_h:
        return 1.0
    w = 0.5 ** ((t - plateau_h) / halflife_h)
    return max(w, floor)


def immersion_band(water_temp_c: float) -> tuple[float, float]:
    """``(time_to_exhaustion_h, expected_survival_h)`` for this water temperature (IMMERSION_TABLE)."""
    t = float(water_temp_c)
    for upper_c, exhaustion_h, survival_h in IMMERSION_TABLE:
        if t <= upper_c:
            return exhaustion_h, survival_h
    last = IMMERSION_TABLE[-1]
    return last[1], last[2]


def time_to_exhaustion_h(water_temp_c: float) -> float:
    """§2.5 line 209's rank key: how long until an immersed survivor is exhausted at this water temperature."""
    return immersion_band(water_temp_c)[0]


def immersed_weight(elapsed_h: float, water_temp_c: float = DEFAULT_WATER_TEMP_C) -> float:
    """w = 1.0 while the survivor is inside the published exhaustion window, then decay to the floor.

    §2.5 line 209 states the plateau ("w = 1.0") and the exhaustion clock but no decay constant, because the
    class it describes is the one whose *urgency* is meant to carry the ordering. The decay after exhaustion is
    ADOPTED with a half-life equal to the published survival span (expected survival minus time to exhaustion),
    which is the only time constant the chart supplies. It never reaches zero (§1.4, §5.8 line 836).
    """
    exhaustion_h, survival_h = immersion_band(water_temp_c)
    halflife_h = max(survival_h - exhaustion_h, 1e-3)
    return _plateau_then_halflife(elapsed_h, exhaustion_h, halflife_h, SURVIVAL_FLOOR)


def trapped_weight(elapsed_h: float) -> float:
    """§2.5 line 208 verbatim: 1.0 for 48 h, then half-life 3 days, floor 0.05."""
    return _plateau_then_halflife(elapsed_h, TRAPPED_PLATEAU_H, TRAPPED_HALFLIFE_H, SURVIVAL_FLOOR)


def stranded_weight(elapsed_h: float) -> float:
    """§2.5 line 210 verbatim: 1.0 for 24 h, then half-life 4 days. Floor per SURVIVAL_FLOOR (never zero)."""
    return _plateau_then_halflife(elapsed_h, STRANDED_PLATEAU_H, STRANDED_HALFLIFE_H, SURVIVAL_FLOOR)


def survival_weight(
    situation: str,
    elapsed_h: float,
    water_temp_c: float = DEFAULT_WATER_TEMP_C,
    cls: str = "human",
) -> float:
    """``w_class(t)`` of §5.8. ``cls == "animal"`` applies the 0.3 x factor of §2.5 line 211."""
    if situation == "immersed":
        w = immersed_weight(elapsed_h, water_temp_c)
    elif situation == "trapped":
        w = trapped_weight(elapsed_h)
    elif situation == "stranded":
        w = stranded_weight(elapsed_h)
    else:  # "unknown" or anything unrecognised falls back to the base situation (§5.5a rule 1)
        w = stranded_weight(elapsed_h)
    if cls == "animal":
        w *= ANIMAL_FACTOR
    return w


def urgency_weight(
    situation: str,
    elapsed_h: float = 0.0,
    water_temp_c: float = DEFAULT_WATER_TEMP_C,
    head_only: bool = False,
) -> float:
    """``urgency_class`` of §5.8: immersed > trapped > stranded, head-only above every other immersed record."""
    if situation == "immersed":
        base = HEAD_ONLY_URGENCY if head_only else URGENCY_BASE["immersed"]
        exhaustion_h = time_to_exhaustion_h(water_temp_c)
        fraction = clamp01(max(0.0, float(elapsed_h)) / exhaustion_h) if exhaustion_h > 0 else 1.0
        return base * (1.0 + IMMERSION_DEADLINE_GAIN * fraction)
    return URGENCY_BASE.get(situation, URGENCY_BASE["stranded"])
