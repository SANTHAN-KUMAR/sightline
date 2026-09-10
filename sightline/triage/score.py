"""F14 - the §5.8 priority score, with every term stored on the record.

    score = P(living | record) x w_class(t) x urgency_class x (1 + 0.1 * count_estimate)
    -- SOLUTION_DOC.md §5.8, line ~832

`P(living | record)` is the fused detection confidence, **raised** by thermal-positive evidence in the first
hours and by observed motion, and **left unchanged (never lowered)** by a thermal-negative RGB detection after
hours (§5.8 line 835, and the practical rule at §2.5 line 201:

    a thermal-positive detection is strong evidence of *life* in the first hours and weak evidence after ~24 h;
    a thermal-negative RGB detection after hours is consistent with either a live hypothermic person or a body,
    and is never demoted to `cleared`.

The four terms are written into :class:`~sightline.schemas.ScoreComponents` and never collapsed: `Record.score`
is only ever ``components.total()``, and :func:`sightline.triage.rank.explain` is what the UI prints beside it
(§5.8 line 838: "Every term is stored on the record and shown in the UI; the number is never shown alone").
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sightline.schemas import Record, ScoreComponents
from sightline.triage import curves
from sightline.triage.curves import DEFAULT_WATER_TEMP_C, clamp01
from sightline.triage.posture import DEFAULT_POSTURE_CONF_THRESHOLD, PostureVerdict, situation_of

# --- P(living | record) boosts ------------------------------------------------------------------------------
#: §2.5 line 201 + the thermal timeline table at §2.5 lines 196-198: a thermal-positive detection is strong
#: evidence of life through the first hours and weak evidence by ~24 h. ADOPTED magnitude 1.6 and the 6 h / 24 h
#: knees; the document states the shape ("strong in the first hours, weak after ~24 h") and the timeline rows
#: 0-1 h / 1-6 h / 6-24 h, not a coefficient.
THERMAL_BOOST_MAX = 1.6
THERMAL_FULL_H = 6.0  # full strength through the 0-1 h and 1-6 h rows
THERMAL_FADE_H = 24.0  # faded to nothing by the end of the 6-24 h row

#: §5.8 line 835: p_living is raised "by observed motion". Movement is direct evidence of life, so it is the
#: larger of the two boosts. ADOPTED magnitude.
MOTION_BOOST = 1.5

#: §5.8 line 832: the count term, verbatim.
COUNT_BONUS_PER_HEAD = 0.1


@dataclass(frozen=True, slots=True)
class TriageContext:
    """Everything the score needs that is not on the record.

    ``incident_t0_utc`` is the clock the survival curves run on: §2.5's ``w(t)`` is time since the *incident*,
    not time since the detection, so a record's priority keeps ageing after the drone has flown on.
    ``now_utc`` defaults to wall-clock at score time; tests and demos pass it explicitly so the numbers repeat.
    """

    incident_t0_utc: float
    now_utc: float | None = None
    water_temp_c: float = DEFAULT_WATER_TEMP_C
    posture_conf_threshold: float = DEFAULT_POSTURE_CONF_THRESHOLD
    domain: Literal["sim", "real"] = "sim"

    def resolved_now_utc(self) -> float:
        return time.time() if self.now_utc is None else float(self.now_utc)

    def elapsed_h(self) -> float:
        """Hours since the incident, clamped at zero (a record cannot be older than the disaster)."""
        return max(0.0, (self.resolved_now_utc() - float(self.incident_t0_utc)) / 3600.0)


def thermal_boost(record: Record, elapsed_h: float) -> float:
    """>= 1.0 always. A thermal-NEGATIVE observation returns exactly 1.0 - it can never lower ``p_living``."""
    if not record.thermal_hot:
        return 1.0
    t = max(0.0, float(elapsed_h))
    if t <= THERMAL_FULL_H:
        fraction = 1.0
    elif t >= THERMAL_FADE_H:
        fraction = 0.0
    else:
        fraction = (THERMAL_FADE_H - t) / (THERMAL_FADE_H - THERMAL_FULL_H)
    return 1.0 + (THERMAL_BOOST_MAX - 1.0) * fraction


def motion_boost(record: Record) -> float:
    """>= 1.0 always. "still" and "unknown" both return 1.0: absence of motion is not evidence of death."""
    return MOTION_BOOST if record.motion_state == "moving" else 1.0


def p_living(confidence: float, boost: float = 1.0) -> float:
    """Raise a fused confidence by ``boost >= 1`` without ever exceeding 1 or falling below the input.

    ``1 - (1 - p) ** boost`` is the "extra independent looks" form: boost = 1 is the identity, boost > 1 shrinks
    the residual probability that the detection is not a living being. It is monotone in both arguments, which
    is what makes "never lowered" checkable rather than hoped for.
    """
    p = clamp01(confidence)
    b = max(1.0, float(boost))
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** b


def count_bonus(count_estimate: int) -> float:
    """``1 + 0.1 * count_estimate`` (§5.8 line 832). Negative counts are treated as one being."""
    n = max(1, int(count_estimate))
    return 1.0 + COUNT_BONUS_PER_HEAD * n


def urgency_class_label(record: Record, verdict: PostureVerdict) -> str:
    """The label stored on the record. Animals get their own list (§2.5 line 211), humans get the situation."""
    return "animal" if record.cls == "animal" else verdict.situation


def score_record(record: Record, ctx: TriageContext) -> Record:
    """Fill ``record.components`` and ``record.score`` in place and return the record.

    Idempotent: scoring the same record twice with the same context gives the same numbers. Dismissed records
    are scored exactly like any other - R10 keeps them in the list, so they keep their components too.
    """
    elapsed_h = ctx.elapsed_h()
    verdict = situation_of(record, ctx.posture_conf_threshold)

    t_boost = thermal_boost(record, elapsed_h)
    m_boost = motion_boost(record)
    p = p_living(record.confidence, t_boost * m_boost)
    w = curves.survival_weight(verdict.situation, elapsed_h, ctx.water_temp_c, record.cls)
    u = curves.urgency_weight(verdict.situation, elapsed_h, ctx.water_temp_c, verdict.head_only)
    cb = count_bonus(record.count_estimate)

    record.components = ScoreComponents(
        p_living=p,
        w_class=w,
        urgency=u,
        count_bonus=cb,
        urgency_class=urgency_class_label(record, verdict),  # type: ignore[arg-type]
        elapsed_h=elapsed_h,
        thermal_boost=t_boost,
        motion_boost=m_boost,
        posture_promoted=verdict.promoted,
    )
    record.score = record.components.total()
    return record


def score_records(records: Iterable[Record], ctx: TriageContext) -> list[Record]:
    """Score every record in place; returns the same objects as a list, in the order given."""
    return [score_record(r, ctx) for r in records]


def base_score(record: Record, ctx: TriageContext) -> float:
    """The score this record would get with the posture/submersion head switched off entirely.

    §5.5a rule 2 says a posture prediction "can never push it below the base rank". This function computes that
    base numerically, so the property can be asserted against every possible head output instead of being
    argued about. It does not touch ``record``.
    """
    blind = dataclasses.replace(record, posture="unknown", posture_conf=0.0, submersion="unknown", submersion_conf=0.0)
    return score_record(blind, ctx).score


def posture_demotions(records: Iterable[Record], ctx: TriageContext) -> list[tuple[str, float, float]]:
    """Any record whose posture-informed score fell below its posture-blind base. Must always be empty.

    Returns ``(record_id, scored, base)`` triples. Exposed rather than kept in the test file so the API server
    and the evaluation harness can assert the same safety property on live data.
    """
    out: list[tuple[str, float, float]] = []
    for r in records:
        scored = score_record(r, ctx).score
        base = base_score(r, ctx)
        if scored < base - 1e-12:
            out.append((r.record_id, scored, base))
    return out
