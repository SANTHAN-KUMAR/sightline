"""F14 (score) + F21 (guardrails): the §5.8 triage score and the R10 guardrail.

    score = P(living | record) x w_class(t) x urgency_class x (1 + 0.1 * count_estimate)

Typical use::

    from sightline.triage import TriageContext, rank_records, explain_table

    ctx = TriageContext(incident_t0_utc=t0, water_temp_c=24.0, domain="sim")
    ranked = rank_records(records, ctx)          # scores, orders, fills priority_rank
    print(explain_table(ranked))                 # never the number alone (§5.8)

Dismissal is the only way a record leaves the top of the list, and it keeps the record::

    from sightline.triage import dismiss
    dismiss(rec, reason="operator confirmed it is a tarpaulin", by="IC-1")
"""

from sightline.triage.curves import (
    ANIMAL_FACTOR,
    DEFAULT_WATER_TEMP_C,
    HEAD_ONLY_URGENCY,
    IMMERSION_DEADLINE_GAIN,
    IMMERSION_TABLE,
    SITUATION_RANK,
    SITUATIONS,
    STRANDED_HALFLIFE_H,
    STRANDED_PLATEAU_H,
    SURVIVAL_FLOOR,
    TRAPPED_HALFLIFE_H,
    TRAPPED_PLATEAU_H,
    URGENCY_BASE,
    Situation,
    immersed_weight,
    immersion_band,
    stranded_weight,
    survival_weight,
    time_to_exhaustion_h,
    trapped_weight,
    urgency_weight,
)
from sightline.triage.guardrails import (
    ALLOWED_STATUSES,
    LANE_SOURCE_DIRS,
    GuardrailError,
    Violation,
    assert_no_record_deletion,
    dismiss,
    is_dismissed,
    partition_dismissed,
    retention_check,
    scan_lane_sources,
    scan_source,
    set_status,
    undismiss,
)
from sightline.triage.posture import (
    BASE_SITUATION,
    DEFAULT_POSTURE_CONF_THRESHOLD,
    POSTURE_PROMOTION,
    SUBMERSION_PROMOTION,
    PostureVerdict,
    infer_situation,
    situation_of,
)
from sightline.triage.rank import explain, explain_table, rank_records, sort_key, top_n
from sightline.triage.score import (
    MOTION_BOOST,
    THERMAL_BOOST_MAX,
    THERMAL_FADE_H,
    THERMAL_FULL_H,
    TriageContext,
    base_score,
    count_bonus,
    motion_boost,
    p_living,
    posture_demotions,
    score_record,
    score_records,
    thermal_boost,
)

__all__ = [
    "ALLOWED_STATUSES",
    "ANIMAL_FACTOR",
    "BASE_SITUATION",
    "DEFAULT_POSTURE_CONF_THRESHOLD",
    "DEFAULT_WATER_TEMP_C",
    "HEAD_ONLY_URGENCY",
    "IMMERSION_DEADLINE_GAIN",
    "IMMERSION_TABLE",
    "LANE_SOURCE_DIRS",
    "MOTION_BOOST",
    "POSTURE_PROMOTION",
    "SITUATIONS",
    "SITUATION_RANK",
    "STRANDED_HALFLIFE_H",
    "STRANDED_PLATEAU_H",
    "SUBMERSION_PROMOTION",
    "SURVIVAL_FLOOR",
    "THERMAL_BOOST_MAX",
    "THERMAL_FADE_H",
    "THERMAL_FULL_H",
    "TRAPPED_HALFLIFE_H",
    "TRAPPED_PLATEAU_H",
    "URGENCY_BASE",
    "GuardrailError",
    "PostureVerdict",
    "Situation",
    "TriageContext",
    "Violation",
    "assert_no_record_deletion",
    "base_score",
    "count_bonus",
    "dismiss",
    "explain",
    "explain_table",
    "immersed_weight",
    "immersion_band",
    "infer_situation",
    "is_dismissed",
    "motion_boost",
    "p_living",
    "partition_dismissed",
    "posture_demotions",
    "rank_records",
    "retention_check",
    "scan_lane_sources",
    "scan_source",
    "score_record",
    "score_records",
    "set_status",
    "situation_of",
    "sort_key",
    "stranded_weight",
    "survival_weight",
    "thermal_boost",
    "time_to_exhaustion_h",
    "top_n",
    "trapped_weight",
    "undismiss",
    "urgency_weight",
]
