"""Lane B4 - triage scoring (F14) and the R10 guardrail (F21).

Every number asserted here is traceable to `docs/SOLUTION_DOC.md`; the section is named in each test. Offline,
CPU only, no simulator. Run with::

    D:\\Tools\\uv\\uv.exe run pytest tests/test_triage.py -q
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from sightline.schemas import POSTURES, SUBMERSIONS, Record
from sightline.triage import (
    ALLOWED_STATUSES,
    ANIMAL_FACTOR,
    BASE_SITUATION,
    HEAD_ONLY_URGENCY,
    IMMERSION_DEADLINE_GAIN,
    IMMERSION_TABLE,
    MOTION_BOOST,
    STRANDED_HALFLIFE_H,
    STRANDED_PLATEAU_H,
    SURVIVAL_FLOOR,
    THERMAL_BOOST_MAX,
    THERMAL_FADE_H,
    THERMAL_FULL_H,
    TRAPPED_HALFLIFE_H,
    TRAPPED_PLATEAU_H,
    URGENCY_BASE,
    GuardrailError,
    TriageContext,
    base_score,
    count_bonus,
    dismiss,
    explain,
    immersed_weight,
    immersion_band,
    infer_situation,
    motion_boost,
    p_living,
    partition_dismissed,
    posture_demotions,
    rank_records,
    retention_check,
    scan_lane_sources,
    scan_source,
    score_record,
    set_status,
    situation_of,
    stranded_weight,
    survival_weight,
    thermal_boost,
    time_to_exhaustion_h,
    trapped_weight,
    undismiss,
    urgency_weight,
)

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "_artifacts" / "lanes" / "triage_export"

T0 = 1789000000.0  # arbitrary fixed incident epoch; every test pins now_utc so nothing depends on wall clock
HOUR = 3600.0

#: The times the survival curves are probed at. Deliberately spans plateau, one half-life, several, and the
#: far tail where the floor has to hold.
PROBE_HOURS = (0.0, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0, 120.0, 192.0, 360.0, 1000.0, 100_000.0)


def ctx_at(hours_after_t0: float, **kw) -> TriageContext:
    kw.setdefault("water_temp_c", 24.0)
    return TriageContext(incident_t0_utc=T0, now_utc=T0 + hours_after_t0 * HOUR, **kw)


def make_record(rid: str = "r", **kw) -> Record:
    """A plain record with the fields triage reads; everything else keeps its schema default."""
    defaults = dict(
        record_id=rid,
        status="confirmed",
        cls="human",
        lat=11.5,
        lon=76.1,
        alt_msl_m=760.0,
        h_acc_m=2.6,
        confidence=0.8,
        confidence_max_det=0.8,
        n_observations=5,
        first_seen_utc=T0 + HOUR,
        last_seen_utc=T0 + 2 * HOUR,
        count_estimate=1,
    )
    defaults.update(kw)
    return Record(**defaults)


# ==============================================================================================================
# 1. Survival-decay curves - SOLUTION_DOC §2.5 line 208-211, §5.8 line 836
# ==============================================================================================================
def test_trapped_curve_hits_the_documented_values():
    """§2.5 line 208: w(t) = 1.0 for t < 48 h, then exponential with half-life ~= 3 days, floor 0.05."""
    assert TRAPPED_PLATEAU_H == 48.0 and TRAPPED_HALFLIFE_H == 72.0
    for t in (0.0, 1.0, 24.0, 47.9, 48.0):
        assert trapped_weight(t) == 1.0, t
    assert trapped_weight(48.0 + 72.0) == pytest.approx(0.5, abs=1e-12)
    assert trapped_weight(48.0 + 144.0) == pytest.approx(0.25, abs=1e-12)
    assert trapped_weight(48.0 + 216.0) == pytest.approx(0.125, abs=1e-12)
    # the floor is reached after log2(1/0.05) = 4.3219 half-lives and never moves again
    floor_at_h = 48.0 + 72.0 * math.log2(1.0 / SURVIVAL_FLOOR)
    assert trapped_weight(floor_at_h - 1.0) > SURVIVAL_FLOOR
    assert trapped_weight(floor_at_h + 1.0) == SURVIVAL_FLOOR


def test_stranded_curve_hits_the_documented_values():
    """§2.5 line 210: w(t) = 1.0 for t < 24 h, then slow decay (half-life ~4 days)."""
    assert STRANDED_PLATEAU_H == 24.0 and STRANDED_HALFLIFE_H == 96.0
    for t in (0.0, 6.0, 23.9, 24.0):
        assert stranded_weight(t) == 1.0, t
    assert stranded_weight(24.0 + 96.0) == pytest.approx(0.5, abs=1e-12)
    assert stranded_weight(24.0 + 192.0) == pytest.approx(0.25, abs=1e-12)
    assert stranded_weight(24.0 + 288.0) == pytest.approx(0.125, abs=1e-12)


def test_immersed_curve_plateaus_until_time_to_exhaustion():
    """§2.5 line 209: w = 1.0 with time-to-exhaustion as the rank key; decay half-life = the survival span."""
    exhaustion_h, survival_h = immersion_band(24.0)
    assert (exhaustion_h, survival_h) == (3.0, 48.0)  # the 21.1-26.7 C row
    halflife = survival_h - exhaustion_h
    for t in (0.0, 1.0, 2.99, 3.0):
        assert immersed_weight(t, 24.0) == 1.0, t
    assert immersed_weight(exhaustion_h + halflife, 24.0) == pytest.approx(0.5, abs=1e-12)
    assert immersed_weight(exhaustion_h + 2 * halflife, 24.0) == pytest.approx(0.25, abs=1e-12)
    # colder water decays faster: 15.6-21.1 C is "exhaustion 2-7 h" (§2.5 line 209)
    assert immersion_band(18.0)[0] == 2.0
    assert immersed_weight(30.0, 18.0) < immersed_weight(30.0, 24.0)


def test_the_two_published_immersion_bands_are_reproduced_exactly():
    """§2.5 line 209 quotes two rows of the USCG chart; those are the two the score must reproduce."""
    assert time_to_exhaustion_h(16.0) == 2.0 and time_to_exhaustion_h(21.0) == 2.0  # "15-21 C: exhaustion 2-7 h"
    assert time_to_exhaustion_h(22.0) == 3.0 and time_to_exhaustion_h(26.0) == 3.0  # "21-27 C: exhaustion 3-12 h"


def test_immersion_table_is_monotone_in_water_temperature():
    """Colder water must never be less urgent: exhaustion and the survival span both rise with temperature."""
    exhaustions = [row[1] for row in IMMERSION_TABLE]
    halflives = [row[2] - row[1] for row in IMMERSION_TABLE]
    assert exhaustions == sorted(exhaustions), exhaustions
    assert halflives == sorted(halflives), halflives
    assert all(h > 0 for h in halflives)


def test_no_curve_ever_reaches_zero():
    """§5.8 line 836 "The floor is never zero"; §1.4 "never lowers a record's priority to zero"."""
    for t in PROBE_HOURS + (1e9,):
        for situation in ("immersed", "trapped", "stranded", "unknown"):
            w = survival_weight(situation, t, 24.0, "human")
            assert w >= SURVIVAL_FLOOR > 0.0, (situation, t, w)
            w_animal = survival_weight(situation, t, 24.0, "animal")
            assert w_animal >= SURVIVAL_FLOOR * ANIMAL_FACTOR > 0.0, (situation, t, w_animal)


def test_curves_never_increase_with_time():
    for situation in ("immersed", "trapped", "stranded"):
        previous = math.inf
        for t in PROBE_HOURS:
            w = survival_weight(situation, t, 24.0)
            assert w <= previous + 1e-15, (situation, t)
            previous = w


def test_animal_curve_is_exactly_three_tenths_of_the_human_curve():
    """§2.5 line 211: "Separate list; w = 0.3 x human curve"."""
    assert ANIMAL_FACTOR == 0.3
    for situation in ("immersed", "trapped", "stranded"):
        for t in PROBE_HOURS:
            human = survival_weight(situation, t, 24.0, "human")
            animal = survival_weight(situation, t, 24.0, "animal")
            assert animal == pytest.approx(ANIMAL_FACTOR * human, rel=1e-12)


def test_unknown_situation_falls_back_to_the_stranded_curve():
    """§5.5a rule 1: the unknown case is never the cheapest case."""
    for t in PROBE_HOURS:
        assert survival_weight("unknown", t, 24.0) == survival_weight(BASE_SITUATION, t, 24.0)
        assert survival_weight("nonsense-label", t, 24.0) == survival_weight(BASE_SITUATION, t, 24.0)


# ==============================================================================================================
# 2. Urgency ordering - SOLUTION_DOC §5.8 line 837
# ==============================================================================================================
def test_urgency_orders_immersed_above_trapped_above_stranded_always():
    """§5.8 line 837, as a structural property: true for every elapsed time and every water temperature."""
    for water_c in (0.0, 5.0, 12.0, 18.0, 24.0, 30.0):
        for t in PROBE_HOURS:
            u_immersed = urgency_weight("immersed", t, water_c)
            assert u_immersed > urgency_weight("trapped", t, water_c) > urgency_weight("stranded", t, water_c)
            assert urgency_weight("unknown", t, water_c) == URGENCY_BASE["stranded"]


def test_head_only_sits_above_every_other_immersed_record():
    """§5.8 line 837 / §2.5 line 209: "head-only detections go to the top of the list" - with no overlap."""
    worst_case_immersed = URGENCY_BASE["immersed"] * (1.0 + IMMERSION_DEADLINE_GAIN)
    assert worst_case_immersed <= HEAD_ONLY_URGENCY
    for water_c in (0.0, 12.0, 24.0, 30.0):
        for t in PROBE_HOURS:
            assert urgency_weight("immersed", t, water_c, head_only=True) >= HEAD_ONLY_URGENCY
            assert urgency_weight("immersed", t, water_c, head_only=False) <= worst_case_immersed
            assert urgency_weight("immersed", t, water_c, True) > urgency_weight("immersed", t, water_c, False)


def test_time_to_exhaustion_is_the_rank_key_inside_the_immersed_block():
    """§2.5 line 209: colder water = shorter clock = the record overtakes a warm-water one at the same age."""
    cold, warm = urgency_weight("immersed", 2.0, 12.0), urgency_weight("immersed", 2.0, 24.0)
    assert cold > warm
    # and urgency rises monotonically as a record approaches its own exhaustion time
    series = [urgency_weight("immersed", t, 24.0) for t in (0.0, 0.75, 1.5, 2.25, 3.0, 10.0)]
    assert series == sorted(series)
    assert series[0] == URGENCY_BASE["immersed"]
    assert series[-1] == URGENCY_BASE["immersed"] * (1.0 + IMMERSION_DEADLINE_GAIN)


# ==============================================================================================================
# 3. P(living | record) - SOLUTION_DOC §5.8 line 835, §2.5 line 201
# ==============================================================================================================
def test_thermal_negative_never_lowers_p_living():
    """§5.8 line 835: "left unchanged (never lowered) by a thermal-negative RGB detection after hours"."""
    for hours in (0.0, 1.0, 6.0, 12.0, 24.0, 72.0, 240.0):
        rec = make_record(thermal_hot=False, confidence=0.62)
        assert thermal_boost(rec, hours) == 1.0
        score_record(rec, ctx_at(hours))
        assert rec.components.thermal_boost == 1.0
        assert rec.components.p_living == pytest.approx(0.62, abs=1e-12)
        # and a thermal-hot twin is never worse off
        hot = make_record(thermal_hot=True, confidence=0.62)
        score_record(hot, ctx_at(hours))
        assert hot.components.p_living >= rec.components.p_living


def test_thermal_boost_is_strong_early_and_faded_by_24h():
    """§2.5 line 201: strong evidence of life in the first hours, weak evidence after ~24 h."""
    hot = make_record(thermal_hot=True)
    assert thermal_boost(hot, 0.0) == THERMAL_BOOST_MAX == 1.6
    assert thermal_boost(hot, THERMAL_FULL_H) == THERMAL_BOOST_MAX
    assert thermal_boost(hot, THERMAL_FADE_H) == 1.0
    assert thermal_boost(hot, 500.0) == 1.0
    assert thermal_boost(hot, 15.0) == pytest.approx(1.0 + 0.6 * (24.0 - 15.0) / 18.0, abs=1e-12)
    series = [thermal_boost(hot, t) for t in (0.0, 3.0, 6.0, 9.0, 15.0, 24.0, 100.0)]
    assert series == sorted(series, reverse=True)
    assert all(b >= 1.0 for b in series)


def test_motion_raises_p_living_and_stillness_does_not_lower_it():
    """§5.8 line 835: p_living is raised "by observed motion"."""
    assert motion_boost(make_record(motion_state="moving")) == MOTION_BOOST == 1.5
    assert motion_boost(make_record(motion_state="still")) == 1.0
    assert motion_boost(make_record(motion_state="unknown")) == 1.0
    moving = score_record(make_record(motion_state="moving", confidence=0.5), ctx_at(2.0))
    still = score_record(make_record(motion_state="still", confidence=0.5), ctx_at(2.0))
    unknown = score_record(make_record(motion_state="unknown", confidence=0.5), ctx_at(2.0))
    assert moving.components.p_living > still.components.p_living
    assert still.components.p_living == unknown.components.p_living == pytest.approx(0.5, abs=1e-12)


def test_p_living_is_bounded_and_never_below_the_input_confidence():
    for conf in (0.0, 0.05, 0.3, 0.5, 0.87, 0.999, 1.0):
        for boost in (1.0, 1.1, 1.5, 1.6, 2.4, 10.0):
            p = p_living(conf, boost)
            assert conf - 1e-12 <= p <= 1.0, (conf, boost, p)
    assert p_living(0.5, 1.0) == pytest.approx(0.5, abs=1e-12)
    assert p_living(0.5, 2.0) == pytest.approx(0.75, abs=1e-12)  # 1 - 0.5^2
    assert p_living(0.9, 0.1) == pytest.approx(0.9, abs=1e-12)  # a boost below 1 cannot demote


def test_count_bonus_matches_the_formula():
    """§5.8 line 832: (1 + 0.1 * count_estimate)."""
    assert count_bonus(1) == pytest.approx(1.1)
    assert count_bonus(5) == pytest.approx(1.5)
    assert count_bonus(12) == pytest.approx(2.2)
    assert count_bonus(0) == count_bonus(-4) == pytest.approx(1.1)  # a record is at least one being


def test_score_is_exactly_the_product_of_its_four_stored_terms():
    """§5.8: every term is stored on the record and the number is never shown alone."""
    rec = score_record(make_record(count_estimate=3, thermal_hot=True, motion_state="moving"), ctx_at(2.0))
    c = rec.components
    assert rec.score == pytest.approx(c.p_living * c.w_class * c.urgency * c.count_bonus, rel=1e-12)
    assert rec.score == pytest.approx(c.total(), rel=1e-12)
    for value in (c.p_living, c.w_class, c.urgency, c.count_bonus):
        assert value > 0.0
    text = explain(rec)
    for token in ("p_living", "w_class", "urgency", "count", "posture", "submersion"):
        assert token in text


# ==============================================================================================================
# 4. Record-level ordering - the reason this is a triage list (§5.8 line 837)
# ==============================================================================================================
def _trio(hours: float, **kw):
    immersed = make_record("immersed", submersion="half", submersion_conf=0.9, **kw)
    trapped = make_record("trapped", posture="trapped", posture_conf=0.9, **kw)
    stranded = make_record("stranded", posture="standing", posture_conf=0.9, **kw)
    return rank_records([stranded, trapped, immersed], ctx_at(hours))


def test_immersed_outranks_trapped_outranks_stranded_at_equal_confidence():
    """The ordering §5.8 exists for, checked across the whole operational window."""
    for hours in (0.0, 1.0, 6.0, 12.0, 24.0, 47.0):
        ranked = _trio(hours)
        assert [r.record_id for r in ranked] == ["immersed", "trapped", "stranded"], hours
        assert [r.priority_rank for r in ranked] == [0, 1, 2]
        assert ranked[0].score > ranked[1].score > ranked[2].score


def test_head_only_goes_to_the_very_top():
    """§5.8 line 837 / §2.5 line 209."""
    head = make_record("head_only", submersion="head_only", submersion_conf=0.8)
    immersed = make_record("immersed", submersion="half", submersion_conf=0.99, confidence=0.95)
    trapped = make_record("trapped", posture="trapped", posture_conf=0.99, confidence=0.95)
    ranked = rank_records([trapped, immersed, head], ctx_at(3.0))
    assert ranked[0].record_id == "head_only"
    assert ranked[0].components.urgency >= HEAD_ONLY_URGENCY


def test_the_survival_curves_cross_over_after_days_and_that_is_deliberate():
    """After ~8 days a trapped-with-air survivor is likelier alive than an immersed one (§2.5 lines 208-209).

    This is the survival curve doing its job, not a broken ordering: the *urgency* term still ranks immersed
    above trapped (asserted above), but w_class has decayed past it. Pinned so a future edit has to mean it.
    """
    ranked = _trio(200.0)
    assert [r.record_id for r in ranked] == ["trapped", "immersed", "stranded"]
    assert all(r.score > 0 for r in ranked)


# ==============================================================================================================
# 5. The §5.5a posture safety rule - the part that must not be got wrong
# ==============================================================================================================
def test_rule_1_unknown_posture_uses_the_stranded_base():
    """§5.5a rule 1: unknown is never the cheapest case."""
    verdict = infer_situation(posture="unknown", posture_conf=0.0, submersion="unknown", submersion_conf=0.0)
    assert verdict.situation == BASE_SITUATION == "stranded"
    assert verdict.promoted is False and verdict.head_only is False

    unknown = score_record(make_record("u"), ctx_at(6.0))
    stranded = score_record(make_record("s", posture="standing", posture_conf=0.95), ctx_at(6.0))
    assert unknown.score == pytest.approx(stranded.score, rel=1e-12)
    assert unknown.components.urgency == URGENCY_BASE["stranded"]


def test_rule_1_low_confidence_posture_uses_the_base():
    """§5.5a rule 1: "or the head's confidence is below a threshold"."""
    ctx = ctx_at(6.0, posture_conf_threshold=0.5)
    for conf in (0.0, 0.1, 0.49, 0.4999):
        rec = make_record("low", submersion="head_only", submersion_conf=conf, posture="half_submerged", posture_conf=conf)
        score_record(rec, ctx)
        assert rec.components.urgency_class == "stranded", conf
        assert rec.components.urgency == URGENCY_BASE["stranded"], conf
        assert rec.components.posture_promoted is False
        assert rec.score == pytest.approx(base_score(rec, ctx), rel=1e-12)
    # at the threshold the promotion takes effect
    at_threshold = make_record("ok", submersion="head_only", submersion_conf=0.5)
    score_record(at_threshold, ctx)
    assert at_threshold.components.urgency_class == "immersed"
    assert at_threshold.components.posture_promoted is True


def test_rule_2_no_posture_prediction_can_push_a_record_below_its_base():
    """§5.5a rule 2, exhaustively: every posture x submersion x confidence, against the posture-blind base."""
    ctx = ctx_at(6.0)
    checked = 0
    for posture in POSTURES:
        for submersion in SUBMERSIONS:
            for conf in (0.0, 0.3, 0.5, 0.75, 1.0):
                rec = make_record(
                    f"{posture}-{submersion}-{conf}",
                    posture=posture,
                    posture_conf=conf,
                    submersion=submersion,
                    submersion_conf=conf,
                )
                base = base_score(rec, ctx)
                score_record(rec, ctx)
                assert rec.score >= base - 1e-12, (posture, submersion, conf, rec.score, base)
                assert rec.components.urgency >= URGENCY_BASE[BASE_SITUATION] - 1e-12
                assert rec.components.w_class >= SURVIVAL_FLOOR
                checked += 1
    assert checked == len(POSTURES) * len(SUBMERSIONS) * 5 == 210


def test_rule_2_holds_against_adversarial_head_output():
    """Malformed, out-of-vocabulary and confidently-wrong predictions must all be no-ops, not demotions."""
    ctx = ctx_at(6.0)
    adversarial = [
        dict(posture="standing", posture_conf=1.0, submersion="dry", submersion_conf=1.0),  # confidently wrong
        dict(posture="unknown", posture_conf=float("nan"), submersion="unknown", submersion_conf=float("nan")),
        dict(posture="unknown", posture_conf=-5.0, submersion="unknown", submersion_conf=-5.0),
        dict(posture="unknown", posture_conf=17.0, submersion="unknown", submersion_conf=17.0),
        dict(posture="dead", posture_conf=0.99, submersion="buried", submersion_conf=0.99),  # not in the vocab
        dict(posture="", posture_conf=0.99, submersion="", submersion_conf=0.99),
        dict(posture="prone", posture_conf=0.99, submersion="wet", submersion_conf=0.99),
        dict(posture="supine", posture_conf=0.99, submersion="dry", submersion_conf=0.99),
    ]
    baseline = score_record(make_record("plain"), ctx).score
    for i, kw in enumerate(adversarial):
        rec = make_record(f"adv{i}", **kw)
        base = base_score(rec, ctx)
        score_record(rec, ctx)
        assert rec.score >= base - 1e-12, kw
        assert rec.score >= baseline - 1e-12, kw
        assert rec.components.urgency_class in ("stranded", "trapped", "immersed")
    assert posture_demotions([make_record(f"a{i}", **kw) for i, kw in enumerate(adversarial)], ctx) == []


def test_rule_2_a_promotion_really_promotes():
    """The safety rule must not be satisfied by making the head inert."""
    ctx = ctx_at(6.0)
    for kw, expected in (
        (dict(submersion="head_only", submersion_conf=0.8), "immersed"),
        (dict(submersion="half", submersion_conf=0.8), "immersed"),
        (dict(submersion="partial", submersion_conf=0.8), "immersed"),
        (dict(posture="half_submerged", posture_conf=0.8), "immersed"),
        (dict(posture="trapped", posture_conf=0.8), "trapped"),
    ):
        rec = make_record("p", **kw)
        base = base_score(rec, ctx)
        score_record(rec, ctx)
        assert rec.components.urgency_class == expected, kw
        assert rec.components.posture_promoted is True, kw
        assert rec.score > base, kw


def test_rule_3_the_prediction_and_its_confidence_stay_on_the_record():
    """§5.5a rule 3: the commander sees "half-submerged, 0.61" and can overrule it."""
    rec = make_record("r3", posture="half_submerged", posture_conf=0.61, submersion="half", submersion_conf=0.61)
    score_record(rec, ctx_at(6.0))
    assert rec.posture == "half_submerged" and rec.posture_conf == 0.61
    assert rec.submersion == "half" and rec.submersion_conf == 0.61
    assert rec.components.posture_promoted is True
    line = explain(rec)
    assert "half_submerged 0.61" in line and "posture promoted" in line
    verdict = situation_of(rec)
    assert verdict.situation == "immersed" and "immersed" in verdict.reason


def test_animals_carry_their_own_urgency_class_and_the_0_3_factor():
    """§2.5 line 211: a separate list, w = 0.3 x the human curve - counted exactly once."""
    ctx = ctx_at(6.0)
    human = score_record(make_record("h", cls="human"), ctx)
    animal = score_record(make_record("a", cls="animal"), ctx)
    assert animal.components.urgency_class == "animal"
    assert animal.components.urgency == human.components.urgency  # not double-counted
    assert animal.components.w_class == pytest.approx(ANIMAL_FACTOR * human.components.w_class, rel=1e-12)
    assert animal.score == pytest.approx(ANIMAL_FACTOR * human.score, rel=1e-12)


# ==============================================================================================================
# 6. Ranking determinism (§5.8 - the list has to repeat for a demo and a test)
# ==============================================================================================================
def _population(n: int = 40) -> list[Record]:
    rng = random.Random(20260910)
    postures = list(POSTURES)
    submersions = list(SUBMERSIONS)
    out: list[Record] = []
    for i in range(n):
        out.append(
            make_record(
                f"rec-{i:03d}",
                cls="animal" if i % 7 == 0 else "human",
                confidence=round(rng.uniform(0.3, 0.99), 3),
                posture=rng.choice(postures),
                posture_conf=round(rng.uniform(0.0, 1.0), 3),
                submersion=rng.choice(submersions),
                submersion_conf=round(rng.uniform(0.0, 1.0), 3),
                count_estimate=rng.randint(1, 6),
                thermal_hot=bool(i % 3),
                motion_state=rng.choice(["still", "moving", "unknown"]),
                n_observations=rng.randint(3, 30),
            )
        )
    return out


def test_ranking_is_deterministic_under_shuffling():
    ctx = ctx_at(9.0)
    reference = [r.record_id for r in rank_records(_population(), ctx)]
    rng = random.Random(7)
    for _ in range(50):
        pop = _population()
        rng.shuffle(pop)
        got = rank_records(pop, ctx)
        assert [r.record_id for r in got] == reference
        assert [r.priority_rank for r in got] == list(range(len(got)))


def test_exact_ties_break_on_record_id():
    ctx = ctx_at(9.0)
    twins = [make_record(rid) for rid in ("zulu", "alpha", "mike")]
    ranked = rank_records(twins, ctx)
    assert len({r.score for r in ranked}) == 1
    assert [r.record_id for r in ranked] == ["alpha", "mike", "zulu"]


def test_animals_form_their_own_block_below_the_humans():
    """§2.5 line 211: "Separate list"."""
    ctx = ctx_at(6.0)
    pop = [make_record("weak-human", confidence=0.31), make_record("strong-animal", cls="animal", confidence=0.99)]
    ranked = rank_records(pop, ctx)
    assert [r.record_id for r in ranked] == ["weak-human", "strong-animal"]
    mixed = rank_records(_population(), ctx)
    classes = [r.cls for r in mixed]
    assert classes == sorted(classes, key=lambda c: c == "animal"), "animals must be one contiguous trailing block"
    # switching the policy off keeps every record, it just interleaves them
    interleaved = rank_records(_population(), ctx, separate_animals=False)
    assert len(interleaved) == len(mixed)


def test_ranking_never_loses_a_record():
    ctx = ctx_at(6.0)
    pop = _population()
    dismiss(pop[3], "duplicate of rec-002", "IC-1", t_utc=T0 + 10 * HOUR)
    ranked = rank_records(pop, ctx)
    assert len(ranked) == len(pop)
    assert {r.record_id for r in ranked} == {r.record_id for r in pop}
    assert retention_check(pop, ranked) == []


def test_dismissed_records_sink_but_keep_their_score():
    """§5.8 R10 / §1.4: dismissal changes attention, not evidence; the priority is never zeroed."""
    ctx = ctx_at(6.0)
    top = make_record("was-top", submersion="head_only", submersion_conf=0.9, confidence=0.99)
    other = make_record("ordinary", confidence=0.4)
    ranked_before = rank_records([top, other], ctx)
    assert ranked_before[0].record_id == "was-top"
    score_before = top.score

    dismiss(top, "operator inspected the crop: it is a tarpaulin", "IC-1", t_utc=T0 + 7 * HOUR)
    ranked_after = rank_records([top, other], ctx)
    assert [r.record_id for r in ranked_after] == ["ordinary", "was-top"]
    assert top.score == pytest.approx(score_before, rel=1e-12)
    assert top.score > 0.0
    assert top.components.total() == pytest.approx(score_before, rel=1e-12)
    assert top.priority_rank == 1  # still ranked, still present
    assert "dismissed" in explain(top) and "tarpaulin" in explain(top)


def test_scoring_is_idempotent():
    ctx = ctx_at(6.0)
    rec = make_record("idem", submersion="half", submersion_conf=0.8, thermal_hot=True, motion_state="moving")
    first = score_record(rec, ctx).score
    for _ in range(5):
        assert score_record(rec, ctx).score == first


# ==============================================================================================================
# 7. Guardrail R10 (F21)
# ==============================================================================================================
def test_dismiss_requires_a_non_empty_reason_and_an_operator():
    for bad in ("", "   ", "\n\t", None):
        rec = make_record("g")
        with pytest.raises(GuardrailError, match="reason"):
            dismiss(rec, bad, "IC-1")
        assert rec.status == "confirmed" and rec.dismissed_reason == ""
    with pytest.raises(GuardrailError, match="operator"):
        dismiss(make_record("g2"), "a real reason", "")


def test_dismiss_keeps_the_record_and_records_the_audit_trail():
    rec = make_record("g3")
    version_before = rec.version
    dismiss(rec, "  duplicate of rec-002  ", "IC-1", t_utc=T0 + 5 * HOUR)
    assert rec.status == "dismissed"
    assert rec.dismissed_reason == "duplicate of rec-002"  # stripped, not blanked
    assert rec.dismissed_by == "IC-1"
    assert rec.dismissed_utc == T0 + 5 * HOUR
    assert rec.version == version_before + 1  # the outbox key changes so the dismissal is re-shipped
    assert rec.record_id and rec.lat and rec.lon  # nothing else was touched


def test_status_vocabulary_has_no_cleared_and_rejects_one():
    assert set(ALLOWED_STATUSES) == {"candidate", "confirmed", "stale", "dismissed"}
    rec = make_record("g4")
    for forbidden in ("cleared", "resolved", "done", "deleted", "closed"):
        with pytest.raises(GuardrailError):
            set_status(rec, forbidden)
    assert rec.status == "confirmed"
    set_status(rec, "stale")
    assert rec.status == "stale"
    with pytest.raises(GuardrailError, match="reason"):
        set_status(rec, "dismissed")  # routed through dismiss(), so it cannot skip its reason


def test_undismiss_reinstates_and_keeps_the_history():
    rec = make_record("g5")
    dismiss(rec, "thought it was debris", "IC-1", t_utc=T0 + 5 * HOUR)
    undismiss(rec, "IC-2")
    assert rec.status == "candidate"
    assert "thought it was debris" in rec.notes and "IC-1" in rec.notes and "IC-2" in rec.notes
    with pytest.raises(GuardrailError):
        undismiss(rec, "IC-2", status="dismissed")


def test_partition_and_retention_helpers():
    pop = _population(10)
    dismiss(pop[2], "false positive: reflective sheet", "IC-1", t_utc=T0)
    active, dismissed = partition_dismissed(pop)
    assert len(active) + len(dismissed) == len(pop) and len(dismissed) == 1
    assert retention_check(pop, active, dismissed) == []
    assert retention_check(pop, active) == [pop[2].record_id]


def test_no_delete_shaped_code_anywhere_in_this_lane():
    """The mechanical half of R10: this fails the build if a future edit adds a deletion."""
    violations = scan_lane_sources(REPO)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_the_scanner_is_not_vacuous():
    """A scanner that never fires proves nothing, so make it fire on every pattern class it claims to catch."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    probe = SCRATCH / "r10_probe_bad.py"
    probe.write_text(
        "\n".join(
            [
                "import os, shutil",
                "def purge_records(records, store, path):",
                "    records.remove(records[0])",
                "    store.pop('rec')",
                "    del records[1]",
                "    records.clear()",
                "    os.remove(path)",
                "    shutil.rmtree(path)",
                "    store.execute('DELETE FROM records WHERE id = ?')",
                "    store.execute('DROP TABLE records')",
                "    segment.status = 'cleared'",
            ]
        ),
        encoding="utf-8",
    )
    kinds = {v.kind for v in scan_source(probe)}
    for expected in (
        "container removal",
        "container pop",
        "del statement",
        "container reset",
        "filesystem removal",
        "recursive tree removal",
        "SQL row deletion",
        "SQL table drop",
        "delete-shaped function",
    ):
        assert expected in kinds, f"{expected} not detected; kinds={sorted(kinds)}"
    assert any("R10" in v.kind for v in scan_source(probe)), "the segment-is-done literal was not detected"


def test_the_scanner_ignores_forbidden_spellings_inside_strings_and_comments():
    """Otherwise the pattern table could not be written down, and prose would break the build."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    probe = SCRATCH / "r10_probe_ok.py"
    probe.write_text(
        "\n".join(
            [
                '"""We never call .pop() or .remove(), and never del a record."""',
                "# a comment mentioning os.remove( and shutil.rmtree( and del x",
                'PATTERNS = (".pop(", ".remove(", "del ", "rmtree(")',
                "def keep(records):",
                "    return list(records)",
            ]
        ),
        encoding="utf-8",
    )
    assert scan_source(probe) == []
