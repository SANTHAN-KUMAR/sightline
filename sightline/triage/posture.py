"""The §5.5a safety rule: posture may RAISE urgency, never lower it.

SOLUTION_DOC.md §5.5a (around line 659), quoted because it is the part that must not be got wrong:

    **The safety rule that makes it usable: posture may raise urgency, never lower it.** A wrong posture
    prediction must never demote a real survivor. Three constraints enforce this:

    1. If ``posture = unknown`` or the head's confidence is below a threshold, the triage score uses the
       **base** weight for a stranded survivor. The unknown case is never the cheapest case.
    2. Posture can promote a record (a head-only or half-submerged prediction moves it to the top of the list)
       but can never push it below the base rank.
    3. The predicted class is displayed on the record card with its confidence, so a commander sees
       "half-submerged, 0.61" and can overrule it, rather than seeing a rank with no reason.

How each constraint is made structural rather than remembered:

1. ``BASE_SITUATION`` is ``"stranded"`` and is the *starting value*, not a fallback branch. A posture the model
   is unsure about, or has never heard of, simply fails to appear in the promotion tables and the base stands.
2. The resolved situation is ``max(base, candidate)`` over :data:`sightline.triage.curves.SITUATION_RANK`.
   There is no code path that returns something below the base, so no input - adversarial, malformed or
   confidently wrong - can demote a record. :func:`sightline.triage.score.base_score` recomputes the
   posture-blind score so a test can assert the property numerically as well.
3. ``posture``/``posture_conf``/``submersion``/``submersion_conf`` stay on the record untouched and
   :func:`sightline.triage.rank.explain` renders them next to the score components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sightline.schemas import Record
from sightline.triage.curves import SITUATION_RANK, Situation

#: §5.5a rule 1: "the triage score uses the base weight for a stranded survivor".
BASE_SITUATION: Situation = "stranded"

#: §5.5a rule 1: "or the head's confidence is below a threshold". ADOPTED value - the document names a
#: threshold but not a number. 0.5 is the natural operating point for a softmax head over 7 posture classes:
#: below it the head is closer to a coin flip than to a prediction.
DEFAULT_POSTURE_CONF_THRESHOLD = 0.5

#: Posture -> the situation it may promote to. Absent postures (standing, sitting, prone, supine, unknown)
#: promote nothing, so they leave the base standing. Prone and supine deliberately map to nothing: §5.5a notes
#: they are hard to separate at nadir and "need not be separated, since both map to the same urgency".
POSTURE_PROMOTION: dict[str, Situation] = {
    "half_submerged": "immersed",
    "trapped": "trapped",
}

#: Submersion -> the situation it may promote to. "dry" and "wet" are not immersion ("wet" is a rain-soaked or
#: just-out-of-the-water survivor); "unknown" is deliberately absent per rule 1.
SUBMERSION_PROMOTION: dict[str, Situation] = {
    "head_only": "immersed",
    "half": "immersed",
    "partial": "immersed",
}

#: The submersion label that goes to the very top of the list (§5.8 line 837, §2.5 line 209).
HEAD_ONLY_SUBMERSION = "head_only"


@dataclass(frozen=True, slots=True)
class PostureVerdict:
    """What the posture/submersion head was allowed to change, and why - rule 3's raw material."""

    situation: Situation
    base_situation: Situation
    promoted: bool
    head_only: bool
    posture_used: bool
    submersion_used: bool
    conf_used: float
    reason: str

    def __str__(self) -> str:
        return self.reason


def infer_situation(
    posture: str = "unknown",
    posture_conf: float = 0.0,
    submersion: str = "unknown",
    submersion_conf: float = 0.0,
    threshold: float = DEFAULT_POSTURE_CONF_THRESHOLD,
) -> PostureVerdict:
    """Resolve the entrapment situation. The result is never below :data:`BASE_SITUATION`.

    The implementation is a monotone maximum over :data:`SITUATION_RANK` seeded with the base, which is what
    makes §5.5a rule 2 a property of the code rather than of the test suite.
    """
    base: Situation = BASE_SITUATION
    situation: Situation = base
    promoted = False
    head_only = False
    posture_used = False
    submersion_used = False
    conf_used = 0.0
    reasons: list[str] = []

    def _try(candidate: Situation | None, conf: float, label: str) -> bool:
        nonlocal situation, promoted, conf_used
        if candidate is None:
            return False
        if not _confident(conf, threshold):
            reasons.append(f"{label} below threshold {threshold:.2f} -> base")
            return False
        if SITUATION_RANK[candidate] > SITUATION_RANK[situation]:
            situation = candidate
            promoted = True
            conf_used = max(conf_used, float(conf))
            reasons.append(f"{label} -> {candidate}")
            return True
        # A confident prediction that maps at or below the current situation changes nothing. This is the
        # branch a confidently-wrong "standing, dry" prediction lands in, and it must stay a no-op.
        reasons.append(f"{label} does not raise {situation}")
        return True

    posture_used = _try(POSTURE_PROMOTION.get(str(posture)), posture_conf, f"posture={posture}")
    submersion_used = _try(SUBMERSION_PROMOTION.get(str(submersion)), submersion_conf, f"submersion={submersion}")

    if str(submersion) == HEAD_ONLY_SUBMERSION and _confident(submersion_conf, threshold):
        head_only = True
        reasons.append("head-only -> top of the list")

    if not reasons:
        reasons.append(f"no usable posture/submersion evidence -> base {base}")

    return PostureVerdict(
        situation=situation,
        base_situation=base,
        promoted=promoted,
        head_only=head_only,
        posture_used=posture_used,
        submersion_used=submersion_used,
        conf_used=conf_used,
        reason="; ".join(reasons),
    )


def _confident(conf: float, threshold: float) -> bool:
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return False
    return not math.isnan(c) and c >= threshold  # a NaN confidence is not a confident prediction


def situation_of(record: Record, threshold: float = DEFAULT_POSTURE_CONF_THRESHOLD) -> PostureVerdict:
    """:func:`infer_situation` applied to a :class:`~sightline.schemas.Record`."""
    return infer_situation(
        posture=record.posture,
        posture_conf=record.posture_conf,
        submersion=record.submersion,
        submersion_conf=record.submersion_conf,
        threshold=threshold,
    )
