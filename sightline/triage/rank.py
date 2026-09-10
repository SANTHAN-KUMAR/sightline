"""F14 - turning scored records into *the ranked list* (SOLUTION_DOC §5.8, §1.3 "Product 1").

The ordering, not the score, is what makes this a triage list rather than a sorted confidence dump, so the sort
key is written out explicitly and every tie is broken by something that cannot vary between runs.

Two ordering policies come from the document rather than from taste:

* §2.5 line 211 - animals are a **separate list**. They keep their real score (already multiplied by 0.3) and
  are ranked into their own contiguous block after the humans, so a high-confidence goat can never sit above a
  low-confidence person on the commander's screen.
* §5.8 "Guardrail implementation (R10)" - dismissed records **stay** in the log and in the exported bundle.
  They are sorted to the end of the display list, with their score and components untouched: dismissal is an
  operator decision about attention, not a change to the evidence (§1.4: "never lowers a record's priority to
  zero").
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sightline.schemas import Record
from sightline.triage.score import TriageContext, score_records


def sort_key(record: Record, *, separate_animals: bool = True, sink_dismissed: bool = True) -> tuple:
    """The total order the list is built on. Deterministic: the last component is a unique string."""
    return (
        1 if (sink_dismissed and record.status == "dismissed") else 0,
        1 if (separate_animals and record.cls == "animal") else 0,
        -float(record.score),
        -float(record.confidence),
        -int(record.n_observations),
        str(record.record_id),
    )


def rank_records(
    records: Iterable[Record],
    ctx: TriageContext | None = None,
    *,
    separate_animals: bool = True,
    sink_dismissed: bool = True,
) -> list[Record]:
    """Return the records ordered by priority, with ``priority_rank`` filled in (0 = highest priority).

    Pass ``ctx`` to score them first; pass ``None`` to rank scores that were computed earlier. The returned
    list holds the *same* Record objects (mutated with their rank) - nothing is copied and nothing is removed,
    so ``len(rank_records(xs)) == len(xs)`` for every input, dismissed records included (R10).
    """
    items: list[Record] = list(records)
    if ctx is not None:
        score_records(items, ctx)
    ordered = sorted(items, key=lambda r: sort_key(r, separate_animals=separate_animals, sink_dismissed=sink_dismissed))
    for i, rec in enumerate(ordered):
        rec.priority_rank = i
    return ordered


def top_n(records: Sequence[Record], n: int) -> list[Record]:
    """The first ``n`` of an already-ranked list. A *view* for display only - the rest of the list still exists.

    R10: this never removes anything from any store. It is the caller's responsibility to keep the full list,
    which is why this returns a new list and does not touch ``records``.
    """
    return list(records[: max(0, int(n))])


def explain(record: Record) -> str:
    """One line a commander can read: the score with all four of its terms, never the number alone (§5.8).

    §5.5a rule 3 is satisfied here - the predicted posture and submersion are printed with their confidences,
    so "half_submerged 0.61" is visible and can be overruled.
    """
    c = record.components
    rank = "unranked" if record.priority_rank < 0 else f"#{record.priority_rank + 1}"
    parts = [
        f"{rank} {record.cls} score {record.score:.4f}",
        f"= p_living {c.p_living:.3f}",
        f"x w_class {c.w_class:.3f}",
        f"x urgency {c.urgency:.2f} ({c.urgency_class})",
        f"x count {c.count_bonus:.2f} (n={record.count_estimate})",
        f"| t+{c.elapsed_h:.1f} h",
        f"| thermal x{c.thermal_boost:.2f}{' (hot)' if record.thermal_hot else ''}",
        f"motion x{c.motion_boost:.2f} ({record.motion_state})",
        f"| posture {record.posture} {record.posture_conf:.2f}",
        f"submersion {record.submersion} {record.submersion_conf:.2f}",
    ]
    if c.posture_promoted:
        parts.append("[posture promoted]")
    if record.status == "dismissed":
        parts.append(f"[dismissed by {record.dismissed_by or '?'}: {record.dismissed_reason or '?'} - record retained]")
    return " ".join(parts)


def explain_table(records: Sequence[Record]) -> str:
    """The whole ranked list, one :func:`explain` line each. Used by the demo and the lane report."""
    return "\n".join(explain(r) for r in records)
