"""The slice grid and the guard that makes dishonest reporting structurally hard (§5.12, hard rule 5).

Two rules are enforced here, in code rather than in prose:

1. **Every number that leaves this package is a `MetricRow`**, and a `MetricRow` cannot exist without a
   `SliceKey`, and a `SliceKey` cannot exist without a `domain`. `metric_row()` is the only constructor the rest
   of the package uses and it validates the domain and every bin label against a closed vocabulary.
2. **A `sim` row and a `real` row may never be averaged.** Every aggregation path (`MetricSet.aggregate`,
   `combine_rows`, `ratio_row`) calls `require_single_domain()` first and raises `DomainMixError` otherwise.
   Concatenating rows into one table is allowed (a report table has both domains); *collapsing* them is not.

Bin vocabularies come from `SliceKey`'s own docstring in `schemas.py` plus §5.12/§6.2. `"<30"` is an addition:
`SliceKey` lists four altitude bands starting at 30 m and the scenario flies as low as 25 m in confirmation
passes, so a below-30 sample gets its own bin instead of being silently folded into "30-45".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Iterator, Literal, Sequence

from sightline.schemas import OCCLUSION_BINS, POSTURES, MetricRow, SliceKey

# --- closed vocabularies -----------------------------------------------------------------------------------
DOMAINS: tuple[str, ...] = ("sim", "real")
ALTITUDE_BANDS: tuple[str, ...] = ("<30", "30-45", "45-60", "60-90", "90+")
PIXEL_SIZE_BINS: tuple[str, ...] = ("<20", "20-40", "40-80", "80+")
TIMES_OF_DAY: tuple[str, ...] = ("dawn", "day", "dusk", "night")
OCCLUSION_LABELS: tuple[str, ...] = tuple(str(b) for b in OCCLUSION_BINS)
MODALITIES: tuple[str, ...] = ("rgb", "thermal", "fused")
ZONES: tuple[str, ...] = ("fan", "settlement", "channel", "hillslope", "unknown")

#: Axis name -> the labels it may take, "all" excluded. Used to validate and to enumerate report tables.
AXIS_VALUES: dict[str, tuple[str, ...]] = {
    "zone": ZONES,
    "altitude_band": ALTITUDE_BANDS,
    "time_of_day": TIMES_OF_DAY,
    "occlusion": OCCLUSION_LABELS,
    "posture": POSTURES,
    "pixel_size": PIXEL_SIZE_BINS,
    "modality": MODALITIES,
}

#: The value each axis has when it is NOT pinned. `SliceKey` defaults `zone` to "unknown" and everything else
#: to "all"; `SliceKey.label()` hides both, so `is_pinned()` uses exactly this table rather than guessing.
#: Consequence, stated once: a frame whose zone is genuinely unknown cannot be given its own zone row, so
#: `slice_rows()` counts it as unlabelled on that axis instead of pretending it is the overall row.
AXIS_DEFAULT: dict[str, str] = {ax: ("unknown" if ax == "zone" else "all") for ax in AXIS_VALUES}

#: Axes that are a property of the FRAME. Precision and FP/min are only defined on these, because a false
#: positive has no ground-truth box and therefore no occlusion, posture or pixel-size bin (§5.12).
FRAME_AXES: tuple[str, ...] = ("zone", "altitude_band", "time_of_day", "modality")
#: Axes that are a property of the ground-truth BOX. Only recall-style metrics are defined on these.
BOX_AXES: tuple[str, ...] = ("occlusion", "posture", "pixel_size")


class DomainMixError(ValueError):
    """Raised when a `sim` number and a `real` number would be collapsed into one (hard rule 5, §5.5c).

    "Never average a simulation number with a real number, and never present a simulation number without the
    word." The two domains may sit side by side in a table; they may not be pooled.
    """


class SliceError(ValueError):
    """A slice label outside its closed vocabulary — a typo that would silently create a bogus bin."""


# --- binning -----------------------------------------------------------------------------------------------
def altitude_band(agl_m: float) -> str:
    """§5.12 slice axis. Bands are half-open [lo, hi): 45.0 m is "45-60", not "30-45"."""
    if not math.isfinite(agl_m):
        raise SliceError(f"agl_m must be finite, got {agl_m!r}")
    if agl_m < 30.0:
        return "<30"
    if agl_m < 45.0:
        return "30-45"
    if agl_m < 60.0:
        return "45-60"
    if agl_m < 90.0:
        return "60-90"
    return "90+"


def pixel_size_bin(size_px: float) -> str:
    """Binned on the LONGEST side (`Detection.size_px`). The 20 px edge is the §2.5 operating floor."""
    if not math.isfinite(size_px):
        raise SliceError(f"size_px must be finite, got {size_px!r}")
    if size_px < 20.0:
        return "<20"
    if size_px < 40.0:
        return "20-40"
    if size_px < 80.0:
        return "40-80"
    return "80+"


def time_of_day_bin(value: str | float) -> str:
    """Accepts a bin name, an ISO-8601 scene time (`Telemetry.time_of_day`), "HH:MM", or an hour float.

    dawn 05:00-07:59, day 08:00-16:59, dusk 17:00-19:59, night otherwise. The crossover window (§2.3) sits
    inside dawn and dusk on purpose: it is sliced by `modality`, not by time.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in TIMES_OF_DAY:
            return v
        if not v:
            raise SliceError("time_of_day is empty; pass a bin name, an ISO time or an hour")
        digits = v.replace("t", " ").split()
        clock = digits[-1] if len(digits) > 1 else v
        try:
            hour = int(clock.split(":")[0][-2:])
        except (ValueError, IndexError) as exc:  # pragma: no cover - defensive
            raise SliceError(f"cannot read a time of day from {value!r}") from exc
    else:
        hour = int(value)
    hour %= 24
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 17:
        return "day"
    if 17 <= hour < 20:
        return "dusk"
    return "night"


def occlusion_label(occlusion: int | None) -> str:
    """`OCCLUSION_BINS` 0/1/2 -> "0"/"1"/"2"; None -> "all" (unknown occlusion is not a bin, it is no bin)."""
    if occlusion is None:
        return "all"
    if occlusion not in OCCLUSION_BINS:
        raise SliceError(f"occlusion must be one of {OCCLUSION_BINS}, got {occlusion!r}")
    return str(occlusion)


def posture_label(posture: str) -> str:
    if posture not in POSTURES:
        raise SliceError(f"posture must be one of {POSTURES}, got {posture!r}")
    return posture


# --- slice construction ------------------------------------------------------------------------------------
def make_slice(domain: str, **axes: str) -> SliceKey:
    """The ONLY way this package builds a `SliceKey`. Validates the domain and every axis label.

    `domain` has no default on purpose: forgetting it is a `TypeError`, not a silently mislabelled number.
    """
    if domain not in DOMAINS:
        raise SliceError(f"domain must be one of {DOMAINS} (hard rule 5), got {domain!r}")
    for name, value in axes.items():
        if name not in AXIS_VALUES:
            raise SliceError(f"unknown slice axis {name!r}; known axes are {sorted(AXIS_VALUES)}")
        if value != "all" and value not in AXIS_VALUES[name]:
            raise SliceError(f"{name}={value!r} is not in {AXIS_VALUES[name]}")
    return SliceKey(domain=domain, **axes)  # type: ignore[arg-type]


def narrow(key: SliceKey, **axes: str) -> SliceKey:
    """Return a copy of `key` with some axes pinned. The domain can never be changed by narrowing."""
    if "domain" in axes:
        raise SliceError("narrow() cannot change the domain of a slice")
    return make_slice(key.domain, **{**{k: v for k, v in asdict(key).items() if k != "domain"}, **axes})


def slice_axis_value(key: SliceKey, axis: str) -> str:
    return str(getattr(key, axis))


def is_pinned(key: SliceKey, axis: str) -> bool:
    """True when this slice actually restricts `axis` (i.e. the value is not that axis's unpinned default)."""
    return slice_axis_value(key, axis) != AXIS_DEFAULT[axis]


# --- the row constructor -----------------------------------------------------------------------------------
def metric_row(name: str, value: float, key: SliceKey, n: int = 0, **detail: Any) -> MetricRow:
    """Build a `MetricRow`. Every public function in `sightline.eval` returns these, never bare floats."""
    if key.domain not in DOMAINS:
        raise SliceError(f"MetricRow slice has no valid domain: {key!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(
            f"metric {name!r} is {v}; report an explicit n=0 row with a note instead of a NaN or an infinity"
        )
    return MetricRow(name=name, value=v, slice=key, n=int(n), detail=dict(detail))


def require_single_domain(rows: Sequence[MetricRow]) -> str:
    """Guard in front of every aggregation. Returns the one domain, or raises `DomainMixError`."""
    domains = {r.slice.domain for r in rows}
    if not domains:
        raise DomainMixError("cannot aggregate an empty set of rows: there is no domain to report")
    if len(domains) > 1:
        names = sorted({r.name for r in rows})
        raise DomainMixError(
            f"refusing to combine domains {sorted(domains)} for {names}: a simulation number may never be "
            "averaged with a real number (SOLUTION_DOC 5.5c, HANDBOOK hard rule 5). Report them side by side."
        )
    return domains.pop()


def combine_rows(rows: Sequence[MetricRow], name: str, how: str = "weighted_mean") -> MetricRow:
    """Collapse rows into one. Raises `DomainMixError` if they do not all share a domain.

    `weighted_mean` weights by `n` (the sample-size-correct pooling for recall-like ratios); `mean` is the
    unweighted macro average; `sum` adds counts.
    """
    if not rows:
        raise DomainMixError("cannot combine zero rows")
    domain = require_single_domain(rows)
    keys = [r.slice for r in rows]
    shared = {}
    for axis in AXIS_VALUES:
        values = {slice_axis_value(k, axis) for k in keys}
        shared[axis] = values.pop() if len(values) == 1 else "all"
    key = make_slice(domain, **shared)
    n = sum(r.n for r in rows)
    if how == "sum":
        value = sum(r.value for r in rows)
    elif how == "mean":
        value = sum(r.value for r in rows) / len(rows)
    elif how == "weighted_mean":
        if n <= 0:
            raise ValueError(f"weighted_mean of {name!r} needs n > 0 on at least one row")
        value = sum(r.value * r.n for r in rows) / n
    else:
        raise ValueError(f"unknown aggregation {how!r}")
    return metric_row(name, value, key, n, combined_from=len(rows), how=how)


def ratio_row(name: str, numerator: MetricRow, denominator: MetricRow, **detail: Any) -> MetricRow:
    """A ratio of two measured rows (e.g. raw FP/min divided by record FP/min: "what tracking bought").

    Both rows must share a domain — dividing a sim number by a real one is the same category error as averaging.
    """
    require_single_domain([numerator, denominator])
    if denominator.value == 0:
        raise ZeroDivisionError(f"{name}: denominator {denominator.name} is zero (n={denominator.n})")
    key = numerator.slice if numerator.slice == denominator.slice else make_slice(numerator.slice.domain)
    return metric_row(
        name,
        numerator.value / denominator.value,
        key,
        n=max(numerator.n, denominator.n),
        numerator=numerator.name,
        denominator=denominator.name,
        **detail,
    )


# --- the row container -------------------------------------------------------------------------------------
@dataclass(slots=True)
class MetricSet:
    """An ordered bag of `MetricRow`. Holds both domains; refuses to collapse across them.

    Concatenation (`+`, `extend`) is safe and is how a report table is built. `aggregate()` is the only path
    that produces a single number, and it goes through `require_single_domain()`.
    """

    rows: list[MetricRow] = field(default_factory=list)

    def __iter__(self) -> Iterator[MetricRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __add__(self, other: "MetricSet") -> "MetricSet":
        return MetricSet(list(self.rows) + list(other.rows))

    def add(self, row: MetricRow) -> "MetricSet":
        if not isinstance(row, MetricRow):
            raise TypeError(f"MetricSet holds MetricRow, not {type(row).__name__} — build it with metric_row()")
        if row.slice.domain not in DOMAINS:
            raise SliceError(f"row {row.name!r} has no valid domain")
        self.rows.append(row)
        return self

    def extend(self, rows: Iterable[MetricRow]) -> "MetricSet":
        for row in rows:
            self.add(row)
        return self

    def names(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            seen.setdefault(r.name, None)
        return list(seen)

    def domains(self) -> list[str]:
        return sorted({r.slice.domain for r in self.rows})

    def filter(self, name: str | None = None, domain: str | None = None, **axes: str) -> "MetricSet":
        out = []
        for r in self.rows:
            if name is not None and r.name != name:
                continue
            if domain is not None and r.slice.domain != domain:
                continue
            if any(slice_axis_value(r.slice, ax) != val for ax, val in axes.items()):
                continue
            out.append(r)
        return MetricSet(out)

    def by_domain(self) -> dict[str, "MetricSet"]:
        """Split into one `MetricSet` per domain. This is how a report renders two tables, never one."""
        out: dict[str, MetricSet] = {}
        for r in self.rows:
            out.setdefault(r.slice.domain, MetricSet()).add(r)
        return out

    def get(self, name: str, domain: str, **axes: str) -> MetricRow:
        """Exactly one row, or an error. Used by callers that need a specific measured number."""
        hits = self.filter(name=name, domain=domain, **axes).rows
        if len(hits) != 1:
            raise KeyError(f"expected exactly 1 row for {name!r} domain={domain} {axes}, found {len(hits)}")
        return hits[0]

    def aggregate(self, name: str, how: str = "weighted_mean", domain: str | None = None, **axes: str) -> MetricRow:
        """Collapse the matching rows into one. Raises `DomainMixError` if they span both domains."""
        hits = self.filter(name=name, domain=domain, **axes).rows
        if not hits:
            raise KeyError(f"no rows named {name!r} to aggregate")
        return combine_rows(hits, name, how=how)

    def overall(self, domain: str | None = None) -> "MetricSet":
        """Only the rows whose slice pins no axis — the headline numbers."""
        out = [r for r in self.rows if not any(is_pinned(r.slice, ax) for ax in AXIS_VALUES)]
        if domain is not None:
            out = [r for r in out if r.slice.domain == domain]
        return MetricSet(out)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {"name": r.name, "value": r.value, "n": r.n, "domain": r.slice.domain,
             **{ax: slice_axis_value(r.slice, ax) for ax in AXIS_VALUES}, "detail": r.detail}
            for r in self.rows
        ]

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict[str, Any]]) -> "MetricSet":
        out = cls()
        for d in dicts:
            key = make_slice(d["domain"], **{ax: d.get(ax, "all") for ax in AXIS_VALUES})
            out.add(metric_row(d["name"], d["value"], key, int(d.get("n", 0)), **dict(d.get("detail") or {})))
        return out

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.rows)


# --- partitioning ------------------------------------------------------------------------------------------
def partition(items: Sequence[Any], key_fn) -> dict[str, list[Any]]:
    """Group items by a label function. Every item lands in exactly one group — the property the slice grid
    depends on and `tests/test_eval.py` asserts (no double counting, no dropped items)."""
    out: dict[str, list[Any]] = {}
    for item in items:
        out.setdefault(str(key_fn(item)), []).append(item)
    total = sum(len(v) for v in out.values())
    if total != len(items):
        raise AssertionError(f"partition lost or duplicated items: {total} != {len(items)}")
    return out


Domain = Literal["sim", "real"]

__all__ = [
    "AXIS_DEFAULT", "AXIS_VALUES", "ALTITUDE_BANDS", "BOX_AXES", "DOMAINS", "Domain", "DomainMixError",
    "FRAME_AXES", "MODALITIES", "MetricSet", "OCCLUSION_LABELS", "PIXEL_SIZE_BINS", "SliceError",
    "TIMES_OF_DAY", "ZONES", "altitude_band", "combine_rows", "is_pinned", "make_slice", "metric_row", "narrow",
    "occlusion_label", "partition", "pixel_size_bin", "posture_label", "ratio_row", "replace",
    "require_single_domain", "slice_axis_value", "time_of_day_bin",
]
