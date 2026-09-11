"""F19: the acceptance report over a flown campaign (SOLUTION_DOC 5.12, and 5.5c's reporting rule).

**The acceptance slice is fixed here, in code, before any model exists.** 5.5c step 5 says to define it before
measuring it, and `docs/TRACKER.md` did so on 2026-09-11 02:35:

    "40-60 m, daylight, occlusion below 50 %, non-submerged presentations. Report that number as the
     acceptance figure. Report the hard slices as separate numbers in the same table."

:func:`role_of` derives each pass's role from its own data card by those rules, rather than from a list of
pass names, so a pass flown later gets the right role automatically and nobody can promote a flattering slice
after the fact.

**Why the roles cannot be left to the slice grid.** ``SliceKey`` has an ``altitude_band`` axis, and it does not
separate the campaign:

    altitude_band(45.0) == "45-60"      <- alt45rain, the RAIN pass
    altitude_band(55.0) == "45-60"      <- alt55, the ACCEPTANCE pass

Both land in the same band. There is no weather or condition axis, so any aggregation over the grid folds a
rain pass into the headline figure and nothing in the type system objects. That is what :class:`SliceRoleError`
is for: passes with different roles are never pooled, in the same way and for the same reason that
:class:`~sightline.eval.slicing.DomainMixError` refuses to pool ``sim`` with ``real``. The protection is
structural rather than a convention someone has to remember.

**What runs without a model.** :func:`census` reports the DENOMINATORS — how many boxes the campaign actually
produced, per terrain, per pixel-size bin, per occlusion, per posture — which is answerable the moment the
frames land and is what decides whether a >= 90 % claim on the acceptance slice can carry any weight at all.
The detection half drops in unchanged once predictions exist.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from sightline.eval.context import (
    SceneGeometry,
    measure_dataset_contexts,
)
from sightline.eval.groundtruth import EvalDataset
from sightline.eval.slicing import (
    MetricSet,
    SliceError,
    make_slice,
    metric_row,
    narrow,
    pixel_size_bin,
)

PassRole = Literal["acceptance", "hard", "separate"]

#: The nominal slice, quoted from SOLUTION_DOC 5.5c and pinned before measurement.
NOMINAL_SLICE: dict[str, Any] = {
    "agl_m": (40.0, 60.0),
    "conditions": "daylight, clear",
    "occlusion_below": 0.50,
    "presentations": "non-submerged",
    "source": "SOLUTION_DOC 5.5c step 5; fixed in docs/TRACKER.md 2026-09-11 02:35",
}

#: 5.12 names the terrain types "water, debris, vegetation, roof". `roof` is spelled `structure` in the
#: `GtBox.context` vocabulary; printed by the report so the requirement stays legible.
TERRAIN_WORDING: dict[str, str] = {"structure": "roof / structure", "water": "water", "debris": "debris",
                                   "vegetation": "vegetation", "open_ground": "open ground",
                                   "vehicle": "vehicle"}


class SliceRoleError(RuntimeError):
    """Raised on any attempt to pool passes that were declared to play different roles.

    The sibling of `DomainMixError`. A rain pass and the acceptance pass share an altitude band, so without
    this the headline figure silently absorbs weather it was never defined to include.
    """


@dataclass(frozen=True, slots=True)
class PassSpec:
    """One flown pass and the role it was declared to play, with the reason recorded."""

    name: str
    agl_m: float
    condition: str
    role: PassRole
    why: str
    frames: int = 0
    minutes: float = 0.0
    boxes: int = 0
    weather: tuple[tuple[str, float], ...] = ()

    @property
    def is_acceptance(self) -> bool:
        return self.role == "acceptance"


def role_of(card: dict[str, Any], name: str = "") -> PassSpec:
    """Decide a pass's role from its own data card, by the rules fixed in :data:`NOMINAL_SLICE`."""
    agl = float(card.get("altitude_m_agl", 0.0) or 0.0)
    condition = str(card.get("condition", "") or "")
    weather = {k: float(v) for k, v in (card.get("weather") or {}).items() if v}
    lo, hi = NOMINAL_SLICE["agl_m"]

    if not (lo <= agl <= hi):
        side = "below" if agl < lo else "above"
        role: PassRole = "hard"
        why = (f"{agl:g} m is {side} the {lo:g}-{hi:g} m nominal band, so it is a named HARD slice. "
               f"5.12: the >= 90 % target is expected only at <= 60 m AGL.")
    elif weather:
        role = "separate"
        why = (f"in the {lo:g}-{hi:g} m band but flown in {', '.join(f'{k} {v:g}' for k, v in weather.items())}"
               f"; the nominal slice says daylight and clear, so this is reported SEPARATELY, never folded in. "
               f"NOTE: it shares an altitude_band with the acceptance pass, so the slice grid alone would "
               f"have merged them.")
    else:
        role = "acceptance"
        why = f"{agl:g} m, {condition or 'clear'} — inside the nominal slice. This is THE acceptance figure."

    return PassSpec(
        name=name or str(card.get("clip_id") or ""), agl_m=agl, condition=condition, role=role, why=why,
        frames=int(card.get("frames", 0) or 0), minutes=float(card.get("minutes", 0.0) or 0.0),
        boxes=int(card.get("total_boxes", 0) or 0),
        weather=tuple(sorted(weather.items())),
    )


def refuse_mixed_roles(specs: Sequence[PassSpec]) -> None:
    """Guard every aggregation. Pooling passes of different roles is a reporting error, not a choice."""
    roles = {s.role for s in specs}
    if len(roles) > 1:
        names = ", ".join(f"{s.name}({s.role})" for s in specs)
        raise SliceRoleError(
            f"refusing to pool passes with different roles: {names}. The acceptance figure comes from the "
            f"acceptance pass alone (5.5c); hard slices and weather passes are reported as their own rows in "
            f"the same table. Note that altitude_band does NOT separate them — a 45 m rain pass and a 55 m "
            f"clear pass are both '45-60'.")


def acceptance_spec(specs: Sequence[PassSpec]) -> PassSpec:
    """The single pass the headline figure may come from. Zero or many is a defect, not a default."""
    hits = [s for s in specs if s.is_acceptance]
    if len(hits) != 1:
        raise SliceRoleError(
            f"expected exactly one acceptance pass, found {len(hits)} "
            f"({', '.join(s.name for s in hits) or 'none'}). With none, there is no figure to report; with "
            f"several, someone would have to choose one after seeing the numbers.")
    return hits[0]


# --------------------------------------------------------------------------------- loading

def camera_track(root: Path) -> dict[int, tuple[float, float, float]]:
    """frame_idx -> (east_m, north_m, alt_msl_m) from the run's telemetry.

    `detect.dataset.load_run` keeps `alt_msl_m`, `lat` and `lon` but drops `east_m`/`north_m`, and a prediction
    cannot be projected to the ground without them. Read here rather than by widening `CaptureFrame`, which
    belongs to another lane.
    """
    out: dict[int, tuple[float, float, float]] = {}
    p = root / "telemetry.csv"
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["frame_idx"])] = (float(row["east_m"]), float(row["north_m"]),
                                              float(row["alt_msl_m"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_campaign_pass(root: str | Path, *, geom: SceneGeometry | None = None,
                       f_px: float = 0.0, cx_px: float = 0.0, cy_px: float = 0.0,
                       split: str = "test") -> tuple[EvalDataset, PassSpec, dict[str, int]]:
    """One flown pass, as an `EvalDataset` with camera poses and MEASURED terrain contexts.

    Reuses `detect.dataset.load_run` / `to_eval_dataset` — there is one capture reader, not two — then adds
    the two things the eval lane needs and the capture format does not carry: where the camera was, and what
    each box is actually sitting on.
    """
    from sightline.detect.dataset import load_run, to_eval_dataset  # local: heavy module, optional dependency

    root = Path(root)
    run = load_run(root)
    spec = role_of(run.card, name=root.name)
    ds = to_eval_dataset(run, split=split)

    track = camera_track(root)
    for fr in ds.frames:
        pose = track.get(fr.frame_idx)
        if pose is not None:
            fr.camera_east_m, fr.camera_north_m, fr.camera_asl_m = pose

    stats = {"measured": 0, "changed": 0, "unmeasured": 0}
    if geom is not None and f_px > 0.0:
        stats = measure_dataset_contexts(ds, geom, f_px=f_px, cx_px=cx_px, cy_px=cy_px)
    return ds, spec, stats


def load_campaign(dataset_dir: str | Path, *, include: Sequence[str] | None = None,
                  geom: SceneGeometry | None = None, f_px: float = 0.0,
                  cx_px: float = 0.0, cy_px: float = 0.0
                  ) -> list[tuple[EvalDataset, PassSpec, dict[str, int]]]:
    """Load the campaign. `include` NAMES the passes that constitute it.

    Which runs make up a campaign is a declaration, not something to infer from a directory listing: a
    superseded run sitting next to the real ones looks identical to `role_of`. `_artifacts/dataset/` currently
    holds `train_seed23`, an abandoned 45 m pass, which scores as `acceptance` on its own card and would make
    :func:`acceptance_spec` ambiguous. That ambiguity is REPORTED rather than resolved by guessing — the
    remedy is to name the passes.
    """
    dirs = [d for d in sorted(Path(dataset_dir).iterdir())
            if d.is_dir() and (d / "data_card.json").is_file()]
    if include is not None:
        want = list(include)
        by_name = {d.name: d for d in dirs}
        missing = [n for n in want if n not in by_name]
        if missing:
            raise SliceRoleError(f"named passes not found under {dataset_dir}: {missing}; "
                                 f"available: {sorted(by_name)}")
        dirs = [by_name[n] for n in want]
    return [load_campaign_pass(d, geom=geom, f_px=f_px, cx_px=cx_px, cy_px=cy_px) for d in dirs]


# --------------------------------------------------------------------------------- the model-free half

def census(ds: EvalDataset, spec: PassSpec) -> MetricSet:
    """The DENOMINATORS. What the campaign actually produced, before any detector exists.

    5.5c's >= 90 % expectation was written against "10-20k rendered tiles"; this campaign is deliberately much
    smaller. A recall figure over n boxes cannot resolve better than 1/n, so `recall_resolution` is reported
    next to the count — a 90 % claim on 40 boxes moves by 2.5 points if one box flips.
    """
    base = make_slice(ds.domain)
    ms = MetricSet()
    boxes = [b for f in ds.frames for b in f.boxes]
    scored = [b for b in boxes if not (b.uncertain or b.ignore or b.is_group)]
    n = len(scored)

    ms.add(metric_row("frames", float(len(ds.frames)), base, len(ds.frames), pass_=spec.name, role=spec.role))
    ms.add(metric_row("boxes_scored", float(n), base, n, pass_=spec.name, role=spec.role,
                      agl_m=spec.agl_m, condition=spec.condition))
    ms.add(metric_row("boxes_excluded", float(len(boxes) - n), base, len(boxes) - n, pass_=spec.name,
                      note="uncertain / ignore / group boxes, per 6.3 — not scored either way"))
    ms.add(metric_row("minutes", spec.minutes, base, n, pass_=spec.name))
    ms.add(metric_row("unique_gt_ids", float(len({b.gt_id for b in scored if b.gt_id >= 0})), base, n,
                      pass_=spec.name))
    # `metric_row` refuses a non-finite value and tells you to emit an explicit n=0 row with a note instead.
    # An empty pass has no resolution rather than an infinite one, and the note has to say so.
    ms.add(metric_row("recall_resolution", (1.0 / n) if n else 0.0, base, n, pass_=spec.name,
                      note=("one box flipping moves recall by this much" if n else
                            "undefined: this pass scored no boxes, so no recall can be computed from it")))

    for axis, label in (("context", lambda b: b.context),
                        ("pixel_size", lambda b: pixel_size_bin(b.size_px)),
                        ("occlusion", lambda b: str(b.occlusion) if b.occlusion is not None else "all"),
                        ("posture", lambda b: str(b.posture))):
        counts: dict[str, int] = {}
        for b in scored:
            v = label(b)
            counts[v] = counts.get(v, 0) + 1
        for value, k in sorted(counts.items()):
            try:
                key = narrow(base, **{axis: value})
            except SliceError:
                continue  # a label outside the axis vocabulary is counted nowhere rather than mislabelled
            ms.add(metric_row("boxes_scored", float(k), key, k, pass_=spec.name, role=spec.role))
    return ms


def render_census(results: Iterable[tuple[EvalDataset, PassSpec, MetricSet]]) -> str:
    """The campaign census as markdown. Roles first, so no number is read without its role."""
    rows = list(results)
    specs = [s for _, s, _ in rows]
    L: list[str] = ["# Campaign census — what the flown data can support",
                    "",
                    "Domain: **sim**. Every number below is a count of ground truth, not a detector result.",
                    "",
                    "## Pass roles, fixed before measuring (5.5c step 5)",
                    "",
                    "| pass | AGL | condition | role | boxes | minutes | why |",
                    "|---|---|---|---|---|---|---|"]
    for s in specs:
        L.append(f"| `{s.name}` | {s.agl_m:g} m | {s.condition or '-'} | **{s.role}** | {s.boxes} | "
                 f"{s.minutes:.1f} | {s.why} |")

    try:
        acc = acceptance_spec(specs)
        L += ["", f"**Acceptance figure comes from `{acc.name}` alone.** Everything else is a named row in the "
                  f"same table and is never averaged in."]
    except SliceRoleError as e:
        L += ["", f"> **No single acceptance pass.** {e}"]

    for ds, spec, ms in rows:
        L += ["", f"## `{spec.name}` — {spec.role}", ""]
        overall = [r for r in ms.rows if r.name != "boxes_scored" or r.slice.label() == f"domain={ds.domain}"]
        for r in overall:
            # `MetricRow.undefined` pins `value = 0.0` purely so the row survives strict JSON. Printing that
            # zero as a count would report "0 boxes measured" for a pass nobody measured. Same contract as
            # `eval/report.py::_value_str`; red-team review found this renderer had been missed.
            if not r.is_defined:
                L.append(f"- **{r.name}**: undefined — {r.undefined_reason}")
            elif r.name in ("frames", "boxes_scored", "boxes_excluded", "minutes", "unique_gt_ids"):
                L.append(f"- **{r.name}**: {r.value:g}")
            elif r.name == "recall_resolution":
                L.append(f"- **{r.name}**: {r.value:.4f} — one box flipping moves recall by "
                         f"{100 * r.value:.2f} points")
        for axis, title in (("context", "terrain (5.12: FP/min per terrain type)"),
                            ("pixel_size", "pixel height (the operating floor)"),
                            ("occlusion", "occlusion"), ("posture", "posture")):
            got = [r for r in ms.rows if r.name == "boxes_scored"
                   and getattr(r.slice, axis) not in ("all", "unknown")]
            if not got:
                continue
            L += ["", f"**{title}**", ""]
            for r in sorted(got, key=lambda r: -r.value):
                v = getattr(r.slice, axis)
                label = TERRAIN_WORDING.get(v, v) if axis == "context" else v
                L.append(f"- {label}: {int(r.value) if r.is_defined else 'undefined'}")
    return "\n".join(L) + "\n"


__all__ = ["NOMINAL_SLICE", "PassRole", "PassSpec", "SliceRoleError", "TERRAIN_WORDING", "acceptance_spec",
           "camera_track", "census", "load_campaign", "load_campaign_pass", "refuse_mixed_roles",
           "render_census", "role_of"]
