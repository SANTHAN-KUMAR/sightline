"""The human-readable evaluation report (§5.12): the slice table, with a domain column on every row.

    "Every table in this report carries a domain column (`sim` or `real`), and no figure is quoted without it
    (§5.5c). The acceptance figures come from the simulator; any real-footage figure is reported beside them,
    never averaged with them."

So the renderer:

  * takes a `MetricSet`, never floats, and prints `MetricRow`s;
  * renders **one section per domain**, and if both domains are present says so at the top instead of merging;
  * writes the domain into every table row as well, so a copied-out row still carries it;
  * refuses to render a headline claim without the word "in simulation" / "on real footage" attached.

`render_markdown()` returns a string; `write_report()` puts it under `_artifacts/eval/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sightline.eval.harness import EvalResult
from sightline.eval.slicing import AXIS_VALUES, MetricSet, is_pinned, slice_axis_value
from sightline.schemas import SCHEMA_VERSION, MetricRow

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO / "_artifacts" / "eval"

DOMAIN_PHRASE = {"sim": "in simulation", "real": "on real footage"}


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v)}"
    return f"{v:.4g}"


def _cell(text: Any) -> str:
    """A markdown table cell: pipes escaped (several metric bases contain `|x - y|`) and newlines removed."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _detail_str(detail: dict[str, Any], keys: Sequence[str] = ()) -> str:
    if not detail:
        return ""
    items = [(k, detail[k]) for k in (keys or detail) if k in detail]
    parts = []
    for k, v in items:
        if isinstance(v, float):
            v = _fmt(v)
        elif isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
            if len(v) > 60:
                v = v[:57] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def slice_table(rows: Iterable[MetricRow], axes: Sequence[str] = tuple(AXIS_VALUES),
                detail_keys: Sequence[str] = ()) -> str:
    """A markdown table. Column 1 is `domain` and it is never optional (hard rule 5)."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_\n"
    used = [ax for ax in axes if any(is_pinned(r.slice, ax) for r in rows)]
    head = ["domain", "metric", "value", "n", *used]
    show_detail = bool(detail_keys) or any(r.detail for r in rows)
    if show_detail:
        head.append("detail")
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [r.slice.domain, r.name, _fmt(r.value), str(r.n)]
        # `SliceKey.label()` treats "unknown" as "axis not pinned" (it is the Zone default), so the table does
        # the same and shows "all"; a frame whose zone is genuinely unknown is reported through `detail`.
        cells += [("all" if slice_axis_value(r.slice, ax) == "unknown" else slice_axis_value(r.slice, ax))
                  for ax in used]
        if show_detail:
            cells.append(_detail_str(r.detail, detail_keys))
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def _headline(ms: MetricSet, domain: str) -> list[str]:
    """The acceptance sentence, built so the domain word cannot be dropped from it."""
    out = []
    phrase = DOMAIN_PHRASE[domain]
    for name, label in (("recall@nominal", "nominal-slice recall at IoU 0.5"),
                        ("recall@nominal@iou0.25", "nominal-slice recall at IoU 0.25"),
                        ("recall@iou0.5", "whole-clip recall at IoU 0.5"),
                        ("recall@iou0.25", "whole-clip recall at IoU 0.25")):
        hits = ms.filter(name=name, domain=domain).overall(domain).rows
        if not hits:
            continue
        r = hits[0]
        conf = r.detail.get("at_conf")
        at = f" at the frozen operating confidence {_fmt(float(conf))}" if conf is not None else ""
        out.append(f"- **{r.value * 100:.1f} % {label}, {phrase}**{at} (n = {r.n} ground-truth boxes).")
    return out


def render_markdown(results: Sequence[EvalResult], title: str = "Sightline evaluation report") -> str:
    """Render one or more clips. Two domains produce two sections; nothing is ever pooled across them."""
    all_metrics = MetricSet()
    for r in results:
        all_metrics.extend(r.metrics.rows)
    domains = all_metrics.domains()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"Generated {now} · schema {SCHEMA_VERSION} · `sightline.eval` (F19)")
    L.append("")
    L.append("**Reading rule (SOLUTION_DOC 5.5c, HANDBOOK hard rule 5).** Every number below carries its "
             "domain. A `sim` number and a `real` number are never averaged and never appear in the same "
             "aggregate; where both exist they are shown side by side in separate sections.")
    L.append("")
    if len(domains) > 1:
        L.append(f"> This report contains **{' and '.join(domains)}** measurements. They are reported "
                 "separately, by design. Do not compute a combined figure from them.")
        L.append("")

    # --- runs ---------------------------------------------------------------------------------------
    L.append("## Runs")
    L.append("")
    L.append("| clip | domain | split | seed group | frames | fps processed | minutes | randomisation |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        ds = r.dataset
        L.append(f"| {ds.clip_id or '(unnamed)'} | {ds.domain} | {ds.split} | {ds.seed_group or '-'} | "
                 f"{ds.n_frames} | {ds.fps_processed:g} | {ds.minutes_processed:.2f} | "
                 f"{'ON' if ds.randomisation else 'off'} |")
    L.append("")

    for domain in domains:
        dm = all_metrics.filter(domain=domain)
        L.append(f"## Domain: {domain} ({DOMAIN_PHRASE[domain]})")
        L.append("")
        head = _headline(dm, domain)
        if head:
            L.append("### Headline")
            L.append("")
            L.extend(head)
            L.append("")

        # operating point
        op_rows = dm.filter(name="operating_conf").rows
        if op_rows:
            op = op_rows[0]
            achieved = op.detail.get("achieved")
            L.append("### Frozen operating threshold (SOLUTION_DOC 5.5)")
            L.append("")
            L.append(f"Confidence **{_fmt(op.value)}**, chosen as the highest confidence at which human recall "
                     f">= {op.detail.get('target_recall')} on `{op.detail.get('frozen_on', '?')}`, then frozen. "
                     f"Target {'met' if achieved else '**NOT met**'}. The four `operating_*` rows below are "
                     "measured on that clip; every other number in this report is measured on the test clip at "
                     "this same frozen confidence.")
            if not achieved and op.detail.get("note"):
                L.append("")
                L.append(f"> {op.detail['note']}")
            mf1 = op.detail.get("max_f1_conf")
            if mf1 is not None:
                L.append("")
                L.append(f"For comparison, the max-F1 confidence is {_fmt(float(mf1))} — that is where "
                         "Ultralytics quotes per-class precision and recall. The numbers below are **not** "
                         "taken there.")
            L.append("")
            L.append(slice_table(dm.filter(name="operating_recall").rows
                                 + dm.filter(name="operating_precision").rows
                                 + dm.filter(name="operating_fp_per_min").rows
                                 + dm.filter(name="recall@max_f1_conf").rows
                                 + dm.filter(name="precision@max_f1_conf").rows,
                                 detail_keys=("at_conf", "iou_thr", "frozen_on", "warning")))
            L.append("")

        sections: list[tuple[str, list[MetricRow], Sequence[str]]] = [
            ("Detection level, whole clip",
             [r for n in ("recall@iou0.5", "recall@iou0.25", "precision@iou0.5", "precision@iou0.25",
                          "fp_per_min@iou0.5", "fp_per_min@iou0.25", "f1@iou0.5")
              for r in dm.filter(name=n).overall(domain).rows],
             ("at_conf", "tp", "fp", "fn", "minutes")),
            ("Nominal slice (the acceptance claim, SOLUTION_DOC 5.5c step 5)",
             [r for n in ("recall@nominal", "recall@nominal@iou0.25", "precision@nominal", "fp_per_min@nominal")
              for r in dm.filter(name=n).rows],
             ("definition", "at_conf", "tp", "fn", "note")),
            ("Record level after tracking and dedup (SOLUTION_DOC 5.6)",
             [r for n in ("record_precision", "record_recall", "record_duplicate_rate",
                          "record_duplicate_rate_near", "fp_per_min@record", "fp_per_min_reduction",
                          "fp_per_min_eliminated", "count_mae", "count_bias", "count_exact_rate",
                          "record_localisation_median_m", "buried_survivors_excluded",
                          "r10_records_retained", "r10_dismissals_without_reason")
              for r in dm.filter(name=n).rows],
             ("basis", "radius_m", "matched", "note", "expectation", "guardrail")),
            ("Track level (SOLUTION_DOC 5.6, motmetrics)",
             [r for n in ("hota", "deta", "assa", "idf1", "idp", "idr", "id_switches", "fragmentations",
                          "mota", "motp", "mostly_tracked", "mostly_lost", "track_gt_identities")
              for r in dm.filter(name=n).rows],
             ("source", "max_iou")),
            ("Geolocation (SOLUTION_DOC 5.7)",
             [r for n in ("geo_error_median_m", "geo_error_p90_m", "geo_error_mean_m", "geo_error_max_m",
                          "geo_ce90_containment", "geo_published_ce90_mean_m",
                          "geo_error_median_m_by_off_nadir")
              for r in dm.filter(name=n).rows],
             ("basis", "expected", "noise_injected", "warning", "off_nadir_bin",
              "budget_reference_60m_nadir_1sigma_m")),
            ("Search-quality map calibration (SOLUTION_DOC 5.3, 5.12)",
             [r for n in ("pod_ece", "pod_mce", "pod_bias", "pod_bins_on_diagonal", "pod_observed_find_rate")
              for r in dm.filter(name=n).rows],
             ("presentation", "pod_bin", "predicted", "ci95", "n_found", "on_diagonal", "basis")),
            ("Posture / submersion head (SOLUTION_DOC 5.5a)",
             [r for n in ("posture_accuracy", "submersion_accuracy", "rank_corr_spearman_with_head",
                          "rank_corr_spearman_without_head", "rank_corr_spearman_delta",
                          "rank_corr_kendall_with_head", "rank_corr_kendall_without_head",
                          "rank_corr_kendall_delta", "posture_head_improves_ranking", "posture_demotions",
                          "posture_promotions")
              for r in dm.filter(name=n).rows],
             ("attribute", "gt_class", "axis", "verdict", "basis", "spearman_delta")),
        ]
        for heading, rows, keys in sections:
            if not rows:
                continue
            L.append(f"### {heading}")
            L.append("")
            L.append(slice_table(rows, detail_keys=keys))
            L.append("")

        # the mandatory slice grid
        pinned = [r for r in dm.rows if any(is_pinned(r.slice, ax) for ax in AXIS_VALUES)]
        grid = [r for r in pinned if r.name.startswith(("recall@iou", "precision@iou", "fp_per_min@iou"))]
        grid += dm.filter(name="slice_unbinned_gt").rows
        if grid:
            L.append("### Slice grid (mandatory: zone x altitude x time of day x occlusion x posture x "
                     "pixel size x modality)")
            L.append("")
            L.append("Box-level axes (occlusion, posture, pixel size) carry **recall only**: a false positive "
                     "belongs to a frame but to no ground-truth box, so precision and FP/min are undefined "
                     "there and are not invented.")
            L.append("")
            L.append(slice_table(sorted(grid, key=lambda r: (r.detail.get("axis", ""), r.name)),
                                 detail_keys=("axis", "at_conf", "tp", "fn", "fp")))
            L.append("")

        px = dm.filter(name="recall_by_pixel_height").rows
        if px:
            L.append("### Missed-detection analysis: recall vs pixel height (the operating floor)")
            L.append("")
            L.append("| domain | pixel-height bin | recall | n |")
            L.append("|---|---|---|---|")
            for r in px:
                L.append(f"| {r.slice.domain} | {r.detail.get('px_bin')} | {_fmt(r.value)} | {r.n} |")
            L.append("")

    # --- failure clusters ---------------------------------------------------------------------------
    clusters = [(r.dataset, c) for r in results for c in r.failure_clusters]
    if clusters:
        L.append("## Top failure clusters")
        L.append("")
        L.append("| domain | clip | submersion | zone | altitude | missed | median size px |")
        L.append("|---|---|---|---|---|---|---|")
        for ds, c in clusters:
            L.append(f"| {ds.domain} | {ds.clip_id or '-'} | {c['submersion']} | {c['zone']} | "
                     f"{c['altitude']} | {c['n_missed']} | {c['median_size_px']:.0f} |")
        L.append("")

    L.append("## How to reproduce")
    L.append("")
    L.append("```")
    L.append("D:\\Tools\\uv\\uv.exe run python -m sightline.eval --demo    # synthetic self-check + this report")
    L.append("D:\\Tools\\uv\\uv.exe run pytest tests/test_eval.py -q       # the metric correctness tests")
    L.append("```")
    L.append("")
    L.append("Browse the errors (FiftyOne lives in its own environment, `envs/fiftyone`):")
    L.append("")
    L.append("```")
    L.append("D:\\Sightline\\envs\\fiftyone\\.venv\\Scripts\\python.exe \\")
    L.append("    D:\\Sightline\\sightline\\eval\\fiftyone_browse.py <manifest.json>")
    L.append("```")
    L.append("")
    return "\n".join(L)


def write_report(results: Sequence[EvalResult], out_dir: Path | str = DEFAULT_REPORT_DIR,
                 stem: str = "", title: str = "Sightline evaluation report") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or f"report_{time.strftime('%Y%m%d_%H%M%S')}"
    path = out / f"{stem}.md"
    path.write_text(render_markdown(results, title), encoding="utf-8")
    json_path = out / f"{stem}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.time(),
        "runs": [{"clip_id": r.dataset.clip_id, "domain": r.dataset.domain, "split": r.dataset.split,
                  "seed_group": r.dataset.seed_group, "randomisation": r.dataset.randomisation,
                  "operating": asdict(r.operating) if r.operating else None,
                  "failure_clusters": r.failure_clusters} for r in results],
        "rows": [row for r in results for row in r.metrics.to_dicts()],
    }
    json_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


__all__ = ["DEFAULT_REPORT_DIR", "DOMAIN_PHRASE", "render_markdown", "slice_table", "write_report"]
