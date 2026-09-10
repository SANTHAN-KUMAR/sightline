"""F19 — the evaluation harness (SOLUTION_DOC 5.12), and the guard that keeps every number honest.

**The public API returns `MetricRow`, never a bare float.** A `MetricRow` cannot be built without a `SliceKey`,
a `SliceKey` cannot be built without a `domain`, and every aggregation path calls `require_single_domain()`
first, so a `sim` number and a `real` number cannot be averaged by accident (HANDBOOK hard rule 5,
SOLUTION_DOC 5.5c). `DomainMixError` is what you get if you try.

What the other lanes call:

    from sightline.eval import EvalDataset, GtBox, GtFrame, GtSurvivor      # ground truth in
    from sightline.eval import EvalConfig, freeze_on_validation             # 5.5 operating threshold
    from sightline.eval import run_detection_eval, add_record_eval          # 5.12 numbers out
    from sightline.eval import evaluate_tracking, frames_from_tracks        # 5.6 HOTA / IDF1 / IDsw
    from sightline.eval import evaluate_geolocation, samples_from_records   # 5.7 error + CE90 honesty
    from sightline.eval import evaluate_pod_calibration                     # 5.3 reliability diagram
    from sightline.eval import attribute_accuracy_rows, rank_correlation_rows   # 5.5a posture head
    from sightline.eval import write_manifest                               # FiftyOne error browser
    from sightline.eval import render_markdown, write_report                # the report

Typical order (and the order matters — see `harness.py`):

    op = freeze_on_validation(val_dataset, val_detections)       # highest conf with recall >= 0.92, frozen
    res = run_detection_eval(test_dataset, test_detections, op.conf, operating=op)
    res = add_record_eval(res, records)
    res.metrics.extend(evaluate_tracking(frames_from_tracks(test_dataset, tracks), res.base_key()))
    path = write_report([res])

`python -m sightline.eval --demo` runs the whole thing on a synthetic clip and writes a report, with no model,
no GPU, no editor and no AirSim.
"""

from sightline.eval.calibration import (
    PodBin,
    evaluate_pod_calibration,
    reliability_bins,
    survivors_found_by_cell,
    wilson_interval,
)
from sightline.eval.detection import (
    IOU_PRIMARY,
    IOU_RELAXED,
    TARGET_RECALL,
    DetCounts,
    MatchResult,
    OperatingPoint,
    PrCurve,
    false_positives,
    freeze_operating_threshold,
    headline_rows,
    match_dataset,
    missed_detections,
    recall_vs_pixel_height,
    slice_rows,
    sweep_confidence,
)
from sightline.eval.fiftyone_export import (
    ExportStats,
    build_manifest,
    failure_clusters,
    launch_command,
    write_manifest,
)
from sightline.eval.geoloc import GeoSample, evaluate_geolocation, samples_from_fixes, samples_from_records
from sightline.eval.groundtruth import (
    URGENCY_ORDER,
    EvalDataset,
    GtBox,
    GtFrame,
    GtSurvivor,
    check_split_disjoint,
    load_dataset,
    save_dataset,
)
from sightline.eval.harness import (
    NOMINAL,
    EvalConfig,
    EvalResult,
    add_record_eval,
    freeze_on_validation,
    nominal_masks,
    run_detection_eval,
)
from sightline.eval.matching import FrameMatch, gated_assignment, hungarian, iod_matrix, iou_matrix, match_frame
from sightline.eval.posture import (
    AttributeSample,
    RankingComparison,
    attribute_accuracy_rows,
    kendall_tau_b,
    promotion_audit_rows,
    rank_correlation_rows,
    spearman_rho,
)
from sightline.eval.records import (
    RecordMatch,
    dedup_radius_m,
    evaluate_records,
    match_records,
    r10_rows,
    tracking_gain_row,
)
from sightline.eval.report import render_markdown, slice_table, write_report
from sightline.eval.slicing import (
    AXIS_VALUES,
    BOX_AXES,
    DOMAINS,
    FRAME_AXES,
    DomainMixError,
    MetricSet,
    SliceError,
    altitude_band,
    combine_rows,
    make_slice,
    metric_row,
    narrow,
    occlusion_label,
    partition,
    pixel_size_bin,
    posture_label,
    ratio_row,
    require_single_domain,
    time_of_day_bin,
)
from sightline.eval.tracking import TrackFrame, accumulate, evaluate_tracking, frames_from_tracks

__all__ = [
    "AXIS_VALUES", "AttributeSample", "BOX_AXES", "DOMAINS", "DetCounts", "DomainMixError", "EvalConfig",
    "EvalDataset", "EvalResult", "ExportStats", "FRAME_AXES", "FrameMatch", "GeoSample", "GtBox", "GtFrame",
    "GtSurvivor", "IOU_PRIMARY", "IOU_RELAXED", "MatchResult", "MetricSet", "NOMINAL", "OperatingPoint",
    "PodBin", "PrCurve", "RankingComparison", "RecordMatch", "SliceError", "TARGET_RECALL", "TrackFrame",
    "URGENCY_ORDER", "accumulate", "add_record_eval", "altitude_band", "attribute_accuracy_rows",
    "build_manifest", "check_split_disjoint", "combine_rows", "dedup_radius_m", "evaluate_geolocation",
    "evaluate_pod_calibration", "evaluate_records", "evaluate_tracking", "failure_clusters", "false_positives",
    "frames_from_tracks", "freeze_on_validation", "freeze_operating_threshold", "gated_assignment",
    "headline_rows", "hungarian", "iod_matrix", "iou_matrix", "kendall_tau_b", "launch_command", "load_dataset",
    "make_slice", "match_dataset", "match_frame", "match_records", "metric_row", "missed_detections", "narrow",
    "nominal_masks", "occlusion_label", "partition", "pixel_size_bin", "posture_label",
    "promotion_audit_rows", "r10_rows", "rank_correlation_rows", "ratio_row", "recall_vs_pixel_height",
    "reliability_bins", "render_markdown", "require_single_domain", "run_detection_eval", "samples_from_fixes",
    "samples_from_records", "save_dataset", "slice_rows", "slice_table", "spearman_rho",
    "survivors_found_by_cell", "sweep_confidence", "time_of_day_bin", "tracking_gain_row", "wilson_interval",
    "write_manifest", "write_report",
]
