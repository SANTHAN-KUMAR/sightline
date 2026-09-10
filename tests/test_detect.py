"""F8 / F8b / F9 / F10 — offline tests for the detection lane. No GPU, no torch, no editor, no AirSim.

Every number asserted here is either taken from `docs/SOLUTION_DOC.md` (§5.5, §5.5a, §5.5c, §5.11, §6.3) or
worked out by hand in the test itself and shown in a comment. Nothing is compared against the code's own output.

The tests that matter most, and what they would catch:

* `test_tile_roundtrip_is_exact_*` — an off-by-one in the inverse mapping puts every box a pixel out, which
  looks fine on a contact sheet and quietly costs IoU on 20 px targets.
* `test_seam_*` — a target on a tile boundary counted twice, or lost entirely.
* `test_yolo_line_matches_the_capture_runs_own_label_file` — the +1 in `capture_box_to_xyxy`. If this is wrong
  the model trains on boxes shifted half a pixel from the ones it is scored against.
* `test_split_by_seed_*` — frame-level leakage, which §5.5c says makes every downstream number worthless.
* `test_sweep_picks_the_highest_confidence_above_target` — the §5.5 operating rule, on a curve built by hand.
* `test_rgb_only_fallback_returns_the_same_objects` — the structural fallback of §5.5, checked by identity.
* `test_posture_can_never_demote` — the §5.5a safety rule, over the whole prediction cross-product.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sightline.detect.dataset import (
    CLASSES,
    MIN_BOX_PX,
    CaptureFrame,
    CaptureLabel,
    CaptureRun,
    assert_no_seed_leak,
    build_yolo_dataset,
    capture_box_to_xyxy,
    clip_box_to_tile,
    load_run,
    split_by_seed,
    summarise_boxes,
    tile_boxes,
    to_eval_dataset,
    urgency_for,
    yolo_line,
)
from sightline.detect.fusion import (
    Registration,
    fuse_detections,
    fuse_frame,
    homography_from_intrinsics,
    thermal_weight,
    warp_boxes,
)
from sightline.detect.overlay import detections_from_json, detections_to_json, overlay_frame, zoom_panel
from sightline.detect.rgb import DetectorConfig, boxes_to_detections, detect_frame
from sightline.detect.threshold import (
    TARGET_RECALL,
    FrameEval,
    GroundTruthBox,
    choose_operating_threshold,
    fp_per_minute,
    freeze_operating_point,
    load_operating_point,
    match_frame,
    metric_rows,
    sliced_operating_points,
    sweep,
    tally,
)
from sightline.detect.tiler import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_PX,
    TileGrid,
    containment_xyxy,
    iou_xyxy,
    map_box_to_frame,
    merge_tile_detections,
    tile_batches,
    to_frame,
)
from sightline.detect.verifier import (
    BASE_URGENCY,
    URGENCY_RANK,
    VerifierOutput,
    apply_batch,
    apply_verifier_output,
    crop_for,
    suppress_false_positives,
    urgency_of,
)
from sightline.schemas import OCCLUSION_BINS, POSTURES, SUBMERSIONS, Detection, FrameBundle, Intrinsics, Telemetry

REPO = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 3840, 2160


def det(x1, y1, x2, y2, score=0.9, cls="human", tile_idx=-1) -> Detection:
    return Detection(bbox_px=(float(x1), float(y1), float(x2), float(y2)), score=float(score), cls=cls,
                     tile_idx=tile_idx)


def gt(x1, y1, x2, y2, **kw) -> GroundTruthBox:
    return GroundTruthBox(bbox_px=(float(x1), float(y1), float(x2), float(y2)), **kw)


# =============================================================================================================
# 1. Tiling (F8, §5.5, §5.11)
# =============================================================================================================
def test_default_grid_over_4k_is_the_designs_grid():
    """3840x2160 at 1024 tiles / 0.2 overlap. Worked out by hand:

    x: stride = round(1024*0.8) = 819; n = ceil((3840-1024)/819)+1 = ceil(3.438)+1 = 5
    y: n = ceil((2160-1024)/819)+1 = ceil(1.387)+1 = 3   ->  15 tiles
    """
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    assert (g.n_cols, g.n_rows, len(g)) == (5, 3, 15)
    # last tile flush with the far edge, first at the origin
    assert g[0].x0 == 0 and g[0].y0 == 0
    assert g[len(g) - 1].x1 == FRAME_W and g[len(g) - 1].y1 == FRAME_H
    # the realised overlap is never LESS than requested
    ox, oy = g.effective_overlap
    assert ox >= DEFAULT_OVERLAP - 1e-9 and oy >= DEFAULT_OVERLAP - 1e-9
    # every pixel of the frame is inside at least one tile
    cover = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    for t in g:
        cover[t.y0:t.y1, t.x0:t.x1] = True
    assert cover.all()


def test_native_resolution_means_one_tile_pixel_is_one_frame_pixel():
    """The §5.5c claim the whole recipe rests on: no rescale between capture and training/inference."""
    frame = np.arange(FRAME_H * FRAME_W, dtype=np.int64).reshape(FRAME_H, FRAME_W).astype(np.uint8)
    frame = np.dstack([frame] * 3)
    g = TileGrid.for_frame(frame)
    t = g[7]
    crop = t.crop(frame)
    assert crop.shape[:2] == (1024, 1024)
    assert np.array_equal(crop, frame[t.y0:t.y0 + 1024, t.x0:t.x0 + 1024])


def test_max_safe_target_px_is_the_overlap_band():
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    ox, oy = g.overlap_px
    assert g.max_safe_target_px == min(ox, oy)
    # A target that size, placed anywhere, is WHOLE inside at least one tile. Check by brute force on a lattice.
    s = g.max_safe_target_px
    for x in range(0, FRAME_W - s, 137):
        for y in range(0, FRAME_H - s, 149):
            box = (x, y, x + s, y + s)
            assert any(t.contains_box(box) for t in g), f"{box} is whole in no tile"


def test_tile_roundtrip_is_exact_for_a_box_inside_a_tile():
    """frame -> tile-local -> frame must be bit-exact, or every 20 px target loses IoU to an off-by-one."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    for t in g:
        box = (t.x0 + 13.25, t.y0 + 7.5, t.x0 + 41.75, t.y0 + 63.0)
        local, frac = clip_box_to_tile(box, t)
        assert frac == 1.0
        back = map_box_to_frame(t, local)
        assert back == box


def test_tile_roundtrip_is_exact_for_every_tile_of_a_clipped_box():
    """A box straddling a seam: each fragment maps back to exactly the intersection with its tile."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    from sightline.detect.tiler import intersection_area

    seam = g[0].x1  # 1024
    box = (seam - 30.0, 400.0, seam + 30.0, 470.0)
    hits = [t for t in g if intersection_area(box, t.rect) > 0]
    assert len(hits) >= 2
    for t in hits:
        local, _ = clip_box_to_tile(box, t)
        back = map_box_to_frame(t, local)
        assert back == (max(box[0], t.x0), max(box[1], t.y0), min(box[2], t.x1), min(box[3], t.y1))


def test_to_frame_stamps_tile_provenance_and_frame_index():
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    per_tile = [[det(10, 10, 40, 70, 0.8)] for _ in range(len(g))]
    out = to_frame(per_tile, g, frame_idx=17)
    assert len(out) == len(g)
    for d, t in zip(out, g):
        assert d.tile_idx == t.idx and d.frame_idx == 17
        assert d.bbox_px == (t.x0 + 10.0, t.y0 + 10.0, t.x0 + 40.0, t.y0 + 70.0)


def test_to_frame_supports_a_subset_of_tiles():
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    chosen = [3, 9]
    out = to_frame([[det(5, 5, 25, 45)], [det(1, 2, 21, 42)]], g, chosen)
    assert [d.tile_idx for d in out] == chosen
    assert out[0].bbox_px == (g[3].x0 + 5.0, g[3].y0 + 5.0, g[3].x0 + 25.0, g[3].y0 + 45.0)
    with pytest.raises(ValueError):
        to_frame([[det(0, 0, 1, 1)]], g, [1, 2])


def test_seam_fragments_of_one_target_merge_into_the_union():
    """A target WIDER than the overlap band is in no tile whole: both tiles see a clipped fragment (§5.5)."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    left, right = g[0], g[1]
    seam_lo, seam_hi = right.x0, left.x1          # the overlap band [819, 1024)
    box = (seam_lo - 300.0, 500.0, seam_hi + 300.0, 700.0)   # 824 px wide > max_safe_target_px (205)
    a = det(box[0], box[1], left.x1, box[3], 0.80, tile_idx=left.idx)     # clipped at the left tile's right edge
    b = det(right.x0, box[1], box[2], box[3], 0.70, tile_idx=right.idx)   # clipped at the right tile's left edge
    merged = merge_tile_detections([a, b], g)
    assert len(merged) == 1
    assert merged[0].bbox_px == box
    assert merged[0].score == 0.80  # the higher-scoring fragment donates its score


def test_seam_union_does_not_merge_two_separate_targets():
    """Two people on opposite sides of a seam, each WHOLE in its own tile: only one side is clipped, so no union."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    left, right = g[0], g[1]
    a = det(left.x1 - 120.0, 500.0, left.x1 - 90.0, 560.0, 0.8, tile_idx=left.idx)
    b = det(right.x0 + 90.0, 505.0, right.x0 + 120.0, 565.0, 0.7, tile_idx=right.idx)
    merged = merge_tile_detections([a, b], g)
    assert len(merged) == 2


def test_duplicate_across_two_tiles_is_suppressed_once():
    """One survivor inside the overlap band is detected by both tiles; the merge must keep exactly one."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    left, right = g[0], g[1]
    box = (900.0, 600.0, 928.0, 658.0)  # inside [819, 1024): whole in both tiles
    assert left.contains_box(box) and right.contains_box(box)
    a = det(*box, 0.91, tile_idx=left.idx)
    b = det(box[0] + 1, box[1] + 1, box[2] + 1, box[3] + 1, 0.72, tile_idx=right.idx)
    assert iou_xyxy(a.bbox_px, b.bbox_px) > 0.55
    merged = merge_tile_detections([a, b], g)
    assert len(merged) == 1 and merged[0].score == 0.91 and merged[0].tile_idx == left.idx


def test_edge_fragment_is_absorbed_by_the_whole_box_from_the_other_tile():
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    left, right = g[0], g[1]
    whole = det(1000.0, 600.0, 1060.0, 700.0, 0.9, tile_idx=right.idx)
    sliver = det(1000.0, 600.0, left.x1, 700.0, 0.6, tile_idx=left.idx)  # 24 px wide fragment
    assert containment_xyxy(whole.bbox_px, sliver.bbox_px) == pytest.approx(1.0)
    assert iou_xyxy(whole.bbox_px, sliver.bbox_px) < 0.55
    merged = merge_tile_detections([whole, sliver], g)
    assert len(merged) == 1 and merged[0].bbox_px == whole.bbox_px


def test_two_nested_boxes_from_the_SAME_tile_are_left_alone():
    """Containment is a cross-tile repair. Inside one tile, nesting is the detector's own output."""
    g = TileGrid.build(FRAME_W, FRAME_H, DEFAULT_TILE_PX, DEFAULT_OVERLAP)
    big = det(100.0, 100.0, 300.0, 400.0, 0.9, tile_idx=0)
    small = det(150.0, 150.0, 200.0, 250.0, 0.8, tile_idx=0)
    assert containment_xyxy(big.bbox_px, small.bbox_px) == pytest.approx(1.0)
    assert len(merge_tile_detections([big, small], g)) == 2


def test_tile_batches_partition_exactly():
    assert tile_batches(15, 6) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14]]
    assert sum(len(b) for b in tile_batches(15, 4)) == 15
    with pytest.raises(ValueError):
        tile_batches(4, 0)


def test_grid_refuses_a_frame_of_the_wrong_size():
    g = TileGrid.build(FRAME_W, FRAME_H)
    with pytest.raises(ValueError):
        g.crops(np.zeros((100, 100, 3), np.uint8))


# =============================================================================================================
# 2. detect_frame with a fake inferencer (F8, CPU)
# =============================================================================================================
def test_detect_frame_maps_a_planted_box_back_exactly():
    """One synthetic target at a known place; a fake inferencer 'finds' it in whichever tile contains it."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    target = (1500.0, 900.0, 1528.0, 958.0)  # 28 x 58 px: a standing survivor at 45 m (§5.5c step 1)
    g = TileGrid.for_frame(frame)
    holders = [t for t in g if t.contains_box(target)]
    assert holders, "the target should be whole in at least one tile"

    calls = {"i": 0}

    def infer(crops):
        res = []
        for _ in crops:
            t = g[calls["i"]]
            calls["i"] += 1
            if t.contains_box(target):
                res.append([det(target[0] - t.x0, target[1] - t.y0, target[2] - t.x0, target[3] - t.y0, 0.77)])
            else:
                res.append([])
        return res

    out = detect_frame(frame, infer, frame_idx=5)
    assert len(out) == 1
    assert out[0].bbox_px == target
    assert out[0].frame_idx == 5 and out[0].tile_idx in {t.idx for t in holders}


def test_detect_frame_batching_does_not_change_the_answer():
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    seq = {"i": 0}

    def infer(crops):
        res = []
        for _ in crops:
            i = seq["i"]
            seq["i"] += 1
            res.append([det(10, 10, 38, 68, 0.5 + i / 100.0)])
        return res

    a = detect_frame(frame, infer, cfg=DetectorConfig(max_batch=0))
    seq["i"] = 0
    b = detect_frame(frame, infer, cfg=DetectorConfig(max_batch=4))
    assert [d.bbox_px for d in a] == [d.bbox_px for d in b]
    assert [d.tile_idx for d in a] == [d.tile_idx for d in b]


def test_detect_frame_rejects_a_ragged_inferencer():
    frame = np.zeros((2048, 2048, 3), np.uint8)
    with pytest.raises(ValueError):
        detect_frame(frame, lambda crops: [[]])


def test_boxes_to_detections_maps_class_ids():
    d = boxes_to_detections(np.array([[0, 0, 10, 20], [5, 5, 9, 9]]), np.array([0.4, 0.9]), np.array([1, 0]))
    assert [x.cls for x in d] == ["animal", "human"]
    assert d[0].score == pytest.approx(0.4)
    with pytest.raises(ValueError):
        boxes_to_detections(np.zeros((2, 4)), np.zeros(1), np.zeros(2))


# =============================================================================================================
# 3. Dataset conversion (F8b, §5.5c, §6.3)
# =============================================================================================================
def _fake_run(seed: int, n_frames: int = 3, clip: str = "") -> CaptureRun:
    frames = []
    for i in range(n_frames):
        labels = [CaptureLabel(bbox_px=(500.0 + 100 * i, 400.0, 528.0 + 100 * i, 458.0), cls="human",
                               actor_id=10 + i, size_px=58, pose="standing", submersion="dry", occlusion=0,
                               zone="settlement")]
        frames.append(CaptureFrame(stem=f"c{seed}_{i:05d}", frame_idx=i, image_path=Path(f"/x/{i}.png"),
                                   labels=labels, agl_m=45.0, gsd_cm_px=1.766, clip_id=clip or f"clip{seed}"))
    return CaptureRun(root=Path(f"/runs/{seed}"), clip_id=clip or f"clip{seed}", scenario_seed=seed,
                      frames=frames, card={"scenario_seed": seed, "altitude_m_agl": 45.0, "randomisation": "off"})


def test_capture_box_to_xyxy_adds_the_inclusive_pixel():
    assert capture_box_to_xyxy([922, 9, 975, 66]) == (922.0, 9.0, 976.0, 67.0)


def _captured_runs() -> list[Path]:
    """Whatever the capture lane has written so far. The run directory names change between sessions, so this
    discovers them instead of hard-coding one."""
    root = REPO / "_artifacts" / "dataset"
    return sorted(p.parent for p in root.glob("*/labels") if (p.parent / "images").is_dir())


def test_yolo_line_matches_the_capture_runs_own_label_file():
    """The +1 convention, checked against `tools/capture/labels.to_yolo` output on the REAL captured run.

    This is a cross-lane agreement test: if the capture lane changes its box convention, training boxes and
    scoring boxes silently diverge and this fails.
    """
    runs = _captured_runs()
    if not runs:
        pytest.skip("no captured run under _artifacts/dataset")
    run_dir = runs[-1]
    run = load_run(run_dir)
    checked = 0
    # A capture may be running: its newest frames can have a written .json and a half-written .txt. Drop the
    # last few rather than racing them; the convention is a property of the writer, not of the newest file.
    for fr in run.frames[:-3]:
        txt = run_dir / "labels" / f"{fr.stem}.txt"
        lines = [x for x in txt.read_text(encoding="utf-8").split("\n") if x.strip()] if txt.exists() else []
        if not lines:
            continue
        assert len(lines) == len(fr.labels)
        for line, m in zip(lines, fr.labels):
            cid, cx, cy, bw, bh = line.split()
            x1, y1, x2, y2 = m.bbox_px
            assert int(cid) == CLASSES.index(m.cls)
            assert float(cx) == pytest.approx((x1 + x2) / 2 / fr.width_px, abs=1e-6)
            assert float(cy) == pytest.approx((y1 + y2) / 2 / fr.height_px, abs=1e-6)
            assert float(bw) == pytest.approx((x2 - x1) / fr.width_px, abs=1e-6)
            assert float(bh) == pytest.approx((y2 - y1) / fr.height_px, abs=1e-6)
            checked += 1
    assert checked > 0, "the run has no labelled boxes to check the convention against"


def test_tile_boxes_keeps_a_whole_target_and_drops_a_sliver():
    g = TileGrid.build(FRAME_W, FRAME_H)
    t = g[0]
    whole = CaptureLabel(bbox_px=(500.0, 400.0, 528.0, 458.0))
    sliver = CaptureLabel(bbox_px=(t.x1 - 4.0, 400.0, t.x1 + 56.0, 458.0))  # 4 of 60 px inside -> 6.7 %
    keep, dropped = tile_boxes([whole, sliver], t)
    assert [b.source for b in keep] == [whole]
    assert keep[0].visible_frac == 1.0 and keep[0].truncated is False
    assert dropped == [sliver]


def test_tile_boxes_visible_fraction_is_the_area_ratio():
    g = TileGrid.build(FRAME_W, FRAME_H)
    t = g[0]
    # 100 x 100 box with exactly half its width inside the tile -> frac 0.5
    lab = CaptureLabel(bbox_px=(t.x1 - 50.0, 300.0, t.x1 + 50.0, 400.0))
    keep, _dropped = tile_boxes([lab], t, min_visible_frac=0.5)
    assert len(keep) == 1 and keep[0].visible_frac == pytest.approx(0.5) and keep[0].truncated
    keep2, dropped2 = tile_boxes([lab], t, min_visible_frac=0.51)
    assert keep2 == [] and dropped2 == [lab]


def test_tile_boxes_enforces_the_8px_floor_of_6_3():
    g = TileGrid.build(FRAME_W, FRAME_H)
    t = g[0]
    tiny = CaptureLabel(bbox_px=(100.0, 100.0, 100.0 + MIN_BOX_PX - 1, 100.0 + MIN_BOX_PX - 1))
    assert tile_boxes([tiny], t)[0] == []
    ok = CaptureLabel(bbox_px=(100.0, 100.0, 100.0 + MIN_BOX_PX, 100.0 + MIN_BOX_PX))
    assert len(tile_boxes([ok], t)[0]) == 1


def test_yolo_line_is_hand_computable():
    g = TileGrid.build(FRAME_W, FRAME_H)
    t = g[0]
    lab = CaptureLabel(bbox_px=(t.x0 + 512.0, t.y0 + 256.0, t.x0 + 552.0, t.y0 + 356.0))  # 40 x 100 at (512,256)
    box = tile_boxes([lab], t)[0][0]
    line = yolo_line(box, t.width_px, t.height_px)
    cid, cx, cy, bw, bh = line.split()
    assert cid == "0"
    # six decimals, exactly as `tools/capture/labels.to_yolo` writes them
    assert float(cx) == pytest.approx((512 + 552) / 2 / 1024, abs=1e-6)   # 0.520508
    assert float(cy) == pytest.approx((256 + 356) / 2 / 1024, abs=1e-6)   # 0.298828
    assert float(bw) == pytest.approx(40 / 1024, abs=1e-6)                # 0.039062(5)
    assert float(bh) == pytest.approx(100 / 1024, abs=1e-6)               # 0.097656(25)


def test_yolo_line_refuses_a_box_outside_the_tile():
    from sightline.detect.dataset import TiledBox

    bad = TiledBox(0, (10.0, 10.0, 2000.0, 40.0), 1.0, False, CaptureLabel(bbox_px=(0.0, 0.0, 1.0, 1.0)))
    with pytest.raises(ValueError):
        yolo_line(bad, 1024, 1024)


def test_implausible_boxes_catches_a_mask_colour_leak():
    """A person cannot be 68 m across. This is the check that caught the 2026-09-10 `train_seed23` capture,
    in which the segmentation mask collapsed to a single colour and the terrain was labelled `Human_019`.

    At the captured GSD of 1.766 cm/px, a 3840 px box is 3840 * 0.01766 = 67.81 m of ground.
    """
    from sightline.detect.dataset import MAX_TARGET_GROUND_M, assert_boxes_are_plausible, implausible_boxes

    run = _fake_run(1, n_frames=2)
    for fr in run.frames:
        fr.gsd_cm_px = 1.766
    assert implausible_boxes([run]) == []          # 58 px = 1.02 m: a person

    run.frames[1].labels = [CaptureLabel(bbox_px=(0.0, 0.0, 3840.0, 2160.0), actor_id=19, name="Human_019")]
    bad = implausible_boxes([run])
    assert len(bad) == 1
    assert bad[0]["ground_m"] == pytest.approx(3840 * 0.01766, abs=0.01)
    assert bad[0]["ground_m"] > MAX_TARGET_GROUND_M
    assert bad[0]["name"] == "Human_019"
    with pytest.raises(ValueError, match="DO NOT TRAIN"):
        assert_boxes_are_plausible([run])


def test_build_yolo_dataset_refuses_an_implausible_box():
    run = _fake_run(1, n_frames=1)
    run.frames[0].gsd_cm_px = 1.766
    run.frames[0].labels = [CaptureLabel(bbox_px=(0.0, 0.0, 3840.0, 2160.0), actor_id=19, name="Human_019")]
    with pytest.raises(ValueError, match="DO NOT TRAIN"):
        build_yolo_dataset({"train": [run], "val": [_fake_run(2, n_frames=1)]}, "D:/nope", dry_run=True)
    # ... and the override exists, is explicit, and is recorded in the manifest
    man = build_yolo_dataset({"train": [run], "val": [_fake_run(2, n_frames=1)]}, "D:/nope", dry_run=True,
                             allow_implausible_boxes=True)
    assert man["implausible_boxes_allowed"] is True


def test_a_frame_with_no_gsd_is_not_silently_passed():
    """Telemetry is written when a capture finishes. Without it the check cannot run, and must not pretend to."""
    from sightline.detect.dataset import implausible_boxes

    run = _fake_run(1, n_frames=1)
    run.frames[0].gsd_cm_px = 0.0
    run.frames[0].labels = [CaptureLabel(bbox_px=(0.0, 0.0, 3840.0, 2160.0), actor_id=19)]
    assert implausible_boxes([run]) == []       # skipped, not passed
    run.frames[0].gsd_cm_px = 1.766
    assert len(implausible_boxes([run])) == 1   # the same box fails as soon as the GSD is known


def test_the_real_capture_run_is_checked_for_plausible_boxes():
    """Runs the guard against whatever the capture lane has on disk and reports what it finds.

    This test does NOT require the capture to be clean -- it requires the guard to have an opinion about it,
    with the arithmetic shown, so a broken capture can never quietly become a training set.
    """
    from sightline.detect.dataset import implausible_boxes

    runs = _captured_runs()
    if not runs:
        pytest.skip("no captured run under _artifacts/dataset")
    loaded = [load_run(p) for p in runs]
    with_gsd = [f for r in loaded for f in r.frames if f.gsd_cm_px > 0]
    if not with_gsd:
        pytest.skip("no telemetry yet: the GSD needed for the plausibility check is not written until a "
                    "capture finishes")
    bad = implausible_boxes(loaded)
    total = sum(r.n_boxes for r in loaded)
    for b in bad:
        # `ground_m` is rounded to 2 dp for reporting, so a box just over the bar can print as exactly 3.0
        assert b["ground_m"] >= 3.0 and b["longest_px"] > 0    # the arithmetic is real, not a flag
        assert b["longest_px"] * b["gsd_cm_px"] / 100.0 > 3.0  # ... and the unrounded value really is over
    assert total >= 0
    # The finding itself is reported by `docs/lanes/detect.md`; the assertion here is only that the guard ran.
    assert isinstance(bad, list)


def test_split_by_seed_never_puts_one_seed_in_two_splits():
    runs = [_fake_run(s) for s in (11, 23, 37, 41)]
    splits = split_by_seed(runs, val_seeds=[23])
    assert [r.scenario_seed for r in splits["val"]] == [23]
    assert {r.scenario_seed for r in splits["train"]} == {11, 37, 41}
    assert_no_seed_leak(splits)


def test_assert_no_seed_leak_actually_fails():
    r = _fake_run(7)
    with pytest.raises(ValueError, match="more than one split"):
        assert_no_seed_leak({"train": [r], "val": [r]})


def test_split_by_seed_refuses_a_single_seed():
    with pytest.raises(ValueError, match="at least two"):
        split_by_seed([_fake_run(5), _fake_run(5, clip="second")])


def test_split_by_seed_is_deterministic():
    runs = [_fake_run(s) for s in (1, 2, 3, 4)]
    a = split_by_seed(runs, seed=3)
    b = split_by_seed(runs, seed=3)
    assert [r.clip_id for r in a["val"]] == [r.clip_id for r in b["val"]]


def test_build_yolo_dataset_dry_run_counts_are_consistent():
    """Hand count. The 5 column starts of a 3840-wide frame at 1024/0.2 are 0, 704, 1408, 2112, 2816.

    `_fake_run` puts one 28 px box at x = 500, 600, 700 in successive frames. The first two fall in column 0
    only; the third (700-728) also falls in column 1 (704-1728), keeping 24 of its 28 px = 86 %, which is above
    MIN_VISIBLE_FRAC. So a run of 3 frames yields 4 positive tiles and 4 boxes, and two runs yield 8 of each.
    A target inside the overlap band appearing in BOTH tiles is the intended behaviour: it is what makes the
    same target recoverable at inference from either tile.
    """
    splits = {"train": [_fake_run(1), _fake_run(2)], "val": [_fake_run(3)]}
    man = build_yolo_dataset(splits, "D:/nonexistent/never-written", dry_run=True, negative_frac=0.0)
    tr = man["splits"]["train"]
    assert tr["frames"] == 6
    assert tr["boxes"] == 8 and tr["positive_tiles"] == 8 and tr["background_tiles"] == 0
    assert tr["dropped_fragments"] == 0
    assert tr["median_box_px"] == pytest.approx(58.0)
    assert tr["seeds"] == [1, 2] and man["splits"]["val"]["seeds"] == [3]
    assert "train: images/train" in man["data_yaml_text"] and "0: human" in man["data_yaml_text"]
    assert not Path("D:/nonexistent/never-written").exists()


def test_build_yolo_dataset_refuses_a_leaked_seed():
    r = _fake_run(9)
    with pytest.raises(ValueError):
        build_yolo_dataset({"train": [r], "val": [r]}, "D:/nope", dry_run=True)


def test_background_tiles_are_sampled_not_flooded():
    """15 tiles per frame, one of them positive. With negative_frac=0 no background tile is written."""
    man = build_yolo_dataset({"train": [_fake_run(1, n_frames=1)]}, "D:/nope", dry_run=True, negative_frac=0.0)
    assert man["splits"]["train"]["background_tiles"] == 0
    man2 = build_yolo_dataset({"train": [_fake_run(1, n_frames=1)]}, "D:/nope", dry_run=True, negative_frac=1.0)
    s = man2["splits"]["train"]
    assert s["background_tiles"] + s["positive_tiles"] + s["skipped_fragment_tiles"] == 15


def test_build_yolo_dataset_writes_a_real_tile(tmp_path):
    """Actually cut pixels once, so the write path is exercised, not just the counting path."""
    import cv2

    img_dir = tmp_path / "src"
    img_dir.mkdir()
    frame = np.zeros((2160, 3840, 3), np.uint8)
    frame[400:458, 500:528] = 255
    p = img_dir / "f_00000.png"
    cv2.imwrite(str(p), frame)
    run = _fake_run(1, n_frames=1)
    run.frames[0].image_path = p
    val = _fake_run(2, n_frames=1)
    val.frames[0].image_path = p
    out = tmp_path / "yolo"
    man = build_yolo_dataset({"train": [run], "val": [val]}, out, negative_frac=0.0)
    labs = sorted((out / "labels" / "train").glob("*.txt"))
    imgs = sorted((out / "images" / "train").glob("*.png"))
    assert len(labs) == 1 and len(imgs) == 1
    assert cv2.imread(str(imgs[0])).shape == (1024, 1024, 3)
    line = labs[0].read_text().strip().split()
    assert line[0] == "0"
    assert (out / "data.yaml").exists() and json.loads((out / "manifest.json").read_text())["domain"] == "sim"
    assert man["splits"]["train"]["boxes"] == 1


def test_to_eval_dataset_marks_a_non_detectable_actor_uncertain():
    run = _fake_run(1, n_frames=1)
    run.frames[0].labels = list(run.frames[0].labels) + [
        CaptureLabel(bbox_px=(1000.0, 1000.0, 1030.0, 1060.0), actor_id=62, aerially_detectable=False,
                     pose="prone", submersion="dry", zone="fan")]
    ds = to_eval_dataset(run, split="test")
    boxes = ds.frames[0].boxes
    assert len(boxes) == 2
    assert [b.uncertain for b in boxes] == [False, True]
    assert [b.scored for b in boxes] == [True, False]
    assert ds.domain == "sim" and ds.randomisation is False and ds.seed_group == "seed1"


def test_to_eval_dataset_survivors_carry_the_buried_flag():
    """`data/scene/actors.json` marks 55 and 62 as not aerially detectable (§2.7)."""
    actors = REPO / "data" / "scene" / "actors.json"
    if not actors.exists():
        pytest.skip("no scene actors.json")
    ds = to_eval_dataset(_fake_run(1, n_frames=1))
    buried = {s.gt_id for s in ds.buried_survivors()}
    assert buried, "at least one actor must be marked buried, or §2.7 has been lost"
    assert all(not s.findable for s in ds.buried_survivors())
    assert len(ds.findable_survivors()) == len(ds.survivors) - len(buried)
    for s in ds.survivors[:5]:
        assert 11.4 < s.lat < 11.6 and 76.0 < s.lon < 76.3   # the FloodValley scene's own geopoint


def test_urgency_mapping_matches_5_8():
    assert urgency_for("standing", "dry") == "stranded"
    assert urgency_for("standing", "half") == "immersed"
    assert urgency_for("half_submerged", "dry") == "immersed"
    assert urgency_for("trapped", "dry") == "trapped"
    assert urgency_for("standing", "dry", "animal") == "animal"


def test_summarise_boxes_reports_the_pixel_census():
    s = summarise_boxes([_fake_run(1, n_frames=4)])
    assert s["n_boxes"] == 4 and s["domain"] == "sim"
    assert s["median_px"] == pytest.approx(58.0)
    assert s["below_20_px"] == 0


def test_load_run_reads_the_real_capture():
    runs = _captured_runs()
    if not runs:
        pytest.skip("no captured run under _artifacts/dataset")
    run = load_run(runs[-1])
    assert run.frames
    f = run.frames[0]
    assert (f.width_px, f.height_px) == (3840, 2160)
    assert run.n_boxes == sum(len(x.labels) for x in run.frames)
    # telemetry.csv is written when a capture finishes, so an in-progress run has none yet. When it is there,
    # it must actually join onto the frames -- a silent join failure would leave every altitude slice empty.
    tel = runs[-1] / "telemetry.csv"
    if tel.exists() and tel.stat().st_size > 0:
        assert f.agl_m > 0 and f.gsd_cm_px > 0
        assert 11.0 < f.lat < 12.0 and 76.0 < f.lon < 77.0
        assert -95.0 <= f.gimbal_pitch_deg <= -85.0    # a nadir survey camera (§5.4)


# =============================================================================================================
# 4. The operating threshold (§5.5)
# =============================================================================================================
def _pr_frames(n_targets: int = 25):
    """A curve that falls by exactly 1/n per 0.01 of confidence: target k is found with score 1.00 - 0.01*k.

    Recall at confidence c = (number of targets whose score >= c) / n. With n = 25, recall >= 0.92 needs 23
    targets, i.e. scores 1.00 down to 0.78 -- so the highest confidence achieving it is 0.78. (Scores are
    1.00, 0.99, ... ; the 23rd is 1.00 - 0.22 = 0.78.)
    """
    frames = []
    for k in range(n_targets):
        box = (100.0, 100.0, 140.0, 200.0)
        frames.append(FrameEval(predictions=[det(*box, round(1.00 - 0.01 * k, 6))], ground_truth=[gt(*box)],
                                frame_idx=k))
    return frames


def test_sweep_picks_the_highest_confidence_above_target():
    frames = _pr_frames(25)
    op, _points, t = choose_operating_threshold(frames, fps_processed=5.0)
    assert t.n_gt == 25
    assert op.conf == pytest.approx(0.78)
    assert op.recall == pytest.approx(23 / 25)          # 0.92 exactly
    assert op.recall >= TARGET_RECALL and op.achieved
    # one notch higher must MISS the target -- otherwise 0.78 was not the highest
    higher = sweep(t, [0.79])[0]
    assert higher.recall < TARGET_RECALL
    assert op.precision == pytest.approx(1.0)           # every prediction is a true positive here


def test_recall_and_precision_at_a_threshold_are_hand_computable():
    """Four targets; predictions at IoU 1.00, 0.36, 0.10 and none, plus two boxes on nothing.

    At IoU 0.5 only the first matches -> tp 1, fn 3, fp 3 (the 0.36, the 0.10 and one pure FP... see below).
    """
    a = (0.0, 0.0, 100.0, 100.0)
    b = (200.0, 0.0, 300.0, 100.0)
    c = (400.0, 0.0, 500.0, 100.0)
    d = (600.0, 0.0, 700.0, 100.0)
    # IoU(a, a) = 1; shift b by 40 -> inter 60x100, union 140x100 -> 0.4286; shift c by 82 -> 18/182 = 0.0989
    preds = [det(*a, 0.9), det(240.0, 0.0, 340.0, 100.0, 0.8), det(482.0, 0.0, 582.0, 100.0, 0.7)]
    assert iou_xyxy(preds[1].bbox_px, b) == pytest.approx(60 / 140, abs=1e-9)
    assert iou_xyxy(preds[2].bbox_px, c) == pytest.approx(18 / 182, abs=1e-9)
    fr = FrameEval(predictions=preds, ground_truth=[gt(*a), gt(*b), gt(*c), gt(*d)])
    t = tally([fr], iou_thr=0.5)
    assert t.counts_at(0.0) == (1, 2, 3)      # tp 1, fp 2, fn 3
    t25 = tally([fr], iou_thr=0.25)
    assert t25.counts_at(0.0) == (2, 1, 2)    # the 0.4286 box now matches


def test_ignore_boxes_are_neither_a_miss_nor_a_false_positive():
    """§6.3: a detection overlapping an `ignore` region is dropped before scoring; missing it is not a miss."""
    box = (10.0, 10.0, 60.0, 110.0)
    fr = FrameEval(predictions=[det(*box, 0.9)], ground_truth=[gt(*box, ignore=True)])
    t = tally([fr])
    assert t.n_gt == 0 and t.pred_scores.size == 0
    assert t.counts_at(0.0) == (0, 0, 0)


def test_an_unreachable_target_is_reported_not_hidden():
    """Only 20 of 25 targets are findable at any confidence. The rule must NOT lower the bar."""
    frames = _pr_frames(25)
    for f in frames[20:]:
        f.predictions = []
    op, _points, _t = choose_operating_threshold(frames)
    assert op.achieved is False
    assert op.target_recall == TARGET_RECALL
    assert op.recall == pytest.approx(20 / 25)
    assert "no confidence reaches recall" in op.note


def test_max_f1_is_not_the_operating_point():
    """§5.5: 'Ultralytics' reported per-class P/R are taken at the max-F1 confidence, not at your operating
    point'. Constructed so the two genuinely disagree, and by how much.

    25 targets: 20 easy ones found at scores 1.00, 0.99 ... 0.81; 5 hard ones found only at 0.10 ... 0.06.
    Plus 10 false positives at 0.50 ... 0.41.

    * The §5.5 rule needs 23 of 25 targets, so it must drop to the 23rd-highest ground-truth score, 0.08 ->
      recall 0.92, precision 23/(23+10) = 0.697.
    * max-F1 sits at 0.81: tp 20, fp 0, precision 1.00, recall 0.80, F1 0.889 (against 0.794 at 0.08).

    Quoting the max-F1 row would advertise recall 0.80 as though it were the system's recall.
    """
    frames = []
    for k in range(20):
        box = (100.0, 100.0, 140.0, 200.0)
        frames.append(FrameEval(predictions=[det(*box, round(1.00 - 0.01 * k, 6))], ground_truth=[gt(*box)]))
    for k in range(5):
        box = (100.0, 100.0, 140.0, 200.0)
        frames.append(FrameEval(predictions=[det(*box, round(0.10 - 0.01 * k, 6))], ground_truth=[gt(*box)]))
    frames.append(FrameEval(
        predictions=[det(1000.0 + 50 * j, 1000.0, 1040.0 + 50 * j, 1100.0, round(0.50 - 0.01 * j, 6))
                     for j in range(10)],
        ground_truth=[]))

    op, points, t = choose_operating_threshold(frames)
    max_f1 = max(points, key=lambda p: (p.f1, p.conf))
    assert t.n_gt == 25
    assert op.conf == pytest.approx(0.08) and op.recall == pytest.approx(0.92)
    assert op.precision == pytest.approx(23 / 33)
    assert max_f1.conf == pytest.approx(0.81)
    assert max_f1.recall == pytest.approx(0.80) and max_f1.precision == pytest.approx(1.0)
    assert max_f1.conf != pytest.approx(op.conf)
    assert max_f1.recall < op.recall     # the library's number understates the recall the pipeline delivers
    assert max_f1.precision > op.precision  # ... and overstates its precision


def test_fp_per_minute_is_the_documented_ratio():
    """§5.12: FP/min = FP_total / (frames / fps / 60). 3 FPs over 10 frames at 5 fps = 90 FP/min."""
    assert fp_per_minute(3, 10, 5.0) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        fp_per_minute(3, 10, 0.0)


def test_greedy_matching_is_score_ordered():
    """Swapping two scores moves the true positive: the higher-scoring prediction takes the ground truth."""
    box = (0.0, 0.0, 100.0, 100.0)
    p1 = det(0.0, 0.0, 100.0, 100.0, 0.9)
    p2 = det(5.0, 5.0, 105.0, 105.0, 0.4)
    m = match_frame([p1, p2], [gt(*box)], 0.5)
    assert list(m.pred_is_tp) == [True, False]
    m2 = match_frame([det(*p1.bbox_px, 0.4), det(*p2.bbox_px, 0.9)], [gt(*box)], 0.5)
    assert list(m2.pred_is_tp) == [True, False]  # order is by score, so the 0.9 (second box) is visited first
    assert m2.pred_scores[0] == pytest.approx(0.9)


def test_metric_rows_all_carry_a_domain():
    from sightline.schemas import SliceKey

    op, _p, _t = choose_operating_threshold(_pr_frames(25), fps_processed=5.0)
    rows = metric_rows(op, SliceKey(domain="sim"))
    assert rows and all(r.slice.domain == "sim" for r in rows)
    assert {r.name for r in rows} >= {"recall@IoU0.5", "precision@IoU0.5", "fp_per_frame", "fp_per_minute"}
    assert "domain=sim" in str(rows[0])


def test_sliced_operating_points_partition_by_posture():
    a = (0.0, 0.0, 40.0, 100.0)
    b = (200.0, 0.0, 240.0, 100.0)
    frames = [
        FrameEval(predictions=[det(*a, 0.9)], ground_truth=[gt(*a, posture="standing"), gt(*b, posture="prone")],
                  altitude_band="45", time_of_day="day", zone="settlement"),
    ]
    rows = sliced_operating_points(frames, 0.5)
    by = {r.slice.label(): r for r in rows}
    assert by["domain=sim posture=standing"].value == pytest.approx(1.0)
    assert by["domain=sim posture=prone"].value == pytest.approx(0.0)
    assert all(r.slice.domain == "sim" for r in rows)


def test_freeze_and_load_roundtrip(tmp_path):
    op, points, _ = choose_operating_threshold(_pr_frames(25), fps_processed=5.0, model_version="test-v1")
    p = freeze_operating_point(op, tmp_path / "operating_point.json", points=points)
    back = load_operating_point(p)
    assert back.conf == pytest.approx(op.conf) and back.model_version == "test-v1"
    payload = json.loads(p.read_text())
    assert "highest confidence" in payload["rule"] and len(payload["sweep"]) == len(points)


# =============================================================================================================
# 5. Fusion (F9, §5.5) — the structural RGB-only fallback
# =============================================================================================================
def _bundle(thermal=None, thermal_intr=None) -> FrameBundle:
    intr = Intrinsics.from_hfov(3840, 2160, 74.0)
    return FrameBundle(frame_idx=0, t_utc=0.0, telemetry=Telemetry(t_utc=0.0, lat=11.487, lon=76.145,
                                                                   alt_msl_m=1100.0, agl_m=45.0),
                       intrinsics=intr, rgb=None, thermal=thermal, thermal_intrinsics=thermal_intr)


def test_rgb_only_fallback_returns_the_same_objects():
    """§5.5: 'the fallback is literally the RGB output'. Checked by OBJECT IDENTITY, not by value."""
    rgb = [det(0, 0, 40, 100, 0.8), det(500, 500, 540, 600, 0.6)]
    res = fuse_detections(rgb, None, frame_w=3840, frame_h=2160)
    assert res.is_rgb_only and res.used_thermal is False and res.reason == "no thermal frame"
    assert len(res.detections) == len(rgb)
    for a, b in zip(res.detections, rgb):
        assert a is b                      # the same objects
    assert [id(x) for x in res.detections] == [id(x) for x in rgb]   # in the same order


def test_rgb_only_fallback_when_thermal_produced_no_boxes():
    rgb = [det(0, 0, 40, 100, 0.8)]
    res = fuse_detections(rgb, [], frame_w=3840, frame_h=2160)
    assert res.detections[0] is rgb[0] and res.reason == "thermal produced no boxes"


def test_rgb_only_fallback_when_registration_is_rejected():
    rgb = [det(0, 0, 40, 100, 0.8)]
    bad = Registration(np.eye(3), 99.0, "altitude_table", False, "residual too large")
    res = fuse_detections(rgb, [det(1, 1, 41, 101, 0.7)], frame_w=3840, frame_h=2160, registration=bad)
    assert res.detections[0] is rgb[0] and "registration rejected" in res.reason


def test_fuse_frame_falls_back_byte_for_byte_without_a_thermal_partner():
    rgb = [det(10, 10, 50, 110, 0.77), det(900, 900, 940, 1000, 0.31)]
    res = fuse_frame(_bundle(thermal=None), rgb, None)
    assert res.is_rgb_only
    assert all(a is b for a, b in zip(res.detections, rgb))
    assert res.registration is None  # "there is no thermal frame" is a different state from "it is misaligned"


def test_fusion_with_thermal_keeps_a_single_modality_posterior():
    """ProbEn marginalisation: a box seen by only one modality keeps its own score (§5.5), unlike WBF's n/N."""
    rgb = [det(100.0, 100.0, 140.0, 200.0, 0.60)]
    th = [det(1000.0, 1000.0, 1040.0, 1100.0, 0.55)]  # nowhere near the RGB box
    res = fuse_detections(rgb, th, frame_w=3840, frame_h=2160, w_rgb=1.0, w_thermal=1.0)
    assert res.used_thermal
    scores = sorted(round(d.score, 6) for d in res.detections)
    assert scores == [0.55, 0.60]
    assert {d.modality for d in res.detections} == {"rgb", "thermal"}


def test_fusion_of_an_agreeing_pair_raises_the_score_above_both():
    rgb = [det(100.0, 100.0, 140.0, 200.0, 0.60)]
    th = [det(101.0, 101.0, 141.0, 201.0, 0.55)]
    res = fuse_detections(rgb, th, frame_w=3840, frame_h=2160)
    assert len(res.detections) == 1
    d = res.detections[0]
    assert d.modality == "fused" and d.score > 0.60
    # ProbEn: p = ab / (ab + (1-a)(1-b)) = .33 / (.33 + .18) = 0.647059
    assert d.score == pytest.approx(0.6 * 0.55 / (0.6 * 0.55 + 0.4 * 0.45), rel=1e-6)


def test_homography_from_intrinsics_is_exact_for_a_shared_pose():
    th = Intrinsics.from_hfov(640, 512, 61.0)
    rgb = Intrinsics.from_hfov(3840, 2160, 84.0)
    H = homography_from_intrinsics(th, rgb)
    # the thermal principal point must map to the RGB principal point exactly
    got = warp_boxes(H, [(th.cx, th.cy, th.cx + 1, th.cy + 1)])[0]
    assert got[0] == pytest.approx(rgb.cx) and got[1] == pytest.approx(rgb.cy)
    assert H[0, 0] == pytest.approx(rgb.fx / th.fx) and H[2, 2] == 1.0


def test_thermal_weight_follows_the_documented_schedule():
    """§5.5: lowered at midday and in the crossover windows, raised at night."""
    assert thermal_weight("night") > thermal_weight("day") > thermal_weight("midday")
    assert thermal_weight("dawn") == thermal_weight("dusk") < thermal_weight("day")
    # §5.5b: radiometry replaces the heuristic with a measurement
    assert thermal_weight(radiometric_contrast_c=0.2) < thermal_weight(radiometric_contrast_c=8.0)
    assert thermal_weight(radiometric_contrast_c=8.0) == pytest.approx(1.8)


# =============================================================================================================
# 6. The verifier and the §5.5a safety rule (F10)
# =============================================================================================================
def test_posture_can_never_demote():
    """§5.5a rule 2, over the FULL cross-product of postures x submersions x confidences."""
    base = det(100, 100, 140, 200, 0.5)
    for p in POSTURES:
        for s in SUBMERSIONS:
            for conf in (0.0, 0.3, 0.49, 0.5, 0.51, 0.99, 1.0):
                out = VerifierOutput(is_real=0.9, posture=p, posture_conf=conf, submersion=s,
                                     submersion_conf=conf)
                dec = apply_verifier_output(base, out)
                assert dec.demoted is False, f"{p}/{s}@{conf} demoted the record"
                assert URGENCY_RANK[dec.final_urgency] >= URGENCY_RANK[BASE_URGENCY]


def test_a_low_confidence_prediction_is_treated_as_unknown_at_the_base_weight():
    """§5.5a rule 1: 'the unknown case is never the cheapest case'."""
    base = det(100, 100, 140, 200, 0.5)
    out = VerifierOutput(is_real=0.8, posture="prone", posture_conf=0.2, submersion="dry", submersion_conf=0.2)
    dec = apply_verifier_output(base, out, min_conf=0.5)
    assert dec.detection.posture == "unknown" and dec.detection.posture_conf == 0.0
    assert dec.final_urgency == BASE_URGENCY
    assert "below min_conf" in dec.rejected_reason
    assert dec.output.posture == "prone"  # rule 3: the rejected prediction is still available to show


def test_a_confident_immersed_prediction_promotes():
    base = det(100, 100, 140, 200, 0.5)
    out = VerifierOutput(is_real=0.9, posture="half_submerged", posture_conf=0.61,
                         submersion="head_only", submersion_conf=0.7)
    dec = apply_verifier_output(base, out)
    assert dec.promoted and dec.final_urgency == "immersed"
    assert dec.detection.posture == "half_submerged" and dec.detection.posture_conf == pytest.approx(0.61)
    assert dec.detection.submersion == "head_only"
    assert dec.detection.is_real == pytest.approx(0.9)


def test_urgency_of_never_falls_below_the_base():
    for p in POSTURES:
        for s in SUBMERSIONS:
            assert URGENCY_RANK[urgency_of(p, s)] >= URGENCY_RANK[BASE_URGENCY]
    assert urgency_of("half_submerged", "dry") == "immersed"
    assert urgency_of("standing", "half") == "immersed"
    assert urgency_of("trapped", "dry") == "trapped"
    assert urgency_of("standing", "dry") == "stranded"


def test_verifier_output_rejects_an_out_of_vocabulary_answer():
    with pytest.raises(ValueError):
        VerifierOutput(is_real=0.5, posture="swimming")
    with pytest.raises(ValueError):
        VerifierOutput(is_real=0.5, submersion="soaked")
    with pytest.raises(ValueError):
        VerifierOutput(is_real=1.5)
    with pytest.raises(ValueError):
        VerifierOutput(is_real=0.5, occlusion=7)
    assert all(VerifierOutput(is_real=0.5, occlusion=o) for o in OCCLUSION_BINS)


def test_apply_batch_checks_the_lengths():
    with pytest.raises(ValueError):
        apply_batch([det(0, 0, 1, 1)], [])


def test_suppression_returns_both_lists_and_keeps_the_unverified():
    a = det(0, 0, 40, 100, 0.9)
    b = det(100, 0, 140, 100, 0.9)
    c = det(200, 0, 240, 100, 0.9)
    a.is_real, b.is_real, c.is_real = 0.9, 0.1, None
    kept, dropped = suppress_false_positives([a, b, c], min_is_real=0.35)
    assert kept == [a, c] and dropped == [b]   # None is kept: absence of evidence is not evidence


def test_crop_for_pads_and_clamps_at_the_frame_edge():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    d = det(10.0, 5.0, 38.0, 63.0)
    crop, rect = crop_for(frame, d, pad_frac=0.4)
    assert rect[0] == 0 and rect[1] == 0          # clamped, not negative
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    mid = det(1900.0, 1000.0, 1928.0, 1058.0)
    crop2, r2 = crop_for(frame, mid, pad_frac=0.4)
    # half-side = 58 * (0.5 + 0.4) = 52.2 -> a 105 px window around the centre
    assert crop2.shape[0] == r2[3] - r2[1] and crop2.shape[1] == r2[2] - r2[0]
    assert 100 <= crop2.shape[0] <= 110


def test_crop_verifier_refuses_to_load_without_trained_heads(tmp_path):
    from sightline.detect.verifier import CropVerifier

    with pytest.raises(FileNotFoundError, match="not trained yet"):
        CropVerifier(tmp_path / "no_such_heads.pt")


# =============================================================================================================
# 7. Overlay — the visual-proof tool itself must be right
# =============================================================================================================
def test_overlay_stats_are_hand_computable():
    frame = np.zeros((600, 800, 3), np.uint8)
    a = (100.0, 100.0, 140.0, 200.0)
    b = (300.0, 100.0, 340.0, 200.0)
    preds = [det(*a, 0.9), det(500.0, 300.0, 540.0, 400.0, 0.7), det(600.0, 400.0, 640.0, 500.0, 0.1)]
    img, st = overlay_frame(frame, preds, [gt(*a), gt(*b)], conf=0.25, frame_name="f")
    assert (st.n_gt, st.n_pred_total, st.n_pred_kept) == (2, 3, 2)
    assert (st.tp, st.fp, st.fn) == (1, 1, 1)
    assert "gt=2 pred=2/3" in st.caption()
    assert img.ndim == 3 and img.shape[0] > frame.shape[0] * 0.2


def test_overlay_with_no_ground_truth_reports_every_prediction_as_unmatched():
    frame = np.zeros((400, 400, 3), np.uint8)
    img, st = overlay_frame(frame, [det(10, 10, 50, 110, 0.9)], [], conf=0.5)
    assert (st.tp, st.fp, st.fn) == (0, 1, 0)
    assert img is not None


def test_zoom_panel_magnifies_a_tiny_target():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    frame[900:958, 1500:1528] = 200
    strip = zoom_panel(frame, [((1500.0, 900.0, 1528.0, 958.0), (0, 0, 255), "28x58")], out_px=220)
    assert strip is not None and strip.shape[0] == 242 and strip.shape[1] == 220
    assert strip.max() > 100  # the bright target actually survived into the panel
    assert zoom_panel(frame, []) is None


def test_prediction_json_roundtrips(tmp_path):
    by = {"f0": [det(1.5, 2.5, 41.5, 102.5, 0.42, tile_idx=3)]}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(detections_to_json(by, {"weights": "x.pt"})), encoding="utf-8")
    back, meta = detections_from_json(p)
    assert meta["weights"] == "x.pt"
    assert back["f0"][0].bbox_px == (1.5, 2.5, 41.5, 102.5)
    assert back["f0"][0].tile_idx == 3 and back["f0"][0].score == pytest.approx(0.42)


# =============================================================================================================
# 8. Evaluation wiring (F19) — the slice table, every row domain=sim
# =============================================================================================================
def _run_with_scores(seed: int, scores, *, n_frames=6, hit=True) -> tuple[CaptureRun, dict]:
    """A run with one 40x100 target per frame, and a prediction on it with the given score."""
    run = _fake_run(seed, n_frames=n_frames)
    preds: dict[str, list[Detection]] = {}
    for i, fr in enumerate(run.frames):
        fr.labels = [CaptureLabel(bbox_px=(500.0, 400.0, 540.0, 500.0), actor_id=1, size_px=100,
                                  pose="standing", submersion="dry", occlusion=0, zone="settlement")]
        fr.agl_m = 45.0
        fr.time_of_day = "day"
        s = scores[i % len(scores)]
        box = (500.0, 400.0, 540.0, 500.0) if hit else (2000.0, 400.0, 2040.0, 500.0)
        preds[fr.stem] = [det(*box, s)]
    return run, preds


def test_run_evaluation_produces_a_sim_only_slice_table(tmp_path):
    from sightline.detect.evaluate import bundle_from_runs, run_evaluation

    val, pv = _run_with_scores(11, [0.95, 0.90, 0.85, 0.80, 0.75, 0.70])
    test, pt = _run_with_scores(23, [0.93, 0.88, 0.84, 0.81, 0.79, 0.72])
    preds = {**pv, **pt}
    bundle = bundle_from_runs([val], [test], preds)
    summary = run_evaluation(bundle, out_dir=tmp_path)
    res = summary.pop("_result")

    assert summary["domain"] == "sim"
    assert summary["seeds"] == {"val": [11], "test": [23]}
    assert summary["frozen_on"]["split"] == "val" and summary["measured_on"]["split"] == "test"
    # 6 val targets, all found. The highest confidence keeping recall >= 0.92 keeps all six -> 0.70.
    assert summary["operating_conf"] == pytest.approx(0.70)
    assert summary["operating_recall"] == pytest.approx(1.0)
    assert summary["target_met"] is True
    assert all(r.slice.domain == "sim" for r in res.metrics.rows)
    assert all(math.isfinite(r.value) for r in res.metrics.rows)

    table = (tmp_path / "slice_table.md").read_text(encoding="utf-8")
    assert "| domain | metric | slice | value | n |" in table
    for line in table.splitlines():
        if line.startswith("| ") and "domain |" not in line and "---" not in line:
            assert line.startswith("| sim |"), line
    assert "in simulation" in table.lower()
    assert (tmp_path / "detect_eval.md").exists()


def test_run_evaluation_reports_an_unmet_target_as_a_finding(tmp_path):
    from sightline.detect.evaluate import bundle_from_runs, run_evaluation

    val, pv = _run_with_scores(11, [0.9] * 6)
    for k in list(pv)[:3]:
        pv[k] = []           # only 3 of 6 val targets are findable at any threshold -> recall 0.5
    test, pt = _run_with_scores(23, [0.9] * 6)
    summary = run_evaluation(bundle_from_runs([val], [test], {**pv, **pt}), out_dir=tmp_path)
    summary.pop("_result")
    assert summary["target_met"] is False
    assert summary["operating_recall"] == pytest.approx(0.5)
    assert "fly lower" in summary["finding"].lower()
    assert summary["target_recall"] == pytest.approx(0.92)   # the bar was NOT moved


def test_bundle_refuses_a_seed_in_both_val_and_test():
    from sightline.detect.evaluate import bundle_from_runs

    run, preds = _run_with_scores(7, [0.9])
    with pytest.raises(ValueError, match="share scenario seed"):
        bundle_from_runs([run], [run], preds)


def test_submersion_rows_partition_the_boxes(tmp_path):
    from sightline.detect.evaluate import bundle_from_runs, run_evaluation

    val, pv = _run_with_scores(11, [0.9] * 4, n_frames=4)
    test, pt = _run_with_scores(23, [0.9] * 4, n_frames=4)
    for i, fr in enumerate(test.frames):
        fr.labels[0] = CaptureLabel(bbox_px=fr.labels[0].bbox_px, actor_id=1, pose="standing",
                                    submersion=("half" if i < 2 else "dry"), occlusion=0, zone="settlement")
    summary = run_evaluation(bundle_from_runs([val], [test], {**pv, **pt}), out_dir=tmp_path)
    res = summary.pop("_result")
    rows = {r.name: r for r in res.metrics.rows if r.name.startswith("recall@submersion")}
    assert set(rows) == {"recall@submersion=half", "recall@submersion=dry"}
    assert sum(r.n for r in rows.values()) == 4
    assert all(r.slice.domain == "sim" for r in rows.values())


def test_evaluation_refuses_to_freeze_on_a_non_val_split():
    from sightline.eval.groundtruth import EvalDataset
    from sightline.eval.harness import freeze_on_validation

    ds = EvalDataset(domain="sim", split="test")
    with pytest.raises(ValueError, match="validation split"):
        freeze_on_validation(ds, [])


# =============================================================================================================
# 9. Training / export guards — these must FAIL when the machine is not ready
# =============================================================================================================
def test_train_args_are_the_verified_8gb_settings():
    from sightline.detect.train import BASE_ARGS, TrainConfig, build_train_args

    cfg = TrainConfig(data_yaml="D:/x/data.yaml")
    a = build_train_args(cfg)
    assert a["imgsz"] == 1024 and a["batch"] == -1 and a["amp"] is True
    assert a["cache"] == "disk" and a["workers"] <= 4
    assert 30 <= a["epochs"] <= 50                       # §5.5c
    assert a["hsv_h"] == a["hsv_s"] == a["hsv_v"] == 0.0  # randomisation OFF (§5.5c)
    assert a["flipud"] == 0.5 and a["degrees"] == 180.0   # nadir has no canonical up
    assert str(a["project"]).lower().startswith("d:")
    assert BASE_ARGS["pretrained"] is True                # COCO weights are the starting point


def test_train_args_reject_ram_cache_and_too_many_workers():
    from sightline.detect.train import TrainConfig, build_train_args

    with pytest.raises(ValueError, match="cache='ram'"):
        build_train_args(TrainConfig(data_yaml="D:/x.yaml", overrides={"cache": "ram"}))
    with pytest.raises(ValueError, match="workers"):
        build_train_args(TrainConfig(data_yaml="D:/x.yaml", overrides={"workers": 8}))
    with pytest.raises(ValueError, match="must be on D:"):
        build_train_args(TrainConfig(data_yaml="D:/x.yaml", project="C:/models"))


def test_preflight_refuses_a_missing_dataset():
    from sightline.detect.train import TrainConfig, preflight

    with pytest.raises(FileNotFoundError):
        preflight(TrainConfig(data_yaml="D:/definitely/not/here/data.yaml"))


def test_preflight_refuses_a_leaked_split(tmp_path):
    from sightline.detect.train import TrainConfig, preflight

    (tmp_path / "data.yaml").write_text("names:\n  0: human\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"splits": {"train": {"seeds": [1, 2]}, "val": {"seeds": [2]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="share scenario seed"):
        preflight(TrainConfig(data_yaml=str(tmp_path / "data.yaml"), allow_unvalidated=True))


def test_preflight_refuses_a_dataset_with_no_val_split(tmp_path):
    from sightline.detect.train import TrainConfig, preflight

    (tmp_path / "data.yaml").write_text("names:\n  0: human\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"splits": {"train": {"seeds": [1]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no val split"):
        preflight(TrainConfig(data_yaml=str(tmp_path / "data.yaml"), allow_unvalidated=True))


def test_dataset_is_validated_refuses_an_empty_run_list():
    from sightline.detect.train import dataset_is_validated

    ok, why = dataset_is_validated([])
    assert ok is False and "came from nowhere" in why


def test_export_int8_requires_a_calibration_set():
    from sightline.detect.export import ExportConfig

    with pytest.raises(ValueError, match="tiny/occluded positives"):
        ExportConfig(weights="x.pt", int8=True).args()
    a = ExportConfig(weights="x.pt", int8=True, data="D:/x/data.yaml").args()
    assert a["int8"] is True and a["half"] is False


def test_int8_calibration_refuses_a_set_without_tiny_positives(tmp_path):
    from sightline.detect.export import int8_calibration_manifest

    lab = tmp_path / "labels" / "train"
    lab.mkdir(parents=True)
    for i in range(40):  # every box 200 px on a 1024 tile: far above the "tiny" bar
        (lab / f"t{i}.txt").write_text("0 0.5 0.5 0.1953125 0.1953125\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tiny/occluded positives"):
        int8_calibration_manifest(tmp_path)
    for i in range(40, 80):  # 20 px boxes
        (lab / f"t{i}.txt").write_text("0 0.5 0.5 0.01953125 0.01953125\n", encoding="utf-8")
    man = int8_calibration_manifest(tmp_path)
    assert man["fraction_tiny"] >= 0.2 and man["domain"] == "sim"


def test_compare_recall_refuses_to_mix_domains():
    from sightline.detect.export import compare_recall
    from sightline.eval.slicing import DomainMixError, make_slice, metric_row

    a = [metric_row("recall@iou0.5", 0.94, make_slice("sim"), 100)]
    b = [metric_row("recall@iou0.5", 0.91, make_slice("real"), 100)]
    with pytest.raises(DomainMixError):
        compare_recall(a, b)
    c = [metric_row("recall@iou0.5", 0.938, make_slice("sim"), 100)]
    ok = compare_recall(a, c)
    assert ok["acceptable"] is True and ok["domain"] == "sim"
    d = [metric_row("recall@iou0.5", 0.90, make_slice("sim"), 100)]
    bad = compare_recall(a, d)
    assert bad["acceptable"] is False and "DO NOT SHIP" in bad["verdict"]


def test_importing_the_lane_does_not_import_torch():
    """CONTRACTS §3.2: nothing but the ML lane's own model classes may pull torch in."""
    import subprocess
    import sys as _sys

    code = ("import sightline.detect, sightline.detect.train, sightline.detect.export, "
            "sightline.detect.overlay, sightline.detect.evaluate, sys; "
            "print('torch' in sys.modules or 'ultralytics' in sys.modules)")
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO),
                       check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", r.stdout
