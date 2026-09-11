"""An `ignore` box must never become a training target and must never be scored as a miss.

This project has already shipped one gate that read as protection and did nothing: `validate.py` printed
"all checks passed - dataset is clean" and exited 0 while silently skipping the check that mattered. Marking
18 boxes `ignore` is worth exactly nothing unless both consumers act on it, so both are pinned here.

The boxes in question sit on fern canopy and concrete slabs with no subject visible in RGB at all. The
instance mask reported the survivors underneath as fully visible because Cosys-AirSim disables the
InstancedFoliage and InstancedGrass show flags in its annotation render, and every plant and slab in this
scene is an instanced component. Emitting one as a YOLO target teaches the detector to find people in leaf
texture; counting one as a miss understates recall against survivors no camera could have seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sightline.detect.dataset import CaptureLabel, Tile, tile_boxes  # noqa: E402

def _tile() -> Tile:
    return Tile(idx=0, row=0, col=0, x0=0, y0=0, x1=1024, y1=1024, frame_w=3840, frame_h=2160)


def _label(**kw) -> CaptureLabel:
    base = dict(bbox_px=(400.0, 400.0, 460.0, 500.0), cls="human", actor_id=53,
                name="Human_053", size_px=100.0, pose="prone", visible_px=2900)
    base.update(kw)
    return CaptureLabel(**base)


def test_an_ignored_box_is_not_emitted_as_a_training_target() -> None:
    keep, dropped = tile_boxes([_label(ignore=True)], _tile())
    assert keep == [], "a box with no visible subject must not become a YOLO target"
    assert len(dropped) == 1, "it must be accounted for as dropped, not vanish silently"


def test_the_same_box_without_the_flag_is_emitted() -> None:
    """Guards against the test above passing for the wrong reason (e.g. the box missing the tile)."""
    keep, _ = tile_boxes([_label(ignore=False)], _tile())
    assert len(keep) == 1, "the flag must be what excludes it, not the geometry"


def test_ignore_defaults_to_false_so_untouched_datasets_are_unaffected() -> None:
    assert _label().ignore is False


def test_an_ignored_box_is_uncertain_ground_truth_not_a_recall_target() -> None:
    """section 6.3: an unresolvable instance is neither a true positive nor a miss."""
    from sightline.eval.groundtruth import GtBox

    gt = GtBox(bbox_px=(400.0, 400.0, 460.0, 500.0), cls="human", ignore=True)
    assert not gt.scored, "an ignored survivor must not be counted as missed"
    assert GtBox(bbox_px=(400.0, 400.0, 460.0, 500.0), cls="human").scored,         "and an ordinary box must still be scored, or the assertion above proves nothing"


def test_the_flag_survives_a_json_round_trip() -> None:
    import json

    assert json.loads(json.dumps({"ignore": True}))["ignore"] is True
