"""The §6.3 size and truncation flags must be set the way the guideline defines them.

This matters because `sightline/eval/groundtruth.py:GtBox.is_recall_target` returns False for an `uncertain`
or `ignore` box. Until these flags were populated, a 5 px smudge was scored as a **missed survivor** rather
than as unresolvable - understating recall on exactly the slice where §6.3 says judgement should be withheld.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.capture.enrich_labels import IGNORE_BELOW_PX, UNCERTAIN_BELOW_PX, enrich  # noqa: E402

W, H = 3840, 2160


def box(x1: int, y1: int, x2: int, y2: int) -> dict:
    return {"bbox_px": [x1, y1, x2, y2], "size_px": max(x2 - x1 + 1, y2 - y1 + 1)}


def test_a_normal_box_gets_no_flags() -> None:
    f = enrich(box(1000, 800, 1060, 900), W, H)
    assert not f["ignore"] and not f["uncertain"] and not f["truncated"]


def test_a_box_under_four_px_is_ignored_not_scored() -> None:
    """§6.3's own pre-export sanity check is 'no box < 4 px'."""
    f = enrich(box(1000, 800, 1002, 802), W, H)      # 3 px
    assert f["ignore"] and not f["uncertain"]


def test_four_to_eight_px_is_uncertain_rather_than_dropped_or_scored() -> None:
    """§6.3: 'anything 4-8 px gets uncertain = 1 rather than being skipped'."""
    for size in range(IGNORE_BELOW_PX, UNCERTAIN_BELOW_PX):
        f = enrich(box(1000, 800, 1000 + size - 1, 800 + size - 1), W, H)
        assert f["uncertain"], f"{size} px should be uncertain"
        assert not f["ignore"], f"{size} px should not be ignored"


def test_eight_px_is_a_real_target() -> None:
    f = enrich(box(1000, 800, 1007, 807), W, H)      # exactly 8 px
    assert not f["uncertain"] and not f["ignore"]


def test_every_border_is_detected() -> None:
    assert enrich(box(0, 800, 60, 900), W, H)["truncated"]          # left
    assert enrich(box(1000, 0, 1060, 90), W, H)["truncated"]        # top
    assert enrich(box(W - 1, 800, W - 1, 900), W, H)["truncated"]   # right
    assert enrich(box(1000, H - 40, 1060, H - 1), W, H)["truncated"]  # bottom
    assert not enrich(box(1, 1, 60, 90), W, H)["truncated"]         # one pixel clear of it


def test_truncated_does_not_claim_to_be_the_fifty_percent_rule() -> None:
    """Honesty check: the field says what it actually measured.

    §6.3 defines truncation as '>= 50 % of the visible body cut', which needs an amodal extent this pipeline
    does not have per frame. Claiming the flag means that would be a lie a downstream reader could not spot.
    """
    f = enrich(box(0, 800, 60, 900), W, H)
    assert "border contact" in f["truncated_basis"]
    assert "50%" in f["truncated_basis"] or "50 %" in f["truncated_basis"]


def test_flags_are_plain_bools_so_json_round_trips() -> None:
    import json

    f = enrich(box(0, 0, 2, 2), W, H)
    back = json.loads(json.dumps(f))
    assert back["ignore"] is True and back["truncated"] is True


def test_an_existing_ignore_is_never_cleared_by_the_size_test() -> None:
    """This function knows ONE reason to ignore a box. It must not overrule the others.

    `flag_occluded_boxes.py` sets `ignore` on boxes drawn over fern canopy and concrete slabs where no
    subject is visible in RGB at all - the instance mask cannot see instanced foliage or rubble. Those boxes
    are full-size, so a size test says "keep", and `L.update(flags)` then cleared all 18 of them on the next
    gate run and reported a clean dataset. The flag is a union of reasons, not this function's opinion.
    """
    big = box(1000, 800, 1100, 900)
    big["ignore"] = True
    assert enrich(big, W, H)["ignore"] is True

    plain = box(1000, 800, 1100, 900)
    assert enrich(plain, W, H)["ignore"] is False, "and it must still be False when nobody set it"
