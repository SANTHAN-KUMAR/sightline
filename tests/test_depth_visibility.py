"""The depth gate decides which survivors are real, so its behaviour is pinned here.

Cosys-AirSim renders the instance mask with `SetInstancedFoliage(false)`, so every plant in this scene - all
of them instances on a HISM, forced by the commit limit - is invisible to the mask. A survivor under a fern
therefore appears in the mask whole and unoccluded. `apply_depth_visibility` is what re-imposes physical
visibility, using the depth buffer, which does render the canopy.

Getting this wrong in either direction is a serious defect: too strict and real hard cases are deleted from
the training set, too loose and the detector is taught to find people in leaf texture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.capture.labels import MaskLabel, apply_depth_visibility  # noqa: E402

RGB = (200, 40, 90)
CAM_ALT = 1100.0
BASE_ASL = 1062.0
BODY_CM = 180.0
D_GROUND = CAM_ALT - BASE_ASL          # 38.0 m of planar depth to the ground the survivor lies on
CROWN_M = 8.0                          # a fern/tree crown standing 8 m above them


def truth(occlusion: int = 0) -> dict:
    return {"actors": [{"id": 7, "name": "Human_007", "base_asl_m": BASE_ASL, "height_cm": BODY_CM,
                        "occlusion": occlusion}]}


def scene(vis_cols: slice) -> tuple[np.ndarray, np.ndarray, MaskLabel]:
    """A 20x20 frame with a 10x10 actor blob; columns in `vis_cols` are at body depth, the rest under crown."""
    mask = np.zeros((20, 20, 3), np.uint8)
    mask[5:15, 5:15] = RGB
    depth = np.full((20, 20), D_GROUND, np.float32)
    depth[5:15, 5:15] = D_GROUND - CROWN_M          # start fully occluded
    depth[5:15, vis_cols] = D_GROUND - 0.5          # then expose some of the body
    lab = MaskLabel(actor_id=7, name="Human_007", cls="human", bbox_px=(5, 5, 14, 14),
                    visible_px=100, rgb=RGB)
    return mask, depth, lab


def test_an_unoccluded_survivor_is_kept_whole() -> None:
    mask, depth, lab = scene(slice(5, 15))
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not occ and len(vis) == 1
    m = vis[0]
    assert m.bbox_px == (5, 5, 14, 14)
    assert m.visible_px == 100
    assert m.visible_fraction == pytest.approx(1.0)
    assert m.occlusion == 0


def test_a_survivor_entirely_under_canopy_is_dropped_not_labelled() -> None:
    """The defect this whole gate exists for: 7 of 110 boxes in seed23_alt35 sat on pure leaf texture."""
    mask, depth, lab = scene(slice(0, 0))
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not vis and len(occ) == 1
    assert occ[0].visible_px == 0
    assert occ[0].visible_fraction == pytest.approx(0.0)


def test_a_half_covered_survivor_keeps_only_the_visible_extent() -> None:
    mask, depth, lab = scene(slice(5, 10))          # left half of the body exposed
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not occ and len(vis) == 1
    m = vis[0]
    assert m.bbox_px == (5, 5, 9, 14), "the box must shrink to the pixels the camera can actually see"
    assert m.amodal_bbox_px == (5, 5, 14, 14), "the full-body extent is preserved, not discarded"
    assert m.visible_fraction == pytest.approx(0.5)
    assert m.occlusion == 1


def test_occlusion_levels_follow_the_section_63_scale() -> None:
    for cols, want in ((slice(5, 15), 0), (slice(5, 10), 1), (slice(5, 7), 2)):
        mask, depth, lab = scene(cols)
        vis, _ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
        assert vis[0].occlusion == want, f"{cols} -> {vis[0].visible_fraction:.2f}"


def test_the_body_is_not_clipped_by_its_own_height() -> None:
    """A standing person spans 1.8 m of depth. Every pixel of them must pass, head and feet alike."""
    mask, depth, lab = scene(slice(5, 15))
    depth[5:15, 5:10] = D_GROUND - BODY_CM / 100.0 + 0.01     # head end, nearly 1.8 m above the ground
    depth[5:15, 10:15] = D_GROUND - 0.05                      # feet, essentially on the ground
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not occ
    assert vis[0].visible_fraction == pytest.approx(1.0)


def test_a_crown_just_above_the_body_still_occludes() -> None:
    """The gate must not be so slack that a low occluder passes. 2.5 m clears the 1.8 m body plus slack."""
    mask, depth, lab = scene(slice(0, 0))
    depth[5:15, 5:15] = D_GROUND - (BODY_CM / 100.0 + 1.0 + 0.2)
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not vis and len(occ) == 1


def test_infinite_depth_is_not_counted_as_visible() -> None:
    mask, depth, lab = scene(slice(0, 0))
    depth[5:15, 5:15] = np.inf
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not vis and len(occ) == 1


def test_a_mask_actor_missing_from_the_ground_truth_is_an_error() -> None:
    """Silently skipping would leave an ungated box in the frame - exactly the bug being fixed."""
    mask, depth, lab = scene(slice(5, 15))
    with pytest.raises(KeyError):
        apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT,
                               actors_json={"actors": [{"id": 99, "base_asl_m": 0, "height_cm": 1}]})


def test_a_depth_frame_of_the_wrong_size_is_an_error() -> None:
    mask, depth, lab = scene(slice(5, 15))
    with pytest.raises(ValueError):
        apply_depth_visibility([lab], mask, depth[:10], cam_alt_asl_m=CAM_ALT, actors_json=truth())


def test_a_body_almost_entirely_hidden_is_dropped_even_with_enough_pixels() -> None:
    """A pixel floor alone is not enough, and this case was measured, not imagined.

    On seed47_alt55 a survivor under a jacaranda leaked pixels through the gaps between fronds. That cleared
    the old `min_px = 4` floor and kept a 22 px box open over solid foliage with no subject visible in RGB at
    all. So the gate needs BOTH floors, and this isolates the fraction one: enough absolute pixels to pass
    `MIN_VISIBLE_PX`, but far too small a share of the body to be a view of anybody.
    """
    big = np.zeros((60, 60, 3), np.uint8)
    big[10:40, 10:40] = RGB                       # a 900 px body
    d = np.full((60, 60), D_GROUND, np.float32)
    d[10:40, 10:40] = D_GROUND - CROWN_M          # all of it under the crown ...
    d[10:14, 10:15] = D_GROUND - 0.5              # ... except 20 px: over MIN_VISIBLE_PX, 2.2 % of the body
    lab = MaskLabel(actor_id=7, name="Human_007", cls="human", bbox_px=(10, 10, 39, 39),
                    visible_px=900, rgb=RGB)
    vis, occ = apply_depth_visibility([lab], big, d, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert not vis and len(occ) == 1, "20 visible px of 900 is 2.2 % - the fraction floor must drop it"
    assert occ[0].visible_fraction == pytest.approx(20 / 900, abs=1e-3)


def test_the_floors_do_not_delete_a_genuine_partial_view() -> None:
    """Guard the other direction: section 6.3 wants hard partial cases KEPT, not tidied away."""
    mask, depth, lab = scene(slice(5, 10))        # half the body visible
    vis, occ = apply_depth_visibility([lab], mask, depth, cam_alt_asl_m=CAM_ALT, actors_json=truth())
    assert vis and not occ
    assert vis[0].visible_fraction == pytest.approx(0.5)
