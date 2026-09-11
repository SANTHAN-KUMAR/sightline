"""Apply the manual visibility QA to the two completed passes: section 6.3 `ignore` on unseeable survivors.

    uv run python tools/capture/flag_occluded_boxes.py                  # report + regenerate the evidence
    uv run python tools/capture/flag_occluded_boxes.py --write

WHAT THIS IS, STATED PLAINLY. Cosys-AirSim renders the instance mask with the InstancedFoliage and
InstancedGrass show flags off (`Source/Annotation/ObjectAnnotator.cpp:SetViewForAnnotationRender`). Every
plant in this scene is a HISM instance and every rubble slab an ISM instance, so the mask sees through all of
them: a survivor under a fern, or under a concrete slab, is reported whole and unoccluded, and the
auto-labeller writes a confident box over scenery with nobody visible in it.

`tools/capture/labels.py:apply_depth_visibility` fixes this AT SOURCE using the depth buffer, and every
capture from now on is gated by it. It cannot fix frames already on disk, because those frames were taken
without the depth channel and the exact shutter pose is not recoverable: `survey.py` samples the kinematics
at the top of its loop and only calls `grab()` afterwards, so the recorded pose is a few hundred milliseconds
- up to ~0.3 m at 11 m/s - ahead of the shutter. Attempting to re-park the camera on a recorded pose
reproduced the stored silhouette to IoU 0.40-0.83, which is not good enough to rewrite a box from and was
therefore abandoned rather than fudged.

So the 33 boxes whose silhouette contrast fell below dE 8 were reviewed BY EYE against the actual frames -
`_artifacts/dataset/<run>/visibility_worst.png`, three panels each, RGB next to mask - and the verdict for
every one of them is recorded in VERDICTS below. That is not a proxy for looking: it IS looking, which is
what `docs/QUALITY_GATE.md` requires and what caught all seven of session 3's defects. The dE number beside
each is the measurement that selected it for review, not the thing that decided it.

`ignore` is section 6.3's own answer for an unresolvable instance - neither a recall target nor a false
positive. Nothing is deleted (guardrail R10): the box, its class and its truth join all remain, and
`sightline/eval/groundtruth.py` already declines to score an ignored box while
`sightline/detect/dataset.py` declines to emit one as a training target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: (run, frame index, actor id) -> (dE measured, what the eye saw)
#: HIDDEN entries get `ignore`. The VISIBLE ones are listed too, deliberately: they are the boxes the same
#: measurement flagged and the eye cleared, and recording them is what stops a future reader assuming a low
#: dE alone condemns a box. Several are exactly the hard low-contrast cases the dataset exists to contain.
VERDICTS: dict[tuple[str, int, int], tuple[float, str, str]] = {
    # ---- seed23_alt35 -------------------------------------------------------------------------------
    ("seed23_alt35", 175, 20): (1.04, "hidden", "solid green vegetation; no subject anywhere in the box"),
    ("seed23_alt35", 341, 53): (1.36, "hidden", "dense fern canopy; complete crisp silhouette in the mask"),
    ("seed23_alt35",  43, 49): (2.12, "hidden", "fern fronds only"),
    ("seed23_alt35", 327, 44): (3.08, "visible", "head and teal top clearly visible at the water line"),
    ("seed23_alt35",  41, 49): (3.11, "hidden", "fern and a branch; no body"),
    ("seed23_alt35",  42, 49): (4.47, "hidden", "fern only"),
    ("seed23_alt35", 174, 20): (4.74, "hidden", "green vegetation; a single orange speck, not a person"),
    ("seed23_alt35", 342, 53): (4.85, "hidden", "fern only"),
    ("seed23_alt35", 101, 12): (5.10, "visible", "person on pale ground, low contrast but plainly there"),
    ("seed23_alt35", 101, 13): (6.73, "visible", "person visible"),
    ("seed23_alt35", 326, 44): (7.62, "visible", "person visible"),
    # ---- seed23_alt55 -------------------------------------------------------------------------------
    ("seed23_alt55", 171, 55): (0.44, "hidden", "concrete slab; RUBBLE, not foliage - same ISM blindness"),
    ("seed23_alt55",  41, 49): (1.26, "hidden", "fern canopy"),
    ("seed23_alt55", 118, 20): (1.41, "hidden", "green vegetation"),
    ("seed23_alt55", 287, 53): (1.76, "hidden", "green vegetation"),
    ("seed23_alt55", 172, 55): (1.99, "hidden", "concrete slab"),
    ("seed23_alt55", 173, 55): (2.03, "hidden", "concrete slab"),
    ("seed23_alt55",  96, 12): (2.36, "visible", "pale figure on pale ground - a genuine hard case, keep"),
    ("seed23_alt55", 119, 20): (2.59, "hidden", "green vegetation"),
    ("seed23_alt55",  40, 49): (2.78, "hidden", "green moss"),
    ("seed23_alt55", 286, 53): (2.79, "hidden", "green vegetation"),
    ("seed23_alt55",  10, 41): (3.10, "visible", "half-submerged in water; water is NOT an instanced "
                                                 "occluder, so the mask is trustworthy here"),
    ("seed23_alt55", 150, 22): (4.06, "visible", "figure on mud, low contrast - keep"),
    ("seed23_alt55", 285, 53): (4.24, "hidden", "green vegetation"),
    ("seed23_alt55",  95, 12): (4.28, "visible", "person on pale ground"),
    ("seed23_alt55", 161, 24): (4.34, "hidden", "dark green vegetation"),
    # ---- seed47_alt55 (depth-gated at source; these are the residue the gate did not catch) ----------
    # The gate dropped 15 observations by itself. These two are what the eye found in the remainder, and
    # they are the reason `apply_depth_visibility` now needs a visible-FRACTION floor and not just a pixel
    # floor: a handful of pixels through gaps between fronds held a full box open.
    ("seed47_alt55",  68, 70): (0.60, "hidden", "solid vegetation under a jacaranda; the old pixel-only "
                                                "floor let specks seen through frond gaps hold the box"),
    ("seed47_alt55",  45, 12): (0.80, "visible", "person in dark clothing on pale ground - low contrast "
                                                 "against the surface, but plainly a person; KEEP"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    hidden = {k: v for k, v in VERDICTS.items() if v[1] == "hidden"}
    print(f"{len(VERDICTS)} boxes reviewed by eye, {len(hidden)} judged to have no visible subject")

    touched, applied, missing = 0, 0, []
    for run in sorted({k[0] for k in VERDICTS}):
        root = REPO / "_artifacts" / "dataset" / run
        by_frame: dict[int, list] = {}
        for (r, idx, aid), v in VERDICTS.items():
            if r == run:
                by_frame.setdefault(idx, []).append((aid, v))
        n_run = 0
        for idx, entries in sorted(by_frame.items()):
            hits = sorted(root.glob(f"labels/*_{idx:05d}.json"))
            if not hits:
                missing.append(f"{run} frame {idx:05d}: no label file")
                continue
            lf = hits[0]
            labs = json.loads(lf.read_text(encoding="utf-8"))
            dirty = False
            for aid, (de, verdict, why) in entries:
                tgt = next((m for m in labs if m.get("actor_id") == aid), None)
                if tgt is None:
                    missing.append(f"{run} frame {idx:05d}: actor {aid} not in the label file")
                    continue
                if verdict != "hidden":
                    continue
                tgt["ignore"] = True
                tgt["ignore_reason"] = (
                    f"no visible subject (reviewed by eye; RGB silhouette contrast dE {de:.2f}): {why}. "
                    "The instance mask is blind to instanced foliage and rubble - Cosys-AirSim disables the "
                    "InstancedFoliage/InstancedGrass show flags in its annotation render - so this box was "
                    "written over an occluder.")
                tgt["ignore_basis"] = "manual visual QA, tools/capture/flag_occluded_boxes.py"
                dirty = True
                applied += 1
                n_run += 1
            if dirty and a.write:
                lf.write_text(json.dumps(labs, indent=1), encoding="utf-8")
                touched += 1
        print(f"  {run}: {n_run} box(es) flagged ignore")

    if missing:
        print("\nFAIL: the verdict table does not match the data on disk:")
        for m in missing:
            print(f"  {m}")
        return 2

    print(f"\n{applied} box(es) marked ignore"
          + (f", {touched} label file(s) written" if a.write else "  (report only - pass --write)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
