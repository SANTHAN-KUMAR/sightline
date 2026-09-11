"""Add the §6.3 attributes the capture cannot know at shutter time: `uncertain`, `ignore`, `truncated`.

    uv run python tools/capture/enrich_labels.py _artifacts/dataset/<run>            # report
    uv run python tools/capture/enrich_labels.py _artifacts/dataset/<run> --write

The evaluation harness already honours all three - `sightline/eval/groundtruth.py:GtBox.is_recall_target`
returns False for an `uncertain` or `ignore` box, so it is neither a true positive nor a miss, exactly as
§6.3 specifies. The capture side simply never set them, which means **a 5 px smudge is currently scored as a
missed survivor**. That understates recall on precisely the slice where the guideline says judgement should
be withheld, and it does so silently.

This runs after capture rather than inside `labels.py` for two reasons: `labels.py` is imported by a survey
that may be mid-flight, and these are judgements about a finished frame rather than facts the shutter knows.

The three rules, from §6.3:

* **`ignore`** - "no box < 4 px" is one of the guideline's own pre-export sanity checks. Below 4 px there is
  not enough signal to call it anything, so the box is neither a target nor a false positive.
* **`uncertain`** - "Minimum size >= 8 px on the longer side ... anything 4-8 px gets `uncertain = 1` rather
  than being skipped." TinyPerson's [2, 8] bin is described as near-hopeless; the guideline's answer is to
  withhold judgement rather than to drop the instance or to score it.
* **`truncated`** - the box touches the frame border. §6.3 defines truncation as ">= 50 % of the visible body
  cut", which needs an amodal extent this pipeline does not have per frame. **So this sets border CONTACT,
  which is a fact that can be measured exactly, and does not claim to be the 50 % rule.** It is safe to be
  inclusive here because `truncated` is carried for slicing and reporting only - `matching.py` and
  `detection.py` never read it, so a border-touching box is still scored normally.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

IGNORE_BELOW_PX = 4      # §6.3 sanity check: "no box < 4 px"
UNCERTAIN_BELOW_PX = 8   # §6.3: "Minimum size >= 8 px ... anything 4-8 px gets uncertain = 1"


def enrich(label: dict, width: int, height: int) -> dict:
    """Return the three flags for one label. Pure, so the test can drive it directly."""
    x1, y1, x2, y2 = label["bbox_px"]
    size = float(label.get("size_px", max(x2 - x1 + 1, y2 - y1 + 1)))
    touches = bool(x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1)
    return {
        # `ignore` is a UNION, never an overwrite. This function knows one reason to ignore a box - it is
        # too small to resolve - and it is not the only reason. `tools/capture/flag_occluded_boxes.py` sets
        # `ignore` on boxes drawn over fern canopy and concrete slabs with no subject visible in RGB, which
        # is invisible to a size test. Recomputing the flag from size alone silently cleared all 18 of them
        # and handed a clean-looking dataset straight back to training.
        "ignore": bool(size < IGNORE_BELOW_PX) or bool(label.get("ignore", False)),
        "uncertain": bool(IGNORE_BELOW_PX <= size < UNCERTAIN_BELOW_PX),
        "truncated": touches,
        "truncated_basis": "border contact, not the section 6.3 >=50% rule",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run

    card = run / "data_card.json"
    w, h = 3840, 2160
    if card.exists():
        c = json.loads(card.read_text())
        w = int(c.get("width_px", w) or w)
        h = int(c.get("height_px", h) or h)
    else:
        import csv
        tel = run / "telemetry.csv"
        if tel.exists():
            rows = list(csv.DictReader(tel.open(newline="", encoding="utf-8")))
            if rows:
                w, h = int(rows[0].get("width_px", w)), int(rows[0].get("height_px", h))
    print(f"frame {w}x{h}")

    files = sorted(glob.glob(str(run / "labels" / "*.json")))
    tally, total = Counter(), 0
    changed_files = 0
    for fp in files:
        labs = json.loads(Path(fp).read_text())
        if not labs:
            continue
        dirty = False
        for L in labs:
            flags = enrich(L, w, h)
            total += 1
            for k in ("ignore", "uncertain", "truncated"):
                if flags[k]:
                    tally[k] += 1
            if any(L.get(k) != flags[k] for k in ("ignore", "uncertain", "truncated")):
                dirty = True
            L.update(flags)
        if dirty and a.write:
            Path(fp).write_text(json.dumps(labs, indent=1), encoding="utf-8")
            changed_files += 1

    print(f"{total} boxes")
    print(f"  ignore    (< {IGNORE_BELOW_PX} px)      : {tally['ignore']}  "
          f"- neither a recall target nor a false positive")
    print(f"  uncertain ({IGNORE_BELOW_PX}-{UNCERTAIN_BELOW_PX} px)      : {tally['uncertain']}  "
          f"- judgement withheld; previously scored as MISSES")
    print(f"  truncated (touching border) : {tally['truncated']}  - reported, not excluded")
    if a.write:
        print(f"\nwritten: {changed_files} label file(s) updated")
    else:
        print("\n(report only - pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
