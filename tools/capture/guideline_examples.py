"""Cut worked examples out of the REAL dataset for `docs/ANNOTATION_GUIDELINE.md` (SOLUTION_DOC 6.3).

    uv run python tools/capture/guideline_examples.py _artifacts/dataset/seed23_alt35 [more runs ...]

Section 6.3 asks the guideline to ship "8-12 example crops per rule". Drawing those by hand, or picking them
by eye, would defeat the purpose: the guideline exists so that a human reviewer and the auto-labeller agree,
so its examples must be the auto-labeller's ACTUAL output on real frames, not an illustration of what it is
supposed to do.

So this selects examples by the attribute the rule is about - one per pose, one per submersion class, one per
occlusion level, plus the smallest and largest boxes in the set - and writes each as a crop with the box the
pipeline actually produced drawn on it, captioned with every attribute that box carries. Any disagreement
between the rule and the picture is then visible rather than arguable.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def crops(runs: list[str], out_dir: Path, pad: int = 90) -> list[dict]:
    boxes: list[tuple] = []
    for spec in runs:
        rp = Path(spec) if Path(spec).is_absolute() else REPO / spec
        for lj in sorted(glob.glob(str(rp / "labels" / "*.json"))):
            stem = Path(lj).stem
            img = next((p for p in (rp / "images" / f"{stem}.jpg", rp / "images" / f"{stem}.png")
                        if p.exists()), None)
            if img is None:
                continue
            for L in json.loads(Path(lj).read_text()):
                boxes.append((img, L))
    if not boxes:
        return []

    picks: dict[str, tuple] = {}
    for img, L in boxes:
        for key in (f"pose={L.get('pose')}", f"submersion={L.get('submersion')}",
                    f"occlusion={L.get('occlusion')}", f"zone={L.get('zone')}"):
            # Prefer the LARGEST example of each attribute: the rule is easier to judge when the subject is
            # legible, and a rule illustrated by an unreadable 9 px smudge teaches nobody anything.
            if key not in picks or L.get("size_px", 0) > picks[key][1].get("size_px", 0):
                picks[key] = (img, L)
    sm = min(boxes, key=lambda t: t[1].get("size_px", 1e9))
    lg = max(boxes, key=lambda t: t[1].get("size_px", 0))
    picks["extreme=smallest box in the set"] = sm
    picks["extreme=largest box in the set"] = lg

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for key, (img, L) in sorted(picks.items()):
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w, _ = im.shape
        x1, y1, x2, y2 = L["bbox_px"]
        cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
        cx1, cy1 = min(w, x2 + pad + 1), min(h, y2 + pad + 1)
        c = im[cy0:cy1, cx0:cx1].copy()
        cv2.rectangle(c, (x1 - cx0, y1 - cy0), (x2 - cx0, y2 - cy0), (0, 0, 255), 2)
        # a caption band so the crop is self-describing when it is looked at months later
        cap = (f"{L['name']} {L.get('pose')}/{L.get('submersion')} occl={L.get('occlusion')} "
               f"{L.get('zone')} {L.get('size_px')}px vis={L.get('visible_px')}")
        band = np.full((26, c.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(band, cap[:110], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1,
                    cv2.LINE_AA)
        c = np.vstack([c, band])
        fn = out_dir / (key.replace("=", "_").replace(" ", "_").replace("/", "-") + ".png")
        cv2.imwrite(str(fn), c)
        made.append({"rule": key, "file": fn.name, "frame": img.name, "attrs": {
            k: L.get(k) for k in ("name", "pose", "submersion", "occlusion", "zone", "size_px",
                                  "visible_px", "bbox_px", "aerially_detectable")}})
        print(f"  {key:44s} -> {fn.name}  ({L.get('size_px')} px)")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="_artifacts/guideline")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    made = crops(a.runs, out)
    if not made:
        print("no boxes found in the given runs")
        return 1
    (out / "examples.json").write_text(json.dumps(
        {"by": "tools/capture/guideline_examples.py", "runs": a.runs, "domain": "sim",
         "note": "each crop shows the box the auto-labeller actually produced, not a hand illustration",
         "examples": made}, indent=1), encoding="utf-8")
    print(f"\n{len(made)} example crops -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
