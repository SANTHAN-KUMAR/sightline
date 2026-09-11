"""Correct the occlusion of the specific survivors that `check_occlusion_truth.py` proves are covered.

    uv run python tools/scene/fix_occlusion_truth.py            # show the diff
    uv run python tools/scene/fix_occlusion_truth.py --write    # apply it

This is the narrow counterpart to `tools/scene/reconcile_occlusion.py`, which **refuses** to write: that one
models every occluder as an opaque disc with no height test and does not load the buildings at all, so
writing it would downgrade seven rooftop survivors from `occlusion: 1` to `0` and destroy correct ground
truth. It reports 25 contradictions where only 4 survive a height test.

This one touches ONLY survivors that the height-gated checker positively identifies, and only ever RAISES
their occlusion. It cannot downgrade anybody, so a gap in the model can cost us a missing correction but
never a fabricated one - the safe direction for a number the whole evaluation is measured against.

Why these four are real rather than a modelling artefact: `gen_rubble.py` deliberately leans slabs back over
the fan survivors - "the plate roofs the void without filling it" - to create the 25-75 % occlusion slice the
solution doc asks for, and `gen_props.py` drops branches on the trapped. The geometry is correct and
intended. Nothing ever told `actors.json`, so its `occlusion: 0` was a promise the scene had stopped keeping.

Level 1 (partial), not 2: a void wall roofs part of a prone body and leaves the rest visible, which is
exactly the partial slice. Level 2 is reserved for the section 2.7 burials, which are already recorded and
which carry `aerially_detectable: false` as well.

Nothing needs re-placing afterwards. This changes a LABEL, not a position: `gen_vegetation.py`'s clearance
rule keeps trees off survivors recorded `occlusion: 0`, and moving four survivors out of that set only makes
the protected set smaller, so the invariant still holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/scene"))

from check_occlusion_truth import measure  # noqa: E402

RAISE_TO = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    S = REPO / "data/scene"
    doc = json.loads((S / "actors.json").read_text(encoding="utf-8"))
    actors = doc["actors"]

    # Use the checker's OWN measurement, not a copy of it. A second implementation of these loops
    # drifted from the checker within minutes of being written and found 3 of the 4 violations.
    over = measure(actors)

    changes = []
    for x in actors:
        cov = over[x["name"]]
        if cov and x.get("occlusion", 0) < RAISE_TO and x.get("aerially_detectable", True):
            changes.append((x, cov))

    if not changes:
        print("nothing to correct - every survivor's occlusion already matches the geometry above them")
        return 0

    print(f"{len(changes)} survivor(s) declared occlusion 0 with geometry proven ABOVE them:\n")
    for x, cov in changes:
        srcs = ", ".join(f"{s}:{n}@{d:.1f}m" for s, n, d in cov[:3])
        print(f"  {x['name']:12s} {x['pose']:9s} {x['zone']:11s} "
              f"occlusion {x.get('occlusion', 0)} -> {RAISE_TO}   {srcs}")

    if not a.write:
        print("\n(report only - pass --write to apply. This tool only ever RAISES occlusion; it cannot "
              "downgrade a survivor, so a gap in the model costs a missed correction, never a false one.)")
        return 1

    for x, cov in changes:
        x["occlusion_intent"] = x.get("occlusion", 0)
        x["occlusion"] = RAISE_TO
        x["occluded_by"] = sorted({s for s, _n, _d in cov})
    doc["occlusion_corrected"] = {
        "by": "tools/scene/fix_occlusion_truth.py",
        "rule": "raise-only, height-gated; never downgrades",
        "changed": [x["name"] for x, _ in changes],
    }
    (S / "actors.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nwritten: data/scene/actors.json ({len(changes)} corrected). "
          f"This is a LABEL change only - no actor moves, nothing needs re-placing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
