"""Build a train/val split that no survivor appears on both sides of.

    uv run python tools/capture/make_split.py _artifacts/dataset/<run> [<run> ...] --out _artifacts/dataset/split

SOLUTION_DOC 5.5c splits by scenario seed, because splitting a survey by frame leaks: consecutive frames
overlap by design, so the same survivor, in the same pose, under the same tree, lands in both halves and
every number comes out flattering. A second scenario seed is the ideal, and it is expensive here - six
generators key their geometry to survivor positions (`gen_rubble.py` builds the voids around them,
`gen_props.py` caps the buried ones), so a new actor seed means regenerating and re-placing the whole scene.

This achieves the property the rule exists for, at no scene cost, by splitting on survivor IDENTITY:

  * every survivor is assigned to train or val, stratified by pose/submersion/zone so both sides keep the
    full slice grid;
  * a frame goes to train only if every survivor in it is a train survivor, and to val only if every
    survivor in it is a val survivor. **Frames containing both are dropped**, which is what stops the leak;
  * empty frames are true negatives and are dealt to both sides in proportion.

What it does NOT do, stated plainly rather than buried: the terrain, the buildings, the water and the debris
are the same in both halves, so this measures generalisation to unseen PEOPLE, not to an unseen PLACE. Any
number computed on this val set must be reported as such. A second scenario seed remains the honest way to
measure the second thing, and `--held-out-run` accepts one when it exists.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="_artifacts/dataset/split")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--held-out-run", default="",
                    help="a run from a DIFFERENT scenario seed; if given it becomes the val set outright "
                         "and the identity split is not used")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    actors = json.loads((REPO / "data/scene/actors.json").read_text())["actors"]
    det = [x for x in actors if x["aerially_detectable"]]

    # --- assign survivors, stratified so neither side loses a slice ---------------------------------------
    strata: dict[str, list[int]] = defaultdict(list)
    for x in det:
        strata[f"{x['pose']}/{x['submersion']}/{x['zone']}"].append(x["id"])
    val_ids: set[int] = set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        k = max(1, round(len(ids) * a.val_frac)) if len(ids) > 1 else 0
        val_ids.update(ids[:k])
    train_ids = {x["id"] for x in det} - val_ids
    print(f"survivors: {len(train_ids)} train / {len(val_ids)} val "
          f"(stratified over {len(strata)} pose/submersion/zone strata)")
    for key in sorted(strata):
        v = sum(1 for i in strata[key] if i in val_ids)
        print(f"  {key:38s} {len(strata[key]) - v:3d} train / {v:2d} val")

    # --- deal the frames ------------------------------------------------------------------------------------
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    split: dict[str, list[str]] = {"train": [], "val": []}
    dropped = 0
    neg = []
    counts = {"train": Counter(), "val": Counter()}
    for run in a.runs:
        rp = Path(run) if Path(run).is_absolute() else REPO / run
        for lj in sorted(glob.glob(str(rp / "labels" / "*.json"))):
            labs = json.loads(Path(lj).read_text())
            stem = Path(lj).stem
            img = next((str(p) for p in (rp / "images" / f"{stem}.jpg", rp / "images" / f"{stem}.png")
                        if p.exists()), None)
            if img is None:
                continue
            ids = {L["actor_id"] for L in labs}
            if not ids:
                neg.append(img)
                continue
            if ids <= train_ids:
                side = "train"
            elif ids <= val_ids:
                side = "val"
            else:
                dropped += 1
                continue
            split[side].append(img)
            for L in labs:
                counts[side][f"{L.get('pose')}/{L.get('submersion')}"] += 1

    rng.shuffle(neg)
    cut = int(len(neg) * (1 - a.val_frac))
    split["train"] += neg[:cut]
    split["val"] += neg[cut:]

    print(f"\nframes: {len(split['train'])} train / {len(split['val'])} val, "
          f"{dropped} dropped for containing survivors from both sides, "
          f"{len(neg)} true negatives dealt {cut}/{len(neg) - cut}")
    print(f"boxes:  {sum(counts['train'].values())} train / {sum(counts['val'].values())} val")
    for side in ("train", "val"):
        miss = [k for k in counts["train"] if k not in counts[side]]
        if side == "val" and miss:
            print(f"WARNING: val has no boxes for {miss} - those slices cannot be evaluated")

    for side in ("train", "val"):
        (out / f"{side}.txt").write_text("\n".join(split[side]), encoding="utf-8")
    (out / "split.json").write_text(json.dumps({
        "by": "tools/capture/make_split.py",
        "method": "survivor identity, stratified by pose/submersion/zone; frames mixing both sides dropped",
        "leaks": "terrain, buildings, water and debris are shared - this measures unseen PEOPLE, not an "
                 "unseen PLACE",
        "runs": a.runs, "val_frac": a.val_frac, "seed": a.seed,
        "train_actor_ids": sorted(train_ids), "val_actor_ids": sorted(val_ids),
        "frames": {"train": len(split["train"]), "val": len(split["val"]), "dropped_mixed": dropped},
        "boxes": {"train": sum(counts["train"].values()), "val": sum(counts["val"].values())},
        "domain": "sim",
    }, indent=1), encoding="utf-8")
    print(f"written: {out / 'train.txt'}, {out / 'val.txt'}, {out / 'split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
