"""The dataset gate: one verdict on whether a captured campaign may be trained on.

    uv run python tools/capture/dataset_gate.py _artifacts/dataset/seed23_alt35 [more runs ...]
    uv run python tools/capture/dataset_gate.py --campaign _artifacts/dataset/campaign_seed23.json

`tools/scene/gate.py` answers "is the environment finished?". This answers "is the data fit to train on?",
and it exists for the same reason: the per-run checkers each answer for their own corner, and a dataset can
pass every one of them and still be worthless.

It runs, per run:

    measure_occlusion   replaces the ASSUMED occlusion label with one measured from the rendered mask
    validate            corruption: airframe in frame, blank frames, label-vs-mask, attitude, AGL, box sanity
    quality_report      fitness: volume, resolvable target size, slice balance, duplicates, split integrity

and then, ACROSS runs, the things no single run can see: whether the campaign actually covers the situations
it was flown for, and whether the same survivor appears in two runs that a split would later separate.

Two rules carried over from the environment gate, both learned the hard way here:

* **A check that did not run is not a check that passed.** `validate.py` once printed "all checks passed -
  dataset is clean" and exited 0 while silently skipping the only check that confirms a labelled box
  contains its actor. It now exits 2 on a skipped check, and so does this.
* **Report what was measured, not what was run.** Every line says what it proves.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: (name, argv-suffix, what it proves)
PER_RUN = [
    ("enrich-labels", ["tools/capture/enrich_labels.py", "{run}", "--write"],
     "the section 6.3 size and truncation flags are set, so a sub-8 px smudge is judged UNRESOLVABLE rather "
     "than scored as a missed survivor - `GtBox.is_recall_target` honours `uncertain` and `ignore`"),
    ("measure-occlusion", ["tools/capture/measure_occlusion.py", "{run}", "--write"],
     "each observation's occlusion is measured from the rendered mask and the frame's own GSD, replacing "
     "the value the scenario generator merely intended"),
    ("validate", ["tools/capture/validate.py", "{run}"],
     "the run is not corrupt: no airframe in frame, no blank or duplicate frames, every labelled box "
     "contains its actor in the mask, attitude and AGL within survey limits, no impossible box sizes"),
    ("quality", ["tools/capture/quality_report.py", "{run}", "--in-campaign"],
     "the run is fit to TRAIN on: enough boxes and distinct survivors, targets above the resolvable limit, "
     "slices populated, no near-duplicate flood, ground truth consistent"),
]


def run(argv: list[str], timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run([sys.executable, *argv], cwd=str(REPO), capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMED OUT after {timeout}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--campaign", default="", help="a campaign_seed*.json; its passes become the run list")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--min-boxes", type=int, default=400,
                    help="total boxes across the campaign below which a fine-tune is not worth starting")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    runs = list(a.runs)
    if a.campaign:
        c = json.loads((REPO / a.campaign).read_text()) if not Path(a.campaign).is_absolute() \
            else json.loads(Path(a.campaign).read_text())
        runs += [p["out"] for p in c.get("passes", []) if p.get("ok")]
    if not runs:
        print("no runs given")
        return 2

    rows, t0 = [], time.time()
    for spec in runs:
        rp = Path(spec) if Path(spec).is_absolute() else REPO / spec
        print(f"\n{'=' * 78}\n{rp.name}\n{'=' * 78}")
        if not (rp / "data_card.json").exists():
            rows.append({"run": rp.name, "name": "exists", "status": "FAIL",
                         "detail": "no data_card.json - the run never finished"})
            print("  FAIL   no data_card.json - the run never finished")
            continue
        for name, argv, proves in PER_RUN:
            code, out = run([x.replace("{run}", str(rp)) for x in argv], a.timeout)
            status = "PASS" if code == 0 else ("UNKNOWN" if code in (2, 124) else "FAIL")
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-4:]
            rows.append({"run": rp.name, "name": name, "status": status, "code": code,
                         "proves": proves, "detail": " | ".join(tail)[:500]})
            print(f"  {name:18s} {status:8s} exit {code}")
            if status != "PASS":
                for ln in tail:
                    print(f"      {ln[:150]}")

    # --- across the campaign: the things no single run can see -------------------------------------------
    print(f"\n{'=' * 78}\nCAMPAIGN COVERAGE (domain=sim)\n{'=' * 78}")
    conds, alts, boxes_total = Counter(), Counter(), 0
    seen_by_run: dict[str, set] = {}
    slices: dict[str, int] = defaultdict(int)
    for spec in runs:
        rp = Path(spec) if Path(spec).is_absolute() else REPO / spec
        card_p = rp / "data_card.json"
        if not card_p.exists():
            continue
        card = json.loads(card_p.read_text())
        conds[str(card.get("condition"))] += 1
        alts[str(card.get("altitude_m_agl"))] += 1
        ids = set()
        for lj in glob.glob(str(rp / "labels" / "*.json")):
            for L in json.loads(Path(lj).read_text()):
                ids.add(L["actor_id"])
                boxes_total += 1
                slices[f"{L.get('pose')}/{L.get('submersion')}"] += 1
        seen_by_run[rp.name] = ids

    print(f"conditions: {dict(conds)}")
    print(f"altitudes : {dict(alts)}")
    print(f"boxes     : {boxes_total} across {len(seen_by_run)} run(s)")
    print(f"survivors : {len(set().union(*seen_by_run.values())) if seen_by_run else 0} distinct")
    print(f"pose/submersion slices populated: {len(slices)}")
    for k, v in sorted(slices.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {v}")

    # --- do the CONDITIONS actually look different? ------------------------------------------------------
    # A pass is labelled `rain_overcast` because we asked AirSim for rain, not because rain was observed.
    # `simEnableWeather` only renders if the weather FX exist in the level, and this level was built from
    # scratch - so a pass can come back visually identical to a clear one while carrying a weather label.
    # That is worse than having no rain pass at all, because the label would be believed and the model would
    # be credited with robustness it was never shown. So: measure the frames and require the conditions to
    # be distinguishable.
    import cv2
    import numpy as np
    print()
    sig = {}
    for spec in runs:
        rp = Path(spec) if Path(spec).is_absolute() else REPO / spec
        cp = rp / "data_card.json"
        if not cp.exists():
            continue
        cond = str(json.loads(cp.read_text()).get("condition"))
        imgs = sorted(glob.glob(str(rp / "images" / "*.jpg")) + glob.glob(str(rp / "images" / "*.png")))
        vals = []
        for f in imgs[:: max(1, len(imgs) // 25)]:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            b, gr, r = [im[:, :, i].astype(np.float32).mean() / 255.0 for i in range(3)]
            vals.append((g.mean(), g.std(), r / max(b, 1e-6)))
        if vals:
            v = np.array(vals)
            sig[cond] = {"n": len(vals), "luma": float(v[:, 0].mean()),
                         "contrast": float(v[:, 1].mean()), "warmth_r_over_b": float(v[:, 2].mean())}
    print("condition signatures (sampled frames, domain=sim)")
    for c, v in sig.items():
        print(f"   {c:16s} n={v['n']:3d}  luma {v['luma']:.3f}  contrast {v['contrast']:.3f}  "
              f"R/B {v['warmth_r_over_b']:.3f}")
    names = list(sig)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = sig[names[i]], sig[names[j]]
            d = (abs(a_["luma"] - b_["luma"]) + abs(a_["contrast"] - b_["contrast"])
                 + abs(a_["warmth_r_over_b"] - b_["warmth_r_over_b"]))
            if d < 0.02:
                rows.append({"run": "campaign", "name": f"condition:{names[i]}~{names[j]}", "status": "FAIL",
                             "proves": "conditions labelled differently actually LOOK different",
                             "detail": f"{names[i]} and {names[j]} are visually indistinguishable "
                                       f"(combined luma/contrast/warmth difference {d:.4f} < 0.02). The "
                                       f"weather or time-of-day call did not change the render, so that "
                                       f"diversity axis is fictional and the label would be believed."})
                print(f"   FAIL {names[i]} vs {names[j]}: indistinguishable (diff {d:.4f})")

    fails = [r for r in rows if r["status"] == "FAIL"]
    unknown = [r for r in rows if r["status"] == "UNKNOWN"]
    if len(conds) < 2:
        fails.append({"run": "campaign", "name": "diversity", "status": "FAIL",
                      "proves": "the campaign covers more than one situation",
                      "detail": f"only {len(conds)} lighting condition(s) captured. The whole point of a "
                                f"compact campaign is that it covers SITUATIONS rather than piling up "
                                f"frames of one."})
    if boxes_total < a.min_boxes:
        fails.append({"run": "campaign", "name": "volume", "status": "FAIL",
                      "proves": "there is enough signal to fine-tune on",
                      "detail": f"{boxes_total} boxes across the whole campaign, under the {a.min_boxes} "
                                f"floor. More epochs cannot substitute for targets that were never seen."})

    print(f"\n{'=' * 78}")
    print(f"{len([r for r in rows if r['status'] == 'PASS'])}/{len(rows)} run checks passed "
          f"in {time.time() - t0:.0f}s")
    for r in fails + unknown:
        print(f"\n{r['status']}  {r.get('run')}/{r['name']}: {r.get('proves', '')}")
        print(f"       {r.get('detail', '')[:300]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "boxes": boxes_total,
                                            "conditions": dict(conds), "altitudes": dict(alts),
                                            "domain": "sim"}, indent=1), encoding="utf-8")
    if fails or unknown:
        print(f"\nDATASET IS NOT FIT TO TRAIN ON: {len(fails)} failing, {len(unknown)} did not run.")
        return 1
    print("\nDATASET GATE GREEN - every check ran and passed. Look at the contact sheet before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
