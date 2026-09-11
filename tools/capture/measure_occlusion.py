"""Replace the assumed occlusion label with one MEASURED from the rendered mask.

    uv run python tools/capture/measure_occlusion.py _artifacts/dataset/<run>            # report
    uv run python tools/capture/measure_occlusion.py _artifacts/dataset/<run> --write    # update labels

Every label currently carries the `occlusion` that `gen_actors.py` INTENDED when it laid the scenario out,
copied straight through by `labels.attach_truth`. That number is a plan, not an observation: on the
2026-09-10 layouts, 25 of 71 survivors had an intended level that the finished geometry contradicted, in
both directions - slabs leaned over survivors marked `occlusion: 0`, and survivors marked `occlusion: 1`
standing in the open with nothing above them.

Geometry can be measured (see `tools/scene/reconcile_occlusion.py`), but that treats a tree crown as an
opaque disc, which it is not. The mask does not have to assume anything: it records exactly how many pixels
of each survivor the camera actually saw.

    visible_m2  = visible_px * (gsd_m_per_px ** 2)          per observation, from the frame's own GSD
    reference   = p90 of visible_m2 for that pose over observations with no geometry above them
    visible_fraction = visible_m2 / reference
    occlusion   = 0 if vf >= 0.90,  1 if 0.40 <= vf < 0.90,  2 if vf < 0.40      (schema OCCLUSION_BINS)

The reference is drawn from the run's own clean observations, so it needs no extra render pass and no
hand-tuned constant. A pose with fewer than MIN_REF clean observations gets NO reference and its labels keep
`visible_fraction: null` - the same refusal `labels.attach_visible_fraction` already makes rather than
inventing a number.

Submersion is deliberately NOT factored out: `half_submerged` and `head_only` are their own poses, so water
occlusion is inside the pose's own reference and debris occlusion is what shows up in the fraction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MIN_REF = 8          # clean observations needed before a pose gets a reference silhouette at all
MIN_PX = 20          # below the section 5.5 resolvable limit the pixel count is mostly quantisation noise
CLEAN_COVER = 0.02   # geometric coverage below which an observation counts as "nothing above them"
VF_CLEAR, VF_HEAVY = 0.90, 0.40


def geometric_cover(run: Path) -> dict[int, float]:
    """actor_id -> fraction of footprint covered by scene geometry, from reconcile_occlusion's model."""
    sys.path.insert(0, str(REPO / "tools/scene"))
    try:
        from reconcile_occlusion import overhead  # noqa: PLC0415
    except Exception:
        return {}
    import math
    from tools.capture.labels import truth_for_run  # noqa: PLC0415
    actors = truth_for_run(run)[0]["actors"]
    cov, own_r = overhead(actors)
    out = {}
    for a in actors:
        area = math.pi * own_r[a["name"]] ** 2
        out[a["id"]] = min(1.0, sum(cov[a["name"]].values()) / area)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else REPO / a.run

    gsd_m: dict[int, float] = {}
    tele = run / "telemetry.csv"
    if not tele.exists():
        print(f"FAIL: {tele} not found - the per-frame GSD is the whole basis of this measurement")
        return 1
    for r in csv.DictReader(tele.open(newline="", encoding="utf-8")):
        gsd_m[int(r["frame_idx"])] = float(r["gsd_cm_px"]) / 100.0

    cover = geometric_cover(run)
    if not cover:
        print("WARNING: could not load the geometric model; every observation will be treated as clean, "
              "which INFLATES the reference and understates occlusion")

    files = sorted(glob.glob(str(run / "labels" / "*.json")))
    obs = []          # (file, index_in_file, actor_id, pose, visible_m2, size_px, clean)
    for fp in files:
        idx = int(Path(fp).stem.rsplit("_", 1)[-1])
        g = gsd_m.get(idx)
        if g is None:
            continue
        for i, L in enumerate(json.loads(Path(fp).read_text())):
            vm2 = L["visible_px"] * g * g
            obs.append((fp, i, L["actor_id"], L.get("pose", "?"), vm2, L.get("size_px", 0),
                        cover.get(L["actor_id"], 0.0) < CLEAN_COVER))
    if not obs:
        print("FAIL: no observations")
        return 1

    by_pose: dict[str, list[float]] = defaultdict(list)
    for _f, _i, _a, pose, vm2, px, clean in obs:
        if clean and px >= MIN_PX:
            by_pose[pose].append(vm2)

    ref: dict[str, float] = {}
    print(f"reference silhouette per pose (p90 of clean observations, n >= {MIN_REF} required)")
    poses = sorted({o[3] for o in obs})
    for pose in poses:
        v = by_pose.get(pose, [])
        if len(v) >= MIN_REF:
            ref[pose] = float(np.percentile(v, 90))
            print(f"  {pose:16s} n={len(v):4d}  p50 {np.median(v):5.2f}  p90 {ref[pose]:5.2f} m2")
        else:
            print(f"  {pose:16s} n={len(v):4d}  NO REFERENCE - labels keep visible_fraction null")

    per_file: dict[str, dict[int, tuple[float | None, int | None]]] = defaultdict(dict)
    lvl_count: dict[int, int] = defaultdict(int)
    none_count = 0
    changed = 0
    old_lvl: dict[int, int] = defaultdict(int)
    for fp, i, aid, pose, vm2, px, _clean in obs:
        r = ref.get(pose)
        if r is None or r <= 0 or px < MIN_PX:
            per_file[fp][i] = (None, None)
            none_count += 1
            continue
        vf = float(min(1.0, vm2 / r))
        lvl = 0 if vf >= VF_CLEAR else (1 if vf >= VF_HEAVY else 2)
        per_file[fp][i] = (round(vf, 4), lvl)
        lvl_count[lvl] += 1

    print()
    tot = sum(lvl_count.values())
    print(f"measured occlusion over {tot} observations "
          f"({none_count} left unmeasured: no pose reference, or under {MIN_PX} px)")
    for k in sorted(lvl_count):
        print(f"  level {k}: {lvl_count[k]:5d}  ({100 * lvl_count[k] / max(1, tot):5.1f} %)")

    for fp in files:
        labs = json.loads(Path(fp).read_text())
        for i, L in enumerate(labs):
            vf, lvl = per_file.get(fp, {}).get(i, (None, None))
            old_lvl[L.get("occlusion") if L.get("occlusion") is not None else -1] += 1
            if lvl is not None and lvl != L.get("occlusion"):
                changed += 1
    print(f"intended occlusion in the labels now: "
          f"{ {k: v for k, v in sorted(old_lvl.items())} }")
    print(f"{changed} label(s) would change level")

    if a.write:
        n = 0
        for fp in files:
            labs = json.loads(Path(fp).read_text())
            if not labs:
                continue
            for i, L in enumerate(labs):
                vf, lvl = per_file.get(fp, {}).get(i, (None, None))
                L["occlusion_intent"] = L.get("occlusion")
                L["visible_fraction"] = vf
                if lvl is not None:
                    L["occlusion"] = lvl
            Path(fp).write_text(json.dumps(labs, indent=1), encoding="utf-8")
            n += 1
        (run / "occlusion_reference.json").write_text(json.dumps({
            "by": "tools/capture/measure_occlusion.py",
            "reference_m2_p90_by_pose": {k: round(v, 4) for k, v in sorted(ref.items())},
            "clean_observations_by_pose": {k: len(v) for k, v in sorted(by_pose.items())},
            "bins": {"clear_at": VF_CLEAR, "heavy_below": VF_HEAVY},
            "min_px": MIN_PX, "min_reference_n": MIN_REF,
            "measured": tot, "unmeasured": none_count, "changed": changed, "domain": "sim",
        }, indent=1), encoding="utf-8")
        print(f"written: {n} label files updated, {run / 'occlusion_reference.json'}")
    else:
        print("(report only - pass --write to update the labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
