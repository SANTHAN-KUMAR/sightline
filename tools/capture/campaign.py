"""Fly the whole capture campaign: several altitudes and lighting conditions, one scenario seed at a time.

    uv run python tools/capture/campaign.py --dry-run          # print the plan, fly nothing
    uv run python tools/capture/campaign.py --seed 23
    uv run python tools/capture/campaign.py --seed 23 --passes alt35,alt55,alt80

Why this exists rather than a shell loop: a capture campaign that half-finishes is worse than one that never
started, because the half that exists looks like a dataset. Each pass here writes its own run directory with
its own data card, and the campaign records which passes succeeded, so a crashed editor costs you one pass
and not the afternoon.

The pass list is the argument the whole thing turns on. The 2026-09-11 full-area survey returned 56 boxes
from 285 frames because the 5-95 % quantile box is 607 x 960 m and the 69 survivors occupy a small part of
it; the yield matched the geometric prediction exactly, so the flight was executing a bad plan correctly.
These passes instead use `--plan patches`, which single-links the survivors into clusters and flies a small
lawnmower over each, and they vary ALTITUDE and LIGHT rather than adding more frames of the same thing:

    alt35   1.4 cm/px   the small-target end of the section 5.12 pixel-size slices
    alt55   2.2 cm/px   the nominal survey altitude
    alt80   3.2 cm/px   the coarse end, where recall starts to fall off the cliff

Sun angle changes the image far more than another three hundred frames of the same light does, and it costs
no extra flying, so the passes carry different times of day. Every condition is stamped into the data card
as a slice axis, so the evaluation can report per condition instead of averaging over them.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: name -> altitude (m AGL), time of day and weather. The shutter spacing is DERIVED, not chosen: see
#: `shutter_m` below.
PASSES: dict[str, dict] = {
    "alt35": dict(alt=35.0, tod="2026-09-11 09:20:00", cond="clear_morning"),
    "alt55": dict(alt=55.0, tod="2026-09-11 12:40:00", cond="clear_midday"),
    "alt80": dict(alt=80.0, tod="2026-09-11 16:10:00", cond="hazy_afternoon", fog=0.15),
    # A fourth pass earns its place only because it is a DIFFERENT SITUATION, not more of the same: rain on
    # the lens and a low, flat overcast is the condition a real flood search actually flies in, and it is the
    # one where a detector trained on clear midday falls over. Training is not the constraint here - 60
    # epochs on this dataset is ~15 min on a 4090 - so the cost of this pass is 13 minutes of FLYING, which
    # is the scarce resource.
    "alt45rain": dict(alt=45.0, tod="2026-09-11 14:30:00", cond="rain_overcast", rain=0.55, fog=0.25),
}
DEFAULT_ORDER = ["alt35", "alt55", "alt80", "alt45rain"]
HFOV_FALLBACK = 73.983


def shutter_m(alt: float, speed: float, hfov_deg: float, min_hits: int = 3, window_s: float = 2.0,
              width: int = 3840, height: int = 2160) -> float:
    """Shutter spacing derived from the TRACKER'S confirmation gate, not picked to look reasonable.

    SOLUTION_DOC 5.6 rule 3, encoded in `sightline/track/config.py` as `min_hits=3` and
    `min_hits_window_s=2.0`: a track is emitted only after 3 hits inside 2 seconds. A survey whose cadence
    cannot deliver that produces detections and geolocations and then NOTHING - no track, no record, no
    triage, no map pin. Measured on the 2026-09-11 run: 285 frames -> 56 detections -> 56 geolocated ->
    **0 tracks, 0 records**, because the shutter was 26 m at 11 m/s (2.4 s between frames, and each survivor
    seen about 1.5 times).

    Two conditions have to hold at once, and the spacing is the tighter of them:

        s <= speed * window_s / (min_hits - 1)   the hits must fall inside the 2 s window
        s <= frame_height_m / min_hits           the target must still be in frame for all 3

    At 12 m/s and 3 hits in 2 s the first gives 12 m; the second gives frame_height/3, which binds at low
    altitude where the frame is short. Both are properties of the mission, so neither is tunable by taste.
    """
    gw = 2.0 * alt * math.tan(math.radians(hfov_deg) / 2.0)
    gh = gw * height / width
    by_time = speed * window_s / max(1, min_hits - 1)
    by_frame = gh / min_hits
    # FLOOR to a decimetre, never round. At 35 m the frame is 29.67 m tall, so by_frame is 9.89 m; rounding
    # that to 9.9 puts the target in frame for 2.996 shots and the gate needs 3. Rounding a constraint in the
    # direction that breaks it is how a plan passes arithmetic and fails in the air.
    return math.floor(min(by_time, by_frame) * 10.0) / 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None,
                    help="scenario seed to STAMP in the run name; the level must already hold that "
                         "scenario (this does not re-place actors)")
    ap.add_argument("--passes", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--out-root", default="_artifacts/dataset")
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--jpeg", type=int, default=95, help="0 keeps 4K PNG at ~11.9 MB a frame")
    ap.add_argument("--max-minutes", type=float, default=30.0, help="per pass")
    ap.add_argument("--force", action="store_true",
                    help="fly even if the scene-freeze checks fail (throwaway run)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cal_p = REPO / "data/scene/camera_survey.json"
    hfov = float(json.loads(cal_p.read_text())["hfov_deg"]) if cal_p.exists() else HFOV_FALLBACK
    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    seed = a.seed if a.seed is not None else truth["seed"]
    if seed != truth["seed"]:
        print(f"REFUSING: --seed {seed} but data/scene/actors.json holds scenario seed {truth['seed']}. "
              f"The run name would claim a scenario the level does not contain, and the split would then "
              f"be built on a lie. Re-place the actors for seed {seed} first, or drop --seed.")
        return 2

    # --- the scene must be FROZEN before a single frame is captured -------------------------------------
    # Any change to the scene invalidates every capture taken before it, because the training images have to
    # come from the scene we actually demo. So the campaign is flown once, at the end, and these are the two
    # checks that say the scene is finished rather than merely recent:
    #   assert_qa_fresh      - nothing in the scene changed after the last time a human looked at a render
    #   check_occlusion_truth- no survivor's label contradicts the geometry standing over them
    # Both exit non-zero on failure and both are cheap. --force exists for a deliberate throwaway run.
    if not a.dry_run and not a.force:
        for script, why in (("tools/scene/assert_qa_fresh.py",
                             "the scene changed after the last QA render, so nobody has looked at what "
                             "you are about to photograph"),
                            ("tools/scene/check_occlusion_truth.py",
                             "a survivor's ground truth contradicts the geometry above them, so the labels "
                             "would be wrong in every frame")):
            r = subprocess.run([sys.executable, script], cwd=str(REPO), capture_output=True, text=True)
            if r.returncode != 0:
                print(f"REFUSING TO FLY: {script} exited {r.returncode} - {why}.")
                print((r.stdout or "")[-1500:])
                print((r.stderr or "")[-500:])
                print("Fix it, or pass --force for a deliberately throwaway run.")
                return 3
        print("scene freeze checks passed: QA render is current and the occlusion ground truth agrees "
              "with the geometry")

    names = [p.strip() for p in a.passes.split(",") if p.strip()]
    unknown = [n for n in names if n not in PASSES]
    if unknown:
        print(f"unknown pass(es) {unknown}; known: {sorted(PASSES)}")
        return 2

    results, t0 = [], time.time()
    for name in names:
        cfg = PASSES[name]
        sm = shutter_m(cfg["alt"], a.speed, hfov)
        out = f"{a.out_root}/seed{seed}_{name}"
        cmd = [sys.executable, "-m", "sightline.mission.survey",
               "--alt", str(cfg["alt"]), "--speed", str(a.speed),
               "--plan", "patches", "--shutter-m", str(sm),
               "--out", out, "--max-minutes", str(a.max_minutes),
               "--condition", cfg["cond"], "--time-of-day", cfg["tod"]]
        for k in ("rain", "fog", "dust"):
            if cfg.get(k) is not None:
                cmd += [f"--{k}", str(cfg[k])]
        if a.jpeg:
            cmd += ["--jpeg", str(a.jpeg)]
        if a.dry_run:
            cmd += ["--dry-run"]
        print(f"\n=== pass {name}: {cfg['alt']:.0f} m AGL, shutter {sm} m, {cfg['cond']} ===")
        print("    " + " ".join(cmd[2:]))
        r = subprocess.run(cmd, cwd=str(REPO))
        card = REPO / out / "data_card.json"
        ok = r.returncode == 0 and (a.dry_run or card.exists())
        got = json.loads(card.read_text()) if card.exists() else {}
        results.append({"pass": name, "out": out, "ok": ok, "returncode": r.returncode,
                        "frames": got.get("frames"), "boxes": got.get("total_boxes"),
                        "survivors": got.get("unique_actors_seen")})
        if not ok and not a.dry_run:
            print(f"pass {name} FAILED (exit {r.returncode}). Continuing with the rest; the campaign "
                  f"summary will say which passes exist.")

    print(f"\n=== campaign summary  (seed {seed}, domain=sim, {(time.time() - t0) / 60:.1f} min) ===")
    for x in results:
        print(f"  {x['pass']:8s} {'ok ' if x['ok'] else 'FAIL'}  frames {x['frames']}  "
              f"boxes {x['boxes']}  survivors {x['survivors']}  {x['out']}")
    tot = sum(x["boxes"] or 0 for x in results)
    print(f"  total boxes: {tot}")
    if not a.dry_run:
        (REPO / a.out_root / f"campaign_seed{seed}.json").write_text(json.dumps(
            {"seed": seed, "passes": results, "total_boxes": tot, "domain": "sim",
             "note": "every pass is one altitude and one lighting condition; both are slice axes"},
            indent=1), encoding="utf-8")
        print(f"\nNext: measure_occlusion.py --write on each run, then validate.py, then quality_report.py, "
              f"then LOOK at contact_sheet.py output.")
    return 0 if all(x["ok"] for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
