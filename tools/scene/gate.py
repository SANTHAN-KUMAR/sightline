"""The environment gate: run every asset check and give ONE verdict. The scene is not finished until this
exits 0.

    uv run python tools/scene/gate.py              # everything that does not need the editor
    uv run python tools/scene/gate.py --all        # including the checks that drive the live editor
    uv run python tools/scene/gate.py --json out.json

Nine separate checkers grew up in this project, one per lane, each excellent and each answering only for its
own corner. That is exactly how a scene ends up with every lane reporting green while the thing as a whole is
broken - and it already happened here: on 2026-09-11 the vegetation, rubble and utilities lanes all reported
success while the canopy was on engine placeholder materials, 29 % of the settlement was crushed to black,
and the dataset had been captured from a scene that no longer existed.

So this runs all of them and refuses to summarise generously. A checker that cannot run is NOT a pass - it is
reported as UNKNOWN and fails the gate, because `tools/capture/validate.py` spent a session printing
"all checks passed - dataset is clean" while silently skipping the one check that mattered.

Each entry says what it proves, so a red line points at a thing rather than at a script name.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: (name, argv, needs_editor, what it proves)
CHECKS: list[tuple[str, list[str], bool, str]] = [
    ("realism-dryrun", ["tools/scene/check_realism.py"], False,
     "every build_*.py script runs to completion against a strict mock of the UE API, so a mis-typed engine "
     "name cannot reach the editor"),
    ("occlusion-truth", ["tools/scene/check_occlusion_truth.py"], False,
     "no survivor recorded `occlusion: 0` has geometry above them, and every survivor recorded "
     "aerially_detectable=false does - the labels and the geometry agree"),
    ("env-assets", ["tools/scene/check_env_assets.py"], False,
     "every downloaded asset is complete and matches the size and md5 the source reported"),
    ("lighting", ["tools/scene/measure_lighting.py",
                  "_artifacts/editor_shots/qa_1_valley.png",
                  "_artifacts/editor_shots/qa_4_nadir45.png",
                  "_artifacts/editor_shots/qa_2_settlement.png"], False,
     "shadows are inside the physical range for an outdoor scene and no meaningful area is crushed to "
     "black, so a survivor in shadow is still visible to the detector and to a human"),
    ("qa-fresh", ["tools/scene/assert_qa_fresh.py"], False,
     "nothing in the scene has changed since a human last looked at a render of it"),
    # --- these drive the live editor, so they are opt-in and must hold the editor lock -------------------
    ("vegetation", ["tools/scene/check_vegetation.py"], True,
     "all 5,508 canopy instances match the layout and no crown sits over an unoccluded survivor"),
    ("rubble", ["tools/scene/check_rubble.py"], True,
     "every rubble variant's instance count matches the layout, Nanite is off, and every material compiles"),
    ("utilities", ["EDITOR:tools/scene/check_utilities_placed.py"], True,
     "poles, wires, boats and the foam ribbon are where utilities.json says, measured through the actor "
     "transform rather than a bounding box"),
    ("foliage", ["tools/scene/check_foliage_materials.py"], True,
     "no tree material is an engine placeholder, none is translucent, and the canopy is not crushed"),
    ("sky", ["tools/scene/check_sky.py"], True,
     "the visible sky is consistent with the light that is actually falling on the scene"),
]


def run(argv: list[str], timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run([sys.executable, *argv], cwd=str(REPO), capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or ""))
    except subprocess.TimeoutExpired:
        return 124, f"TIMED OUT after {timeout}s"
    except FileNotFoundError as exc:
        return 127, f"not found: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run the checks that drive the live editor")
    ap.add_argument("--only", default="", help="comma-separated check names")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", default="")
    ap.add_argument("--record", default="", help="NAME=PASS|FAIL for a check run inside the editor")
    a = ap.parse_args()

    if a.record:
        nm, _, st = a.record.partition("=")
        stamp = REPO / "_artifacts" / "env_gate_editor.json"
        cur = json.loads(stamp.read_text()) if stamp.exists() else {}
        cur[nm] = {"status": st.upper() or "PASS", "t_utc": time.time(),
                   "detail": "recorded by hand after an in-editor run"}
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        print(f"recorded {nm} = {st.upper() or 'PASS'}")
        return 0

    only = {x.strip() for x in a.only.split(",") if x.strip()}
    rows, t0 = [], time.time()
    for name, argv, needs_editor, proves in CHECKS:
        if only and name not in only:
            continue
        script = REPO / argv[0].split(":", 1)[-1]      # strip an EDITOR: prefix before checking the path
        if not script.exists():
            rows.append({"name": name, "status": "UNKNOWN", "code": 127, "proves": proves,
                         "detail": f"{argv[0]} does not exist"})
            print(f"{name:16s} UNKNOWN  ({argv[0]} does not exist)")
            continue
        if needs_editor and not a.all:
            rows.append({"name": name, "status": "SKIPPED", "code": None, "proves": proves,
                         "detail": "needs the live editor; re-run with --all"})
            print(f"{name:16s} skipped  (needs the editor - re-run with --all)")
            continue
        if argv[0].startswith("EDITOR:"):
            # This checker imports `unreal` at module scope, so it only runs INSIDE the editor. Running it
            # as a host subprocess gives ModuleNotFoundError, which is NOT evidence about the scene. The
            # gate reads a stamp instead, and a stamp older than the level does not count.
            script = argv[0].split(":", 1)[1]
            stamp = REPO / "_artifacts" / "env_gate_editor.json"
            rec = json.loads(stamp.read_text()).get(name) if stamp.exists() else None
            umap = REPO / "sim/SightlineSim/Content/Sightline/Maps/FloodValley.umap"
            fresh = rec and umap.exists() and rec.get("t_utc", 0) >= umap.stat().st_mtime
            if rec and rec.get("status") == "PASS" and fresh:
                rows.append({"name": name, "status": "PASS", "code": 0, "proves": proves,
                             "detail": f"stamped {rec.get('detail', '')}"})
                print(f"{name:16s} PASS     (stamped in-editor run, newer than the level)")
            else:
                why = ("no stamp" if not rec else
                       "stamp is older than the level" if not fresh else f"stamp says {rec.get('status')}")
                rows.append({"name": name, "status": "UNKNOWN", "code": 127, "proves": proves,
                             "detail": f"{why}; run in the editor: ue_python exec {script}, then "
                                       f"tools/scene/gate.py --record {name}=PASS"})
                print(f"{name:16s} UNKNOWN  ({why} - run in-editor, then --record {name}=PASS)")
            continue
        code, out = run(argv, a.timeout)
        status = "PASS" if code == 0 else ("UNKNOWN" if code in (124, 127) else "FAIL")
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-3:]
        rows.append({"name": name, "status": status, "code": code, "proves": proves,
                     "detail": " | ".join(tail)[:400]})
        print(f"{name:16s} {status:8s} exit {code}")
        if status != "PASS":
            for ln in tail:
                print(f"                 {ln[:150]}")

    ran = [r for r in rows if r["status"] in ("PASS", "FAIL", "UNKNOWN")]
    bad = [r for r in ran if r["status"] != "PASS"]
    skipped = [r for r in rows if r["status"] == "SKIPPED"]

    print("\n" + "=" * 78)
    print(f"{len([r for r in ran if r['status'] == 'PASS'])}/{len(ran)} checks passed "
          f"in {time.time() - t0:.0f}s  (domain=sim)")
    for r in bad:
        print(f"\n{r['status']}  {r['name']}: {r['proves']}")
        print(f"       {r['detail'][:300]}")
    if skipped:
        print(f"\nnot run (needs the editor): {', '.join(r['name'] for r in skipped)}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"by": "tools/scene/gate.py", "t_utc": time.time(), "domain": "sim",
             "passed": len([r for r in ran if r["status"] == "PASS"]), "ran": len(ran),
             "checks": rows}, indent=1), encoding="utf-8")

    if bad:
        print(f"\nENVIRONMENT IS NOT FINISHED: {len(bad)} check(s) not passing. "
              f"An UNKNOWN counts against the gate - a check that did not run is not a check that passed.")
        return 1
    if skipped and not a.all:
        print("\nHost-side checks pass. The editor-side checks have NOT run - use --all with the editor up "
              "before calling the environment finished.")
        return 2
    print("\nENVIRONMENT GATE GREEN - every asset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
