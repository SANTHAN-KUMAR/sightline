"""Give the simulator a physically-meaningful thermal image (SOLUTION_DOC 5.1 step 7, 5.5b; day-1 test 3).

    uv run python tools/capture/thermal_ids.py            # assign, verify, write the table
    uv run python tools/capture/thermal_ids.py --time night

Cosys-AirSim's `Infrared` image type is NOT radiometry: it renders each object's segmentation ID straight into
grey, so object ID 42 comes out as (42, 42, 42) (docs/CONTEXT.md section 6). That is usually written off as a
toy, but it is exactly enough to carry a temperature field: choose the IDs so that ID *is* temperature under a
fixed, documented scale, and the 8-bit Infrared pass becomes a quantised thermal image that can be decoded back
to degrees Celsius.

    id = round((T_C - T_MIN) / T_STEP)          T_MIN = -20 C, T_STEP = 0.5 C  ->  0..255 covers -20..107.5 C

That scale is stored in `data/scene/thermal_table.json` alongside every object's assigned temperature, so the
capture pipeline can invert it and emit a **radiometric** frame in centi-kelvin, which is what
`FrameBundle.thermal_is_radiometric` promises and what section 5.5b's absolute-temperature filter needs.

The temperatures come from section 2.3's causal model rather than being decorative:
  * skin sits near 33-35 C, and that is the whole basis of thermal detection;
  * **wet skin and wet clothing read far cooler** through evaporative cooling - a half-submerged survivor is
    barely warmer than the water, which is why thermal is not a free win in a flood;
  * flood water is a large, uniform, cool background (26-28 C in a monsoon);
  * **sun-heated metal and concrete roofs reach 50-70 C**, far hotter than a person. At midday they swamp a
    survivor's contrast, which is precisely the false-positive slice section 5.5 warns about, and the reason
    section 5.5b wants absolute temperature rather than a contrast-stretched 8-bit picture.
Switching `--time` between day/dusk/night moves the surfaces but not the people, so the crossover windows of
section 2.3 rows 20 and 23 fall out of the model instead of being hand-waved.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

with contextlib.redirect_stdout(io.StringIO()):
    import cosysairsim as airsim

T_MIN, T_STEP, ID_MAX = -20.0, 0.5, 255


def to_id(t_c: float) -> int:
    return int(max(0, min(ID_MAX, round((t_c - T_MIN) / T_STEP))))


def to_c(i: int) -> float:
    return T_MIN + i * T_STEP


#: surface -> temperature in C, by time of day. Sun-driven surfaces swing; bodies and water barely do.
THERMAL: dict[str, dict[str, float]] = {
    #                       day    dusk   night
    "skin_dry":       dict(day=34.0, dusk=34.0, night=33.5),
    "skin_wet":       dict(day=29.0, dusk=28.5, night=28.0),   # evaporative cooling: the hard case
    "skin_immersed":  dict(day=27.5, dusk=27.5, night=27.2),   # barely above the water
    "water":          dict(day=27.5, dusk=27.2, night=26.8),
    "mud":            dict(day=32.0, dusk=28.5, night=24.5),
    "vegetation":     dict(day=29.0, dusk=26.5, night=22.5),
    "concrete":       dict(day=52.0, dusk=34.0, night=24.0),   # swamps a person at midday
    "metal_roof":     dict(day=64.0, dusk=33.0, night=20.5),   # the classic thermal false positive
    "tile_roof":      dict(day=48.0, dusk=33.0, night=23.0),
    "wood":           dict(day=38.0, dusk=30.0, night=23.0),
    "plastic":        dict(day=44.0, dusk=31.0, night=22.0),
    "rock":           dict(day=40.0, dusk=31.0, night=23.5),
    "vehicle":        dict(day=55.0, dusk=33.0, night=22.0),
    "background":     dict(day=30.0, dusk=27.0, night=23.0),
}

#: object-name regex -> surface. First match wins, so the specific patterns come first.
RULES: list[tuple[str, str]] = [
    (r"^Human_\d+", "skin_dry"),          # refined per-actor below from its submersion class
    (r"^Animal_\d+", "skin_dry"),
    (r"FloodWater", "water"),
    (r"Ground", "mud"),
    (r"SM_[A-E]_(tile)_", "tile_roof"),
    (r"SM_[A-E]_(sheet)_", "metal_roof"),
    (r"SM_[A-E]_(flat)_", "concrete"),
    (r"covered_car|car_", "vehicle"),
    (r"fern|leaf|leaves|grass", "vegetation"),
    (r"rock|stone|boulder|namaqualand", "rock"),
    (r"trunk|stump|branch|plank|crate|wood", "wood"),
    (r"barrel|jerrycan|plastic|cardboard|bag|tyre|tire", "plastic"),
]


def surface_for(name: str) -> str:
    for pat, surf in RULES:
        if re.search(pat, name, re.I):
            return surf
    return "background"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", choices=("day", "dusk", "night"), default="day")
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()

    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    sub_by_id = {f"Human_{x['id']:03d}": x["submersion"] for x in truth["actors"]}

    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()
    names = c.simListInstanceSegmentationObjects()

    assigned: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for n in names:
        surf = surface_for(n)
        m = re.match(r"^(Human_\d+)_\d+$", n)
        if m:
            # A wet or immersed survivor is much cooler than a dry one: this is the physics that decides
            # whether thermal helps at all in a flood (section 2.3).
            sub = sub_by_id.get(m.group(1), "unknown")
            surf = {"half": "skin_immersed", "head_only": "skin_immersed",
                    "partial": "skin_wet", "wet": "skin_wet"}.get(sub, "skin_dry")
        t = THERMAL[surf][a.time]
        i = to_id(t)
        ok = c.simSetSegmentationObjectID(n, i, False)
        if not ok:                                  # some engine-internal objects refuse an ID; not fatal
            counts["refused"] = counts.get("refused", 0) + 1
            continue
        assigned[n] = {"surface": surf, "temp_c": t, "id": i}
        counts[surf] = counts.get(surf, 0) + 1

    table = {
        "generated_by": "tools/capture/thermal_ids.py",
        "time_of_day": a.time,
        "encoding": {"t_min_c": T_MIN, "t_step_c": T_STEP, "id_max": ID_MAX,
                     "decode": "temp_c = t_min_c + id * t_step_c",
                     "note": ("Cosys Infrared renders the segmentation ID directly as grey, so grey == id and "
                              "this scale makes the 8-bit IR pass decodable to absolute temperature.")},
        "surfaces": {k: v[a.time] for k, v in THERMAL.items()},
        "counts": counts,
        "objects": assigned,
    }
    (REPO / "data/scene/thermal_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"time of day: {a.time}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        t = THERMAL.get(k, {}).get(a.time)
        print(f"  {k:16s} {v:5d} objects" + (f"  {t:5.1f} C -> id {to_id(t):3d}" if t is not None else ""))

    if a.verify:
        req = [airsim.ImageRequest("survey", airsim.ImageType.Infrared, False, False)]
        with contextlib.redirect_stdout(io.StringIO()):
            c.simGetImages(req)                     # warm-up: the first call per process is unconverted
            r = c.simGetImages(req)[0]
        ir = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)[:, :, 0]
        vals, cnt = np.unique(ir, return_counts=True)
        print(f"\nIR frame {r.width}x{r.height}: {len(vals)} distinct greys")
        for v, n in sorted(zip(vals, cnt), key=lambda t: -t[1])[:6]:
            hit = [k for k, o in assigned.items() if o["id"] == int(v)]
            print(f"  grey {int(v):3d} -> {to_c(int(v)):6.1f} C  {100 * n / ir.size:5.1f}% of frame"
                  f"  e.g. {hit[0] if hit else '(unassigned)'}")
        skin = to_id(THERMAL["skin_dry"][a.time])
        if skin in set(int(v) for v in vals):
            print(f"\n  dry skin ({THERMAL['skin_dry'][a.time]} C, id {skin}) IS present in this frame")
        else:
            print(f"\n  no dry-skin pixels in this frame (id {skin}) - aim at a survivor to see one")
    print("\nwritten to data/scene/thermal_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
