"""Radiometric thermal via a Cosys-AirSim ANNOTATION layer (F9b, SOLUTION_DOC §5.5b). Replaces thermal_ids.py.

    ue must be running with `-settings=D:\\Sightline\\sim\\settings\\dataset_thermal.json`, PIE ON
    uv run python tools/capture/thermal_annotation.py                 # assign, verify, write the table
    uv run python tools/capture/thermal_annotation.py --time night

================================================================================================
WHY THIS EXISTS: `thermal_ids.py` WORKED BY DESTROYING THE DATASET
================================================================================================
The old script encoded temperature into the **instance-segmentation ID**, because Cosys-AirSim's `Infrared`
pass renders an object's segmentation id straight into grey. It is a clever trick and it is fatal: every
object sharing a temperature shares an id, and therefore a colour. Run mid-flight on 2026-09-10 it collapsed
the instance palette, the flood plane rendered in a survivor's colour, and 78 % of the resulting boxes were
physically impossible - one "person" spanned 67.8 x 38.1 m of ground. A 336-frame dataset was lost. The
script now refuses to run without `--i-know-this-destroys-instance-segmentation`.

There is a proper channel and it was there all along. Cosys-AirSim replaced `Infrared` with configurable
**annotation layers** (`ImageType.Annotation = 11`), and a GREYSCALE layer (`AnnotatorType.Greyscale = 1`) is
exactly a per-object scalar field. It is a SEPARATE channel from instance segmentation, so temperature and
instance ids coexist and neither can corrupt the other. That is the whole difference: not a better encoding,
a channel that was never shared.

This is also why `Infrared` renders 100 % grey 0 in this project despite every object accepting a
segmentation id - the feature it depends on has been superseded.

================================================================================================
THE PHYSICS (unchanged from thermal_ids.py, which had it right)
================================================================================================
Temperatures come from §2.3's causal model rather than being decorative:
  * skin sits near 33-35 C, and that is the whole basis of thermal detection;
  * **wet skin and wet clothing read far cooler** through evaporative cooling - a half-submerged survivor is
    barely warmer than the water, which is why thermal is not a free win in a flood;
  * flood water is a large, uniform, cool background (26-28 C in a monsoon);
  * **sun-heated metal and concrete roofs reach 50-70 C**, far hotter than a person, and at midday they swamp
    a survivor's contrast. That is the false-positive slice §5.5 warns about and the reason §5.5b wants
    absolute temperature rather than a contrast-stretched 8-bit picture.

    id = round((T_C - T_MIN) / T_STEP)     T_MIN = -20 C, T_STEP = 0.5 C  ->  0..255 covers -20..107.5 C

The scale is written to `data/scene/thermal_table.json` so the capture can invert it and emit a
**radiometric** frame in centi-kelvin, which is what `FrameBundle.thermal_is_radiometric` promises.
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

LAYER = "thermal"
T_MIN, T_STEP, ID_MAX = -20.0, 0.5, 255


def to_grey(t_c: float) -> int:
    return int(max(0, min(ID_MAX, round((t_c - T_MIN) / T_STEP))))


def to_c(g: int) -> float:
    return T_MIN + g * T_STEP


#: surface -> temperature in C, by time of day. Sun-driven surfaces swing; bodies and water barely do.
THERMAL: dict[str, dict[str, float]] = {
    "skin_dry":      dict(day=34.0, dusk=34.0, night=33.5),
    "skin_wet":      dict(day=29.0, dusk=28.5, night=28.0),   # evaporative cooling: the hard case
    "skin_immersed": dict(day=27.5, dusk=27.5, night=27.2),   # barely above the water
    "water":         dict(day=27.5, dusk=27.2, night=26.8),
    "mud":           dict(day=32.0, dusk=28.5, night=24.5),
    "vegetation":    dict(day=29.0, dusk=26.5, night=22.5),
    "concrete":      dict(day=52.0, dusk=34.0, night=24.0),   # swamps a person at midday
    "metal_roof":    dict(day=64.0, dusk=33.0, night=20.5),   # the classic thermal false positive
    "tile_roof":     dict(day=48.0, dusk=33.0, night=23.0),
    "wood":          dict(day=38.0, dusk=30.0, night=23.0),
    "plastic":       dict(day=44.0, dusk=31.0, night=22.0),
    "rock":          dict(day=40.0, dusk=31.0, night=23.5),
    "vehicle":       dict(day=55.0, dusk=33.0, night=22.0),
    "background":    dict(day=30.0, dusk=27.0, night=23.0),
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
    (r"Veg_|fern|leaf|leaves|grass|tree", "vegetation"),
    (r"Rubble_|rock|stone|boulder|namaqualand|concrete|masonry|slab", "rock"),
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
    ap.add_argument("--layer", default=LAYER)
    a = ap.parse_args()

    truth = json.loads((REPO / "data/scene/actors.json").read_text())
    sub_by_id = {f"Human_{x['id']:03d}": x["submersion"] for x in truth["actors"]}

    with contextlib.redirect_stdout(io.StringIO()):
        c = airsim.MultirotorClient()
        c.confirmConnection()

    # The layer must exist in settings.json BEFORE the editor starts. If it does not, every set below is
    # accepted and does nothing - which is exactly the silent failure this project keeps being bitten by.
    try:
        known = list(c.simListAnnotationObjects(a.layer))
    except Exception as exc:                                  # noqa: BLE001
        print(f"FATAL: annotation layer {a.layer!r} is not available ({type(exc).__name__}: {exc}).\n\n"
              f"The layer has to be declared in the settings JSON the editor was STARTED with, and the\n"
              f"editor must be restarted for it to take effect:\n"
              f'    "Annotation": [{{"Name": "{a.layer}", "Type": 1, "Default": true, "SetDirect": true}}]\n'
              f"plus a capture entry with \"ImageType\": 11, \"Annotation\": \"{a.layer}\".\n"
              f"`sim/settings/dataset_thermal.json` already has both; relaunch with -settings= that file.")
        return 2
    print(f"annotation layer {a.layer!r} is live: {len(known)} objects registered")

    names = c.simListInstanceSegmentationObjects()
    assigned: dict[str, dict] = {}
    counts: dict[str, int] = {}
    refused = 0
    for n in names:
        surf = surface_for(n)
        m = re.match(r"^(Human_\d+)_\d+$", n)
        if m:
            # A wet or immersed survivor is much cooler than a dry one: this is the physics that decides
            # whether thermal helps at all in a flood (§2.3).
            sub = sub_by_id.get(m.group(1), "unknown")
            surf = {"half": "skin_immersed", "head_only": "skin_immersed",
                    "partial": "skin_wet", "wet": "skin_wet"}.get(sub, "skin_dry")
        t = THERMAL[surf][a.time]
        if not c.simSetAnnotationObjectValue(a.layer, n, to_grey(t)):
            refused += 1
            continue
        assigned[n] = {"surface": surf, "temp_c": t, "grey": to_grey(t)}
        counts[surf] = counts.get(surf, 0) + 1

    print(f"time of day: {a.time}   assigned {len(assigned)}, refused {refused}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        t = THERMAL[k][a.time]
        print(f"  {k:16s} {v:5d} objects   {t:5.1f} C -> grey {to_grey(t):3d}")

    # --- PROVE IT: read a value back, and render a frame -------------------------------------------------
    # `simSetAnnotationObjectValue` returning True is not evidence. Neither is a frame that is merely
    # non-empty: the old Infrared pass returned a perfectly valid, perfectly grey-0 image for weeks.
    probe = next((n for n in assigned if n.startswith("Human_")), None)
    if probe:
        got = c.simGetAnnotationObjectValue(a.layer, probe)
        want = assigned[probe]["grey"]
        print(f"\nread-back {probe}: {got} (set {want}) -> {'OK' if int(got) == want else 'MISMATCH'}")

    req = [airsim.ImageRequest("survey", airsim.ImageType.Annotation, False, False, a.layer)]
    with contextlib.redirect_stdout(io.StringIO()):
        c.simGetImages(req)                                   # warm-up: the first call is unconverted
        r = c.simGetImages(req)[0]
    im = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)[:, :, 0]
    vals, cnt = np.unique(im, return_counts=True)
    print(f"annotation frame {r.width}x{r.height}: {len(vals)} distinct greys")
    for v, n_ in sorted(zip(vals, cnt), key=lambda t: -t[1])[:6]:
        hit = [k for k, o in assigned.items() if o["grey"] == int(v)]
        print(f"  grey {int(v):3d} -> {to_c(int(v)):6.1f} C  {100 * n_ / im.size:5.1f}% of frame"
              f"  e.g. {hit[0] if hit else '(unassigned)'}")
    if len(vals) <= 1:
        print("\nFAIL: the annotation frame carries a single value. That is the same symptom Infrared had.\n"
              "      The layer is declared but nothing is rendering it - check the camera's capture entry\n"
              '      has {"ImageType": 11, "Annotation": "%s"}.' % a.layer)
        return 1

    (REPO / "data/scene/thermal_table.json").write_text(json.dumps({
        "generated_by": "tools/capture/thermal_annotation.py",
        "channel": f"annotation layer {a.layer!r} (greyscale), ImageType.Annotation = 11",
        "why_not_segmentation_ids": "thermal_ids.py encoded temperature as the SEGMENTATION id, which is "
                                    "shared across objects at the same temperature and destroyed instance "
                                    "segmentation; an annotation layer is a separate channel",
        "time_of_day": a.time,
        "encoding": {"t_min_c": T_MIN, "t_step_c": T_STEP, "id_max": ID_MAX,
                     "decode": "temp_c = t_min_c + grey * t_step_c"},
        "surfaces": {k: v[a.time] for k, v in THERMAL.items()},
        "counts": counts, "objects": assigned,
    }, indent=1), encoding="utf-8")
    print(f"\nwritten to data/scene/thermal_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
