# Structural-collapse lane — the rubble field on the deposit fan

Written offline on 2026-09-11 (no editor, no simulator, no `mcp__sightline__*` / `mcp__unreal__*` calls). The
host-side generator was run for real and every number below is measured, not estimated. The target is
`docs/SCENE_REFERENCE.md` **Reference B**: a collapsed multi-storey building — broken floor slabs at every angle,
blocks, masonry, protruding rebar, voids between the plates, dust-desaturated tan/grey, belongings as the only
colour.

## Files added (nothing existing was edited)

| File | Side | Purpose |
|---|---|---|
| `tools/scene/gen_rubble.py` | host (needs numpy) | 40 OBJ meshes + `data/scene/rubble_layout.json` + the offline renders + the checks |
| `tools/scene/build_rubble.py` | editor (no numpy) | imports the OBJs, builds `M_Rubble_Master` + 7 instances, places the field, idempotent |
| `docs/lanes/rubble.md` | — | this report |

Generated outputs: `data/scene/rubble/*.obj` (40 files, 14.0 MB), `data/scene/rubble_layout.json` (666,752 bytes),
`_artifacts/rubble/*.png` (11 images).

## Run order (exact)

```
# 1. host (safe while the editor is busy; ~2 min with renders, ~10 s with --no-render; no GPU, no torch)
D:\Tools\uv\uv.exe run python tools\scene\gen_rubble.py            # --seed 71 default; renders + checks
#    it EXITS NON-ZERO if any check fails. Do not proceed past a non-zero exit.

# 2. LOOK at _artifacts\rubble\*.png before importing anything

# 3. editor, PIE STOPPED (idempotent, re-runnable)
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_rubble.py').read())"

# 4. LOOK again
ue_python code="exec(open(r'D:\Sightline\tools\scene\qa_shots.py').read())"
```

`build_rubble.py` has no ordering dependency on the other build scripts. It owns `/Game/Sightline/Rubble/**` and
touches nothing else — in particular it does **not** rebuild `M_PBR_Master` (owned by `build_buildings.py` /
`build_roof_materials.py`) or the terrain/water materials (`build_materials.py`). It builds its own
`M_Rubble_Master`, so the "one owner per material" rule from the scene-polish lane still holds.

## What the generator produces

**Mesh library — 40 meshes, 113,524 triangles**, centimetres, +Z up, metre-true planar UVs, `usemtl` groups that
become UE material slots (`concrete`, `masonry`, `rebar`, `fabric`, `wood`, `metal`):

| Family | Variants | Triangles each | What it is |
|---|---|---|---|
| `P_*` pile | 12 | 3,684 – 14,016 | an assembled rubble pile: a mound of blocks with floor slabs leaning up it, 5.0–13.6 m across, 3.4–5.7 m tall |
| `D_*` mat | 8 | 1,544 – 1,804 | ~90 small fragments strewn over an irregular 7–11 m patch: the continuous ground cover between the piles |
| `S_*` slab | 6 | 324 – 684 | the signature shape: a thin irregular plate 1.4–3.7 m across, 0.18–0.28 m thick, gently sagged, with bent rebar protruding 0.48–1.38 m from the fracture edges |
| `B_*` block | 4 | 48 | angular concrete lumps 0.20–0.87 m |
| `M_*` masonry | 3 | 48 – 144 | one to three bonded, chipped masonry units |
| `R_*` rebar | 3 | 156 – 264 | a knot of bar torn out of a slab, 1.2–2.0 m across |
| `G_*` belonging | 4 | 24 – 72 | mattress, cabinet, appliance, chair — the only colour in the field |

The slab is built as a star-shaped polar outline: 4–6 corner anchors joined by chords, some kept clean (an
original cast edge) and the rest given radial fracture noise, then extruded through ring strips so a sag/twist
can be applied. That is what makes it read as a broken floor slab rather than an extruded blob.

**Layout — 1,799 actors, 2,407,920 triangles (96 % of the 2.5 M budget)**

| group | n | rule and the line behind it |
|---|---|---|
| `pile` | 130 | the field. Confined to the upper 50 % of the fan, density `(1-f/0.5)^1.2`, allowed to overlap. Doc §2.2 zone 1 / `gen_props.py`: a debris flow drops its coarsest, heaviest load first, so the structural load lands at the apex and thins downstream. Mean downstream fraction **0.229**. |
| `mat` | 543 | ground cover between the piles, upper 62 % of the fan (see "What the renders changed" below) |
| `loose` | 900 | slabs/blocks/masonry/rebar/belongings scattered across the whole fan floor, same apex bias |
| `obstacle` | 30 | rafted against the **upstream (+north) face** of every house on or within 30 m of the fan — the same wrack-line logic `gen_props.py` uses for floating debris |
| `bank` | 90 | jammed against the cut bank of the incised channel where the fan meets it |
| `void` | 100 | a ring of slabs and blocks around every aerially-detectable survivor on the fan, with a **1.35 m clear radius**: doc §2.3 `trapped` posture and the 25–75 % occlusion slices. Nothing is placed inside the clear radius, so the ground truth stays honest |
| `burial` | 6 | three collapsed floor slabs over each of `Human_055` and `Human_062`, the two actors marked `aerially_detectable: false`. Doc **§2.7**: a concrete slab is the physical reason aerial search cannot clear that cell; the boulder cap `gen_props.py` already builds was a weaker claim |

The **channel is left scoured clean** (0 items), the rubble stays inside a fan-floor mask (slope < 18°, less than
25 m above flood stage — the raw `fan` mask in `gen_terrain.py` is an ellipse of added thickness that also covers
valley wall up to 105 m above flood stage), and nothing sits inside a house footprint.

**Triangle cost, measured (all `sim`):** 2,407,920 rubble triangles added to the ~524 k terrain + ~4.15 M debris
props already in the scene → ~7.08 M total. The generator hard-caps at 2.5 M and refuses to exceed it; each
placement stage gets an explicit share of the budget (piles 49 %, mats to 88 %, loose to 94 %, then the
mission-critical `obstacle`/`bank`/`void`/`burial` groups keep the full budget).

## The renders — paths, and what I actually saw

All in `D:\Sightline\_artifacts\rubble\`. They come from a z-buffered flat-shaded software rasteriser written
into `gen_rubble.py` (numpy + PIL), with near-plane clipping and a 1.75 m human for scale. **Flat shading, no
textures — this is a geometry check, not a look check.**

| Image | What I saw |
|---|---|
| `variants_00.png`, `variants_01.png` | the 12 piles from iso/side/top. Dense mounds of overlapping tilted plates with rebar spikes around the crest, blocks at the margins, a blue mattress in `P_large_2`. 3.4–5.7 m tall against the 1.75 m figure — correct scale for a collapsed multi-storey building |
| `variants_02.png` | the 6 slabs. Genuinely irregular broken outlines, visible edge thickness, bent rebar rods protruding from the fracture edges and clearly readable in the top view. No slivers, no degenerate plates |
| `variants_03.png`, `variants_04.png` | blocks and the 8 mats. The mats read as irregular 7–11 m carpets of small angular fragments lying on the ground, < 1 m tall |
| `variants_05.png` | masonry (broken bonded brick units) and the rebar tangles (a concrete lump with 3–6 bars springing out, up to 1.6 m tall) |
| `variants_06.png` | the four belongings |
| **`scene_ground_close.png`** | **the Reference-B shot.** Eye at 1.6 m, 32 m back. A continuous chaotic mass of broken concrete plates at every angle, blocks and masonry filling between them, rebar protruding everywhere, mud showing through in the foreground, one blue mattress as the only colour. This is the picture that says the lane worked |
| `scene_ground_oblique.png` | eye 1.8 m, 75 m back. The field fills the frame from a rescuer's standoff and thins to bare mud in the foreground |
| `scene_aerial_45m.png` | 45 m AGL oblique, the survey slice. 1.89 M triangles in frame. The rubble forms one coherent lobe across the fan floor, thinning east against the valley wall and upstream — it reads as a debris-flow deposit carrying a settlement's structural load |
| `scene_aerial_nadir_60m.png` | 60 m nadir. A continuous irregular field of plate fragments with rebar flecks, thinning to bare mud on the east edge |

## What the renders changed (the point of looking)

Four things passed every programmatic check and were still wrong. Each was found by opening a PNG:

1. **The first nadir render showed 210 isolated grey "flowers" on clean mud.** The piles were spread evenly over
   the whole fan with nothing covering the ground between them, and each pile's slabs radiated from its centre,
   so from directly above they read as rosettes. Fixes: the `D_*` mat family (~90 fragments per 1.6 k triangles —
   the only way to buy continuous ground cover inside the budget); piles confined to the upper half of the fan
   and allowed to overlap; and three slab attitudes per pile (60 % lean-tos, 25 % near-horizontal caps, 15 %
   wedged on edge) instead of lean-tos only. This is now enforced by a check: **the apex third of the fan is 48 %
   covered, the mid fan 14 %.**
2. **Cap blocks floated in mid-air** above the piles — they were placed on the analytic dome height, which is not
   where the slab surface actually is. They are now sunk into the crest.
3. **The renderer itself was lying.** It walked the item list in order and stopped at a triangle cap; because the
   piles are written first they ate the whole cap and almost every mat and loose fragment was silently dropped
   from the picture. It now gathers nearest-first with per-view frustum-sized radii.
4. **The ground-level camera sampled its altitude at the focus, not at its own position**, so it sat 0.52 m above
   the terrain under it and the near ground fell behind the near plane — the render showed rubble floating over a
   strip of sky. Fixed, plus proper near-plane clipping instead of dropping the triangle.

## The checks (they can fail, and they did)

`gen_rubble.py` runs 22 checks at the end of every run and **exits non-zero** if any fails. Final run, all `sim`:

```
  [PASS] basis() maps local +x onto the requested direction
  [PASS] no non-finite vertices  -- 0 meshes with NaN/inf
  [PASS] no zero-area faces (min triangle >= 0.2 cm^2)
  [PASS] no degenerate slivers (aspect ratio < 400)
  [PASS] every usemtl group is a known material slot
  [PASS] slab variants are thin plates 1.5-4 m across, 0.15-0.25 m thick
  [PASS] rebar protrudes from most slab variants  -- 5/6 slab variants carry a rebar slot
  [PASS] rebar sticks out at least 0.15 m past the concrete  -- 0.48 to 1.38 m
  [PASS] block variants are 0.2-0.8 m angular lumps
  [PASS] every rubble pile contains vertical voids >= 0.35 m  -- 7-25 % of sampled columns per pile
  [PASS] every field item sits inside the fan-floor mask  -- 0 outside
  [PASS] the scoured channel is left clean  -- 0 in the channel
  [PASS] piles only on ground flatter than 10 deg  -- 0 too steep
  [PASS] nothing floats or sinks more than 1 m from the fan surface  -- 0 bad
  [PASS] the 1.35 m void around every detectable survivor is respected  -- 0 intrusions
  [PASS] every 2.7 burial-boundary survivor carries a slab over them  -- Human_055: 3, Human_062: 3
  [PASS] nothing is placed inside a house footprint  -- 0 inside
  [PASS] triangle budget respected (2,407,920 <= 2,500,000)
  [PASS] piles are biased to the fan apex (mean downstream fraction < 0.45)  -- mean f = 0.2291
  [PASS] the field is large enough to read as a collapse zone (>= 800 items, >= 120 piles)
  [PASS] the apex third of the fan is >= 45 % covered by rubble  -- apex third 48 %, mid fan 14 %
  [PASS] coverage still thins downstream (apex third denser than the mid fan)  -- 48 % vs 14 %
```

**Proof they can fail** (`gen_rubble.py --no-render --max-tris 400000`, exit code 1):

```
  [FAIL] every 2.7 burial-boundary survivor carries a slab over them  -- [('Human_055', 0), ('Human_062', 0)]
  [FAIL] the field is large enough to read as a collapse zone (>= 800 items, >= 120 piles)
  [FAIL] the apex third of the fan is >= 45 % covered by rubble  -- apex third 11 % covered, mid fan 2 %
3 CHECK(S) FAILED - do not import this rubble field.
EXIT CODE = 1
```

They also failed for real four times during development and each failure was a genuine bug:
* slab bounding boxes included the protruding rebar, so the "thin plate" test was measuring the wrong box — and
  behind it, the sag coefficients summed and folded a 3.4 m slab into a 1.13 m taco;
* an unclipped corner jitter collapsed two corners of a small lump into a 0.09 cm² sliver;
* two of 1,300 loose items were written **1.7 m underground**: the terrain was sampled at full precision and the
  position stored rounded to 2 dp, which occasionally lands in a neighbouring 4 m cell;
* the ground-cover mats ate 100 % of the triangle budget and the six §2.7 burial slabs were never placed at all.

**Determinism proven.** Two consecutive runs: OBJ bytes `sha256 4722af67e880e984…` identical, and
`rubble_layout.json` byte-identical (666,752 bytes both runs). One `default_rng(seed)` split with `.spawn(2)` so
the mesh library and the layout draw from independent streams.

**Units verified.** `data/scene/rubble/S_slab_3.obj` header reads `CENTIMETRES, +Z up`, vertices are in the
hundreds (e.g. `v -111.72 -28.48 -11.36`), and the mesh measures 3.35 × 3.23 × 0.83 m. `P_large_1` measures
13.63 × 12.30 × 5.66 m with `min z = -1.57 m` — the piles are deliberately buried 0.4 + 0.11 · radius metres so
they never show a gap where the fan surface slopes away under them.

## What only an editor can confirm — I could not check these

1. **That the materials compile and the field is not the grey WorldGridMaterial checker.** `build_rubble.py`
   asserts it with `MaterialEditingLibrary.get_statistics()` (zero instructions = failed compile) for the master
   and all seven instances, and asserts the sRGB/compression settings of all six source textures before binding
   them — but the assertion has never been executed.
2. **That the OBJ material-slot names survive the import** as `concrete` / `masonry` / `rebar` / `fabric` /
   `wood` / `metal`. The script raises on any unrecognised slot. This is the same mechanism `build_buildings.py`
   already proved for `wall` / `roof_tile` / …, so the risk is low but not zero.
3. **That Nanite really is off and no geometry was lost on import.** The script compares
   `get_num_triangles(0)` against the count the generator recorded, per mesh, and raises on a >2 % mismatch —
   which also catches the known trap where a Nanite-enabled mesh reports its fallback (347 triangles for a 524 k
   asset).
4. **The sign of `pitch` under the OBJ handedness flip.** The importer maps OBJ +y to UE −Y, so a UE pitch about
   UE-Y is a rotation about the *negated* local axis. Every pitch and roll in the layout is drawn from a
   symmetric distribution, and the burial slabs are limited to ±12° so either sign leaves them lying flat over
   the survivor — but the exact sign is unverified.
5. **Frame rate and VRAM with 1,799 more actors and 2.4 M more triangles** on an 8 GB card. `docs/CONTEXT.md`
   records the GPU at 75–82 % busy with the *bare* scene; this is the first big addition since. If it is too
   heavy, `--piles`, `--mats` and `--loose` scale it down and `--max-tris` hard-caps it.
6. **Whether the survivors on the fan still read correctly.** The layout keeps a 1.35 m void around every
   detectable one, but their `base_asl_m` came from `gen_actors.py` sampling the *terrain*, not the rubble, so a
   survivor beside a pile may now be at the foot of a 3 m mound rather than in a void within it. Worth a look in
   `qa_shots.py` and possibly a re-seat.

## Decisions worth knowing

* **Collision is OFF on all rubble** (`RUBBLE_COLLISION = False` at the top of `build_rubble.py`). Two concrete
  reasons: complex collision over 2.4 M triangles is real memory on a 16 GB / 8 GB-VRAM machine, and the
  sightline server fails a flight on any collision report whose actor is not named `Ground`, so a drone
  descending over the fan would fail on rubble contact. Flip the constant to get query-only complex collision if
  something needs to line-trace onto the rubble surface.
* **Dust is a material parameter, not a texture choice.** Reference B's palette line is "dust-covered,
  desaturated tan/grey, very low colour contrast", so `M_Rubble_Master` carries `DustAmount` / `DustColor` that
  lerp the base colour toward dust, flatten the normal and raise roughness together. Concrete is at 0.50, rebar
  at 0.18.
* **Instance segmentation IDs are not set** on the rubble actors (neither does `build_props.py`). If the rubble
  needs its own class in the auto-labels, that is a separate change.
* **`data/scene/rubble/*.obj` (14 MB) is NOT covered by `.gitignore`**, whereas `data/scene/*.obj` and
  `data/scene/buildings/*.obj` are. I was not allowed to edit `.gitignore`; the orchestrator may want to add
  `data/scene/rubble/*.obj` to it, since the files are deterministic from the seed. (They appear to have been
  committed already by a concurrent run.)
* Nothing here deletes a record or marks a segment cleared (guardrail R10). The only interaction with the
  coverage story is additive: the §2.7 burial polygons now have physical concrete over them.
