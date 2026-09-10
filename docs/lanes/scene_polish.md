# Scene polish lane — anti-tiling, flood damage, waterline foam

Written offline (no editor, no sim) on 2026-09-10 for the FloodValley scene. Everything below was written to be
run by the orchestrator; the host-side generators were run here and the numbers are real.

## Files added (nothing existing was edited)

| File | Side | Purpose |
|---|---|---|
| `tools/scene/build_roof_materials.py` | editor | rebuilds `M_PBR_Master` with anti-tiling **and** the flood tide line; restores and re-verifies all 8 instances |
| `tools/scene/gen_damage.py` | host | writes `data/scene/damage/*.obj` (14 damaged archetype meshes) + `data/scene/damage.json` |
| `tools/scene/build_damage.py` | editor | imports the variants, swaps them onto the planned houses, rafts debris onto flat roofs |
| `tools/scene/gen_foam.py` | host | writes `data/scene/foam.obj` (the shoreline ribbon) + `data/scene/foam.json` |
| `tools/scene/build_foam.py` | editor | imports the ambientCG foam textures, builds `M_Foam`/`MI_Foam_Shore`, places `FloodFoam` |
| `docs/lanes/scene_polish.md` | — | this report |

## Run order (exact)

```
# 1. host side (safe while the editor is busy; ~2 s each, no GPU, no torch)
D:\Tools\uv\uv.exe run python tools\scene\gen_damage.py          # --seed 41 default
D:\Tools\uv\uv.exe run python tools\scene\gen_foam.py            # --seed 53 default

# 2. editor, PIE STOPPED, in this order (each is idempotent)
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_roof_materials.py').read())"
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_damage.py').read())"
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_foam.py').read())"

# 3. LOOK
ue_python code="exec(open(r'D:\Sightline\tools\scene\qa_shots.py').read())"
```

**Ordering constraints that matter**

* `build_roof_materials.py` must run **after** `build_buildings.py`, and re-running `build_buildings.py` later
  **reverts** the master material — re-run `build_roof_materials.py` after it, every time.
* `build_damage.py` must run after `build_roof_materials.py` only so that `MI_Rubble` inherits the finished
  master; the reverse order also works, it just needs a second pass.
* `build_foam.py` is independent of both. It never touches `M_FloodWater` or `M_FloodValleyTerrain`.
* **One owner per material.** `build_materials.py` owns the terrain and the water; `build_roof_materials.py` owns
  `M_PBR_Master`; `build_foam.py` owns `M_Foam`. That is why the tide line ended up in script A rather than in
  `build_damage.py` even though the brief listed it under damage: two scripts clearing and rebuilding the same
  graph would silently undo each other, and the failure mode (a grey checker) is invisible to the Python API.

---

## A. `build_roof_materials.py`

Rebuilds `M_PBR_Master` from scratch, keeping every existing parameter name (`UVScale`, `BaseColor`, `Normal`,
`ARM`, `Tint`) so the 8 instances keep working, and adds three independent period-breakers plus a tide line.

**Why it was needed, measured.** The generated OBJ UVs are metre-tiles (`gen_buildings.TILE_M`: wall 2.0 m,
roof_tile 1.5 m, roof_sheet 2.0 m), so a clay-tile roof repeats every 85 px at the 1.77 cm/px survey GSD, and 76
houses drawn from five archetypes carry the *identical* texture. What actually repeats is the low-frequency
content of the diffuse map — std of the 64 px mip of the sRGB luma:

| set | mip5 (64 px) luma std | mip3 (256 px) |
|---|---|---|
| worn_mossy_plasterwall | **0.0648** | 0.0815 |
| dirty_concrete | 0.0516 | 0.0599 |
| clay_roof_tiles | **0.0432** (and it is on 46 % of roofs) | 0.0644 |
| rusty_corrugated_iron | 0.0215 | 0.0408 |
| weathered_planks | 0.0212 | 0.0338 |
| painted_plaster_wall | 0.0110 | 0.0142 |

Those numbers set the per-instance `AntiTileStrength` (painted plaster barely repeats, so it gets 0.45; mossy
plaster and concrete get 1.0).

**1. Per-object UV jitter.** `frac(dot(ObjectPositionWS.xy, k))` is constant across one actor, so it is a free
per-house random UV offset with **zero** distortion — unlike a smoothly varying UV warp, whose gradient (≈0.5
tiles/m for a 3-tile warp over a 22 m macro period, against a base gradient of 0.67 tiles/m for a 1.5 m tile)
would visibly shear the tile grid. Houses are ≥ 17 m apart, so `dot` moves ≥ 19 units between neighbours and
`frac` fully decorrelates them. Guarded with `getattr` — if `MaterialExpressionObjectPositionWS` is absent the
script prints a warning and continues without the jitter.

**2. Two UV scales blended by a low-frequency macro field** — the same treatment the terrain got.
*The macro channel was measured, not assumed* (the terrain bug was a mask whose p10 was already 0.878, so the
remap was 1.0 everywhere). Statistics of the **32 px mip** — the level the shader actually samples — of the raw
JPEG, which is exactly what a TC_MASKS texture (sRGB off) returns:

| texture / channel | mean | std | p10 | p50 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| **brown_mud_rocks_01 ARM.G** | **0.545** | **0.068** | 0.463 | 0.541 | 0.631 | 0.349 | 0.804 |
| clay_roof_tiles ARM.R | 0.651 | 0.051 | 0.580 | 0.651 | 0.718 | 0.506 | 0.784 |
| red_laterite ARM.R | 0.626 | 0.020 | 0.604 | 0.624 | 0.651 | 0.569 | 0.765 |
| brown_mud_02 ARM.G | 0.951 | 0.028 | 0.914 | 0.957 | 0.980 | 0.828 | 1.000 |
| rusty_corrugated_iron ARM.G | 0.936 | 0.004 | 0.933 | 0.937 | 0.941 | 0.925 | 0.945 |
| painted_plaster_wall ARM.G | 0.919 | 0.002 | 0.918 | 0.918 | 0.922 | 0.910 | 0.929 |
| worn_mossy_plasterwall ARM.G | 0.854 | 0.022 | 0.828 | 0.851 | 0.886 | 0.792 | 0.925 |
| dirty_concrete ARM.G | 0.863 | 0.018 | 0.847 | 0.859 | 0.890 | 0.820 | 0.933 |

`brown_mud_rocks_01_ARM.G` is the only channel that is both centred and broad, so the remap is
**`saturate((G − 0.44) × 4.8)`**: it puts the measured p10/p90 (0.463 / 0.631) at 0.11 / 0.92 and the mean at
0.50, i.e. the field swings across the whole 0–1 range instead of clamping to one end. Sampled twice, at mip 6,
at world scales `MacroTileCm = 2600` (blend weight + brightness) and `MacroFineTileCm = 1200` (tide-line wobble).

**3. Far-UV mode, also measured.** 1-D profiles of the source maps (std of the row/column means, dominant period
of each profile):

| map | std along u | std along v | dominant period |
|---|---|---|---|
| rusty_corrugated_iron normal | 0.0694 | **0.0005** | 59 cycles along u (corrugations), flat along v |
| clay_roof_tiles normal | 0.0746 | 0.0083 | 27 cycles along u, 16 along v (tile lattice) |
| everything else | — | — | unstructured |

Those two roof maps are strict lattices, so a *scale* change would blend two different ridge pitches and mush
the corrugations in oblique views. Instead they get `UVScaleFar = 1.0` plus a **lattice-locked** `UVFarOffset` —
an offset of an integer number of lattice periods (20/59 = 0.339 for the iron; 9/27 and 5/16 for the tiles) —
which keeps the ridges perfectly in phase while sampling a different part of the macro content. Everything else
gets the true two-scale blend at `UVScaleFar = 0.53`. (At survey altitude the lattice itself is invisible anyway:
1.5 m / 27 = 5.6 cm ≈ 3.1 px, 2.0 m / 59 = 3.4 cm ≈ 1.9 px. The macro blotch is what repeats.)

**4. Macro brightness variation** `1 ± MacroTintStrength` (default 0.12) from the same field: period-free by
construction and, from 45 m, the single most effective cue against "the same roof, 76 times".

**5. Flood tide line and mud staining.** World-Z driven, so it lines up with `FloodWater` automatically:
`SiltLineZ` defaults to `flood_water_z_cm + 8` = **1575.45 cm** (from `flood_valley.json`); `below` fades in over
`SiltFadeCm` = 90 cm under the line, `band` is a ±22 cm darker accent at the line itself, and the whole thing is
modulated by the fine macro field (0.55–1.00) and wobbled ±11 cm so it is not a laser-straight line across 76
buildings. Silt colour is `T_brown_mud_02_D` (measured linear mean RGB **0.0802 / 0.0633 / 0.0421** = dark wet
mud); `SiltTint` (1.50, 1.45, 1.35) lifts it to linear (0.120, 0.092, 0.057), a light grey-brown dried stain.
Roughness is lerped to 0.86 and **metallic is lerped to 0** under the stain, so the rusty-iron roofs lose their
metal sheen below the waterline.

Cost: 9 texture samples (all shared-wrap, so no sampler slots are consumed), **0 triangles**.

**Per-instance settings the script applies**

| instance | AntiTileStrength | far-UV mode |
|---|---|---|
| MI_RoofTile | 1.00 | lattice-locked (1.0, 0.3333, 0.3125) |
| MI_RoofSheet_Rusty | 0.80 | lattice-locked (1.0, 0.3390, 0.4100) |
| MI_Wall_Weathered, MI_Concrete | 1.00 | two-scale 0.53 |
| MI_Wood | 0.70 | two-scale 0.53 |
| MI_Wall_Cream / Yellow / Mint | 0.45 | two-scale 0.53 |

**Safety.** Every instance's texture/vector/scalar overrides are **captured before** the graph is cleared and
restored afterwards, then each one is re-read and compared *by package path* (checking only for `None` is not
enough — a lost override silently falls back to the parent's dirty-concrete default, which is a texture). The
master and all 8 instances go through `assert_compiles()`.

---

## B. `gen_damage.py` + `build_damage.py`

`gen_damage.py` reuses `gen_buildings.Mesh`, `openings()` and the archetype dimensions, so the damaged houses are
the same buildings with pieces removed — no new textures, no new assets.

**Seven variants × 2 mirror flavours = 14 meshes, 134–316 triangles each**

| variant | archetype | damage |
|---|---|---|
| `C_sheet_1s_open` | corrugated shed | 58 % of the sheeting gone from the upstream eave, 3 purlins exposed, one sheet peeled back at 55°, silt-filled interior |
| `C_sheet_1s_peel` | corrugated shed | 26 % gone, one peeled sheet |
| `A_tile_1s_holed` | 1-storey tile | 4.2 × 2.6 m hole in the upstream slope, through both the tiles and the underside, 2 purlins |
| `D_tile_2s_holed` | 2-storey tile | 5.4 × 3.0 m hole, interior floor at the upper storey |
| `A_tile_1s_endgone` | 1-storey tile | roof gone over 3 m at one end, jagged end wall, rafters, rubble heaps inside and out |
| `B_flat_2s_corner` | 2-storey RCC | 3.8 × 3.2 m of the roof slab and its parapet collapsed, slab fragment hanging into the hole, rubble on the floor below, stair cabin relocated clear |
| `E_flat_1s_slump` | 1-storey RCC | the upstream wall pushed 1.1 m out and 0.9 m down, slab sagging over it, rubble along the base |

**Orientation is exact, not approximate.** `build_buildings.py` imports the OBJ (UE flips it to UE X = u,
UE Y = −v) and then spawns the actor at UE yaw = 90 − θ. Composing the two gives
`east = u·cosθ − v·sinθ`, `north = u·sinθ + v·cosθ` — exactly the convention `gen_actors.py` and `gen_props.py`
already use, so there is no mirror ambiguity. The `_a`/`_b` flavours (`mirror_v`, which negates v and reverses
every winding) let each torn roof face **upstream**: `_a` when cos θ > 0, `_b` otherwise.

**Interiors are real.** UE materials are single-sided, so an opened roof would otherwise show straight through
the far walls to the terrain. Each damaged variant carries an `inner_room()` liner — inward-facing quads
*coincident* with the outer walls (zero-thickness walls, so exactly one of each pair is front-facing from any
viewpoint and the other is backface-culled: no z-fighting) — with `weathered_planks` walls over a
`brown_mud_rocks_01` floor at the top storey's slab. That reads as a dark, silted-up room from directly above,
which is also the partial-occlusion case the detector should be tested on.

**Survivors are protected per-variant, not per-house.** 26 of the 76 houses carry roof survivors, and blocking
every occupied house would rule out most of the flooded terrace — exactly where the damage belongs. Each variant
declares the local (u, v) rectangle it removes (`UNSAFE`), and a house is only eligible for a variant when no
survivor stands inside it plus a 0.8 m margin. Roof debris additionally keeps 1.5 m clear of every survivor.

**Measured output (seed 41, settlement seed 11, terrain seed 7)**

```
variants: 14 meshes (134-316 tris each)
damaged houses: 18 / 76  (5 rolled damage but no variant was clear of their survivors;
                          5 are damaged AND carry survivors, well away from the damage)
  by variant:   A_tile_1s_endgone_b 2, A_tile_1s_holed_a 1, B_flat_2s_corner_a 2, B_flat_2s_corner_b 3,
                C_sheet_1s_open_a 1, C_sheet_1s_peel_b 2, D_tile_2s_holed_a 2, D_tile_2s_holed_b 2,
                E_flat_1s_slump_a 1, E_flat_1s_slump_b 2
  by archetype: C_sheet_1s 3/10, A_tile_1s 3/13, D_tile_2s 4/13, B_flat_2s 5/24, E_flat_1s 3/16
  by zone:      terrace 17, bank 1
roof debris:  67 items, 211,163 tris (budget 220,000)
NET added triangles: 212,237  (variant meshes 4,316 replace 3,242 pristine)
```

Determinism verified: two consecutive runs produced a byte-identical `damage.json`
(sha256 `27e0d254ea2232be…`). `hash()` is deliberately not used anywhere — Python salts string hashes per
process, which would have made the meshes differ between runs; `zlib.crc32` is used instead.

`build_damage.py` is self-repairing: it **resets every** `House_###` actor to its pristine archetype mesh from
`settlement.json` first (so a house dropped from the plan really does become pristine again), re-asserts its
location, then swaps in the variants. It never renames an actor — roof debris is cleared by outliner folder
(`Damage`), the pattern from `build_props.py`.

---

## C. `gen_foam.py` + `build_foam.py`

A cell of the height grid is a **shoreline cell** when the water level falls between the min and max of its four
corners. Each is turned into one horizontal quad oriented by the local terrain gradient — `across` runs downhill,
`along` runs parallel to the shore — with the centre pushed 30 % of the ribbon width downhill so most of the
quad lies over water rather than buried in the bank.

UV0 carries **(metres along the shore, 0–1 across the ribbon)**. The material turns the second component into
`saturate(1 − |2v − 1|)`, and each quad's strength is baked in by **shrinking its v range about 0.5** — free, and
because the falloff is symmetric about v = 0.5 it is immune to the V flip UE's OBJ importer applies. The u origin
is jittered per quad so no two patches show the same bubbles. Quads sit at `water_level + 3 cm` with a 0–3 cm
jitter so overlapping patches are never exactly coplanar.

**Measured output (seed 53, spacing 4 m)**

```
shoreline cells: 2,346  ->  2,346 quads after thinning (~9.38 km of shoreline)
quads: 2,344 shoreline + 204 upstream-of-house + 25 obstacle = 2,573
triangles: 5,146   vertices: 10,292
shoreline slope (dz/dx): mean 0.2245, p10 0.0505, p50 0.1833, p90 0.3495
```

Quad strength is `clip(0.95 − 0.80·min(slope,1), 0.35, 0.95) × U(0.82,1.0)`, which given the measured slope
distribution puts the gentle margins (p10 = 0.05) at ~0.91 and the steep cut banks (p90 = 0.35) at ~0.67 — a wide
foam raft where the water laps a flat terrace, a thin line where it cuts a bank.

**The alpha clip value is measured, not guessed.** Coverage of `Foam001_2K-JPG_Opacity.jpg`:

| clip | 0.08 | 0.10 | **0.12** | 0.14 | 0.16 | 0.20 | 0.24 | 0.30 | 0.36 | 0.48 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coverage | 0.730 | 0.679 | **0.630** | 0.581 | 0.533 | 0.431 | 0.351 | 0.248 | 0.164 | 0.052 | 0.011 |

(mip 0; the 4th mip tracks it to within 0.02 — 0.570 vs 0.552 at clip 0.15 — so the alpha test does **not** thin
out with distance, which is the usual failure of masked foam at survey altitude.) `opacity_mask_clip_value =
0.12` gives 63 % coverage at the ribbon centre, ~35 % where the falloff has dropped to 0.5, and nothing below
falloff 0.25: a band that fades out rather than ending at a line. `FoamGain` (default 1.0) shifts the whole curve
per-instance without touching the material.

**Colour is measured too.** Over the texels the alpha test keeps, Foam001's colour map means sRGB
(0.479, 0.510, 0.517) = linear (0.202, 0.233, 0.240) — mid-grey with a **blue** cast, because it is a scan of
foam on dark water. River foam on a silt-laden monsoon flood is a dirty warm cream, so `FoamTint` (2.08, 1.67,
1.38) inverts that ordering rather than merely brightening it, landing on linear (0.42, 0.39, 0.33). The set's
roughness map (mean **0.17**) is a wet *water* value and is deliberately unused; `FoamRoughness` is 0.75.

Foam002 is imported with identical settings so switching is a one-parameter change on `MI_Foam_Shore`
(`FoamColor`/`FoamOpacity`/`FoamNormal`); Foam002's opacity is brighter (mean 0.243 vs 0.199), so drop
`FoamGain` to ~0.82 if you switch.

The `FloodFoam` actor is MOVABLE, `NoCollision`, `cast_shadow = False` (a flat sheet at the surface must not
shade the water, and foam must never block `sim_fly` or a spawner's trace).

---

## Cost

| item | triangles | notes |
|---|---|---|
| damaged house variants | **+1,074 net** | 4,316 replace 3,242 pristine |
| roof debris (67 props) | **+211,163** | budget 220,000, enforced in the generator |
| foam ribbon | **+5,146** | one actor, one draw call |
| all three materials | 0 | |
| **total added** | **≈ 217,400** | against the stated ~1.5 M ceiling; scene goes ~4.5 M → ~4.72 M |

Added texture memory: Foam001 (colour BC1 + opacity BC7/masks + normal BC5, 2K with mips) ≈ **14 MB** VRAM.
Foam002 is imported but unreferenced, so it is not resident. No new building textures at all — `MI_Rubble` reuses
`brown_mud_rocks_01`, already imported for the terrain.

Overdraw: the foam band covers 9.4 km × 7 m ≈ 66,000 m² of a 4.2 M m² map (1.6 %), masked, no shadow.

---

## What I could NOT verify (no editor in this lane), and what to do if it bites

All three editor scripts were dry-run twice against a mock `unreal` module (both branches of the texture-object
probe, and both a pivot-preserving and a re-pivoting importer): **0 failures**. That validates our control flow,
the JSON handling and all the bounds/pivot arithmetic. It cannot validate UE API names or shader compilation.

| risk | why it might bite | what the script does about it | if it still fails |
|---|---|---|---|
| `MaterialExpressionTextureObjectParameter` + the `Tex` input pin | pin names are not documented and vary by version | probes the pin with a throwaway node pair, falls back to two `TextureSampleParameter2D` nodes sharing one parameter name | nothing to do — the fallback is automatic; the script prints which path it took |
| `MaterialExpressionObjectPositionWS` missing | class name could differ | `getattr` guard; prints a warning and drops the per-object jitter | anti-tiling still works via the two-scale blend and the macro tint |
| shader compile failure of `M_PBR_Master` / `M_Foam` | a sampler-type / default-texture mismatch renders the grey checker and reports nothing | `assert_compiles()` on the master, all 8 instances, `MI_Rubble`, `M_Foam` and `MI_Foam_Shore` | the error names the material; check its default texture's sRGB flag against its sampler type |
| a lost instance texture override | rebuilding the master could drop overrides | captured before the clear, restored after, then compared **by package path** | error names the instance and parameter |
| OBJ importer re-pivots off-centre meshes | never proven in this project (terrain and pristine archetypes are both symmetric in u/v, so they could not expose it) | both scripts measure the imported bounds against the generator's own bbox and translate the actor onto the expected position | it self-corrects and prints the shift; if the *extent* is wrong `build_foam.py` refuses to continue (that means a wrong yaw) |
| triangle count vs Nanite fallback | imported meshes arrive with Nanite ON | asserted with a 2 % tolerance (still catches the fallback by three orders of magnitude) | the error prints both numbers |
| stale terrain in the level | foam baked for terrain seed 7 | `build_foam.py` line-traces 8 shoreline probes against the generator's heights and needs 5 of 8 within 40 cm | re-run `gen_terrain.py` + `build_flood_valley.py`, then `gen_foam.py` |

**Judgement calls that only a render can settle** — all are single parameters, no rebuild needed:

| what to look at | knob | where |
|---|---|---|
| roofs still show a grid | `AntiTileStrength` ↑ (max ~1.0), `MacroTintStrength` ↑ to 0.18 | per instance under `/Game/Sightline/Buildings/Materials` |
| roofs look mushy / double-ridged | `UVScaleFar` → 1.0 and give it a lattice-locked `UVFarOffset` (or `AntiTileStrength` ↓) | same |
| tide line too high/low or too wide | `SiltLineZ`, `SiltFadeCm`, `SiltBandCm`, `SiltAmount` | `M_PBR_Master` defaults or per instance |
| tide line too dark/light | `SiltTint` | same |
| foam too crunchy at the edges | `FoamGain` ↑ (each +0.1 adds ~5 points of coverage) | `MI_Foam_Shore` |
| too much / too little foam | `FoamGain`, or regenerate with `--spacing` | `MI_Foam_Shore` / `gen_foam.py` |
| foam patches too big/small | `FoamTileM` (2.2 m default), `FoamMacroCm` | `MI_Foam_Shore` |
| not enough damaged houses | `DAMAGE_P` in `gen_damage.py`, or a different `--seed` | host |

If the masked foam edges shimmer at 4K, the alternative is a translucent `M_Foam` — but note that overlapping
translucent quads double-blend where they overlap, which masked quads do not, so lowering `FoamGain` is the
cheaper first move.

## Loose ends for the orchestrator

* **`.gitignore`**: `data/scene/foam.obj` is already covered by `data/scene/*.obj`, but
  `data/scene/damage/*.obj` is not, and a commit made during this session has already tracked all 14 of them
  (~130 KB). To match the `data/scene/buildings/*.obj` convention, add `data/scene/damage/*.obj` to `.gitignore`
  and `git rm --cached data/scene/damage` — I did neither: `.gitignore` is outside this lane's file list and this
  lane runs no state-changing git commands. They are deterministic, so leaving them tracked is harmless.
  `damage.json` and `foam.json` are small and should stay committed, like `settlement.json`.
* **Flood level**: the foam ribbon is baked for `water_level_m = 1061.6814259297319`. If a scenario changes
  FloodLevel, re-run `gen_foam.py --water-level <asl>` and `build_foam.py`; moving the actor in Z is not enough
  because the shoreline *shape* changes.
* **Not oriented per house**: `A_tile_1s_endgone` always wrecks the local −u end, which is along the house's long
  axis and therefore not flow-aligned. The roof tears (`_open`, `_peel`, `_holed`) and the wall slump *are*
  flow-aligned via the `_a`/`_b` mirror.
* **Windows on the slumped wall** are culled rather than carried out with the leaning wall (`cull()` in
  `gen_damage.py`), so `E_flat_1s_slump` shows a blank leaning panel. Deliberate: carrying the openings would
  need the openings generator rewritten, and at survey altitude a blank leaning wall reads correctly.
* **Tracker**: this lane touched none of `docs/TRACKER.md` / `docs/CONTEXT.md`. Backlog items 4 (roof tiling) and
  6 (damage state) are addressed here, and item 3's "foam at the waterline" half is done; the water's own
  large-scale break-up (item 3's first half) was left to `build_materials.py`'s owner.
