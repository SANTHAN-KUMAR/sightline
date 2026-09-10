# Scene realism lane — canopy, utilities, boats, and the water that was never being rendered

Written offline on 2026-09-10 (no editor, no simulator, no `mcp__sightline__*` / `mcp__unreal__*`). Every host
generator below was actually run; every number is from that run. Nothing existing was edited: six new files.

## TL;DR — the one thing to read

**The flood water has never rendered as water.** `M_FloodWater` never connects `MP_Opacity`, and in UE's
single-layer-water shader `WaterVisibility = 1 - Opacity` with an unconnected Opacity defaulting to **1.0**.
So the entire volumetric term — every `Scattering` / `Absorption` / `PhaseG` / `ColorScaleBehindWater` value
`build_materials.py` sets — is multiplied by **zero** and discarded, and what we have been looking at is an
opaque DefaultLit card whose colour is `BaseColor`. That is the "uniform silty tan". Separately, the
coefficients were in the wrong unit by 100× (`1/cm`, not `1/m`), so even with the volume switched on there
could never have been any depth variation. `tools/scene/tune_water.py` fixes both. Engine source, line by
line, in section C.

## Files added (nothing existing was edited)

| File | Side | Purpose |
|---|---|---|
| `tools/scene/gen_vegetation.py` | host | tree + understorey layout → `data/scene/vegetation.json`, `vegetation_canopy.png` |
| `tools/scene/build_vegetation.py` | editor | imports/repairs the tree meshes, forces Nanite + opaque + two-sided, places 5,528 actors |
| `tools/scene/gen_utilities.py` | host | poles/wires/service drops/boats → `data/scene/utilities/*.obj`, `utilities.json`, 5 preview PNGs |
| `tools/scene/build_utilities.py` | editor | builds `M_Wire` + 4 boat-paint instances, imports and places them |
| `tools/scene/tune_water.py` | editor | retunes `M_FloodWater` in place and asserts it still compiles |
| `tools/scene/check_realism.py` | host | the failing-capable check for all of the above (217 checks + 13 dry runs) |

## Run order (exact)

```
# 1. host (safe while the editor is busy; ~25 s total, no GPU, no torch)
D:\Tools\uv\uv.exe run python tools\scene\gen_vegetation.py      # --seed 67 --max-trees 4600
D:\Tools\uv\uv.exe run python tools\scene\gen_utilities.py       # --seed 83
D:\Tools\uv\uv.exe run python tools\scene\check_realism.py       # MUST exit 0 before step 2

# 2. editor, PIE STOPPED, in this order
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_utilities.py').read())"
ue_python code="exec(open(r'D:\Sightline\tools\scene\tune_water.py').read())"
ue_python code="exec(open(r'D:\Sightline\tools\scene\build_vegetation.py').read())"   # SLOW, see warning

# 3. LOOK
ue_python code="exec(open(r'D:\Sightline\tools\scene\qa_shots.py').read())"
```

**Ordering constraints that matter**

* `build_utilities.py` must run **after** `build_buildings.py` — it reuses `MI_Concrete`,
  `MI_RoofSheet_Rusty`, `MI_Wood`, `M_Window` and `M_PBR_Master`, and raises by name if any is missing.
* `tune_water.py` must be re-run **every time** `build_materials.py` runs. `build_materials.py` clears and
  rebuilds the `M_FloodWater` graph from scratch, which discards this tuning silently — the same relationship
  `build_roof_materials.py` already has with `build_buildings.py`.
* `build_vegetation.py` must run **after** `import_assets.py`, which has now imported the five species:
  it turns Nanite **off** and adds a **box collision** to every prop, both of which are wrong for a 353 k-
  triangle, 24 m-wide tree. `build_vegetation.py` turns Nanite back on and sets the actors to `NoCollision`,
  so running it after repairs whatever the other script did — and it must be re-run if that script re-runs.
* `build_vegetation.py` is last because it is by far the heaviest step; if the editor is going to run out of
  memory it should do so after everything else has been saved.

---

# A. Vegetation — `gen_vegetation.py` + `build_vegetation.py`

## A.1 What I measured about the source assets (this changed the whole design)

The trees were downloaded by the assets lane while I worked. They are **not** the low-poly leaf-card trees the
triangle budget in the brief assumed. Measured directly from the glTF accessors (`count`, per-attribute
`min`/`max`), not from a catalogue:

| id | full scan | **imported (`_2k_lite`)** | height | crown | leaf primitive (full) | lite .bin |
|---|---|---|---|---|---|---|
| `island_tree_01` | 1,599,403 | **162,400** | 5.03 m | 4.82 m | 1,060,032 | 6.1 MB |
| `island_tree_02` | 1,072,213 | **106,821** | 3.41 m | 4.21 m | - | 4.1 MB |
| `island_tree_03` | 2,085,320 | **111,522** | 2.62 m | 2.97 m | 1,408,704 | 4.4 MB |
| **`jacaranda_tree`** | 3,863,832 | **352,726** | 19.47 m | 24.42 m | 2,402,434 | 13.6 MB |
| `tree_small_02` | 2,062,487 | **156,172** | 4.56 m | 4.29 m | - | 6.0 MB |
| `fern_02` (understorey) | 6,232 (4 meshes) | - | 0.43 m | 0.99 m | - | 0.2 MB |

The assets lane decimated each scan to `<pid>_2k_lite.gltf` and imported **those** (12-24x reduction, 208 MB
to 13.6 MB for the jacaranda), so the lite counts are what the level carries and what this budget is built
from; the full scans are measured only as a cross-check. `gen_vegetation.py` prefers the lite variant, falls
back to the full scan when it is absent, and prefers `props.json`'s editor-measured `tris`/`size_m` when the
species is there - unless the two disagree by more than 2x, which would mean Nanite is reporting a **fallback**
mesh (CONTEXT: a 524 k-triangle terrain reported 347) and then the glTF is treated as truth.

glTF is Y-up, so height is `bbox.y` and the crown spread is `bbox.x`/`bbox.z`. That mapping is not assumed —
it is checked against something already in the project: `dead_tree_trunk`'s glTF bbox is (3.05, 0.29, 0.28)
and `data/scene/props.json` records it as UE `size_m [3.05, 0.28, 0.29]`, i.e. UE(x,y,z) = glTF(x,z,y). The
same check identifies `dead_tree_trunk` as a **fallen** trunk (3 m along X, 0.29 m tall), which is correct.

The generator re-measures on every run and **rejects a species whose buffer is missing or short**. That fired
for real: at 23:38 `tree_small_02` had a 9 KB glTF and no `.bin` at all, and was dropped with
`SKIPPED - missing buffer tree_small_02.bin (download incomplete)`; by 23:39 the 95 MB buffer had landed and
it was included. That is exactly the silent failure this project keeps hitting, caught by a file-size compare.

## A.2 Consequence: Nanite is still not optional

4,377 crowns x 107-353 k triangles = **1,083,718,780 instanced source triangles**. An imported static mesh has
no LOD chain, so without Nanite every instance rasterises its full LOD0 - a billion triangles a frame. Nanite
stores the geometry once per mesh (**889,641 unique triangles** for all five species) and rasterises clusters
sized to the pixels on screen, so instance count is nearly free. Pushed raw, that would be 217x the entire
rest of FloodValley (terrain 524 k + debris 4.15 M + damage 212 k + houses + foam ~ 5 M).

`build_vegetation.py` turns Nanite **on** for the tree meshes only (the understorey ferns stay non-Nanite:
784-2,384 triangles each, where Nanite is all cost), reads the flag back **from the saved asset**, and if it is
still off **and** the layout exceeds `NO_NANITE_MAX_TRIS = 8,000,000` it **refuses to place anything** and says
so. That refusal is exercised by the dry-run suite.

This is a deliberate deviation from `docs/SOLUTION_DOC.md` section 4's "Nanite off on imported meshes", which
is right for a 12-53 k rock and inverted here. It should be recorded in `docs/CONTEXT.md`.

## A.3 The second silent failure: the leaf materials import TRANSLUCENT

Every Poly Haven tree glTF declares its leaf material `"alphaMode": "BLEND"`, and **every `*_diff_2k.jpg` is a
3-channel RGB JPEG with no alpha channel** (checked with PIL on all five sets — mode `RGB`, 2048×2048). The
transparency is a no-op in the source and a disaster in UE:

* Nanite does not render translucent materials — it falls back to the fallback mesh, i.e. **bald trees**;
* translucent surfaces do not write the depth/stencil that Cosys-AirSim's instance segmentation reads;
* thousands of translucent crowns is unbounded overdraw.

The leaves are **real modelled geometry** (1.06–2.40 M triangles in the leaf primitive alone), so forcing
Opaque loses nothing. `build_vegetation.py` forces `blend_mode = BM_OPAQUE` and `two_sided = True` on every
tree material (the glTF sets `doubleSided: true` on all three slots; if the importer drops it, half of every
crown disappears from above). It handles both a plain `Material` and a `MaterialInstanceConstant` (which needs
`base_property_overrides`), and both branches are dry-run tested.

## A.4 How the layout is built, and the measured result

Per zone a target **canopy closure** is converted to a Poisson intensity `lambda = -ln(1 - closure) / A_mean`
— crowns overlap, so closure is `1 - exp(-lambda·A)`, not `lambda·A`. Each 4 m terrain cell draws
`Poisson(lambda·16 m²)` trees at uniform positions inside it.

| zone | closure | cells | mean crown | λ /ha | placed | why |
|---|---|---|---|---|---|---|
| hillslope | 0.72 | 35,556 | 247.6 m² | 51.4 | 2,939 | dense Western Ghats forest |
| garden | 0.55 | 638 | 224.7 m² | 35.5 | 36 | Kerala homegardens above water |
| **flooded** | **0.55** | 15,707 | 201.9 m² | 39.6 | **942** | the flood drowns a homegarden's floor, it does not fell it |
| bank | 0.35 | 588 | 186.2 m² | 23.1 | 28 | cut banks |
| floodplain | 0.20 | 10,788 | 140.0 m² | 15.9 | 242 | drowned valley floor outside the terrace |
| fan | 0.08 | 14,057 | 107.6 m² | 7.8 | 178 | 2–7 m of fresh debris buried everything |
| channel | 0.02 | 2,752 | 30.8 m² | 6.6 | 12 | scoured clean |

**Result (sim, seed 67 / terrain 7 / settlement 11): 4,377 trees + 1,151 understorey clumps.**
Canopy closure, rasterised at 2 m over the region of interest: **45.6 % of the 1.3725 km² ROI, 42.1 % over the
settlement.** Species: jacaranda 2,233, island_01 745, island_02 606, tree_small_02 491, island_03 302.

It is not 50 % because two large zones are *deliberately* bare: the deposit fan is 22.7 % of the ROI at 8 %
closure and the channel is scoured. That is what a debris-flow flood looks like from the air, and raising it
would be making the scene prettier and less true.

**The `floodplain` zone exists because I looked at the picture.** The first run left a bare grey halo all the
way round the water; measuring the masks showed **7.2 % of the ROI** was drowned ground outside the terrace
rectangle and fell through every zone. No count would have shown that.

## A.5 "Crowns still emerging from shallow water"

Trees in the flooded zones lean **downstream** (the valley drains towards −north) by 6–22°, and one in five is
*undercut* — root plate scoured out, settled 0.4–1.3 m. The emergent height above the surface, measured:

```
n = 1,150 drowned trees   min 0.43 m   p10 2.17 m   p50 5.99 m   p90 20.00 m   max 24.02 m
94 of them show less than 2 m of crown above the surface, 203 less than 3 m
```

So there are hundreds of crowns sitting 0.5–3 m proud of the water (the 3–5 m island trees in 2–3.7 m of
flood) under a canopy of 20 m jacarandas whose trunks are visible. 51 candidates were rejected because their
crown would have finished **below** the surface — up to 6.7 m of water on the drowned valley floor — which
would have been 107–353 k triangles each of invisible geometry.

The lean is encoded as **pitch alone with the yaw aimed at the lean azimuth** (a tree is rotationally
arbitrary about its trunk, so this avoids UE's roll-sign convention entirely). `check_realism.py` verifies it
by rebuilding UE's own `FRotationMatrix` third row and asserting the recovered tilt and azimuth match to
0.06° (the stored rounding). Of the 923 trees that carry a scour lean, **98.7 % lean within 60° of due south
and 100 % within 90°**, mean deviation from due south 19.9°.

**`PITCH_SIGN` at the top of `build_vegetation.py` is the one thing here that a render must settle.** UE is
left-handed, so a positive pitch raises +X and tips local +Z toward −X; with yaw = azimuth that leans the
trunk along the azimuth. If the capture shows the drowned trees leaning **upstream**, flip that one constant.

## A.6 Clearances (all enforced and re-verified by the checker)

house footprint + 2.0 m (50 rejected) · survivor + 3.0 m (1 rejected) · launch pad 70 m · slope < 46°.
Crowns *may* overhang a roof — that is realistic and is also the partial-occlusion case the detector should
see — but no trunk stands in a wall and no crown centre sits over a survivor, for the same reason
`gen_foam.py` keeps foam off them: it would contradict the `aerially_detectable` ground truth.

Trees are spawned **`NoCollision`**. `sim_fly` fails a takeoff on contact with any actor not named `Ground`,
and a spawner trace that landed on a leaf would put a survivor in mid-air. The layout keeps 70 m clear of the
launch pad so nothing near the ground station changes either way.

## A.7 Cost

| | |
|---|---|
| tree actors | 4,377 |
| understorey actors | 1,151 (1,793,258 triangles, non-Nanite) |
| instanced source triangles | 1,083,718,780 |
| **unique triangles UE actually stores** | **889,641** (Nanite, once per mesh) |
| decimated source buffers to import | ~34 MB total (jacaranda 13.6 MB) |
| new texture memory | 5 sets x 3 maps x 2K ~ 9 x 14 MB ~ **120 MB** VRAM |

Because the imported meshes are the decimated variants the import itself is cheap (~34 MB); the **spawn loop
and the level save** are now the slow part at 5,528 actors, and `build_vegetation.py` says so before it starts.
`--max-trees` reduces the actor count.

---

# B. Utilities — `gen_utilities.py` + `build_utilities.py`

Poles, conductors, service drops and boats. No CC0 asset exists for a Kerala PSC pole or a country boat, so
they are generated as real geometry in **centimetres** with `usemtl` groups that become UE material slots —
the convention `gen_buildings.py` established for the 76 houses — plus vertex normals, which
`gen_buildings.Mesh` does not write and a tapered pole and a lofted hull both need.

## B.1 The wire width is a measured decision, not a guess

`data/scene/camera_survey.json` gives `f_px = 2548.723`, so at the 45 m survey altitude the GSD is
**1.7656 cm/px**. A bare 8 mm LV conductor would be **0.45 px** wide: TAA and the 4K resample would erase it
and we would ship wires that exist in the mesh and not in the image. Kerala LV distribution is overwhelmingly
aerial bundled cable — three or four 50 mm² cores twisted into one sheath, 30–38 mm overall — so **3.6 cm** is
both physically right and **2.04 px**. Service drops are 2.4 cm = **1.36 px**, deliberately fainter. The three
crossarm conductors are 0.30 m apart = **17 px**, so they resolve as separate lines rather than a smear.
`check_realism.py` recomputes the GSD from `camera_survey.json` and fails below 1.5 px.

## B.2 Layout

A distribution line is a **sequence**, not a cloud. Sites are eligible when they are 7–26 m from the nearest
house, clear of every footprint + 1.6 m, out of the channel, on slope < 20° and on ground between 4 m below
and 8 m above flood stage. They are then chained greedily from the south end, scoring each next site by how
close its span is to the ideal 36 m *and* by how little it turns the line (`+2.2·(1 - cos θ)`) — a pure
nearest-neighbour walk zig-zags and reads as scattered posts from the air. Every added pole retires all sites
within 25.8 m, so two lines never run 12 m apart (the checker caught exactly that in the first version).

**Measured (sim, seed 83): 60 poles in 3 lines of 28 / 26 / 6, 57 spans × 4 conductors, 40 service drops.**
Pole heights 8.4–9.2 m, buried 1.2 m so sloping sites never show a gap. Catenary sag `0.020 × span` clipped to
0.15–1.6 m (0.72 m on a 36 m span), 10 segments. **All 60 poles stand in the flood**, median depth 1.95 m —
which is what the reference photograph shows. One pole (2 crossarms, 6 insulators, light bracket) is
**172 triangles**; the whole network including every wire is **22,704 triangles / 44,856 vertices**.

## B.3 Boats

Three archetypes lofted from 15 stations, `z = 0` at the **design waterline** so a boat placed at the flood
surface floats at the right draught with no per-boat fudge:

| variant | triangles | L × B | draught | freeboard | slots |
|---|---|---|---|---|---|
| `vallam` (Kerala country boat, pointed both ends) | 422 | 6.20 × 1.28 m | 0.20 m | 0.42 m | wood |
| `dinghy` (transom stern) | 430 | 3.60 × 1.46 m | 0.17 m | 0.41 m | wood |
| `rescue_punt` (sheet metal, squared ends) | 438 | 4.80 × 1.60 m | 0.16 m | 0.46 m | roof_sheet + wood |

Every hull carries an **inward-facing copy of its own surface**, coincident with the outer skin: UE materials
are single-sided, so without it a nadir camera looks straight through an open boat to the terrain (the fix
`gen_damage.py` needed for opened roofs). Zero thickness means exactly one of each pair faces the camera, so
no z-fighting. A gunwale cap 5.5 cm wide runs the sheer, because the rim is what a nadir camera actually sees
of an open boat, and three thwarts sit 20 % of the depth below it.

**19 boats placed**: 13 moored against flooded houses (2.4–4.6 m off a wall, aligned to it ±14°), 6 adrift in
the wrack. 8 punts, 7 dinghies, 4 vallams. All float in > 0.5 m of water, all ≥ 2.5 m from any survivor,
±3 cm of heave and ±2–4° of trim/heel so no two are coplanar. **8,202 triangles total.**

Painted per actor: `build_utilities.py` creates four instances of `M_PBR_Master` that **copy `MI_Wood`'s
texture overrides** (rather than hard-coding a texture path — the terrain sets are named `_N` and the building
sets `_NRM`, and guessing wrong is a silent grey material) and vary only `Tint`: cream, blue, green, red.

## B.4 Cost

network 22,704 + boats 8,202 = **30,906 triangles**, one new material (`M_Wire`, a constant-colour black
polymer — it is 2 px wide, a texture would only alias), four new material instances, **zero new textures**.

---

# C. Water — `tune_water.py`

## C.1 The volume was being multiplied by zero (engine source)

`D:\UE_5.8\Engine\Shaders\Private\BasePassPixelShader.usf:1140-1141`

```hlsl
const float BaseMaterialCoverageOverWater = Opacity;
const float WaterVisibility = 1.0 - BaseMaterialCoverageOverWater;
```

`SingleLayerWaterShading.ush:238`

```hlsl
Output.Luminance = WaterVisibility * (ScatteredLuminance + Transmittance * (BehindWaterSceneLuminance * ColorScaleBehindWater));
```

`Material.cpp:7977` — `MP_Opacity` is **active** for `MSM_SingleLayerWater` even in an Opaque blend mode.
`MaterialAttributeDefinitionMap.cpp:401` — an unconnected `Opacity` defaults to **`FVector4(1,0,0,0)` = 1.0**.

`build_materials.py` connects BaseColor, Roughness, Specular and Normal, and never Opacity. Therefore
`WaterVisibility = 0`, the entire single-layer-water term is discarded, and the flood renders as an opaque
DefaultLit surface: `lerp(BaseColor, BaseColorSilt, silt_mix)` as a Lambertian diffuse lobe plus a specular
highlight. A flat lit card. **No amount of tuning `Scattering` could have changed it.**

The Substrate path arrives at the same place: `SubstrateLegacyConversion.ush:633` passes `Opacity` through as
the SLW BSDF's `TopMaterialOpacity`, and `Substrate.ush:1964` uses it to blend `BaseColor` over the water.

**Fix:** connect `MP_Opacity` to `lerp(WaterOpacity = 0.03, WaterOpacitySilt = 0.22, silt_mix)`. Opacity now
means what it means physically here — the fraction of *surface film* (scum, floating silt, sheen) drawn over
the water — so the volume is 97 % visible in clear water and 78 % in the heaviest plume.

## C.2 The coefficients were 100× too large: the unit is 1/cm

`Engine/Source/Runtime/Engine/Public/Materials/MaterialExpressionSingleLayerWaterMaterialOutput.h`:
*"Valid range is [0,+inf[. **Unit is 1/cm.**"* — and the shader multiplies them by `WaterVolumeDepth`, a scene
depth difference in UE units (centimetres).

The old values `Scattering (7.0, 5.0, 2.6)` + `Absorption (1.2, 1.9, 3.4)` are an extinction of **820 per
metre** in red: optical depth 1 at **1.2 mm**. Even with the volume switched on, `Transmittance` would be zero
everywhere and `SafeScatteringAmount` would collapse to a constant albedo — one flat colour, no depth
variation anywhere. Both halves of the symptom, from one 100× unit error and one missing connection.

## C.3 The new parameterisation, and why

**Turbidity.** A turbid monsoon flood has a Secchi depth of roughly 0.35 m; `K_d · z_SD ≈ 1.44` gives an
effective attenuation of ~4.1 /m in the green.

**Hue.** Green-teal is what a turbid inland flood *is*: pure water absorbs strongly in the red (0.35 /m at
650 nm) and CDOM/yellow substance absorbs strongly in the blue, so green is the channel that survives.

| | extinction /m (R,G,B) | albedo ω = S/(S+A) | Scattering 1/cm | Absorption 1/cm |
|---|---|---|---|---|
| clear water | 6.2, 4.1, 5.0 | 0.34, 0.62, 0.52 | .0211 .0254 .0260 | .0409 .0156 .0240 |
| silt plume | 7.5, 5.0, 6.3 | 0.39, 0.60, 0.39 | .0293 .0300 .0246 | .0458 .0200 .0384 |

Clear water is 0.55 : 1.00 : 0.84 — green-cyan. The plume is 0.65 : 1.00 : 0.65 — warmer, browner and more
opaque, which is what a fresh sediment plume looks like from the air.

**Depth.** Green transmittance is 0.51 at 0.17 m, 0.13 at 0.50 m, 0.017 at 1.0 m, 3e-4 at 2.0 m. The
settlement stands in 0.5–3.7 m (`settlement.json`), so the deep terrace reads as opaque teal while submerged
roads, kerbs, garden walls and the shallow margins show through — the depth variation the reference has.

**`ColorScaleBehindWater` was 0.0.** The shader uses `lerp(1.0, param, saturate(depth·0.02))`, so beyond 50 cm
of depth *everything* behind the surface was blacked out. Set to 1.0.

| parameter | old | new | why |
|---|---|---|---|
| (new) `WaterOpacity` / `WaterOpacitySilt` → `MP_Opacity` | unconnected ⇒ 1.0 | 0.03 / 0.22 | switches the water volume on at all |
| `Scattering` | 7.0, 5.0, 2.6 | .0211 .0254 .0260 | unit is 1/cm; 820/m ⇒ opaque in 1.2 mm |
| `Absorption` | 1.2, 1.9, 3.4 | .0409 .0156 .0240 | red-absorbing, green-transmitting = teal |
| (new) `ScatteringSilt` / `AbsorptionSilt` | — | see table | a plume is a different *water*, not a tint |
| `ColorScaleBehindWater` | 0.0 | 1.0 | 0 blacked out every depth cue past 50 cm |
| `PhaseG` | 0.35 | 0.10 | Schlick's phase is evaluated sun↔refracted-view: for a nadir camera that is backscatter, where a forward lobe is small. 0.35 starved the sun and made the result depend on sun azimuth |
| `Specular` | 0.50 | 0.25 | UE `F0 = 0.08 × Specular`; water at IOR 1.33 has F0 = 0.02 |
| `Roughness` / `RoughnessSilt` | 0.06 / 0.16 | 0.035 / 0.19 | sharper sun disc; scum kills the mirror in the plume |
| `NormalStrength` | 0.35 | 0.50 | broadens the glitter path — see the caveat below |
| `BaseColor` | 0.30, 0.215, 0.125 | 0.052, 0.070, 0.060 | it is now the 3 % surface film, not the water |
| `BaseColorSilt` | 0.46, 0.355, 0.225 | 0.165, 0.150, 0.112 | pale silt scum at 22 % coverage |

**The silt-plume work is not removed, it is promoted.** The ~180 m plume field and ~55 m along-flow streaks
`build_materials.py` builds still drive `BaseColor` and `Roughness`, and now also drive `Opacity` and the
scattering/absorption coefficients themselves. `tune_water.py` recovers that field by asking
`get_material_property_input_node(w, MP_BASE_COLOR)` for the lerp and reading its `Alpha` input — an
undocumented Python path, so it is **probed, not assumed**: if it fails the script still runs with constant
coefficients and constant surface coverage, and prints which path it took. Both paths are dry-run tested.

**`NormalStrength` caveat.** `build_materials.py` records that 0.6 "read as large dark blobs under a low dawn
sun". That was measured while the surface was an opaque **diffuse** card, where the normal modulates N·L
directly. At Opacity 0.03 the normal drives refraction and specular instead, where the same amplitude reads as
ripple. It is still the first knob to lower if the water looks lumpy.

**Brightness, honestly estimated.** Old: `BaseColor/π · N·L · E ≈ 0.0955 × 0.7 × E ≈ 0.067 E`. New:
`albedo_G × (E_sky·1/4π + E_sun·SchlickPhase(0.10)) ≈ 0.62 × 0.085 × E ≈ 0.053 E`, plus sky reflection and
refraction that did not exist before. So expect it slightly darker and far less flat. If it is too dark,
raise all three `Scattering` channels together; that is the one knob that moves brightness without moving hue.

**Idempotency.** Every node `tune_water.py` adds carries `desc = "SIGHTLINE_TUNE_WATER"`; a re-run creates the
new set, rewires, and only then deletes the previous set — so a failure part-way can never leave the SLW pins
dangling. If `MaterialExpression.desc` turns out not to be readable through Python it refuses to run a second
time rather than stacking duplicate lerps, and tells you to re-run `build_materials.py` first.

---

# D. What I looked at, and what I saw

Per the quality gate: no editor in this lane, so the visual proof is of the **inputs**.

| artifact | what I saw |
|---|---|
| `data/scene/vegetation_canopy.png` | Dense canopy on both hillslopes; a clear bare corridor down the channel; the settlement (red house markers) under a mostly-continuous canopy with gaps; a sparse deposit fan upper-left; the 70 m launch-pad disc bare. **First version had a bare grey halo all round the water** — that is how the `floodplain` zone was found. Nothing about it was visible in the counts. |
| `data/scene/utilities/preview_pole.png` | A 10 m tapered octagonal shaft (bbox 1.90 × 0.27 × 10.00 m), two crossarms, six pin insulators, an angled street-light bracket. Solid in all three orthographic views — **not** a flat quad. |
| `data/scene/utilities/preview_span.png` | Four poles at 36 m with four conductors each, visibly sagging ~0.7 m between them. Added specifically because that sag is invisible in a 540 m-wide render of the whole network. |
| `data/scene/utilities/preview_network.png` | Top view: one long line wandering through the settlement with short service-drop stubs, plus two shorter lines. Reads as a street-following distribution line, not a lattice and not a zig-zag. The first version was a 238-pole lattice of 21 feeders; that is why `MAX_POLES` exists. |
| `data/scene/utilities/preview_boat_vallam.png` | 6.20 × 1.28 × 0.62 m double-ended canoe, sheer rising at both ends, three thwarts, open interior. (The checkerboard shading is my preview renderer drawing both coincident faces; UE backface-culls one.) |
| `data/scene/utilities/preview_boat_dinghy.png` | 3.60 × 1.46 m, squared transom at one end, pointed at the other, visible rim and thwarts. |
| `data/scene/utilities/preview_boat_rescue_punt.png` | 4.80 × 1.60 m, near-parallel sides amidships, squared at both ends. First version was a bulbous oval; `fullness` was raised from 2.4 to 3.6 after looking at it. |
| the five tree glTFs | Parsed every accessor: leaf primitives are 1.06–2.40 M triangles of **real modelled leaves**, not alpha cards, and every diffuse JPEG is `RGB` mode with no alpha — which is what makes forcing Opaque safe and necessary. |

# E. The check that can fail — `check_realism.py`

```
$ D:\Tools\uv\uv.exe run python tools\scene\check_realism.py
vegetation: 4,377 trees + 1,151 understorey, closure 45.6 % ROI / 42.1 % settlement
utilities:  60 poles, 57 spans, 40 drops, 19 boats, 30,906 tris
217 checks run
all checks passed
```

It asserts, among 217 checks: source glTF integrity and triangle counts still matching the layout; every
clearance (house / survivor / launch pad); scale inside the species range; `base = ground − undercut − sink`
exactly, and the recorded ground inside the local terrain cell range; the lean encoding round-tripping through
UE's `FRotationMatrix`; every drowned crown above water; canopy closure > 30 %; OBJ index bounds, UV and
normal counts, **zero degenerate triangles**, no axis-degenerate bounding box; the network OBJ shifted to
`asl − base_z`; pole heights and separations; the wire ≥ 1.5 px at the GSD recomputed from
`camera_survey.json`; boat draught/freeboard against `z = 0`; boats floating in > 0.5 m of water.

It found six real defects while I wrote it: two poles 12 m apart (chain starts were not retiring nearby
sites), 3 zero-area triangles in two hulls (a lofted stem collapses to a point — UE would weld them and the
triangle-count assertion in `build_utilities.py` would then fail), a tree crown 6.22 m *below* the water, a
lean-encoding tolerance tighter than the stored rounding, a wrong ground-height comparison, and (via the
catalogue integrity check) `tree_small_02` shipped with a 9 KB glTF and no buffer at all.

**Proof it can fail** (`--self-test` corrupts the data four ways and requires all four to be caught):

```
  caught: tree inside a house -> 1 trees stand inside a house footprint + 2.0 m
  caught: tree floating in the air -> 1 trees have base != ground - undercut - sink
  caught: scale outside the species range -> 1 trees carry a scale outside their species range
  caught: triangle total does not add up -> instanced triangle total does not match the sum over items
self-test: 4/4 deliberate corruptions caught
```

**Determinism** is verified twice: in-process (two `build()` calls compared as sorted JSON) and across
processes — `vegetation.json` `aa024ad0…`, `utilities.json` `2b5eedf3…`, `network.obj` `b3fedd6b…`,
`boat_vallam.obj` `82f8c7e0…`, byte-identical over two full re-runs. `hash()` is not used anywhere (Python
salts string hashes per process).

**Editor-script dry runs**: all three editor scripts are exec'd against a mock `unreal` module in 13
configurations — plain `Material` vs `MaterialInstanceConstant`, asset present vs resolved-by-folder vs
imported, Nanite refusing to stick, a material reporting 0 instructions, a 2× wrong actor footprint, a missing
texture override, the plume probe failing, a missing water parameter, no SLW output node. Each configuration
asserts the script either completes or raises **for the right reason**. All 13 behave as specified.

# F. What I could NOT verify without the editor

The dry runs validate our control flow, JSON handling and bounds arithmetic. They cannot validate UE API
names, shader compilation or anything visual.

| risk | why it might bite | what the script does | if it still fails |
|---|---|---|---|
| **the water is now too dark / too green** | the sun-and-sky illuminance scale is not predictable analytically; only the hue ratios and the depth profile are | prints the extinction and albedo it applied | raise all three `Scattering` channels together for brightness; raise `Absorption.G` to pull green back; scale `Scattering + Absorption` together by the same factor to move where the shallows show through |
| **`MaterialExpression.desc`** may not be exposed to Python | it is the marker that makes `tune_water.py` idempotent | probed; if absent, the script refuses a second run rather than stacking duplicate lerps | re-run `build_materials.py`, then `tune_water.py` once |
| **the plume-field probe** (`ExpressionInput.expression`) is undocumented | pin introspection varies by version | falls back to constant coefficients and constant opacity, and prints which path it took | nothing to do; the Opacity fix and every retuned value still apply |
| **`PITCH_SIGN`** — the lean direction | UE's left-handed pitch sign | one named constant at the top of `build_vegetation.py` | flip it if the drowned trees lean upstream |
| **5,528 spawns and the level save** | this is the slow part now, not the import (the lite meshes total ~34 MB) | prints a warning before starting; actors are cleared by folder so a re-run is cheap | lower `--max-trees`, or convert the folder to HISM (below) |
| **Nanite + masked/two-sided leaves** | Nanite programmable raster is slower for these | asserts Nanite stuck; refuses to place above 8 M triangles without it | if the frame time is unusable, lower `--max-trees`; converting the `Vegetation` folder to HISM/foliage is the next optimisation and would also collapse 5,528 actors into ~6 components (at the cost of one segmentation colour for all foliage, doc §5.1 — acceptable, trees are background) |
| **glTF material slot names** | the importer may name slots differently | `build_vegetation.py` never matches tree slots by name; it fixes every material on the mesh | — |
| **imported mesh asset names** | `<pid>_LOD0` vs `<pid>_2k` differs between Poly Haven exports | `asset_hint` is a hint only; the real resolution is hint → largest mesh in the prop folder by bounding volume → import | prints which route it took per species |
| **`props.json` is owned by another lane** | it now carries the five species; a re-import could rename the assets | `asset_hint` is a hint only, resolution falls back to the folder, and the checker re-measures the glTF against the recorded count | re-run `gen_vegetation.py`, which re-reads props.json |
| **wires as dotted lines** | 2.04 px is thin | — | raise `WIRE_D_M` in `gen_utilities.py` and regenerate |

# G. Loose ends for the orchestrator

* **A second boat generator exists.** Another agent added `tools/scene/gen_boat.py` (+ `gltf_tools.py`,
  `gen_rubble.py`, `fetch_env_assets.py`) this session, producing a 7.5 m vallam as a glTF. Mine produces
  three archetypes as OBJs plus the **placement layout** (`utilities.json` → `boats.items`). They do not
  conflict — nothing is shared — but shipping both would put two different vallams in the scene. Cheapest
  merge: keep my placement and point `SM_Boat_vallam` at their imported glTF mesh (one `unreal.load_asset`
  swap in `build_utilities.py`); their hull is the higher-fidelity one, mine adds the dinghy and the punt.
* **The assets lane decimated the trees while I worked** (`<pid>_2k_lite.gltf`, 12-24x) and imported them, so
  `data/scene/props.json` now carries all five species under role `tree` with real asset paths. My generator
  picks that up automatically and every number above is from the lite meshes. Nothing needs doing; it is
  recorded here because an earlier draft of this report carried counts 11x larger.
* **`.gitignore`**: `data/scene/utilities/*.obj` is not covered by the existing `data/scene/*.obj` rule (nor
  is `data/scene/damage/*.obj`, already noted in `scene_polish.md`). Four small deterministic files, ~120 KB.
  `utilities.json`, `vegetation.json` and the preview PNGs should stay committed.
* **`docs/CONTEXT.md` deserves three new entries**, all measured here and all silent failures:
  1. the SLW `Opacity` / `WaterVisibility` trap and the `1/cm` coefficient unit;
  2. Poly Haven tree glTFs declare `alphaMode: BLEND` on leaf materials whose textures have no alpha —
     translucent leaves disable Nanite and break instance segmentation;
  3. Nanite must be **on** for the tree scans, which reverses doc §4's blanket "Nanite off on imported meshes".
* **Flood level**: the boats, the poles' eligibility band and the drowned-tree selection are all baked for
  `water_level_m = 1061.6814259297319`. If a scenario changes FloodLevel, re-run `gen_utilities.py` and
  `gen_vegetation.py`; moving actors in Z is not enough because the *eligible sets* change.
* **Not done, and named as such**: no banana/areca/coconut asset exists, so the mid-storey is the island-tree
  scans scaled 1.15–2.10× rather than the palms a Kerala homegarden would actually have. No hanging or
  storm-broken wires. No street furniture beyond the light brackets. The understorey is `fern_02` only.
