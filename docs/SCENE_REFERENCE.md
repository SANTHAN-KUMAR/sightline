# Scene reference — what the simulation must look like

The user supplied two reference photographs and said: **"make everything exist for now, but our goal is to match
this level of quality."** Breadth first, then fidelity. This file is the visual target; every scene lane works
to it. Read it with `docs/QUALITY_GATE.md`.

---

## Reference A — flooded settlement (the F1 flood zone)

An aerial view of a real flooded neighbourhood, roughly 40-60 m up, looking down at ~30 deg off nadir.

**What dominates the frame, in order:**

| Element | In the photo | In our scene today |
|---|---|---|
| **Mature broadleaf tree canopy** | **~half the frame.** Large rounded crowns, mid-green, many standing IN the water with only the crown showing | **PLACED 2026-09-11**: 4,342 trees + 1,166 understorey as 5,508 HISM instances; 44.9 % closure over the ROI, 40.0 % over the settlement; drowned trees lean downstream and crowns emerge from the water. sunlit crowns read mid-green; crown interiors and backlit crowns go dark under GI = None |
| Water | green-teal, not brown; strong specular sun glint in patches; depth readable through it near edges | uniform silty tan, flat, no depth cue |
| Houses | pitched shingle and metal roofs, water partway up the walls, some only roofs showing | present (73), correct behaviour |
| Utility poles + wires | poles standing in water, wires strung between them across the frame | absent |
| Boats | a small boat with people, moving between structures | absent |
| Debris | rafted against trees and structures, a bright pile top-right | present (919 items) |
| Ground vegetation | dense between structures wherever land is above water | ferns only |

**Reading:** the flood is *in a wooded neighbourhood*. Trees are the texture of the whole image and the main
occluder a survivor can be under. Their absence is why our frames read as empty mudflats.

---

## Reference B — collapsed structure / debris field (the F1 deposit-fan zone)

A ground-level view of a collapsed multi-storey building: a rubble field with rescuers walking on it.

**What it contains:**

| Element | In the photo | In our scene today |
|---|---|---|
| **Concrete rubble** | dense, chaotic: broken floor slabs at all angles, blocks, masonry, protruding **rebar** | **absent** — the fan has scattered natural boulders on clean mud |
| Slab geometry | large flat fragments stacked and tilted, forming voids — the voids are where survivors are | absent |
| Personal belongings | fabric, furniture, an appliance, scattered colour among the grey | absent |
| Collapsed buildings | multi-storey structures behind, floors pancaked, facades sheared | all 73 houses are pristine (damage scripts written, not yet run) |
| Palette | dust-covered, desaturated tan/grey, very low colour contrast | fan is red laterite / brown mud |
| People | rescuers standing ON the rubble, high-vis helmets | survivors exist, but on clean ground |

**Reading:** the debris-flow fan should read as a *structural collapse field*, not a rocky riverbed. This is also
where the hardest detection cases live: a person in a rubble void is the `trapped` posture and the
partial-occlusion slice, and it is the visual justification for section 2.7's burial boundary — some survivors
in that field genuinely cannot be seen from the air.

---

## Priority order (breadth before polish)

1. **Trees** — Poly Haven CC0: `island_tree_01/02/03`, `jacaranda_tree`. Biggest single gain.
2. **Concrete rubble field** on the fan — generated slab/block geometry is acceptable and is how the 73 houses
   were built; source rubble textures already exist (`dirty_concrete`, `brown_mud_rocks_01`).
3. **Building damage** — scripts exist (`gen_damage.py` / `build_damage.py`), not yet run.
4. **Water colour** — green-teal with glint and depth variation, keeping the silt-plume work.
5. **Poles and wires**, then **boats**.
6. Ground vegetation density.

## Asset sourcing — Fab does NOT work for this project

Fab's "Add to Project" lists nothing even with "Show all projects" ticked: the assets have no build for
**UE 5.8.2** (most top out at 5.4-5.6). The project *is* registered with the Launcher and the engine *is* an
installed build, so this is the asset side, not ours. Use instead, in order:

1. **Poly Haven** (CC0, no login, HTTP API `https://api.polyhaven.com/assets?t=models`) — already the source of
   27 prop sets and 120 textures. 521 models; 79 are tree/plant-like.
2. **ambientCG** (CC0, no login) — textures and some models.
3. **Generate the geometry** — proven here for 73 houses and 63 pose assets. Correct for rubble, poles, wires.
4. Fab only if the user downloads an asset manually and points at the folder.

## Gap review, 2026-09-11 00:52 — orchestrator eyeball on the utilities lane's renders

Two renders opened and read directly (not their manifest, the pixels). Both at
`_artifacts/editor_shots/`.

### `util_7_nadir45.png` — 45 m nadir, the altitude the dataset is actually flown at
Working: the flood water renders as a VOLUME for the first time (the `MP_Opacity` fix in `tune_water.py`
landed — `WaterVisibility` 0.97 instead of 0.00), foam appears at the house edges, a boat floats at the
surface rather than through it, and a survivor is legible on a flat cream roof.

Wrong:
1. **The water is a saturated tropical green.** Reference A is brown, silty monsoon flood water; this reads as
   a swimming pool. The hue has to move towards brown-green — red absorption up relative to green, or the
   green albedo down. Judge it against the photograph, not against a coefficient.
2. **Shadows on the water are crushed to near-black.** A survivor inside a house shadow at nadir is invisible
   to a detector and to a human reviewer, which silently destroys recall in the shadow slice. Exposure, sun
   angle or an over-strong AO — needs diagnosis, not a brightness nudge.
3. **The red tile roof carries a visible vertical seam** down the middle where the tiling changes. Building
   material / UV problem, owned by `build_buildings.py` + `build_roof_materials.py`.

### `util_9_shallow.png` — waterline close-up
4. **The shallow margin does not exist**, and this is the largest single realism gap in the scene. The water
   goes from fully opaque teal to dark mud across a hard geometric line. Measured from the terrain rather
   than asserted: the waterline slope is p10 0.022, p50 0.086, p90 0.280 m/m, so the 0 -> 0.3 m depth band is
   **3.5 m wide on the ground at the median slope — 195 pixels at 45 m AGL** — and 13.5 m (760 px) along the
   gentlest tenth of the shoreline. With the extinction `tune_water.py` reported (green transmittance 0.13 at
   0.50 m) that whole ribbon should show the drowned ground through it. It shows nothing, so the depth
   feeding the extinction is not the real water depth.
   For scale, the depth histogram says only 2.8 % of the flood is under 0.3 m and 67 % is over 2 m, so the
   deep terrace reading as opaque teal is CORRECT — it is specifically the margin that is broken.
5. **The foam is bright white, wide, uniform, and traces the waterline exactly** — clean sea surf. Flood scum
   is dirtier, narrower, broken up, and collects against obstacles rather than following a perfect contour.
6. **The grass above the line is a lush saturated lawn** with orange litter. Reference A is muddy and drab.
   Terrain material, owned by `build_materials.py`.

Items 1, 2, 4 and 5 are the ones that decide whether a frame looks like a flood. Items 3 and 6 are visible at
survey altitude and matter for the demo but not for label correctness.

### Corrections to the two entries above, 2026-09-11 01:05 — measured by the utilities lane

Two of the orchestrator's eyeball calls were WRONG and are corrected here rather than quietly dropped.

* **"The foam is bright white / blown out" — wrong.** Measured mean (174, 168, 149) with **0.0 % of pixels
  clipped**, which is essentially exactly the warm cream (174, 168, 156) its material intends. It only reads
  white by contrast against dark water. No change was needed and none was made.
* **"The depth feeding the extinction is not the real water depth" — wrong diagnosis, right symptom.** The
  depth was being sampled correctly; the extinction was simply about 3x too strong, so everything past a few
  centimetres was already opaque. `tune_water.py` now carries `TURBIDITY_DIVISOR = 3.0` (Secchi 0.35 m ->
  1.05 m via K_d ~= 1.44 / z_SD), which leaves the single-scattering albedo — and therefore the HUE —
  exactly unchanged and moves only depth: green transmittance 0.51 at 0.5 m, 0.26 at 1.0 m, 0.017 at 3.0 m.
  Submerged walls and window openings now read through the surface and fade with depth.
  Note the 7 m foam ribbon covers part of the shallow margin, so it masks some of the new depth cue right at
  the water's edge.
* Also found in passing: **`tune_water.py`'s own summary was printing hard-coded literals** (`clear R 6.2
  G 4.1 B 5.0`) about a material it had just written with 2.07 / 1.37 / 1.67. The numbers quoted in the
  entry above came from that lying summary. It now derives them from the values actually written.

The terrain finding is unchanged and was independently confirmed: near-black shadowed slopes with
fluorescent green ridge patches. `tools/scene/tune_terrain.py` is written and waiting for the editor.

## Whole-valley review, 2026-09-11 00:57:50 — `_artifacts/editor_shots/qa_1_valley.png`

Level state at the render: 1,093 actors (71 survivors, 73 houses, 919 debris, ~21 utility actors). No
vegetation and no rubble yet. Read directly from the image, ranked by how much each one costs us:

1. **The terrain is a uniform fluorescent green** across the whole 4.2 km2 — one flat saturated colour with
   faint orange leaf-litter speckles, reading as a golf course rather than a monsoon valley. Reference A is
   drab, muddy and broken up. This is the largest single realism gap and it is a terrain-material problem
   (`build_materials.py`), not a missing-asset problem: no amount of vegetation on top will fix the ground
   colour between the trees. Wants: desaturation, large-scale colour variation (macro variation / noise at
   50-200 m), and mud-to-grass transition driven by height above the flood line.
2. **A soft-edged black blob covers roughly a quarter of the frame** on the right, with a red-brown smear at
   its upper edge. Too smooth and too large to be terrain shadow. Candidates: a distance-field shadow or DFAO
   artifact, an over-large cloud-shadow texture on the directional light, or a shadow-cascade boundary. Needs
   diagnosis. It is the same crushed-to-black behaviour visible at 45 m nadir, seen at valley scale, and a
   survivor inside it is invisible to the detector and to a human reviewer.
3. **The shoreline is a hard white ring** tracing the water boundary exactly — a bathtub rim. Same root as
   the measured shallow-margin defect above.
4. **The mud patch upper-left is a hard-edged blob with visible tiling stripes** — the tide/mud layer needs a
   noisy mask boundary and a larger, less repetitive tiling scale.
5. **The sky is a flat blue gradient with a grey horizon band.** Cheap to improve and it sets the light for
   every frame the dataset captures.

Items 1 and 2 change every pixel of every training frame, so they outrank everything else in the polish
backlog.

## Lighting, water and terrain pass, 2026-09-11 01:35 — orchestrator, measured

### The lighting defect was ABSENT ambient, not weak ambient
Queried in the editor: `SkyLight intensity 1.0, source SLS_CAPTURED_SCENE, real_time_capture True,
cubemap None, lower_hemisphere_is_black True`, against `Sun intensity 8.0`. A captured-scene SkyLight in a
level whose `BP_Sky_Sphere` is hidden captures **black**, and `real_time_capture=True` re-captures every
frame and OVERRIDES any specified cubemap - which is why assigning the HDRI alone changed nothing (26.8 ->
21.6). Turning real-time capture off and cranking intensity to 20 collapsed the ratio to 1.5:1, proving the
SkyLight worked and isolating the cause.

Fix: Poly Haven CC0 `overcast_soil_puresky_2k` imported as a **TextureCube** (asserted - a Texture2D import
would be accepted and silently ignored), `SLS_SPECIFIED_CUBEMAP`, `real_time_capture` off,
`lower_hemisphere_is_black` off with a ground-bounce colour, intensity **3.0** solved from the 20x data
point rather than dialled by eye. GI stays OFF; a SkyLight is an ambient cubemap lookup, not GI, so this
costs nothing from the 8 GB VRAM budget.

| view | sun:shadow before | after | terrain below sRGB 0.05 |
|---|---|---|---|
| qa_1_valley | 26.8 : 1 | **4.1 : 1** | 3.3 % -> **0.0 %** |
| qa_4_nadir45 (survey altitude) | 45.0 : 1 | **6.8 : 1** | 8.1 % -> **0.0 %** |
| qa_2_settlement | 239.5 : 1 | **17.3 : 1** | 29.0 % -> **0.3 %** |

`tools/scene/measure_lighting.py` is the gate. It carries **no shadow-colour test**: two were tried and both
were wrong ("shadow is blue" holds only under a clear sky, and this scene is overcast; "shadow is cooler
than sun" then failed because vegetation bounce warms a shadow, which is correct physics). A test that
cannot separate a defect from correct physics was removed rather than tuned until it passed. The ratio is a
FAILURE only when accompanied by ground actually crushed to black - the harm the tool exists to prevent.

### Water: jade -> silt
Once real ambient arrived, the water read as pale milky jade. Cause was in the coefficients, not the sky:
single-scattering albedo `w = S/(S+A)` was R 0.34 / G 0.62 / B 0.52 - green-cyan. Silt is a mineral
suspension: it scatters long wavelengths and absorbs short ones, so red must survive and blue must die.
Re-derived to **R 0.56 / G 0.44 / B 0.17**, extinction held near 1.8-2.4 /m so the utilities lane's depth
cue survives. Written into `tune_water.py` itself, so a `build_materials.py` rebuild cannot revert it.

### Terrain
`GrassTint` 0.35,1.00,0.45 -> 0.26,0.50,0.50: grass linear G/R **2.11 -> 1.42**, G/B **4.87 -> 2.19**,
inside the 1.3-1.5 / 2.0-2.5 range real aerial vegetation occupies. Plus macro colour variation (1 +- 0.18
at ~130 m) and a 0.22 desaturation that takes the Mars-red out of the laterite. 543 -> 631 instructions.

### Still open after this pass, ranked
1. ~~**Distant trees render as dark skeletal twigs** at valley scale~~ - **CLOSED 2026-09-11 01:50 (sky lane).**
   Cause was Nanite's build-time simplification keeping the OPAQUE branch geometry and dropping the
   BLEND_MASKED leaf surfaces. Fixed with `shape_preservation = PRESERVE_AREA` on the 5 tree meshes
   (`tools/scene/build_nanite_foliage.py`): canopy coverage in the 450 m demo frame **32.3 % -> 60.3 %**
   against a 58.4 % near-field reference, at **zero runtime cost**. `VOXELIZE` and streaming-pool size both
   did nothing; see CONTEXT §7 for the measured curve. `check_sky.py`'s numeric LOD bar is still red at
   0.702 vs 0.85 and was deliberately not relaxed - the sweep shows no lever reaches it (TRACKER).
2. **Tree leaves are translucent and on engine placeholder materials** (`/InterchangeAssets/gltf/M_Default`)
   - foliage lane is on it.
3. ~~**The sky is a flat navy gradient.**~~ - **CLOSED 2026-09-11 01:50 (sky lane).** The navy turned out to be
   the **ExponentialHeightFog** inheriting the SkyAtmosphere's colour and saturating over distance, not the
   atmosphere being visible directly; `is_sky` excludes aerial perspective but not height fog. A 40 km unlit
   `is_sky` dome sampling the SkyLight's own TextureCube, plus `fog_cutoff_distance` 20 km, took the visible
   sky from **5.10x** the driving HDRI's blue-over-red to **1.02x**. The SkyLight was not touched.

## Still open, ranked (after the sky lane, 2026-09-11 01:50)
1. **The horizon shows the HDRI's lower hemisphere** wherever terrain does not cover it (visible in
   `_artifacts/sky_lane/sky_horizon.png` looking out past the 2 km terrain edge). It reads as haze / a distant
   rain curtain, which is plausible for a flood, but it is the HDRI's ground, not real geometry, and there is
   no fog past 20 km to soften it. Cosmetic; only affects frames aimed off the terrain.
2. **The terrain mesh ends in a hard straight edge against the sky** in wide oblique frames. Pre-existing.
3. **The lighting gate's terrain mask is `G > B`**, which excluded the old navy sky for free. With a neutral
   grey sky it no longer does so by construction; measured leakage into the decisive "sun" bucket is only
   1.1-1.5 % in the frames that contain sky, so the gate still measures terrain - but a future sky change
   should re-check that rather than assume it.
