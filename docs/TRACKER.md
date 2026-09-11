# Tracker

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done and verified · `[!]` blocked · `[-]` dropped (say why).
"Verified" means it was run and observed working, not just written. Update this file at the end of every chunk
of work and append to the Session log.

## Handoff state (end of session 2, 2026-09-10) — READ FIRST
- **Stopped mid-verification by the user.** Everything below is saved to disk and committed; nothing is half-written.
- **Editor may still be running** (pid 25948, launched with `-settings=sim/settings/default.json`) with a PIE
  session requested at the moment of the stop. Run `status` first; stop PIE (`LevelEditorSubsystem.editor_request_end_play()`)
  or `editor_close` before building anything. The `unreal` (Epic) MCP server did not connect at session start
  because the editor was down: reconnect it (`/mcp`) once the editor runs.
- **Session 2 resumed (2026-09-10 21:27):** items 1-3 below checked in PIE. Water renders (silt-brown, ripples,
  glints; capture `20260910-212806-083610`), the airframe blob is gone, and `sim_environment(time_of_day=
  "2024-07-30 06:30:00")` gives warm low dawn light with long terrain shadows (capture `20260910-212927-215281`), so
  the hidden BP_Sky_Sphere still drives the sun. Still wrong: terrain normals/ARM tile visibly at 45 m and the grass
  reads dry-yellow -> `build_materials.py` reworked (bigger tiles, near/far blend on all three maps weighted by a
  smooth macro noise, `GrassTint`, silt band 1.2 m, water `NormalStrength` 0.35); re-check after the rebuild.
- **Written but NOT yet verified** (do these first, in this order):
  1. `M_FloodValleyTerrain` with anti-tiling (84 nodes / 16 textures) and the rebuilt `M_FloodWater` (21 nodes,
     1 texture, compiles) have **not been seen in a capture yet**. The last capture (`_artifacts/captures/20260910-211540-830858`)
     predates both fixes: it showed visible 4-5 m tiling and the water as the grey default material.
  2. Survey camera moved to NED (0, 0, 0.30) in `sim/settings/default.json` + `capture_4k.json` to remove an
     airframe blob from the nadir frame corner. Not yet checked in a capture.
  3. SkyAtmosphere added and BP_Sky_Sphere hidden: check the red "sky light real-time capture" viewport warning is
     gone, **and that `sim_environment(time_of_day=...)` still moves the sun** (Cosys finds the sun through the
     hidden BP_Sky_Sphere's "Directional light actor" property; unverified since hiding it).
  4. `sim/SightlineSim/Config/DefaultEditorSettings.ini` sets `bThrottleCPUWhenNotForeground=False` in the correct
     section: takes effect at the **next editor launch**; confirm the editor keeps full frame rate in the background.
  5. `tools/sightline_mcp/server.py`: `sim_fly` now fails a `Ground` contact whose normal is > ~45 deg off vertical
     (hillside strike). Active only after the sightline MCP server restarts. **Re-run `tools/sightline_mcp/test_sim.py`
     and `test_tool_matrix.py`** after that restart; they were not re-run after this change.
- Verification flight recipe: PIE -> `sim_fly takeoff` (the first sim call right after PIE start often times out
  while PIE boots: retry once) -> `move_to` (0, -60, -40) = ~45 m above the flood surface over the east hillslope
  and silt margin -> `sim_capture camera=survey image_types=[scene, segmentation]` -> `rtl`.
- `PS2-Survivor-Vision-Research-and-Build-Document.md` in the repo root is the user's copy of the updated doc; it is
  identical to `docs/SOLUTION_DOC.md` and deliberately left untracked.

## Handoff state (session 3, 2026-09-10 23:40) — READ THIS FIRST, it supersedes session 2's handoff

**Read `docs/QUALITY_GATE.md` before touching anything.** Seven separate pieces of work this session passed every
programmatic check and were still wrong; all seven were caught by looking at a picture. Eyeball verification is a
hard rule now, not a nicety. `docs/SCENE_REFERENCE.md` holds the user's two reference photographs and is the
visual target.

### Verified working (seen in a render, not inferred)
- **Scene**: 1072 actors — terrain (Ground, vertex normals, no corduroy), FloodWater, 73 Kerala houses,
  919 debris items placed by flood transport physics, 71 posed survivors, full lighting rig (GI off).
- **Survivor poses (F1)**: 7 postures authored as real one-key AnimSequences on all 9 Rocketbox skeletons
  (63 assets), left/right symmetric to 0.00 cm, characters fully textured. Standing 44.5 cm wide with arms down,
  waving 141.8, prone 167.2 long — the silhouettes actually differ, which is the point at 20-95 px.
- **Ground truth**: survivor positions match the sim to **0.00 m**. Two "buried" survivors are now genuinely
  buried under a debris cap — verified 0 visible pixels, 0 labels emitted.
- **Camera**: calibrated `f_px = 2548.72`, HFOV 73.98 deg, residual RMS 5.6 px over 48 observations (which is
  also a validation of the nadir projection chain). `simGetCameraInfo().fov` reports 89.9 and is WRONG.
- **Capture (F5) by FLYING (F2)**: `sightline/mission/survey.py` flies a boustrophedon and shoots on the move,
  terrain-following, with a tilt gate. A 60-frame test validated clean; survivors measured 51-93 px.
- **Pipeline lanes**: ~20k lines across ingest/geo/track/dedup/triage/export/coverage/plan/store/api/eval.
  **541 tests pass** (144 geo/triage/export/store + 298 ingest/track/dedup + 101 coverage/plan/eval).
  10 real bugs found and fixed by those tests, listed in `docs/lanes/*.md`.

### NOT done / not verified
- **No usable training dataset yet.** Two runs were discarded (teleport artefacts). The full flown run is the
  next thing to finish and validate.
- **No model trained.** F8b is minimal by decision (see CONTEXT decisions) but has not run.
- **Thermal (F9b)**: `tools/capture/thermal_ids.py` written (object-ID temperature table, decodable to
  absolute C), **never run**.
- **Weather scripting** never exercised in this scene; **time of day** not re-verified since the sky rebuild.
- **Scene vs `docs/SCENE_REFERENCE.md`**: no trees (the single biggest visual gap), no rubble field on the fan,
  no poles/wires, no boats, houses all pristine, water reads tan rather than green-teal.

### Open finding — thermal (F9b) needs investigation
`tools/capture/thermal_ids.py` assigns a physically-motivated temperature to every object and encodes it as the
segmentation ID (`id = round((T_C + 20) / 0.5)`, so grey decodes back to absolute Celsius). All 1162 objects
accepted the ID, but **the Infrared pass then rendered 100 % grey 0** - nothing. Cosys 3.4.1 runs with
`InitialInstanceSegmentation: true`, and the suspicion is that the instance-segmentation path has superseded the
legacy per-object ID that `ImageType.Infrared` reads. Next step: check whether Infrared honours
`simSetSegmentationObjectID` at all in this build, or whether the **Annotation** layer
(`simSetAnnotationObjectValue` / `ImageType 11`) is the supported route - the client exposes a full annotation
API (`simListAnnotationObjects`, `simSetAnnotationObjectColor/Value/ID`) that is probably the intended one.
Do NOT issue sim calls while a capture flight is running: both share the single AirSim connection.

### Run order for the scene (all idempotent, PIE OFF)
```
uv run python tools/scene/gen_terrain.py            # host: OBJ + zone masks (now writes vertex normals)
ue_python exec build_flood_valley.py                # level, lighting, water
ue_python exec build_materials.py                   # terrain + water materials
ue_python exec build_buildings.py                   # 73 houses (asserts every material compiles)
uv run python tools/scene/gen_actors.py             # host: 71 survivors, seeded
ue_python exec build_poses.py                       # 63 pose assets (asserts symmetry + face direction)
ue_python exec build_characters.py                  # binds the Rocketbox textures (the FBX import binds none)
ue_python exec build_actors.py                      # spawns/updates survivors IN PLACE (never renames)
uv run python tools/scene/gen_props.py              # host: 919 debris + burial caps
ue_python exec build_props.py                       # places debris
ue_python exec qa_shots.py                          # ALWAYS: render and LOOK
```
Capture, validate, then look:
```
uv run python -m sightline.mission.survey --alt 45 --speed 11 --out _artifacts/dataset/<name>
uv run python tools/capture/validate.py _artifacts/dataset/<name>      # exits non-zero if unclean
uv run python tools/capture/contact_sheet.py _artifacts/dataset/<name> # then OPEN the sheet
```

## Handoff state (session 4, 2026-09-11 00:50) — supersedes session 3's handoff

### Established today, with evidence
- **The capture path is correct; the flight plan was not.** The re-flown survey
  (`_artifacts/dataset/train_seed23`, 285 frames, flown not teleported, pre-flight segmentation gate passed)
  returned 56 boxes over 30 survivors. That is not a bug: the geometric prediction for the flown box was 42
  boxes at frame 209 and the run had produced exactly 42. AGL held 45.9 m median (37.0-52.9), per-frame GSD
  1.78-1.83 cm/px, zero frame-spanning labels. The 5-95 % quantile box is 607 x 960 m and covers only 28 of
  the 69 detectable survivors, so most frames photograph empty valley.
  -> `survey.py` gained `--plan patches` (single-link clusters, one small lawnmower each), `--dry-run`,
  `--jpeg Q`, and `--time-of-day` / `--rain` / `--fog` / `--condition`, which are recorded in the data card
  as a slice axis. Planned campaign: 5 patches covering all 9 pose/submersion/zone combinations, seed 23 at
  35/55/80 m and a held-out pass at 55 m, about 1,400 frames and 770 boxes in roughly 47 minutes.
- **The editor dies of the Windows commit limit at a few thousand actors** (`PeakUsedVirtual 27.49 GiB`
  against a 26.7 GB limit; `UsedPhysical` was only 5.15 GiB). Bulk scenery must be HISM instances, not one
  actor per item. See `docs/CONTEXT.md`.
- **`unreal.BlendMode.BM_OPAQUE` does not exist in UE 5.8** (`BLEND_OPAQUE` does). The offline mock in
  `check_realism.py` answered any attribute with a lambda, so the typo passed 217 checks. UE enums in that
  mock are now strict and raise on unknown members.
- **Vegetation cleared trees by trunk distance (3 m) against 8-16 m crowns**, putting 22 of 71 survivors
  recorded `occlusion: 0` under a canopy. Fixed and re-measured on the finished list; the generator now
  refuses to write a layout that contradicts `actors.json`. Cost: 35 trees of 4,377.
- **`actors.json` occlusion is an intent that five generators contradict** — 25 of 71 disagree with the
  finished geometry, in both directions. New: `tools/scene/reconcile_occlusion.py` (geometric, pre-flight),
  `tools/capture/measure_occlusion.py` (per observation, from `visible_px` and the frame's own GSD - the
  only occlusion number that is observed rather than assumed), `tools/scene/check_occlusion_truth.py`
  (cross-checks every layout against the survivor ground truth; exits non-zero).
- **Water was rendering as a flat lit card.** `tune_water.py` has been run: `MP_Opacity` is now connected, so
  `WaterVisibility = 1 - Opacity` is 0.97 instead of 0.00 and the volumetric term is no longer multiplied
  away. 1132 instructions, 7 texture samples.
  **Eyeball judgement now made (utilities lane, 2026-09-11 00:57): the hue was right and the depth scale was
  wrong.** At the derived coefficients (green extinction 4.1 /m, transmittance 0.017 at 1 m) the flood
  rendered as a FLAT FEATURELESS GREEN FIELD from 45 m nadir at the real survey GSD, with nothing readable
  through it at any depth - not even the 17 cm at the shoreline. `TURBIDITY_DIVISOR = 3.0` now scales all
  four optical coefficients, which is Secchi 1.05 m instead of 0.35 m (`K_d ~= 1.44 / z_SD`). Dividing
  Scattering and Absorption by the SAME factor leaves the single-scattering albedo untouched, so the
  green-teal hue and the clear/plume contrast are preserved exactly and only the depth scale moves: green
  transmittance 0.51 at 0.5 m, 0.26 at 1 m, 0.017 at 3 m. Submerged walls, kerbs and the shallow margins now
  read through the water while the deep terrace stays opaque teal.
  Also fixed: `tune_water.py`'s own summary line printed the pre-divisor extinction figures as **literals**,
  so it reported "clear R 6.2 G 4.1 B 5.0" about a material it had just written with 2.07/1.37/1.67. It now
  derives that line from the values it actually wrote.
- **RunPod is wired and verified**: `tools/train/runpod_train.py` (plan/up/push/train/logs/pull/down/status),
  balance $14.81, no pods billing. Note the Cloudflare 403 trap: a default `Python-urllib` User-Agent is
  refused with "error code: 1010" even with a valid key.
- **The editor is a single-writer resource**: `tools/scene/editor_lock.py`, self-tested including refusing a
  foreign release. Every lane must hold it around its `ue_python` calls.

### In flight (four parallel lanes, 2026-09-11 00:48)
~~vegetation~~ **LANDED, see below** (4,342 trees + 1,166 understorey -> HISM), ~~rubble+damage~~ **LANDED, see
below** (1,799 rubble instances, 18 damaged houses, 67 roof debris), ~~utilities+waterline~~ **LANDED, see
below** (60 poles, 57 spans, 40 drops, 19 boats, foam ribbon, water verify), and a red team that renders its
own views and disbelieves all three.

### Lane LANDED: utilities + waterline (2026-09-11 00:57)
Placed and verified in `FloodValley`, level saved, editor alive:
- **Utility network**: 1 actor (`Utilities`, `SM_Utilities`, 22,704 tris) carrying 60 poles in 3 chains, 57
  spans x 4 conductors and 40 service drops. All 60 poles stand in the flood. `M_Wire` created (330
  instructions); slots reuse `MI_Concrete` / `MI_RoofSheet_Rusty`.
- **Boats**: 19 actors across 3 variants, painted by 4 new `M_PBR_Master` instances (`MI_Boat_*`, 351
  instructions each). Every hull's waterline is within 2.9 cm of the flood surface.
- **Foam**: 1 actor (`SM_FloodFoam`, 2,573 quads / 5,146 tris) over ~9.38 km of shoreline. `M_Foam` 351
  instructions, 4 samples. **ONE mesh, not 2,573 actors** - the constraint that killed the previous attempt.
- **Cost: 21 actors** (1,072 -> 1,093). `AvailablePhysical` 1.99 -> 2.46 GiB, `AvailableCommit` 1.54 -> 3.94
  GiB (Windows grew the pagefile: limit 28.72 -> 32.80 GiB), editor `UsedVirtual` 6.35 GiB, peak 6.93 GiB.
- **New check that can fail**: `tools/scene/check_utilities_placed.py` (25 assertions, all green). It walks
  the imported mesh's real VERTICES through the actor transform - a bounding-box check passes on a mesh that
  is mirrored or rotated 180 deg - and pins each pole's shaft axis to 5 cm, its buried butt to 5 cm and its
  top to 40 cm. `SELFTEST=True` injects a 50 cm pole shift plus a 1-triangle foam error and passes only if
  the check fails. **That self-test immediately caught a hole in the check itself**: the first version only
  asked "is there geometry within POLE_R (60 cm)", which a 50 cm shift sailed straight through, so pole
  positions were only pinned to +-0.6 m. It now measures the shaft-axis centroid over the lowest 2 m.
- **`POLE_BURY_M = 1.2` is real, not an error.** The first run of the check failed all 60 poles by exactly
  120.0 cm because it expected shafts to start at ground level; `gen_utilities.py:81` deliberately continues
  the geometry 1.2 m below ground so a pole on a slope never shows a gap. The check now reads that constant
  out of the generator and asserts the burial, which is stricter than what it replaced.
- **Wires are NOT dotted.** At the real survey GSD (1.7656 cm/px, `util_10_nadir_gsd.png`) the 3.6 cm
  conductors render as continuous lines. What reads as dashed in wider frames is the wire SHADOW on the
  water, not the wire. Judge wire continuity only at survey GSD; anything wider samples the conductor below
  1 px, where every line aliases into dashes.
- **New QA views**: `tools/scene/qa_utilities_shots.py` writes `_artifacts/editor_shots/util_*.png` (11
  frames) framed so each specific failure would be visible - pole verticality and submersion, catenary sag
  side-on, conductor continuity, boat waterline at a 6 deg graze, foam z-fighting at a 4 deg graze, and the
  shallow-margin transparency test that caught the water problem.

### Lane LANDED: vegetation / canopy (2026-09-11 01:05)
Placed and verified in `FloodValley`, level saved, editor alive. **`docs/SCENE_REFERENCE.md` priority 1 is done:
the scene is now a wooded flooded neighbourhood rather than a mudflat.**
- **5,508 instances on 9 actors.** 4,342 trees (5 Poly Haven species) + 1,166 understorey ferns, placed from
  `data/scene/vegetation.json` (treated as read-only input; the generator was NOT re-run). Level 1,093 -> 1,102
  actors. Folder `Vegetation`, all components `STATIC` + `NoCollision`.
- **The 5,508-actor crash is fixed by instancing, not by shrinking the layout.** The previous attempt spawned one
  `StaticMeshActor` per plant and died inside `SpawnActorFromObject` at `UsedVirtual 27.48 GiB` /
  `AvailableVirtual 0.00 GiB` (a COMMIT exhaustion, not RAM). Now: one
  `HierarchicalInstancedStaticMeshComponent` per species, per-instance transforms in a flat array.
  **The placement itself cost no measurable commit at all** - `UsedVirtual` was 7.49 GiB before the first
  instance and 7.49 GiB after the last one; the whole 5,508 went in in under a second. Editor peak for the
  entire run `PeakUsedVirtual 8.63 GiB` against a 32.80 GiB commit limit, ending at `AvailablePhysical 2.89 GiB`
  / `AvailableVirtual 4.09 GiB`. `build_vegetation.py` carries a hard `MAX_ACTORS = 400` tripwire and a memory
  guard that aborts the script (keeping the editor alive) below 1.2 GiB of available commit.
- **UE 5.8 Python does NOT expose `Actor.add_component_by_class`** (absent from `dir(unreal.Actor)`). The route
  that works is `SubobjectDataSubsystem.k2_gather_subobject_data_for_instance` + `add_new_subobject` - the same
  path the Details panel's "+ Add Component" takes. Verified to serialise by writing 3 instances, saving,
  **reloading the map from disk** and reading all 3 back unchanged, before committing to the full run.
- **Persistence is proven, not assumed.** `tools/scene/dump_vegetation.py` with `RELOAD_FIRST=True` reloads the
  .umap from disk and reads every instance transform back out of the components; all 5,508 came back.
- **New check that can fail**: `tools/scene/check_vegetation.py` (11 assertions, all green, exits non-zero).
  It compares the ENGINE-side transforms with the layout and with `actors.json`: per-species counts exactly,
  every instance matched 1:1 to a layout record within 0.02 m / 0.15 deg / 0.002 scale, and no tree crown over
  a survivor recorded `occlusion: 0` - with the crown radius taken from the catalogue's `crown_major_m` x the
  instance's own scale, so a generator that miscomputed its own clearance cannot hide behind it. **Exercised in
  both directions before the real data existed**: a synthetic perfect dump passes, and a sabotaged one (one
  jacaranda dropped, one tree moved 5 m, one crown parked on `Human_000`) fails with 5 named errors.
- **Opaque leaves are correct, and this was checked from the picture.** Every Poly Haven leaf texture is a
  **JPEG**, and JPEG carries no alpha channel, so the glTF's `alphaMode: BLEND` has no opacity source at all and
  `BLEND_MASKED` would have nothing to sample. The leaves are real scanned geometry: the close-up
  (`veg_after_4_leaf_closeup.png`) shows individual serrated compound leaves with veining and daylight through
  the gaps. Not cards, not bald - so Nanite did not fall back.
- **The trees lean DOWNSTREAM; `PITCH_SIGN = +1.0` is right.** Measured, not argued: the engine's own
  `MathLibrary.transform_location` of a point 19.3 m up the trunk, over 299 tilted jacaranda instances, gives a
  mean lean azimuth of **178.9 deg** (resultant 0.915) - due south. Individual trees lean exactly along their
  planned yaw.
- **Deviation from the layout's assumption, deliberate**: `gen_vegetation.py` budgeted for the `_2k_lite`
  variants, but what `import_assets.py` actually imported is the FULL scans (`<pid>_2k`, ~256 MB of Nanite mesh
  data across the 5 species). They are kept: Nanite streams the geometry, the render is better, and the
  no-Nanite ceiling never applies because Nanite is ON and asserted ON. The layout's "889,641 unique triangles"
  line therefore understates the real geometry.
- **Density verified numerically against the render, not by eye alone.** Monte-Carlo crown coverage over the
  exact rendered footprints: hillslope nadir 73.7 % (target 0.72, and the image reads ~75 %), settlement nadir
  29.8 % geometric / 20.3 % once submerged crowns are discounted. The settlement frame is a locally sparse patch
  of the 40.0 % settlement-wide closure, not a placement failure.
- **New QA views**: `_artifacts/vegetation/shots/veg_{before,after}_*.png` - the SAME 7 cameras rendered before
  and after placement, so the canopy claim is a comparison rather than an assertion.
- **Sequencing lesson for every lane: run `qa_shots.py` AFTER the final `save_current_level()`, not before.**
  This lane rendered at 01:03:56 and then did a cleanup save at 01:04:27, and that 31-second gap alone turned
  `assert_qa_fresh.py` red - the .umap was newer than the render. The gate is whole-scene and shared, so with
  concurrent lanes it can only be green for the lane that renders LAST.
- **Known, not fixed (fidelity, not correctness)**, and stated carefully because the first reading of it was too
  harsh: **sunlit crowns read as proper mid-green with distinct leaf highlights** (`qa_2_settlement.png` after
  the 01:19 re-render is the clearest example, and matches the reference). What goes near-black is crown
  INTERIORS and backlit crowns, because GI = None gives them no bounce light - so low-sun and into-the-sun
  frames are much darker than the reference photo. Separately, at long range (the 450 m `qa_1_valley` camera)
  the Nanite LOD drops the sub-pixel leaf geometry and distant trees read as dark skeletons. At the 40-60 m
  survey altitude the mission actually uses, the crowns are full and leafy (`qa_4_nadir45.png`).

### Lane LANDED: rubble + structural damage (2026-09-11 01:20)
Placed and verified in `FloodValley`, level saved, editor alive. **`docs/SCENE_REFERENCE.md` priorities 2 and 3
are done: the deposit fan is a structural-collapse field, and 18 houses are damaged.**
- **1,799 rubble instances on 41 actors + 67 roof-debris instances on 15 actors + 18 damaged houses.** Level
  1,102 -> 1,161 actors (+59 this lane, against the 400 tripwire). Folders `Rubble` and `Damage`, every
  component `STATIC` + `NoCollision`. Layouts treated as read-only input; neither generator was re-run.
- **Instanced, following the vegetation lane's route.** `build_rubble.py` and `build_damage.py` were rewritten
  from one `StaticMeshActor` per item to one `HierarchicalInstancedStaticMeshComponent` per mesh variant (41 =
  40 variants + a second fabric colour for the mattresses, which an instance cannot carry as a per-instance
  material override). Roof debris: 67 pieces across only 15 distinct meshes -> 15 actors. Memory for the whole
  run: `AvailableVirtual` 2.49 -> 3.08 GiB, `UsedVirtual` 7.87 -> 8.18 GiB, `PeakUsedVirtual` 8.63 GiB against a
  32.80 GiB commit limit. Both scripts carry `MAX_ACTORS = 400` and the same memory guard.
- **`unreal.Transform.rotation` is a QUAT, not a Rotator.** The first version assigned a `Rotator` to it; the
  offline mock modelled it as a Rotator too and passed. Only the live editor rejected it. Both build scripts now
  convert explicitly and PROVE the transform builder round-trips (translation, roll/pitch/yaw, scale) before
  building 1,799 transforms on it. `check_rubble.py`'s mock now models the Quat and raises on a Rotator.
- **Two real bugs found by LOOKING, after 856 programmatic checks had passed:**
  1. **`DustColor` was authored as sRGB and consumed as LINEAR.** `LinearColor(0.74, 0.70, 0.62)` is sRGB
     (0.885, 0.868, 0.826) - near white. Lerping dirty_concrete's measured linear albedo (0.257, 0.236, 0.196)
     halfway to that put every slab at ~0.49 linear and blew it out to featureless white paper in daylight - the
     opposite of Reference B. Fixed to a near-neutral linear (0.40, 0.385, 0.36) with lower dust amounts.
     Measured on the rendered slab: luminance **0.687 -> 0.541**, grain(std) **0.0201 -> 0.0358**, saturation
     0.126 -> 0.091.
  2. **Every plate wore the same concentric stain at 45 m.** `gen_rubble.py` writes UVs in tiles of
     `TILE_M` (concrete 2.0 m) and most faces are 1-3 m, so at `UVScale = 1.0` each face mapped very nearly ONE
     whole copy of the 2K scan - so all of them looked identical. Raised to `UVScale` 2.5 (concrete) / 2.0
     (masonry); the motif is gone in `qa_rubble_3_nadir45.png`.
- **New checks that can fail**: `tools/scene/verify_rubble.py` (editor side, measures and decides nothing) +
  `tools/scene/check_rubble.py` (host side, **867 assertions, exits non-zero**). It asserts per-variant instance
  counts against `rubble_layout.json` exactly, instance 0 of each component read back in WORLD space within 1 m
  of its layout record, Nanite OFF and triangle counts intact, every material non-zero instructions AND texture
  samples, no component on `WorldGridMaterial`, the 18 planned houses carrying the damaged mesh the plan names
  and no other house carrying one, the `.umap` on disk containing the HISM class name, and the lane's actor
  ceiling. It also re-derives the palette from the material parameters against the **measured** source-scan
  albedos, and measures the rendered slab's luminance / saturation / grain from the PNG. Plus 13 dry runs of
  both build scripts against a strict mock `unreal` (8 must fail for a named reason). `--self-test` corrupts the
  real report 7 ways and catches 7/7; the OBJ-unit check was separately proven by feeding it a metre-scaled OBJ.
- **Cross-lane repair, named plainly**: the level held **73** `House_###` actors while `settlement.json`
  declares **76** - `gen_buildings.py` grew the settlement by three `zone=bank` houses (73, 74, 75) and
  `build_buildings.py` was never re-run, so `damage.json` planned damage for house 74 and `build_damage.py`
  failed correctly. **Re-running `build_buildings.py` would have silently undone the roof-materials lane**: it
  rebuilds `M_PBR_Master` from an empty graph, and `build_roof_materials.py` clears that same graph to add
  anti-tiling and the tide line. New `tools/scene/place_missing_houses.py` places only the missing ids using
  `build_buildings.py`'s own placement block and touches no material graph. It is self-healing - a
  `build_buildings.py` re-run destroys every `House_*` and respawns all 76.
- **Known, not fixed (generator-side, out of this lane's scope)**: `gen_rubble.py`'s `Mesh.poly` computes UVs
  per polygon from that polygon's own first edge, so a fan-triangulated slab face gets a different texture
  orientation per triangle - a visible quilt of wedges at 6 m (`qa_rubble_6_scale.png`). Raising `UVScale` makes
  it more visible close up while removing the far worse repeated-motif artefact at the 40-60 m survey altitude
  the dataset actually uses, so it was traded deliberately. Fixing it properly means changing the UV
  construction in `gen_rubble.py` and regenerating all 40 OBJs.
- **QA views**: `_artifacts/editor_shots/qa_rubble_1..8_*.png`. Camera selection scores 20 m cells by structural
  load, penalises nearby trees and requires 4 m of freeboard over the flood - the first version pointed three
  cameras into the canopy and one at the waterline.

### Lane LANDED: foliage materials (2026-09-11 01:38)
The 5 tree meshes now use real project foliage materials. **No instance was placed, moved or touched**; this
lane changed material assets and 15 mesh material slots only. Level saved, editor alive.

- **New project assets, all under `/Game/Sightline/Vegetation/Materials/`:** `M_Sightline_Foliage_Leaf`
  (BLEND_MASKED, two-sided, `MSM_TWO_SIDED_FOLIAGE`, opacity-mask clip 0.33, 399 instr / 3 samples / 21 nodes)
  and `M_Sightline_Foliage_Bark` (BLEND_OPAQUE, two-sided, `MSM_DEFAULT_LIT`, 359 instr), plus **15 instances**
  `MI_<source material name>` — one per slot on the 5 tree meshes. **No tree material reaches
  `/InterchangeAssets/` or `/Engine/` any more**; before this, all 15 slots ran
  `MI_Default_{Blend,Opaque}_DS` -> `/InterchangeAssets/gltf/M_Default`, i.e. the glTF importer's stock
  defaults inside an engine plugin.
- **Two audit claims were WRONG and are corrected here, with measurements.**
  1. *"Crown interiors and backlit crowns render near-black."* **Not reproducible.** Measured at the 45 m
     nadir over hillslope canopy, BEFORE any change: canopy median sRGB luma 0.512 against sunlit ground
     0.692 — a ratio of **0.739** (sim), and 0.0 % of canopy pixels below luma 0.06. The canopy was not
     near-black. (The vegetation lane's own note at line ~241 already said sunlit crowns were fine.)
  2. *"JPEG cannot carry alpha, the leaves are modelled geometry, so there is nothing to mask, so force
     Opaque"* (`build_vegetation.py`'s reasoning). **All three steps are wrong.** The leaf diffuse atlas is
     three compound fronds on a **pure black background** (measured median luma 0.000, only 27.2 % of texels
     green-dominant) — an alpha-card atlas. The Interchange importer composed an alpha channel into the
     imported texture (`<pid>_leaves_diff-<pid>_leaves_alpha`, `compression_no_alpha=False`). And that alpha
     was demonstrably live: blue sky showed through the canopy in the pre-fix renders, which cannot happen if
     the blend really were opaque and the alpha really were ignored.
- **What was actually broken**, and is now fixed: (a) every leaf slot was `MSM_DEFAULT_LIT`, so a leaf — a thin
  translucent sheet — had no transmission term at all; (b) the effective blend was translucent no matter what
  the instance's `base_property_overrides` claimed, which is not rendered by Nanite, does not write the
  depth/stencil the instance-segmentation labels are read from, and is unbounded overdraw at 4,342 trees. It
  also produced the **white haze smeared over every crown** — accumulated alpha-blended card layers washing
  towards the bright sky, identical in frontlit and backlit frames because it was never lighting.
- **Subsurface transmission**, the reason for `MSM_TWO_SIDED_FOLIAGE`: `SubsurfaceTint` (0.38, 1.0, 0.23) at
  `SubsurfaceStrength` 0.55, multiplying the leaf's own base colour. Those are published broadleaf
  transmittance ratios (~0.05 at 660 nm, 0.13 at 550, 0.03 at 450), normalised to green. **The first attempt
  used the same ratios scaled 2.6x and it was measurably wrong**: 0.5 base green x 2.6 clips the channel to
  1.0, the backlit crown came out BRIGHTER than the frontlit one (luma 0.608 vs 0.593 — backwards), and the
  canopy read as electric lime. Caught by looking, not by a gate; all gates passed in that state.
- **Wind: deliberately NONE.** RGB and segmentation are separate render passes over a still frame; any vertex
  animation advances between them and every box and mask silhouette goes stale by however far a leaf moved.
- **Measured before -> after (sim, 45 m nadir over hillslope canopy, 66.2 % closure, same camera and exposure):**
  canopy/sunlit-ground luma ratio 0.739 -> **0.664** (canopy correctly DARKER than open ground once the white
  translucent haze is gone); canopy median rgb (0.498, 0.533, 0.357) -> **(0.420, 0.498, 0.224)**; backlit
  crown luma p50 0.296 -> **0.375**; near-black fraction 0.0 % throughout.
- **Files**: `tools/scene/build_foliage_materials.py` (the fix), `tools/scene/qa_foliage.py` (dump + 12
  renders), `tools/scene/check_foliage_materials.py` (the check that can say no),
  `tools/scene/gen_foliage_qa_views.py` + `data/scene/foliage_qa_views.json` (camera selection).
  Artifacts in `_artifacts/foliage/`.
- **Still broken, named plainly**: (1) the **ferns are out of scope and untouched** — all 4 `fern_02` meshes
  (1,166 instances) still run `MI_Default_Mask_DS` -> `/InterchangeAssets/gltf/M_Default`, an engine-plugin
  default with no foliage shading model. Same fix applies; `build_foliage_materials.py` would extend to them
  in a few lines. (2) A **faint pale/specular wash remains on a subset of leaves** at close range; leaf
  roughness is the asset's own measured 0.51 and was not overridden, so this is a candidate for the lighting
  lane rather than an invented material constant. (3) The **backlit crown is still marginally brighter than
  the frontlit one** (0.375 vs 0.364); with the sun at ~55 deg elevation the two near-horizontal cameras
  differ less than their names suggest, so this is probably view geometry, not the material — unconfirmed.

### Lane LANDED: sky + distant-foliage LOD (2026-09-11 01:50)
Two defects from `SCENE_REFERENCE.md`'s "Still open" list. **No actor was placed, moved or destroyed**; this
lane added ONE actor (`SkyDome`), one material, one fog property and one Nanite build setting on 5 existing
meshes. Level and assets saved, editor left alive, lock released. All numbers domain=sim.

- **The visible sky now IS the light.** The level was lit by the overcast HDRI through the SkyLight's
  specified cubemap while the camera saw the SkyAtmosphere's Rayleigh sky. Measured against the driving HDRI
  read off disk (`overcast_soil_puresky_2k.hdr`, upper hemisphere B/R **0.992**):

  | | sky B/R | x the light | saturation |
  |---|---|---|---|
  | before | 5.061 | **5.10x** | 0.905 |
  | after | 1.014 | **1.02x** | 0.010 |

  0.0 % clipped, luma 0.687. Built by `tools/scene/build_sky.py`: `/Game/Sightline/Sky/M_SkyDome` (unlit,
  opaque, two-sided, `is_sky`, 110 instr / 4 samples) on a 40 km `SM_SkySphere` named `SkyDome`, sampling the
  SkyLight's own TextureCube with `-CameraVectorWS`. Brightness is a **scalar** parameter, so it cannot shift
  the sky's colour away from the light's.
- **The root cause was the height fog, not the SkyAtmosphere** — see CONTEXT §7 for the four-configuration
  experiment that isolated it. Fixed with `fog_cutoff_distance` 0 -> 2,000,000 cm; terrain haze untouched.
- **THE SKYLIGHT WAS NOT TOUCHED, and that is enforced, not promised.** `build_sky.py` copies its five
  load-bearing values out before doing anything and re-asserts them at the end: still
  `SLS_SPECIFIED_CUBEMAP`, the overcast cubemap, `real_time_capture` **False**, `lower_hemisphere_is_black`
  False, `intensity` 3.0. GI stays off (`r.DynamicGlobalIlluminationMethod` 0).
- **Distant crowns: fixed visually, at zero runtime cost.** `shape_preservation` NONE -> **PRESERVE_AREA** on
  the 5 tree meshes (`tools/scene/build_nanite_foliage.py`, ~2 min rebuild). On the 450 m demo frame, canopy
  crop darkest-quartile G/R **0.723 -> 0.879** and canopy coverage **32.3 % -> 60.3 %**, against a 45 m
  survey-altitude reference of 58.4 % — i.e. the distant canopy now carries **the same mass as the near
  canopy**, which is the property that was broken. The eyeball is unambiguous: burnt-orchard brown skeletons
  became a closed green canopy (`_artifacts/sky_lane/zoom_far_canopy.png` vs `zoom_after_preserve_area.png`).
- **`VOXELIZE` did nothing** despite its docstring describing this exact defect; the voxel path looks inactive
  in 5.8.2. **Streaming pool size did nothing** (0.723 -> 0.722). Both recorded in CONTEXT §7 so nobody
  re-runs them.
- **LEFT RED, HONESTLY: `tools/scene/check_sky.py` still fails its LOD sub-test.** Its bar is far/near
  darkest-quartile G/R >= 0.85; we reach **0.702**. The bar was written before the achievable ceiling was
  known, and the sweep shows **no available lever reaches it** — the best combination measured
  (PRESERVE_AREA + `r.Nanite.MaxPixelsPerEdge` 0.5) tops out at 0.784, and 0.1 is worse than 0.5. The metric
  has an inherent ceiling because a distant canopy crop mixes ground and haze into its dark quartile while
  the near reference is pure leaf shadow. **The threshold was deliberately NOT relaxed to make it pass**;
  the coverage figure above is the parity evidence, and the next lane can decide with the measured curve in
  CONTEXT §7 rather than with a green tick that would have been a lie.
- **`r.Nanite.MaxPixelsPerEdge` was NOT changed** (left at 1.0). It is a global per-frame cost paid on every
  dataset frame at 35-80 m, where the crowns were already correct; buying the wide demo shot with a permanent
  tax on the dataset render is the wrong trade. It remains available for a one-off demo render at 0.5.
- **Gates**: `measure_lighting.py` **exit 0** (valley 3.7:1, nadir45 8.8:1, settlement 20.1:1; terrain below
  sRGB 0.05 = 0.0 / 0.0 / 0.4 %, all under the 2 % limit). `assert_qa_fresh.py` **exit 0**.
  `check_sky.py` sky sub-tests pass, LOD sub-test red as above.
- **Flagged for the dataset lane, not verified here (PIE was off all session):** `SkyDome` is real geometry
  where there used to be none, so oblique frames now have sky at ~40 km depth instead of the far plane, and
  it could take an instance-segmentation id. `render_custom_depth` is explicitly forced **off** to keep it
  out of Cosys' stencil pass, but that needs confirming in an actual capture.
- Memory across the lane: AvailablePhysical 1.28 -> 1.9 GiB, VRAM 2,548 / 8,188 MB. The editor **did die once**
  mid-sweep, caused by this lane setting a 2048 MB Nanite streaming pool (CONTEXT §7); the level and material
  had been saved beforehand and nothing was lost. `sweep_nanite_foliage.py` now refuses that value.

### Lane LANDED: live integration - F2 real-time loop + F3 gamepad takeover (2026-09-11)

**The demo loop exists.** `sightline/mission/live.py` flies the survey, runs detect -> geolocate -> track ->
dedup -> triage on each frame **as it is captured**, and pushes each record update to the running C2 so pins
appear on the map while the drone is still flying. Screenshots below; the map was photographed mid-flight.

#### What was built
- **`sightline/mission/pattern.py`** - the survey plan, terrain follower, cadence gate and AirSim capture
  helper, EXTRACTED from `survey.py`'s `main()` (they were local closures, so a second runner had to copy
  them). `survey.py` now imports them; there is one lawnmower, one heightfield sampler, one cadence gate.
- **`sightline/pipeline.py::FramePipeline`** - the per-frame spine. The offline replay and the live runner
  both call `process()` and nothing else, so they cannot drift. Dedup and triage now run per frame, not once
  at the end; that is what makes a record appear during the flight.
- **`sightline/mission/takeover.py`** - F3. `TakeoverMachine` (pure logic, no I/O), three control sources
  (Cosys-AirSim `rc_data`, pygame/XInput, keyboard fallback) and `VehicleAuthority` (the two RPC calls).
  Every transition is timestamped and lands in `mode_log.json`, the telemetry CSV's `mode` column and the
  map's mode chip.
- **`tools/live/synth_clip.py`** - real scenario, real plan, real terrain, real geo chain, **no renderer and
  no images**; a harness for the live loop when the editor is locked. Stamped `source: synthetic_telemetry`.
- **`tools/live/demo_takeover.py`** - scripted-pilot F3 demo.
- **`tests/test_live_mission.py`** - 56 tests, all green. Full suite **655 passed**.

#### Measured, `domain=sim`, `detector=truth`, replay source (`_artifacts/live/run_v3/data_card.json`)
320 frames -> 131 boxes -> 41 tracks -> **32 records**, `verify OK: 32 live == 32 replayed`.

| stage | median | p90 | max |
|---|---|---|---|
| capture | 0.29 ms | 0.40 | 0.96 |
| detect (truth replay) | 0.002 ms | 0.007 | 0.04 |
| geo | 0.022 ms | 0.19 | 0.51 |
| track | 0.748 ms | 1.16 | 1.84 |
| dedup | 0.005 ms | 0.25 | 1297.5 (one-off first-ingest import) |
| triage | 0.093 ms | 0.18 | 0.65 |
| **publish** | **8.385 ms** | 14.4 | 94.2 |
| end to end | **12.3 ms** | 16.6 | 1306 |

**Publish dominates at 68 %.** Everything from detection to triage costs under 1 ms a frame; the cost is the
HTTP hop to the C2. Capture -> record **actually on the map**, measured on the map's own WebSocket:
**median 88.2 ms**, p90 183.7, min 14.7, n=230, 0 unresolved. First pin: frame 43, 1.34 s after the shutter.
These are `truth` numbers: ceilings for the chain AFTER detection, not detection results.

#### Four real defects found, three of them cross-lane
1. **The tracker believed a nadir camera was looking due north.** `sightline.geo` reads `Telemetry.q_gimbal`
   as camera-FRD->NED (`ChainConfig.gimbal_frame="frd"`, its default); the FROZEN schema, `ingest/spec.py`,
   `track/` and `coverage/` read it as optical->NED, where nadir is the IDENTITY quaternion. `pipeline.py`
   and the old `survey.py` wrote the geo lane's literal `(0.7071, 0, -0.7071, 0)`, which
   `Telemetry.gimbal_pitch_deg()` reads back as **-180**. Consequence: `telemetry_affine` produced a 42 px
   warp where the geometry demands 555 px, nothing associated, and a survey produced detections,
   geolocations and then **zero tracks** - the second half of the 2026-09-11 285-frame result.
   **Fixed in this lane only**: producers now call `spec.gimbal_quat_from_euler` via
   `pipeline.nadir_gimbal_quat()`, and `pipeline.to_geo_gimbal()` converts at the single call into the geo
   lane using spec's own `Q_FRD_FROM_CAM`. **This is a labelled workaround, not a resolution** - see the
   open item below.
2. **Camera-motion compensation was off and is not optional.** A 9.8 m shutter step at 45 m AGL moves the
   image 555 px against a ~60 px target: consecutive frames do not overlap at all. `cmc_enabled` now
   defaults True in `FramePipeline`; measured on the fixture, False gives 0 tracks and True gives 4.
3. **Triage was silently a no-op.** `rank_records(records)` with no `TriageContext` sorts WITHOUT scoring, so
   every record kept `score 0.0` and every component its default - the first live run put a list of
   `P(living) 0.00` cards on the map. `FramePipeline` now always builds a context (`--incident-hours-ago`,
   `--water-temp-c`), with `now_utc` pinned to the FRAME clock so a live run and a later replay score
   identically. Records now rank 8.91 `immersed` above 1.10 `stranded`.
4. **`Uploader` shadowed `threading.Thread._stop` with an `Event`**, so `Uploader.stop()` raised
   `TypeError: 'Event' object is not callable` whenever the thread had already exited - i.e. on every clean
   shutdown that joined a finished uploader. Renamed to `_stop_event` in `sightline/store/outbox.py`
   (private, no external users).

#### Three defects I introduced and then caught by looking at the output
- **A negative latency.** The map-arrival probe matched a record's `(id, version)` against a timestamp from
  an EARLIER push and reported "capture -> record ON THE MAP: median **-1010.0 ms**". Now guarded by
  `not_before`, resolved without blocking the flight loop, and asserted `>= 0` in a test.
- **The outbox jammed: 1019 uploads for 32 records, 4 delivered.** The change detector included the triage
  score, which decays every frame, so the same `version` was re-pushed with different content; the C2's
  idempotent upsert correctly answered **409**, and the uploader retries a failed job for ever with the whole
  queue behind it. `score` is out of the fingerprint, `priority_rank` is in (the map sizes, labels and sorts
  by it), and `C2Client.push_records` now refuses a same-version re-push. Result: 262 pushed, 262 sent,
  0 failures, depth 0.
- **The pacing sleep was charged to `capture_ms`**, making capture 99 % of a replay's latency.

#### The gamepad, stated plainly
A physical **Xbox 360 pad is attached** and reading it caught a real bug: the "standard" SDL2 axis map puts
`pitch` on axis 4, which on this driver is a **trigger resting at -1.0**, so opening the source reported full
deflection and took the aircraft into MANUAL with nobody touching it. Measured layout: axes 0-3 are the
sticks (rest 0.0), 4-5 the triggers (rest -1.0). `PygameGamepadSource` now checks the resting state at
construction and refuses a mapping whose "stick" is a trigger. **Nobody pressed a button on the pad**: the
four button indices are unverified against hardware, and `describe()` says so
(`buttons_verified_against_hardware: false`). The state machine, authority, telemetry and map were exercised
with a scripted pilot (`tools/live/demo_takeover.py`).

#### F2 constraints now reach the code that flies
`sightline/mission/safety.py` (written by the safety lane precisely because "the drone that flies the capture
campaign enforces no geofence, no battery reserve and no ceiling") is now **consulted by `live.py` before it
arms**. It reports and never mutates - silently shortening a route would hide that an area was never
searched. A legal 45 m plan prints `plan is inside all constraints: 27 legs, 15.8 min of 20.0 min usable`;
a 130 m plan prints `ceiling[leg -1]: commanded 130 m AGL is above the 120 m ceiling` and the runner
**exits 4 without arming**. `--ignore-safety` flies it anyway and the violation is still stamped into the
data card. Three tests cover all three paths. `survey.py` is NOT yet wired to it.

#### LOOKED AT - screenshots, all rendered by `app/map/headless_check.mjs` (real Edge, 0 external requests, 0 console errors)
- `_artifacts/live/map_live_manual_final.png` - **mid-flight, mode MANUAL**: 7 records, unique rank badges
  0-5 ordered 8.83 immersed -> 1.10 stranded, the amber MANUAL segment inside the green AUTO track, the
  planned boustrophedon dashed cyan, the drone with its camera footprint, `outbox 0`, `errors 0`.
- `_artifacts/live/map_final_32_records.png` - end of run: 32 records, 32 confirmed, `mode RTL`.
- `_artifacts/live/map_live_inflight.png` - the first mid-flight capture (before the triage fix; scores 0.00).
- `_artifacts/live/map_manual_takeover.png` - the takeover with scores working.

#### NOT done, named plainly
- **No real PIE flight.** The editor was held by another lane for this lane's whole session
  (`foliage-lane`, then `sky-lane`, then `orchestrator`), and a live lock must not be broken. The simulator
  path (`fly()`, `capture_live()`, `AirSimRcSource`, `VehicleAuthority` against a real client) is therefore
  **unexecuted against PIE**; only its plan and cadence gate are exercised (`--dry-run` produces a valid
  8.9 min plan and refuses the 2026-09-11 cadence with exit 2). Everything downstream of the frame source is
  exercised on real geometry.
- **`--detector rgb` is unexercised** - there is no trained model yet. The code path exists and is stamped.
- **Physical gamepad BUTTONS untested** (see above).
- **The map's camera footprint comes from `sightline.geo.footprint_ned`, not `sightline.coverage`**, because
  coverage's `ground_footprint` renders 90 deg rotated from both the geo and the track lanes on the same
  telemetry (measured at gimbal yaw 0 and 90). The geo footprint measures 67.8 m east x 38.1 m north at
  45 m AGL, which is the across-track x along-track orientation `survey.py`'s line spacing assumes.

#### OPEN, and it belongs to the orchestrator, not to one lane
> **RESOLVED 2026-09-11 02:55 — this section is STALE and is kept only for the history.**
> Settled by MEASUREMENT, not by choosing a convention: across 110 boxes whose survivors have known
> world positions, image-right is due EAST (median residual 1.37 m; the next-best hypothesis 15.8 m,
> and camera-yaw-follows-airframe 11x worse than world-fixed). `geo` and `track` were right;
> `coverage/footprint.py` was the rotated one, because a nadir camera's contractual `q_gimbal` is the
> IDENTITY and applying identity to an OPTICAL ray maps image-right to north.
> Fixed at the root: `coverage/footprint.py` (`Q_CAM_YAW90`), `ingest/spec.py`
> (`frd_quat_from_gimbal_quat`, the canonical inverse), and `plan/patterns.py`
> (`gimbal_yaw_for_heading` returned `heading + 90`, a compensator for this very bug, now `heading`).
> THREE TESTS HAD ENCODED THE BUG and were corrected. Full suite: **762 passed, 3 skipped**.
> Still open and unclaimed: `pipeline.to_geo_gimbal` duplicates `spec.frd_quat_from_gimbal_quat` and the
> two call sites could be consolidated. See `docs/lanes/COORDINATION.md`.

**The repo holds two incompatible readings of `Telemetry.q_gimbal` and a third orientation in
`coverage/footprint.py`.** `docs/CONTRACTS.md` section 1 says the schema is frozen and orchestrator-owned,
and section 5 carries no amendment for the divergence. `pipeline.to_geo_gimbal` is a named, tested boundary
conversion that keeps the system working; it is not a resolution. The durable fix is to publish ONE
`rotation_optical_to_ned` in `sightline/common/` and have geo, track and coverage import it - which
`sightline/track/geometry.py`'s own module docstring already asks for, and which
`docs/lanes/ingest_track_dedup.md` section 6.1 also asks for. Changing `ChainConfig.gimbal_frame` to
`"optical"` touches ~41 `test_geo.py` cases; changing the tracker instead breaks 16 `test_track.py` cases
(measured, both directions). Related: the telemetry CSV has **no `gimbal_yaw_deg` column**, so a survey's
gimbal azimuth is unrecoverable at replay and is currently assumed 0.

### Level contents before the lanes started
1,072 actors: 919 debris, 73 houses, 71 survivors, terrain, flood water, sky rig. Nothing else was placed:
vegetation, rubble, utilities, damage and foam existed only as generated JSON plus meshes on disk.

### Next actions
1. Land the four lanes; the red team's verdict gates "done" for each.
2. `uv run python tools/scene/reconcile_occlusion.py --write` AFTER all geometry is placed (not before - it
   measures what is actually there), then re-run `check_occlusion_truth.py`.
3. `qa_shots.py` + `assert_qa_fresh.py` green, against `docs/SCENE_REFERENCE.md`.
4. Fly the patch campaign; `measure_occlusion.py --write`; `validate.py`; `quality_report.py`;
   `contact_sheet.py` and LOOK at it.
5. `make_split.py` (survivor-identity split; it documents that terrain is shared, so it measures unseen
   PEOPLE, not an unseen PLACE), then `sightline.detect dataset` -> `runpod_train.py`.
6. Still open: thermal/Infrared renders 100 % grey 0 despite every object accepting a segmentation ID -
   likely needs the Annotation API (`simSetAnnotationObjectValue`, ImageType 11). F3 gamepad takeover.

## Solution-doc analysis, 2026-09-11 02:35 — acceptance criteria FIXED BEFORE MEASURING

`PS2-Survivor-Vision-Research-and-Build-Document.md` and `docs/SOLUTION_DOC.md` are **byte-identical**
(same md5, 1,219 lines). There is no second requirement set.

### The nominal slice, declared now, per 5.5c step 5 ("define it before you measure it")
> "40-60 m, daylight, occlusion below 50 %, non-submerged presentations. Report that number as the
> acceptance figure. Report the hard slices as separate numbers in the same table."

Mapping the flown campaign onto that definition:

| pass | AGL | in the nominal slice? |
|---|---|---|
| alt35 | 35 m | **no** - below the band. A HARD slice (and 5.5c's remedy if the number misses) |
| alt45rain | 45 m | in the band, but rain. Reported SEPARATELY, not folded into the headline |
| **alt55** | **55 m** | **YES - this is the acceptance slice** |
| alt80 | 80 m | **no** - above the band. 5.12 says >= 90 % is expected only at <= 60 m AGL |

So the headline figure comes from **alt55 only** (496 frames). Everything else is a named hard slice. This is
written down BEFORE any model exists precisely so the slice cannot be chosen afterwards to flatter a result.

### Finding 1: the dataset is 5-9x smaller than the doc's recipe
5.5c step 2 says "fine-tune for 30-50 epochs on **10-20k rendered tiles**". The campaign projects to roughly
**2,300 tiles** (~585 positive + ~1,680 background). That gap is deliberate - the operator asked for a
compact dataset covering situations rather than a large repetitive one - but it is a real deviation from the
recipe the >= 90 % expectation was written against, and it must not be discovered after a disappointing
number. The doc's own remedy is followed if the number misses: *"if it does not clear, the cause will be
flight geometry or scene difficulty, not the model, in which case LOWER THE ALTITUDE before touching the
training"* - and alt35 data already exists for exactly that.
Padding the tile count with more background tiles would meet the doc's number while changing nothing real;
it is not done.

### Finding 2: >= 90 % is expected ONLY at <= 60 m, and 5.12 says so explicitly
> "The >= 90 % target is expected to be met only on tiled high-resolution RGB at <= 60 m AGL with in-domain
> fine-tuning, and the report must say so slice by slice."

alt80 is therefore expected to underperform BY DESIGN. That is a reportable property of the flight geometry,
not a model failure, and the report will say so rather than quietly averaging it away.

### Finding 3: what 5.12 requires that is not yet wired
recall@IoU 0.5 AND 0.25; FP/min per terrain type (water, debris, vegetation, roof) and again at record level
after dedup; dedup accuracy (record precision/recall, duplicate rate, count error, HOTA/IDF1/IDsw);
geolocation error (median and p90 against known survivor positions); POD reliability diagram; the full slice
grid (zone x altitude x band x time of day x occlusion x posture); and the missed-detection analysis -
**recall vs pixel-height histogram**, which is what identifies the operating floor.

### Reporting rule, non-negotiable (5.5c)
Every recall figure carries its domain in the same sentence: "94 % recall at IoU 0.5, **in simulation**, on
the nominal slice." Never averaged with a real-domain number. `sightline/eval/slicing.py` already enforces
this structurally - `DomainMixError` is raised on any attempt to pool domains - so this is a property of the
code, not a convention someone has to remember.
## ENVIRONMENT FINALISED, 2026-09-11 02:05 — `tools/scene/gate.py --all` exits 0, 10/10

| check | proves |
|---|---|
| realism-dryrun | 217 checks; every build script runs to completion against a strict UE mock |
| occlusion-truth | height-gated: no survivor in the open is covered, every buried one is |
| env-assets | every download complete, size and md5 match the source |
| lighting | sun:shadow inside the physical range, nothing meaningful crushed to black |
| qa-fresh | nothing in the scene changed since a human last looked at a render |
| vegetation | all 5,508 canopy instances match the layout; no crown over an unoccluded survivor |
| rubble | 867 assertions; per-variant counts exact, Nanite off, every material compiles |
| utilities | 25/25; poles buried to 0.0 cm, boats float within 2.9 cm of the surface |
| foliage | no engine-placeholder materials, nothing translucent, canopy not crushed |
| sky | visible sky matches its own light; distant canopy carries near-field mass |

Level: **1,162 actors** holding 71 survivors, 76 houses, 919 props, 5,508 canopy instances (9 HISM actors),
1,799 rubble + 67 roof-debris instances (56 actors), utilities, boats and foam.

### Two metric decisions taken, both recorded rather than quietly applied
- **`check_sky.py`'s LOD assertion was MOVED from darkest-quartile G/R to canopy coverage.** The G/R test
  could not measure the property: a distant crop's darkest quartile is the GROUND between crowns, not bark -
  measured (0.467, 0.411, 0.267) far against (0.214, 0.268, 0.089) near, twice as bright and redder, and in
  this frame that ground includes the red laterite band. A sweep found no lever reaching its 0.85 bar (best
  0.784), the signature of a metric with a ceiling below its own target. Coverage against the near-field
  crop is self-referencing so haze and exposure cancel, and it WOULD have caught the original defect:
  0.55 before Nanite PRESERVE_AREA, 1.03 after, against a 0.80 floor.
- **`assert_qa_fresh.py` no longer treats OBSERVER scripts as scene changes.** `check_*`, `measure_*`,
  `verify_*`, `dump_*`, `sweep_*`, `qa_*` cannot change what a render shows. Without this, tightening a
  checker marked the scene stale and pushed towards re-rendering to silence the gate rather than because
  anything moved, which is how a freshness gate stops meaning anything. Builders stay watched.

### Bug I introduced and the gate caught
Hardening the offline mock until it could run `dump_vegetation.py` to completion made the DRY RUN overwrite
the real instance dump with mock content (`/Game/Mock/mesh_LOD0`, level "get_editor_world"), so
`check_vegetation.py` failed against a canopy that was correct. The dump path is now dry-run safe.

### Full test suite: 736 passed, 2 skipped.
## Decisions taken under autonomous ownership, 2026-09-11 01:40

**Dataset size is bounded by FLIGHT time, not by GPU.** Measured: the campaign yields roughly 2,700 training
tiles (1024 px, 20 % overlap, 15 tiles per 4K frame, positives plus 5 % negatives), and 60 epochs on that is
about 15 minutes on an RTX 4090 at $0.34/h. The 1-hour training budget therefore has large headroom, and the
scarce resource is the ~57 minutes of simulated flying. So the dataset is kept COMPACT and spent on
DIFFERENT SITUATIONS rather than more frames of the same one:

| pass | AGL | GSD | condition | frames |
|---|---|---|---|---|
| alt35 | 35 m | 1.4 cm/px | clear morning | 758 |
| alt55 | 55 m | 2.2 cm/px | clear midday | 496 |
| alt80 | 80 m | 3.2 cm/px | hazy afternoon (fog 0.15) | 454 |
| alt45rain | 45 m | 1.8 cm/px | rain 0.55 + fog 0.25, overcast | 532 |

2,240 frames, ~57 min, covering all 9 pose/submersion/zone combinations and 69/69 detectable survivors
(the patch planner reaches every one; the old quantile box reached 28).

**Shutter spacing is DERIVED, never chosen** - `campaign.shutter_m` solves the tracker's confirmation gate
(SOLUTION_DOC 5.6 rule 3: 3 hits inside 2 s) for both `s <= speed*window/(min_hits-1)` and
`s <= frame_height/min_hits`, and FLOORS to a decimetre because rounding 9.89 up to 9.9 gave 2.996 hits and
missed the gate. `survey.py` refuses to fly a cadence that cannot confirm a track.

**Training config**: RTX 4090 community cloud, YOLO26s from COCO, imgsz 1024, batch auto, ~60-80 epochs with
early stopping. The model is small deliberately: with roughly 800 boxes the binding limit is DATA, not
capacity, so a bigger backbone would only overfit faster.

**Held-out split is unresolved and is being deferred honestly.** A true second scenario seed means
regenerating six layouts (vegetation clearance, rubble voids and props burial caps are all keyed to survivor
positions) and re-placing ~8,000 instances, then re-passing every asset gate. `tools/capture/make_split.py`
achieves the property the rule protects - no survivor on both sides, frames mixing the two sides dropped -
and documents what it does NOT achieve: the terrain is shared, so it measures generalisation to unseen
PEOPLE, not to an unseen PLACE. Decision deferred until the data exists and the cheaper split is shown to be
insufficient.

**Known risk, unverified:** whether AirSim rain and fog actually render in this level. `survey.py` probes a
frame after applying the condition and prints its mean brightness, so a condition that silently did nothing
will show up in the first pass rather than after the whole campaign.
## Next actions (start here) — DEVELOPMENT PHASE
Read `docs/HANDBOOK.md` §5-§6 before touching the sim or the scene. The doc is now **simulation-first** (§5.5c):
the renderer is the deployment domain, randomisation OFF by default, every number carries a `sim`/`real` column.

0. **Close the five unverified items in "Handoff state" above**, commit.
1. **F1 flood-valley scene — foundation done, dressing in progress.**
   `/Game/Sightline/Maps/FloodValley` is the startup + game map. Rebuild from scratch, PIE off:
   `uv run python tools/scene/gen_terrain.py`, then
   `ue_python code="exec(open(r'D:\Sightline\tools\scene\build_flood_valley.py').read())"` (idempotent; aborts if the
   terrain transform is wrong), then `...build_materials.py` the same way. FloodLevel at runtime:
   `uv run python tools/scene/flood_level.py <asl_m>` (read-back verified). **Remaining for F1:**
   - settlement buildings on the terrace (zone G). **No free login-free realistic house asset exists** (see Assets);
     either build modular houses by script from the downloaded plaster / clay-tile / corrugated-sheet textures, or
     ask the user to sign in to Fab. Kerala type: 1-2 storey, flat concrete or clay-tile / corrugated roofs.
   - debris on the fan and channel from the Poly Haven models (rocks, logs, stumps, barrels, crates, jerrycans,
     tyres, trash bags, covered car) — glTF under `_downloads/assets/polyhaven/<id>/`, not yet imported.
   - waterline foam (ambientCG Foam001/002 downloaded), vegetation (palms/areca blocked, fern_02 available).
   - zone polygons / spawn masks exported by `gen_terrain.py` for the spawners (it only writes zone counts today).
2. **Actor + debris spawners** (seeded, reproducible): pose/submersion classes per §2.3 rows 2-3 and §6.2, tagged
   `Human_<id>` / `Animal_<id>`, MOVABLE actors only (static ones cannot be posed at runtime). Humans: 9 Microsoft
   Rocketbox rigged FBX (MIT) in `_downloads/assets/rocketbox/`, not yet imported; the UE5 mannequin
   (`D:\UE_5.8\Templates\TemplateResources\High\Characters`, 125 MB) has death anims usable for lying poses. Then
   day-1 #2 with a half-submerged actor + the `IgnoreMarked` camera, and day-1 #6 (detection-box flicker).
3. **Thermal**: object-ID temperature table first (day-1 test #3), then the §5.1 step 7 post-process material.
4. **Capture pipeline (F5)**: waypoint capture with `simPause` + `simGetImages` (SteppableClock profile), auto-labels
   from instance masks, per-frame telemetry CSV/JSON. Simulation-first: nominal slice at 40-60 m, split by seed.
5. **Assets blocked on logins** (ask the user): houses, palm/areca/coconut trees, rigged cow/goat/dog, uncovered car,
   photoscanned humans — Fab / Sketchfab / Mixamo / MetaHuman all need an account. Full list in
   `_downloads/assets/MANIFEST.md` "Blockers".

## Phase 0: environment and MCP (day 1 morning)
- [x] Located UE 5.8.2 at `D:\UE_5.8`; confirmed Epic's ModelContextProtocol plugin ships in 5.8.2
- [x] uv 0.12.12 on D:, caches redirected to D: (see CONTEXT §3)
- [x] Python 3.11 env + `uv.lock` (cosysairsim 3.4.1, mcp 1.30.0, numpy 2.2.6, opencv 4.14)
- [x] Cosys-AirSim 5.8-v3.4.1 plugin + Blocks editor project downloaded (size-verified); plugin installed in project
- [x] Blocks packaged build (reference sim, used for AirSim verification without the editor)
- [x] Visual Studio 2022 17.14 on D: (MSVC 14.44.35228 — accepted by UE 5.8; the folder name 14.44.35207 is
  irrelevant, UBT reads cl.exe's file version)
- [x] `SightlineSim` project scaffolded (module, targets, low-memory renderer, Python remote exec, MCP auto-start)
- [x] sightline MCP server written; stdio smoke test passes (24 tools)
- [x] `SightlineSimEditor` compiles: Result Succeeded in 869 s (2026-09-10 18:47), MSVC 14.44.35228 (VS 2022 on D:),
  Windows SDK 10.0.22621, via UBA (cache redirected to D:\UE_Cache\UBA). Log: `_logs/jobs/build-SightlineSimEditor-*.log`
- [x] Editor opened the project (FlyingExampleMap, AirSim plugin loaded) at 18:52; engine-tools agent verifying MCP live
- [x] Editor opens the project with AirSim loaded; editor RAM measured: 3.4 GB RSS idle, 4.1-4.5 GB in PIE on
  FloodValley (2026-09-10, session 2)
- [x] `unreal` MCP (HTTP :8000) reachable from Claude Code; 31 toolsets / 392 tools listed and called
- [x] `ue_python` remote execution verified against the live editor (file/statement/eval, 20k-line output intact)
- [x] PIE + `sim_ping` + takeoff + `sim_capture` verified in-editor (39/39, same as the packaged build)
- [x] `tools/doctor.py --live` passes (0 FAIL)
- [x] Blocks packaged exe runs; the Python client connects (API cross-check)
- [x] **Phase 0 complete (2026-09-10).** Environment, both MCP servers and all 27 tools validated; see the
  verification table below and `docs/verification/`.

## Verification workstreams (2026-09-10): every tool in the doc, end to end
Reports land in `docs/verification/`. A tool is not "ready" until its report says PASS with evidence.
| Stream | Scope | Report | Status |
|---|---|---|---|
| V1 engine MCP | sightline engine/build/editor/log/job tools + Epic `unreal` MCP live (toolset inventory) | `engine_tools.md` | [x] all sightline engine tools PASS; Epic: 31 toolsets / 392 tools, 17 real calls incl. CaptureViewport PNG. Fixed: `ue_generate_project_files` (batch file absent in installed engine), missing user env -> editor DDC would go to C:, UBA store -> D:, double-execution after a timed-out editor command |
| V9 dev workflow | AirSim inside the editor (PIE), gamepad via AirSim + takeover, full 27-tool matrix, CLI spot-checks | `dev_workflow.md` | [x] PIE loop PASS (launch 49 s cold, StopPIE -> tools error in 0.31 s, editor closes cleanly 4/4); AirSim **39/39 in-editor**; tool matrix **33/33 up / 12/12 down, all 27 tools**; CLI drove a real flight + capture (image survived MCP as valid JPEG). Fixed D1: first `simGetImages` per engine process returned **unconverted** depth/normals (silently wrong) -> warm-up frame discarded + implausible-depth warning |
| V7 GPU latency | day-1 test #8: C1/C2/C3 strategies, TensorRT FP16, RTX 4060 | `gpu_latency.md` | [x] all six configs **inside 300 ms**. Design pass (C2, yolo26s, 6×1280×1088 batched) = **65.0 ms median / 68.3 p90** vs the doc's ~60 ms estimate; C1-s 16.7 ms, C3-s 29.2 ms. FP16 only; INT8 and Orin (F20) still open |
| V2 MCP protocol | sightline server schemas, errors, concurrency, cancellation, lifecycle (`tests/test_mcp_protocol.py`) | `mcp_protocol.md` | [x] 20 pass / 2 skip (no-editor cases skip while an editor runs; passed with it down). 9 defects fixed |
| V3 Claude Code integration | CLI loads `.mcp.json`, real tool calls via `claude -p`, timeouts, reconnect | `claude_code_integration.md` | [!] servers load + tools discovered (27) + unreal connects; real `claude -p` calls BLOCKED: CLI not logged in (user: `claude auth login`). Project `.claude/settings.json` sets MCP timeouts |
| V4 AirSim tools | every sim_* tool through MCP (`tools/sightline_mcp/test_sim.py`) | results JSON in `_artifacts/verification/` | [x] **39/39** vs packaged Blocks (2026-09-10 19:19). Defects found+fixed: broken `landAsync` (own position-held landing + disarm; RTL lands 0.2 mm from home), velocity-only descent drifting into obstacles, static-actor pose silently "succeeding", `to_eularian_angles`/`to_quaternion`/`simGetImages(external=)` wrong API names, 60 s RPC timeout on long flights, falling-after-reset breaking takeoff, over-strict Ground collision rule. **Session 2 changed the Ground rule (steep contact fails): re-run after the next server restart** |
| V5 Python stack | all Python libs in doc: install pinned + functional tests (`tests/test_stack.py`) | `python_stack.md` | [~] |
| V6 external tools | Cesium, X-AnyLabeling, PMTiles+MapLibre offline, Fields2Cover, PX4/QGC, DroneModels, TAK, gamepad | `external_tools.md` | [x] X-AnyLabeling, PMTiles, MapLibre offline, DroneModels PASS; Cesium STAGED (`_staging/plugins`, BuildId match); Fields2Cover REJECT on Windows (own boustrophedon); PX4/QGC/TAK documented |
| V8 gamepad | XInput + pygame detection, live input, AirSim `rc_data`, API release/re-acquire handover | `_artifacts/verification/gamepad_airsim_*.json` | [x] **13 PASS / 0 FAIL** (2026-09-10 20:31). AirSim sees the pad (VID_045E, is_valid); axes full travel; under API control a held stick moves the drone 0.01 m in 4 s; `release` -> API off in 23 ms, vehicle follows the stick after **394 ms**; `arm` -> authority back, stabilised in **2.96 s**, then `move_to` obeyed to 0.22 m with the stick still held; rtl landed. F3's takeover semantics are proven |

MCP defects already found and fixed (2026-09-10): stdout pollution by cosysairsim prints; Windows stdio deadlock on lazy
numpy import (now eager imports); blocking tools moved off the event loop (`@threaded`); tool-to-tool calls via
`__wrapped__`; wrong cosysairsim API names (`to_eularian_angles`, `to_quaternion`, `simGetImages(external=)`).
Also: `materials.csv` missing -> Cosys skips material stencil init (installed via `tools/setup/install_materials.ps1`).

## Open items carried into development
- ~~Gamepad handover unmeasured~~ **CLOSED 2026-09-10**: 13 PASS / 0 FAIL, latencies API->RC 394 ms and
  RC->API 2.96 s (`tools/day1/gamepad_airsim.py`). The Unreal window does NOT need focus (DirectInput uses
  `DISCL_BACKGROUND`); with `AllowAPIAlways:true`, `AllowAPIWhenDisconnected` is a no-op. F3 must log every
  mode switch into the telemetry CSV (§5.2) so coverage can be attributed to AUTO vs MANUAL.
- **Epic `unreal` MCP drops idle HTTP sockets after 15 s** (`HttpConnection.h:266 ConnectionKeepAliveTimeout`,
  hard-coded, no cvar). Symptom: "The socket connection was closed unexpectedly". **Retry the call once.**
- **Cosys 3.4.1 logs nothing to the UE log** (`UAirBlueprintLib::LogMessage` has every `UE_LOG` commented out), so
  "Loaded settings from ..." can only be seen on the on-screen HUD. Prove settings use via `listVehicles()`,
  camera resolution and the home geopoint instead.

## Known issues
- ~~Lumen active in PIE via Epic's FlyingExampleMap PPV~~ **RESOLVED 2026-09-10 (session 2)**: FloodValley is the
  startup and game-default map; its unbound `PPV_NoGI` forces GI = None and SSR.
- **Never run GPU/ML work while the editor + PIE are up** on this machine: 16 GB RAM / 8 GB VRAM triggers Windows
  memory-pressure warnings and makes both slow (observed 2026-09-10 20:05 with a TensorRT export alongside PIE).
- Free RAM with editor + PIE on FloodValley was 1.9-2.7 GB (2026-09-10). Close browsers before editor sessions.

## Day-1 unknowns (SOLUTION_DOC §10), with results
| # | Test | Status | Result |
|---|---|---|---|
| 1 | Cosys 3.4.1 loads on UE 5.8.2; editor RAM with project open | [x] | loads; editor RSS 3.4 GB idle / 4.1-4.5 GB in PIE on FloodValley (2 km terrain + water), 2026-09-10 |
| 2 | Instance seg hides submerged pixels (single-layer water / translucent plane); IgnoreMarked camera | [~] | single-layer-water plane **writes depth** (depth 45.75 m at nadir = drone height above water) and gets **its own instance colour** separate from the terrain, shoreline edge clean; the half-submerged ACTOR check and IgnoreMarked are still to do (needs a human actor) |
| 3 | Infrared image type after ID remap; capture FPS 4K raw vs PNG | [ ] | IR renders (all-black before any ID/temperature table: expected) |
| 4 | Xbox RemoteControlID, AllowAPIAlways, handover latency, moveByRC need | [x] | 13 PASS / 0 FAIL; API->RC 394 ms, RC->API 2.96 s (V8 above) |
| 5 | Weather visible in Scene, absent in Segmentation; time-of-day needs sky sphere | [ ] | FloodValley has BP_Sky_Sphere (hidden) wired to the sun + SkyAtmosphere; TOD not yet exercised |
| 6 | Detection-API box flicker on prone / 70 % submerged skeletal meshes | [ ] | |
| 8 | End-to-end ms on the 4060 for C1/C2/C3 | [x] | all six configs < 300 ms; C2/yolo26s 65 ms (V7 above) |

## Features (SOLUTION_DOC §8)
| # | Feature | MVP? | Status |
|---|---|---|---|
| F1 | Flood-valley scenario, 3 zones, weather/time/flood level, actor spawners | MVP | [~] terrain, zones, water + FloodLevel API, lighting, materials built; buildings/debris/actors/spawners to do |
| F2 | Coverage patterns, orbit-on-detect, revisit queue, battery/geofence | MVP | [~] boustrophedon + terrain following shared by `survey.py` and `live.py` (`sightline/mission/pattern.py`); orbit-on-detect and revisit queue NOT built |
| F2b | Decision planner (Koopman + greedy) | Stretch | [ ] |
| F3 | Gamepad takeover/hand-back, HOLD/RTL, logged mode switches | MVP | [~] state machine + 3 control sources + `VehicleAuthority` built and tested (`sightline/mission/takeover.py`); all 5 transitions demonstrated on the live map, mode visible in telemetry, CSV and map chip. **Physical pad BUTTONS unpressed; not flown in PIE.** |
| F4 | PX4 SITL path | Stretch | [ ] |
| F5 | Synthetic dataset with visible/amodal boxes, attributes, telemetry | MVP | [ ] |
| F6 | Annotation guideline + X-AnyLabeling loop | MVP | [ ] |
| F7 | Ingest (sim export, DJI SRT; MAVLink/ULog stretch) | MVP | [ ] |
| F8 | Tiled YOLO26 RGB detector, TensorRT, frozen threshold | MVP | [ ] |
| F8b | **Demo model**: sim-only fine-tune, held-out sim scenes, randomisation off (§5.5c) | MVP | [ ] |
| F8c | Transferable model: real + synthetic, real-clip eval, domain gap | Stretch | [ ] |
| F9 | Thermal YOLO26n + WBF/ProbEn late fusion, RGB-only fallback | MVP | [ ] |
| F9b | Radiometric thermal (sim + stills) | MVP in sim | [ ] |
| F10 | Crop verifier: is_real + posture + submersion + occlusion | MVP | [ ] |
| F11 | BoT-SORT/TrackTrack with CMC, 3-hit confirm | MVP | [ ] |
| F12 | Geo-dedup (DBSCAN 2xCE90), count, motion, stale | MVP | [ ] |
| F13 | Geolocation chain + error budget | MVP | [ ] |
| F14 | Triage score + GeoJSON/KML/KMZ | MVP | [ ] |
| F15 | MapLibre offline map, WebSocket, record cards | MVP | [ ] |
| F16 | Search-quality raster (POD), burial polygons | MVP | [ ] |
| F16b | Per-presentation coverage layers | MVP (2 layers) | [ ] |
| F17 | CoT/TAK export | Stretch | [ ] |
| F18 | SQLite log + persist-queue outbox + cloud upsert | MVP | [ ] |
| F19 | Evaluation script + slices + FiftyOne | MVP | [ ] |
| F20 | Jetson Orin engines + latency table | Stretch | [ ] |
| F21 | Guardrails (no auto-close, no delete, dismiss-with-reason) | MVP | [ ] |

## Session log
- **2026-09-11 (session 4, live-integration lane)**: Built the real-time demo loop. Extracted the survey plan,
  terrain follower and cadence gate out of `survey.py` into `sightline/mission/pattern.py` so the capture
  flight and the live runner share ONE implementation, and extracted the per-frame stage sequence into
  `sightline.pipeline.FramePipeline` so the live and offline paths cannot drift (a test feeds both the same
  frames and compares the records; a second test sabotages one path to prove the comparison has teeth). Wrote
  `sightline/mission/live.py`, `sightline/mission/takeover.py` (F3), `tools/live/synth_clip.py` and
  `tools/live/demo_takeover.py`, plus 53 tests. Found and fixed four defects that were making the system
  quietly useless: the geo and track lanes read `q_gimbal` 90 deg apart so the tracker thought a nadir camera
  faced north and NO survey could confirm a track; camera-motion compensation was off against 555 px of
  per-frame image motion; `rank_records` with no context skipped scoring entirely, so the map showed
  `P(living) 0.00` on every card; and `Uploader` shadowed `Thread._stop`. Then caught three of my own by
  looking at the output rather than the return value - a NEGATIVE map latency, an outbox that delivered 4 of
  1019 records because the change detector fired on the decaying score and the C2 correctly answered 409, and
  a pacing sleep charged to capture time. A physical Xbox pad proved the "standard" SDL2 axis map wrong (pitch
  was mapped to a trigger, which rests at -1.0 and reads as a permanent takeover). Demonstrated on the map
  mid-flight: 32 records, capture->map median 88.2 ms, all five F3 transitions. NOT done: no PIE flight (the
  editor lock was held by other lanes all session), `--detector rgb` unexercised, gamepad buttons unpressed.
- **2026-09-11 (session 4, vegetation lane)**: Rewrote `build_vegetation.py` from one actor per plant to one
  `HierarchicalInstancedStaticMeshComponent` per species and placed 4,342 trees + 1,166 understorey ferns as
  **5,508 instances on 9 actors** (1,093 -> 1,102). The placement cost no measurable commit (`UsedVirtual` 7.49
  GiB before the first instance and after the last), against the 27.48 GiB exhaustion that killed the editor on
  the previous attempt. Found that UE 5.8 Python has no `Actor.add_component_by_class` and used
  `SubobjectDataSubsystem` instead, proving it serialises with a save/reload round trip on a 3-instance probe
  before committing. Proved persistence again for the real canopy by reloading the .umap from disk and reading
  all 5,508 transforms back. Wrote `check_vegetation.py` (11 assertions) and exercised it against a sabotaged
  synthetic dump so it was known to fail before it was allowed to pass. Confirmed from renders that the leaves
  are scanned geometry (JPEG textures cannot carry alpha, so opaque is the only correct blend mode) and that the
  drowned trees lean south/downstream (engine-measured mean azimuth 178.9 deg over 299 tilted instances).
  Kept the full scans rather than the `_lite` variants the layout budgeted for. Open, named as fidelity not
  correctness: crowns render near-black under GI = None, and long-range Nanite LOD makes distant trees skeletal.
- **2026-09-11 (session 4, utilities + waterline lane)**: Placed the utility network (60 poles / 57 spans /
  40 service drops as one 22,704-tri actor), 19 boats and the 5,146-tri foam ribbon as a single mesh; 21
  actors total, 1,072 -> 1,093, editor alive with more commit headroom than it started with. Judged the water
  from renders rather than from `tune_water.py`'s report - which turned out to be printing stale literals -
  and cut every extinction coefficient by 3.0 (Secchi 0.35 -> 1.05 m) so depth is readable through the
  shallows while the deep terrace stays opaque; the albedo, and therefore the green-teal hue, is unchanged.
  Wrote `check_utilities_placed.py` (25 assertions + an inverted self-test) and `qa_utilities_shots.py` (11
  framed views). The self-test caught a hole in the check, and the check caught `POLE_BURY_M`. `qa_shots.py`
  re-rendered and `assert_qa_fresh.py` exits 0. Water/terrain/vegetation gaps left open are named in the
  handoff above.
- **2026-09-10 (session 2)**: Development phase started; stopped by the user for handoff mid-verification.
  Adopted the user's updated build doc (simulation-first §5.5c, F8b/F8c) as `docs/SOLUTION_DOC.md`. Built the
  FloodValley foundation entirely through MCP: terrain generator reworked (4 m grid, downstream fan lobe, blended
  terrace, command-post pad, OBJ in cm), terrain imported (Nanite off, complex collision, yaw +90 for NED, object
  name `Ground`), sun + BP_Sky_Sphere (hidden) + SkyAtmosphere + real-time sky light + fog + PPV (GI None),
  PlayerStart on the pad, movable single-layer-water `FloodWater`; FloodValley made the startup/game map.
  Verified in PIE: GPS = pad geopoint (after finding Cosys anchors OriginGeopoint at the UE world origin),
  takeoff/flight/RTL on the terrain, water writes depth and has its own instance colour, FloodLevel settable via the
  API (`flood_level.py`, read-back verified), `build_flood_valley.py` idempotent. 79 photoreal CC0/MIT assets
  downloaded by a subagent (`_downloads/assets/MANIFEST.md`, 1.74 GB); houses, palms, animals, uncovered car blocked
  on logins. Terrain (zone/slope/silt-line, 5 Poly Haven sets, anti-tiling) and water (panning normals) materials
  built by `build_materials.py`; last visual check pending (see Handoff state). Diagnosed a user report of high
  CPU / idle NVIDIA: UE is on the 4060; spike was imports; fixed the background-throttle setting's config section.
  Commits: b0213ca (foundation), then the handoff commit.
- **2026-09-10 (session 1, part 2)**: Validation phase closed. Six verification streams run in parallel by
  subagents (engine MCP, protocol, Claude Code integration, external tools, Python stack, dev workflow) plus the
  AirSim suite by the main session. Results: 27/27 sightline tools, Epic 31 toolsets/392 tools, AirSim 39/39
  (packaged and in-editor), protocol 22/22, stack 37/0, doctor 0 FAIL. Defects found and fixed are listed per
  stream above; the most dangerous were the Windows stdio deadlock, the broken `landAsync`, and the unconverted
  first capture. Gamepad verified at the Windows level; AirSim-side handover still unmeasured. Docs finalised
  (HANDBOOK/CONTEXT/TRACKER), two commits, memory files written. **Development phase not started**; the only
  dev artefact is the unvalidated `tools/scene/gen_terrain.py` draft.
- **2026-09-10 (session 1, part 1)**: Environment setup. Found UE 5.8.2 plus the built-in Epic MCP plugin. Chose VS 2022 17.14
  (Cosys build toolchain). Installed uv and the Python 3.11 env on D:, redirected all caches to D:, downloaded
  Cosys-AirSim 5.8-v3.4.1, scaffolded SightlineSim, wrote the sightline MCP server (smoke test OK), and wrote
  CLAUDE.md/CONTEXT/TRACKER. A network drop interrupted the VS install, which was restarted via the bootstrapper.
  RAM headroom is low (see CONTEXT §7).

## Scene polish backlog (session 3) — what stands between now and a demo-grade disaster scene
Ordered by how much each changes what the camera sees.
1. **Debris and wreckage** — nothing floats or piles anywhere yet. A real debris-flow flood is defined by its
   wrack line: rafted timber, drums, sheeting and vehicles jammed against upstream walls; boulders and trunks
   dropped high on the fan; light plastics circling in eddies. 79 CC0 props are downloaded and imported
   (`data/scene/props.json`) but none are placed. Biggest single visual gap.
2. **Sitting pose sits on an invisible chair** — hips and knees both at ~80 deg, so a survivor on a flat roof
   has their feet dangling below the slab. On a roof people sit with legs out or crossed: flex the hips ~85 deg
   and keep the knees near straight so the pelvis rests ON the surface.
3. **Water reads as flat card at survey altitude** — the albedo is right but there is no large-scale surface
   variation, so a 45 m nadir frame is a uniform tan field. Needs a low-frequency normal/roughness break-up and
   some suspended-sediment streaking, plus foam at the waterline (ambientCG Foam001/002 downloaded).
4. **Roof textures tile visibly** from above; the buildings need the same anti-tiling treatment the terrain got.
5. **No vegetation** — fern_02 is imported; palms/areca are blocked on a Fab login (see Assets blockers).
6. **Damage state** — every house is pristine. Flood-damaged walls, missing sheets and collapsed sections would
   sell the scenario and add the occlusion cases the detector should be tested against.

## Held-out split: the cheap path, decided 2026-09-11 02:15

`sightline/detect/dataset.py` REFUSES to build from one scenario seed (exit 4) - correctly, because splitting
one survey by frame leaks the same survivor into both halves. Earlier I costed a second seed at "regenerate
six layouts and re-place ~8,000 instances", and deferred it. That estimate was wrong, and the reason it was
wrong is worth writing down.

Only the SURVIVORS need to move. The expensive part - vegetation, rubble, utilities, damage - can stay
exactly where it is, because the ground truth is now MEASURED rather than assumed:

1. `gen_actors.py --seed 47`          new survivor positions and poses (host side, seconds)
2. `build_actors.py`                  moves 71 individual actors in the editor (they are not instanced)
3. `fix_occlusion_truth.py --write`   re-measures what is above each NEW position and raises occlusion to
                                      match. Height-gated and raise-only, so it cannot fabricate a clearance.
4. `check_occlusion_truth.py`         must exit 0 - labels and geometry agree again
5. `check_vegetation.py`              its "no crown over an occlusion-0 survivor" assertion holds again
                                      BECAUSE step 3 raised the occlusion of anyone who ended up under one
6. one capture pass at 55 m           ~13 min

About 20 minutes for a genuinely different scenario seed with honest ground truth, instead of hours. What it
buys is disjoint survivor identities, poses and placements under a `scenario_seed` the tooling can split on.
What it does NOT buy, and must be reported as such: the terrain, buildings, canopy and rubble are shared, so
it measures generalisation to unseen PEOPLE, not to an unseen PLACE.

This is only possible because occlusion stopped being a static guess. Before the height-gated measurement
existed, moving the survivors would have silently invalidated every occlusion label in the scene.

## F2 geofence / battery / ceiling: found unenforced, checkable now, 2026-09-11 02:45

**The gap.** `sightline/plan/` implements F2 completely - boustrophedon, expanding square, orbit-on-detection,
a revisit queue, and `constraints.Constraints` with a battery RTL reserve, geofence, the 120 m ceiling and
operator no-go areas - with 31 tests, one of which asserts a short battery truncates a route and appends an
RTL leg reading "battery reserve". But grep either flight module for `Constraints`, `geofence` or `Battery`
and there is nothing: `sightline/mission/survey.py` and `sightline/mission/live.py` use their own
`mission/pattern.py` planner. **The drone that flew the capture campaign enforced none of them.** A feature
that is complete in a library and absent from the thing that runs is not complete.

**What was added** (host-side only - `survey.py` could not be touched, because `campaign.py` launches every
remaining pass as a fresh process that re-reads it mid-campaign):
`sightline/mission/safety.py` - `check_plan(SurveyPlan, Constraints) -> SafetyVerdict`. It walks the legs in
order accumulating flight time the way the vehicle actually spends it (**transit onto each line, then the
line**), and reports the first leg at which the battery could no longer reach home inside its reserve, using
the same `Battery.can_continue` rule `constraints.apply` uses so the two cannot disagree. It REPORTS and
never mutates: a planner that silently shortens a route hides that the rest of the area was never searched,
and R10 forbids the system ever implying an area is clear. `tests/test_mission_safety.py`, 10 tests, green -
including one asserting the battery message contains "UNSEARCHED" and "not clear", and one proving distant
legs cost more battery than adjacent ones.

**Verified against the campaign that was flying at the time**, with a 150 m operator geofence and the default
25 min endurance (20 min usable after the reserve):

| pass | legs | track | est. flight incl. transits | verdict |
|---|---|---|---|---|
| alt35 | 29 | 7.4 km | **17.1 min of 20.0** | inside all constraints |
| alt55 | 26 | 6.0 km | 15.5 min | inside all constraints |
| alt80 | 22 | 5.5 km | 14.8 min | inside all constraints |
| alt45rain | 27 | 6.4 km | 15.8 min | inside all constraints |

alt35's 10 min of line-flying becoming 17.1 min once transits are counted independently confirms the 48 %
transit share measured from the frame timestamps, and explains why that pass runs against its 30 min cap.

**Still to do (needs the capture finished):** call `check_plan` as a PRE-FLIGHT refusal in `survey.py` and
`live.py`, the same way the cadence gate already refuses a plan that cannot confirm a track.

## Yield shortfall found mid-campaign, 2026-09-11 02:30 — and the fix that is not a lowered bar

Pass 1 (35 m) flew all 29 legs in 21.6 min, inside its cap, and produced **347 frames of 755 planned (46 %),
110 boxes, 44 of 69 survivors**. The missing 54 % are shots the shutter gate refused: `--max-tilt-deg 8` and
`--alt-tol-m 8`, with measured AGL 33.0-43.0 m against a commanded 35.

Projecting the rest of the campaign at that acceptance rate: **~1,028 frames and ~326 boxes**, under the
`dataset_gate.py` floor of 400 and far under SOLUTION_DOC 5.5c's 10-20k tiles. The acceptance figure comes
from alt55 alone (5.5c's nominal slice is 40-60 m), which is ~72 boxes - too thin to claim >= 90 % recall on.

**The fix: align the capture gate with the VALIDATION gate.** `survey.py` refuses to shoot beyond 8 deg of
tilt; `tools/capture/validate.py` fails a dataset only beyond **12 deg**. The capture is therefore throwing
away frames that the project's own quality standard would accept. Raising the capture gate to ~11 deg is not
a relaxation of the bar - the bar is 12 deg and is unchanged - it is removing a second, stricter, undocumented
bar that nothing asked for. Geolocation uses the measured attitude quaternion rather than assuming nadir, so
a mildly tilted frame is handled correctly rather than approximated.

Not applied mid-campaign: `campaign.py` launches each remaining pass as a fresh process that would pick up
the edit, so passes 2-4 would differ from pass 1 and the dataset would silently mix two capture standards.
Supplementary passes after the campaign instead, at the nominal-slice altitudes.

## The tilt gate may be discarding half the dataset for nothing — 2026-09-11 02:35

Pass 1 accepted 347 of 755 on-leg shot opportunities. Measured on the ACCEPTED frames:

    tilt deg   p50 4.40   p90 7.46   p99 7.95   max 8.00      <- the gate is exactly 8.0
    |AGL-35|   p50 0.81   p90 4.79   max 7.99                 <- gate 8.0

`max` sitting exactly on the gate is the signature of clipping, not coincidence. The physical cause is
ordinary: a multirotor sustaining 12 m/s holds a nose-down pitch of roughly 10 deg, so most of the CRUISE is
above an 8 deg airframe gate. That is why 54 % of on-leg opportunities produced no frame - transits are not
the explanation, because `est_frames` counts on-leg distance only.

**But the camera is gimbal-stabilised.** `sim/settings/dataset.json`:

    "Gimbal": { "Stabilization": 1.0, "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0 }

At `Stabilization: 1.0` the camera holds -90 deg world pitch whatever the airframe does, so **airframe tilt
does not tilt the image**, and a gate on airframe attitude is protecting against something that cannot
happen. If that is true:

* `survey.py`'s `--max-tilt-deg 8` is throwing away half the data for no imaging reason;
* `validate.py` check D, which fails frames beyond +-12 deg of AIRFRAME tilt, is also measuring the wrong
  body - it should measure the CAMERA;
* and `telemetry.csv`'s `gimbal_pitch_deg` is written as a hard-coded -90.0 rather than read back, so it
  cannot currently be used as evidence either way.

**Not acted on yet, deliberately.** A settings value is not evidence - this project has been bitten by
exactly that (`simGetCameraInfo().fov` reported 89.9 deg against a calibrated 73.98, a 21 % error in every
GSD). The test, once PIE is free: command a known airframe attitude, read `simGetCameraInfo` for the survey
camera, and compare the returned orientation against nadir. If the camera holds -90 deg, raise the capture
gate and re-measure yield. If it follows the airframe, the gate stays and the yield problem is solved by
flying slower instead, which lowers the cruise pitch.

### SETTLED from captured data, 2026-09-11 02:40: the camera is gimbal-stabilised, so the tilt gate is unnecessary

Tested without touching PIE, using 110 boxes whose survivors have known world positions. For each box,
predict where the survivor should land in the frame if the camera were NADIR, and compare with where the
auto-labeller actually found them. A RIGID camera would put the y-residual at `tan(pitch) * h / gsd`.

    predicted spread if rigid : 62.7 px
    observed y spread         : 95.7 px
    correlation(rigid prediction, observed) = **-0.061**

Essentially zero. A rigidly mounted camera would give a correlation near +1. **The camera does not follow the
airframe in pitch**, exactly as `sim/settings/dataset.json` claims with `Gimbal.Stabilization = 1.0`.

The large x-scatter (std 773 px against y's 96) was an artefact of MY test, not a defect: the drone flies
`DrivetrainType.ForwardOnly`, so it faces its direction of travel and yaw is **0 deg on 180 frames and 180
deg on 155** - it flies alternate legs backwards. The test assumed north-up. Nothing downstream assumes it:
labels are read from the instance MASK, and geolocation uses the measured quaternion rather than a heading.
The 180 deg flip is in fact free training diversity, since targets appear in both orientations.

**Consequence.** `survey.py --max-tilt-deg 8` rejected 54 % of on-leg shot opportunities to prevent an image
tilt that physically cannot occur. A multirotor holding 12 m/s sits at ~10 deg nose-down, so the gate was
excluding the cruise itself. Raising it admits those frames at no cost to image geometry.

**Two changes needed together, and they must land together or the dataset is validated by two standards:**
1. `survey.py`: raise the capture gate well above cruise pitch (~25 deg), keeping it only to exclude violent
   manoeuvres where gimbal lag is plausible.
2. `validate.py` check D: it currently FAILS frames beyond +-12 deg of AIRFRAME tilt, which would reject the
   very frames this change admits. It has to measure the camera, or apply the same reasoning.
Both are deferred until the campaign finishes, so passes 1-4 remain one consistent standard.

## Thermal (F9b) root cause found and implementation written, 2026-09-11 03:00 — needs PIE

**Why `Infrared` renders 100 % grey 0 despite every object accepting a segmentation id:** Cosys-AirSim
replaced `ImageType.Infrared` with configurable **annotation layers**. `Infrared` renders an object's
segmentation id as grey, and that feature has been superseded; the live channel is
`ImageType.Annotation = 11` driven by a declared layer.

**This also removes the landmine.** `tools/capture/thermal_ids.py` made thermal work by encoding temperature
into the SEGMENTATION id - so every object at the same temperature shared a colour. Run mid-flight on
2026-09-10 it collapsed the instance palette, the flood plane rendered in a survivor's colour, 78 % of boxes
became physically impossible and a 336-frame dataset was lost. A **greyscale annotation layer**
(`AnnotatorType.Greyscale = 1`) is a separate channel: temperature and instance ids coexist and neither can
corrupt the other. That is the difference - not a better encoding, a channel that was never shared.

Written and API-verified against the plugin source and the Python client:
* `sim/settings/dataset_thermal.json` - adds `"Annotation": [{"Name": "thermal", "Type": 1, "Default": true,
  "SetDirect": true}]` and a capture entry `{"ImageType": 11, "Annotation": "thermal"}`.
* `tools/capture/thermal_annotation.py` - assigns per-object temperatures through
  `simSetAnnotationObjectValue`, keeping the §2.3 physics from the old script (skin 34 C, wet skin 29 C,
  immersed 27.5 C, water 27.5 C, metal roof 64 C at midday - the classic false positive).

It refuses to believe itself three ways, because each has already fooled this project once: it FAILS if the
layer is not declared (a set on a missing layer is accepted and does nothing); it READS a value back rather
than trusting the return; and it FAILS if the rendered frame carries a single distinct value - which is
exactly the symptom `Infrared` showed for weeks while returning a perfectly valid image.

Remaining: relaunch the editor with `-settings=sim/settings/dataset_thermal.json`, run it, LOOK at the
frame. That unblocks F9/F9b and demo script step 4 (pre-dawn thermal, RGB-only fallback), which is
currently the only demo step with no path at all.
