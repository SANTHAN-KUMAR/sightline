# Lane report: environment assets (canopy, ground vegetation, utility poles, boats, rafted debris)

Session 2026-09-10/11. Editor never opened (orchestrator owns it); no `mcp__sightline__*` / `mcp__unreal__*`
calls; no GPU work. Everything downloaded to `D:\Sightline\_downloads\assets`. All numbers below are **sim**
(source-asset geometry), measured from the files on disk, not read off an API.

---

## 1. What was acquired — 24 Poly Haven downloads (CC0 1.0) + 1 generated asset

820.6 MB downloaded, every file size- **and md5-checked against the Poly Haven API** by
`tools/scene/fetch_env_assets.py`. 23 of the 24 downloads are used; `ship_pinnace` was downloaded, inspected
and rejected (§5). With the generated boat that is **25 new manifest rows**, and the palette the project uses
(`data/scene/env_assets.json`) holds **24 assets / 254 meshes**. Licences and source URLs are in
`_downloads/assets/MANIFEST.md` (now 104 assets / 2598.8 MB) and `MANIFEST.json`.

`tris` is what the project imports; `(full N)` means the import uses a thinned `*_2k_lite.gltf` and the
original stays on disk beside it. `size` is one mesh's real-world bounding box in Unreal axis order
(x, y, z-up).

### Canopy — the reference photo's biggest gap
| asset | author | tris imported | full | size (m) | what I saw |
|---|---|---|---|---|---|
| `jacaranda_tree` | Rob Tuytel, Rico Cilliers | **352,726** | 3,863,832 | 24.6 x 18.8 x **19.4** | the only mature tree on Poly Haven. Thick multi-stem trunk, high spreading crown, dense dark-green foliage. This is the photo's dominant element. |
| `island_tree_01` | Rob Tuytel, Rico Cilliers | **162,400** | 1,599,403 | 4.6 x 4.6 x **5.0** | stout short trunk, rounded crown — a yard/roadside tree, not a canopy tree |
| `tree_small_02` | Rico Cilliers | **156,172** | 2,062,487 | 2.9 x 4.3 x **4.5** | young single-leader tree, open crown, lighter green (Burkea africana) |
| `island_tree_02` | Rob Tuytel, Rico Cilliers | **106,821** | 1,072,213 | 4.0 x 4.1 x **3.3** | leaning multi-stem, **grows out of a rock** — riverbank only |
| `island_tree_03` | Rob Tuytel, Rico Cilliers | **111,522** | 2,085,320 | 2.9 x 2.9 x **2.5** | shrubby clump on a **grey rock slab** — waterline only; thinnest of the five after reduction |

**The names are misleading and this matters for placement:** only `jacaranda_tree` is a mature tree. The three
"island trees" are 2.5–5.0 m. Measured, not assumed — see the pictures in §2.

### Ground vegetation
| asset | meshes | tris (all meshes) | per-mesh range | tallest (m) |
|---|---|---|---|---|
| `grass_bermuda_01` | 21 | 941 | 8–268 | 0.15 |
| `grass_medium_02` | 5 | 7,842 | 714–2,489 | 0.43 |
| `shrub_03` | 4 | 8,287 | 1,790–2,385 | 0.40 |
| `weed_plant_02` | 5 | 11,116 | 1,766–2,967 | 0.07 (flat rosettes) |
| `calathea_orbifolia_01` | 5 | 16,688 | 1,802–5,904 | 0.53 |
| `grass_medium_01` | 17 | 24,730 | 28–6,422 | 0.33 |
| `shrub_02` | 4 | 27,254 | 5,188–9,234 | 1.71 |
| `shrub_04` | 1 | 27,327 | — | 0.22 (a 0.58 m strip) |
| `nettle_plant` | 6 | 31,304 | 1,088–8,080 | 0.22 |
| `anthurium_botany_01` | 6 | 67,152 | 7,664–15,808 | 1.14 (broad glossy tropical leaves) |
| `pachira_aquatica_01` | 8 | 76,914 | 584–24,341 | 1.65 (Malabar chestnut, a wetland species) |
| `shrub_01` | 1 | **156,012** | — | 0.40 (a **2.6 m verge strip in one mesh**) |

Two traps found by measurement, both invisible in a return value:
* `pachira_aquatica_01` imports as **8 separate meshes: 4 `_bark` and 4 `_leaves`**. Placing a `_bark` mesh on
  its own gives a bare stem. Each plant is a bark+leaves pair that must share one transform.
* `shrub_01` and `shrub_04` are single meshes containing a **row of plants** (2.59 m and 0.58 m long), not one
  plant. They are verge/hedge strips. `shrub_01` costs 156k triangles per placement.

### Utility poles and wires
| asset | meshes | tris | notes |
|---|---|---|---|
| `modular_electricity_poles` | 103 | 200,610 | a real kit: pole shafts **`pole_small` 288 tris / 5.0 m, `pole` 360 tris / 6.0 m, `pole_large` 400 tris / 7.0 m**, plus crossarm assemblies (`connection_large_01` 13,376 tris, `connection_small_01` 9,818), `fastener` 1,802, and 90-odd insulators / bolts / caps / transformers (median 198 tris). Three pre-assembled `preset_01..03` sets are included. |
| `modular_electric_cables` | 49 | 42,078 | **NOT overhead spans** — conduit, elbows, junction and meter boxes for house walls. Useful dressing; does not solve wires. |

### Boats and waterfront
| asset | meshes | tris | size (m) | notes |
|---|---|---|---|---|
| `country_boat_01` | 1 | **2,340** | 7.50 x 1.55 x 0.99 | **generated** by `tools/scene/gen_boat.py` — a Kerala vallam: lofted hull with rocker and sheer, open interior, 3 thwarts, plank strakes fore-and-aft. Skinned with the CC0 `weathered_planks` set (green channel flipped DX→GL for glTF). |
| `modular_wooden_pier` | 7 | 84,780 | sections 2.5 x 3.4 x 4.2 | plank jetty on poles with a hoist frame |
| `lifebuoy` | 1 | 10,728 | 0.78 x 0.16 x 0.84 | red/white ring buoy |
| `life_jacket` | 1 | 8,968 | 0.70 x 0.18 x 0.99 | orange life jacket |
| `ship_pinnace` | 7 | 168,317 | **9.5 x 39.8 x 32.6** | **downloaded, inspected, NOT used** — a three-masted square-rigged ship. See §5. |

### Rafted debris
| asset | meshes | tris | full | size (m) |
|---|---|---|---|---|
| `bark_debris_01` | 4 | 34,440 | 193,406 | slabs ~0.14 x 0.6 x 0.1 |

---

## 2. What I saw (pictures, and what is in them)

I cannot open the editor, so the visual proof is of the **source assets**: `tools/scene/gltf_tools.py render`
surface-samples the real triangles of the downloaded file, projects them through a perspective camera,
resolves occlusion with a painter's sort, and shades them with the model's **own base-colour texture**. Each
frame is captioned with the file name, the measured triangle count, the measured bounding box and a scale bar.

| picture | what is in it |
|---|---|
| `_artifacts/env_assets/sheet_trees.png` | the five originals, upright and to scale. Jacaranda is a real mature tree; the three island trees are 2.5–5.0 m and **two of them sit on visible grey rock outcrops**; `tree_small_02` is a slender young tree. |
| `_artifacts/env_assets/sheet_trees_lite.png` | the five reduced trees actually imported. Leafy, same silhouettes, bounding boxes within 3 % of the originals. `island_tree_03` is the weakest — visible bare branches at 111k. |
| `_artifacts/env_assets/cmp_jacaranda.png` | side by side: 3,863,832 → 352,726 tris (11x). Crown shape, density and size hold (24.33x19.09x19.45 → 24.60x18.64x19.38 m). Leaf clusters are chunkier and the crown edge is slightly gappier. |
| `_artifacts/env_assets/cmp_island_tree_01.png` | side by side at the final 162,400 tris. **An earlier attempt at 89,554 tris is what taught me the rule** — it rendered as a half-bare winter tree, so the budget split was changed to spend 82 % on leaves and starve the twigs, which are hidden under foliage in a live crown. |
| `_artifacts/env_assets/sheet_groundcover.png` | all 13 vegetation sets. This is where I saw that `grass_*`, `shrub_02/03`, `nettle`, `weed`, `anthurium`, `calathea` and `pachira` lay their variants out in a **row** (separate meshes), while `shrub_01`/`shrub_04` are one mesh holding a row of plants. |
| `_artifacts/env_assets/sheet_utility_water.png` | poles kit (3 assembled poles + loose crossarms, insulators, transformers, bare shafts), the cables kit (conduit and boxes — **no spans**), the pier, the pinnace (unmistakably a tall ship), the lifebuoy and the life jacket. |
| `_artifacts/env_assets/country_boat_01.png` | the generated boat from above: pointed ends, plank strakes running fore-and-aft, three thwarts, open interior. |
| `_artifacts/env_assets/cmp_bark_debris.png` | bark 193,406 → 34,440 tris, visually identical at 0.6 m, same bbox. |

---

## 3. The check that can fail, and its output

`tools/scene/check_env_assets.py` re-measures every asset from the bytes on disk and asserts: the glTF loads
and its `.bin` is long enough; every referenced texture exists **and decodes with PIL at >= 512 px**; no
triangle index points past the end of the vertex array and no position/UV is non-finite (the `_lite` buffers
are written by our own code, so they have to be proved); every mesh has triangles; every mesh's real-world
size is inside the band its role allows; each asset is under its role's triangle ceiling.

```
$ uv run python tools/scene/check_env_assets.py
jacaranda_tree               tree            1 mesh    352,726 tris  biggest [24.61, 18.785, 19.397]
island_tree_01               tree            1 mesh    162,400 tris  biggest [4.648, 4.613, 5.014]
...
bark_debris_01               debris_bark     4 mesh     34,440 tris  biggest [0.203, 0.463, 0.088]

by role: boat=2,340 tris, debris_bark=34,440 tris, ground_cover=455,567 tris, tree=889,641 tris,
         utility=242,688 tris, waterfront=104,476 tris

PASS: 24 assets, 254 meshes, 1,729,152 unique triangles, every texture decodes, every size in band
exit=0
```

**Proof it can fail** — four fixtures, run against the same code with `--table/--root`:

```
$ uv run python tools/scene/check_env_assets.py --table <neg>/table.json --root <neg>
FAIL (4):
  trunc: glTF will not load: t.gltf: buffer t.bin is 1101333 B, glTF declares 3304000 B - truncated download
  badtex: texture lifebuoy_diff_2k.jpg does not decode (cannot identify image file ...)   # a 4 KB HTML error page
  boat_as_groundcover: source glTF missing: ...\country_boat_01.gltf
  missing: source glTF missing: ...\nope\nope.gltf
exit code = 1

$ ... (the boat declared as ground_cover, file present)
FAIL (1):
  boat_as_groundcover/country_boat_01: size [7.5, 1.55, 0.995] m outside the ground_cover band 0.01-7.0 m
exit code = 1
```

### A regression this check chain caught before it shipped
`bark_debris_01` was first filed under the existing `woody` debris role. Running `gen_props.py` then produced
**553 items / 6,499,992 tris (budget exhausted, no vehicles and no hillslope vegetation placed at all)** instead
of the baseline 919 items / 4,153,401 tris — because `gen_props.pick()` biases towards the cheapest 70 % of a
role's pool and the 38–65k bark slabs dragged the woody average up. Fix: bark gets its own role
`debris_bark` (inert until the orchestrator wires it) and a `_lite` build at 34,440 tris. After the fix,
`data/scene/props_layout.json` re-generates **byte-identical to the baseline** (sha256
`c32a3c1b16193c1e4908f1e666e96be4ae70756d819df5ed298a5176a1f843f7`). The existing debris field is untouched.

---

## 4. How to import it

The editor-side importer already knows about everything; it reads `data/scene/env_assets.json`, the same table
the host-side check reads.

```
mcp__sightline__ue_python  code="exec(open(r'D:\\Sightline\\tools\\scene\\import_assets.py').read())"
```

That imports the 27 original prop sets + 9 Rocketbox humans **plus** the 24 new assets into
`/Game/Sightline/Props/<id>/<file stem>/StaticMeshes/<node name>`, turns Nanite off (§4), and rewrites
`data/scene/props.json`. New behaviour added for this lane:

* **LOD chains.** Assets marked `"lods": true` get a 4-step reduction chain
  (`StaticMeshReductionSettings.percent_triangles` = 1.0 / 0.40 / 0.15 / 0.06, auto screen sizes) via
  `StaticMeshEditorSubsystem.set_lods`. Wrapped in try/except: a failure is recorded in
  `report["failures"]` and never aborts the import. **This is the one thing in this lane I could not run** —
  it needs the editor. Check `lod_count` in the printed table after the run; if `set_lods` is unavailable in
  5.8 the assets still import, just without LODs.
* **No collision on foliage.** `tree` and `ground_cover` get no simple collision box: a 19 m crown's box
  would swallow the street, and a survivor must never be blocked by a leaf. Existing roles are unchanged.

To rebuild the sources from scratch:
```
uv run python tools/scene/fetch_env_assets.py                       # download + md5 verify (820.6 MB)
uv run python tools/scene/gen_boat.py                               # the vallam
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/jacaranda_tree/jacaranda_tree_2k.gltf --target 300000
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/island_tree_01/island_tree_01_2k.gltf --target 160000
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/island_tree_02/island_tree_02_2k.gltf --target 110000
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/island_tree_03/island_tree_03_2k.gltf --target 110000
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/tree_small_02/tree_small_02_2k.gltf  --target 140000
uv run python tools/scene/gltf_tools.py lite _downloads/assets/polyhaven/bark_debris_01/bark_debris_01_2k.gltf --target 40000
uv run python tools/scene/check_env_assets.py --write               # verify + merge the palette into props.json
```

### How the trees were reduced (and why not a quadric decimator)
Measured first: the leaves are **real modelled geometry, not alpha cards** — jacaranda has 116,084 leaf
components averaging 20.7 triangles each, and 33,145 twig components averaging 37. So:
* **leaves and twigs** are thinned by deleting **whole connected components**, largest first so structural
  limbs survive, and the surviving leaves are grown about their own centroid by 1/sqrt(keep) (capped at 2.4x)
  so the crown keeps its coverage. A quadric collapse over leaf geometry shreds it into spikes; deleting whole
  leaves is what a foliage artist does by hand.
* **trunks and bark** are decimated by **vertex clustering** on a metric lattice (jacaranda trunk 230,112 →
  58,364 at a 5.1 cm cell), which keeps the silhouette.
* 82 % of the foliage budget goes to leaves, 18 % to twigs — because thinning both evenly produced a tree that
  read as **dead** (`cmp_island_tree_01.png`, first attempt).

---

## 5. What the reference photo needs that I could not source

| wanted | status | why |
|---|---|---|
| **Overhead wires strung between poles/houses** | **not sourceable** | Poly Haven has poles and crossarms but no catenary span; `modular_electric_cables` is wall conduit and boxes (inspected). ambientCG has no models of this kind. **Recommendation:** generate them — a catenary `y = a·cosh(x/a)` swept as a 6-sided tube between two crossarm attachment points is ~120 tris per span, so ~40 spans ≈ 5k triangles. The `fastener` (1,802 tris, 1.45 x 3.06 m) is the crossarm to anchor to. |
| **Small boat** | **generated instead** | Poly Haven's only vessel is `ship_pinnace`, measured at 39.8 m long and 32.6 m tall with masts — a 17th-century square-rigged ship. Scaling it to 7 m would put a warship in a Kerala village, so I built `country_boat_01` procedurally (the same approach as the 73 houses). |
| **Coconut / areca palm** | still blocked | The single biggest remaining species gap: a Kerala settlement with no palms. Nothing photoreal and CC0/login-free exists; Fab and Sketchfab both need a login and Fab has no UE 5.8.2 build. |
| **Banana / plantain plant** | still blocked | Poly Haven `bananas` is a bunch of fruit, not a plant. |
| **Boats in motion / wakes** | not an asset problem | needs a material or a moving actor, not a mesh. |

---

## 6. Triangle cost and recommended counts

The scene already carries ~524k (terrain, **sim**) + 4,153,401 (919 debris instances, **sim**) and the machine
has 8 GB VRAM. Two currencies matter and they are not the same:

* **LOD0-accounted** = `mesh LOD0 triangles x instances`. This is exactly what `gen_props.py` sums into its
  `max_tris` budget, so it is the currency the brief's "~2M more" is in.
* **Rendered** = what the GPU actually draws after LOD selection. With the 4-step chain the importer now
  builds, a 19 m tree at 40–100 m AGL lands on LOD1–LOD2, i.e. **0.15–0.40x LOD0**.

Unique geometry residency for the whole new set is **1,729,152 triangles / 254 meshes** — roughly 60 MB of
VRAM. That is not the constraint. The ~130 new 2K textures (≈ 3–6 MB each once BC-compressed) are the real
VRAM cost, on the order of 400–700 MB.

### Plan A — fits inside ~2.0M LOD0-accounted triangles
| what | per instance | count | subtotal |
|---|---|---|---|
| `jacaranda_tree` | 352,726 | 2 | 705,452 |
| `island_tree_01` | 162,400 | 1 | 162,400 |
| `tree_small_02` | 156,172 | 1 | 156,172 |
| `island_tree_02` | 106,821 | 1 | 106,821 |
| `island_tree_03` | 111,522 | 1 | 111,522 |
| `grass_bermuda_01` (all 21, avg ~45) | ~45 | 800 | 36,000 |
| `grass_medium_01`, 11 cheap variants (avg ~250) | ~250 | 400 | 100,000 |
| `grass_medium_02` (avg 1,568) | ~1,568 | 40 | 62,720 |
| `weed_plant_02` (avg 2,223) | ~2,223 | 30 | 66,690 |
| `shrub_03` (avg 2,072) | ~2,072 | 30 | 62,160 |
| `calathea_orbifolia_01` (avg 3,338) | ~3,338 | 15 | 50,070 |
| poles: shaft + fastener + insulators | ~3,000 | 20 | 60,000 |
| poles with a full crossarm assembly | ~14,000 | 4 | 56,000 |
| `country_boat_01` | 2,340 | 12 | 28,080 |
| `lifebuoy` / `life_jacket` | ~9,850 | 8 | 78,800 |
| `modular_wooden_pier`, one 4-section jetty | ~63,000 | 1 | 63,000 |
| `bark_debris_01` (avg 8,610) | ~8,610 | 12 | 103,320 |
| **total** | | | **≈ 2,009,207** |

That is **6 trees**. It is a village with trees in it; it is not the reference photo.

### Plan B — what the photo actually looks like
| species | count | LOD0-accounted | est. rendered at 40–100 m AGL |
|---|---|---|---|
| `jacaranda_tree` (19.4 m, yaw random, scale 0.8–1.25) | 24 | 8,465,424 | 1.3M–3.4M |
| `island_tree_01` (5.0 m; **scale 2.2–2.6x for second-storey canopy**) | 30 | 4,872,000 | 0.7M–1.9M |
| `tree_small_02` (4.5 m, gardens) | 20 | 3,123,440 | 0.5M–1.2M |
| `island_tree_02` (3.3 m, riverbank — has a rock base) | 12 | 1,281,852 | 0.2M–0.5M |
| `island_tree_03` (2.5 m, waterline only — rock slab base) | 8 | 892,176 | 0.1M–0.4M |
| **trees total** | **94** | **18,634,892** | **≈ 2.8M–7.4M** |

plus ground cover at roughly 4x Plan A's counts (≈ 1.5M LOD0-accounted, mostly sub-3k meshes), 24–30 poles,
12–20 boats.

The rendered column is an **estimate**, not a measurement — I could not open the editor. It comes from the
LOD percentages the importer applies and the screen fraction a 19 m tree subtends at 40–100 m. Recommended
next step for whoever owns the editor: place Plan A first, fly the survey path, read `stat rhi` /
`stat scenerendering`, then scale towards Plan B until the frame budget bites.

Two changes are needed for Plan B to be legal in the current pipeline, and both belong to the orchestrator:
1. `gen_props.py`'s `max_tris` accounting is LOD0-based and would reject Plan B on paper while the GPU is idle.
   Foliage should be budgeted separately from debris, or budgeted at its LOD2 cost.
2. Trees should be **HISM / Foliage instances**, not one `StaticMeshActor` each, or 94 trees add 94 draw calls
   on top of the 919 debris actors `build_props.py` already spawns.

### Placement notes that come out of the measurements
* Only `jacaranda_tree` is a canopy tree. To vary the skyline, use random yaw plus 0.8–1.25 non-uniform scale,
  and use `island_tree_01` **scaled 2.2–2.6x** (→ 11–13 m) as the second-storey canopy. Its leaves become
  ~2.5x life size; at 40 m AGL a leaf is a few pixels, so this is invisible from the air and wrong on the
  ground.
* `island_tree_02` and `island_tree_03` include **rock outcrops at the base**. Plant them on the riverbank or
  at the waterline, never in a garden.
* `pachira_aquatica_01` is bark + leaves as separate meshes: place each pair on one shared transform.
* `shrub_01` (2.6 m) and `shrub_04` (0.58 m) are strips of plants in a single mesh — lay them along verges and
  fence lines, not as single bushes. `shrub_01` costs 156k per placement; keep it to 2–3, or rely on its LOD.
* `bark_debris_01` is deliberately in its own role. To fold it into the wrack line, change its role to `woody`
  **and** re-check the debris budget — see the regression in §3.

---

## 7. Files this lane touched

Created: `tools/scene/fetch_env_assets.py`, `tools/scene/gltf_tools.py` (stats / render / lite / sheet),
`tools/scene/gen_boat.py`, `tools/scene/check_env_assets.py`, `data/scene/env_assets.json`,
`_artifacts/env_assets/*.png`, `_downloads/assets/polyhaven/<24 ids>/`,
`_downloads/assets/generated/country_boat_01/`.

Modified: `tools/scene/import_assets.py` (env-asset import, LOD chains, foliage collision rule),
`data/scene/props.json` (27 → 51 props; the 24 new entries carry measured `tris` and `size_m`),
`_downloads/assets/MANIFEST.md`, `_downloads/assets/MANIFEST.json`.

Not touched, as instructed: `tools/scene/gen_props.py`, `tools/scene/build_props.py`, anything under
`sightline/`. `data/scene/props_layout.json` was regenerated only to prove it came back byte-identical.

**Stubs and known-unverified items, named plainly:**
1. `build_lods()` in `import_assets.py` has never been executed — it needs the editor. It is defensive
   (failures are recorded, the import continues) but it is unproven.
2. The predicted Unreal asset paths in `props.json` are inferred from the naming rule the existing 27 prop
   sets demonstrably produced (file stem for single-mesh files, node name for multi-mesh). `import_assets.py`
   overwrites them with the real paths when it runs; if a name differs, `build_props.py` will report it as a
   missing asset rather than failing silently.
3. `country_boat_01` is procedural geometry, not a scan. It is labelled as generated in both manifests.
4. The "est. rendered" column in Plan B is an estimate, not a measurement.
