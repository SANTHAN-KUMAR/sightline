# Aerial Survivor Triage System — Research, Stack and Build Document

**Problem statement:** PS 2, AI for Disaster Response & Public Safety — *Real-Time Vision System for Identifying Survivors in Flood, Landslide and Tsunami Zones*.
**Prepared:** 10 September 2026. **Target dev machine:** Windows 11, NVIDIA RTX 4060 (8 GB VRAM), 16 GB RAM. **Deployment target named by the PS:** NVIDIA Jetson Orin, < 300 ms per frame.

**Contents:** 0. How to read · 1. The problem, re-framed · 2. The disaster we simulate (scenario choice, causal attribute model, correlated priors, sensor numbers, the burial boundary) · 3. Solution overview · 4. Hardware envelope · 5. The stack (5.1 simulation, 5.2 flight control and takeover, 5.3 search planning, search-quality map, decision planner and per-presentation coverage, 5.4 ingest, 5.5 detection and fusion, 5.5a posture and submersion head, 5.5b radiometric thermal, 5.5c simulation-first training, 5.6 tracking and dedup, 5.7 geolocation, 5.8 triage output, 5.9 map and C2, 5.10 offline and cloud, 5.11 edge latency, 5.12 evaluation) · 6. Data plan and annotation guideline · 7. Full user flow · 8. Feature list · 9. Build plan and demo script · 10. Risks and day-1 tests · Appendices A–E (reuse ledger, formulas, FMCW radar paper, gods-eye-view, papers).

---

## 0. How to read this document

This document is linear. Each section is built on the one before it: the problem is re-framed (§1), one disaster scenario is chosen and decomposed into the physical attributes that make detection hard (§2), the solution is stated (§3), the hardware envelope is fixed (§4), the stack is chosen component by component with verified reuse candidates (§5), the data plan follows from the stack (§6), the user flow and feature list follow from the data plan (§7–8), and the build plan and risks close it (§9–10). Appendices hold the reuse ledger, formulas and notes on the two references you gave that needed identification.

**One framing decision governs everything below: this is a simulation-first project.** The drone, sensors, scene and disaster are rendered, the demo runs on rendered frames, and the acceptance thresholds are demonstrated on rendered frames. The renderer is the deployment domain, not a training aid. §5.5c states what that means for training, for the recall target and for how every number must be reported; read it before acting on the data plan in §6, which is written for the optional real-world variant.

Three conventions are used throughout:

- **Verified** means a repository page, release, documentation page, spec sheet or paper was fetched on 10 Sep 2026 and the number or claim was read from it. Star counts, licences and last-activity dates are as of that day.
- **Estimate** means a value computed from verified inputs (for example a ground sampling distance computed from a published focal length) or an engineering estimate that has not yet been measured on your hardware. Estimates are labelled.
- **Unverified** means the research could not confirm it; these are collected in §10 so you can test them on day 1 instead of discovering them on day 5.

Every tool named here was checked for four things before being recommended: it runs on Windows 11 without a Linux-only toolchain (or the exception is called out), it fits 8 GB of VRAM and 16 GB of RAM, its licence is compatible with an open hackathon repository, and it is still maintained. Tools that fail one of these are listed as *rejected* with the reason rather than silently omitted, so the team can see the decision instead of re-deriving it.

---

## 1. The problem, re-framed

### 1.1 What the problem statement actually asks for

The PS lists seven modules and a set of thresholds. Restating them as numbered requirements makes it possible to trace every design decision and every feature back to a line in the PS.

| ID | Requirement (from the PS) | Threshold / constraint |
|---|---|---|
| R1 | **Ingest**: read a video stream or recorded flight plus a telemetry log (CSV or MAVLink), synchronised on timestamp | 4K at ≥ 30 FPS input |
| R2 | **Detection**: detect living beings under occlusion and non-standard pose; classes `human`, `animal` | Recall ≥ 90 % at IoU 0.5 on `human`; recall prioritised over precision |
| R3 | **Fusion**: fuse RGB with thermal where available; fall back cleanly to RGB-only | YOLOv8 / EfficientDet / RTMDet or equivalent "with a fusion head" |
| R4 | **Tracking and deduplication**: persistent IDs so one survivor produces one record | Deduplication accuracy reported |
| R5 | **Geolocation**: project image coordinates to lat/long from telemetry (altitude, attitude) and camera intrinsics | State the error budget for the assumed altitude |
| R6 | **Triage output**: confidence-ranked GeoJSON or KML with location, confidence, movement vs stillness, estimated count, evidence thumbnail, rendered on a map | GeoJSON/KML plus map view, evidence crop per record |
| R7 | **Offline queue**: buffer locally when connectivity is lost, synchronise on reconnect | — |
| R8 | **Evaluation script**: recall at IoU 0.5, false positives per minute, deduplication accuracy on a held-out clip | Recall ≥ 90 %, FP/min reported explicitly |
| R9 | **Latency**: per-frame processing on Jetson Orin, cloud fallback, offline queueing | < 300 ms per frame |
| R10 | **Guardrail**: the system recommends only; it must never close out a search area automatically | Hard rule, not a setting |
| R11 | **Annotated dataset** with occlusion and pose diversity, plus the annotation guideline for partially visible subjects | Deliverable |
| R12 | **Replay harness or live camera input** producing the ranked list on a live map | Deliverable |

Your framing adds two things the PS does not ask for but which make the solution coherent rather than a bag of modules:

- **A13 (added): the simulation is the product's test bench.** The drone, sensors, scene and disaster are simulated in Unreal Engine or Unity, with a drone that flies search patterns autonomously *and* can be taken over by a human on a gamepad at any moment.
- **A14 (added): a quantified coverage map.** Alongside the triage list, the system outputs a map of *where nobody has looked well enough yet*. This is the differentiator, and §5.3 shows it drops out of search theory almost for free once the geolocation module exists.

### 1.2 Why this is hard: the five failure modes

The PS names the difficulties in prose. Below they are turned into the five concrete failure modes the system must be designed against, because each one maps to a different part of the stack.

1. **The target is not a pedestrian.** Street-level detectors learn upright, walking, fully visible humans. Here the subject is prone, supine, curled, half-submerged, or reduced to a head above brown water or a limb under a sheet of roofing. A 2025 thermal dataset built for exactly this (POP, DJI M30T at 30–70 m) found that detectors trained on COCO or on a standard thermal UAV set lose about 0.70 mAP50 when moved to occluded, posed subjects; the same paper found accuracy holds until the occluded fraction exceeds ~70 %, *provided occluded examples were in the training data*. The fix is data (pose- and occlusion-diverse, real plus synthetic) and an annotation rule that boxes what is visible — not a cleverer architecture.
2. **The target is small and its size is not fixed.** At 60 m with a 4K wide camera a shoulder width of 0.45 m is about 20 px; at 120 m it is 10 px. A 640 × 512 thermal core at 60 m gives under 6 px on the same target. Below ~20 px along the critical dimension, learned detectors drop off sharply (C2A, HERIDAL, TinyPerson evidence in §2.5). The fix is *resolution management*: tiled inference on the full 4K frame, an altitude policy that keeps targets above the pixel floor, and a thermal camera used for cueing rather than confirmation above ~60 m.
3. **The environment attacks both bands.** Silt-brown water hides everything below the surface in RGB and in thermal alike; sun glint saturates RGB; rain, fog and wet clothing collapse thermal contrast; thermal crossover around sunrise and sunset can erase a person on a roof for an hour; sun-heated roofing sheets and rocks read as warm as a person at midday. §2.3 lists 27 of these attributes with their physical cause, their effect on each band and on the detector, and the simulator knob that reproduces each one. The fix is to *randomise* those knobs in the synthetic data and to *slice* the evaluation by them, so the team knows in which conditions the 90 % recall claim holds and in which it does not.
4. **The cost is asymmetric, in both directions.** A missed survivor may be fatal; but a flood of false positives makes responders stop trusting the list. Recall is prioritised at the detector, and precision is recovered *downstream*: temporal persistence across frames, a second-stage verifier on candidate crops, RGB–thermal agreement, and geo-deduplication so that a false positive costs one map marker rather than forty. FP per minute is reported as a first-class metric, at the operating threshold actually used.
5. **The output is a decision aid, not a video overlay.** A survivor visible in frame 14,000 of an unwatched flight is never found. The deliverable is a ranked, deduplicated, geolocated list with evidence, plus the coverage map that tells the incident commander where the drone has *not* yet produced a trustworthy negative. The system never converts "nothing detected" into "area cleared" (R10), and search theory gives the formal reason: probability of detection is always below 1, so after an unsuccessful pass the probability that the subject is in a segment shrinks but never reaches zero.

### 1.3 The two products

The system produces two artefacts for the incident commander, both rendered on the same map:

**Product 1 — the triage list.** One record per living being, ranked. Each record carries: location (lat/long with an error radius), class (`human`/`animal`), fused confidence, movement vs stillness, estimated count (for clusters on a roof), posture and submersion attributes when inferable, the evidence thumbnail from the best frame, the first- and last-seen timestamps, the flight and frame it came from, and the components of its priority score shown separately (not collapsed into an opaque number). It is exported as GeoJSON (primary) and KML (for Google Earth/ATAK users), and as Cursor-on-Target messages for teams running TAK (§5.9).

**Product 2 — the search-quality map.** A raster over the area of operations where each cell accumulates *how well* it has been observed, not merely whether the drone flew over it. The per-pass quality of a cell is a function of the attributes in §2.3 that the system can measure or knows from telemetry: ground sampling distance (from altitude and camera), motion blur (from speed and exposure), band available (RGB, thermal, both), time-of-day thermal contrast, view angle, and occlusion from canopy or structures. The cumulative value is converted to a probability-of-detection using the Koopman exponential model (§5.3), so the map reads as "if a person were here, what is the probability we would have seen them by now". Cells under deep burial polygons are marked *aerial search cannot clear*, because the 2024 Wayanad search showed that neither thermal drones nor respiration radar find buried victims (§2.1).

### 1.4 Non-goals and guardrails

- The system recommends and ranks; it never closes a segment, never deletes a record, and never lowers a record's priority to zero. Operators can mark a record *resolved* with a reason; the record stays in the log.
- No identification of individuals: no face recognition, no re-identification across incidents. Re-identification embeddings are used only to merge tracks of the same person within a flight, at aerial scale where faces are not resolvable anyway.
- **Fully buried victims — no visible part in any band — are out of scope for the vision system, and no camera changes that.** A person with a hand, head, shoulder or shin exposed is emphatically *in* scope and is one of the highest-value cases the detector is trained for. §2.7 draws that boundary with the physics and states what the system does instead of pretending a negative result is a clearance. The FMCW radar paper you referenced (Appendix C) is the right *complementary* sensor for the fully-buried case at 1–2 m standoff; it is not an aerial-survey sensor.
- The simulation is a test bench and demo vehicle, not a claim of field readiness. Every accuracy number produced in simulation is labelled as such; the real-footage replay path (R12) exists so that the same pipeline can be run on recorded DJI footage with its embedded telemetry.

---

## 2. The disaster we simulate

The PS spans floods, landslides and tsunamis. A hackathon cannot simulate three disasters convincingly, and the three differ in what the drone sees. One scenario must be chosen that covers as much of the others as possible.

### 2.1 Scenario comparison

The research compared the three on five criteria that matter for *this* project: how survivors present to a camera, how hard the scene is to build in a game engine, how much real aerial data exists to anchor it, frequency and relevance in India, and how much a thermal camera contributes. Scores are 1 (poor) to 5 (excellent).

| Criterion | Flash / riverine flood with debris | Landslide / debris flow | Tsunami |
|---|---|---|---|
| Survivor presentation from the air | **5** — rooftops (≈30,000 people at once in Kerala 2018), upper floors, trees, elevated roads, clinging to floating debris, wading, head-only; postures from upright and waving to prone and exhausted; animals abundant (Assam: 8,529 animals washed away in one 2024 bulletin) | **2** — the people you most want are buried and invisible to any camera. In Wayanad 2024 an Army respiration radar reported "three breath signals", teams dug 3 ft and found nothing ("possibly a frog or snake"); a private thermal-drone survey of Mundakkai found "zero human presence"; >200 bodies or parts were recovered along a 40 km stretch of the Chaliyar instead | **3** — a debris field plus water, but the rescue phase is short and most work is recovery |
| Ease of convincing simulation | **5** — a water surface, a river spline, buoyant debris, scattered props and flooded building interiors; UE5 and Unity both have this built in | **3** — terrain deformation, mud material, boulder scatter, destroyed buildings; and a convincing "burial" is meaningless to a camera | **2** — coastline inundation and very large debris volumes; hard to make it not look like a flood |
| Real aerial data and footage | **5** — FloodNet (2,343 UAV images, 1.5 cm GSD), RescueNet, C2A flood backgrounds, floating-object datasets, abundant Indian drone news footage | **3** — Wayanad news drone footage; C2A rubble backgrounds; no labelled person-in-landslide aerial set found | **1** — 2004 predates consumer drones |
| Frequency and relevance in India | **5** — annual (Assam, Bihar, Kerala, Himachal, Uttarakhand); 12 % of the landmass is flood-prone | **4** — every monsoon in the Western Ghats and Himalaya; Wayanad 2024 the deadliest | **1** — one event, 2004 |
| Value of thermal | **4** — night rooftops and trees against cool water: strong; immersed bodies: contrast collapses to roughly one third with the torso immersed, which gives a clean *failure slice*; sun-heated roofing sheets are hot distractors | **3** — surface survivors on evaporatively-cooled wet mud contrast well at night; buried: nothing; midday boulders: false positives | **3** — as flood |
| **Total** | **24** | **15** | **10** |

Sources for the table are in the flood-domain research file (Wikipedia, Onmanorama, Tribune, npj Natural Hazards, Springer *Landslides*, Eos, Asia Times, Deccan Herald, Outlook, FloodNet and RescueNet papers). Key event facts: Wayanad debris flow 30 Jul 2024, ~0.83 million m³ of material, 6.9–8 km runout, flow height 7.3 m, deposit ~4 m thick 4 km downstream, 427 of 1,018 structures destroyed, 1,531 rescuers, casualty totals 224–420 dead and 118–276 missing depending on source and date. Kerala 2018: 483 dead, ~30,000 on rooftops at the peak, 40 helicopters and 500 boats. Sikkim GLOF 2023: a night event whose flood travelled 385 km and left bodies in Bangladesh.

### 2.2 The chosen scenario and what it still covers

**Simulate one map: a debris-flow-fed monsoon flash flood in a hill-valley settlement — the "Wayanad → Chaliyar" family (also Dharali 2025, Sikkim–Teesta 2023).** One terrain, one weather system, one time-of-day clock, three zones:

1. **Upstream deposit fan** (this is the landslide coverage): 2–7 m of mud and boulders over part of the settlement footprint, broken houses, uprooted trees, mud-coated survivors on upper floors, roofs and the fan margins. The ground truth also contains *buried* actors that are not visible from any angle, so the evaluation can check that the system reports nothing for them and that the coverage map does not claim the polygon cleared.
2. **Flooded settlement** (riverine flood coverage): rooftops with single people and crowds, upper-floor windows and balconies, trees, an elevated road, stranded vehicles with people on the roof, cattle and goats and dogs on high ground.
3. **Channel and banks downstream** (flash flood and drift coverage): fast silt-brown water carrying timber, roofing sheets and a vehicle; people clinging to debris or wading; recirculation zones, bends and a debris dam where floating people and bodies accumulate (drowning-victim drift studies report 0.5–8 km typical drift with accumulation on the shores of recirculation systems).

What this covers: riverine and flash floods fully; the *surface-visible* part of a landslide (which is the only part any camera can see); the water-plus-debris look of a tsunami. What it does not cover: coastal inundation geometry and the specific salt-water, beach and estuary presentation of a tsunami. That gap is accepted; a second scene is not worth the build time.

Why this matters for the ML pipeline: the three zones give three distinct background distributions (mud/boulder, built environment/rooftops, moving brown water/debris) and three distinct survivor presentations (mud-coated and partially buried; upright, waving, clustered; immersed and clinging). A detector that scores 90 % recall on rooftops but 40 % on the channel is a different product from one that holds across all three, and the evaluation must be sliced by zone to tell the difference.

### 2.3 Physics-to-pixels: the causal attribute model

This table is the core of the data plan. Each row is a physical attribute of the scene that *causes* a change in what the camera records, which in turn *causes* a change in detector behaviour. For every row the table gives the simulator parameter that reproduces the attribute, a realistic randomisation range, and whether the attribute is kept as a fixed evaluation slice (Y) or only randomised during training (N). "R" is recall, "P" is precision. Numbers with a source are verified; ranges marked *proposed* are engineering choices anchored on the evidence.

| # | Physical attribute | What causes it | Effect on the RGB image | Effect on the LWIR thermal image | Effect on the detector | Simulator parameter | Randomisation range | Slice |
|---|---|---|---|---|---|---|---|---|
| 1 | **Water turbidity and colour** (silt-brown) | Monsoon suspended sediment; landslide fines | Water opaque, brown, low saturation; submerged limbs invisible; body, mud and water colours converge | Water is opaque to LWIR regardless of turbidity; only the surface skin temperature is seen | Submerged parts lost in both bands → only head/shoulders/arms above water detectable → R↓ for immersed people; brown water ≈ mud ≈ brown clothing → P↓ | Water material absorption/scattering colour; opacity depth (0.05–0.3 m) | 60–900 NTU (Indian monsoon rivers: Panchet 60→700, Brahmaputra 73–875) | Y (clear vs turbid) |
| 2 | **Submersion fraction of the body** | Where the person is: channel, wading, clinging, roof | Visible silhouette shrinks from full body (1.7 × 0.45 m) to head and shoulders (~0.25–0.45 m) | Immersed skin reaches water temperature within minutes (water removes heat ~25× faster than air); contrast falls to ~⅓ with torso immersed, less with head only | Effective target size falls below 20 px far earlier; limb-based pose cues vanish → R↓ steeply with altitude | Actor placement class: on-roof 0 %, wading 30–60 %, clinging 60–90 %, head-only ~95 %; water level vs terrain | *Proposed* mix: 50 % roof/upper floor, 20 % tree/road/vehicle roof, 20 % wading/clinging, 10 % head-only (no published statistics exist; Kerala 2018 "30,000 on rooftops" and flood-fatality studies are the proxies) | Y (per class) |
| 3 | **Body posture** | Exhaustion, injury, hypothermia, sleep, death; signalling | Nadir footprint changes shape: upright ≈ 0.45 × 0.25 m; prone ≈ 1.7 × 0.45 m; fetal ≈ 0.8 × 0.6 m; lying bodies resemble logs and bags | Prone body on a hot roof sheet loses contrast; upright body against water has the strongest contrast | Pedestrian-trained detectors fail on lying and bent subjects (C2A evidence); <20 px targets detected less often and at lower confidence | Animation/ragdoll pose set; per-actor pose-class sampling | C2A taxonomy: bent, kneeling, sitting, upright, lying; *proposed* 35 % upright, 20 % sitting, 15 % crouched/fetal, 15 % prone, 10 % supine, 5 % kneeling; deceased: floating face-down | Y (upright vs lying; waving vs static) |
| 4 | **Debris density and type** | Entrained houses, plantations; debris-flow load ("oversaturated soil, vegetation, large boulders") | Partial occlusion; human-sized elongated objects (logs, sheets, bags); texture clutter | Galvanised sheets (emissivity 0.20–0.30) mirror the cold sky and read cold; rusted sheets (0.70–0.95) and boulders read hot at midday | Occlusion → R↓; human-like debris → P↓; but debris also *correlates positively* with survivors (people cling to it) | Debris spawner: density, size distribution, material class, buoyancy; per-actor occlusion fraction | 0–30 items per 100 m² (*proposed*); Wayanad: 427/1,018 structures destroyed → high built-debris fraction; material mix ~40 % vegetation, 25 % timber, 15 % roof sheet, 10 % plastics/cloth, 5 % vehicles, 5 % boulders (*proposed*) | Y (low/med/high; occlusion 0/25/50/75 %) |
| 5 | **Sun glint on water** | Fresnel reflection of the sun off wavelets; worse with high sun and calm water | Saturated white streaks; contrast loss; auto-exposure darkens the rest of the frame | Water emissivity ~0.95–0.98 near nadir but drops sharply above ~70° incidence → oblique views reflect the cold sky; little LWIR glint | Bright blobs → P↓; under-exposure → R↓; oblique thermal keeps contrast | Sun azimuth/elevation, wind-driven wave spectrum, camera pitch, exposure model | Glint variability worst above ~54° solar elevation; recommended survey window 25–47°; Indian monsoon noon sun ≈ 80° | Y (glint on/off) |
| 6 | **Rain** | Monsoon convective cells (Wayanad: 570 mm in the two days before) | Streaks, lens droplets, veiling haze; visibility falls with rate | Droplets absorb IR; wet surfaces cool and equalise; FLIR: IR degradation from rain "is very range sensitive", dropping off dramatically at 100–500 m | R↓ in both bands; thermal contrast ↓ (wet clothing, evaporative cooling) | Rain particles + lens-droplet post-process + atmospheric extinction + surface wetness/temperature coupling | Light 2–10, heavy 20–50, extreme 100 mm/h (LWIR dB/km per rain rate is unverified; treated qualitatively) | Y (dry/light/heavy) |
| 7 | **Fog / mist** | Valley humidity after rain; dawn | Contrast and colour desaturate; targets vanish | LWIR helps only in light fog: ~4× visual range at 610 m visibility; no advantage at ≤ 305 m | R↓ sharply below ~300 m visibility in both bands | Exponential height fog / volumetric fog density | Visibility 1,220 / 610 / 305 / 92 m (FLIR Cat I–IIIc) | Y (Cat I/II/IIIa) |
| 8 | **Dust** after a landslide | Dry debris, secondary collapses | Brown veiling haze | Mild attenuation | Local R↓; minor in monsoon conditions | Brown-tinted particle fog | Short events (unverified; treat as minor) | N |
| 9 | **Low light / night** | Time of day; overcast | Full moon 0.05–0.3 lux, overcast night 0.0001 lux, twilight 3.4 lux, overcast day 1,000 lux; RGB useless below a few lux | Unaffected by light; pre-dawn is the best window (ground has shed heat, the person has not) | RGB R→0 at night; thermal-only path; night thermal targets at 10 px give mAP50 ≈ 0.55 (AIResQ) | Sky/sun rig; exposure and ISO noise model | 0.0001–100,000 lux | Y (day/dusk/night) |
| 10 | **Motion blur** | Platform speed × exposure ÷ GSD | blur_px = v·t_exp/GSD: 4K at 60 m (2.25 cm/px), 5 m/s, 1/500 s → 0.4 px; 1/60 s at dusk → 3.7 px | Microbolometer integration ~10 ms (unverified) → 0.6 px at 60 m and 5 m/s | Blur > 1 px smears 10–20 px targets → R↓ | Exposure time, platform speed, motion-blur post-process | Exposure 1/60–1/2000 s; speed 2–12 m/s | N, plus one "dusk auto-exposure" slice |
| 11 | **Altitude → GSD → pixels on target** | Flight profile; 120 m legal cap in green zones | 4K wide: 0.45 m target = 40/20/13/10 px at 30/60/90/120 m | 640 × 512 thermal at 60 m: 5.7 px across 0.45 m, 21 px along a prone body | HERIDAL ~60 px targets → R 92.9 %; C2A: <20 px detected less often; AIResQ ~10 px → mAP50 0.55 | Camera altitude AGL; sensor model | 30–120 m | Y (30/60/90/120 m) |
| 12 | **Rolling shutter / vibration** | CMOS line readout; motor vibration | Skew and jello on fast pans | Microbolometers are global but slow; gimbal damps vibration | Small-target geometry distortion; tracking jitter | Per-row time offset; IMU jitter noise on camera pose | Row readout 10–30 ms; jitter 0.05–0.3° (*proposed*) | N |
| 13 | **Gimbal pitch** (nadir vs oblique) | Operator choice; DJI gimbals reach −120°…+45° or more | Oblique reveals people under eaves and trees and shows body height, but stretches the footprint, varies GSD across the frame and grows geolocation error | Oblique water reflects the cold sky → warm heads contrast better; roofs and walls become visible | Nadir: consistent scale, poor occlusion reveal; oblique: better R under overhangs, worse geolocation | Camera pitch | −90° (nadir) to −30° | Y (nadir / −60° / −45°) |
| 14 | **Wind gusts → attitude and position error** | Valley gusts; DJI rated 12 m/s | — | — | Ground error ≈ H·δθ at nadir: 1° at 100 m → 1.75 m; 5° → 8.7 m; at 45° oblique ≈ 2H·δθ; plus GNSS ±1.5 m (GPS) vs ±0.1 m (RTK) | Attitude and GNSS noise on the camera pose | δθ 0.5–5°; GNSS 0.1–1.5 m; wind 0–12 m/s | Y (RTK vs GPS; calm vs gust) |
| 15 | **Background similarity / camouflage** | Mud-coated bodies; brown or khaki clothing; wet dark clothing | Contrast collapse (drowning-detection studies flag dark-clothed victims as the low-contrast case) | Wet mud-coated clothing tends to ambient; skin stays ~33–35 °C (emissivity 0.99) | R↓ in RGB; thermal partially compensates at night | Clothing colour sampler tied to the scene palette; mud-coat decal probability | Clothing–background L\*a\*b distance 5–60; mud coat 0–100 % (*proposed*) | Y (camouflaged vs contrasting) |
| 16 | **Human-like distractors** (mannequins, clothing, bags, logs, carcasses) | Flood carries household goods; drowned livestock | Elongated 1–2 m objects; skin-coloured plastics; clothes on lines | Carcasses cool to ambient within hours; sun-heated objects read hot | P↓; classic SAR false positives: sun-warmed rocks, decomposing vegetation, warm roofs, warm vehicle engines | Distractor spawner with class labels | 0–10 per 100 m² (*proposed*) | Y (distractor-rich) |
| 17 | **Animals** (cattle, buffalo, goats, dogs, wildlife) | Indian floods carry livestock (Assam 2024 bulletin: 35,025 affected, 8,529 washed away; Kaziranga: 174 wild animals dead, 135 rescued) | Quadruped shapes 1–2.5 m; heads above water resemble human heads | Warm-blooded 37–39 °C; same blob size as a person above 60 m | Confusion class → P↓ unless `animal` is trained; the PS *wants* animals in the list | Animal actors with stand/swim/lying animations | 0.2–2 animals per person (*proposed*) | Y |
| 18 | **Crowds / clusters on rooftops** | Whole households on roofs | Overlapping bodies; counting, not detection | Merged warm blob | NMS suppresses neighbours → under-count; needs a count estimate per cluster | Rooftop spawner with cluster sizes | 1–40 per roof (*proposed*) | Y (single / 2–5 / >10) |
| 19 | **Body–water temperature difference vs time** | Immersion cooling, exhaustion, hypothermia | — | Immersed skin → water temperature in minutes; head-only contrast small; after death the body reaches ambient in hours | Thermal recall for immersed people decays with time in water; after death only RGB shape remains | Per-actor thermal state: T_skin(t) = T_water + (33 − T_water)·e^(−t/τ), τ_immersed ≈ 10–20 min, τ_air ≈ 60 min (*proposed*) | Water 15–30 °C; air 20–32 °C | Y (fresh / 6 h / 24 h) |
| 20 | **Thermal crossover at dawn/dusk** | Backgrounds pass through the person's temperature | — | Measured contrast ratio ≈ 1.0 at ~06:50 and ~18:05 for low-inertia surfaces; metal loses contrast for up to 4 h across two cycles; water (high inertia) keeps contrast | Thermal R near zero on roofs and rock in the crossover window | Diurnal surface-temperature model per material (thermal inertia) | Two 1–2 h windows per day | Y (crossover window) |
| 21 | **Wet clothing / evaporative cooling** | Rain, immersion, spray | Darker, clingier clothing | Wet surfaces cool rapidly; wet clothes raise heat loss ~5×; outline softens | Thermal contrast ↓ | Wetness → surface temperature and albedo | 0–100 % | N |
| 22 | **Emissivity and sky reflection** | Material physics | — | Water 0.95–0.98, skin 0.99, wet soil 0.95–1.0, concrete 0.95, cotton 0.8, galvanised steel 0.20–0.30, oxidised steel 0.70–0.95; low-emissivity roofs mirror the cold sky | Apparent-temperature clutter: "hot roof" and "cold roof" both possible → P↓ | Per-material emissivity in the thermal render; sky radiance term | As listed | N |
| 23 | **Sun-heated distractors** (roofs, rocks, asphalt, dry soil) | Solar loading; worst at midday | — | Rock, asphalt and dry soil "can all read as warm as a person"; best window overnight and pre-dawn; overcast often helps | P↓ at midday; thermal cues must be RGB-confirmed | Material solar absorptance + thermal inertia; time of day | Midday background 0–15 K above air (*proposed*) | Y (midday sun vs overcast) |
| 24 | **Time since event → survival probability** | Immersion, burial, trauma, exposure | — | Dead bodies lose thermal signature within hours | Affects triage weight, not the detector (§5.8) | Scenario clock t₀; per-actor alive/dead state from survival curves | 0–72 h | Y (2 h / 12 h / 48 h) |
| 25 | **Canopy and structural occlusion** (trees, eaves, tarpaulins) | Tea slopes, coconut and areca canopy; people shelter under roofs | Partial or full occlusion | Canopy blocks LWIR too | R↓ proportional to occluded fraction; oblique views recover some | Per-actor occlusion fraction from ray casting | 0–100 %; slices at 0/25/50/75 % | Y |
| 26 | **Thermal frame rate and resolution tier** | Export-control tiers: ≤ 9 Hz vs 30 Hz; 640 × 512 vs 1280 × 1024 | — | 9 Hz thermal video is choppy → tracking gaps | Association at 9 vs 30 Hz; dedup quality | Sensor frame rate and resolution parameters | 9 / 30 Hz; 640 × 512 / 1280 × 1024 | Y (9 vs 30 Hz) |
| 27 | **Thermal noise (NETD)** | Sensor quality (DJI ≤ 50 mK) | — | Noise floor hides ΔT below ~0.1 K after atmosphere | Thermal R↓ in humid or crossover conditions | Gaussian + fixed-pattern noise | 30–80 mK | N |

Two practical consequences follow directly from this table:

- **The simulator needs a thermal model with material emissivity and a diurnal temperature state, not just a "hot people, cold world" filter.** Rows 19–23 are where the interesting failures live. §5.1 shows that the AirSim family's infrared camera already works by mapping object IDs to temperatures, which is the right starting point; the per-actor cooling curve and per-material diurnal temperature are small scripts on top of it.
- **The evaluation report is a grid, not a number.** Recall must be reported per slice (zone × altitude × band × time-of-day × occlusion × posture), and the "recall ≥ 90 %" claim must state the slice it holds on. That is also what makes the search-quality map honest: the per-cell probability of detection uses the recall measured on the slice matching that cell's conditions.

### 2.4 Correlated priors (not causal, but useful)

These attributes do not change the pixels; they change *where survivors are likely to be* or *how detectable they are*. They feed the probability-of-area layer of the search-quality map and the ordering of the initial search pattern.

| Attribute | Why it correlates with survivor presence | Use in the prior map |
|---|---|---|
| Proximity to buildings, especially multi-storey | Upper-floor occupants survived a landslide inundation at 95 % while ground-floor occupants were 12× likelier to be killed; Kerala 2018 rooftops held ~30,000 people | Raise the prior on roofs and upper floors of intact or partially intact buildings inside the flooded footprint, weighted by roof area and storeys |
| Trees, elevated roads, embankments, hilltops | Kedarnath 2013 sorties picked people "from rooftops or marooned on hilltops"; embankments are Assam's refuges | Boost any terrain above flood level within ~200 m of inundated homes |
| Vehicles in or near water | 32.8 % of analysed flood deaths were drownings in vehicles | Prior on stranded vehicles (occupants inside or on the roof); also a distractor class |
| Debris density, debris dams, channel bends, recirculation zones | People cling to debris; floating bodies accumulate on the shores of recirculation systems; drift typically 0.5–8 km | Downstream prior along the channel decaying with distance; local maxima at bends, bridges, debris dams and eddies (Wayanad organised its search along 40 km of river) |
| Last-known phone positions and drone-photo GPS | The Wayanad district administration fused residents' last phone locations with drone-derived coordinates to place teams | Seed the prior with last-known-position points, exactly as wilderness SAR does |
| Time of day | Thermal contrast best pre-dawn; crossover ±1 h around sunrise and sunset; midday solar loading | Not a location prior: a *detectability* prior that scales thermal probability of detection |
| Timing of deaths | 87 % of flood deaths occur during the impact phase | After the first hours, survivors are mostly stranded rather than drowning → prioritise elevated refuges over the channel |
| Population at t₀ (night vs day) | Wayanad struck at 01:00–04:00 when people were in homes | Building footprints × night occupancy seed the prior for night events |
| Livestock density | Assam: 8,529 animals washed away in one bulletin | Animal priors co-locate with human refuges (people move animals to high ground) |

### 2.5 Sensor realism numbers the design depends on

The simulator's camera models and the altitude policy are anchored on real DJI enterprise cameras, because the replay path (R12) will eventually run on footage from one of them. Values are computed from the official spec pages (Mavic 3T, Matrice 30T, Zenmuse H20T, Zenmuse H30T).

**Ground sampling distance and pixels on target (nadir; computed from published focal lengths and sensor sizes):**

| Camera (mode) | GSD @ 30 m | @ 60 m | @ 90 m | @ 120 m | Swath @ 60 m | px across 0.45 m @ 30/60/90/120 m | px along 1.7 m @ 30/60/90/120 m |
|---|---|---|---|---|---|---|---|
| M3T / M30T thermal 640 × 512 | 3.96 cm | 7.91 cm | 11.9 cm | 15.8 cm | 50.6 m | 11 / 6 / 4 / 3 | 43 / 22 / 14 / 11 |
| H20T thermal 640 × 512 | 2.67 cm | 5.33 cm | 8.00 cm | 10.7 cm | 34.1 m | 17 / 8 / 6 / 4 | 64 / 32 / 21 / 16 |
| H30T thermal 1280 × 1024 | 1.52 cm | 3.05 cm | 4.57 cm | 6.09 cm | 39.0 m | 30 / 15 / 10 / 7 | 112 / 56 / 37 / 28 |
| M3T / M30T / H30T wide, 4K video | ~1.1 cm | ~2.2 cm | ~3.3 cm | ~4.5 cm | ~85 m | 40 / 20 / 13 / 10 | 153 / 77 / 51 / 38 |
| M3T tele, 4K video (confirmation only) | 0.16 cm | 0.33 cm | 0.49 cm | 0.66 cm | 12.6 m | 273 / 137 / 91 / 68 | — |

**How many pixels a detector needs.** Johnson's criteria (50 % probability for a *human observer*) need ~2 px for detection, ~8 for recognition and ~13 for identification across the critical dimension. Learned detectors need far more: HERIDAL persons at ~60 × 60 px gave 92.9 % recall; C2A found objects under 20 px "detected with less frequency and lower confidence"; AIResQ thermal persons at ~10 px gave mAP50 0.55; TinyPerson (mean size 18 px) is a research benchmark, not an operating point. **Design rule adopted: ≥ 20 px along the critical dimension for "90 % recall plausible"; 8–20 px is cue-only; < 8 px is a blob.**

**Altitude ceilings implied by that rule:**

| Camera (mode) | Ceiling for a compact target (0.45 m ≥ 20 px) | Ceiling for a prone body (1.7 m ≥ 20 px) |
|---|---|---|
| M3T / M30T thermal 640 × 512 | 17 m (impractical) | 64 m |
| H20T thermal 640 × 512 | 25 m | 96 m |
| H30T thermal 1280 × 1024 | 44 m | 120 m (legal cap) |
| RGB 4K wide | ~60 m | 120 m (legal cap) |

Consequences that flow into §5: fly the *detection* pass at 40–60 m AGL with 4K RGB tiled at full resolution; treat the 640 × 512 thermal as a *cueing* sensor above ~40 m and as a confirmation sensor only below it; use 90–120 m only for a coarse thermal sweep with tele or low-pass confirmation. Motion blur constrains speed: keep shutter ≤ GSD / ground speed, which at 60 m and 1/500 s allows ≤ 11 m/s but at 1/60 s (dusk) only ≤ 1.35 m/s.

**Thermal contrast timeline (for the triage weight and for the simulator's thermal state machine):**

| Time since event | Person on a roof at night | Person on a roof at midday | Person immersed to the chest (water 22–28 °C) | Deceased, floating |
|---|---|---|---|---|
| 0–1 h | Skin 33–35 °C vs cooled roof → ΔT 5–10 K; best contrast of the timeline | Roof sheet or rock heated above skin temperature → person colder than background or invisible | Immersed skin reaches water temperature in minutes; only head/shoulders warm; ΔT ≈ ⅓ of dry value | Still near 33–35 °C at the surface, cooling |
| 1–6 h | Clothing surface tends to air temperature; ΔT 3–8 K | After the ~18:05 crossover the background cools faster → contrast returns by ~21:00 | Exhaustion or unconsciousness at 3–12 h in 21–27 °C water; the person may slip lower → head only | Cools ~2 °C in the first hour then ~1 °C/h in air, faster in water → indistinguishable from debris within hours |
| 6–24 h | Pre-dawn is the best window; crossover ~06:50 | Midday false-positive peak | Survival 3 h–indefinite at 21–27 °C; hypothermic skin is cooler → contrast ↓ | Thermal signature gone; RGB shape only; drift 0.5–8 km |
| 24–72 h | Cycle repeats; fewer waving cues | Same | Most immersion survivors are out of the water or dead | May sink and resurface after days |

Practical rule carried into the triage score: a thermal-positive detection is strong evidence of *life* in the first hours and weak evidence after ~24 h; a thermal-negative RGB detection after hours is consistent with either a live hypothermic person or a body, and is never demoted to "cleared".

**Survival-probability decay (for ranking, never for closing):**

| Entrapment class | Evidence | Weight curve adopted |
|---|---|---|
| Fully buried in mud or debris | Avalanche analogue: survival ~30 % within 35 min; Chuuk 2002 debris flows: ~90 % of deaths by suffocation | Not detectable from the air → not modelled in the detector; the coverage map flags "aerial search cannot clear this polygon" |
| Trapped in a collapsed or partly buried structure with air | Most documented live rescues within 5–6 days; Haiti 2010 day-7 rescues; longest 14 days; INSARAG rejects any fixed cut-off | w(t) = 1.0 for t < 48 h, then exponential with half-life ≈ 3 days, floor 0.05 (never zero) |
| Immersed in water | 21–27 °C water: exhaustion 3–12 h, survival 3 h–indefinite; 15–21 °C: exhaustion 2–7 h; flowing water adds trauma | Highest urgency: w = 1.0 with time-to-exhaustion as the rank key; head-only detections go to the top of the list |
| Stranded on a roof, tree or high ground | Kerala 2018 people waited days on roofs; "rule of fours" for water | w(t) = 1.0 for t < 24 h, then slow decay (half-life ~4 days); large clusters and children/elderly raise priority |
| Animals | Assam livestock losses | Separate list; w = 0.3 × human curve (*proposed*) |

### 2.6 India-specific operating constraints

- **Drone Rules 2021** (PIB backgrounder, Jan 2022): green zone up to 120 m AGL without permission (60 m within 8–12 km of an airport); yellow zone needs ATC permission; red zone needs Central Government permission; micro category 250 g–2 kg, small 2–25 kg; Digital Sky registration and No-Permission-No-Takeoff. Hill-valley disaster sites are usually green zone, so the 120 m ceiling in the tables above is also the legal ceiling. The exact disaster-relief exemption clause could not be read (gazette PDF unreachable) and is listed in §10.
- **NDMA/NDRF practice**: NDMA first used drones in the 2013 Uttarakhand floods (4 UAVs, 50 areas); a TEC/DoT 2024 study paper recommends IR thermographic cameras for night landslide work; C-DAC ran a 40-hour drone bootcamp for 42 NDRF personnel in July–August 2026 with the stated concept "UAVs deployed ahead of NDRF personnel to assess disaster sites, locate victims and identify hazards". No NDMA drone-specific SOP was found (unverified whether one exists).
- **Wayanad's own structure to mimic in the demo**: six search zones, 40 teams, riverbank search by eight police stations over 40 km, last-known positions from phones fused with drone-photo GPS. The demo's incident should be laid out the same way: zones as polygons, one assignment per zone, priors seeded from last-known positions.
---

### 2.7 The burial boundary: what the cameras can and cannot see

This is the single most misread line in the design, so it is drawn explicitly here. "Buried" does not mean "occluded". The two are different problems with different answers.

**Partly visible is in scope, and is the point of the project.** A person under a collapsed roof sheet with one arm out, a body wedged in a debris dam with the head and shoulders clear, a survivor prone in silt with the back and one leg exposed, a head above brown water: all of these are detectable, all of them are what the annotation guideline's visible-extent rule and the 0/25/50/75 % occlusion slices exist for, and all of them are trained on. The evidence says this works when the training data contains such cases: the POP thermal dataset found detection accuracy holds until the occluded fraction exceeds about 70 %, *provided* occluded examples were in the training set, while models trained on ordinary pedestrian data lose roughly 0.70 mAP50 when moved to occluded, posed subjects. The system is built for exactly this band.

**Fully buried — no visible part in any band — is out of scope, and it is physics, not a resolution limit.** Long-wave infrared is a *surface* instrument. It measures radiance leaving the top few micrometres of whatever lies on top, and mud, rubble, silt and water are all opaque to it. A better sensor does not help. For a buried person to appear, their body heat must first conduct to the surface and then win against everything else heating and cooling that surface. Both stages fail.

*Stage one, the heat arrives far too late.* Thermal diffusion time scales as depth squared divided by diffusivity, and saturated mud is a poor conductor (order 0.5 × 10⁻⁶ m²/s).

| Burial depth | Order-of-magnitude time for a thermal signal to reach the surface (*computed*) |
|---|---|
| 10 cm | several hours |
| 30 cm | ~2 days |
| 1 m | ~3 weeks |
| 4 m (the Wayanad deposit 4 km downstream) | far beyond any rescue horizon |

These are computed from a standard wet-soil diffusivity, not measured on debris-flow material, and are given as magnitudes rather than predictions. The conclusion does not depend on the exact constant: at the deposit depths recorded at Wayanad (7.3 m flow height, ~4 m deposit at 4 km) no plausible value brings the signal into a rescue timeframe.

*Stage two, the signal loses to clutter even if it arrives.* A person radiates on the order of 100 W, less when injured or hypothermic. Midday solar loading on mud and rock is hundreds of watts per square metre, the diurnal surface swing is 10–15 K, wet mud is evaporatively cooling, and rain is falling. The camera's ≤ 50 mK noise floor is irrelevant when the background varies by kelvins across the frame (§2.3 rows 20–23). A related idea sometimes proposed — detecting *disturbed ground* by its altered thermal inertia, as in forensic grave surveys — also fails here for a structural reason: in a debris flow the entire surface is disturbed, so the anomaly has no contrast to stand against.

**The field evidence agrees.** At Wayanad in 2024 a private thermal-drone survey of Mundakkai reported "zero human presence". The Army's respiration radar, which genuinely does penetrate rubble, reported three breath signals; teams dug three feet and found nothing, "possibly a frog or snake". More than 200 bodies and body parts were recovered from a 40 km stretch of the Chaliyar instead. Both the thermal negative and the radar positive were wrong in the ways their physics predicts.

**What the system does about it.** Four things, none of which is silence:

1. **Marks, never clears.** Cells inside a burial polygon are drawn hatched and labelled *aerial search cannot clear*; their probability of detection is not updated by overflights, so no number of passes can make the map look searched. In the per-presentation coverage model of §5.3b this stops being a special case: fully-buried is simply the presentation class whose coverage layer is identically zero, so the label is derived rather than painted on. A false clearance is worse than no information, because it diverts teams from ground that still holds people.
2. **Keeps them in the ground truth.** The simulator places fully buried actors that appear in the truth as *not visible*. The evaluation checks that the system reports nothing for them and that the coverage map does not claim the polygon cleared. Getting a negative right is a scored behaviour, not an absence of behaviour.
3. **Redirects the search where the evidence points.** The prior map pushes effort downstream along the channel — bends, bridges, debris dams, recirculation zones — because that is where bodies and clinging survivors actually accumulate, and it is where Wayanad's recoveries actually happened (§2.4).
4. **Hands the case to the right sensor.** A record over a burial polygon can carry a "radar check requested" action for a close-in FMCW or respiration-radar pass at 1–2 m standoff (Appendix C), or for cadaver dogs and acoustic listening devices. The result is logged as evidence, never as proof, with the Wayanad false alarm as the standing reason.

**Where thermal does earn its place**, so the limitation is not mistaken for a dismissal: surface-visible survivors at night on rooftops, in trees and on mud margins, with pre-dawn the strongest window because the ground has shed its heat and the person has not. Its own failure modes are catalogued rather than hidden — crossover around 06:50 and 18:05, midday solar loading, and immersed people whose exposed skin reaches water temperature within minutes so contrast falls to roughly one third with the torso submerged (§2.5). Those are evaluation slices, not surprises.

---

## 3. Solution overview

### 3.1 Thesis

A simulated search drone flies coverage patterns over a debris-flow-fed flood valley, hands control to a human on a gamepad at any moment and takes it back, and streams 4K RGB plus thermal video with synchronised telemetry into a vision pipeline that runs on the development PC now and on a Jetson Orin later. The pipeline detects humans and animals under occlusion and abnormal pose with a tiled high-resolution RGB detector, adds a thermal detector through box-level late fusion so the RGB-only path is untouched, tracks and geo-clusters detections so each living being becomes one record, projects each record to latitude/longitude with a published error radius, ranks records by survival urgency with the score components visible, and renders them on an offline map next to a search-quality raster that shows where the search has *not* yet produced a trustworthy negative. Records are written to a local outbox first and synchronised when a link exists. Nothing in the system closes an area; the commander does.

The same code path runs on recorded real drone footage with its embedded telemetry, so simulation results and real-footage results are produced by one pipeline and reported with the same slice grid.

### 3.2 System diagram

```
┌──────────────────────────── SIMULATION (Windows PC, UE 5.8 + Cosys-AirSim) ────────────────────────────┐
│  Flood-valley level (3 zones) · weather/time-of-day API · human/animal actors with pose + thermal state │
│  Drone (SimpleFlight) ◄── Python mission script (coverage pattern, orbit-on-detect, revisit queue)      │
│                       ◄── Xbox gamepad (takeover / hand-back, logged as mode switches)                  │
│  Cameras: 4K RGB · thermal (IR image type / post-process) · instance segmentation (labels only)         │
│  Exports: video frames + telemetry.csv (pose, GPS, attitude, gimbal, intrinsics, weather, t)           │
└───────────────┬────────────────────────────────────────────────────────────┬────────────────────────────┘
                │ live stream (or recorded flight + telemetry: DJI SRT / MAVLink / ULog)   auto-labels ─► dataset
                ▼
┌──────────────────────────────── VISION PIPELINE (RTX 4060 now · Jetson Orin later) ────────────────────┐
│ INGEST: NVDEC decode → GPU frames · telemetry parse → per-frame pose (interp/SLERP) · decimate to k FPS │
│ DETECT: tile 4K → YOLO26s (TensorRT) · thermal 640×512 → YOLO26n · WBF late fusion (RGB-only fallback)  │
│ VERIFY: crop classifier on low-confidence candidates (precision recovery)                               │
│ TRACK:  BoT-SORT + camera-motion compensation (processed-frame units) · confirm after 3 hits            │
│ GEO:    pixel → ray → gimbal/NED → ground plane/DEM → lat/lon + h_acc_m (per-pixel geometry)             │
│ DEDUP:  DBSCAN in geo space (eps = 2×CE90) → one record per living being · count · motion state          │
│ TRIAGE: score = P(living) × w_class(t) × urgency × (1 + 0.1·count) — components stored, never collapsed   │
│ COVER:  per-cell search quality → POD = 1 − e^(−kC) · buried polygons = "aerial search cannot clear"     │
└───────────────┬───────────────────────────────────────────────────────────────┬─────────────────────────┘
                │ SQLite (WAL) record log + evidence thumbnails                  │ outbox (persist-queue)
                ▼                                                               ▼
┌──────────── MAP / C2 (laptop) ────────────┐                 ┌──────── CLOUD (optional, when linked) ────────┐
│ MapLibre GL JS + PMTiles offline basemap  │ ◄── WebSocket   │ FastAPI: same TensorRT/ONNX engine (/detect)   │
│ triage list · evidence · POD raster       │                 │ idempotent record upsert · shared map           │
│ GeoJSON / KML / CoT (ATAK, WinTAK) export │                 └────────────────────────────────────────────────┘
│ EVAL dashboard: recall@0.5, FP/min, dedup │
└───────────────────────────────────────────┘
```

### 3.3 Two operating modes

| Mode | Input | Purpose | What it proves |
|---|---|---|---|
| **Live simulation** | Frames and telemetry pulled from the simulator API at k FPS while the drone flies (autonomous or gamepad) | The demo: autonomy, takeover, detection, map and coverage updating live | End-to-end latency, human-in-the-loop behaviour, the coverage map filling in |
| **Replay harness** (R12) | A recorded flight: MP4 + telemetry (simulator export, DJI `.SRT`, MAVLink `.tlog`/`.bin`, PX4 `.ulg`) | Evaluation on held-out clips, real-footage validation, reproducible numbers | Recall, FP/min, dedup accuracy, geolocation error on the same code path |

Both modes feed the same pipeline object; the only difference is the frame source. The replay harness exposes a per-clip time-offset parameter so video-to-log alignment errors can be measured and corrected.

### 3.4 What is reused and what is built

| Component | Reused (verified status) | Built by the team |
|---|---|---|
| Engine and drone plugin | Unreal Engine 5.8; Cosys-AirSim v3.4.1 (MIT, precompiled Windows plugin, Jul 2026); fallback Project AirSim v1.0.1 (MIT, Sep 2026) | Flood level, actor spawners, thermal material, capture and label scripts |
| Flight control and takeover | SimpleFlight controller with Xbox RC support (in the plugin); optional PX4 SITL v1.18-beta in WSL2 | Mission script (patterns, orbit, revisit), takeover state machine, telemetry exporter |
| Search theory | Koopman/USCG formulas, measured sweep widths (literature) | Per-cell search-quality raster and its calibration |
| Ingest | PyNvVideoCodec (MIT), PyAV, pymavlink, pyulog, DJI SRT format documentation, dji-log-parser (MIT) | 30-line SRT parser, alignment and decimation logic |
| Detection | Ultralytics YOLO26 (AGPL-3.0) or RF-DETR (Apache-2.0); SAHI/supervision tiling (MIT); Weighted Boxes Fusion (MIT); ProbEn score rule (Apache-2.0) | Training recipes, tiler, fusion weights, verifier |
| Tracking / dedup | Ultralytics BoT-SORT/TrackTrack or roboflow/trackers (Apache-2.0); scikit-learn DBSCAN; py-motmetrics (MIT) | Static-target tuning, geo-clustering, record merge rules |
| Geolocation | Theta-Limited DroneModels.json (Apache-2.0); Copernicus DEM GLO-30 via dem-stitcher (Apache-2.0); pyproj, rasterio | 80-line projection chain, noise injection, error-budget reporting |
| Output / map | MapLibre GL JS (BSD-3), PMTiles/Protomaps, FastAPI, simplekml, geopandas/pyogrio, pytak | Record schema, map layers, exports, evaluation dashboard |
| Offline / cloud | persist-queue SQLiteAckQueue (BSD-3); FastAPI | Outbox, idempotent upsert, fallback policy |
| Edge | TensorRT via Ultralytics export; JetPack 6.2.x; GStreamer/DeepStream on Jetson | Engine builds, batching, benchmarks |
| Data | NOMAD, POP, WiSARD, SARD, AIResQ, SeaDronesSee, AFO, HIT-UAV, C2A, BIRDSAI, livestock sets (§6) | Synthetic dataset, annotation guideline, auto-label loop, slices |

The full ledger with URLs, stars, dates and licences is Appendix A.

---

## 4. Hardware envelope and feasibility

**The binding constraint is 16 GB of system RAM, not the 8 GB of VRAM.** Epic's UE 5.8 specification recommends 32 GB RAM and 8 GB+ VRAM; the editor with the Cosys Blocks project open is expected to use 8–11 GB (measure on day 1). Consequences and mitigations:

| Concern | What to do | Evidence / status |
|---|---|---|
| UE 5.8 editor memory | Dynamic GI = None (no Lumen), reflections = screen space, shadow maps not virtual shadow maps, Nanite off on imported meshes, `r.Streaming.PoolSize ~2500`, scalability groups at 1–2, TSR/TAA. Keep the area of operations ≤ 2 × 2 km. Use a **packaged build** with `ViewMode: NoDisplay` for data runs. | UE 5.8 hardware and scalability docs; Lumen needs RTX 2000+ but 8 GB VRAM with 4K captures, water and foliage will not sustain it |
| Running the simulator and training at once | Do not. Sequence: capture data (engine running) → close engine → train overnight. Run the vision pipeline against the *packaged* simulator, not the editor, during demos. | Engineering judgement from the memory figures above |
| 4K capture throughput from the simulator | `simGetImages` at 3840 × 2160 uncompressed is 24.9 MB per frame; measure FPS; fall back to 1080p video recording plus 4K stills for the dataset if throughput stalls. | Day-1 test |
| Training on 8 GB VRAM | YOLO26n/s at 1024 with batch 8 and AMP fits (estimate 5–7 GB); YOLO26m at 1024 batch 4 is borderline; P2 head halves batch; RF-DETR Nano/Small at 384–512 fits per its docs. | §5.5 |
| Inference on the RTX 4060 | Six 1280 tiles of YOLO26m FP16 ≈ 113 ms detect, ≈ 130–145 ms end-to-end: comfortably inside 300 ms; the 4060 is an upper bound for Orin, not a proxy. | §5.11 |
| Jetson Orin target | Orin NX 16 GB: six 1280 tiles of YOLO26s INT8 at 5 processed FPS (≈ 170–200 ms). Orin Nano Super: coarse pass plus ROI re-detect, or YOLO26n tiles at 5 FPS. Build engines on the device with the JetPack-supplied TensorRT. | §5.11; Ultralytics Jetson tables |
| Rejected on hardware grounds | NVIDIA Isaac Sim 5.1 (Pegasus, OmniDrones): minimum RTX 4080, 16 GB VRAM, 32 GB RAM. Flightmare, FlightGoggles, RotorS, Aerial Gym: Linux-only or stale. AirGen/GRID and Duality Falcon free tiers: cloud sessions only. | Simulation research, verified spec pages |
| Disk | UE 5.8 + VS 2026 + plugin + assets ≈ 60–80 GB; datasets 20–60 GB (WiSARD alone is 40 GB); keep datasets on the SSD. | Estimate |

**Feasibility verdict.** Every recommended component runs on the stated machine with the settings above. The two genuine risks are toolchain freshness (Cosys-AirSim's UE 5.8 + Visual Studio 2026 build is weeks old; Project AirSim on UE 5.7 + VS 2022 is the tested fallback and both engines can coexist) and wall-clock time for training (mitigated by overnight runs and 30–50-epoch fine-tunes).

---

## 5. The stack, component by component

Each component below states its role, the chosen tool with verified status, the alternatives that were rejected and why, the integration surface, and the technical, operational, environmental, integration and feasibility factors that shaped the choice.

### 5.1 Simulation and synthetic data (A13, R11)

**Role.** Provide the drone, the sensors, the scene, the disaster, the weather, the ground truth and the telemetry, on the Windows PC, with autonomous flight and gamepad takeover.

**Chosen: Cosys-AirSim v3.4.1 on Unreal Engine 5.8 (Windows 11, Visual Studio 2026), SimpleFlight controller, Python client (`pip install cosysairsim`), Xbox controller through the plugin's RC block.** Verified on 10 Sep 2026: 420 stars, MIT, last push 13 Aug 2026, release `5.8-v3.4.1` (17 Jul 2026) with a precompiled Windows plugin (705 MB), a packaged Blocks executable and the Blocks editor project. The README carries an "as is, will not be actively updated" disclaimer yet the lab ships UE 5.8 builds and merged pull requests in August 2026.

Why this one, feature by feature (all read from the repo docs and source, not from memory):

| Need | Cosys-AirSim feature | Note |
|---|---|---|
| RGB 4K, thermal, labels from one camera pose | Image types Scene=0, DepthPerspective=2, **Segmentation=5**, **Infrared=7**, **Annotation=11**; per-camera `CaptureSettings` (width/height/FOV, auto-exposure, motion-blur amount) | Infrared and Scene captures of the same camera are pixel-aligned, so thermal labels reuse the RGB masks |
| Per-instance labels for partially visible people | **Instance segmentation with 2,744,000 unique colours** (v3.4 supports skeletal Nanite meshes); `segmentation_generate_list.py` dumps object → colour | Landscape, foliage and brush get a default colour; use static-mesh terrain or accept them as background |
| Whole-body (amodal) boxes | Detection API: `simAddDetectionFilterMeshName(cam, type, "Human*")`, `simGetDetections()` → 2D box (projection of the actor's 3D bounds) + 3D box + geo point; visibility by line traces to the box corners and 10 interior points | Boxes are amodal and can flicker (upstream PR #3472); visible boxes come from masks, see below |
| Visible fraction | `MarkedIgnore` actor tag + `IgnoreMarked` camera setting: a second segmentation camera ignores tagged water/debris/foliage and renders the person unoccluded | Day-1 test |
| Weather and light | `simEnableWeather`, `simSetWeatherParameter` with Rain, Snow, Dust, Fog (0–1); `simSetTimeOfDay(...)` (needs a sky sphere and directional light in the level); `simSetWind`; spawnable point lights | Road-wetness effects need the WeatherFX materials; skip |
| Telemetry | `getMultirotorState()` (kinematics, attitude, GPS lat/lon/alt derived from `OriginGeopoint`, nanosecond timestamp); built-in Recording → `airsim_rec.txt` (timestamp, NED position, quaternion, image file) | Set `OriginGeopoint` to the Wayanad valley so exported coordinates are real |
| Gamepad | `remote_control.md`: Xbox controller supported for SimpleFlight via `"RC": {"RemoteControlID": 0}`; API allowed when `AllowAPIAlways` (default true) | Takeover semantics in §5.2 |
| PX4 | `px4_sitl_wsl2.md`: SITL in WSL2, TCP 4560 lockstep, `ControlIp: "remote"` | Optional |
| Domain randomisation | Dynamic Objects (AI humans walking between waypoints, seeded), runtime texture swapping | Useful for the dry margins and clothing colours |
| Deterministic capture | `ClockType: SteppableClock`, `simPause`/`simContinueForTime` | Multi-image captures at one pose |

**Fallback 1: Project AirSim v1.0.1 on UE 5.7 + VS 2022** (IAMAI; 836 stars, MIT, weekly releases through 7 Sep 2026; prebuilt Windows environments and a UE 5.7 plugin). It has the best maintenance cadence, built-in 2D/3D box annotations in every image message, a data-collection module that sweeps weather and time and writes COCO, an Xbox script (`xbox_rc.py`, X button toggles API control), and PX4/ArduPilot SITL. It lacks a thermal image type, has only 256 segmentation IDs, its 2D boxes have no occlusion test (checked in `UnrealCamera.cpp`), and it has no server-side recording. Switch to it if the Cosys plugin does not load on the installed UE 5.8 point release. Both engines and both Visual Studio versions can coexist on one PC.

**Fallback 2 (panic): plain UE 5.8** with a Blueprint drone pawn, SceneCapture2D for RGB/segmentation/thermal, Enhanced Input for the gamepad and CSV telemetry from a Blueprint tick. Loses PX4 and the API ecosystem; zero third-party build risk.

**Rejected.** Colosseum: **archived 11 Jul 2026**; main targets UE 5.6 only; UE 5.7.1 build broken (issue #143, unresolved); no prebuilt Windows binaries. microsoft/AirSim: not archived but frozen at v1.8.1 (2022), UE 4.27; its docs remain the canonical description of the API Cosys keeps. Unity path: Unity Perception is officially discontinued (last release 2022), AirSim-Unity needs Unity 2019.3.12, and no maintained Unity drone simulator with PX4/MAVLink surfaced; Unity 6.3 LTS with HDRP Water and Cesium for Unity exists if someone insists, but the labelling and sensor stack would be built from scratch. Isaac Sim/Pegasus/OmniDrones: hardware floor above this PC. Flightmare, FlightGoggles, RotorS, Aerial Gym: Linux and stale. Gazebo: Windows port experimental, low photorealism. Webots R2025a: runs anywhere and has a Mavic 2 Pro model, but a 400 × 240 camera and no thermal, water or labels; keep as a one-day panic fallback only. PteroSim: free, Windows binaries, PX4/ArduPilot SITL, but its Python API returns a BGR camera only (no depth, segmentation, thermal, weather or boxes). AirGen/GRID and Duality Falcon: cloud sessions. MATLAB UAV Toolbox: paid, and MathWorks states no native thermal in Unreal.

**Building the flood scene (concrete).**

1. Start from the Cosys **Blocks editor project** (guaranteed plugin compatibility) and add a new level with the low-memory settings from §4.
2. Terrain: a small Landscape with Edit Layers enabled, sculpted as a valley with a river channel and a deposit fan; or **Cesium for Unreal v2.29.1** (Apache-2.0, supports UE 5.6–5.8, free Community ion tier with a 15 GB/month streaming cap) streaming Cesium World Terrain plus imagery georeferenced on the Mundakkai–Chooralmala valley for real relief. Cesium tiles are static meshes, so water cannot carve them; use a custom water body or plane there.
3. Water: fastest is a large single-layer-water plane with brown absorption colour, panning normal maps and foam decals, driven by a Blueprint timeline that raises Z over time and exposes a `FloodLevel` variable settable from Python; better-looking is the UE Water System (Lake/Ocean body + Water Zone) with Buoyancy Components on debris. Set `r.Water.EnableUnderwaterPostProcess 0` and raise `r.Water.WaterSplineResampleMaxDistance` to 400. The Water plugin has no documented runtime level API; moving the actor in Z is the day-1 test.
4. Debris, mud, collapsed houses: Chaos-fracture a few house meshes, simulate once and freeze; mud as a landscape layer with wet, dark albedo and silt-line decals; a debris spawner Blueprint seeded from Python for reproducible layouts. Free assets: Fab's free filter, already-claimed Quixel packs, CC-licensed Sketchfab items (Megascans have been paid on Fab since 1 Jan 2025; items claimed earlier remain usable).
5. Weather and light: Rain, Dust, Fog parameters through the API; an Exponential Height Fog actor for visibility categories; the time-of-day API for dawn, dusk and night; spawned point lights for torches and vehicle lights.
6. Humans and animals: Mixamo characters and animations (free with an Adobe ID as of Sep 2026; clips for standing, waving, prone, supine, crawling, sitting, treading water); each as its own actor tagged `Human_<id>` / `Animal_<id>`; MetaHuman Creator (in-engine since UE 5.6; licence page login-walled, unverified) only for two or three hero shots at low LOD; UE mannequins as the cheap fallback. Half-submerged: pelvis 20–40 cm below `FloodLevel`; the water surface must write depth/stencil so submerged pixels are hidden in the label passes (single-layer water is opaque-pass and should; a translucent plane is not, in which case enable Render CustomDepth or use an opaque proxy for the label pass). Prone and occluded: spawn under debris and canopy clusters; the mask captures the occlusion naturally. Buried: actors placed fully under the deposit mesh, present in the ground truth as *not visible*.
7. Thermal: day 1 uses the built-in Infrared image type, which maps each object's segmentation ID to a digital count computed by integrating Planck radiance over 8–14 µm times emissivity (the `create_ir_segmentation_map.py` recipe); write the temperature/emissivity table for skin (305–310 K, 0.99), wet clothing, water (0.96–0.98), mud, concrete, galvanised sheet (0.20–0.30), rusted sheet, vegetation. It is per-object uniform. Day 2+ upgrades to a **post-process material**: every material carries temperature and emissivity parameters; a dedicated capture converts (T, ε) to band radiance, applies automatic gain, Gaussian blur, fixed-pattern and temporal noise and distance attenuation; humans get limb gradients from a vertex-colour texture; a per-actor cooling state implements the immersion curve of §2.3 row 19; a per-material diurnal temperature implements crossover and solar loading (rows 20–23). The survey literature confirms this "temperature → radiance → render" family is the standard game-engine approach and that purely synthetic thermal underperforms hybrid training, so the thermal detector is validated against WiSARD and HIT-UAV frames before it is trusted.
8. Capture plan (Python): for each seed, randomise poses, positions, debris layout, flood level, weather grid, time of day (5 settings), altitude 15–60 m (plus 90/120 m cueing passes), gimbal pitch −45° to −90°; fly a lawnmower with `moveOnPathAsync`; at each waypoint `simPause(True)` → `simGetImages([Scene 4K, Segmentation, Infrared, DepthPerspective])` + `simGetDetections` + `getMultirotorState` → `simPause(False)`. Video: record continuously at 1080p/30; 4K stills at waypoints unless 4K video throughput proves adequate.

**Auto-labels for partially visible subjects.**

- **Visible box (the training and evaluation box):** per frame, map each unique instance colour to its actor, keep `Human*`/`Animal*`, run connected components per colour, take the tight box of visible pixels, drop components below N px. This is exactly a visible-extent box, robust to water, debris, prone poses and frame-edge truncation.
- **Amodal box (stored as an attribute):** `simGetDetections` 2D box.
- **Visible fraction:** ratio of visible mask pixels to the unoccluded mask from the `IgnoreMarked` camera; fallback approximation = visible-box area / amodal-box area; store `occlusion` in VisDrone bins (0: 0 %, 1: 1–50 %, 2: > 50 % hidden).
- **Pose attribute:** from the assigned animation state, in the C2A vocabulary (upright, bent, kneeling, sitting, lying) extended with `half_submerged` and `trapped`, so C2A can be mixed in.
- **Submersion attribute:** from actor pelvis height vs `FloodLevel`.
- **Thermal labels:** reuse the RGB masks (same camera pose).
- **Telemetry per frame:** timestamp, NED position and quaternion, lat/lon/alt, Euler attitude, camera intrinsics (W, H, FOV), gimbal pitch, weather vector, sun time, flood level, mode (AUTO/MANUAL). CSV plus per-frame JSON; the CSV timestamp base equals the video frame index for later synchronisation. Noise injection (§5.7) is applied in a separate pass so the clean truth is kept.

**Sim-to-real evidence that this plan rests on.** Archangel-Synthetic plus only 50 real images lifted SARD AP from 17.7 to 45.8 and HERIDAL from 12.1 to 36.9, and the same study found **pose variety in the synthetic pool is the dominant factor** while altitude/angle sweeps matter less. C2A (10,215 composited images, 360k humans in five poses over flood and rubble backgrounds) reached mAP50 0.893 and shows the pose taxonomy works. Maritime synthetic sets (SynBASe, DGTA-SeaDronesSee) improved over real-only training but were "still not competitive" alone. Consensus: synthetic alone underperforms; synthetic pre-training plus a few hundred real images, or a 60:40 real:synthetic mix, is the working recipe; pose diversity and small-object realism are the levers.

**Factors.**

| Factor class | What matters |
|---|---|
| Technical | UE 5.8 + VS 2026 toolchain freshness; instance segmentation vs water depth writes; 4K capture throughput; thermal model fidelity (per-object uniform vs per-material) |
| Operational | Packaged build for demos; seeded, reproducible layouts; a `settings.json` per scenario |
| Environmental | Weather, time of day, flood level, turbidity and debris density are all scriptable parameters, which is what makes §2.3's randomisation ranges executable |
| Integration | The Python client is the single integration surface for mission control, capture, labels and telemetry; the vision pipeline consumes exported video plus CSV exactly as it consumes real footage |
| Feasibility | Runs on the PC with Lumen/Nanite off; two fallbacks exist; the day-1 tests in §10 decide the path by midday |

### 5.2 Flight control, autonomy and takeover (A13)

*Note on sourcing: the dedicated control-stack research run was cut off by the session limit before it produced a report. The facts below about SimpleFlight, the RC block, PX4-on-WSL2 and the Project AirSim Xbox script were verified by the simulation research on 10 Sep 2026; the MAVSDK, QGroundControl, MAVLink and ROS 2 details are from the standing PX4/MAVLink documentation and were not re-fetched today, and are marked as such.*

**Chosen: SimpleFlight (the plugin's built-in controller) driven by the Python API for autonomy, with the Xbox controller mapped through the plugin's RC block for takeover. PX4 SITL in WSL2 is optional and is attempted only if the demo narrative needs QGroundControl and MAVLink.**

Why: SimpleFlight plus the API gives waypoint autonomy, velocity control and gamepad takeover with zero networking, no second operating system and no lockstep clock, and it is what every AirSim tutorial assumes. PX4 buys realism (real flight modes, failsafes, MAVLink telemetry a real ground station can read) at the cost of WSL2, a TCP 4560 lockstep link, restarting PX4 every time Unreal stops, and known port-binding confusion (verified in the Cosys docs and upstream issues). ArduPilot has archived its AirSim page and now points UE users to PteroSim, so ArduPilot with AirSim is unsupported.

**Takeover state machine.**

```
          ┌──────────── stick deflection > deadband or TAKEOVER button ────────────┐
          ▼                                                                       │
   ┌─────────────┐   RESUME button (and sticks centred ≥ 1 s)   ┌───────────────┐ │
   │   MANUAL    │ ─────────────────────────────────────────────►│  AUTO-RESUME  │ │
   │ (gamepad)   │                                               │ re-plan from   │ │
   └─────────────┘                                               │ current pose;  │ │
          ▲                                                      │ continue the   │ │
          │ any time                                             │ pattern        │ │
   ┌─────────────┐   pattern complete / low battery / geofence   └──────┬────────┘ │
   │    AUTO     │ ◄───────────────────────────────────────────────────┘          │
   │ (mission)   │ ── HOLD button ──► HOVER (position hold) ── RESUME ──► AUTO     │
   └─────────────┘ ── RTL button  ──► RETURN-TO-LAUNCH (always available) ─────────┘
```

Implementation on SimpleFlight: keep `AllowAPIAlways: true`; the mission loop polls the gamepad at ~50 Hz (`pygame` joystick or the `inputs` library on Windows; both read XInput controllers); on deflection above the deadband it calls `enableApiControl(False)`, after which SimpleFlight obeys the RC channels directly; on RESUME it calls `enableApiControl(True)`, re-plans the remaining pattern from the current pose, and continues. Every transition is written into the telemetry CSV with a timestamp, so the coverage map can attribute passes to AUTO or MANUAL and the evaluation can slice by mode. Day-1 tests: the controller index (`RemoteControlID`), handover latency, whether `moveByRC` is needed instead of raw RC pass-through, and stick deadband tuning.

Implementation on PX4 (optional path): QGroundControl's Joystick tab reads the Xbox controller through SDL and sends MAVLink `MANUAL_CONTROL`; PX4 parameter `COM_RC_IN_MODE` selects joystick input; flight modes Position/Altitude/Manual for the pilot, Hold and Return for HOLD/RTL, Offboard or mission mode for autonomy; the mission itself is uploaded and monitored with MAVSDK-Python (mission and offboard plugins) or pymavlink. *(PX4/MAVLink documentation, not re-fetched today.)* The camera and label stream still comes from the AirSim client exactly as in the SimpleFlight path; only the flight stack changes.

**Autonomy behaviours (all in the mission script).**

1. **Coverage pattern** over each search segment polygon: boustrophedon (lawnmower) lines spaced by the camera swath at the chosen altitude with 20–30 % side overlap, flown at a ground speed that keeps motion blur under one pixel (§2.5), gimbal nadir for the detection pass. Expanding-square and sector patterns for a last-known-position start. Footprint from FOV and altitude (Appendix B).
2. **Orbit-on-detection:** when a candidate reaches confirmation, the drone can (operator-approved, or automatically in AUTO if enabled) pause the pattern, fly to put the target near the frame centre at nadir, dwell for N seconds to collect observations (this is what shrinks the random part of the geolocation error and produces the best evidence thumbnail), then resume. Never mandatory; the commander can disable it.
3. **Revisit queue:** records marked `stale` and cells whose search quality is below a threshold are queued as revisit waypoints after the pattern completes.
4. **Constraints:** simulated battery budget with return-to-launch reserve, a geofence around the area of operations, the 120 m ceiling, and a no-fly buffer around the buried-polygon areas only if the commander sets one (not automatic).

**Rejected or deferred.** ROS 2 on Windows: binaries exist for recent releases, but nothing in this system needs a ROS graph, and it adds a heavy install and a second message layer; plain Python with the AirSim RPC client and a WebSocket to the map is enough. DroneKit: long unmaintained (not re-verified today); pymavlink and MAVSDK are the live libraries. Coverage-path-planning libraries such as Fields2Cover were not verified today; the boustrophedon generator is ~60 lines and is written by the team.

**Factors.** Technical: handover latency and deadband, re-planning from an arbitrary pose, pattern spacing tied to swath and overlap. Operational: HOLD and RTL always one button away; every mode switch logged. Environmental: wind (simulated `simSetWind`) affects track-keeping and blur; the pattern speed adapts. Integration: the mission script is the only writer of the drone state and the only reader of the gamepad, which keeps the takeover deterministic. Feasibility: SimpleFlight path has no external dependency; the PX4 path is time-boxed to two hours and dropped if it fights back.

### 5.3 Search planning and the search-quality map (A14, R10)

**Role.** Decide where to fly first, and tell the commander how much each place has been searched, in units that mean something.

**Probability of area (where to look first).** A raster prior over the area of operations built from §2.4: rooftops and upper floors of buildings inside the flood footprint (weighted by roof area and storeys), trees and terrain above flood level within ~200 m of inundated homes, stranded vehicles, the channel downstream with local maxima at bends, bridges, debris dams and recirculation zones, and last-known-position points from phones or reports. Segments (polygons) are drawn over the prior, Wayanad-style, and each becomes one assignment.

**Probability of detection (how well a place has been searched).** Search theory, as used by the US Coast Guard and land SAR, defines an effective sweep width W, a coverage C = effort × W / area, and a probability of detection that rises with coverage: the random-search lower bound is POD = 1 − e^(−C), and the empirical USCG fit is POD = 1 − e^(−1.3C). Measured sweep widths for adult-sized subjects with ground searchers in temperate forest are 64 m by day and 22 m at night, and searchers were unable to self-evaluate their own POD, which is the whole argument for computing it instead of asking.

For a drone, W is not a constant: it depends on the same attributes that §2.3 lists. The system therefore computes a **per-cell, per-pass search quality**:

```
q_pass(cell) = R_slice(GSD, band, time_of_day, blur, view_angle) × V(cell)
```

where `R_slice` is the detector recall *measured on the evaluation slice matching those conditions* (§5.12), and `V(cell)` is the visibility factor for that cell (1.0 open ground; lower under canopy or structures, from the terrain/land-cover layer; 0 for buried polygons). The cumulative coverage is `C(cell) = Σ_passes q_pass(cell)`, and the displayed value is `POD(cell) = 1 − e^(−k·C(cell))` with k calibrated on the held-out clip by a reliability diagram (§5.12) and clamped so POD never reaches 1.

**Bayesian update and the guardrail.** After an unsuccessful pass over a segment its probability shrinks by (1 − POD) and the prior is renormalised. Because POD < 1 always, no segment ever reaches zero; that is the formal basis for "the system recommends, never closes", reinforced by INSARAG's rejection of fixed cut-off times and by the Wayanad respiration-radar false alarm. Cells under deep-burial polygons are drawn hatched with the label *aerial search cannot clear*, and their POD is not updated by flights.

**Outputs.** A GeoTIFF/PNG raster overlay (POD, and the underlying coverage count) and a vector layer of segments with an ICS-204-style assignment record (segment, priority, assigned asset, passes flown, current POD, recommended next action). The recommendation text is generated from the numbers ("Segment B: POD 0.41 after two passes at 60 m; night thermal pass recommended before 06:00 to beat crossover"), and the commander accepts, edits or ignores it.

**Factors.** Technical: calibration of k and of `R_slice` per condition; raster resolution (5–10 m cells). Operational: the map must be readable at a glance (POD colour ramp, hatched no-clear polygons). Environmental: time of day and weather change `R_slice` between passes, so the same flight line at noon and at 05:30 contributes different quality. Integration: uses the geolocation footprint per frame (which cells were in view, at what GSD) and the detector's slice table. Feasibility: ~200 lines of numpy/rasterio.

### 5.3a The decision planner: turning the coverage map into flight decisions

Everything in §5.3 so far describes a *map*. This subsection makes it a *planner*, because the same two quantities that colour the map — probability of area and probability of detection — are exactly the objective function a search planner maximises. Without this, the drone flies generated geometry; with it, the drone chooses.

**What already exists, checked on 10 Sep 2026.**

| Name | URL | What it is | Status | Verdict |
|---|---|---|---|---|
| **Fields2Cover** | github.com/Fields2Cover/Fields2Cover | Modular coverage-path-planning library: headland generator, swath generator, route planner, path planner, for polygons with vehicle constraints | 887★, **BSD-3-Clause**, pushed 4 Sep 2026, C++ with Python bindings | **ADOPT** for the geometric layer (swaths and route order inside a segment); it is the maintained, permissive option and saves writing a boustrophedon generator plus turn planning |
| ethz-asl/polygon_coverage_planning | github.com/ethz-asl/polygon_coverage_planning | Exact cellular decomposition for polygons with holes | 656★, **GPL-3.0**, last push 13 Nov 2023, ROS 1 | REFERENCE (licence, staleness, ROS 1) |
| **Koopman optimal allocation** | Koopman 1953; Stone, *Theory of Optimal Search*, 1975 | Closed-form optimal distribution of a fixed search effort across cells under an exponential detection function | Classical, no code needed | **ADOPT the formula** (below) |
| **SAROPS** (US Coast Guard) | dco.uscg.mil SAROPS fact sheet; Kratzke & Stone | The operational maritime search planner since 2007: Monte-Carlo particle filter over target-location hypotheses plus environmental data, driving search-unit allocation; successor to CASP (1974) | Not open source | **REFERENCE** — the doctrine this planner imitates in miniature. Our prior raster is the particle cloud's poor cousin; the allocation step is the same idea |
| Slope Probability Search | *Sensors* 2025, Wang et al., PMC12787386 | Modified A* over a dynamic probability map that links terrain slope to lost-person behaviour, with searched areas decaying then recovering over time | Peer-reviewed, **no code released** | **ADAPT the idea.** Reported 88.9 % success versus lawnmower 75.4 %, spiral 68.4 %, traditional A* 50.5 %, random 45.6 % — direct evidence that a prior-driven planner beats a pattern |
| capstone-insper/drone-swarm-search | github.com/capstone-insper/drone-swarm-search | PettingZoo reinforcement-learning environment for SAR search patterns, no vision | 81★, MIT, pushed 31 Aug 2026 | REFERENCE |
| dmar-bonn/ipp-rl-3d | github.com/dmar-bonn/ipp-rl-3d | RA-L 2024 deep RL for adaptive informative path planning, **with a released model checkpoint** | 34★, MIT, pushed 11 Jul 2024 | REFERENCE — the checkpoint is trained for 3-D monitoring with a different sensor and reward, so it does not transfer; read it for the graph-based state design |
| uzh-rpg/agile_flight | github.com/uzh-rpg/agile_flight | Vision-based agile flight and racing benchmark | 196★, MIT, last push 23 May 2022 | **REJECT** — solves high-speed obstacle avoidance, not search allocation |

**Finding on pre-trained "autopilot" weights: there are none worth using.** The published learned-flight policies are for agile racing and obstacle avoidance at speed, a different problem from a 45–60 m survey pattern, and the one informative-path-planning checkpoint that exists is trained on a different sensor model and reward. The *control* problem is already solved by SimpleFlight or PX4, which are real cascaded controllers, not scripts. What is missing is the layer above them, and search theory answers it analytically and inspectably — which is worth more here than a learned policy nobody can interrogate.

**Layer 1 — strategic allocation (closed form).** Given a total flight-time budget and the prior `p_i` over cells, Koopman's result for an exponential detection function gives the optimal coverage directly:

    c_i* = max( ln p_i − λ , 0 ),   with λ chosen so that Σ c_i* = total effort available

This is a water-filling solution: cells whose prior falls below `e^λ` receive **zero** effort, and the rest receive effort proportional to the log of their prior. It runs in milliseconds by bisection on λ, and it tells the commander how many minutes each segment deserves before anyone takes off.

**Layer 2 — tactical scoring (greedy, replanned every 30–60 s).** Allocation says how much; this says what to do next, accounting for transit and for conditions that change during the flight. Score each candidate action and take the best:

    value(a) = [ Σ_cells POA(cell) · ΔPOD(cell, a) ] / ( t_transit(a) + t_execute(a) )

with diminishing returns built into the increment, so a repeat pass in identical conditions scores near zero:

    ΔPOD(cell, a) = e^(−k·C_before) − e^(−k·C_after),   C_after = C_before + q_pass(cell, a)

and `q_pass` is the measured-recall term from §5.3, not a constant sweep width. §5.3b splits this increment by presentation class, which is what makes the planner propose a low confirmation pass on its own.

**Candidate action set** (finite, enumerable, ~20 items per decision): continue the current swath; jump to segment *k*; change altitude band (30 / 45 / 60 / 90 / 120 m); change band (RGB, thermal, both); orbit and dwell on candidate record *r*; loiter until time *T* (the pre-dawn thermal window); return to launch. Hard constraints prune the set before scoring: battery reserve for return, geofence, the 120 m ceiling, and any operator no-go area.

**Behaviours this produces that nobody coded.**

1. It **abandons a half-finished pattern** when a detection elsewhere raises that segment's prior enough to repay the transit time.
2. It **treats altitude as a trade, not a setting**: lower means higher `q_pass` per cell but fewer cells per minute, and the ratio in `value(a)` weighs them.
3. It **schedules the thermal pass for pre-dawn on its own**, because `q_pass` carries the time-of-day term from §2.3 rows 19–23, so a loiter-until-05:00 action can outscore flying now.
4. It **declines to re-fly a cell** in identical conditions, because ΔPOD ≈ 0, and it can say so numerically.

That last one is the demonstration moment: a drone that refuses a task and justifies the refusal is recognisably autonomous in a way a flawless lawnmower is not.

**Graceful degradation, which is the safety property.** When the prior is flat and no cell has been searched, every cell has equal `p_i` and equal ΔPOD, so the argmax reduces to "cover the nearest unsearched area efficiently" — that is, the boustrophedon pattern. The planner does not have a separate fallback mode; **the pattern is what the planner outputs when it has no information to be clever with.** If the planner is disabled entirely, §5.2's pattern generator still flies the mission.

**Explainability and the guardrail.** Every decision logs its top three candidates with scores, so the timeline shows "chose segment B at 0.031 expected finds per minute over continuing swath 7 at 0.004". The commander can pin an action, veto one, or take the gamepad, and any override is logged and attributed in the coverage map. The planner proposes; it never closes a segment, and cells inside burial polygons contribute zero ΔPOD by construction (§2.7), so it will never waste effort pretending to clear them.

**Build cost and placement.** Layer 1 is roughly 40 lines, Layer 2 roughly 150, plus Fields2Cover for swath geometry. It depends entirely on the coverage map, so it is scheduled after it (day 5–6, stretch), and the system is fully demonstrable without it.

**Factors.** Technical: replan cadence versus flight smoothness; the myopic horizon (a greedy scorer can be led away by a rich neighbour, which the allocation layer bounds). Operational: every decision explainable in one line; a pin-and-veto control. Environmental: `q_pass` changes with weather and light between passes, which is what makes loiter-until-dawn a rational action. Integration: reads the prior raster, the coverage raster and the slice table; writes waypoints to the same mission executor as §5.2. Feasibility: closed-form allocation plus a bounded greedy search over ~20 actions is milliseconds per decision on the flight laptop.

### 5.3b Per-presentation coverage: the map must say *searched for what*

**Why a single probability of detection is dishonest.** §5.3 computes one number per cell: if a person were here, would we have seen them. But "a person" is not one target. Seen from nadir, the critical dimension of a human ranges from 1.7 m for a prone body down to about 0.18 m for a hand protruding from mud — a factor of seventeen. A cell flown once at 60 m is thoroughly searched for a body lying on a roof and barely searched at all for a partly buried casualty, and a map that reports one number for both is telling the commander something false about the second case.

**The presentation classes.** These are not new; they are the `pose` and `submersion` values the annotation guideline already records (§6.3) and the verifier head already predicts (§5.5a).

| Presentation | Critical dimension | Where it dominates |
|---|---|---|
| Prone or supine body | 1.70 m | Deposit fan, rooftops, mud margins |
| Cluster of three or more | ~1.5 m | Rooftops, elevated roads |
| Upright, sitting or crouched | 0.45 m | Rooftops, upper floors, high ground |
| Wading or clinging, torso above water | 0.45 m | Channel, flooded streets |
| Head and shoulders only | 0.25 m | Fast channel, deep water |
| Limb only (hand, foot, forearm) | 0.18 m | Deposit fan, debris dams, collapsed structures |
| Fully buried | 0 | Deposit fan |

**What that does to the altitude ceiling.** Applying the ≥ 20 px rule from §2.5 to each class (*computed* from the same GSD figures):

| Presentation | px at 30 / 60 / 90 / 120 m, 4K wide | Highest altitude still ≥ 20 px, 4K wide | Same, 640 × 512 thermal |
|---|---|---|---|
| Prone or supine body | 150 / 76 / 50 / 38 | 120 m (legal cap; geometric ~227 m) | 64 m |
| Cluster | 133 / 67 / 44 / 33 | 120 m (cap) | 56 m |
| Upright | 40 / 20 / 13 / 10 | 60 m | 17 m |
| Wading or clinging | 40 / 20 / 13 / 10 | 60 m | 17 m |
| Head and shoulders | 22 / 11 / 7 / 6 | 33 m | 10 m |
| Limb only | 16 / 8 / 5 / 4 | **24 m** | 7 m |
| Fully buried | 0 | never | never |

A survey flown at the design altitude of 45–60 m is therefore *complete* for bodies and clusters, *adequate* for upright and wading survivors, and *nearly blind* to head-only and limb-only presentations. That is a true statement the current map cannot make.

**The change to the machinery, which is small.** The coverage raster becomes a short stack, one layer per presentation class *j*:

    q_pass(cell, j) = R_slice(j ; GSD, band, time, blur, view) × V(cell, j)
    C_j(cell)       = Σ_passes q_pass(cell, j)
    POD_j(cell)     = 1 − e^(−k · C_j(cell))

`R_slice` is already conditioned on flight and light conditions; adding presentation as one more slice axis costs nothing at evaluation time because posture and submersion are already annotation attributes, so the held-out clips already carry the labels. Six or seven numpy layers replace one.

**Fully buried becomes the limiting member of the family, not a special case.** Its layer is identically zero everywhere by construction, so burial polygons stop being a hard-coded exception in the code and become what they physically are: the presentation class for which no aerial coverage ever accumulates (§2.7). The "aerial search cannot clear" label is then generated from the data rather than painted on.

**What the commander sees.** Three surfaces is too many to read, so the default view is a single mixture weighted by the presentation mix expected in that cell:

    POD_effective(cell) = Σ_j w_j(zone) · POD_j(cell)

and the mix `w_j(zone)` comes straight from the placement distribution already specified in §2.3 row 2 and §6.2: rooftop and upright presentations dominate the flooded settlement, head-only and clinging dominate the channel, prone and limb-only dominate the deposit fan. A selector lets the commander switch to any single layer, and the tooltip states both numbers. The observed presentations from the verifier head can update the assumed mix as the flight proceeds, so a fan that turns out to be producing limb-only detections re-weights its own map.

**The payoff is in the planner, and it is free.** Replace `ΔPOD` in §5.3a's action score with the mixture-weighted increment `Σ_j w_j · ΔPOD_j`. Nothing else changes, and the planner immediately acquires a behaviour nobody wrote: when a segment's prone-body layer is saturated but its limb-only layer is thin, a low pass at 25 m scores higher than another pass at 60 m, so the drone proposes the descent on its own. The descend-to-confirm behaviour that §5.5's hand case needs falls out of the coverage model rather than being a special rule.

**Statements the map can now make**, which is the whole point:

> Segment B is 0.81 searched for prone bodies and 0.12 for limb-only presentations. One pass at 25 m is recommended before it is treated as searched for partly buried casualties.

**Evaluation.** The reliability diagram of §5.12 becomes one diagram per class, checking that cells predicting 0.4 for limb-only really do surface about 40 % of limb-only ground truth. Expect the limb-only and head-only layers to be the poorly calibrated ones at first, because they have the fewest positives; report their confidence intervals rather than a bare number.

**Factors.** Technical: more layers means more slices and thinner statistics per slice, so calibrate the well-populated classes first and widen the intervals on the rest. Operational: never show seven heat maps; show the mixture with a selector, and put both numbers in the tooltip. Environmental: the ceilings above assume nadir and clear air; rain, fog and crossover pull every class down through `R_slice` already. Integration: one extra axis on the slice table, one extra dimension on the raster, one substitution in the planner's score. Feasibility: an afternoon, given that §5.3 and §5.3a exist.

### 5.4 Ingest and synchronisation (R1)

**Role.** Turn a video stream or recorded flight plus a telemetry log into per-frame (image, pose, intrinsics, timestamp) tuples at the processed frame rate.

**Sources and parsers (verified unless marked).**

| Source | What it carries | Parser | Gotchas |
|---|---|---|---|
| Simulator export | frame index = timestamp, NED pose, quaternion, GPS, attitude, gimbal, intrinsics, weather, mode | own CSV/JSON reader; `airsim_rec.txt` | Inject the §5.7 noise model; keep clean truth for evaluation |
| DJI `.SRT` (per-frame subtitle telemetry) | Mavic 3 / Air 3: ISO, shutter, `focal_len`, lat/lon, `rel_alt`, `abs_alt` — **no gimbal angles**; Matrice 30/M350: adds `F.PRY` (flight pitch/roll/yaw) and `G.PRY` (gimbal pitch/roll/yaw); newer models use an HTML-bracket format | 30-line regex parser written against the format documentation in `dji-drone-metadata-embedder` (MIT, v2.15.0, Sep 2026) | Field units differ by generation (`fnum 170` = f/1.7 on older models); consumer Mavic SRT has no gimbal attitude |
| DJI flight log `.txt` | GPS, aircraft attitude, **gimbal pitch/roll/yaw**, timestamps | `dji-log-parser` (MIT, v0.5.7, Rust core with Python bindings) | Logs v13+ are encrypted and need a DJI developer API key: apply early |
| MAVLink `.tlog` / ArduPilot `.bin` | `GLOBAL_POSITION_INT`, `ATTITUDE`, `GPS_RAW_INT` (h_acc), `GIMBAL_DEVICE_ATTITUDE_STATUS` (quaternion + frame flags), `SYSTEM_TIME`, `CAMERA_INFORMATION`, `CAMERA_IMAGE_CAPTURED` | `pymavlink` (v2.4.49) | No frame-locked video timestamp in MAVLink; alignment below |
| PX4 `.ulg` | `vehicle_global_position`, `vehicle_attitude`, `sensor_gps`, gimbal topics | `pyulog` (BSD-3, v1.2.4, Aug 2026) | Microsecond timestamps |

**Time alignment.** SRT is indexed by frame count (one entry per frame), so no interpolation is needed. For MAVLink/ULog, build a monotonic series, map `time_boot_ms` to UTC with `SYSTEM_TIME`, then interpolate position and altitude (`np.interp`) and SLERP attitude and gimbal quaternions to each frame's presentation timestamp from the container (PyAV or torchcodec expose PTS). Estimate the per-clip offset by cross-correlating barometric altitude with the visible take-off, store it as `t_offset_s` per clip, and expose it in the evaluation script; the Roboflow DJI georeferencing project lists exactly this as its open problem.

**Decode.** Windows/RTX: PyNvVideoCodec (NVIDIA, MIT, v2.2.2, Windows wheels) straight to CUDA tensors, zero-copy into PyTorch via DLPack, decoder kept alive across clips. Jetson: GStreamer `nvv4l2decoder` into DeepStream or an NVMM appsink; Orin Nano has hardware decode but software encode only. PyAV for PTS and metadata and for thermal MP4s. Rejected: decord (stale), OpenCV cudacodec (custom build).

**Decimation.** Process every k-th frame (k = 3 → 10 FPS, k = 6 → 5 FPS); the tracker's buffers are set in processed-frame units (§5.6). Every decoded frame is still available for the evidence thumbnail of the best observation.

**Factors.** Technical: FOV/crop ambiguity of 4K video vs the 4:3 sensor, datum of altitude fields, PTS vs wall clock. Operational: one `t_offset_s` per clip, logged. Environmental: none directly. Integration: the ingest module emits one tuple type for all sources, so every downstream module is source-agnostic. Feasibility: all pip-installable on Windows; the SRT parser is trivial; the DJI API key for encrypted logs is the only external dependency.

### 5.5 Detection and fusion (R2, R3)

**Role.** Turn each processed 4K RGB frame (and the 640 × 512 thermal frame when present) into a list of `human` / `animal` boxes with calibrated confidences, at ≥ 90 % recall on humans, under the tiling budget from §5.11.

**Chosen tools (verified 10 Sep 2026).**

| Component | Choice | Status | Why this one |
|---|---|---|---|
| RGB detector | **Ultralytics YOLO26s** (fallback YOLO26n on Orin Nano; YOLO26m on the RTX 4060), trained at 1024–1280 on tiles, exported to TensorRT FP16/INT8 | Ultralytics v8.4.146 (9 Sep 2026), 61.5k stars, **AGPL-3.0**; YOLO26 released Jan 2026 with an NMS-free end-to-end head, small-target-aware label assignment, and a `yolo26-p2.yaml` (stride-4 head) in the repo | The only family that combines Windows-friendly training on 8 GB, a published Jetson Orin TensorRT table (YOLO26s 6.41 ms FP16 on Orin NX at 640), built-in tracking, SAHI support and an INT8 path. AGPL is fine for an open hackathon repository; a closed commercial deployment would need the Enterprise licence, and that is flagged rather than hidden. |
| Permissive alternative / ensemble member | **RF-DETR Nano–Large** (Roboflow, ICLR 2026) | v1.10.1 (7 Sep 2026), 9.4k stars, Apache-2.0 for Nano/Small/Medium/Large; Orin NX TensorRT FP16 Nano at 384 px = 6.0 ms | Best accuracy-per-latency with a permissive licence; native resolutions are small (384–704) so it must be tiled harder; use it if AGPL becomes a blocker or as the second model in an ensemble. |
| Thermal detector | **YOLO26n**, trained thermal-only on HIT-UAV (CC-BY-4.0) + AIResQ (2026) + the simulator's thermal frames | Same toolchain | Cheap (4.6 ms on Orin Nano Super at 640), and thermal-only training needs no paired RGB–thermal data. |
| Fusion | **Late fusion at the box level**: Weighted Boxes Fusion (ZFTurbo/Weighted-Boxes-Fusion, MIT, 1.8k stars, pushed Jul 2026) with the ProbEn probabilistic score rule (Apache-2.0) | Pure numpy, millisecond cost | The RGB-only fallback is *structural*: if thermal is absent or misaligned the fusion step is skipped and the output is exactly the RGB model's output. No thermal-zeroed training tricks, no out-of-distribution inputs. |
| Tiling | Batched native-resolution tiles (own 30-line tiler, or `supervision.InferenceSlicer` / SAHI for offline) | supervision MIT 49.9k stars; SAHI MIT 5.5k stars | SAHI's ICIP 2022 paper: slicing-aware fine-tuning adds +12.7 to +14.5 AP on VisDrone/xView. |
| Second-stage verifier (precision recovery) | Small classifier on candidate crops (DINOv2/DINOv3 ViT-S features + logistic head trained on true/false-positive crops) | Optional | Suppresses logs, roofing sheets and glint blobs at conf 0.05–0.3 without lowering recall; ~10 crops per frame is affordable on Orin. |

**Alternatives rejected, with reasons.** RTMDet via mmyolo/mmdetection: last mmyolo release Aug 2023, mmdetection last push Aug 2024, no prebuilt mmcv wheels for Windows, mmyolo is GPL-3.0. EfficientDet: archived (google/automl, 2021), slow on TensorRT. DEIMv2: best accuracy per size but its licence is non-commercial only. Gold-YOLO: GPL-3.0. PP-YOLOE: needs the Paddle framework. Aerial-specific forks (TPH-YOLO, Drone-YOLO, FBRT-YOLO, LEAF-YOLO): unlicensed or stale research code; their recipe (high-resolution features, P2 head, multi-scale context) is what `yolo26-p2.yaml` plus 1024–1280 input plus tiling already gives. Open-vocabulary detectors (YOLO-World, Grounding DINO, OWLv2, SAM 3): zero-shot recall on aerial small targets is in the single digits to tens of percent (Grounding DINO 4–9 AP50 on an aerial benchmark; YOLO-World 12.9 AP50 on VisDrone), so they are used *only* for offline auto-labelling on tiles with human review (§6). 4-channel early fusion in Ultralytics: natively supported since v8.3.112 (a `channels:` key in the dataset YAML, multi-page TIFF or `.npy` input, N-channel ONNX/TensorRT export), but code reading shows HSV and Albumentations augmentation are silently disabled for non-3-channel input, video loaders are hard-wired to 3 channels, the pretrained stem is lost, and the fallback would be a *different* model than the RGB one. Two-stream mid-fusion detectors (CFT 206 M params, ICAFusion 120 M, DAMSDet 79 M): too heavy for Orin and, in the tightly coupled case, collapse when a modality is missing (DAMS-DETR drops from 54.0 to 15.0 mAP on LLVIP with RGB only).

**Why late fusion is not a sacrifice.** On KAIST with a YOLOv4 base, late fusion scored 5.35 % miss rate against 4.91 % for the best mid-fusion. ProbEn's late fusion of independently trained detectors beat the mid-fusion baseline on FLIR-aligned (83.76 vs 80.53 AP50). And every "aligned" RGB–thermal benchmark was aligned once with a global homography; 20–35 % of DroneVehicle boxes still show 0–15 px offsets. Box-level fusion at IoU 0.5 tolerates that; a 4-channel stem does not.

**Registration recipe for real DJI payloads (M30T, Mavic 3T, H20T).** The 61° thermal field sits inside the 84° wide field: the thermal frame maps to roughly a 2250 × 1800 px central window of the 4K frame, about 3.5 wide-camera pixels per thermal pixel. Computed parallax at 30–100 m is sub-pixel in thermal units, so the real enemies are un-calibrated scale/rotation, frame-timestamp skew (a 33 ms offset at 5 m/s is 2–4 thermal px), and DJI's zoom-sync changing the crop mid-flight. Lock thermal zoom at 1× and zoom-sync off; fit one homography per altitude band from ≥ 8 clicked ground correspondences; optionally refine online with ECC on gradient images; pair frames by nearest timestamp; fuse boxes in RGB coordinates. In the simulator the two cameras share a pose, so the homography is exact and this step can be switched off; it is exercised only on the real-footage replay path.

**Inference sketch.**

```
RGB 4K frame ── tile to N×1280 (native res) ──► YOLO26s TensorRT ──► boxes_rgb
thermal 640×512 ────────────────────────────► YOLO26n TensorRT ──► boxes_t ── H(T→RGB) ──► boxes_t'
if thermal present and registration residual < τ:
    fused = weighted_boxes_fusion([boxes_rgb, boxes_t'], weights=[w_rgb, w_t(time_of_day, contrast)],
                                  iou_thr=0.5, skip_box_thr=0.05, score rule = ProbEn)
else:
    fused = boxes_rgb            # the fallback is literally the RGB output
```

The ProbEn rule keeps a box seen by only one modality at that modality's posterior (marginalisation), which is what lets a thermal-only hot spot survive at night and an RGB-only detection survive over midday water. The thermal weight is lowered at midday and during the crossover windows (§2.3 rows 20 and 23) and raised at night.

**Training recipe for 8 GB VRAM (Windows 11).** *This three-stage recipe builds the transferable model that generalises to real footage. For the demo model that runs in the simulator and produces the acceptance figures, use the single-run recipe in §5.5c instead.* Verified facts: Ultralytics `batch=-1` auto-sizes to ~60 % of GPU memory; AMP is on by default; use `cache='disk'` not RAM with 16 GB; set `workers ≤ 4` and wrap training in `if __name__ == "__main__":` on Windows.

| Stage | Data | Setting | Fits 8 GB? |
|---|---|---|---|
| 1. Aerial pre-train | COCO-pretrained YOLO26s → VisDrone person classes + SARD + HERIDAL + NOMAD, sliced to 1024 tiles with 0.2 overlap | imgsz 1024, batch 8, AMP, 60–100 epochs, mosaic on, `flipud 0.5`, `degrees 0–180` for nadir sets | Yes (estimate 5–7 GB) |
| 2. Domain fine-tune | Flood/landslide real tiles (FloodNet/RescueNet backgrounds with pasted humans, LADI, drowning/water sets) + simulator tiles + C2A-style paste tiles, real:synthetic ≈ 60:40, synthetic ≤ 40 % of the mix | imgsz 1024, batch 8, 30–50 epochs, `close_mosaic 10` | Yes |
| 3. Optional P2 head | `yolo26s-p2.yaml` if persons are still < 20 px after tiling | batch 4 (P2 ≈ 1.5–2× activation memory) | Yes |
| 4. Thermal model | YOLO26n on HIT-UAV + AIResQ + simulator thermal, imgsz 640 | batch 16 | Yes |
| 5. Export | `model.export(format="engine", half=True)`; INT8 only after checking recall on the held-out clip | — | — |

Epoch time is an estimate: YOLO26s at 1024 on the RTX 4060 ≈ 3–5 min per 5k tiles, so 100 epochs ≈ 5–8 h; plan overnight runs. On VisDrone at 640 px the independent benchmark puts YOLO26 n–x at mAP50 0.26–0.38 and YOLOv8 at 0.27–0.37: architecture choice moves little, **resolution and tiling move a lot**, which is why the effort goes into the data and the tiling, not into architecture search.

**Operating threshold.** Sweep confidence on the validation split, pick the highest confidence at which human recall ≥ 0.92 (a 2-point margin over the 0.90 target), freeze it, and report precision and FP/min *at that threshold*. Ultralytics' reported per-class P/R are taken at the max-F1 confidence, not at your operating point, so read recall from the curves or run `val(conf=c)`. Also report recall at IoU 0.25 as a secondary "found the person" metric (TinyPerson practice), because on a limb-only target the box extent is ill-defined.

**Factors.**

| Factor class | What matters here |
|---|---|
| Technical | Pixels on target (≥ 20 px rule), tiling overlap (0.2), NMS-free head behaviour on tiny objects (an independent benchmark found YOLO26 ≈ YOLOv8 at 640 on VisDrone; verify on tiles), INT8 calibration set must include tiny/occluded positives |
| Operational | Threshold frozen per model version; RGB-only fallback exercised in every demo; thermal weight schedule tied to time of day |
| Environmental | Glint, turbid water, crossover and midday solar loading are evaluation slices, not surprises; the verifier is trained on the false positives those slices produce |
| Integration | Same ONNX for edge and cloud; box output in frame coordinates with tile provenance; thermal boxes mapped through H before fusion |
| Feasibility | Everything trains on 8 GB; the risk is training *time*, mitigated by overnight runs and 30–50-epoch fine-tunes |

### 5.5a The posture and submersion head: closing the hole in the triage score

**The problem this fixes.** §5.8 ranks records by `w_class(t) × urgency_class`, and that ordering — immersed above trapped above stranded — is the whole reason the list is a triage list rather than a sorted confidence dump. But nothing in the pipeline as described so far *predicts* which class a detection belongs to. The score has to assume it. That is a hole in the design's own logic, and the labels to close it already exist.

**Where the labels come from, at no extra cost.** The annotation guideline (§6.3) already records `pose ∈ {standing, sitting, prone, supine, half_submerged, trapped, unknown}` and `occlusion ∈ {0,1,2}` on every box. The simulator knows both exactly: posture from the animation state assigned to the actor, submersion from the pelvis height against `FloodLevel`, occlusion from the mask ratio (§5.1). Real data supplies more: SARD carries six pose classes, NOMAD carries activity labels including laying, hiding, swimming and drowning, C2A carries five poses, and POP carries a per-box occlusion rate. No new annotation work is required.

**Where it goes: fold it into the second-stage verifier.** Three placements were considered.

| Option | Verdict |
|---|---|
| Extra classes on the detector (`human_prone`, `human_upright`, …) | **Reject.** Splits the human class and therefore splits the recall metric that the ≥ 90 % requirement is measured on |
| Multi-task head on the detector backbone | Workable, but couples posture training to detector training and forces a retrain of the recall-critical model whenever the attribute schema changes |
| **Multi-output classifier on candidate crops** | **Adopt.** It is the same crop model already planned as the optional verifier (F10) |

The verifier was previously justified only by precision recovery, which made it easy to cut. Giving it a second job changes that: one small crop model, run on perhaps ten crops per frame, outputs `is_real`, `posture`, `submersion` and `occlusion` together. That is one forward pass buying both a cleaner list and an evidence-driven ranking, which is why **F10 should be promoted from stretch to MVP**.

Architecture: a frozen DINOv2 or DINOv3 ViT-S feature extractor with four small linear heads, trained on crops from the simulator (abundant, perfectly labelled) plus real crops from SARD, NOMAD, POP and C2A. It trains in minutes on the RTX 4060 and adds a few milliseconds per frame on Orin.

**The safety rule that makes it usable: posture may raise urgency, never lower it.** A wrong posture prediction must never demote a real survivor. Three constraints enforce this:

1. If `posture = unknown` or the head's confidence is below a threshold, the triage score uses the **base** weight for a stranded survivor. The unknown case is never the cheapest case.
2. Posture can promote a record (a head-only or half-submerged prediction moves it to the top of the list) but can never push it below the base rank.
3. The predicted class is displayed on the record card with its confidence, so a commander sees "half-submerged, 0.61" and can overrule it, rather than seeing a rank with no reason.

**Evaluation.** Report posture accuracy per class **binned by target pixel size**, because at 20–40 px posture is genuinely hard and the honest report must say where it works. The metric that actually matters is not raw accuracy but whether the ordering improves: measure rank correlation between the system's ranked list and the ground-truth urgency ordering, with and without the head. If the correlation does not improve, the head is decoration and should be switched off.

**Factors.** Technical: small-target posture is the weak case; foreshortening at nadir makes prone and supine hard to separate (they need not be separated, since both map to the same urgency). Operational: predictions are advisory and visible, never silent. Environmental: submersion prediction depends on the water context, which the coarse segmentation of §5.5b supplies. Integration: one crop model, one call site, four outputs. Feasibility: labels are free, training is minutes.

### 5.5b Radiometric thermal: absolute temperature as a physical filter

**The idea.** An 8-bit thermal picture is a *contrast-stretched* image: automatic gain control maps whatever temperature range is currently in frame onto 0–255. A radiometric frame instead carries an absolute temperature per pixel. Human skin sits near 33–35 °C, wet clothing lower, and a sun-heated galvanised roof can reach 60 °C or more. With absolute temperature you reject the roof by physics rather than asking a classifier to learn the difference.

**The bigger win is training stability, not just filtering.** Under automatic gain the same person takes different pixel values in different frames depending on what else is in view: a hot roof entering the frame darkens everything else. A detector trained on those images is learning a moving target. In radiometric units a person at 34 °C is 34 °C in every frame, so the thermal detector's input distribution stops drifting. This matters more than the filter itself.

**What the data actually supports, verified 10 Sep 2026.** This is where the idea meets reality and has to be scoped honestly.

| Path | Radiometric available? | How |
|---|---|---|
| **Simulator** | **Yes, free** | The thermal render already computes radiance from per-material temperature and emissivity (§5.1), so a 16-bit temperature channel is an extra output, not new work |
| **DJI stills (R-JPEG)** | **Yes** | Raw 16-bit sensor values live in the JPEG's APP3 segment; extracted with the DJI Thermal SDK (v1.8, `dji_irp -a measure`) or the pure-Python `thermal_parser` (MIT, 104★, pushed Mar 2025). Generic image tools read only the false-colour layer and silently miss the temperatures |
| **DJI thermal video** | **In practice no** | Operator reports show the M30T's `_T` video files cannot be opened in DJI's own analysis tool or converted while preserving thermal data. Treat video as 8-bit AGC |

**So the design uses it where it exists and degrades cleanly where it does not.** Video keeps running the AGC-trained thermal detector exactly as described in §5.5. Radiometric enters at the two moments that matter most:

1. **The dwell.** When the tactical planner (§5.3a) chooses "orbit and dwell" on a candidate, that dwell captures an R-JPEG still. The temperature check then arrives precisely at the confirmation step, which is where a hard filter is worth most and where a few hundred milliseconds of extra processing costs nothing.
2. **The simulator, throughout**, so the approach can be developed, evaluated and demonstrated end to end even though real video will not carry it.

**The filter.** Given an apparent temperature per pixel, a candidate box is scored on the distribution inside it: reject if the warm pixels sit far above plausible human surface temperature (sun-loaded metal, fire, engine), or far below (already at water or ambient temperature, meaning the thermal channel carries no evidence either way — which **downgrades the thermal contribution, it does not reject the record**, because a hypothermic or immersed person reads exactly like this, per §2.5). Absolute temperature also feeds the fusion weight `w_t` directly, replacing the time-of-day heuristic with a measurement.

**Honest caveats, all of which belong in the report.**

- Accuracy on DJI enterprise thermal is about ±2 °C, so thresholds must be bands, not edges.
- Apparent temperature depends on the assumed **emissivity** of the surface. Skin is around 0.98 and is easy; wet mud-coated clothing is not. Getting emissivity wrong shifts the reading, which is exactly why low-emissivity galvanised sheet (0.20–0.30) mirrors the cold sky and can read *colder* than air (§2.3 row 22).
- Reflected apparent temperature and atmospheric attenuation both matter at 60–120 m, modestly.
- The filter must never be allowed to reject a record outright on its own. It adjusts the thermal weight and can demote a *distractor*; a record supported by RGB survives regardless. This mirrors the standing rule that a thermal negative is never a clearance.

**Factors.** Technical: emissivity assumptions, ±2 °C accuracy, stills-only on real hardware. Operational: the temperature reading is shown on the record card as evidence a commander can read. Environmental: this is the direct countermeasure to the midday solar-loading false positives of §2.3 row 23, which are the dominant precision failure. Integration: one extra parser on the stills path, one extra channel in the simulator, one term in the fusion weight. Feasibility: the parser is an afternoon; the simulator channel is nearly free.

### 5.5c Simulation-first: the primary target is the simulator, and how ≥ 90 % recall is reached cheaply there

**Read this before §6's data plan, because it changes what that plan is for.** Sections 5.5 and 6 are written for a model that generalises to real disaster footage, and they are correct for that goal. But **the deliverable of this project is a simulated system.** The drone flies in Unreal, the cameras are rendered, the demo runs on rendered frames, and the acceptance thresholds will be demonstrated on rendered frames. The renderer is therefore not a training aid, it is the **deployment domain**, and a model must have seen its deployment domain during training. A model tuned only for real photographs and evaluated on renders is optimised for a target nobody in this project will ever run.

This subsection states the simulation-first objective explicitly, gives the recipe that reaches the ≥ 90 % recall threshold with a single short training run, and sets the reporting rule that keeps the claim defensible.

**Why ≥ 90 % is cheap in simulation and expensive in reality.** Detector recall is dominated by three quantities, and in a simulator you own all three.

| Quantity | In the real world | In simulation |
|---|---|---|
| Pixels on target | Set by the drone, weather and altitude you happen to get | You choose the altitude and therefore the pixel count |
| Domain gap between training and test data | Large and unavoidable: different cameras, sensors, lighting, seasons | **Zero.** Both sets come from the same renderer, actors, materials and lighting rig |
| Scene difficulty | Whatever the disaster produced | You choose the occlusion, submersion and posture distribution |

The middle row is the whole story. Almost all of the difficulty in the real recipe — the three-stage pipeline, the six real datasets, the sixty-forty mixing rule, the two to three days of curation — exists to survive a domain gap that does not exist when you train and test on the same renderer. Remove that gap and the problem becomes an ordinary, easy detection task.

**The simulation-first recipe. One training run, no dataset curation.**

1. **Fly for the number.** This is the single largest lever and it is not machine learning. Fly the evaluation passes at **40–50 m** and tile the 4K frame at native resolution. That puts an upright person at 25–30 px and a prone body well over 60 px, comfortably above the 20 px floor of §2.5. Flying at 100 m and downscaling the frame will defeat any model you can train.
2. **Fine-tune from COCO on simulator frames only.** Skip stage 1 of §5.5 entirely. Take stock YOLO26s weights, fine-tune for 30–50 epochs on 10–20k rendered tiles, at 1024. That is one overnight run on the RTX 4060, or two to four hours on a rented GPU. No real datasets are downloaded, no formats are converted, no licences are read.
3. **Split by scenario seed, never by frame.** Train and test scenes must use different seeds, different actor placements and different weather draws. Same renderer, different scenes. Splitting by frame would be leakage and the resulting number would be worthless even as a simulation claim.
4. **Run a low confidence threshold and recover precision downstream.** Pick the lowest threshold that keeps false positives per minute tolerable, not the highest that keeps recall up. The verifier of §5.5a, the three-hits-in-two-seconds confirmation of §5.6 and geographic deduplication all exist to make the resulting false positives cheap.
5. **Define the nominal slice before you measure it.** The headline claim is recall on a stated slice: 40–60 m, daylight, occlusion below 50 %, non-submerged presentations. Report that number as the acceptance figure. Report the hard slices — head-only, limb-only, crossover-window thermal, above 90 m — as separate numbers in the same table. §5.3b already defines these classes.

Expect the nominal slice to clear 90 % comfortably with the recipe above, because the model is being tested on the distribution it was trained on. **Measure it rather than assuming it**, and if it does not clear, the cause will be flight geometry or scene difficulty, not the model, in which case lower the altitude before touching the training.

**The domain-randomisation trade, stated so it is a choice and not an accident.** §5.1 lists runtime texture swapping and camera noise, chromatic aberration, lens distortion and motion blur as features. They are in fact the two levers that control which way this project points, and they pull against the number you are trying to demonstrate.

| Setting | In-simulation recall | Real-footage capability |
|---|---|---|
| Randomisation off, fixed materials and clean renders | **Highest.** Train and test are nearly identical | Near zero. The model has learned this renderer |
| Randomisation on: texture swapping, sensor noise, aberration, blur | Lower, by an amount you should measure | Meaningfully better. Texture randomisation alone is worth double-digit mean average precision in the drone literature |

For a demonstration built and judged in simulation, **randomisation off is the correct default** and gives the acceptance number. Turning it on is the stretch that buys a claim about the real world. Do not switch it on midway through and then compare numbers across the switch.

**Two models, or one model and two numbers.** The primary artefact is the **demo model**: simulator-only training, evaluated on held-out simulator scenes, and this is what runs in the demo and produces the acceptance figures. The optional secondary artefact is the **transferable model** from §5.5 and §6: real data plus simulator data at roughly sixty-forty, evaluated on a real clip. Build the first; build the second only if time remains. They come from the same scripts and differ by a dataset path.

**The reporting rule, which is a competitive advantage rather than a formality.** Every recall figure carries the domain it was measured in, in the same sentence as the number: "94 % recall at IoU 0.5, in simulation, on the nominal slice." A judge or reviewer will ask whether it works on real footage. A team that answers "we measured 94 % in simulation, we measured the real-footage gap on one clip, and here is what closes it" is in a far stronger position than one that quotes a single unqualified figure and cannot say how it was obtained. Never average a simulation number with a real number, and never present a simulation number without the word.

**What this changes elsewhere in the document.** §5.5's three-stage recipe becomes the *transferable-model* path rather than the default. §6's dataset table becomes optional for the demo and required only for the transferable model. §6.5's sixty-forty rule applies to the transferable model only; the demo model is one hundred percent simulator data by design. §5.12's evaluation harness is unchanged, but every table gains a domain column.

### 5.6 Tracking and deduplication (R4)

**Role.** Give each living being one persistent record across frames, passes and re-visits, so the commander sees one marker, not forty. Survivors are mostly *static* and the *camera* moves, which is the opposite of the pedestrian-tracking assumption most trackers are tuned for.

**Chosen tools.**

| Component | Choice | Status | Why |
|---|---|---|---|
| Frame-to-frame tracker | Ultralytics built-in **BoT-SORT** (`botsort.yaml`, `gmc_method: sparseOptFlow`, `with_reid: False`) or the new default **TrackTrack** | Ships in Ultralytics 8.4.146 (six trackers: BoT-SORT, ByteTrack, OC-SORT, DeepOC-SORT, FastTrack, TrackTrack) | Camera-motion compensation is the one feature that matters for a moving camera; ByteTrack's low-confidence second-stage match keeps a flickering 15-px survivor alive |
| Licence-clean alternative | `roboflow/trackers` BoT-SORT with CMC (Apache-2.0, 3.8k stars, v2.6.0 Aug 2026) + `supervision` (MIT) | Maintained | Same behaviour if the AGPL of Ultralytics/BoxMOT is a problem |
| Reference | BoxMOT (AGPL-3.0, 8.3k stars, v25.0.0, Sep 2026; nine trackers) | Maintained | Use for tracker A/B tests, not in the product |
| Cross-pass identity | **Geo-clustering**: per-track weighted-median lat/lon → `sklearn.cluster.DBSCAN(eps = 2 × CE90, min_samples = 1, metric = 'haversine')` | scikit-learn | Position is the stable identity of a static survivor; appearance is not (a 20–60 px blob has no usable appearance) |
| Metrics | `py-motmetrics` (pip, maintained; HOTA/IDF1/ID switches) + own record-level script | MIT | TrackEval is the canonical HOTA code but was last pushed Jul 2024; vendor it only if needed |

**Rejected.** Re-identification models (OSNet/torchreid, fast-reid, CLIP-ReID): at 20–60 px an embedding adds 1–3 ms per crop and mostly noise; Ultralytics' `model: auto` (reuse detector features, near-zero cost) is the only ReID worth an A/B. UAVMOT: unlicensed and stale. UCMCTrack (AAAI 2024): its idea, tracking on the ground plane via camera parameters, is exactly right for a gimballed drone and is *ported as an idea*: the dedup step works in geo space.

**Static-survivor tuning (validate on day 1).**

1. Run the tracker on *processed* frames (2–10 FPS after decimation) and set `track_buffer` in processed-frame units: 5 FPS × 30 s = 150, not the default 30 that assumes 30 FPS.
2. Keep `track_low_thresh` 0.1 and `fuse_score` on, so the second-stage match rescues low-confidence frames.
3. Require confirmation before emitting a record: 3 hits within 2 s (TrackTrack `min_track_len: 3`).
4. Run `sparseOptFlow` camera-motion compensation on a 640-px-wide downscale; ORB/ECC on 4K is far too slow. Rippling water breaks optical-flow compensation, so fall back to a *telemetry-predicted* homography (the previous frame's boxes warped analytically from attitude and GPS change).
5. After compensation the residual motion of a still target is small; if IDs still flip, lower `match_thresh` (0.8 → 0.6) before raising buffers.
6. What the UAV literature says: on VisDrone-MOT and UAVDT, tracker choice moves HOTA by ~1 point while MOTA (false-positive-sensitive) swings wildly; detector quality dominates. Gate confirmation rather than tuning association endlessly.

**Deduplication mechanism.** Every confirmed track yields observations (lat, lon, σ, conf, frame). Take the weighted-median position and the per-track CE90 from the geolocation budget for that geometry. Cluster all tracks from this and previous passes with DBSCAN at radius 2 × CE90 (about 6–8 m at 60 m nadir with consumer GNSS; ~10 m oblique). One cluster = one record. Confidence = 1 − Π(1 − conf_track); `count_estimate` = the maximum number of *simultaneous* distinct tracks inside the cluster (groups on a roof); `motion.state = moving` if any member track's geo-displacement over ≥ 5 s exceeds 3 × CE90, else `still`. A record not re-observed when its cell is re-imaged becomes `stale`, never deleted. Record IDs are never renumbered.

**"Deduplication accuracy", defined so the evaluation script can compute it.** Match output records to ground-truth survivor IDs by distance ≤ r (Hungarian on distance). Report record precision (duplicates count as false positives), record recall, duplicate rate = (records − unique matched GT) / unique matched GT, count error per cluster, and track-level HOTA / IDF1 / ID switches. Also report FP/min twice: at the raw-detection level and at the record level after tracking and dedup. The ratio is the value the tracker adds.

**Factors.** Technical: frame decimation vs buffer units, CMC over water, group counting. Operational: stable IDs across a re-visit two minutes later (test case). Environmental: water texture defeats optical flow; canopy causes gaps. Integration: dedup runs in geo space so tracker ID switches stop mattering for the output. Feasibility: all pip-installable; tracking costs 10–25 ms on Orin.

### 5.7 Geolocation (R5)

**Role.** Turn a pixel in a frame into latitude/longitude with a stated error radius. A detection without a coordinate cannot be actioned, and the error radius sets the dedup radius and the marker's uncertainty circle.

**Reuse verdict.** No maintained Python library does this end-to-end for video. OpenAthena's Python is archived and carries an export-control addendum (do not vendor). What *is* adopted: Theta-Limited's **DroneModels.json** (Apache-2.0, pushed Sep 2026) for DJI camera intrinsics, Brown–Conrady distortion coefficients and a target-location-error model per slant range; **Copernicus DEM GLO-30** (free, < 4 m LE90 vertical) fetched with `dem-stitcher` (Apache-2.0, v3.2.0 Aug 2026); `pyproj` for geodesics; `rasterio` for sampling. The implementation itself is ~80 lines of numpy.

**The chain (frames: image → optical → gimbal FRD → body → NED → geodetic).**

1. Intrinsics from field of view when uncalibrated: `f_px = (W/2) / tan(HFOV/2)`. DJI publishes *diagonal* FOV; convert with the frame aspect, and beware that 4K 16:9 video may be a crop of the 4:3 sensor (day-1 test: measure the effective HFOV, or read the per-frame `focal_len` field from the SRT).
2. Undistort the pixel with the DroneModels coefficients (`cv2.undistortPoints`); skipping this costs up to ~1° at the frame edge of a wide lens.
3. Ray in the optical frame: `d = normalize([(u − c_x)/f_x, (v − c_y)/f_y, 1])`; re-order to forward-right-down.
4. Rotate by the gimbal attitude. For DJI SRT/XMP the gimbal angles are already earth-referenced (yaw relative to north), so the body attitude is not needed. For MAVLink `GIMBAL_DEVICE_ATTITUDE_STATUS` the frame flag decides whether to compose with the vehicle attitude. For the simulator, the camera pose is exact.
5. Intersect the ray with the ground: flat plane at the barometric AGL (`t = h / d_z`, reject near-horizon rays with `d_z ≤ 0.1`), or ray-march a DEM in 2–5 m steps and bisect the crossing when terrain relief matters. Over flood water the "ground" is the water surface: prefer barometric AGL plus a manual water-level offset over DEM marching, and treat DEM error as at least the flood depth.
6. Convert the north/east offset to lat/lon with a WGS-84 geodesic (`pyproj.Geod.fwd`); never use a single "111 km per degree" constant for longitude.
7. Emit lon, lat and `h_acc_m` for *this pixel's geometry* (off-nadir angle), so the dedup radius is per record.

**Error budget.** Sensitivities with θ = angle between the ray and the vertical: pointing error `h·sec²θ·δθ`; heading error `h·tanθ·δψ`; altitude error `tanθ·δh`; pixel error `h·sec²θ·(δu/f)`; GNSS `δp`; time-sync `v·δt`. Inputs used (with sources in the research file): consumer GNSS 2.5 m (RTK 0.05 m); vehicle pitch/roll 0.5° (pessimistic 1.7°); magnetometer yaw 1.5° (pessimistic 3°); gimbal report 0.2°; boresight 0.3°; AGL 1 m barometric (4 m if from a DEM); 5 px localisation; 40–50 ms sync at 5–10 m/s.

| Geometry | 30 m AGL | 60 m AGL | 100 m AGL |
|---|---|---|---|
| Nadir, image centre, consumer GNSS (1σ) | **2.5 m** | **2.6 m** | **2.7 m** |
| Nadir, image centre, RTK | 0.4 m | 0.7 m | 1.1 m |
| Nadir, frame edge (θ = 36°), consumer, AGL ±1 m | 2.7 m | 3.0 m | 3.6 m |
| Nadir, frame edge, AGL from DEM (±4 m) | 3.9 m | 4.1 m | 4.6 m |
| Oblique −45°, image centre, consumer | **2.9 m** | **3.4 m** | **4.4 m** |
| Oblique −45°, RTK | 1.5 m | 2.3 m | 3.6 m |
| Oblique −45°, pessimistic attitude (1.7° / 3°), consumer | 3.9 m | 5.5 m | 8.5 m |

Multiply by ≈ 2.1 for CE90. Cross-checks: a 2018 low-cost-UAV study derives σ ≈ 0.10–0.12 h at 30° camera pitch; OpenAthena's field testing reports 0.02–0.035 m per metre of slant range; BYU reached ~3 m with recursive least squares and bias estimation at small-UAV altitudes. Reading: **at nadir the budget is GNSS-dominated (~2.5–3 m) and RTK buys 5×; at 45° oblique and ≥ 60 m the attitude and yaw terms dominate and RTK barely helps.** So: fly nadir-ish (gimbal −70° to −90°) for geolocation, take the final fix with the target near the frame centre, and calibrate the compass (yaw is the cheapest accuracy win). Averaging N frames shrinks the random terms by ≈ √N but not the biases (yaw, boresight), which is why single-pass records still carry the full yaw term.

**The stated error budget for the assumed altitude (R5).** The design altitude is 60 m AGL, nadir gimbal, consumer GNSS: **≈ 2.6 m (1σ) at the image centre, ≈ 3.0 m at the frame edge, CE90 ≈ 6 m; with RTK ≈ 0.7–1.7 m.** Each record publishes its own `h_acc_m`.

**In the simulator.** Ground-truth pose makes sim geolocation error ~0, which would tune the dedup radius wrong for real footage. Inject the noise model above into the exported telemetry (GNSS 2.5 m random walk, 1.5° yaw bias, 0.5° pitch/roll, 1 m barometric), and keep a switch to turn it off for debugging.

**Factors.** Technical: FOV/crop ambiguity, datum consistency (DJI `abs_alt` is not reliably ellipsoidal/orthometric across models; use DEM(takeoff) + `rel_alt`), near-horizon rays. Operational: compass calibration, nadir policy for fixes. Environmental: water surface vs DEM, wind-induced attitude error (row 14 of §2.3). Integration: `h_acc_m` feeds the dedup radius and the map circle. Feasibility: trivial compute; the work is in getting the telemetry fields right per platform.

### 5.8 Triage scoring and output (R6)

**Role.** Produce the ranked list: one GeoJSON Feature per record with location, confidence, movement vs stillness, count, evidence thumbnail, and the *components* of its priority so the commander can see why it ranks where it does.

**Priority score (proposal, anchored on §2.5).**

```
score = P(living | record) × w_class(t) × urgency_class × (1 + 0.1 · count_estimate)
```

- `P(living | record)` is the fused detection confidence, raised by thermal-positive evidence in the first hours (strong evidence of life) and by observed motion, and left unchanged (never lowered) by a thermal-negative RGB detection after hours.
- `w_class(t)` is the survival-decay curve for the inferred entrapment class: immersed (rank key = time-to-exhaustion at the water temperature), trapped-in-structure (1.0 for 48 h, then half-life ≈ 3 days, floor 0.05), stranded (1.0 for 24 h, then half-life ≈ 4 days), animal (0.3 × human curve). The floor is never zero.
- `urgency_class` orders immersed > trapped > stranded, with head-only detections at the top of the list.
- Every term is stored on the record and shown in the UI; the number is never shown alone.

**Record schema (GeoJSON Feature, RFC 7946, coordinates `[lon, lat, alt]`, 6 decimals).** Properties: `record_id`, `cluster_id`, `status ∈ {candidate, confirmed, stale, dismissed}`, `priority_rank` and the score components, `confidence` and `confidence_max_det`, `n_observations`, `n_tracks_merged`, `seen_in_passes`, `first_seen_utc`, `last_seen_utc`, `position {lat, lon, alt_msl_m, h_acc_m, h_acc_basis, method, dem_source, agl_m, off_nadir_deg}`, `motion {state, displacement_m, window_s}`, `count {estimate, min, max, basis}`, `class` (`human`/`animal`), `pose` and `submersion` attributes when inferable, `modality` and `thermal_hot`, `pixel_size_px`, `gsd_cm_px`, `evidence[] {thumb_uri, clip_id, frame_idx, frame_time_utc, bbox_px, det_conf, camera}`, `source {platform, telemetry, sim, aoi_id}`, `notes`. Thumbnails are file references in the live feed (keeps messages under 5 KB) and are embedded only in exported bundles (KMZ `files/` or a single-file HTML report).

**Formats.** GeoJSON is primary (Python `geojson` or plain `json`). KML/KMZ via `simplekml` (stale but complete; embed thumbnails with `addfile()`), for Google Earth users. Cursor-on-Target for TAK users (§5.9). GeoPackage export via geopandas/pyogrio for GIS hand-off. The record log itself is SQLite in WAL mode; DuckDB-spatial for analytics if spatial SQL is wanted. SpatiaLite on Windows needs a DLL and is skipped; PostGIS needs a server and is rejected for a laptop.

**Guardrail implementation (R10).** There is no code path that deletes a record or sets a segment to "cleared". Operators can set `status = dismissed` with a reason string; dismissed records stay in the log and in the exported bundle under a separate layer. The search-quality map (§5.3) shows probability of detection, never a "done" flag.

### 5.9 Map UI and command-and-control integration (R6, R12)

**Chosen.** **MapLibre GL JS** (BSD-3, v6.9.0 Sep 2026) in a single static HTML page, an offline vector basemap in **PMTiles** extracted for the area of operations from the Protomaps daily build (`pmtiles extract --bbox … --maxzoom 15`; attribute OpenStreetMap), and a FastAPI backend pushing GeoJSON over a WebSocket at each record update. Layers: triage markers (size = rank, colour = class, ring = `h_acc_m`), evidence pop-up with the thumbnail and score components, the search-quality raster (probability-of-detection heat map with the "aerial search cannot clear" polygons hatched), the drone's live position and footprint, the planned pattern, and the flight track.

**Rejected.** CesiumJS (3D not needed; heavy). kepler.gl (still asks for a Mapbox token; fine for post-hoc analysis only). Streamlit + folium (re-renders the whole map per update, not for a 5 Hz feed; fine for the evaluation dashboard). deck.gl only if 100k+ points are ever needed.

**Responder integration: TAK.** Real teams increasingly run ATAK/WinTAK. The detections are emitted as Cursor-on-Target (CoT) events using `pytak` so a WinTAK/ATAK client or a TAK server shows the same markers. This is the same integration OpenAthena uses to push target coordinates into ATAK. Details and licences are in §5.2/§5.10's control-and-C2 research; the gods-eye-view repository you cited is a CesiumJS globe fusing public feeds (aircraft, ships, satellites, earthquakes, fires, CCTV); its reusable value here is the UI pattern for streaming geolocated feeds onto a map, not any drone or vision code (Appendix D).

**India-specific note.** OSM tiles render on-the-ground borders; for a public demo in India crop the basemap to the area of operations or use Survey of India boundary data. The 2021 DST geospatial guidelines set a 1 m horizontal / 3 m vertical threshold for finer data restrictions; the outputs here are coarser than that.

### 5.10 Offline queue and cloud fallback (R7, R9)

**Chosen.** A local **SQLite outbox** using `persist-queue`'s `SQLiteAckQueue` (BSD-3, v1.1.0 Oct 2025): every record and thumbnail is written to the local SQLite log first, then an upload job keyed by `(clip_id, record_id, version)` is enqueued; an uploader thread retries with back-off and acknowledges on success; the server upserts idempotently. Delivery is at-least-once and replayable, the local pipeline never blocks on the network, and the same mechanism carries the search-quality raster tiles. MQTT with persistent sessions (Mosquitto, QoS 1) is the natural field-radio transport and can replace HTTP without changing the outbox; ZeroMQ, CouchDB/PouchDB replication, Syncthing and rqlite were considered and are heavier than needed for one laptop and one server.

**Cloud fallback for detection.** A FastAPI service with the *same* Ultralytics TensorRT/ONNX engine exposes `POST /detect` for JPEG tiles. If the edge misses its per-frame deadline it ships the downscaled frame (~200 KB JPEG) and merges results by frame index; when the link is down it keeps running the cheaper local configuration (single 1920 pass, or coarse pass plus ROI re-detect). Triton (BSD-3, v2.72) and BentoML are overkill for a hackathon; Roboflow Inference needs an account for custom weights.

### 5.11 Edge deployment and the latency budget (R9)

**Platform status (Sep 2026).** JetPack 6.2.x covers the whole Orin family; JetPack 7.2.1 (Aug 2026: Ubuntu 24.04, CUDA 13.2, TensorRT 10.16) added Orin support; DeepStream 9.1 requires JetPack 7.2. Ultralytics' Orin benchmarks were run on JetPack 6.x, so develop on 6.2 and treat 7.2 as a day-1 compatibility test. Orin Nano Super: 67 sparse INT8 TOPS; Orin NX 16 GB: 157 TOPS. The RTX 4060 (242 AI TOPS, one 5th-gen NVDEC) is a comfortable superset of any Orin for development, which is why the same engines are built on both and the *measured* numbers on the 4060 are used only as an upper bound.

**Video decode.** Windows/RTX: **PyNvVideoCodec** (NVIDIA, MIT, v2.2.2, Windows wheels) decoding straight to CUDA memory; one Ada NVDEC does ~1,641 fps of 1080p HEVC, roughly 410 fps at 4K, so a 4K30 stream uses ~7 % of the decoder. Jetson: GStreamer `nvv4l2decoder` (or DeepStream); Orin NX is rated for two 4K60 streams, Orin Nano for one. Decode is never the bottleneck; the host copy and colour conversion are, so frames stay on the GPU and only crops are downloaded for thumbnails. Rejected: decord (stale), OpenCV cudacodec (custom build).

**Detector cost by 4K strategy** (TensorRT FP16 unless noted; scaled from the published 640-px Ultralytics tables by pixel count, so these are estimates until measured on day 1):

| Strategy | Keeps a prone person at 60 m ≥ 20 px? | Orin NX 16 GB (n / s / m) | Orin Nano Super (n / s / m) | RTX 4060 (n / s / m, T4 proxy) |
|---|---|---|---|---|
| C1: single 1920 × 1088 letterbox | yes ≤ 60 m, marginal at 100 m | 21 / 33 / 60 ms | 23 / 37 / 69 | 9 / 13 / 24 |
| C2: 6 native tiles 1280 × 1088 (2 × 3, ~10 % overlap) | yes to 100 m | 99 / 154 / 280 (s INT8: 115) | 110 / 172 / 326 (s INT8: 126) | 41 / 60 / 113 |
| C3: C1 + re-detect 4 ROIs at 640 native | yes where the coarse pass fires | 38 / 58 / 106 | 42 / 65 / 124 | 15 / 23 / 43 |
| Thermal 640 × 512 native | thermal person at 60 m ≈ 20 × 6 px | 4–12 | 5–14 | 2–5 |

**Per-frame budget (one processed 4K frame, ms).**

| Stage | Orin NX 16 GB | Orin Nano Super | RTX 4060 |
|---|---|---|---|
| NVDEC decode (pipelined) | 8–17 | ~17 | ~2.5 |
| Colour convert, tile, normalise (GPU) | 3–8 | 3–8 | 1–3 |
| Detect, C2 YOLO26s INT8 | 115 | 126 | 60 |
| NMS-free post-processing | 1–2 | 1–2 | < 1 |
| Thermal detect + fusion | 5–12 | 6–14 | 3–5 |
| Track (BoT-SORT + sparse optical flow on a 640-wide frame) | 10–25 | 10–25 | 3–10 |
| Geolocate | < 1 | < 1 | < 1 |
| Write (SQLite, GeoJSON, 128-px thumbnail, WebSocket) | 2–5 | 2–5 | 1–3 |
| Python/framework overhead | 15–30 | 20–40 | 5–10 |
| **Total, C2 YOLO26s INT8** | **≈ 170–200** | **≈ 190–230** | **≈ 80–95** |
| Total, C1 YOLO26m FP16 | ≈ 105–135 | ≈ 125–165 | ≈ 40–50 |
| Fits < 300 ms? | C2-s yes; C2-m no (INT8 borderline); C1/C3 yes | C2-n yes; C2-s INT8 yes, FP16 borderline; C1/C3 yes | everything, including C2-m and C2-l |

Caveat that decides the day-1 test: users have measured 2–3.5× the published Orin numbers end-to-end when pre/post-processing stays in Python (16 ms measured vs 4.5 ms published for YOLO11n on Orin Nano Super). Batch the six tiles into one engine call and keep pre-processing on the GPU, or the 300 ms target is at risk on the Nano.

**Throughput vs latency.** The 300 ms figure is per-frame *latency*; real-time throughput at 30 FPS needs decimation. A static survivor tolerates 2–5 processed FPS: at 60 m nadir the 16:9 footprint is ≈ 89 × 50 m, so at 10 m/s the along-track dwell is ≈ 5 s and 5 FPS gives ~25 observations. Plan: **Orin NX → C2 YOLO26s INT8 at 5 FPS** (or C3 at 8–10 FPS); **Orin Nano Super → C3 YOLO26s or C2 YOLO26n at 5 FPS**; **RTX 4060 → C2 YOLO26m at 10 FPS**. Thermal runs at native 640 on every frame it is available (cheap) and its hits seed RGB regions of interest. Dynamic tiling (skip open-water tiles with no texture or pure canopy; always process tiles with any coarse hit or active track) is expected to save 30–60 % of tiles over flood scenes.

**Memory.** RTX 4060 (8 GB) and Orin NX (16 GB) hold a YOLO26m 1280 batch-6 engine comfortably; Orin Nano's 8 GB is shared with the OS and decoder, so build engines with `batch=6, half=True` and a workspace ≤ 2 GB.

### 5.12 Evaluation harness (R8)

**Tools (all pip on Windows).** `torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True, extended_summary=True)` for recall at IoU 0.5 per class (pycocotools 2.0.11 has Windows wheels as the backend); Ultralytics `model.val()` for quick checks and curves; `py-motmetrics` for HOTA/IDF1/ID switches; **FiftyOne** (Apache-2.0, v1.21, Windows supported) to browse false positives on glint and false negatives under canopy; an own 100-line script for FP/min, record-level precision/recall, duplicate rate and slices.

**Metrics, defined once.**

- **Recall at IoU 0.5 (human):** from the extended summary tensor `recall[iou=0.5, class=human, area=all, maxDet=100]`, at the frozen operating threshold; also at IoU 0.25.
- **FP per minute:** per processed frame, greedy-match predictions to ground truth at IoU ≥ 0.5, count unmatched predictions; `FP/min = FP_total / (frames_processed / fps_processed / 60)`; reported per clip and per terrain type (water, debris, vegetation, roof), and separately at the record level after dedup (expected ≥ 10× lower).
- **Deduplication accuracy:** record precision, record recall, duplicate rate, count error, HOTA/IDF1/IDsw (§5.6).
- **Geolocation error:** median and 90th-percentile distance from ground-truth survivor positions vs the §5.7 prediction.
- **Search-quality map calibration:** for cells with a ground-truth survivor, the fraction detected, binned by the cell's predicted probability of detection (a reliability diagram; the map is honest if the bins lie near the diagonal).

**Every table in this report carries a domain column** (`sim` or `real`), and no figure is quoted without it (§5.5c). The acceptance figures come from the simulator; any real-footage figure is reported beside them, never averaged with them.

**Held-out clip protocol.** One simulator clip (noise-injected telemetry, known survivor IDs and positions, all three zones, at least one buried actor and one crossover-window segment) plus one real DJI clip with boxes hand-labelled every 10th frame. Publish every number above with its slice grid (zone × altitude × band × time of day × occlusion × posture), and the missed-detection analysis: recall vs pixel-height histogram to find the operating floor, and the top failure clusters from FiftyOne (expected: head-only in turbid water at > 60 m, midday roofing-sheet false positives, crossover-window thermal misses).

**What the numbers are expected to look like** (so the team knows what "good" is before running anything): HERIDAL-class imagery at 2 cm GSD gives 86.1 % recall at IoU 0.5 (92.9 % relaxed); the POP thermal set gives 0.78 recall for YOLOv8s; VisDrone all-classes at 640 gives 0.36–0.41. **The ≥ 90 % target is expected to be met only on tiled high-resolution RGB at ≤ 60 m AGL with in-domain fine-tuning, and the report must say so slice by slice.**
---

## 6. Data plan (R11)

### 6.1 Real datasets to pull first

Verified on 10 Sep 2026 (sizes, classes, licences, link liveness); the full 29-row table with per-dataset notes is in the research file. No public dataset labels "half-submerged" or "trapped in debris" persons from real drone footage; that gap is exactly what the synthetic set fills.

| Priority | Dataset | What it gives this project | Size | Modality / altitude | Licence | Access |
|---|---|---|---|---|---|---|
| CORE | **NOMAD** (WACV 2024) | Graded visibility labels in 10 bins, laying and hiding sequences, a swimming/drowning subset | 42,825 frames, 5.4K video | RGB, 10–90 m | CC BY 4.0 | Google Drive live |
| CORE | **POP** (Sci. Data 2025) | Thermal persons lying supine/lateral under vegetation with a per-box occlusion rate 0–100 %, four weathers; tight visible-region boxes | 8,768 images, 25,811 boxes | Thermal 1280 × 1024, 30–70 m (M30T) | CC BY-NC-ND 4.0 | OSF |
| CORE | **WiSARD** (IROS 2022) | Synchronised RGB + thermal pairs with sitting/laying/waving subjects, above/below-canopy views, terrain and time-of-day metadata | 15,453 pairs (55,942 labelled images) | RGB 4K + LWIR 640 × 512, 20–120 m | MIT-style | Google Drive, 40 GB; pairs are *not* spatially registered |
| CORE | **SARD** (IEEE Access 2021) | Real drone RGB with six pose classes incl. lying and sitting, "positions typical of exhausted or injured persons" | 1,981 frames, 6,532 boxes | RGB 1080p, 5–50 m | paper CC BY 4.0; dataset licence behind login | IEEE DataPort (login); Roboflow/Kaggle mirrors lose the pose labels |
| CORE (thermal) | **AIResQ** (Sci. Data 2026) | High-resolution thermal SAR persons in atypical poses plus a real-drone benchmark; the honest thermal baseline (mAP50 0.55) | 9,788 images, 17,550 boxes (+1,988 benchmark) | Thermal up to 2048 × 1536, 50–120 m | CC BY 4.0 (article) | Zenodo (unreachable on the research day) |
| CORE (water) | **SeaDronesSee** ODv2 | Swimmers in open water with per-frame altitude and gimbal metadata | 8,930 / 1,547 / 3,750 images | RGB up to 4K, 5–260 m | CC0 | Site login; Roboflow mirrors |
| CORE (water) | **AFO** | Floating persons and objects | 3,647 images, 39,991 objects (33,174 human) | RGB 720p–4K | CC BY-NC-SA 3.0 | Kaggle |
| SUPPLEMENT | HIT-UAV | Thermal pre-training (persons, cars, bicycles), day and night | 2,898 images, 24,899 boxes | Thermal 640 × 512, 60–130 m | CC BY 4.0 (conflicting CC0 copy) | GitHub |
| SUPPLEMENT | C2A | 360k composited humans in five poses over flood and rubble backgrounds (pose prior only; occlusion not modelled) | 10,215 images | RGB composite | unclear (none / MIT / CC BY 4.0) | Google Drive, Kaggle |
| SUPPLEMENT | HERIDAL, Lacmus LADD, Okutama-Action, ForestPersons | Tiny wilderness persons; forest SAR at 40–100 m; a "lying" action at 10–45 m; canopy visibility bins | 68k patches; 1,365 img; 77k frames; 204k boxes | RGB | CC BY 3.0; GPL-3.0; CC BY-NC-SA 3.0; CC BY-NC-SA 4.0 (gated) | mirrors (HERIDAL origin site is dead) |
| SUPPLEMENT (water) | MOBDrone | 113k person-in-water boxes from one sea scene | 126,170 frames | RGB 1080p, 10–60 m | unverified (Zenodo down) | Zenodo |
| ANIMAL | BIRDSAI; aerial-sheep; aerial-cows; Koger wildlife | Thermal aerial animals with an occlusion flag; livestock from above; wildlife with humans | 62k frames; 1.7k; 1.7k; 2k images | Thermal / RGB | CDLA-Permissive; Public Domain; CC BY 4.0; CC0 | LILA, HF/Roboflow |
| REFERENCE | VisDrone-DET, TinyPerson | Pre-training and the occlusion/truncation and size-bin conventions | 10k images / 471k boxes; 1.6k / 72k | RGB | unstated / MIT repo | Drive |
| Backgrounds | FloodNet, RescueNet, AIDER, LADI v2 | Flood and rubble textures for compositing and for the simulator's material reference (no person labels) | 2.3k; 4.5k; 2.5k; 10k | RGB | CDLA-P; conflicting; GPL-3.0; CC BY 4.0 | Dropbox/HF |

Licence hygiene for the deliverable dataset: the team's own synthetic data is released under CC BY 4.0; real data is *referenced* with its licence and a download script, never redistributed; POP, AFO, ForestPersons and C2A are non-commercial or unclear and are kept in the training mix only with that flag recorded in the data card.

### 6.2 Synthetic generation plan

Target: **~20,000 labelled 4K frames** (≈ 300k tiles at 1024) across the three zones, with the attribute distribution below, plus ~10,000 thermal frames from the same poses. Generation is scripted (§5.1 capture plan) with a seed per scenario so any frame can be regenerated.

| Axis | Distribution (from §2.3) |
|---|---|
| Zone | 35 % flooded settlement, 35 % channel and banks, 30 % deposit fan |
| Placement / submersion | 50 % roof or upper floor, 20 % tree/road/vehicle roof, 20 % wading or clinging, 10 % head-only; plus buried actors present in truth as not visible |
| Posture | 35 % upright, 20 % sitting, 15 % crouched/fetal, 15 % prone, 10 % supine, 5 % kneeling; 20 % of upright actors waving |
| Occlusion (visible fraction) | uniform over 100 %, 75 %, 50 %, 25 %, 10 % |
| Altitude | 30, 45, 60, 90, 120 m (weighted 20/30/30/10/10) |
| Gimbal pitch | nadir 60 %, −60° 25 %, −45° 15 % |
| Time of day | day 45 %, dusk/dawn incl. crossover 25 %, night 30 % |
| Weather | dry 50 %, light rain 25 %, heavy rain 10 %, fog Cat I/II 15 % |
| Water | turbidity 60–900 NTU; glint on/off by sun elevation |
| Debris density | low/medium/high; distractors 0–10 per 100 m² |
| Animals | 0.2–2 per human, cattle/goat/dog |
| Clusters | rooftop groups of 1, 2–5 and > 10 |

Every attribute is written into the frame's metadata so the evaluation slices are free.

### 6.3 Annotation guideline for partially visible subjects (deliverable)

The guideline below is the project's rule set. It is grounded in published conventions (VisDrone occlusion and truncation bins, COCO `iscrowd`, KITTI occlusion levels, CrowdHuman visible vs full-body boxes, CityPersons ignore regions for mannequins and reflections, TinyPerson `uncertain`/`ignore` and its "sea person" rule, NOMAD's visibility budget, POP's tight visible-region boxes) and applies equally to real and synthetic data.

**Box extent.** Draw the **visible-extent box**: the tightest axis-aligned box around all visible pixels of the individual, including a lone limb or head. Do not draw amodal full-body boxes (a limb-only prediction would fail IoU 0.5 against an amodal box of a mostly buried body; CrowdHuman itself warns full-body boxes suffer high annotator variance). If the body is split into disconnected visible parts by debris or water and the gap is under one body-width, draw one box over the union; otherwise box the larger part and set `occlusion = 2`. At the frame edge box only what is inside; set `truncated = 1` if ≥ 50 % of the visible body is cut (VisDrone drops > 50 % truncation from evaluation).

**Minimum visible fraction and size.** Label whenever at least one identifiable body part (head, torso, arm, leg, hand or foot with a limb) is visible, regardless of fraction; record the fraction in `occlusion` instead of dropping the instance. Minimum size ≥ 8 px on the longer side at the resolution the detector sees (one stride-8 feature cell; TinyPerson's [2, 8] bin is near-hopeless). With 1024 tiles cut from 4K without resizing, that is ≥ 8 px in the 4K frame; anything 4–8 px gets `uncertain = 1` rather than being skipped.

**Classes.** `human` (any person, any pose, alive or not: do not judge). `animal` (dog, cat, goat, sheep, cattle, buffalo, pig, horse, poultry; no species split; free-text `animal_type` when obvious). Hard-negative classes, labelled so they enter training as negatives and are scored separately but excluded from survivor metrics: `mannequin_or_statue`, `clothing_only`, `debris_limb_like`, `animal_carcass`, `reflection`.

**Uncertain and ignore.** `uncertain = 1` on a box the annotator cannot resolve (TinyPerson "uncertain", Caltech "Person?"): neither positive nor negative in evaluation; detections overlapping it are dropped before scoring and missing it is not a miss; used as a positive only for the recall-first model. `ignore` regions (polygon, no class) for unresolvable clutter, blur or saturation; matched by intersection-over-detection.

**Attributes per box.** `occlusion ∈ {0: fully visible, 1: 1–50 % hidden, 2: > 50 % hidden}` (VisDrone bins, so VisDrone and NOMAD merge; estimate relative to the whole body). `truncated ∈ {0, 1}`. `pose ∈ {standing, sitting, prone, supine, half_submerged, trapped, unknown}` with `half_submerged` = water covers ≥ 50 % of the body (TinyPerson's rule) and `trapped` = pinned under or inside debris with ≥ 1 limb visible. `context ∈ {open_ground, water, debris, vegetation, structure, vehicle}`. `visibility_modality ∈ {rgb_only, thermal_only, both}` when a thermal frame is paired: box each modality independently; a thermal-only person gets a copied box on the RGB frame with `thermal_only` and `uncertain = 1`, so the RGB-only evaluation ignores it and the fusion evaluation counts it. `visible_parts` (multi-select head/torso/limb). `group_count` (see below). `source ∈ {auto:<model,prompt,threshold>, manual}`.

**Clusters.** One box per individual by default, even when overlapping. If ≥ 5 individuals are heavily overlapping and physically touching and cannot be separated (Open Images' group threshold), draw one `human_group` box with `group_count`; evaluate it like COCO `iscrowd = 1` (detections inside are neither TP nor FP; the group counts as one recall target if any detection overlaps it at IoD ≥ 0.5).

**QA.** 10 % of tiles, stratified by flight and by occlusion level, get a second blind annotation; targets: box match at IoU ≥ 0.5 recall ≥ 0.9 for occlusion 0–1 and ≥ 0.75 for occlusion 2, class agreement ≥ 0.95, occlusion exact agreement ≥ 0.8; disagreements adjudicated by a third person and fed back as guideline examples. Sanity checks before export: no box < 4 px, none outside the image, no duplicates at IoU > 0.9, class histogram per flight.

**Files.** Master COCO JSON at frame level (visible-extent `bbox`, `iscrowd = 1` for groups and ignore regions, attributes inside each annotation, per-image `meta` with `flight_id`, `altitude_m`, `gimbal_pitch_deg`, `modality`, `tile_offset`, and the §6.2 scene attributes for synthetic frames). Training export as YOLO txt per tile in two variants: clean (`human`, `animal`; uncertain, ignore and groups removed) and recall-first (adds the hard-negative classes). Ship the export script and 8–12 example crops per rule in `ANNOTATION_GUIDELINE.md`.

### 6.4 Auto-label and review loop for real footage

**Tool: X-AnyLabeling v4.0.6** (GPL-3.0, 10.4k stars, Windows CUDA 12 executable, released 5 Sep 2026): one desktop app with Grounding DINO, SAM 3, YOLO-World/YOLOE, RF-DETR and YOLO26 built in, text-prompt batch auto-labelling over a folder, and COCO/YOLO export. Label Studio (Apache-2.0, pip) is the reserve if more than three reviewers work concurrently; CVAT on Windows needs Docker Desktop + WSL2 + NVIDIA WSL drivers + Nuclio for auto-labelling and is not worth it for a sprint; LabelImg is archived; Roboflow Auto Label publishes imagery on the free plan.

Loop: sample 1–2 FPS per flight and drop near-duplicates → tile 4K frames to 1024 with 25 % overlap (a 25-px prone person becomes 6 px if the whole frame is resized to 1024; in a tile it stays 25 px) → pass A: Grounding DINO or SAM 3 with a ≤ 5-phrase vocabulary (`person . animal .`, then `person lying down . partially visible person . human head . dog . cattle`) at a low box threshold (~0.2); pass B: YOLO-World/YOLOE or `yolo26x.pt` for `person` at `conf 0.15`; union with NMS → human review on the tile folder (delete, tighten to visible extent, add misses, set attributes; one reviewer per flight) → export COCO (frame) and YOLO (tile) → **round 2 is model-in-the-loop:** train the project's own YOLO on round-1 tiles and pre-label the rest with it at `conf 0.1`, because a fine-tuned detector beats every zero-shot open-vocabulary model on aerial persons by a wide margin (a 2026 comparison measured OWLv2 at recall 0.25 and Grounding DINO at recall 0.04 zero-shot on an aerial benchmark; in-domain training adds ~+31 AP50). Expect 150–300 tiles per reviewer-hour (estimate).

### 6.5 Mixing and splits

*This mixing rule applies to the transferable model only. The demo model is trained on 100 % simulator frames by design — see §5.5c.* Real:synthetic ≈ 60:40 in the fine-tune stage, synthetic never above 40 % of the mix; pose-diverse synthetic first (Archangel finding). Splits are **by flight or by scenario seed, never by frame**, to avoid leakage between near-identical frames. Held-out: one simulator clip (all three zones, noise-injected telemetry, buried actors, a crossover-window segment) and one real DJI clip labelled every 10th frame. Every reported number carries its slice.

---

## 7. Full user flow

Three roles use the system: the **incident commander** (owns segments, priorities and decisions), the **drone operator** (flies, takes over, hands back), and the **analyst** (runs the pipeline, checks evidence, exports). In a hackathon demo one person can play all three; the screens are the same.

**Step 0 — Incident setup (commander, 5 min).** Open the map. Draw the area of operations. Load the flood polygon and the deposit-fan polygon (in the demo, from the scenario; in the field, from satellite or a first overflight). Mark burial polygons as *aerial search cannot clear*. Drop last-known-position pins from phone data or reports. The probability-of-area prior renders. Draw search segments over it (or accept the auto-segmentation) and set an initial priority per segment. Choose the mission profile: detection pass at 45–60 m nadir at ≤ 8 m/s, or thermal cueing at 90–120 m. The screen shows the expected GSD, pixels on a 0.45 m target, swath, and the time to cover each segment.

**Step 1 — Launch (operator, 1 min).** Start the simulator scenario (packaged build) or connect the real feed. The pipeline reports decode FPS, telemetry source, per-frame latency and the fallback state (RGB-only or fused). Arm, take off, AUTO begins the first segment's pattern. The map shows the live drone, its footprint and the trail; the search-quality raster starts filling under the footprint.

**Step 2 — Detection to record (automatic, seconds).** A candidate appears as a faint marker after its first hit and becomes a numbered record after three hits in two seconds. The record card shows the evidence thumbnail from the best frame, class, fused confidence, whether thermal agreed, motion state, count, position with the error ring, and the priority components. Records are ranked in the side list; nothing is removed automatically.

**Step 3 — Takeover (operator, at will).** The operator deflects a stick or presses TAKEOVER; the mode chip flips to MANUAL and the trail changes colour. They fly closer, look under an eave, or orbit a roof. Every frame in MANUAL still runs detection, geolocation and coverage. RESUME hands control back; the mission re-plans from the current pose and continues. HOLD and RTL are always available. All transitions are in the telemetry and visible on the timeline.

**Step 4 — Orbit and evidence (automatic or operator-approved).** For a confirmed record, the system offers "orbit to confirm"; if accepted (or if AUTO-orbit is enabled), the drone centres the target at nadir and dwells; the record's position error shrinks as observations accumulate, the thumbnail improves, and count and motion estimates update.

**Step 5 — Commander review (continuous).** The commander sorts by priority, opens records, and marks them *confirmed*, *dispatched* (with a team name) or *dismissed* (with a reason); dismissed records remain in the log under a separate layer. The commander looks at the search-quality raster, reads the generated recommendation for each segment ("POD 0.41 after two passes; night thermal pass recommended before 06:00"), and re-prioritises. The system never changes a segment's status by itself.

**Step 6 — Revisit and hand-off.** Stale records and low-POD cells are queued as revisit waypoints; the operator approves the queue. Exports are one click: GeoJSON and KML/KMZ with thumbnails for the rescue teams, Cursor-on-Target events to a TAK server for teams running ATAK/WinTAK, and a GeoPackage for GIS. If the link is down the exports are written locally and the outbox synchronises when it returns; the status bar shows the queue depth.

**Step 7 — After the flight (analyst).** Run the replay harness on the recorded flight; the evaluation dashboard shows recall at IoU 0.5 and 0.25, FP/min at the detection and record level, dedup precision/recall and duplicate rate, geolocation error against known positions, the POD reliability diagram, and the missed-detection browser (FiftyOne) grouped by slice. The report is regenerated per model version.

---

## 8. Feature list, mapped to requirements

| # | Feature | Requirement | MVP / Stretch |
|---|---|---|---|
| F1 | Flood-valley scenario with three zones, scriptable weather, time of day, flood level, actor spawners with pose, submersion and thermal state | A13, R11 | MVP |
| F2 | Autonomous coverage patterns (boustrophedon, expanding square), orbit-on-detection, revisit queue, battery/geofence constraints | A13 | MVP (orbit: stretch if time is short) |
| F2b | Decision planner: Koopman closed-form effort allocation + greedy action scoring over the probability-of-success surface; explainable choices; degrades to the plain pattern when the prior is flat | A14, A13 | Stretch (depends on F16) |
| F3 | Gamepad takeover and hand-back with logged mode switches; HOLD and RTL | A13 | MVP |
| F4 | PX4 SITL path with QGroundControl joystick and MAVSDK mission | A13 | Stretch |
| F5 | Synthetic dataset with visible/amodal boxes, visible fraction, pose, submersion, thermal frames, per-frame telemetry and scene attributes | R11 | MVP |
| F6 | Annotation guideline and X-AnyLabeling review loop for real footage | R11 | MVP |
| F7 | Ingest from simulator export, DJI SRT, MAVLink, ULog; time alignment; NVDEC decode; decimation | R1 | MVP (SRT + sim); MAVLink/ULog stretch |
| F8 | Tiled YOLO26 RGB detector at native 4K resolution; TensorRT export; operating threshold frozen for recall ≥ 0.92 | R2, R9 | MVP |
| F8b | **Demo model**: single fine-tune on simulator frames only, evaluated on held-out simulator scenes, randomisation off; produces the acceptance figures (§5.5c) | R2, A13 | **MVP** |
| F8c | Transferable model: three-stage recipe on real + synthetic, evaluated on a real clip; measures the domain gap | R2 | Stretch |
| F9 | Thermal YOLO26n + WBF/ProbEn late fusion with structural RGB-only fallback; per-altitude homography for real payloads | R3 | MVP (fusion); registration stretch |
| F9b | Radiometric thermal on the stills/dwell path and throughout the simulator: absolute-temperature filter for sun-heated distractors, and a measured fusion weight (§5.5b) | R3 | MVP in sim; stretch on real hardware (video is AGC-only) |
| F10 | Crop verifier, multi-output: `is_real` + posture + submersion + occlusion per candidate. Doubles as the triage-attribute predictor (§5.5a) | R2, R6 | **MVP** (promoted: it feeds the triage ranking, not just precision) |
| F11 | BoT-SORT/TrackTrack with camera-motion compensation, processed-frame buffers, 3-hit confirmation | R4 | MVP |
| F12 | Geo-space deduplication (DBSCAN at 2 × CE90), count and motion estimates, stable record IDs, stale status | R4, R6 | MVP |
| F13 | Geolocation chain with DroneModels intrinsics, DEM option, per-record `h_acc_m`, published error budget at 60 m | R5 | MVP |
| F14 | Triage score with visible components; GeoJSON/KML/KMZ exports with evidence thumbnails | R6 | MVP |
| F15 | MapLibre offline map with PMTiles basemap, live WebSocket updates, record cards, timeline | R6, R12 | MVP |
| F16 | Search-quality raster (per-cell POD), burial polygons, segment assignments and generated recommendations | A14, R10 | MVP (raster); generated text stretch |
| F16b | Per-presentation coverage layers (prone, cluster, upright, wading, head-only, limb-only, buried) with a zone-weighted default view and a layer selector (§5.3b) | A14 | MVP for two layers (body, limb-only); full stack stretch |
| F17 | Cursor-on-Target export to TAK | R6 | Stretch |
| F18 | SQLite record log + persist-queue outbox + idempotent cloud upsert; FastAPI `/detect` fallback | R7, R9 | MVP (outbox); cloud detect stretch |
| F19 | Evaluation script: recall@0.5/0.25, FP/min (detection and record level), dedup metrics, geolocation error, POD calibration, slices, FiftyOne error browser | R8 | MVP |
| F20 | Jetson Orin engine build and measured latency table | R9 | Stretch (hardware permitting); RTX 4060 numbers are MVP |
| F21 | Guardrails: no auto-close, no delete, dismiss-with-reason, no identification of individuals | R10 | MVP |

---

## 9. Build plan and demo script

The order is chosen so that a demonstrable slice exists at the end of every day and the riskiest unknowns are retired first.

| Day | Goal | Exit criterion |
|---|---|---|
| 1 | Toolchain and unknowns: install UE 5.8 + VS 2026 + Cosys plugin (fallback to Project AirSim by midday if it fails); Blocks flies; Xbox takeover works; 4K/segmentation/infrared captures work; water hides submerged pixels in the label pass; measure editor RAM and capture FPS; pull NOMAD, SARD, WiSARD sample, HIT-UAV, C2A; run YOLO26s on one 4K frame with TensorRT on the 4060 | A frame with RGB + thermal + instance mask + telemetry line saved from a gamepad-flown drone |
| 2 | Flood level: terrain, water, debris, three zones, actor spawners, weather and time-of-day scripting, thermal table; capture script producing labelled frames; start aerial pre-training (overnight) | 1,000 labelled synthetic frames; training running |
| 3 | Pipeline spine: ingest (sim export + SRT), tiled detection, tracking, geolocation with noise injection, SQLite log, GeoJSON out; map page with live markers | Live sim flight produces ranked markers on the map |
| 4 | Dedup, triage score, search-quality raster, evidence thumbnails, exports; thermal detector training; fusion; RGB-only fallback switch | One survivor = one record across two passes; POD raster fills; fusion on/off toggles cleanly |
| 5 | Autonomy: patterns, orbit, revisit; **decision planner if the coverage map is working**; takeover state machine polished; outbox and reconnect; evaluation script and dashboard; held-out clips labelled | Evaluation report v1 with slices; offline/online sync demonstrated |
| 6 | Fine-tune on synthetic + real; threshold selection; FP/min; missed-detection analysis; real DJI clip through the replay harness; Jetson build if hardware exists | Report v2; README; dataset card and annotation guideline |
| 7 | Demo rehearsal, packaging, buffer for the unverified items in §10 | Demo runs twice from a clean start |

**Demo script (8 minutes).** (1) Map with the Wayanad-type scenario, burial polygons hatched, prior and segments drawn. (2) Launch; AUTO pattern over the flooded settlement; first records appear with thumbnails; the POD raster fills. (3) Operator takes over on the gamepad, flies under an eave, finds a prone person the pattern missed; hands back; the mission resumes. (4) Switch time of day to pre-dawn; thermal-positive records rank up; pull the thermal feed: the RGB-only fallback keeps working. (5) Second pass over the same roof: still one record, count updated. (6) Disconnect the network: outbox depth rises; reconnect: it drains; export KML and open it. (7) Show the evaluation dashboard for the held-out clip, saying the domain out loud: "94 % recall at IoU 0.5, in simulation, on the nominal slice", then the slices where it drops (head-only at 90 m), stated honestly. (8) Close on the guardrail: the commander dismisses a record with a reason; nothing is deleted; no segment is ever "closed".

---

## 10. Risks, unknowns and day-1 tests

**Risks with mitigations.**

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| Cosys-AirSim plugin fails on the installed UE 5.8 point release or VS 2026 toolset | medium / high | Pin the MSVC toolset via `BuildConfiguration.xml`; switch to Project AirSim on UE 5.7 + VS 2022 by midday of day 1 |
| 16 GB RAM: editor + captures thrash | medium / medium | Packaged builds for data runs; Lumen/Nanite/VSM off; small AOI; never train while the engine runs |
| 4K capture too slow | medium / medium | 1080p video + 4K stills; measure first |
| Thermal realism too low to train on | medium / medium | Per-object infrared on day 1, post-process material later; validate against WiSARD/HIT-UAV; thermal is a cueing sensor, RGB stays primary |
| Recall ≥ 90 % not reached on hard slices | high / medium (reputational if hidden) | Report per slice; state the operating envelope (≤ 60 m, tiled RGB, in-domain fine-tune); recall at IoU 0.25 as the secondary metric |
| Orin end-to-end 2–3× the published numbers | high / medium | Batch tiles, GPU pre/post-processing, INT8, C3 strategy on Nano; the 300 ms target is stated per configuration |
| Licences (AGPL Ultralytics/BoxMOT; non-commercial POP/AFO/C2A; DeepDataSpace API) | low / medium | Open repo makes AGPL fine; flag Enterprise for closed deployment; `roboflow/trackers` + RF-DETR as permissive swaps; data card records each licence |
| DJI encrypted logs (v13+) block gimbal angles on consumer drones | medium / low | Apply for the DJI developer key early; assume nadir if absent; M30T SRT carries gimbal angles |
| Wayanad casualty and event details vary by source | certain / low | Quote ranges; the scenario is a *type*, not a reconstruction |

**Unverified items to test or confirm on day 1** (consolidated from all research files):

1. Cosys 3.4.1 precompiled plugin loads on your UE 5.8.x with VS 2026 (release notes mention VS 2022 17.14 builds); editor RAM with Blocks open (expect 8–11 GB).
2. Instance segmentation hides submerged pixels under single-layer water and under a translucent plane; `IgnoreMarked` second-camera trick renders the unoccluded person.
3. Infrared image type renders after the ID-remap with your own temperature table; capture FPS at 4K uncompressed vs PNG.
4. Xbox `RemoteControlID`, `AllowAPIAlways` semantics, handover latency of `enableApiControl(False)`, whether `moveByRC` is needed; the same pad through QGC if PX4 is attempted.
5. Weather visible in Scene captures and absent from Segmentation; time-of-day API needs a sky sphere in the level.
6. Detection-API box flicker on skeletal meshes in prone poses and when 70 % submerged.
7. Cesium streaming cost and VRAM for the Wayanad AOI; segmentation colours of Cesium tiles.
8. Actual end-to-end ms on the 4060 for C1/C2/C3 with YOLO26n/s/m batch-6 at 1280; on Orin if available; JetPack 6.2 vs 7.2.1 compatibility of TensorRT export and decode paths; PyNvVideoCodec on Orin (only Thor is listed).
9. Effective HFOV of 4K 16:9 video vs the 4:3 sensor for the DJI model used; per-frame `focal_len` field.
10. Compass calibration and a known-point check for heading error (the dominant oblique term); `t_offset_s` estimation on a MAVLink clip.
11. Camera-motion compensation over rippling water: sparseOptFlow vs none vs telemetry homography, compared by ID switches.
12. `track_buffer` in processed-frame units and the two-survivors-5 m-apart dedup case (they merge unless `h_acc_m` < 2.5 m, i.e. RTK, and then report count ≥ 2).
13. Thermal-to-RGB registration on real M30T/M3T footage (or geolocate each independently and fuse in geo space).
14. Ultralytics 4-channel TensorRT export was not exercised (not needed for the late-fusion design; relevant only to the optional stage-2 fusion head).
15. Licence texts that could not be read: MetaHuman EULA (login-walled), SARD DataPort licence, MOBDrone/AIDER/AIResQ Zenodo licences (Zenodo returned 504 all day), RescueNet and HIT-UAV licence conflicts, C2A licence, WAID licence, DeepDataSpace pricing.
16. Domain numbers with no primary source: fraction of flood survivors on roofs vs in water and the posture distribution (proposed mixes only); LWIR attenuation per rain rate; immersion contrast falling to ~⅓ (industry article); microbolometer integration time; SAR mission speeds and overlaps; the Drone Rules 2021 disaster-relief exemption clause and whether an NDMA drone SOP exists; FEMA/USCG human-in-the-loop wording.
17. Control-stack items not re-fetched today: MAVSDK-Python Windows support and plugin names, QGroundControl joystick mapping details, `COM_RC_IN_MODE` semantics, DroneKit status, Fields2Cover licence and Windows build, pytak/FreeTAKServer status and licences.

---

## Appendix A. Reuse ledger (verified 10 Sep 2026 unless marked)

| Component | Name | URL | Status | Licence | Verdict |
|---|---|---|---|---|---|
| Engine | Unreal Engine 5.8 | unrealengine.com | Released 17 Jun 2026; 32 GB RAM recommended, 8 GB+ VRAM; VS 2026 | Epic EULA | ADOPT |
| Sim plugin | Cosys-AirSim | github.com/Cosys-Lab/Cosys-AirSim | 420★, push 13 Aug 2026, release 5.8-v3.4.1 (17 Jul 2026), precompiled Windows plugin | MIT | ADOPT (primary) |
| Sim plugin | Project AirSim (IAMAI) | github.com/iamaisim/ProjectAirSim | 836★, v1.0.1 (7 Sep 2026), UE 5.7 plugin, prebuilt envs | MIT | ADAPT (fallback) |
| Sim plugin | Colosseum | github.com/CodexLabsLLC/Colosseum | 670★, **archived 11 Jul 2026**, UE 5.6 main | MIT | REJECT (reference docs) |
| Sim | microsoft/AirSim | github.com/microsoft/AirSim | 18.5k★, frozen at v1.8.1 (2022) | MIT | REFERENCE (API docs) |
| Sim | Project AirSim / Cosys docs on PX4 WSL2 | in repos | verified | — | REFERENCE |
| Terrain | Cesium for Unreal | github.com/CesiumGS/cesium-unreal | 1,236★, v2.29.1 (1 Sep 2026), UE 5.6–5.8 | Apache-2.0 | ADOPT (optional) |
| Sim (rejected) | Isaac Sim + Pegasus | github.com/PegasusSimulator/PegasusSimulator | 875★, v5.1.0; Isaac Sim needs RTX 4080/16 GB VRAM/32 GB RAM | BSD-3 | REJECT (hardware) |
| Sim (rejected) | Flightmare / FlightGoggles / RotorS / Aerial Gym / OmniDrones | uzh-rpg, mit-aera, ethz-asl, ntnu-arl, btx0424 | Linux-only and/or stale | MIT/BSD | REJECT |
| Sim (rejected) | Unity Perception | github.com/Unity-Technologies/com.unity.perception | 994★, discontinued, last release 2022 | Apache-2.0 | REJECT |
| Sim (rejected) | Webots R2025a | github.com/cyberbotics/webots | 4,604★, Mavic 2 Pro model, 400×240 camera | Apache-2.0 | REJECT (panic fallback only) |
| Sim (rejected) | PteroSim | github.com/PteroLabsAI/PteroSim-UAV-Simulator | v0.2.0 (Jul 2026), BGR camera only | proprietary, free non-commercial | REFERENCE |
| Sim (rejected) | AirGen/GRID, Duality Falcon | docs.generalrobotics.dev; duality.ai | cloud-only free tiers | — | REJECT |
| Planning | Fields2Cover | github.com/Fields2Cover/Fields2Cover | 887★, push 4 Sep 2026 | BSD-3-Clause | ADOPT (swath geometry) |
| Planning | ethz-asl/polygon_coverage_planning | github.com/ethz-asl/polygon_coverage_planning | 656★, push 13 Nov 2023, ROS 1 | GPL-3.0 | REFERENCE |
| Planning | SAROPS (USCG); Koopman/Stone optimal allocation | dco.uscg.mil; Stone 1975 | operational since 2007; classical | not open source; n/a | REFERENCE; ADOPT the formula |
| Planning | Slope Probability Search (Sensors 2025) | PMC12787386 | no code released | article licence | ADAPT the idea (88.9 % vs lawnmower 75.4 %) |
| Planning | drone-swarm-search; dmar-bonn/ipp-rl-3d; uzh-rpg/agile_flight | see §5.3a | 81★/34★/196★ | MIT | REFERENCE; REFERENCE; REJECT |
| Flight stack | PX4 Autopilot | github.com/PX4/PX4-Autopilot | v1.18.0-beta2 (Aug 2026); Windows = WSL2 | BSD-3 | OPTIONAL |
| Flight stack | ArduPilot | github.com/ArduPilot/ardupilot | 4.7.1 (Sep 2026); AirSim page archived | GPL-3.0 | REJECT with AirSim |
| Assets | Mixamo; MetaHuman Creator; Fab/Megascans | adobe.com/mixamo; UE ≥ 5.6; fab.com | Mixamo free (Sep 2026); MetaHuman EULA unverified; Megascans paid since 1 Jan 2025 | various | ADOPT / verify |
| Detector | Ultralytics YOLO26 | github.com/ultralytics/ultralytics | v8.4.146 (9 Sep 2026), 61.5k★ | AGPL-3.0 | ADOPT |
| Detector | RF-DETR | github.com/roboflow/rf-detr | v1.10.1 (7 Sep 2026), 9.4k★ | Apache-2.0 (N/S/M/L) | ADOPT (alt) |
| Detector | D-FINE / DEIM | github.com/Peterande/D-FINE; Intellindust-AI-Lab/DEIM | 3.3k★ / 1.6k★, 2026 pushes | Apache-2.0 | ADAPT |
| Detector | DEIMv2 | github.com/Intellindust-AI-Lab/DEIMv2 | 2.0k★ | non-commercial | REFERENCE |
| Detector | RTMDet (mmyolo/mmdet) | github.com/open-mmlab | frozen since 2024; no Windows mmcv wheels | GPL-3.0 / Apache-2.0 | REJECT |
| Detector | EfficientDet | github.com/google/automl | archived 2021 | Apache-2.0 | REJECT |
| Tiling | SAHI; supervision | github.com/obss/sahi; roboflow/supervision | v0.12.6; v0.30.2 | MIT; MIT | ADOPT |
| Fusion | Weighted Boxes Fusion | github.com/ZFTurbo/Weighted-Boxes-Fusion | 1.8k★, push Jul 2026 | MIT | ADOPT |
| Fusion | ProbEn | github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling | 176★ | Apache-2.0 | ADOPT (score rule) |
| Fusion (stage 2) | YOLOv11-RGBT | github.com/wandahangFY/YOLOv11-RGBT | 720★, Dec 2025 | AGPL-3.0 | ADAPT (optional) |
| Fusion (rejected) | CFT, ICAFusion, DAMSDet, TFDet, MS-DETR, MMPedestron, Scarf-DETR | see research file | heavy and/or non-permissive | mixed | REFERENCE |
| Auto-label | X-AnyLabeling | github.com/CVHub520/X-AnyLabeling | v4.0.6 (5 Sep 2026), 10.4k★, Windows CUDA exe | GPL-3.0 | ADOPT |
| Auto-label | Label Studio; CVAT; Autodistill; Roboflow | HumanSignal; cvat-ai; autodistill; roboflow.com | 1.23 / 2.74.1 / stale 2024 / SaaS | Apache / MIT / Apache / SaaS | reserve / REJECT on Windows / REJECT / REJECT |
| Zero-shot labelers | Grounding DINO (HF), OWLv2, SAM 2, SAM 3, YOLO-World/YOLOE | IDEA-Research; google; facebookresearch; AILab-CVC | open weights; SAM 3 gated, custom licence | Apache / Apache / Apache / SAM licence / GPL-AGPL | offline labelling only |
| Tracking | Ultralytics trackers (BoT-SORT, TrackTrack…) | docs.ultralytics.com/modes/track | in 8.4.146 | AGPL-3.0 | ADOPT |
| Tracking | roboflow/trackers | github.com/roboflow/trackers | 3.8k★, v2.6.0 (Aug 2026) | Apache-2.0 | ADOPT (alt) |
| Tracking | BoxMOT | github.com/mikel-brostrom/boxmot | 8.3k★, v25.0.0 (Sep 2026) | AGPL-3.0 | ADAPT (A/B only) |
| Tracking | UCMCTrack; UAVMOT | corfyi; LiuShuaiyr | 2024 research; unlicensed | MIT; none | REFERENCE; REJECT |
| ReID | torchreid/OSNet, fast-reid, CLIP-ReID | KaiyangZhou; JDAI-CV; Syliz517 | stale | MIT/Apache | REJECT (aerial scale) |
| Metrics | py-motmetrics; TrackEval | cheind; JonathonLuiten | v1.4.0, push Jul 2026; push Jul 2024 | MIT | ADOPT; ADAPT |
| Geolocation | Theta-Limited DroneModels | github.com/Theta-Limited/DroneModels | push 1 Sep 2026 | Apache-2.0 | ADOPT |
| Geolocation | OpenAthena (Python) | github.com/Theta-Limited/OpenAthena | archived; export-control addendum | custom | REFERENCE only |
| Geolocation | roboflow/dji-aerial-georeferencing | github.com/roboflow/dji-aerial-georeferencing | 341★, 2023, JS | Apache-2.0 | REFERENCE |
| Geolocation | OpenDroneMap/WebODM, OpenSfM | OpenDroneMap; mapillary | batch photogrammetry | AGPL / BSD | REJECT (not real-time) |
| DEM | Copernicus DEM GLO-30; dem-stitcher; rasterio; pyproj | AWS registry; ACCESS-Cloud-Based-InSAR; rasterio; pyproj4 | < 4 m LE90; v3.2.0; 1.5.1; 3.8.0 | free / Apache / BSD / MIT | ADOPT |
| DEM | CartoDEM (Bhuvan); SRTM; py3dep | bhuvan.nrsc.gov.in; USGS; hyriver | ~8 m LE90, 20 tiles/day; EGM96; US-only | — | optional / reference / REJECT |
| Thermal radiometry | DJI Thermal SDK v1.8; thermal_parser | dji.com/downloads; github.com/SanNianYiSi/thermal_parser | SDK Aug 2025; parser 104★, push 31 Mar 2025 | DJI EULA; MIT | ADOPT (stills path) |
| Telemetry | dji-drone-metadata-embedder (SRT formats); dji-log-parser; pymavlink; pyulog | CallMarcus; lvauvillier; ArduPilot; PX4 | v2.15.0; v0.5.7; v2.4.49; v1.2.4 | MIT; MIT; LGPL; BSD-3 | ADOPT |
| Decode | PyNvVideoCodec; torchcodec; PyAV; decord | NVIDIA; pytorch; PyAV-Org; dmlc | v2.2.2; v0.16.0; v18.1.0; stale | MIT; BSD-3; BSD-3; Apache | ADOPT; ADAPT; ADAPT; REJECT |
| Edge | JetPack 6.2.x / 7.2.1; DeepStream 9.1; jetson-inference; torch2trt | developer.nvidia.com; dusty-nv; NVIDIA-AI-IOT | 7.2.1 Aug 2026; DS 9.1 needs JP 7.2; torch2trt stale | various | ADOPT / ADAPT / REFERENCE / REJECT |
| Serving | FastAPI + ONNX/TensorRT; Triton; BentoML; Roboflow Inference | — ; triton-inference-server; bentoml; roboflow | v2.72; v1.4.39; v1.5.2 | — ; BSD-3; Apache; Apache+enterprise | ADOPT; REFERENCE; REFERENCE; REJECT |
| Outbox | persist-queue | github.com/peter-wangxu/persist-queue | v1.1.0 (Oct 2025) | BSD-3 | ADOPT |
| Map | MapLibre GL JS; PMTiles/Protomaps; Leaflet; deck.gl; CesiumJS; kepler.gl; folium/leafmap/streamlit-folium | maplibre; protomaps; Leaflet; visgl; CesiumGS; keplergl; python-visualization/opengeos | v6.9.0; daily builds; v1.9.4; v9.4.0; 1.145; 3.3-alpha; current | BSD-3; BSD; BSD-2; MIT; Apache; MIT; MIT | ADOPT; ADOPT; ADAPT; REFERENCE; REJECT; REJECT; ADAPT (eval) |
| Formats/DB | geojson; simplekml; fastkml; geopandas/pyogrio; DuckDB spatial; SQLite | jazzband; pypi; cleder; geopandas; duckdb | current; 1.3.6 (2021); current; current; current | BSD/LGPL/LGPL/BSD/MIT | ADOPT |
| C2 | pytak; ATAK-CIV; FreeTAKServer; WinTAK | snstac; deptofdefense; FreeTAKTeam; tak.gov | not re-fetched today | Apache-2.0 (pytak, per project) | ADOPT (stretch), verify |
| Eval | torchmetrics; pycocotools; FiftyOne | Lightning-AI; pypi; voxel51 | v1.9.0; 2.0.11 (Windows wheels); v1.21.0 | Apache; BSD; Apache | ADOPT |
| Datasets | see §6.1 | — | — | — | — |
| End-to-end SAR repos | Lacmus; Thermal-Imaging-Drone-SAR; SeaDronesSee YOLO forks; SUAS teams; misc | see research file | none complete or maintained | mixed/none | REFERENCE only |

## Appendix B. Formulas used

- **Ground sampling distance (nadir):** GSD = 2·h·tan(HFOV/2) / W_px. Horizontal FOV from a diagonal spec: HFOV = 2·atan(tan(DFOV/2)·W/√(W²+H²)).
- **Pixels on target:** n = L_target / GSD. Design floor: n ≥ 20 for "90 % recall plausible"; 8–20 cue-only.
- **Footprint (nadir):** width = 2·h·tan(HFOV/2), height = 2·h·tan(VFOV/2); line spacing = width × (1 − side overlap).
- **Motion blur:** blur_px = v·t_exp / GSD; keep t_exp ≤ GSD / v.
- **Geolocation sensitivities** (θ = ray angle from vertical): ε_ang = h·sec²θ·δθ; ε_yaw = h·tanθ·δψ; ε_h = tanθ·δh; ε_px = h·sec²θ·(δu/f); ε_gps = δp; ε_t = v·δt; CE(1σ) = √Σε²; CE90 ≈ 2.1 × CE(1σ). Ray–plane: t = h / d_z; NED offset → geodesic via WGS-84.
- **Optimal effort allocation (Koopman/Stone):** c_i* = max(ln p_i − λ, 0) with λ set so Σ c_i* = budget; cells below prior e^λ get zero effort.
- **Tactical action score:** value(a) = Σ_cells POA·ΔPOD(cell,a) / (t_transit + t_execute), with ΔPOD = e^(−k·C_before) − e^(−k·C_after).
- **Search theory:** coverage C = effort·W / area; POD = 1 − e^(−C) (random search) or 1 − e^(−1.3C) (USCG empirical); POS = POA × POD; posterior after a miss: POA′ ∝ POA·(1 − POD).
- **Per-cell search quality:** q_pass = R_slice(GSD, band, time, blur, view) × V(cell); C = Σ q_pass; POD = 1 − e^(−k·C), k calibrated, POD < 1 enforced.
- **Per-presentation coverage (§5.3b):** q_pass(cell,j) = R_slice(j; …) × V(cell,j); POD_j = 1 − e^(−k·C_j); displayed as POD_eff = Σ_j w_j(zone)·POD_j. Fully-buried is the j whose layer is identically zero.
- **Presentation critical dimensions (nadir):** prone/supine 1.70 m · cluster ~1.5 m · upright 0.45 m · wading 0.45 m · head-only 0.25 m · limb-only 0.18 m. Ceiling at ≥20 px, 4K wide: 120 / 120 / 60 / 60 / 33 / 24 m.
- **Dedup radius:** r = 2 × CE90 of the record's geometry; DBSCAN(haversine, eps = r / 6,371,000 rad, min_samples = 1).
- **Record confidence:** 1 − Π(1 − conf_track). **Triage score:** P(living) × w_class(t) × urgency × (1 + 0.1·count), components stored.
- **FP/min:** unmatched predictions (IoU ≥ 0.5) / (frames_processed / fps_processed / 60), detection-level and record-level.
- **Thermal state (sim):** T_skin(t) = T_water + (33 − T_water)·e^(−t/τ), τ_immersed ≈ 10–20 min, τ_air ≈ 60 min (proposed).

## Appendix C. The FMCW radar reference (IEEE document 11059497)

The IEEE link you gave resolves to *"FMCW Radar for Human Detection in Collapsed Structures for Post-Disaster Search and Rescue"* (Abdelhamid, Safa, Ismail, Mohamed; IWCMC 2025, Abu Dhabi; DOI 10.1109/IWCMC65282.2025.11059497; read in full on 10 Sep 2026). A 24 GHz FMCW radar with 200 MHz bandwidth (0.75 m range resolution) is mounted on a low-altitude UAV with a Jetson Nano; range profiles are computed by FFT per 16 × 64 frame, the beat-frequency peak's phase is extracted as a feature, and a Random Forest / Gradient Boosting classifier separates human-present from human-absent frames in a simulated rubble setup at **1.6 m and 2.0 m** standoff, reaching 93.3–93.9 % accuracy with majority voting over frames. Reflections weaken with altitude; the method targets micro-motion (respiration) of a person *under* rubble, which no camera can see.

How it fits this system: as a **close-in confirmation sensor for the deposit-fan zone**, flown at 1–2 m over a suspected burial point after the vision system and the prior map have proposed it, not as a survey sensor. It cannot be simulated meaningfully in a game engine, so the design treats it as a stub: a "radar check requested" action on a record, with a simulated binary result and a stated false-alarm rate, and with the Wayanad respiration-radar false alarm cited as the reason the result is evidence, not proof. The coverage map keeps burial polygons as "aerial search cannot clear" regardless of radar results.

## Appendix D. The gods-eye-view reference

`bilawalsidhu/gods-eye-view` (MIT, ~21.6k stars, trending Aug 2026) is a browser-based geospatial dashboard: vanilla JavaScript on CesiumJS with Google Photorealistic 3D Tiles, fusing live public feeds (11,000+ aircraft via OpenSky/adsb.lol, AIS vessels, 838+ satellites from CelesTrak, USGS earthquakes, NASA FIRMS fires, ~800 public CCTV feeds projected into 3D cities, OSM traffic, launches, radio) and an OpenAI Realtime voice interface, with keyless operation and an explicit refusal to build named-person search or face recognition. It contains no drone, video or vision code. What transfers: the pattern of streaming many geolocated feeds onto one map with per-source layers and a live update loop, and the ethics stance on individual identification, which this document adopts in §1.4. This project uses MapLibre (2D, offline) rather than CesiumJS because 3D globe rendering is not needed on a field laptop and offline PMTiles basemaps are simpler; if a 3D view of the valley is ever wanted, CesiumJS with the same terrain used in the simulator (Cesium World Terrain) is the natural extension.

## Appendix E. Key papers and evidence used

| Topic | Paper / source | What it established for this design |
|---|---|---|
| Occlusion in thermal SAR | POP dataset (Sci. Data 2025, PMC11840078) | Accuracy holds until > 70 % occlusion *if* occluded examples are trained on; COCO/HIT-UAV models lose ~0.70 mAP50 on it; boxes drawn on the visible region |
| Pixel size vs recall | HERIDAL / AIR (IJCV 2019; arXiv 2111.09406) | ~60 px persons at 2 cm GSD → 92.9 % recall (relaxed), 86.1 % at IoU 0.5 |
| Pose diversity | C2A (ICPR 2024, arXiv 2408.04922); "Exploring the Impact of Synthetic Data for Aerial-view Human Detection" (arXiv 2405.15203) | Pose-diverse synthetic data is the dominant lever; +28 AP on SARD with 50 real images |
| Thermal SAR baseline | AIResQ (Sci. Data 2026, PMC13338175) | Real SAR thermal at 50–120 m tops out at mAP50 0.55; summer contrast hardest |
| Thermal timing | Thermal crossover study (PMC5298629); Burke et al. 2019 (arXiv 1812.05498) | Crossover ~06:50 / ~18:05; ground indistinguishable from animals after ~08:00 |
| Small objects | SAHI (ICIP 2022, arXiv 2202.06934); TinyPerson (WACV 2020) | Slicing-aware fine-tuning +12.7 to +14.5 AP; IoU 0.25 as a SAR-relevant secondary metric |
| Detector landscape | YOLO26 docs and Jetson benchmarks; independent VisDrone benchmark (arXiv 2605.24831); RF-DETR (ICLR 2026) | Orin latencies; architecture moves little at 640 px, resolution moves a lot |
| Late fusion | ProbEn (ECCV 2022); Sensors 2021 attention-fusion table; Scarf-DETR (2025); MMPedestron (ECCV 2024) | Late fusion ≈ mid fusion accuracy; tightly coupled fusion collapses without thermal; modality dropout preserves RGB-only |
| Alignment | AR-CNN (ICCV 2019); TSRA (ECCV 2022); CoDAF (CVPR 2025); RGBT-Tiny (TPAMI 2025) | 20–35 % of boxes offset 0–15 px even in curated RGB-T sets |
| Tracking on UAVs | OATrack (Sensors 2026, PMC13517331); UCMCTrack (AAAI 2024) | Tracker choice moves HOTA ~1 point; detector quality dominates; track on the ground plane |
| Zero-shot on aerial | arXiv 2601.22164; OS-W2S (arXiv 2505.03334); UAV-OVD (arXiv 2509.06011) | Zero-shot open-vocabulary detectors have single-digit to tens-of-percent recall on aerial small targets; small vocabularies help 15× |
| Geolocation error | Sensors 2016 (PMC5298606); Sensors 2018 (PMC6263998); NUAA 2018; OpenAthena accuracy testing; Barber et al. 2006 | Error scales with h·sec²θ·δθ; yaw dominates oblique; 0.02–0.035 m per metre of slant range in the field |
| Search theory | Koester 2020 (Journal of SAR); USCG detection experiments 2004; DTIC ADA511658 | POD = 1 − e^(−C) / 1 − e^(−1.3C); sweep widths 64 m day / 22 m night; searchers cannot self-evaluate POD |
| Survival intervals | INSARAG Annex E; systematic review of 18 earthquakes (PubMed 16602260); cold-water survival charts (USCG) | No fixed cut-off; live rescues up to day 14; immersion exhaustion 3–12 h at 21–27 °C |
| Wayanad 2024 | npj Natural Hazards (s44304-024-00044-5); Springer Landslides (s10346-025-02484-0); Onmanorama, Tribune, Wikipedia | Event physics, search organisation, radar false alarm, thermal-drone negative, riverbank recoveries |
| Radar under rubble | Abdelhamid et al., IWCMC 2025 (DOI 10.1109/IWCMC65282.2025.11059497) | 24 GHz FMCW at 1.6–2.0 m detects human presence under simulated rubble at ~93 % accuracy |
