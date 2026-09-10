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
| **Mature broadleaf tree canopy** | **~half the frame.** Large rounded crowns, mid-green, many standing IN the water with only the crown showing | **absent — the single biggest gap** |
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
