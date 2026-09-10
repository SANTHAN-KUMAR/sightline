# Quality gate — mandatory for every lane, no exceptions

Every agent on this project must pass this gate before reporting work as done. It exists because this session
produced a long run of work that passed every programmatic check and was still wrong:

| What was "verified" | What was actually true |
|---|---|
| 71 survivors spawned, object names verified, instance colours unique, mask sizes measured | every one rendered as a bind-pose **T shape** — `set_animation()` does not serialise |
| poses authored, FK geometry asserted, 63 assets written | arms were **asymmetric** — one at the side, one straight out (mirrored bind axes) |
| characters imported, materials assigned, textures 2048², compression correct | every character was an **untextured white mannequin** — the FBX import bound no textures |
| 731-frame 4K dataset, labels extracted, boxes measured, JSON valid | the drone's **own propellers filled most frames**; one frame was pure sky; 42 boxes of ~110 |
| building materials wired, slots correct, parameters resolving | the master material **failed to compile**; everything rendered as the grey checker |
| terrain material compiled, 113 nodes, 36 texture samples | the **grass layer was 100 % replaced** by leaf litter; the tint provably did nothing |
| two survivors marked "aerial search cannot find them" (§2.7) | nothing was **on top of them**; they appeared in the mask |

Every one of those was found by looking at a picture, and none by reading a return value.

## The gate

**1. Produce a visual artifact, and actually look at it.**
No change to the scene, the simulator, the dataset, or any UI is done until you have rendered it and inspected
the image. Not the log, not the JSON, not the return value — the image.
* scene / materials / actors → `tools/scene/qa_shots.py`, then read the PNGs
* captured dataset → `tools/capture/contact_sheet.py`, then read the sheet
* map / web UI → headless screenshot (see `app/map/headless_check.mjs`), then read the PNG
* plots, rasters, exports → render and read them

**2. Write a check that can FAIL, and run it.**
A test that cannot fail is not a test. Assert against numbers from `docs/SOLUTION_DOC.md` or against a case you
worked out by hand. `tools/capture/validate.py` is the model: it exits non-zero and says *do not train on this*.

**3. Never infer success from an API return value.**
`True`, a non-empty list, "success", a saved asset — none of these mean the thing works. UE and Cosys-AirSim
both fail silently in the specific ways listed in `docs/CONTEXT.md`. Assume silent failure; prove otherwise.

**4. Report what you SAW, not what you did.**
"Ran the script, it returned success" is not a report. "Rendered three views; the roofs tile visibly at 45 m;
here is the file" is. Include the artifact paths so the orchestrator can look too.

**5. State the domain and the slice on every number** (project hard rule 5): `sim` or `real`, in the same
sentence. Never average across domains.

**6. If it is broken and you cannot fix it, say so plainly.** Never weaken a test to make it pass (hard rule 2).
A labelled stub in the report is worth far more than a green tick that is a lie.

## Definition of done

- [ ] visual artifact rendered **and inspected**, path in the report
- [ ] a failing-capable check written and run, output pasted
- [ ] no claim rests on a return value alone
- [ ] numbers carry their slice and domain
- [ ] stubs and known-broken items named explicitly
- [ ] the report says what was seen
