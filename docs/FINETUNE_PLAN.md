# YOLO fine-tuning plan (F8b, the demo model)

**Source of truth: `docs/SOLUTION_DOC.md` §5.5c** ("Simulation-first"), with §5.5 for the recipe mechanics,
§5.12 for evaluation and §6.3 for label semantics. This plan states what will be run, the number chosen for
every knob, and *why that number rather than the default*. Where our situation departs from the document, the
departure is named and argued rather than quietly absorbed.

Status: written while `seed47_alt55` is still flying. Everything here is decided; nothing waits on discussion.

---

## 0. What §5.5c actually asks for

> 1. Fly for the number — 40–50 m, tile the 4K frame at native resolution.
> 2. Fine-tune from COCO on simulator frames only. Stock YOLO26s, 30–50 epochs, 10–20k rendered tiles, 1024.
> 3. Split by scenario seed, never by frame.
> 4. Run a low confidence threshold and recover precision downstream.
> 5. Define the nominal slice before you measure it.

And the reporting rule: *"Every recall figure carries the domain it was measured in, in the same sentence as
the number."* Our headline is therefore of the form **"X % recall at IoU 0.5, in simulation, on the nominal
slice"** — never a bare number.

---

## 1. Where we comply, and where we do not

| §5.5c requirement | Our state | Verdict |
|---|---|---|
| Fly at 40–50 m | alt55 (acceptance) + alt35 (hard slice) | **Deviation, argued** — see below |
| Tile 4K at native resolution | 1024 px tiles, 0.2 overlap, no resize | ✅ exact |
| Stock YOLO26s from COCO | `yolo26s.pt` | ✅ |
| 30–50 epochs | **100 with patience 25** | **Deviation, argued** — see §4 |
| 10–20k rendered tiles | **~1–2k tiles, ~495 instances** | **Shortfall, argued** — see §3 |
| Split by scenario seed | seed 23 train / seed 47 val | ✅ structurally enforced |
| Randomisation off | off | ✅ |
| Low conf threshold | swept on val, not guessed | ✅ see §6 |

**The altitude deviation.** §5.5c says 40–50 m; `ALTITUDE_BANDS` puts the nominal slice at 40–60 m. alt55 sits
at the top of that band and is the acceptance pass; alt35 is *below* it and is reported as a named hard slice,
never pooled. alt80 was outside the band entirely, which is why killing it cost little. `SliceRoleError` in
`sightline/eval/campaign.py` refuses to pool passes of different roles, so this cannot be averaged away by
accident.

**The instance shortfall is the real risk in this plan and is not hidden.** The document sizes the recipe for
10–20k tiles. We will have roughly a tenth of that, and — more importantly — only ~495 labelled instances
across both seeds. Detection quality is bounded by *instances*, not tiles. The mitigations are in §4; the
honest expectation is in §7.

---

## 2. Dataset quality gate — RUN BEFORE TRAINING, blocking

Training does not start until every one of these passes. Each maps to a tool that exits non-zero, not to a
judgement call.

| # | Check | How | Blocking? |
|---|---|---|---|
| 1 | Structure + config | `sightline.detect dataset` writes `data.yaml`; assert `nc`, `names`, split paths resolve | yes |
| 2 | Seed-held-out split | `assert_no_seed_leak` — **already proven to refuse a single-seed build** | yes |
| 3 | Label correctness | `tools/capture/validate.py` (exits 2 on any skipped check) | yes |
| 4 | Boxes contain their actor | validate.py check C, against the recorded palette | yes |
| 5 | No impossible boxes | `assert_boxes_are_plausible` — fails on any box implying a person > 3 m on the ground | yes |
| 6 | Visibility | `tools/capture/check_mask_visibility.py` — every scored box must have an RGB silhouette | yes |
| 7 | Occlusion truth | `tools/scene/check_occlusion_truth.py` — ground truth vs placed geometry | yes |
| 8 | Duplicates | `quality_report.py` near-duplicate dHash | yes |
| 9 | Corrupt/missing samples | every tile image opens, every label line parses, no empty-image-with-label | yes |
| 10 | Class definitions | see §2.1 — resolved deliberately, not by default | yes |
| 11 | Class balance + coverage | pose/submersion/occlusion histograms per split | report |
| 12 | Buried survivors absent | validate.py fails if a §2.7 buried actor is labelled | yes |
| 13 | `ignore` honoured | `tests/test_ignore_is_honoured.py` — no ignored box becomes a YOLO target | yes |

### 2.1 Class definitions — a decision, not a default

`CLASSES = ("human", "animal")`, but the captured data contains **295 human and 0 animal instances**. A
two-class head trained with an empty class is a metric hazard: it can emit class-1 predictions that no ground
truth can ever match, and any naive mean over classes averages in a meaningless number.

**Decision: train the two-class head as defined, and report `human` AP explicitly rather than mAP over
classes.** Rationale: the inference pipeline, `urgency_for()` and the triage schema all key on the class
vocabulary, and silently retraining to `nc=1` would desynchronise the model from every consumer for a metric
convenience. The empty class is recorded in the data card and any `animal` prediction is counted as a false
positive, which is exactly what it is. If Ultralytics reports mAP over both classes, the `human` row is the
one quoted.

---

## 3. Dataset configuration

```
train: seed 23  — seed23_alt35 (347 frames) + seed23_alt55 (293 frames), 295 scored boxes
val:   seed 47  — seed47_alt55, held out entirely
```

* **Tiling**: 1024 px, overlap 0.2, cut at native resolution (`TileGrid`). 5 cols × 3 rows = **15 tiles per
  4K frame**. No resize — §5.5c is explicit that downscaling a 4K frame to 1024 turns a 25 px prone person
  into 6 px and "will defeat any model you can train".
* **Negatives**: `--negative-frac 0.05`. Empty tiles are the overwhelming majority (395 of 640 seed-23 frames
  carry no box at all); feeding them all would swamp the positives. 5 % keeps a background signal without
  drowning the loss. The 18 `ignore` boxes become background tiles by design — a survivor hidden under a fern
  genuinely is "nothing visible", and is a useful hard negative.
* **Split enforcement**: by scenario seed, structurally. `assert_no_seed_leak` refuses otherwise, and has
  already been observed refusing a single-seed build.

---

## 4. Training parameters — every departure from the default argued

| Knob | Value | Why this and not the default |
|---|---|---|
| `model` | `yolo26s.pt` | §5.5c names it. `s` also has a published Orin TensorRT table, which F20 needs. |
| `imgsz` | **1024** | One tile pixel = one frame pixel. Any other value breaks that identity. |
| `epochs` | **100** | §5.5c says 30–50 *for 10–20k tiles*. We have ~10× fewer, so 40 epochs is ~10× fewer gradient steps than the recipe assumes. 100 epochs restores a comparable step count; it is not "more training", it is the same training on less data. |
| `patience` | **25** | With ~495 instances the model will overfit well before the epoch budget. Early stopping on val mAP50 is what actually sets the length; 100 is a ceiling, not a target. |
| `batch` | **-1** | Verified Ultralytics behaviour: auto-sizes to ~60 % of VRAM. Hand-tuning it on a rented card we have not profiled is guesswork. |
| `close_mosaic` | **15** | §5.5 uses 10. Raised because mosaic is doing more work here (scarce instances) and its artefacts need longer to wash out of a longer run. |
| `degrees` | **180** | A nadir frame has no canonical up. This is the single most appropriate augmentation for this dataset. |
| `flipud` / `fliplr` | **0.5 / 0.5** | Same reason. |
| `mosaic` | **1.0** | The main defence against instance scarcity: it manufactures new spatial contexts per batch. |
| `copy_paste` | **0.0, unless verified** | Flagged by `runpod_train.py` as "the most valuable augmentation when INSTANCES are the scarce resource" — which is exactly our case. **But Ultralytics' copy-paste operates on segmentation polygons; on a detection-only dataset it can silently do nothing.** It will be enabled only if a one-epoch smoke run shows it actually changes the batch. Claiming an augmentation that is silently inert is the exact failure mode this project keeps catching. |
| `cache` | **'disk'** | On the pod RAM is plentiful, but 'disk' is what the local recipe is validated with and removes a variable. |
| `workers` | **8** | The Windows ≤ 4 ceiling is a Windows constraint; the pod is Linux. |
| `cos_lr` | **True** | A long run on a small set benefits from a smooth decay to a small final LR rather than a step. |
| `seed` | **0**, `deterministic=True` | So the run can be reproduced and compared against a re-run. |

---

## 5. Compute configuration

| | |
|---|---|
| Provider | RunPod, via `tools/train/runpod_train.py` (SECURE cloud default) |
| GPU | **NVIDIA GeForce RTX 4090, 24 GB, $0.34/h** |
| Why | Highest clock per dollar in the list for small-batch 1024-px training. A6000/A40 (48 GB, $0.33–0.35) buy VRAM we cannot use at this dataset size; the 3090 at $0.22 is ~40 % slower for ~35 % less. The 4090 is the best *time-to-result* per dollar, and time is the binding constraint. |
| Estimated run | ~1–2k tiles, batch ~16 at 1024 → ~15–25 s/epoch → **100 epochs ≈ 25–40 min** |
| Budget | Well inside the 1-hour ceiling; ≈ **$0.15–0.25**. Balance $10.78. |
| Pod hygiene | `up` records pod id/price/start; every subcommand prints accrued cost; `down` is idempotent and runs unconditionally, including on failure. |

**Fallback**: if the 4090 is unavailable at request time, take the RTX 4080/3090 rather than waiting; the run
is short enough that a 40 % slower card still finishes inside the hour.

---

## 6. Validation strategy, metrics and the operating threshold

**Validation set**: seed 47, held out by construction. Never seen in training, different actor placement,
different occluders.

**Metrics** (§5.12, §5.5c step 4):

1. **Recall at IoU 0.5** on the nominal slice — the acceptance figure. Reported as
   *"X % recall at IoU 0.5, in simulation, on the nominal slice (40–60 m, daylight, occlusion < 50 %,
   non-submerged)"*.
2. **Recall at IoU 0.25** as the secondary "found the person" metric — §5.5c and TinyPerson practice, because
   on a limb-only target the box extent is ill-defined.
3. **Precision and FP/min at the frozen operating threshold** — not at Ultralytics' max-F1 confidence, which
   is a different point and would flatter the result.
4. **Per-slice tables**: altitude band, pose, submersion, occlusion level, pixel-size bin. Hard slices
   (head-only, occlusion 2, sub-20 px) reported **separately, never pooled** — `SliceRoleError` and
   `DomainMixError` enforce this structurally.

**Operating threshold procedure** (§5.5 "Operating threshold", verbatim intent): sweep confidence on the
**validation** split, pick the **highest** confidence at which recall ≥ **0.92** (a 2-point margin over the
0.90 target), **freeze it**, and report precision and FP/min at that frozen value. Ultralytics' per-class P/R
are taken at max-F1, so recall is read from the curves or via `val(conf=c)` — not from the summary line.

**Known distribution warning, from the second session's census, carried here so it is not a surprise**: at
alt55 only 6 boxes reach 80 px and 22 fall under 20 px, against 57 and 3 at alt35. Whatever the acceptance
figure is, it will be dominated by the 40–80 px band, and the sub-20 px tail sits at or below the §2.5
resolvable floor. That is flight geometry, not model quality, and §5.12 requires the report to say so.

---

## 7. What "good" looks like, stated before measuring

§5.5c: *"Expect the nominal slice to clear 90 % comfortably … because the model is being tested on the
distribution it was trained on. Measure it rather than assuming it."*

Stated in advance so the result cannot be rationalised afterwards:

* **≥ 90 % recall at IoU 0.5 on the nominal slice** — the target. Plausible: zero domain gap.
* **75–90 %** — a real result worth reporting with the shortfall named. Most likely cause given ~495
  instances is the small-target tail, not the recipe.
* **< 75 %** — treat as a defect, not a number. §5.5c is explicit that the cause would be *flight geometry or
  scene difficulty, not the model*, and the fix is to lower the altitude before touching training.

Hard slices are expected to be materially worse and that is the honest finding, not a failure: head-only and
occlusion-2 targets are the cases §2.7 and §5.3b exist to characterise.

---

## 8. Process, in order

```bash
# 1. build the tiled dataset (refuses without two seeds)
uv run python -m sightline.detect dataset \
    _artifacts/dataset/seed23_alt35 _artifacts/dataset/seed23_alt55 \
    _artifacts/dataset/seed47_alt55 --out _artifacts/yolo/sim --val-seed 47

# 2. dataset quality gate - BLOCKING (section 2 above)
uv run python tools/capture/dataset_gate.py _artifacts/dataset/seed23_alt35 \
    _artifacts/dataset/seed23_alt55 _artifacts/dataset/seed47_alt55

# 3. cost arithmetic, rents nothing
uv run python tools/train/runpod_train.py plan --data _artifacts/yolo/sim

# 4. rent, push, train
uv run python tools/train/runpod_train.py up --gpu "NVIDIA GeForce RTX 4090"
uv run python tools/train/runpod_train.py push --data _artifacts/yolo/sim
uv run python tools/train/runpod_train.py train --epochs 100 --imgsz 1024 --patience 25

# 5. retrieve and RELEASE (down runs even if train failed)
uv run python tools/train/runpod_train.py pull
uv run python tools/train/runpod_train.py down
```

**Artefacts preserved**: `best.pt`, `last.pt`, `results.csv` (the loss/metric curves), `args.yaml` (the exact
resolved hyperparameters), the confusion matrix and PR curves, plus `data.yaml` and the dataset manifest. The
frozen operating threshold is written beside them so inference cannot drift from the evaluated point.

---

## 9. Training-run verification — "completed" is not "succeeded"

Checked from `results.csv` after the run, not inferred from RunPod's exit status:

* **Losses decrease** — box, cls and dfl. A flat or rising cls loss means the label mapping is wrong.
* **Train/val divergence** — val mAP50 rising then falling is overfitting; with ~495 instances this is the
  most likely failure and `patience 25` should have stopped it. Verify the stop was early, not at the ceiling.
* **Instability** — NaN/inf loss, or a mAP that collapses to 0 after an epoch, means LR or AMP trouble.
* **Underfitting** — mAP50 still climbing at the last epoch means the ceiling bound the run; re-run longer.
* **The checkpoint is actually usable** — load `best.pt` locally, run it on a held-out tile, and **look at the
  boxes on the image**. A checkpoint that loads and predicts nothing is a passing job and a failed model.
* **Sanity against the baseline** — run stock `yolo26s.pt` (COCO `person`) on the same val tiles. The
  fine-tune must beat it clearly; if it does not, the dataset or the label mapping is wrong, not the recipe.

---

## 10. Integration readiness (hand-off to the real-time demo)

* `best.pt` is the artefact; the inference path is `sightline/detect/` at 1024 tiles with the **same**
  `TileGrid` used for training, so a training pixel and a deployment pixel are the same pixel.
* The frozen confidence threshold ships with the weights.
* TensorRT export (F8/F20) is a later optimisation; the demo runs the PyTorch checkpoint on the 4060, which
  §5.11 sizes at ~113 ms for six 1280 tiles — inside the 300 ms budget.
* Detections feed the §5.6 tracker (3 hits in 2 s), §5.7 geolocation and the §5.8 triage output unchanged.

---

## 11. Open risks

| Risk | Severity | Handling |
|---|---|---|
| ~495 instances vs the recipe's 10–20k tiles | **high** | More epochs, aggressive geometric augmentation, early stopping. Named in the report; not concealed by a good-looking mAP. |
| Sub-20 px tail below the resolvable floor | medium | Reported as its own slice; §2.5 says these are not resolvable, so they are characterised, not fixed. |
| Train labels hand-patched, val labels depth-gated | medium | Asymmetry is in the safe direction: the *validation* number is the accurate one. Documented. |
| `trapped` boxes are amodal on seed 23 | medium | Rubble is an ISM and invisible to the mask; unfixable on captured frames. Localisation on that class is optimistic; stated. |
| `copy_paste` silently inert on a detection dataset | low | Disabled unless a smoke run proves it changes batches. |
| Empty `animal` class | low | Kept for pipeline compatibility; `human` AP quoted; any animal prediction is a false positive. |
