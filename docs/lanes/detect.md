# Lane report — detection + evaluation (F8, F8b, F9, F10, F19)

Written 2026-09-10 by the detection/evaluation agent. Scope: `sightline/detect/`, the wiring of
`sightline/eval/`, and `tests/test_detect.py`. The Unreal editor was running for this entire session, so
**nothing was trained and nothing was exported**; what follows separates what is measured from what is blocked.

---

## 0. The headline, stated the way hard rule 5 requires

**There is no F8b model yet, and the dataset it would be trained on is broken.** The lane is complete and tested
around that hole: the tiler, the dataset converter, the seed-held-out split, the threshold rule, late fusion, the
verifier safety rule and the evaluation wiring are all written, unit-tested on CPU, and exercised end to end on
real captured 4K frames using a stock COCO detector. The two things that are missing are a free GPU and a clean
capture with at least two scenario seeds.

The measured baseline, with its slice, in one sentence: **stock COCO YOLO26s, run over the native 1024-px tile
grid, reaches recall 0.556 at IoU 0.5 in simulation** on 24 frames of `train_seed23` at 45–51 m AGL,
1.766 cm/px, daylight, randomisation off, n = 27 scored human boxes — at confidence 0.058, precision 0.128,
1275 FP/min at 5 processed fps — and **the §5.5 target of 0.92 is not reachable at any confidence**, because 12
of the 27 targets are never found at all. That is the number the fine-tune has to beat, and it is a lower bound
on a clip whose ground truth is itself wrong (§3).

---

## 1. The prediction overlay — what I actually saw

**`_artifacts/detect/overlay_zeroshot_yolo26s.png`** (24 frames, 12 MB) and the close-up
**`_artifacts/detect/overlay_zeroshot_fp_closeup.png`** (frames 00135 and 00141).
Green = a prediction that matched ground truth, red = an unmatched prediction, yellow = a missed ground truth,
grey = a prediction below the drawn threshold. Every panel carries a strip of **magnified insets**, one per box,
because a 60 px survivor in a 3840×2160 frame is invisible on a downscaled contact sheet and every box looks
correct there. The matching in the picture is the lane's own `threshold.match_frame`, so the drawing and the
metric cannot disagree.

I opened them. What is in the pixels:

* **The model does find people, at the sizes §2.5 predicts.** A prone body in light clothing on gravel:
  `TP 0.92, IoU 0.95, 110 px`. A person on a rooftop: `TP 0.88, IoU 0.98, 68 px`. A person on a gravel bank at
  `TP 0.34, IoU 0.81, 64 px`. The Rocketbox characters are textured and posed in these crops — no white
  mannequins, no T-poses.
* **It misses the hard presentations.** A dark-clothed waving figure on grass at 74 px, and every `head_only`
  target (a faint pale smudge on silt in the inset) are unmatched yellow boxes.
* **Some of the red "false positives" really are shadows and rocks.** On the deposit fan, five boxes of
  740–786 px at conf 0.31–0.68 sit on mud/rock outcrops; on frame 00141 a 784 px box wraps an entire building.
  This is precisely the failure the quality gate names, and it is what the F10 crop verifier exists to remove.
* **But many red boxes are people the ground truth does not have.** On frame 00141, the insets for `FP 0.89`
  (82 px) and `FP 0.74` (86 px) each show an unmistakable human figure lying on a roof — light-blue shirt with
  limbs, and a dark prone figure — with **no ground-truth box anywhere near them**. The precision figure above is
  therefore an underestimate, and I am not treating it as a measurement of the detector.

That last bullet is the finding that matters, and it is why the visual gate exists: the numbers looked like "a
bad detector", the picture says "a broken label set".

---

## 2. What is verified on CPU right now

`D:\Tools\uv\uv.exe run pytest tests/test_detect.py -q` → **88 passed** (plus `tests/test_eval.py` 34 passed,
unbroken). No torch, no ultralytics, no editor, no AirSim; one test asserts that by spawning a subprocess and
checking `torch` never reaches `sys.modules` after importing every module in the lane.

| Property (doc reference) | How it is checked |
|---|---|
| **The 4K tile grid is the design's grid** | 3840×2160 at 1024/0.2 gives exactly 5×3 = 15 tiles, the last flush with the far edge, realised overlap never below the requested 0.2, and **every pixel of the frame inside at least one tile** (checked by rasterising the union). |
| **Native resolution means one tile pixel = one frame pixel** | A crop is asserted bit-identical to the corresponding slice of the frame — the identity the whole §5.5c recipe rests on. |
| **`max_safe_target_px` is real** | The claim "a target below the overlap band is whole in some tile" is brute-forced over a lattice of positions, not asserted. |
| **Inverse mapping is exact** | frame → tile-local → frame round-trips bit-exactly for a box inside a tile, and for **every** tile a seam-crossing box touches, to exactly the intersection rect. An off-by-one here silently costs IoU on every 20 px target. |
| **A target split across a seam is reassembled** | An 824 px target (wider than the 320 px overlap band, so whole in no tile) is seen as two clipped fragments; the merge returns **one** box equal to their union, with the higher fragment's score. |
| …and two separate targets across a seam are **not** merged | Two people either side of a seam, each whole in its own tile, stay two detections. |
| **A duplicate in the overlap band collapses to one** | Same survivor detected by both tiles at IoU > 0.55 → one survivor, keeping the higher score and its `tile_idx` provenance. An edge *fragment* is absorbed into the whole box another tile saw by intersection-over-smaller (IoU 0.09, containment 1.0). |
| Nesting inside one tile is left alone | Cross-tile repair must not rewrite the detector's own output. |
| **`detect_frame` maps a planted target back exactly** | With a hand-written inferencer, a 28×58 px target at (1500, 900) comes back at exactly (1500, 900, 1528, 958) with the right `frame_idx` and `tile_idx`; batching the grid 3-at-a-time gives an identical answer. |
| **The capture's box convention** | `capture_box_to_xyxy` (+1 on the inclusive index) is checked against **the capture lane's own `labels/*.txt`** on the real run, to 1e-6, for every labelled frame. If `tools/capture/labels.py` ever changes convention, training boxes and scoring boxes diverge and this fails. |
| Tile-label clipping | A whole target keeps `visible_frac == 1.0`; a 4-of-60 px sliver is dropped **and reported**, so the caller can skip the tile rather than teach it that a visible arm is background; the 0.5 boundary is checked from both sides; §6.3's 8 px floor is enforced from both sides. |
| YOLO line arithmetic | Hand-computed `cx, cy, w, h` for a 40×100 box at (512, 256) in a 1024 tile, at the same 6 decimals `labels.to_yolo` writes; a box outside the tile raises. |
| **Split by scenario seed, never by frame** | `split_by_seed` assigns whole runs; `assert_no_seed_leak` raises when a seed appears twice (checked by actually making it happen); a single-seed capture **raises** rather than silently splitting by frame; the draw is deterministic for a seed. |
| Dataset build counts | Hand-derived: the 5 column starts of a 3840-wide frame are 0/704/1408/2112/2816, so a run of 3 frames with boxes at x = 500/600/700 yields 4 positive tiles and 4 boxes (the third box is in the overlap band and legitimately appears in two tiles). Background tiles are sampled at `negative_frac`, never flooded. One test writes real pixels and checks the crop is 1024². |
| **A buried actor is `uncertain`, not a target** | `aerially_detectable = False` → `GtBox(uncertain=True)`: not a recall target, and a detection on it is dropped rather than counted as a false positive (§2.7 + R10). Survivors from `data/scene/actors.json` carry `buried` and real lat/lon. |
| **The §5.5 threshold rule, on a curve built by hand** | 25 targets scored 1.00 … 0.76: recall ≥ 0.92 needs 23, so the answer is **0.78 with recall exactly 0.92**, and 0.79 is verified to miss the target. |
| Recall/precision at a threshold | Four targets with predictions at IoU 1.00 / 0.4286 / 0.0989 / none: `(tp, fp, fn) = (1, 2, 3)` at IoU 0.5 and `(2, 1, 2)` at IoU 0.25 — the constructed IoUs are themselves verified first. |
| **max-F1 is not the operating point** | Constructed so the two genuinely disagree: the §5.5 rule lands at conf 0.08 (recall 0.92, precision 0.697) while max-F1 sits at 0.81 (recall 0.80, precision 1.00). Quoting the library's row would advertise recall 0.80 as the system's recall. |
| An unreachable target is a finding | With only 20 of 25 findable, `achieved=False`, `target_recall` stays 0.92, and the note says so. The bar is never moved. |
| FP/min | 3 FPs over 10 frames at 5 fps = exactly 90 FP/min; a zero fps raises. |
| `ignore` regions | Neither a miss nor a false positive (§6.3). |
| Every metric carries its domain | `metric_rows` and `sliced_operating_points` return `MetricRow`s whose `SliceKey.domain` is set; `str(row)` contains `domain=sim`. |
| **The RGB-only fallback is structural** | `fuse_detections` with no thermal returns **the same `Detection` objects, by identity (`a is b`), in the same order** — checked for three separate fallback causes (no thermal frame, thermal found nothing, registration rejected) and through `fuse_frame`. |
| ProbEn, not WBF's `n/N` | A single-modality box keeps its own posterior (0.60 stays 0.60, 0.55 stays 0.55); an agreeing pair fuses to `ab/(ab+(1−a)(1−b)) = 0.6471`, hand-computed. |
| Registration is exact in sim | `homography_from_intrinsics` maps the thermal principal point onto the RGB principal point exactly, with scale `fx_r/fx_t`. |
| Thermal weight schedule | night > day > midday, dawn = dusk < day, and radiometric contrast overrides the heuristic (§5.5b). |
| **§5.5a: posture may raise urgency, never lower it** | Over the **full cross-product** of 8 postures × 6 submersions × 7 confidences (336 cases), `decision.demoted` is never True and the final urgency never falls below the `stranded` base. A low-confidence prediction becomes `unknown` at the base weight while the rejected prediction is preserved for display (rule 3). A confident `half_submerged` / `head_only` promotes to `immersed`. |
| Suppression is never silent | `suppress_false_positives` returns **both** lists, and a detection the verifier never saw (`is_real is None`) is always kept. |
| The verifier will not fake being trained | `CropVerifier` raises `FileNotFoundError("… not trained yet …")` rather than returning uniform guesses. |
| Training guards | `imgsz 1024`, `batch=-1`, AMP on, `cache='disk'`, `workers ≤ 4`, 30–50 epochs, `project` on D:, and **HSV augmentation zeroed** (randomisation off, §5.5c). `cache='ram'`, `workers=8` and a C: project path each raise. Preflight raises on a missing dataset, on a train/val seed overlap, and on a dataset with no val split. |
| Export guards | INT8 without a calibration set raises; a calibration set with under 20 % tiny positives raises (and passes once 20 px boxes are added); `compare_recall` raises `DomainMixError` on a sim/real comparison and says "DO NOT SHIP IT" when INT8 loses more than 0.005 recall. |
| The overlay tool itself | tp/fp/fn hand-computed on a constructed frame; the zoom panel is checked to actually contain the bright target it magnified. |
| **The evaluation wiring** | `run_evaluation` freezes on `val`, measures on `test`, and every row comes back `domain="sim"` with a finite value; the generated slice table's every data row starts with `| sim |`; an unmet target becomes a *finding* with `target_recall` still 0.92; a seed present in both val and test raises; `freeze_on_validation` refuses a non-`val` split. |

---

## 3. FINDING: the current capture must not be trained on

`_artifacts/dataset/train_seed23`, read at 88 → 184 → 204+ frames while it was still being written.

**3.1 The segmentation mask collapses to a single colour over long stretches of the flight.** Frames 62 and 63
have **exactly one unique colour in the entire 3840×2160 mask**. Frame 63's single colour is the palette entry
for `Human_019`, so the label file says `Human_019` occupies `[0, 0, 3839, 2159]` with `visible_px = 8 294 400`.
At the run's own GSD of 1.766 cm/px that is **67.8 m × 38.1 m of ground for one sitting person**. It is not an
isolated frame: at the last read, **368 of 409 labelled boxes** in the run were larger than 3 m on the ground.
Frame 72's mask has 9 colours but the background is still `Human_019`'s colour, and `Human_068` gets a 655×719 px
box (12.7 m).

Evidence: `_artifacts/detect/capture_defect_check.png` — frames 30 / 62 / 63 / 72 / 77 with their own labels
drawn. Frame 30 is healthy (2 small actor blobs, 99.99 % background). Frames 62–77 are near-featureless silt and
water with a full-frame green box on them.

**3.2 Most survivors are never labelled at all.** `tools/capture/validate.py` reports **11 of 69 detectable
survivors seen, 58 never seen**. The overlay confirms the consequence directly: frame 00141 shows at least three
people lying on rooftops, clearly visible in the RGB, and the truth has one box.

**3.3 `tools/capture/validate.py` fails the run — but for the smaller reason.** It exits 1 on
`BURIED survivors appeared in labels: [62]` (correct, §2.7). Its check F *prints* `size px min 7 p50 2160 p90
3840 max 3840` and does not fail on it, and its check C reports "0 mismatched" because a whole-frame box does
contain that actor's colour. **The dominant defect is printed and passed.** That is a gap for the capture lane
to close; in the meantime this lane closes it on its own side.

**3.4 What I added so it cannot slip through.** `dataset.implausible_boxes()` / `assert_boxes_are_plausible()`
turn every box into metres of ground via the frame's own GSD (falling back to the run's median GSD, so a
live capture with unwritten telemetry is still checked) and refuse anything over `MAX_TARGET_GROUND_M = 3.0`.
`build_yolo_dataset()` calls it and raises *"DO NOT TRAIN ON THIS DATASET"* with the worst offenders named;
`--allow-implausible-boxes` exists, is explicit, and is recorded in the manifest. `dataset.plausible_labels()` is
the read-side counterpart, so a broken capture can still be *looked at* without the leak boxes being drawn or
scored as truth. Three tests cover it, including one that asserts a frame with no GSD is **skipped, not passed**.

```
$ uv run python -m sightline.detect dataset _artifacts/dataset/train_seed23 --dry-run
WARNING: 368 labelled box(es) are too large to be people; ... Worst: {... 'name': 'Human_019',
  'longest_px': 3840.0, 'ground_m': 67.81, 'gsd_cm_px': 1.766, 'agl_m': 50.59 ...}
REFUSED: only 1 scenario seed(s) captured ([-1]); a seed-held-out split needs at least two.
         Capture another run with a different scenario seed before training (SOLUTION_DOC 5.5c step 3).
```

**What the capture lane needs to do:** pin the segmentation instance colours per mesh so the terrain and the
water plane cannot take an actor's ID (the colour changes between frames — 62 is `(255,127,255)`, 63 is
`(95,159,95)` — which is the signature of unpinned IDs), then re-capture. And fail on box size in `validate.py`,
not just print it.

---

## 4. The zero-shot baseline (domain = sim), and the exact numbers

`_artifacts/eval/detect/zeroshot_slice_table.md`, `_artifacts/detect/zeroshot_operating_point.json`.
Stock `models/yolo26s.pt` (COCO, **no fine-tune**), tiles 1024 / overlap 0.2, raw conf 0.05, FP16 on the RTX
4060; 24 frames of `train_seed23`, 45–51 m AGL, 1.766 cm/px, daylight, randomisation off, n = 27 scored human
boxes, 5 processed fps. **Measured in simulation. Every caveat of §3 applies — the ground truth is wrong in both
directions, so this is a lower bound, not a measurement of the detector.**

| domain | conf | recall | precision | tp | fp | fn |
|---|---|---|---|---|---|---|
| sim | 0.05 | 0.556 | 0.116 | 15 | 114 | 12 |
| sim | 0.10 | 0.519 | 0.179 | 14 | 64 | 13 |
| sim | 0.25 | 0.333 | 0.273 | 9 | 24 | 18 |
| sim | 0.50 | 0.259 | 0.467 | 7 | 8 | 20 |
| sim | 0.80 | 0.148 | 0.667 | 4 | 2 | 23 |

Operating point by the §5.5 rule: `conf=0.058 recall=0.5556 (target 0.92, NOT ACHIEVED) precision=0.1282 in sim,
n_gt=27`; recall at IoU 0.25 also 0.5556; **1275 FP/min**. The 0.92 target is unreachable at *any* confidence —
12 of 27 targets are never found — so no threshold recovers them. The bar was not moved.

Slices at that frozen confidence (all `domain=sim`, all n small):

| metric | slice | value | n |
|---|---|---|---|
| recall@IoU0.5 | whole clip | 0.5556 | 27 |
| recall@IoU0.5 | altitude_band=30-45 | 0.5455 | 11 |
| recall@IoU0.5 | altitude_band=45-60 | 0.5625 | 16 |
| recall@IoU0.5 | zone=fan | 1.0000 | 2 |
| recall@IoU0.5 | zone=settlement | 0.5200 | 25 |
| **recall@IoU0.5** | **occlusion=0** | **0.9167** | 12 |
| **recall@IoU0.5** | **occlusion=1** | **0.2667** | 15 |
| recall@IoU0.5 | posture=half_submerged | 0.2143 | 14 |
| recall@IoU0.5 | posture=prone | 1.0000 | 3 |
| recall@IoU0.5 | posture=standing | 0.6667 | 3 |
| recall@IoU0.5 | posture=waving | 1.0000 | 6 |
| recall@IoU0.5 | posture=trapped | 1.0000 | 1 |

The shape is exactly what §5.5c predicts and is the useful part of this table: **an unoccluded target at 45 m is
already found 92 % of the time by a model that has never seen this renderer**; occluded and head-only targets are
not. Pixels on target and occlusion dominate — which is why the recipe is "fly for the number, then one short
fine-tune", not architecture search.

---

## 5. Training: the exact command, and what it is blocked on

```bat
:: 1. build the dataset (CPU, safe while the editor runs). Needs >= 2 scenario seeds.
D:\Tools\uv\uv.exe run python -m sightline.detect dataset ^
    _artifacts\dataset\<run_seedA> _artifacts\dataset\<run_seedB> [_artifacts\dataset\<run_seedC>] ^
    --out D:\Sightline\_artifacts\yolo\sim --val-seed <B> --negative-frac 0.05

:: 2. preflight only (proves the guards pass before an hour of GPU)
D:\Tools\uv\uv.exe run python -m sightline.detect.train D:\Sightline\_artifacts\yolo\sim\data.yaml ^
    --run _artifacts\dataset\<run_seedA> --run _artifacts\dataset\<run_seedB> --preflight-only

:: 3. THE fine-tune. CLOSE THE EDITOR FIRST - this refuses to start otherwise.
D:\Tools\uv\uv.exe run python -m sightline.detect.train D:\Sightline\_artifacts\yolo\sim\data.yaml ^
    --model yolo26s.pt --name f8b_sim ^
    --run _artifacts\dataset\<run_seedA> --run _artifacts\dataset\<run_seedB>
```

Settings applied (all from §5.5/§5.5c, all asserted by a test): `imgsz=1024`, `epochs=40`, `batch=-1` (auto-size
to ~60 % of VRAM), `amp=True`, `cache='disk'`, `workers=4`, `close_mosaic=10`, `pretrained=True`,
`degrees=180 / flipud=0.5 / fliplr=0.5` (a nadir frame has no canonical up), **`hsv_h=hsv_s=hsv_v=0`** —
domain randomisation is off by decision, and colour jitter would reintroduce it by the back door. Output goes to
`D:\Sightline\models\detect\f8b_sim`; a `train_manifest.json` is written beside the weights.

**Expected runtime.** §5.5's own figure is ≈3–5 min per 5 000 tiles per epoch for YOLO26s at 1024 on an RTX 4060.
A 10–20 k-tile set (≈700–1 400 4K frames at the measured ~14 positive-or-background tiles per frame) is therefore
**5–8 hours for 40 epochs** — one overnight run, as §5.5c says. A first sanity run at `--epochs 5` on a couple of
thousand tiles is ~30–45 min and is the right thing to do before committing the night.

Then, in order: freeze the threshold on **val** (`python -m sightline.detect threshold <val runs> --preds …`),
predict on **test**, and `python -m sightline.detect evaluate --val <run> --test <run> --preds …` for the slice
table. Export (`sightline.detect.export`) also refuses to run beside the editor; **do not re-benchmark latency** —
`docs/verification/gpu_latency.md` already measured this exact machine (yolo26s FP16, 6 × 1280×1088 native tiles,
one batched call: **65.0 ms median**, R9 budget 300 ms).

### Blocked on

1. **A GPU.** `UnrealEditor.exe` ran throughout (3.6–3.9 GB VRAM free of 8, RAM 85–90 % used). Training and
   TensorRT export are gated behind `train.editor_is_running()` and were not attempted. The zero-shot inference
   above was 24 single-frame passes at ~0.45 s each with a headroom check first — small enough not to disturb the
   capture, and it is the only GPU work this lane did.
2. **A clean capture** — §3.
3. **A second scenario seed.** Only `seed 23` exists, and `split_by_seed` refuses to split one seed by frame.
   Without a second seed there is no honest val split, so §5.5's threshold cannot be frozen, and
   `run_evaluation` correctly refuses to produce a slice table.
4. **F10 head weights.** The verifier's safety rule, cropping and batching are written and fully tested; the
   DINOv2 + 4-linear-head training on simulator crops is minutes of GPU and has not been run. `CropVerifier`
   raises rather than pretending — this is a **labelled stub**.

---

## 6. Stubs and honest limits

1. **`CropVerifier` (F10) has no trained heads.** Labelled in the class docstring and enforced by a
   `FileNotFoundError`. Everything around it — `crop_for`, `apply_verifier_output`, the promote-only rule,
   `suppress_false_positives` — is real and tested.
2. **The thermal detector (F9's second model) does not exist.** `fusion.py` is complete and tested, but with no
   thermal model every call takes the structural RGB-only path — which is the designed behaviour, not a gap.
   `AltitudeHomographyTable` (the real-footage registration path) has no fitted bands and refuses to guess.
3. **`evaluate.submersion_rows` adds an axis `slicing.BOX_AXES` does not have.** §5.12's slice grid names
   submersion; rather than edit another module's frozen axis tuple I emit the rows here. If the eval lane wants
   it in `BOX_AXES`, that is a one-line change there and this function should then be deleted.
4. **No record-level, tracking, geolocation or POD numbers from this lane.** `add_record_eval`,
   `evaluate_tracking` and friends are wired and tested by `tests/test_eval.py`, but they need `Record`s and
   `Track`s from lanes B3/B4, which this lane does not produce.
5. **The zero-shot numbers in §4 are on a broken clip.** They are reported because the *shape* of the slice table
   is informative and because they prove the whole chain runs on real frames, not because they measure a
   detector. They must not be quoted as an acceptance figure and they must never be compared with a real-footage
   figure.
6. **`MAX_TARGET_GROUND_M = 3.0` is a judgement, not a measurement.** A prone adult is 1.8 m; 3.0 m allows for
   off-nadir foreshortening, a shadow caught in the mask and a sprawled pose. A genuine rooftop *group* labelled
   as one box would trip it — the capture labels one box per actor today, so it does not, but if group labelling
   is ever introduced this constant needs a per-class value.

---

## 7. What other lanes should know

* **Capture lane (F5):** §3 — pin the segmentation instance colours per mesh, re-capture, and make
  `tools/capture/validate.py` **fail** on box size (it already computes the distribution). Also: `data_card.json`
  and `telemetry.csv` are written only at the end of a run, so a live capture cannot be validated; writing
  `telemetry.csv` incrementally (it now is) and a partial `data_card.json` early would help a lot.
* **Everyone:** `sightline.detect` imports with no torch and no GPU. `from sightline.detect import detect_frame,
  merge_tile_detections, choose_operating_threshold, fuse_detections` are all safe to import anywhere.
* **Ingest (B1):** `detect_bundle(bundle, infer)` is the `FrameBundle` → `list[Detection]` entry point in
  `docs/CONTRACTS.md`. Detections come back in full-frame pixel coordinates with `tile_idx` provenance and
  `frame_idx` stamped, score-ordered.
* **Triage (B4):** do **not** re-implement the §5.5a safety rule. `verifier.apply_verifier_output` returns a
  `VerifierDecision` whose `.detection` is already safe to score and whose `.output` carries the rejected
  prediction for display. `URGENCY_RANK` there mirrors `eval.groundtruth.URGENCY_ORDER`.
* **Eval lane (B7):** nothing in `sightline/eval/` was changed. `sightline/detect/evaluate.py` is the bridge that
  drives it from capture runs; `tests/test_eval.py` still passes (34 tests).
* **No package was added**, and `pyproject.toml`, `uv.lock`, `schemas.py`, `sightline/common/`, `tools/**` and
  every other lane's directory were not touched. No state-changing git command was run.

## 8. Artifacts

| Path | What it is |
|---|---|
| `_artifacts/detect/overlay_zeroshot_yolo26s.png` | 24-frame prediction overlay with magnified insets — **the picture the gate asks for** |
| `_artifacts/detect/overlay_zeroshot_fp_closeup.png` | frames 00135 / 00141 close-up: the rock/roof false positives, and the unlabelled people |
| `_artifacts/detect/capture_defect_check.png` | frames 30 / 62 / 63 / 72 / 77 with their own labels: the mask-collapse defect |
| `_artifacts/detect/preds_zeroshot_yolo26s.json` | the raw predictions, so the overlay and the numbers can be re-derived on CPU |
| `_artifacts/detect/zeroshot_operating_point.json` | the frozen zero-shot operating point plus the full sweep |
| `_artifacts/eval/detect/zeroshot_slice_table.md` | the sweep and slice table of §4, every row `domain=sim` |
