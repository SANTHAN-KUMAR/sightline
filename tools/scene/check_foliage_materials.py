"""Check the tree materials that are ACTUALLY in the project, and the pixels they actually render.

    uv run python tools/scene/check_foliage_materials.py before    # expected to FAIL (the defect)
    uv run python tools/scene/check_foliage_materials.py after     # must PASS
    uv run python tools/scene/check_foliage_materials.py after --compare before

Exits non-zero on any failure. Host side; needs numpy + Pillow. Reads what `tools/scene/qa_foliage.py` dumped:
`_artifacts/foliage/state_<tag>.json` plus the PNGs beside it. It never looks at a return value from the editor.

WHAT IT ASSERTS, AND WHY EACH ONE IS A FAILURE THAT HAS ACTUALLY HAPPENED HERE
------------------------------------------------------------------------------
  A  the dump is newer than the builder (tag "after")     a stale render passing for fresh work
  B  all 5 tree meshes present, each with a leaf slot      a mesh silently resolving to the wrong asset
  C  no material in any slot's PARENT CHAIN is in          the defect: project-looking instances whose parent
     /InterchangeAssets/ (or any /Engine/ path)            is the glTF importer's stock default in a plugin
  D  every tree slot material lives under                  a material authored into the wrong folder, where
     /Game/Sightline/Vegetation/Materials/                 another lane's script would later clear it
  E  no tree slot is translucent                           translucency is not rendered by Nanite, does not
                                                           write the depth/stencil the instance-segmentation
                                                           mask is read from, and is unbounded overdraw at
                                                           4,342 trees
  F  every tree slot compiles: non-zero instructions       a material that fails to compile reports ZERO and
                                                           nothing else, and renders as the grey checker
  G  leaf slots are two-sided and MSM_TWO_SIDED_FOLIAGE    without it a leaf lit from behind has no term to
                                                           return to the camera and renders black
  H  rendered canopy luminance vs SUNLIT GROUND luminance  the red team's evidence: "sunlit grass brilliant
     in the SAME nadir frame                               green, canopy beside it near-black at the same
                                                           exposure"
  I  backlit crown pixels are neither black nor grey       the specific thing subsurface transmission fixes

H AND I ARE MEASURED, NOT ASSERTED FROM METADATA
------------------------------------------------
A material can carry every correct flag in C-G and still render black. So H and I read the PNGs.

Pixels are classified as canopy or ground WITHOUT reference to their brightness, because a test that defined
canopy as "the dark part of the image" would be circular. `qa_foliage.py` captures each view four times: beauty
and world-NORMAL, each with the canopy visible and with it hidden. A pixel is CANOPY where the world normal
changed when the trees were hidden -- geometry, independent of albedo and of lighting. A pixel is CLEAN GROUND
where the normal did NOT change AND the beauty frame also did not change; that second condition discards every
pixel whose lighting changed when the trees went away, i.e. every tree SHADOW, which would otherwise be scored
as canopy and bias the measurement dark with the very thing being measured.

THE NUMBERS, AND WHERE THEY COME FROM
-------------------------------------
All in sRGB luma (0.2126 R' + 0.7152 G' + 0.0722 B' on the gamma-encoded 0..1 PNG values), because the
judgement being reproduced was a visual one made on a displayed PNG. Linear luminance is reported alongside.

`docs/SCENE_REFERENCE.md` Reference A is a flooded wooded neighbourhood whose crowns read MID-GREEN. Mid-green
canopy in an aerial photograph sits near sRGB (60, 110, 55) -> luma 0.38; brilliant sunlit grass near
(120, 190, 80) -> luma 0.65. That is a ratio of about 0.58. A canopy that reads "near-black beside brilliant
grass" is below 0.15. The gate is set at 0.35 -- comfortably between the two states it has to tell apart, and
deliberately lenient, since a canopy legitimately IS darker than open sunlit grass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "_artifacts/foliage"
MAT_DIR = "/Game/Sightline/Vegetation/Materials/"
BUILDER = REPO / "tools/scene/build_foliage_materials.py"

EXPECT_PIDS = {"island_tree_01", "island_tree_02", "island_tree_03", "jacaranda_tree", "tree_small_02"}
BANNED_PREFIXES = ("/InterchangeAssets/", "/Engine/")
TRANSLUCENT = ("BLEND_TRANSLUCENT", "BLEND_ADDITIVE", "BLEND_MODULATE", "BLEND_ALPHACOMPOSITE",
               "BLEND_ALPHAHOLDOUT", "BLEND_TRANSLUCENTCOLOREDTRANSMITTANCE")
OK_BLEND = ("BLEND_OPAQUE", "BLEND_MASKED")

# --- gates (declared here, before any measurement, so they cannot be tuned to whatever came out) -------------
NRM_T = 10 / 255.0          # world normal moved this much -> vegetation is in front of what was there
BEAUTY_T = 6 / 255.0        # beauty frame agrees this closely -> lighting did not change -> not a tree shadow
MASK_COVER_MIN = 0.10       # a degenerate mask makes H meaningless: fail loudly instead of passing quietly
MASK_COVER_MAX = 0.95
GROUND_MIN_PIX = 20000

RATIO_MIN = 0.35            # canopy median luma / sunlit ground luma. See the docstring.
CANOPY_P25_MIN = 0.06       # crown interiors: the darker quartile must clear "near-black"
NEAR_BLACK = 0.06
NEAR_BLACK_FRAC_MAX = 0.25
BACKLIT_P50_MIN = 0.04      # a backlit crown is legitimately dark, but it must not be a black silhouette


def norm_enum(v) -> str:
    """Tolerate a dump written before the enum names were cleaned up: `str(BlendMode.BLEND_OPAQUE)` is
    '<BlendMode.BLEND_OPAQUE: 0>', which naive splitting leaves as 'BLEND_OPAQUE: 0>'."""
    return str(v).strip("<>").split(":")[0].split(".")[-1].strip().upper()


def luma_srgb(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load(p: Path) -> np.ndarray:
    if not p.exists():
        raise SystemExit(f"MISSING IMAGE {p} - run tools/scene/qa_foliage.py first")
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


class Checker:
    def __init__(self):
        self.fails: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            self.fails.append(f"{label}  {detail}".strip())
        return ok

    def note(self, s: str):
        print(f"         {s}")
        self.notes.append(s)


def masks(tag: str, view: str, c: Checker):
    """canopy / clean-ground boolean masks for one view. See the module docstring."""
    beauty = load(ART / f"{view}_beauty_{tag}.png") if (ART / f"{view}_beauty_{tag}.png").exists() \
        else load(ART / f"{view}_beauty.png")
    noveg = load(ART / f"{view}_noveg.png")
    nrm = load(ART / f"{view}_nrm.png")
    nrm_nv = load(ART / f"{view}_nrm_noveg.png")

    dn = np.abs(nrm - nrm_nv).max(axis=2)
    db = np.abs(beauty - noveg).max(axis=2)
    canopy = dn > NRM_T
    ground = (dn <= NRM_T) & (db <= BEAUTY_T)
    return beauty, noveg, canopy, ground, dn, db


def stats(rgb: np.ndarray, m: np.ndarray) -> dict:
    if m.sum() == 0:
        return {"n": 0}
    px = rgb[m]
    L = luma_srgb(px)
    Ll = luma_srgb(to_linear(px))
    return {"n": int(m.sum()),
            "p10": float(np.percentile(L, 10)), "p25": float(np.percentile(L, 25)),
            "p50": float(np.median(L)), "p75": float(np.percentile(L, 75)),
            "lin_p50": float(np.median(Ll)),
            "r": float(np.median(px[:, 0])), "g": float(np.median(px[:, 1])), "b": float(np.median(px[:, 2])),
            "near_black_frac": float((L < NEAR_BLACK).mean())}


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "(no pixels)"
    return (f"n={s['n']:,} luma p25={s['p25']:.3f} p50={s['p50']:.3f} p75={s['p75']:.3f} "
            f"(linear p50={s['lin_p50']:.4f}) rgb=({s['r']:.3f},{s['g']:.3f},{s['b']:.3f}) "
            f"near-black {s['near_black_frac'] * 100:.1f} %")


def write_mask_png(tag, view, beauty, canopy, ground):
    """Save what the mask actually selected, so the classification can be EYEBALLED instead of trusted."""
    vis = (beauty * 255).astype(np.uint8).copy()
    vis[canopy] = (vis[canopy] * 0.35 + np.array([255, 40, 200]) * 0.65).astype(np.uint8)
    vis[ground] = (vis[ground] * 0.35 + np.array([60, 200, 255]) * 0.65).astype(np.uint8)
    p = ART / f"mask_{view}_{tag}.png"
    Image.fromarray(vis).save(p)
    return p


def selftest(tag: str) -> int:
    """Prove gate H is not vacuous.

    Measured on the real scene, H PASSES even before this lane's fix: the canopy at 45 m nadir came out at
    ratio 0.739 of the sunlit ground, not "near-black". That is an honest result -- the audit's headline
    symptom was not reproducible -- but it leaves the gate unexercised, and a gate that has never been seen to
    fail is not yet a test.

    So: take the REAL nadir frame and the REAL canopy mask, and darken only the canopy pixels to the state the
    audit described ("canopy beside it near-black at the same exposure"). Everything else -- the mask, the
    ground reference, the thresholds -- is untouched. If the gates still pass on that, they are worthless.
    """
    c = Checker()
    print(f"SELF-TEST of gate H against a synthetic near-black canopy (source frames: tag {tag!r})\n")
    beauty, _noveg, canopy, ground, _dn, _db = masks(tag, "fol_1_nadir", c)
    real = stats(beauty, canopy)
    gs = stats(beauty, ground)
    print(f"  real canopy   {fmt(real)}")
    print(f"  real ground   {fmt(gs)}")
    print(f"  real ratio    {real['p50'] / gs['p75']:.3f}\n")

    dark = beauty.copy()
    dark[canopy] *= 0.12                       # the audit's described state, applied to the real canopy pixels
    Image.fromarray((dark * 255).astype(np.uint8)).save(ART / f"selftest_nadir_darkened_{tag}.png")
    ds = stats(dark, canopy)
    ratio = ds["p50"] / max(gs["p75"], 1e-6)
    print(f"  darkened canopy {fmt(ds)}")
    print(f"  darkened ratio  {ratio:.3f}\n")

    fired = []
    for label, ok, detail in (
            ("canopy/sunlit-ground ratio", ratio >= RATIO_MIN, f"{ratio:.3f} (gate >= {RATIO_MIN})"),
            ("crown interiors not near-black", ds["p25"] >= CANOPY_P25_MIN,
             f"p25 {ds['p25']:.3f} (gate >= {CANOPY_P25_MIN})"),
            ("near-black fraction", ds["near_black_frac"] <= NEAR_BLACK_FRAC_MAX,
             f"{ds['near_black_frac'] * 100:.1f} % (gate <= {NEAR_BLACK_FRAC_MAX * 100:.0f} %)")):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}")
        if not ok:
            fired.append(label)

    print()
    if len(fired) < 2:
        print(f"SELF-TEST FAILED: only {len(fired)} of 3 gates fired on a canopy darkened to 12 % — gate H is "
              f"too loose to detect the state it exists for.")
        return 1
    print(f"SELF-TEST PASSED: {len(fired)} of 3 gates fire on the synthetic near-black canopy "
          f"({', '.join(fired)}), so gate H can detect that state (sim).")
    print(f"  darkened frame written to {ART / f'selftest_nadir_darkened_{tag}.png'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="after")
    ap.add_argument("--compare", default=None, help="also print the same measurements from another tag")
    ap.add_argument("--selftest", action="store_true",
                    help="prove gate H can fail: re-measure the real frame with the canopy pixels darkened "
                         "to the state the audit described, and show the gates firing")
    a = ap.parse_args()
    tag = a.tag
    if a.selftest:
        return selftest(tag)

    sp = ART / f"state_{tag}.json"
    if not sp.exists():
        print(f"MISSING {sp} - run qa_foliage.py with FOLIAGE_TAG='{tag}' first")
        return 2
    state = json.loads(sp.read_text())
    c = Checker()
    print(f"check_foliage_materials: tag={tag!r} dumped {state.get('when_local')} "
          f"level={state.get('level')}\n")

    # --- A. freshness ------------------------------------------------------------------------------------
    print("A. freshness")
    if tag == "after":
        if BUILDER.exists():
            ok = sp.stat().st_mtime >= BUILDER.stat().st_mtime
            c.check(ok, "state dump is newer than build_foliage_materials.py",
                    f"dump {sp.stat().st_mtime:.0f} vs builder {BUILDER.stat().st_mtime:.0f}")
        else:
            c.check(False, "build_foliage_materials.py exists", str(BUILDER))
    else:
        c.note(f"freshness not enforced for tag {tag!r} (a pre-fix snapshot is meant to be older)")

    # --- B-G. the material state -------------------------------------------------------------------------
    print("\nB. meshes and slots")
    pids = {m["pid"] for m in state["meshes"]}
    c.check(pids == EXPECT_PIDS, "all 5 tree meshes dumped", f"got {sorted(pids)}")
    leaf_slots, all_slots = [], []
    for m in state["meshes"]:
        ls = [s for s in m["slots"] if s.get("role") == "leaf"]
        c.check(len(ls) == 1, f"{m['pid']}: exactly one leaf slot", f"found {len(ls)}")
        for s in m["slots"]:
            s["_pid"] = m["pid"]
        leaf_slots += ls
        all_slots += [s for s in m["slots"] if s.get("material_path")]
    print(f"         {len(all_slots)} material slots across {len(pids)} meshes, {len(leaf_slots)} leaf")

    print("\nC. no engine-plugin material anywhere in any parent chain")
    for s in all_slots:
        bad = [p for p in s.get("chain", []) if any(p.startswith(b) for b in BANNED_PREFIXES)]
        c.check(not bad, f"{s['_pid']}/{s['material_name']} ({s['role']}) chain is project-owned",
                f"BANNED: {bad}" if bad else f"chain depth {len(s.get('chain', []))}")

    print(f"\nD. every tree material lives under {MAT_DIR}")
    for s in all_slots:
        c.check(str(s["material_path"]).startswith(MAT_DIR),
                f"{s['_pid']}/{s['material_name']} in the project foliage folder", s["material_path"])

    print("\nE. no tree slot is translucent, instance AND parent")
    # Both, deliberately. `build_vegetation.py` set an OPAQUE override on every leaf instance whose parent was
    # the importer's translucent `MI_Default_Blend_DS`, the dump read the override back as BLEND_OPAQUE -- and
    # the canopy still rendered blended, with sky visible through the cards and a white haze of accumulated
    # layers over every crown. Checking only the instance's own value would have called that state clean.
    for s in all_slots:
        b = norm_enum(s.get("blend_mode", "?"))
        p = norm_enum(s.get("parent_blend_mode", "?"))
        ok = b in OK_BLEND and b not in TRANSLUCENT
        okp = p in OK_BLEND or p == "?"
        c.check(ok and okp, f"{s['_pid']}/{s['material_name']} blend mode",
                f"instance={b} parent={p}" + ("" if okp else "  <- PARENT IS TRANSLUCENT"))

    print("\nF. every tree slot compiles (zero instructions = failed compile = grey checker)")
    for s in all_slots:
        c.check(int(s.get("instructions", 0)) > 0,
                f"{s['_pid']}/{s['material_name']} compiles",
                f"{s.get('instructions')} instructions, {s.get('texture_samples')} texture samples")
        if int(s.get("texture_samples", 0)) == 0:
            c.note(f"!! {s['_pid']}/{s['material_name']} samples NO texture - it will render flat")

    print("\nG. leaf slots are two-sided foliage")
    for s in leaf_slots:
        c.check(norm_enum(s.get("shading_model")) == "MSM_TWO_SIDED_FOLIAGE",
                f"{s['_pid']}/{s['material_name']} shading model", norm_enum(s.get("shading_model")))
        c.check(bool(s.get("two_sided")), f"{s['_pid']}/{s['material_name']} is two-sided",
                str(s.get("two_sided")))

    # --- H. the pixels: nadir canopy vs sunlit ground ------------------------------------------------------
    print("\nH. rendered canopy vs SUNLIT GROUND, measured in the same nadir frame")
    measured = {}
    beauty, noveg, canopy, ground, dn, db = masks(tag, "fol_1_nadir", c)
    cover = float(canopy.mean())
    gcover = float(ground.mean())
    mp = write_mask_png(tag, "fol_1_nadir", beauty, canopy, ground)
    print(f"         canopy mask {cover * 100:.1f} % of frame, clean ground {gcover * 100:.1f} %  -> {mp}")
    exp = state.get("view_plan", {}).get("nadir", {}).get("expected_closure")
    if exp:
        c.note(f"layout predicts {exp * 100:.1f} % canopy closure in this footprint (sim)")
    ok_mask = c.check(MASK_COVER_MIN < cover < MASK_COVER_MAX, "canopy mask is not degenerate",
                      f"{cover * 100:.1f} % (gate {MASK_COVER_MIN * 100:.0f}-{MASK_COVER_MAX * 100:.0f} %)")
    ok_mask &= c.check(ground.sum() >= GROUND_MIN_PIX, "enough clean ground pixels to compare against",
                       f"{int(ground.sum()):,} (need {GROUND_MIN_PIX:,})")

    cs = stats(beauty, canopy)
    gs = stats(beauty, ground)
    print(f"         canopy  {fmt(cs)}")
    print(f"         ground  {fmt(gs)}")
    if ok_mask:
        sunlit = gs["p75"]          # "sunlit grass" = the lit part of the open ground in this same frame
        ratio = cs["p50"] / max(sunlit, 1e-6)
        measured = {"canopy_p50": cs["p50"], "canopy_p25": cs["p25"], "sunlit_ground": sunlit,
                    "ratio": ratio, "near_black_frac": cs["near_black_frac"], "cover": cover}
        c.check(ratio >= RATIO_MIN,
                "canopy median luma / sunlit ground luma", f"{ratio:.3f} (gate >= {RATIO_MIN})")
        c.check(cs["p25"] >= CANOPY_P25_MIN, "crown interiors (canopy luma p25) are not near-black",
                f"{cs['p25']:.3f} (gate >= {CANOPY_P25_MIN})")
        c.check(cs["near_black_frac"] <= NEAR_BLACK_FRAC_MAX,
                f"fraction of canopy below luma {NEAR_BLACK}",
                f"{cs['near_black_frac'] * 100:.1f} % (gate <= {NEAR_BLACK_FRAC_MAX * 100:.0f} %)")
        c.check(cs["g"] > cs["r"] and cs["g"] > cs["b"], "canopy reads GREEN, not grey or black",
                f"median rgb ({cs['r']:.3f}, {cs['g']:.3f}, {cs['b']:.3f})")
    else:
        c.note("H not evaluated: the mask is unusable, which is itself the failure above")

    # --- I. the backlit crown ------------------------------------------------------------------------------
    print("\nI. backlit crown (the frame the subsurface term exists for)")
    for view, gate in (("fol_2_crown_backlit", BACKLIT_P50_MIN), ("fol_3_crown_frontlit", None)):
        b2, nv2, can2, gnd2, _, _ = masks(tag, view, c)
        s2 = stats(b2, can2)
        p = write_mask_png(tag, view, b2, can2, gnd2)
        print(f"         {view}: canopy {can2.mean() * 100:.1f} % of frame -> {p}")
        print(f"           {fmt(s2)}")
        measured[view] = s2
        if not s2.get("n"):
            c.check(False, f"{view}: crown pixels found", "canopy mask is empty")
            continue
        if gate is not None:
            c.check(s2["p50"] >= gate, f"{view}: crown is not a black silhouette",
                    f"luma p50 {s2['p50']:.3f} (gate >= {gate})")
            c.check(s2["g"] > s2["r"] and s2["g"] > s2["b"],
                    f"{view}: backlit crown transmits GREEN",
                    f"median rgb ({s2['r']:.3f}, {s2['g']:.3f}, {s2['b']:.3f})")

    (ART / f"measured_{tag}.json").write_text(json.dumps(measured, indent=1))

    if a.compare:
        cp = ART / f"measured_{a.compare}.json"
        if cp.exists():
            old = json.loads(cp.read_text())
            print(f"\n--- {a.compare} -> {tag} ---")
            for k in ("canopy_p50", "canopy_p25", "sunlit_ground", "ratio", "near_black_frac"):
                if k in old and k in measured:
                    print(f"  {k:18s} {old[k]:.4f}  ->  {measured[k]:.4f}")
            for v in ("fol_2_crown_backlit", "fol_3_crown_frontlit"):
                if v in old and v in measured and old[v].get("n") and measured[v].get("n"):
                    print(f"  {v:18s} luma p50 {old[v]['p50']:.4f}  ->  {measured[v]['p50']:.4f}   "
                          f"near-black {old[v]['near_black_frac'] * 100:.1f} %  ->  "
                          f"{measured[v]['near_black_frac'] * 100:.1f} %")
        else:
            print(f"\n(no {cp} to compare against)")

    print("\n" + "=" * 100)
    if c.fails:
        print(f"FAILED: {len(c.fails)} check(s) — tag {tag!r} (sim)")
        for f in c.fails:
            print(f"  - {f}")
        print("DO NOT call the foliage materials done.")
        return 1
    print(f"PASSED: every foliage-material check is green — tag {tag!r} (sim)")
    print("Still LOOK at the beauty PNGs and the mask_*.png overlays; this checks numbers, not taste.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
