"""F8b: the ONE fine-tune. COCO-pretrained YOLO26s -> simulator tiles, seed-held-out (SOLUTION_DOC §5.5c).

**This is deliberately a small amount of work, and that is the design, not a compromise.** §5.5c: the renderer is
the deployment domain, so the domain gap that motivates §5.5's three-stage recipe and six real datasets does not
exist here. "Fine-tune from COCO on simulator frames only. Skip stage 1 of §5.5 entirely. Take stock YOLO26s
weights, fine-tune for 30-50 epochs on 10-20k rendered tiles, at 1024." One overnight run.

The verified 8 GB / Windows facts from §5.5, all encoded in :data:`BASE_ARGS`:

* ``batch=-1`` auto-sizes to about 60 % of GPU memory (do not hand-tune it);
* AMP is on by default -- left on;
* ``cache='disk'``, never ``'ram'``: this machine has 16 GB and the editor may be using most of it;
* ``workers<=4`` on Windows;
* training must sit under ``if __name__ == "__main__":`` (Windows spawn) -- :func:`main` is that entry point;
* ``project``/``name`` point at **D:** so nothing lands in ``C:\\Users``.

Two guards run before a single epoch, because both failures are silent and expensive:

1. :func:`editor_is_running` -- the handbook forbids GPU work while Unreal is up; training beside it either OOMs
   or produces meaningless epoch times.
2. :func:`dataset_is_validated` -- `tools/capture/validate.py` must have passed on every capture run in the
   dataset. Training on a run whose drone photographed its own propellers is the exact failure
   `docs/QUALITY_GATE.md` was written about.

Nothing in this module imports torch or ultralytics at module scope.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BASE_ARGS",
    "TrainConfig",
    "build_train_args",
    "dataset_is_validated",
    "editor_is_running",
    "main",
    "preflight",
    "train",
]

REPO = Path(__file__).resolve().parents[2]

#: The verified-safe Ultralytics arguments for this machine. Anything not here is left at the library default.
BASE_ARGS: dict[str, Any] = {
    "imgsz": 1024,          # §5.5c: tiles are cut at 1024 and inference runs at 1024 -> one tile px = one frame px
    "epochs": 40,           # §5.5c: "30-50 epochs"
    "batch": -1,            # auto-size to ~60 % of VRAM (verified behaviour, §5.5)
    "amp": True,
    "cache": "disk",        # NOT 'ram': 16 GB shared with the editor
    "workers": 4,           # Windows dataloader ceiling
    "device": 0,
    "patience": 15,
    "close_mosaic": 10,     # §5.5 stage 2
    "pretrained": True,     # COCO weights are the starting point; that is the whole recipe
    "seed": 0,
    "deterministic": True,
    "val": True,
    "plots": True,
    "exist_ok": True,
    # Augmentation: geometry only. §5.5c has domain randomisation OFF, so colour jitter would be re-introducing
    # by the back door exactly the variation the decision says to leave out. Flips and rotation are pose
    # variation a nadir camera genuinely sees.
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 180.0,       # a nadir frame has no canonical up
    "flipud": 0.5, "fliplr": 0.5,
    "translate": 0.1, "scale": 0.2, "shear": 0.0, "perspective": 0.0,
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
}


@dataclass(slots=True)
class TrainConfig:
    data_yaml: str
    model: str = "yolo26s.pt"
    project: str = str(REPO / "models" / "detect")
    name: str = "f8b_sim"
    runs: list[str] = field(default_factory=list)  # capture runs the dataset was built from (for the guard)
    overrides: dict[str, Any] = field(default_factory=dict)
    allow_editor_running: bool = False
    allow_unvalidated: bool = False

    def args(self) -> dict[str, Any]:
        return build_train_args(self)


def build_train_args(cfg: TrainConfig) -> dict[str, Any]:
    a = dict(BASE_ARGS)
    a.update({"data": str(Path(cfg.data_yaml).resolve()), "project": cfg.project, "name": cfg.name})
    a.update(cfg.overrides)
    if str(a.get("cache")) == "ram":
        raise ValueError("cache='ram' will OOM this 16 GB machine; SOLUTION_DOC 5.5 says use 'disk'")
    if int(a.get("workers", 0)) > 4:
        raise ValueError("workers > 4 is unreliable on Windows (SOLUTION_DOC 5.5)")
    if not str(a["project"]).lower().startswith("d:"):
        raise ValueError(f"project must be on D: (project rule), got {a['project']!r}")
    return a


# --- guards --------------------------------------------------------------------------------------------------
def editor_is_running() -> list[str]:
    """Names of any running Unreal process. Empty list = the GPU is ours."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is pinned, this is belt and braces
        return []
    names = []
    for p in psutil.process_iter(["name"]):
        n = (p.info.get("name") or "")
        if "Unreal" in n or n.startswith("SightlineSim"):
            names.append(n)
    return sorted(set(names))


def dataset_is_validated(run_dirs: list[str] | list[Path], *, timeout_s: float = 900) -> tuple[bool, str]:
    """Run `tools/capture/validate.py` on every capture run. Returns (ok, report).

    This shells out on purpose: `tools/capture/**` belongs to another lane and must not be imported and
    monkey-patched from here. Its exit code is the contract ("exits non-zero and tells you not to train").
    """
    if not run_dirs:
        return False, "no capture runs given: refusing to certify a dataset that came from nowhere"
    lines = []
    ok = True
    uv = shutil.which("uv") or r"D:\Tools\uv\uv.exe"
    for d in run_dirs:
        cmd = [uv, "run", "python", str(REPO / "tools" / "capture" / "validate.py"), str(d)]
        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                               timeout=timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:  # pragma: no cover - environment failure
            return False, f"could not run validate.py on {d}: {e}"
        lines.append(f"--- {d} -> exit {r.returncode}\n{r.stdout[-4000:]}")
        ok = ok and r.returncode == 0
    return ok, "\n".join(lines)


def preflight(cfg: TrainConfig) -> dict[str, Any]:
    """Everything that must be true before an epoch runs. Raises with the reason; never warns and continues."""
    data = Path(cfg.data_yaml)
    if not data.exists():
        raise FileNotFoundError(f"data yaml not found: {data}. Build it with sightline.detect.dataset first.")
    manifest_p = data.parent / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    splits = manifest.get("splits", {})
    train_seeds = set(splits.get("train", {}).get("seeds", []))
    val_seeds = set(splits.get("val", {}).get("seeds", []))
    if manifest and train_seeds & val_seeds:
        raise ValueError(f"train and val share scenario seed(s) {sorted(train_seeds & val_seeds)}: "
                         "SOLUTION_DOC 5.5c step 3 forbids it")
    if manifest and not val_seeds:
        raise ValueError("the dataset has no val split; the operating threshold must be frozen on validation "
                         "data (SOLUTION_DOC 5.5)")

    editors = editor_is_running()
    if editors and not cfg.allow_editor_running:
        raise RuntimeError(
            f"the Unreal editor is running ({', '.join(editors)}). This machine has 8 GB of VRAM and the "
            "handbook forbids GPU work beside the editor. Close it, or pass allow_editor_running=True and say "
            "so in the report."
        )

    runs = cfg.runs or manifest.get("capture_runs", [])
    if not cfg.allow_unvalidated:
        ok, report = dataset_is_validated(runs)
        if not ok:
            raise RuntimeError("tools/capture/validate.py did not pass on every capture run - DO NOT TRAIN.\n"
                               + report)
    return {"manifest": manifest, "editors": editors, "runs": [str(r) for r in runs]}


# --- the run -------------------------------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run the fine-tune. Writes `train_manifest.json` beside the weights so the number is reproducible."""
    info = preflight(cfg)
    args = build_train_args(cfg)

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(cfg.model)
    results = model.train(**args)
    elapsed = time.time() - t0

    save_dir = Path(getattr(results, "save_dir", Path(args["project"]) / args["name"]))
    out = {
        "product": "sightline.detect.train",
        "domain": "sim",
        "recipe": "SOLUTION_DOC 5.5c: one fine-tune from COCO on simulator frames only, randomisation off",
        "model": cfg.model,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
        "elapsed_s": elapsed,
        "save_dir": str(save_dir),
        "weights": str(save_dir / "weights" / "best.pt"),
        "manifest": info["manifest"],
        "note": "Ultralytics' own P/R are quoted at max-F1 confidence. They are NOT the operating point; "
                "use sightline.detect.threshold.choose_operating_threshold on the val split.",
    }
    (save_dir / "train_manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="F8b: fine-tune YOLO26s on simulator tiles (SOLUTION_DOC 5.5c)")
    ap.add_argument("data_yaml")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--name", default="f8b_sim")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--run", action="append", default=[], help="capture run dir to validate (repeatable)")
    ap.add_argument("--allow-editor-running", action="store_true")
    ap.add_argument("--allow-unvalidated", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)

    overrides: dict[str, Any] = {}
    for k in ("epochs", "imgsz", "batch"):
        if getattr(a, k) is not None:
            overrides[k] = getattr(a, k)
    cfg = TrainConfig(data_yaml=a.data_yaml, model=a.model, name=a.name, runs=a.run, overrides=overrides,
                      allow_editor_running=a.allow_editor_running, allow_unvalidated=a.allow_unvalidated)
    if a.preflight_only:
        print(json.dumps({"preflight": preflight(cfg), "args": build_train_args(cfg)}, indent=2, default=str))
        return 0
    print(json.dumps(train(cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":  # Windows spawn: training MUST be under this guard (SOLUTION_DOC 5.5)
    sys.exit(main())
