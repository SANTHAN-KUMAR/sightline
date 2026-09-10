r"""Load a Sightline evaluation manifest into FiftyOne and open the error browser (SOLUTION_DOC 5.12).

**Run this with the FiftyOne interpreter, not the main env.** FiftyOne pins conflict with the main
environment (opencv-python-headless vs opencv-python, starlette, pymongo), so it lives in `envs/fiftyone`.
This file imports only `fiftyone` and the standard library, and nothing from `sightline`, so the two
environments never have to agree on anything except the JSON manifest.

    D:\Sightline\envs\fiftyone\.venv\Scripts\python.exe ^
        D:\Sightline\sightline\eval\fiftyone_browse.py D:\Sightline\_artifacts\eval\fiftyone\<clip>.json

Options:
    --no-app        load the dataset and print a summary, do not open the browser (CI / headless check)
    --persist       keep the dataset in the FiftyOne database after the process exits
    --name NAME     override the dataset name from the manifest

Views created for the §5.12 error analysis:
    has_fp   frames with an unmatched prediction  (glint, roofing sheets, debris)
    has_fn   frames with a missed survivor        (canopy, head-only in turbid water)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FO_ROOT = REPO / "data" / "fiftyone"
# Keep every byte on D: (project rule) and off the telemetry endpoint.
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(FO_ROOT / "db"))
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", str(FO_ROOT / "datasets"))
os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(FO_ROOT / "zoo" / "datasets"))
os.environ.setdefault("FIFTYONE_MODEL_ZOO_DIR", str(FO_ROOT / "zoo" / "models"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("MPLCONFIGDIR", r"D:\Tools\cache\matplotlib")

# FiftyOne deliberately lives in its own environment (envs/fiftyone; its pins conflict with the main env), so a
# hard import here would break `import sightline.eval` for every other caller. Bind it lazily and fail with a
# useful message only when this module is actually used.
try:
    import fiftyone as fo  # noqa: E402
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on which env is active
    _FO_IMPORT_ERROR = _exc
    fo = None  # type: ignore[assignment]
else:
    _FO_IMPORT_ERROR = None


def require_fiftyone():
    """Raise a directed error instead of a bare ModuleNotFoundError deep inside a call."""
    if fo is None:
        raise RuntimeError(
            "fiftyone is not installed in this environment. It lives in its own project at envs/fiftyone "
            "(its pins conflict with the main env). Run this module with that environment's interpreter."
        ) from _FO_IMPORT_ERROR
    return fo

ATTRS = ("gt_id", "occlusion", "posture", "submersion", "visible_fraction", "size_px", "uncertain",
         "eval", "modality", "tile_idx", "thermal_c")


def _detections(items, crowd_key="iscrowd"):
    dets = []
    for d in items:
        kwargs = {k: d[k] for k in ATTRS if k in d and d[k] is not None}
        if d.get(crowd_key):
            kwargs["iscrowd"] = True
        det = fo.Detection(label=d["label"], bounding_box=list(d["bounding_box"]), **kwargs)
        if d.get("confidence") is not None:
            det.confidence = float(d["confidence"])
        dets.append(det)
    return fo.Detections(detections=dets)


def load(manifest_path: Path, name: str | None = None, persist: bool = False) -> fo.Dataset:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "sightline-fiftyone-manifest/1":
        raise SystemExit(f"{manifest_path} is not a sightline-fiftyone-manifest/1 document")
    ds_name = name or manifest["dataset_name"]
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)
    ds = fo.Dataset(ds_name, persistent=persist)
    ds.info = {k: v for k, v in manifest.items() if k != "samples"}
    # Domain is stamped on the dataset AND on every sample: a sim view and a real view are never one view.
    samples, skipped = [], 0
    for s in manifest["samples"]:
        fp = s.get("filepath") or ""
        if not fp or not Path(fp).exists():
            skipped += 1
            continue
        sample = fo.Sample(filepath=fp, tags=list(s.get("tags", [])))
        for field in ("frame_idx", "clip_id", "domain", "zone", "agl_m", "time_of_day", "modality",
                      "gimbal_pitch_deg", "gsd_cm_px", "n_fp", "n_fn"):
            if field in s:
                sample[field] = s[field]
        sample["ground_truth"] = _detections(s.get("ground_truth", []))
        sample["predictions"] = _detections(s.get("predictions", []))
        samples.append(sample)
    if samples:
        ds.add_samples(samples)
    if skipped:
        print(f"[warn] {skipped} frame(s) had no readable image and were skipped "
              "(export the frames alongside the manifest to browse them)", file=sys.stderr)
    return ds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--no-app", action="store_true", help="do not open the browser")
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    ds = load(args.manifest, name=args.name, persist=args.persist)
    info = ds.info
    print(json.dumps({
        "dataset": ds.name,
        "domain": info.get("domain"),
        "clip_id": info.get("clip_id"),
        "operating_conf": info.get("operating_conf"),
        "samples": len(ds),
        "with_false_positives": len(ds.match_tags("has_fp")),
        "with_false_negatives": len(ds.match_tags("has_fn")),
        "database_dir": fo.config.database_dir,
    }))
    if args.no_app:
        return 0
    session = fo.launch_app(ds.match_tags(["has_fp", "has_fn"]))
    print("FiftyOne is showing the frames with an error. Ctrl-C to close.")
    session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
