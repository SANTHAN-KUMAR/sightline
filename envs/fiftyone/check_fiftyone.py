"""Functional FiftyOne check, run with envs/fiftyone/.venv. Prints one JSON line (parsed by tests/test_stack.py).

Creates a dataset with 3 samples carrying detections, runs an evaluation, deletes it, and proves the MongoDB
database (fiftyone-db) lives on D:.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FO_ROOT = REPO / "data" / "fiftyone"
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(FO_ROOT / "db"))
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", str(FO_ROOT / "datasets"))
os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(FO_ROOT / "zoo" / "datasets"))
os.environ.setdefault("FIFTYONE_MODEL_ZOO_DIR", str(FO_ROOT / "zoo" / "models"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("MPLCONFIGDIR", r"D:\Tools\cache\matplotlib")

import fiftyone as fo
import numpy as np
from PIL import Image

img_dir = FO_ROOT / "check_images"
img_dir.mkdir(parents=True, exist_ok=True)
samples = []
for i in range(3):
    p = img_dir / f"frame_{i}.png"
    Image.fromarray((np.random.default_rng(i).random((64, 96, 3)) * 255).astype(np.uint8)).save(p)
    s = fo.Sample(filepath=str(p))
    s["ground_truth"] = fo.Detections(detections=[fo.Detection(label="human", bounding_box=[0.1, 0.1, 0.2, 0.3])])
    s["predictions"] = fo.Detections(detections=[
        fo.Detection(label="human", bounding_box=[0.11, 0.1, 0.2, 0.3], confidence=0.9),
        fo.Detection(label="human", bounding_box=[0.6, 0.6, 0.1, 0.1], confidence=0.3)])  # a false positive
    samples.append(s)

name = "sightline-stack-check"
if fo.dataset_exists(name):
    fo.delete_dataset(name)
ds = fo.Dataset(name)
ds.add_samples(samples)
res = ds.evaluate_detections("predictions", gt_field="ground_truth", eval_key="ev", iou=0.5)
tp, fp, fn = ds.sum("ev_tp"), ds.sum("ev_fp"), ds.sum("ev_fn")
n = len(ds)
ds.delete()
deleted = not fo.dataset_exists(name)
db_dir = fo.config.database_dir
db_files = sum(1 for _ in Path(db_dir).rglob("*")) if Path(db_dir).exists() else 0
home_fo = Path.home() / ".fiftyone"
print(json.dumps({"version": fo.__version__, "samples": n, "tp": tp, "fp": fp, "fn": fn, "deleted": deleted,
                  "database_dir": db_dir, "db_files": db_files, "home_dotfiftyone_exists": home_fo.exists(),
                  "python": sys.version.split()[0]}))
