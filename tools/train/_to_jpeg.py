"""Re-encode the tiled dataset's PNG crops as JPEG so the upload is not 7x larger than it needs to be.

    uv run python tools/train/_to_jpeg.py _artifacts/yolo/sim 95

The capture wrote its frames with `--jpeg 95`, and the tiler then cut PNG crops out of those JPEGs. So these
pixels have already been through JPEG once; re-encoding at q95 adds one more generation on data that is
already lossy, which is not a meaningful change for detection - and it turns a 2,148 MB upload into roughly
300 MB. On a home uplink that is the difference between a quarter of an hour and three minutes.

Labels are untouched: YOLO labels are normalised, so nothing about the boxes changes with the container.
Ultralytics finds images by stem, so the extension swap needs no edit anywhere else.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

root = Path(sys.argv[1] if len(sys.argv) > 1 else "_artifacts/yolo/sim")
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 95
pngs = sorted(root.glob("images/*/*.png"))
print(f"{len(pngs)} PNG tiles -> JPEG q{Q}")


def conv(p: Path) -> int:
    im = cv2.imread(str(p))
    if im is None:
        print(f"  UNREADABLE, kept as PNG: {p.name}")
        return 0
    out = p.with_suffix(".jpg")
    if not cv2.imwrite(str(out), im, [int(cv2.IMWRITE_JPEG_QUALITY), Q]):
        print(f"  WRITE FAILED, kept as PNG: {p.name}")
        return 0
    n = out.stat().st_size
    p.unlink()
    return n


with ThreadPoolExecutor(max_workers=8) as ex:
    total = sum(ex.map(conv, pngs))

print(f"wrote {total / 1048576:.0f} MB of JPEG")
for split in ("train", "val"):
    d = root / "images" / split
    n_jpg, n_png = len(list(d.glob("*.jpg"))), len(list(d.glob("*.png")))
    n_txt = len(list((root / "labels" / split).glob("*.txt")))
    print(f"  {split}: {n_jpg} jpg, {n_png} png left, {n_txt} labels")
    if n_png:
        print("  WARNING: PNGs remain - they still upload at full size")
