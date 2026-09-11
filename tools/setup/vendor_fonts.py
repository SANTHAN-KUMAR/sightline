"""Vendor the UI typefaces into the repo so the dashboard stays fully offline.

    uv run python tools/setup/vendor_fonts.py

`app/map/headless_check.mjs` asserts **zero external network requests**: the C2 map has to work on a field
laptop with no internet, which is the whole point of the PMTiles basemap and the vendored MapLibre. Linking
Google Fonts would break that test and, more importantly, the capability it protects - the first thing that
happens in a disaster is that the network goes away.

So the woff2 files are downloaded once, at build time, and `@font-face` points at local paths. Only the
**latin** subset is taken: the full family is a dozen files per weight covering scripts this UI never renders,
and the dashboard must stay small enough to ship on a laptop.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "app/map/vendor/fonts"
CSS_URL = ("https://fonts.googleapis.com/css2?"
           "family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Poppins:wght@400;500&display=swap")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")
#: the latin block. Google emits one @font-face per subset; everything else is dead weight here.
LATIN = "U+0000-00FF"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    css = fetch(CSS_URL).decode("utf-8")

    blocks = re.findall(r"@font-face\s*\{[^}]*\}", css)
    print(f"{len(blocks)} @font-face blocks; keeping the {LATIN} (latin) subset only")

    kept, out_css = 0, []
    for b in blocks:
        if LATIN not in b:
            continue
        fam = re.search(r"font-family:\s*'([^']+)'", b)
        wt = re.search(r"font-weight:\s*(\d+)", b)
        url = re.search(r"url\((https://[^)]+\.woff2)\)", b)
        if not (fam and wt and url):
            continue
        slug = fam.group(1).lower().replace(" ", "-")
        name = f"{slug}-{wt.group(1)}.woff2"
        data = fetch(url.group(1))
        (OUT / name).write_bytes(data)
        kept += 1
        print(f"  {name:34s} {len(data) / 1024:6.1f} KB")
        out_css.append(
            "@font-face{\n"
            f"  font-family:'{fam.group(1)}';\n"
            "  font-style:normal;\n"
            f"  font-weight:{wt.group(1)};\n"
            "  font-display:swap;\n"
            f"  src:url('fonts/{name}') format('woff2');\n"
            f"  unicode-range:{LATIN},U+0131,U+0152-0153,U+02BB-02BC,U+2000-206F,U+2122,U+2192,U+2212;\n"
            "}"
        )

    if not kept:
        print("FAIL: no latin subset found - the CSS format changed; inspect it before trusting this.")
        return 2

    header = ("/* Vendored by tools/setup/vendor_fonts.py. The dashboard must make ZERO external requests\n"
              "   (app/map/headless_check.mjs asserts it), because a field laptop in a disaster has no\n"
              "   network. Do not replace these with a CDN link. */\n")
    (OUT.parent / "fonts.css").write_text(header + "\n".join(out_css) + "\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.glob("*.woff2")) / 1024
    print(f"\n{kept} faces, {total:.0f} KB total -> app/map/vendor/fonts.css")
    return 0


if __name__ == "__main__":
    sys.exit(main())
