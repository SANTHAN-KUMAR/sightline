"""DJI R-JPEG radiometric reader (doc 5.5b) on top of thermal_parser (MIT, installed from git, see pyproject).

Why a shim: thermal_parser's setup.py installs its DJI Thermal SDK DLLs and exiftool via `data_files` into
`<venv>\\plugins\\...`, but its code looks for them in `<site-packages>\\plugins`. We point it at the real
location instead of copying proprietary DLLs around. The DJI Thermal SDK binaries are under the DJI EULA.
"""
from __future__ import annotations

import sys
from pathlib import Path

SDK = "dji_thermal_sdk_v1.7_20241205"


def plugins_dir() -> Path:
    import thermal_parser

    candidates = [Path(sys.prefix) / "plugins", Path(thermal_parser.__file__).resolve().parents[1] / "plugins"]
    for c in candidates:
        if (c / SDK).exists():
            return c
    raise FileNotFoundError(f"thermal_parser plugins not found in {candidates}")


def default_filepaths() -> list[str]:
    root = plugins_dir()
    rel = root / SDK / "windows" / "release_x64"
    return [str(rel / "libdirp.dll"), str(rel / "libv_dirp.dll"), str(rel / "libv_iirp.dll"),
            str(root / "exiftool-12.35.exe")]


def make_thermal(dtype=None):
    """Return a thermal_parser.Thermal whose DLL/exiftool paths point at the installed plugins folder."""
    import numpy as np
    import thermal_parser.thermal as tt

    tt.get_default_filepaths = default_filepaths  # used by Thermal.__init__
    return tt.Thermal(dtype=dtype or np.float32)
