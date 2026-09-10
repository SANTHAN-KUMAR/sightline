"""DEM sampling for the geolocation chain (SOLUTION_DOC §5.7 step 5).

Copernicus DEM GLO-30 (< 4 m LE90 vertical, free) is the elevation source the doc adopts. A tile covering the
Wayanad area of operations is committed at ``data/dem/wayanad_glo30_76.10_11.45_76.18_11.53.tif``.

Design notes
------------
* The whole tile is read **once** into a numpy array and cached on the sampler (289 x 289 float32 = 334 KB for
  the committed tile), so a ray-march does thousands of samples with no I/O.
* Sampling is bilinear on pixel *centres*, with NaN for out-of-bounds and for the raster's nodata value. A NaN
  never silently becomes 0 m — the caller must decide to fall back.
* GLO-30 heights are orthometric (EGM2008). `Telemetry.alt_msl_m` is also mean-sea-level, so the two are
  directly comparable; a switch to ellipsoidal altitude would need a geoid correction (not implemented, and the
  chain reports `dem_source` so a fix is traceable).
* The FloodValley simulator terrain is **synthetic** and does not match this real tile (sim base 1046.007 m ASL
  vs GLO-30 1060.3 m at the same geopoint). The DEM path is therefore exercised against the real tile for
  correctness, while sim runs use `flat_plane` / `water_plane` against the ground-truth AGL.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = ["DemSampler", "DEFAULT_DEM_PATH", "load_default_dem"]

#: The tile committed for the Wayanad area of operations (doc §5.1 scenario area).
DEFAULT_DEM_PATH = Path(__file__).resolve().parents[2] / "data" / "dem" / "wayanad_glo30_76.10_11.45_76.18_11.53.tif"

#: Copernicus GLO-30 vertical error, 1σ. LE90 < 4 m -> σ ≈ 4 / 1.6449 ≈ 2.4 m; the doc's budget uses the
#: pessimistic "AGL from a DEM = ±4 m" figure, so that is what the chain defaults to.
GLO30_SIGMA_M = 4.0


class DemSampler:
    """A cached, bilinearly-interpolated digital elevation model in EPSG:4326.

    Heights are metres above mean sea level (orthometric). `elevation()` returns NaN outside the tile or on
    nodata; it never extrapolates.
    """

    def __init__(self, path: str | Path, band: int = 1, name: str = "") -> None:
        import rasterio  # local import: keeps `sightline.geo` importable without the geo dependency group

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"DEM not found: {self.path}")
        with rasterio.open(self.path) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 4326:
                raise ValueError(f"DEM must be EPSG:4326 lat/lon, got {ds.crs} ({self.path.name})")
            self._z = ds.read(band).astype(np.float64)
            self._transform = ds.transform
            self._inv = ~ds.transform
            self._nodata = ds.nodata
            self.bounds = tuple(ds.bounds)  # (left, bottom, right, top) = (lon_min, lat_min, lon_max, lat_max)
            self.shape = (int(ds.height), int(ds.width))
            self.res_deg = (float(abs(ds.transform.a)), float(abs(ds.transform.e)))
        if self._nodata is not None and not (isinstance(self._nodata, float) and math.isnan(self._nodata)):
            self._z = np.where(self._z == self._nodata, np.nan, self._z)
        self.name = name or self.path.name
        #: approximate ground sample distance of the DEM in metres, for choosing a march step
        self.res_m = float(self.res_deg[0] * 111_320.0 * math.cos(math.radians((self.bounds[1] + self.bounds[3]) / 2)))

    # -- introspection ---------------------------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DemSampler({self.name!r}, shape={self.shape}, bounds={tuple(round(b, 5) for b in self.bounds)})"

    def contains(self, lat: float, lon: float) -> bool:
        lon_min, lat_min, lon_max, lat_max = self.bounds
        return bool(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max)

    @property
    def array(self) -> np.ndarray:
        """The cached height grid (read-only view). Rows run north -> south."""
        v = self._z.view()
        v.flags.writeable = False
        return v

    # -- sampling --------------------------------------------------------------------------------------------
    def elevation(self, lat: float, lon: float) -> float:
        """Bilinear height in metres MSL at one point; NaN outside the tile or on nodata."""
        return float(self.elevations(np.array([lat], float), np.array([lon], float))[0])

    def elevations(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Vectorised bilinear sampling. Inputs are broadcast to a common shape; output is float64 with NaNs."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        # inverse affine applied by hand: `~transform * (x, y)` is deprecated for array operands, and this
        # keeps the whole march allocation-free. (a, b, c, d, e, f) is the row-major affine.
        a, b, c0_, d, e, f0_ = self._inv.a, self._inv.b, self._inv.c, self._inv.d, self._inv.e, self._inv.f
        col_f = a * lon + b * lat + c0_
        row_f = d * lon + e * lat + f0_
        # move to pixel-centre coordinates so bilinear weights are correct
        c = col_f - 0.5
        r = row_f - 0.5
        n_row, n_col = self.shape

        c0 = np.floor(c).astype(np.int64)
        r0 = np.floor(r).astype(np.int64)
        fc = c - c0
        fr = r - r0
        inside = (c >= -0.5) & (c <= n_col - 0.5) & (r >= -0.5) & (r <= n_row - 0.5)

        c0c = np.clip(c0, 0, n_col - 1)
        r0c = np.clip(r0, 0, n_row - 1)
        c1c = np.clip(c0 + 1, 0, n_col - 1)
        r1c = np.clip(r0 + 1, 0, n_row - 1)

        z00 = self._z[r0c, c0c]
        z01 = self._z[r0c, c1c]
        z10 = self._z[r1c, c0c]
        z11 = self._z[r1c, c1c]
        top = z00 * (1.0 - fc) + z01 * fc
        bot = z10 * (1.0 - fc) + z11 * fc
        out = top * (1.0 - fr) + bot * fr
        return np.where(inside, out, np.nan)

    # -- convenience -----------------------------------------------------------------------------------------
    def slope_deg(self, lat: float, lon: float) -> float:
        """Local surface slope in degrees, from centred differences one DEM cell apart. NaN when unsampleable."""
        dlat, dlon = self.res_deg[1], self.res_deg[0]
        n = self.elevation(lat + dlat, lon)
        s = self.elevation(lat - dlat, lon)
        e = self.elevation(lat, lon + dlon)
        w = self.elevation(lat, lon - dlon)
        if any(math.isnan(v) for v in (n, s, e, w)):
            return float("nan")
        m_lat = 2.0 * dlat * 111_132.0
        m_lon = 2.0 * dlon * 111_320.0 * math.cos(math.radians(lat))
        return math.degrees(math.atan(math.hypot((n - s) / m_lat, (e - w) / m_lon)))


_DEFAULT: DemSampler | None = None


def load_default_dem() -> DemSampler:
    """The committed Wayanad GLO-30 tile, loaded once per process."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DemSampler(DEFAULT_DEM_PATH, name="copernicus_glo30/wayanad")
    return _DEFAULT
