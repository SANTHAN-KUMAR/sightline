"""Ingest and synchronisation — feature F7, SOLUTION_DOC §5.4.

> "Turn a video stream or recorded flight plus a telemetry log into per-frame (image, pose, intrinsics,
>  timestamp) tuples at the processed frame rate."

**One public entry point.** `open_clip(path, **opts)` auto-detects the source and returns a `Clip`, which is
an iterator of `FrameBundle` (the frozen contract in `sightline/schemas.py`). Every downstream module —
detection, geolocation, tracking, coverage, evaluation — consumes `FrameBundle` and therefore never learns
which of the five sources it came from, which is the §5.4 "Integration" requirement::

    from sightline.ingest import open_clip

    clip = open_clip(r"D:\\Sightline\\_artifacts\\captures\\floodvalley_pass1", decimate=3)   # 30 -> 10 FPS
    for bundle in clip:
        detections = detect(bundle.rgb, bundle.intrinsics)
        fix = geolocate(detections[0], bundle.telemetry, bundle.intrinsics)
    thumb = clip.load_frame(best_observation.frame_idx)     # ANY decoded frame, decimated or not
    clip.close()

Sources, in the order `detect_source()` tries them:

| detected | trigger | reader |
|---|---|---|
| `sim`        | a directory (or file) containing `capture.json` | `sim.SimExportReader` — the primary path |
| `capture_run`| a directory containing `telemetry.csv` with `tools/capture/run.py`'s header | `sim.CaptureRunReader` |
| `airsim_rec` | a directory (or file) containing `airsim_rec.txt` | `sim.AirSimRecReader` |
| `dji_srt`    | a `.srt` file, or a video with a sibling `.srt` | `dji_srt.parse_srt` |
| `mavlink`    | a `.tlog` / `.bin` file | `mavlink.read_mavlink` (stretch) |
| `ulog`       | a `.ulg` file | `ulog.read_ulog` (stretch) |
| `video`      | a video file with no telemetry beside it | metadata only; telemetry must be supplied |

**The gimbal quaternion convention** used by every reader is defined once, in `spec.gimbal_quat_from_euler`
/ `spec.gimbal_quat_from_frd_quat`, and is the only one consistent with the frozen schema's own
`Telemetry.gimbal_pitch_deg()` helper (-90 = straight down). Producers must not hand-roll it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from sightline.ingest import sim as sim_mod
from sightline.ingest import spec
from sightline.ingest.align import BootClock, TelemetrySeries, cross_correlate_lag, estimate_t_offset
from sightline.ingest.decimate import FrameIndex, FrameRef, decimate, decimation_for_fps, K_5FPS, K_10FPS
from sightline.ingest.decode import DecodedFrame, VideoReader, open_video, read_image_bgr, read_thermal
from sightline.ingest.dji_srt import SrtEntry, SrtParseReport, parse_srt, srt_to_telemetry
from sightline.ingest.sim import (
    AirSimRecReader,
    CaptureRunReader,
    NoiseModel,
    SimExportReader,
    inject_noise,
    is_airsim_rec,
    is_capture_run,
    is_sim_export,
)
from sightline.ingest.spec import CaptureFormatError, CaptureManifest, SimCaptureWriter
from sightline.schemas import FrameBundle, Intrinsics, Telemetry

__all__ = [
    "open_clip",
    "Clip",
    "detect_source",
    "SOURCE_TYPES",
    # re-exported so callers need one import
    "TelemetrySeries",
    "BootClock",
    "CaptureRunReader",
    "SimExportReader",
    "AirSimRecReader",
    "FrameIndex",
    "FrameRef",
    "DecodedFrame",
    "VideoReader",
    "open_video",
    "read_image_bgr",
    "read_thermal",
    "NoiseModel",
    "inject_noise",
    "SimCaptureWriter",
    "CaptureManifest",
    "CaptureFormatError",
    "SrtEntry",
    "SrtParseReport",
    "parse_srt",
    "srt_to_telemetry",
    "estimate_t_offset",
    "cross_correlate_lag",
    "decimate",
    "decimation_for_fps",
    "K_5FPS",
    "K_10FPS",
    "spec",
]

SOURCE_TYPES = ("sim", "capture_run", "airsim_rec", "dji_srt", "mavlink", "ulog", "video")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v", ".lrf")


def detect_source(path: str | os.PathLike[str]) -> str:
    """Classify a path into one of `SOURCE_TYPES`. Raises `FileNotFoundError` when nothing matches."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if is_sim_export(p):
        return "sim"
    if is_capture_run(p):
        return "capture_run"
    if is_airsim_rec(p):
        return "airsim_rec"
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.suffix.lower() == ".srt":
                return "dji_srt"
        raise CaptureFormatError(
            f"{p} is a directory but holds no {spec.MANIFEST_NAME}, {sim_mod.CAPTURE_RUN_TELEMETRY}, "
            f"{spec.AIRSIM_REC} or .srt file"
        )
    suffix = p.suffix.lower()
    if suffix == ".srt":
        return "dji_srt"
    if suffix in (".tlog", ".bin", ".log"):
        return "mavlink"
    if suffix == ".ulg":
        return "ulog"
    if suffix in _VIDEO_SUFFIXES:
        return "dji_srt" if _sibling_srt(p) else "video"
    raise CaptureFormatError(f"{p}: unrecognised source (suffix {suffix!r})")


def _sibling_srt(video: Path) -> Path | None:
    for cand in (video.with_suffix(".SRT"), video.with_suffix(".srt")):
        if cand.is_file():
            return cand
    return None


def _sibling_video(other: Path) -> Path | None:
    for suffix in _VIDEO_SUFFIXES:
        for cand in (other.with_suffix(suffix), other.with_suffix(suffix.upper())):
            if cand.is_file():
                return cand
    return None


class Clip:
    """One recorded flight, from any source, as a stream of `FrameBundle`.

    A `Clip` **is** an `Iterator[FrameBundle]`; iterating it twice restarts the stream. It also exposes the
    things the replay harness (§3.3, R12) and the evidence path (§5.4 "Decimation") need:

    * `t_offset_s` — settable per clip, applied to every telemetry lookup from that moment on;
    * `estimate_t_offset()` — the take-off cross-correlation, when the clip has decodable video;
    * `frame_index` — every DECODED frame, including the ones decimation skipped;
    * `load_frame(frame_idx)` — the pixels of any of them, for `Evidence.thumb_uri`;
    * `telemetry_truth` — the clean simulator pose, kept beside the noised one for evaluation (§5.7).
    """

    def __init__(self, *, source_type: str, clip_id: str, telemetry: TelemetrySeries,
                 intrinsics: Intrinsics, frame_index: FrameIndex, source_path: str,
                 telemetry_truth: TelemetrySeries | None = None,
                 thermal_intrinsics: Intrinsics | None = None, thermal_is_radiometric: bool = False,
                 video_reader: VideoReader | None = None, video_start_utc: float = 0.0,
                 k: int = 1, load_images: bool = True, load_thermal: bool = True,
                 max_frames: int = 0, reports: dict[str, Any] | None = None,
                 domain: str = "sim") -> None:
        self.source_type = source_type
        self.clip_id = clip_id
        self.source_path = source_path
        self.telemetry = telemetry
        self.telemetry_truth = telemetry_truth if telemetry_truth is not None else telemetry
        self.intrinsics = intrinsics
        self.thermal_intrinsics = thermal_intrinsics
        self.thermal_is_radiometric = thermal_is_radiometric
        self.frame_index = frame_index
        self.video_reader = video_reader
        self.video_start_utc = video_start_utc
        self.k = max(1, int(k))
        self.load_images = load_images
        self.load_thermal = load_thermal
        self.max_frames = max_frames
        self.reports = reports or {}
        self.domain = domain
        self._iter: Iterator[FrameBundle] | None = None

    # -- the R12 offset ----------------------------------------------------------------------------------
    @property
    def t_offset_s(self) -> float:
        """Per-clip video-to-telemetry offset. `t_telemetry = t_frame + t_offset_s` (§5.4, R12)."""
        return self.telemetry.t_offset_s

    @t_offset_s.setter
    def t_offset_s(self, value: float) -> None:
        self.telemetry.t_offset_s = float(value)
        self.telemetry_truth.t_offset_s = float(value)

    def estimate_t_offset(self, *, max_lag_s: float = 10.0, max_frames: int = 600,
                          step: int = 2, apply: bool = True) -> tuple[float, float]:
        """Cross-correlate the visible take-off against barometric altitude and return `(offset_s, peak)`.

        Needs decodable video (`self.video_reader`). `apply=True` stores the result on the clip.
        """
        from sightline.ingest.align import video_log_height_series

        if self.video_reader is None:
            raise ValueError("this clip has no video reader; t_offset_s must be set by hand")
        frames: list[np.ndarray] = []
        times: list[float] = []
        for df in self.video_reader.frames(step=step):
            if df.image is None:
                continue
            frames.append(df.image)
            times.append(df.pts_s + self.video_start_utc)
            if len(frames) >= max_frames:
                break
        t_frames, log_h = video_log_height_series(frames, times)
        offset, peak = estimate_t_offset(t_frames, log_h, self.telemetry.t, self.telemetry.alt_msl_m,
                                         max_lag_s=max_lag_s)
        if apply:
            self.t_offset_s = offset
        return offset, peak

    # -- iteration ---------------------------------------------------------------------------------------
    def frames(self) -> Iterator[FrameBundle]:
        """Yield one `FrameBundle` per PROCESSED frame (every k-th; §5.4 "Decimation")."""
        n = 0
        for ref in decimate(self.frame_index, self.k):
            if self.max_frames and n >= self.max_frames:
                return
            bundle = self._bundle(ref)
            if bundle is None:
                continue
            self.frame_index.mark_processed(ref.frame_idx)
            n += 1
            yield bundle

    def _bundle(self, ref: FrameRef) -> FrameBundle | None:
        telemetry = self.telemetry.at(ref.t_utc, frame_idx=ref.frame_idx)
        rgb = None
        if self.load_images:
            rgb = self.frame_index.load(ref.frame_idx, video_reader=self.video_reader)
        thermal = None
        if self.load_thermal and ref.thermal_path:
            thermal = read_thermal(ref.thermal_path, radiometric=self.thermal_is_radiometric)
        return FrameBundle(
            frame_idx=ref.frame_idx,
            t_utc=telemetry.t_utc,
            telemetry=telemetry,
            intrinsics=self.intrinsics,
            rgb=rgb,
            thermal=thermal,
            thermal_intrinsics=self.thermal_intrinsics,
            thermal_is_radiometric=self.thermal_is_radiometric,
            clip_id=self.clip_id,
            source_path=ref.path or ref.video_path or self.source_path,
        )

    def __iter__(self) -> "Clip":
        self._iter = self.frames()
        return self

    def __next__(self) -> FrameBundle:
        if self._iter is None:
            self._iter = self.frames()
        return next(self._iter)

    def __len__(self) -> int:
        """Number of PROCESSED frames (after decimation)."""
        total = (len(self.frame_index) + self.k - 1) // self.k
        return min(total, self.max_frames) if self.max_frames else total

    # -- evidence ----------------------------------------------------------------------------------------
    def load_frame(self, frame_idx: int) -> np.ndarray | None:
        """Pixels of ANY decoded frame — the §5.4 guarantee that decimation loses no evidence."""
        return self.frame_index.load(frame_idx, video_reader=self.video_reader)

    def telemetry_at(self, t_utc: float) -> Telemetry:
        return self.telemetry.at(t_utc)

    def close(self) -> None:
        if self.video_reader is not None:
            self.video_reader.close()
            self.video_reader = None

    def __enter__(self) -> "Clip":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"Clip(source={self.source_type!r}, clip_id={self.clip_id!r}, frames={len(self.frame_index)}, "
                f"k={self.k}, t_offset_s={self.t_offset_s:.3f}, domain={self.domain!r})")


# --- the entry point ---------------------------------------------------------------------------------------
def open_clip(path: str | os.PathLike[str], *, decimate_k: int | None = None, target_fps: float | None = None,
              source_fps: float | None = None, load_images: bool = True, load_thermal: bool = True,
              noise: bool | NoiseModel | None = None, t_offset_s: float | None = None,
              clip_id: str = "", max_frames: int = 0, video: str | os.PathLike[str] | None = None,
              video_start_utc: float | None = None, decode_backend: str = "auto",
              tz_offset_h: float = 0.0, gps_order: str = "auto", takeoff_alt_msl_m: float | None = None,
              assume_nadir: bool = True, intrinsics: Intrinsics | None = None,
              origin: tuple[float, float, float] | None = None, hfov_deg: float | None = None,
              frame_size: tuple[int, int] | None = None, validate: bool = True) -> Clip:
    """Open any supported recording as a stream of `FrameBundle`. See `detect_source` for the source table.

    Common options:

    * `decimate_k` / `target_fps` — process every k-th frame (§5.4: k=3 -> 10 FPS, k=6 -> 5 FPS). Give
      `target_fps` with `source_fps` (or a source that knows its own rate) instead of computing k by hand.
    * `noise` — `True` or a `NoiseModel` applies the §5.7 sensor-error model to a **simulator** clip and keeps
      the clean series on `Clip.telemetry_truth`. Ignored (with the truth series unchanged) for real sources,
      which already carry real noise.
    * `t_offset_s` — the R12 per-clip alignment offset; `Clip.estimate_t_offset()` can measure it instead.
    * `video`, `video_start_utc` — required for telemetry-only sources (MAVLink, ULog, a bare `.srt`) when the
      video is not a sibling file with the same stem.
    * `origin`, `hfov_deg`, `frame_size` — required by `airsim_rec` captures, which record none of them.
    """
    p = Path(path)
    source_type = detect_source(p)
    builder = {
        "sim": _open_sim,
        "capture_run": _open_capture_run,
        "airsim_rec": _open_airsim_rec,
        "dji_srt": _open_dji_srt,
        "mavlink": _open_mavlink,
        "ulog": _open_ulog,
        "video": _open_video_only,
    }[source_type]
    clip: Clip = builder(
        p,
        clip_id=clip_id, load_images=load_images, load_thermal=load_thermal, max_frames=max_frames,
        video=video, video_start_utc=video_start_utc, decode_backend=decode_backend,
        tz_offset_h=tz_offset_h, gps_order=gps_order, takeoff_alt_msl_m=takeoff_alt_msl_m,
        assume_nadir=assume_nadir, intrinsics=intrinsics, origin=origin, hfov_deg=hfov_deg,
        frame_size=frame_size, validate=validate,
    )
    if t_offset_s is not None:
        clip.t_offset_s = float(t_offset_s)
    if noise:
        if clip.domain != "sim":
            clip.reports.setdefault("warnings", []).append(
                "noise= was requested on a non-simulator clip and was ignored: real telemetry already "
                "carries real error (SOLUTION_DOC 5.7)."
            )
        else:
            model = noise if isinstance(noise, NoiseModel) else NoiseModel()
            clip.telemetry_truth = clip.telemetry
            clip.telemetry = inject_noise(clip.telemetry, model)
            clip.telemetry.t_offset_s = clip.telemetry_truth.t_offset_s
            clip.reports["noise"] = {"model": model.label, "seed": model.seed,
                                     "gnss_sigma_m": model.gnss_sigma_m, "yaw_bias_sigma_deg":
                                     model.yaw_bias_sigma_deg}
    k = _resolve_k(decimate_k, target_fps, source_fps, clip)
    clip.k = k
    return clip


def _resolve_k(decimate_k: int | None, target_fps: float | None, source_fps: float | None, clip: Clip) -> int:
    if decimate_k is not None:
        return max(1, int(decimate_k))
    if target_fps is None:
        return 1
    src = source_fps
    if src is None and clip.video_reader is not None:
        src = clip.video_reader.info.fps
    if src is None or src <= 0:
        times = clip.frame_index.times()
        if times.size > 2:
            dt = float(np.median(np.diff(np.sort(times))))
            src = 1.0 / dt if dt > 0 else 0.0
    return decimation_for_fps(src or 0.0, target_fps)


# --- per-source builders -----------------------------------------------------------------------------------
def _open_sim(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
              validate: bool, **_: Any) -> Clip:
    reader = SimExportReader(p, validate=validate)
    rows = reader.rows
    series = reader.telemetry_series()
    index = reader.frame_index()
    first = rows[0] if rows else None
    video_reader = None
    manifest = reader.manifest
    if manifest.video:
        video_reader = open_video(reader.root / manifest.video, decode_images=load_images)
    return Clip(
        source_type="sim",
        clip_id=clip_id or reader.clip_id,
        telemetry=series,
        intrinsics=reader.intrinsics(first),
        thermal_intrinsics=reader.thermal_intrinsics(first),
        thermal_is_radiometric=reader.thermal_is_radiometric(first),
        frame_index=index,
        source_path=str(reader.root),
        video_reader=video_reader,
        video_start_utc=manifest.video_start_utc or 0.0,
        load_images=load_images,
        load_thermal=load_thermal,
        max_frames=max_frames,
        domain=manifest.domain,
        reports={"manifest": manifest.to_dict()},
    )


def _open_capture_run(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
                      **_: Any) -> Clip:
    """The F5 dataset run written by `tools/capture/run.py` (`telemetry.csv` + images/labels/masks)."""
    reader = CaptureRunReader(p, clip_id=clip_id)
    first = reader.rows[0]
    return Clip(
        source_type="capture_run",
        clip_id=reader.clip_id,
        telemetry=reader.telemetry_series(),
        intrinsics=reader.intrinsics(first),
        frame_index=reader.frame_index(),
        source_path=str(reader.root),
        load_images=load_images,
        load_thermal=load_thermal,
        max_frames=max_frames,
        domain=reader.domain,
        reports={"data_card": reader.card, "assumptions": reader.assumptions,
                 "gsd_cm_px": reader.gsd_cm_px(first)},
    )


def _open_airsim_rec(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
                     origin: tuple[float, float, float] | None, hfov_deg: float | None,
                     frame_size: tuple[int, int] | None, **_: Any) -> Clip:
    if origin is None or hfov_deg is None or frame_size is None:
        raise ValueError(
            "airsim_rec.txt records no GPS, no intrinsics and no camera pose. Pass origin=(lat, lon, "
            "alt_msl_m) from the AirSim settings OriginGeopoint, hfov_deg from CaptureSettings.FOV_Degrees "
            "and frame_size=(width_px, height_px). Prefer a sightline-sim-capture clip, which records them."
        )
    reader = AirSimRecReader(p, origin_lat=origin[0], origin_lon=origin[1], origin_alt_msl_m=origin[2],
                             hfov_deg=hfov_deg, width_px=frame_size[0], height_px=frame_size[1],
                             clip_id=clip_id)
    return Clip(
        source_type="airsim_rec",
        clip_id=reader.clip_id,
        telemetry=reader.telemetry_series(),
        intrinsics=reader.intrinsics,
        frame_index=reader.frame_index(),
        source_path=str(reader.path),
        load_images=load_images,
        load_thermal=load_thermal,
        max_frames=max_frames,
        domain="sim",
        reports={"assumptions": reader.assumptions},
    )


def _open_dji_srt(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
                  video: str | os.PathLike[str] | None, video_start_utc: float | None,
                  decode_backend: str, tz_offset_h: float, gps_order: str,
                  takeoff_alt_msl_m: float | None, assume_nadir: bool,
                  intrinsics: Intrinsics | None, **_: Any) -> Clip:
    srt_path = p if p.suffix.lower() == ".srt" else _sibling_srt(p)
    if p.is_dir():
        srt_path = next(c for c in sorted(p.iterdir()) if c.suffix.lower() == ".srt")
    if srt_path is None:
        raise CaptureFormatError(f"{p}: no .srt beside the video")
    entries = parse_srt(srt_path, tz_offset_h=tz_offset_h, gps_order=gps_order)
    if not entries:
        raise CaptureFormatError(f"{srt_path}: parsed zero entries")
    name = clip_id or srt_path.stem
    samples, report = srt_to_telemetry(entries, clip_id=name, takeoff_alt_msl_m=takeoff_alt_msl_m,
                                       assume_nadir=assume_nadir, tz_offset_h=tz_offset_h)
    if not samples:
        raise CaptureFormatError(f"{srt_path}: no entry carried both a position and a time")
    series = TelemetrySeries.from_telemetry(samples, clip_id=name)

    video_path = Path(video) if video else (p if p.suffix.lower() in _VIDEO_SUFFIXES else _sibling_video(srt_path))
    reader = None
    index = FrameIndex(name)
    if video_path is not None and video_path.is_file():
        reader = open_video(video_path, backend=decode_backend, decode_images=load_images)
        start_utc = video_start_utc if video_start_utc is not None else float(series.t[0])
        for i, pts in enumerate(reader.pts_seconds()):
            index.add(FrameRef(frame_idx=i, t_utc=float(pts) + start_utc, pts_s=float(pts),
                               video_path=str(video_path), video_index=i))
    else:
        start_utc = video_start_utc if video_start_utc is not None else 0.0
        # Index from the SAMPLES, not by zipping them against the entries: `srt_to_telemetry` drops any entry
        # with no position or no time, so a positional zip would pair sample i with entry i+1 from the first
        # gap onwards and stamp every later frame with the wrong index and PTS.
        pts_by_frame = {e.frame_idx: e.start_s for e in entries}
        for s in samples:
            index.add(FrameRef(frame_idx=s.frame_idx, t_utc=s.t_utc, pts_s=pts_by_frame.get(s.frame_idx, 0.0)))
    intr = intrinsics or _srt_intrinsics(entries, reader)
    return Clip(
        source_type="dji_srt",
        clip_id=name,
        telemetry=series,
        intrinsics=intr,
        frame_index=index,
        source_path=str(srt_path),
        video_reader=reader,
        video_start_utc=start_utc,
        load_images=load_images and reader is not None,
        load_thermal=load_thermal,
        max_frames=max_frames,
        domain="real",
        reports={"srt": report},
    )


def _srt_intrinsics(entries: Sequence[SrtEntry], reader: VideoReader | None) -> Intrinsics:
    """Intrinsics from the frame size plus the SRT's own `focal_len`, else a 4K-wide DJI default HFOV.

    §5.7 "Factors" warns that 4K 16:9 video may be a crop of the 4:3 sensor, so a focal length in millimetres
    cannot be turned into pixels without the sensor width. Without a `DroneModels.json` lookup (the geo lane
    owns that) the honest fallback is an HFOV, flagged by `Intrinsics.source`.
    """
    width = reader.info.width_px if reader is not None else 3840
    height = reader.info.height_px if reader is not None else 2160
    return Intrinsics.from_hfov(width or 3840, height or 2160, 73.7, source="fov")


def _open_mavlink(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
                  video: str | os.PathLike[str] | None, video_start_utc: float | None,
                  decode_backend: str, intrinsics: Intrinsics | None, **_: Any) -> Clip:
    from sightline.ingest.mavlink import read_mavlink

    name = clip_id or p.stem
    series, report = read_mavlink(p, clip_id=name)
    return _telemetry_only_clip("mavlink", p, name, series, report, video, video_start_utc, decode_backend,
                                intrinsics, load_images, load_thermal, max_frames)


def _open_ulog(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
               video: str | os.PathLike[str] | None, video_start_utc: float | None,
               decode_backend: str, intrinsics: Intrinsics | None, **_: Any) -> Clip:
    from sightline.ingest.ulog import read_ulog

    name = clip_id or p.stem
    series, report = read_ulog(p, clip_id=name)
    return _telemetry_only_clip("ulog", p, name, series, report, video, video_start_utc, decode_backend,
                                intrinsics, load_images, load_thermal, max_frames)


def _telemetry_only_clip(source_type: str, p: Path, name: str, series: TelemetrySeries, report: Any,
                         video: str | os.PathLike[str] | None, video_start_utc: float | None,
                         decode_backend: str, intrinsics: Intrinsics | None, load_images: bool,
                         load_thermal: bool, max_frames: int) -> Clip:
    video_path = Path(video) if video else _sibling_video(p)
    reader = None
    index = FrameIndex(name)
    start_utc = video_start_utc if video_start_utc is not None else float(series.t[0])
    if video_path is not None and video_path.is_file():
        reader = open_video(video_path, backend=decode_backend, decode_images=load_images)
        for i, pts in enumerate(reader.pts_seconds()):
            index.add(FrameRef(frame_idx=i, t_utc=float(pts) + start_utc, pts_s=float(pts),
                               video_path=str(video_path), video_index=i))
    else:
        for i, t in enumerate(series.t):     # metadata-only replay: one bundle per telemetry sample
            index.add(FrameRef(frame_idx=i, t_utc=float(t)))
    w = reader.info.width_px if reader is not None else 1920
    h = reader.info.height_px if reader is not None else 1080
    intr = intrinsics or getattr(report, "intrinsics", None) or Intrinsics.from_hfov(w, h, 73.7, source="fov")
    return Clip(
        source_type=source_type, clip_id=name, telemetry=series, intrinsics=intr, frame_index=index,
        source_path=str(p), video_reader=reader, video_start_utc=start_utc,
        load_images=load_images and reader is not None, load_thermal=load_thermal, max_frames=max_frames,
        domain="real", reports={source_type: report},
    )


def _open_video_only(p: Path, *, clip_id: str, load_images: bool, load_thermal: bool, max_frames: int,
                     decode_backend: str, intrinsics: Intrinsics | None, video_start_utc: float | None,
                     **_: Any) -> Clip:
    raise CaptureFormatError(
        f"{p} is a bare video with no telemetry beside it. Ingest emits (image, POSE, intrinsics, time) "
        f"tuples (SOLUTION_DOC 5.4), so a clip with no pose cannot be opened. Put the matching .srt / .tlog "
        f"/ .ulg next to it, or open the telemetry file and pass video={p.name!r}."
    )
