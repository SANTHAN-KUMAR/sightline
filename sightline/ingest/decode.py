"""Video and image decode (SOLUTION_DOC §5.4 "Decode").

* **PyAV is the reference path.** It is the only backend that reliably gives us presentation timestamps, and
  §5.4 needs PTS to hang the telemetry interpolation on. `av` 18.1.0 is installed and verified
  (`tests/test_stack.py`); `torchcodec` is NOT usable on this machine (it ships no FFmpeg on Windows and
  borrowing PyAV's MinGW DLLs corrupts the heap — see `tools/ffmpeg_shim.py`).
* **PyNvVideoCodec is the NVDEC path** for 4K throughput on the RTX 4060. The code is written against the
  v2.2.2 `CreateDemuxer` / `CreateDecoder` API and is **guarded**: it is only selected when the caller asks
  for it explicitly (`backend="nvdec"`) or sets `SIGHTLINE_ALLOW_NVDEC=1`, and ANY failure degrades to PyAV
  with a warning rather than raising. See the "not executed here" note on `NvdecReader`.

Every reader yields `DecodedFrame(index, pts_s, image)` where `image` is HxWx3 uint8 **BGR** (the convention
`Detection`/OpenCV already use) or None in metadata-only mode.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "DecodedFrame",
    "VideoInfo",
    "VideoReader",
    "PyAVReader",
    "NvdecReader",
    "open_video",
    "read_image_bgr",
    "read_thermal",
    "nvdec_available",
]


@dataclass(slots=True)
class DecodedFrame:
    """One decoded frame. `pts_s` is the container presentation timestamp in seconds (not a UTC time)."""

    index: int
    pts_s: float
    image: np.ndarray | None = None
    key_frame: bool = False


@dataclass(slots=True)
class VideoInfo:
    path: str
    width_px: int = 0
    height_px: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    n_frames: int = 0          # container-reported count; 0 when unknown (it often is)
    codec: str = ""
    pix_fmt: str = ""
    start_time_s: float = 0.0
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class VideoReader:
    """Minimal reader interface. Subclasses must be usable as context managers and be re-iterable."""

    info: VideoInfo

    def frames(self, start_index: int = 0, step: int = 1) -> Iterator[DecodedFrame]:
        raise NotImplementedError

    def frame_at(self, index: int) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --- PyAV --------------------------------------------------------------------------------------------------
class PyAVReader(VideoReader):
    """PyAV decode with real PTS. `decode_images=False` walks the container for timestamps only (cheap)."""

    def __init__(self, path: str | os.PathLike[str], *, decode_images: bool = True, threads: str = "AUTO") -> None:
        import av  # local import so `sightline.ingest` imports without av installed

        self._av = av
        self.path = str(path)
        self.decode_images = decode_images
        self._container = av.open(self.path)
        streams = self._container.streams.video
        if not streams:
            self._container.close()
            raise ValueError(f"{self.path}: no video stream")
        self._stream = streams[0]
        self._stream.thread_type = threads
        tb = self._stream.time_base
        self._time_base = float(tb) if tb is not None else 0.0
        st = self._stream
        avg = st.average_rate or st.guessed_rate
        self.info = VideoInfo(
            path=self.path,
            width_px=int(st.codec_context.width or 0),
            height_px=int(st.codec_context.height or 0),
            fps=float(avg) if avg else 0.0,
            duration_s=(float(st.duration * self._time_base) if st.duration and self._time_base
                        else float((self._container.duration or 0) / 1_000_000.0)),
            n_frames=int(st.frames or 0),
            codec=str(st.codec_context.name or ""),
            pix_fmt=str(st.codec_context.pix_fmt or ""),
            start_time_s=(float(st.start_time * self._time_base) if st.start_time and self._time_base else 0.0),
            backend="pyav",
        )

    def frames(self, start_index: int = 0, step: int = 1) -> Iterator[DecodedFrame]:
        if step < 1:
            raise ValueError("step must be >= 1")
        self._container.seek(0, stream=self._stream)  # re-iterable: rewind on every call
        idx = 0
        for frame in self._container.decode(self._stream):
            if idx >= start_index and (idx - start_index) % step == 0:
                pts = (float(frame.pts * self._time_base) if frame.pts is not None
                       else float(idx / max(self.info.fps, 1e-9)))
                img = frame.to_ndarray(format="bgr24") if self.decode_images else None
                yield DecodedFrame(index=idx, pts_s=pts, image=img, key_frame=bool(frame.key_frame))
            idx += 1

    def frame_at(self, index: int) -> np.ndarray | None:
        """Random access for the evidence thumbnail (§5.4: every decoded frame stays reachable).

        Seeks to the nearest preceding keyframe and decodes forward, so it is O(GOP) rather than O(clip).
        """
        if index < 0:
            return None
        fps = self.info.fps or 30.0
        target_pts = int(round((index / fps) / self._time_base)) if self._time_base else 0
        try:
            self._container.seek(max(target_pts, 0), stream=self._stream, backward=True, any_frame=False)
        except Exception:  # noqa: BLE001 - a non-seekable container just means we scan from wherever we are
            self._container.seek(0, stream=self._stream)
        best: np.ndarray | None = None
        want_t = index / fps
        for frame in self._container.decode(self._stream):
            t = float(frame.pts * self._time_base) if frame.pts is not None else 0.0
            if t <= want_t + 0.5 / fps:
                best = frame.to_ndarray(format="bgr24")
            if t >= want_t:
                break
        return best

    def pts_seconds(self) -> np.ndarray:
        """Every frame's PTS in seconds, without decoding pixels — the alignment input of §5.4."""
        self._container.seek(0, stream=self._stream)
        out = [float(p.pts * self._time_base) for p in self._container.demux(self._stream)
               if p.pts is not None and p.size > 0]
        return np.asarray(sorted(out), dtype=np.float64)

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:  # noqa: BLE001
            pass


# --- PyNvVideoCodec (NVDEC) --------------------------------------------------------------------------------
def nvdec_available() -> bool:
    """True when PyNvVideoCodec imports. Does NOT touch the GPU (importing does not create a CUDA context)."""
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class NvdecReader(VideoReader):
    """NVDEC decode via PyNvVideoCodec 2.2.2 (§5.4 "Windows/RTX: PyNvVideoCodec ... decoder kept alive").

    **NOT EXERCISED ON THIS MACHINE.** The build agent runs alongside the Unreal editor on an 8 GB RTX 4060
    and the project's hard rule is that no lane but the ML lane may allocate GPU memory (HANDBOOK §4). This
    class is therefore written from the documented API and never executed by `tests/test_ingest.py`; the
    first person with a free GPU should run it against a real MP4 before trusting it. `open_video()` will not
    select it unless asked, and falls back to PyAV on any error.

    PTS: the demuxer's packets carry the container timestamps, which is why PyAV is still used to build the
    PTS table when precise per-frame times are needed.
    """

    def __init__(self, path: str | os.PathLike[str], *, gpu_id: int = 0, use_device_memory: bool = False) -> None:
        import PyNvVideoCodec as nvc

        self._nvc = nvc
        self.path = str(path)
        self._demuxer = nvc.CreateDemuxer(filename=self.path)
        self._decoder = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=self._demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=use_device_memory,
        )
        self._use_device_memory = use_device_memory
        self.info = VideoInfo(path=self.path, backend="nvdec")
        with PyAVReader(self.path, decode_images=False) as probe:  # metadata from the container, cheaply
            self.info = dataclasses.replace(probe.info, backend="nvdec")

    def frames(self, start_index: int = 0, step: int = 1) -> Iterator[DecodedFrame]:
        if step < 1:
            raise ValueError("step must be >= 1")
        fps = self.info.fps or 30.0
        idx = 0
        for packet in self._demuxer:
            for frame in self._decoder.Decode(packet):
                if idx >= start_index and (idx - start_index) % step == 0:
                    yield DecodedFrame(index=idx, pts_s=idx / fps, image=self._to_bgr(frame))
                idx += 1

    def _to_bgr(self, frame: Any) -> np.ndarray | None:
        """NV12 (device or host) -> HxWx3 BGR uint8. Device frames are copied to the host via CAI."""
        import cv2

        arr = np.asarray(frame) if not self._use_device_memory else _cuda_to_host(frame)
        if arr is None:
            return None
        h = self.info.height_px or (arr.shape[0] * 2 // 3)
        if arr.ndim == 2 and arr.shape[0] == h * 3 // 2:
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)
        return arr

    def frame_at(self, index: int) -> np.ndarray | None:
        for f in self.frames(start_index=index, step=1):
            return f.image
        return None

    def close(self) -> None:
        self._decoder = None
        self._demuxer = None


def _cuda_to_host(frame: Any) -> np.ndarray | None:
    """Copy a device frame to the host through `__cuda_array_interface__` without importing torch."""
    try:
        import cupy  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return cupy.asnumpy(cupy.asarray(frame))


def open_video(path: str | os.PathLike[str], *, backend: str = "auto", decode_images: bool = True,
               gpu_id: int = 0) -> VideoReader:
    """Open a video. `backend` is "auto" | "pyav" | "nvdec".

    "auto" resolves to PyAV unless `SIGHTLINE_ALLOW_NVDEC=1` is set, because this project's build machine runs
    the Unreal editor on the same 8 GB GPU (HANDBOOK §4). NVDEC failures always fall back to PyAV.
    """
    backend = backend.lower()
    if backend == "auto":
        backend = "nvdec" if os.environ.get("SIGHTLINE_ALLOW_NVDEC") == "1" and nvdec_available() else "pyav"
    if backend == "nvdec":
        try:
            return NvdecReader(path, gpu_id=gpu_id)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"NVDEC decode unavailable ({exc.__class__.__name__}: {exc}); falling back to PyAV",
                          RuntimeWarning, stacklevel=2)
    if backend not in ("pyav", "nvdec"):
        raise ValueError(f"unknown backend {backend!r}; expected auto|pyav|nvdec")
    return PyAVReader(path, decode_images=decode_images)


# --- still images ------------------------------------------------------------------------------------------
def read_image_bgr(path: str | os.PathLike[str]) -> np.ndarray:
    """Read an RGB frame file as HxWx3 uint8 BGR. Raises rather than returning cv2's silent None."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not decode image {path}")
    return img


def read_thermal(path: str | os.PathLike[str], *, radiometric: bool) -> np.ndarray:
    """Read a thermal frame (§5.5b).

    `radiometric=True` returns HxW **uint16 centi-kelvin** exactly as stored (a 16-bit PNG); `False` returns
    HxW uint8 AGC grey. The 16-bit read needs `IMREAD_UNCHANGED`: `IMREAD_GRAYSCALE` silently truncates a
    16-bit PNG to 8 bits, which would turn 34.0 C into noise.
    """
    import cv2

    flag = cv2.IMREAD_UNCHANGED if radiometric else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"could not decode thermal image {path}")
    if img.ndim == 3:
        img = img[..., 0]
    if radiometric and img.dtype != np.uint16:
        raise ValueError(f"{path}: declared radiometric but stored as {img.dtype}; expected uint16 centi-kelvin")
    return img


def pts_table(path: str | os.PathLike[str]) -> np.ndarray:
    """Every frame's PTS in seconds for a container, decoding no pixels (§5.4 alignment input)."""
    with PyAVReader(path, decode_images=False) as r:
        return r.pts_seconds()


def frame_times_utc(pts_s: Sequence[float], video_start_utc: float, t_offset_s: float = 0.0) -> np.ndarray:
    """PTS -> UTC. `video_start_utc` is the UTC of PTS 0.0; `t_offset_s` is the per-clip R12 correction."""
    return np.asarray(pts_s, dtype=np.float64) + float(video_start_utc) + float(t_offset_s)
