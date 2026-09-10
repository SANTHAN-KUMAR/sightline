"""DJI `.SRT` per-frame subtitle telemetry (SOLUTION_DOC §5.4 row 2).

Written against the format documentation of **dji-drone-metadata-embedder** (MIT, v2.15.0), read 10 Sep 2026
at https://callmarcus.github.io/dji-drone-metadata-embedder/SRT_FORMATS/ , which enumerates six real variants.
All six are handled here; the parser tags each entry with the variant it matched so a bug report can name it.

* `format1`  — Mini 3 Pro, Mini 4 Pro, Mavic Air 2, FPV:
  bare `[key: value]` groups with no `<font>` wrapper.
* `format2`  — Mavic Pro, Phantom 4, Avata 2:
  `GPS(lat,lon,alt) BAROMETER(91.2) HOME(lat,lon) D=5.2m H=1.5m`.
* `format2b` — Matrice 300 RTK and adjacent enterprise:
  as format2 but `GPS(...,0.0M) BAROMETER:0.3M` (unit suffixes, colon notation).
* `format2c` — Phantom 4 RTK/Pro, Matrice 350/30 lineage:
  `F/5.6, SS 400, ISO 100, EV 0, GPS (...), HOME (...), D, H, H.S, V.S, F.PRY (p,r,y), G.PRY (p,r,y)`.
* `format3`  — Mavic 3, Air 2S, Air 3:
  `<font>SrtCnt : N, DiffTime : Nms` + a date line + `[key : value]` groups, **legacy integer units**.
* `format3b` — Mini 5 Pro, Avata 360, Neo 2, Air 3S, Mavic 4 Pro:
  as format3 but `FrameCnt`, **decimal** fnum/focal_len, and optional `gb_yaw`/`gb_pitch`/`gb_roll`.

**Unit quirks the documentation calls out, and how they are resolved here.**

* `fnum : 170` means f/1.7 and `fnum: 1.8` means f/1.8; `focal_len : 240` means 24 mm and `focal_len: 24.00`
  means 24 mm. The generation cannot be read from the number alone (a 240 mm enterprise zoom is legal), so the
  rule used is **"a decimal point means the value is literal; an integer means the legacy scaled encoding"**,
  with magnitude as a fallback when the token has no dot. Both the raw token and the rule that fired are kept
  on the entry (`raw`, `unit_rule`) so this is auditable rather than magic.
* Consumer Mavic-class SRT carries **no gimbal attitude at all**. `gimbal_pitch_deg` is then `None`, and
  `srt_to_telemetry(..., assume_nadir=True)` substitutes the −90° nadir assumption **and marks it**
  (`Telemetry.weather` is untouched; the assumption is recorded in the returned `SrtParseReport`).
* The date line has no timezone. DJI writes aircraft-local time. `tz_offset_h` converts it; the default of 0
  treats the stamp as UTC and is almost certainly wrong for real footage — the replay harness must set it.
* `GPS(a,b,c)` argument order is genuinely inconsistent between the documented examples (format2's example is
  lat-first, format2c's is lon-first). `gps_order="auto"` uses the only reliable discriminator — |latitude|
  cannot exceed 90 — and falls back to the per-variant default when both values are under 90.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sightline.common.geodesy import euler_to_quat
from sightline.ingest.spec import gimbal_quat_from_euler
from sightline.schemas import Telemetry

__all__ = ["SrtEntry", "SrtParseReport", "parse_srt", "parse_srt_text", "srt_to_telemetry", "detect_srt_format"]

# --- regexes -----------------------------------------------------------------------------------------------
_TIMECODE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_HTML_TAG = re.compile(r"<[^>]*>")
_BRACKET_GROUP = re.compile(r"\[([^\]]*)\]")
#: `key : value` inside a bracket group. A group may hold several pairs: `[rel_alt: 5.0 abs_alt: 130.0]`,
#: `[dzoom_ratio: 10000, delta: 0]`, `[ct: 5562, tint: 0]`, `[gb_yaw: 1.0 gb_pitch: -90.0 gb_roll: 0.0]`.
_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*([^\s,\]]+)")
_FRAME_CNT = re.compile(r"(?:FrameCnt|SrtCnt)\s*:\s*(\d+)", re.IGNORECASE)
_DIFF_TIME = re.compile(r"DiffTime\s*:\s*(-?\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)
_DATE_LINE = re.compile(
    r"(\d{4})[-./](\d{2})[-./](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[,.](\d{1,3}))?(?:,(\d{1,3}))?"
)
_NUM = r"[-+]?\d+(?:\.\d+)?"
_GPS_FN = re.compile(rf"GPS\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*(?:,\s*({_NUM})\s*[A-Za-z]*)?\)", re.IGNORECASE)
_HOME_FN = re.compile(rf"HOME\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*(?:,\s*({_NUM})\s*[A-Za-z]*)?\)", re.IGNORECASE)
_BARO = re.compile(rf"BAROMETER\s*[:(]\s*({_NUM})\s*([A-Za-z]*)\)?", re.IGNORECASE)
_DIST_HOME = re.compile(rf"\bD\s*[=: ]\s*({_NUM})\s*m", re.IGNORECASE)
_HEIGHT_HOME = re.compile(rf"(?<![.\w])H\s*[=: ]\s*({_NUM})\s*m", re.IGNORECASE)
_H_SPEED = re.compile(rf"H\.S\s*[=: ]\s*({_NUM})\s*m/s", re.IGNORECASE)
_V_SPEED = re.compile(rf"V\.S\s*[=: ]\s*({_NUM})\s*m/s", re.IGNORECASE)
_F_PRY = re.compile(rf"F\.PRY\s*\(\s*({_NUM})\s*[^,]*,\s*({_NUM})\s*[^,]*,\s*({_NUM})", re.IGNORECASE)
_G_PRY = re.compile(rf"G\.PRY\s*\(\s*({_NUM})\s*[^,]*,\s*({_NUM})\s*[^,]*,\s*({_NUM})", re.IGNORECASE)
_F_STOP = re.compile(rf"\bF/\s*({_NUM})")
_SS = re.compile(rf"\bSS\s+({_NUM})")
_ISO_BARE = re.compile(rf"\bISO\s+({_NUM})")
_EV_BARE = re.compile(rf"\bEV\s+({_NUM})")


# --- one entry ---------------------------------------------------------------------------------------------
@dataclass(slots=True)
class SrtEntry:
    """One SRT block = one video frame. Fields that the source did not carry stay `None` (never 0.0)."""

    index: int                        # 1-based SRT block number
    frame_idx: int                    # 0-based frame index: FrameCnt/SrtCnt - 1 when present, else index - 1
    start_s: float                    # subtitle start, seconds from the beginning of the clip
    end_s: float
    srt_format: str = "unknown"
    t_utc: float | None = None        # from the date line, shifted by tz_offset_h
    # geodesy
    lat: float | None = None
    lon: float | None = None
    rel_alt_m: float | None = None    # height above the take-off point
    abs_alt_m: float | None = None    # DJI's "absolute" altitude; datum is NOT consistent across models (§5.7)
    baro_m: float | None = None
    home_lat: float | None = None
    home_lon: float | None = None
    dist_home_m: float | None = None
    height_home_m: float | None = None
    h_speed_ms: float | None = None
    v_speed_ms: float | None = None
    # attitude, degrees. DJI gimbal angles are EARTH-referenced (§5.7 step 4).
    gimbal_pitch_deg: float | None = None
    gimbal_roll_deg: float | None = None
    gimbal_yaw_deg: float | None = None
    flight_pitch_deg: float | None = None
    flight_roll_deg: float | None = None
    flight_yaw_deg: float | None = None
    # camera
    iso: float | None = None
    shutter_s: float | None = None
    fnum: float | None = None
    focal_len_mm: float | None = None
    ev: float | None = None
    ct_k: float | None = None
    color_md: str = ""
    dzoom_ratio: float | None = None
    diff_time_ms: float | None = None
    raw: dict[str, str] = field(default_factory=dict)
    unit_rule: dict[str, str] = field(default_factory=dict)

    @property
    def has_gimbal(self) -> bool:
        return self.gimbal_pitch_deg is not None


@dataclass(slots=True)
class SrtParseReport:
    """What the parser had to assume. Anything in here belongs in the clip's provenance, not in a silent log."""

    n_entries: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    n_with_gimbal: int = 0
    n_with_time: int = 0
    assumed_nadir: bool = False
    assumed_gps_order: str = ""
    tz_offset_h: float = 0.0
    alt_basis: str = ""
    warnings: list[str] = field(default_factory=list)


# --- parsing -----------------------------------------------------------------------------------------------
def parse_srt(path: str | os.PathLike[str], **kw: Any) -> list[SrtEntry]:
    """Parse a `.SRT` file. See `parse_srt_text` for the keyword arguments."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return parse_srt_text(fh.read(), **kw)


def parse_srt_text(text: str, *, tz_offset_h: float = 0.0, gps_order: str = "auto") -> list[SrtEntry]:
    """Parse SRT text into one `SrtEntry` per block.

    `tz_offset_h` is the aircraft-local UTC offset of the date line (e.g. 5.5 for IST); `t_utc` is
    `naive_stamp_as_utc - tz_offset_h * 3600`. `gps_order` is "auto" | "latlon" | "lonlat" (see module docs).
    """
    entries: list[SrtEntry] = []
    for block in _split_blocks(text):
        entry = _parse_block(block, tz_offset_h=tz_offset_h, gps_order=gps_order)
        if entry is not None:
            entries.append(entry)
    return entries


def _split_blocks(text: str) -> Iterable[tuple[int, list[str]]]:
    """Yield `(block_no, lines)`. Blocks are separated by blank lines; CRLF and a missing final blank are fine."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    current: list[str] = []
    block_no = 0
    for line in lines:
        if line.strip() == "":
            if current:
                block_no += 1
                yield block_no, current
                current = []
        else:
            current.append(line)
    if current:
        yield block_no + 1, current


def detect_srt_format(payload: str, had_font_tag: bool) -> str:
    """Classify one block's payload into the six documented variants."""
    if _G_PRY.search(payload) or _F_PRY.search(payload) or _SS.search(payload) or _F_STOP.search(payload):
        return "format2c"
    if _GPS_FN.search(payload):
        return "format2b" if re.search(r"BAROMETER\s*:", payload, re.IGNORECASE) else "format2"
    if had_font_tag or _FRAME_CNT.search(payload):
        # format3 uses the legacy integer encodings; format3b writes decimals for fnum/focal_len.
        if re.search(r"\b(?:fnum|focal_len)\s*:\s*\d+\.\d", payload):
            return "format3b"
        if re.search(r"\bFrameCnt\b", payload):
            return "format3b"
        return "format3"
    if _BRACKET_GROUP.search(payload):
        return "format1"
    return "unknown"


def _parse_block(block: tuple[int, list[str]], *, tz_offset_h: float, gps_order: str) -> SrtEntry | None:
    block_no, lines = block
    index = block_no
    body_start = 0
    if lines and lines[0].strip().isdigit():
        index = int(lines[0].strip())
        body_start = 1
    start_s = end_s = 0.0
    tc = _TIMECODE.search(lines[body_start]) if len(lines) > body_start else None
    if tc:
        start_s = _tc_seconds(tc.group(1), tc.group(2), tc.group(3), tc.group(4))
        end_s = _tc_seconds(tc.group(5), tc.group(6), tc.group(7), tc.group(8))
        body_start += 1
    payload_raw = "\n".join(lines[body_start:])
    if not payload_raw.strip():
        return None
    had_font = "<font" in payload_raw.lower()
    payload = _HTML_TAG.sub(" ", payload_raw)

    srt_format = detect_srt_format(payload, had_font)
    entry = SrtEntry(index=index, frame_idx=index - 1, start_s=start_s, end_s=end_s, srt_format=srt_format)

    m = _FRAME_CNT.search(payload)
    if m:
        entry.frame_idx = int(m.group(1)) - 1          # FrameCnt/SrtCnt are 1-based
    m = _DIFF_TIME.search(payload)
    if m:
        entry.diff_time_ms = float(m.group(1))

    # --- date line -> t_utc
    m = _DATE_LINE.search(payload)
    if m:
        y, mo, d, hh, mm, ss = (int(m.group(i)) for i in range(1, 7))
        frac = 0.0
        if m.group(7):
            frac += int(m.group(7)) / 10 ** len(m.group(7))
        if m.group(8):  # the "…,123,456" millis,micros form
            frac += int(m.group(8)) / 10 ** (len(m.group(8)) + 3)
        naive = datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)
        entry.t_utc = naive.timestamp() + frac - tz_offset_h * 3600.0

    # --- bracketed key/value pairs (formats 1, 3, 3b)
    kv: dict[str, str] = {}
    for group in _BRACKET_GROUP.findall(payload):
        for key, value in _KV.findall(group):
            kv[key.lower()] = value
    entry.raw.update(kv)

    _set_float(entry, "lat", kv.get("latitude"))
    _set_float(entry, "lon", kv.get("longitude"))
    _set_float(entry, "rel_alt_m", kv.get("rel_alt"))
    _set_float(entry, "abs_alt_m", kv.get("abs_alt"))
    _set_float(entry, "iso", kv.get("iso"))
    _set_float(entry, "ev", kv.get("ev"))
    _set_float(entry, "ct_k", kv.get("ct"))
    _set_float(entry, "dzoom_ratio", kv.get("dzoom_ratio"))
    _set_float(entry, "gimbal_yaw_deg", kv.get("gb_yaw"))
    _set_float(entry, "gimbal_pitch_deg", kv.get("gb_pitch"))
    _set_float(entry, "gimbal_roll_deg", kv.get("gb_roll"))
    entry.color_md = kv.get("color_md", "")
    if "shutter" in kv:
        entry.shutter_s = _shutter_seconds(kv["shutter"])
    if "fnum" in kv:
        entry.fnum, rule = _scaled_value(kv["fnum"], scale=100.0, literal_max=25.0)
        entry.unit_rule["fnum"] = rule
    if "focal_len" in kv:
        entry.focal_len_mm, rule = _scaled_value(kv["focal_len"], scale=10.0, literal_max=100.0)
        entry.unit_rule["focal_len"] = rule

    # --- function-call style (formats 2, 2b, 2c)
    m = _GPS_FN.search(payload)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lat, lon = _order_gps(a, b, gps_order, srt_format)
        entry.lat, entry.lon = lat, lon
        if m.group(3) is not None:
            entry.abs_alt_m = float(m.group(3))
        entry.raw["gps"] = m.group(0)
    m = _HOME_FN.search(payload)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        entry.home_lat, entry.home_lon = _order_gps(a, b, gps_order, srt_format)
    m = _DIST_HOME.search(payload)
    if m:
        entry.dist_home_m = float(m.group(1))
    m = _HEIGHT_HOME.search(payload)
    if m:  # "H 85.80m" is height above the home point: the best AGL a format2/2c SRT carries
        entry.height_home_m = float(m.group(1))
        if entry.rel_alt_m is None:
            entry.rel_alt_m = entry.height_home_m
    m = _BARO.search(payload)
    if m:
        entry.baro_m = float(m.group(1))
        # `BAROMETER(91.2)` (format2) is a PRESSURE reading, `BAROMETER:0.3M` (format2b) is metres. Only the
        # unit-suffixed form may stand in for rel_alt; guessing on the bare number would inject a 90 m error.
        if entry.rel_alt_m is None and m.group(2).lower().startswith("m"):
            entry.rel_alt_m = entry.baro_m
    m = _H_SPEED.search(payload)
    if m:
        entry.h_speed_ms = float(m.group(1))
    m = _V_SPEED.search(payload)
    if m:
        entry.v_speed_ms = float(m.group(1))
    m = _F_PRY.search(payload)  # documented as pitch, roll, yaw in that order
    if m:
        entry.flight_pitch_deg = float(m.group(1))
        entry.flight_roll_deg = float(m.group(2))
        entry.flight_yaw_deg = float(m.group(3))
    m = _G_PRY.search(payload)
    if m:
        entry.gimbal_pitch_deg = float(m.group(1))
        entry.gimbal_roll_deg = float(m.group(2))
        entry.gimbal_yaw_deg = float(m.group(3))
    if entry.fnum is None:
        m = _F_STOP.search(payload)
        if m:
            entry.fnum, entry.unit_rule["fnum"] = float(m.group(1)), "f_stop_token"
    if entry.shutter_s is None:
        m = _SS.search(payload)
        if m:
            denom = float(m.group(1))
            entry.shutter_s = (1.0 / denom) if denom > 0 else None
    if entry.iso is None:
        m = _ISO_BARE.search(payload)
        if m:
            entry.iso = float(m.group(1))
    if entry.ev is None:
        m = _EV_BARE.search(payload)
        if m:
            entry.ev = float(m.group(1))
    return entry


def _tc_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 10 ** len(ms)


def _set_float(entry: SrtEntry, attr: str, text: str | None) -> None:
    if text is None:
        return
    try:
        setattr(entry, attr, float(text))
    except ValueError:
        pass


def _shutter_seconds(token: str) -> float | None:
    """`1/1000.0` -> 0.001; `1/30.0` -> 0.0333; a bare `400` is read as 1/400 s (the `SS` token form)."""
    token = token.strip()
    if "/" in token:
        num, _, den = token.partition("/")
        try:
            n, d = float(num), float(den)
        except ValueError:
            return None
        return n / d if d else None
    try:
        v = float(token)
    except ValueError:
        return None
    return (1.0 / v) if v > 1.0 else v


def _scaled_value(token: str, *, scale: float, literal_max: float) -> tuple[float | None, str]:
    """Resolve DJI's legacy integer encodings. Returns `(value, rule_that_fired)`.

    `fnum 170` -> 1.7 (scale 100) and `focal_len 240` -> 24.0 (scale 10), but `fnum 1.8` and `focal_len 24.00`
    are already literal. A decimal point is the reliable discriminator; magnitude is the fallback.
    """
    token = token.strip()
    try:
        v = float(token)
    except ValueError:
        return None, "unparsable"
    if "." in token:
        return v, "literal_decimal"
    if v <= literal_max:
        return v, "literal_small_integer"
    return v / scale, f"legacy_x{int(scale)}"


def _order_gps(a: float, b: float, gps_order: str, srt_format: str) -> tuple[float, float]:
    """Resolve the `GPS(a,b)` argument order into `(lat, lon)`."""
    if gps_order == "latlon":
        return a, b
    if gps_order == "lonlat":
        return b, a
    if abs(a) > 90.0 >= abs(b):
        return b, a          # a cannot be a latitude
    if abs(b) > 90.0 >= abs(a):
        return a, b
    # Both plausible. Fall back to the per-variant default from the documented examples.
    return (b, a) if srt_format == "format2c" else (a, b)


# --- to the pipeline contract ------------------------------------------------------------------------------
def srt_to_telemetry(entries: Sequence[SrtEntry], *, clip_id: str = "", clip_start_utc: float | None = None,
                     takeoff_alt_msl_m: float | None = None, assume_nadir: bool = True,
                     h_acc_m: float = 2.5, v_acc_m: float = 1.0,
                     mode: str = "AUTO") -> tuple[list[Telemetry], SrtParseReport]:
    """Turn parsed SRT entries into `Telemetry` samples plus a report of every assumption that was made.

    Altitude (§5.7 "Factors"): DJI `abs_alt` is not reliably ellipsoidal or orthometric across models, so when
    `takeoff_alt_msl_m` is supplied the altitude used is `takeoff_alt_msl_m + rel_alt`, and `abs_alt` is only a
    fallback. `agl_m` is `rel_alt` (height above the take-off point) — correct over flat ground and the only
    thing a consumer SRT can support; a DEM correction is the geo lane's job (F13).

    Attitude: the gimbal quaternion is built with `spec.gimbal_quat_from_euler`, the single definition shared
    with the simulator export, and is earth-referenced. Consumer Mavic-class SRT has no gimbal at all: with
    `assume_nadir=True` the −90° nadir assumption is applied and recorded in the report.
    """
    report = SrtParseReport(n_entries=len(entries), tz_offset_h=0.0)
    out: list[Telemetry] = []
    for e in entries:
        report.formats[e.srt_format] = report.formats.get(e.srt_format, 0) + 1
        if e.has_gimbal:
            report.n_with_gimbal += 1
        if e.t_utc is not None:
            report.n_with_time += 1
    if report.n_with_gimbal == 0:
        report.assumed_nadir = bool(assume_nadir)
        if assume_nadir:
            report.warnings.append(
                "no gimbal attitude in this SRT (consumer Mavic-class): assuming a -90 deg nadir gimbal. "
                "Geolocation error grows as sec^2(theta) with any real off-nadir angle."
            )
        else:
            report.warnings.append("no gimbal attitude in this SRT and assume_nadir=False: gimbal left level.")
    report.alt_basis = "takeoff_msl + rel_alt" if takeoff_alt_msl_m is not None else "abs_alt (datum unverified)"
    if takeoff_alt_msl_m is None:
        report.warnings.append(
            "takeoff_alt_msl_m not supplied: falling back to the SRT abs_alt field, whose datum is not "
            "consistent across DJI models (SOLUTION_DOC 5.7 'Factors')."
        )

    for e in entries:
        if e.lat is None or e.lon is None:
            continue
        t = e.t_utc
        if t is None:
            if clip_start_utc is None:
                continue
            t = clip_start_utc + e.start_s
        rel = e.rel_alt_m if e.rel_alt_m is not None else 0.0
        if takeoff_alt_msl_m is not None:
            alt_msl = takeoff_alt_msl_m + rel
        elif e.abs_alt_m is not None:
            alt_msl = e.abs_alt_m
        else:
            alt_msl = rel
        pitch = e.gimbal_pitch_deg
        if pitch is None:
            pitch = -90.0 if assume_nadir else 0.0
        roll = e.gimbal_roll_deg if e.gimbal_roll_deg is not None else 0.0
        yaw = e.gimbal_yaw_deg
        if yaw is None:
            yaw = e.flight_yaw_deg if e.flight_yaw_deg is not None else 0.0
        q_body = euler_to_quat(
            e.flight_roll_deg if e.flight_roll_deg is not None else 0.0,
            e.flight_pitch_deg if e.flight_pitch_deg is not None else 0.0,
            e.flight_yaw_deg if e.flight_yaw_deg is not None else yaw,
        )
        vel = (0.0, 0.0, 0.0)
        if e.h_speed_ms is not None or e.v_speed_ms is not None:
            hs = e.h_speed_ms or 0.0
            heading = math.radians(e.flight_yaw_deg if e.flight_yaw_deg is not None else yaw)
            vel = (hs * math.cos(heading), hs * math.sin(heading), -(e.v_speed_ms or 0.0))
        out.append(Telemetry(
            t_utc=float(t),
            lat=float(e.lat),
            lon=float(e.lon),
            alt_msl_m=float(alt_msl),
            agl_m=float(rel),
            q_body=q_body,
            q_gimbal=gimbal_quat_from_euler(roll, pitch, yaw),
            gimbal_is_earth_referenced=True,
            ned_m=None,
            vel_ned_ms=vel,
            h_acc_m=h_acc_m,
            v_acc_m=v_acc_m,
            mode=mode,  # type: ignore[arg-type]
            clip_id=clip_id,
            frame_idx=e.frame_idx,
            noise_injected=False,
        ))
    return out, report
