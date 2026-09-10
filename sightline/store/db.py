"""F18 record log: SQLite in WAL mode, append-only, with the R10 guardrail enforced by the database itself.

SOLUTION_DOC §5.8 ("the record log itself is SQLite in WAL mode") and §5.10 (the outbox writes locally first).

Shape of the log
----------------
``records``          one row per record_id, always at its LATEST version (what the map and the API read).
``record_versions``  one row per (record_id, version) ever written — the full history, append-only.
``evidence``         one row per evidence item per version. Not a JSON blob: the map opens thumbnails from it
                     and the evaluation lane joins on ``clip_id``/``frame_idx``.
``audit``            one row per state transition (create / update / status / dismiss), with actor and reason.

The score components of §5.8 are **flattened into real columns** (``p_living``, ``w_class``, ``urgency``,
``count_bonus``, ``urgency_class``, ``elapsed_h``, ``thermal_boost``, ``motion_boost``, ``posture_promoted``)
so the priority score can be explained and queried, never shown as a bare number.

GUARDRAIL R10 (project hard rule 4)
-----------------------------------
There is no ``DELETE`` and no ``DROP`` in this module. On top of that the schema installs BEFORE DELETE triggers
on every table that ``RAISE(ABORT)``, so a delete is refused even if some other process opens the same file:

    sqlite3.IntegrityError: R10: records are never deleted

Two more triggers hold the rest of the rule: a record whose ``status`` becomes ``dismissed`` must carry a
non-empty ``dismissed_reason``, and ``version`` may never move backwards nor ``record_id`` be renumbered.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from sightline.schemas import (
    SCHEMA_VERSION,
    Evidence,
    Record,
    RecordStatus,
    ScoreComponents,
    ce90_m,
    feature_collection,
)

__all__ = [
    "RecordStore",
    "StaleVersionError",
    "GuardrailError",
    "RECORD_COLUMNS",
    "COMPONENT_COLUMNS",
    "STATUSES",
]

STATUSES: tuple[str, ...] = ("candidate", "confirmed", "stale", "dismissed")

#: Record fields that get their own TABLE rather than a column.
_NON_SCALAR = {"components", "evidence"}

#: Record fields stored as a JSON TEXT column (small, and SQLite's json_each() still makes them queryable).
_JSON_FIELDS = {"seen_in_passes", "source"}


def _scalar_columns() -> list[tuple[str, str]]:
    """Derived from the frozen `Record` dataclass so a schema drift is impossible to miss."""
    out: list[tuple[str, str]] = []
    for f in fields(Record):
        if f.name in _NON_SCALAR:
            continue
        ann = str(f.type)
        if f.name in _JSON_FIELDS:
            t = "TEXT"
        elif "bool" in ann:
            t = "INTEGER"
        elif "int" in ann and "Literal" not in ann:
            t = "INTEGER"
        elif "float" in ann:
            t = "REAL"
        else:
            t = "TEXT"
        out.append((f.name, t))
    return out


RECORD_COLUMNS: list[tuple[str, str]] = _scalar_columns()

#: §5.8: every term of the score is a column of its own, never one opaque blob.
COMPONENT_COLUMNS: list[tuple[str, str]] = [
    ("p_living", "REAL"),
    ("w_class", "REAL"),
    ("urgency", "REAL"),
    ("count_bonus", "REAL"),
    ("urgency_class", "TEXT"),
    ("elapsed_h", "REAL"),
    ("thermal_boost", "REAL"),
    ("motion_boost", "REAL"),
    ("posture_promoted", "INTEGER"),
]

#: Bookkeeping columns the store adds on top of the schema.
_EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("clip_id", "TEXT"),  # derived: source["clip_id"] or evidence[0].clip_id — the outbox key needs it
    ("ce90_m", "REAL"),  # derived: CE90_FACTOR * h_acc_m, so SQL/DuckDB can filter on it directly
    ("updated_utc", "REAL"),
]

_ALL_COLUMNS = RECORD_COLUMNS + COMPONENT_COLUMNS + _EXTRA_COLUMNS
_ALL_NAMES = [c for c, _ in _ALL_COLUMNS]


class StaleVersionError(RuntimeError):
    """A write carried a version at or below the stored one but different content."""


class GuardrailError(RuntimeError):
    """A write would have broken guardrail R10 (delete, or dismissal without a reason)."""


def _cols_sql(extra_pk: str = "") -> str:
    body = ",\n  ".join(f"{n} {t}" for n, t in _ALL_COLUMNS)
    return body + (",\n  " + extra_pk if extra_pk else "")


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS records (
  {_cols_sql("PRIMARY KEY (record_id)")}
);
CREATE TABLE IF NOT EXISTS record_versions (
  {_cols_sql("PRIMARY KEY (record_id, version)")}
);
CREATE TABLE IF NOT EXISTS evidence (
  record_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  thumb_uri TEXT,
  clip_id TEXT,
  frame_idx INTEGER,
  frame_time_utc REAL,
  bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
  det_conf REAL,
  camera TEXT,
  PRIMARY KEY (record_id, version, idx)
);
CREATE TABLE IF NOT EXISTS audit (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  t_utc REAL NOT NULL,
  action TEXT NOT NULL,
  actor TEXT NOT NULL DEFAULT '',
  reason TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_records_rank   ON records (priority_rank);
CREATE INDEX IF NOT EXISTS idx_records_status ON records (status);
CREATE INDEX IF NOT EXISTS idx_records_clip   ON records (clip_id);
CREATE INDEX IF NOT EXISTS idx_evidence_clip  ON evidence (clip_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_audit_record   ON audit (record_id);

-- ---- GUARDRAIL R10, enforced by the database (project hard rule 4) --------------------------------------
CREATE TRIGGER IF NOT EXISTS r10_records_no_delete BEFORE DELETE ON records
BEGIN SELECT RAISE(ABORT, 'R10: records are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS r10_versions_no_delete BEFORE DELETE ON record_versions
BEGIN SELECT RAISE(ABORT, 'R10: record history is never deleted'); END;
CREATE TRIGGER IF NOT EXISTS r10_evidence_no_delete BEFORE DELETE ON evidence
BEGIN SELECT RAISE(ABORT, 'R10: evidence is never deleted'); END;
CREATE TRIGGER IF NOT EXISTS r10_audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'R10: the audit log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS r10_dismiss_needs_reason_ins BEFORE INSERT ON records
WHEN NEW.status = 'dismissed' AND (NEW.dismissed_reason IS NULL OR trim(NEW.dismissed_reason) = '')
BEGIN SELECT RAISE(ABORT, 'R10: dismissal requires a reason'); END;
CREATE TRIGGER IF NOT EXISTS r10_dismiss_needs_reason_upd BEFORE UPDATE ON records
WHEN NEW.status = 'dismissed' AND (NEW.dismissed_reason IS NULL OR trim(NEW.dismissed_reason) = '')
BEGIN SELECT RAISE(ABORT, 'R10: dismissal requires a reason'); END;

CREATE TRIGGER IF NOT EXISTS r10_version_monotonic BEFORE UPDATE ON records
WHEN NEW.version < OLD.version
BEGIN SELECT RAISE(ABORT, 'R10: a record version never moves backwards'); END;
CREATE TRIGGER IF NOT EXISTS r10_id_immutable BEFORE UPDATE ON records
WHEN NEW.record_id <> OLD.record_id
BEGIN SELECT RAISE(ABORT, 'R10: record ids are never renumbered'); END;
"""


def _clip_id_of(rec: Record) -> str:
    cid = str(rec.source.get("clip_id", "") or "")
    if not cid and rec.evidence:
        cid = rec.evidence[0].clip_id
    return cid


def _row_from_record(rec: Record, now: float) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, _ in RECORD_COLUMNS:
        v = getattr(rec, name)
        row[name] = int(v) if isinstance(v, bool) else v
    row["seen_in_passes"] = json.dumps(list(rec.seen_in_passes))
    row["source"] = json.dumps(rec.source, default=str, sort_keys=True)
    c = rec.components
    for name, _ in COMPONENT_COLUMNS:
        v = getattr(c, name)
        row[name] = int(v) if isinstance(v, bool) else v
    row["clip_id"] = _clip_id_of(rec)
    row["ce90_m"] = ce90_m(rec.h_acc_m)
    row["updated_utc"] = now
    return row


def _record_from_row(row: sqlite3.Row, ev: Sequence[sqlite3.Row]) -> Record:
    kw: dict[str, Any] = {}
    for name, _ in RECORD_COLUMNS:
        kw[name] = row[name]
    kw["seen_in_passes"] = json.loads(row["seen_in_passes"] or "[]")
    kw["source"] = json.loads(row["source"] or "{}")
    kw["thermal_hot"] = bool(row["thermal_hot"])
    kw["occlusion"] = None if row["occlusion"] is None else int(row["occlusion"])
    kw["components"] = ScoreComponents(
        p_living=row["p_living"],
        w_class=row["w_class"],
        urgency=row["urgency"],
        count_bonus=row["count_bonus"],
        urgency_class=row["urgency_class"],
        elapsed_h=row["elapsed_h"],
        thermal_boost=row["thermal_boost"],
        motion_boost=row["motion_boost"],
        posture_promoted=bool(row["posture_promoted"]),
    )
    kw["evidence"] = [
        Evidence(
            thumb_uri=e["thumb_uri"] or "",
            clip_id=e["clip_id"] or "",
            frame_idx=int(e["frame_idx"] if e["frame_idx"] is not None else -1),
            frame_time_utc=float(e["frame_time_utc"] or 0.0),
            bbox_px=(e["bbox_x1"], e["bbox_y1"], e["bbox_x2"], e["bbox_y2"]),
            det_conf=float(e["det_conf"] or 0.0),
            camera=e["camera"] or "rgb",
        )
        for e in ev
    ]
    return Record(**kw)


class RecordStore:
    """The append-only record log. Thread-safe; WAL lets the API read while the pipeline writes.

    ``subscribe(cb)`` registers a callback ``cb(record, event)`` fired after every committed write; the API
    layer uses it to push the live feed. Callbacks run on the writing thread and must not block.
    """

    def __init__(self, path: str | os.PathLike[str], *, thumbs_dir: str | os.PathLike[str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir = Path(thumbs_dir) if thumbs_dir else self.path.parent / "thumbs"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._subs: list[Callable[[Record, str], None]] = []
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
            )
            self._conn.commit()

    # ---- lifecycle -----------------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "RecordStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def journal_mode(self) -> str:
        with self._lock:
            return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    def subscribe(self, cb: Callable[[Record, str], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def _notify(self, rec: Record, event: str) -> None:
        for cb in list(self._subs):
            try:
                cb(rec, event)
            except Exception:  # a broken subscriber must never break the log
                pass

    # ---- writes --------------------------------------------------------------------------------------
    def put(
        self,
        rec: Record,
        *,
        actor: str = "pipeline",
        reason: str = "",
        action: str | None = None,
    ) -> Record:
        """Idempotent upsert at ``rec.version``.

        * new record_id            -> insert;
        * same (record_id, version), identical content -> no-op, returns the stored copy (at-least-once
          delivery from the outbox lands here twice and must not duplicate anything);
        * version > stored         -> new version, history kept;
        * version <= stored with different content -> ``StaleVersionError``.
        """
        if rec.status == "dismissed" and not rec.dismissed_reason.strip():
            raise GuardrailError("R10: dismissal requires a reason")
        if rec.status not in STATUSES:
            raise ValueError(f"unknown status {rec.status!r}; expected one of {STATUSES}")
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT * FROM records WHERE record_id = ?", (rec.record_id,)).fetchone()
            if cur is not None and rec.version <= int(cur["version"]):
                stored = self.get(rec.record_id, version=rec.version)
                if stored is not None and stored == rec:
                    return stored  # exact replay: idempotent
                raise StaleVersionError(
                    f"{rec.record_id}: incoming version {rec.version} <= stored {cur['version']} "
                    "with different content"
                )
            row = _row_from_record(rec, now)
            cols = ",".join(_ALL_NAMES)
            marks = ",".join("?" for _ in _ALL_NAMES)
            vals = [row[n] for n in _ALL_NAMES]
            self._conn.execute(f"INSERT OR REPLACE INTO record_versions ({cols}) VALUES ({marks})", vals)
            if cur is None:
                self._conn.execute(f"INSERT INTO records ({cols}) VALUES ({marks})", vals)
            else:
                sets = ",".join(f"{n}=?" for n in _ALL_NAMES if n != "record_id")
                self._conn.execute(
                    f"UPDATE records SET {sets} WHERE record_id=?",
                    [row[n] for n in _ALL_NAMES if n != "record_id"] + [rec.record_id],
                )
            self._write_evidence(rec)
            act = action or ("create" if cur is None else "update")
            self._audit(rec.record_id, rec.version, act, actor, reason, detail=f"status={rec.status}")
            self._conn.commit()
        self._notify(rec, act)
        return rec

    def _write_evidence(self, rec: Record) -> None:
        for i, e in enumerate(rec.evidence):
            x1, y1, x2, y2 = e.bbox_px
            self._conn.execute(
                "INSERT OR REPLACE INTO evidence (record_id, version, idx, thumb_uri, clip_id, frame_idx,"
                " frame_time_utc, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_conf, camera)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.record_id, rec.version, i, e.thumb_uri, e.clip_id, e.frame_idx, e.frame_time_utc,
                 x1, y1, x2, y2, e.det_conf, e.camera),
            )

    def _audit(self, rid: str, version: int, action: str, actor: str, reason: str, detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO audit (record_id, version, t_utc, action, actor, reason, detail)"
            " VALUES (?,?,?,?,?,?,?)",
            (rid, version, time.time(), action, actor, reason, detail),
        )

    def update(self, record_id: str, *, actor: str = "operator", reason: str = "", **changes: Any) -> Record:
        """Apply field changes, bump ``version``, keep the old version in history."""
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        known = {f.name for f in fields(Record)}
        bad = set(changes) - known
        if bad:
            raise ValueError(f"unknown Record fields: {sorted(bad)}")
        for k, v in changes.items():
            setattr(cur, k, v)
        cur.version = cur.version + 1
        return self.put(cur, actor=actor, reason=reason, action="update")

    def set_status(
        self, record_id: str, status: RecordStatus, *, actor: str = "operator", reason: str = ""
    ) -> Record:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if status == "dismissed":
            return self.dismiss(record_id, reason=reason, by=actor)
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        cur.status = status
        cur.version += 1
        return self.put(cur, actor=actor, reason=reason, action="status")

    def dismiss(self, record_id: str, *, reason: str, by: str = "operator") -> Record:
        """R10: the ONLY way a record leaves the active list. The row stays; the reason is mandatory."""
        if not reason or not reason.strip():
            raise GuardrailError("R10: dismissal requires a non-empty reason")
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        cur.status = "dismissed"
        cur.dismissed_reason = reason.strip()
        cur.dismissed_by = by
        cur.dismissed_utc = time.time()
        cur.version += 1
        return self.put(cur, actor=by, reason=reason.strip(), action="dismiss")

    def add_note(self, record_id: str, note: str, *, actor: str = "operator") -> Record:
        """Free-text annotation (used for 'dispatched to <team>', §7 step 5 — see the lane report)."""
        cur = self.get(record_id)
        if cur is None:
            raise KeyError(record_id)
        cur.notes = (cur.notes + "\n" if cur.notes else "") + note
        cur.version += 1
        return self.put(cur, actor=actor, reason=note, action="note")

    def put_thumbnail(self, record_id: str, version: int, data: bytes, *, ext: str = ".jpg") -> str:
        """Write the evidence crop to disk FIRST (§5.10) and return the URI the live feed carries."""
        name = f"{record_id}_v{version}{ext}"
        (self.thumbs_dir / name).write_bytes(data)
        return f"/api/thumbs/{name}"

    def thumbnail_path(self, name: str) -> Path | None:
        """Resolve a thumbnail name safely (no traversal out of `thumbs_dir`)."""
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None
        p = (self.thumbs_dir / name).resolve()
        try:
            p.relative_to(self.thumbs_dir.resolve())
        except ValueError:
            return None
        return p if p.is_file() else None

    # ---- reads ---------------------------------------------------------------------------------------
    def get(self, record_id: str, *, version: int | None = None) -> Record | None:
        with self._lock:
            if version is None:
                row = self._conn.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
                ver = None if row is None else int(row["version"])
            else:
                row = self._conn.execute(
                    "SELECT * FROM record_versions WHERE record_id=? AND version=?", (record_id, version)
                ).fetchone()
                ver = version
            if row is None:
                return None
            ev = self._conn.execute(
                "SELECT * FROM evidence WHERE record_id=? AND version=? ORDER BY idx", (record_id, ver)
            ).fetchall()
        return _record_from_row(row, ev)

    def history(self, record_id: str) -> list[Record]:
        with self._lock:
            vs = [
                int(r["version"])
                for r in self._conn.execute(
                    "SELECT version FROM record_versions WHERE record_id=? ORDER BY version", (record_id,)
                ).fetchall()
            ]
        return [r for v in vs if (r := self.get(record_id, version=v)) is not None]

    def records(
        self, *, include_dismissed: bool = True, status: str | None = None, limit: int | None = None
    ) -> list[Record]:
        """Ranked list: priority_rank ascending (1 = most urgent), unranked (-1) last, then score desc."""
        sql = "SELECT record_id FROM records"
        where, args = [], []
        if status is not None:
            where.append("status = ?")
            args.append(status)
        elif not include_dismissed:
            where.append("status != 'dismissed'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY CASE WHEN priority_rank < 0 THEN 1 ELSE 0 END, priority_rank ASC, score DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            ids = [r["record_id"] for r in self._conn.execute(sql, args).fetchall()]
        return [r for i in ids if (r := self.get(i)) is not None]

    def feature_collection(self, **kw: Any) -> dict[str, Any]:
        return feature_collection(self.records(**kw))

    def audit_log(self, record_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit"
        args: list[Any] = []
        if record_id:
            sql += " WHERE record_id = ?"
            args.append(record_id)
        sql += " ORDER BY audit_id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            q = lambda s, a=(): self._conn.execute(s, a).fetchone()[0]  # noqa: E731
            by_status = {
                r["status"]: r["n"]
                for r in self._conn.execute("SELECT status, COUNT(*) n FROM records GROUP BY status")
            }
            return {
                "records": q("SELECT COUNT(*) FROM records"),
                "versions": q("SELECT COUNT(*) FROM record_versions"),
                "evidence": q("SELECT COUNT(*) FROM evidence"),
                "audit": q("SELECT COUNT(*) FROM audit"),
                "by_status": by_status,
                "journal_mode": str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]),
                "schema_version": SCHEMA_VERSION,
                "path": str(self.path),
            }

    def query(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        """Read-only escape hatch for the evaluation lane (score components / evidence live in real columns)."""
        s = sql.strip().lower()
        if not (s.startswith("select") or s.startswith("with")):
            raise GuardrailError("RecordStore.query() is read-only: SELECT / WITH only")
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, list(args)).fetchall()]
