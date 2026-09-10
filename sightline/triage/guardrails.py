"""F21 / requirement R10 - the guardrail, enforced mechanically instead of remembered.

    The system recommends and ranks; it never closes a segment, never deletes a record, and never lowers a
    record's priority to zero. Operators can mark a record *resolved* with a reason; the record stays in the
    log.  -- SOLUTION_DOC.md §1.4

    **Guardrail implementation (R10).** There is no code path that deletes a record or sets a segment to
    `cleared`. Operators can set ``status = dismissed`` with a reason string; dismissed records stay in the log
    and in the exported bundle under a separate layer.  -- SOLUTION_DOC.md §5.8

Two halves:

1. :func:`dismiss` is the *only* way a record leaves the commander's attention. It demands a non-empty reason,
   records who did it and when, bumps ``version`` so the outbox re-ships it, and changes nothing else - not the
   score, not the components, not the evidence. :func:`undismiss` exists because reinstating a record is the
   safe direction; there is deliberately no opposite.
2. :func:`scan_lane_sources` reads this lane's own source and fails on delete-shaped code. The check runs in
   ``tests/test_triage.py`` and ``tests/test_export.py``, so a future edit that adds a deletion breaks the
   build rather than the search.

The scanner runs two passes. The first blanks out every string literal and comment with :mod:`tokenize` before
matching, so the pattern table below - which necessarily contains the forbidden spellings - is invisible to it.
The second pass matches raw text and uses regexes written so that their own source text cannot match them.
"""

from __future__ import annotations

import io
import re
import time
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from sightline.schemas import Record, RecordStatus

#: The statuses a record may ever hold, taken from the frozen schema so this can never drift from it. Note what
#: is absent: there is no "done", no "resolved-and-gone", no closed segment.
ALLOWED_STATUSES: tuple[str, ...] = tuple(get_args(RecordStatus))

#: The directories this lane owns and therefore must keep clean.
LANE_SOURCE_DIRS: tuple[str, ...] = ("sightline/triage", "sightline/export")


class GuardrailError(RuntimeError):
    """Raised when an operation would violate R10."""


# --- 1. the operator API ------------------------------------------------------------------------------------
def dismiss(record: Record, reason: str, by: str, t_utc: float | None = None) -> Record:
    """Mark a record dismissed. The record is **kept**, everywhere, forever.

    Args:
        record: the record to dismiss; mutated in place and returned.
        reason: why. Must be non-empty after stripping - R10's whole point is that a dismissal is accountable.
        by: operator identity for the audit trail. Must be non-empty.
        t_utc: when, UTC POSIX seconds; defaults to now.

    Raises:
        GuardrailError: if ``reason`` or ``by`` is blank.
    """
    text = (reason or "").strip()
    who = (by or "").strip()
    if not text:
        raise GuardrailError("R10: a dismissal must carry a non-empty reason; the record is never removed")
    if not who:
        raise GuardrailError("R10: a dismissal must name the operator who made it")
    record.status = "dismissed"
    record.dismissed_reason = text
    record.dismissed_by = who
    record.dismissed_utc = time.time() if t_utc is None else float(t_utc)
    record.version = int(record.version) + 1
    return record


def undismiss(record: Record, by: str, status: str = "candidate", t_utc: float | None = None) -> Record:
    """Reinstate a dismissed record. Reinstating is always allowed; the reverse needs a reason.

    The original dismissal reason is preserved in ``notes`` so the audit trail survives the reinstatement.
    """
    who = (by or "").strip()
    if not who:
        raise GuardrailError("R10: a reinstatement must name the operator who made it")
    if status not in ALLOWED_STATUSES or status == "dismissed":
        raise GuardrailError(f"R10: reinstatement status must be one of {ALLOWED_STATUSES!r} and not dismissed")
    if record.status == "dismissed" and record.dismissed_reason:
        stamp = record.dismissed_utc or (time.time() if t_utc is None else float(t_utc))
        trail = f"[reinstated by {who}] previously dismissed by {record.dismissed_by or '?'} at {stamp:.0f}: {record.dismissed_reason}"
        record.notes = f"{record.notes}\n{trail}".strip() if record.notes else trail
    record.status = status  # type: ignore[assignment]
    record.version = int(record.version) + 1
    return record


def set_status(record: Record, status: str, reason: str = "", by: str = "") -> Record:
    """Set any allowed status. Dismissal is routed through :func:`dismiss` so it cannot skip its reason."""
    if status not in ALLOWED_STATUSES:
        raise GuardrailError(f"R10: {status!r} is not a permitted status; allowed: {ALLOWED_STATUSES!r}")
    if status == "dismissed":
        return dismiss(record, reason, by)
    record.status = status  # type: ignore[assignment]
    record.version = int(record.version) + 1
    return record


def is_dismissed(record: Record) -> bool:
    return record.status == "dismissed"


def partition_dismissed(records: Iterable[Record]) -> tuple[list[Record], list[Record]]:
    """``(active, dismissed)``. Both halves are returned; nothing is ever dropped on the floor."""
    items = list(records)
    return [r for r in items if not is_dismissed(r)], [r for r in items if is_dismissed(r)]


def retention_check(before: Sequence[Record], *after: Iterable[Record]) -> list[str]:
    """Record ids present in ``before`` but missing from the union of ``after``. Must always be empty.

    The source scanner catches *syntactic* deletion. This catches the semantic kind - a filter or a layer split
    that quietly loses a record - and is what every export path is asserted against.
    """
    seen: set[str] = set()
    for group in after:
        for rec in group:
            seen.add(str(rec.record_id))
    return [str(r.record_id) for r in before if str(r.record_id) not in seen]


# --- 2. the source scanner ----------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line_no: int
    line: str
    pattern: str
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line_no}: {self.kind} [{self.pattern}] :: {self.line.strip()}"


#: Pass 1 - matched against source with every string literal and comment blanked out, so these spellings are
#: only dangerous when they appear as real code.
_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<![\w.])del\s+[\w\[(]", "del statement"),
    (r"\.remove\s*\(", "container removal"),
    (r"\.pop\s*\(", "container pop"),
    (r"\.popitem\s*\(", "container popitem"),
    (r"\.clear\s*\(", "container reset"),
    (r"\.discard\s*\(", "set discard"),
    (r"\.truncate\s*\(", "truncate"),
    (r"\.drop\s*\(", "table/frame drop"),
    (r"\brmtree\s*\(", "recursive tree removal"),
    (r"\.unlink\s*\(", "filesystem unlink"),
    (r"\bos\s*\.\s*(remove|unlink|rmdir|removedirs)\b", "filesystem removal"),
    (r"\bdef\s+(delete|purge|drop|remove|expunge|wipe|erase|clear|close_out|finali[sz]e_segment)\w*", "delete-shaped function"),
)

#: Pass 2 - matched against the raw text, so it also sees string literals and comments. Each regex is written so
#: that its own source spelling cannot match it: the SQL ones put a metacharacter where the space would be, and
#: the status one puts a character class where the fifth letter would be.
_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"DELETE\s+FROM", "SQL row deletion"),
    (r"TRUNCATE\s+TABLE", "SQL table truncation"),
    (r"DROP\s+TABLE", "SQL table drop"),
    (r"[\"']clear[e]d[\"']", "a segment-is-done literal, forbidden by R10"),
    (r"[\"']searched_done[\"']", "a segment-is-done literal, forbidden by R10"),
)

_STRINGY_TOKENS = {tokenize.STRING, tokenize.COMMENT}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):  # 3.12+ splits f-strings into their own tokens
    _tt = getattr(tokenize, _name, None)
    if _tt is not None:
        _STRINGY_TOKENS.add(_tt)


def _blank_strings_and_comments(source: str) -> list[str]:
    """Return the source lines with string literals and comments replaced by spaces (line numbers preserved)."""
    lines = source.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return lines
    for tok in tokens:
        if tok.type not in _STRINGY_TOKENS:
            continue
        (start_row, start_col), (end_row, end_col) = tok.start, tok.end
        for row in range(start_row, min(end_row, len(lines)) + 1):
            if row < 1 or row > len(lines):
                continue
            line = lines[row - 1]
            a = start_col if row == start_row else 0
            b = end_col if row == end_row else len(line)
            a, b = max(0, min(a, len(line))), max(0, min(b, len(line)))
            if b > a:
                lines[row - 1] = line[:a] + " " * (b - a) + line[b:]
    return lines


def scan_source(path: Path | str) -> list[Violation]:
    """Scan one Python file for delete-shaped operations on records."""
    p = Path(path)
    source = p.read_text(encoding="utf-8")
    raw_lines = source.splitlines()
    code_lines = _blank_strings_and_comments(source)
    found: list[Violation] = []
    for pattern, kind in _CODE_PATTERNS:
        rx = re.compile(pattern)
        for i, line in enumerate(code_lines, start=1):
            if rx.search(line):
                found.append(Violation(str(p), i, raw_lines[i - 1], pattern, kind))
    for pattern, kind in _TEXT_PATTERNS:
        rx = re.compile(pattern)
        for i, line in enumerate(raw_lines, start=1):
            if rx.search(line):
                found.append(Violation(str(p), i, line, pattern, kind))
    return found


def scan_lane_sources(repo_root: Path | str | None = None, dirs: Sequence[str] = LANE_SOURCE_DIRS) -> list[Violation]:
    """Scan every ``.py`` file under this lane's directories. An empty result is the R10 pass condition."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    found: list[Violation] = []
    for rel in dirs:
        directory = root / rel
        if not directory.is_dir():
            raise GuardrailError(f"R10 scanner: {directory} does not exist; the lane layout changed")
        for py in sorted(directory.rglob("*.py")):
            found.extend(scan_source(py))
    return found


def assert_no_record_deletion(repo_root: Path | str | None = None, dirs: Sequence[str] = LANE_SOURCE_DIRS) -> None:
    """Raise :class:`GuardrailError` listing every violation found by :func:`scan_lane_sources`."""
    found = scan_lane_sources(repo_root, dirs)
    if found:
        detail = "\n".join(str(v) for v in found)
        raise GuardrailError(f"R10 violated - {len(found)} delete-shaped operation(s) in the triage/export lane:\n{detail}")
