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
2. :func:`scan_lane_sources` reads the source of **every lane in the system** and fails on delete-shaped code.
   The check runs in ``tests/test_triage.py``, ``tests/test_export.py``, ``tests/test_live_mission.py`` and
   ``tests/test_guardrails_scan.py``, so a future edit that adds a deletion breaks the build rather than the
   search. ``python -m sightline.triage.guardrails`` runs it as a standalone gate and exits non-zero.

Scope, and why it changed (2026-09-11)
--------------------------------------
``LANE_SOURCE_DIRS`` used to be ``("sightline/triage", "sightline/export")``: 2 of the 14 directories in
``docs/CONTRACTS.md`` §2. ``docs/lanes/AUDIT.md`` finding **S10** ran the scanner over the whole tree, got 32
hits, read every one by hand and confirmed **none** is a record deletion. This module now scans all fourteen
lane directories plus ``sightline/pipeline.py`` and ``sightline/schemas.py``, and carries the audit's reasoning
as :data:`ALLOWANCES` - one entry per site, each saying *why* that particular line is not a record deletion. A
delete-shaped line with no matching entry still fails, so a new deletion cannot ride in on the coat-tails of an
old benign one.

The three passes
----------------
1. **Code pass.** Every string literal and comment is blanked with :mod:`tokenize` before matching, so the
   pattern table below - which necessarily contains the forbidden spellings - is invisible to it.
2. **Raw pass.** SQL statements, matched against unblanked text so they are caught inside the query strings
   where they actually live. Each regex is written so that its own source text cannot match it.
3. **Wording pass** (:func:`_wording_violations`), token-aware. R10 is a rule about *words* as well as code: no
   operator may be told an area is finished. The naive version of this check - a raw grep for the
   segment-is-done vocabulary - fired on eight lines of prose that promise the exact opposite ("nothing in this
   package has such a state", "the map shows POD, never that"), and that false positive is what turned
   ``tests/test_api.py::test_this_lane_never_says_a_segment_is_cleared`` red on 2026-09-10 at 23:57
   (``docs/lanes/AUDIT.md``, S10 and "Transients"). What distinguishes a claim from its denial is not the word,
   it is three things, and this pass uses all three:

   * **Where the word sits.** A docstring or ``#`` comment is prose *about* the rule. A string literal whose
     entire body is one word from the vocabulary is a *state value* - it is what gets written to a field and
     rendered on a map - and no amount of surrounding prose makes that acceptable. A bare identifier in code is
     the same thing under a different name.
   * **Negation and modality.** "never", "nothing", "cannot", "no ... state", "forbidden" turn an assertion
     into its denial.
   * **How far away the denial is.** It has to govern the same sentence. A paragraph that says "no" somewhere
     does not license a claim four sentences later, so the lookback stops at ``.!?;:``, at a blank line, at the
     end of the comment/string the word lives in, and after 3 physical lines or 400 characters.

   The pass therefore *allows* ``never "..."`` and *fails* on ``status = "..."`` even when the same line also
   says "never" - which is exactly the asymmetry :func:`self_test` mutation W3 checks.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
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

#: The two directories lane B4 owns. Kept as its own name because ``tests/test_export.py`` scans exactly this
#: subset, and because the wider list below is a *system* guard rather than a lane guard.
TRIAGE_LANE_DIRS: tuple[str, ...] = ("sightline/triage", "sightline/export")

#: Every source path the R10 source guard covers: the fourteen lane directories of ``docs/CONTRACTS.md`` §2 in
#: table order, then the two loose modules. Entries may be directories (scanned recursively for ``*.py``) or
#: single ``.py`` files. Widened from ``TRIAGE_LANE_DIRS`` on 2026-09-11 per AUDIT.md S10.
LANE_SOURCE_DIRS: tuple[str, ...] = (
    "sightline/common",
    "sightline/ingest",
    "sightline/detect",
    "sightline/geo",
    "sightline/track",
    "sightline/dedup",
    "sightline/triage",
    "sightline/export",
    "sightline/coverage",
    "sightline/plan",
    "sightline/store",
    "sightline/api",
    "sightline/eval",
    "sightline/mission",
    "sightline/pipeline.py",
    "sightline/schemas.py",
)


#: **What this scan does NOT do, stated plainly.** It is a pattern scan over source text, and semantic record
#: loss is invisible to it. The clearest example, from red-team review: ``records = [r for r in records if
#: r.score >= t]`` deletes records with no delete keyword anywhere in the line, and is the single most likely
#: way a real engineer would drop records by accident. Eight statements of that exact shape already exist in
#: the scanned tree (all benign -- metric slices and per-clip working state) and the scan is silent on every
#: one. Detecting them properly means dataflow analysis, not another regex. Treat this scan as a tripwire for
#: delete-SHAPED code; the enforcement that cannot be talked past is the store's eight SQLite ``RAISE(ABORT)``
#: triggers, which act on the record log itself rather than on the text that manipulates it.
_SCAN_IS_A_TRIPWIRE_NOT_A_PROOF = True


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
    rel: str = ""

    def __str__(self) -> str:
        return f"{self.rel or self.path}:{self.line_no}: {self.kind} [{self.pattern}] :: {self.line.strip()}"


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
    # `sightline/api/` came into scope in 2026-09-11's widening. An HTTP verb that removes a record is the most
    # plausible way R10 would ever be broken there, and it is invisible to every pattern above.
    (r"@\s*\w[\w.]*\s*\.\s*delete\s*\(", "HTTP delete route"),
    # --- added 2026-09-11 after red-team review planted eight real R10 violations and the gate reported
    # `0 unallowed`. Each line below is one evasion that actually worked against a sandbox copy of the tree.
    (r"\.delete\s*\(", "delete call"),                       # store.delete(rid), session.delete(rec)
    (r"methods\s*=\s*\[[^\]]*[Dd][Ee][Ll][Ee][Tt][Ee]", "HTTP delete route"),  # @router.api_route(..., methods=["DELETE"])
    (r"(?<![\w.])del\s*\(", "del statement"),                 # del(x) is valid Python; the old rule needed a space
    (r"\.__delitem__\s*\(", "container removal"),
    (r"(?<![\w.])(?:remove|unlink)\s*\(", "filesystem removal"),   # `from os import remove` then remove(p)
    # Only an assignment of an EMPTY container to a slice is a deletion; `rgb[:rows] = (...)` is an ordinary
    # write and `x[:3] == y` is a comparison. The first draft of this rule matched all three.
    (r"\[\s*[\w.]*\s*:\s*[\w.]*\s*\]\s*=\s*(?:\[\s*\]|\(\s*\)|set\s*\(\s*\))", "slice emptying"),
)

#: Pass 2 - matched against the raw text, so it also sees string literals and comments. Each regex is written so
#: that its own source spelling cannot match it: the SQL ones put a metacharacter where the space would be.
_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"DELETE\s+FROM", "SQL row deletion"),
    (r"TRUNCATE\s+TABLE", "SQL table truncation"),
    (r"DROP\s+TABLE", "SQL table drop"),
)

#: Pass 3 - the segment-is-done vocabulary, matched token-aware (see the module docstring). Every alternative is
#: written with a character class or a group where a plain letter would be, so this tuple's own source text can
#: never match the regex built from it. ``tests/test_guardrails_scan.py`` asserts that property directly rather
#: than trusting the comment.
_DONE_WORDS: tuple[str, ...] = (
    r"clear[e]d",
    r"all[ _\-]?clear",
    r"search(?:ed)?[ _\-]?c[o]mplete",
    r"search(?:ed)?[ _\-]?d[o]ne",
    r"area[ _\-]?c[o]mplete",
    r"segment[ _\-]?c[o]mplete",
)
#: Prose boundary: `\b`, so "unclearable" and "clearable" are left alone and a word is a word.
_WORDING_RX = re.compile(r"\b(?:" + "|".join(_DONE_WORDS) + r")\b", re.IGNORECASE)

#: Code boundary: an underscore separates words in an identifier but is a *word character* to `\b`, so the
#: strict regex above cannot see `def mark_cleared(...)` at all - it was silently missed until an edge probe
#: went looking for it. This variant treats `_` and `-` as boundaries and tolerates a past-tense suffix, and it
#: is used ONLY where the match is real code or a value; in prose it would fire on identifiers quoted inside
#: sentences (this module's own docstring names a test whose function name embeds the vocabulary). Measured
#: over the 104 scanned files, the looser boundary changes the reported output by ZERO lines: there is not a
#: single loose-only match in code context anywhere in the tree. (An earlier note here claimed "exactly one
#: line". That was wrong and is corrected rather than quietly dropped: re-measured three ways after red-team
#: review, the counts are 4 lines matching loose-but-not-strict, 3 that a loose-in-PROSE rule would newly
#: report, and 0 added to today's actual output. None of them is 1. The rule is still worth having -- it is
#: what catches `def mark_cleared` -- but its value is prospective, not something it is currently finding.)
_WORDING_LOOSE_RX = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(_DONE_WORDS) + r")d?(?![A-Za-z0-9])", re.IGNORECASE
)

#: A literal shaped like an identifier or enum member rather than a sentence: no spaces, no punctuation.
_IDENTIFIER_LIKE = re.compile(r"[A-Za-z0-9_\-]{1,40}")

#: Negation and modality that turn a claim into its denial. Deliberately narrow: it holds the words that make a
#: sentence mean the opposite, not every hedge. "would", for instance, is *not* here - a counterfactual is not a
#: denial, which is why ``sightline/eval/groundtruth.py`` needs an explicit entry in :data:`ALLOWANCES` below
#: rather than being waved through by the regex.
_DENIAL_RX = re.compile(
    r"(?:\b(?:no|not|none|nothing|nor|neither|never|nowhere|without|unable|un[a-z]*able|forbid[a-z]*"
    r"|refus[a-z]*|den(?:y|ies|ied|ial)|prevent[a-z]*|prohibit[a-z]*)\b|\bcan(?:no|')?t\b)",
    re.IGNORECASE,
)

#: A denial governs only its own sentence. These end one.
_SENTENCE_END = ".!?;:"
_LOOKBACK_LINES = 3
_LOOKBACK_CHARS = 400

KIND_STATE_LITERAL = "a segment-is-done status literal, forbidden by R10"
KIND_MESSAGE = "an unqualified segment-is-done claim in a message string, forbidden by R10"
KIND_PROSE = "an unqualified segment-is-done claim in prose, forbidden by R10"
KIND_IDENTIFIER = "a segment-is-done identifier in code, forbidden by R10"

_STRINGY_TOKENS = {tokenize.STRING, tokenize.COMMENT}
_FSTRING_TOKENS: set[int] = set()
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):  # 3.12+ splits f-strings into their own tokens
    _tt = getattr(tokenize, _name, None)
    if _tt is not None:
        _STRINGY_TOKENS.add(_tt)
        _FSTRING_TOKENS.add(_tt)

#: Tokens that mean "a new statement starts here", used to tell a docstring from an ordinary string literal.
_STATEMENT_BREAKS = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}


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


# --- 2a. the wording pass -----------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Span:
    """One comment or string token, with the two facts the wording pass needs about it."""

    srow: int
    scol: int
    erow: int
    ecol: int
    prose: bool  # a ``#`` comment or a docstring: text *about* the code, not text the code emits
    bare: bool  # the whole literal body is one word of the segment-is-done vocabulary, i.e. a state value

    def contains(self, row: int, col: int) -> bool:
        return (self.srow, self.scol) <= (row, col) < (self.erow, self.ecol)


def _literal_body(tok_text: str) -> str:
    """The inside of a string literal: prefix letters and matching quotes stripped."""
    s = tok_text.lstrip("rRbBuUfF")
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            return s[len(q) : -len(q)]
    return s


def _is_state_value(body: str) -> bool:
    """Is this literal a *value* meaning "finished", rather than a sentence about one?

    Two shapes count: the literal is exactly a word of the vocabulary, or it is identifier-shaped - no spaces,
    no punctuation, the way an enum member or a status column is written - and contains one. A literal that
    carries its own denial ("never ...") is neither; it is a message, and goes through the sentence rule with
    everything else. Getting that exemption wrong in the permissive direction is the whole defect this module
    was rewritten to fix, so it is deliberately the narrowest of the three tests here.
    """
    s = body.strip()
    if not s or _DENIAL_RX.search(s):
        return False
    if _WORDING_LOOSE_RX.fullmatch(s):
        return True
    return bool(_IDENTIFIER_LIKE.fullmatch(s) and _WORDING_LOOSE_RX.search(s))


def _text_spans(source: str) -> list[_Span]:
    """Locate every comment and string token and classify it prose / message / bare state value.

    A string is treated as a docstring - i.e. prose - when it opens a statement, which is what a module, class,
    function or attribute docstring does. ``x = "..."`` is not a docstring, and neither is any f-string.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    spans: list[_Span] = []
    at_statement_start = True
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            spans.append(_Span(tok.start[0], tok.start[1], tok.end[0], tok.end[1], prose=True, bare=False))
            continue
        if tok.type in _STRINGY_TOKENS:
            is_f = tok.type in _FSTRING_TOKENS
            body = "" if is_f else _literal_body(tok.string)
            spans.append(
                _Span(
                    tok.start[0],
                    tok.start[1],
                    tok.end[0],
                    tok.end[1],
                    prose=at_statement_start and not is_f,
                    bare=_is_state_value(body),
                )
            )
            at_statement_start = False
            continue
        if tok.type in _STATEMENT_BREAKS:
            at_statement_start = True
        elif tok.type != tokenize.NL:
            at_statement_start = False
    return spans


def _span_at(spans: Sequence[_Span], row: int, col: int) -> _Span | None:
    for sp in spans:
        if sp.contains(row, col):
            return sp
    return None


def _sentence_before(lines: Sequence[str], spans: Sequence[_Span], row: int, col: int) -> str:
    """The text of the sentence up to ``(row, col)``, walked backwards across wrapped lines.

    Stops at ``.!?;:``, at a blank line, at a line whose tail is in a different token than the match (so a
    denial in a neighbouring *code* line cannot license prose), and after ``_LOOKBACK_LINES`` /
    ``_LOOKBACK_CHARS``. Shortening the window can only ever make the check stricter, so the crude sentence
    split - which will also cut at "§5.3b." and "0.99" - is safe in the direction that matters.
    """
    here = _span_at(spans, row, col)
    parts: list[str] = []
    r, c = row, col
    while True:
        seg = lines[r - 1][:c]
        cut = max(seg.rfind(t) for t in _SENTENCE_END)
        if cut >= 0:
            parts.append(seg[cut + 1 :])
            break
        parts.append(seg)
        if len(parts) >= _LOOKBACK_LINES or r <= 1:
            break
        prev = lines[r - 2]
        if not prev.strip():
            break
        prev_span = _span_at(spans, r - 1, max(0, len(prev) - 1))
        if prev_span is not here:
            break
        r, c = r - 1, len(prev)
    return " ".join(reversed(parts))[-_LOOKBACK_CHARS:]


def _wording_violations(path: Path, rel: str, source: str, raw_lines: Sequence[str]) -> list[Violation]:
    """R10 in words. See the module docstring for why this is not a grep."""
    spans = _text_spans(source)
    found: list[Violation] = []
    for i, line in enumerate(raw_lines, start=1):
        for m in _WORDING_LOOSE_RX.finditer(line):
            sp = _span_at(spans, i, m.start())
            if sp is None:
                # A bare identifier - an attribute, a variable or a function name built out of the vocabulary.
                # That is state meaning "finished" whatever it is called, and no surrounding prose excuses it.
                found.append(Violation(str(path), i, line, _WORDING_LOOSE_RX.pattern, KIND_IDENTIFIER, rel))
                continue
            if sp.bare:
                # The literal *is* the value: a status on its way to a field or a map legend. Deliberately
                # checked before both the boundary rule (so `"segment_cleared"` counts) and the denial rule
                # (so a "never" on the same line cannot launder it).
                found.append(Violation(str(path), i, line, _WORDING_RX.pattern, KIND_STATE_LITERAL, rel))
                continue
            if not _WORDING_RX.match(line, m.start()):
                # Inside a comment or a string the underscore-tolerant boundary is wrong: it fires on
                # identifiers *quoted* in a sentence, e.g. the name of a test. Prose gets the strict rule.
                continue
            if _DENIAL_RX.search(_sentence_before(raw_lines, spans, i, m.start())):
                continue
            kind = KIND_PROSE if sp.prose else KIND_MESSAGE
            found.append(Violation(str(path), i, line, _WORDING_RX.pattern, kind, rel))
    return found


# --- 2b. the reasoned allowances ----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Allowance:
    """One reviewed site that is delete-shaped but is not a record deletion.

    Matching is by ``(path, kind, snippet)`` and deliberately **not** by line number: line numbers drift with
    every edit above them, and an allowance that silently slides onto a different line would be worse than no
    allowance at all. ``sites`` is how many lines in that file the entry may excuse, so pasting a second copy of
    an allowed line is itself a failure - the reviewer has to look at the new one.
    """

    path: str  #: repo-relative, posix separators
    kind: str  #: the :attr:`Violation.kind` this excuses
    snippet: str  #: substring of the offending line after whitespace normalisation
    sites: int  #: how many lines in that file this entry covers
    why: str  #: the argument that it is not a record deletion
    seen_at: str  #: line number(s) at review time, for a human following along - never used for matching


_AUDIT = "docs/lanes/AUDIT.md S10 (32 hits read by hand); re-read line by line 2026-09-11 for this widening"

#: Every delete-shaped site in the scanned tree, with the reason it stays. Read `docs/lanes/AUDIT.md` S10 first:
#: it is the specification this table implements. The rule for adding an entry is that the reason has to survive
#: a hostile reader - "it is only working state" is an argument, "it looked fine" is not.
ALLOWANCES: tuple[Allowance, ...] = (
    # --- category 1: benign container work ---------------------------------------------------------------
    Allowance(
        "sightline/api/coverage_feed.py", "container pop", "open_rects.pop(key)", 1,
        "Run-length rectangle merge over a boolean mask. `open_rects` is a scratch dict of still-open "
        "rectangles keyed by (j0, j1); the popped start row is appended to `closed` on the same line, so the "
        "rectangle is carried forward rather than lost. No Record is in scope in this function.",
        "line 307",
    ),
    Allowance(
        "sightline/api/demo_control.py", "container reset", "self._log.clear()", 1,
        "`self._log` is a bounded deque of STDOUT LINES from a demo subprocess, shown on the demo console "
        "so a judge can watch a run without a terminal. It is reset when a NEW demo starts so the console "
        "does not show the previous run's output as if it were this one's. No Record, no segment and no "
        "evidence is in scope in this module: it starts and stops processes, and the record log lives in "
        "`sightline/store` and is never opened here. The guard is right to flag the shape - a bare "
        "container reset is indistinguishable from a record purge without reading it - which is why this "
        "entry exists rather than the call being rewritten to dodge the pattern.",
        "line 115, Runner.start",
    ),
    Allowance(
        "sightline/api/live.py", "set discard", "self._clients.discard(ws)", 1,
        "`unregister` drops a disconnected websocket from the subscriber set. It removes a *listener*, not "
        "data; the record log is `sightline/store` and is untouched by a client going away.",
        "line 88",
    ),
    Allowance(
        "sightline/api/wire.py", "container pop", 'props.pop("score_components", None)', 1,
        "Pops from `props = dict(feature.get('properties'))`, a private copy of the inbound wire payload made "
        "one line earlier. The value is put straight back as `kw['components']`, so the round trip is "
        "lossless; `tests/test_api.py` asserts the reconstructed Record equals the original.",
        "line 39",
    ),
    Allowance(
        "sightline/api/wire.py", "container pop", 'props.pop("evidence", None)', 1,
        "The same private copy of the inbound wire payload, one line further down: the popped list is rebuilt "
        "into `kw['evidence']` as Evidence objects on the next statement, so no thumbnail reference is lost "
        "in the round trip. Nothing is removed from a Record; the Record does not exist yet at this point.",
        "line 40",
    ),
    Allowance(
        "sightline/detect/__main__.py", "container pop", 'man.pop("data_yaml_text", None)', 1,
        "CLI display only: drops a multi-kilobyte YAML blob out of the manifest dict immediately before "
        "`json.dumps(man)` prints it to stdout. `build_yolo_dataset` has already written the manifest to disk "
        "and that file keeps the field.",
        "line 64",
    ),
    Allowance(
        "sightline/detect/__main__.py", "del statement", "del img", 2,
        "Releases the decoded 4K frame - roughly 25 MB per array - so the predict and overlay CLIs do not hold "
        "every frame of a 731-frame run at once on a 16 GB machine. `del` on a local name, which unbinds a "
        "reference; nothing is removed from any container.",
        "lines 95 and 127",
    ),
    Allowance(
        "sightline/detect/__main__.py", "container pop", 'summary.pop("_result", None)', 1,
        "Drops the non-JSON-serialisable `_result` object before `json.dumps(summary)`. The evaluation itself "
        "has already been written to `out_dir` by `run_evaluation`.",
        "line 174",
    ),
    Allowance(
        "sightline/detect/dataset.py", "container pop", "zones.pop() if len(zones) == 1", 1,
        "Reads the single member out of a one-element set, guarded by `len(zones) == 1` on the same line - the "
        "idiomatic way to unwrap a set. The set is a local built from `fr.labels` two lines above.",
        "line 678",
    ),
    Allowance(
        "sightline/eval/slicing.py", "container pop", "return domains.pop()", 1,
        "Unwraps the one-element set of domains after `require_single_domain` has already raised "
        "`DomainMixError` for len 0 and len > 1. Reading a set, not shrinking a result.",
        "line 205",
    ),
    Allowance(
        "sightline/eval/slicing.py", "container pop", "values.pop() if len(values) == 1", 1,
        "Same one-element-set unwrap, per slice axis, guarded on the same line; the many-valued case falls "
        "through to the string 'all' rather than dropping anything.",
        "line 221",
    ),
    Allowance(
        "sightline/geo/noise.py", "container pop", 'kw.pop("seed", 0)', 1,
        "Reads a keyword default out of `**kw` so `SIGHTLINE_GEO_NOISE_SEED` can override it, then passes the "
        "result to the constructor. A dict-get with a default, spelled with pop because the key must not also "
        "reach `cls(...)` twice.",
        "line 77",
    ),
    Allowance(
        "sightline/geo/noise.py", "container reset", "self.samples.clear()", 1,
        "`reseed()` empties the *diagnostic* sample buffer of the geolocation noise model so that rerunning a "
        "seed reproduces an identical list - determinism, which `tests/test_geo.py` asserts. These are drawn "
        "error terms kept for plotting, not detections and not records.",
        "line 122",
    ),
    Allowance(
        "sightline/plan/revisit.py", "container pop", "self.items.pop(0) if self.items else None", 1,
        "`RevisitQueue.pop` dequeues the next revisit *task*. The Record the task points at is untouched and "
        "stays in the store; the module docstring makes the same point at line 9. A dequeued task means "
        "'this one is being flown now', which is the opposite of removing it from attention.",
        "line 62",
    ),
    Allowance(
        "sightline/plan/revisit.py", "container pop", "a, b = stack.pop()", 1,
        "The work stack of the pure-python flood fill used to label connected components when scipy is "
        "missing. Pixels of a mask, inside an `except ImportError` fallback.",
        "line 165",
    ),
    Allowance(
        "sightline/schemas.py", "container pop", 'props.pop("lat"), props.pop("lon"), props.pop("alt_msl_m")', 1,
        "`Record.to_feature` pops from `asdict(self)`, a fresh dict built on the line above. All three values "
        "reappear two lines later inside `geometry.coordinates`; RFC 7946 requires the position in the "
        "geometry rather than in properties. The Record itself is a frozen-contract dataclass and is not "
        "mutated.",
        "line 340",
    ),
    Allowance(
        "sightline/schemas.py", "container pop", 'props.pop("components")', 1,
        "Same `asdict` copy: renames `components` to `score_components` in the emitted Feature. The value is "
        "assigned on the same line, so nothing is dropped.",
        "line 341",
    ),
    Allowance(
        "sightline/store/db.py", "container removal", "self._subs.remove(cb)", 1,
        "The unsubscribe closure returned by `subscribe()`; it removes a callback from a listener list. Row "
        "deletion in this lane is prevented structurally rather than by convention - SQLite triggers "
        "r10_records_no_delete, r10_versions_no_delete, r10_evidence_no_delete and r10_audit_no_delete - and "
        "AUDIT.md mutation M21 confirms that dropping one of those triggers turns `tests/test_store.py` red.",
        "line 294",
    ),
    Allowance(
        "sightline/store/outbox.py", "container pop", "self._attempts.pop(key, None)", 1,
        "Clears the retry counter for a key that has just been acknowledged by the receiver. The outbox queue "
        "itself has no eviction and no maximum depth, which AUDIT.md S10 checked explicitly; only the "
        "in-memory attempt tally is reset, and only on success.",
        "line 210",
    ),
    # --- category 2: track pruning (tracks are working state, records are not) ----------------------------
    Allowance(
        "sightline/track/tracker.py", "del statement", "del self._frame_times[k]", 1,
        "Trims the `frame_idx -> t_utc` lookup to `cfg.track_buffer_frames + 8` entries, oldest first. Frame "
        "timestamps used to backdate a track's creation; a dropped one only costs the backdating fallback on "
        "the next line, which is `bundle.t_utc - age / fps`.",
        "line 226",
    ),
    Allowance(
        "sightline/track/tracker.py", "container pop", "st = self._state.pop(backend_id)", 1,
        "`_prune` drops tracks that were never confirmed and have gone quiet past `unconfirmed_prune_s`. "
        "Tracks are per-clip working state; a Record is only ever created by `sightline/dedup` once a track "
        "*is* confirmed, so nothing that reaches the record log can be pruned here. The function's own "
        "docstring states this and the count is published as `stats.tracks_pruned_unconfirmed` rather than "
        "being silent.",
        "line 256",
    ),
    Allowance(
        "sightline/track/tracker.py", "container pop", "self._tracks.pop(st.global_id, None)", 1,
        "The wrapper-side half of the same never-confirmed prune, one line below, counted by the same stat.",
        "line 257",
    ),
    # --- category 3: the bounded LRU merge buffer ---------------------------------------------------------
    Allowance(
        "sightline/api/detect_fallback.py", "container popitem", "self._frames.popitem(last=False)", 2,
        "`FallbackMerger` is an `OrderedDict` LRU bounded at `maxlen=300` frames that holds *detections per "
        "frame* while an on-board result waits for the cloud result of the same frame. It is a merge buffer, "
        "not the record log. Eviction loses no evidence: a cloud reply that arrives after its frame was "
        "evicted is still accepted, re-inserted and counted in `self.late`, and the class docstring at line "
        "118 says so. AUDIT.md S10 went looking for exactly this as an unbounded cache and found it bounded "
        "and observable.",
        "lines 135 and 146",
    ),
    # --- category 4: prose the wording rule cannot read as a denial ---------------------------------------
    Allowance(
        "sightline/eval/groundtruth.py", KIND_PROSE, "hiding it would overstate what the search", 1,
        "`GtBox.findable`'s docstring, arguing why a buried actor is excluded from recall and reported on its "
        "own row. The sentence is counterfactual - it describes what *would* happen if the box were hidden - "
        "so it asserts nothing about any real segment. `_DENIAL_RX` deliberately holds no hedges, only real "
        "negation, so a subjunctive does not license the vocabulary automatically; this one is licensed here, "
        "by name, instead.",
        "line 145",
    ),
)


def _norm(line: str) -> str:
    return " ".join(line.split())


def _rel_of(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _matches(a: Allowance, v: Violation) -> bool:
    """Does this allowance cover this violation?

    Two rules here are load-bearing, and both were added after red-team review defeated the first version:

    * **The blessed snippet is tested against the CODE-BLANKED line, never the raw one.** `v.line` is raw, so
      `self._records.discard(rid)  # mirrors self._clients.discard(ws)` was granted an allowance *by its own
      comment* -- a real deletion smuggled in on the strength of a code shape that only appeared in a comment.
      That needed no contortion at all, which made it the worst of the three laundering routes found.
    * **`in` was a substring test.** The blessed `del img` therefore also covered
      `del img_store.records[fr.stem]`, because the allowed snippet is a prefix of the new statement. The
      match is now anchored to the whole normalised statement.
    """
    if v.rel != a.path or v.kind != a.kind:
        return False
    # Drop the COMMENT, keep the code — including its string literals, which several honest allowances quote
    # (`kw.pop("seed", 0)`). Comments are the laundering vector, not strings: `_matches` used to test the RAW
    # line, so `self._records.discard(rid)  # mirrors self._clients.discard(ws)` was granted an allowance by
    # its own comment, smuggling a real deletion in on a code shape that existed only in a comment.
    # `_blank_strings_and_comments` turns a comment into trailing spaces, so the end of the code region is
    # exactly where the blanked line stops having content.
    if not v.line:
        return False
    blanked = _blank_strings_and_comments(v.line)[0]
    line = _norm(v.line[:len(blanked.rstrip())])
    snip = _norm(a.snippet)
    if not snip:
        return False
    at = line.find(snip)
    if at < 0:
        return False
    # The snippet has to end on a TOKEN BOUNDARY. A bare substring test let the blessed `del img` cover
    # `del img_store.records[fr.stem]`, because the allowed statement is a prefix of the new one. Requiring
    # the next character not to continue the expression keeps every honest allowance (they end at a `)` or
    # end-of-statement) while refusing an extension of the blessed name.
    nxt = line[at + len(snip):at + len(snip) + 1]
    return nxt == "" or not (nxt.isalnum() or nxt in "_.[(")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scan_source(path: Path | str, repo_root: Path | str | None = None) -> list[Violation]:
    """Scan one Python file for delete-shaped operations on records. Allowances are **not** applied here.

    Kept allowance-free on purpose: ``tests/test_triage.py`` and ``tests/test_live_mission.py`` both plant a
    violation in a temp file and assert this returns something, and an allowance filter in this function would
    make those probes depend on the table.
    """
    p = Path(path)
    root = Path(repo_root) if repo_root is not None else _repo_root()
    rel = _rel_of(p, root)
    source = p.read_text(encoding="utf-8")
    raw_lines = source.splitlines()
    code_lines = _blank_strings_and_comments(source)
    found: list[Violation] = []
    for pattern, kind in _CODE_PATTERNS:
        rx = re.compile(pattern)
        for i, line in enumerate(code_lines, start=1):
            if rx.search(line):
                found.append(Violation(str(p), i, raw_lines[i - 1], pattern, kind, rel))
    # SQL is CASE-INSENSITIVE and `DELETE`/`FROM` can sit on different lines of a triple-quoted query. The
    # per-line, case-sensitive version of this loop missed a lowercase SQL row-deletion against `records`
    # outright -- lowercase SQL is ordinary style, so the guard was blind to the single most direct way to
    # break R10. Matching the JOINED source with IGNORECASE catches both; the line number is recovered from
    # the match offset so the report still points at a line.
    for pattern, kind in _TEXT_PATTERNS:
        rx = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        for m in rx.finditer(source):
            i = source.count(chr(10), 0, m.start()) + 1
            line = raw_lines[i - 1] if i - 1 < len(raw_lines) else ""
            found.append(Violation(str(p), i, line, pattern, kind, rel))
    found.extend(_wording_violations(p, rel, source, raw_lines))
    return found


def _iter_sources(root: Path, entries: Sequence[str]) -> list[Path]:
    """Every ``.py`` file named by ``entries``, which may hold directories or single files."""
    out: list[Path] = []
    for rel in entries:
        target = root / rel
        if target.is_dir():
            out.extend(sorted(target.rglob("*.py")))
        elif target.is_file() and target.suffix == ".py":
            out.append(target)
        else:
            raise GuardrailError(f"R10 scanner: {target} does not exist; the lane layout changed")
    return out


def scan_lane_sources(
    repo_root: Path | str | None = None,
    dirs: Sequence[str] = LANE_SOURCE_DIRS,
    *,
    include_allowed: bool = False,
) -> list[Violation]:
    """Scan every ``.py`` file under ``dirs``. An empty result is the R10 pass condition.

    ``dirs`` entries may be directories or single ``.py`` files. By default the reviewed sites in
    :data:`ALLOWANCES` are filtered out; pass ``include_allowed=True`` for the raw list, which is what the
    reporting CLI and the tests that count hits use.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    found: list[Violation] = []
    for py in _iter_sources(root, dirs):
        found.extend(scan_source(py, root))
    if include_allowed:
        return found
    return [v for v in found if not any(_matches(a, v) for a in ALLOWANCES)]


def _in_scope(a: Allowance, entries: Sequence[str]) -> bool:
    """Is this allowance's file inside the paths being scanned? Keeps subset scans from auditing the rest."""
    return any(a.path == e or a.path.startswith(e.rstrip("/") + "/") for e in entries)


def audit_allowances(
    repo_root: Path | str | None = None, dirs: Sequence[str] = LANE_SOURCE_DIRS
) -> list[str]:
    """Check the allowance table against reality. Empty means the table still describes the tree.

    Two ways it can rot, both reported:

    * an entry that matches **nothing** - the code it excused was moved or rewritten, so the reasoning has not
      been checked against what is there now and the entry has to go or be re-argued;
    * an entry that matches **more** lines than its ``sites`` count - somebody added another delete-shaped line
      that happens to look like an already-blessed one. That is exactly the case a blanket suppression would
      swallow, and it is the reason ``sites`` exists.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    raw = scan_lane_sources(root, dirs, include_allowed=True)
    problems: list[str] = []
    for a in ALLOWANCES:
        if not _in_scope(a, dirs):
            continue
        hits = [v for v in raw if _matches(a, v)]
        if not hits:
            problems.append(
                f"stale allowance: {a.path} [{a.kind}] :: {a.snippet!r} matches nothing (was {a.seen_at}). "
                "Re-read the site and update or drop the entry; do not widen it."
            )
        elif len(hits) > a.sites:
            lines = ", ".join(str(v.line_no) for v in hits)
            problems.append(
                f"over-broad allowance: {a.path} [{a.kind}] :: {a.snippet!r} covers {a.sites} site(s) but "
                f"matched {len(hits)} (lines {lines}). A new delete-shaped line appeared - review it on its "
                "own merits before touching this number."
            )
    return problems


def assert_no_record_deletion(
    repo_root: Path | str | None = None, dirs: Sequence[str] = LANE_SOURCE_DIRS
) -> None:
    """Raise :class:`GuardrailError` listing every unallowed violation and every rotten allowance."""
    found = scan_lane_sources(repo_root, dirs)
    problems = audit_allowances(repo_root, dirs)
    if not found and not problems:
        return
    parts = []
    if found:
        parts.append(
            f"{len(found)} delete-shaped operation(s) with no reviewed allowance:\n"
            + "\n".join(f"  {v}" for v in found)
        )
    if problems:
        parts.append(f"{len(problems)} allowance(s) no longer describe the tree:\n" + "\n".join(f"  {p}" for p in problems))
    raise GuardrailError("R10 violated - " + "\n".join(parts))


# --- 3. the gate ---------------------------------------------------------------------------------------------
#: (name, probe source, kind fragment that must appear). Each is a mutation the scanner has to catch; if any of
#: them stops firing the gate has gone toothless and `--self-test` exits non-zero.
#:
#: Every forbidden spelling here is split across a ``+`` so that *this* file never contains it adjacently while
#: the probe written to disk does. Python folds the concatenation at compile time, so the probe is byte-exact.
#: Getting that wrong is not theoretical: the first version of this table wrote the ``+`` inside the probe
#: string instead of outside it, five cases silently tested nothing, and `--self-test` is what caught it.
_SELF_TEST_CASES: tuple[tuple[str, str, str], ...] = (
    ("D1 delete-shaped function", "def purge_records(store):\n    pass\n", "delete-shaped function"),
    ("D2 list removal", "def go(records):\n    records.remove(records[0])\n", "container removal"),
    ("D3 dict pop", "def go(store):\n    store.pop('rec-001')\n", "container pop"),
    ("D4 del statement", "def go(records):\n    del records[1]\n", "del statement"),
    ("D5 container reset", "def go(records):\n    records.clear()\n", "container reset"),
    ("D6 filesystem removal", "import os\ndef go(p):\n    os.remove(p)\n", "filesystem removal"),
    ("D7 tree removal", "import shutil\ndef go(p):\n    shutil.rmtree(p)\n", "recursive tree removal"),
    ("D8 SQL row deletion", "def go(c):\n    c.execute('DELETE" + " FROM records')\n", "SQL row deletion"),
    ("D9 SQL table drop", "def go(c):\n    c.execute('DROP" + " TABLE records')\n", "SQL table drop"),
    ("D10 HTTP delete route", "@app.delete('/api/records/{i}')\ndef go(i):\n    pass\n", "HTTP delete route"),
    # The wording half. W1/W2 are the claims; W3 is the one that matters most, because it is the case a
    # negation-only rule gets wrong: a state value on a line that also carries the word "never".
    ("W1 state literal", "def go(seg):\n    seg.status = 'clear" + "ed'\n", "status literal"),
    ("W2 message string", "def go(ui):\n    ui.say('Segment 4 is now clear" + "ed')\n", "message string"),
    (
        "W3 denial must not launder a state value",
        "STATUS = 'never'\ndef go(seg):\n    seg.status = 'clear" + "ed'  # we never do this\n",
        "status literal",
    ),
    ("W4 code identifier", "def go(seg):\n    seg.clear" + "ed = True\n", "identifier in code"),
    # W5/W6 are the gap an edge probe found after the first version shipped: `_` is a word character, so `\b`
    # cannot see the vocabulary inside an identifier at all. `def mark_cleared(...)` scanned clean.
    ("W5 function named for the state", "def mark_clear" + "ed(seg):\n    seg.done = True\n", "identifier in code"),
    ("W6 compound state literal", "def go(seg):\n    seg.status = 'segment_clear" + "ed'\n", "status literal"),
)

#: Sources that must stay clean: prose that denies, and the container idioms the pattern table names in text.
_SELF_TEST_CLEAN: tuple[tuple[str, str], ...] = (
    (
        "C1 prose denying a claim",
        '"""Nothing in this package has a "clear' + 'ed" state (guardrail R10).\n\n'
        'The map shows POD, never "clear' + 'ed", and no amount of effort raises it to 1.\n"""\n',
    ),
    (
        "C2 denial wrapped across two lines",
        '"""There is no code path that deletes a record or sets a segment to\n`clear' + 'ed`. It stays."""\n',
    ),
    (
        "C3 forbidden spellings quoted as data",
        (
            '"""We never call .pop() or .remove(), and never del a record."""\n'
            "# a comment mentioning os.remove( and shutil.rmtree( and del x\n"
            'PATTERNS = (".pop(", ".remove(", "del ", "rmtree(")\n'
            "def keep(records):\n    return list(records)\n"
        ),
    ),
    (
        "C4 a denying message string",
        "def legend():\n    return {'note': 'POD is a probability of detection, never a clear' + 'ed flag.'}\n",
    ),
)


def self_test() -> list[str]:
    """Prove the scanner still bites. Returns a list of failures; empty is the pass condition.

    Hard rule from ``docs/QUALITY_GATE.md`` §2: a check that cannot fail is not a check. This plants each class
    of violation in a temporary file and insists it is caught, then plants the prose that a naive grep gets
    wrong and insists it is **not**. It runs from the CLI (``--self-test``) and from
    ``tests/test_guardrails_scan.py``.
    """
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="r10_selftest_") as td:
        tmp = Path(td)
        for name, source, want in _SELF_TEST_CASES:
            p = tmp / "probe.py"
            p.write_text(source, encoding="utf-8")
            kinds = [v.kind for v in scan_source(p, tmp)]
            if not any(want in k for k in kinds):
                failures.append(f"{name}: expected a {want!r} violation, got {kinds or 'nothing'}")
        for name, source in _SELF_TEST_CLEAN:
            p = tmp / "probe.py"
            p.write_text(source, encoding="utf-8")
            got = scan_source(p, tmp)
            if got:
                failures.append(f"{name}: expected no violation, got {[str(v) for v in got]}")
    # The scanner must be clean on its own source, or the pattern table could not be written down at all.
    own = scan_lane_sources(dirs=("sightline/triage/guardrails.py",), include_allowed=True)
    if own:
        failures.append("the scanner flags its own source: " + "; ".join(str(v) for v in own))
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sightline.triage.guardrails``. Exit 0 clean, 1 violated, 2 the gate itself is broken."""
    ap = argparse.ArgumentParser(prog="python -m sightline.triage.guardrails", description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="repository root (default: inferred from this file)")
    ap.add_argument("--dirs", nargs="*", default=None, help="override the scanned paths")
    ap.add_argument("--show-allowed", action="store_true", help="also list the reviewed sites and their reasons")
    ap.add_argument("--self-test", action="store_true", help="prove the scanner still catches planted violations")
    a = ap.parse_args(argv)

    if a.self_test:
        failures = self_test()
        for f in failures:
            print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
        print(f"R10 self-test: {len(_SELF_TEST_CASES)} planted violations, {len(_SELF_TEST_CLEAN)} clean "
              f"sources, {len(failures)} failure(s)")
        return 2 if failures else 0

    root = Path(a.root) if a.root else _repo_root()
    dirs = tuple(a.dirs) if a.dirs else LANE_SOURCE_DIRS
    raw = scan_lane_sources(root, dirs, include_allowed=True)
    unallowed = [v for v in raw if not any(_matches(al, v) for al in ALLOWANCES)]
    problems = audit_allowances(root, dirs)

    n_files = len(_iter_sources(root, dirs))
    print(f"R10 source guard: {n_files} file(s) over {len(dirs)} path(s); {len(raw)} raw hit(s), "
          f"{len(raw) - len(unallowed)} allowed by review, {len(unallowed)} unallowed.")
    if a.show_allowed:
        for al in ALLOWANCES:
            if not _in_scope(al, dirs):
                continue
            hits = [v for v in raw if _matches(al, v)]
            where = ", ".join(str(v.line_no) for v in hits) or "-"
            print(f"  ALLOWED {al.path}:{where} [{al.kind}] :: {al.snippet}")
            print(f"          why: {al.why}")
    for v in unallowed:
        print(f"  VIOLATION {v}", file=sys.stderr)
    for p in problems:
        print(f"  ALLOWANCE {p}", file=sys.stderr)
    return 1 if (unallowed or problems) else 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
