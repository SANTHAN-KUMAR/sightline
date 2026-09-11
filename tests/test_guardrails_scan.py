"""F21 / R10 - the *system-wide* source guard in `sightline/triage/guardrails.py`.

`tests/test_triage.py` already covers the operator half of R10 (dismiss/undismiss/retention) and asserts the
scan comes back empty. This file covers the thing that assertion depends on: **that the scan looks at the whole
system, and that it would go red if it were broken.**

Why it exists
-------------
`docs/lanes/AUDIT.md` finding S10: the shipped scanner guarded `sightline/triage` and `sightline/export`, 2 of
the 14 lane directories in `docs/CONTRACTS.md` §2. The audit ran it over the whole tree, got **32 hits, read
every one by hand, and found no record deletion**. Eight of those 32 were the wording check firing on prose that
promises the *opposite* - "nothing in this package has such a state", "the map shows POD, never that" - and that
false positive is what turned `tests/test_api.py::test_this_lane_never_says_a_segment_is_cleared` red at 23:57.

Measured here, on this tree (2026-09-11):

* before: 2 paths scanned, 32 raw hits when pointed at all 16, 8 of them false positives on denying prose;
* after: 16 paths scanned (14 lane dirs + `pipeline.py` + `schemas.py`), 104 files, **25 raw hits, 25 covered
  by a reasoned per-site allowance, 0 unallowed** - and the 8 false positives are gone while a genuine
  segment-is-done message is still caught.

The hard rule from `docs/QUALITY_GATE.md` §2 is that a check which cannot fail is not a check, so every
behaviour below is asserted **both ways**: the thing is caught, and something that only looks like it is not.
Section 3 plants violations, section 4 proves the allowances are load-bearing rather than decorative, and
section 5 mutates the real prose that started this and insists the check fires on the mutation.

Offline, CPU only, no simulator, no GPU. Run with::

    D:\\Tools\\uv\\uv.exe run pytest tests/test_guardrails_scan.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sightline.triage import guardrails as G
from sightline.triage.guardrails import (
    ALLOWANCES,
    LANE_SOURCE_DIRS,
    TRIAGE_LANE_DIRS,
    Allowance,
    GuardrailError,
    assert_no_record_deletion,
    audit_allowances,
    scan_lane_sources,
    scan_source,
    self_test,
)

REPO = Path(__file__).resolve().parents[1]

#: The lane directories `docs/CONTRACTS.md` §2 hands out, written here independently of the module so that this
#: is a real cross-check and not a tautology. If the contract table grows a lane, this list is what fails.
CONTRACT_LANE_DIRS: tuple[str, ...] = (
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
)

#: A delete-shaped line that matches no allowance anywhere, used to prove a path is actually being read.
PLANT = "def go(records):\n    records.remove(records[0])\n"


@pytest.fixture(scope="module")
def raw_hits() -> list:
    """Every hit over the whole system, allowances *not* applied. One scan, shared (it costs ~1 s)."""
    return scan_lane_sources(REPO, include_allowed=True)


@pytest.fixture(scope="module")
def unallowed() -> list:
    return scan_lane_sources(REPO)


def _line_with(rel: str, needle: str) -> int:
    """1-based line number of the single line in `rel` containing `needle`. Fails loudly if it is not unique.

    Used instead of hard-coded line numbers so this file does not go red merely because someone added an import
    above the site. The *content* is the anchor, which is also how `Allowance.snippet` works.
    """
    hits = [i for i, ln in enumerate((REPO / rel).read_text(encoding="utf-8").splitlines(), 1) if needle in ln]
    assert len(hits) == 1, f"{needle!r} appears {len(hits)} time(s) in {rel}; expected exactly 1"
    return hits[0]


# ==============================================================================================================
# 1. The widening is real: the scan reaches every lane, not just the two it used to
# ==============================================================================================================
def test_the_scan_list_covers_every_contract_lane_plus_the_two_loose_modules():
    """S10's complaint, turned into an assertion: 2 of 14 directories is not a system guard."""
    missing = [d for d in CONTRACT_LANE_DIRS if d not in LANE_SOURCE_DIRS]
    assert not missing, f"lane directories with no R10 source guard: {missing}"
    for loose in ("sightline/pipeline.py", "sightline/schemas.py"):
        assert loose in LANE_SOURCE_DIRS, f"{loose} is outside every lane directory and must be named explicitly"
    assert set(TRIAGE_LANE_DIRS) <= set(LANE_SOURCE_DIRS), "the old narrow pair must remain a subset"


def test_every_configured_path_is_actually_read(tmp_path: Path):
    """The list above is only worth what the scanner does with it.

    Builds a skeleton tree with the same shape as the repo, plants one delete-shaped line in **every** entry,
    and insists on one violation per entry. Before the widening this returned 2; a path that silently fails to
    be walked - a typo, a directory entry that is really a file - shows up here as a missing rel path.
    """
    for entry in LANE_SOURCE_DIRS:
        target = tmp_path / entry
        if entry.endswith(".py"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(PLANT, encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "probe.py").write_text(PLANT, encoding="utf-8")

    found = scan_lane_sources(tmp_path)
    assert len(found) == len(LANE_SOURCE_DIRS), [str(v) for v in found]
    reached = {v.rel for v in found}
    for entry in LANE_SOURCE_DIRS:
        want = entry if entry.endswith(".py") else f"{entry}/probe.py"
        assert want in reached, f"{entry} was never read; reached={sorted(reached)}"


def test_a_missing_path_is_an_error_not_a_silent_pass(tmp_path: Path):
    """A scan that quietly walks nothing is the worst failure mode a gate has."""
    with pytest.raises(GuardrailError, match="does not exist"):
        scan_lane_sources(tmp_path, dirs=("sightline/nowhere",))


# ==============================================================================================================
# 2. The tree is clean, and the allowance table honestly describes it
# ==============================================================================================================
def test_no_unallowed_delete_shaped_code_anywhere_in_the_system(unallowed):
    """The pass condition. Every hit is either absent or explained by name in `ALLOWANCES`."""
    assert unallowed == [], "R10: unexplained delete-shaped code\n" + "\n".join(f"  {v}" for v in unallowed)


def test_the_allowance_table_still_describes_the_tree():
    """Stale entry (the code moved) or over-broad entry (a second copy appeared) - both are failures."""
    problems = audit_allowances(REPO)
    assert problems == [], "\n".join(problems)


def test_every_raw_hit_is_covered_exactly_once(raw_hits):
    """No hit is double-covered and no allowance is dead weight.

    Measured on this tree: 25 raw hits over 23 allowances (two of which cover 2 sites each). If this number
    moves, a delete-shaped line was added or removed and somebody has to look at it - which is the point.
    """
    assert sum(a.sites for a in ALLOWANCES) == len(raw_hits), (
        f"{len(raw_hits)} raw hit(s) vs {sum(a.sites for a in ALLOWANCES)} allowed site(s)"
    )
    for v in raw_hits:
        covering = [a for a in ALLOWANCES if G._matches(a, v)]
        assert len(covering) == 1, f"{v} is covered by {len(covering)} allowance(s), expected exactly 1"


def test_every_allowance_states_a_reason_a_hostile_reader_could_check():
    """R10's allowances exist to be audited. A blank or one-word `why` is a blanket suppression in disguise."""
    for a in ALLOWANCES:
        assert a.sites >= 1 and a.seen_at.strip(), a
        assert len(a.why.split()) >= 15, f"{a.path} [{a.kind}]: reason is too thin to audit: {a.why!r}"
        assert (REPO / a.path).is_file(), f"{a.path} does not exist"


# ==============================================================================================================
# 3. Teeth: plant violations and prove they are caught; plant innocents and prove they are not
# ==============================================================================================================
def test_the_modules_own_self_test_passes():
    """`--self-test` plants 14 violation classes and 4 clean sources. Empty means the gate still bites.

    It is not decoration: the first draft of that table built its probes wrong, five cases tested nothing at
    all, and this is what said so.
    """
    assert self_test() == []


@pytest.mark.parametrize(
    ("name", "source", "kind"),
    [
        ("record deletion by name", "def delete_record(store, rid):\n    store.records.pop(rid)\n",
         "delete-shaped function"),
        ("list removal", "def go(records, r):\n    records.remove(r)\n", "container removal"),
        ("del on a container", "def go(store, rid):\n    del store.records[rid]\n", "del statement"),
        ("wipe the log", "def go(records):\n    records.clear()\n", "container reset"),
        ("SQL", "def go(c):\n    c.execute('DELETE" + " FROM records WHERE id = ?')\n", "SQL row deletion"),
        ("HTTP verb", "@app.delete('/api/records/{rid}')\nasync def go(rid):\n    ...\n", "HTTP delete route"),
        ("segment marked done", "def go(seg):\n    seg.status = 'clear" + "ed'\n", "status literal"),
        ("operator told an area is done",
         "def banner(i):\n    return f'Segment {i}: search" + " complete'\n", "message string"),
        # The gap an edge probe found after the first draft: `_` is a word character, so `\b` cannot see the
        # vocabulary inside an identifier and `def mark_cleared(...)` scanned completely clean.
        ("function named for the state", "def mark_clear" + "ed(seg):\n    seg.done = True\n",
         "identifier in code"),
        ("attribute named for the state", "def go(seg):\n    seg.area_clear" + "ed = True\n",
         "identifier in code"),
        ("compound state literal", "def go(seg):\n    seg.status = 'segment_clear" + "ed'\n",
         "status literal"),
    ],
)
def test_a_planted_violation_in_a_temp_tree_is_caught(tmp_path: Path, name: str, source: str, kind: str):
    """The inverted self-test the brief demands, one case per class of R10 breach."""
    (tmp_path / "sightline" / "geo").mkdir(parents=True)
    (tmp_path / "sightline" / "geo" / "planted.py").write_text(source, encoding="utf-8")
    found = scan_lane_sources(tmp_path, dirs=("sightline/geo",))
    assert any(kind in v.kind for v in found), f"{name}: not caught; got {[v.kind for v in found]}"


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("prose denying it outright",
         '"""Nothing in this package has a "clear' + 'ed" state (guardrail R10)."""\n'),
        ("prose denying it across a wrapped line",
         '"""There is no code path that deletes a record or sets a segment to\n`clear' + 'ed`."""\n'),
        ("a comment denying it",
         "# R10: this module never marks an area clear" + "ed; POD is a probability.\nX = 1\n"),
        ("a message string that denies it",
         "def legend():\n    return 'POD is a probability of detection, never a clear" + "ed flag.'\n"),
        ("a modal denial",
         '"""No overflight can ever leave a segment clear' + 'ed; only POD rises."""\n'),
        ("the vocabulary quoted as data, not used",
         ('"""We never call .pop() or .remove(), and never del a record."""\n'
          'PATTERNS = (".pop(", ".remove(", "del ", "rmtree(")\n')),
    ],
)
def test_innocent_prose_that_denies_clearing_is_not_caught(tmp_path: Path, name: str, source: str):
    """The defect this work exists to fix: sentences promising R10 were being reported as breaking it."""
    p = tmp_path / "innocent.py"
    p.write_text(source, encoding="utf-8")
    assert scan_source(p, tmp_path) == [], f"{name}: false positive {[str(v) for v in scan_source(p, tmp_path)]}"


def test_a_denial_on_the_line_cannot_launder_a_state_value(tmp_path: Path):
    """The asymmetry that makes the fix safe rather than merely quieter.

    A negation-only rule would wave this through because the line says "never". A bare literal is not prose: it
    is the value that lands in a field and on a map, so it fails whatever the comment beside it claims.
    """
    p = tmp_path / "sneaky.py"
    p.write_text("def go(seg):\n    seg.status = 'clear" + "ed'  # we never do this\n", encoding="utf-8")
    kinds = [v.kind for v in scan_source(p, tmp_path)]
    assert any("status literal" in k for k in kinds), kinds


def test_a_denial_in_a_previous_sentence_does_not_license_a_later_claim(tmp_path: Path):
    """"No" somewhere in the paragraph is not consent. The lookback stops at the sentence boundary."""
    p = tmp_path / "far.py"
    p.write_text('"""No record is ever removed. Segment 4 is clear' + 'ed."""\n', encoding="utf-8")
    assert scan_source(p, tmp_path), "a claim two sentences after an unrelated negation slipped through"


def test_a_denial_three_lines_up_does_not_license_a_claim(tmp_path: Path):
    """The lookback is bounded at 3 physical lines so a distant "never" cannot govern."""
    body = '"""We never delete.\n\nThe operator is told the area is clear' + 'ed."""\n'
    p = tmp_path / "distant.py"
    p.write_text(body, encoding="utf-8")
    assert scan_source(p, tmp_path), "a blank line should have ended the sentence lookback"


def test_the_loose_code_boundary_does_not_leak_into_prose(tmp_path: Path):
    """Identifiers quoted inside a sentence are prose, not state.

    The underscore-tolerant boundary that catches `def mark_cleared` would otherwise fire on any docstring
    naming a test or a field whose name embeds the vocabulary - including this module's own docstring, which
    cites `test_this_lane_never_says_a_segment_is_cleared` by name. Measured over the 104 scanned files, that
    docstring is the *only* line the loose boundary adds, which is why prose keeps the strict rule.
    """
    p = tmp_path / "prose.py"
    p.write_text(
        '"""S10 turned ``test_this_lane_never_says_a_segment_is_clear' + 'ed`` red at 23:57."""\n',
        encoding="utf-8",
    )
    assert scan_source(p, tmp_path) == [], [str(v) for v in scan_source(p, tmp_path)]


def test_the_scanner_is_clean_on_its_own_source():
    """The pattern table has to be writable. If the scanner cannot pass its own check it cannot be maintained."""
    own = scan_source(Path(G.__file__), REPO)
    assert own == [], "\n".join(str(v) for v in own)


# ==============================================================================================================
# 4. The allowances are load-bearing, not decorative
# ==============================================================================================================
def test_dropping_an_allowance_turns_its_site_red(monkeypatch):
    """If removing an entry changes nothing, the entry was never doing any work."""
    target = next(a for a in ALLOWANCES if a.path == "sightline/track/tracker.py" and "_state.pop" in a.snippet)
    monkeypatch.setattr(G, "ALLOWANCES", tuple(a for a in ALLOWANCES if a is not target))
    found = scan_lane_sources(REPO, dirs=("sightline/track",))
    assert len(found) == 1 and target.snippet in " ".join(found[0].line.split()), [str(v) for v in found]


def test_an_allowance_does_not_cover_a_different_deletion_in_the_same_file(tmp_path: Path):
    """Per-site, not per-file. A blessed `.pop()` must not smuggle in a `.remove()` two lines below it."""
    d = tmp_path / "sightline" / "track"
    d.mkdir(parents=True)
    (d / "tracker.py").write_text(
        "def _prune(self, backend_id, rec):\n"
        "    st = self._state.pop(backend_id)\n"   # the reviewed line, allowed
        "    self.records.remove(rec)\n",          # brand new, must fail
        encoding="utf-8",
    )
    found = scan_lane_sources(tmp_path, dirs=("sightline/track",))
    assert len(found) == 1 and found[0].kind == "container removal", [str(v) for v in found]


def test_a_second_copy_of_an_allowed_line_is_reported_as_over_broad(tmp_path: Path):
    """`sites` is the count that stops an allowance turning into a blanket suppression by accident."""
    d = tmp_path / "sightline" / "track"
    d.mkdir(parents=True)
    (d / "tracker.py").write_text(
        "def _prune(self, backend_id):\n"
        "    st = self._state.pop(backend_id)\n"
        "    st = self._state.pop(backend_id)\n",  # a second, unreviewed prune wearing the same clothes
        encoding="utf-8",
    )
    problems = audit_allowances(tmp_path, dirs=("sightline/track",))
    assert any("over-broad" in p and "_state.pop" in p for p in problems), problems


def test_a_stale_allowance_is_reported(monkeypatch):
    """An entry whose code has been rewritten has not been audited against what is there now."""
    bogus = Allowance("sightline/track/tracker.py", "container pop", "self._ghosts.pop(nothing)", 1,
                      "deliberately fictional, planted by tests/test_guardrails_scan.py", "nowhere")
    monkeypatch.setattr(G, "ALLOWANCES", ALLOWANCES + (bogus,))
    problems = audit_allowances(REPO, dirs=("sightline/track",))
    assert any("stale allowance" in p and "_ghosts" in p for p in problems), problems


def test_a_subset_scan_does_not_audit_allowances_it_never_looked_at():
    """`tests/test_export.py` scans only `sightline/export`; the other 22 entries must not read as stale."""
    assert audit_allowances(REPO, dirs=("sightline/export",)) == []
    assert audit_allowances(REPO, dirs=TRIAGE_LANE_DIRS) == []


# ==============================================================================================================
# 5. The eight false positives that started this - and proof the check still fires on the real prose
# ==============================================================================================================
#: The lines AUDIT.md S10 named, anchored by content rather than by line number. Each denies clearing; each was
#: being reported as a violation before this fix.
DENYING_PROSE: tuple[tuple[str, str], ...] = (
    ("sightline/api/coverage_feed.py", "by an overflight, only ever marked unclearable"),
    ("sightline/api/coverage_feed.py", "R10: nothing here can mark a cell"),
    ("sightline/coverage/__init__.py", "Nothing in this package has a"),
    ("sightline/coverage/accumulate.py", "There is **no "),
    ("sightline/coverage/accumulate.py", "figure \u2014 a POD statistic"),
    ("sightline/coverage/export.py", "Guardrail R10: nothing exported here says"),
    ("sightline/dedup/cluster.py", "There is no delete and no "),
    ("sightline/eval/calibration.py", "the map shows POD, never "),
)


def test_the_prose_that_broke_the_api_test_is_no_longer_flagged(raw_hits):
    """S10: eight lines promising R10 were being reported as breaking R10. None of them may appear."""
    flagged = {(v.rel, v.line_no) for v in raw_hits}
    for rel, needle in DENYING_PROSE:
        line_no = _line_with(rel, needle)
        assert (rel, line_no) not in flagged, f"{rel}:{line_no} is a denial and must not be a violation"


def test_the_wider_wording_vocabulary_also_leaves_the_denials_alone(raw_hits):
    """These four are only visible to the widened vocabulary; all four deny, so all four must stay quiet."""
    flagged = {(v.rel, v.line_no) for v in raw_hits}
    for rel, needle in (
        ("sightline/api/coverage_feed.py", '"legend_note": "POD is a probability'),
        ("sightline/coverage/export.py", '"legend_note": ("POD is a probability'),
        ("sightline/mission/takeover.py", "R10 note: nothing in this module deletes"),
        ("sightline/plan/segments.py", "says a segment is "),
    ):
        assert (rel, _line_with(rel, needle)) not in flagged, f"{rel}: denial flagged"


def test_removing_the_denial_from_that_real_prose_makes_the_check_fire(tmp_path: Path):
    """The mutation that proves the fix is a fix and not a mute button.

    Takes the actual text of `sightline/coverage/__init__.py`, deletes the one word that makes it a denial, and
    insists the wording pass then reports it. If this ever goes green the check has stopped reading sentences.
    """
    src = (REPO / "sightline" / "coverage" / "__init__.py").read_text(encoding="utf-8")
    assert scan_source(REPO / "sightline" / "coverage" / "__init__.py", REPO) == []

    mutated = src.replace("Nothing in this package has a", "This package has a", 1)
    assert mutated != src, "the anchor sentence moved; re-read sightline/coverage/__init__.py"
    p = tmp_path / "mutant.py"
    p.write_text(mutated, encoding="utf-8")
    found = scan_source(p, tmp_path)
    assert any("prose" in v.kind for v in found), f"the denial was load-bearing but nothing fired: {found}"


# ==============================================================================================================
# 6. It still works as a gate, and the old narrow contract is intact
# ==============================================================================================================
def test_the_original_two_lane_subset_still_scans_clean():
    """`tests/test_export.py` and the B4 lane depend on this exact call still meaning what it did."""
    assert scan_lane_sources(REPO, dirs=("sightline/export",)) == []
    assert scan_lane_sources(REPO, dirs=TRIAGE_LANE_DIRS) == []


def test_scan_source_never_applies_allowances(tmp_path: Path):
    """`tests/test_triage.py` and `tests/test_live_mission.py` plant probes and call this directly.

    If allowances leaked into the single-file scan, a probe that happened to sit at an allowed path would come
    back empty and those two tests would pass for the wrong reason.
    """
    d = tmp_path / "sightline" / "track"
    d.mkdir(parents=True)
    p = d / "tracker.py"
    p.write_text("def go(self, backend_id):\n    st = self._state.pop(backend_id)\n", encoding="utf-8")
    assert scan_source(p, tmp_path), "scan_source filtered an allowance; it must return the raw list"
    assert scan_lane_sources(tmp_path, dirs=("sightline/track",)) == []


def test_assert_no_record_deletion_raises_and_names_the_file(tmp_path: Path):
    (tmp_path / "sightline" / "store").mkdir(parents=True)
    (tmp_path / "sightline" / "store" / "db.py").write_text(PLANT, encoding="utf-8")
    with pytest.raises(GuardrailError, match="R10 violated"):
        assert_no_record_deletion(tmp_path, dirs=("sightline/store",))
    assert_no_record_deletion(REPO)  # the real tree must not raise


def test_the_cli_is_a_gate(tmp_path: Path, capsys):
    """Exit 0 clean, 1 violated, 0 on `--self-test`. This is what CI would call."""
    assert G.main([]) == 0
    assert G.main(["--self-test"]) == 0

    (tmp_path / "sightline" / "eval").mkdir(parents=True)
    (tmp_path / "sightline" / "eval" / "bad.py").write_text(PLANT, encoding="utf-8")
    capsys.readouterr()
    assert G.main(["--root", str(tmp_path), "--dirs", "sightline/eval"]) == 1
    assert "VIOLATION" in capsys.readouterr().err
