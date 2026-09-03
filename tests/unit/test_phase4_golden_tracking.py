#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_golden_tracking.py - Phase 4 correction pass items 5 and 6:
fail-CLOSED coverage for the committed golden dataset, and proof that
every required file is stageable by git without ``-f``, in a way that
stays valid BOTH before and after the Phase 4 golden tree is committed.

Item 5 (fail-open skip removal): :func:`test_phase4_golden_tree_is_complete`
directly exercises the same check every Phase 4 parity module now runs at
collection time (``require_golden_tree()``/``missing_golden_files()``), so a
broken checkout gets a SECOND, independent, loud failure naming the exact
missing file(s) here too, rather than relying solely on collection-time
side effects.

Item 6 (git trackability) — STATE-INDEPENDENT CONTRACT (correction pass 3):
the previous version of this module asserted two things that are only true
while the Phase 4 golden tree is untracked:

- ``git check-ignore`` (without ``--no-index``) consults the index, so a
  path that is ALREADY TRACKED is reported "not ignored" (exit 1) even when
  a gitignore rule would otherwise exclude it. That happens to read as
  "pass" for an already-committed, correctly-tracked file, but it also masks
  a real defect: it cannot tell a genuinely-not-ignored path apart from a
  tracked-but-still-pattern-matched one. ``--no-index`` evaluates gitignore
  patterns against the path alone, ignoring the index entirely, so it is the
  only form that is meaningful regardless of tracked state. Reproduced
  directly in :func:`test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file`
  and :func:`test_git_check_ignore_no_index_reports_the_real_pattern_match_for_a_tracked_ignored_file`
  below, against a disposable temp repo — never the real project index.
- ``git add --dry-run`` prints an ``add '<path>'`` line only for a path git
  would actually touch. A path that is ALREADY TRACKED AND UNCHANGED has
  nothing to add, so git prints NOTHING for it and still exits 0 — "every
  required file appears in dry-run stdout" silently starts failing the
  moment the golden tree is committed and clean. Reproduced directly in
  :func:`test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file`
  below.

The real, state-independent contract every required Phase 4 golden file
must satisfy, implemented by
:func:`test_phase4_golden_files_are_trackable_without_dash_f`:

1. NOT ignored per ``git check-ignore --no-index`` (meaningful regardless of
   tracked state);
2. EITHER already tracked (``git ls-files --error-unmatch`` succeeds) OR, if
   untracked, included by a normal, non-forced ``git add --dry-run``.

Neither branch of (2) ever needs ``-f``: an already-tracked file needs no
``add`` at all, and an untracked-but-not-ignored file is addable without
force by definition.

None of this needs the live MSVC/CMake/Ninja toolchain - it is pure
filesystem/git-plumbing inspection - so this module is CORE, like the
other golden-CSV-driven Phase 4 modules. Every git operation against the
REAL project repository is read-only/non-mutating (``--dry-run``,
``check-ignore``, ``ls-files``) and bounded via
:func:`~tests.cpp_parity_live._proc.run_bounded`. The two tests that need a
TRACKED file to probe tracked-state behaviour build and commit into their
own disposable temp repository (a pytest ``tmp_path``), never the real
project index - the real repository is never staged, committed, or
modified by this module.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import pathlib

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._phase4_contract import (
    PHASE4_MODES,
    _required_golden_files,
    golden_dir,
    missing_golden_files,
)
from tests.cpp_parity_live._proc import BoundedResult, run_bounded

#: Real, present-on-disk PHASE4_MODES ``golden_dir()``s. Read once at
#: collection time so a whole-tree traversal is not repeated per test.
_GOLDEN_TREE_ROOT = os.path.dirname(golden_dir(PHASE4_MODES[0]))

#: Bound for every ``git`` subprocess this module spawns - plumbing calls on
#: a small, already-on-disk file set, so a generous bound catches a truly
#: hung process without flaking on normal load.
_GIT_TIMEOUT_S = 30.0


def _all_required_files() -> list:
    """
    Return every file :func:`~tests.cpp_parity_live._phase4_contract.
    _required_golden_files` names across all six Phase 4 modes.

    :returns: Absolute paths, one list entry per required file.
    """
    files = []
    for mode in PHASE4_MODES:
        files.extend(_required_golden_files(mode))
    return files


def _all_tree_files() -> list:
    """
    Return every file that actually exists under the Phase 4 golden tree
    root, walked directly from disk.

    :returns: Absolute paths, sorted.
    """
    found = []
    for root, _dirs, files in os.walk(_GOLDEN_TREE_ROOT):
        for name in files:
            found.append(os.path.join(root, name))
    return sorted(found)


def _check_ignore_no_index(paths: list, cwd: str = PROJECT_ROOT) -> set:
    """
    Run ``git check-ignore --no-index`` against *paths* and return the
    subset it reports as still ignored, FAILING CLOSED on any return code
    other than git's own two documented outcomes for this subcommand.

    ``git check-ignore`` documents exactly two meaningful return codes: 0
    when one or more of the given paths ARE ignored, and 1 when NONE of
    them are. Any other code (2 or higher) means the git invocation itself
    failed operationally (bad arguments, corrupted repository, environment
    problem, etc.) - stdout being empty in that case must never be read as
    "nothing is ignored", since an operational failure produces the exact
    same empty stdout a genuine "nothing ignored" result would. Conflating
    the two would let a broken ``git`` invocation silently pass this check.

    :param paths: Repo-relative paths to check, forward-slash separated.
    :param cwd: Working directory the ``git`` subprocess runs in.
    :returns: The subset of *paths* git reports as still ignored (only
        possible when the return code is 0; always empty when it is 1).
    :raises AssertionError: If the return code is anything other than 0
        or 1, including both stdout and stderr in the message.
    """
    result = _git("check-ignore", "--no-index", "--", *paths, cwd=cwd)
    assert result.returncode in (0, 1), (
        "git check-ignore --no-index failed operationally "
        f"(rc={result.returncode}, expected 0 or 1); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    if result.returncode == 1:
        # Git's own documented behaviour: rc=1 means NONE of the given
        # paths are ignored, so there is nothing to parse from stdout.
        return set()
    return set(result.stdout.splitlines())


def _git(*args: str, cwd: str = PROJECT_ROOT) -> BoundedResult:
    """
    Run a bounded ``git`` subprocess.

    :param args: ``git`` subcommand and arguments.
    :param cwd: Working directory for the subprocess - defaults to the
        real project root, but the tracked-state probe tests below pass a
        disposable temp-repo path instead so the real project index is
        never touched.
    :returns: The :class:`~tests.cpp_parity_live._proc.BoundedResult`.
    """
    return run_bounded(["git", *args], timeout=_GIT_TIMEOUT_S, cwd=cwd)


def _init_temp_repo(root: pathlib.Path, gitignore_body: str) -> None:
    """
    Initialize a disposable git repository at *root* with a committed
    ``.gitignore``.

    Used only by the tracked-state probe tests, so they can prove real git
    behaviour on a TRACKED file without ever touching the actual project
    repository's index.

    :param root: Directory to initialize (a pytest ``tmp_path``).
    :param gitignore_body: Literal ``.gitignore`` contents to commit.
    :returns: None.
    :raises AssertionError: If any setup step fails.
    """
    result = _git("init", "-q", cwd=str(root))
    assert result.returncode == 0, result.stderr
    (root / ".gitignore").write_text(gitignore_body, encoding="utf-8")
    result = _git("add", ".gitignore", cwd=str(root))
    assert result.returncode == 0, result.stderr
    result = _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "commit", "-q", "-m", "init", cwd=str(root),
    )
    assert result.returncode == 0, result.stderr


def test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file(
        tmp_path: pathlib.Path,
):
    """
    Ground-truth reproduction (disposable temp repo, real project repo
    untouched): once a file matching an active ``.gitignore`` rule is
    force-tracked and left unchanged, ``git add --dry-run`` on it exits 0
    but prints NOTHING - there is nothing to add. This is exactly why
    "every required file appears in ``git add --dry-run`` stdout" is not a
    valid universal check once the Phase 4 golden tree is committed.

    :param tmp_path: Pytest-provided disposable directory.
    :returns: None.
    """
    _init_temp_repo(tmp_path, "*.csv\n")
    tracked = tmp_path / "blocked.csv"
    tracked.write_text("data\n", encoding="utf-8")
    result = _git("add", "-f", "blocked.csv", cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    result = _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "commit", "-q", "-m", "track", cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    result = _git("add", "--dry-run", "--", "blocked.csv", cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        "expected git add --dry-run to print nothing for a tracked, "
        f"unchanged file, got: {result.stdout!r}"
    )


def test_git_check_ignore_no_index_reports_the_real_pattern_match_for_a_tracked_ignored_file(
        tmp_path: pathlib.Path,
):
    """
    Ground-truth reproduction (disposable temp repo, real project repo
    untouched): ``git check-ignore --no-index`` on a force-tracked file
    that still matches an active ``.gitignore`` rule correctly reports the
    real pattern match (exit 0, path on stdout) regardless of the file's
    tracked status - the form :func:`test_phase4_golden_files_are_not_
    gitignored` actually relies on.

    :param tmp_path: Pytest-provided disposable directory.
    :returns: None.
    """
    _init_temp_repo(tmp_path, "*.csv\n")
    tracked = tmp_path / "blocked.csv"
    tracked.write_text("data\n", encoding="utf-8")
    result = _git("add", "-f", "blocked.csv", cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    result = _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "commit", "-q", "-m", "track", cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    result = _git(
        "check-ignore", "--no-index", "--", "blocked.csv", cwd=str(tmp_path),
    )
    assert result.returncode == 0, (
        "expected git check-ignore --no-index to report the tracked file "
        f"as still pattern-matched (exit 0); got rc={result.returncode}"
    )
    assert result.stdout.strip() == "blocked.csv"


def test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file(
        tmp_path: pathlib.Path,
):
    """
    Ground-truth reproduction (disposable temp repo, real project repo
    untouched): plain ``git check-ignore`` (no ``--no-index``) consults the
    index, so it reports a force-tracked file as NOT ignored (exit 1, no
    stdout) even though the exact same ``.gitignore`` rule would exclude
    that path if it were untracked. This is the concrete evidence for why
    :func:`test_phase4_golden_files_are_not_gitignored` uses ``--no-index``
    rather than the plain form the earlier version of this module used.

    :param tmp_path: Pytest-provided disposable directory.
    :returns: None.
    """
    _init_temp_repo(tmp_path, "*.csv\n")
    tracked = tmp_path / "blocked.csv"
    tracked.write_text("data\n", encoding="utf-8")
    result = _git("add", "-f", "blocked.csv", cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    result = _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "commit", "-q", "-m", "track", cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    result = _git("check-ignore", "--", "blocked.csv", cwd=str(tmp_path))
    assert result.returncode == 1, (
        "expected plain git check-ignore (no --no-index) to report the "
        f"tracked file as NOT ignored (exit 1); got rc={result.returncode}, "
        f"stdout={result.stdout!r}"
    )
    assert result.stdout.strip() == ""


def test_phase4_golden_files_are_not_gitignored():
    """
    Item 6(b): every required Phase 4 golden file resolves as NOT ignored
    by git - ``git check-ignore --no-index``, a real subprocess, never a
    re-implementation of gitignore's pattern-matching rules.

    ``--no-index`` is used deliberately rather than the plain form: plain
    ``git check-ignore`` reports a path as "not ignored" whenever it is
    already tracked, regardless of whether a gitignore rule still matches
    it (reproduced directly in
    :func:`test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file`),
    so it cannot distinguish "genuinely not excluded" from "excluded but
    saved by being tracked". ``--no-index`` evaluates the patterns against
    the path alone and is meaningful whether the Phase 4 golden tree is
    currently tracked or still untracked in this checkout.

    The return-code handling itself (fail closed on anything but git's own
    documented 0/1 outcomes for this subcommand, rather than silently
    trusting empty stdout) lives in :func:`_check_ignore_no_index` and is
    exercised in isolation by
    :func:`test_phase4_golden_files_are_not_gitignored_fails_closed_on_git_operational_failure`.

    :returns: None.
    """
    required = [
        os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        for path in _all_required_files()
    ]
    still_ignored = _check_ignore_no_index(required)
    assert not still_ignored, (
        f"still ignored by git (git check-ignore --no-index): "
        f"{sorted(still_ignored)}"
    )


def test_phase4_golden_files_are_not_gitignored_fails_closed_on_git_operational_failure(
        tmp_path: pathlib.Path,
):
    """
    Regression for the fail-open gap this correction pass fixes: BEFORE
    the fix, :func:`test_phase4_golden_files_are_not_gitignored` read
    ``result.stdout.splitlines()`` with no return-code check at all, so a
    ``git check-ignore --no-index`` invocation that failed OPERATIONALLY
    (git itself erroring out, e.g. rc=128) - which also produces empty
    stdout - was indistinguishable from "genuinely nothing is ignored" and
    would have PASSED incorrectly. AFTER the fix, :func:`_check_ignore_no_
    index` asserts the return code is exactly 0 or 1 before trusting
    stdout at all.

    This test triggers a REAL git operational failure - no mocking of the
    subprocess call - by pointing ``_check_ignore_no_index`` at *tmp_path*,
    a pytest-provided directory that (per pytest's own base-temp
    convention) sits outside any git repository, real or disposable. Any
    ``git`` subcommand invoked with that as its working directory exits
    128 with ``fatal: not a git repository (or any of the parent
    directories): .git`` on stderr and nothing on stdout - confirmed
    directly against the real ``git`` binary before writing this test.

    Without the fix, this exact call would have silently returned an
    empty set (read as "nothing ignored", i.e. a false pass) instead of
    raising. With the fix, it raises ``AssertionError`` naming the bad
    return code, which this test asserts on directly.

    :param tmp_path: Pytest-provided disposable directory, guaranteed to
        be outside any git repository.
    :returns: None.
    """
    outside_repo_check = _git(
        "check-ignore", "--no-index", "--", "anything.csv", cwd=str(tmp_path),
    )
    assert outside_repo_check.returncode not in (0, 1), (
        "test precondition failed: expected tmp_path to be outside any "
        f"git repository (rc should be 128, not 0/1), got "
        f"rc={outside_repo_check.returncode} stdout={outside_repo_check.stdout!r} "
        f"stderr={outside_repo_check.stderr!r} - tmp_path may itself be "
        "inside a git repository in this environment, invalidating the "
        "injected-failure scenario"
    )
    assert outside_repo_check.stdout == "", (
        "test precondition failed: expected empty stdout from the "
        "operational failure (the exact condition that made the pre-fix "
        f"code read it as a false pass); got {outside_repo_check.stdout!r}"
    )

    try:
        _check_ignore_no_index(["anything.csv"], cwd=str(tmp_path))
    except AssertionError as exc:
        assert "operationally" in str(exc)
        assert str(outside_repo_check.returncode) in str(exc)
    else:
        raise AssertionError(
            "_check_ignore_no_index did not fail closed on a real git "
            "operational failure (rc="
            f"{outside_repo_check.returncode}, empty stdout) - this is "
            "exactly the fail-open gap this regression test exists to "
            "catch"
        )


def test_phase4_golden_files_are_trackable_without_dash_f():
    """
    Item 6(d), rewritten as a STATE-INDEPENDENT contract (correction pass
    3): every required Phase 4 golden file is committable by a normal,
    non-forced ``git add`` regardless of whether this checkout currently
    has the Phase 4 golden tree tracked or still untracked.

    For each required file:

    - it must not be ignored (:func:`test_phase4_golden_files_are_not_
      gitignored` covers this independently and this test does not repeat
      it, but relies on it being true);
    - EITHER it is already tracked (``git ls-files --error-unmatch``
      succeeds - nothing further to add, ``-f`` is moot), OR it is
      untracked and a real ``git add --dry-run`` would stage it (an
      ``add '...'`` stdout line, no "ignored" warning on stderr).

    The prior version of this test asserted ONLY the untracked branch
    (every file must appear in ``git add --dry-run`` stdout), which is
    exactly why it broke once files were committed: a tracked, unchanged
    file prints nothing under ``--dry-run`` (reproduced in
    :func:`test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file`)
    even though it needs no add at all.

    :returns: None.
    """
    required_abs = _all_required_files()
    required_rel = [
        os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        for path in required_abs
    ]

    tracked_result = _git("ls-files", "-z", "--", *required_rel)
    assert tracked_result.returncode == 0, tracked_result.stderr
    tracked = {p for p in tracked_result.stdout.split("\0") if p}

    untracked_rel = [p for p in required_rel if p not in tracked]
    staged_by_dry_run: set = set()
    if untracked_rel:
        dry_run = _git("add", "--dry-run", "--", *untracked_rel)
        assert dry_run.returncode == 0, (
            f"git add --dry-run failed (rc={dry_run.returncode}): "
            f"{dry_run.stderr}"
        )
        assert "ignored" not in dry_run.stderr.lower(), (
            "git add --dry-run reported (an) ignored path(s), so a real "
            f"'git add' would silently omit them without -f: "
            f"{dry_run.stderr}"
        )
        for line in dry_run.stdout.splitlines():
            line = line.strip()
            if line.startswith("add '") and line.endswith("'"):
                staged_by_dry_run.add(line[len("add '"):-1])

    uncovered = [
        p for p in required_rel
        if p not in tracked and p not in staged_by_dry_run
    ]
    assert not uncovered, (
        "these required Phase 4 golden files are neither already tracked "
        f"nor stageable by a real 'git add --dry-run' without -f: "
        f"{sorted(uncovered)}"
    )


def test_phase4_golden_tree_has_no_stale_or_extra_files():
    """
    Item 6(c): the Phase 4 golden tree on disk contains exactly the
    required files - nothing stale or extra.

    A stray file left over from an interrupted or manual regeneration
    would otherwise go unnoticed by :func:`test_phase4_golden_tree_is_
    complete` (which only checks REQUIRED files are present, not that
    nothing else exists) and, once tracked, would sit in the repository
    forever.

    :returns: None.
    """
    required = {os.path.normpath(p) for p in _all_required_files()}
    on_disk = {os.path.normpath(p) for p in _all_tree_files()}
    extra = on_disk - required
    assert not extra, f"unexpected file(s) in the Phase 4 golden tree: {sorted(extra)}"


def test_phase4_golden_tree_is_complete():
    """
    Item 5: fail CLOSED with the exact missing-file list if the committed
    Phase 4 golden dataset is incomplete.

    Duplicates the check every Phase 4 parity module already runs at
    collection time (via ``require_golden_tree()``) as an independent,
    directly-runnable assertion, so this module alone proves the dataset
    is intact even if a future change altered how the parity modules wire
    their own collection-time guard.

    :returns: None.
    """
    missing = missing_golden_files()
    assert not missing, (
        "the committed Phase 4 golden dataset is incomplete - missing or "
        f"empty: {missing}"
    )
