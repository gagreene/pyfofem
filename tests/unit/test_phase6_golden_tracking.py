#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase6_golden_tracking.py - Phase 6 counterpart to
``test_phase4_golden_tracking.py``'s completeness and git-trackability
coverage, applied to the ``default-emissions-equivalence`` golden tree
(``tests/test_data/test_golden_output/phase6/``).

This module deliberately does NOT re-derive or re-prove the underlying git
behaviour quirks (``git check-ignore``'s plain-vs-``--no-index`` tracked-file
blind spot, ``git add --dry-run``'s silence for an already-tracked-and-
unchanged file) - those are properties of ``git`` itself, not of the Phase 6
dataset, and are already proven once, directly and reproducibly, in
``test_phase4_golden_tracking.py``'s
``test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file`` /
``test_git_check_ignore_no_index_reports_the_real_pattern_match_for_a_tracked_ignored_file`` /
``test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file``. This
module applies the CORRECT, already-proven pattern (``--no-index`` for
ignore-checking; tracked-OR-dry-run-stageable for the "committable without
``-f``" contract) directly to the Phase 6 tree, the same state-independent
contract (valid whether or not the tree is yet committed) Phase 4's module
established.

None of this needs the live MSVC/CMake/Ninja toolchain - it is pure
filesystem/git-plumbing inspection - so this module is CORE, like the other
golden-CSV-driven Phase 4/5/6 modules. Every git operation is read-only/non-
mutating (``--dry-run``, ``check-ignore``, ``ls-files``) against the real
project repository and bounded via
:func:`~tests.cpp_parity_live._proc.run_bounded`.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import git_safe_directory_value
from tests.cpp_parity_live._phase6_contract import (
    _required_golden_files,
    golden_dir,
    missing_golden_files,
    PHASE6_MODES,
)
from tests.cpp_parity_live._proc import BoundedResult, run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

#: Real, present-on-disk PHASE6_MODES ``golden_dir()``s. Read once at
#: collection time so a whole-tree traversal is not repeated per test.
_GOLDEN_TREE_ROOT = os.path.dirname(golden_dir(PHASE6_MODES[0]))

#: Bound for every ``git`` subprocess this module spawns - plumbing calls on
#: a small, already-on-disk file set, so a generous bound catches a truly
#: hung process without flaking on normal load.
_GIT_TIMEOUT_S = 30.0


def _all_required_files() -> list:
    """
    Return every file :func:`~tests.cpp_parity_live._phase6_contract.
    _required_golden_files` names across all Phase 6 modes.

    :returns: Absolute paths, one list entry per required file.
    """
    files = []
    for mode in PHASE6_MODES:
        files.extend(_required_golden_files(mode))
    return files


def _all_tree_files() -> list:
    """
    Return every file that actually exists under the Phase 6 golden tree
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
    subset it reports as still ignored, failing closed on any return code
    other than git's own two documented outcomes for this subcommand (see
    ``test_phase4_golden_tracking.py``'s identical helper for the full
    rationale).

    :param paths: Repo-relative paths to check, forward-slash separated.
    :param cwd: Working directory the ``git`` subprocess runs in.
    :returns: The subset of *paths* git reports as still ignored.
    :raises AssertionError: If the return code is anything other than 0
        or 1.
    """
    result = _git("check-ignore", "--no-index", "--", *paths, cwd=cwd)
    assert result.returncode in (0, 1), (
        "git check-ignore --no-index failed operationally "
        f"(rc={result.returncode}, expected 0 or 1); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    if result.returncode == 1:
        return set()
    return set(result.stdout.splitlines())


def _git(*args: str, cwd: str = PROJECT_ROOT) -> BoundedResult:
    """
    Run a bounded ``git`` subprocess, qualified with a per-command
    ``-c safe.directory=<forward-slash form of cwd>`` override so it
    succeeds regardless of the running account's global Git
    configuration or the repository's file ownership - never written to
    any config file.

    **Final acceptance-gate correction (2026-09-09)**: reproduced
    directly under forced ownership mistrust
    (``GIT_TEST_ASSUME_DIFFERENT_OWNER=1``) that calls against the real
    project repository failed with ``fatal: detected dubious ownership``
    before this fix. The trusted path is *cwd* itself, never an
    unconditional :data:`~tests._support.PROJECT_ROOT`, so this helper
    stays correct if a future test passes a non-default ``cwd`` too -
    reuses the established
    :func:`~tests.cpp_parity_live._golden_manifest.git_safe_directory_value`
    helper rather than duplicating its forward-slash normalization logic.

    :param args: ``git`` subcommand and arguments.
    :param cwd: Working directory for the subprocess.
    :returns: The :class:`~tests.cpp_parity_live._proc.BoundedResult`.
    """
    safe_directory = git_safe_directory_value(cwd)
    return run_bounded(
        ["git", "-c", f"safe.directory={safe_directory}", *args],
        timeout=_GIT_TIMEOUT_S, cwd=cwd,
    )


def test_git_helper_succeeds_against_the_real_project_under_forced_ownership_mistrust(
        monkeypatch,
):
    """
    ``_git()`` must succeed against the REAL project repository (a
    read-only ``rev-parse``) even when Git's dubious-ownership check is
    forced via ``GIT_TEST_ASSUME_DIFFERENT_OWNER=1`` with blank,
    repository-local global/system config - the exact Phase 7
    acceptance-gate failure independent review reported, reproduced
    directly before this fix (``fatal: detected dubious ownership in
    repository at '<PROJECT_ROOT>'``).

    :param monkeypatch: Pytest fixture; used only to set 3 environment
        variables for this test's own subprocess calls, auto-restored on
        teardown.
    :returns: None.
    """
    with scratch_tempdir("phase6_golden_tracking_ownership") as base:
        blank_global = os.path.join(base, "blank_global.gitconfig")
        blank_system = os.path.join(base, "blank_system.gitconfig")
        open(blank_global, "w", encoding="utf-8").close()
        open(blank_system, "w", encoding="utf-8").close()
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", blank_global)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", blank_system)

        result = _git("rev-parse", "HEAD")
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.strip()) == 40


def test_phase6_golden_files_are_not_gitignored():
    """
    Every required Phase 6 golden file resolves as NOT ignored by git
    (``git check-ignore --no-index``, a real subprocess).

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


def test_phase6_golden_files_are_trackable_without_dash_f():
    """
    Every required Phase 6 golden file is committable by a normal,
    non-forced ``git add``, whether this checkout currently has the
    Phase 6 golden tree tracked or still untracked: EITHER it is already
    tracked (``git ls-files`` lists it, nothing further to add), OR it is
    untracked and a real ``git add --dry-run`` would stage it.

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
        "these required Phase 6 golden files are neither already tracked "
        f"nor stageable by a real 'git add --dry-run' without -f: "
        f"{sorted(uncovered)}"
    )


def test_phase6_golden_tree_has_no_stale_or_extra_files():
    """
    The Phase 6 golden tree on disk contains exactly the required files -
    nothing stale or extra (e.g. a leftover side file that should have
    been cleaned up by the generator).

    :returns: None.
    """
    required = {os.path.normpath(p) for p in _all_required_files()}
    on_disk = {os.path.normpath(p) for p in _all_tree_files()}
    extra = on_disk - required
    assert not extra, f"unexpected file(s) in the Phase 6 golden tree: {sorted(extra)}"


def test_phase6_golden_tree_is_complete():
    """
    Fail CLOSED with the exact missing-file list if the committed Phase 6
    golden dataset is incomplete.

    :returns: None.
    """
    missing = missing_golden_files()
    assert not missing, (
        "the committed Phase 6 golden dataset is incomplete - missing or "
        f"empty: {missing}"
    )
