#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase7_golden_tracking.py - Phase 7 counterpart to
``test_phase4_golden_tracking.py``'s completeness and git-trackability
coverage, applied to the item-E additional-``run_burnup`` golden tree
(``tests/test_data/test_golden_output/phase7/``).

This module deliberately does NOT re-derive or re-prove the underlying git
behaviour quirks (``git check-ignore``'s plain-vs-``--no-index`` tracked-file
blind spot, ``git add --dry-run``'s silence for an already-tracked-and-
unchanged file) - those are properties of ``git`` itself, not of the Phase 7
dataset, and are already proven once, directly and reproducibly, in
``test_phase4_golden_tracking.py``'s
``test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file`` /
``test_git_check_ignore_no_index_reports_the_real_pattern_match_for_a_tracked_ignored_file`` /
``test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file``. This
module applies the CORRECT, already-proven pattern (``--no-index`` for
ignore-checking; tracked-OR-dry-run-stageable for the "committable without
``-f``" contract) directly to the Phase 7 tree, the same state-independent
contract (valid whether or not the tree is yet committed) Phase 4's module
established.

None of this needs the live MSVC/CMake/Ninja toolchain - it is pure
filesystem/git-plumbing inspection - so this module is CORE, like the other
golden-CSV-driven Phase 4/5/6/7 modules. Every git operation is read-only/
non-mutating (``--dry-run``, ``check-ignore``, ``ls-files``) against the
real project repository and bounded via
:func:`~tests.cpp_parity_live._proc.run_bounded`.

**Environment-independence (Phase 7 correction pass, 2026-09-06, item
5).** An independent review found this module's ``git add --dry-run``
call attempted to create the real repository's ``.git/index.lock`` -
which fails in an execution environment that lacks write access to the
real project index (e.g. a reviewer account without ownership of the
checkout) - and that every ``git`` call here could also fail closed on
git's own "dubious ownership" safety check when the running account
does not own the repository, unless the environment already has a
matching ``safe.directory`` entry. Fixed WITHOUT modifying any global
Git configuration and WITHOUT adding this repository to it:

- Every ``git`` invocation now carries a PER-COMMAND
  ``-c safe.directory=<forward-slash-normalized PROJECT_ROOT>``
  override (see :func:`_git` and
  ``_golden_manifest.git_safe_directory_value``) - this is
  process-local config for that one invocation only, never a write to
  any config file, so it works whether or not the running account's
  global config already trusts this checkout. **Corrected in the
  2026-09-07 correction pass**: the value was originally the raw
  ``PROJECT_ROOT`` string (a Windows BACKSLASH path), which git does
  not recognize as matching the checkout for ``safe.directory``
  purposes - the override was silently ineffective. Now
  forward-slash-normalized.
- ``git add --dry-run`` now runs with ``GIT_INDEX_FILE`` pointed at a
  disposable, repository-local scratch file (under
  ``tests/cpp_parity_live/_scratch.py``'s scratch root, never the
  system/user temp directory) instead of the real ``.git/index`` - so it
  can never attempt to create the real ``.git/index.lock``, while still
  running against the REAL working tree (``cwd=PROJECT_ROOT``) and
  therefore the REAL ``.gitignore``/``.gitattributes`` content, which is
  unaffected by which index file git happens to be updating. Directly
  confirmed (this module's own probe, and see
  :func:`test_git_add_dry_run_with_a_disposable_index_never_touches_the_real_git_index_lock`)
  that pointing ``GIT_INDEX_FILE`` at a nonexistent scratch path makes
  ``git add --dry-run`` report ``add '<path>'`` for a real, non-ignored
  file and the real "ignored by one of your .gitignore files" message
  for a real, ignored one, with zero footprint under the real ``.git/``.
- :func:`test_phase7_golden_tracking_never_touches_the_real_git_index`
  captures the real ``.git/index``'s own content hash before and after
  this module's full test run and asserts it is byte-for-byte unchanged.
- Dedicated, environment-independent git-behavior proofs
  (:func:`test_git_ignore_detection_works_in_a_disposable_process_owned_repo`,
  :func:`test_dry_run_add_reports_ignored_status_in_a_disposable_process_owned_repo`)
  exercise the same ignore-detection/dry-run-add mechanisms this
  module's real-repo tests depend on inside a throwaway, this-process-
  owned Git repository created under the Phase 7 scratch root - so they
  pass regardless of the real repository's ownership or any global
  ``safe.directory`` configuration, following the same environment-
  independence precedent Phase 6's ``massman_fof_dll_probe.py`` probe-
  contract tests established.

None of this weakens what is actually being tested: the two tests that
check the REAL ``.gitignore``/``.gitattributes`` content
(:func:`test_phase7_golden_files_are_not_gitignored` and
:func:`test_phase7_golden_files_are_trackable_without_dash_f`) still run
against the real working tree and its real ignore rules - only the
INDEX (a book-keeping file, not a rule source) and the Git config
lookup path are made environment-independent.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import os

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import git_safe_directory_value
from tests.cpp_parity_live._phase7_contract import (
    _required_golden_files,
    golden_dir,
    missing_golden_files,
    PHASE7_MODES,
)
from tests.cpp_parity_live._proc import BoundedResult, run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

#: Real, present-on-disk PHASE7_MODES ``golden_dir()``s. Read once at
#: collection time so a whole-tree traversal is not repeated per test.
_GOLDEN_TREE_ROOT = os.path.dirname(golden_dir(PHASE7_MODES[0]))

#: Bound for every ``git`` subprocess this module spawns - plumbing calls on
#: a small, already-on-disk file set, so a generous bound catches a truly
#: hung process without flaking on normal load.
_GIT_TIMEOUT_S = 30.0

#: Real ``.git/index`` path of the project repository - read ONLY to prove
#: byte-for-byte non-mutation (see
#: :func:`test_phase7_golden_tracking_never_touches_the_real_git_index`);
#: never opened for writing anywhere in this module.
_REAL_GIT_INDEX_PATH = os.path.join(PROJECT_ROOT, ".git", "index")


def _all_required_files() -> list:
    """
    Return every file :func:`~tests.cpp_parity_live._phase7_contract.
    _required_golden_files` names across all Phase 7 modes.

    :returns: Absolute paths, one list entry per required file.
    """
    files = []
    for mode in PHASE7_MODES:
        files.extend(_required_golden_files(mode))
    return files


def _all_tree_files() -> list:
    """
    Return every file that actually exists under the Phase 7 golden tree
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


def _git(*args: str, cwd: str = PROJECT_ROOT, env: "dict | None" = None) -> BoundedResult:
    """
    Run a bounded ``git`` subprocess against the real project repository,
    with a PER-COMMAND ``-c safe.directory=<forward-slash form of
    PROJECT_ROOT>`` override (see
    :func:`~tests.cpp_parity_live._golden_manifest.git_safe_directory_value`)
    so this call never depends on the running account's global Git
    configuration already trusting this checkout (git's own "dubious
    ownership" safety check would otherwise fail closed for every
    subcommand, not only index-writing ones, in an environment where the
    process does not own the repository). This is process-local config
    for this one invocation only - it never writes to any config file
    and never adds this repository to global configuration.

    Phase 7 correction pass (2026-09-07): the safe-directory value was
    previously the raw ``PROJECT_ROOT`` string, which on Windows is a
    BACKSLASH path - git does not recognize that form as matching the
    checkout for ``safe.directory`` purposes, so the override was
    silently ineffective. Now forward-slash-normalized via
    :func:`git_safe_directory_value`.

    :param args: ``git`` subcommand and arguments.
    :param cwd: Working directory for the subprocess.
    :param env: Environment for the subprocess (``None`` inherits this
        process's own environment, matching
        :func:`~tests.cpp_parity_live._proc.run_bounded`'s own default).
    :returns: The :class:`~tests.cpp_parity_live._proc.BoundedResult`.
    """
    return run_bounded(
        ["git", "-c", f"safe.directory={git_safe_directory_value(PROJECT_ROOT)}", *args],
        timeout=_GIT_TIMEOUT_S, cwd=cwd, env=env,
    )


def _hash_file(path: str) -> str:
    """
    Return the SHA-256 hex digest of *path*'s current bytes.

    :param path: File to hash.
    :return: Hex digest string.
    """
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_dry_run_add_reports_ignored_status_in_a_disposable_process_owned_repo():
    """
    Environment-independence proof (Phase 7 correction pass item 5): the
    ``GIT_INDEX_FILE``-redirected ``git add --dry-run`` mechanism this
    module's real-repo trackability test depends on is exercised here
    inside a throwaway, THIS-PROCESS-OWNED Git repository (``git init``
    under the Phase 7 scratch root) - never the real project repository
    - so this proof holds regardless of the real repository's ownership
    or any global ``safe.directory`` configuration. Both directions are
    checked: a real, non-ignored file reports ``add '<path>'``, and a
    real, ignored one reports the real "ignored by one of your
    .gitignore files" message - with a disposable ``GIT_INDEX_FILE`` that
    never existed before this call.

    :returns: None.
    """
    with scratch_tempdir("golden_tracking", prefix="disposable-repo-dryrun") as repo_dir:
        init = _git("init", "-q", cwd=repo_dir)
        assert init.returncode == 0, init.stderr

        with open(os.path.join(repo_dir, ".gitignore"), "w", encoding="utf-8") as fh:
            fh.write("*.ignored\n")
        with open(os.path.join(repo_dir, "keep.txt"), "w", encoding="utf-8") as fh:
            fh.write("keep\n")
        with open(os.path.join(repo_dir, "drop.ignored"), "w", encoding="utf-8") as fh:
            fh.write("drop\n")

        with scratch_tempdir("golden_tracking", prefix="disposable-index") as index_dir:
            disposable_index = os.path.join(index_dir, "index")
            child_env = dict(os.environ)
            child_env["GIT_INDEX_FILE"] = disposable_index
            assert not os.path.exists(disposable_index)

            kept = _git("add", "--dry-run", "--", "keep.txt", cwd=repo_dir, env=child_env)
            assert kept.returncode == 0, kept.stderr
            assert "add 'keep.txt'" in kept.stdout

            dropped = _git(
                "add", "--dry-run", "--", "drop.ignored", cwd=repo_dir, env=child_env,
            )
            assert dropped.returncode == 1
            assert "ignored" in dropped.stderr.lower()


def test_git_helper_uses_a_per_command_safe_directory_override_not_global_config(
        monkeypatch,
):
    """
    Environment-independence proof (Phase 7 correction pass item 5):
    :func:`_git` must pass ``-c safe.directory=<PROJECT_ROOT>`` as part
    of the actual argv on every invocation, never relying on this
    process's global Git configuration already trusting the checkout.
    Verified by spying on the real
    :func:`~tests.cpp_parity_live._proc.run_bounded` (delegating to the
    real implementation so the call still genuinely executes) and
    inspecting the captured argv, rather than merely reading the source.

    :param monkeypatch: Pytest fixture, restores the real ``run_bounded``
        automatically even if this test fails.
    :returns: None.
    """
    import tests.unit.test_phase7_golden_tracking as this_module

    captured = {}
    real_run_bounded = this_module.run_bounded

    def _spy(args, **kwargs):
        captured["args"] = list(args)
        return real_run_bounded(args, **kwargs)

    monkeypatch.setattr(this_module, "run_bounded", _spy)
    result = _git("status", "--porcelain")

    assert result.returncode == 0, result.stderr
    args = captured["args"]
    assert args[0] == "git"
    assert "-c" in args
    safe_dir_index = args.index("-c") + 1
    expected = f"safe.directory={git_safe_directory_value(PROJECT_ROOT)}"
    assert args[safe_dir_index] == expected
    assert "\\" not in args[safe_dir_index], (
        "safe.directory value must be forward-slash-normalized, not a "
        f"raw Windows path: {args[safe_dir_index]!r}"
    )


def test_git_ignore_detection_works_in_a_disposable_process_owned_repo():
    """
    Environment-independence proof (Phase 7 correction pass item 5): the
    ``git check-ignore --no-index`` mechanism this module's real-repo
    ignore-check test depends on is exercised here inside a throwaway,
    THIS-PROCESS-OWNED Git repository - never the real project
    repository - so this proof holds regardless of the real
    repository's ownership or any global ``safe.directory``
    configuration.

    :returns: None.
    """
    with scratch_tempdir("golden_tracking", prefix="disposable-repo-ignore") as repo_dir:
        init = _git("init", "-q", cwd=repo_dir)
        assert init.returncode == 0, init.stderr

        with open(os.path.join(repo_dir, ".gitignore"), "w", encoding="utf-8") as fh:
            fh.write("*.ignored\n")
        with open(os.path.join(repo_dir, "keep.txt"), "w", encoding="utf-8") as fh:
            fh.write("keep\n")
        with open(os.path.join(repo_dir, "drop.ignored"), "w", encoding="utf-8") as fh:
            fh.write("drop\n")

        still_ignored = _check_ignore_no_index(
            ["keep.txt", "drop.ignored"], cwd=repo_dir,
        )
        assert still_ignored == {"drop.ignored"}


def test_phase7_golden_files_are_not_gitignored():
    """
    Every required Phase 7 golden file resolves as NOT ignored by git
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


def test_phase7_golden_files_are_trackable_without_dash_f():
    """
    Every required Phase 7 golden file is committable by a normal,
    non-forced ``git add``, whether this checkout currently has the
    Phase 7 golden tree tracked or still untracked: EITHER it is already
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
        # A disposable, repository-local GIT_INDEX_FILE (never the real
        # .git/index) means this dry-run add can never attempt to create
        # the real .git/index.lock - it still runs against the REAL
        # working tree (cwd=PROJECT_ROOT), so it still evaluates the REAL
        # .gitignore/.gitattributes rules; only the index bookkeeping
        # file is redirected. See this module's own docstring.
        with scratch_tempdir("golden_tracking", prefix="dry-run-index") as index_dir:
            disposable_index = os.path.join(index_dir, "index")
            child_env = dict(os.environ)
            child_env["GIT_INDEX_FILE"] = disposable_index
            dry_run = _git("add", "--dry-run", "--", *untracked_rel, env=child_env)
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
        "these required Phase 7 golden files are neither already tracked "
        f"nor stageable by a real 'git add --dry-run' without -f: "
        f"{sorted(uncovered)}"
    )


def test_phase7_golden_tracking_never_touches_the_real_git_index():
    """
    Item 5's headline proof: this module's own two real-repo git tests
    (:func:`test_phase7_golden_files_are_not_gitignored`,
    :func:`test_phase7_golden_files_are_trackable_without_dash_f`) must
    leave the real project repository's ``.git/index`` byte-for-byte
    unchanged - captured as a SHA-256 hash before and after running both,
    called directly here as ordinary functions.

    :returns: None.
    """
    assert os.path.isfile(_REAL_GIT_INDEX_PATH), (
        f"expected a real .git/index at {_REAL_GIT_INDEX_PATH!r}"
    )
    before = _hash_file(_REAL_GIT_INDEX_PATH)

    test_phase7_golden_files_are_not_gitignored()
    test_phase7_golden_files_are_trackable_without_dash_f()

    after = _hash_file(_REAL_GIT_INDEX_PATH)
    assert before == after, (
        "the real .git/index changed after running the Phase 7 golden-"
        "tracking tests - it must never be written to"
    )


def test_phase7_golden_tree_has_no_stale_or_extra_files():
    """
    The Phase 7 golden tree on disk contains exactly the required files -
    nothing stale or extra (e.g. a leftover side file that should have
    been cleaned up by the generator).

    :returns: None.
    """
    required = {os.path.normpath(p) for p in _all_required_files()}
    on_disk = {os.path.normpath(p) for p in _all_tree_files()}
    extra = on_disk - required
    assert not extra, f"unexpected file(s) in the Phase 7 golden tree: {sorted(extra)}"


def test_phase7_golden_tree_is_complete():
    """
    Fail CLOSED with the exact missing-file list if the committed Phase 7
    golden dataset is incomplete.

    :returns: None.
    """
    missing = missing_golden_files()
    assert not missing, (
        "the committed Phase 7 golden dataset is incomplete - missing or "
        f"empty: {missing}"
    )
