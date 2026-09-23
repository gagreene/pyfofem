#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_golden_tracking.py - fail-CLOSED completeness coverage and git
trackability proof for every committed C++-oracle golden dataset
(``canonical`` is exempt - see below - ``expanded_matrix``,
``soil_campbell``, ``emissions_equivalence``, ``burnup_extended``).

MERGED (test-suite renaming pass) from the four near-identical
``test_phase{4,5,6,7}_golden_tracking.py`` modules. Every per-dataset test
(completeness, no-stale-files, not-gitignored, trackable-without-``-f``) is
now a single function parametrized over :data:`_DATASETS`. The underlying
GIT BEHAVIOR proofs (``check-ignore``'s plain-vs-``--no-index`` tracked-file
blind spot, ``git add --dry-run``'s silence for an already-tracked-and-
unchanged file, environment-independence under forced ownership mistrust,
and the disposable-process-owned-repo mechanism proofs originally only in
the Phase 7 module) are properties of ``git`` itself, not of any one
dataset, so they are kept as SHARED, non-parametrized tests that run once -
duplicating them per dataset would prove nothing new each time.

The former Phase 4 module's simpler ``git add --dry-run`` (against the real
``.git/index``) and the former Phase 7 module's hardened,
``GIT_INDEX_FILE``-redirected form are NOT both kept: the hardened form is
strictly more correct (never risks creating the real ``.git/index.lock`` in
an environment where the running account lacks write access to it) and is
now used UNIFORMLY for all four datasets, including the two
(``expanded_matrix``, ``emissions_equivalence``) whose original modules used
the simpler form.

``canonical`` (the frozen Phase 2 dataset) is deliberately NOT covered by
this module - it force-tracks its own goldens directly (``git add -f``, see
``.gitattributes``) and was never covered by any of the four merged
modules either; it has no counterpart contract module exposing
``_required_golden_files``/``missing_golden_files`` the way the other four
do.

None of this needs the live MSVC/CMake/Ninja toolchain - it is pure
filesystem/git-plumbing inspection - so this module is CORE, like its four
predecessors. Every git operation is read-only/non-mutating (``--dry-run``,
``check-ignore``, ``ls-files``) against the real project repository and
bounded via :func:`~tests.cpp_parity_live._proc.run_bounded`; the two
per-dataset tests that touch the real repository's ignore rules are proven
to leave the real ``.git/index`` byte-for-byte unchanged by
:func:`test_golden_tracking_never_touches_the_real_git_index`.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Tuple

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import git_safe_directory_value
from tests.cpp_parity_live._proc import BoundedResult, run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

from tests.cpp_parity_live._expanded_matrix_contract import (
    EXPANDED_MATRIX_MODES,
    _required_golden_files as _expanded_matrix_required_golden_files,
    golden_dir as _expanded_matrix_golden_dir,
    missing_golden_files as _expanded_matrix_missing_golden_files,
)
from tests.cpp_parity_live._soil_campbell_contract import (
    SOIL_CAMPBELL_MODES,
    _required_golden_files as _soil_campbell_required_golden_files,
    golden_dir as _soil_campbell_golden_dir,
    missing_golden_files as _soil_campbell_missing_golden_files,
)
from tests.cpp_parity_live._emissions_equivalence_contract import (
    EMISSIONS_EQUIVALENCE_MODES,
    _required_golden_files as _emissions_equivalence_required_golden_files,
    golden_dir as _emissions_equivalence_golden_dir,
    missing_golden_files as _emissions_equivalence_missing_golden_files,
)
from tests.cpp_parity_live._burnup_extended_contract import (
    BURNUP_EXTENDED_MODES,
    _required_golden_files as _burnup_extended_required_golden_files,
    golden_dir as _burnup_extended_golden_dir,
    missing_golden_files as _burnup_extended_missing_golden_files,
)


@dataclass(frozen=True)
class _Dataset:
    """One golden dataset's identity and contract-module bindings."""

    label: str
    modes: Tuple[str, ...]
    required_golden_files: Callable[[str], List[str]]
    golden_dir: Callable[[str], str]
    missing_golden_files: Callable[[], List[str]]

    def all_required_files(self) -> List[str]:
        """Every file required across all of this dataset's modes."""
        files: List[str] = []
        for mode in self.modes:
            files.extend(self.required_golden_files(mode))
        return files

    def tree_root(self) -> str:
        """This dataset's golden-tree root directory."""
        return os.path.dirname(self.golden_dir(self.modes[0]))

    def all_tree_files(self) -> List[str]:
        """Every file that actually exists under this dataset's tree root."""
        found: List[str] = []
        for root, _dirs, files in os.walk(self.tree_root()):
            for name in files:
                found.append(os.path.join(root, name))
        return sorted(found)


#: Every dataset this module covers (``canonical`` excluded - see module
#: docstring). Order is arbitrary but stable, for deterministic test IDs.
_DATASETS: Tuple[_Dataset, ...] = (
    _Dataset(
        "expanded_matrix", EXPANDED_MATRIX_MODES,
        _expanded_matrix_required_golden_files, _expanded_matrix_golden_dir,
        _expanded_matrix_missing_golden_files,
    ),
    _Dataset(
        "soil_campbell", SOIL_CAMPBELL_MODES,
        _soil_campbell_required_golden_files, _soil_campbell_golden_dir,
        _soil_campbell_missing_golden_files,
    ),
    _Dataset(
        "emissions_equivalence", EMISSIONS_EQUIVALENCE_MODES,
        _emissions_equivalence_required_golden_files,
        _emissions_equivalence_golden_dir,
        _emissions_equivalence_missing_golden_files,
    ),
    _Dataset(
        "burnup_extended", BURNUP_EXTENDED_MODES,
        _burnup_extended_required_golden_files, _burnup_extended_golden_dir,
        _burnup_extended_missing_golden_files,
    ),
)

_DATASET_IDS: Tuple[str, ...] = tuple(d.label for d in _DATASETS)

#: Bound for every ``git`` subprocess this module spawns - plumbing calls on
#: a small, already-on-disk file set, so a generous bound catches a truly
#: hung process without flaking on normal load.
_GIT_TIMEOUT_S = 30.0

#: Real ``.git/index`` path of the project repository - read ONLY to prove
#: byte-for-byte non-mutation (see
#: :func:`test_golden_tracking_never_touches_the_real_git_index`); never
#: opened for writing anywhere in this module.
_REAL_GIT_INDEX_PATH = os.path.join(PROJECT_ROOT, ".git", "index")


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


def _git(*args: str, cwd: str = PROJECT_ROOT, env: "dict | None" = None) -> BoundedResult:
    """
    Run a bounded ``git`` subprocess, qualified with a per-command
    ``-c safe.directory=<forward-slash form of cwd>`` override so it
    succeeds regardless of the running account's global Git configuration
    or the repository's file ownership - never written to any config file.

    Reproduced directly under forced ownership mistrust
    (``GIT_TEST_ASSUME_DIFFERENT_OWNER=1``) that every call here failed
    with ``fatal: detected dubious ownership`` before this fix - both
    against the real project repository (the default ``cwd``) AND against
    disposable temp repositories some tests below create (a non-default
    ``cwd``). The trusted path is ALWAYS *cwd* itself, never an
    unconditional :data:`~tests._support.PROJECT_ROOT` - reuses the
    established
    :func:`~tests.cpp_parity_live._golden_manifest.git_safe_directory_value`
    helper rather than duplicating its forward-slash normalization logic.

    :param args: ``git`` subcommand and arguments.
    :param cwd: Working directory for the subprocess.
    :param env: Environment for the subprocess (``None`` inherits this
        process's own environment).
    :returns: The :class:`~tests.cpp_parity_live._proc.BoundedResult`.
    """
    safe_directory = git_safe_directory_value(cwd)
    return run_bounded(
        ["git", "-c", f"safe.directory={safe_directory}", *args],
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


def _not_gitignored(dataset: _Dataset) -> None:
    """
    Every required golden file for *dataset* resolves as NOT ignored by git
    (``git check-ignore --no-index``, a real subprocess).

    :param dataset: The dataset to check.
    :returns: None.
    """
    required = [
        os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        for path in dataset.all_required_files()
    ]
    still_ignored = _check_ignore_no_index(required)
    assert not still_ignored, (
        f"{dataset.label}: still ignored by git (git check-ignore "
        f"--no-index): {sorted(still_ignored)}"
    )


def _trackable_without_dash_f(dataset: _Dataset) -> None:
    """
    Every required golden file for *dataset* is committable by a normal,
    non-forced ``git add``, whether this checkout currently has that
    dataset's golden tree tracked or still untracked: EITHER it is already
    tracked (``git ls-files`` lists it, nothing further to add), OR it is
    untracked and a real ``git add --dry-run`` would stage it.

    ``git add --dry-run`` runs with ``GIT_INDEX_FILE`` pointed at a
    disposable, repository-local scratch file (never the real
    ``.git/index``) so it can never attempt to create the real
    ``.git/index.lock`` - it still runs against the REAL working tree
    (``cwd=PROJECT_ROOT``) and therefore the REAL ``.gitignore``/
    ``.gitattributes`` content, which is unaffected by which index file git
    happens to be updating.

    :param dataset: The dataset to check.
    :returns: None.
    """
    required_abs = dataset.all_required_files()
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
        f"{dataset.label}: these required golden files are neither already "
        f"tracked nor stageable by a real 'git add --dry-run' without -f: "
        f"{sorted(uncovered)}"
    )


def test_dry_run_add_reports_ignored_status_in_a_disposable_process_owned_repo():
    """
    Environment-independence proof: the ``GIT_INDEX_FILE``-redirected
    ``git add --dry-run`` mechanism :func:`_trackable_without_dash_f`
    depends on is exercised here inside a throwaway, THIS-PROCESS-OWNED
    Git repository (``git init`` under the shared scratch root) - never
    the real project repository - so this proof holds regardless of the
    real repository's ownership or any global ``safe.directory``
    configuration. Both directions are checked: a real, non-ignored file
    reports ``add '<path>'``, and a real, ignored one reports the real
    "ignored by one of your .gitignore files" message - with a disposable
    ``GIT_INDEX_FILE`` that never existed before this call.

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


def test_git_add_dry_run_is_silent_for_a_tracked_unchanged_ignored_file(
        tmp_path: pathlib.Path,
):
    """
    Ground-truth reproduction (disposable temp repo, real project repo
    untouched): once a file matching an active ``.gitignore`` rule is
    force-tracked and left unchanged, ``git add --dry-run`` on it exits 0
    but prints NOTHING - there is nothing to add. This is exactly why
    "every required file appears in ``git add --dry-run`` stdout" is not a
    valid universal check once a golden tree is committed.

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
    tracked status - the form :func:`_not_gitignored` actually relies on.

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
    :func:`_not_gitignored` uses ``--no-index`` rather than the plain form.

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


def test_git_helper_succeeds_against_a_disposable_repository_under_forced_ownership_mistrust(
        monkeypatch,
):
    """
    ``_git(cwd=<disposable repo>)`` must ALSO succeed under forced
    ownership mistrust - the trusted ``safe.directory`` value tracks the
    ACTUAL *cwd* passed to ``_git()``, never an unconditional
    :data:`~tests._support.PROJECT_ROOT`, since this module's own
    tracked-state probe tests create disposable repositories at a
    different path.

    :param monkeypatch: Pytest fixture; used only to set 3 environment
        variables for this test's own subprocess calls, auto-restored on
        teardown.
    :returns: None.
    """
    with scratch_tempdir("golden_tracking_ownership", prefix="disposable") as base:
        blank_global = os.path.join(base, "blank_global.gitconfig")
        blank_system = os.path.join(base, "blank_system.gitconfig")
        open(blank_global, "w", encoding="utf-8").close()
        open(blank_system, "w", encoding="utf-8").close()
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", blank_global)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", blank_system)

        repo = os.path.join(base, "disposable_repo")
        os.makedirs(repo)
        # Without the fix, this reproduces the exact reported failure:
        # "fatal: detected dubious ownership in repository at '<repo>'".
        result = _git("init", "-q", cwd=repo)
        assert result.returncode == 0, result.stderr


def test_git_helper_succeeds_against_the_real_project_under_forced_ownership_mistrust(
        monkeypatch,
):
    """
    ``_git()`` must succeed against the REAL project repository (a
    read-only ``rev-parse``) even when Git's dubious-ownership check is
    forced via ``GIT_TEST_ASSUME_DIFFERENT_OWNER=1`` with blank,
    repository-local global/system config.

    :param monkeypatch: Pytest fixture; used only to set 3 environment
        variables for this test's own subprocess calls, auto-restored on
        teardown.
    :returns: None.
    """
    with scratch_tempdir("golden_tracking_ownership", prefix="real") as base:
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


def test_git_helper_uses_a_per_command_safe_directory_override_not_global_config(
        monkeypatch,
):
    """
    Environment-independence proof: :func:`_git` must pass ``-c
    safe.directory=<PROJECT_ROOT>`` as part of the actual argv on every
    invocation, never relying on this process's global Git configuration
    already trusting the checkout. Verified by spying on the real
    :func:`~tests.cpp_parity_live._proc.run_bounded` (delegating to the
    real implementation so the call still genuinely executes) and
    inspecting the captured argv, rather than merely reading the source.

    :param monkeypatch: Pytest fixture, restores the real ``run_bounded``
        automatically even if this test fails.
    :returns: None.
    """
    import tests.unit.cpp.test_golden_tracking as this_module

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
    Environment-independence proof: the ``git check-ignore --no-index``
    mechanism :func:`_not_gitignored` depends on is exercised here inside
    a throwaway, THIS-PROCESS-OWNED Git repository - never the real
    project repository - so this proof holds regardless of the real
    repository's ownership or any global ``safe.directory`` configuration.

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


def test_golden_files_are_not_gitignored_fails_closed_on_git_operational_failure(
        tmp_path: pathlib.Path,
):
    """
    Regression: BEFORE this module's fail-closed fix, the not-gitignored
    check read ``result.stdout.splitlines()`` with no return-code check at
    all, so a ``git check-ignore --no-index`` invocation that failed
    OPERATIONALLY (git itself erroring out, e.g. rc=128) - which also
    produces empty stdout - was indistinguishable from "genuinely nothing
    is ignored" and would have PASSED incorrectly. AFTER the fix,
    :func:`_check_ignore_no_index` asserts the return code is exactly 0 or
    1 before trusting stdout at all.

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


def test_golden_tracking_never_touches_the_real_git_index():
    """
    Headline proof: every per-dataset real-repo git test in this module
    (:func:`_not_gitignored`, :func:`_trackable_without_dash_f`, run here
    for every dataset in :data:`_DATASETS`) must leave the real project
    repository's ``.git/index`` byte-for-byte unchanged - captured as a
    SHA-256 hash before and after running all of them, called directly
    here as ordinary functions.

    :returns: None.
    """
    assert os.path.isfile(_REAL_GIT_INDEX_PATH), (
        f"expected a real .git/index at {_REAL_GIT_INDEX_PATH!r}"
    )
    before = _hash_file(_REAL_GIT_INDEX_PATH)

    for dataset in _DATASETS:
        _not_gitignored(dataset)
        _trackable_without_dash_f(dataset)

    after = _hash_file(_REAL_GIT_INDEX_PATH)
    assert before == after, (
        "the real .git/index changed after running the golden-tracking "
        "tests - it must never be written to"
    )


@pytest.mark.parametrize("dataset", _DATASETS, ids=_DATASET_IDS)
def test_golden_tree_has_no_stale_or_extra_files(dataset: _Dataset):
    """
    *dataset*'s golden tree on disk contains exactly the required files -
    nothing stale or extra (e.g. a leftover side file that should have
    been cleaned up by the generator).

    :param dataset: The dataset to check.
    :returns: None.
    """
    required = {os.path.normpath(p) for p in dataset.all_required_files()}
    on_disk = {os.path.normpath(p) for p in dataset.all_tree_files()}
    extra = on_disk - required
    assert not extra, (
        f"unexpected file(s) in the {dataset.label} golden tree: {sorted(extra)}"
    )


@pytest.mark.parametrize("dataset", _DATASETS, ids=_DATASET_IDS)
def test_golden_tree_is_complete(dataset: _Dataset):
    """
    Fail CLOSED with the exact missing-file list if *dataset*'s committed
    golden dataset is incomplete.

    Duplicates the check every parity module for this dataset already runs
    at collection time (via ``require_golden_tree()``) as an independent,
    directly-runnable assertion, so this module alone proves the dataset
    is intact even if a future change altered how the parity modules wire
    their own collection-time guard.

    :param dataset: The dataset to check.
    :returns: None.
    """
    missing = dataset.missing_golden_files()
    assert not missing, (
        f"the committed {dataset.label} golden dataset is incomplete - "
        f"missing or empty: {missing}"
    )


@pytest.mark.parametrize("dataset", _DATASETS, ids=_DATASET_IDS)
def test_required_golden_files_are_not_gitignored(dataset: _Dataset):
    """
    Every required golden file for *dataset* resolves as NOT ignored by
    git - ``git check-ignore --no-index``, a real subprocess, never a
    re-implementation of gitignore's pattern-matching rules.

    ``--no-index`` is used deliberately rather than the plain form: plain
    ``git check-ignore`` reports a path as "not ignored" whenever it is
    already tracked, regardless of whether a gitignore rule still matches
    it (reproduced directly in
    :func:`test_git_check_ignore_plain_is_unreliable_for_a_tracked_ignored_file`),
    so it cannot distinguish "genuinely not excluded" from "excluded but
    saved by being tracked". ``--no-index`` evaluates the patterns against
    the path alone and is meaningful whether *dataset*'s golden tree is
    currently tracked or still untracked in this checkout.

    The return-code handling itself (fail closed on anything but git's own
    documented 0/1 outcomes for this subcommand, rather than silently
    trusting empty stdout) lives in :func:`_check_ignore_no_index` and is
    exercised in isolation by
    :func:`test_golden_files_are_not_gitignored_fails_closed_on_git_operational_failure`.

    :param dataset: The dataset to check.
    :returns: None.
    """
    _not_gitignored(dataset)


@pytest.mark.parametrize("dataset", _DATASETS, ids=_DATASET_IDS)
def test_required_golden_files_are_trackable_without_dash_f(dataset: _Dataset):
    """
    Every required golden file for *dataset* is committable by a normal,
    non-forced ``git add`` regardless of whether this checkout currently
    has that dataset's golden tree tracked or still untracked.

    For each required file:

    - it must not be ignored (:func:`test_required_golden_files_are_not_
      gitignored` covers this independently and this test does not repeat
      it, but relies on it being true);
    - EITHER it is already tracked (``git ls-files --error-unmatch``
      succeeds - nothing further to add, ``-f`` is moot), OR it is
      untracked and a real ``git add --dry-run`` would stage it (an
      ``add '...'`` stdout line, no "ignored" warning on stderr).

    :param dataset: The dataset to check.
    :returns: None.
    """
    _trackable_without_dash_f(dataset)
