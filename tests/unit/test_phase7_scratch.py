#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase7_scratch.py - Phase 7 correction pass item 4: contract tests
for ``tests/cpp_parity_live/_scratch.py``, the repository-local
scratch-directory utility every Phase 7 generator/test now uses instead
of unqualified ``tempfile.TemporaryDirectory()`` or pytest's built-in
``tmp_path`` (whose root may be a user/system temporary directory).

Every test here operates only on throwaway paths under this module's own
scratch subtree (cleaned up via each test's own context manager) or on a
non-existent path used purely to prove rejection - nothing here ever
creates state outside the repository.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid

import pytest

import tests.cpp_parity_live._scratch as _scratch_module
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._scratch import (
    PHASE7_SCRATCH_ROOT,
    scratch_root,
    scratch_tempdir,
    verify_under_project_root,
)


@pytest.fixture(scope="module", autouse=True)
def _assert_no_residue_after_module():
    """
    Module-level postcondition (Phase 7 correction pass, 2026-09-08, item
    2): after every test in this module has run, :data:`PHASE7_SCRATCH_ROOT`
    must contain no artifact this module's own tests created - not merely
    "no individual test's own assertion caught a leak", but one real,
    automatic, always-checked proof covering the whole module, including
    any cross-test leakage a single test's own in-body assertions might
    miss. Runs once per module (``scope="module"``), after the LAST test
    finishes, not per test.

    :return: None (a ``yield``-based fixture; the assertion runs during
        teardown).
    """
    yield
    if os.path.isdir(PHASE7_SCRATCH_ROOT):
        leftover = os.listdir(PHASE7_SCRATCH_ROOT)
        assert not leftover, (
            "PHASE7_SCRATCH_ROOT contains leftover entries after "
            f"test_phase7_scratch.py's own tests ran: {leftover}"
        )


def _cleanup_leaked_scratch_path(path: str) -> None:
    """
    Remove a scratch directory a test deliberately left behind by
    monkeypatching ``shutil.rmtree`` to fail (to prove fail-closed
    cleanup genuinely raises), using the REAL removal machinery - the
    monkeypatch must already be restored by the caller before this runs,
    since the ONLY reason the directory survived is the fake failure, not
    a genuine permission problem. Then prunes the now-possibly-empty
    parent the same way :func:`~tests.cpp_parity_live._scratch.
    scratch_tempdir`'s own successful-cleanup path does, so no orphaned
    empty parent remains either.

    :param path: The exact path the test itself created and recorded -
        verified under ``PROJECT_ROOT`` before any deletion, never a path
        merely assumed to be safe.
    :return: None.
    """
    verify_under_project_root(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    _scratch_module._prune_empty_ancestors(os.path.dirname(path))  # noqa: SLF001


def _init_tiny_git_repo(path: str) -> None:
    """
    Initialize a minimal, real local git repository with one commit at
    *path* - used only to reproduce the read-only ``.git/objects/*``
    files a genuine checkout leaves on Windows (the exact scenario
    :func:`~tests.cpp_parity_live._scratch._make_tree_writable`'s
    chmod-and-retry cleanup path was fixed for), never a network
    operation.

    :param path: Existing directory to initialize as a git repository.
    :return: None.
    """
    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=path, check=True,
            capture_output=True, text=True, timeout=30,
        )

    _git("init", "-q", ".")
    _git("config", "user.email", "t@t.com")
    _git("config", "user.name", "t")
    with open(os.path.join(path, "f.txt"), "w", encoding="utf-8") as handle:
        handle.write("x")
    _git("add", "f.txt")
    _git("commit", "-q", "-m", "init")


def _unique_subdir(label: str) -> str:
    """
    A per-call unique scratch subdir name, so pruning tests never
    interfere with any other test module's own concurrently-existing
    scratch subtree (which legitimately keeps :data:`PHASE7_SCRATCH_ROOT`
    non-empty during a real full-suite run).

    :param label: Human-readable label folded into the unique name.
    :return: ``f"{label}-{uuid4 hex}"``.
    """
    return f"{label}-{uuid.uuid4().hex}"


def test_make_tree_writable_never_chmods_a_simulated_generic_reparse_point(monkeypatch):
    """A simulated generic Windows reparse point - the THIRD detection
    path in ``_is_link_like`` (neither a symlink nor an
    ``os.path.isjunction`` hit, only the raw ``st_file_attributes &
    stat.FILE_ATTRIBUTE_REPARSE_POINT`` bit) - must never be chmod'd.

    Portable on any platform/Python version: this patches ``os.lstat``'s
    RETURN VALUE directly (wrapping the real result and injecting a
    fake ``st_file_attributes``) rather than relying on a real
    ``stat_result`` exposing that attribute at all (it is Windows-only on
    a genuine filesystem) - no real reparse point is created."""
    with scratch_tempdir(_unique_subdir("test-linklike-reparse")) as base:
        fake_reparse = os.path.join(base, "fake_reparse")
        with open(fake_reparse, "w", encoding="utf-8") as handle:
            handle.write("x")

        ordinary_dir = os.path.join(base, "ordinary_dir")
        os.makedirs(ordinary_dir)
        ordinary_file = os.path.join(ordinary_dir, "f.txt")
        with open(ordinary_file, "w", encoding="utf-8") as handle:
            handle.write("x")
        # Strip write first so _grant() has an actual bit to add back -
        # otherwise a freshly-created file is already fully writable and
        # _grant()'s own new_mode != current_mode check would skip
        # chmod entirely, making the "was it chmod'd" assertion vacuous.
        os.chmod(ordinary_file, os.stat(ordinary_file).st_mode & ~stat.S_IWRITE)

        real_lstat = os.lstat

        class _FakeStatResult:
            def __init__(self, real_result, attrs):
                self._real = real_result
                self.st_file_attributes = attrs

            def __getattr__(self, name):
                return getattr(self._real, name)

        def fake_lstat(path, *args, **kwargs):
            real_result = real_lstat(path, *args, **kwargs)
            if os.path.abspath(path) == os.path.abspath(fake_reparse):
                return _FakeStatResult(real_result, stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return real_result

        chmod_calls = []
        real_chmod = os.chmod

        def spy_chmod(path, mode, *args, **kwargs):
            chmod_calls.append(os.path.abspath(path))
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", fake_lstat)
        monkeypatch.setattr(os, "chmod", spy_chmod)

        _scratch_module._make_tree_writable(base)  # noqa: SLF001

        assert os.path.abspath(fake_reparse) not in chmod_calls
        assert os.path.abspath(ordinary_file) in chmod_calls


def test_make_tree_writable_never_chmods_or_traverses_a_simulated_symlink(monkeypatch):
    """A simulated symlink (real ``os.path.islink`` patched to report
    ``True`` for one specific path) must never be chmod'd, while an
    ordinary sibling file IS still processed. No real symlink is created
    (avoids any Windows elevated-privilege dependency)."""
    with scratch_tempdir(_unique_subdir("test-linklike-symlink")) as base:
        fake_link = os.path.join(base, "fake_symlink")
        with open(fake_link, "w", encoding="utf-8") as handle:
            handle.write("x")

        ordinary = os.path.join(base, "ordinary.txt")
        with open(ordinary, "w", encoding="utf-8") as handle:
            handle.write("x")
        os.chmod(ordinary, os.stat(ordinary).st_mode & ~stat.S_IWRITE)

        real_islink = os.path.islink

        def fake_islink(path):
            if os.path.abspath(path) == os.path.abspath(fake_link):
                return True
            return real_islink(path)

        chmod_calls = []
        real_chmod = os.chmod

        def spy_chmod(path, mode, *args, **kwargs):
            chmod_calls.append(os.path.abspath(path))
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os.path, "islink", fake_islink)
        monkeypatch.setattr(os, "chmod", spy_chmod)

        _scratch_module._make_tree_writable(base)  # noqa: SLF001

        assert os.path.abspath(fake_link) not in chmod_calls
        assert os.path.abspath(ordinary) in chmod_calls


def test_make_tree_writable_never_traverses_a_simulated_junction(monkeypatch):
    """A simulated Windows directory junction (real ``os.path.isjunction``
    patched to report ``True`` for one specific directory) must never be
    chmod'd AND its children must never be visited/chmod'd at all -
    proving TRAVERSAL itself is blocked, not merely chmod skipped.
    Reproduces the exact gap independent review found: a real junction
    is not a symlink, so ``os.walk``'s own ``followlinks=False`` check
    alone does not stop it recursing into one (confirmed directly against
    a real ``mklink /J`` junction before this fix).

    ``raising=False`` lets this patch apply even on a Python version
    where ``os.path.isjunction`` does not exist as a real attribute
    (added in 3.12; this repository's floor is 3.10) - the test still
    exercises exactly the same ``_is_link_like`` code path either way,
    without needing a skip. No real junction is created (avoids any
    ``mklink``/Windows-junction-creation dependency)."""
    with scratch_tempdir(_unique_subdir("test-linklike-junction")) as base:
        fake_junction = os.path.join(base, "fake_junction")
        os.makedirs(fake_junction)
        hidden_file = os.path.join(fake_junction, "should_never_be_visited.txt")
        with open(hidden_file, "w", encoding="utf-8") as handle:
            handle.write("x")

        ordinary_dir = os.path.join(base, "ordinary_dir")
        os.makedirs(ordinary_dir)
        ordinary_file = os.path.join(ordinary_dir, "f.txt")
        with open(ordinary_file, "w", encoding="utf-8") as handle:
            handle.write("x")
        os.chmod(ordinary_file, os.stat(ordinary_file).st_mode & ~stat.S_IWRITE)

        real_isjunction = getattr(os.path, "isjunction", lambda _p: False)

        def fake_isjunction(path):
            if os.path.abspath(path) == os.path.abspath(fake_junction):
                return True
            return real_isjunction(path)

        chmod_calls = []
        real_chmod = os.chmod

        def spy_chmod(path, mode, *args, **kwargs):
            chmod_calls.append(os.path.abspath(path))
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os.path, "isjunction", fake_isjunction, raising=False)
        monkeypatch.setattr(os, "chmod", spy_chmod)

        _scratch_module._make_tree_writable(base)  # noqa: SLF001

        assert os.path.abspath(fake_junction) not in chmod_calls
        assert os.path.abspath(hidden_file) not in chmod_calls
        assert os.path.abspath(ordinary_file) in chmod_calls


def test_make_tree_writable_preserves_pre_existing_mode_bits():
    """``_make_tree_writable()`` must ADD only the write bit a file needs
    for deletion, never replace its mode wholesale - reproduces the
    exact prior bug (``stat.S_IWRITE | stat.S_IREAD`` discarded every
    other bit via a plain assignment instead of an OR)."""
    with scratch_tempdir(_unique_subdir("test-mode-preserve-file")) as base:
        target = os.path.join(base, "f.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("x")
        initial_mode = os.stat(target).st_mode

        # Strip write, mimicking a git object file's read-only attribute.
        os.chmod(target, initial_mode & ~stat.S_IWRITE)
        assert not (os.stat(target).st_mode & stat.S_IWRITE)

        _scratch_module._make_tree_writable(base)  # noqa: SLF001

        final_mode = os.stat(target).st_mode
        assert final_mode & stat.S_IWRITE, "write permission was not restored"
        # Every bit the file had before it was stripped read-only must
        # still be present - proves an additive fix-up, not a wholesale
        # replacement with a fixed, narrow mode value.
        assert (initial_mode & final_mode) == initial_mode, (
            f"pre-existing mode bits were discarded: "
            f"initial={oct(initial_mode)} final={oct(final_mode)}"
        )


def test_make_tree_writable_retains_or_adds_directory_search_permission():
    """Directories must retain (or gain) owner read/write/execute
    (search) permission - the prior bug replaced a directory's ENTIRE
    mode with ``stat.S_IWRITE | stat.S_IREAD`` (``0o600``), discarding
    owner execute and making the directory unlistable/untraversable on
    POSIX (which in turn made ``os.walk``'s own next ``listdir()`` of it
    fail, silently leaving deeper entries never visited).

    The pre-strip step is best-effort only: Windows' ``os.chmod``/
    ``os.stat`` emulation for directories always reports the owner
    execute bit as set regardless of what is chmod'd (confirmed
    directly - there is no real POSIX execute/search concept for
    directories on Windows, so this precondition cannot be observed
    there), so this test asserts only the one thing that IS meaningful
    and portable: after :func:`~tests.cpp_parity_live._scratch.
    _make_tree_writable` runs, the directory's owner read/write/execute
    bits are all set - the real invariant the fix guarantees, checked
    directly rather than inferred from an unobservable precondition."""
    with scratch_tempdir(_unique_subdir("test-mode-preserve-dir")) as base:
        nested = os.path.join(base, "nested")
        os.makedirs(nested)
        current = os.stat(nested).st_mode
        os.chmod(nested, current & ~stat.S_IXUSR)  # best-effort; see above

        _scratch_module._make_tree_writable(base)  # noqa: SLF001

        final_mode = os.stat(nested).st_mode
        assert final_mode & stat.S_IXUSR, "owner execute/search was not restored"
        assert final_mode & stat.S_IWUSR, "owner write was not restored"
        assert final_mode & stat.S_IRUSR, "owner read was not restored"


def test_phase7_scratch_root_is_under_project_root():
    """:data:`PHASE7_SCRATCH_ROOT` itself must resolve under
    ``PROJECT_ROOT``."""
    resolved = os.path.realpath(PHASE7_SCRATCH_ROOT)
    root = os.path.realpath(PROJECT_ROOT)
    assert resolved == root or resolved.startswith(root + os.sep)


def test_scratch_root_creates_and_returns_a_repo_local_directory():
    """``scratch_root(*subdirs)`` creates (if needed) and returns an
    existing directory under :data:`PHASE7_SCRATCH_ROOT`."""
    created = scratch_root(_unique_subdir("test_phase7_scratch_probe"))
    try:
        assert os.path.isdir(created)
        assert os.path.realpath(created).startswith(
            os.path.realpath(PROJECT_ROOT) + os.sep
        )
    finally:
        shutil.rmtree(created)


def test_scratch_tempdir_cleans_up_after_a_raised_exception():
    """The directory created by ``scratch_tempdir`` must be removed even
    when the ``with`` block raises - the ``finally``-based cleanup must
    not be skipped by an exception (simulating an interrupted/failed
    generation run)."""
    captured_path = None
    with pytest.raises(ValueError, match="simulated failure"):
        with scratch_tempdir("test-cleanup-exception") as path:
            captured_path = path
            assert os.path.isdir(path)
            raise ValueError("simulated failure")
    assert captured_path is not None
    assert not os.path.exists(captured_path)


def test_scratch_tempdir_cleans_up_after_success():
    """The directory created by ``scratch_tempdir`` must be removed after
    a normal, successful ``with`` block exit."""
    with scratch_tempdir("test-cleanup-success") as path:
        assert os.path.isdir(path)
        marker = os.path.join(path, "marker.txt")
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("x")
        assert os.path.isfile(marker)
        captured_path = path
    assert not os.path.exists(captured_path)


def test_scratch_tempdir_cleans_up_after_timeout_style_interruption():
    """A ``TimeoutError`` raised inside the ``with`` block (simulating a
    bounded subprocess call that timed out) must still trigger cleanup,
    exactly like any other exception - the ``finally`` block does not
    special-case exception type."""
    captured_path = None
    with pytest.raises(TimeoutError):
        with scratch_tempdir("test-cleanup-timeout") as path:
            captured_path = path
            assert os.path.isdir(path)
            raise TimeoutError("simulated bounded-process timeout")
    assert captured_path is not None
    assert not os.path.exists(captured_path)


def test_scratch_tempdir_cleanup_failure_preserves_original_exception_as_context():
    """When the ``with`` block ITSELF raises and cleanup ALSO fails,
    neither cause may be silently dropped - the raised cleanup
    ``OSError`` must chain the original exception as ``__context__``
    (Python's own implicit exception-chaining, exercised directly, not
    merely assumed)."""
    def _fail_rmtree(_path, *a, **k):
        raise OSError("simulated deletion failure")

    original = _scratch_module.shutil.rmtree
    _scratch_module.shutil.rmtree = _fail_rmtree
    captured_path = None
    try:
        with pytest.raises(OSError, match="cleanup failed") as exc_info:
            with scratch_tempdir(_unique_subdir("test-cleanup-fail-both")) as path:
                captured_path = path
                assert os.path.isdir(path)
                raise ValueError("original failure")
    finally:
        _scratch_module.shutil.rmtree = original
        # The fake failure is the ONLY reason this path survived - remove
        # it for real now that the working rmtree is restored, so this
        # test (which deliberately induces an unrecoverable-looking
        # failure to prove fail-closed cleanup raises) leaves no scratch
        # residue of its own behind.
        if captured_path is not None:
            _cleanup_leaked_scratch_path(captured_path)

    # The raised OSError's __cause__ is the SECOND (retry) rmtree failure
    # (explicit "raise ... from"); since the retry's own try/except sits
    # INSIDE the first attempt's except block (never fully exited before
    # the retry runs), that second failure's own __context__ is the FIRST
    # rmtree failure, not the ValueError directly - and the first
    # failure's own __context__ IS the ValueError, since it was raised
    # while the ValueError was still propagating through the enclosing
    # finally block. Walking __cause__ then __context__ twice finds it,
    # proving no cause is silently dropped anywhere in the 2-attempt
    # (chmod-then-retry) cleanup chain.
    cause = exc_info.value.__cause__
    assert cause is not None, "cleanup OSError must chain __cause__"
    assert isinstance(cause.__context__, OSError)
    assert isinstance(cause.__context__.__context__, ValueError)
    assert "original failure" in str(cause.__context__.__context__)


def test_scratch_tempdir_cleanup_failure_raises_instead_of_silently_swallowing():
    """A genuine ``shutil.rmtree`` failure on an otherwise-successful
    ``with`` block must RAISE, not be swallowed by
    ``ignore_errors=True`` (the exact fail-closed correction item 4
    requires) - proven via a monkeypatched ``shutil.rmtree`` rather than
    fragile real-filesystem permission manipulation."""
    def _fail_rmtree(_path, *a, **k):
        raise OSError("simulated deletion failure")

    original = _scratch_module.shutil.rmtree
    _scratch_module.shutil.rmtree = _fail_rmtree
    captured_path = None
    try:
        with pytest.raises(OSError, match="simulated deletion failure"):
            with scratch_tempdir(_unique_subdir("test-cleanup-fail")) as path:
                captured_path = path
                assert os.path.isdir(path)
    finally:
        _scratch_module.shutil.rmtree = original
        if captured_path is not None:
            _cleanup_leaked_scratch_path(captured_path)


def test_scratch_tempdir_does_not_remove_a_parent_still_owned_by_another_active_context():
    """Exiting an INNER ``scratch_tempdir`` must not remove the shared
    parent subdirectory while an OUTER ``scratch_tempdir`` under the
    same unique subdir is still open and using it - pruning must observe
    real filesystem state (``os.rmdir`` fails on a non-empty directory),
    not merely the inner call's own view."""
    subdir = _unique_subdir("test-prune-still-active-parent")
    parent = os.path.join(PHASE7_SCRATCH_ROOT, subdir)
    with scratch_tempdir(subdir) as outer_path:
        with scratch_tempdir(subdir) as inner_path:
            assert inner_path != outer_path
        # Inner context exited and was removed, but the parent must
        # survive - the outer context's own directory still lives there.
        assert os.path.isdir(parent), (
            "parent subdirectory was pruned while still owned by the "
            "outer scratch_tempdir context"
        )
        assert os.path.isdir(outer_path)
    # Now both contexts are closed - the parent must be pruned away.
    assert not os.path.exists(parent)


def test_scratch_tempdir_is_collision_safe_across_concurrent_calls():
    """Two nested/sequential ``scratch_tempdir`` calls with the SAME
    prefix and subdirs must never collide - ``tempfile.mkdtemp``'s own
    uniqueness guarantee, exercised directly."""
    with scratch_tempdir("test-collision", prefix="dup") as path_a:
        with scratch_tempdir("test-collision", prefix="dup") as path_b:
            assert path_a != path_b
            assert os.path.isdir(path_a)
            assert os.path.isdir(path_b)
        assert not os.path.exists(path_b)
    assert not os.path.exists(path_a)


def test_scratch_tempdir_prunes_empty_parent_after_normal_exit():
    """After the LAST ``scratch_tempdir`` under a given unique subdir
    exits normally, the now-empty ``scratch_root(subdir)`` parent
    directory itself must be pruned away, not left as an orphaned empty
    directory (the exact reviewer-observed regression item 4 corrects -
    nine such directories were found after a focused run)."""
    subdir = _unique_subdir("test-prune-empty-parent")
    parent = os.path.join(PHASE7_SCRATCH_ROOT, subdir)
    with scratch_tempdir(subdir) as path:
        assert os.path.isdir(parent)
        assert os.path.isdir(path)
    assert not os.path.exists(parent), (
        f"empty parent {parent!r} was not pruned after the only active "
        "scratch_tempdir context under it exited normally"
    )


def test_scratch_tempdir_pruning_never_removes_phase7_scratch_root_itself():
    """Even when a caller's own scratch subdirectory chain fully empties
    out, :data:`PHASE7_SCRATCH_ROOT` itself must never be pruned away -
    pruning stops strictly before it, since other test modules'
    concurrently-existing scratch subtrees legitimately keep it
    non-empty during a real suite run, and it is the fixed, gitignored
    root every Phase 7 scratch path is verified against."""
    subdir = _unique_subdir("test-prune-root-boundary")
    with scratch_tempdir(subdir):
        pass
    assert os.path.isdir(PHASE7_SCRATCH_ROOT), (
        "PHASE7_SCRATCH_ROOT itself must never be removed by pruning"
    )


def test_scratch_tempdir_removes_a_nested_read_only_git_like_tree():
    """The chmod-and-retry cleanup path must successfully remove a
    scratch directory containing a REAL git checkout (whose
    ``.git/objects/*`` files are read-only on Windows) - the exact
    scenario that motivated :func:`~tests.cpp_parity_live._scratch.
    _make_tree_writable`, exercised end to end via a genuine
    ``git init``/``commit``, not simulated."""
    with scratch_tempdir(_unique_subdir("test-cleanup-real-git-tree")) as outer:
        repo_dir = os.path.join(outer, "repo")
        os.makedirs(repo_dir)
        _init_tiny_git_repo(repo_dir)
        assert os.path.isdir(os.path.join(repo_dir, ".git"))
    # Normal exit must have removed the entire tree, including the
    # read-only git object files, without raising.
    assert not os.path.exists(outer)


def test_verify_under_project_root_accepts_a_real_child_path():
    """A path that genuinely resolves under ``PROJECT_ROOT`` must be
    accepted (no exception), returning its resolved form."""
    child = os.path.join(PROJECT_ROOT, "tests", "unit", "does_not_need_to_exist")
    resolved = verify_under_project_root(child)
    assert resolved == os.path.realpath(child)


def test_verify_under_project_root_rejects_a_path_traversal_escape():
    """A path that reaches outside ``PROJECT_ROOT`` via literal ``..``
    traversal segments (e.g. from deep inside
    :data:`PHASE7_SCRATCH_ROOT`) must be rejected - ``os.path.realpath``
    collapses the ``..`` segments before the containment check runs, so
    this is a real, not merely textual, escape attempt."""
    escaped = os.path.join(PHASE7_SCRATCH_ROOT, "..", "..", "..", "escaped_dir")
    assert not os.path.realpath(escaped).startswith(
        os.path.realpath(PROJECT_ROOT) + os.sep
    ), "test fixture assumption violated: this traversal must actually escape PROJECT_ROOT"
    with pytest.raises(RuntimeError, match="outside PROJECT_ROOT"):
        verify_under_project_root(escaped)


def test_verify_under_project_root_rejects_a_path_whose_resolved_form_escapes_project_root(monkeypatch):
    """A path that LOOKS like it is under :data:`PHASE7_SCRATCH_ROOT`
    textually, but whose RESOLVED (real) form escapes ``PROJECT_ROOT`` -
    exactly what a symlink, NTFS reparse point/junction, or bind mount
    pointing outside the repository would produce - must be rejected.

    Portable by construction (Phase 7 correction pass, 2026-09-08, item
    4): creating a real symlink needs elevated privilege on Windows only,
    and a directory junction has no POSIX equivalent, so neither can run
    unconditionally on Windows, Linux, and macOS without a skip - which
    the complete Phase 7 file surface forbids
    (``test_phase7_contract_hygiene.py`` asserts zero
    ``pytest.skip()``/``importorskip()`` calls anywhere). Instead, this
    test targets the containment DECISION directly: ``_resolved()`` (the
    module's own single ``os.path.realpath()`` choke point) is
    monkeypatched to report an escaped location for exactly ONE
    escape-candidate path, while every other path (including
    ``PROJECT_ROOT`` itself, checked in the same call) still resolves via
    the REAL ``os.path.realpath`` - a precise, targeted simulation of
    "this path's resolved form is outside PROJECT_ROOT", not a blanket
    stub, and nothing is read from or written to any real location
    outside the repository."""
    outside_target = os.path.join(
        os.path.dirname(os.path.realpath(PROJECT_ROOT)),
        "phase7-scratch-escape-target-never-created",
    )
    escape_candidate = os.path.join(PHASE7_SCRATCH_ROOT, "looks_like_a_real_subdir")
    assert not os.path.exists(escape_candidate)
    assert not os.path.exists(outside_target)

    real_resolved = _scratch_module._resolved

    def _fake_resolved(path):
        if os.path.abspath(path) == os.path.abspath(escape_candidate):
            return outside_target
        return real_resolved(path)

    monkeypatch.setattr(_scratch_module, "_resolved", _fake_resolved)

    with pytest.raises(RuntimeError, match="outside PROJECT_ROOT"):
        verify_under_project_root(escape_candidate)


def test_verify_under_project_root_rejects_a_sibling_directory():
    """A path resolving to a directory OUTSIDE the repository but not
    the system temp directory (e.g. the repository's own parent) must
    also be rejected - proves the check is a real containment test, not
    merely a system-temp-directory blocklist."""
    sibling = os.path.dirname(os.path.realpath(PROJECT_ROOT))
    with pytest.raises(RuntimeError, match="outside PROJECT_ROOT"):
        verify_under_project_root(sibling)


def test_verify_under_project_root_rejects_the_real_system_temp_directory():
    """A path resolving to the real system/user temporary directory must
    be REJECTED - the exact filesystem-boundary violation item 4 exists
    to prevent (unqualified ``tempfile.TemporaryDirectory()``, pytest's
    built-in ``tmp_path``)."""
    import tempfile
    system_temp = tempfile.gettempdir()
    # Guard against the pathological case where the real system temp
    # directory happens to be inside this checkout (never true in this
    # project, but asserted rather than assumed).
    assert not os.path.realpath(system_temp).startswith(
        os.path.realpath(PROJECT_ROOT) + os.sep
    )
    with pytest.raises(RuntimeError, match="outside PROJECT_ROOT"):
        verify_under_project_root(system_temp)
