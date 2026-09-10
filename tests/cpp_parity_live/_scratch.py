#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_scratch.py - Repository-local scratch-directory utilities for Phase 7,
introduced during the Phase 7 correction pass (2026-09-06) to close item
4 of that pass: every Phase 7 temporary directory (golden generation,
``--verify-only``, generator-driver tests, golden-tracking tests,
``gen_burnup_in_file``/fire-environment-bounds tests, and the Git
environment-independence tests) must live beneath this repository, never
the system/user temporary directory - unqualified ``tempfile.
TemporaryDirectory()`` and pytest's built-in ``tmp_path`` (whose root is
platform/environment-dependent and may be outside the repository) both
violate that boundary.

This is new, Phase-7-owned infrastructure - it does not modify
``tests/_support.py`` (used by every earlier phase) or any other shared
module, so no earlier phase's ``GENERATOR_SOURCE_FILES(_RELATIVE)`` list
is affected by this file's existence.

**Phase 7 correction pass (2026-09-07), item 4**: :func:`scratch_tempdir`
previously called ``shutil.rmtree(path, ignore_errors=True)``, silently
accepting a cleanup failure (a real, reviewer-observed consequence: nine
empty leftover directories were found under :data:`PHASE7_SCRATCH_ROOT`
after a focused run). Cleanup is now fail-closed - a real deletion
failure raises, chained (via ``raise ... from``, which also preserves
any exception still propagating from the ``with`` block as
``__context__``) so BOTH causes remain visible, never one silently
dropped - and successful cleanup now also prunes empty caller
subdirectories and their ancestors, stopping strictly before (never
removing) :data:`PHASE7_SCRATCH_ROOT` itself, using ``os.rmdir`` (which
only succeeds on a genuinely empty directory) so a directory another
concurrent or nested scratch context still owns is left untouched - the
OS's own atomicity is the single source of truth on "still in use",
never a separate snapshot check that could race.

**Phase 7 correction pass (2026-09-08), item 1 fallout**: a real git
checkout (e.g. a disposable local clone used to test
``prepare_cpp_reference.py`` without touching the pinned submodule)
stores its ``.git/objects/*`` files read-only on Windows, so a first
``shutil.rmtree`` attempt on a scratch directory containing one
genuinely fails with ``WinError 5: Access is denied`` even though this
process created and fully owns the directory - reproduced directly.
:func:`scratch_tempdir` now retries once after clearing read-only
attributes throughout the tree (:func:`_make_tree_writable`) before
treating a removal failure as fail-closed-raise-worthy; a failure that
survives the retry still raises exactly as before - this closes a real
false-positive in the fail-closed cleanup, it does not weaken it.

Function order: private helpers first, then public functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import stat
import tempfile
from typing import Iterator

from tests._support import PROJECT_ROOT, TESTS_DIR

#: Root of every Phase 7 scratch directory - a single, gitignored
#: directory tree under ``tests/``, never the system/user temp directory.
PHASE7_SCRATCH_ROOT: str = os.path.join(TESTS_DIR, ".phase7_scratch")


def _is_link_like(entry: str) -> bool:
    """
    Return ``True`` if *entry* is a symlink, a Windows directory junction,
    or any other detectable filesystem reparse point - without ever
    following it to inspect or resolve its target.

    **Narrow correction pass (2026-09-08, second round)**: the prior
    implementation checked only :func:`os.path.islink`, but a Windows
    directory JUNCTION is not a symlink - :func:`os.path.islink` reports
    ``False`` for one (verified directly), and - more critically -
    ``os.walk()`` with its own default ``followlinks=False`` still
    DESCENDS into a junction (also verified directly: ``entry.
    is_symlink()`` is ``False`` for a junction, which is the exact
    condition ``os.walk`` itself checks to decide whether to skip
    recursion). A junction could therefore have let this module's own
    traversal/chmod escape the verified scratch tree - the precise
    boundary this utility exists to prevent.

    Checks, in order, only what each platform actually exposes (no
    platform raises here - each check is skipped, not attempted, where
    unsupported):

    - :func:`os.path.islink` - ordinary symlinks, every platform.
    - :func:`os.path.isjunction` - Windows directory junctions
      specifically (Python 3.12+; present only on Windows, guarded via
      ``hasattr``).
    - :func:`os.lstat`'s ``st_file_attributes`` bit
      ``stat.FILE_ATTRIBUTE_REPARSE_POINT`` - any OTHER Windows reparse
      point neither check above catches (``st_file_attributes`` is a
      Windows-only ``stat_result`` attribute; both it and the constant
      are read via ``getattr``/``hasattr`` so this degrades to a no-op
      on platforms lacking them, never raising).

    Every check inspects *entry* itself without following it:
    :func:`os.path.islink`/:func:`os.path.isjunction` never dereference
    their argument, and :func:`os.lstat` is the no-follow stat call - the
    target of a real link/reparse point is never opened, resolved, or
    chmod'd by this function.

    :param entry: Path to classify (need not exist).
    :return: ``True`` if *entry* is link-like/reparse and must be
        neither chmod'd nor traversed into.
    """
    if os.path.islink(entry):
        return True
    if hasattr(os.path, "isjunction") and os.path.isjunction(entry):
        return True
    reparse_bit = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", None)
    if reparse_bit is not None:
        try:
            attrs = os.lstat(entry).st_file_attributes
        except (OSError, AttributeError):
            attrs = 0
        if attrs & reparse_bit:
            return True
    return False


def _make_tree_writable(path: str) -> None:
    """
    Grant the owner write permission (and, for directories, read+execute
    permission) on *path* and everything beneath it, preserving every
    other existing mode bit - an additive fix-up, never a wholesale mode
    replacement.

    A real git checkout stores its ``.git/objects/*`` files read-only on
    Windows, which makes a plain ``shutil.rmtree`` fail with
    ``WinError 5: Access is denied`` even when this process created and
    fully owns the directory - not a genuine permission problem, just a
    normal git artifact. Called once, as a retry step, before treating a
    removal failure as fail-closed-raise-worthy.

    **Narrow correction pass (2026-09-08)**: the original implementation
    replaced each entry's ENTIRE mode with ``stat.S_IWRITE | stat.S_IREAD``
    (``0o600``), unconditionally discarding every other bit - on POSIX
    this can turn a normal ``0o755`` directory into ``0o600``, stripping
    owner EXECUTE (search) permission. Reproduced directly: chmod'ing a
    directory this way makes ``os.walk``'s own subsequent ``listdir()``
    of it fail with ``PermissionError`` (silently suppressed by
    ``os.walk``, which just yields nothing further for that subtree),
    leaving deeper entries never visited and never made writable - the
    retry then fails again, permanently, on those untouched files. Fixed
    to read each entry's EXISTING mode first (``os.stat`` with
    ``follow_symlinks=False``, so a symlink's own mode is inspected, not
    its target's) and OR in only the bits deletion actually needs: owner
    write for files; owner write, read, AND execute for directories
    (read+execute are both independently required merely to list a
    directory's own contents on POSIX, and execute additionally to
    traverse into it - the prior bug came from losing execute).
    Symlinks/junctions/reparse points (see :func:`_is_link_like`) are
    never chmod'd, and directories identified as link-like are removed
    from ``os.walk``'s own ``dirs`` list IN PLACE before it can descend
    into them - not merely skipped when this function's own ``_grant``
    would otherwise chmod them. **Narrow correction pass (2026-09-08,
    second round)**: reproduced directly that ``os.walk()``'s default
    ``followlinks=False`` does NOT stop it descending into a Windows
    junction (a junction is not a symlink, so ``os.walk``'s own
    ``is_symlink()`` check that gates recursion never fires for one) -
    a bare "don't chmod a link-like entry" guard was therefore
    insufficient; traversal itself had to be blocked. A symlink's own
    removal needs only its parent directory's permissions (never the
    target's), and following any of these to chmod or list its target
    could reach a location outside the verified scratch tree.

    :param path: Directory tree to make writable (need not exist; a
        missing path is silently skipped per-entry via
        :func:`os.walk`/``os.chmod``'s own ``OSError``).
    :return: None.
    """
    def _grant(entry: str) -> None:
        if _is_link_like(entry):
            return
        try:
            current_mode = os.stat(entry, follow_symlinks=False).st_mode
        except OSError:
            return
        added = stat.S_IWUSR
        if stat.S_ISDIR(current_mode):
            added |= stat.S_IRUSR | stat.S_IXUSR
        new_mode = current_mode | added
        if new_mode != current_mode:
            try:
                os.chmod(entry, new_mode)
            except OSError:
                pass

    _grant(path)
    for root, dirs, files in os.walk(path, topdown=True):
        # Filter link-like subdirectories out of `dirs` IN PLACE before
        # os.walk can descend into them on its next iteration - a
        # Windows junction is not a symlink, so os.walk's own
        # followlinks=False check alone does not stop it recursing into
        # one (verified directly).
        dirs[:] = [name for name in dirs if not _is_link_like(os.path.join(root, name))]
        for name in dirs:
            _grant(os.path.join(root, name))
        for name in files:
            _grant(os.path.join(root, name))


def _prune_empty_ancestors(start_dir: str) -> None:
    """
    Starting at *start_dir*, remove it and each successive empty parent
    directory, stopping strictly before (never removing)
    :data:`PHASE7_SCRATCH_ROOT` itself.

    Best-effort and race-safe by construction: each removal uses
    ``os.rmdir``, which only succeeds on a genuinely empty directory, so
    a directory another concurrent or nested :func:`scratch_tempdir`
    call still owns (or that a concurrent process just repopulated) is
    left untouched rather than force-removed - pruning simply stops at
    the first non-empty (or already-gone) directory. This is silent,
    expected behaviour, not a cleanup failure: the caller's own
    mkdtemp'd directory was already removed successfully before this
    function is ever called.

    :param start_dir: Directory to start pruning from (ordinarily the
        parent of a just-removed :func:`tempfile.mkdtemp` directory).
    """
    root_real = _resolved(PHASE7_SCRATCH_ROOT)
    current = _resolved(start_dir)
    while current != root_real and current.startswith(root_real + os.sep):
        verify_under_project_root(current)
        try:
            os.rmdir(current)
        except OSError:
            return
        current = os.path.dirname(current)


def _resolved(path: str) -> str:
    """
    Return the fully resolved (real, absolute) form of *path*.

    :param path: Path to resolve (need not yet exist).
    :return: ``os.path.realpath(path)``.
    """
    return os.path.realpath(path)


def scratch_root(*subdirs: str) -> str:
    """
    Return (creating if needed) a Phase 7 scratch directory under
    :data:`PHASE7_SCRATCH_ROOT`, verified to resolve under
    :data:`~tests._support.PROJECT_ROOT` before creation.

    :param subdirs: Path segments appended to :data:`PHASE7_SCRATCH_ROOT`.
    :return: The resolved, existing directory path.
    :raises RuntimeError: If the resolved path is not under
        :data:`~tests._support.PROJECT_ROOT`.
    """
    path = os.path.join(PHASE7_SCRATCH_ROOT, *subdirs)
    verify_under_project_root(path)
    os.makedirs(path, exist_ok=True)
    return path


@contextlib.contextmanager
def scratch_tempdir(*subdirs: str, prefix: str = "tmp") -> Iterator[str]:
    """
    Context manager yielding a unique, collision-safe, repository-local
    temporary directory under ``scratch_root(*subdirs)`` - a drop-in
    replacement for ``tempfile.TemporaryDirectory()`` that never touches
    the system/user temp directory.

    The directory is verified to resolve under
    :data:`~tests._support.PROJECT_ROOT` immediately before both creation
    and removal, and is removed on exit regardless of whether the ``with``
    block succeeded, raised, or was interrupted by a timeout in the
    calling process (the removal happens in a ``finally`` block here; a
    hard process kill from outside is the one case no in-process
    ``finally`` can guarantee against, identical to the limitation
    ``tempfile.TemporaryDirectory()`` itself has).

    **Fail-closed cleanup (Phase 7 correction pass, 2026-09-07, item 4)**:
    a real removal failure RAISES (chained via ``raise ... from`` - if
    the ``with`` block itself was also raising, that original exception
    remains visible as the chained exception's ``__context__``, so
    neither cause is silently dropped) instead of being swallowed. On a
    successful removal, empty caller subdirectories are also pruned via
    :func:`_prune_empty_ancestors`, stopping strictly before
    :data:`PHASE7_SCRATCH_ROOT` itself, so a directory a concurrent or
    nested :func:`scratch_tempdir` call still owns is never touched.

    :param subdirs: Path segments appended to :data:`PHASE7_SCRATCH_ROOT`
        to select this caller's own scratch subtree (e.g.
        ``"generate"``, ``"pytest-tmp"``, ``"git-env"``).
    :param prefix: Prefix for the unique directory name
        (:func:`tempfile.mkdtemp`'s own collision-safe uniqueness).
    :return: Yields the resolved, existing directory path.
    :raises RuntimeError: If the resolved path is not under
        :data:`~tests._support.PROJECT_ROOT`, at either creation or
        removal time.
    :raises OSError: If removing *path* itself fails (fail-closed - never
        silently swallowed).
    """
    parent = scratch_root(*subdirs)
    path = tempfile.mkdtemp(prefix=f"{prefix}-", dir=parent)
    verify_under_project_root(path)
    try:
        yield path
    finally:
        verify_under_project_root(path)
        try:
            shutil.rmtree(path)
        except OSError:
            # Retry once after clearing read-only attributes (a real git
            # checkout's .git/objects/* files are read-only on Windows,
            # which otherwise fails removal even though this process
            # fully owns the directory it created) - only a failure that
            # survives this retry is treated as fail-closed-raise-worthy.
            _make_tree_writable(path)
            try:
                shutil.rmtree(path)
            except OSError as cleanup_exc:
                raise OSError(
                    f"scratch_tempdir cleanup failed for {path!r}: {cleanup_exc}"
                ) from cleanup_exc
        _prune_empty_ancestors(parent)


def verify_under_project_root(path: str) -> str:
    """
    Resolve *path* and raise ``RuntimeError`` unless it is
    :data:`~tests._support.PROJECT_ROOT` itself or a descendant of it -
    called before every scratch directory create OR delete in this
    module, so a bug can never make Phase 7 operate outside the
    checkout (mirrors ``massman_fof_dll_probe.py``'s own
    ``_verify_under_project_root()``, generalized here for reuse).

    :param path: Path to verify (need not yet exist).
    :return: The resolved (``os.path.realpath``) absolute path.
    :raises RuntimeError: If *path* resolves outside
        :data:`~tests._support.PROJECT_ROOT`.
    """
    real = _resolved(path)
    root_real = _resolved(PROJECT_ROOT)
    if real != root_real and not real.startswith(root_real + os.sep):
        raise RuntimeError(
            f"refusing to operate outside PROJECT_ROOT: {real!r} is not "
            f"under {root_real!r}"
        )
    return real
