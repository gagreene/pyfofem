#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_prepare_cpp_reference_git_ownership.py - Phase 7 correction pass
(2026-09-08) item 1: regression coverage proving every Git invocation in
``tests/prepare_cpp_reference.py`` (clone into a missing destination, the
refresh path's fetch/checkout/reset, and ``_print_status()``'s
``rev-parse``) succeeds under Git's "dubious ownership" safety check -
the exact gap independent review reproduced directly (``ensure_built()``
-> ``prepare_cpp_reference.py`` -> unqualified git calls -> ``fatal:
detected dubious ownership`` under an untrusted checkout).

Ownership mistrust is forced via the real git-provided
``GIT_TEST_ASSUME_DIFFERENT_OWNER=1`` test-only environment variable (not
a simulated/mocked check), combined with ``GIT_CONFIG_GLOBAL``/
``GIT_CONFIG_SYSTEM`` redirected to repository-local, throwaway config
files under this module's own Phase 7 scratch subtree - never the real
``~/.gitconfig``/system-wide config, and no ``safe.directory=*`` wildcard
anywhere. The refresh/clone tests operate on a disposable, network-free
local repository under scratch - never ``reference/fofem_cpp`` itself.
``_print_status()``'s own test targets the REAL checkout, but only via a
read-only ``git rev-parse HEAD``, never mutating it.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import tests.prepare_cpp_reference as prepare_cpp_reference
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._proc import run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

#: Bounded timeouts (seconds) for the subprocess calls this module makes.
_GIT_TIMEOUT_S = 30
_GENERATOR_SUITE_TIMEOUT_S = 180


def _blank_config_env(blank_global: str, blank_system: str) -> dict:
    """
    Return a full environment copy forcing Git ownership mistrust and
    redirecting Git's global/system configuration to throwaway,
    repository-local files - never the real ``~/.gitconfig`` or any
    system-wide file.

    :param blank_global: Path to a file to use as Git's "global" config
        for this environment (need not exist; may be pre-populated with
        throwaway ``safe.directory`` entries for disposable test fixtures
        only - never for the paths actually under test).
    :param blank_system: Same, for the "system" config.
    :return: ``os.environ`` copy with the 3 relevant variables set.
    """
    env = dict(os.environ)
    env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
    env["GIT_CONFIG_GLOBAL"] = blank_global
    env["GIT_CONFIG_SYSTEM"] = blank_system
    return env


def _init_disposable_origin(path: str) -> None:
    """
    Initialize a minimal, disposable local git repository with one commit
    at *path* - used only as a network-free clone source in these tests,
    never the real ``reference/fofem_cpp`` remote.

    :param path: Existing directory to initialize as a git repository.
    :return: None.
    """
    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=path, check=True,
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
        )

    _git("init", "-q", ".")
    _git("config", "user.email", "t@t.com")
    _git("config", "user.name", "t")
    with open(os.path.join(path, "f.txt"), "w", encoding="utf-8") as handle:
        handle.write("x")
    _git("add", "f.txt")
    _git("commit", "-q", "-m", "init")


def _write_blank_configs(base: str, *, trusted_local_origins: tuple = ()) -> tuple:
    """
    Write a throwaway, repository-local "global"/"system" Git config pair
    under *base* and return their paths.

    :param base: Existing scratch directory to write the 2 config files
        into.
    :param trusted_local_origins: Local repository directories to
        allowlist via ``safe.directory`` in the throwaway global config -
        used ONLY to make a disposable, network-free local origin
        clonable/fetchable in these tests (real production code always
        targets a real remote URL, which Git's ownership check does not
        apply to at all, so this never allowlists anything the fix under
        test is supposed to cover). Both the repository root AND its
        ``.git`` subdirectory are allowlisted - reproduced directly that
        a LOCAL clone/fetch source is checked against the ``.git`` path
        specifically, not the working-tree root alone.
    :return: ``(blank_global_path, blank_system_path)``.
    """
    blank_global = os.path.join(base, "blank_global.gitconfig")
    blank_system = os.path.join(base, "blank_system.gitconfig")
    lines = []
    for origin in trusted_local_origins:
        value = prepare_cpp_reference._safe_directory_value(pathlib.Path(origin))  # noqa: SLF001
        lines.append(f"\tdirectory = {value}")
        lines.append(f"\tdirectory = {value}/.git")
    with open(blank_global, "w", encoding="utf-8") as handle:
        if lines:
            handle.write("[safe]\n" + "\n".join(lines) + "\n")
    with open(blank_system, "w", encoding="utf-8") as handle:
        handle.write("")
    return blank_global, blank_system


def test_ensure_cpp_repo_clone_succeeds_under_forced_ownership_mistrust_into_a_missing_destination(monkeypatch):
    """
    ``_ensure_cpp_repo(refresh=False)``'s clone step must succeed under
    forced ownership mistrust when its destination does not yet exist -
    proving :func:`tests.prepare_cpp_reference._ensure_cpp_repo` needs no
    ``safe.directory`` override for the clone destination itself (there
    is no existing repository there for Git's check to reject).

    The disposable local origin repository IS trusted via a throwaway,
    repository-local ``GIT_CONFIG_GLOBAL`` entry (real production code
    never clones from a local path - only a real remote URL, which this
    ownership check does not apply to at all - so this allowlist entry
    exists purely to make the network-free test fixture itself clonable,
    not to weaken the protection this test is proving). The origin path
    handed to ``git clone`` on the command line is given in forward-slash
    form - reproduced directly that Git's local-clone-source ownership
    check does not match a backslash-form Windows path against an
    identical forward-slash ``safe.directory`` entry, even though both
    resolve to the same location (unlike ``-C <dir>``/``fetch``/
    ``rev-parse``, which normalize consistently regardless of the input's
    own separator style - the exact 3 call sites this correction pass's
    ``repo_path=`` fix covers).
    """
    with scratch_tempdir("prepare_cpp_reference_git_ownership", prefix="clone") as base:
        origin_dir = os.path.join(base, "origin")
        os.makedirs(origin_dir)
        _init_disposable_origin(origin_dir)

        dest_dir = os.path.join(base, "cpp_dir_equivalent")
        assert not os.path.exists(dest_dir)

        blank_global, blank_system = _write_blank_configs(
            base, trusted_local_origins=(origin_dir,),
        )
        origin_forward_slash = origin_dir.replace(os.sep, "/")
        monkeypatch.setattr(prepare_cpp_reference, "CPP_DIR", pathlib.Path(dest_dir))
        monkeypatch.setattr(prepare_cpp_reference, "REFERENCE_DIR", pathlib.Path(base))
        monkeypatch.setattr(
            prepare_cpp_reference, "_read_gitmodules_url", lambda: origin_forward_slash,
        )
        for key, value in _blank_config_env(blank_global, blank_system).items():
            monkeypatch.setenv(key, value)

        prepare_cpp_reference._ensure_cpp_repo(refresh=False)  # noqa: SLF001

        assert os.path.isdir(os.path.join(dest_dir, ".git"))


def test_ensure_cpp_repo_refresh_path_succeeds_under_forced_ownership_mistrust(monkeypatch):
    """
    ``_ensure_cpp_repo(refresh=True)``'s ``fetch``/``checkout``/``reset``
    sequence must succeed under forced ownership mistrust against a
    disposable, already-cloned local checkout - proving the
    ``repo_path=`` qualification added to each of those 3 calls works for
    a genuinely existing repository. The checkout itself gets NO
    ``safe.directory`` allowlist entry - the ``repo_path=`` fix under
    test is the ONLY thing making the fetch/checkout/reset sequence
    succeed against it. The disposable local ORIGIN remote is trusted the
    same way the clone test's fixture is (``git fetch`` against a local
    filesystem path remote also opens that remote's own ``.git``
    directly, independent of the checkout's own qualification): real
    production ``fetch`` always targets a real remote URL, which this
    ownership check does not apply to at all. As in the clone test, the
    origin path is given to git in forward-slash form (a backslash-form
    Windows path does not match the identical forward-slash
    ``safe.directory`` entry for a local clone/fetch source).
    """
    with scratch_tempdir("prepare_cpp_reference_git_ownership", prefix="refresh") as base:
        origin_dir = os.path.join(base, "origin")
        os.makedirs(origin_dir)
        _init_disposable_origin(origin_dir)

        blank_global, blank_system = _write_blank_configs(
            base, trusted_local_origins=(origin_dir,),
        )
        env = _blank_config_env(blank_global, blank_system)

        checkout_dir = os.path.join(base, "cpp_dir_equivalent")
        subprocess.run(
            ["git", "clone", "-q", origin_dir.replace(os.sep, "/"), checkout_dir],
            check=True, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
            env=env,
        )
        assert os.path.isdir(os.path.join(checkout_dir, ".git"))

        monkeypatch.setattr(prepare_cpp_reference, "CPP_DIR", pathlib.Path(checkout_dir))
        monkeypatch.setattr(
            prepare_cpp_reference, "BUILD_DIR",
            pathlib.Path(checkout_dir) / "build-test",
        )
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        # Without the fix, this raises subprocess.CalledProcessError with
        # "fatal: detected dubious ownership" - reproduced directly
        # before this fix existed.
        prepare_cpp_reference._ensure_cpp_repo(refresh=True)  # noqa: SLF001


def test_generator_driver_suite_passes_under_forced_ownership_mistrust():
    """
    The REAL burnup_extended (formerly Phase 7) cases of the consolidated
    generator-driver suite (``test_generate_goldens.py``, which exercises
    ``generate_all()`` -> ``ensure_built()`` ->
    ``tests/prepare_cpp_reference.py`` as a real subprocess, then
    ``check_pinned_sha()``/``git_dirty_status()`` in
    ``_golden_manifest.py``) must pass end to end under forced ownership
    mistrust - not a mocked helper test, the actual module, spawned as a
    real subprocess so the hostile environment propagates through the
    full ``ensure_built()`` -> ``prepare_cpp_reference.py`` -> git chain
    exactly as it would in a genuinely untrusted checkout. Filtered to the
    ``burnup_extended`` dataset parametrization (not the whole, now
    dataset-parametrized module) to keep this specifically about the
    fail-closed-on-missing-toolchain path Phase 7 established, at the
    original module's runtime cost.
    """
    with scratch_tempdir("prepare_cpp_reference_git_ownership", prefix="generator-suite") as base:
        blank_global, blank_system = _write_blank_configs(base)
        env = _blank_config_env(blank_global, blank_system)
        result = run_bounded(
            [
                sys.executable, "-m", "pytest",
                "tests/cpp_parity_live/test_generate_goldens.py",
                "-k", "burnup_extended", "-q",
            ],
            timeout=_GENERATOR_SUITE_TIMEOUT_S, cwd=PROJECT_ROOT, env=env,
        )
        assert result.returncode == 0, (
            f"generator-driver suite failed under forced ownership mistrust:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


def test_print_status_succeeds_under_forced_ownership_mistrust_against_the_real_checkout(monkeypatch):
    """
    ``_print_status()``'s ``git rev-parse HEAD`` call against the REAL
    ``reference/fofem_cpp`` checkout must succeed under forced ownership
    mistrust - the exact call site (line ~191 before this fix)
    independent review named directly. Read-only (``rev-parse`` never
    mutates the checkout), so this is safe to run against the real,
    pinned submodule.
    """
    with scratch_tempdir("prepare_cpp_reference_git_ownership", prefix="print-status") as base:
        blank_global, blank_system = _write_blank_configs(base)
        for key, value in _blank_config_env(blank_global, blank_system).items():
            monkeypatch.setenv(key, value)

        # Must not raise - before this fix, this reproduced the exact
        # reported "fatal: detected dubious ownership" failure.
        prepare_cpp_reference._print_status()  # noqa: SLF001


def test_safe_directory_args_matches_the_established_helper():
    """``_safe_directory_args(path)`` must return exactly
    ``["-c", "safe.directory=<value>"]`` using
    :func:`tests.prepare_cpp_reference._safe_directory_value`."""
    path = pathlib.Path(PROJECT_ROOT) / "reference" / "fofem_cpp"
    args = prepare_cpp_reference._safe_directory_args(path)  # noqa: SLF001
    assert args[0] == "-c"
    assert args[1] == f"safe.directory={prepare_cpp_reference._safe_directory_value(path)}"  # noqa: SLF001


def test_safe_directory_value_uses_forward_slashes():
    """``_safe_directory_value(path)`` must normalize to forward slashes
    for both the parent repository and the pinned ``reference/fofem_cpp``
    submodule - the exact backslash gap already fixed once for
    ``tests/cpp_parity_live/_golden_manifest.py`` and reproduced/fixed
    again here for this script's own, separate git calls."""
    for path in (
            pathlib.Path(PROJECT_ROOT),
            pathlib.Path(PROJECT_ROOT) / "reference" / "fofem_cpp",
    ):
        value = prepare_cpp_reference._safe_directory_value(path)  # noqa: SLF001
        assert "\\" not in value, (path, value)
        assert value == str(path.resolve()).replace(os.sep, "/")
