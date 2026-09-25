#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare the local FOFEM C++ reference checkout used by full/parity tests.

This script intentionally stays separate from ``run_unified_tests.py`` so the
default test runner remains publishing-safe and offline-friendly.

Capabilities
------------
1. Optionally refresh ``reference/fofem_cpp`` from the upstream git remote.
2. Reapply the local overlay from ``reference/fofem_cpp_overlay/source``.
3. Optionally build the ``fofem_test`` C++ harness.

Examples
--------
python tests/prepare_cpp_reference.py
python tests/prepare_cpp_reference.py --refresh
python tests/prepare_cpp_reference.py --refresh --build
python tests/prepare_cpp_reference.py --build --build-system batch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


# _repo_root() is defined and called first, ahead of alphabetical order,
# because the module-level path constants below call it immediately at
# import time and Python requires the definition to exist before that call.
def _repo_root() -> Path:
    """
    Resolve the repository root directory from this file's location.

    :return: Absolute path to the repository root (two levels up from
        ``tests/prepare_cpp_reference.py``).
    """
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
REFERENCE_DIR = REPO_ROOT / "reference"
CPP_DIR = REFERENCE_DIR / "fofem_cpp"
OVERLAY_DIR = REFERENCE_DIR / "fofem_cpp_overlay"
OVERLAY_SOURCE_DIR = OVERLAY_DIR / "source"
BUILD_DIR = CPP_DIR / "build-test"


def _apply_overlay() -> None:
    """
    Copy overlay source files on top of the C++ reference checkout.

    :return: None. Copies files from ``OVERLAY_SOURCE_DIR`` into ``CPP_DIR``
        as a side effect and prints a summary count.
    :raises RuntimeError: If the overlay source or C++ reference directory
        does not exist.
    """
    if not OVERLAY_SOURCE_DIR.is_dir():
        raise RuntimeError(f"Overlay source directory not found: {OVERLAY_SOURCE_DIR}")
    if not CPP_DIR.is_dir():
        raise RuntimeError(f"C++ reference directory not found: {CPP_DIR}")
    copied = _copy_tree_contents(OVERLAY_SOURCE_DIR, CPP_DIR)
    print(f"[prepare-cpp] applied overlay files: {len(copied)}")


def _build_harness(build_system: str) -> None:
    """
    Build the ``fofem_test`` C++ harness using the requested build system.

    :param build_system: One of ``'cmake'``, ``'batch'``, or ``'auto'``
        (prefers CMake, falls back to the Windows batch script).
    :return: None. Builds the harness as a side effect.
    :raises RuntimeError: If *build_system* is ``'batch'`` on a non-Windows
        OS, or if ``'auto'`` finds no usable build system.
    """
    if build_system == "cmake":
        _build_with_cmake()
        return
    if build_system == "batch":
        if os.name != "nt":
            raise RuntimeError("Batch build is only supported on Windows.")
        _build_with_batch()
        return

    # auto
    if _which("cmake") is not None:
        _build_with_cmake()
        return
    if os.name == "nt" and (CPP_DIR / "compile_test.bat").is_file():
        _build_with_batch()
        return
    raise RuntimeError(
        "No usable build system found. Install CMake or provide compile_test.bat on Windows."
    )


def _build_with_batch() -> None:
    """
    Build the C++ harness via the Windows ``compile_test.bat`` script.

    :return: None. Runs the batch build as a side effect.
    :raises RuntimeError: If ``compile_test.bat`` is not found.
    """
    batch_path = CPP_DIR / "compile_test.bat"
    if not batch_path.is_file():
        raise RuntimeError(f"Batch build file not found: {batch_path}")
    _run(["cmd", "/c", str(batch_path)], cwd=CPP_DIR)


def _build_with_cmake() -> None:
    """
    Build the C++ harness via CMake.

    :return: None. Configures and builds the ``fofem_test`` target as a
        side effect.
    :raises RuntimeError: If ``cmake`` is not found on ``PATH``.
    """
    if _which("cmake") is None:
        raise RuntimeError("cmake was not found on PATH.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _run(["cmake", "-S", str(CPP_DIR), "-B", str(BUILD_DIR)], cwd=REPO_ROOT)
    _run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "fofem_test"], cwd=REPO_ROOT)


def _copy_tree_contents(src: Path, dst: Path) -> List[Path]:
    """
    Copy all files (recursively) from *src* into *dst*, preserving relative paths.

    :param src: Source directory to copy files from.
    :param dst: Destination directory to copy files into.
    :return: List of destination file paths that were written.
    """
    copied: List[Path] = []
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _ensure_cpp_repo(refresh: bool) -> None:
    """
    Clone the C++ reference repository if missing, or refresh it in place.

    :param refresh: If ``True`` and the repo already exists, fetch and hard
        reset it to ``origin/master`` before returning.
    :return: None. Clones or refreshes ``CPP_DIR`` as a side effect.
    :raises RuntimeError: If *refresh* is ``True`` but ``CPP_DIR`` exists
        and is not a git checkout.
    """
    remote_url = _read_gitmodules_url()
    if not CPP_DIR.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        # Deliberately no `repo_path=` here: `git clone` targets a
        # not-yet-existing directory, so there is no existing repository
        # for Git's dubious-ownership check to reject — verified directly
        # (a real clone against a remote URL into a fresh destination
        # succeeds under GIT_TEST_ASSUME_DIFFERENT_OWNER=1 with no
        # safe.directory override at all; see
        # test_prepare_cpp_reference_git_ownership.py).
        _run(["git", "clone", "--depth", "1", remote_url, str(CPP_DIR)], cwd=REPO_ROOT)
        return

    if not refresh:
        return

    if not (CPP_DIR / ".git").exists():
        raise RuntimeError(
            f"{CPP_DIR} exists but is not a git checkout. "
            "Refresh requires a clone or submodule checkout."
        )

    _run(["git", "fetch", "origin"], cwd=CPP_DIR, repo_path=CPP_DIR)
    _run(["git", "checkout", "master"], cwd=CPP_DIR, repo_path=CPP_DIR)
    _run(["git", "reset", "--hard", "origin/master"], cwd=CPP_DIR, repo_path=CPP_DIR)
    # Remove previous generated build output so the next build is clean.
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def _print_status() -> None:
    """
    Print the current C++ reference checkout HEAD and directory locations.

    :return: None. Prints status lines to stdout as a side effect.
    """
    if (CPP_DIR / ".git").exists():
        proc = subprocess.run(
            ["git", *_safe_directory_args(CPP_DIR), "rev-parse", "HEAD"],
            cwd=str(CPP_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[prepare-cpp] fofem_cpp HEAD: {proc.stdout.strip()}")
    print(f"[prepare-cpp] cpp dir: {CPP_DIR}")
    print(f"[prepare-cpp] overlay dir: {OVERLAY_DIR}")


def _read_gitmodules_url() -> str:
    """
    Read the C++ reference repository URL from ``.gitmodules``.

    :return: The remote URL from ``.gitmodules``, or a hardcoded fallback
        URL if ``.gitmodules`` is missing or has no matching entry.
    """
    gitmodules = REPO_ROOT / ".gitmodules"
    if gitmodules.is_file():
        text = gitmodules.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("url ="):
                return line.split("=", 1)[1].strip()
    return "https://github.com/bran-jnw/fofem_wuinity.git"


def _run(cmd: Iterable[str], *, cwd: Path | None = None, repo_path: Path | None = None) -> None:
    """
    Print and execute a subprocess command, raising on non-zero exit.

    :param cmd: Command and arguments to execute.
    :param cwd: Working directory to run the command in. Defaults to
        ``REPO_ROOT`` when omitted.
    :param repo_path: When given and *cmd* is a ``git`` invocation, a
        per-command ``-c safe.directory=<forward-slash form of repo_path>``
        is inserted immediately after ``git`` (never written to any config
        file) so the command succeeds regardless of the running account's
        global Git configuration or the repository's file ownership. Pass
        the existing repository this specific invocation OPERATES ON (e.g.
        ``CPP_DIR`` for a ``fetch``/``checkout``/``reset`` against the C++
        reference checkout) — omit it for commands (like ``git clone`` into
        a not-yet-existing directory) that do not need it; see
        :func:`_safe_directory_args`.
    :return: None. Runs the command as a side effect.
    :raises subprocess.CalledProcessError: If the command exits non-zero.
    """
    cmd = list(cmd)
    if repo_path is not None and cmd and cmd[0] == "git":
        cmd = [cmd[0], *_safe_directory_args(repo_path), *cmd[1:]]
    shown_cwd = str(cwd or REPO_ROOT)
    print(f"[prepare-cpp] ({shown_cwd})$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _safe_directory_args(repo_path: Path) -> List[str]:
    """
    Return the ``-c safe.directory=<value>`` argument pair for *repo_path*.

    Git's "dubious ownership" safety check rejects operating on a
    repository whose directory is (or is merely reported as, under
    ``GIT_TEST_ASSUME_DIFFERENT_OWNER``) owned by a different account than
    the running process, unless that exact path is listed in
    ``safe.directory`` configuration. A per-command ``-c`` override applies
    only to this one invocation — it never writes to any global, system, or
    local Git configuration file, and never uses the ``*`` wildcard that
    would blanket-disable the protection for every repository.

    :param repo_path: The existing repository directory this git
        invocation operates on.
    :return: ``["-c", "safe.directory=<value>"]``.
    """
    return ["-c", f"safe.directory={_safe_directory_value(repo_path)}"]


def _safe_directory_value(repo_path: Path) -> str:
    """
    Return *repo_path* resolved to an absolute, forward-slash-normalized
    path suitable as a ``safe.directory`` config value on every platform.

    Git matches a configured ``safe.directory`` value against its own
    internally-normalized (forward-slash) form of the repository path even
    on Windows — a raw Windows path with backslashes is not recognized as
    matching the checkout, silently making the override ineffective (the
    same gap already fixed for ``tests/cpp_parity_live/_golden_manifest.py``
    and reproduced directly here again for this script's own git calls).

    :param repo_path: Directory path to normalize (need not yet exist).
    :return: Absolute, forward-slash-normalized path string.
    """
    return str(repo_path.resolve()).replace(os.sep, "/")


def _which(executable: str) -> str | None:
    """
    Locate an executable on ``PATH``.

    :param executable: Executable name to search for.
    :return: Absolute path to the executable, or ``None`` if not found.
    """
    return shutil.which(executable)


def main() -> int:
    """
    Parse CLI arguments and prepare the local FOFEM C++ reference checkout.

    :return: Process exit code — 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(description="Prepare local FOFEM C++ reference assets.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch and hard-reset reference/fofem_cpp to origin/master before applying the overlay.",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Do not copy files from reference/fofem_cpp_overlay/source into the C++ checkout.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the fofem_test harness after preparing the checkout.",
    )
    parser.add_argument(
        "--build-system",
        choices=("auto", "cmake", "batch"),
        default="auto",
        help="How to build the harness when --build is set.",
    )
    args = parser.parse_args()

    try:
        _ensure_cpp_repo(refresh=bool(args.refresh))
        if not args.skip_overlay:
            _apply_overlay()
        if args.build:
            _build_harness(build_system=str(args.build_system))
        _print_status()
    except subprocess.CalledProcessError as exc:
        print(f"[prepare-cpp] command failed with exit code {exc.returncode}", file=sys.stderr)
        return int(exc.returncode or 1)
    except Exception as exc:  # pragma: no cover - environment/setup failures
        print(f"[prepare-cpp] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
