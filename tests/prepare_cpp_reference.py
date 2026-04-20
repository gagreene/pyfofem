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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
REFERENCE_DIR = REPO_ROOT / "reference"
CPP_DIR = REFERENCE_DIR / "fofem_cpp"
OVERLAY_DIR = REFERENCE_DIR / "fofem_cpp_overlay"
OVERLAY_SOURCE_DIR = OVERLAY_DIR / "source"
BUILD_DIR = CPP_DIR / "build-test"


def _read_gitmodules_url() -> str:
    gitmodules = REPO_ROOT / ".gitmodules"
    if gitmodules.is_file():
        text = gitmodules.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("url ="):
                return line.split("=", 1)[1].strip()
    return "https://github.com/bran-jnw/fofem_wuinity.git"


def _run(cmd: Iterable[str], *, cwd: Path | None = None) -> None:
    cmd = list(cmd)
    shown_cwd = str(cwd or REPO_ROOT)
    print(f"[prepare-cpp] ({shown_cwd})$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _copy_tree_contents(src: Path, dst: Path) -> List[Path]:
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
    remote_url = _read_gitmodules_url()
    if not CPP_DIR.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", remote_url, str(CPP_DIR)], cwd=REPO_ROOT)
        return

    if not refresh:
        return

    if not (CPP_DIR / ".git").exists():
        raise RuntimeError(
            f"{CPP_DIR} exists but is not a git checkout. "
            "Refresh requires a clone or submodule checkout."
        )

    _run(["git", "fetch", "origin"], cwd=CPP_DIR)
    _run(["git", "checkout", "master"], cwd=CPP_DIR)
    _run(["git", "reset", "--hard", "origin/master"], cwd=CPP_DIR)
    # Remove previous generated build output so the next build is clean.
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def _apply_overlay() -> None:
    if not OVERLAY_SOURCE_DIR.is_dir():
        raise RuntimeError(f"Overlay source directory not found: {OVERLAY_SOURCE_DIR}")
    if not CPP_DIR.is_dir():
        raise RuntimeError(f"C++ reference directory not found: {CPP_DIR}")
    copied = _copy_tree_contents(OVERLAY_SOURCE_DIR, CPP_DIR)
    print(f"[prepare-cpp] applied overlay files: {len(copied)}")


def _which(executable: str) -> str | None:
    return shutil.which(executable)


def _build_with_cmake() -> None:
    if _which("cmake") is None:
        raise RuntimeError("cmake was not found on PATH.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _run(["cmake", "-S", str(CPP_DIR), "-B", str(BUILD_DIR)], cwd=REPO_ROOT)
    _run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "fofem_test"], cwd=REPO_ROOT)


def _build_with_batch() -> None:
    batch_path = CPP_DIR / "compile_test.bat"
    if not batch_path.is_file():
        raise RuntimeError(f"Batch build file not found: {batch_path}")
    _run(["cmd", "/c", str(batch_path)], cwd=CPP_DIR)


def _build_harness(build_system: str) -> None:
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


def _print_status() -> None:
    if (CPP_DIR / ".git").exists():
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(CPP_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[prepare-cpp] fofem_cpp HEAD: {proc.stdout.strip()}")
    print(f"[prepare-cpp] cpp dir: {CPP_DIR}")
    print(f"[prepare-cpp] overlay dir: {OVERLAY_DIR}")


def main() -> int:
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
