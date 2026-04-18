#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified test runner for pyfofem.

This script provides a stable single entrypoint for CI and packaging checks.
It wraps pytest suites and supports two publishing-friendly modes:

1) core  - fast, deterministic tests suitable for PyPI/Conda package checks.
2) full  - core + parity/comparison tests that depend on reference assets.

Examples
--------
python tests/run_unified_tests.py --suite core
python tests/run_unified_tests.py --suite full
python tests/run_unified_tests.py --suite core --installed-only
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List


CORE_TESTS: List[str] = [
    "tests/test_equations_golden.py",
    "tests/test_burnup_golden.py",
    "tests/test_run_fofem_emissions_output_keys.py",
    "tests/test_soil_heating_invalid_soil_family.py",
]

FULL_EXTRA_TESTS: List[str] = [
    "tests/test_compare_cpp_python.py",
    "tests/test_cpp_comparison.py",
    "tests/test_soil_heating_cpp_parity.py",
]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_tests(suite: str) -> List[str]:
    tests = list(CORE_TESTS)
    if suite == "full":
        tests.extend(FULL_EXTRA_TESTS)
    return tests


def _discover_active_test_modules() -> List[str]:
    """Return active pytest modules under tests/ that should be accounted for."""
    tests_dir = Path(_repo_root()) / "tests"
    paths = []
    for path in sorted(tests_dir.glob("test_*.py")):
        rel = path.relative_to(_repo_root()).as_posix()
        if rel == "tests/run_unified_tests.py":
            continue
        paths.append(rel)
    return paths


def _validate_suite_coverage() -> None:
    """Fail fast if a new test module was added but not assigned to a suite."""
    configured = set(CORE_TESTS) | set(FULL_EXTRA_TESTS)
    discovered = set(_discover_active_test_modules())
    missing = sorted(discovered - configured)
    if missing:
        raise RuntimeError(
            "run_unified_tests.py is missing active test modules:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def _verify_pytest_available() -> None:
    try:
        importlib.import_module("pytest")
    except Exception as exc:  # pragma: no cover - environment check
        raise RuntimeError(
            "pytest is required for run_unified_tests.py. "
            "Install test deps first (e.g., `pip install pytest`)."
        ) from exc


def _check_import(installed_only: bool) -> None:
    pyfofem = importlib.import_module("pyfofem")
    module_path = os.path.abspath(getattr(pyfofem, "__file__", ""))
    print(f"[unified-tests] pyfofem import: {module_path}")

    if not installed_only:
        return

    root = _repo_root()
    src_root = os.path.abspath(os.path.join(root, "src"))
    if module_path.startswith(src_root):
        raise RuntimeError(
            "--installed-only was requested, but pyfofem is imported from local "
            f"source tree: {module_path}"
        )


def _run_pytest(test_paths: List[str], verbosity: int) -> int:
    cmd = [sys.executable, "-m", "pytest", "-ra"]
    if verbosity <= 0:
        cmd.append("-q")
    elif verbosity >= 2:
        cmd.append("-vv")
    cmd.extend(test_paths)

    print(f"[unified-tests] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_repo_root())
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified pyfofem test runner.")
    parser.add_argument(
        "--suite",
        choices=("core", "full"),
        default="core",
        help="core: publish-safe default tests; full: includes parity/comparison tests.",
    )
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="Fail if pyfofem resolves to local ./src instead of an installed package.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase pytest verbosity (-v / -vv).",
    )
    args = parser.parse_args()

    _verify_pytest_available()
    _check_import(installed_only=bool(args.installed_only))
    _validate_suite_coverage()

    tests = _resolve_tests(args.suite)
    missing = [p for p in tests if not os.path.isfile(os.path.join(_repo_root(), p))]
    if missing:
        print("[unified-tests] missing test files:")
        for path in missing:
            print(f"  - {path}")
        return 2

    return _run_pytest(test_paths=tests, verbosity=int(args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
