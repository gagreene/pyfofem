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


# Preserves the exact pre-restructure CORE/FULL membership, remapped onto
# the Phase 1 destination paths (see the Directory Restructure section of
# development/plans/2026-08-26-comprehensive-test-suite-plan.md). Membership
# is NOT re-derived by directory: the 3 golden-CSV-driven files that used to
# sit in FULL_EXTRA_TESTS's neighborhood remain CORE because they compare
# against pre-committed golden CSVs and need no live C++ build, exactly as
# before the move.
CORE_TESTS: List[str] = [
    "tests/unit/test_consumption_golden.py",   # was tests/test_equations_golden.py (split half 1)
    "tests/regression/test_equations_golden_fixes.py",  # was tests/test_equations_golden.py (split half 2)
    "tests/unit/test_burnup_golden.py",        # was tests/test_burnup_golden.py
    "tests/unit/test_equation_routing.py",     # was tests/test_emission_equation_ids.py
    "tests/regression/test_pr1_review_regressions.py",  # was tests/test_pr1_review_regressions.py
    "tests/integration/test_run_fofem_emissions.py",    # was tests/test_run_fofem_emissions_output_keys.py
    "tests/integration/test_soil_heating_pipeline.py",  # was tests/test_soil_heating_invalid_soil_family.py
    # New Phase 1 runner/import-contract coverage (itemized, not pre-existing):
    "tests/unit/test_run_unified_tests_contract.py",
    # Phase 2: manifest builder/validator logic only (file hashing + git
    # rev-parse) — no live C++ build required, so this stays in CORE.
    "tests/unit/test_golden_manifest_validator.py",
    # Phase 2 correction pass: bounded subprocess helper (real process-tree
    # kill on timeout) — pure psutil/subprocess, no live C++ build.
    "tests/unit/test_proc.py",
    # Phase 2 correction pass: tolerance_policy.json schema/completeness —
    # pure JSON/static inspection, no live C++ build.
    "tests/unit/test_tolerance_policy_completeness.py",
]

FULL_EXTRA_TESTS: List[str] = [
    "tests/cpp_parity_live/test_compare_cpp_python.py",     # was tests/test_compare_cpp_python.py
    "tests/cpp_parity_live/test_cpp_comparison.py",          # was tests/test_cpp_comparison.py
    "tests/cpp_parity_live/test_soil_heating_cpp_parity.py",  # was tests/test_soil_heating_cpp_parity.py
    # Phase 2: builds and drives the live fofem_test C++ harness binary
    # directly (MSVC/CMake/Ninja required) — not a golden-CSV comparison.
    "tests/cpp_parity_live/test_cpp_harness_contract.py",
    # Phase 2: driver tests for the golden generator/promoter (wrong SHA,
    # qualification failure, harness failure, staleness, corruption,
    # manifest mismatch, determinism) — also needs the live build.
    "tests/cpp_parity_live/test_generate_phase2_goldens.py",
]

#: Environment variable set on the pytest subprocess when ``--installed-only``
#: is requested, so ``tests/conftest.py``'s ``pytest_sessionstart`` hook can
#: verify the import origin *inside* the process that collects/runs tests.
#: Must stay in sync with ``tests/conftest.py``.
_INSTALLED_ONLY_ENV_VAR = "PYFOFEM_INSTALLED_ONLY"


def _check_import(installed_only: bool) -> None:
    """
    Import pyfofem and print the resolved module path.

    :param installed_only: If ``True``, raise when pyfofem resolves to the
        local ``src/`` tree instead of an installed package.
    :return: None. Prints the resolved import path as a side effect.
    :raises RuntimeError: If *installed_only* is ``True`` and pyfofem
        resolves to the local source tree.
    """
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


def _discover_active_test_modules() -> List[str]:
    """
    Return active pytest modules under tests/ that should be accounted for.

    Uses a recursive glob so modules under ``tests/unit/``,
    ``tests/integration/``, ``tests/regression/``, and
    ``tests/cpp_parity_live/`` are not silently dropped from suite-coverage
    validation (the pre-Phase-1 non-recursive glob only ever saw the flat
    ``tests/*.py`` layout and would return an empty list once tests moved
    into subdirectories).

    :return: Sorted list of ``tests/**/test_*.py`` relative paths, excluding
        this runner script itself.
    """
    tests_dir = Path(_repo_root()) / "tests"
    paths = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        rel = path.relative_to(_repo_root()).as_posix()
        if rel == "tests/run_unified_tests.py":
            continue
        paths.append(rel)
    return paths


def _repo_root() -> str:
    """
    Resolve the repository root directory from this file's location.

    :return: Absolute path to the repository root (two levels up from
        ``tests/run_unified_tests.py``).
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_tests(suite: str) -> List[str]:
    """
    Resolve the list of test-file paths for the requested suite.

    :param suite: ``'core'`` for the publish-safe default tests, or
        ``'full'`` to additionally include parity/comparison tests.
    :return: List of ``tests/*.py`` relative paths to run.
    """
    tests = list(CORE_TESTS)
    if suite == "full":
        tests.extend(FULL_EXTRA_TESTS)
    return tests


def _run_pytest(test_paths: List[str], verbosity: int, installed_only: bool) -> int:
    """
    Invoke pytest as a subprocess against the given test paths.

    When *installed_only* is set, propagates :data:`_INSTALLED_ONLY_ENV_VAR`
    into the pytest subprocess's environment so ``tests/conftest.py``'s
    ``pytest_sessionstart`` hook can verify, from *inside* the process that
    actually collects and runs the tests, that ``pyfofem`` does not resolve
    beneath the checkout's ``src/`` tree. This is the child-process half of
    the parent/child installed-only contract; :func:`_check_import`'s
    parent-process check alone cannot see what a freshly spawned subprocess
    will import.

    :param test_paths: Test file paths to run, relative to the repo root.
    :param verbosity: Pytest verbosity level (``<=0`` for ``-q``, ``>=2``
        for ``-vv``, otherwise pytest's default verbosity).
    :param installed_only: If ``True``, set :data:`_INSTALLED_ONLY_ENV_VAR`
        on the subprocess environment.
    :return: The pytest subprocess return code.
    """
    cmd = [sys.executable, "-m", "pytest", "-ra"]
    if verbosity <= 0:
        cmd.append("-q")
    elif verbosity >= 2:
        cmd.append("-vv")
    cmd.extend(test_paths)

    env = dict(os.environ)
    if installed_only:
        env[_INSTALLED_ONLY_ENV_VAR] = "1"
    else:
        env.pop(_INSTALLED_ONLY_ENV_VAR, None)

    print(f"[unified-tests] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_repo_root(), env=env)
    return int(proc.returncode)


def _validate_suite_coverage() -> None:
    """
    Fail fast if a new test module was added but not assigned to a suite.

    :return: None. Raises if uncovered test modules are discovered.
    :raises RuntimeError: If any discovered test module is not present in
        ``CORE_TESTS`` or ``FULL_EXTRA_TESTS``.
    """
    configured = set(CORE_TESTS) | set(FULL_EXTRA_TESTS)
    discovered = set(_discover_active_test_modules())
    missing = sorted(discovered - configured)
    if missing:
        raise RuntimeError(
            "run_unified_tests.py is missing active test modules:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def _verify_pytest_available() -> None:
    """
    Confirm that pytest is importable before attempting to run any suite.

    :return: None. Raises if pytest cannot be imported.
    :raises RuntimeError: If pytest is not installed/importable.
    """
    try:
        importlib.import_module("pytest")
    except Exception as exc:  # pragma: no cover - environment check
        raise RuntimeError(
            "pytest is required for run_unified_tests.py. "
            "Install test deps first (e.g., `pip install pytest`)."
        ) from exc


def main() -> int:
    """
    Parse CLI arguments, validate the environment, and run the selected suite.

    :return: Process exit code — 0 on success, 2 if configured test files
        are missing, or the pytest subprocess return code otherwise.
    """
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

    return _run_pytest(
        test_paths=tests,
        verbosity=int(args.verbose),
        installed_only=bool(args.installed_only),
    )


if __name__ == "__main__":
    raise SystemExit(main())
