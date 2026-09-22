#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified test runner for pyfofem.

This script provides a stable single entrypoint for CI and packaging checks.
It wraps pytest suites and supports two publishing-friendly modes:

1) ci-smoke - fast representative checks for ordinary pull-request updates.
2) core     - deterministic tests suitable for protected-branch integration.
3) full     - core + parity/comparison tests that depend on reference assets.

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
    # Phase 3: Python-only, data, and relation-level public contracts.
    # None of these builds or runs C++ — they are equation/contract tests
    # and hand-derived source-relation cross-checks — so all six are CORE.
    "tests/unit/test_burnup_component_api.py",
    "tests/unit/test_public_constants.py",
    "tests/unit/test_runtime_data_resources.py",
    "tests/unit/test_tree_flame_contracts.py",
    "tests/unit/test_tree_flame_source_relations.py",
    "tests/unit/test_utility_contracts.py",
    # Phase 4: Tier-2 consumption/mortality/tree-structure/emissions parity
    # against the PRE-GENERATED, committed Phase 4 golden dataset. These
    # read committed CSVs and never build or invoke C++, so they are CORE -
    # the same rule that puts the other golden-CSV-driven modules here.
    "tests/unit/test_phase4_consumption_parity.py",
    "tests/unit/test_bark_thickness_contract.py",
    # CON-01/CON-02 post-suite correction: consm_shrub() Eq 16/234 four-term
    # f_WPRE fix and zero-load NaN fix. Reads the committed Phase 4 consume
    # golden for one live-C++-discriminating case - no live C++ build needed
    # at test-run time, same rule as the golden-CSV-driven modules above.
    "tests/unit/test_con01_con02_shrub_eq234.py",
    # F-39 (Coastal Plain sub-finding): consm_litter()/consm_duff()/
    # consm_mineral_soil() forest-floor route (Eq 30/31/32). Reads the
    # already-committed Phase 4 consume golden's se-cp-entire-m050 row -
    # no live C++ build needed at test-run time, same rule as the
    # golden-CSV-driven modules above.
    "tests/unit/test_f39_coastal_plain.py",
    "tests/unit/test_phase4_emissions_parity.py",
    # Phase 6 investigation A: default-emissions-equivalence full-pipeline
    # comparison (F-54) against the committed Phase 6 consume golden. Reads
    # committed CSVs only - no live C++ build needed at test-run time,
    # exactly like the Phase 4/5 golden-driven modules above/below.
    "tests/unit/test_phase6_default_emissions_equivalence.py",
    # Phase 7 item E: additional run_burnup()/burnup() scientific-parity
    # comparison (rotten/sound-mix, calm-wind, hot-ambient, long-residence
    # -time) against the committed Phase 7 consume golden. Reads
    # committed CSVs only - no live C++ build needed at test-run time,
    # exactly like the Phase 4/6 golden-driven modules above.
    "tests/unit/test_phase7_run_burnup_parity.py",
    # Phase 7 item E: fail-closed golden-tree completeness plus
    # git-trackability proof for the Phase 7 dataset, mirroring
    # test_phase4/5/6_golden_tracking.py's state-independent contract.
    # Filesystem/git-plumbing only - no live C++ build.
    "tests/unit/test_phase7_golden_tracking.py",
    # Phase 7 correction pass item 4: repository-local scratch-directory
    # contract tests (_scratch.py) - filesystem-only, no live C++ build.
    "tests/unit/test_phase7_scratch.py",
    # Phase 7 correction pass item 6: AST-based meta-test proving the
    # complete Phase 7 file surface contains no imperative pytest.skip()
    # calls and no skip/skipif markers. Pure source-file static analysis.
    "tests/unit/test_phase7_contract_hygiene.py",
    "tests/unit/test_phase4_mortality_parity.py",
    "tests/unit/test_phase4_tree_structure_parity.py",
    # Phase 4 correction pass: AST-based meta-test that no Phase 4 test
    # module contains an imperative pytest.xfail() call and every
    # pytest.mark.xfail(...) marker is strict=True. Pure source-file
    # static analysis - no live C++ build, no golden data read.
    "tests/unit/test_phase4_contract_hygiene.py",
    # Phase 4 correction pass: fail-closed golden-tree completeness plus
    # git-trackability proof (check-ignore, dry-run add, stale/extra file
    # audit). Filesystem/git-plumbing only - no live C++ build.
    "tests/unit/test_phase4_golden_tracking.py",
    # Phase 5: explicit dataset-mode-ownership guard proving Phase 2/Phase 4
    # own exactly the six original modes and Phase 5 owns exactly
    # soil_campbell. Static membership + source-scan only - no live C++
    # build, no golden data read.
    "tests/unit/test_phase5_dataset_ownership.py",
    # Phase 5 Part 3: soil_heat_campbell Python contract + source-relation
    # coverage (class (a)/(b) only - class (c) lives in the sibling
    # characterization module below, per F-52). Pure Python/SciPy, no live
    # C++ build, no golden data read.
    "tests/unit/test_phase5_soil_campbell_contract.py",
    # Phase 5 Part 3 scientific-triage pass (F-52): class (c)
    # cross-implementation CHARACTERIZATION (not parity) for
    # soil_heat_campbell against the committed Phase 5 golden - reads
    # committed golden CSVs, no live C++ build.
    "tests/unit/test_phase5_soil_campbell_characterization.py",
    # Campbell duff-forcing correction pass (F-69/F-53): class (a) Python
    # contract/source-relation coverage for the new _duff_burn_rate/
    # _duff_heat_fraction/_duff_burn_profile helpers, plus one class (c)
    # Campbell-outcome characterization test (soil_heat_campbell()
    # directly, monkeypatch-reconstructed pre-correction comparison, no
    # C++ golden/tolerance touched). Pure Python/SciPy, no live C++ build.
    "tests/unit/test_duff_forcing_correction.py",
    # Phase 5 Part 3: fail-closed completeness + git-trackability coverage
    # for the committed Phase 5 golden tree, mirroring
    # test_phase4_golden_tracking.py's state-independent contract.
    # Filesystem/git-plumbing only - no live C++ build.
    "tests/unit/test_phase5_golden_tracking.py",
    # Phase 6 investigation A: completeness + git-trackability coverage for
    # the committed Phase 6 golden tree, mirroring test_phase4/5_golden_
    # tracking.py's state-independent contract. Filesystem/git-plumbing
    # only - no live C++ build.
    "tests/unit/test_phase6_golden_tracking.py",
    # Phase 5 correction pass (item-3): AST-based guard preventing
    # test_phase5_soil_campbell_characterization.py from regressing to a
    # raw hardcoded pytest.approx()/sanity-envelope tolerance literal
    # instead of the two named constants in _phase5_contract.py. Pure
    # source-file static analysis - no live C++ build, no golden data read.
    "tests/unit/test_phase5_contract_hygiene.py",
    # Phase 6 investigation B: comprehensive Python contract/source-relation
    # coverage for soil_heat_massman() (Massman HMV, simplified). No C++
    # build, no golden data here - the pinned FOF_DLL solver DOES build/
    # link/run (see tests/cpp_parity_live/massman_fof_dll_probe.py, run
    # manually, never part of this suite), but produces non-finite output
    # under documented-bounds-compliant inputs (F-55/F-57/F-58), so this
    # module carries zero class (c) tests for a scientific, not a build,
    # reason.
    "tests/unit/test_massman_hmv_contract.py",
    # Phase 6 probe-hardening pass: contract tests for
    # massman_fof_dll_probe.py's fail-closed orchestration, using
    # injected/mocked BoundedResult values. Every run_bounded() call is
    # monkeypatched with a stub - no live C++ build, no real compiler
    # invoked, safe on any machine regardless of MSVC availability.
    "tests/cpp_parity_live/test_massman_fof_dll_probe.py",
    # Phase 7 item A: executable coverage for every real burnup_error code
    # _run_burnup_cell() can return - 10-16, 20-29, 90, 91, 99, all 20
    # codes with no carve-out - plus a completeness meta-test. Codes
    # 21/26/27 are genuinely reachable (not unreachable, as an earlier,
    # RETRACTED draft of this coverage originally claimed - see F-59,
    # gate0/04-findings.md): _check_fuel() reads the bound for each
    # FuelParticle attribute from the mutable module-level _FUEL_BOUNDS
    # dict on every call, so patching only the applicable bound entry
    # (never the validation/translation logic itself) drives the real
    # codes. Pure Python, no live C++ build.
    "tests/unit/test_run_burnup_cell_error_codes.py",
    # Phase 7 item B: characterization of the three inconsistent fire-
    # environment bounds-handling paths (_check_fire/_run_burnup_cell/
    # gen_burnup_in_file), including the exact fistart boundary and its
    # float neighbors. Pure Python, no live C++ build.
    "tests/unit/test_fire_environment_bounds.py",
    # Phase 7 item D: gen_burnup_in_file() content/field-order/numeric-
    # serialization/boundary/failure-mode coverage. Pure Python, no live
    # C++ build.
    "tests/unit/test_gen_burnup_in_file.py",
    # Phase 7 item C: 2D-input regression coverage for the former
    # atleast_1d -> ravel bug class, organized by public API family
    # (mortality/consumption/orchestrator), plus calc_carbon's
    # deliberately different shape-preserving contract. Pure Python, no
    # live C++ build.
    "tests/unit/test_2d_input_regression.py",
    # Phase 8 item A: mixed-validity array isolation (burnup per-cell
    # error isolation at the array level, plus mortality unsupported-
    # species non-contamination). Pure Python, no live C++ build.
    "tests/unit/test_phase8_array_isolation.py",
    # Phase 8 item B: serial (num_workers=1) vs parallel (num_workers>1,
    # ProcessPoolExecutor) equivalence for run_fofem_emissions(). Pure
    # Python/multiprocessing, no live C++ build.
    "tests/unit/test_phase8_serial_parallel_equivalence.py",
    # Phase 8 item C: run_fofem_mortality() facade integration coverage
    # (previously zero test coverage at any level). Pure Python, no live
    # C++ build.
    "tests/unit/test_phase8_mortality_facade.py",
    # Phase 8 item D: moisture-regime integration matrix through
    # run_fofem_emissions()/consm_duff(). Pure Python, no live C++ build.
    "tests/unit/test_phase8_moisture_regime_integration.py",
    # Phase 8 item E: SI/Imperial unit-system contract matrix. Pure
    # Python, no live C++ build.
    "tests/unit/test_phase8_unit_system_contract.py",
    # Phase 8 item F: runner/discovery completeness - proves every test
    # module is assigned to CORE or FULL exactly once and plain-pytest
    # discovery matches this file's own registration. Pure Python
    # introspection of this very module, no live C++ build.
    "tests/unit/test_phase8_runner_completeness.py",
    # Phase 8 item G: operational hardening (deterministic repeats,
    # warning behavior, bounded runtime, child-process/scratch/file-
    # handle cleanup, order independence, hostile-git-ownership support
    # reused from Phase 7's established pattern). Pure Python
    # introspection/execution, no live C++ build - the Phase 2-7 golden
    # --verify-only gates this item also requires are exercised directly
    # in the Phase 8 acceptance audit, not via a new CORE test module
    # (they need the live build, which would break CORE's no-toolchain
    # guarantee; the existing FULL_EXTRA_TESTS generator-driver modules
    # already cover them as pytest nodes).
    "tests/unit/test_phase8_operational_hardening.py",
    # 2026-09-18 xfail-disposition audit: meta-test guaranteeing
    # development/plans/gate0/08-xfail-disposition.csv exactly covers
    # the suite's currently-collected xfail nodes. Spawns its own bounded
    # subprocess restricted to 8 already-CORE test files, no live C++
    # build of its own.
    "tests/unit/test_xfail_disposition_audit.py",
    # 2026-09-21 F-62 partial-resolution regression coverage: synthetic,
    # golden-independent proof of the burnup() duff-remaining termination
    # gate fix. Pure Python, no live C++ build.
    "tests/unit/test_burnup_duff_smolder_continuation.py",
]

FULL_EXTRA_TESTS: List[str] = [
    "tests/cpp_parity_live/test_compare_cpp_python.py",     # was tests/test_compare_cpp_python.py
    "tests/cpp_parity_live/test_cpp_comparison.py",          # was tests/test_cpp_comparison.py
    # Opt-in soil_campbell intermediate-state diagnostic plus Python-side
    # comparison against the live reference implementation. Requires the
    # live C++ build like every other FULL_EXTRA_TESTS module here.
    "tests/cpp_parity_live/test_soil_solver_diagnostic_comparison.py",
    # Phase 2: builds and drives the live fofem_test C++ harness binary
    # directly (MSVC/CMake/Ninja required) — not a golden-CSV comparison.
    "tests/cpp_parity_live/test_cpp_harness_contract.py",
    # Phase 2: driver tests for the golden generator/promoter (wrong SHA,
    # qualification failure, harness failure, staleness, corruption,
    # manifest mismatch, determinism) — also needs the live build.
    "tests/cpp_parity_live/test_generate_phase2_goldens.py",
    # Phase 4: driver tests for the Phase 4 golden generator (pinned-SHA
    # gate, determinism, corrupted/missing/extra/mismatched committed files,
    # dataset field, and proof the Phase 2 tree is never touched). Needs the
    # live build.
    "tests/cpp_parity_live/test_generate_phase4_goldens.py",
    # Phase 5 Part 3: driver tests for the Phase 5 soil_campbell golden
    # generator (pinned-SHA gate, determinism, corrupted/missing/extra/
    # mismatched committed files, dataset field, and proof the Phase 2/
    # Phase 4 trees are never touched). Needs the live build.
    "tests/cpp_parity_live/test_generate_phase5_goldens.py",
    # Phase 6 investigation A: driver tests for the default-emissions-
    # equivalence consume golden generator (pinned-SHA gate, determinism,
    # corrupted/missing/extra/mismatched committed files, dataset field, and
    # proof the Phase 2/4/5 trees are never touched). Needs the live build.
    "tests/cpp_parity_live/test_generate_phase6_goldens.py",
    # Phase 7 item E: driver tests for the additional run_burnup golden
    # generator (pinned-SHA gate, determinism, corrupted/missing/extra/
    # mismatched committed files, dataset field, and proof the Phase 2/4/
    # 5/6 trees are never touched). Needs the live build.
    "tests/cpp_parity_live/test_generate_phase7_goldens.py",
    # Phase 7 narrow correction pass item 1: proves every Git invocation in
    # tests/prepare_cpp_reference.py (clone, refresh fetch/checkout/reset,
    # _print_status()'s rev-parse) succeeds under Git's dubious-ownership
    # check via GIT_TEST_ASSUME_DIFFERENT_OWNER=1, including a real
    # subprocess run of the full Phase 7 generator-driver suite under that
    # hostile environment. Needs the live build (same as the module above).
    "tests/unit/test_prepare_cpp_reference_git_ownership.py",
    # F-62 completion/acceptance-recovery pass (2026-09-21): a real
    # wheel-build/isolated-venv-install/import-from-outside-the-checkout
    # regression for the FOF_SPP.CSV packaging defect this pass fixed.
    # Genuinely slow (network/disk-bound wheel build + full dependency
    # install) — kept out of CORE_TESTS, unlike its sibling
    # test_runtime_data_resources.py, which only pins the corrected
    # package-data declaration.
    "tests/unit/test_packaging_wheel_install.py",
]

#: A deliberately small, representative subset of :data:`CORE_TESTS` for
#: ordinary pull-request feedback. Every path remains assigned to CORE or
#: FULL exactly once; this list is only a CI execution tier, never a third
#: ownership bucket.
CI_SMOKE_TESTS: List[str] = [
    "tests/unit/test_burnup_component_api.py",
    "tests/unit/test_consumption_golden.py",
    "tests/unit/test_run_unified_tests_contract.py",
    "tests/unit/test_runtime_data_resources.py",
    "tests/unit/test_tree_flame_contracts.py",
    "tests/unit/test_utility_contracts.py",
    "tests/integration/test_run_fofem_emissions.py",
    "tests/unit/test_2d_input_regression.py",
    "tests/unit/test_phase8_unit_system_contract.py",
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

    :param suite: ``'ci-smoke'`` for representative PR checks, ``'core'``
        for the publish-safe integration suite, or ``'full'`` to additionally
        include parity/comparison tests.
    :return: List of ``tests/*.py`` relative paths to run.
    """
    if suite == "ci-smoke":
        return list(CI_SMOKE_TESTS)

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
        choices=("ci-smoke", "core", "full"),
        default="core",
        help=(
            "ci-smoke: representative PR checks; core: publish-safe "
            "integration tests; full: includes parity/comparison tests."
        ),
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
