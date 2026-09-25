#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_runner_completeness.py - Phase 8 item F: runner/discovery
completeness.

``run_unified_tests.py`` already has a durable completeness mechanism,
``_validate_suite_coverage()``, which recursively globs ``tests/**/
test_*.py`` and fails loudly if any discovered module is absent from
``CORE_TESTS``/``FULL_EXTRA_TESTS`` - it is NOT marker-based or naming-
convention-based, so it cannot be silently defeated by a module that
merely fails to match a pattern. This module (a) proves that mechanism
actually holds for the REAL, current configuration (previously only
exercised implicitly via ``main()``, never asserted directly by a test)
and (b) adds the properties it does NOT itself prove: exactly-once
assignment (no module registered in both lists), plain-pytest-vs-runner
discovery agreement, and that Phase 8's own new modules are registered.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

import tests.run_unified_tests as unified
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._proc import run_bounded

#: The 7 modules Phase 8 itself adds - an explicit, defensive check
#: independent of (in addition to) the glob-based mechanism above, so a
#: future change to the glob/discovery logic cannot silently drop this
#: guarantee for these specific files.
_AUDITED_MODULE_PATHS = (
    "tests/unit/test_array_isolation.py",
    "tests/unit/test_serial_parallel_equivalence.py",
    "tests/unit/test_mortality_facade.py",
    "tests/unit/test_moisture_regime_integration.py",
    "tests/unit/test_unit_system_contract.py",
    "tests/unit/test_runner_completeness.py",
    "tests/unit/test_operational_hardening.py",
)

#: Bound for the plain-pytest collection subprocess this module spawns.
_COLLECT_TIMEOUT_S = 120.0


def _collect_only_file_set() -> set:
    """
    Run ``python -m pytest --collect-only -q`` as a bounded subprocess
    and return the set of test-file paths (repo-relative, forward-slash)
    it collected.

    :return: Set of collected test-file paths.
    :raises AssertionError: If the subprocess fails operationally.
    """
    result = run_bounded(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        timeout=_COLLECT_TIMEOUT_S, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"plain pytest --collect-only failed (rc={result.returncode}): {result.stderr}"
    )
    files = set()
    for line in result.stdout.splitlines():
        if "::" not in line:
            continue
        file_part = line.split("::", 1)[0].strip()
        if file_part.startswith("tests/") and file_part.endswith(".py"):
            files.add(file_part.replace(os.sep, "/"))
    return files


def _has_skip_construct(path: str) -> list:
    """
    AST-scan *path* for an imperative ``pytest.skip()``/
    ``pytest.importorskip()`` call or a ``@pytest.mark.skip``/
    ``@pytest.mark.skipif`` decorator.

    :param path: Repo-relative file path to scan.
    :return: List of human-readable finding strings (empty if clean).
    """
    full_path = os.path.join(PROJECT_ROOT, path)
    tree = ast.parse(open(full_path, encoding="utf-8").read(), filename=path)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("skip", "importorskip"):
                findings.append(f"{path}:{node.lineno}: imperative pytest.{name}() call")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if "skipif" in dec_str or ("'skip'" in dec_str and "mark" in dec_str):
                    findings.append(f"{path}:{dec.lineno}: @pytest.mark.skip/skipif decorator")
    return findings


def test_core_and_full_extra_have_no_duplicate_entries_within_themselves():
    """Neither ``CORE_TESTS`` nor ``FULL_EXTRA_TESTS`` may list the same
    module path twice - each list is a set in spirit and must behave
    like one."""
    assert len(unified.CORE_TESTS) == len(set(unified.CORE_TESTS)), 'CORE_TESTS has duplicate entries'
    assert len(unified.FULL_EXTRA_TESTS) == len(set(unified.FULL_EXTRA_TESTS)), (
        'FULL_EXTRA_TESTS has duplicate entries'
    )


def test_core_and_full_extra_have_zero_overlap():
    """A module assigned to CORE must not ALSO be assigned to
    FULL_EXTRA_TESTS - each module is assigned to exactly one suite
    tier, never both (which would run it twice under ``--suite full``)."""
    overlap = set(unified.CORE_TESTS) & set(unified.FULL_EXTRA_TESTS)
    assert not overlap, f'modules registered in BOTH CORE_TESTS and FULL_EXTRA_TESTS: {sorted(overlap)}'


def test_every_configured_test_file_exists_on_disk():
    """Every path named in ``CORE_TESTS``/``FULL_EXTRA_TESTS`` must
    correspond to a real file - a stale/renamed entry must not silently
    vanish from the effective suite."""
    for path in unified.CORE_TESTS + unified.FULL_EXTRA_TESTS:
        assert os.path.isfile(os.path.join(unified._repo_root(), path)), (
            f'configured test file does not exist: {path}'
        )


def test_no_new_skips_in_the_audited_module_surface():
    """None of Phase 8's own 7 new modules may introduce an imperative
    ``pytest.skip()``/``pytest.importorskip()`` call or a
    ``@pytest.mark.skip``/``@pytest.mark.skipif`` decorator - "no new
    skips" per item F, verified by AST scan, not by convention."""
    all_findings = []
    for path in _AUDITED_MODULE_PATHS:
        all_findings.extend(_has_skip_construct(path))
    assert not all_findings, 'unexpected skip construct(s) found:\n' + '\n'.join(all_findings)


def test_audited_modules_are_registered_in_core():
    """Every one of Phase 8's own 7 new modules must be registered in
    ``CORE_TESTS`` (none require a live C++ build, so none belong in
    ``FULL_EXTRA_TESTS``)."""
    for path in _AUDITED_MODULE_PATHS:
        assert path in unified.CORE_TESTS, f'{path} is not registered in CORE_TESTS'


def test_plain_pytest_collection_matches_the_configured_file_set():
    """Plain ``pytest --collect-only`` (using ``pyproject.toml``'s own
    ``testpaths = ["tests"]``, the same discovery real developers get)
    must collect EXACTLY the same file set
    ``run_unified_tests.py``'s ``CORE_TESTS``/``FULL_EXTRA_TESTS``
    configure - proving plain pytest, core, and full discovery are
    internally consistent, not merely assumed to align."""
    collected = _collect_only_file_set()
    configured = set(unified.CORE_TESTS) | set(unified.FULL_EXTRA_TESTS)
    missing_from_config = collected - configured
    missing_from_collection = configured - collected
    assert not missing_from_config, (
        f'plain pytest collects file(s) not registered in run_unified_tests.py: {sorted(missing_from_config)}'
    )
    assert not missing_from_collection, (
        f'run_unified_tests.py registers file(s) plain pytest never collects: {sorted(missing_from_collection)}'
    )


def test_resolve_tests_core_equals_core_tests_only():
    """``_resolve_tests('core')`` must equal exactly ``CORE_TESTS``,
    excluding every ``FULL_EXTRA_TESTS`` entry (full/live-reference
    work)."""
    assert unified._resolve_tests('core') == unified.CORE_TESTS
    assert not (set(unified._resolve_tests('core')) & set(unified.FULL_EXTRA_TESTS))


def test_ci_smoke_is_a_duplicate_free_subset_of_core():
    """The fast CI tier may select CORE tests but cannot redefine ownership."""
    smoke = unified._resolve_tests('ci-smoke')
    assert smoke == unified.CI_SMOKE_TESTS
    assert len(smoke) == len(set(smoke))
    assert set(smoke) <= set(unified.CORE_TESTS)


def test_resolve_tests_full_equals_core_plus_full_extra_with_no_duplicates():
    """``_resolve_tests('full')`` must equal exactly ``CORE_TESTS +
    FULL_EXTRA_TESTS`` (order preserved, nothing added or dropped), and
    the combined list must contain no duplicate entries."""
    resolved_full = unified._resolve_tests('full')
    assert resolved_full == unified.CORE_TESTS + unified.FULL_EXTRA_TESTS
    assert len(resolved_full) == len(set(resolved_full)), 'full suite resolution has duplicate entries'


def test_validate_suite_coverage_passes_for_the_real_configuration():
    """``_validate_suite_coverage()`` must NOT raise against the real,
    current repository state - no test module was accidentally omitted
    from either suite because of recursive-discovery, naming, marker, or
    registration drift. This exercises the mechanism directly rather
    than only implicitly through ``main()``."""
    unified._validate_suite_coverage()
