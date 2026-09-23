#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_burnup_extended_skip_hygiene.py - Phase 7 correction pass item 6: a
static AST meta-test proving the complete ``burnup_extended``-era test/
infrastructure surface contains NO new skips - no imperative
``pytest.skip()``/``pytest.importorskip()`` calls and no
``@pytest.mark.skip``/``@pytest.mark.skipif`` decorators anywhere.

This scan is AST-based, not a regex or plain substring search,
specifically so a docstring or comment merely MENTIONING ``pytest.skip``
(as several of these modules' own docstrings do, explaining exactly this
rule) can never be mistaken for a real call or marker.

**Phase 7 narrow correction pass (2026-09-08), item 4**: also proves the
complete surface contains no ``ignore_errors=True`` keyword argument on
ANY call - the fail-open cleanup pattern this pass removed from
``_scratch.py``/``test_repo_local_scratch.py`` (which left real, reviewer-
observed leaked directories behind) must not silently return via a
future addition. AST-based for the same reason: a docstring/comment
mentioning the phrase (several now do, describing the fix) must never be
mistaken for a real keyword argument.

Scope: the explicitly-enumerated file surface in
:data:`BURNUP_EXTENDED_MODULE_PATHS` (not a naming-convention glob, since
these files do not all share one filename fragment - "gen_burnup_in_file",
"fire_environment_bounds", and "2d_input_regression" have no "phase7"/
"burnup_extended" in their names at all, unlike the golden-tracking/
burnup-parity/scratch/contract/generator modules). That set's own
completeness is backstopped, not guaranteed, by
:func:`test_module_paths_matches_every_python_file_mentioning_phase_7`
below - see that test's docstring for the precise, honest scope of what
a literal-text scan can and cannot prove. The scan target is still the
literal historical text ``"Phase 7"`` (not "burnup_extended"): that
substring remains the real, unrenamed marker across this repo's dated
project-history prose (test-suite renaming plan, Slice 6, leaves that
narrative untouched), so it is what a real completeness backstop must
search for.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os

import pytest

from tests._support import PROJECT_ROOT

#: The explicitly-enumerated file surface this module audits - every test
#: module AND every piece of infrastructure (contract, generator, scratch
#: utility) introduced across the original Phase 7 pass and every
#: correction pass since, PLUS shared/earlier-phase infrastructure Phase 7
#: itself substantively edited (``_golden_manifest.py`` gained
#: ``git_safe_directory_value()``; ``test_golden_manifest_validator.py``
#: gained its regression tests; ``run_unified_tests.py`` gained Phase 7's
#: own ``CORE_TESTS``/``FULL_EXTRA_TESTS`` entries and comments) - all
#: found and added exactly this way, per direct evidence:
#: :func:`test_module_paths_matches_every_python_file_mentioning_phase_7`
#: below cross-checks this set against every file whose own text CURRENTLY
#: mentions "Phase 7", so an omission that also happens to use that literal
#: marker fails loudly instead of silently narrowing this audit again - a
#: real backstop for the common case, not a guarantee against every
#: possible future omission (a substantive edit that never writes the
#: words "Phase 7" would not be caught this way). Explicit, not globbed:
#: several of these files share no common filename fragment with
#: "phase7"/"burnup_extended". Paths updated for the test-suite renaming
#: plan's Slice 3 (the four golden-tracking modules merged into one
#: ``tests/unit/cpp/test_golden_tracking.py``; the moved
#: burnup_extended-specific modules now live under ``tests/unit/cpp/``).
BURNUP_EXTENDED_MODULE_PATHS = tuple(
    os.path.join(PROJECT_ROOT, *parts) for parts in (
        ("tests", "prepare_cpp_reference.py"),
        ("tests", "run_unified_tests.py"),
        ("tests", "unit", "test_run_burnup_cell_error_codes.py"),
        ("tests", "unit", "test_fire_environment_bounds.py"),
        ("tests", "unit", "test_gen_burnup_in_file.py"),
        ("tests", "unit", "test_2d_input_regression.py"),
        ("tests", "unit", "cpp", "test_golden_tracking.py"),
        ("tests", "unit", "cpp", "test_burnup_extended_parity.py"),
        ("tests", "unit", "cpp", "test_repo_local_scratch.py"),
        ("tests", "unit", "cpp", "test_burnup_extended_skip_hygiene.py"),
        ("tests", "unit", "test_prepare_cpp_reference_git_ownership.py"),
        ("tests", "unit", "test_golden_manifest_validator.py"),
        ("tests", "cpp_parity_live", "_golden_manifest.py"),
        ("tests", "cpp_parity_live", "_burnup_extended_contract.py"),
        ("tests", "cpp_parity_live", "_scratch.py"),
        ("tests", "cpp_parity_live", "generate_burnup_extended_goldens.py"),
        # test_generate_phase7_goldens.py was consolidated (test-suite
        # renaming plan, Slice 1) into a single dataset-parametrized
        # test_generate_goldens.py shared across all four scenario-matrix
        # datasets; it still carries Phase 7's two genuine behavioral
        # differences (fail-closed-not-skip; repo-local tmp_path), so it
        # replaces the old file in this audited set rather than dropping out.
        ("tests", "cpp_parity_live", "test_generate_goldens.py"),
    )
)

#: Real, existing files under ``tests/`` whose own text mentions "Phase 7"
#: but that this module deliberately does NOT audit, each with a verified
#: reason - checked by
#: :func:`test_module_paths_matches_every_python_file_mentioning_phase_7`
#: below, so an exclusion can never silently go stale (e.g. because the
#: file was later, legitimately, made Phase-7-owned).
BURNUP_EXTENDED_MENTION_EXCLUSIONS = frozenset({
    os.path.join(PROJECT_ROOT, *parts) for parts in (
        # Phase 3's own file. Its module docstring names "Phase 7" only as
        # a forward-reference scope boundary ("the three burnup_calcs
        # underscore exports ... are assigned to Phase 7 ... so they are
        # not covered here either") - Phase 7 itself never edited this
        # file's executable content.
        ("tests", "unit", "test_burnup_component_api.py"),
        # Phase 8's own files. Both merely CITE Phase 7 work as context
        # (test_array_isolation.py's docstring points at "Phase 7
        # item A" as the sibling per-cell-error-code coverage it builds
        # on; test_operational_hardening.py's docstring names
        # "the Phase 7 acceptance-gate correction" as the origin of the
        # hostile-Git-ownership pattern it reuses) - neither file was
        # itself edited by Phase 7, so neither belongs in
        # BURNUP_EXTENDED_MODULE_PATHS.
        ("tests", "unit", "test_array_isolation.py"),
        ("tests", "unit", "test_operational_hardening.py"),
    )
})

#: Decorator names that mark a test skipped (rather than xfailed or run).
_SKIP_MARKER_NAMES = frozenset({"skip", "skipif"})

#: Call names that impose an imperative skip at runtime.
_SKIP_CALL_NAMES = frozenset({"skip", "importorskip"})


def _decorator_marker_name(decorator: ast.expr):
    """
    Return the ``pytest.mark.<name>`` marker name a decorator node
    invokes or references, or ``None`` if it is not a ``pytest.mark.*``
    decorator at all.

    :param decorator: One AST decorator expression.
    :return: The marker name (e.g. ``"skipif"``), or ``None``.
    """
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "pytest"
            and target.value.attr == "mark"
    ):
        return target.attr
    return None


def _find_ignore_errors_calls(tree: ast.AST, source_path: str) -> list:
    """
    Return every call site anywhere in *tree* that passes ``ignore_errors=
    True`` as a keyword argument (e.g. a fail-open
    ``shutil.rmtree(path, ignore_errors=True)``) - the exact fail-open
    cleanup pattern this correction pass removed.

    :param tree: Parsed module AST.
    :param source_path: Path used only to build a readable location string.
    :return: List of ``"<path>:<line>: ignore_errors=True"`` strings.
    """
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                    keyword.arg == "ignore_errors"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
            ):
                findings.append(f"{source_path}:{node.lineno}: ignore_errors=True")
    return findings


def _find_skip_calls(tree: ast.AST, source_path: str) -> list:
    """
    Return every imperative ``pytest.skip(...)``/``pytest.importorskip(...)``
    call site found by walking *tree*.

    :param tree: Parsed module AST.
    :param source_path: Path used only to build a readable location string.
    :return: List of ``"<path>:<line>: pytest.<name>(...)"`` strings.
    """
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
                and func.attr in _SKIP_CALL_NAMES
        ):
            findings.append(f"{source_path}:{node.lineno}: pytest.{func.attr}(...)")
    return findings


def _find_skip_markers(tree: ast.AST, source_path: str) -> list:
    """
    Return every ``@pytest.mark.skip``/``@pytest.mark.skipif`` decorator
    found on any function/class definition in *tree*.

    :param tree: Parsed module AST.
    :param source_path: Path used only to build a readable location string.
    :return: List of ``"<path>:<line>: @pytest.mark.<name>"`` strings.
    """
    findings = []
    for node in ast.walk(tree):
        decorators = getattr(node, "decorator_list", None)
        if not decorators:
            continue
        for decorator in decorators:
            name = _decorator_marker_name(decorator)
            if name in _SKIP_MARKER_NAMES:
                findings.append(f"{source_path}:{decorator.lineno}: @pytest.mark.{name}")
    return findings


def _mentioning_python_files() -> set:
    """
    Return every ``.py`` file under ``tests/`` whose own text contains
    the literal substring ``"Phase 7"`` - a real, independently-derived
    candidate set for :data:`BURNUP_EXTENDED_MODULE_PATHS`'s completeness
    (see :func:`test_module_paths_matches_every_python_file_mentioning_phase_7`).

    :return: Set of absolute file paths.
    """
    found = set()
    tests_root = os.path.join(PROJECT_ROOT, "tests")
    for dirpath, _dirs, files in os.walk(tests_root):
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            with open(full, encoding="utf-8") as handle:
                if "Phase 7" in handle.read():
                    found.add(full)
    return found


def test_module_paths_matches_every_python_file_mentioning_phase_7():
    """
    :data:`BURNUP_EXTENDED_MODULE_PATHS` plus the explicit, justified
    :data:`BURNUP_EXTENDED_MENTION_EXCLUSIONS` must account for EXACTLY
    every ``.py`` file under ``tests/`` whose own text CURRENTLY mentions
    "Phase 7" - a mention-based completeness BACKSTOP/candidate-set
    check derived from real, present-tense file contents, not a second
    hand-maintained list copied from the same source used to build
    ``BURNUP_EXTENDED_MODULE_PATHS`` itself.

    **Narrow correction pass (2026-09-08, second round)**: this is
    deliberately NOT called a "completeness oracle" - independent review
    correctly pointed out that framing overclaims what a literal-text
    scan can prove. It proves only that every file mentioning "Phase 7"
    TODAY is accounted for; it cannot prove a FUTURE Phase-7-related file
    will use that literal marker at all (a substantive edit that never
    happens to write the words "Phase 7" would escape this check
    entirely, silently). It is a real, valuable backstop for the common
    case - exactly how ``_golden_manifest.py``/
    ``test_golden_manifest_validator.py``/``run_unified_tests.py`` were
    caught as omissions during this same narrow correction pass - not a
    guarantee covering every possible future omission.

    :returns: None.
    """
    mentioning = _mentioning_python_files()
    audited = set(BURNUP_EXTENDED_MODULE_PATHS)
    overlap = audited & BURNUP_EXTENDED_MENTION_EXCLUSIONS
    assert not overlap, (
        "file(s) are both audited AND explicitly excluded - remove from "
        f"one or the other: {sorted(os.path.relpath(p, PROJECT_ROOT) for p in overlap)}"
    )
    missing = mentioning - (audited | BURNUP_EXTENDED_MENTION_EXCLUSIONS)
    assert not missing, (
        "file(s) mention 'Phase 7' but are neither audited in "
        "BURNUP_EXTENDED_MODULE_PATHS nor explicitly excluded in "
        f"BURNUP_EXTENDED_MENTION_EXCLUSIONS: "
        f"{sorted(os.path.relpath(p, PROJECT_ROOT) for p in missing)}"
    )
    stale_exclusions = BURNUP_EXTENDED_MENTION_EXCLUSIONS - mentioning
    assert not stale_exclusions, (
        "excluded file(s) no longer mention 'Phase 7' at all - the "
        f"exclusion is stale and should be removed: "
        f"{sorted(os.path.relpath(p, PROJECT_ROOT) for p in stale_exclusions)}"
    )


def test_module_surface_is_complete():
    """
    Every path in :data:`BURNUP_EXTENDED_MODULE_PATHS` must actually
    exist - guards against a typo silently narrowing this hygiene check's
    own coverage.

    :returns: None.
    """
    missing = [path for path in BURNUP_EXTENDED_MODULE_PATHS if not os.path.isfile(path)]
    assert not missing, f"BURNUP_EXTENDED_MODULE_PATHS names missing file(s): {missing}"


def test_modules_have_no_ignore_errors_true():
    """
    No module in this audited surface may call anything with
    ``ignore_errors=True`` - every cleanup path must be fail-closed (a
    real removal failure raises), never fail-open (silently swallowed,
    which previously left real, reviewer-observed leaked scratch
    directories behind).

    :returns: None.
    """
    findings = []
    for path in BURNUP_EXTENDED_MODULE_PATHS:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_ignore_errors_calls(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "ignore_errors=True call(s) found:\n" + "\n".join(findings)


#: ``test_generate_goldens.py`` is a narrow, DOCUMENTED exception to this
#: check (test-suite renaming plan, Slice 1): it consolidates the
#: generator-driver tests for FOUR datasets, three of which (expanded_matrix,
#: soil_campbell, emissions_equivalence) legitimately skip on a missing
#: toolchain exactly as they always did as separate modules. Its one
#: ``pytest.skip(...)`` call site lives in ``_toolchain_gate``, a
#: function-scoped ``autouse`` fixture that checks
#: ``case.fail_closed_no_skip`` FIRST and calls ``pytest.fail(...)`` (never
#: reaching the skip line) whenever the case under test is
#: ``burnup_extended`` - the real safety property this hygiene check
#: protects (burnup_extended never silently skips) is preserved by that
#: runtime guard, verified directly by
#: :func:`test_burnup_extended_dataset_always_fails_closed_never_skips`
#: below, not by this AST scan (which cannot see the per-case guard).
_SKIP_CALL_EXEMPT_MODULES = frozenset({
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "test_generate_goldens.py"),
})


def test_burnup_extended_dataset_always_fails_closed_never_skips():
    """
    The real safety property :func:`test_modules_have_no_imperative_
    skip_calls`'s exemption for ``test_generate_goldens.py`` relies on:
    with the toolchain reported unavailable, the ``burnup_extended`` case's
    ``_toolchain_gate`` fixture must call ``pytest.fail`` (never
    ``pytest.skip``) - proved directly by invoking the fixture function
    with a monkeypatched ``toolchain_status``, not by inference from the
    source text.

    :returns: None.
    """
    import tests.cpp_parity_live.test_generate_goldens as tgg

    burnup_extended_case = next(c for c in tgg._CASES if c.label == "burnup_extended")
    real_status = tgg.toolchain_status
    tgg.toolchain_status = lambda: (False, "simulated: toolchain unavailable")
    try:
        with pytest.raises(pytest.fail.Exception):
            tgg._toolchain_gate.__wrapped__(burnup_extended_case)
    finally:
        tgg.toolchain_status = real_status


def test_modules_have_no_imperative_skip_calls():
    """
    No module in this audited surface may call ``pytest.skip()`` or
    ``pytest.importorskip()`` anywhere - missing prerequisites (e.g. the
    live MSVC/CMake/Ninja toolchain) must fail closed with a diagnostic
    instead - EXCEPT the narrow, documented exemption in
    :data:`_SKIP_CALL_EXEMPT_MODULES`.

    :returns: None.
    """
    findings = []
    for path in BURNUP_EXTENDED_MODULE_PATHS:
        if path in _SKIP_CALL_EXEMPT_MODULES:
            continue
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_skip_calls(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "imperative pytest.skip()/importorskip() call(s) found:\n" + "\n".join(
        findings
    )


def test_modules_have_no_skip_markers():
    """
    No module in this audited surface may carry an ``@pytest.mark.skip``
    or ``@pytest.mark.skipif`` decorator anywhere.

    :returns: None.
    """
    findings = []
    for path in BURNUP_EXTENDED_MODULE_PATHS:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_skip_markers(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "@pytest.mark.skip/skipif marker(s) found:\n" + "\n".join(findings)
