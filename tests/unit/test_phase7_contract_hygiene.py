#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase7_contract_hygiene.py - Phase 7 correction pass item 6: a
static AST meta-test proving the complete Phase 7 test/infrastructure
surface contains NO new skips - no imperative ``pytest.skip()``/
``pytest.importorskip()`` calls and no ``@pytest.mark.skip``/
``@pytest.mark.skipif`` decorators anywhere.

This scan is AST-based, not a regex or plain substring search,
specifically so a docstring or comment merely MENTIONING ``pytest.skip``
(as several Phase 7 modules' own docstrings do, explaining exactly this
rule) can never be mistaken for a real call or marker.

**Phase 7 narrow correction pass (2026-09-08), item 4**: also proves the
complete surface contains no ``ignore_errors=True`` keyword argument on
ANY call - the fail-open cleanup pattern this pass removed from
``_scratch.py``/``test_phase7_scratch.py`` (which left real, reviewer-
observed leaked directories behind) must not silently return via a
future addition. AST-based for the same reason: a docstring/comment
mentioning the phrase (several now do, describing the fix) must never be
mistaken for a real keyword argument.

Scope: the explicitly-enumerated Phase 7 file surface in
:data:`PHASE7_MODULE_PATHS` (not a naming-convention glob, since these
files do not all share one filename fragment - "gen_burnup_in_file",
"fire_environment_bounds", and "2d_input_regression" have no "phase7" in
their names at all, unlike the golden-tracking/run_burnup_parity/
scratch/contract/generator modules). That set's own completeness is
backstopped, not guaranteed, by
:func:`test_phase7_module_paths_matches_every_python_file_mentioning_phase_7`
below - see that test's docstring for the precise, honest scope of what
a literal-text scan can and cannot prove.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os

from tests._support import PROJECT_ROOT

#: The explicitly-enumerated Phase 7 file surface this module audits -
#: every test module AND every piece of Phase 7 infrastructure (contract,
#: generator, scratch utility) introduced across the original Phase 7
#: pass and every correction pass since, PLUS shared/earlier-phase
#: infrastructure Phase 7 itself substantively edited (``_golden_manifest.py``
#: gained ``git_safe_directory_value()``; ``test_golden_manifest_validator.py``
#: gained its regression tests; ``run_unified_tests.py`` gained Phase 7's own
#: ``CORE_TESTS``/``FULL_EXTRA_TESTS`` entries and comments; the Phase 4/5/6
#: golden-tracking modules' own ``_git()`` helpers gained the same
#: ``safe.directory`` qualification during the 2026-09-09 final
#: cross-phase acceptance-gate correction - all found and added exactly
#: this way, per direct evidence:
#: :func:`test_phase7_module_paths_matches_every_python_file_mentioning_phase_7`
#: below cross-checks this set against every file whose own text CURRENTLY
#: mentions "Phase 7", so an omission that also happens to use that literal
#: marker fails loudly instead of silently narrowing this audit again - a
#: real backstop for the common case, not a guarantee against every
#: possible future omission (a substantive edit that never writes the
#: words "Phase 7" would not be caught this way). Explicit, not globbed:
#: several of these files share no common filename fragment with "phase7".
PHASE7_MODULE_PATHS = tuple(
    os.path.join(PROJECT_ROOT, *parts) for parts in (
        ("tests", "prepare_cpp_reference.py"),
        ("tests", "run_unified_tests.py"),
        ("tests", "unit", "test_run_burnup_cell_error_codes.py"),
        ("tests", "unit", "test_fire_environment_bounds.py"),
        ("tests", "unit", "test_gen_burnup_in_file.py"),
        ("tests", "unit", "test_2d_input_regression.py"),
        ("tests", "unit", "test_phase4_golden_tracking.py"),
        ("tests", "unit", "test_phase5_golden_tracking.py"),
        ("tests", "unit", "test_phase6_golden_tracking.py"),
        ("tests", "unit", "test_phase7_run_burnup_parity.py"),
        ("tests", "unit", "test_phase7_golden_tracking.py"),
        ("tests", "unit", "test_phase7_scratch.py"),
        ("tests", "unit", "test_phase7_contract_hygiene.py"),
        ("tests", "unit", "test_prepare_cpp_reference_git_ownership.py"),
        ("tests", "unit", "test_golden_manifest_validator.py"),
        ("tests", "cpp_parity_live", "_golden_manifest.py"),
        ("tests", "cpp_parity_live", "_phase7_contract.py"),
        ("tests", "cpp_parity_live", "_scratch.py"),
        ("tests", "cpp_parity_live", "generate_phase7_goldens.py"),
        ("tests", "cpp_parity_live", "test_generate_phase7_goldens.py"),
    )
)

#: Real, existing files under ``tests/`` whose own text mentions "Phase 7"
#: but that this module deliberately does NOT audit, each with a verified
#: reason - checked by
#: :func:`test_phase7_module_paths_matches_every_python_file_mentioning_phase_7`
#: below, so an exclusion can never silently go stale (e.g. because the
#: file was later, legitimately, made Phase-7-owned).
PHASE7_MENTION_EXCLUSIONS = frozenset({
    os.path.join(PROJECT_ROOT, *parts) for parts in (
        # Phase 3's own file. Its module docstring names "Phase 7" only as
        # a forward-reference scope boundary ("the three burnup_calcs
        # underscore exports ... are assigned to Phase 7 ... so they are
        # not covered here either") - Phase 7 itself never edited this
        # file's executable content.
        ("tests", "unit", "test_burnup_component_api.py"),
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


def _phase7_mentioning_python_files() -> set:
    """
    Return every ``.py`` file under ``tests/`` whose own text contains
    the literal substring ``"Phase 7"`` - a real, independently-derived
    candidate set for :data:`PHASE7_MODULE_PATHS`'s completeness (see
    :func:`test_phase7_module_paths_matches_every_python_file_mentioning_phase_7`).

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


def test_phase7_module_paths_matches_every_python_file_mentioning_phase_7():
    """
    :data:`PHASE7_MODULE_PATHS` plus the explicit, justified
    :data:`PHASE7_MENTION_EXCLUSIONS` must account for EXACTLY every
    ``.py`` file under ``tests/`` whose own text CURRENTLY mentions
    "Phase 7" - a mention-based completeness BACKSTOP/candidate-set
    check derived from real, present-tense file contents, not a second
    hand-maintained list copied from the same source used to build
    ``PHASE7_MODULE_PATHS`` itself.

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
    mentioning = _phase7_mentioning_python_files()
    audited = set(PHASE7_MODULE_PATHS)
    overlap = audited & PHASE7_MENTION_EXCLUSIONS
    assert not overlap, (
        "file(s) are both audited AND explicitly excluded - remove from "
        f"one or the other: {sorted(os.path.relpath(p, PROJECT_ROOT) for p in overlap)}"
    )
    missing = mentioning - (audited | PHASE7_MENTION_EXCLUSIONS)
    assert not missing, (
        "file(s) mention 'Phase 7' but are neither audited in "
        "PHASE7_MODULE_PATHS nor explicitly excluded in "
        f"PHASE7_MENTION_EXCLUSIONS: "
        f"{sorted(os.path.relpath(p, PROJECT_ROOT) for p in missing)}"
    )
    stale_exclusions = PHASE7_MENTION_EXCLUSIONS - mentioning
    assert not stale_exclusions, (
        "excluded file(s) no longer mention 'Phase 7' at all - the "
        f"exclusion is stale and should be removed: "
        f"{sorted(os.path.relpath(p, PROJECT_ROOT) for p in stale_exclusions)}"
    )


def test_phase7_module_surface_is_complete():
    """
    Every path in :data:`PHASE7_MODULE_PATHS` must actually exist -
    guards against a typo silently narrowing this hygiene check's own
    coverage.

    :returns: None.
    """
    missing = [path for path in PHASE7_MODULE_PATHS if not os.path.isfile(path)]
    assert not missing, f"PHASE7_MODULE_PATHS names missing file(s): {missing}"


def test_phase7_modules_have_no_ignore_errors_true():
    """
    No Phase 7 module may call anything with ``ignore_errors=True`` -
    every cleanup path must be fail-closed (a real removal failure
    raises), never fail-open (silently swallowed, which previously left
    real, reviewer-observed leaked scratch directories behind).

    :returns: None.
    """
    findings = []
    for path in PHASE7_MODULE_PATHS:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_ignore_errors_calls(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "ignore_errors=True call(s) found:\n" + "\n".join(findings)


def test_phase7_modules_have_no_imperative_skip_calls():
    """
    No Phase 7 module may call ``pytest.skip()`` or
    ``pytest.importorskip()`` anywhere - missing prerequisites (e.g. the
    live MSVC/CMake/Ninja toolchain) must fail closed with a diagnostic
    instead (see ``test_generate_phase7_goldens.py``'s
    ``_require_toolchain`` fixture).

    :returns: None.
    """
    findings = []
    for path in PHASE7_MODULE_PATHS:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_skip_calls(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "imperative pytest.skip()/importorskip() call(s) found:\n" + "\n".join(
        findings
    )


def test_phase7_modules_have_no_skip_markers():
    """
    No Phase 7 module may carry an ``@pytest.mark.skip`` or
    ``@pytest.mark.skipif`` decorator anywhere.

    :returns: None.
    """
    findings = []
    for path in PHASE7_MODULE_PATHS:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        findings.extend(_find_skip_markers(tree, os.path.relpath(path, PROJECT_ROOT)))
    assert not findings, "@pytest.mark.skip/skipif marker(s) found:\n" + "\n".join(findings)
