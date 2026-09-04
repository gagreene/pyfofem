#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase5_contract_hygiene.py - Correction-pass part-2 item-3 meta-test:
prevents ``tests/unit/test_phase5_soil_campbell_characterization.py`` from
regressing back to a raw, hardcoded ``pytest.approx(..., abs=<literal>)``
characterization-regression precision, which must instead reference the
centrally-defined ``CHARACTERIZATION_REGRESSION_PRECISION_DEGC`` constant in
``tests/cpp_parity_live/_phase5_contract.py``.

**Correction-pass part-3 note.** Part 2 also defined a
``CHARACTERIZATION_SANITY_ENVELOPE_DEGC`` "physically sane envelope" bound
and a corresponding hygiene check here. Independent review found the
envelope itself was tuned to the observed result (numerically equal to
``SOI-DUF-06``'s own measured max divergence), not an independently derived
bound, so part 3 deleted both the constant and its dedicated
characterization test rather than keep policing a value that was never
independently justified. This module's scope is therefore narrowed back to
the one constant that IS a genuine, non-tuned regression-stability
precision (see ``_phase5_contract.py``'s own docstring for that constant).

Scope is deliberately narrow: ONLY the characterization module, not its
companion ``test_phase5_soil_campbell_contract.py``. That companion module's
own ``pytest.approx(1230.0)``/``pytest.approx(0.157, rel=0.02)``-style calls
are class (b) source-relation checks against literal, evidence-cited C++
constants (F-51) - a different concern from "regression-stability precision
for a re-measured characterization divergence" this module's item-3 fix
targets. Applying the same rule there would misclassify legitimate,
already-justified literal comparisons as a hygiene violation.

This is an AST-based structural check (not a text/regex scan), so it cannot
be defeated by reformatting, and it cannot false-positive on an unrelated
numeric literal that happens to equal 0.05 elsewhere in the file (a
soil-moisture fraction, say) - it only inspects the ``abs=``/``rel=``
keyword arguments of ``pytest.approx(...)`` calls.

**Correction-pass part-4 addition (2026-09-04, F-53).** The characterization
module now carries one ``@pytest.mark.xfail`` strict-xfail test
(``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``)
pinning F-53's confirmed defect's DESIRED behaviour. Two further checks
mirror ``test_phase4_contract_hygiene.py``'s established pattern: every
``pytest.mark.xfail`` decorator in this module must be ``strict=True`` (a
non-strict xfail would silently keep passing even after the production fix
lands, defeating the whole point of pinning desired behaviour), and no
imperative ``pytest.xfail(...)`` call may appear anywhere in it (an
imperative call is not collected as a distinct, individually-reportable
xfail node the way a decorator is, and is exactly the anti-pattern Phase 4's
correction pass eliminated).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._phase5_contract import CHARACTERIZATION_REGRESSION_PRECISION_DEGC

_CHARACTERIZATION_MODULE_PATH = os.path.join(
    PROJECT_ROOT, "tests", "unit", "test_phase5_soil_campbell_characterization.py"
)


def _parse_characterization_module() -> ast.Module:
    """
    Parse the characterization module's source into an AST.

    :returns: The parsed module.
    """
    with open(_CHARACTERIZATION_MODULE_PATH, encoding="utf-8") as handle:
        source = handle.read()
    return ast.parse(source, filename=_CHARACTERIZATION_MODULE_PATH)


def _pytest_approx_calls(tree: ast.Module):
    """
    Yield every ``pytest.approx(...)`` call node in *tree*.

    :param tree: A parsed module AST.
    :returns: Generator of :class:`ast.Call` nodes.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "approx"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        ):
            yield node


def test_characterization_module_declares_and_imports_the_named_constant():
    """The characterization module must actually import the named constant
    (not merely have it available transitively) - a future edit that stops
    importing it while still hardcoding its value would defeat the whole
    point of centralising it."""
    tree = _parse_characterization_module()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    assert "CHARACTERIZATION_REGRESSION_PRECISION_DEGC" in imported_names, (
        "test_phase5_soil_campbell_characterization.py no longer imports "
        "CHARACTERIZATION_REGRESSION_PRECISION_DEGC from _phase5_contract.py"
    )


def test_characterization_module_has_no_imperative_pytest_xfail_call():
    """No ``pytest.xfail(...)`` call (imperative, mid-test-body form) may
    appear anywhere in the characterization module - only the declarative
    ``@pytest.mark.xfail(...)`` decorator form, which pytest can report,
    collect, and strict-check individually. Mirrors
    ``test_phase4_contract_hygiene.py``'s identical rule."""
    tree = _parse_characterization_module()
    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "xfail"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        ):
            offending.append(f"line {node.lineno}: imperative pytest.xfail(...) call")
    assert not offending, offending


def test_characterization_module_has_no_raw_pytest_approx_tolerance_literal():
    """Every ``pytest.approx(..., abs=...)``/``pytest.approx(..., rel=...)``
    call in the characterization module must reference a named identifier
    (the centrally-defined precision constant), never a raw numeric literal
    - the exact defect the part-2 correction pass fixed (``abs=0.05``
    hardcoded at 6 call sites)."""
    tree = _parse_characterization_module()
    offending = []
    for call in _pytest_approx_calls(tree):
        for keyword in call.keywords:
            if keyword.arg not in ("abs", "rel"):
                continue
            if isinstance(keyword.value, ast.Constant):
                offending.append(
                    f"line {call.lineno}: pytest.approx(..., {keyword.arg}="
                    f"{keyword.value.value!r}) is a raw literal, not a named constant"
                )
    assert not offending, offending
    assert CHARACTERIZATION_REGRESSION_PRECISION_DEGC > 0.0


def test_characterization_module_xfail_markers_are_all_strict():
    """Every ``@pytest.mark.xfail(...)`` decorator in the characterization
    module must set ``strict=True`` - a non-strict xfail silently keeps
    passing (with no signal) once the underlying defect is fixed, defeating
    the purpose of pinning DESIRED behaviour that should fail until F-53 is
    resolved."""
    tree = _parse_characterization_module()
    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            is_xfail_marker = (
                isinstance(func, ast.Attribute)
                and func.attr == "xfail"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "pytest"
            )
            if not is_xfail_marker:
                continue
            strict_kw = next(
                (kw for kw in decorator.keywords if kw.arg == "strict"), None
            )
            is_strict_true = (
                strict_kw is not None
                and isinstance(strict_kw.value, ast.Constant)
                and strict_kw.value.value is True
            )
            if not is_strict_true:
                offending.append(
                    f"{node.name} (line {node.lineno}): xfail marker is not strict=True"
                )
    assert offending == [], offending
    # A future edit that deletes the xfail entirely (rather than fixing the
    # underlying defect) would make this assertion trivially/vacuously true
    # with an empty `offending` list -- guard against that by requiring at
    # least one real xfail marker to have been found and checked.
    xfail_marker_count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "xfail"
    )
    assert xfail_marker_count >= 1, (
        "expected at least one @pytest.mark.xfail marker in the "
        "characterization module (F-53's desired-behaviour pin) - none found"
    )
