#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_expanded_matrix_xfail_hygiene.py - Meta-test enforcing the
``expanded_matrix`` dataset's xfail discipline: every scenario a
production defect prevents from passing must be pinned by a REAL
executable assertion of the DESIRED (post-fix) behaviour, decorated
``@pytest.mark.xfail(strict=True, ...)``, never declared unconditionally
with an imperative ``pytest.xfail(...)`` call.

An imperative ``pytest.xfail()`` call is a permanent, unconditional
surrender: it never executes a real assertion, so it can never turn into a
strict XPASS when the underlying defect is fixed, and a reader cannot tell
whether it is still accurate without re-deriving the defect from source. A
non-``strict`` ``@pytest.mark.xfail`` marker has the softer version of the
same defect: it silently tolerates the defect being fixed (XPASS) without
failing the suite, so nobody is forced to remove the stale marker.

This module performs a static AST scan, not a regex, specifically so a
docstring or comment merely MENTIONING ``pytest.xfail`` (as this module's
own docstring does, and as several ``expanded_matrix`` modules' docstrings
do when explaining the rule) can never be mistaken for a real call.

Scope: an explicit, individually-checked module list (see
:data:`EXPANDED_MATRIX_MODULE_PATHS`) - the four ``tests/unit/cpp/`` parity
modules, this hygiene module itself, and the ``expanded_matrix`` contract/
generator pair under ``tests/cpp_parity_live/``. A glob is no longer viable
here (test-suite renaming plan, Slice 3): these modules now carry
descriptive names with no shared "phase4"/"expanded_matrix" substring, so
the explicit list is the ONLY source of truth, not a fallback for one.
``test_golden_tracking.py`` (merged across all four scenario-matrix
datasets in the same renaming pass) and ``test_generate_goldens.py``
(likewise, from the earlier Slice 1 pass) are intentionally EXCLUDED: both
are shared infrastructure, not ``expanded_matrix``-exclusive.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os

import pytest

from tests._support import PROJECT_ROOT

#: The complete, explicit ``expanded_matrix`` test surface, repo-relative
#: with forward slashes. There is no glob fallback (see module docstring) -
#: this list IS the audited set, checked individually by
#: :func:`test_expanded_matrix_known_required_modules_are_all_present`.
EXPANDED_MATRIX_REQUIRED_MODULE_PATHS = frozenset({
    "tests/cpp_parity_live/_expanded_matrix_contract.py",
    "tests/cpp_parity_live/generate_expanded_matrix_goldens.py",
    "tests/unit/cpp/test_consumption_parity.py",
    "tests/unit/cpp/test_emissions_parity.py",
    "tests/unit/cpp/test_mortality_parity.py",
    "tests/unit/cpp/test_tree_structure_parity.py",
    "tests/unit/cpp/test_expanded_matrix_xfail_hygiene.py",
})

#: Absolute paths for the same set, in deterministic order - what the AST
#: scanners below actually iterate.
EXPANDED_MATRIX_MODULE_PATHS = tuple(
    sorted(
        os.path.join(PROJECT_ROOT, *rel.split("/"))
        for rel in EXPANDED_MATRIX_REQUIRED_MODULE_PATHS
    )
)


def _is_pytest_mark_xfail_call(node: ast.Call) -> bool:
    """
    Return whether *node* is a call to ``pytest.mark.xfail(...)``.

    :param node: An AST ``Call`` node.
    :returns: ``True`` if *node*'s callee is exactly the attribute chain
        ``pytest.mark.xfail``.
    """
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "xfail"):
        return False
    mark = func.value
    if not (isinstance(mark, ast.Attribute) and mark.attr == "mark"):
        return False
    root = mark.value
    return isinstance(root, ast.Name) and root.id == "pytest"


def _is_pytest_xfail_call(node: ast.Call) -> bool:
    """
    Return whether *node* is a call to the imperative ``pytest.xfail(...)``.

    Deliberately does NOT match ``pytest.mark.xfail(...)`` (the marker
    factory): that call's ``func.value`` is the attribute ``pytest.mark``,
    not the bare name ``pytest``, so the two are structurally
    distinguishable without any string/regex heuristic.

    :param node: An AST ``Call`` node.
    :returns: ``True`` if *node*'s callee is exactly ``pytest.xfail``.
    """
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "xfail"):
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id == "pytest"


def _module_ast(path: str) -> ast.Module:
    """
    Parse *path* into an AST module, with the source file's own path
    attached for readable error messages.

    :param path: Absolute path to a ``.py`` file.
    :returns: The parsed module.
    """
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)
    tree.filename = path  # type: ignore[attr-defined]
    return tree


def _strict_true(node: ast.Call) -> bool:
    """
    Return whether a ``pytest.mark.xfail(...)`` call's keyword arguments
    include a literal ``strict=True``.

    Only a literal ``True`` counts: a variable or expression could evaluate
    to ``True`` at runtime, but this is a static hygiene check and must not
    guess at runtime values, so anything other than the literal is treated
    as non-compliant and reported for human review.

    :param node: An AST ``Call`` node for ``pytest.mark.xfail(...)``.
    :returns: ``True`` only if a ``strict`` keyword is present with the
        literal value ``True``.
    """
    for kw in node.keywords:
        if kw.arg == "strict":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def test_expanded_matrix_known_required_modules_are_all_present():
    """
    Every module in :data:`EXPANDED_MATRIX_REQUIRED_MODULE_PATHS` must
    actually exist on disk - the explicit presence guarantee this module's
    fixed, non-glob module list depends on.

    :returns: None.
    """
    missing_on_disk = sorted(
        rel for rel in EXPANDED_MATRIX_REQUIRED_MODULE_PATHS
        if not os.path.isfile(os.path.join(PROJECT_ROOT, rel))
    )
    assert not missing_on_disk, (
        f"required expanded_matrix module(s) missing on disk: {missing_on_disk}"
    )


def test_expanded_matrix_module_set_is_nonempty():
    """
    Guard against :data:`EXPANDED_MATRIX_MODULE_PATHS` being accidentally
    emptied, which would make every other test in this module vacuously
    pass.

    :returns: None.
    """
    assert len(EXPANDED_MATRIX_MODULE_PATHS) == len(EXPANDED_MATRIX_REQUIRED_MODULE_PATHS), (
        "expected exactly the known-required expanded_matrix modules (4 "
        "tests/unit/cpp/ parity modules plus the "
        "tests/cpp_parity_live/{_expanded_matrix_contract,generate_"
        "expanded_matrix_goldens}.py pair, plus this module); got "
        f"{len(EXPANDED_MATRIX_MODULE_PATHS)}: {EXPANDED_MATRIX_MODULE_PATHS!r}"
    )


def test_expanded_matrix_modules_have_no_imperative_pytest_xfail_calls():
    """
    No ``expanded_matrix`` test module may call ``pytest.xfail(...)``
    unconditionally.

    Every such call is a permanent surrender that can never become a strict
    XPASS - see the module docstring. AST-based, so a docstring or comment
    that merely mentions ``pytest.xfail`` (several ``expanded_matrix``
    modules' docstrings do, to explain this very rule) is never mistaken
    for a call.

    :returns: None.
    """
    offenders = []
    for path in EXPANDED_MATRIX_MODULE_PATHS:
        tree = _module_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_pytest_xfail_call(node):
                rel = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "imperative pytest.xfail() calls found (replace each with a real "
        "desired-behaviour assertion decorated "
        "@pytest.mark.xfail(strict=True, reason=...)): "
        + ", ".join(offenders)
    )


def test_expanded_matrix_xfail_markers_are_all_strict():
    """
    Every ``@pytest.mark.xfail(...)`` in an ``expanded_matrix`` test module
    must set ``strict=True``.

    A non-strict marker would silently tolerate the underlying production
    defect being fixed (XPASS) without failing the suite, defeating the
    entire point of pinning desired-behaviour assertions here.

    :returns: None.
    """
    offenders = []
    for path in EXPANDED_MATRIX_MODULE_PATHS:
        tree = _module_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_pytest_mark_xfail_call(node):
                if not _strict_true(node):
                    rel = os.path.relpath(path, PROJECT_ROOT).replace(
                        os.sep, "/"
                    )
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "pytest.mark.xfail(...) call(s) missing a literal strict=True: "
        + ", ".join(offenders)
    )
