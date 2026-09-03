#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_contract_hygiene.py - Meta-test enforcing the Phase 4 correction
pass's xfail discipline: every scenario a production defect prevents from
passing must be pinned by a REAL executable assertion of the DESIRED
(post-fix) behaviour, decorated ``@pytest.mark.xfail(strict=True, ...)``,
never declared unconditionally with an imperative ``pytest.xfail(...)``
call.

An imperative ``pytest.xfail()`` call is a permanent, unconditional
surrender: it never executes a real assertion, so it can never turn into a
strict XPASS when the underlying defect is fixed, and a reader cannot tell
whether it is still accurate without re-deriving the defect from source. A
non-``strict`` ``@pytest.mark.xfail`` marker has the softer version of the
same defect: it silently tolerates the defect being fixed (XPASS) without
failing the suite, so nobody is forced to remove the stale marker.

This module performs a static AST scan, not a regex, specifically so a
docstring or comment merely MENTIONING ``pytest.xfail`` (as this module's
own docstring does, and as several Phase 4 modules' docstrings do when
explaining the rule) can never be mistaken for a real call.

Scope: every ``tests/unit/test_phase4_*.py`` module (the four Tier-2 parity
modules this hygiene check itself lives beside) and every ``*phase4*.py``
module under ``tests/cpp_parity_live/`` (the contract, generator and
generator-driver-test trio) - the complete Phase 4 test surface.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import glob
import os

import pytest

from tests._support import PROJECT_ROOT

#: Every source file this module audits. Globbed rather than hardcoded so a
#: new Phase 4 test module is automatically covered without editing this
#: list.
PHASE4_MODULE_PATHS = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "tests", "unit", "test_phase4_*.py"))
    + glob.glob(
        os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "*phase4*.py")
    )
)

#: The complete, known-required Phase 4 test surface, repo-relative with
#: forward slashes. A count-only guard (``len(PHASE4_MODULE_PATHS) >= 7``)
#: would not catch one of these being deleted or renamed as long as some
#: OTHER Phase 4 module still brought the total back up to 7+, so this
#: explicit set is checked for individually by
#: :func:`test_phase4_known_required_modules_are_all_present`, while the
#: glob above still covers detecting any FUTURE Phase 4 module added beyond
#: this set.
PHASE4_REQUIRED_MODULE_PATHS = frozenset({
    "tests/cpp_parity_live/_phase4_contract.py",
    "tests/cpp_parity_live/generate_phase4_goldens.py",
    "tests/cpp_parity_live/test_generate_phase4_goldens.py",
    "tests/unit/test_phase4_consumption_parity.py",
    "tests/unit/test_phase4_emissions_parity.py",
    "tests/unit/test_phase4_mortality_parity.py",
    "tests/unit/test_phase4_tree_structure_parity.py",
    "tests/unit/test_phase4_contract_hygiene.py",
    "tests/unit/test_phase4_golden_tracking.py",
})


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


def test_phase4_known_required_modules_are_all_present():
    """
    Every module in :data:`PHASE4_REQUIRED_MODULE_PATHS` must actually
    exist on disk AND be picked up by the glob-based
    :data:`PHASE4_MODULE_PATHS`.

    This is the explicit presence guarantee the count-only
    :func:`test_phase4_module_set_is_nonempty` cannot provide: a required
    module being deleted or renamed would not necessarily change
    ``len(PHASE4_MODULE_PATHS)`` below 7 as long as some other Phase 4
    module happened to still total 7 or more, so a bare count check could
    stay green while a specific required file silently vanished. Checking
    each required path's existence directly (not merely trusting the glob
    to have found it) also catches the glob pattern itself being narrowed
    in a way that stops matching one of these files.

    :returns: None.
    """
    glob_found = set(
        os.path.relpath(p, PROJECT_ROOT).replace(os.sep, "/")
        for p in PHASE4_MODULE_PATHS
    )
    missing_on_disk = sorted(
        rel for rel in PHASE4_REQUIRED_MODULE_PATHS
        if not os.path.isfile(os.path.join(PROJECT_ROOT, rel))
    )
    assert not missing_on_disk, (
        f"required Phase 4 module(s) missing on disk: {missing_on_disk}"
    )
    missing_from_glob = sorted(PHASE4_REQUIRED_MODULE_PATHS - glob_found)
    assert not missing_from_glob, (
        "required Phase 4 module(s) exist on disk but were not picked up "
        f"by the glob (PHASE4_MODULE_PATHS): {missing_from_glob}"
    )


def test_phase4_module_set_is_nonempty():
    """
    Guard against a silently-empty glob making the other tests in this
    module vacuously pass. This is deliberately count/pattern-based (not
    the explicit-presence guarantee) so a FUTURE Phase 4 module added
    beyond the known-required 9 is still covered without editing this
    module: see :func:`test_phase4_known_required_modules_are_all_present`
    for the explicit guarantee over the known set.

    :returns: None.
    """
    assert len(PHASE4_MODULE_PATHS) >= len(PHASE4_REQUIRED_MODULE_PATHS), (
        "expected at least the 9 known-required Phase 4 modules (4 "
        "tests/unit/test_phase4_*.py parity/hygiene/tracking modules plus "
        "the tests/cpp_parity_live/{_phase4_contract,generate_phase4_"
        "goldens,test_generate_phase4_goldens}.py trio, plus this module "
        "and the golden-tracking module); got "
        f"{len(PHASE4_MODULE_PATHS)}: {PHASE4_MODULE_PATHS!r}"
    )


def test_phase4_modules_have_no_imperative_pytest_xfail_calls():
    """
    No Phase 4 test module may call ``pytest.xfail(...)`` unconditionally.

    Every such call is a permanent surrender that can never become a strict
    XPASS - see the module docstring. AST-based, so a docstring or comment
    that merely mentions ``pytest.xfail`` (several Phase 4 modules'
    docstrings do, to explain this very rule) is never mistaken for a call.

    :returns: None.
    """
    offenders = []
    for path in PHASE4_MODULE_PATHS:
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


def test_phase4_xfail_markers_are_all_strict():
    """
    Every ``@pytest.mark.xfail(...)`` in a Phase 4 test module must set
    ``strict=True``.

    A non-strict marker would silently tolerate the underlying production
    defect being fixed (XPASS) without failing the suite, defeating the
    entire point of pinning desired-behaviour assertions here.

    :returns: None.
    """
    offenders = []
    for path in PHASE4_MODULE_PATHS:
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
