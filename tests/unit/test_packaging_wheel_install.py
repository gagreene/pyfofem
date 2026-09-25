#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_packaging_wheel_install.py - F-62 completion/acceptance-recovery
pass (2026-09-21): a real build-a-wheel/install-into-a-throwaway-venv/
import-from-outside-the-checkout regression for the packaging defect
this pass found and fixed.

An isolated-wheel installed-only run (required whenever
``src/pyfofem/`` changes, per this repository's established
methodology) reproduced a real ``FileNotFoundError`` on plain
``import pyfofem``: a concurrent, unrelated change to
``components/tree_flame_calcs.py`` added
``_load_canopy_equations()``, which reads
``supporting_data/FOFEM6.7/FOF_SPP.CSV`` at **import time** - a file
``pyproject.toml``'s ``[tool.setuptools.package-data]`` glob
(``supporting_data/*.csv``, deliberately non-recursive so the rest of
the bundled ``FOFEM6.7/`` vendor distribution - ``FOF_GUI.exe``, two
Microsoft DLLs, a help PDF - stays out of the wheel) never reached,
because it lives one directory deeper.

Fixed with the smallest possible correction: one additional EXACT,
non-glob, non-recursive package-data literal naming
``supporting_data/FOFEM6.7/FOF_SPP.CSV`` specifically - proven, not
merely declared, by this module's own test building a real wheel from
the corrected ``pyproject.toml``, installing it into a fully isolated
throwaway venv (own numpy/pandas/scipy/tqdm - no
``--system-site-packages``, so this checkout's own editable ``pyfofem``
install in the base environment cannot substitute for the wheel under
test), and importing ``pyfofem.components.tree_flame_calcs`` from a
real child process whose ``cwd`` is outside this checkout entirely.

Registered in ``FULL_EXTRA_TESTS`` only (``tests/run_unified_tests.py``),
never ``CORE_TESTS`` - unlike every test in
``test_runtime_data_resources.py`` (this module's sibling, which pins
the corrected package-data *declaration* without paying for a real
wheel build), this test performs a genuine network/disk-bound wheel
build plus a full dependency install, which would slow every
``--suite core`` invocation if placed there.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):
category (a), Python contract/packaging test - a real, executable
proof of the installed-artifact resource-resolution contract, not a
C++ parity comparison.

Function order: private helpers first, then public test functions,
each group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import sys

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._proc import run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

#: Timeouts (seconds) for this module's own subprocess steps - each is a
#: real network/disk-bound operation (wheel build, venv creation,
#: dependency + wheel install), so these are deliberately generous.
_WHEEL_BUILD_TIMEOUT_S = 300.0
_VENV_CREATE_TIMEOUT_S = 120.0
_WHEEL_INSTALL_TIMEOUT_S = 480.0
_IMPORT_PROBE_TIMEOUT_S = 180.0


def _venv_python_path(venv_dir: str) -> str:
    """
    Resolve the interpreter path inside a venv created at *venv_dir*.

    :param venv_dir: Root directory of the venv.
    :return: Absolute path to that venv's own ``python``/``python.exe``.
    """
    return (
        os.path.join(venv_dir, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )


@pytest.mark.installed_artifact
@pytest.mark.slow
def test_fof_spp_csv_resolves_from_an_installed_wheel_outside_the_checkout():
    """
    Build a REAL wheel from this checkout's corrected ``pyproject.toml``,
    install it (fully isolated - a throwaway venv with its own
    numpy/pandas/scipy/tqdm, no ``--system-site-packages``) into a
    repository-local scratch directory, and import
    ``pyfofem.components.tree_flame_calcs`` from a real child process
    whose ``cwd`` is that scratch directory - well outside this
    checkout - proving ``supporting_data/FOFEM6.7/FOF_SPP.CSV`` resolves
    under the installed package's own ``site-packages`` directory and
    that ``_CANOPY_EQUATIONS`` (the dict it populates) is non-empty.

    Reverting this pass's one-line ``pyproject.toml`` fix reproduces
    ``FileNotFoundError: ... FOF_SPP.CSV`` here directly (verified
    manually before this test was written, not merely asserted).

    Uses :func:`tests.cpp_parity_live._scratch.scratch_tempdir` (never
    the system/user temp directory) and
    :func:`tests.cpp_parity_live._proc.run_bounded` (a hard timeout and
    real process-tree teardown for every subprocess step) throughout.

    :return: None. Raises via ``assert`` on mismatch or subprocess
        failure.
    """
    with scratch_tempdir("packaging_wheel_probe") as scratch:
        dist_dir = os.path.join(scratch, "dist")
        venv_dir = os.path.join(scratch, "venv")
        os.makedirs(dist_dir, exist_ok=True)

        build = run_bounded(
            [
                sys.executable, "-m", "pip", "wheel", PROJECT_ROOT,
                "--no-deps", "-w", dist_dir, "-q",
            ],
            timeout=_WHEEL_BUILD_TIMEOUT_S,
            cwd=scratch,
        )
        assert build.returncode == 0, build.stderr

        wheels = [name for name in os.listdir(dist_dir) if name.endswith(".whl")]
        assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
        wheel_path = os.path.join(dist_dir, wheels[0])

        venv_create = run_bounded(
            [sys.executable, "-m", "venv", venv_dir],
            timeout=_VENV_CREATE_TIMEOUT_S,
            cwd=scratch,
        )
        assert venv_create.returncode == 0, venv_create.stderr

        venv_python = _venv_python_path(venv_dir)
        assert os.path.isfile(venv_python)

        install = run_bounded(
            [
                venv_python, "-m", "pip", "install", "-q",
                "numpy", "pandas", "scipy", "tqdm", wheel_path,
            ],
            timeout=_WHEEL_INSTALL_TIMEOUT_S,
            cwd=scratch,
        )
        assert install.returncode == 0, install.stderr

        probe = (
            "import os, pyfofem;"
            "from pyfofem.components import tree_flame_calcs;"
            "pkg = os.path.dirname(os.path.abspath(pyfofem.__file__));"
            "spp = os.path.normpath(os.path.join("
            "os.path.dirname(os.path.abspath(tree_flame_calcs.__file__)),"
            "'..','supporting_data','FOFEM6.7','FOF_SPP.CSV'));"
            "print(pkg);print(spp);print(os.path.isfile(spp));"
            "print(len(tree_flame_calcs._CANOPY_EQUATIONS))"
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)

        result = run_bounded(
            [venv_python, "-c", probe],
            timeout=_IMPORT_PROBE_TIMEOUT_S,
            cwd=scratch,
            env=env,
        )
        assert result.returncode == 0, result.stderr

        pkg_dir, spp_path, exists_line, count_line = result.stdout.strip().splitlines()
        assert exists_line == "True"
        assert int(count_line) > 0
        assert os.path.isabs(spp_path)
        assert os.path.commonpath([pkg_dir, spp_path]) == pkg_dir
        # pkg_dir must resolve inside THIS venv's own site-packages (proving
        # the wheel install, not something else, satisfied the import) and
        # must NOT be the checkout's own src/pyfofem package directory
        # (proving the checkout's editable install - present in the base
        # environment this venv was NOT created with --system-site-packages
        # from - could not have shadowed or substituted for the wheel).
        # scratch_tempdir is deliberately repository-local (never the
        # system/user temp directory), so pkg_dir legitimately lives inside
        # PROJECT_ROOT - the checkout-package-dir check, not a
        # PROJECT_ROOT-exclusion check, is what actually proves isolation.
        assert os.path.commonpath([venv_dir, pkg_dir]) == venv_dir
        checkout_pkg_dir = os.path.join(PROJECT_ROOT, "src", "pyfofem")
        assert pkg_dir != os.path.normpath(checkout_pkg_dir)
        assert os.path.commonpath([checkout_pkg_dir, pkg_dir]) != os.path.normpath(checkout_pkg_dir)
