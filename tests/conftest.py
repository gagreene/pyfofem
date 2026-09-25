#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared pytest configuration for the pyfofem test suite.

Owns fixtures, marker/session configuration, and pytest-session validation
only. Path constants and other reusable non-pytest helpers live in
:mod:`tests._support`. This module must never insert the checkout's
``src/`` directory onto ``sys.path`` — see :mod:`tests._support`.
"""
from __future__ import annotations

import importlib
import os

import pytest

from tests._support import SRC_DIR

#: Environment variable ``run_unified_tests.py --installed-only`` sets on
#: the pytest subprocess to request the child-process import-origin check
#: below. Must stay in sync with ``tests/run_unified_tests.py``.
INSTALLED_ONLY_ENV_VAR = "PYFOFEM_INSTALLED_ONLY"


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Fail the session before collection if ``pyfofem`` resolves under ``src/``
    while installed-only validation was requested.

    This is the child-process half of the parent/child installed-only
    contract: ``run_unified_tests.py --installed-only`` performs an early
    parent-process import diagnostic, then sets
    :data:`INSTALLED_ONLY_ENV_VAR` on the pytest subprocess so this hook can
    verify the import origin *inside* the process that will actually collect
    and run the tests. The parent-process check alone is not sufficient
    because it says nothing about what the subprocess resolves.

    :param session: The pytest ``Session`` object for this run.
    :return: None. Raises ``pytest.UsageError`` to abort the session before
        any test is collected.
    :raises pytest.UsageError: If :data:`INSTALLED_ONLY_ENV_VAR` is ``"1"``
        and ``pyfofem`` resolves to the local checkout's ``src/`` tree
        instead of an installed package.
    """
    if os.environ.get(INSTALLED_ONLY_ENV_VAR) != "1":
        return

    pyfofem = importlib.import_module("pyfofem")
    module_path = os.path.abspath(getattr(pyfofem, "__file__", ""))
    src_root = os.path.abspath(SRC_DIR)

    if module_path == src_root or module_path.startswith(src_root + os.sep):
        raise pytest.UsageError(
            f"{INSTALLED_ONLY_ENV_VAR}=1 was set, but pyfofem resolved to "
            f"the local checkout's src/ tree: {module_path}"
        )
