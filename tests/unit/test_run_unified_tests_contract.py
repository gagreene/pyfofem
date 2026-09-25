#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression coverage for the ``run_unified_tests.py --installed-only``
parent-to-child contract added during the Phase 1 directory restructure.

The contract has two halves that must both hold, or installed-artifact
validation can silently pass when it shouldn't:

1. **Parent** (``tests/run_unified_tests.py::_run_pytest``): when
   ``--installed-only`` is requested, the pytest subprocess must actually
   receive ``PYFOFEM_INSTALLED_ONLY=1`` in its environment. When it is not
   requested, the flag must not leak from a stale parent-process environment
   variable into the subprocess.
2. **Child** (``tests/conftest.py::pytest_sessionstart``): the flag must be
   opt-in (session proceeds untouched when unset, even if pyfofem happens to
   resolve under ``src/``) and must actually abort the session when set and
   pyfofem resolves under ``src/``.

Without both halves wired correctly, a future change to either file could
silently defeat installed-artifact validation without any test noticing.
"""
import importlib
import os
from unittest.mock import patch

import pytest

import tests.conftest as pyfofem_conftest
import tests.run_unified_tests as unified

pytestmark = pytest.mark.installed_artifact


class _FakeCompletedProcess:
    """Minimal stand-in for :class:`subprocess.CompletedProcess`."""

    def __init__(self, returncode: int = 0):
        """
        :param returncode: Fake process return code to expose.
        :return: None.
        """
        self.returncode = returncode


class TestChildEnforcesInstalledOnlyFlag:
    """``pytest_sessionstart`` must enforce the flag only when it is set."""

    def test_flag_set_does_not_raise_when_import_resolves_elsewhere(
        self, monkeypatch, tmp_path
    ):
        """
        With the env flag set, the hook must NOT raise when ``pyfofem``
        resolves outside the (patched) ``src/`` directory — proving the
        next test's ``pytest.raises`` isn't passing because the hook raises
        unconditionally.

        :param monkeypatch: pytest fixture used to set the env flag and
            patch ``SRC_DIR``.
        :param tmp_path: pytest fixture providing an isolated directory that
            cannot contain the real ``pyfofem`` install.
        :return: None. Raises via ``assert`` (test failure) if the hook
            raises unexpectedly.
        """
        monkeypatch.setattr(pyfofem_conftest, "SRC_DIR", str(tmp_path))
        monkeypatch.setenv(pyfofem_conftest.INSTALLED_ONLY_ENV_VAR, "1")
        pyfofem_conftest.pytest_sessionstart(session=None)

    def test_flag_set_raises_when_import_resolves_under_src(self, monkeypatch):
        """
        With the env flag set to ``"1"``, the hook must raise
        ``pytest.UsageError`` when ``pyfofem`` resolves beneath the
        directory the hook is told is ``src/`` — the exact failure mode
        installed-only validation exists to catch.

        Patches ``SRC_DIR`` to wherever ``pyfofem`` *actually* resolves
        right now, rather than assuming this test always runs in a
        dev/editable checkout: that keeps the test deterministic whether it
        is executed against the local src/ tree or, in an installed-wheel
        CI lane, against site-packages.

        :param monkeypatch: pytest fixture used to set the env flag and
            patch ``SRC_DIR``.
        :return: None. Raises via ``assert`` (test failure) if the hook does
            not raise.
        """
        pyfofem = importlib.import_module("pyfofem")
        actual_dir = os.path.dirname(os.path.abspath(pyfofem.__file__))
        monkeypatch.setattr(pyfofem_conftest, "SRC_DIR", actual_dir)
        monkeypatch.setenv(pyfofem_conftest.INSTALLED_ONLY_ENV_VAR, "1")
        with pytest.raises(pytest.UsageError):
            pyfofem_conftest.pytest_sessionstart(session=None)

    def test_flag_unset_does_not_raise(self, monkeypatch):
        """
        With the env flag unset, the session-start hook must return without
        raising, regardless of where ``pyfofem`` currently resolves.

        :param monkeypatch: pytest fixture used to clear the env flag.
        :return: None. Raises via ``assert`` (test failure) if the hook
            raises unexpectedly.
        """
        monkeypatch.delenv(
            pyfofem_conftest.INSTALLED_ONLY_ENV_VAR, raising=False
        )
        pyfofem_conftest.pytest_sessionstart(session=None)


class TestParentPropagatesInstalledOnlyFlag:
    """``_run_pytest`` must set/unset the child env flag, never leak it."""

    def test_installed_only_false_does_not_set_env_flag(self):
        """
        ``installed_only=False`` must not set the child env flag, even if a
        stale value from the parent process's own environment is present —
        the flag must never leak into a run that did not request it.

        :return: None. Raises via ``assert`` on mismatch.
        """
        with patch.dict(
            unified.os.environ, {unified._INSTALLED_ONLY_ENV_VAR: "1"}
        ):
            with patch.object(
                unified.subprocess, "run", return_value=_FakeCompletedProcess(0)
            ) as mock_run:
                unified._run_pytest(
                    test_paths=[], verbosity=0, installed_only=False
                )

        _, kwargs = mock_run.call_args
        env = kwargs["env"]
        assert unified._INSTALLED_ONLY_ENV_VAR not in env

    def test_installed_only_true_sets_env_flag(self):
        """
        ``installed_only=True`` must set ``PYFOFEM_INSTALLED_ONLY=1`` on the
        subprocess environment passed to ``subprocess.run``.

        :return: None. Raises via ``assert`` on mismatch.
        """
        with patch.object(
            unified.subprocess, "run", return_value=_FakeCompletedProcess(0)
        ) as mock_run:
            unified._run_pytest(test_paths=[], verbosity=0, installed_only=True)

        _, kwargs = mock_run.call_args
        env = kwargs["env"]
        assert env.get(unified._INSTALLED_ONLY_ENV_VAR) == "1"
