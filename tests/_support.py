#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reusable, non-pytest path constants for the pyfofem test suite.

This module owns shared filesystem paths only. It must never insert the
checkout's ``src/`` directory onto ``sys.path`` — every test relies on an
editable or wheel install of ``pyfofem`` already being importable without
per-file path manipulation (see the Phase 1 directory-restructure plan,
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``).
"""
from __future__ import annotations

import os

#: Absolute path to the repository root (one level up from ``tests/``).
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Absolute path to the ``tests/`` package directory itself.
TESTS_DIR: str = os.path.join(PROJECT_ROOT, "tests")

#: Absolute path to the checkout's ``src/`` tree. Reserved for the
#: installed-only session validation in :mod:`tests.conftest`; never insert
#: this onto ``sys.path``.
SRC_DIR: str = os.path.join(PROJECT_ROOT, "src")

#: Absolute path to ``tests/test_data/``.
TEST_DATA_DIR: str = os.path.join(TESTS_DIR, "test_data")

#: Absolute path to ``tests/test_data/test_inputs/``.
TEST_INPUTS_DIR: str = os.path.join(TEST_DATA_DIR, "test_inputs")

#: Absolute path to ``tests/test_data/test_golden_output/``.
TEST_GOLDEN_DIR: str = os.path.join(TEST_DATA_DIR, "test_golden_output")

#: Absolute path to the pinned C++ reference checkout.
CPP_REFERENCE_DIR: str = os.path.join(PROJECT_ROOT, "reference", "fofem_cpp")
