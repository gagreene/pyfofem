#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Equation-level unit tests for pyfofem consumption functions.

Each test case is driven by a row in:
    tests/test_data/test_golden_output/equation_unit_tests_golden.csv

The CSV encodes the function name, a JSON inputs dict, the output key to check,
the expected value, and the absolute tolerance.  This makes it trivial to add
new analytical test cases without modifying the test code.

Functions under test
--------------------
- consm_duff     : duff consumption equations 1, 2, 3, 4, 5, 6, 15, 17
                   + low-moisture floor (Fix D) and pile-burning fix (Fix C)
                   + Eq-3/7 correct moisture variable (Fix A)
- consm_herb     : equations 22, 221 (Spring-only fix B), 222, 223
- consm_shrub    : equations 23, 231, 232, 233, 234, 235
- consm_canopy   : equations 37, 38

Fix A-D's own named regression classes live in
``tests/regression/test_equations_golden_fixes.py`` (split out during the
Phase 1 directory restructure); this module keeps only the golden-CSV-driven
parametrized coverage plus the hand-written ``consm_canopy`` sanity checks
below, which test the same equations (37/38) but are not CSV-driven.
"""

import json
import os

import pandas as pd
import pytest

from pyfofem import consm_canopy, consm_duff, consm_herb, consm_shrub
from tests._support import TEST_GOLDEN_DIR

_GOLDEN_CSV = os.path.join(TEST_GOLDEN_DIR, 'equation_unit_tests_golden.csv')

# ---------------------------------------------------------------------------
# Function dispatcher
# ---------------------------------------------------------------------------
_FUNCTION_MAP = {
    'consm_duff':   consm_duff,
    'consm_herb':   consm_herb,
    'consm_shrub':  consm_shrub,
    'consm_canopy': consm_canopy,
}

# Keys that consm_shrub returns as a percent (slc_pct is derived from the
# raw float return value which IS the percent).
_SHRUB_PCT_KEY = 'slc_pct'


def _call_function(func_name: str, inputs: dict, expected_key: str):
    """
    Call the named function with the given inputs and return the scalar result.

    :param func_name: Key into :data:`_FUNCTION_MAP` naming the consumption
        function under test.
    :param inputs: Keyword arguments to forward to the named function.
    :param expected_key: Result dict key to extract (ignored for functions
        that return a bare scalar).
    :return: The scalar result value to compare against the golden expectation.
    :raises ValueError: If *func_name* is not a recognised function name.
    """
    if func_name not in _FUNCTION_MAP:
        raise ValueError(f'Unknown function: {func_name}')
    func = _FUNCTION_MAP[func_name]

    if func_name == 'consm_duff':
        result = func(**inputs)
        return result[expected_key]

    if func_name == 'consm_herb':
        result = func(**inputs)
        # consm_herb returns the consumed load directly (not a dict)
        return float(result)

    if func_name == 'consm_shrub':
        result = func(**inputs)
        # consm_shrub returns the percent consumed (float)
        return float(result)

    if func_name == 'consm_canopy':
        result = func(**inputs)
        return result[expected_key]

    raise ValueError(f'Unknown function: {func_name}')


# ---------------------------------------------------------------------------
# Load golden table and parametrize
# ---------------------------------------------------------------------------
_golden_df = pd.read_csv(_GOLDEN_CSV, comment='#')

_test_cases = [
    pytest.param(
        row['test_id'],
        row['function'],
        json.loads(row['inputs_json']),
        row['expected_key'],
        float(row['expected_value']),
        float(row['atol']),
        id=row['test_id'],
    )
    for _, row in _golden_df.iterrows()
]


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'test_id,func_name,inputs,expected_key,expected_value,atol',
    _test_cases,
)
def test_equation_golden(test_id, func_name, inputs, expected_key, expected_value, atol):
    """
    Verify one golden-CSV row: each row becomes one pytest case.

    :param test_id: Golden-CSV test identifier (used as the pytest case ID).
    :param func_name: Name of the consumption function under test.
    :param inputs: Keyword arguments to forward to the function.
    :param expected_key: Result dict key to extract and compare.
    :param expected_value: Expected scalar value from the golden CSV.
    :param atol: Absolute tolerance for the comparison.
    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = _call_function(func_name, inputs, expected_key)
    assert abs(actual - expected_value) <= atol, (
        f'[{test_id}] {func_name}({inputs})[{expected_key}]: '
        f'got {actual:.6f}, expected {expected_value:.6f} '
        f'(diff={abs(actual - expected_value):.6f}, atol={atol})'
    )


# ---------------------------------------------------------------------------
# Additional consm_canopy sanity checks
# ---------------------------------------------------------------------------
class TestCanopyEquations:
    """Additional consm_canopy sanity checks (equations 37, 38)."""

    def test_zero_crown_burn(self):
        """0% crown burn consumes no foliage or branch load.

        :return: None. Raises via ``assert`` on mismatch.
        """
        result = consm_canopy(0.0, 10.0, 5.0, units='Imperial')
        assert result['flc'] == pytest.approx(0.0)
        assert result['blc'] == pytest.approx(0.0)

    def test_full_crown_burn(self):
        """100% crown burn consumes all foliage and half the branch load.

        :return: None. Raises via ``assert`` on mismatch.
        """
        result = consm_canopy(100.0, 10.0, 6.0, units='Imperial')
        assert result['flc'] == pytest.approx(10.0, abs=0.001)
        assert result['blc'] == pytest.approx(3.0, abs=0.001)  # 6.0 * 0.5 * 1.0

    def test_branch_always_50pct_of_foliage_fraction(self):
        """Branch consumed = 50% of what foliage fraction says (Eq 38)."""
        for pct in (25.0, 50.0, 75.0):
            result = consm_canopy(pct, 8.0, 4.0, units='Imperial')
            assert abs(result['blc'] - result['flc'] * 0.5 * 4.0 / 8.0) < 0.001
