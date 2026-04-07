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
"""

import json
import os
import sys

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from pyfofem import consm_duff, consm_herb, consm_shrub, consm_canopy  # noqa: E402

_GOLDEN_CSV = os.path.join(
    _TESTS_DIR, 'test_data', 'test_golden_output', 'equation_unit_tests_golden.csv'
)

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
    """Call the named function with the given inputs and return the scalar result."""
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
    """Each row in equation_unit_tests_golden.csv becomes one pytest case."""
    actual = _call_function(func_name, inputs, expected_key)
    assert abs(actual - expected_value) <= atol, (
        f'[{test_id}] {func_name}({inputs})[{expected_key}]: '
        f'got {actual:.6f}, expected {expected_value:.6f} '
        f'(diff={abs(actual - expected_value):.6f}, atol={atol})'
    )


# ---------------------------------------------------------------------------
# Fix-specific named tests (explicit, readable, not driven by the CSV)
# ---------------------------------------------------------------------------

class TestFixA_Eq3UsesCorrectMoisture:
    """Fix A: Eq 3 (nfdth) must use dw1000_moist, not duff_moist."""

    def test_eq3_uses_dw1000_moist(self):
        """Result must differ when dw1000_moist ≠ duff_moist."""
        result_correct = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=20.0, units='Imperial',
        )
        result_wrong = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=None,  # falls back to duff_moist=80 → 114.7-4.2*80 → clamped 0
            units='Imperial',
        )
        # Correct: 114.7 - 4.2*20 = 30.7
        assert abs(result_correct['pdc'] - 30.7) < 0.01
        # Fallback: 114.7 - 4.2*80 = -221.3 → clamped to 0 by np.clip
        assert result_wrong['pdc'] == pytest.approx(0.0, abs=0.01)

    def test_eq7_uses_dw1000_moist(self):
        """Eq 7 depth (nfdth) must use dw1000_moist."""
        result = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=20.0, d_pre=3.0, units='Imperial',
        )
        # Eq 7: 1.773 - 0.1051*20 + 0.399*3 = 1.773 - 2.102 + 1.197 = 0.868
        expected = 1.773 - 0.1051 * 20.0 + 0.399 * 3.0
        assert abs(result['ddc'] - expected) < 0.001


class TestFixB_GrassHerbSeason:
    """Fix B: GrassGroup Eq 221 (10%) applies only in Spring."""

    def test_grass_spring_is_10pct(self):
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Spring', units='Imperial')
        assert abs(hlc - 0.2) < 0.001  # 2.0 * 0.1 = 0.2

    def test_grass_summer_is_100pct(self):
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Summer', units='Imperial')
        assert abs(hlc - 2.0) < 0.001  # 100 % consumed

    def test_grass_fall_is_100pct(self):
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Fall', units='Imperial')
        assert abs(hlc - 2.0) < 0.001

    def test_grass_winter_is_100pct(self):
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Winter', units='Imperial')
        assert abs(hlc - 2.0) < 0.001

    def test_grass_no_season_is_100pct(self):
        """No season provided → defaults to non-Spring behaviour (100%)."""
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         units='Imperial')
        assert abs(hlc - 2.0) < 0.001


class TestFixC_PileBurning:
    """Fix C: pile burning (Eq 17) must return pdc=10%, not 90%."""

    def test_pile_pdc_is_10_percent(self):
        result = consm_duff(pre_dl=10.0, duff_moist=50.0,
                            pile=True, units='Imperial')
        assert abs(result['pdc'] - 10.0) < 0.01, (
            f"Pile burning pdc={result['pdc']:.2f}, expected 10.0"
        )

    def test_pile_consumed_amount(self):
        """Consumed amount = pre_dl * 10% = 1.0 T/ac."""
        result = consm_duff(pre_dl=10.0, duff_moist=50.0,
                            pile=True, units='Imperial')
        pdc = result['pdc']
        consumed = 10.0 * pdc / 100.0
        assert abs(consumed - 1.0) < 0.01, (
            f'Pile consumed={consumed:.3f} T/ac, expected 1.0'
        )


class TestFixD_LowMoistureFloor:
    """Fix D: duff_moist ≤ 10 forces pdc=100%."""

    def test_floor_at_exactly_10(self):
        result = consm_duff(pre_dl=10.0, duff_moist=10.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        assert abs(result['pdc'] - 100.0) < 0.01

    def test_floor_below_10(self):
        result = consm_duff(pre_dl=10.0, duff_moist=5.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        assert abs(result['pdc'] - 100.0) < 0.01

    def test_no_floor_above_10(self):
        """pdc should NOT be forced to 100% when duff_moist > 10."""
        result = consm_duff(pre_dl=10.0, duff_moist=11.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        # Eq 2: 83.7 - 0.426*11 = 79.014
        expected = 83.7 - 0.426 * 11.0
        assert abs(result['pdc'] - expected) < 0.01
        assert result['pdc'] < 100.0


# ---------------------------------------------------------------------------
# Additional consm_canopy sanity checks
# ---------------------------------------------------------------------------
class TestCanopyEquations:
    def test_zero_crown_burn(self):
        result = consm_canopy(0.0, 10.0, 5.0, units='Imperial')
        assert result['flc'] == pytest.approx(0.0)
        assert result['blc'] == pytest.approx(0.0)

    def test_full_crown_burn(self):
        result = consm_canopy(100.0, 10.0, 6.0, units='Imperial')
        assert result['flc'] == pytest.approx(10.0, abs=0.001)
        assert result['blc'] == pytest.approx(3.0, abs=0.001)  # 6.0 * 0.5 * 1.0

    def test_branch_always_50pct_of_foliage_fraction(self):
        """Branch consumed = 50% of what foliage fraction says (Eq 38)."""
        for pct in (25.0, 50.0, 75.0):
            result = consm_canopy(pct, 8.0, 4.0, units='Imperial')
            assert abs(result['blc'] - result['flc'] * 0.5 * 4.0 / 8.0) < 0.001


