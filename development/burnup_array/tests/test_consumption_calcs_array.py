#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for burnup_calcs.py — the array-based burnup consumption pipeline.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_PROJECT_ROOT))
sys.path.insert(0, _PROJECT_ROOT)

from burnup_array_calcs import (
    BurnupConsumptionResult,
    run_burnup_array,
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_INPUTS_DIR = os.path.join(_REPO_ROOT, 'tests', 'test_data', 'test_inputs')
_INPUT_CSV = os.path.join(_INPUTS_DIR, 'burnup_input.csv')
_TAC_TO_KGM2 = 1.0 / 4.4609


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def canonical_inputs():
    """Load canonical test inputs from burnup_input.csv."""
    return pd.read_csv(_INPUT_CSV, comment='#')


@pytest.fixture(scope='module')
def array_result():
    """Run run_burnup_array with canonical inputs (C=1)."""
    inp = pd.read_csv(_INPUT_CSV, comment='#')
    loads = {row['component']: row['load_tac'] * _TAC_TO_KGM2
             for _, row in inp.iterrows()}
    moists = {row['component']: row['moisture_fraction']
              for _, row in inp.iterrows()}

    result = run_burnup_array(
        litter=np.array([loads.get('litter', 0.0)]),
        dw1=np.array([loads.get('dw1', 0.0)]),
        dw10=np.array([loads.get('dw10', 0.0)]),
        dw100=np.array([loads.get('dw100', 0.0)]),
        dwk_3_6=np.array([loads.get('dwk_3_6', 0.0)]),
        dwk_6_9=np.array([loads.get('dwk_6_9', 0.0)]),
        dwk_9_20=np.array([loads.get('dwk_9_20', 0.0)]),
        dwk_20=np.array([loads.get('dwk_20', 0.0)]),
        litter_moist=np.array([moists.get('litter', 0.10)]),
        dw1_moist=np.array([moists.get('dw1', 0.10)]),
        dw10_moist=np.array([moists.get('dw10', 0.10)]),
        dw100_moist=np.array([moists.get('dw100', 0.10)]),
        dwk_moist=np.array([moists.get('dwk_3_6', 0.10)]),
        intensity=np.array([50.0]),
        ig_time=np.array([60.0]),
        windspeed=np.array([0.0]),
        depth=np.array([0.3]),
        ambient_temp=np.array([21.0]),
        duff_loading=np.array([0.0]),
        duff_moist_frac=np.array([2.0]),
        dt=15.0,
        max_timesteps=3000,
    )
    return result


# ---------------------------------------------------------------------------
# Tests — basic functionality
# ---------------------------------------------------------------------------
class TestRunBurnupArrayBasic:
    def test_no_error(self, array_result):
        assert array_result.burnup_error[0] == 0

    def test_returns_correct_type(self, array_result):
        assert isinstance(array_result, BurnupConsumptionResult)

    def test_has_class_keys(self, array_result):
        for key in ('litter', 'dw1', 'dw10', 'dw100',
                     'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20'):
            assert key in array_result.consumed

    def test_durations_positive(self, array_result):
        assert array_result.fla_dur[0] >= 0
        assert array_result.smo_dur[0] >= 0

    def test_fine_fuels_consumed(self, array_result):
        """Litter and 1-hr should be fully consumed."""
        assert array_result.frac_remaining['litter'][0] < 0.01
        assert array_result.frac_remaining['dw1'][0] < 0.01

    def test_coarse_fuels_intact(self, array_result):
        """1000-hr fuels should be mostly intact."""
        for key in ('dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20'):
            assert array_result.frac_remaining[key][0] > 0.99


# ---------------------------------------------------------------------------
# Tests — multi-cell
# ---------------------------------------------------------------------------
class TestRunBurnupArrayMultiCell:
    def test_two_identical_cells(self):
        """Two identical cells should produce identical results."""
        C = 2
        result = run_burnup_array(
            litter=np.full(C, 0.224),
            dw1=np.full(C, 0.224),
            dw10=np.full(C, 0.224),
            dw100=np.full(C, 0.224),
            dwk_3_6=np.full(C, 0.028),
            dwk_6_9=np.full(C, 0.028),
            dwk_9_20=np.full(C, 0.028),
            dwk_20=np.full(C, 0.028),
            litter_moist=np.full(C, 0.18),
            dw1_moist=np.full(C, 0.18),
            dw10_moist=np.full(C, 0.20),
            dw100_moist=np.full(C, 0.22),
            dwk_moist=np.full(C, 0.20),
            intensity=np.full(C, 50.0),
            ig_time=np.full(C, 60.0),
            windspeed=np.full(C, 0.0),
            depth=np.full(C, 0.3),
            ambient_temp=np.full(C, 21.0),
            duff_loading=np.full(C, 0.0),
            duff_moist_frac=np.full(C, 2.0),
            dt=15.0,
            max_timesteps=3000,
        )

        assert result.burnup_error[0] == 0
        assert result.burnup_error[1] == 0

        for key in ('litter', 'dw1', 'dw10', 'dw100'):
            np.testing.assert_allclose(
                result.consumed[key][0],
                result.consumed[key][1],
                atol=1e-10,
                err_msg=f'{key} consumed mismatch',
            )

    def test_zero_fuel_cell(self):
        """A cell with zero fuel should get error code 90."""
        C = 2
        result = run_burnup_array(
            litter=np.array([0.224, 0.0]),
            dw1=np.array([0.224, 0.0]),
            dw10=np.array([0.224, 0.0]),
            dw100=np.array([0.224, 0.0]),
            dwk_3_6=np.array([0.028, 0.0]),
            dwk_6_9=np.array([0.028, 0.0]),
            dwk_9_20=np.array([0.028, 0.0]),
            dwk_20=np.array([0.028, 0.0]),
            litter_moist=np.full(C, 0.18),
            dw1_moist=np.full(C, 0.18),
            dw10_moist=np.full(C, 0.20),
            dw100_moist=np.full(C, 0.22),
            dwk_moist=np.full(C, 0.20),
            intensity=np.full(C, 50.0),
            ig_time=np.full(C, 60.0),
            windspeed=np.full(C, 0.0),
            depth=np.full(C, 0.3),
            ambient_temp=np.full(C, 21.0),
            duff_loading=np.full(C, 0.0),
            duff_moist_frac=np.full(C, 2.0),
            dt=15.0,
        )

        assert result.burnup_error[0] == 0
        assert result.burnup_error[1] == 90  # no fuel
        # Error cell should have zeroed consumption
        for key in ('litter', 'dw1', 'dw10'):
            assert result.consumed[key][1] == 0.0

    def test_clipping_adjustment_codes(self):
        """Inputs exceeding bounds should be clipped with adjustment codes."""
        result = run_burnup_array(
            litter=np.array([0.224]),
            dw1=np.array([0.224]),
            dw10=np.array([0.224]),
            dw100=np.array([0.224]),
            dwk_3_6=np.array([0.028]),
            dwk_6_9=np.array([0.028]),
            dwk_9_20=np.array([0.028]),
            dwk_20=np.array([0.028]),
            litter_moist=np.array([0.18]),
            dw1_moist=np.array([0.18]),
            dw10_moist=np.array([0.20]),
            dw100_moist=np.array([0.22]),
            dwk_moist=np.array([0.20]),
            intensity=np.array([200000.0]),  # exceeds max → should be clipped
            ig_time=np.array([60.0]),
            windspeed=np.array([0.0]),
            depth=np.array([0.3]),
            ambient_temp=np.array([21.0]),
            duff_loading=np.array([0.0]),
            duff_moist_frac=np.array([2.0]),
            dt=15.0,
            max_timesteps=100,
        )

        # Adjustment code should be non-zero (intensity clipped → digit 1)
        assert result.burnup_limit_adjust[0] > 0
        # But burnup should still succeed
        assert result.burnup_error[0] == 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_run_burnup_array_smoke():
    """Basic smoke test: run_burnup_array completes without exception."""
    result = run_burnup_array(
        litter=np.array([0.224]),
        dw1=np.array([0.224]),
        dw10=np.array([0.224]),
        dw100=np.array([0.224]),
        dwk_3_6=np.array([0.028]),
        dwk_6_9=np.array([0.028]),
        dwk_9_20=np.array([0.028]),
        dwk_20=np.array([0.028]),
        litter_moist=np.array([0.18]),
        dw1_moist=np.array([0.18]),
        dw10_moist=np.array([0.20]),
        dw100_moist=np.array([0.22]),
        dwk_moist=np.array([0.20]),
        intensity=np.array([50.0]),
        ig_time=np.array([60.0]),
        windspeed=np.array([0.0]),
        depth=np.array([0.3]),
        ambient_temp=np.array([21.0]),
        duff_loading=np.array([0.0]),
        duff_moist_frac=np.array([2.0]),
        dt=15.0,
        max_timesteps=100,
    )

    assert isinstance(result, BurnupConsumptionResult)
    assert result.burnup_error[0] == 0

