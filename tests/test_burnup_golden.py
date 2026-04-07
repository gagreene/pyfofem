#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for run_burnup against golden output derived from the
canonical C++ FOFEM burnup reference files (load.txt / emis.txt).

Golden files
------------
- tests/test_data/test_inputs/burnup_input.csv
    Fuel loadings and moistures reconstructed from load.txt (sound-wood only,
    1.0 T/ac each for litter/DW1/DW10/DW100, 0.125 T/ac each for four
    1000-hr size classes, fire env: 50 kW/m² intensity, 60 s ig_time).

- tests/test_data/test_golden_output/burnup_load_golden.csv
    Expected per-component summary (pre-load, post-load, frac_remaining,
    ignition time, burnout time, moisture).

- tests/test_data/test_golden_output/burnup_timeseries_golden.csv
    Expected per-timestep overall remaining fraction (wdf) and flaming
    fraction (ff) for every recorded timestep.

Tolerances
----------
- Per-component frac_remaining : atol=0.005  (0.5 percentage points)
- Per-component t_ignite_s     : atol=2.0    seconds
- Per-step wdf                 : atol=0.005
- Per-step ff                  : atol=0.02
"""

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

from pyfofem import run_burnup  # noqa: E402

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_INPUTS_DIR = os.path.join(_TESTS_DIR, 'test_data', 'test_inputs')
_GOLDEN_DIR = os.path.join(_TESTS_DIR, 'test_data', 'test_golden_output')

_INPUT_CSV         = os.path.join(_INPUTS_DIR, 'burnup_input.csv')
_LOAD_GOLDEN_CSV   = os.path.join(_GOLDEN_DIR, 'burnup_load_golden.csv')
_TS_GOLDEN_CSV     = os.path.join(_GOLDEN_DIR, 'burnup_timeseries_golden.csv')

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
_ATOL_FRAC   = 0.005   # frac_remaining tolerance (0.5 pp)
_ATOL_TIGN   = 2.0     # ignition time tolerance (s)
_ATOL_WDF    = 0.005   # overall remaining-weight fraction
_ATOL_FF     = 0.02    # flaming fraction

_TAC_TO_KGM2 = 1.0 / 4.4609  # T/ac → kg/m²


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def burnup_results():
    """Run run_burnup once with the canonical inputs; cache for all tests."""
    inp = pd.read_csv(_INPUT_CSV, comment='#')

    fuel_loadings = {
        row['component']: row['load_tac'] * _TAC_TO_KGM2
        for _, row in inp.iterrows()
    }
    fuel_moistures = {
        row['component']: row['moisture_fraction']
        for _, row in inp.iterrows()
    }

    results, summary, class_order = run_burnup(
        fuel_loadings=fuel_loadings,
        fuel_moistures=fuel_moistures,
        intensity=50.0,
        ig_time=60.0,
        windspeed=0.0,
        depth=0.3,
        ambient_temp=21.0,
        max_times=3000,
        timestep=15.0,
    )
    return results, summary, class_order


@pytest.fixture(scope='module')
def load_golden():
    return pd.read_csv(_LOAD_GOLDEN_CSV, comment='#')


@pytest.fixture(scope='module')
def ts_golden():
    return pd.read_csv(_TS_GOLDEN_CSV, comment='#')


# ---------------------------------------------------------------------------
# Tests — per-component summary (load.txt equivalent)
# ---------------------------------------------------------------------------
class TestBurnupLoadGolden:
    """Validate per-component summary against golden load data."""

    def test_number_of_components(self, burnup_results, load_golden):
        _, summary, class_order = burnup_results
        assert len(class_order) == len(load_golden), (
            f'Expected {len(load_golden)} components, got {len(class_order)}'
        )

    def test_component_order(self, burnup_results, load_golden):
        _, _, class_order = burnup_results
        expected = list(load_golden['component'])
        assert class_order == expected, (
            f'Component order mismatch.\nExpected: {expected}\nGot: {class_order}'
        )

    @pytest.mark.parametrize('component', [
        'litter', 'dw1', 'dw10', 'dw100',
        'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    ])
    def test_frac_remaining(self, burnup_results, load_golden, component):
        """Remaining fraction should match golden within 0.5 pp."""
        _, summary, class_order = burnup_results
        idx = class_order.index(component)
        actual = summary[idx].frac_remaining
        expected = float(load_golden.loc[load_golden['component'] == component, 'frac_remaining'].iloc[0])
        assert abs(actual - expected) <= _ATOL_FRAC, (
            f'{component}: frac_remaining {actual:.6f} differs from golden '
            f'{expected:.6f} by {abs(actual-expected):.6f} (atol={_ATOL_FRAC})'
        )

    @pytest.mark.parametrize('component', ['litter', 'dw1', 'dw10'])
    def test_ignition_time(self, burnup_results, load_golden, component):
        """Fine fuels should ignite within 2 s of the golden value."""
        _, summary, class_order = burnup_results
        idx = class_order.index(component)
        actual = summary[idx].t_ignite
        expected = float(load_golden.loc[load_golden['component'] == component, 't_ignite_s'].iloc[0])
        assert abs(actual - expected) <= _ATOL_TIGN, (
            f'{component}: t_ignite {actual:.1f} s differs from golden '
            f'{expected:.1f} s by {abs(actual-expected):.1f} s (atol={_ATOL_TIGN})'
        )

    def test_fine_fuels_fully_consumed(self, burnup_results):
        """Litter and 1-hr wood must be 100 % consumed (frac_remaining ≈ 0)."""
        _, summary, class_order = burnup_results
        for comp in ('litter', 'dw1'):
            idx = class_order.index(comp)
            assert summary[idx].frac_remaining < 0.01, (
                f'{comp} should be fully consumed but frac_remaining='
                f'{summary[idx].frac_remaining:.4f}'
            )

    def test_coarse_fuels_mostly_intact(self, burnup_results):
        """1000-hr fuels should have > 99 % remaining (low intensity fire)."""
        _, summary, class_order = burnup_results
        for comp in ('dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20'):
            idx = class_order.index(comp)
            assert summary[idx].frac_remaining > 0.99, (
                f'{comp} frac_remaining={summary[idx].frac_remaining:.4f} '
                f'expected > 0.99 for low-intensity fire'
            )

    def test_prefire_loads_correct(self, burnup_results):
        """Pre-fire wdry values must match the inputs from burnup_input.csv."""
        inp = pd.read_csv(_INPUT_CSV, comment='#')
        _, summary, class_order = burnup_results
        for _, row in inp.iterrows():
            comp = row['component']
            idx = class_order.index(comp)
            expected_kg = row['load_tac'] * _TAC_TO_KGM2
            assert abs(summary[idx].wdry - expected_kg) < 1e-4, (
                f'{comp}: wdry {summary[idx].wdry:.5f} ≠ input {expected_kg:.5f}'
            )


# ---------------------------------------------------------------------------
# Tests — time-series (emis.txt equivalent)
# ---------------------------------------------------------------------------
class TestBurnupTimeseriesGolden:
    """Validate per-timestep wdf and ff against golden time-series data."""

    def test_timestep_count(self, burnup_results, ts_golden):
        results, _, _ = burnup_results
        assert len(results) == len(ts_golden), (
            f'Expected {len(ts_golden)} timesteps, got {len(results)}'
        )

    def test_first_timestep_time(self, burnup_results, ts_golden):
        results, _, _ = burnup_results
        assert abs(results[0].time - float(ts_golden.iloc[0]['time_s'])) < 1.0

    def test_last_timestep_time(self, burnup_results, ts_golden):
        results, _, _ = burnup_results
        assert abs(results[-1].time - float(ts_golden.iloc[-1]['time_s'])) < 1.0

    @pytest.mark.parametrize('step_idx', list(range(24)))
    def test_wdf_at_step(self, burnup_results, ts_golden, step_idx):
        """Overall remaining fraction at each timestep within atol=0.005."""
        results, _, _ = burnup_results
        if step_idx >= len(results):
            pytest.skip('step_idx beyond result length')
        actual = results[step_idx].wdf
        expected = float(ts_golden.iloc[step_idx]['wdf'])
        assert abs(actual - expected) <= _ATOL_WDF, (
            f'Step {step_idx} (t={results[step_idx].time:.0f}s): '
            f'wdf {actual:.6f} vs golden {expected:.6f} '
            f'(diff={abs(actual-expected):.6f}, atol={_ATOL_WDF})'
        )

    @pytest.mark.parametrize('step_idx', [0, 1, 2])
    def test_ff_at_flaming_steps(self, burnup_results, ts_golden, step_idx):
        """Flaming fraction at early (flaming-phase) steps within atol=0.02."""
        results, _, _ = burnup_results
        actual = results[step_idx].ff
        expected = float(ts_golden.iloc[step_idx]['ff'])
        assert abs(actual - expected) <= _ATOL_FF, (
            f'Step {step_idx} (t={results[step_idx].time:.0f}s): '
            f'ff {actual:.6f} vs golden {expected:.6f} '
            f'(diff={abs(actual-expected):.6f}, atol={_ATOL_FF})'
        )

    def test_ff_zero_after_flaming(self, burnup_results, ts_golden):
        """ff must be 0 for all smoldering-only timesteps (step ≥ 3)."""
        results, _, _ = burnup_results
        for i in range(3, len(results)):
            assert results[i].ff == pytest.approx(0.0, abs=_ATOL_FF), (
                f'Step {i} (t={results[i].time:.0f}s): ff={results[i].ff:.4f} '
                f'should be 0 in smoldering phase'
            )

    def test_wdf_monotonically_decreasing(self, burnup_results):
        """Overall remaining fraction must not increase between steps."""
        results, _, _ = burnup_results
        for i in range(1, len(results)):
            assert results[i].wdf <= results[i - 1].wdf + 1e-9, (
                f'wdf increased at step {i}: '
                f'{results[i-1].wdf:.6f} → {results[i].wdf:.6f}'
            )


# ---------------------------------------------------------------------------
# Smoke test — simulation completes without exception
# ---------------------------------------------------------------------------
def test_burnup_runs_without_error():
    """Basic smoke test: run_burnup returns (results, summary, class_order)."""
    inp = pd.read_csv(_INPUT_CSV, comment='#')
    fl = {r['component']: r['load_tac'] * _TAC_TO_KGM2 for _, r in inp.iterrows()}
    fm = {r['component']: r['moisture_fraction'] for _, r in inp.iterrows()}
    results, summary, class_order = run_burnup(
        fuel_loadings=fl, fuel_moistures=fm,
        intensity=50.0, ig_time=60.0, windspeed=0.0,
        depth=0.3, ambient_temp=21.0,
    )
    assert len(results) > 0
    assert len(summary) == len(fl)
    assert len(class_order) == len(fl)

