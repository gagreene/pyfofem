#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression test for burnup_array against the scalar burnup() implementation.

Runs the same canonical inputs through both engines and validates that the
vectorized array version produces results consistent with the scalar version.

The array version uses simplified qdot averaging (latest value instead of
rolling window), so tolerances are wider than the scalar-vs-C++ golden tests.
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
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from pyfofem.components.burnup_scalar import (
    burnup as scalar_burnup,
)
from pyfofem.components.burnup import (
    FuelParticle,
    burnup as burnup_array,
    BurnupArrayResult,
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_INPUTS_DIR = os.path.join(_TESTS_DIR, 'test_data', 'test_inputs')
_INPUT_CSV = os.path.join(_INPUTS_DIR, 'burnup_input.csv')

_TAC_TO_KGM2 = 1.0 / 4.4609


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def canonical_inputs():
    """Load canonical test inputs from burnup_input.csv."""
    inp = pd.read_csv(_INPUT_CSV, comment='#')
    return inp


@pytest.fixture(scope='module')
def scalar_result(canonical_inputs):
    """Run scalar burnup once and cache."""
    inp = canonical_inputs
    particles = []
    for _, row in inp.iterrows():
        particles.append(FuelParticle(
            wdry=row['load_tac'] * _TAC_TO_KGM2,
            htval=1.86e7,
            fmois=row['moisture_fraction'],
            dendry=513.0,
            sigma=row['sav_per_m'],
            cheat=2750.0,
            condry=0.133,
            tpig=327.0,
            tchar=377.0,
            ash=0.05,
        ))
    results, summary = scalar_burnup(
        particles=particles,
        fi=50.0, ti=60.0, u=0.0, d=0.3, tamb=21.0,
        r0=1.83, dr=0.4, dt=15.0, ntimes=3000,
        validate=True,
    )
    return results, summary


@pytest.fixture(scope='module')
def array_result(canonical_inputs):
    """Run array burnup with C=1 and cache."""
    inp = canonical_inputs
    P = len(inp)

    wdry   = np.array([row['load_tac'] * _TAC_TO_KGM2 for _, row in inp.iterrows()]).reshape(1, P)
    htval  = np.full((1, P), 1.86e7)
    fmois  = np.array([row['moisture_fraction'] for _, row in inp.iterrows()]).reshape(1, P)
    dendry = np.full((1, P), 513.0)
    sigma  = np.array([row['sav_per_m'] for _, row in inp.iterrows()]).reshape(1, P)
    cheat  = np.full((1, P), 2750.0)
    condry = np.full((1, P), 0.133)
    tpig   = np.full((1, P), 327.0)
    tchar  = np.full((1, P), 377.0)
    ash    = np.full((1, P), 0.05)

    result = burnup_array(
        wdry=wdry, htval=htval, fmois=fmois, dendry=dendry, sigma=sigma,
        cheat=cheat, condry=condry, tpig=tpig, tchar=tchar, ash=ash,
        fi=np.array([50.0]), ti=np.array([60.0]),
        u=np.array([0.0]), d=np.array([0.3]),
        tamb=np.array([21.0]),
        r0=np.array([1.83]), dr=np.array([0.4]),
        dt=15.0, max_timesteps=3000,
        validate=True,
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBurnupArrayBasic:
    """Basic functionality tests for the array burnup module."""

    def test_no_error(self, array_result):
        """Array burnup should succeed (error_code = 0)."""
        assert array_result.error_code[0] == 0, (
            f'Expected error_code=0, got {array_result.error_code[0]}'
        )

    def test_has_timesteps(self, array_result):
        """Should produce at least one timestep."""
        assert array_result.n_steps[0] > 0

    def test_output_shapes(self, array_result):
        """Output arrays should have correct shapes."""
        assert array_result.time.shape[0] == 1
        assert array_result.wdf.shape[0] == 1
        assert array_result.ff.shape[0] == 1

    def test_wdf_monotonic(self, array_result):
        """wdf should be monotonically non-increasing."""
        n = array_result.n_steps[0]
        wdf = array_result.wdf[0, :n]
        for i in range(1, len(wdf)):
            assert wdf[i] <= wdf[i - 1] + 1e-6, (
                f'wdf increased at step {i}: {wdf[i-1]:.6f} → {wdf[i]:.6f}'
            )


class TestBurnupArrayVsScalar:
    """Compare array burnup (C=1) against scalar burnup."""

    def test_fine_fuels_consumed(self, array_result):
        """Fine fuels (litter, 1-hr) should be nearly fully consumed."""
        # First two components after sorting by decreasing SAV
        # should have very low remaining fraction
        frac = array_result.summary_frac_remaining[0]
        # At least the two finest classes should be mostly consumed
        assert frac[0] < 0.05, f'Finest fuel frac_remaining={frac[0]:.4f}'
        assert frac[1] < 0.05, f'Second finest frac_remaining={frac[1]:.4f}'

    def test_coarse_fuels_intact(self, array_result):
        """Coarse fuels should be mostly intact under low-intensity fire."""
        frac = array_result.summary_frac_remaining[0]
        # Last few components (coarsest) should be > 0.9
        P = len(frac)
        for i in range(max(0, P - 4), P):
            assert frac[i] > 0.90, (
                f'Component {i} frac_remaining={frac[i]:.4f}, expected > 0.90'
            )

    def test_scalar_vs_array_frac_remaining_direction(self, scalar_result,
                                                       array_result):
        """Array result should agree with scalar on which fuels are consumed.

        This is a directional check — both should agree that fine fuels are
        consumed and coarse fuels remain.  Exact values may differ due to
        the simplified qdot averaging in the array version.
        """
        _, summary = scalar_result
        scalar_frac = np.array([s.frac_remaining for s in summary])
        array_frac = array_result.summary_frac_remaining[0]

        # Both should have same number of components
        assert len(scalar_frac) == len(array_frac), (
            f'Component count mismatch: scalar={len(scalar_frac)}, '
            f'array={len(array_frac)}'
        )

        # Fine fuels: both should be < 0.1
        for i in range(2):
            assert scalar_frac[i] < 0.1, f'Scalar component {i} not consumed'
            assert array_frac[i] < 0.1, f'Array component {i} not consumed'

        # Coarse fuels: both should be > 0.5
        for i in range(max(0, len(scalar_frac) - 4), len(scalar_frac)):
            assert scalar_frac[i] > 0.5, f'Scalar component {i} unexpectedly consumed'
            assert array_frac[i] > 0.5, f'Array component {i} unexpectedly consumed'

    def test_scalar_vs_array_frac_remaining_tight(self, scalar_result,
                                                   array_result):
        """Array frac_remaining should match scalar within 0.001 per component.

        With the full qdot rolling-window average, numerical agreement is
        very close — max difference should be well under 1 percentage point.
        """
        _, summary = scalar_result
        scalar_frac = np.array([s.frac_remaining for s in summary])
        array_frac = array_result.summary_frac_remaining[0]

        np.testing.assert_allclose(
            array_frac, scalar_frac, atol=0.001,
            err_msg='Array frac_remaining deviates from scalar by > 0.001',
        )

    def test_scalar_vs_array_step_count(self, scalar_result, array_result):
        """Both engines should produce the same number of timesteps."""
        results, _ = scalar_result
        assert array_result.n_steps[0] == len(results), (
            f'Step count mismatch: scalar={len(results)}, '
            f'array={array_result.n_steps[0]}'
        )

    def test_scalar_vs_array_wdf_timeseries(self, scalar_result, array_result):
        """Per-step wdf should match within atol=0.001."""
        results, _ = scalar_result
        n = min(len(results), array_result.n_steps[0])
        scalar_wdf = np.array([r.wdf for r in results[:n]])
        array_wdf = array_result.wdf[0, :n]
        np.testing.assert_allclose(
            array_wdf, scalar_wdf, atol=0.001,
            err_msg='Array wdf timeseries deviates from scalar by > 0.001',
        )


class TestBurnupArrayMultiCell:
    """Test with multiple cells (C > 1)."""

    def test_two_identical_cells(self, canonical_inputs):
        """Two identical cells should produce identical results."""
        inp = canonical_inputs
        P = len(inp)

        wdry_1d = np.array([row['load_tac'] * _TAC_TO_KGM2 for _, row in inp.iterrows()])
        fmois_1d = np.array([row['moisture_fraction'] for _, row in inp.iterrows()])
        sigma_1d = np.array([row['sav_per_m'] for _, row in inp.iterrows()])

        # Stack two identical cells
        wdry   = np.tile(wdry_1d, (2, 1))
        fmois  = np.tile(fmois_1d, (2, 1))
        sigma  = np.tile(sigma_1d, (2, 1))
        htval  = np.full((2, P), 1.86e7)
        dendry = np.full((2, P), 513.0)
        cheat  = np.full((2, P), 2750.0)
        condry = np.full((2, P), 0.133)
        tpig   = np.full((2, P), 327.0)
        tchar  = np.full((2, P), 377.0)
        ash    = np.full((2, P), 0.05)

        result = burnup_array(
            wdry=wdry, htval=htval, fmois=fmois, dendry=dendry, sigma=sigma,
            cheat=cheat, condry=condry, tpig=tpig, tchar=tchar, ash=ash,
            fi=np.array([50.0, 50.0]), ti=np.array([60.0, 60.0]),
            u=np.array([0.0, 0.0]), d=np.array([0.3, 0.3]),
            tamb=np.array([21.0, 21.0]),
            r0=np.array([1.83, 1.83]), dr=np.array([0.4, 0.4]),
            dt=15.0, max_timesteps=3000,
        )

        # Both cells should have no errors
        assert result.error_code[0] == 0
        assert result.error_code[1] == 0

        # Both cells should have the same number of steps
        assert result.n_steps[0] == result.n_steps[1]

        # Remaining fractions should be identical
        np.testing.assert_allclose(
            result.summary_frac_remaining[0],
            result.summary_frac_remaining[1],
            atol=1e-10,
        )

        # wdf timeseries should be identical
        n = result.n_steps[0]
        np.testing.assert_allclose(
            result.wdf[0, :n],
            result.wdf[1, :n],
            atol=1e-10,
        )

    def test_mixed_valid_invalid_cells(self, canonical_inputs):
        """A batch with one valid and one invalid cell should handle both."""
        inp = canonical_inputs
        P = len(inp)

        wdry_1d = np.array([row['load_tac'] * _TAC_TO_KGM2 for _, row in inp.iterrows()])
        fmois_1d = np.array([row['moisture_fraction'] for _, row in inp.iterrows()])
        sigma_1d = np.array([row['sav_per_m'] for _, row in inp.iterrows()])

        # Cell 0: valid; Cell 1: zero fuel
        wdry = np.zeros((2, P))
        wdry[0, :] = wdry_1d
        # wdry[1, :] stays 0 — no fuel

        fmois = np.tile(fmois_1d, (2, 1))
        sigma = np.tile(sigma_1d, (2, 1))
        htval  = np.full((2, P), 1.86e7)
        dendry = np.full((2, P), 513.0)
        cheat  = np.full((2, P), 2750.0)
        condry = np.full((2, P), 0.133)
        tpig   = np.full((2, P), 327.0)
        tchar  = np.full((2, P), 377.0)
        ash    = np.full((2, P), 0.05)

        result = burnup_array(
            wdry=wdry, htval=htval, fmois=fmois, dendry=dendry, sigma=sigma,
            cheat=cheat, condry=condry, tpig=tpig, tchar=tchar, ash=ash,
            fi=np.array([50.0, 50.0]), ti=np.array([60.0, 60.0]),
            u=np.array([0.0, 0.0]), d=np.array([0.3, 0.3]),
            tamb=np.array([21.0, 21.0]),
            r0=np.array([1.83, 1.83]), dr=np.array([0.4, 0.4]),
            dt=15.0, max_timesteps=3000,
        )

        # Cell 0 should succeed
        assert result.error_code[0] == 0
        assert result.n_steps[0] > 0

        # Cell 1 should have error code 90 (no fuel)
        assert result.error_code[1] == 90
        assert result.n_steps[1] == 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_burnup_array_smoke(canonical_inputs):
    """Basic smoke test: burnup_array completes without exception."""
    inp = canonical_inputs
    P = len(inp)

    wdry = np.array([row['load_tac'] * _TAC_TO_KGM2 for _, row in inp.iterrows()]).reshape(1, P)
    fmois = np.array([row['moisture_fraction'] for _, row in inp.iterrows()]).reshape(1, P)
    sigma = np.array([row['sav_per_m'] for _, row in inp.iterrows()]).reshape(1, P)

    result = burnup_array(
        wdry=wdry,
        htval=np.full((1, P), 1.86e7),
        fmois=fmois,
        dendry=np.full((1, P), 513.0),
        sigma=sigma,
        cheat=np.full((1, P), 2750.0),
        condry=np.full((1, P), 0.133),
        tpig=np.full((1, P), 327.0),
        tchar=np.full((1, P), 377.0),
        ash=np.full((1, P), 0.05),
        fi=np.array([50.0]),
        ti=np.array([60.0]),
        u=np.array([0.0]),
        d=np.array([0.3]),
        tamb=np.array([21.0]),
        r0=np.array([1.83]),
        dr=np.array([0.4]),
        dt=15.0,
        max_timesteps=100,  # short for smoke test
    )

    assert isinstance(result, BurnupArrayResult)
    assert result.error_code[0] == 0
    assert result.n_steps[0] > 0




