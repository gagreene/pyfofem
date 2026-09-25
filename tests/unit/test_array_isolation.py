#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_array_isolation.py - Phase 8 item A: mixed-validity array
isolation.

Exercises the real public integration routes with multi-row inputs
containing both valid and invalid cells, proving that a bad cell cannot
alter or contaminate a valid cell's result. Two genuinely distinct
"invalid cell" mechanisms exist in this codebase's public surface and
are both covered here:

- **Burnup** (``run_fofem_emissions(use_burnup=True)``): a per-cell
  fire-intensity value below ``_FIRE_BOUNDS['fistart'][0]`` is REJECTED
  by ``_run_burnup_cell()`` with a nonzero ``BurnupError`` code, and all
  of that cell's per-cell consumption/duration outputs are zeroed by
  ``run_fofem_emissions()``'s own step 5b - the exact per-cell error
  isolation Phase 7 item A already covers at the ``_run_burnup_cell()``
  unit level (``test_run_burnup_cell_error_codes.py``). This module
  covers the ARRAY-LEVEL question Phase 7 did not: given a batch with
  invalid cells interleaved among valid ones, does a valid cell's
  result match the SAME cell run in isolation, and does reordering
  independent cells (then restoring the original order) change anything.
- **Mortality** (``mort_bolchar``): an unsupported species code produces
  ``np.nan`` for that cell only (a vectorized boolean-mask computation,
  not an exception), with a printed warning. This module proves the
  warning text does not leak information about VALID cells and that
  repeated calls carry no persistent module-level accumulator across
  calls.

**Explicitly NOT a genuine invalid-cell case, confirmed by direct source
inspection rather than assumed**: the ``consm_*`` consumption-family
functions (``consm_duff``, ``consm_litter``, ``consm_herb``,
``consm_shrub``, ``consm_canopy``) have no per-cell error/invalid state
of their own - an unrecognized ``reg``/``cvr_grp`` string silently falls
through to a documented default equation branch (see each function's own
routing table) rather than raising or flagging that cell as invalid.
There is therefore no "bad consumption cell" to isolate at that layer
independent of the burnup/mortality mechanisms already covered above;
forcing an artificial one would not reflect this codebase's real
contract, so none was added.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem import mort_bolchar, run_fofem_emissions
from pyfofem.components.burnup import _FIRE_BOUNDS
from tests.cpp_parity_live._driver_support import (
    KNOWN_PERCELL_FIELDS,
    assert_known_percell_field_inventory,
    percell_fields,
)

#: Fields NOT expected to be zeroed for an invalid (BurnupError != 0)
#: cell, confirmed by direct measurement against this module's own
#: fixture: the two status fields themselves (``BurnupError`` is
#: SUPPOSED to be nonzero for an invalid cell; ``BurnupLimitAdj`` is an
#: independent clip-code field, unaffected by the error path), every
#: pre-fire ``*Pre`` load (never zeroed - it reflects the INPUT, not
#: burnup's output), and every ``*-Equ`` equation-ID field (resolved
#: earlier in the pipeline than burnup, so an invalid burnup cell does
#: not affect it).
_NOT_ZEROED_FOR_INVALID = frozenset(
    {'BurnupError', 'BurnupLimitAdj'}
    | {f for f in KNOWN_PERCELL_FIELDS if f.endswith('Pre') or f.endswith('-Equ')}
)

#: A fire intensity guaranteed to be rejected by ``_run_burnup_cell()``'s
#: own lower-bound check (``burnup_error`` code 10) - well below the real,
#: currently-verified ``_FIRE_BOUNDS['fistart'][0]`` lower bound (40.0),
#: read from the live module rather than hardcoded so this test tracks the
#: bound if it is ever corrected again.
_INVALID_HFI = _FIRE_BOUNDS['fistart'][0] / 2.0

#: A fire intensity comfortably inside the valid ``fistart`` range.
_VALID_HFI = 500.0

#: Per-cell litter loads (T/ac), deliberately distinct so a shuffled/
#: contaminated result cannot masquerade as correct by coincidence.
_LITTER_5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

#: hfi pattern with invalid cells BEFORE (index 0), BETWEEN (index 2, sitting
#: between the two valid cells at 1 and 3), and AFTER (index 4) valid cells.
_HFI_PATTERN = np.array([_INVALID_HFI, _VALID_HFI, _INVALID_HFI, _VALID_HFI, _INVALID_HFI])

_VALID_INDICES = (1, 3)
_INVALID_INDICES = (0, 2, 4)


def _run_batch(litter: np.ndarray, hfi: np.ndarray) -> dict:
    """
    Run ``run_fofem_emissions()`` for a batch of cells with per-cell
    litter loads and fire intensities, all other loads/moistures fixed.

    :param litter: Per-cell pre-fire litter load (T/ac).
    :param hfi: Per-cell head fire intensity (kW/m).
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    n = len(litter)
    zeros = np.zeros(n)
    return run_fofem_emissions(
        litter=litter, duff=zeros, duff_depth=zeros, herb=zeros, shrub=zeros,
        crown_foliage=zeros, crown_branch=zeros, pct_crown_burned=zeros,
        region=np.array(['InteriorWest'] * n),
        use_burnup=True, num_workers=1, moisture_regime='Dry', units='Imperial',
        hfi=hfi,
    )


def _run_single_cell(litter_value: float, hfi_value: float) -> dict:
    """
    Run ``run_fofem_emissions()`` for exactly one cell (scalar inputs).

    :param litter_value: Pre-fire litter load (T/ac) for the single cell.
    :param hfi_value: Head fire intensity (kW/m) for the single cell.
    :return: The full ``run_fofem_emissions()`` result dict (scalar values).
    """
    return run_fofem_emissions(
        litter=litter_value, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region='InteriorWest',
        use_burnup=True, num_workers=1, moisture_regime='Dry', units='Imperial',
        hfi=hfi_value,
    )


def test_mixed_validity_batch_invalid_cells_are_flagged_and_zeroed():
    """Every invalid cell (positions 0, 2, 4 - before, between, and after
    the two valid cells) must carry a nonzero ``BurnupError`` and every
    OTHER per-cell output (the complete field set, not a hand-picked
    subset - see :data:`_NOT_ZEROED_FOR_INVALID` for the 3 field
    CLASSES that are legitimately exempt, confirmed by direct
    measurement, not assumption) zeroed; every valid cell (1, 3) must
    carry ``BurnupError == 0`` and a finite, positive ``LitCon``."""
    result = _run_batch(_LITTER_5, _HFI_PATTERN)
    assert_known_percell_field_inventory(result, len(_LITTER_5))

    burnup_error = np.asarray(result['BurnupError'])
    for idx in _INVALID_INDICES:
        assert burnup_error[idx] != 0, f'cell {idx} (invalid) should carry a nonzero BurnupError'
        for key in KNOWN_PERCELL_FIELDS - _NOT_ZEROED_FOR_INVALID:
            value = np.asarray(result[key])[idx]
            assert value == 0.0, f'cell {idx} (invalid) {key} should be zeroed, got {value!r}'

    lit_con = np.asarray(result['LitCon'])
    for idx in _VALID_INDICES:
        assert burnup_error[idx] == 0, f'cell {idx} (valid) should carry BurnupError == 0'
        assert np.isfinite(lit_con[idx]) and lit_con[idx] > 0.0, (
            f'cell {idx} (valid) LitCon should be finite and positive'
        )
    # Discriminating check: a non-consumption downstream (emissions) field
    # is genuinely nonzero for a valid cell, so the zeroing assertion above
    # could not pass merely because every field happens to be zero anyway.
    assert np.asarray(result['PM10F'])[_VALID_INDICES[0]] > 0.0


def test_mixed_validity_batch_reordering_valid_cells_preserves_results():
    """Permute the 5 cells into a different order, run the batch again,
    then apply the inverse permutation to restore the original order -
    the restored result must exactly match the original (unpermuted)
    batch's result for every valid cell, across the COMPLETE per-cell
    field set, proving cell processing order has no effect on
    outcome."""
    baseline = _run_batch(_LITTER_5, _HFI_PATTERN)

    perm = np.array([3, 1, 4, 0, 2])  # a genuine shuffle, not identity
    shuffled_litter = _LITTER_5[perm]
    shuffled_hfi = _HFI_PATTERN[perm]
    shuffled_result = _run_batch(shuffled_litter, shuffled_hfi)

    inverse_perm = np.argsort(perm)
    for key in percell_fields(baseline, len(_LITTER_5)):
        restored = np.asarray(shuffled_result[key])[inverse_perm]
        baseline_arr = np.asarray(baseline[key])
        np.testing.assert_allclose(
            restored, baseline_arr, atol=1e-9,
            err_msg=f'{key} changed after reordering cells and restoring the original order',
        )


def test_mixed_validity_batch_valid_cells_match_the_same_cell_run_alone():
    """A valid cell's result inside the mixed batch must exactly match
    the SAME cell (same litter load, same fire intensity) run as an
    isolated, single-cell call - across the COMPLETE per-cell field set
    (not a hand-picked subset) - proving no cross-cell contamination
    from the interleaved invalid cells."""
    batch = _run_batch(_LITTER_5, _HFI_PATTERN)
    fields = percell_fields(batch, len(_LITTER_5))

    for idx in _VALID_INDICES:
        solo = _run_single_cell(float(_LITTER_5[idx]), _VALID_HFI)
        for key in fields:
            batch_val = np.asarray(batch[key])[idx]
            solo_val = solo[key]
            assert batch_val == pytest.approx(solo_val, abs=1e-9), (
                f"cell {idx}'s batch {key}={batch_val} does not match its "
                f'isolated single-cell run {key}={solo_val}'
            )


def test_mort_bolchar_repeated_calls_carry_no_persistent_accumulator():
    """Calling ``mort_bolchar`` twice with DIFFERENT unsupported species
    exposed in between must not change the result for an identical
    valid-species cell across the two calls - proving there is no
    module-level accumulator/warning-state that persists or leaks across
    separate calls."""
    result_1 = mort_bolchar(
        np.array(['ZZZZ1', 'QUAL']), np.array([10.0, 20.0]), np.array([1.0, 2.0]),
    )
    result_2 = mort_bolchar(
        np.array(['ZZZZ2', 'QUAL']), np.array([10.0, 20.0]), np.array([1.0, 2.0]),
    )
    assert np.isnan(result_1[0]) and np.isnan(result_2[0])
    assert result_1[1] == pytest.approx(result_2[1]), (
        'the QUAL cell result changed depending on which unsupported species '
        'code was exposed in an earlier call - suggests leaked/persistent state'
    )


def test_mort_bolchar_unsupported_cells_do_not_contaminate_supported_cells(capsys):
    """A 5-tree batch with unsupported species BEFORE, BETWEEN, and AFTER
    two supported (QUAL) cells with distinct dbh/char_ht must return NaN
    only for the unsupported cells, and the supported cells' values must
    match the same cells run in isolation. The printed warning must name
    only the unsupported codes, never mentioning the supported species -
    proving the warning does not leak information across cells."""
    spp = np.array(['UNK1', 'QUAL', 'UNK2', 'QUAL', 'UNK3'])
    dbh = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    char_ht = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = mort_bolchar(spp, dbh, char_ht)

    for idx in (0, 2, 4):
        assert np.isnan(result[idx]), f'cell {idx} (unsupported species) should be NaN'
    for idx in (1, 3):
        assert np.isfinite(result[idx]), f'cell {idx} (QUAL) should be a finite probability'
        solo = mort_bolchar('QUAL', float(dbh[idx]), float(char_ht[idx]))
        assert result[idx] == pytest.approx(solo), (
            f'cell {idx} (QUAL) result differs from the same cell run in isolation'
        )

    captured = capsys.readouterr()
    assert 'QUAL' not in captured.out, (
        'the unsupported-species warning must not mention the supported QUAL cells'
    )
    for unk in ('UNK1', 'UNK2', 'UNK3'):
        assert unk in captured.out, f'warning text should name the unsupported code {unk}'
