#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_2d_input_regression.py - Phase 7 item C: 2D-input regression
coverage for the former ``np.atleast_1d`` -> ``np.ravel`` bug class
(fixed 2026-08-26; see ``docs/CODEBASE.md`` and
``tests/regression/test_pr1_review_regressions.py``), organized by
PUBLIC API FAMILY rather than one brittle test per historical internal
call site.

Three families are covered, each proven with genuinely varying (not
uniformly-filled) per-cell input values so the assertions actually
demonstrate C-order cell-ordering, not merely a shape match:

- **Mortality family** (``mortality_calcs.py``): ``mort_bolchar``,
  ``mort_crcabe``, ``mort_crnsch`` - all originally affected by the
  boolean-species-mask-vs-1D-output-array ``IndexError`` this bug class
  produced.
- **Consumption family** (``consumption_calcs.py``): ``consm_canopy``
  (already regression-tested pre-Phase-7, generalized here) and
  ``consm_duff`` (a second, structurally distinct signature - region/
  cover-group code plus numeric loading/moisture).
- **Orchestrator broadcast step** (``pyfofem.py::run_fofem_emissions()``):
  the top-level public entry point whose own broadcast-to-length-n step
  uses the same fixed helpers.

A fourth, DELIBERATELY DIFFERENT case is also characterized:
``consumption_calcs.py::calc_carbon()`` (utility family) never needed
the ``atleast_1d`` -> ``ravel`` fix and does NOT flatten 2D input at
all - it is a pure elementwise ``loading * carbon_fraction`` multiply
with no boolean-mask indexing, so it preserves the caller's original
array shape. This is verified directly here (not assumed) precisely
because item C requires characterizing "flattening/broadcast semantics
... as defined by the current public contract", and that contract is
NOT uniform across every public function - conflating the two would be
a real assertion made without evidence in the opposite direction the
Phase 3 findings (F-24/F-28/F-33/F-34) already made a project-wide
practice.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""

from __future__ import annotations
import numpy as np
import pytest
from pyfofem import (
    calc_carbon,
    consm_canopy,
    consm_duff,
    mort_bolchar,
    mort_crcabe,
    mort_crnsch,
    run_fofem_emissions,
)
_SHAPES = ((6,), (1, 6), (6, 1), (2, 3))


def _reshape_all(flat: np.ndarray):
    """Return ``{shape: flat.reshape(shape)}`` for every shape in
    :data:`_SHAPES`.

    :param flat: A 1D array of exactly 6 elements.
    :return: Dict mapping each shape tuple to the reshaped view.
    """
    assert flat.shape == (6,)
    return {shape: flat.reshape(shape) for shape in _SHAPES}


def _run_fofem_emissions_with_shape(shape):
    """Call ``run_fofem_emissions()`` with a 6-cell, genuinely-varying
    ``litter`` loading reshaped to *shape*, ``use_burnup=False`` (burnup
    itself is a per-cell, non-vectorized simulation covered separately
    by items A/E/F, not part of this broadcast-step regression), and a
    fixed ``moisture_regime`` so no moisture input is required.

    :param shape: Target shape for every per-cell input array.
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    litter_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    zeros_flat = np.zeros(6)
    region_flat = np.array(['InteriorWest'] * 6)
    return run_fofem_emissions(
        litter=litter_flat.reshape(shape),
        duff=zeros_flat.reshape(shape),
        duff_depth=zeros_flat.reshape(shape),
        herb=zeros_flat.reshape(shape),
        shrub=zeros_flat.reshape(shape),
        crown_foliage=zeros_flat.reshape(shape),
        crown_branch=zeros_flat.reshape(shape),
        pct_crown_burned=zeros_flat.reshape(shape),
        region=region_flat.reshape(shape),
        use_burnup=False,
        moisture_regime='Dry',
        units='Imperial',
    )


def test_calc_carbon_preserves_input_shape_rather_than_flattening():
    """``calc_carbon`` (utility family) is a pure elementwise
    ``loading * carbon_fraction`` multiply with no boolean-mask
    indexing anywhere in its implementation, so it never needed (and
    does not have) the ``atleast_1d`` -> ``ravel`` fix: a genuinely 2D
    input loading array comes back as the SAME 2D shape, not flattened
    - directly confirmed by execution, the opposite of every other
    function in this module."""
    duff_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    duff_2d = duff_flat.reshape(2, 3)
    result_2d = calc_carbon({'duff': duff_2d})['duff']
    result_flat = calc_carbon({'duff': duff_flat})['duff']
    assert result_2d.shape == (2, 3)
    np.testing.assert_array_equal(result_2d, result_flat.reshape(2, 3))


def test_calc_carbon_scalar_input_returns_a_plain_float():
    """Scalar-array convention for the one function in this module that
    does not flatten arrays: an all-scalar loading dict must still
    return plain ``float`` values, not 0-d arrays."""
    result = calc_carbon({'duff': 1.0})
    assert isinstance(result['duff'], float)


def test_consm_canopy_ravels_to_1d_preserving_cell_order():
    """Generalizes the pre-Phase-7 ``test_consm_canopy_accepts_2d_input``
    (which used uniform fills, so it could not distinguish "flattened
    correctly" from "flattened to the wrong order but every cell happens
    to be identical") to genuinely varying per-cell values across all
    four shapes."""
    crown_burn_flat = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    pre_fl_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pre_bl_flat = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    cb_shapes = _reshape_all(crown_burn_flat)
    fl_shapes = _reshape_all(pre_fl_flat)
    bl_shapes = _reshape_all(pre_bl_flat)

    baseline = consm_canopy(cb_shapes[(6,)], fl_shapes[(6,)], bl_shapes[(6,)], units='Imperial')
    assert baseline['flc'].shape == (6,)
    assert baseline['blc'].shape == (6,)

    for shape in _SHAPES[1:]:
        result = consm_canopy(cb_shapes[shape], fl_shapes[shape], bl_shapes[shape], units='Imperial')
        assert result['flc'].shape == (6,), f'shape {shape} flc did not flatten to (6,)'
        assert result['blc'].shape == (6,), f'shape {shape} blc did not flatten to (6,)'
        np.testing.assert_array_equal(result['flc'], baseline['flc'])
        np.testing.assert_array_equal(result['blc'], baseline['blc'])


def test_consm_duff_ravels_to_1d_preserving_cell_order():
    """A second, structurally distinct consumption-family signature
    (region/cover-group STRING codes plus numeric loading/moisture, one
    of several ``reg=``/``cvr_grp=`` keyword-style ``consm_*``
    functions) - proven with genuinely varying loading AND moisture
    values, since (as directly confirmed by probing this function ahead
    of writing this test) ``pdc`` depends on both, not loading alone."""
    dl_flat = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    dm_flat = np.array([30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    dl_shapes = _reshape_all(dl_flat)
    dm_shapes = _reshape_all(dm_flat)
    kwargs = dict(reg='InteriorWest', cvr_grp=None, duff_moist_cat='edm',
                  d_pre=2.0, units='Imperial')

    baseline = consm_duff(dl_shapes[(6,)], dm_shapes[(6,)], **kwargs)['pdc']
    assert baseline.shape == (6,)
    assert len(set(baseline.tolist())) == 6, 'fixture values collapsed to <6 distinct results'

    for shape in _SHAPES[1:]:
        result = consm_duff(dl_shapes[shape], dm_shapes[shape], **kwargs)['pdc']
        assert result.shape == (6,), f'shape {shape} pdc did not flatten to (6,)'
        np.testing.assert_array_equal(result, baseline)


def test_consm_duff_scalar_and_array_pdc_directions_are_consistent():
    """Companion to :func:`test_consm_duff_scalar_moisture_with_array_
    loading_broadcasts_to_6` below: exercises the wholly-scalar
    direction (plain ``float`` output, not an array) and the REVERSE
    mixed direction (scalar ``pre_dl``, array ``duff_moist``), proving
    the latter genuinely varies per element (``pdc`` DOES depend on
    ``duff_moist``), unlike the array-``pre_dl``/scalar-``duff_moist``
    direction below where every broadcast element is identical."""
    scalar_result = consm_duff(
        2.0, 50.0, reg='InteriorWest', cvr_grp=None, duff_moist_cat='edm',
        d_pre=2.0, units='Imperial',
    )['pdc']
    assert isinstance(scalar_result, float)

    dm = np.array([30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    reverse_result = consm_duff(
        2.0, dm, reg='InteriorWest', cvr_grp=None, duff_moist_cat='edm',
        d_pre=2.0, units='Imperial',
    )['pdc']
    assert reverse_result.shape == (6,)
    assert len(set(reverse_result.tolist())) == 6, (
        'fixture duff_moist values collapsed to <6 distinct pdc results'
    )


def test_consm_duff_scalar_moisture_with_array_loading_broadcasts_to_6():
    """Mixing an ARRAY ``pre_dl`` (6 distinct loads) with a SCALAR
    ``duff_moist`` now correctly broadcasts the scalar moisture across
    every array position, matching this codebase's Scalar-Array
    Convention (``docs/CODEBASE.md``) and NumPy's own elementwise
    broadcasting rules.

    **Regression history** (an F-62-adjacent acceptance-recovery pass,
    2026-09-21, investigated this directly - not assumed): an EARLIER
    version of this test pinned the OPPOSITE, defective behavior as a
    "current-contract characterization" - ``pdc``'s shape used to be
    derived purely from ``duff_moist``'s own raveled shape
    (``pdc = np.full_like(duff_moist, ...)``), so a scalar
    ``duff_moist`` collapsed ``pdc`` to shape ``(1,)`` regardless of
    ``pre_dl``'s actual length, a genuine broadcast-contract defect.
    Direct inspection of ``consm_duff()`` (``consumption_calcs.py``)
    found this fixed as a side effect of an unrelated, well-documented,
    concurrently-added physical-boundary-condition line -
    ``pdc = np.where(pre_dl <= 0.0, 0.0, pdc)`` ("there is nothing to
    consume when the duff load is zero") - whose own ordinary NumPy
    broadcasting (a shape ``(1,)`` ``pdc`` against a shape ``(6,)``
    ``pre_dl`` condition) now upgrades ``pdc`` to shape ``(6,)``. This
    was verified as a genuine, desired fix (not a coincidental shape
    match): each broadcast element below is checked against a true
    single-scalar-input call, proving real elementwise broadcast, not
    an accidental reshape; ``ddc``/``rdd`` (which also broadcast, via
    ``d_pre``, a separate optional scalar parameter) are checked too;
    and the wholly-scalar and array-``duff_moist``-with-scalar-
    ``pre_dl`` directions are exercised by
    :func:`test_consm_duff_scalar_and_array_pdc_directions_are_consistent`
    above. No production code was changed by this pass - only this
    stale test assertion, which pinned the now-fixed defect rather than
    the corrected behavior."""
    dl = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    result = consm_duff(
        dl, 50.0, reg='InteriorWest', cvr_grp=None, duff_moist_cat='edm',
        d_pre=2.0, units='Imperial',
    )
    pdc, ddc, rdd = result['pdc'], result['ddc'], result['rdd']
    assert pdc.shape == (6,), (
        f'expected the scalar duff_moist to broadcast across all 6 '
        f'pre_dl positions (matching the Scalar-Array Convention), got '
        f'shape {pdc.shape} - the pre-2026-09-21 defect collapsed this '
        f'to (1,)'
    )
    assert ddc.shape == (6,)
    assert rdd.shape == (6,)

    single = consm_duff(
        5.0, 50.0, reg='InteriorWest', cvr_grp=None, duff_moist_cat='edm',
        d_pre=2.0, units='Imperial',
    )
    # pdc/ddc/rdd for THIS route depend only on duff_moist/d_pre (both
    # scalar here), not on pre_dl's own per-element value, so every
    # broadcast element must equal the true single-scalar-input result -
    # proving genuine elementwise broadcast, not a coincidental reshape.
    assert np.all(pdc == single['pdc'])
    assert np.all(ddc == single['ddc'])
    assert np.all(rdd == single['rdd'])


def test_mort_bolchar_mismatched_species_and_dbh_length_raises():
    """Invalid-shape/current-contract characterization: a SCALAR species
    string is NOT broadcast against an array ``dbh`` - it is internally
    converted to a length-1 array, so the boolean species mask no longer
    matches ``dbh``'s length and a real ``IndexError`` is raised. This
    is the exact original bug-class failure mode, now characterized as
    the function's documented (if surprising) current contract rather
    than silently fixed further."""
    with pytest.raises(IndexError):
        mort_bolchar('QUAL', np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]))


def test_mort_bolchar_ravels_to_1d_preserving_cell_order():
    """``mort_bolchar`` must flatten every shape variant to ``(6,)``, with
    values in the SAME order as the underlying flat sequence - proving
    real per-cell correctness, not just a matching shape (all 6 dbh/
    char_ht pairs are distinct, so a shuffled or truncated result would
    fail this)."""
    spp_flat = np.array(['QUAL'] * 6)
    dbh_flat = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    char_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spp_shapes = _reshape_all(spp_flat)
    dbh_shapes = _reshape_all(dbh_flat)
    char_shapes = _reshape_all(char_flat)

    baseline = mort_bolchar(spp_shapes[(6,)], dbh_shapes[(6,)], char_shapes[(6,)])
    assert baseline.shape == (6,)
    assert np.all(np.isfinite(baseline))

    for shape in _SHAPES[1:]:
        result = mort_bolchar(spp_shapes[shape], dbh_shapes[shape], char_shapes[shape])
        assert result.shape == (6,), f'shape {shape} did not flatten to (6,)'
        np.testing.assert_array_equal(result, baseline)


def test_mort_bolchar_scalar_inputs_return_a_plain_float():
    """Scalar-array convention: all-scalar inputs must return a plain
    ``float``, never a length-1 ``np.ndarray``."""
    result = mort_bolchar('QUAL', 30.0, 2.0)
    assert isinstance(result, float)


def test_mort_crcabe_ravels_to_1d_preserving_cell_order():
    """``mort_crcabe`` must flatten every shape variant to ``(6,)`` in
    C-order, matching its pre-Phase-7 2D regression test's contract but
    with genuinely distinct per-cell inputs (rather than uniform fills)
    so the ordering assertion is meaningful."""
    spp_flat = np.array(['PIPO'] * 6)
    dbh_flat = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    ht_flat = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    cd_flat = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ckr_flat = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    scorch_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fields = {
        'spp': _reshape_all(spp_flat), 'dbh': _reshape_all(dbh_flat),
        'ht': _reshape_all(ht_flat), 'cd': _reshape_all(cd_flat),
        'ckr': _reshape_all(ckr_flat), 'scorch': _reshape_all(scorch_flat),
    }

    baseline = mort_crcabe(
        fields['spp'][(6,)], fields['dbh'][(6,)], fields['ht'][(6,)],
        fields['cd'][(6,)], fields['ckr'][(6,)], fields['scorch'][(6,)],
    )
    assert baseline.shape == (6,)
    assert np.all(np.isfinite(baseline))
    assert len(set(baseline.tolist())) == 6, 'fixture values collapsed to <6 distinct results'

    for shape in _SHAPES[1:]:
        result = mort_crcabe(
            fields['spp'][shape], fields['dbh'][shape], fields['ht'][shape],
            fields['cd'][shape], fields['ckr'][shape], fields['scorch'][shape],
        )
        assert result.shape == (6,), f'shape {shape} did not flatten to (6,)'
        np.testing.assert_array_equal(result, baseline)


def test_mort_crnsch_ravels_to_1d_and_2d_shape_is_valid():
    """``mort_crnsch`` must flatten every shape variant to ``(6,)`` - the
    third mortality-family member, whose own keyword-heavy signature
    (``bark_thickness``/``scorch_ht``/``flame_length``) is a distinct
    call shape from ``mort_bolchar``/``mort_crcabe``'s positional-only
    style."""
    spp_flat = np.array(['PSME'] * 6)
    dbh_flat = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    ht_flat = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    cd_flat = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    bark_flat = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    scorch_flat = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    fields = {
        'spp': _reshape_all(spp_flat), 'dbh': _reshape_all(dbh_flat),
        'ht': _reshape_all(ht_flat), 'cd': _reshape_all(cd_flat),
        'bark': _reshape_all(bark_flat), 'scorch': _reshape_all(scorch_flat),
    }

    baseline = mort_crnsch(
        fields['spp'][(6,)], fields['dbh'][(6,)], fields['ht'][(6,)], fields['cd'][(6,)],
        bark_thickness=fields['bark'][(6,)], scorch_ht=fields['scorch'][(6,)],
        flame_length=1.0,
    )
    assert baseline.shape == (6,)
    assert np.all(np.isfinite(baseline))

    for shape in _SHAPES[1:]:
        result = mort_crnsch(
            fields['spp'][shape], fields['dbh'][shape], fields['ht'][shape], fields['cd'][shape],
            bark_thickness=fields['bark'][shape], scorch_ht=fields['scorch'][shape],
            flame_length=1.0,
        )
        assert result.shape == (6,), f'shape {shape} did not flatten to (6,)'
        np.testing.assert_array_equal(result, baseline)


def test_run_fofem_emissions_ravels_to_1d_preserving_cell_order():
    """The top-level public orchestrator's own broadcast-to-length-n step
    must flatten every shape variant to ``(6,)`` in C-order, matching
    the same contract as the lower-level ``consm_*``/``mort_*``
    functions it calls internally."""
    baseline = _run_fofem_emissions_with_shape((6,))['LitPre']
    assert baseline.shape == (6,)
    np.testing.assert_array_equal(baseline, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    for shape in _SHAPES[1:]:
        result = _run_fofem_emissions_with_shape(shape)['LitPre']
        assert result.shape == (6,), f'shape {shape} LitPre did not flatten to (6,)'
        np.testing.assert_array_equal(result, baseline)


def test_run_fofem_emissions_scalar_inputs_return_plain_floats():
    """Scalar-array convention at the orchestrator level: all-scalar
    inputs must return plain ``float`` values in the result dict, not
    length-1 arrays."""
    result = run_fofem_emissions(
        litter=1.0, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region='InteriorWest', use_burnup=False, moisture_regime='Dry',
        units='Imperial',
    )
    assert isinstance(result['LitPre'], float)
