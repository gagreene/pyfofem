#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase8_mortality_facade.py - Phase 8 item C: ``run_fofem_mortality``
facade integration coverage.

Before Phase 8, ``run_fofem_mortality`` had ZERO test coverage at any
level (confirmed by a repo-wide grep for the symbol across ``tests/``
before this module existed - only ``docs/CODEBASE.md``, the production
module itself, and its own re-export in ``__init__.py`` mentioned it).
This module covers the facade's real, verified ``(mort_function: str,
params: dict)`` interface directly against the three real dispatched
functions.

**Documentation defect found and pinned, not fixed** (item C explicitly
asks for this reconciliation): ``run_fofem_mortality``'s own docstring
``Examples::`` block shows calling it as
``run_fofem_mortality('crnsch', spp='PIPO', dbh=25.0, ...)`` - i.e. as
if the second positional slot were ``**kwargs``. The REAL signature is
``run_fofem_mortality(mort_function: str, params: dict)`` - exactly two
positional/keyword parameters, no ``**kwargs`` catch-all - so the
documented call form raises
``TypeError: run_fofem_mortality() got an unexpected keyword argument
'spp'`` before ever reaching the dispatch table. Confirmed directly by
execution; see
:func:`test_documented_examples_block_call_form_is_broken_but_the_real_params_dict_form_works`.
Not fixed here (a docstring correction to `src/pyfofem/pyfofem.py` is
out of Phase 8's "no production code" scope) - flagged for separate
authorization, same as every other docstring-inaccuracy gotcha this
project records rather than silently fixes.

**Known bark-thickness defect (F-19/F-20) remains visible through the
facade** - not re-derived here (the direct-call KeyError is already a
strict xfail in ``test_phase4_mortality_parity.py::
test_crnsch_without_bark_thickness_is_dead_on_arrival``); this module
adds exactly one facade-specific proof that ``run_fofem_mortality``
does not mask or change that defect, citing F-19/F-20 rather than
duplicating the finding.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem import mort_bolchar, mort_crcabe, mort_crnsch, run_fofem_mortality


def _bolchar_params(**overrides) -> dict:
    """
    Build a valid ``mort_bolchar`` params dict, with overrides applied.

    :param overrides: Keys to override in the base params dict.
    :return: A params dict suitable for
        ``run_fofem_mortality('bolchar', params)``.
    """
    base = {'spp': 'ACRU', 'dbh': 12.0, 'char_ht': 1.5}
    base.update(overrides)
    return base


def _crcabe_params(**overrides) -> dict:
    """
    Build a valid ``mort_crcabe`` params dict, with overrides applied.

    :param overrides: Keys to override in the base params dict.
    :return: A params dict suitable for
        ``run_fofem_mortality('crcabe', params)``.
    """
    base = {
        'spp': 'PIPO', 'dbh': 25.0, 'ht': 15.0, 'crown_depth': 5.0,
        'ckr': 2.0, 'scorch_ht': 3.0,
    }
    base.update(overrides)
    return base


def _crnsch_params(**overrides) -> dict:
    """
    Build a valid ``mort_crnsch`` params dict (with an explicit
    ``bark_thickness`` to avoid the known F-19/F-20 dead-on-arrival
    default path), with overrides applied.

    :param overrides: Keys to override in the base params dict.
    :return: A params dict suitable for
        ``run_fofem_mortality('crnsch', params)``.
    """
    base = {
        'spp': 'PSME', 'dbh': 25.0, 'ht': 15.0, 'crown_depth': 5.0,
        'bark_thickness': 1.5, 'flame_length': 1.0, 'scorch_ht': 3.0,
    }
    base.update(overrides)
    return base


def test_documented_examples_block_call_form_is_broken_but_the_real_params_dict_form_works():
    """``run_fofem_mortality``'s own docstring ``Examples::`` block calls
    it as ``run_fofem_mortality('crnsch', spp=..., dbh=..., ...)`` - the
    real ``(mort_function, params)`` signature has no ``**kwargs``
    catch-all, so that exact documented form raises ``TypeError``. The
    real, working form (a single ``params`` dict) is proven immediately
    after."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'spp'"):
        run_fofem_mortality(
            'crnsch', spp='PIPO', dbh=25.0, ht=15.0, crown_depth=5.0,
            fire_intensity=500.0,
        )

    result = run_fofem_mortality('crnsch', _crnsch_params())
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_facade_bolchar_mixed_valid_and_unsupported_species_rows():
    """Array-input dispatch through the facade must preserve the same
    mixed-validity contract as calling ``mort_bolchar`` directly - an
    unsupported species yields ``NaN`` for that cell only, without
    disturbing a supported cell's own result."""
    params = _bolchar_params(
        spp=np.array(['UNKSPP', 'ACRU']), dbh=np.array([10.0, 12.0]),
        char_ht=np.array([1.0, 1.5]),
    )
    result = run_fofem_mortality('bolchar', params)
    assert np.isnan(result[0])
    assert np.isfinite(result[1])
    assert result[1] == pytest.approx(run_fofem_mortality('bolchar', _bolchar_params()))


def test_facade_crcabe_crown_damage_override_forwards_verbatim():
    """
    The facade exposes CRCABE's direct C++-style crown-damage input.

    Forwarding a direct ``crown_damage`` value must produce the exact direct
    ``mort_crcabe`` result, rather than silently dropping the public option.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    params = _crcabe_params(crown_damage=30.0)
    facade_value = run_fofem_mortality('crcabe', params)
    direct_value = mort_crcabe(**params)
    assert facade_value == pytest.approx(direct_value, abs=1e-12)


def test_facade_crcabe_mixed_valid_and_unsupported_species_rows():
    """Phase 8 correction pass item 6: the mixed-validity check must
    also cover ``crcabe`` (not only ``bolchar``) - ``mort_crcabe`` has
    the SAME unsupported-species-yields-NaN contract, confirmed by
    direct source inspection of its own ``mask_supported`` check."""
    params = _crcabe_params(
        spp=np.array(['UNKSPP', 'PIPO']), dbh=np.array([10.0, 25.0]),
        ht=np.array([10.0, 15.0]), crown_depth=np.array([3.0, 5.0]),
        ckr=np.array([1.0, 2.0]), scorch_ht=np.array([3.0, 3.0]),
    )
    result = run_fofem_mortality('crcabe', params)
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_facade_crnsch_has_no_unsupported_species_nan_path():
    """Stop-and-report honesty (item 6's "do not claim array support
    where disproven" instruction, extended here to the mixed-validity
    axis): confirmed by direct source inspection that ``mort_crnsch``'s
    species dispatch ends in a catch-all ``mask_other`` branch (FOFEM
    Eq 1, "all other species") rather than a NaN-for-unsupported path -
    EVERY species code, including a genuinely unrecognized one,
    receives a finite probability. There is therefore no
    mixed-valid/unsupported-species case to test for ``crnsch`` the way
    there is for ``bolchar``/``crcabe`` - this test proves that fact
    directly rather than silently omitting the check."""
    params = _crnsch_params(
        spp=np.array(['UNKSPP', 'PSME']), dbh=np.array([25.0, 25.0]),
        ht=np.array([15.0, 15.0]), crown_depth=np.array([5.0, 5.0]),
        bark_thickness=np.array([1.5, 1.5]), flame_length=np.array([1.0, 1.0]),
        scorch_ht=np.array([3.0, 3.0]),
    )
    result = run_fofem_mortality('crnsch', params)
    assert np.all(np.isfinite(result)), (
        'mort_crnsch unexpectedly produced a non-finite result for an '
        'unrecognized species - the Eq 1 catch-all fallback may have changed'
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        'F-19/F-20 (see test_phase4_mortality_parity.py::'
        'test_crnsch_without_bark_thickness_is_dead_on_arrival for the '
        "direct-call proof): mort_crnsch(bark_thickness=None) raises "
        "KeyError('FOFEM_BrkThck_Vsp') via calc_bark_thickness, and the "
        'facade forwards this call unmodified - it does not mask or '
        'change the defect.'
    ),
)
def test_facade_crnsch_without_bark_thickness_is_dead_on_arrival_too():
    """Asserts the DESIRED behaviour (a finite probability) for
    ``run_fofem_mortality('crnsch', ...)`` with ``bark_thickness``
    omitted - proving the SAME F-19/F-20 defect reaches callers through
    the facade, not only through a direct ``mort_crnsch`` call. Genuinely
    executes and genuinely fails under ``--runxfail`` (real ``KeyError``,
    not vacuous)."""
    result = run_fofem_mortality('crnsch', _crnsch_params(bark_thickness=None))
    assert np.isfinite(result)


def test_facade_forwards_every_param_without_silent_drop_or_rename():
    """The facade must forward the FULL params dict verbatim - proven by
    comparing its result against a direct call with the same params for
    all three dispatchers - and must NOT silently accept/drop an
    unrecognized parameter name (an extra bogus key must raise
    ``TypeError``, not be silently ignored)."""
    bolchar_params = _bolchar_params()
    assert run_fofem_mortality('bolchar', bolchar_params) == pytest.approx(
        mort_bolchar(**bolchar_params)
    )

    crcabe_params = _crcabe_params(beetles=True, cvk=10.0)
    assert run_fofem_mortality('crcabe', crcabe_params) == pytest.approx(
        mort_crcabe(**crcabe_params)
    )

    crnsch_params = _crnsch_params()
    assert run_fofem_mortality('crnsch', crnsch_params) == pytest.approx(
        mort_crnsch(**crnsch_params)
    )

    with pytest.raises(TypeError, match='unexpected keyword argument'):
        run_fofem_mortality('bolchar', _bolchar_params(not_a_real_param=1))


@pytest.mark.parametrize('canonical_key,params_builder', [
    ('bolchar', _bolchar_params), ('crcabe', _crcabe_params), ('crnsch', _crnsch_params),
])
def test_facade_keys_are_case_insensitive(canonical_key, params_builder):
    """``mort_function`` dispatch must be case-insensitive and tolerant
    of surrounding whitespace, per the facade's own documented
    ``.strip().lower()`` normalization - parametrized across ALL THREE
    dispatch keys (Phase 8 correction pass item 6: the prior version
    only exercised ``'bolchar'``), each checked against canonical,
    UPPERCASE, MixedCase, and surrounding-whitespace variants."""
    params = params_builder()
    baseline = run_fofem_mortality(canonical_key, params)
    variants = (
        canonical_key.upper(),
        canonical_key[0].upper() + canonical_key[1:],
        f' {canonical_key} ',
        ''.join(c.upper() if i % 2 else c for i, c in enumerate(canonical_key)),
    )
    for variant in variants:
        assert run_fofem_mortality(variant, params) == pytest.approx(baseline), (
            f"key variant '{variant}' did not dispatch identically to '{canonical_key}'"
        )


def test_facade_malformed_argument_type_propagates_a_real_error():
    """A malformed (non-numeric) argument value must propagate a real,
    diagnosable error - not be silently coerced or swallowed."""
    with pytest.raises(ValueError, match='could not convert string to float'):
        run_fofem_mortality('bolchar', _bolchar_params(dbh='not-a-number'))


def test_facade_missing_required_argument_raises_type_error():
    """Omitting a required positional argument from ``params`` must raise
    the same ``TypeError`` the underlying function itself would raise -
    the facade adds no swallowing/defaulting behaviour of its own."""
    with pytest.raises(TypeError, match='missing'):
        run_fofem_mortality('bolchar', {'spp': 'ACRU'})


@pytest.mark.parametrize('key,params_builder,array_overrides', [
    ('bolchar', _bolchar_params, {
        'spp': np.array(['ACRU', 'ACRU', 'ACRU']),
        'dbh': np.array([10.0, 12.0, 14.0]),
        'char_ht': np.array([1.0, 1.5, 2.0]),
    }),
    ('crcabe', _crcabe_params, {
        'spp': np.array(['PIPO', 'PIPO', 'PIPO']),
        'dbh': np.array([15.0, 25.0, 35.0]),
        'ht': np.array([10.0, 15.0, 20.0]),
        'crown_depth': np.array([3.0, 5.0, 7.0]),
        'ckr': np.array([1.0, 2.0, 3.0]),
        'scorch_ht': np.array([8.0, 12.0, 16.0]),
    }),
    ('crnsch', _crnsch_params, {
        'spp': np.array(['PSME', 'PSME', 'PSME']),
        'dbh': np.array([15.0, 25.0, 35.0]),
        'ht': np.array([15.0, 20.0, 25.0]),
        'crown_depth': np.array([5.0, 6.0, 7.0]),
        'bark_thickness': np.array([1.0, 1.5, 2.0]),
        'flame_length': np.array([0.8, 1.0, 1.2]),
        'scorch_ht': np.array([12.0, 18.0, 22.0]),
    }),
])
def test_facade_return_type_shape_and_percell_correspondence_for_array_input(key, params_builder, array_overrides):
    """For all three dispatchers (Phase 8 correction pass item 6: the
    prior version only exercised ``'bolchar'``): scalar params must
    return a plain ``float``; array params of matching length must
    return an ``np.ndarray`` of that same length, with values
    per-cell-identical to calling the underlying function directly
    (genuinely varying fixture values per cell, so a shuffled/wrong-
    order result would fail this)."""
    scalar_result = run_fofem_mortality(key, params_builder())
    assert isinstance(scalar_result, float)

    array_params = params_builder(**array_overrides)
    array_result = run_fofem_mortality(key, array_params)
    assert isinstance(array_result, np.ndarray)
    assert array_result.shape == (3,)
    assert len(set(np.asarray(array_result).tolist())) == 3, (
        f'{key}: fixture values collapsed to <3 distinct results - not a meaningful per-cell check'
    )

    direct_func = {'bolchar': mort_bolchar, 'crcabe': mort_crcabe, 'crnsch': mort_crnsch}[key]
    direct_result = direct_func(**array_params)
    np.testing.assert_array_equal(np.asarray(array_result), np.asarray(direct_result))


def test_facade_supports_all_three_documented_dispatch_keys():
    """Every key in the facade's own documented dispatch table
    (``'bolchar'``, ``'crnsch'``, ``'crcabe'``) must dispatch to its
    real underlying function and return a probability in ``[0, 1]``."""
    assert 0.0 <= run_fofem_mortality('bolchar', _bolchar_params()) <= 1.0
    assert 0.0 <= run_fofem_mortality('crcabe', _crcabe_params()) <= 1.0
    assert 0.0 <= run_fofem_mortality('crnsch', _crnsch_params()) <= 1.0


def test_facade_unknown_key_raises_key_error_naming_valid_options():
    """An unrecognized ``mort_function`` key must raise ``KeyError``
    naming the exact set of valid options, not a generic/undiagnosable
    failure."""
    with pytest.raises(KeyError, match=r"bolchar.*crnsch.*crcabe"):
        run_fofem_mortality('not_a_real_model', {})
