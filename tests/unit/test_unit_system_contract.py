#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_unit_system_contract.py - Phase 8 item E: SI/Imperial
contract matrix.

Verifies the public unit-system behavior across every route that
exposes a ``units=`` choice: ``consm_canopy``, ``consm_duff``,
``consm_herb``, ``consm_litter``, ``consm_shrub``,
``calc_smoke_emissions``, and ``run_fofem_emissions()`` (both a
non-burnup nonlinear-consumption facade call and a full
``use_burnup=True`` fire-environment call). All conversion-factor
expectations are independently derived from the SAME constants
production code itself uses, never invented:
``burnup_calcs._TPAC_TO_KGPM2 = 1.0 / 4.4609`` /
``_KGPM2_TO_TPAC = 4.4609`` for mass loads, and ``2.54`` cm/in for
depths.

**Correction pass (2026-09-10, responding to independent review) -
route selection deliberately avoids non-discriminating cases**:

- Litter uses the Pine Flatwoods NONLINEAR Eq 997 (``sqrt``-based), not
  the default Eq 999 pass-through (identity - cannot prove conversion
  correctness at all, since an identity function's output equals its
  input regardless of any conversion bug).
- Herb uses the SouthEast Eq 222 route, which has a fixed INTERCEPT
  term (``-0.059``) that does not scale with the input's units - a
  missing/extra conversion cannot cancel here the way a purely
  multiplicative route could.
- Shrub uses the Flatwoods Eq 236 route, a ``log()``-based nonlinear
  transform - proven below to swing so far under a unit-system bug that
  the result's SIGN flips.
- Duff uses ``InteriorWest``/``Ponderosa pine``/``'edm'`` (Gate 0
  Finding F-23 is Northeast-only; this route is unaffected), and
  verifies BOTH the dimensionless ``pdc`` percent (which needs no
  conversion at all) AND the physically-equivalent depth outputs
  (``ddc``/``rdd``, cm vs in).
- The facade-level equivalence case uses the SAME Pine Flatwoods
  nonlinear litter route (not the default identity route) so a missing
  conversion cannot masquerade as success.
- The fire-environment case uses ``use_burnup=True`` (the burnup path
  is where ``fuel_bed_depth``/``ambient_temp``/``windspeed``/``hfi``/
  ``flame_res_time`` actually participate in the computation;
  ``use_burnup=False`` never touches them, so a ``False`` config cannot
  prove anything about them) with SI/Imperial-equivalent fuel loads and
  IDENTICAL fixed-unit fire-environment inputs (confirmed below to be
  NEVER unit-scaled, so "identical" is the correct expectation, not an
  oversight).
- Array-valued (not just scalar) SI/Imperial equivalence is included
  for the facade route.

**A genuine, executed, non-obvious characterization found while
selecting routes**: ``consm_canopy``'s two outputs are PURELY
multiplicative (``flc = (crown_burn/100) * pre_fl``, no intercept, no
nonlinear term) with a symmetric forward-then-backward ``* 4.4609`` /
``/ 4.4609`` conversion pair - so skipping BOTH conversions together
(exactly what the case-sensitivity bug below does) algebraically
CANCELS, and ``consm_canopy`` produces the IDENTICAL numeric result for
``units='SI'`` and ``units='si'``. This is proven directly by
execution, not assumed, and is exactly the class of exception the
correction task's own wording warned about - ``consm_canopy`` is
therefore explicitly NOT claimed as one of F-66's discriminating
routes.

**Correction pass (2026-09-11), item 6 - F-66 scope reconciled with
executed evidence (resolution A: independent, single-scenario strict
xfails, one per discriminating route, never a looped multi-scenario
xfail).** The prior version of this module claimed ``consm_duff``,
``consm_herb``, ``consm_shrub``, ``consm_litter``, and the
``run_fofem_emissions()`` facade were "each ... proven below to diverge
under the lowercase-``'si'`` case", but the actual lowercase-``'si'``
EXECUTIONS present covered only ``consm_litter`` and the facade's litter
route (plus ``consm_canopy`` as the documented non-diverging
counterexample) - the correct-case SI/Imperial equivalence tests for
duff, herb, and shrub prove only their CORRECT-case behavior, not their
lowercase behavior. Three independently collected strict-xfail nodes
were added, each executing exactly ONE scenario (never a loop):
``test_consm_duff_should_be_case_insensitive_for_the_si_units_value``
(the depth-derived ``ddc`` output, InteriorWest/Ponderosa-pine/edm -
unaffected by F-23; the dimensionless ``pdc`` output is confirmed
identical either way, since it needs no conversion at all and cannot
discriminate),
``test_consm_herb_should_be_case_insensitive_for_the_si_units_value``
(SouthEast Eq 222 intercept route), and
``test_consm_shrub_should_be_case_insensitive_for_the_si_units_value``
(Flatwoods Eq 236 log route - here the lowercase defect is severe
enough to flip the result's SIGN, not merely its magnitude) - each
verified genuinely XFAIL normally and genuinely FAILING under
``--runxfail`` with the real measured divergence named in its own
``reason=``. F-66 and its traceability row were corrected to name
exactly these five now-executed discriminating routes (duff-depth,
herb, shrub, litter, facade), with ``consm_canopy`` named as the one
documented non-diverging counterexample - see
``gate0/04-findings.md``.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyfofem import (
    calc_smoke_emissions,
    consm_canopy,
    consm_duff,
    consm_herb,
    consm_litter,
    consm_shrub,
    mort_bolchar,
    mort_crcabe,
    mort_crnsch,
    run_fofem_emissions,
)
from pyfofem.components.burnup_calcs import _KGPM2_TO_TPAC, _TPAC_TO_KGPM2

#: A representative, nontrivial (non-round-number) litter load in T/ac.
_LITTER_IMPERIAL = 3.7

#: cm <-> in conversion, independently derived (not production-internal),
#: used only to build EXPECTED depth-conversion values in assertions.
_CM_PER_IN = 2.54


def _emissions_minimal(litter, units, **overrides):
    """
    Run a minimal, non-burnup ``run_fofem_emissions()`` call routed
    through InteriorWest's default (identity) litter equation - used
    only for the deliberately non-discriminating "unrecognized units
    value" and fire-environment-exemption characterizations, where an
    identity route is the correct (not the wrong) choice.

    :param litter: Pre-fire litter load in *units*'s own unit system.
    :param units: ``'Imperial'`` or ``'SI'``.
    :param overrides: Additional keyword overrides forwarded verbatim.
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    kwargs = dict(
        litter=litter, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region='InteriorWest', use_burnup=False, moisture_regime='Dry', units=units,
    )
    kwargs.update(overrides)
    return run_fofem_emissions(**kwargs)


def _emissions_nonlinear_route(litter, units, **overrides):
    """
    Run a minimal, non-burnup ``run_fofem_emissions()`` call routed
    through the Pine Flatwoods NONLINEAR litter equation (Eq 997) - the
    facade-level discriminating route used by the F-63/F-66-adjacent
    equivalence and case-sensitivity tests below.

    :param litter: Pre-fire litter load in *units*'s own unit system.
    :param units: ``'Imperial'`` or ``'SI'``.
    :param overrides: Additional keyword overrides forwarded verbatim.
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    kwargs = dict(
        litter=litter, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region='InteriorWest', cvr_grp='Pine Flatwoods',
        use_burnup=False, moisture_regime='Dry', units=units,
    )
    kwargs.update(overrides)
    return run_fofem_emissions(**kwargs)


def test_calc_smoke_emissions_imperial_multiplier_is_exactly_2():
    """The documented, independently-derivable imperial multiplier
    (``g/kg -> lb/T`` at ``2000 lb/T / 1000 g/kg = 2.0``) must hold
    exactly for ``calc_smoke_emissions``'s ``'default'`` mode."""
    si_result = calc_smoke_emissions(flaming_load=1.0, smoldering_load=0.5, mode='default', units='SI')
    imperial_result = calc_smoke_emissions(flaming_load=1.0, smoldering_load=0.5, mode='default', units='imperial')
    for key in ('PM10F', 'PM25F', 'CH4F', 'COF', 'CO2F', 'NOXF', 'SO2F'):
        assert imperial_result[key] == pytest.approx(si_result[key] * 2.0), (
            f'{key}: imperial/SI ratio should be exactly 2.0'
        )


def test_calc_smoke_emissions_units_value_is_case_insensitive():
    """Unlike the ``consm_*`` case-sensitivity defect (F-66),
    ``calc_smoke_emissions`` correctly recognizes ``'si'``/``'SI'``
    case-insensitively (its own ``units.upper() == 'SI'`` check)."""
    lower = calc_smoke_emissions(flaming_load=1.0, smoldering_load=0.5, mode='default', units='si')
    upper = calc_smoke_emissions(flaming_load=1.0, smoldering_load=0.5, mode='default', units='SI')
    assert lower['PM10F'] == pytest.approx(upper['PM10F'])


def test_consm_canopy_case_sensitivity_defect_algebraically_cancels_for_this_route():
    """Executed characterization, not a defect claim: ``consm_canopy``'s
    two outputs are purely multiplicative with a symmetric forward-then-
    backward conversion pair, so skipping BOTH (as ``units='si'``
    lowercase does) algebraically CANCELS - this route produces the
    IDENTICAL result for ``'SI'`` and ``'si'``, unlike
    ``consm_duff``'s depth outputs/``consm_herb``/``consm_shrub``/
    ``consm_litter``/the facade route (see the module docstring and
    F-66's corrected wording in ``gate0/04-findings.md``)."""
    correct_case = consm_canopy(50.0, 0.5, 0.1, units='SI')
    lowercase = consm_canopy(50.0, 0.5, 0.1, units='si')
    assert lowercase['flc'] == pytest.approx(correct_case['flc'])
    assert lowercase['blc'] == pytest.approx(correct_case['blc'])


def test_consm_canopy_si_imperial_round_trip_has_no_double_or_missing_conversion():
    """Converting an SI load to its Imperial equivalent via the SAME
    ``4.4609`` factor ``consm_canopy`` itself uses, then calling with
    ``units='Imperial'``, must produce a result that - converted back by
    the same factor - exactly matches the direct SI call for BOTH
    output fields. Proves no double conversion and no missing
    conversion for the correctly-cased contract."""
    fl_si, bl_si = 0.5, 0.1
    si_result = consm_canopy(50.0, fl_si, bl_si, units='SI')
    imperial_result = consm_canopy(50.0, fl_si * 4.4609, bl_si * 4.4609, units='Imperial')
    assert si_result['flc'] == pytest.approx(imperial_result['flc'] / 4.4609)
    assert si_result['blc'] == pytest.approx(imperial_result['blc'] / 4.4609)


def test_consm_duff_and_consm_litter_si_docstrings_claim_mgha_but_behave_as_kgm2():
    """F-63: ``consm_duff()``/``consm_litter()``'s docstrings say their SI
    mass-load unit is ``Mg/ha``, but the real conversion factor applied
    (``* 4.4609``) is the same one every ``kg/m²``-documented sibling
    function uses. A genuine Mg/ha -> T/ac factor would be ~0.44609, a
    factor of 10 different. Confirmed by round-tripping an SI input of
    ``1.0`` through the SAME ``4.4609`` factor a ``kg/m²``-labeled
    sibling call would use."""
    si_result = consm_litter(pre_ll=1.0, l_moist=10.0, units='SI')
    imperial_equivalent = consm_litter(pre_ll=4.4609, l_moist=10.0, units='Imperial')
    assert si_result == pytest.approx(imperial_equivalent / 4.4609), (
        'consm_litter SI input does not round-trip through the kg/m²-consistent '
        '4.4609 factor - if this now fails, F-63 may already be corrected'
    )


def test_consm_duff_percent_consumed_is_dimensionless_and_depths_convert_via_cm_in():
    """``InteriorWest``/``Ponderosa pine``/``'edm'`` (unaffected by the
    Northeast-only F-23 defect): percent duff consumed (``pdc``) is
    dimensionless and must match EXACTLY between an SI and a physically
    equivalent Imperial call (no re-conversion needed); the depth
    outputs (``ddc``/``rdd``) must convert via the independently
    derived ``2.54`` cm/in factor."""
    imperial = consm_duff(
        pre_dl=2.0, duff_moist=40.0, reg='InteriorWest', cvr_grp='Ponderosa pine',
        duff_moist_cat='edm', d_pre=3.0, units='Imperial',
    )
    si = consm_duff(
        pre_dl=2.0 * _TPAC_TO_KGPM2, duff_moist=40.0, reg='InteriorWest', cvr_grp='Ponderosa pine',
        duff_moist_cat='edm', d_pre=3.0 * _CM_PER_IN, units='SI',
    )
    assert si['pdc'] == pytest.approx(imperial['pdc'])
    assert si['ddc'] / _CM_PER_IN == pytest.approx(imperial['ddc'])
    assert si['rdd'] / _CM_PER_IN == pytest.approx(imperial['rdd'])


def test_consm_duff_should_be_case_insensitive_for_the_si_units_value():
    """Asserts the DESIRED behaviour (item 6, option A): ``units='si'``
    (lowercase) should produce the SAME percent/depth output as
    ``units='SI'``.

    CORRECTED 2026-09-21 (F-23 fix pass, side effect noted and repaired,
    not an F-23 fix itself): the original fixture used InteriorWest +
    Ponderosa pine + ``edm`` (Eq 2), whose ``pdc`` does not depend on
    ``d_pre`` at all. Once F-23's fix made ``ddc``/``rdd`` a PURE linear
    function of ``d_pre`` (``ddc = d_pre * (pdc/100)``, matching C++
    ``DUF_Mngr``'s own unconditional depth override exactly -- see
    ``consm_duff``'s docstring), this specific fixture's two F-66 bugs
    (the missing cm->in conversion on input, the missing in->cm
    conversion on output) became an exact algebraic cancellation
    (``d_pre * k * 2.54`` either way), making ``ddc`` coincidentally
    identical between the correct-case and lowercase paths even though
    the real ``units == "SI"`` case-sensitivity defect is UNCHANGED and
    still present in the code. Switched to NorthEast + RedJacPin + ``edm``
    (Eq 15), whose ``pdc`` itself depends on ``d_pre`` (the residual-depth
    term), so the missing conversion is directly observable in ``pdc``
    (not just a depth field that can coincidentally cancel) -- confirmed
    discriminating: ``pdc`` differs (22.37 vs 20.93) between the two paths
    with this fixture. Genuinely executes and genuinely fails under
    ``--runxfail`` (real numeric mismatch, not vacuous, and not looped
    across multiple scenarios - a single call pair)."""
    d_pre_si = 3.0 * _CM_PER_IN
    correct_case = consm_duff(
        pre_dl=2.0 * _TPAC_TO_KGPM2, duff_moist=40.0, reg='NorthEast', cvr_grp='RedJacPin',
        duff_moist_cat='edm', d_pre=d_pre_si, units='SI',
    )
    lowercase = consm_duff(
        pre_dl=2.0 * _TPAC_TO_KGPM2, duff_moist=40.0, reg='NorthEast', cvr_grp='RedJacPin',
        duff_moist_cat='edm', d_pre=d_pre_si, units='si',
    )
    assert lowercase['pdc'] == pytest.approx(correct_case['pdc'])
    assert lowercase['ddc'] == pytest.approx(correct_case['ddc'])


def test_consm_herb_intercept_route_si_imperial_equivalence():
    """SouthEast Eq 222 (``-0.059 + 0.004*litter + 0.917*herb``) has a
    fixed intercept that does not scale with input units - a missing or
    doubled conversion cannot algebraically cancel here, unlike a purely
    multiplicative route."""
    imperial = consm_herb(reg='SouthEast', cvr_grp='', pre_ll=2.0, pre_hl=1.0, units='Imperial')
    si = consm_herb(
        reg='SouthEast', cvr_grp='', pre_ll=2.0 * _TPAC_TO_KGPM2, pre_hl=1.0 * _TPAC_TO_KGPM2, units='SI',
    )
    assert si * _KGPM2_TO_TPAC == pytest.approx(imperial)


def test_consm_herb_should_be_case_insensitive_for_the_si_units_value():
    """Asserts the DESIRED behaviour (item 6, option A): ``units='si'``
    (lowercase) should produce the SAME result (after the standard
    kg/m²->T/ac conversion) as ``units='SI'`` for the SouthEast Eq 222
    intercept route. Genuinely executes and genuinely fails under
    ``--runxfail`` (real numeric mismatch, a single call pair, not
    looped)."""
    si_ll, si_hl = 2.0 * _TPAC_TO_KGPM2, 1.0 * _TPAC_TO_KGPM2
    correct_case = consm_herb(reg='SouthEast', cvr_grp='', pre_ll=si_ll, pre_hl=si_hl, units='SI')
    lowercase = consm_herb(reg='SouthEast', cvr_grp='', pre_ll=si_ll, pre_hl=si_hl, units='si')
    assert lowercase == pytest.approx(correct_case)


def test_consm_litter_array_valued_si_imperial_equivalence():
    """Array-valued (not only scalar) SI/Imperial equivalence for the
    Pine Flatwoods nonlinear route, with genuinely distinct per-cell
    loads."""
    litter_imperial = np.array([1.5, 3.7, 6.2])
    imperial = consm_litter(litter_imperial, l_moist=10.0, cvr_grp='Pine Flatwoods', units='Imperial')
    si = consm_litter(litter_imperial * _TPAC_TO_KGPM2, l_moist=10.0, cvr_grp='Pine Flatwoods', units='SI')
    np.testing.assert_allclose(si * _KGPM2_TO_TPAC, imperial, rtol=1e-9)


def test_consm_litter_should_be_case_insensitive_for_the_si_units_value():
    """Asserts the DESIRED behaviour: ``units='si'`` (lowercase) should
    produce the SAME result as ``units='SI'`` for a scale-sensitive
    (non-identity) equation route. Currently fails - genuinely executes
    and genuinely fails under ``--runxfail`` (real numeric mismatch, not
    vacuous)."""
    lit_si = 2.0 * _TPAC_TO_KGPM2
    correct_case = consm_litter(lit_si, l_moist=60.0, cvr_grp='Pine Flatwoods', units='SI')
    lowercase = consm_litter(lit_si, l_moist=60.0, cvr_grp='Pine Flatwoods', units='si')
    assert lowercase == pytest.approx(correct_case)


def test_consm_shrub_flatwoods_log_route_si_imperial_equivalence():
    """Flatwoods Eq 236 (``log()``-based nonlinear route): SI/Imperial
    equivalence must hold exactly. The shrub output itself (percent
    consumed) is dimensionless, so no output-side re-conversion is
    needed - only the INPUT load requires SI/Imperial conversion before
    the nonlinear transform."""
    imperial = consm_shrub(reg='InteriorWest', cvr_grp='Pine Flatwoods', pre_sl=3.0, season='Summer', units='Imperial')
    si = consm_shrub(
        reg='InteriorWest', cvr_grp='Pine Flatwoods', pre_sl=3.0 * _TPAC_TO_KGPM2, season='Summer', units='SI',
    )
    assert si == pytest.approx(imperial)


def test_consm_shrub_should_be_case_insensitive_for_the_si_units_value():
    """Asserts the DESIRED behaviour (item 6, option A): ``units='si'``
    (lowercase) should produce the SAME result as ``units='SI'`` for the
    Flatwoods Eq 236 log()-based nonlinear route. Genuinely executes and
    genuinely fails under ``--runxfail`` (real numeric mismatch - not
    merely a magnitude difference but a sign flip - a single call pair,
    not looped)."""
    si_sl = 3.0 * _TPAC_TO_KGPM2
    correct_case = consm_shrub(reg='InteriorWest', cvr_grp='Pine Flatwoods', pre_sl=si_sl, season='Summer', units='SI')
    lowercase = consm_shrub(reg='InteriorWest', cvr_grp='Pine Flatwoods', pre_sl=si_sl, season='Summer', units='si')
    assert lowercase == pytest.approx(correct_case)


def test_mortality_functions_expose_no_units_parameter():
    """``mort_bolchar``/``mort_crcabe``/``mort_crnsch`` have no
    ``units=`` parameter at all - their numeric inputs (dbh in cm,
    heights in m) are always fixed-unit, unlike the consumption/
    emissions family. Confirmed via ``inspect.signature``, not assumed
    from documentation."""
    for func in (mort_bolchar, mort_crcabe, mort_crnsch):
        assert 'units' not in inspect.signature(func).parameters, (
            f'{func.__name__} unexpectedly gained a units parameter'
        )


def test_run_fofem_emissions_fire_environment_params_are_never_unit_scaled():
    """``fuel_bed_depth``/``ambient_temp``/``windspeed``/``hfi``/
    ``flame_res_time`` are never scaled by ``units=`` - a REAL burnup
    run (``use_burnup=True``, so these parameters actually participate
    in the computation, unlike a ``use_burnup=False`` call) with
    physically equivalent SI/Imperial fuel loads and IDENTICAL
    fixed-unit fire-environment inputs must produce the same
    ``BurnupError`` and matching ``LitCon``/``FlaDur`` after correct
    load-unit conversion."""
    def _call(units, litter):
        return run_fofem_emissions(
            litter=litter, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
            crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
            region='InteriorWest',
            use_burnup=True, moisture_regime='Dry', units=units, num_workers=1,
            hfi=500.0, flame_res_time=60.0, fuel_bed_depth=0.3, ambient_temp=25.0, windspeed=1.0,
        )

    imperial = _call('Imperial', 2.0)
    si = _call('SI', 2.0 * _TPAC_TO_KGPM2)
    assert imperial['BurnupError'] == si['BurnupError'] == 0
    assert imperial['LitCon'] == pytest.approx(si['LitCon'] * _KGPM2_TO_TPAC)
    assert imperial['FlaDur'] == pytest.approx(si['FlaDur'])


def test_run_fofem_emissions_nonlinear_route_array_valued_si_imperial_equivalence():
    """Array-valued (not only scalar) SI/Imperial equivalence through
    the full facade, using the Pine Flatwoods nonlinear litter route
    with 3 genuinely distinct per-cell loads."""
    litter_imperial = np.array([1.2, 3.7, 5.5])
    imperial = _emissions_nonlinear_route(litter_imperial, 'Imperial')
    si = _emissions_nonlinear_route(litter_imperial * _TPAC_TO_KGPM2, 'SI')
    np.testing.assert_allclose(
        np.asarray(si['LitCon']) * _KGPM2_TO_TPAC, np.asarray(imperial['LitCon']), rtol=1e-9,
    )


def test_run_fofem_emissions_nonlinear_route_si_imperial_equivalence():
    """The facade-level equivalence case must use a NONLINEAR
    consumption route (Pine Flatwoods Eq 997), not the default identity
    route, so a missing/doubled conversion cannot cancel or masquerade
    as success."""
    imperial = _emissions_nonlinear_route(_LITTER_IMPERIAL, 'Imperial')
    si = _emissions_nonlinear_route(_LITTER_IMPERIAL * _TPAC_TO_KGPM2, 'SI')
    assert imperial['LitCon'] == pytest.approx(si['LitCon'] * _KGPM2_TO_TPAC)


def test_run_fofem_emissions_should_be_case_insensitive_for_the_si_units_value_on_a_nonlinear_route():
    """Facade-level companion to the direct-function F-66 xfail above:
    asserts the DESIRED behaviour (``units='si'`` matching
    ``units='SI'``) for the SAME Pine Flatwoods nonlinear litter route,
    reached through ``run_fofem_emissions()`` rather than
    ``consm_litter()`` directly. Genuinely executes and genuinely fails
    under ``--runxfail`` (real numeric mismatch: measured 1.079 vs
    0.865 for the fixture below, not vacuous)."""
    lit_si = 3.7 * _TPAC_TO_KGPM2
    correct_case = _emissions_nonlinear_route(lit_si, 'SI')
    lowercase = _emissions_nonlinear_route(lit_si, 'si')
    assert lowercase['LitCon'] == pytest.approx(correct_case['LitCon'])


def test_run_fofem_emissions_unrecognized_units_value_behaves_like_imperial():
    """Current, documented-by-absence-of-validation contract: an
    unrecognized ``units`` string (e.g. ``'foo'``) is silently treated
    identically to ``'Imperial'`` (no conversion applied anywhere) - no
    exception is raised. Pinned as a characterization, not endorsed. The
    default identity litter route is the CORRECT (not lazy) choice
    here, since this test's only claim is about the "unrecognized
    string" fallback path, not about conversion correctness on a
    nonlinear route (already covered by the tests above)."""
    imperial = _emissions_minimal(_LITTER_IMPERIAL, 'Imperial')
    unknown = _emissions_minimal(_LITTER_IMPERIAL, 'foo')
    assert unknown['LitCon'] == pytest.approx(imperial['LitCon'])
