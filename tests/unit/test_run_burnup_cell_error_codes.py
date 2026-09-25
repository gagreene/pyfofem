#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_run_burnup_cell_error_codes.py - Phase 7 item A: executable coverage
for every ``burnup_error`` code ``_run_burnup_cell()``
(``src/pyfofem/components/burnup_calcs.py``) can actually return: **10-16,
20-29, 90, 91, 99** (the production error table this module documents in
its own docstring and ``README.md`` - NOT the separate, non-production
``development/burnup_array/burnup_array_calcs.py`` prototype, whose own
``burnup_error == 90`` test in
``development/burnup_array/tests/test_consumption_calcs_array.py`` does
not exercise this function at all and must not be mistaken for coverage
of it).

Every test below drives the REAL, unmodified ``_run_burnup_cell()``
through a genuine ``ckw`` dict and/or the real underlying ``burnup()``
simulation - never a mock of the error-translation logic itself. Three
kinds of "dependency" are patched, and only where the production code
provides no other seam - the validation (``_check_fuel()``) and
exception-translation (``_run_burnup_cell()``'s own ``except
BurnupValidationError`` block) logic that actually produces every code
below runs for real and unmodified in every test:

- Codes 22/25/28/29 (``htval``/``sigma``/``tpig``/``tchar`` out of
  ``_FUEL_BOUNDS``): ``_run_burnup_cell()`` builds every ``FuelParticle``
  from a small set of HARDCODED values (``_HTVAL``, ``_SAV_DEFAULTS``,
  ``_SOUND_TPIG``, ``_TCHAR``) that a caller cannot reach through ``ckw``
  at all. Monkeypatching the named module-level constant the function
  itself reads drives these branches without editing production code.
- Codes 21/26/27 (``ash``/``cheat``/``condry`` out of ``_FUEL_BOUNDS``):
  these three ``FuelParticle`` fields are ALSO hardcoded inline literals
  (``ash=0.05``, ``cheat=2750.0``, ``condry=0.133``) with no
  caller-facing parameter - but ``_check_fuel()`` (``burnup.py``) reads
  the bound for each attribute from the mutable MODULE-LEVEL
  ``_FUEL_BOUNDS`` dict on every call, not a value frozen at import time.
  Monkeypatching only the applicable ``_FUEL_BOUNDS`` entry so the
  UNCHANGED inline value falls outside it drives the real
  ``_check_fuel()`` -> ``BurnupValidationError`` -> ``_run_burnup_cell()``
  translation path, exactly as codes 22/25/28/29 do via a different
  hardcoded name. (An earlier pass wrongly claimed these three codes were
  structurally unreachable and could only be closed by editing production
  code - directly disproved by executing this exact patch: see
  ``gate0/04-findings.md``'s F-59, retracted.)
- Every other code (10-16, 20, 23, 24, 90, 91, 99) is reached purely
  through ``ckw`` field values - no monkeypatching at all.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, Optional

import pytest

import pyfofem.components.burnup  # noqa: F401  (ensures sys.modules entry exists)
# pyfofem.components.__init__ rebinds the `burnup` ATTRIBUTE on the
# `pyfofem.components` package to the re-exported `burnup()` FUNCTION (its
# own two-hop re-export), which shadows `from pyfofem.components import
# burnup as x` (an attribute lookup under the hood) - sys.modules is
# unaffected by that rebind, so it is the only reliable way to reach the
# real module object whose mutable _FUEL_BOUNDS dict codes 21/26/27 patch.
_burnup_module = sys.modules['pyfofem.components.burnup']
from pyfofem.components import burnup_calcs as bc
from pyfofem.components.burnup import _BURNUP_LIMIT_ERROR

#: Every code this module claims real, executable coverage for. Cross-
#: checked against the full registered ``_BURNUP_LIMIT_ERROR`` table and
#: the collected test parametrization by the completeness meta-tests
#: below, so a duplicated/missing/newly-introduced production error code
#: cannot silently escape coverage.
_COVERED_CODES = (
    10, 11, 12, 13, 14, 15, 16,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    90, 91, 99,
)


def _assert_error_result(result: Dict[str, Any], expected_code: int) -> None:
    """
    Assert *result* is a well-formed FAILURE result: exactly the
    documented ``{'burnup_limit_adjust': int, 'burnup_error': <code>}``
    shape/state - no per-cell consumption keys leak through on failure
    (``_run_burnup_cell()``'s own docstring: "On failure, a dict with
    only 'burnup_limit_adjust' and a nonzero 'burnup_error' code").

    :param result: Return value of :func:`_run_burnup_cell`.
    :param expected_code: The exact ``burnup_error`` code expected.
    """
    assert set(result) == {'burnup_limit_adjust', 'burnup_error'}
    assert result['burnup_error'] == expected_code
    assert isinstance(result['burnup_limit_adjust'], int)


def _base_ckw(**overrides: Any) -> Dict[str, Any]:
    """
    Build a minimal, otherwise-valid ``ckw`` dict for ``_run_burnup_cell()``
    - a single ``litter`` particle with nominal fire/fuel inputs entirely
    within every documented bound, so exactly one override at a time
    drives exactly one error/success branch.

    :param overrides: Top-level ``ckw`` key overrides (values replace the
        defaults below; a nested ``bkw`` override REPLACES the whole
        ``bkw`` sub-dict, it is not merged).
    :return: A complete ``ckw`` dict suitable for
        :func:`~pyfofem.components.burnup_calcs._run_burnup_cell`.
    """
    ckw: Dict[str, Any] = dict(
        fuel_loadings_bu={'litter': 1.0},
        fuel_moistures_bu={'litter': 0.10},
        rotten_keys={},
        density_map={},
        intensity_kw=100.0,
        frt_s=60.0,
        ws=1.0,
        fb_depth=0.3,
        amb_temp=20.0,
        duf_loading_si=0.0,
        duf_moist_frac=0.5,
        duf_pct_consumed=-1.0,
        hsf_consumed_si=0.0,
        brafol_consumed_si=0.0,
        burnup_dt=15.0,
        bkw=dict(r0=1.83, dr=0.4, max_times=3000, fint_switch=15.0, validate=True),
    )
    ckw.update(overrides)
    return ckw


def test_code_10_fistart_below_minimum_is_rejected_not_clipped():
    """Code 10: igniting fire intensity below ``_FIRE_BOUNDS['fistart']``'s
    minimum (40 kW/m²) is REJECTED outright (unlike the upper bound,
    which is clipped, see adjust-code 1) - direct pre-check at
    ``burnup_calcs.py``'s own ``if intensity < _fi_lo2:`` line, reached
    before ``burnup()`` is ever called."""
    result = bc._run_burnup_cell(_base_ckw(intensity_kw=39.9))
    _assert_error_result(result, 10)


def test_code_11_ti_below_minimum_is_rejected():
    """Code 11: surface fire residence time below the minimum (10 s)."""
    result = bc._run_burnup_cell(_base_ckw(frt_s=9.9))
    _assert_error_result(result, 11)


def test_code_12_windspeed_below_minimum_is_rejected():
    """Code 12: windspeed below the minimum (0 m/s) - exercised with a
    negative value since 0 itself is the documented (valid) boundary."""
    result = bc._run_burnup_cell(_base_ckw(ws=-1.0))
    _assert_error_result(result, 12)


def test_code_13_ambient_temperature_below_minimum_is_rejected():
    """Code 13: ambient temperature below the minimum (-40 degC)."""
    result = bc._run_burnup_cell(_base_ckw(amb_temp=-40.1))
    _assert_error_result(result, 13)


def test_code_14_duff_moisture_above_maximum_is_rejected_only_when_duff_present():
    """Code 14: duff moisture above the maximum (1.972 fraction) is
    rejected ONLY when duff is actually present (``duf_loading_si > 0``) -
    the same out-of-range moisture with zero duff loading must NOT reject,
    since duff moisture is meaningless without duff."""
    rejected = bc._run_burnup_cell(
        _base_ckw(duf_loading_si=5.0, duf_moist_frac=2.5)
    )
    _assert_error_result(rejected, 14)

    accepted = bc._run_burnup_cell(
        _base_ckw(duf_loading_si=0.0, duf_moist_frac=2.5)
    )
    assert accepted['burnup_error'] == 0


def test_code_15_igniting_fire_cannot_dry_fuel():
    """Code 15: a real ``BurnupValidationError("Igniting fire cannot dry
    fuel")`` raised by ``burnup()`` itself, translated via the
    ``'cannot dry fuel'`` substring match - reached with documented-
    bounds-compliant inputs (minimum fire intensity, minimum ambient
    temperature) plus a genuinely small mixing parameter ``r0``, which
    keeps the estimated fire-environment temperature below the drying
    threshold (``_TPDRY + 10`` K) - not an out-of-bounds input."""
    result = bc._run_burnup_cell(_base_ckw(
        intensity_kw=40.0, frt_s=200.0, ws=0.0, fb_depth=0.1, amb_temp=-40.0,
        bkw=dict(r0=0.05, dr=0.4, max_times=3000, fint_switch=15.0, validate=True),
    ))
    _assert_error_result(result, 15)


def test_code_16_no_fuel_ignited():
    """Code 16: a real ``BurnupValidationError("No fuel ignited")`` -
    reached with the minimum documented-bounds-compliant residence time
    (10 s) and a coarse, slow-heating fuel class (``dwk_20``, the lowest
    SAV in :data:`~pyfofem.components.burnup_calcs._SAV_DEFAULTS`), which
    genuinely cannot finish drying and igniting within that residence
    time under otherwise-minimal fire conditions."""
    result = bc._run_burnup_cell(_base_ckw(
        fuel_loadings_bu={'dwk_20': 5.0}, fuel_moistures_bu={'dwk_20': 0.10},
        intensity_kw=40.0, frt_s=10.0, ws=0.0, fb_depth=0.1, amb_temp=-40.0,
    ))
    _assert_error_result(result, 16)


def test_code_20_dry_loading_out_of_fuel_bounds():
    """Code 20: ``wdry`` (dry loading) above ``_FUEL_BOUNDS``'s maximum
    (1e6 kg/m²) - reached via the real, caller-supplied
    ``fuel_loadings_bu`` value, a genuine ``_check_fuel()`` validation
    failure inside ``burnup()``, translated via the ``'dry loading'``
    fragment match."""
    result = bc._run_burnup_cell(_base_ckw(fuel_loadings_bu={'litter': 2.0e6}))
    _assert_error_result(result, 20)


def test_code_21_ash_content_out_of_fuel_bounds(monkeypatch):
    """Code 21: ``ash`` (ash content) out of ``_FUEL_BOUNDS`` -
    ``_run_burnup_cell()`` hardcodes every particle's ``ash`` to the
    inline literal ``0.05``, but ``_check_fuel()`` (``burnup.py``) reads
    the ``'ash'`` bound tuple from the mutable module-level
    ``_FUEL_BOUNDS`` dict on every call. Patching ONLY that bound entry
    (never ``_run_burnup_cell``, ``_check_fuel``, the raised exception,
    or the translation table) so the unchanged ``0.05`` literal falls
    outside it drives the real ``_check_fuel()`` ->
    ``BurnupValidationError`` -> ``'ash content'`` fragment match. The
    nominal fixture is proven to succeed both before AND after the
    patch (the latter via restoring the original bound), so the failure
    is attributable to the patch alone."""
    baseline = bc._run_burnup_cell(_base_ckw())
    assert baseline['burnup_error'] == 0

    original = _burnup_module._FUEL_BOUNDS['ash']
    monkeypatch.setitem(
        _burnup_module._FUEL_BOUNDS, 'ash', (0.06, 0.1, 'ash content (fraction)'),
    )
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 21)
    monkeypatch.undo()

    restored = bc._run_burnup_cell(_base_ckw())
    assert restored['burnup_error'] == 0
    assert _burnup_module._FUEL_BOUNDS['ash'] == original


def test_code_22_heat_content_out_of_fuel_bounds(monkeypatch):
    """Code 22: ``htval`` (heat content) out of ``_FUEL_BOUNDS`` -
    ``_run_burnup_cell()`` hardcodes every particle's ``htval`` to the
    module-level :data:`~pyfofem.components.burnup_calcs._HTVAL`
    constant, so that constant is monkeypatched out of range (never the
    validation/translation logic itself) to drive this branch through the
    real ``_check_fuel()`` -> ``BurnupValidationError`` ->
    ``'heat content'`` fragment match."""
    monkeypatch.setattr(bc, '_HTVAL', 5.0e7)  # _FUEL_BOUNDS['htval'] is (1e7, 3e7)
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 22)


def test_code_23_fuel_moisture_out_of_fuel_bounds():
    """Code 23: ``fmois`` (fuel moisture fraction) above the maximum
    (3.0) - reached via the real, caller-supplied ``fuel_moistures_bu``
    value."""
    result = bc._run_burnup_cell(_base_ckw(fuel_moistures_bu={'litter': 5.0}))
    _assert_error_result(result, 23)


def test_code_24_dry_mass_density_out_of_fuel_bounds():
    """Code 24: ``dendry`` (dry mass density) below the minimum
    (200 kg/m³) - reached via the real, caller-supplied ``density_map``
    override."""
    result = bc._run_burnup_cell(_base_ckw(density_map={'litter': 50.0}))
    _assert_error_result(result, 24)


def test_code_25_sav_out_of_fuel_bounds(monkeypatch):
    """Code 25: ``sigma`` (surface-area-to-volume ratio) out of
    ``_FUEL_BOUNDS`` - ``_run_burnup_cell()`` hardcodes every particle's
    SAV from :data:`~pyfofem.components.burnup_calcs._SAV_DEFAULTS`
    (keyed by fuel-class name, not caller-overridable), so that table
    entry is monkeypatched out of range to drive this branch through the
    real ``_check_fuel()`` -> ``BurnupValidationError`` -> ``'sav'``
    fragment match."""
    monkeypatch.setitem(bc._SAV_DEFAULTS, 'litter', 2.0)  # bound is (4.0, 1e4)
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 25)


def test_code_26_heat_capacity_out_of_fuel_bounds(monkeypatch):
    """Code 26: ``cheat`` (heat capacity) out of ``_FUEL_BOUNDS`` - the
    same patch-the-bound-not-the-value mechanism as code 21 above,
    applied to the ``'cheat'`` entry against the inline ``cheat=2750.0``
    literal, driving the real ``_check_fuel()`` ->
    ``BurnupValidationError`` -> ``'heat capacity'`` fragment match.
    Nominal success proven before and after the patch."""
    baseline = bc._run_burnup_cell(_base_ckw())
    assert baseline['burnup_error'] == 0

    original = _burnup_module._FUEL_BOUNDS['cheat']
    monkeypatch.setitem(
        _burnup_module._FUEL_BOUNDS, 'cheat',
        (1000.0, 2000.0, 'heat capacity (J/kg*K)'),
    )
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 26)
    monkeypatch.undo()

    restored = bc._run_burnup_cell(_base_ckw())
    assert restored['burnup_error'] == 0
    assert _burnup_module._FUEL_BOUNDS['cheat'] == original


def test_code_27_thermal_conductivity_out_of_fuel_bounds(monkeypatch):
    """Code 27: ``condry`` (thermal conductivity) out of ``_FUEL_BOUNDS``
    - the same patch-the-bound-not-the-value mechanism as codes 21/26
    above, applied to the ``'condry'`` entry against the inline
    ``condry=0.133`` literal, driving the real ``_check_fuel()`` ->
    ``BurnupValidationError`` -> ``'thermal conductivity'`` fragment
    match. Nominal success proven before and after the patch."""
    baseline = bc._run_burnup_cell(_base_ckw())
    assert baseline['burnup_error'] == 0

    original = _burnup_module._FUEL_BOUNDS['condry']
    monkeypatch.setitem(
        _burnup_module._FUEL_BOUNDS, 'condry',
        (0.2, 0.25, 'thermal conductivity (W/m*K)'),
    )
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 27)
    monkeypatch.undo()

    restored = bc._run_burnup_cell(_base_ckw())
    assert restored['burnup_error'] == 0
    assert _burnup_module._FUEL_BOUNDS['condry'] == original


def test_code_28_ignition_temperature_out_of_fuel_bounds(monkeypatch):
    """Code 28: ``tpig`` (ignition temperature) out of ``_FUEL_BOUNDS`` -
    ``_run_burnup_cell()`` hardcodes sound-class particles' ``tpig`` to
    the module-level :data:`~pyfofem.components.burnup_calcs._SOUND_TPIG`
    constant, monkeypatched out of range here to drive this branch
    through the real ``_check_fuel()`` -> ``BurnupValidationError`` ->
    ``'ignition temperature'`` fragment match."""
    monkeypatch.setattr(bc, '_SOUND_TPIG', 500.0)  # _FUEL_BOUNDS['tpig'] is (200, 400)
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 28)


def test_code_29_char_temperature_out_of_fuel_bounds(monkeypatch):
    """Code 29: ``tchar`` (char temperature) out of ``_FUEL_BOUNDS`` -
    ``_run_burnup_cell()`` hardcodes every particle's ``tchar`` to the
    module-level :data:`~pyfofem.components.burnup_calcs._TCHAR`
    constant, monkeypatched out of range here to drive this branch
    through the real ``_check_fuel()`` -> ``BurnupValidationError`` ->
    ``'char temperature'`` fragment match."""
    monkeypatch.setattr(bc, '_TCHAR', 600.0)  # _FUEL_BOUNDS['tchar'] is (250, 500)
    result = bc._run_burnup_cell(_base_ckw())
    _assert_error_result(result, 29)


def test_code_90_no_fuel_particles():
    """Code 90: every recognised fuel-class loading is ``<= 0`` (here,
    none supplied at all) - the earliest possible failure, checked before
    any fire-environment input."""
    result = bc._run_burnup_cell(_base_ckw(fuel_loadings_bu={}))
    _assert_error_result(result, 90)


def test_code_91_ntimes_not_positive():
    """Code 91: ``ntimes`` (``bkw['max_times']``) ``<= 0`` - a real
    ``BurnupValidationError("ntimes must be > 0")`` from ``burnup()``
    itself, translated via the ``'ntimes'`` substring match."""
    result = bc._run_burnup_cell(_base_ckw(
        bkw=dict(r0=1.83, dr=0.4, max_times=0, fint_switch=15.0, validate=True),
    ))
    _assert_error_result(result, 91)


def test_code_99_unexpected_exception_falls_back_to_catch_all():
    """Code 99: a genuinely unexpected exception (here, a real
    ``KeyError`` from a malformed ``bkw`` dict missing the required
    ``'fint_switch'`` key - a real programming-error/malformed-input
    scenario, not a physical out-of-range value) is caught by
    ``_run_burnup_cell()``'s bare ``except Exception:`` catch-all and
    translated to code 99, exactly matching
    ``_BURNUP_LIMIT_ERROR[99]``'s own "unexpected burnup exception"
    description - it does NOT propagate and abort the caller."""
    malformed_bkw = dict(r0=1.83, dr=0.4, max_times=3000, validate=True)
    result = bc._run_burnup_cell(_base_ckw(bkw=malformed_bkw))
    _assert_error_result(result, 99)


def test_error_code_table_completeness():
    """Meta-test: every code this module's docstring claims to cover
    (:data:`_COVERED_CODES`) must equal EXACTLY the full registered
    ``_BURNUP_LIMIT_ERROR`` table, with no duplicates, no gaps, and no
    stale/removed codes - catching an accidentally duplicated, missing,
    or newly introduced production error code before it silently
    escapes test coverage. There is no "unreachable" carve-out: every
    registered code, including 21/26/27, has real executable coverage
    (see F-59's retraction)."""
    covered = set(_COVERED_CODES)
    assert len(covered) == len(_COVERED_CODES), "duplicate code in _COVERED_CODES"
    registered = set(_BURNUP_LIMIT_ERROR)
    assert covered == registered, (
        f"mismatch between this module's tracked codes and the real "
        f"_BURNUP_LIMIT_ERROR table: "
        f"missing from this module={registered - covered}, "
        f"stale in this module={covered - registered}"
    )


def test_every_covered_code_has_a_dedicated_test_function():
    """Meta-test: every code in :data:`_COVERED_CODES` must have exactly
    one dedicated ``test_code_<N>_...`` function actually collected in
    this module - guards against a code being *listed* as covered here
    without a real test ever being written (or being silently deleted
    later without updating the list)."""
    import sys
    module = sys.modules[__name__]
    test_names = [name for name in dir(module) if name.startswith('test_code_')]
    found_codes = set()
    for name in test_names:
        match = re.match(r'test_code_(\d+)_', name)
        assert match, f"{name} does not match the expected test_code_<N>_... pattern"
        found_codes.add(int(match.group(1)))
    assert found_codes == set(_COVERED_CODES), (
        f"test_code_<N>_ functions do not match _COVERED_CODES: "
        f"missing test={set(_COVERED_CODES) - found_codes}, "
        f"extra test={found_codes - set(_COVERED_CODES)}"
    )


def test_success_path_returns_zero_and_full_output_shape():
    """Companion positive-path contract test: a fully nominal ``ckw``
    (identical to :func:`_base_ckw`'s own defaults, which every one of
    the above negative tests perturbs exactly one field away from) must
    return ``burnup_error == 0`` and the complete success-path output
    shape/state ``_run_burnup_cell()``'s own docstring documents - proving
    the negative tests above are each isolating exactly one real failure
    condition, not merely exercising an already-broken baseline."""
    result = bc._run_burnup_cell(_base_ckw())
    assert result['burnup_error'] == 0
    assert set(result) == {
        'bcon', 'fla_dur', 'smo_dur', 'burnup_times_s', 'burnup_fi_wl',
        'burnup_fi_hs', 'class_order', 'burnup_limit_adjust', 'burnup_error',
    }
    assert result['class_order'] == ['litter']
    assert 'litter' in result['bcon']
    assert set(result['bcon']['litter']) == {
        'consumed', 'flaming', 'smoldering', 'frac_remaining',
    }
