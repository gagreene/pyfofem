#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_burnup_component_api.py - Phase 3 direct coverage for the
documented **component-level** scientific API that Gate 0 classified as
*supported* (``development/plans/gate0/02-api-inventory.md`` §3):
``pyfofem.components.FuelParticle``, ``BurnResult``,
``BurnSummaryRow``, ``BurnupValidationError`` and the ``burnup``
entry point they belong to.

Scope is exactly that inventory §3 list. The three
``_component_helpers`` underscore exports are classified **accidental**
and are deliberately not covered here; the three ``burnup_calcs``
underscore exports are classified *intentionally semi-public* but are
assigned to **Phase 7** (the error-code table, the bounds-handling
divergence and the rate-to-mass translation), so they are not covered
here either.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):

(a) *Python contract/equation test* - **every test in this module.**
(b) *Source-relation cross-check* - **none here.**
(c) *Executable C++ parity* - **none here.** ``run_burnup``'s existing
    golden-CSV comparison lives in ``tests/unit/test_burnup_golden.py``;
    nothing in this module builds or runs C++.

**The documented rate-vs-mass defect is pinned, not fixed.**
``BurnResult.comp_flaming`` / ``comp_smoldering`` are *documented* as
"cumulative mass consumed (kg/m2)" but are in fact **rates**
(kg/m2/s) - the real consumer,
``burnup_calcs._extract_burnup_consumption``, multiplies each value by
the record's own time interval. Gate 0 §3 requires the field-unit tests
to assert the real (rate) semantics so that a later docstring fix
cannot silently become a behaviour change;
:func:`test_burn_result_component_fields_are_rates_not_cumulative_mass`
does that with a dimensional discriminator rather than by restating
either wording.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import dataclasses

import pytest

from pyfofem import components
from pyfofem.components import (
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    FuelParticle,
    burnup,
)

#: A physically valid single fuel particle, well inside every bound in
#: ``burnup._FUEL_BOUNDS``. Used as the baseline for the construction
#: and simulation tests below.
_VALID_PARTICLE_KWARGS = {
    "wdry": 0.5,
    "htval": 1.85e7,
    "fmois": 0.1,
    "dendry": 400.0,
    "sigma": 2000.0,
}

#: A fire environment that reliably ignites :data:`_VALID_PARTICLE_KWARGS`
#: and completes in a handful of time steps.
_VALID_FIRE_KWARGS = {
    "fi": 200.0,
    "ti": 30.0,
    "u": 1.0,
    "d": 0.3,
    "tamb": 20.0,
    "r0": 1.83,
    "dr": 0.4,
    "dt": 15.0,
    "ntimes": 40,
}


def _run_reference_burn():
    """
    Run the baseline single-particle simulation used by several tests.

    :return: The ``(results, summary)`` 2-tuple returned by
        :func:`pyfofem.components.burnup`.
    """
    return burnup(particles=[FuelParticle(**_VALID_PARTICLE_KWARGS)], **_VALID_FIRE_KWARGS)


def test_burn_result_component_fields_are_rates_not_cumulative_mass():
    """
    Pin the real (rate) semantics of ``comp_flaming`` / ``comp_smoldering``
    against the field's own docstring, which calls them cumulative mass.

    The two readings are separated dimensionally, not by restating either
    wording. Summing ``value * interval`` over the records reconstructs
    the mass the summary row reports as consumed to within a few percent
    (rate reading); summing the raw values instead - the reading the
    docstring implies - lands more than an order of magnitude low,
    because it drops the seconds. Measured on this scenario: 1.04x
    versus 0.037x of the consumed mass.

    Bounds are deliberately loose (0.5-2.0x versus < 0.2x) so the test
    asserts the *dimension*, not a tuned numerical value.

    :return: None. Raises via ``assert`` on mismatch.
    """
    results, summary = _run_reference_burn()
    consumed = summary[0].wdry * (1.0 - summary[0].frac_remaining)
    assert consumed > 0.0

    rate_reading = 0.0
    mass_reading = 0.0
    previous_time = 0.0
    for record in results:
        interval = record.time - previous_time
        previous_time = record.time
        step_total = record.comp_flaming[0] + record.comp_smoldering[0]
        rate_reading += step_total * interval
        mass_reading += step_total

    assert 0.5 <= rate_reading / consumed <= 2.0
    assert mass_reading / consumed < 0.2


def test_burn_result_defaults_are_immutable_and_not_shared():
    """
    ``BurnResult``'s four optional fields default to ``None``, not to a
    shared mutable list - the classic dataclass footgun. Verify the
    absence of sharing directly: two default-constructed instances must
    not alias, and assigning a list to one must not reach the other.

    :return: None. Raises via ``assert`` on mismatch.
    """
    first = BurnResult(time=0.0, wdf=1.0, ff=0.5)
    second = BurnResult(time=0.0, wdf=1.0, ff=0.5)

    for field in ("comp_flaming", "comp_smoldering", "fi_wl", "fi_hs"):
        assert getattr(first, field) is None
        assert getattr(second, field) is None

    first.comp_flaming = [1.0, 2.0]
    assert second.comp_flaming is None


def test_burn_result_field_order_and_storage():
    """
    ``BurnResult`` must declare its seven fields in the documented order
    and store positional arguments without transformation.

    :return: None. Raises via ``assert`` on mismatch.
    """
    names = [field.name for field in dataclasses.fields(BurnResult)]
    assert names == [
        "time",
        "wdf",
        "ff",
        "comp_flaming",
        "comp_smoldering",
        "fi_wl",
        "fi_hs",
    ]

    record = BurnResult(12.5, 0.75, 0.4, [1.0], [2.0, 0.0], 300.0, 10.0)
    assert record.time == 12.5
    assert record.wdf == 0.75
    assert record.ff == 0.4
    assert record.comp_flaming == [1.0]
    assert record.comp_smoldering == [2.0, 0.0]
    assert record.fi_wl == 300.0
    assert record.fi_hs == 10.0


def test_burn_result_smoldering_list_carries_one_extra_duff_slot():
    """
    ``comp_smoldering``'s documented layout is one entry per fuel
    component **plus** a trailing duff-smouldering slot at index
    ``number``, whereas ``comp_flaming`` has exactly one entry per
    component. Verify that against a real simulation rather than trusting
    the docstring.

    :return: None. Raises via ``assert`` on mismatch.
    """
    results, summary = _run_reference_burn()
    number = len(summary)
    assert number == 1

    for record in results:
        assert len(record.comp_flaming) == number
        assert len(record.comp_smoldering) == number + 1


def test_burn_summary_row_field_order_and_storage():
    """
    ``BurnSummaryRow`` must declare its eight fields in the documented
    order and store them verbatim.

    :return: None. Raises via ``assert`` on mismatch.
    """
    names = [field.name for field in dataclasses.fields(BurnSummaryRow)]
    assert names == [
        "component",
        "wdry",
        "fmois",
        "diam",
        "t_ignite",
        "t_burnout",
        "remaining",
        "frac_remaining",
    ]

    row = BurnSummaryRow(1, 0.5, 0.1, 0.002, 0.0, 37.5, 0.05, 0.1)
    assert row.component == 1
    assert row.wdry == 0.5
    assert row.fmois == 0.1
    assert row.diam == 0.002
    assert row.t_ignite == 0.0
    assert row.t_burnout == 37.5
    assert row.remaining == 0.05
    assert row.frac_remaining == 0.1


def test_burn_summary_row_has_no_defaults_and_requires_every_field():
    """
    Every ``BurnSummaryRow`` field is required - the class is an output
    record, so a silently defaulted field would be an unnoticed hole in
    a reported result.

    :return: None. Raises via ``assert`` on mismatch.
    """
    for field in dataclasses.fields(BurnSummaryRow):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING

    with pytest.raises(TypeError):
        BurnSummaryRow(1, 0.5, 0.1)


def test_burn_summary_row_returned_by_burnup_is_internally_consistent():
    """
    Oracle-independent invariant on the real summary row: the remaining
    loading must equal ``wdry * frac_remaining``, the remaining fraction
    must lie in ``[0, 1]``, burnout must not precede ignition, and the
    component index must be 1-based.

    :return: None. Raises via ``assert`` on mismatch.
    """
    _, summary = _run_reference_burn()
    row = summary[0]

    assert row.component == 1
    assert 0.0 <= row.frac_remaining <= 1.0
    assert row.remaining == pytest.approx(row.wdry * row.frac_remaining, abs=1e-12)
    assert row.t_burnout >= row.t_ignite
    assert row.diam > 0.0


def test_burnup_returns_the_documented_two_tuple_of_record_lists():
    """
    ``burnup`` must return ``(results, summary)`` - a list of
    :class:`BurnResult` and a list of :class:`BurnSummaryRow`, one
    summary row per input particle.

    Pinned because ``burnup_calcs.run_burnup`` returns a **3**-tuple; a
    caller that confuses the two arities gets a silent unpacking bug.

    :return: None. Raises via ``assert`` on mismatch.
    """
    returned = _run_reference_burn()
    assert isinstance(returned, tuple)
    assert len(returned) == 2

    results, summary = returned
    assert isinstance(results, list) and results
    assert isinstance(summary, list) and len(summary) == 1
    assert all(isinstance(record, BurnResult) for record in results)
    assert all(isinstance(row, BurnSummaryRow) for row in summary)


def test_burnup_validation_error_is_a_value_error_subclass():
    """
    ``BurnupValidationError`` must subclass ``ValueError`` so existing
    ``except ValueError`` handlers keep working, and must remain a
    distinct type so callers can catch it specifically.

    Its classification is load-bearing beyond ergonomics:
    ``burnup_calcs._run_burnup_cell`` catches it to translate a failed
    cell into a numeric ``BurnupError`` code, and a change of base class
    would silently change that behaviour.

    :return: None. Raises via ``assert`` on mismatch.
    """
    assert issubclass(BurnupValidationError, ValueError)
    assert issubclass(BurnupValidationError, Exception)
    assert BurnupValidationError is not ValueError

    error = BurnupValidationError("boom")
    assert isinstance(error, ValueError)
    assert str(error) == "boom"

    with pytest.raises(ValueError):
        raise BurnupValidationError("caught as ValueError")


def test_burnup_validation_error_is_raised_for_structural_input_faults():
    """
    The two structural preconditions - at least one particle, and a
    positive step count - must raise ``BurnupValidationError`` with a
    specific message, and must do so regardless of ``validate``, since
    they are checked before the range validation that ``validate``
    governs.

    :return: None. Raises via ``assert`` on mismatch.
    """
    particle = FuelParticle(**_VALID_PARTICLE_KWARGS)

    for validate in (True, False):
        kwargs = dict(_VALID_FIRE_KWARGS)
        kwargs["validate"] = validate

        with pytest.raises(BurnupValidationError, match="at least one fuel particle"):
            burnup(particles=[], **kwargs)

        no_steps = dict(kwargs, ntimes=0)
        with pytest.raises(BurnupValidationError, match="ntimes must be > 0"):
            burnup(particles=[particle], **no_steps)


@pytest.mark.parametrize(
    ("field", "bad_value", "message_fragment"),
    [
        ("ash", 0.5, "ash content (fraction)"),
        ("cheat", 100.0, "heat capacity"),
        ("condry", 1.0, "thermal conductivity"),
        ("dendry", 50.0, "dry mass density"),
        ("fmois", 5.0, "fuel moisture (fraction)"),
        ("htval", 1.0e5, "heat content (J/kg)"),
        ("sigma", 1.0, "SAV (1/m)"),
        ("tchar", 900.0, "char temperature"),
        ("tpig", 50.0, "ignition temperature"),
        ("wdry", 0.0, "dry loading"),
    ],
)
def test_burnup_validation_rejects_out_of_range_particle_fields(
        field,
        bad_value,
        message_fragment,
):
    """
    With ``validate=True`` (the default), every out-of-range
    :class:`FuelParticle` field must raise ``BurnupValidationError``
    naming the offending quantity, its value and its bounds.

    :param field: ``FuelParticle`` attribute to push out of range.
    :param bad_value: An out-of-range value for that attribute.
    :param message_fragment: Text that must appear in the raised
        message, identifying which quantity failed.
    :return: None. Raises via ``assert`` on mismatch.
    """
    kwargs = dict(_VALID_PARTICLE_KWARGS)
    kwargs[field] = bad_value
    particle = FuelParticle(**kwargs)

    with pytest.raises(BurnupValidationError) as excinfo:
        burnup(particles=[particle], **_VALID_FIRE_KWARGS)

    message = str(excinfo.value)
    assert message_fragment in message
    assert "Fuel class 0" in message
    assert "out of range" in message


def test_burnup_validation_skipped_when_validate_is_false():
    """
    ``validate=False`` must skip the particle range checks: the same
    out-of-range particle that raises with the default reaches the
    simulation instead.

    It still raises ``BurnupValidationError`` here - but with the
    *runtime* "No fuel ignited" message rather than a range message.
    That is pinned deliberately: the same exception type carries both an
    input-validation and a runtime-outcome meaning, which a caller
    distinguishing the two needs to know.

    :return: None. Raises via ``assert`` on mismatch.
    """
    soaked = FuelParticle(**dict(_VALID_PARTICLE_KWARGS, fmois=5.0))

    with pytest.raises(BurnupValidationError, match="fuel moisture"):
        burnup(particles=[soaked], **_VALID_FIRE_KWARGS)

    with pytest.raises(BurnupValidationError, match="No fuel ignited"):
        burnup(particles=[soaked], **dict(_VALID_FIRE_KWARGS, validate=False))


def test_fuel_particle_defaults_match_the_documented_values():
    """
    ``FuelParticle``'s five optional fields must keep their documented
    defaults, and its five required fields must have no default at all.

    The defaults are model inputs, not conveniences: a changed
    ``cheat``/``condry``/``tpig``/``tchar``/``ash`` silently changes
    every standalone burnup result.

    :return: None. Raises via ``assert`` on mismatch.
    """
    defaults = {
        field.name: field.default for field in dataclasses.fields(FuelParticle)
    }
    assert defaults["cheat"] == 2750.0
    assert defaults["condry"] == 0.133
    assert defaults["tpig"] == 300.0
    assert defaults["tchar"] == 350.0
    assert defaults["ash"] == 0.05

    for required in ("wdry", "htval", "fmois", "dendry", "sigma"):
        assert defaults[required] is dataclasses.MISSING


def test_fuel_particle_equality_and_repr_are_value_based():
    """
    ``FuelParticle`` is a plain dataclass, so equality is field-by-field
    and ``repr`` shows every field. Both are part of the documented
    component-level API surface and are pinned here.

    :return: None. Raises via ``assert`` on mismatch.
    """
    first = FuelParticle(**_VALID_PARTICLE_KWARGS)
    same = FuelParticle(**_VALID_PARTICLE_KWARGS)
    different = FuelParticle(**dict(_VALID_PARTICLE_KWARGS, wdry=0.6))

    assert first == same
    assert first is not same
    assert first != different
    assert first != object()

    text = repr(first)
    assert text.startswith("FuelParticle(")
    for field in dataclasses.fields(FuelParticle):
        assert f"{field.name}=" in text


def test_fuel_particle_field_order_and_verbatim_storage():
    """
    ``FuelParticle`` must declare its ten fields in the documented order
    and store every value verbatim.

    Verbatim storage matters for the temperature fields specifically:
    ``tpig``/``tchar`` are documented in **degrees Celsius** and the
    Celsius-to-Kelvin conversion happens inside :func:`burnup`, not at
    construction. A future conversion moved into ``__post_init__`` would
    double-convert, and this test would catch it.

    :return: None. Raises via ``assert`` on mismatch.
    """
    names = [field.name for field in dataclasses.fields(FuelParticle)]
    assert names == [
        "wdry",
        "htval",
        "fmois",
        "dendry",
        "sigma",
        "cheat",
        "condry",
        "tpig",
        "tchar",
        "ash",
    ]

    particle = FuelParticle(0.5, 1.85e7, 0.1, 400.0, 2000.0, 2600.0, 0.14, 310.0, 360.0, 0.04)
    assert particle.wdry == 0.5
    assert particle.htval == 1.85e7
    assert particle.fmois == 0.1
    assert particle.dendry == 400.0
    assert particle.sigma == 2000.0
    assert particle.cheat == 2600.0
    assert particle.condry == 0.14
    assert particle.tpig == 310.0
    assert particle.tchar == 360.0
    assert particle.ash == 0.04


def test_supported_component_symbols_are_exported_from_pyfofem_components():
    """
    All five supported component-level symbols must be reachable from
    ``pyfofem.components`` and declared in its ``__all__``, and each
    must be the kind of object the inventory records (four classes plus
    one function).

    :return: None. Raises via ``assert`` on mismatch.
    """
    expected = {
        "FuelParticle": FuelParticle,
        "BurnResult": BurnResult,
        "BurnSummaryRow": BurnSummaryRow,
        "BurnupValidationError": BurnupValidationError,
        "burnup": burnup,
    }
    for name, obj in expected.items():
        assert name in components.__all__
        assert getattr(components, name) is obj

    for name in ("FuelParticle", "BurnResult", "BurnSummaryRow"):
        assert dataclasses.is_dataclass(expected[name])
    assert issubclass(BurnupValidationError, Exception)
    assert callable(burnup)
