#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_utility_contracts.py - Phase 3 coverage for the two Python/data
utility functions Gate 0 classified as having no executable C++ target:
``calc_carbon`` and ``get_moisture_regime``.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):

(a) *Python contract/equation test* - every ``get_moisture_regime`` test
    below, plus ``calc_carbon``'s key-domain, units-are-a-no-op,
    array-support and input-immutability tests.
(b) *Source-relation cross-check* - ``calc_carbon``'s factor values and
    per-component factor assignment, hand-derived from the pinned
    FOF_GUI report equations.
(c) *Executable C++ parity* - **none in this module.**

**calc_carbon.** Gate 0 tier 5: the two carbon factors exist in the
pinned reference only as preprocessor constants inside the report
writer, not as a callable scientific-core function, so there is nothing
to drive as an oracle and no GUI harness is invented here. Verified
directly at C++ SHA ``78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`` in
``reference/fofem_cpp/FOF_GUI/Wnd_Rep.cpp``:

* ``:731`` ``#define e_LitDufCar 0.37``
* ``:732`` ``#define e_LivDeaCar 0.50``

and the per-component assignment in the same file's carbon table:

* ``:916-917`` litter uses ``e_LitDufCar``
* ``:924-930`` wood - ``f_DW1 + f_DW10 + f_DW100 + f_Snd_DW1k +
  f_Rot_DW1k`` - uses ``e_LivDeaCar``
* ``:938-939`` duff uses ``e_LitDufCar``
* ``:946-947`` herbaceous uses ``e_LivDeaCar``
* ``:954-955`` shrub uses ``e_LivDeaCar``
* ``:962-963`` foliage + branch uses ``e_LivDeaCar``

Python splits the 1000-hour class into four diameter bins
(``dwk_3_6``/``dwk_6_9``/``dwk_9_20``/``dwk_20``) where C++ splits it
into sound/rotten. That is a component-granularity difference only:
every 1000-hour bin takes the same 0.50 factor on both sides, so the
factor assignment is equivalent. The tests below assert the factor per
component, not the component partition.

**get_moisture_regime.** Gate 0 tier 5, confirmed Python-only: no C++
counterpart exists. Coverage required by
``gate0/03-cpp-crosswalk.md`` row 16 - all four regimes, name
normalisation, invalid name, and the defensive copy - is provided here.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem import calc_carbon, get_moisture_regime
from pyfofem.components import consumption_calcs

#: Carbon factor for duff and litter, ``e_LitDufCar`` at
#: ``FOF_GUI/Wnd_Rep.cpp:731``.
_DUFF_FACTOR = 0.37

#: Carbon factor for down woody, herb, shrub, foliage and branch,
#: ``e_LivDeaCar`` at ``FOF_GUI/Wnd_Rep.cpp:732``.
_WOODY_FACTOR = 0.50

#: Every fuel-component key ``calc_carbon`` accepts, paired with the
#: pinned C++ factor its component maps to, and the pinned line that
#: establishes the mapping.
_KEY_TO_FACTOR = {
    "litter": (_DUFF_FACTOR, "Wnd_Rep.cpp:916-917"),
    "duff": (_DUFF_FACTOR, "Wnd_Rep.cpp:938-939"),
    "dw1": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dw10": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dw100": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dwk_3_6": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dwk_6_9": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dwk_9_20": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "dwk_20": (_WOODY_FACTOR, "Wnd_Rep.cpp:924-930"),
    "herb": (_WOODY_FACTOR, "Wnd_Rep.cpp:946-947"),
    "shrub": (_WOODY_FACTOR, "Wnd_Rep.cpp:954-955"),
    "foliage": (_WOODY_FACTOR, "Wnd_Rep.cpp:962-963"),
    "branch": (_WOODY_FACTOR, "Wnd_Rep.cpp:962-963"),
}

#: The four documented moisture regimes and their percent values
#: (``consumption_calcs.py``; Lutes 2020 p. 79). Written out here as an
#: independent expectation rather than read from the module under test.
_EXPECTED_REGIMES = {
    "wet": {"duff": 130.0, "10hr": 22.0, "3plus": 40.0, "soil": 25.0},
    "moderate": {"duff": 75.0, "10hr": 16.0, "3plus": 30.0, "soil": 15.0},
    "dry": {"duff": 40.0, "10hr": 10.0, "3plus": 15.0, "soil": 10.0},
    "very dry": {"duff": 20.0, "10hr": 6.0, "3plus": 10.0, "soil": 5.0},
}


def test_carbon_accepts_numpy_arrays_elementwise():
    """
    Category (a). Array loadings are converted elementwise and the
    returned array keeps the input shape, per the scalar-array
    convention.

    Hand-derived: ``[1, 2, 4] * 0.5 = [0.5, 1.0, 2.0]`` and
    ``[10, 100] * 0.37 = [3.7, 37.0]``.

    :return: None. Raises via ``assert`` on mismatch.
    """
    result = calc_carbon(
        {
            "dw10": np.array([1.0, 2.0, 4.0]),
            "duff": np.array([10.0, 100.0]),
        }
    )
    assert isinstance(result["dw10"], np.ndarray)
    assert result["dw10"].shape == (3,)
    np.testing.assert_allclose(result["dw10"], [0.5, 1.0, 2.0], rtol=1e-12)
    np.testing.assert_allclose(result["duff"], [3.7, 37.0], rtol=1e-12)


def test_carbon_does_not_mutate_the_caller_dict():
    """
    Category (a). The input mapping must be left untouched - the
    function builds a new dict rather than converting in place.

    :return: None. Raises via ``assert`` on mismatch.
    """
    loadings = {"litter": 10.0, "dw1": 4.0}
    calc_carbon(loadings)
    assert loadings == {"litter": 10.0, "dw1": 4.0}


def test_carbon_empty_mapping_returns_empty_mapping():
    """
    Category (a). An empty input produces an empty output rather than
    raising or fabricating default components.

    :return: None. Raises via ``assert`` on mismatch.
    """
    assert calc_carbon({}) == {}


@pytest.mark.parametrize(
    ("key", "factor", "citation"),
    [(key, factor, citation) for key, (factor, citation) in sorted(_KEY_TO_FACTOR.items())],
)
def test_carbon_factor_per_component_matches_pinned_report_equations(key, factor, citation):
    """
    Category (b). Each accepted fuel-component key must use the carbon
    factor its C++ counterpart uses in the pinned FOF_GUI carbon table.

    Hand-derived with a loading of exactly 100 units, chosen so the
    expected carbon value is the factor itself times 100 and needs no
    floating-point reasoning: ``100 * 0.37 = 37.0`` for the litter/duff
    components and ``100 * 0.50 = 50.0`` for the live/dead components.

    No executable parity is claimed: the factors are ``#define``
    constants inside the report writer, not a callable scientific-core
    function (Gate 0 tier 5).

    :param key: Fuel-component key accepted by ``calc_carbon``.
    :param factor: Pinned C++ carbon factor for that component.
    :param citation: Pinned ``Wnd_Rep.cpp`` lines establishing the
        mapping, recorded in the test id for traceability.
    :return: None. Raises via ``assert`` on mismatch.
    """
    assert citation.startswith("Wnd_Rep.cpp:")
    result = calc_carbon({key: 100.0})
    assert result[key] == pytest.approx(100.0 * factor, rel=1e-12)


def test_carbon_factors_are_exactly_the_pinned_constants():
    """
    Category (b). Pin the two factor values themselves against the
    pinned ``#define``\\ s, using loadings of 1.0 so the returned carbon
    value *is* the factor.

    ``e_LitDufCar 0.37`` (``FOF_GUI/Wnd_Rep.cpp:731``) and
    ``e_LivDeaCar 0.50`` (``:732``).

    :return: None. Raises via ``assert`` on mismatch.
    """
    result = calc_carbon({"duff": 1.0, "litter": 1.0, "dw1": 1.0, "shrub": 1.0})
    assert result["duff"] == pytest.approx(0.37, rel=1e-15)
    assert result["litter"] == pytest.approx(0.37, rel=1e-15)
    assert result["dw1"] == pytest.approx(0.50, rel=1e-15)
    assert result["shrub"] == pytest.approx(0.50, rel=1e-15)


def test_carbon_preserves_input_keys_and_rejects_unknown_ones():
    """
    Category (a). The output carries exactly the keys supplied, and an
    unrecognised key raises ``ValueError`` naming the offending key
    rather than being silently dropped or defaulted.

    :return: None. Raises via ``assert`` on mismatch.
    """
    supplied = {"litter": 1.0, "herb": 2.0, "dwk_20": 3.0}
    assert set(calc_carbon(supplied)) == set(supplied)

    with pytest.raises(ValueError, match="Unrecognised fuel component key 'sawdust'"):
        calc_carbon({"sawdust": 1.0})


def test_carbon_units_argument_performs_no_conversion():
    """
    Category (a). ``units`` is documented as informational only. Pin
    that it is a genuine no-op so it cannot silently start converting -
    Gate 0 explicitly required this
    (``gate0/03-cpp-crosswalk.md`` row 3).

    :return: None. Raises via ``assert`` on mismatch.
    """
    loadings = {"litter": 10.0, "dw1": 4.0, "duff": 25.0}
    si = calc_carbon(loadings, units="SI")
    imperial = calc_carbon(loadings, units="imperial")
    nonsense = calc_carbon(loadings, units="furlongs-per-fortnight")
    assert si == imperial == nonsense
    assert si == {"litter": 3.7, "dw1": 2.0, "duff": 9.25}


def test_carbon_zero_and_negative_loadings_scale_linearly():
    """
    Category (a). The conversion is a pure multiplication with no
    clamping, so zero maps to zero and a negative loading maps to a
    negative carbon value.

    Pinned as current behaviour: ``calc_carbon`` performs no domain
    validation on the loading values themselves.

    :return: None. Raises via ``assert`` on mismatch.
    """
    result = calc_carbon({"duff": 0.0, "dw1": -8.0})
    assert result["duff"] == 0.0
    assert result["dw1"] == pytest.approx(-4.0, rel=1e-12)


def test_moisture_regime_defensive_copy_protects_the_module_table():
    """
    Category (a). The returned dict must be a copy: mutating it must not
    corrupt the module-level ``_MOISTURE_REGIMES`` table that every
    later caller reads.

    Verified directly - the returned mapping is mutated, then a fresh
    call and the module table itself are both re-checked. Gate 0
    required this check explicitly
    (``gate0/03-cpp-crosswalk.md`` row 16).

    :return: None. Raises via ``assert`` on mismatch.
    """
    first = get_moisture_regime("dry")
    assert first is not consumption_calcs._MOISTURE_REGIMES["dry"]

    first["duff"] = -999.0
    first["injected"] = 1.0
    del first["soil"]

    second = get_moisture_regime("dry")
    assert second == _EXPECTED_REGIMES["dry"]
    assert consumption_calcs._MOISTURE_REGIMES["dry"] == _EXPECTED_REGIMES["dry"]


def test_moisture_regime_each_call_returns_an_independent_object():
    """
    Category (a). Two calls for the same regime must return equal but
    distinct dicts, so one caller's mutation cannot reach another's.

    :return: None. Raises via ``assert`` on mismatch.
    """
    a = get_moisture_regime("wet")
    b = get_moisture_regime("wet")
    assert a == b
    assert a is not b


@pytest.mark.parametrize("regime", sorted(_EXPECTED_REGIMES))
def test_moisture_regime_exact_values_for_each_regime(regime):
    """
    Category (a). Each of the four documented regimes must return its
    documented ``{duff, 10hr, 3plus, soil}`` percentages exactly, as
    floats.

    :param regime: One of the four canonical regime names.
    :return: None. Raises via ``assert`` on mismatch.
    """
    result = get_moisture_regime(regime)
    assert isinstance(result, dict)
    assert result == _EXPECTED_REGIMES[regime]
    assert list(result) == ["duff", "10hr", "3plus", "soil"]
    assert all(isinstance(value, float) for value in result.values())


@pytest.mark.parametrize(
    ("supplied", "canonical"),
    [
        ("WET", "wet"),
        ("Wet", "wet"),
        ("  moderate  ", "moderate"),
        ("DRY\t", "dry"),
        ("Very Dry", "very dry"),
        ("VERY DRY", "very dry"),
        (" very dry ", "very dry"),
    ],
)
def test_moisture_regime_normalises_case_and_surrounding_whitespace(supplied, canonical):
    """
    Category (a). Lookup strips surrounding whitespace and lowercases
    the name before matching.

    :param supplied: Regime name as a caller might type it.
    :param canonical: The canonical regime key it must resolve to.
    :return: None. Raises via ``assert`` on mismatch.
    """
    assert get_moisture_regime(supplied) == _EXPECTED_REGIMES[canonical]


@pytest.mark.parametrize(
    "regime", ["", "damp", "verydry", "very  dry", "very-dry", "wetter", "moist"]
)
def test_moisture_regime_rejects_unknown_names(regime):
    """
    Category (a). An unrecognised regime name raises ``KeyError``, and
    the message lists the valid options.

    ``'verydry'`` and ``'very  dry'`` are included deliberately:
    normalisation collapses surrounding whitespace only, not internal
    whitespace, so neither resolves to ``'very dry'``.

    :param regime: A name that is not one of the four regimes.
    :return: None. Raises via ``assert`` on mismatch.
    """
    with pytest.raises(KeyError, match="Unknown moisture regime"):
        get_moisture_regime(regime)


def test_moisture_regime_values_are_ordered_wet_to_very_dry():
    """
    Category (a). Oracle-independent invariant: for every fuel key the
    four regimes must be strictly ordered wet > moderate > dry >
    very dry, so a future table edit that transposed two rows would
    fail here rather than silently changing model inputs.

    :return: None. Raises via ``assert`` on mismatch.
    """
    order = ["wet", "moderate", "dry", "very dry"]
    for key in ("duff", "10hr", "3plus", "soil"):
        values = [get_moisture_regime(name)[key] for name in order]
        assert values == sorted(values, reverse=True)
        assert len(set(values)) == len(values)
        assert all(value > 0.0 for value in values)
