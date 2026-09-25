#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_tree_flame_source_relations.py - Phase 3 source-relation coverage
for ``calc_char_ht`` and ``calc_crown_length_vol_scorched``.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):

(a) *Python contract/equation test* - the shape/scalar-array, clamping
    and warning-behaviour tests below.
(b) *Source-relation cross-check* - hand-derived expected values taken
    from a pinned C++ expression that **cannot be executed in
    isolation**. The value tests below are category (b).
(c) *Executable C++ parity* - **none in this module, and none is
    possible.**

**Why there is no live parity here.** Both pinned C++ expressions live
inside ``MRT_Calc`` (``reference/fofem_cpp/FOF_UNIX/fof_mrt.cpp``), and
the intermediates they produce - ``f_CK`` (percent crown volume
scorched), ``f_CSL`` (percent crown length scorched) and ``f_Fl`` (the
case-4 flame length) - are **local variables** declared at
``fof_mrt.cpp:276-278``. They are never written to ``d_MO``
(``fof_mrt.h:57-89``), so no harness mode can emit them without
instrumenting the overlay, and recomputing them inside a wrapper would
violate the plan's oracle-independence rule. This is Gate 0 finding
**F-30**. Consequently every value assertion in this module is a
*source-relation* claim: the Python result is compared against a value
derived by hand from the pinned C++ expression, not against output from
a live C++ run.

**Pinned source, verified directly at C++ SHA**
``78f97f093ee7d1c77b3cd2622b2bd7248036c1e4``:

* ``fof_mrt.cpp:315`` ``f_HCR = f_Hgt * ( f_LCR / 10.0 );``
* ``fof_mrt.cpp:316`` ``f_B = f_Scorch - (f_Hgt - f_HCR);``
* ``fof_mrt.cpp:317-320`` clamp ``f_B`` to ``[0, f_HCR]``
* ``fof_mrt.cpp:326`` ``f_CK = 100.0 * (f_B * (2.0 * f_HCR - f_B) /
  ( f_HCR * f_HCR ) );``
* ``fof_mrt.cpp:327`` ``f_CSL = 100.0 * (f_B / f_HCR);``
* ``fof_mrt.cpp:329-333`` the ``f_HCR == 0`` branch: sets
  ``f_CK = 0``, ``f_CSL = 0``, writes the error text
  ``"Mortality Calculaton is attempting to Divide by 0"`` (sic) and
  **returns -1**
* ``fof_mrt.cpp:396-397`` ``f_Fl = Calc_Flame(f_Scorch);
  f_CH = f_Fl / 1.8 ;``

``f_HCR`` *is* crown depth (total height times crown ratio in tenths),
so passing ``crown_depth = ht * CR / 10`` makes Python's algebra
identical to the C++ block above; the tests below exercise exactly that
substitution.

Gate 0 finding **F-29** is pinned as current behaviour: where C++
returns -1 with an error message for a zero crown depth, Python divides
by zero and yields NaN with a NumPy ``RuntimeWarning``.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem import calc_char_ht, calc_crown_length_vol_scorched, calc_flame_length

#: Hand-derived crown-scorch cases, each carrying its own derivation.
#: Columns: ``(label, ht_m, crown_ratio_tenths, scorch_ht_m,
#: expected_crown_length_scorched_m, expected_cvs_pct, expected_cls_pct)``.
#: ``crown_depth`` is computed as ``ht * crown_ratio_tenths / 10``, the
#: pinned ``f_HCR`` expression at ``fof_mrt.cpp:315``.
_CROWN_CASES = [
    # ht=10, CR=4 -> HCR=4. B = 8 - (10-4) = 2, inside [0,4].
    # CK = 100*(2*(2*4-2)/4^2) = 100*(2*6/16) = 75.
    # CSL = 100*(2/4) = 50.
    ("partial-mid", 10.0, 4.0, 8.0, 2.0, 75.0, 50.0),
    # ht=30, CR=5 -> HCR=15. B = 20 - (30-15) = 5, inside [0,15].
    # CK = 100*(5*(30-5)/225) = 100*125/225 = 500/9 = 55.5555...
    # CSL = 100*(5/15) = 100/3 = 33.3333...
    ("partial-tall", 30.0, 5.0, 20.0, 5.0, 500.0 / 9.0, 100.0 / 3.0),
    # ht=10, CR=4 -> HCR=4. B = 20 - 6 = 14 > HCR, clamped to 4.
    # CK = 100*(4*(8-4)/16) = 100. CSL = 100*(4/4) = 100.
    ("clamped-high", 10.0, 4.0, 20.0, 4.0, 100.0, 100.0),
    # ht=20, CR=4 -> HCR=8. B = 10 - (20-8) = -2 <= 0, clamped to 0.
    # CK = 0, CSL = 0. This is the exact example recorded in
    # gate0/03-cpp-crosswalk.md row 5.
    ("clamped-low", 20.0, 4.0, 10.0, 0.0, 0.0, 0.0),
    # ht=10, CR=10 -> HCR=10 (crown occupies the whole stem).
    # B = 10 - (10-10) = 10, equals HCR (no clamp needed).
    # CK = 100*(10*(20-10)/100) = 100. CSL = 100.
    ("full-crown", 10.0, 10.0, 10.0, 10.0, 100.0, 100.0),
]


def _assert_close(actual: float, expected: float, rel: float = 1e-12) -> None:
    """
    Assert a scalar result matches a hand-derived expectation.

    The 1e-12 relative tolerance is a floating-point round-off allowance
    only; every expectation in this module is an exact closed-form value
    derived from the pinned C++ expression, so no scientific difference
    is being absorbed.

    :param actual: Value returned by the function under test.
    :param expected: Hand-derived expected value.
    :param rel: Relative tolerance passed through to
        :func:`pytest.approx`.
    :return: None. Raises via ``assert`` on mismatch.
    """
    assert actual == pytest.approx(expected, rel=rel)


def test_char_ht_array_input_returns_matching_shape():
    """
    Category (a). Array input returns an array of the same shape, with
    each element following the same relation.

    :return: None. Raises via ``assert`` on mismatch.
    """
    result = calc_char_ht(np.array([1.8, 3.6, 9.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    _assert_close(float(result[0]), 1.0)
    _assert_close(float(result[1]), 2.0)
    _assert_close(float(result[2]), 5.0)


def test_char_ht_is_the_exact_inverse_of_the_flame_length_char_mode():
    """
    Category (a). ``calc_char_ht`` and ``calc_flame_length(char_ht=...)``
    are exact inverses, so the two Python functions encode one
    self-consistent form of the pinned ``f_CH = f_Fl / 1.8`` relation
    rather than two independently drifting constants.

    :return: None. Raises via ``assert`` on mismatch.
    """
    for char_ht in (0.0, 0.5, 1.0, 2.75, 12.5):
        round_tripped = float(calc_char_ht(calc_flame_length(char_ht=char_ht)))
        assert round_tripped == pytest.approx(char_ht, abs=1e-12)


def test_char_ht_matches_the_pinned_mortality_relation():
    """
    Category (b). ``calc_char_ht`` must implement the pinned relation
    ``f_CH = f_Fl / 1.8`` from ``fof_mrt.cpp:396-397``.

    Hand-derived divisions by 1.8: ``1.8 -> 1.0``, ``3.6 -> 2.0``,
    ``9.0 -> 5.0``, ``0.0 -> 0.0``, ``0.9 -> 0.5``.

    This is a *relation-level* cross-check only. The C++ ``f_Fl``
    intermediate is a ``MRT_Calc`` local and is absent from ``d_MO``
    (finding F-30), so no executable comparison is possible and none is
    claimed. Wrapping ``Calc_Flame(scorch)/1.8`` to serve as an oracle
    would re-implement the caller rather than test it, and is explicitly
    forbidden by the plan.

    :return: None. Raises via ``assert`` on mismatch.
    """
    _assert_close(float(calc_char_ht(1.8)), 1.0)
    _assert_close(float(calc_char_ht(3.6)), 2.0)
    _assert_close(float(calc_char_ht(9.0)), 5.0)
    _assert_close(float(calc_char_ht(0.9)), 0.5)
    assert float(calc_char_ht(0.0)) == 0.0


def test_char_ht_relation_is_unit_free():
    """
    Category (b). The ``/1.8`` divisor carries no units, so the relation
    must be exactly homogeneous of degree one: scaling the input scales
    the output by the same factor. This is the property that would make
    the relation directly comparable to C++ if the intermediate were
    reachable (``gate0/03-cpp-crosswalk.md`` row 4).

    :return: None. Raises via ``assert`` on mismatch.
    """
    base = float(calc_char_ht(2.0))
    for factor in (0.5, 3.0, 1000.0):
        _assert_close(float(calc_char_ht(2.0 * factor)), base * factor)


def test_crown_scorch_array_broadcast_matches_row_by_row():
    """
    Category (a). Array evaluation must equal row-by-row scalar
    evaluation of the same inputs (oracle-independent metamorphic
    property required by the phase plan).

    :return: None. Raises via ``assert`` on mismatch.
    """
    scorch = np.array([8.0, 20.0, 10.0])
    ht = np.array([10.0, 30.0, 20.0])
    depth = np.array([4.0, 15.0, 8.0])

    vec_len, vec_cvs, vec_cls = calc_crown_length_vol_scorched(scorch, ht, depth)
    assert vec_len.shape == vec_cvs.shape == vec_cls.shape == (3,)

    for i in range(3):
        row = calc_crown_length_vol_scorched(scorch[i], ht[i], depth[i])
        _assert_close(float(vec_len[i]), float(row[0]))
        assert float(vec_cvs[i]) == pytest.approx(float(row[1]), abs=1e-12)
        assert float(vec_cls[i]) == pytest.approx(float(row[2]), abs=1e-12)


def test_crown_scorch_cvs_is_never_below_cls_for_partial_scorch():
    """
    Category (a). Oracle-independent invariant implied by the pinned
    geometry: for ``0 < B < HCR`` the volume-scorched percent
    ``100*B*(2*HCR-B)/HCR^2`` always exceeds the length-scorched percent
    ``100*B/HCR``, because ``(2*HCR-B)/HCR > 1`` there. Both percentages
    must also stay inside ``[0, 100]``.

    :return: None. Raises via ``assert`` on mismatch.
    """
    for scorch in (6.5, 7.0, 8.0, 9.0, 9.9):
        _, cvs, cls = calc_crown_length_vol_scorched(scorch, 10.0, 4.0)
        assert 0.0 <= float(cls) <= 100.0
        assert 0.0 <= float(cvs) <= 100.0
        assert float(cvs) > float(cls)


@pytest.mark.parametrize(
    ("label", "ht", "crown_ratio", "scorch_ht", "exp_len", "exp_cvs", "exp_cls"),
    _CROWN_CASES,
    ids=[case[0] for case in _CROWN_CASES],
)
def test_crown_scorch_matches_hand_derived_pinned_geometry(
        label,
        ht,
        crown_ratio,
        scorch_ht,
        exp_len,
        exp_cvs,
        exp_cls,
):
    """
    Category (b). Compare Python against values hand-derived from the
    pinned C++ crown-scorch geometry at ``fof_mrt.cpp:315-327``, using
    the ``crown_depth = ht * crown_ratio / 10`` substitution that makes
    the two algebras identical (``f_HCR`` is crown depth).

    Every expectation's derivation is written out in
    :data:`_CROWN_CASES`. No executable C++ comparison is made or
    claimed - ``f_CK``/``f_CSL`` are ``MRT_Calc`` locals absent from
    ``d_MO`` (finding F-30).

    :param label: Case identifier, used only for the test id.
    :param ht: Total tree height (m), the C++ ``f_Hgt``.
    :param crown_ratio: Crown ratio in tenths, the C++ ``f_LCR``.
    :param scorch_ht: Scorch height (m), the C++ ``f_Scorch``.
    :param exp_len: Hand-derived crown length scorched (m), the clamped
        C++ ``f_B``.
    :param exp_cvs: Hand-derived percent crown volume scorched, the C++
        ``f_CK``.
    :param exp_cls: Hand-derived percent crown length scorched, the C++
        ``f_CSL``.
    :return: None. Raises via ``assert`` on mismatch.
    """
    crown_depth = ht * (crown_ratio / 10.0)
    length, cvs, cls = calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    assert float(length) == pytest.approx(exp_len, abs=1e-12)
    assert float(cvs) == pytest.approx(exp_cvs, abs=1e-12)
    assert float(cls) == pytest.approx(exp_cls, abs=1e-12)


def test_crown_scorch_returns_a_three_tuple_of_arrays():
    """
    Category (a). The documented return contract is a 3-tuple
    ``(crown_length_scorched, cvs, cls)``; pin the arity and ordering so
    a future refactor cannot silently permute them.

    Ordering is pinned by value, not just by position: with ht=10,
    crown_depth=4 and scorch=8 the three results are numerically
    distinct (2.0 m, 75 %, 50 %), so a swap would fail.

    :return: None. Raises via ``assert`` on mismatch.
    """
    result = calc_crown_length_vol_scorched(8.0, 10.0, 4.0)
    assert isinstance(result, tuple)
    assert len(result) == 3
    _assert_close(float(result[0]), 2.0)
    _assert_close(float(result[1]), 75.0)
    _assert_close(float(result[2]), 50.0)


def test_crown_scorch_zero_crown_depth_yields_nan_where_cpp_returns_minus_one():
    """
    Category (a)/(b). Pin Gate 0 finding **F-29**, an error-semantics
    divergence from the pinned reference.

    C++ (``fof_mrt.cpp:329-333``) detects ``f_HCR == 0``, sets
    ``f_CK = f_CSL = 0``, writes the error text ``"Mortality Calculaton
    is attempting to Divide by 0"`` and **returns -1** from
    ``MRT_Calc``. Python instead divides by zero: the clamp forces the
    scorched length to 0, and both percentages become ``0/0 -> NaN``
    with a NumPy ``RuntimeWarning``.

    This test documents the current Python behaviour so any future fix is
    a visible, deliberate change. It makes no parity claim.

    :return: None. Raises via ``assert`` on mismatch.
    """
    with pytest.warns(RuntimeWarning, match="invalid value"):
        length, cvs, cls = calc_crown_length_vol_scorched(8.0, 10.0, 0.0)

    assert float(length) == 0.0
    assert np.isnan(cvs)
    assert np.isnan(cls)
