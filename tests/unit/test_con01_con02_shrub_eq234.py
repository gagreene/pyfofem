#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_con01_con02_shrub_eq234.py - regression coverage for CON-01 and CON-02,
two ``consm_shrub()`` Eq 16 / Eq 234 (SE non-Pocosin) defects found during the
post-comprehensive-test-suite publication audit.

**CON-02** (four-term ``f_WPRE``): the pinned C++ ``Equation_16``/
``Equ_234_Per`` (``reference/fofem_cpp/FOF_UNIX/fof_hsf.cpp:229-274``) compute
``f_WPRE = f_Lit + f_Duff + f_DW10 + f_DW1`` (litter + duff + 10-hr + 1-hr
dead woody fuel). Before this fix, ``consm_shrub()`` used only
``pre_ll + pre_dl`` (litter + duff), with no way for a caller to supply
10-hr/1-hr loads at all. ``consm_shrub()`` now accepts optional ``pre_dw1``/
``pre_dw10`` parameters and uses all four terms.

**CON-01** (zero-load NaN): C++ returns 0 (not NaN/an error) whenever
``f_W == 0`` (``fof_hsf.cpp:234-235``), ``f_WPRE == 0``
(``fof_hsf.cpp:238,271``), or ``f_ShrReg`` (= ``f_Shrub``) ``== 0``
(``fof_hsf.cpp:243``, and again in ``Calc_Shrub`` at
``fof_hsf.cpp:182-186``). Before this fix, a zero litter+duff (+dw10+dw1)
configuration with a nonzero shrub load produced by ``consm_shrub()`` a NaN
percent, not 0.

Test classes (per this repo's established convention):

- **(a) Python contract tests**: shape/broadcast/omitted-parameter/vectorized
  isolation behavior. No parity claim.
- **(b) source-relation cross-check**: the discriminating CON-02 case and the
  CON-01 zero-load cases are verified against an independent, directly
  hand-transcribed re-implementation of the pinned C++ formula text
  (``_eq16_fw`` / ``_eq234_fraction_reference`` below), not against
  ``consm_shrub()``'s own internals. This is NOT executable C++ parity: the
  live ``shrub_herb_eq`` harness mode's 15-column input schema
  (``SHRUB_HERB_EQ_HEADER``, ``reference/fofem_cpp_overlay/source/FOF_UNIX/
  test_harness.cpp:1096-1099``) has no ``dw10``/``dw1`` columns, so a nonzero
  ``pre_dw10``/``pre_dw1`` scenario cannot be driven through the live C++
  harness without a harness schema change, which is out of scope for this
  pass. The zero-``dw10``/``dw1`` case (every existing golden row) IS
  harness-verified: see
  ``test_consumption_parity.py (tests/unit/cpp/)::test_shrub_herb_shrub_percent_matches_cpp``,
  unchanged and still passing after this fix, since 0 + 0 added to the old
  2-term sum reproduces it exactly.
- **(c) manifested executable C++ parity**: the ``consume``-mode
  ``se-gen-entire-m050`` golden row (below) DOES carry nonzero
  ``dw10_tac``/``dw1_tac`` (0.5 each, per the shared Phase 2 canonical base
  row this scenario inherits), so it genuinely discriminates the CON-02 fix
  through the real, compiled C++ pipeline. This is not routed through the
  ``consume_expanded_matrix`` tolerance-policy registry (``EXPANDED_MATRIX_ROUTE_KEYS``/
  ``expanded_matrix_policy_keys``): that registry's own pre-existing ``shrub`` route
  is deliberately ``"status": "unverified"`` because the golden's ``ShrCon``
  is a LOAD while ``consm_shrub`` returns a PERCENT, and converting between
  them in general requires re-deriving C++'s own clamp order
  (``fof_hsf.cpp:177-186``, load clamped to ``[0, f_Shrub]`` before percent
  is derived). This module's one comparison below is a narrow, scenario-
  specific check (this exact case's load never reaches that clamp, since
  ``ShrCon (0.910408) < ShrPre (1.0)``, confirmed by direct inspection of the
  golden row before writing the assertion) and does not generalize that
  "unverified" status to a "verified" one - the ``consume_expanded_matrix.shrub`` policy
  entry is intentionally left unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem.components._component_helpers import _KGPM2_TO_TPAC
from pyfofem.components.consumption_calcs import consm_shrub
from pyfofem.components.emission_pipeline import compute_pre_burnup_consumption
from tests.cpp_parity_live._expanded_matrix_contract import (
    CONSUME_INDEX,
    CONSUME_SCENARIOS,
    golden_rows_by_case,
    expanded_matrix_tolerance,
    require_golden_tree,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

# Fail CLOSED, not open: a missing/incomplete committed Phase 4 golden
# dataset is a repository defect for the one (c) test below, never a silent
# skip - see require_golden_tree().
require_golden_tree()

ATOL = 1e-9
#: Reused, not fabricated: the same atol Phase 4's own
#: ``shrub_herb_eq_expanded_matrix.shrub`` route already applies to a Calc_Shrub-derived
#: percent (test_phase4_consumption_parity.ATOL_SHRUB_PERCENT).
ATOL_SHRUB_PERCENT = expanded_matrix_tolerance("shrub_herb_eq", "shrub")[0]


def _consume_input_value(overrides, column):
    """
    Return one ``consume``-mode input field for a scenario after its
    overrides are applied to the shared canonical base row. Mirrors
    ``test_phase4_consumption_parity._consume_input`` (kept as a small local
    copy rather than a cross-module private import).

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :return: The raw string value that was written to the golden's input CSV.
    """
    row = list(MODES["consume"]["row"])
    for key, value in overrides.items():
        row[CONSUME_INDEX[key]] = value
    return row[CONSUME_INDEX[column]]


def _eq16_fw(wpre: float, duff_moist: float) -> float:
    """
    Direct transcription of C++ ``Equation_16`` (``fof_hsf.cpp:266-275``).

    :param wpre: ``f_WPRE`` (litter + duff + dw10 + dw1).
    :param duff_moist: ``f_MoistDuff``.
    :return: ``f_W``, or 0 when ``wpre == 0`` (``fof_hsf.cpp:271-272``).
    """
    if wpre == 0:
        return 0.0
    return 3.4958 + (0.3833 * wpre) - (0.0237 * duff_moist) - (5.6075 / wpre)


def _eq234_fraction_reference(
        wpre: float, shrub: float, duff_moist: float,
) -> float:
    """
    Direct transcription of C++ ``Equ_234_Per`` (``fof_hsf.cpp:229-258``).

    :param wpre: ``f_WPRE`` (litter + duff + dw10 + dw1).
    :param shrub: ``f_ShrReg`` (= ``f_Shrub``).
    :param duff_moist: ``f_MoistDuff``.
    :return: the raw fraction ``Calc_Shrub`` multiplies directly by
        ``f_Shrub`` (fof_hsf.cpp:207-209) - not itself a 0-100 percent
        despite the ``[0, 100]`` clamp bound (fof_hsf.cpp:253-255).
    """
    f_w = _eq16_fw(wpre, duff_moist)
    if f_w == 0:
        return 0.0
    if wpre == 0:
        return 0.0
    if shrub == 0:
        return 0.0
    f = (
        (3.2484 + (0.4322 * wpre) + (0.6765 * shrub) -
         (0.0276 * duff_moist) - (5.0796 / wpre)) - f_w
    ) / shrub
    return max(0.0, min(100.0, f))


def test_con01_mixed_valid_and_zero_load_array_does_not_cross_contaminate():
    """
    (a)+(b) Vectorized mixed-cell behavior: a zero-litter+duff cell and a
    zero-shrub cell must each return 0.0 without affecting a normal nonzero
    control cell computed in the same array call.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll = np.array([0.0, 1.0, 1.0])
    pre_dl = np.array([0.0, 0.8, 0.8])
    pre_sl = np.array([2.0, 2.0, 0.0])
    duff_moist = np.array([40.0, 40.0, 40.0])

    result = consm_shrub(
        np.array(['SouthEast'] * 3), np.array(['NA'] * 3), pre_sl,
        season=np.array(['Summer'] * 3), pre_ll=pre_ll, pre_dl=pre_dl,
        duff_moist=duff_moist, units='Imperial',
    )
    assert not np.any(np.isnan(result))
    assert float(result[0]) == pytest.approx(0.0, abs=ATOL)  # zero litter+duff
    assert float(result[2]) == pytest.approx(0.0, abs=ATOL)  # zero shrub

    control_expected_pct = 100.0 * _eq234_fraction_reference(
        pre_ll[1] + pre_dl[1], pre_sl[1], duff_moist[1],
    )
    assert float(result[1]) == pytest.approx(control_expected_pct, abs=ATOL)


def test_con01_negative_woody_load_is_not_silently_zeroed():
    """
    (a) A genuinely invalid (physically impossible) negative litter+duff sum
    must NOT be silently converted to a 0 result by the zero-load guard -
    only an exact ``== 0`` sum triggers it. This must still propagate as NaN
    (the pre-existing, unrelated behavior for invalid inputs), confirming the
    CON-01 fix is scoped to the exact zero condition only.

    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = consm_shrub(
        'SouthEast', 'NA', 2.0,
        season='Summer', pre_ll=-5.0, pre_dl=0.0, duff_moist=40.0,
        units='Imperial',
    )
    assert np.isnan(float(actual))


def test_con01_zero_litter_and_duff_returns_zero_not_nan():
    """
    (b) CON-01: zero litter + duff (+ zero dw10/dw1) with a nonzero shrub
    load must return 0.0, matching C++'s ``f_WPRE == 0`` guard
    (``fof_hsf.cpp:238,271``), not NaN.

    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = consm_shrub(
        'SouthEast', 'NA', 2.0,
        season='Summer', pre_ll=0.0, pre_dl=0.0, duff_moist=40.0,
        units='Imperial',
    )
    assert not np.isnan(float(actual))
    assert float(actual) == pytest.approx(0.0, abs=ATOL)


def test_con01_zero_shrub_returns_zero_not_nan():
    """
    (b) CON-01: zero shrub load with nonzero litter/duff must return 0.0,
    matching C++'s ``f_ShrReg == 0`` guard (``fof_hsf.cpp:243``) and
    ``Calc_Shrub``'s own zero-shrub percent guard (``fof_hsf.cpp:182-186``),
    not NaN.

    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = consm_shrub(
        'SouthEast', 'NA', 0.0,
        season='Summer', pre_ll=1.0, pre_dl=0.8, duff_moist=40.0,
        units='Imperial',
    )
    assert not np.isnan(float(actual))
    assert float(actual) == pytest.approx(0.0, abs=ATOL)


def test_con01_equation_16_exact_zero_returns_zero_not_nan():
    """
    (b) A nonzero ``f_WPRE`` whose Equation 16 result is exactly zero must
    take C++'s separate ``f_W == 0`` guard, rather than continue into
    Eq. 234. This is deliberately distinct from the ``f_WPRE == 0`` case.

    :return: None. Raises via ``assert`` on mismatch.
    """
    wpre = 3.0
    duff_moist = (3.4958 + (0.3833 * wpre) - (5.6075 / wpre)) / 0.0237
    assert _eq16_fw(wpre, duff_moist) == 0.0

    actual = consm_shrub(
        'SouthEast', 'NA', 2.0,
        season='Summer', pre_ll=1.0, pre_dl=2.0, duff_moist=duff_moist,
        units='Imperial',
    )
    assert float(actual) == pytest.approx(0.0, abs=ATOL)


def test_consume_se_gen_entire_m050_shrub_matches_cpp_with_nonzero_dw_terms():
    """
    (c) Full-pipeline, live-C++-golden discriminating evidence for CON-02:
    the ``consume`` mode's ``se-gen-entire-m050`` scenario (SE non-Pocosin
    Eq 234, "BR-SHR-234") inherits ``dw10_tac=0.5``/``dw1_tac=0.5`` from the
    shared Phase 2 canonical base row - the pre-fix 2-term formula
    (litter=2.0 + duff=10.0 = 12.0) gives 86.489%; the correct 4-term
    ``f_WPRE`` (+ dw10 + dw1 = 13.0) matches the real compiled C++ oracle's
    ``ShrCon``/``ShrPre`` ratio (0.910408 / 1.0 = 91.0408%) to within
    ``ATOL_SHRUB_PERCENT``.

    ``shrub_herb_eq``'s own harness mode cannot exercise this: its 15-column
    input schema (``SHRUB_HERB_EQ_HEADER``) has no ``dw10``/``dw1`` columns
    at all, so this ``consume``-mode scenario is the only live C++ evidence
    available for the nonzero-dw10/dw1 case.

    :return: None. Raises via ``assert`` on mismatch.
    """
    overrides = next(
        o for c, o, _b in CONSUME_SCENARIOS if c == "se-gen-entire-m050"
    )
    row = golden_rows_by_case("consume", "_summary")["se-gen-entire-m050"]
    shr_pre = float(row["ShrPre"])
    shr_con = float(row["ShrCon"])
    # This scenario's own load never reaches C++'s post-hoc
    # [0, f_Shrub] clamp (fof_hsf.cpp:177-186), so a plain ratio is a valid
    # percent here; see the module docstring for why this is not generalized.
    assert shr_con < shr_pre
    expected_pct = 100.0 * shr_con / shr_pre

    dw10 = float(_consume_input_value(overrides, "dw10_tac"))
    dw1 = float(_consume_input_value(overrides, "dw1_tac"))
    assert dw10 > 0.0 and dw1 > 0.0  # confirms this case truly discriminates

    actual = consm_shrub(
        _consume_input_value(overrides, "region"),
        _consume_input_value(overrides, "cover_group") or "NA",
        float(_consume_input_value(overrides, "shrub_tac")),
        season=_consume_input_value(overrides, "season") or None,
        pre_ll=float(_consume_input_value(overrides, "litter_tac")),
        pre_dl=float(_consume_input_value(overrides, "duff_tac")),
        pre_dw10=dw10,
        pre_dw1=dw1,
        duff_moist=float(_consume_input_value(overrides, "duff_moist_pct")),
        units="Imperial",
    )
    assert float(actual) == pytest.approx(expected_pct, abs=ATOL_SHRUB_PERCENT)


def test_eq234_control_case_nonzero_matches_reference_formula():
    """
    (b) A normal, nonzero SE non-Pocosin Eq 234 case (no zero-load guard
    exercised) must match the independent reference formula, with
    ``pre_dw1``/``pre_dw10`` omitted (equivalent to 0).

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, pre_sl, duff_moist = 1.0, 0.8, 2.0, 40.0
    expected_pct = 100.0 * _eq234_fraction_reference(
        pre_ll + pre_dl, pre_sl, duff_moist,
    )
    actual = consm_shrub(
        'SouthEast', 'NA', pre_sl,
        season='Summer', pre_ll=pre_ll, pre_dl=pre_dl, duff_moist=duff_moist,
        units='Imperial',
    )
    assert float(actual) == pytest.approx(expected_pct, abs=ATOL)


def test_eq234_discriminating_case_nonzero_dw1_dw10_matches_four_term_reference():
    """
    (b) CON-02 discriminating case: nonzero ``pre_dw1``/``pre_dw10`` must
    change the result relative to the old 2-term formula, and the new result
    must match the 4-term reference formula exactly.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, pre_dw10, pre_dw1 = 1.0, 0.8, 0.5, 0.3
    pre_sl, duff_moist = 2.0, 40.0

    wpre_four_term = pre_ll + pre_dl + pre_dw10 + pre_dw1
    wpre_two_term = pre_ll + pre_dl
    expected_pct_four_term = 100.0 * _eq234_fraction_reference(
        wpre_four_term, pre_sl, duff_moist,
    )
    expected_pct_two_term = 100.0 * _eq234_fraction_reference(
        wpre_two_term, pre_sl, duff_moist,
    )
    # The two reference values must genuinely differ, or this case would not
    # discriminate the old (2-term) defect from the fix.
    assert abs(expected_pct_four_term - expected_pct_two_term) > 1.0

    actual = consm_shrub(
        'SouthEast', 'NA', pre_sl,
        season='Summer', pre_ll=pre_ll, pre_dl=pre_dl,
        pre_dw10=pre_dw10, pre_dw1=pre_dw1, duff_moist=duff_moist,
        units='Imperial',
    )
    assert float(actual) == pytest.approx(expected_pct_four_term, abs=ATOL)
    assert float(actual) != pytest.approx(expected_pct_two_term, abs=1e-3)


def test_eq234_omitted_dw_params_equal_explicit_zero():
    """
    (a) Omitting ``pre_dw1``/``pre_dw10`` must be exactly equivalent to
    passing 0 explicitly (backward-compatible default).

    :return: None. Raises via ``assert`` on mismatch.
    """
    kwargs = dict(
        reg='SouthEast', cvr_grp='NA', pre_sl=2.0, season='Summer',
        pre_ll=1.0, pre_dl=0.8, duff_moist=40.0, units='Imperial',
    )
    omitted = consm_shrub(**kwargs)
    explicit_zero = consm_shrub(**kwargs, pre_dw1=0.0, pre_dw10=0.0)
    assert float(omitted) == pytest.approx(float(explicit_zero), abs=ATOL)


def test_eq234_omitted_dw_params_match_pre_fix_two_term_behavior():
    """
    (a) Omitting the new parameters must reproduce the exact pre-fix 2-term
    result (backward compatibility for existing callers that never supply
    dw10/dw1), cross-checked against the reference formula with a 2-term
    ``wpre``.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, pre_sl, duff_moist = 1.0, 0.8, 2.0, 40.0
    expected_pct_two_term = 100.0 * _eq234_fraction_reference(
        pre_ll + pre_dl, pre_sl, duff_moist,
    )
    actual = consm_shrub(
        'SouthEast', 'NA', pre_sl,
        season='Summer', pre_ll=pre_ll, pre_dl=pre_dl, duff_moist=duff_moist,
        units='Imperial',
    )
    assert float(actual) == pytest.approx(expected_pct_two_term, abs=ATOL)


def test_eq234_si_imperial_equivalence_with_dw_terms():
    """
    (a) SI (kg/m2) and Imperial (T/ac) inputs carrying the same physical
    loads, including nonzero ``pre_dw1``/``pre_dw10``, must produce the same
    percent-consumed result. Output stays a percent in both cases.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll_tac, pre_dl_tac, pre_dw10_tac, pre_dw1_tac = 1.0, 0.8, 0.5, 0.3
    pre_sl_tac, duff_moist = 2.0, 40.0
    kg_m2_per_tac = _KGPM2_TO_TPAC

    imperial = consm_shrub(
        'SouthEast', 'NA', pre_sl_tac,
        season='Summer', pre_ll=pre_ll_tac, pre_dl=pre_dl_tac,
        pre_dw10=pre_dw10_tac, pre_dw1=pre_dw1_tac, duff_moist=duff_moist,
        units='Imperial',
    )
    si = consm_shrub(
        'SouthEast', 'NA', pre_sl_tac / kg_m2_per_tac,
        season='Summer', pre_ll=pre_ll_tac / kg_m2_per_tac,
        pre_dl=pre_dl_tac / kg_m2_per_tac,
        pre_dw10=pre_dw10_tac / kg_m2_per_tac,
        pre_dw1=pre_dw1_tac / kg_m2_per_tac, duff_moist=duff_moist,
        units='SI',
    )
    assert float(si) == pytest.approx(float(imperial), abs=1e-6)


def test_pre_burnup_facade_shrub_matches_direct_helper_for_eq234_cells():
    """
    (a) The pre-burnup façade must use the direct four-term helper result for
    Southeast non-Pocosin cells, including a zero-shrub guard and nonzero
    1-hr/10-hr woody loads, without a second formula implementation.

    :return: None. Raises via ``assert`` on mismatch.
    """
    lit = np.array([0.5, 1.0, 1.0])
    duff = np.array([0.5, 0.8, 0.8])
    shrub = np.array([0.0, 2.0, 3.0])
    dw10 = np.array([0.5, 0.5, 0.5])
    dw1 = np.array([0.3, 0.3, 0.3])
    duff_moist = np.array([40.0, 40.0, 40.0])

    direct_pct = np.asarray(
        consm_shrub(
            np.array(['SouthEast'] * 3), np.array(['NA'] * 3), shrub,
            season=np.array(['Summer'] * 3), pre_ll=lit, pre_dl=duff,
            pre_dw10=dw10, pre_dw1=dw1, duff_moist=duff_moist,
            units='Imperial',
        ),
        dtype=float,
    )
    result = compute_pre_burnup_consumption(
        lit_a=lit, l_m_a=np.array([10.0] * 3),
        cvr_a=np.array(['NA'] * 3), reg_a=np.array(['SouthEast'] * 3),
        units='Imperial', her_a=np.zeros(3), sea_a=np.array(['Summer'] * 3),
        shr_a=shrub, pcb_a=np.zeros(3), fol_a=np.zeros(3), bra_a=np.zeros(3),
        ft_a=np.array(['Natural'] * 3), duf_m_a=duff_moist, duf_a=duff,
        duf_dep_a=np.zeros(3), dw10_a=dw10, dw1_a=dw1,
        dw1k_m_a=np.array([20.0] * 3),
    )
    np.testing.assert_allclose(
        result['shr_con_arr'], shrub * direct_pct / 100.0, atol=ATOL,
    )
