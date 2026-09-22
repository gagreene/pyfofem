#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_f39_coastal_plain.py - implementation of F-39's Coastal Plain
sub-finding: ``consm_litter()``/``consm_duff()``/``consm_mineral_soil()``
gained a real Coastal Plain (``cvr_grp`` ``'CP'``/``'CoastPlain'``, a
SouthEast COVER GROUP, not a region) forest-floor route, matching the
pinned C++ ``Equ_CP_Per``/``Equ_CP_Red``/``Equ_CP_MSE``
(``reference/fofem_cpp/FOF_UNIX/fof_duf.cpp:1115-1249``, equation IDs
30/31/32) and ``_CalcCP_Lit``/``_ChkLitMoist`` (``fof_hsf.cpp:83-127,
614-626``).

Test classes (per this repo's established convention):

- **(a) Python contract tests**: alias discrimination, argument
  validation, region guards, scalar/array shape, no-double-counting.
- **(b) source-relation cross-check**: hand-derived-from-C++-source
  reference values (``_reference_forest_floor`` below, an INDEPENDENT
  transcription of ``Equ_CP_Per``, not calling the production
  ``_coastal_plain_forest_floor`` helper), including the exact worked
  example from the C++ source's own docstring comment
  (``fof_duf.cpp:1159-1165``: "prefire Lit=5 and Duf=5; Total
  Consumed=2.5 -> 50% litter, no duff").
- **(c) manifested executable C++ parity**: the ``consume``-mode
  ``se-cp-entire-m050`` golden scenario (already committed under
  ``tests/test_data/test_golden_output/phase4/consume/`` since the
  original Gate 0 executed-oracle pass -- no new golden generation was
  needed for this pass) already carries a real ``cvr_grp='CoastPlain'``
  row with nonzero litter/duff/depth/moisture, discriminating Coastal
  Plain from both Pocosin and ordinary SouthEast. See
  ``test_phase4_consumption_parity.py::test_consume_coastal_plain_litter_matches_cpp``
  for the direct golden-row comparison this module does not duplicate;
  this module's own (c) tests re-derive the same golden row's inputs to
  additionally prove depth/MSE/equation-ID agreement in one place.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem.components.consumption_calcs import (
    MG_HECTARE_PER_TON_ACRE,
    consm_duff,
    consm_litter,
    consm_mineral_soil,
)
from pyfofem.pyfofem import run_fofem_emissions
from tests.cpp_parity_live._phase4_contract import (
    CONSUME_INDEX,
    CONSUME_SCENARIOS,
    golden_rows_by_case,
    require_golden_tree,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

# Fail CLOSED, not open: a missing/incomplete committed Phase 4 golden
# dataset is a repository defect for the (c) tests below, never a silent
# skip - see require_golden_tree().
require_golden_tree()

ATOL = 1e-6


def _consume_input_value(overrides, column):
    """
    Return one ``consume``-mode input field for a scenario after its
    overrides are applied to the shared canonical base row.

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :return: The raw string value that was written to the golden's input CSV.
    """
    row = list(MODES["consume"]["row"])
    for key, value in overrides.items():
        row[CONSUME_INDEX[key]] = value
    return row[CONSUME_INDEX[column]]


def _reference_forest_floor(pre_ll: float, pre_dl: float, l_moist: float):
    """
    Independent hand transcription of C++ ``Equ_CP_Per``
    (``fof_duf.cpp:1171-1225``), NOT calling the production
    ``_coastal_plain_forest_floor`` helper.

    :param pre_ll: Pre-fire litter load (T/ac).
    :param pre_dl: Pre-fire duff load (T/ac).
    :param l_moist: Litter moisture content (%).
    :return: ``(litter_pct, duff_pct)``, each in ``[0, 100]``.
    """
    ll = pre_ll * 0.907184
    dl = pre_dl * 0.907184
    x1_load = ll + dl
    if x1_load <= 0:
        return 0.0, 0.0
    tc = -3.893 + (0.944 * x1_load) - (0.078 * l_moist)
    if tc > x1_load:
        tc = x1_load
    if tc < 0:
        tc = 0.0
    dc = tc - ll
    if dc <= 0:
        dc = 0.0
    lc = tc - dc
    duff_pct = 0.0 if dl <= 0 else 100.0 * (dc / dl)
    litter_pct = 0.0 if ll <= 0 else 100.0 * (lc / ll)
    return min(100.0, litter_pct), min(100.0, duff_pct)


def test_consm_duff_coastal_plain_case_matches_cpp_worked_example():
    """
    (b) The exact worked example from C++'s own ``Equ_CP_Per`` docstring
    comment (``fof_duf.cpp:1159-1165``): "prefire Lit=5 and Duf=5; Total
    Consumed=2.5 -> 50 percent of the litter is consumed, but no Duff."
    The comment's Lit/Duf/Tc=5/5/2.5 are already-converted (Mg/ha) example
    numbers, so *pre_ll*/*pre_dl* here are chosen (T/ac) so that
    ``* MG_HECTARE_PER_TON_ACRE`` reproduces exactly 5.0 Mg/ha each; the
    litter moisture is then hand-solved so Eq 30's own total-consumed
    expression evaluates to exactly 2.5 Mg/ha.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll = pre_dl = 5.0 / MG_HECTARE_PER_TON_ACRE
    x1_load = 10.0  # (pre_ll + pre_dl) * MG_HECTARE_PER_TON_ACRE, exactly
    # Solve -3.893 + 0.944*x1_load - 0.078*l_moist = 2.5 for l_moist.
    l_moist = (-3.893 + 0.944 * x1_load - 2.5) / 0.078
    lit_pct, duf_pct = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert lit_pct == pytest.approx(50.0, abs=1e-6)
    assert duf_pct == pytest.approx(0.0, abs=1e-6)

    res = consm_duff(
        pre_dl, 50.0, reg="SouthEast", cvr_grp="CoastPlain", d_pre=1.0,
        pre_ll=pre_ll, l_moist=l_moist, units="Imperial",
    )
    assert res["pdc"] == pytest.approx(duf_pct, abs=ATOL)
    lit_con = consm_litter(
        pre_ll, l_moist, cvr_grp="CoastPlain", reg="SouthEast",
        pre_dl=pre_dl, units="Imperial",
    )
    assert float(lit_con) == pytest.approx(pre_ll * (lit_pct / 100.0), abs=ATOL)


def test_consm_duff_coastal_plain_duff_moist_le_10_overrides_percent_and_depth():
    """
    (a)+(b) The pre-existing global ``duff_moist <= 10`` override (100%
    duff consumed, full duff-depth consumed) still applies on top of Eq 30
    for Coastal Plain, exactly as for every other SouthEast branch -- but
    litter percent is UNAFFECTED (C++'s override touches only
    ``f_Per``/``f_Red``, never ``f_PerLit``, fof_duf.cpp:363-368).

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, l_moist = 2.0, 10.0, 40.0
    _lit_pct_ref, duf_pct_ref = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert duf_pct_ref != pytest.approx(100.0)  # sanity: not already 100%

    res = consm_duff(
        pre_dl, 10.0, reg="SouthEast", cvr_grp="CP", d_pre=2.0,
        pre_ll=pre_ll, l_moist=l_moist, units="Imperial",
    )
    assert res["pdc"] == pytest.approx(100.0, abs=ATOL)
    assert res["ddc"] == pytest.approx(2.0, abs=ATOL)  # full depth consumed

    lit_con = consm_litter(
        pre_ll, l_moist, cvr_grp="CP", reg="SouthEast", pre_dl=pre_dl,
        units="Imperial",
    )
    expected_lit_pct, _ = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert float(lit_con) == pytest.approx(pre_ll * (expected_lit_pct / 100.0), abs=ATOL)


def test_consm_duff_coastal_plain_equation_ids_are_30_31_32():
    """
    (c) The full ``run_fofem_emissions()`` facade reports equation IDs
    30 (litter+duff percent), 31 (depth), and 32 (mineral soil) for a
    Coastal Plain cell -- matching C++'s ``e_CP_PerEq``/``e_CP_RedEq``/
    ``e_CP_MSEEq`` (``fof_duf.h:11-13``).

    :return: None. Raises via ``assert`` on mismatch.
    """
    res = run_fofem_emissions(
        litter=2.0, duff=10.0, duff_depth=2.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region="SouthEast", cvr_grp="CoastPlain", season="Summer",
        fuel_category="Natural", duff_moist=50.0, l_moist=40.0,
        dw10_moist=15.0, dw1000_moist=25.0, units="Imperial",
        use_burnup=False,
    )
    assert res["Lit-Equ"] == 30
    assert res["DufCon-Equ"] == 30
    assert res["DufRed-Equ"] == 31
    assert res["MSE-Equ"] == 32


def test_consm_duff_coastal_plain_matches_golden_scenario_full_output_set():
    """
    (c) The committed ``se-cp-entire-m050`` golden scenario, compared on
    every applicable scientific field at once (litter, duff percent/load,
    depth, and mineral soil), not just one summary field.

    :return: None. Raises via ``assert`` on mismatch.
    """
    overrides = next(
        o for c, o, _b in CONSUME_SCENARIOS if c == "se-cp-entire-m050"
    )
    row = golden_rows_by_case("consume", "_summary")["se-cp-entire-m050"]

    litter_tac = float(_consume_input_value(overrides, "litter_tac"))
    duff_tac = float(_consume_input_value(overrides, "duff_tac"))
    duff_depth_in = float(_consume_input_value(overrides, "duff_depth_in"))
    duff_moist_pct = float(_consume_input_value(overrides, "duff_moist_pct"))
    litter_moist_pct = float(_consume_input_value(overrides, "litter_moist_pct"))

    lit_con = consm_litter(
        litter_tac, litter_moist_pct, cvr_grp="CoastPlain", reg="SouthEast",
        pre_dl=duff_tac, units="Imperial",
    )
    assert float(lit_con) == pytest.approx(float(row["LitCon"]), abs=1e-5)

    duf_res = consm_duff(
        duff_tac, duff_moist_pct, reg="SouthEast", cvr_grp="CoastPlain",
        d_pre=duff_depth_in, pre_ll=litter_tac, l_moist=litter_moist_pct,
        units="Imperial",
    )
    assert duf_res["pdc"] == pytest.approx(float(row["DufPer"]), abs=1e-5)
    assert duf_res["ddc"] == pytest.approx(float(row["DufDepCon"]), abs=1e-5)

    mse = consm_mineral_soil(
        "SouthEast", "CoastPlain", "Natural", duff_moist_pct, "edm",
        duff_load=duff_tac,
    )
    assert float(mse) == pytest.approx(float(row["MSE"]), abs=1e-5)


def test_consm_duff_coastal_plain_requires_litter_and_moisture():
    """
    (a) Coastal Plain requires both ``pre_ll`` and ``l_moist``.

    :return: None. Raises via ``assert`` on the expected exception.
    """
    with pytest.raises(ValueError, match="pre_ll"):
        consm_duff(
            10.0, 50.0, reg="SouthEast", cvr_grp="CP", l_moist=40.0,
        )
    with pytest.raises(ValueError, match="pre_ll"):
        consm_duff(
            10.0, 50.0, reg="SouthEast", cvr_grp="CP", pre_ll=2.0,
        )


def test_consm_duff_coastal_plain_wrong_region_raises():
    """
    (a) Coastal Plain is a SouthEast cover group, not a region: any other
    ``reg`` fails loudly rather than silently applying an unrelated
    regional equation.

    :return: None. Raises via ``assert`` on the expected exception.
    """
    for cvr in ("CP", "CoastPlain", "cp", "COASTPLAIN"):
        with pytest.raises(ValueError, match="SouthEast"):
            consm_duff(
                10.0, 50.0, reg="InteriorWest", cvr_grp=cvr,
                pre_ll=2.0, l_moist=40.0,
            )


def test_consm_duff_pocosin_pc_still_routes_pocosin_not_coastal_plain():
    """
    (a) ``PC`` (Pocosin) must be unaffected: it still routes to Eq 20, and
    supplying no litter/moisture inputs at all must not raise (Eq 20 never
    needs them).

    :return: None. Raises via ``assert`` on mismatch.
    """
    res_pc = consm_duff(
        10.0, 50.0, reg="SouthEast", cvr_grp="PC", mc_lyr1=50.0, d_pre=2.0,
        units="Imperial",
    )
    res_cp = consm_duff(
        10.0, 50.0, reg="SouthEast", cvr_grp="CP", d_pre=2.0,
        pre_ll=2.0, l_moist=40.0, units="Imperial",
    )
    assert res_pc["pdc"] != pytest.approx(res_cp["pdc"], abs=1e-3)


def test_consm_litter_coastal_plain_duff_only_load():
    """
    (b) Duff-only load (zero litter): litter percent is 0; the total
    consumed budget goes entirely to duff.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, l_moist = 0.0, 5.0, 1.0
    lit_pct_ref, duf_pct_ref = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert lit_pct_ref == pytest.approx(0.0, abs=ATOL)
    assert duf_pct_ref > 0.0

    lit_con = consm_litter(
        pre_ll, l_moist, cvr_grp="CoastPlain", reg="SouthEast", pre_dl=pre_dl,
        units="Imperial",
    )
    assert float(lit_con) == pytest.approx(0.0, abs=ATOL)
    duf_res = consm_duff(
        pre_dl, 50.0, reg="SouthEast", cvr_grp="CoastPlain", d_pre=1.0,
        pre_ll=pre_ll, l_moist=l_moist, units="Imperial",
    )
    assert duf_res["pdc"] == pytest.approx(duf_pct_ref, abs=ATOL)


def test_consm_litter_coastal_plain_litter_only_load():
    """
    (b) Litter-only load (zero duff): all consumption goes to litter,
    duff percent is 0.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, l_moist = 5.0, 0.0, 1.0
    lit_pct_ref, duf_pct_ref = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert duf_pct_ref == pytest.approx(0.0, abs=ATOL)
    assert lit_pct_ref > 0.0

    lit_con = consm_litter(
        pre_ll, l_moist, cvr_grp="CP", reg="SouthEast", pre_dl=pre_dl,
        units="Imperial",
    )
    assert float(lit_con) == pytest.approx(pre_ll * (lit_pct_ref / 100.0), abs=ATOL)
    duf_res = consm_duff(
        pre_dl, 50.0, reg="SouthEast", cvr_grp="CP", d_pre=1.0,
        pre_ll=pre_ll, l_moist=l_moist, units="Imperial",
    )
    assert duf_res["pdc"] == pytest.approx(0.0, abs=ATOL)


def test_consm_litter_coastal_plain_moisture_boundaries():
    """
    (a) ``l_moist`` is enforced to C++'s inclusive ``[1.0, 100.0]`` range:
    exactly 1 and exactly 100 are accepted; values just outside are
    rejected.

    :return: None. Raises via ``assert``/the expected exception.
    """
    for ok_val in (1.0, 100.0):
        value = consm_litter(
            2.0, ok_val, cvr_grp="CP", reg="SouthEast", pre_dl=10.0,
            units="Imperial",
        )
        assert np.isfinite(float(value))

    for bad_val in (0.999999, 100.000001, 0.0, -5.0, 200.0):
        with pytest.raises(ValueError, match="Litter Moisture"):
            consm_litter(
                2.0, bad_val, cvr_grp="CP", reg="SouthEast", pre_dl=10.0,
                units="Imperial",
            )


def test_consm_litter_coastal_plain_requires_duff_load():
    """
    (a) Coastal Plain litter consumption requires ``pre_dl`` (needed for
    Eq 30's combined litter+duff total).

    :return: None. Raises via ``assert`` on the expected exception.
    """
    with pytest.raises(ValueError, match="pre_dl"):
        consm_litter(2.0, 40.0, cvr_grp="CoastPlain", reg="SouthEast")


def test_consm_litter_coastal_plain_scalar_and_array_agree():
    """
    (a) Scalar and length-1-array calls must agree, and a length-3 mixed
    array (one Coastal Plain cell, two ordinary SouthEast cells) must
    isolate the Coastal Plain cell's result without disturbing the others.

    :return: None. Raises via ``assert`` on mismatch.
    """
    scalar_val = consm_litter(
        2.0, 40.0, cvr_grp="CoastPlain", reg="SouthEast", pre_dl=10.0,
        units="Imperial",
    )
    array_val = consm_litter(
        np.array([2.0]), np.array([40.0]), cvr_grp=np.array(["CoastPlain"]),
        reg=np.array(["SouthEast"]), pre_dl=np.array([10.0]),
        units="Imperial",
    )
    assert isinstance(scalar_val, float)
    assert float(array_val[0]) == pytest.approx(scalar_val, abs=ATOL)

    mixed = consm_litter(
        np.array([2.0, 3.0, 4.0]),
        np.array([40.0, 20.0, 20.0]),
        cvr_grp=np.array(["CoastPlain", "", ""]),
        reg=np.array(["SouthEast", "SouthEast", "SouthEast"]),
        pre_dl=np.array([10.0, 0.0, 0.0]),
        units="Imperial",
    )
    assert float(mixed[0]) == pytest.approx(scalar_val, abs=ATOL)
    assert float(mixed[1]) == pytest.approx(3.0 * 0.8, abs=ATOL)  # Eq 998
    assert float(mixed[2]) == pytest.approx(4.0 * 0.8, abs=ATOL)  # Eq 998


def test_consm_litter_coastal_plain_si_and_imperial_agree():
    """
    (a) SI (kg/m²) and Imperial (T/ac) inputs carrying the same physical
    loads must produce the same litter-consumed load, correctly converted.

    :return: None. Raises via ``assert`` on mismatch.
    """
    kg_m2_per_tac = 4.4609
    pre_ll_tac, pre_dl_tac, l_moist = 2.0, 10.0, 40.0

    imperial = consm_litter(
        pre_ll_tac, l_moist, cvr_grp="CoastPlain", reg="SouthEast",
        pre_dl=pre_dl_tac, units="Imperial",
    )
    si = consm_litter(
        pre_ll_tac / kg_m2_per_tac, l_moist, cvr_grp="CoastPlain",
        reg="SouthEast", pre_dl=pre_dl_tac / kg_m2_per_tac, units="SI",
    )
    assert float(si) * kg_m2_per_tac == pytest.approx(float(imperial), abs=1e-6)


def test_consm_litter_coastal_plain_total_consumed_lower_clamp():
    """
    (b) Total-consumed lower clamp (Tc forced to 0): a small load with
    litter moisture at the maximum valid value (100%) drives the raw Eq 30
    total negative, clamped to 0 -- both percents are 0, not negative.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pre_ll, pre_dl, l_moist = 1.0, 1.0, 100.0
    lit_pct_ref, duf_pct_ref = _reference_forest_floor(pre_ll, pre_dl, l_moist)
    assert lit_pct_ref == pytest.approx(0.0, abs=ATOL)
    assert duf_pct_ref == pytest.approx(0.0, abs=ATOL)

    lit_con = consm_litter(
        pre_ll, l_moist, cvr_grp="CP", reg="SouthEast", pre_dl=pre_dl,
        units="Imperial",
    )
    assert float(lit_con) == pytest.approx(0.0, abs=ATOL)


def test_consm_litter_coastal_plain_zero_combined_load_is_zero_not_nan():
    """
    (b) A zero combined litter+duff load returns exactly 0 (matching
    C++'s ``if ((Lit+Duff)<=0) return;`` early exit), never NaN.

    :return: None. Raises via ``assert`` on mismatch.
    """
    value = consm_litter(
        0.0, 50.0, cvr_grp="CoastPlain", reg="SouthEast", pre_dl=0.0,
        units="Imperial",
    )
    assert not np.isnan(float(value))
    assert float(value) == pytest.approx(0.0, abs=ATOL)

    duf_res = consm_duff(
        0.0, 50.0, reg="SouthEast", cvr_grp="CP", d_pre=0.0,
        pre_ll=0.0, l_moist=50.0, units="Imperial",
    )
    assert not np.isnan(duf_res["pdc"])
    assert duf_res["pdc"] == pytest.approx(0.0, abs=ATOL)


def test_consm_mineral_soil_coastal_plain_duff_le_0_overrides_to_100():
    """
    (b) C++'s global ``if (f_Duff <= 0) f_MSEPer = 100.0`` override
    (``fof_duf.cpp:377-378``) applies on top of Coastal Plain's own flat
    5% (``Equ_CP_MSE``).

    :return: None. Raises via ``assert`` on mismatch.
    """
    positive = consm_mineral_soil(
        "SouthEast", "CoastPlain", "Natural", 50.0, "edm", duff_load=10.0,
    )
    zero = consm_mineral_soil(
        "SouthEast", "CP", "Natural", 50.0, "edm", duff_load=0.0,
    )
    negative = consm_mineral_soil(
        "SouthEast", "CP", "Natural", 50.0, "edm", duff_load=-1.0,
    )
    assert float(positive) == pytest.approx(5.0, abs=ATOL)
    assert float(zero) == pytest.approx(100.0, abs=ATOL)
    assert float(negative) == pytest.approx(100.0, abs=ATOL)


def test_consm_mineral_soil_coastal_plain_requires_duff_load():
    """
    (a) Coastal Plain mineral-soil exposure requires ``duff_load``.

    :return: None. Raises via ``assert`` on the expected exception.
    """
    with pytest.raises(ValueError, match="duff_load"):
        consm_mineral_soil("SouthEast", "CoastPlain", "Natural", 50.0, "edm")


def test_consm_mineral_soil_coastal_plain_wrong_region_raises():
    """
    (a) Same SouthEast-only contract as ``consm_duff``/``consm_litter``.

    :return: None. Raises via ``assert`` on the expected exception.
    """
    with pytest.raises(ValueError, match="SouthEast"):
        consm_mineral_soil(
            "InteriorWest", "CoastPlain", "Natural", 50.0, "edm",
            duff_load=10.0,
        )


def test_consm_mineral_soil_pocosin_pc_unaffected():
    """
    (a) ``PC`` (Pocosin) must not be routed through the new Coastal Plain
    guard/branch at all -- no ``duff_load`` is required for it, and its
    result differs from Coastal Plain's.

    :return: None. Raises via ``assert`` on mismatch.
    """
    pocosin = consm_mineral_soil("SouthEast", "PC", "Natural", 50.0, "edm")
    coastplain = consm_mineral_soil(
        "SouthEast", "CoastPlain", "Natural", 50.0, "edm", duff_load=10.0,
    )
    assert np.isfinite(float(pocosin))
    assert float(pocosin) != pytest.approx(float(coastplain), abs=1e-3)


def test_run_fofem_emissions_coastal_plain_does_not_double_count_litter():
    """
    (a) Full-pipeline check: with ``use_burnup=True``, litter consumption
    must come SOLELY from the Coastal Plain forest-floor route (matching
    the ``use_burnup=False`` call exactly), never additionally accumulated
    through Burnup's own (unrelated) fuel-particle accounting.

    :return: None. Raises via ``assert`` on mismatch.
    """
    common = dict(
        litter=2.0, duff=10.0, duff_depth=2.0, herb=0.0, shrub=1.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region="SouthEast", cvr_grp="CoastPlain", season="Summer",
        fuel_category="Natural", duff_moist=50.0, l_moist=40.0,
        dw10=0.5, dw1=0.3, dw100=1.0, dw10_moist=15.0, dw1000_moist=25.0,
        units="Imperial",
    )
    no_burnup = run_fofem_emissions(use_burnup=False, **common)
    with_burnup = run_fofem_emissions(
        use_burnup=True, fuel_bed_depth=0.3, ambient_temp=25.0,
        windspeed=2.0, **common,
    )
    assert with_burnup["BurnupError"] == 0
    assert float(with_burnup["LitPre"]) == pytest.approx(float(no_burnup["LitPre"]), abs=ATOL)
    assert float(with_burnup["LitCon"]) == pytest.approx(float(no_burnup["LitCon"]), abs=ATOL)
    assert float(with_burnup["LitPos"]) == pytest.approx(float(no_burnup["LitPos"]), abs=ATOL)
