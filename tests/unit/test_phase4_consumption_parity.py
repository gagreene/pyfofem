#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_consumption_parity.py - Phase 4 Tier-2 consumption coverage for
``consm_duff``, ``consm_mineral_soil``, ``consm_litter``, ``consm_herb``,
``consm_shrub`` and ``consm_canopy``.

**Assertion class (required module declaration):** every numeric assertion in
this module is class **(c) manifested executable C++ parity** - each expected
value is read from the committed, fully manifested Phase 4 golden dataset
under ``tests/test_data/test_golden_output/phase4/``, which was produced by
the compiled ``fofem_test`` harness running the pinned C++ revision
``78f97f09...``. Nothing here re-implements a C++ equation, and no value is
hand-derived.

Two distinct oracle routes are used, and they are NOT interchangeable:

- ``litter_eq`` and ``shrub_herb_eq`` goldens call the pinned SCALAR C++
  functions directly (``PFW_Litter_Eq997``, ``LitterSouthEast``,
  ``Calc_Shrub``/``Shrub_Equ``, ``Calc_Herb``, ``Calc_CrownFoliage``,
  ``Calc_CrownBranch``), so those are direct executable function parity.
- ``consume`` goldens are produced by the full, faithful
  ``CM_Mngr -> BCM_Mngr -> Burnup`` pipeline, so every ``consume`` assertion
  is a **full-pipeline** comparison, never a claim that a Python function
  equals an isolated C++ call.

The module also contains a small number of Python-only contract tests
(scalar/array shape, invalid input). Those are labelled in their own
docstrings as class **(a) Python contract tests** and make no parity claim.

Every ``xfail`` below is ``strict=True`` and names the exact Gate 0 finding it
reproduces, scoped to the exact scenarios that actually reproduce it -
measured, not assumed. A scenario that merely shares an output column with a
divergent one is NOT marked.

Function order: private helpers first, then public test functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pyfofem.components.consumption_calcs import (
    consm_canopy,
    consm_duff,
    consm_herb,
    consm_litter,
    consm_mineral_soil,
    consm_shrub,
)
from tests.cpp_parity_live._golden_manifest import validate_manifest
from tests.cpp_parity_live._phase4_contract import (
    CONSUME_INDEX,
    CONSUME_SCENARIOS,
    LITTER_EQ_SCENARIOS,
    PHASE4_MODES,
    SHRUB_HERB_EQ_SCENARIOS,
    SHRUB_HERB_INDEX,
    golden_manifest,
    golden_rows_by_case,
    phase4_tolerance,
    require_golden_tree,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

# Fail CLOSED, not open: a missing/incomplete committed Phase 4 golden
# dataset is a repository defect, never a silent skip. This runs at
# collection time so a broken checkout surfaces as a loud, actionable
# error naming the exact missing file(s) - see require_golden_tree().
require_golden_tree()

#: Feet-to-metres and inches-to-centimetres, exactly as the pinned C++ uses
#: them (``FtToMt`` fof_mrt.cpp:542-547, ``InchToCent`` fof_util.cpp:527-530).
FT_TO_M = 0.3048
IN_TO_CM = 2.54

#: The harness's ``duff_moist_method`` token to ``consm_duff``'s
#: ``duff_moist_cat`` vocabulary. C++ spells the four methods
#: Entire/Lower/NFDR/AdjNFDR (fof_ci.h:282-285); Python spells the first three
#: edm/ldm/nfdth and has no AdjNFDR token at all (F-22), so AdjNFDR maps onto
#: the closest Python token, ``nfdth`` - which is precisely why the Adj-NFDR
#: scenario reproduces F-27a below.
DUFF_METHOD_TO_PY_CATEGORY = {
    "Entire": "edm", "Lower": "ldm", "NFDR": "nfdth", "AdjNFDR": "nfdth",
}

#: Absolute tolerance for a percent-valued consume output, retrieved from the
#: centralized policy (never a literal here) - ``consume_p4.duff_percent``,
#: ``consume_p4.mineral_soil``, ``consume_p4.duff_depth`` and
#: ``consume_p4.herb`` all record the SAME 1e-05, because it is not a
#: per-route tuned value: every Phase 4 golden's scientific columns are
#: written through the harness's own ``fmt(v, 6)`` (six decimal places), so
#: this is the output format's shared resolution. Measured maximum |diff|
#: across all agreeing consume percent scenarios is 6e-06, so 1e-05 sits
#: just above the output's own resolution and well below any real divergence
#: (the smallest divergence measured here is 4.9).
ATOL_PERCENT = phase4_tolerance("consume", "mineral_soil")[0]

#: Absolute tolerance for a T/ac or inch-valued consume output, likewise
#: retrieved from the policy - ``consume_p4.herb`` records the same 1e-05
#: output-resolution bound as ``consume_p4.duff_depth``,
#: ``consume_p4.litter_cp`` and ``shrub_herb_eq_p4.*``. Measured maximum
#: |diff| across all agreeing scenarios is 2.9e-07 (duff depth) and 5.6e-17
#: (herb load); 1e-05 again sits just above the six-decimal output
#: resolution.
ATOL_LOAD = phase4_tolerance("consume", "herb")[0]

#: ``consm_duff``'s percent output (``pdc``) vs the golden's ``DufPer``:
#: scenarios that DIVERGE, each with the finding it reproduces. Every other
#: consume scenario agrees to within :data:`ATOL_PERCENT` (measured max |diff|
#: 6e-06 across the 28 agreeing scenarios).
DUFF_PERCENT_XFAIL = {
    "ne-gen-entire-m020": (
        "F-23",
        "NorthEast generic + Entire: C++ Duf_Default -> Equ_2_Per gives "
        "75.180000, Python derives the percent from the Eq-15 residual-depth "
        "relation and gives 55.550000 (measured |diff| 19.63 percentage "
        "points). This is the case-6 defect.",
    ),
    "ne-wph-entire-m050": (
        "F-39",
        "NorthEast + WhiPinHem: C++ DUF_NorthEast delegates to "
        "DUF_InteriorWest (fof_duf.cpp:444-446) giving 62.400000; Python has "
        "no WhiPinHem cover group at all and falls into its NorthEast generic "
        "Eq-15 branch, giving 49.550000 (measured |diff| 12.85).",
    ),
    "se-cp-entire-m050": (
        "F-39",
        "SouthEast + CoastPlain: C++ runs the Coastal Plain duff equations "
        "(fof_duf.cpp:1115-1249) giving 28.871900; Python has no CoastPlain "
        "cover group and falls through to its SouthEast Eq-16 branch, giving "
        "38.623500 (measured |diff| 9.75).",
    ),
    "iw-nat-entire-zero-duff": (
        "F-40",
        "Zero duff load: C++ DUF_Calc yields DufPer 0 when there is no duff "
        "to consume; Python returns the moisture-based 62.400000 regardless "
        "(measured |diff| 62.4).",
    ),
}

#: ``consm_duff``'s depth outputs (``ddc``/``rdd``) vs ``DufDepCon``/
#: ``DufDepPos``: scenarios that DIVERGE. Only 9 of the 32 consume scenarios
#: agree here (measured max |diff| 2.9e-07 among those).
DUFF_DEPTH_XFAIL = {
    case: (
        "F-39",
        "the percent this depth is derived from already diverges for this "
        "cover group (see DUFF_PERCENT_XFAIL), so the depth cannot agree.",
    )
    for case in ("ne-wph-entire-m050", "se-cp-entire-m050")
}
DUFF_DEPTH_XFAIL.update({
    case: (
        "F-23",
        "C++ DUF_Mngr overwrites the depth reduction unconditionally at "
        "fof_duf.cpp:395 with f_DufDep * (f_Per/100) for every non-batch run, "
        "discarding Eqs 5/6/7/15; Python instead returns the raw regression "
        "depth for InteriorWest/PacificWest (and for any row whose region is "
        "IW/PW regardless of cover group). Measured |diff| ranges from 0.006 "
        "to 1.0791 inches across these scenarios.",
    )
    for case in (
        "iw-nat-entire-m050", "iw-nat-entire-m130", "iw-nat-lower-m050",
        "iw-nat-lower-m180", "iw-slash-lower-m050", "iw-slash-lower-m150",
        "iw-slash-nfdr-m020", "iw-nat-nfdr-m020", "iw-pn-lower-m050",
        "iw-pn-entire-m050", "pw-nat-entire-m050", "ne-gen-entire-m020",
        "chaparral-entire-m050", "piles-entire-m050", "iw-nat-entire-m010",
        "iw-crown-burn-000", "iw-crown-burn-100", "iw-crown-zero-load",
        "emis-legacy-iw", "emis-expanded-g258", "emis-expanded-g378",
    )
})

#: ``consm_mineral_soil`` vs the golden's ``MSE``: scenarios that DIVERGE.
#: The other 20 agree to within :data:`ATOL_PERCENT` (measured max |diff|
#: 2.8e-06).
MSE_XFAIL = {
    "iw-nat-lower-m180": (
        "F-26",
        "Eq 13 at 180 % duff moisture: C++ floors the result at 0, Python "
        "returns -18.800000 (60.4 - 0.44*180).",
    ),
    "iw-nat-nfdr-m020": (
        "F-26",
        "Eq 12 at 20 % duff moisture: C++ floors the result at 0, Python "
        "returns -4.900000 (94.3 - 4.96*20).",
    ),
    "iw-slash-adjnfdr-m028": (
        "F-27a",
        "Adj-NFDR: C++ Equ_11_MSE divides the duff moisture by e_Adj = 1.4 "
        "(fof_duf.cpp:25, :903) giving 22.300000; Python has no Adj-NFDR "
        "path, applies Eq 11 to the raw moisture and also omits the 0-floor, "
        "giving -6.100000 (measured |diff| 28.4).",
    ),
    "ne-rjp-entire-m020": (
        "F-41",
        "NorthEast RedJacPin: C++ uses Equ_14_MSE on the percent duff "
        "reduction (-8.98 + 0.44*27.549999 = 3.142000); Python has no "
        "NorthEast branch and falls through to Eq 10, giving 72.734900.",
    ),
    "ne-bbs-entire-m050": (
        "F-41",
        "NorthEast BalBRWSpr: C++ Equ_14_MSE gives 12.822001; Python falls "
        "through to Eq 10, giving 43.780100.",
    ),
    "ne-rjp-lower-m050": (
        "F-41",
        "NorthEast + Lower: no Python np.select branch matches "
        "(~is_iw_pw & ~is_pocosin only covers 'edm' and '%dr'), so Python "
        "silently returns NaN where C++ gives 43.780100.",
    ),
    "ne-bbs-lower-m050": (
        "F-41",
        "NorthEast + Lower: Python silently returns NaN where C++ gives "
        "22.194000.",
    ),
    "se-gen-entire-m050": (
        "F-41",
        "SouthEast generic: C++ uses Equ_14_MSE on the percent duff "
        "reduction (-8.98 + 0.44*38.623539 = 8.014360); Python falls through "
        "to Eq 10, giving 43.780100.",
    ),
    "emis-legacy-se": (
        "F-41",
        "same SouthEast Equ_14_MSE route as se-gen-entire-m050; this "
        "scenario differs only in its emissions dispatch, which does not "
        "affect mineral-soil exposure.",
    ),
    "chaparral-entire-m050": (
        "F-39",
        "ShrubGroupChaparral: C++ Equ_19_MSE returns 100 "
        "(fof_duf.cpp:1325-1363); Python has no chaparral mineral-soil rule "
        "and returns the Eq-10 value 43.780100.",
    ),
    "se-cp-entire-m050": (
        "F-39",
        "CoastPlain: C++ Equ_CP_MSE returns 5.000000; Python has no Coastal "
        "Plain route and returns the Eq-10 value 43.780100.",
    ),
    "iw-nat-entire-zero-duff": (
        "F-40",
        "Zero duff load: C++ DUF_Mngr sets f_MSEPer = 100 when f_Duff <= 0 "
        "(fof_duf.cpp:388-389); Python returns the Eq-10 value 43.780100.",
    ),
}

#: ``consm_herb`` vs the ``shrub_herb_eq`` golden's ``herb_con_tac``:
#: scenarios that DIVERGE. All others agree (measured max |diff| 5.6e-17).
HERB_XFAIL = {
    "shr23-herb221-iw-gg-spring": (
        "F-35",
        "Eq 221: C++ Herb_Eq221 consumes 90 % of the herb load "
        "(fof_hsf.cpp:352-358, `f = f_Herb * 0.9`), Python consumes 10 % "
        "(`pre_hl * 0.1`). Measured 0.450000 vs 0.050000 T/ac.",
    ),
    "shr236-herb223-se-pfw-summer": (
        "F-11",
        "Pine Flatwoods: C++ Calc_Herb tests PinFlaWoo FIRST "
        "(fof_hsf.cpp:299-301) and uses Eq 223 (0.497200); Python's np.select "
        "lists the SouthEast branch first, shadowing flatwoods, and uses "
        "Eq 222 (0.407500).",
    ),
    "shr236-herb223-se-pfw-fall": (
        "F-11",
        "same inverted PinFlaWoo-vs-SouthEast precedence as the summer case; "
        "herb Eq 223 carries no season term, so both seasons diverge "
        "identically.",
    ),
    "herb222-se-clamp-low": (
        "F-12",
        "Eq 222 goes negative for a small herb load with no litter; C++ "
        "clamps the result to 0 (fof_hsf.cpp:322-323), Python does not.",
    ),
    "herb222-se-clamp-high": (
        "F-12",
        "Eq 222 exceeds the pre-fire herb load for a large litter load; C++ "
        "clamps the result to f_Herb (fof_hsf.cpp:320-321), Python does not.",
    ),
}

#: ``consm_shrub`` vs the ``shrub_herb_eq`` golden's ``shrub_pct``: scenarios
#: that DIVERGE. All others agree exactly (measured max |diff| 0.0).
SHRUB_XFAIL = {
    case: (
        "F-13/F-14",
        "SouthEast non-Pocosin Eq 234: C++ Shrub_Equ multiplies Equ_234_Per's "
        "FRACTION by the shrub load and then clamps the consumed amount to "
        "[0, f_Shrub] before deriving the percent; Python returns the same "
        "expression scaled by 100 with no clamp, so it reports percentages "
        "far above 100.",
    )
    for case in (
        "shr234-herb222-se", "shr234-herb222-se-highlit",
        "herb222-se-clamp-low", "herb222-se-clamp-high",
    )
}
SHRUB_XFAIL.update({
    case: (
        "F-13/F-15",
        "Pine Flatwoods Eq 236: C++ Calc_Shrub tests PinFlaWoo before "
        "SouthEast (fof_hsf.cpp:141-165) and PFW_Shrub_Eq236 does the Mg/ha "
        "round-trip and the exp(); Python's SouthEast branch shadows "
        "flatwoods entirely, so the Eq-236 code is never even reached for a "
        "SouthEast row.",
    )
    for case in ("shr236-herb223-se-pfw-summer", "shr236-herb223-se-pfw-fall")
})
SHRUB_XFAIL["zero-shrub-herb-crown"] = (
    "F-38",
    "Zero pre-fire shrub load: C++ Calc_Shrub sets consumed, post and "
    "percent all to 0 when f_Shrub == 0 (fof_hsf.cpp:186-189); Python "
    "returns the Eq-23 percentage 60.0 for a load that does not exist.",
)










































# ===========================================================================
# Class (a) Python contract tests - no parity claim
# ===========================================================================


def _consume_input(overrides, column):
    """
    Return one consume input field for a scenario, after its overrides.

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :returns: The raw string value that was written to the golden's input CSV.
    """
    row = list(MODES["consume"]["row"])
    for key, value in overrides.items():
        row[CONSUME_INDEX[key]] = value
    return row[CONSUME_INDEX[column]]


def _maybe_xfail(request, table, case_id):
    """
    Apply a strict xfail to the running test if *case_id* is a known
    divergence.

    :param request: The pytest ``request`` fixture.
    :param table: One of the ``*_XFAIL`` tables in this module.
    :param case_id: The scenario being compared.
    :returns: None.
    """
    if case_id not in table:
        return
    finding, reason = table[case_id]
    request.node.add_marker(
        pytest.mark.xfail(
            strict=True,
            reason=f"{finding}: {reason}",
        )
    )


def _py_duff(overrides):
    """
    Call ``consm_duff`` with the exact inputs the C++ golden row was given.

    The SouthEast Eq-16 aggregates (``pre_dl110``/``pre_l110``) are supplied
    explicitly and are derived from the SAME input columns C++'s
    ``Equation_16`` sums internally (``f_Lit + f_Duff + f_DW10 + f_DW1`` and
    ``f_Lit + f_DW10 + f_DW1``, fof_hsf.cpp:230-247) - not from any C++
    output, so oracle independence is preserved.

    :param overrides: The scenario's column-name to value overrides.
    :returns: ``consm_duff``'s result dict, in Imperial units.
    """
    litter = float(_consume_input(overrides, "litter_tac"))
    duff = float(_consume_input(overrides, "duff_tac"))
    dw1 = float(_consume_input(overrides, "dw1_tac"))
    dw10 = float(_consume_input(overrides, "dw10_tac"))
    fuel_cat = _consume_input(overrides, "fuel_cat")
    return consm_duff(
        duff,
        float(_consume_input(overrides, "duff_moist_pct")),
        reg=_consume_input(overrides, "region"),
        cvr_grp=_consume_input(overrides, "cover_group") or None,
        duff_moist_cat=DUFF_METHOD_TO_PY_CATEGORY[
            _consume_input(overrides, "duff_moist_method")
        ],
        d_pre=float(_consume_input(overrides, "duff_depth_in")),
        pre_dl110=litter + duff + dw10 + dw1,
        pre_l110=litter + dw10 + dw1,
        dw1000_moist=float(_consume_input(overrides, "dw1000_moist_pct")),
        pile=(fuel_cat == "Piles"),
        units="Imperial",
    )


def _py_mineral_soil(overrides):
    """
    Call ``consm_mineral_soil`` with the golden row's own inputs.

    :param overrides: The scenario's column-name to value overrides.
    :returns: Mineral-soil exposure percent (may be NaN - see F-41).
    """
    fuel_cat = _consume_input(overrides, "fuel_cat")
    return consm_mineral_soil(
        _consume_input(overrides, "region"),
        _consume_input(overrides, "cover_group") or "NA",
        fuel_cat,
        float(_consume_input(overrides, "duff_moist_pct")),
        DUFF_METHOD_TO_PY_CATEGORY[
            _consume_input(overrides, "duff_moist_method")
        ],
        pile=(fuel_cat == "Piles"),
    )


def _shrub_herb_input(overrides, column):
    """
    Return one shrub_herb_eq input field for a scenario, after its overrides.

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :returns: The raw string value written to the golden's input CSV.
    """
    row = list(MODES["shrub_herb_eq"]["row"])
    for key, value in overrides.items():
        row[SHRUB_HERB_INDEX[key]] = value
    return row[SHRUB_HERB_INDEX[column]]


def test_consm_canopy_scalar_and_array_agree():
    """Python contract: scalar inputs and length-1 arrays must agree, and the
    return types must follow the scalar-array convention."""
    scalar = consm_canopy(50.0, 0.5, 0.5, units="Imperial")
    array = consm_canopy(
        np.array([50.0]), np.array([0.5]), np.array([0.5]), units="Imperial"
    )
    assert isinstance(scalar["flc"], float)
    assert isinstance(array["flc"], np.ndarray)
    assert array["flc"][0] == pytest.approx(scalar["flc"])
    assert array["blc"][0] == pytest.approx(scalar["blc"])


def test_consm_duff_scalar_and_array_agree():
    """Python contract: ``consm_duff`` honours the scalar-array convention."""
    kwargs = dict(reg="InteriorWest", cvr_grp=None, duff_moist_cat="edm",
                  d_pre=2.0, units="Imperial")
    scalar = consm_duff(10.0, 50.0, **kwargs)
    array = consm_duff(np.array([10.0]), np.array([50.0]), **kwargs)
    assert isinstance(scalar["pdc"], float)
    assert isinstance(array["pdc"], np.ndarray)
    assert array["pdc"][0] == pytest.approx(scalar["pdc"])


def test_consm_mineral_soil_unrecognised_category_returns_nan_silently():
    """Python contract (F-22): an unrecognised ``duff_moist_cat`` produces no
    error - it silently returns NaN. Pinned for visibility, not endorsed."""
    value = consm_mineral_soil(
        "InteriorWest", "", "Natural", 75.0, "Entire",
    )
    assert math.isnan(float(value))


def test_consm_shrub_returns_percent_not_load():
    """Python contract: ``consm_shrub`` returns a PERCENT, while the C++
    ``Calc_Shrub`` out-parameter ``af_Con`` is a LOAD. Pinned so the two are
    never accidentally compared without the conversion."""
    value = consm_shrub("InteriorWest", "NA", 4.0, season="Summer",
                        units="Imperial")
    assert float(value) == pytest.approx(60.0)


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_canopy_branch_consumption_matches_cpp(case_id):
    """``consm_canopy``'s branch load consumed vs the golden ``BraCon``."""
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    result = consm_canopy(
        float(_consume_input(overrides, "pct_crown_burn")),
        float(_consume_input(overrides, "crown_fol_tac")),
        float(_consume_input(overrides, "crown_bra_tac")),
        units="Imperial",
    )
    assert float(result["blc"]) == pytest.approx(
        float(row["BraCon"]), abs=ATOL_LOAD
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_canopy_foliage_consumption_matches_cpp(case_id):
    """``consm_canopy``'s foliage load consumed vs the golden ``FolCon``."""
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    result = consm_canopy(
        float(_consume_input(overrides, "pct_crown_burn")),
        float(_consume_input(overrides, "crown_fol_tac")),
        float(_consume_input(overrides, "crown_bra_tac")),
        units="Imperial",
    )
    assert float(result["flc"]) == pytest.approx(
        float(row["FolCon"]), abs=ATOL_LOAD
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-39: consm_litter has no Coastal Plain route (CVR_GRP_CODES "
        "contains no CoastPlain token). The closest representable Python "
        "call - reg='SouthEast' (the region the scenario really carries) "
        "with the unrecognised cover_group='CoastPlain' - falls into the "
        "SouthEast equation-998 branch and returns 1.600000 T/ac, where "
        "C++ HSF_Mngr dispatches _CalcCP_Lit (fof_hsf.cpp:83-85, :107-127) "
        "and reports 2.000000 T/ac. Measured |diff| 0.400000 T/ac (20% of "
        "the 2.0 T/ac pre-fire load), closing BR-LIT-CP's prior "
        "EXPECT-INVESTIGATE status with real executed evidence."
    ),
)
def test_consume_coastal_plain_litter_matches_cpp():
    """
    ``consm_litter`` vs the golden ``LitCon`` for the ONE consume-mode row
    whose litter C++ does NOT route through Burnup: ``se-cp-entire-m050``,
    the Coastal Plain scenario.

    Closes ``BR-LIT-CP``: the C++ oracle route is reachable (``HSF_Mngr``
    dispatches ``_CalcCP_Lit`` because ``DUF_Mngr`` set
    ``i_LitEqu == e_CP_PerEq``, per ``consume_p4.litter_cp`` in
    ``tolerance_policy.json``), and this asserts the DESIRED behaviour - a
    real, non-approximated Coastal Plain route - not the current
    SouthEast-branch fallback.
    """
    overrides = next(
        o for c, o, _b in CONSUME_SCENARIOS if c == "se-cp-entire-m050"
    )
    row = golden_rows_by_case("consume", "_summary")["se-cp-entire-m050"]
    value = consm_litter(
        float(_consume_input(overrides, "litter_tac")),
        float(_consume_input(overrides, "litter_moist_pct")),
        reg="SouthEast", cvr_grp="CoastPlain", units="Imperial",
    )
    assert float(value) == pytest.approx(float(row["LitCon"]), abs=ATOL_LOAD)


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_duff_depth_consumed_matches_cpp(case_id, request):
    """``consm_duff``'s ``ddc`` vs the golden ``DufDepCon`` (full pipeline)."""
    _maybe_xfail(request, DUFF_DEPTH_XFAIL, case_id)
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    assert float(_py_duff(overrides)["ddc"]) == pytest.approx(
        float(row["DufDepCon"]), abs=ATOL_LOAD
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_duff_depth_residual_matches_cpp(case_id, request):
    """``consm_duff``'s ``rdd`` vs the golden ``DufDepPos``."""
    _maybe_xfail(request, DUFF_DEPTH_XFAIL, case_id)
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    assert float(_py_duff(overrides)["rdd"]) == pytest.approx(
        float(row["DufDepPos"]), abs=ATOL_LOAD
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_duff_percent_matches_cpp(case_id, request):
    """``consm_duff``'s ``pdc`` vs the golden ``DufPer`` (full pipeline)."""
    _maybe_xfail(request, DUFF_PERCENT_XFAIL, case_id)
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    assert float(_py_duff(overrides)["pdc"]) == pytest.approx(
        float(row["DufPer"]), abs=ATOL_PERCENT
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in CONSUME_SCENARIOS],
)
def test_consume_mineral_soil_matches_cpp(case_id, request):
    """``consm_mineral_soil`` vs the golden ``MSE`` (full pipeline)."""
    _maybe_xfail(request, MSE_XFAIL, case_id)
    overrides = next(o for c, o, _b in CONSUME_SCENARIOS if c == case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    value = float(_py_mineral_soil(overrides))
    assert not math.isnan(value), (
        "consm_mineral_soil returned NaN - no np.select branch matched"
    )
    assert value == pytest.approx(float(row["MSE"]), abs=ATOL_PERCENT)


def test_consume_scenarios_all_produced_an_ok_oracle_row():
    """Every Phase 4 consume scenario must have produced a real, successful
    C++ pipeline row - a suppressed or errored row is not an oracle."""
    rows = golden_rows_by_case("consume", "_summary")
    bad = {
        case: (rows[case]["outcome"], rows[case]["err_text"])
        for case, _o, _b in CONSUME_SCENARIOS
        if rows[case]["outcome"] != "ok" or rows[case]["err_text"].strip()
    }
    assert not bad, bad


@pytest.mark.parametrize("mode", PHASE4_MODES)
def test_every_phase4_manifest_is_structurally_valid(mode):
    """Each committed Phase 4 manifest must pass the shared validator's
    structural, exact-value and internal-consistency checks."""
    manifest = golden_manifest(mode)
    assert manifest is not None, f"no committed Phase 4 manifest for {mode!r}"
    assert manifest.get("dataset") == "phase4", (
        f"{mode}: manifest must record dataset='phase4', got "
        f"{manifest.get('dataset')!r}"
    )
    errors = validate_manifest(manifest, check_against_live_checkout=False)
    assert not errors, errors


@pytest.mark.parametrize(
    "case_id,equ",
    [(case, equ) for case, equ, _l, _m, _b in LITTER_EQ_SCENARIOS],
)
def test_litter_equation_matches_cpp(case_id, equ, request):
    """``consm_litter`` vs the ``litter_eq`` golden's ``con_tac``.

    Equation 997 is a strict xfail on every scenario (F-07/F-08/F-10);
    equation 998 agrees exactly.
    """
    if equ == "997":
        request.node.add_marker(pytest.mark.xfail(
            strict=True,
            reason=(
                "F-07/F-08/F-10: PFW_Litter_Eq997 converts to Mg/ha, "
                "evaluates the polynomial there, converts back and caps the "
                "result at the pre-fire load (fof_hsf.cpp:780-798); "
                "consm_litter does none of the three. Measured |diff| up to "
                "0.555994 T/ac and up to +92 % relative across this "
                "scenario set."
            ),
        ))
    scenario = next(s for s in LITTER_EQ_SCENARIOS if s[0] == case_id)
    _case, _equ, load, moist, _branches = scenario
    row = golden_rows_by_case("litter_eq")[case_id]
    if equ == "997":
        value = consm_litter(float(load), float(moist), cvr_grp="PFL",
                             units="Imperial")
    else:
        value = consm_litter(float(load), 0.0, reg="SouthEast",
                             units="Imperial")
    assert float(value) == pytest.approx(float(row["con_tac"]), abs=ATOL_LOAD)


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in SHRUB_HERB_EQ_SCENARIOS],
)
def test_shrub_herb_crown_branch_matches_cpp(case_id):
    """``consm_canopy``'s ``blc`` vs the golden ``bra_con_tac``."""
    overrides = next(o for c, o, _b in SHRUB_HERB_EQ_SCENARIOS if c == case_id)
    row = golden_rows_by_case("shrub_herb_eq")[case_id]
    result = consm_canopy(
        float(_shrub_herb_input(overrides, "pct_crown_burn")),
        float(_shrub_herb_input(overrides, "crown_fol_tac")),
        float(_shrub_herb_input(overrides, "crown_bra_tac")),
        units="Imperial",
    )
    assert float(result["blc"]) == pytest.approx(
        float(row["bra_con_tac"]), abs=ATOL_LOAD
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in SHRUB_HERB_EQ_SCENARIOS],
)
def test_shrub_herb_crown_foliage_matches_cpp(case_id):
    """``consm_canopy``'s ``flc`` vs the golden ``fol_con_tac``."""
    overrides = next(o for c, o, _b in SHRUB_HERB_EQ_SCENARIOS if c == case_id)
    row = golden_rows_by_case("shrub_herb_eq")[case_id]
    result = consm_canopy(
        float(_shrub_herb_input(overrides, "pct_crown_burn")),
        float(_shrub_herb_input(overrides, "crown_fol_tac")),
        float(_shrub_herb_input(overrides, "crown_bra_tac")),
        units="Imperial",
    )
    assert float(result["flc"]) == pytest.approx(
        float(row["fol_con_tac"]), abs=ATOL_LOAD
    )


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in SHRUB_HERB_EQ_SCENARIOS],
)
def test_shrub_herb_herb_consumption_matches_cpp(case_id, request):
    """``consm_herb`` vs the golden ``herb_con_tac`` (direct Calc_Herb)."""
    _maybe_xfail(request, HERB_XFAIL, case_id)
    overrides = next(o for c, o, _b in SHRUB_HERB_EQ_SCENARIOS if c == case_id)
    row = golden_rows_by_case("shrub_herb_eq")[case_id]
    value = consm_herb(
        _shrub_herb_input(overrides, "region"),
        _shrub_herb_input(overrides, "cover_group") or "NA",
        float(_shrub_herb_input(overrides, "litter_tac")),
        float(_shrub_herb_input(overrides, "herb_tac")),
        season=_shrub_herb_input(overrides, "season"),
        units="Imperial",
    )
    assert float(value) == pytest.approx(
        float(row["herb_con_tac"]), abs=ATOL_LOAD
    )


def test_shrub_herb_scenarios_all_produced_an_ok_oracle_row():
    """Every Phase 4 shrub_herb_eq scenario must have produced a successful
    C++ row."""
    rows = golden_rows_by_case("shrub_herb_eq")
    bad = {
        case: rows[case]["outcome"]
        for case, _o, _b in SHRUB_HERB_EQ_SCENARIOS
        if rows[case]["outcome"] != "ok"
    }
    assert not bad, bad


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o, _b in SHRUB_HERB_EQ_SCENARIOS],
)
def test_shrub_herb_shrub_percent_matches_cpp(case_id, request):
    """``consm_shrub`` vs the golden ``shrub_pct`` (direct Calc_Shrub)."""
    _maybe_xfail(request, SHRUB_XFAIL, case_id)
    overrides = next(o for c, o, _b in SHRUB_HERB_EQ_SCENARIOS if c == case_id)
    row = golden_rows_by_case("shrub_herb_eq")[case_id]
    value = consm_shrub(
        _shrub_herb_input(overrides, "region"),
        _shrub_herb_input(overrides, "cover_group") or "NA",
        float(_shrub_herb_input(overrides, "shrub_tac")),
        season=_shrub_herb_input(overrides, "season"),
        pre_ll=float(_shrub_herb_input(overrides, "litter_tac")),
        pre_dl=float(_shrub_herb_input(overrides, "duff_tac")),
        pre_rl=0.0,
        duff_moist=float(_shrub_herb_input(overrides, "duff_moist_pct")),
        llc=0.0,
        ddc=0.0,
        units="Imperial",
    )
    assert float(value) == pytest.approx(
        float(row["shrub_pct"]), abs=ATOL_PERCENT
    )
