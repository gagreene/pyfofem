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
#: Entire/Lower/NFDR/AdjNFDR (fof_ci.h:282-285); Python's matching vocabulary
#: is ``edm``/``ldm``/``nfdth``/``adjnfdr``.
DUFF_METHOD_TO_PY_CATEGORY = {
    "Entire": "edm", "Lower": "ldm", "NFDR": "nfdth", "AdjNFDR": "adjnfdr",
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

#: The direct C++ shrub oracle serializes ``shrub_pct`` to four decimal
#: places, so its centrally recorded comparison bound reflects that format.
ATOL_SHRUB_PERCENT = phase4_tolerance("shrub_herb_eq", "shrub")[0]

#: ``consm_duff``'s percent output (``pdc``) vs the golden's ``DufPer``:
#: scenarios that DIVERGE, each with the finding it reproduces. Every other
#: consume scenario agrees to within :data:`ATOL_PERCENT` (measured max |diff|
#: 6e-06 across the 31 agreeing scenarios — ``se-cp-entire-m050`` moved from
#: this list to the agreeing set 2026-09-16 when F-39's Coastal Plain
#: sub-finding was resolved; ``ne-wph-entire-m050`` moved 2026-09-18 when
#: F-39's White Pine-Hemlock sub-finding was resolved (``consm_duff`` now
#: routes NorthEast + WhiPinHem into the same InteriorWest branch as the
#: pinned C++ ``DUF_NorthEast`` delegation, ``fof_duf.cpp:444-446``);
#: ``ne-gen-entire-m020`` moved 2026-09-21 when F-23's NorthEast-generic
#: percent-routing defect was resolved (``consm_duff`` now always uses
#: Eq 2 -- C++ ``Duf_Default``'s ``Equ_2_Per`` -- for NorthEast's generic
#: cover-group bucket, matching ``fof_duf.cpp:454-455`` exactly, instead
#: of the wrong Eq-15 branch it previously used whenever
#: ``duff_moist_cat == 'edm'``); see ``test_consume_duff_percent_matches_cpp``).
DUFF_PERCENT_XFAIL = {}

#: ``consm_duff``'s depth outputs (``ddc``/``rdd``) vs ``DufDepCon``/
#: ``DufDepPos``: scenarios that DIVERGE. RESOLVED 2026-09-21 (F-23 fix
#: pass): this dict formerly listed 22 scenarios whose depth diverged
#: because ``consm_duff`` used the abandoned per-region depth-reduction
#: regression equations (Eq 5/6/7/15) instead of C++ ``DUF_Mngr``'s own
#: unconditional non-batch override (``fof_duf.cpp:290-300`` Note-5,
#: ``:395``: ``f_Red = f_DufDep * (f_Per / 100.0)`` for EVERY region and
#: fuel category, not just InteriorWest/PacificWest). ``consm_duff`` now
#: always derives ``ddc``/``rdd`` from the final ``pdc`` this same way;
#: all 32 consume scenarios agree here (measured max |diff| 2.9e-07). Kept
#: as an empty dict (not deleted) so ``_maybe_xfail`` calls below need no
#: further edits and so a future regression has an obvious place to land.
DUFF_DEPTH_XFAIL = {}

#: ``consm_mineral_soil`` vs the golden's ``MSE``: scenarios that DIVERGE.
#: The other 22 agree to within :data:`ATOL_PERCENT` (measured max |diff|
#: 2.8e-06) — ``se-cp-entire-m050`` moved from this list to the agreeing set
#: 2026-09-16 when F-39's Coastal Plain sub-finding was resolved;
#: ``chaparral-entire-m050`` moved 2026-09-18 when F-39's Chaparral-MSE
#: sub-finding was resolved (``consm_mineral_soil`` now returns 100.0 for
#: any Chaparral/SGC cover group, matching the pinned C++ ``Equ_19_MSE``,
#: ``fof_duf.cpp:1336-1339``, exactly, ahead of every other branch).
MSE_XFAIL = {}










































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
        # F-39 (Coastal Plain, Eq 30): harmless for every other scenario -
        # consm_duff only reads these two for a Coastal Plain cover group.
        pre_ll=litter,
        l_moist=float(_consume_input(overrides, "litter_moist_pct")),
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
        pdr=float(_py_duff(overrides)["pdc"]),
        # F-39 (Coastal Plain, Eq 32): harmless for every other scenario -
        # consm_mineral_soil only reads this for a Coastal Plain cover group.
        duff_load=float(_consume_input(overrides, "duff_tac")),
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


def test_consume_coastal_plain_litter_matches_cpp():
    """
    ``consm_litter`` vs the golden ``LitCon`` for the ONE consume-mode row
    whose litter C++ does NOT route through Burnup: ``se-cp-entire-m050``,
    the Coastal Plain scenario.

    Closes ``BR-LIT-CP`` (F-39, Coastal Plain sub-finding RESOLVED
    2026-09-16): ``consm_litter`` now has a real Coastal Plain route
    (Eq 30, ``fof_duf.cpp:1171-1225``) and matches C++'s ``_CalcCP_Lit``
    (``fof_hsf.cpp:83-85,107-127``) result exactly.
    """
    overrides = next(
        o for c, o, _b in CONSUME_SCENARIOS if c == "se-cp-entire-m050"
    )
    row = golden_rows_by_case("consume", "_summary")["se-cp-entire-m050"]
    value = consm_litter(
        float(_consume_input(overrides, "litter_tac")),
        float(_consume_input(overrides, "litter_moist_pct")),
        reg="SouthEast", cvr_grp="CoastPlain", units="Imperial",
        pre_dl=float(_consume_input(overrides, "duff_tac")),
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
def test_litter_equation_matches_cpp(case_id, equ):
    """``consm_litter`` vs the ``litter_eq`` golden's ``con_tac``.

    Both equation 997 and equation 998 agree with the direct C++ oracle.
    """
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
def test_shrub_herb_herb_consumption_matches_cpp(case_id):
    """``consm_herb`` vs the golden ``herb_con_tac`` (direct Calc_Herb)."""
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
def test_shrub_herb_shrub_percent_matches_cpp(case_id):
    """``consm_shrub`` vs the golden ``shrub_pct`` (direct Calc_Shrub)."""
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
        float(row["shrub_pct"]), abs=ATOL_SHRUB_PERCENT
    )
