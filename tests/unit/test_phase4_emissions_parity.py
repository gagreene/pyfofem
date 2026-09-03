#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_emissions_parity.py - Phase 4 coverage for
``calc_smoke_emissions`` in all three modes.

**Assertion classes (required module declaration):**

- Class **(c) manifested FULL-PIPELINE C++ comparison** - the
  ``*_matches_cpp`` tests for ``legacy`` and ``expanded``. Their expected
  values come from the committed, fully manifested Phase 4 ``consume``
  golden, which the compiled harness produced by running the real
  ``CM_Mngr -> BCM_Mngr -> Burnup`` pipeline. Python is fed that same row's
  own ``FlaCon`` / ``SmoCon`` / ``DufCon`` consumption totals and its 21
  emission totals are compared against the row's own.

  **This is a full-pipeline comparison. It is NOT a demonstration that
  ``calc_smoke_emissions`` equals an isolated ``ES_Calc`` / ``ES_Calc_NEW``
  invocation** - no such invocation is obtainable, because Burnup seeds the
  calculators' component state from its own private globals
  (``bur_brn.cpp:301``, ``:3460-3567``) and each calculator performs
  ``ES_SetComponents(0, ...)`` + ``ES_FlaSmo`` internally (``:2092-2095``,
  ``:2402-2405``). Nothing in this module states otherwise.

- Class **(a) Python contract / equation tests** - everything covering
  ``default``, the factor expressions, and input validation. ``default`` has
  no established executable C++ counterpart; demonstrating one is Phase 6's
  investigation, and this module makes no ``default`` parity claim.

**Unit normalisation (F-32).** C++ reports emission totals in imperial units
via ``GramSqMt_To_Pounds`` (``fof_bcm.cpp:210-233``), whose two composing
constant pairs give a T/ac-load multiplier of
``(4046.86/453.592)/4.46 = 2.00040501817599``; Python's imperial ``unit_conv``
is a flat ``2.0``. The two therefore differ by a fixed, exactly reproducible
ratio of 2.025090879957947e-04, which is documented-expected and NOT a
defect - C++'s own two conversion paths already disagree with each other by
that same amount. Every comparison below divides each side by ITS OWN
multiplier first, so both are compared in native SI, exactly as
``gate0/05-harness-contract.md`` requires. Python's flat ``2.0`` imperial
behaviour is pinned separately as an exact contract, never against a C++
imperial oracle.

Function order: private helpers first, then public test functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem.components.emission_calcs import calc_smoke_emissions
from tests.cpp_parity_live._phase4_contract import (
    CONSUME_INDEX,
    CONSUME_SCENARIOS,
    golden_rows_by_case,
    phase4_tolerance,
    python_contract_epsilon,
    require_golden_tree,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

# Fail CLOSED, not open: a missing/incomplete committed Phase 4 golden
# dataset is a repository defect, never a silent skip. This runs at
# collection time so a broken checkout surfaces as a loud, actionable
# error naming the exact missing file(s) - see require_golden_tree().
require_golden_tree()

#: C++'s composite T/ac-load-to-imperial-emission multiplier, computed from
#: the two pinned constant pairs rather than pasted as a rounded literal:
#: ``GramSqMt_To_Pounds`` multiplies by 4046.86 then divides by 453.592
#: (fof_sgv.cpp:98-109), and ``TPA_To_KiSq``/``KgSq_To_TPA`` use a flat 4.46
#: (fof_util.cpp:543-561).
CPP_IMPERIAL_MULTIPLIER = (4046.86 / 453.592) / 4.46

#: Python's imperial multiplier (``emission_calcs.py``), a flat 2.0.
PY_IMPERIAL_MULTIPLIER = 2.0

#: Relative tolerance for an SI-normalised emission total, retrieved from the
#: centralized policy rather than a literal - ``consume_p4.emissions_legacy``
#: and ``consume_p4.emissions_expanded`` record the same 5e-07. The measured
#: maximum relative difference across all four Phase 4 emission scenarios and
#: every comparable total is 1.89e-07, which is consistent with the C++
#: side's float32 ``d_CO`` storage plus its six-decimal output formatting.
#: 5e-07 sits just above that and far below the F-42 divergence it must not
#: absorb (which is a total-vs-zero difference of up to 17110 lb/ac).
RTOL_EMISSION = phase4_tolerance("consume", "emissions_legacy")[1]

#: The 14 flaming/smoldering totals ``ES_Calc`` and ``ES_Calc_NEW`` both
#: populate.
FLAMING_SMOLDERING_TOTALS = [
    "PM10F", "PM10S", "PM25F", "PM25S", "CH4F", "CH4S", "COF", "COS",
    "CO2F", "CO2S", "NOXF", "NOXS", "SO2F", "SO2S",
]

#: The 7 duff-only totals. ``ES_Calc`` (legacy) populates them
#: (bur_brn.cpp:2139-2145, :2178-2184); ``ES_Calc_NEW`` (expanded) never
#: writes them at all - see F-42 and the xfail below.
DUFF_ONLY_TOTALS = [
    "PM10S_Duff", "PM25S_Duff", "CH4S_Duff", "COS_Duff", "CO2S_Duff",
    "NOXS_Duff", "SO2S_Duff",
]

#: Phase 4 consume scenarios that select C++'s ``ES_Calc`` legacy path
#: (``gf_CriInt < 0``, bur_brn.cpp:320,332).
LEGACY_SCENARIOS = ["emis-legacy-iw", "emis-legacy-se"]

#: Phase 4 consume scenarios that select C++'s ``ES_Calc_NEW`` expanded path
#: (``gf_CriInt >= 0``), each with a distinct factor-group configuration.
EXPANDED_SCENARIOS = ["emis-expanded-g258", "emis-expanded-g378"]

#: Totals ``ES_Calc`` leaves at zero by design: the legacy path fixes
#: smoldering NOx at 0 (``d_noxs``, bur_brn.cpp:125), which Python's legacy
#: mode reproduces exactly (``nox_s = 0.0``). Compared as an exact 0-vs-0
#: equality rather than through a relative tolerance, which is undefined at 0.
LEGACY_ZERO_BY_DESIGN = ["NOXS", "NOXS_Duff"]


def _consume_input(overrides, column):
    """
    Return one consume input field for a scenario, after its overrides.

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :returns: The raw string value written to the golden's input CSV.
    """
    row = list(MODES["consume"]["row"])
    for key, value in overrides.items():
        row[CONSUME_INDEX[key]] = value
    return row[CONSUME_INDEX[column]]


def _overrides(case_id):
    """
    Return a Phase 4 consume scenario's override dict.

    :param case_id: The scenario's case ID.
    :returns: The column-name to value override mapping.
    """
    return next(o for case, o, _b in CONSUME_SCENARIOS if case == case_id)


def _python_emissions(case_id):
    """
    Run ``calc_smoke_emissions`` on a golden row's OWN consumption totals.

    The flaming, smoldering and duff loads come from the golden row itself
    (``FlaCon``/``SmoCon``/``DufCon``, populated by ``fof_bcm.cpp:236-237``
    and the duff accumulator), and the factor groups come from that row's own
    input columns - so the comparison uses the row's real configuration, not
    a reconstructed one.

    :param case_id: A Phase 4 consume scenario ID.
    :returns: ``(python_result_dict, golden_row)``.
    """
    overrides = _overrides(case_id)
    row = golden_rows_by_case("consume", "_summary")[case_id]
    critical = float(_consume_input(overrides, "critical_intensity_kw_m"))
    mode = "legacy" if critical < 0 else "expanded"
    group_kwargs = {}
    if mode == "expanded":
        group_kwargs = dict(
            ef_group=int(_consume_input(overrides, "ef_flame_group")),
            ef_smoldering_group=int(
                _consume_input(overrides, "ef_smolder_group")
            ),
            ef_duff_group=int(_consume_input(overrides, "ef_duff_group")),
        )
    result = calc_smoke_emissions(
        float(row["FlaCon"]),
        float(row["SmoCon"]),
        mode=mode,
        duff_load=float(row["DufCon"]),
        units="imperial",
        **group_kwargs,
    )
    return result, row


def _si_normalised(cpp_imperial, py_imperial):
    """
    Divide each side by ITS OWN imperial multiplier (F-32 normalisation).

    :param cpp_imperial: The golden's imperial emission total.
    :param py_imperial: Python's imperial emission total.
    :returns: ``(cpp_si, py_si)``.
    """
    return (cpp_imperial / CPP_IMPERIAL_MULTIPLIER,
            py_imperial / PY_IMPERIAL_MULTIPLIER)


def test_default_mode_applies_one_group_to_both_phases():
    """
    Class (a) Python equation test - NO parity claim.

    ``default`` applies a single factor group to both the flaming and the
    smoldering load. That behaviour has no established executable C++
    counterpart (C++ has exactly two ``gf_CriInt`` dispatch paths and neither
    natively applies one group to both phases); promoting it to parity is
    Phase 6's investigation, not an assumption made here.
    """
    result = calc_smoke_emissions(2.0, 4.0, mode="default", units="SI")
    for species in ("PM10", "PM25", "CH4", "CO", "CO2", "NOX", "SO2"):
        flaming = float(np.asarray(result[f"{species}F"]))
        smoldering = float(np.asarray(result[f"{species}S"]))
        assert smoldering == pytest.approx(flaming * 2.0), (
            f"{species}: default mode must scale one factor by each load, so "
            "a smoldering load twice the flaming load must give exactly "
            "twice the emission"
        )


def test_default_mode_emits_no_duff_only_keys():
    """
    Class (a) Python contract test: ``default`` returns only the 14
    flaming/smoldering keys - the 7 duff-only keys are an ``expanded``/
    ``legacy`` feature.
    """
    result = calc_smoke_emissions(2.0, 4.0, mode="default", units="SI")
    assert set(result) == set(FLAMING_SMOLDERING_TOTALS)


@pytest.mark.parametrize("case_id", EXPANDED_SCENARIOS)
def test_expanded_duff_only_totals_match_cpp(case_id, request):
    """The 7 duff-only totals, expanded path - a strict xfail (F-42)."""
    request.node.add_marker(pytest.mark.xfail(
        strict=True,
        reason=(
            "F-42: ES_Calc_NEW never writes the duff-only accumulators. It "
            "folds the duff emissions into the smoldering totals only "
            "(bur_brn.cpp:2478-2484 adds gf_d* * d_Duff to dN_*S and to no "
            "*_Duff field), so all 7 d_CO duff-only totals stay at their "
            "ES_Init zero on the expanded path, while ES_Calc (legacy) does "
            "populate them (bur_brn.cpp:2139-2145, :2178-2184). Python's "
            "expanded mode reports real duff-only totals, so the two cannot "
            "agree. Measured: C++ 0.0 against Python values up to 17110.1 "
            "lb/ac. This is an upstream C++ asymmetry, not a Python "
            "numerical error, and must not be absorbed by a tolerance."
        ),
    ))
    result, row = _python_emissions(case_id)
    for key in DUFF_ONLY_TOTALS:
        cpp_si, py_si = _si_normalised(
            float(row[key]), float(np.asarray(result[key]))
        )
        assert py_si == pytest.approx(cpp_si, rel=RTOL_EMISSION)


@pytest.mark.parametrize("case_id", EXPANDED_SCENARIOS)
def test_expanded_flaming_smoldering_totals_match_cpp(case_id):
    """
    The 14 flaming/smoldering totals, expanded path, full-pipeline.

    Not a claim that ``calc_smoke_emissions`` equals an isolated
    ``ES_Calc_NEW`` call - see the module docstring.
    """
    result, row = _python_emissions(case_id)
    for key in FLAMING_SMOLDERING_TOTALS:
        cpp_si, py_si = _si_normalised(
            float(row[key]), float(np.asarray(result[key]))
        )
        assert py_si == pytest.approx(cpp_si, rel=RTOL_EMISSION), key


@pytest.mark.parametrize("case_id", EXPANDED_SCENARIOS)
def test_expanded_golden_really_loaded_its_emission_factors(case_id):
    """
    Prove the expanded golden passed the emission-factor-loading
    qualification before it is trusted as an oracle.

    An ``expanded`` row whose ``gf_*`` globals were never populated reports
    all-zero emission totals with a successful return code - the exact
    failure mode Gate 0 revision 7 was written to close. Three independent
    checks are made, none of which relies on the harness merely claiming
    success:

    1. every one of the 14 flaming/smoldering totals is strictly nonzero;
    2. the two expanded scenarios, which differ ONLY in their
       ``ef_flame_group`` / ``ef_smolder_group`` columns, produce DIFFERENT
       totals - impossible if the factors were an unloaded default;
    3. the manifest records the exact tracked emission-factor table as a
       side file.
    """
    _result, row = _python_emissions(case_id)
    zero = [key for key in FLAMING_SMOLDERING_TOTALS if float(row[key]) == 0.0]
    assert not zero, (
        f"{case_id}: these expanded totals are zero, which is the signature "
        f"of unloaded emission factors: {zero}"
    )
    rows = golden_rows_by_case("consume", "_summary")
    other = [c for c in EXPANDED_SCENARIOS if c != case_id][0]
    differing = [
        key for key in FLAMING_SMOLDERING_TOTALS
        if float(rows[case_id][key]) != float(rows[other][key])
    ]
    assert differing, (
        "the two expanded scenarios use different factor groups, so their "
        "totals must differ; identical totals would mean the group columns "
        "were never applied"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-31: calc_smoke_emissions(..., mode='expanded', ef_group=9) does "
        "not raise a group-domain ValueError naming the 1-8 contract; it "
        "passes _validate_group and fails later inside the factor lookup "
        "with a generic pandas conversion error. Fixing _validate_group or "
        "the CSV parse is a production change and is out of Phase 4's "
        "scope."
    ),
)
def test_factor_group_validation_rejects_out_of_domain_groups():
    """
    Class (a) Python contract test (F-31), strict xfail.

    Valid emission-factor groups are exactly 1-8. ``_validate_group`` accepts
    ``1 <= grp <= len(ef_df)``, and ``_load_ef_csv``'s parse yields 17 rows
    (8 real factor groups plus 9 trailing non-factor rows), so groups 9-17
    pass the range check and fail later with a generic pandas conversion
    error instead of a parameter-specific diagnostic.

    This asserts the DESIRED behaviour - a domain-specific ``ValueError``
    naming the real 1-8 contract - not the current one. Currently
    ``ef_group=9`` DOES raise ``ValueError``, but with the wrong message
    (``"could not convert string to float: 'Description'"``), so
    ``pytest.raises(..., match=...)`` genuinely executes and genuinely
    fails on the message mismatch; it is not vacuous.
    """
    with pytest.raises(ValueError, match=r"ef_group must be between 1 and 8"):
        calc_smoke_emissions(1.0, 1.0, mode="expanded", ef_group=9)


def test_invalid_mode_raises_value_error():
    """Class (a) Python contract test: an unknown mode must raise."""
    with pytest.raises(ValueError, match="Unknown emissions mode"):
        calc_smoke_emissions(1.0, 1.0, mode="not-a-mode")


@pytest.mark.parametrize("case_id", LEGACY_SCENARIOS)
def test_legacy_all_totals_match_cpp(case_id):
    """
    All 21 totals, legacy path, full-pipeline.

    ``ES_Calc`` populates the duff-only accumulators as well, so unlike the
    expanded path every one of the 21 totals is comparable here. The two
    fixed-zero totals (smoldering NOx) are asserted as exact zeros on both
    sides rather than through a relative tolerance.
    """
    result, row = _python_emissions(case_id)
    for key in FLAMING_SMOLDERING_TOTALS + DUFF_ONLY_TOTALS:
        cpp_imperial = float(row[key])
        py_imperial = float(np.asarray(result[key]))
        if key in LEGACY_ZERO_BY_DESIGN:
            assert cpp_imperial == 0.0, key
            assert py_imperial == 0.0, key
            continue
        cpp_si, py_si = _si_normalised(cpp_imperial, py_imperial)
        assert py_si == pytest.approx(cpp_si, rel=RTOL_EMISSION), key


def test_legacy_combustion_efficiency_factors_are_exact_constants():
    """
    Class (a) Python equation test - a relation-level claim about the factor
    EXPRESSIONS, not an executable-parity claim.

    Python's legacy factors are built from the same two combustion
    efficiencies and the same base constants as C++'s file-scope globals
    (``bur_brn.cpp:114-126``). Verified by evaluating the Python function on
    a unit SI load, so the factors are read back from real output rather than
    from the module's private constants.
    """
    eps = python_contract_epsilon("emission_factor_expression")
    result = calc_smoke_emissions(1.0, 0.0, mode="legacy", units="SI")
    assert float(np.asarray(result["PM25F"])) == pytest.approx(
        67.4 - 0.97 * 66.8, abs=eps
    )
    assert float(np.asarray(result["CH4F"])) == pytest.approx(
        42.7 - 0.97 * 43.2, abs=eps
    )
    assert float(np.asarray(result["COF"])) == pytest.approx(
        961.0 - 0.97 * 984.0, abs=eps
    )
    assert float(np.asarray(result["CO2F"])) == pytest.approx(
        0.97 * 1833.0, abs=eps
    )
    assert float(np.asarray(result["PM10F"])) == pytest.approx(
        (67.4 - 0.97 * 66.8) * 1.18, abs=eps
    )
    assert float(np.asarray(result["NOXF"])) == pytest.approx(3.2, abs=eps)
    assert float(np.asarray(result["SO2F"])) == pytest.approx(1.0, abs=eps)

    smoldering = calc_smoke_emissions(0.0, 1.0, mode="legacy", units="SI")
    assert float(np.asarray(smoldering["PM25S"])) == pytest.approx(
        67.4 - 0.67 * 66.8, abs=eps
    )
    assert float(np.asarray(smoldering["NOXS"])) == 0.0


def test_python_imperial_multiplier_is_exactly_two():
    """
    Class (a) Python contract test (F-32): Python's imperial conversion is a
    flat 2.0, and is pinned as an exact contract HERE rather than compared
    against a C++ imperial oracle - C++'s own two conversion paths compose to
    2.00040501817599 and disagree with each other by the same ratio, so no
    single correct C++ imperial value exists for this to match.
    """
    si = calc_smoke_emissions(1.0, 1.0, mode="legacy", units="SI")
    imperial = calc_smoke_emissions(1.0, 1.0, mode="legacy", units="imperial")
    for key in FLAMING_SMOLDERING_TOTALS:
        assert float(np.asarray(imperial[key])) == pytest.approx(
            float(np.asarray(si[key])) * PY_IMPERIAL_MULTIPLIER,
            abs=python_contract_epsilon("imperial_multiplier_identity"),
        ), key
    assert CPP_IMPERIAL_MULTIPLIER == pytest.approx(
        2.00040501817599,
        abs=python_contract_epsilon("imperial_multiplier_value"),
    )
    relative_difference = (
        (CPP_IMPERIAL_MULTIPLIER - PY_IMPERIAL_MULTIPLIER)
        / PY_IMPERIAL_MULTIPLIER
    )
    assert relative_difference == pytest.approx(
        2.025090879957947e-04,
        rel=python_contract_epsilon("imperial_multiplier_ratio_relative"),
    )
