#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase7_run_burnup_parity.py - Phase 7 item E: Python-vs-C++ numeric
parity for ``run_fofem_emissions(use_burnup=True, ...)`` (which drives
``run_burnup()``/``burnup()`` internally) against the 4 manifested Phase
7 ``consume`` golden scenarios in
``tests/test_data/test_golden_output/phase7/consume/`` - real class (c)
executable C++ parity, not a source-relation or contract-only check.

**Phase 7 correction pass (2026-09-06), items 2/3: full scientific-column
coverage, corrected root-cause attribution.** The original Phase 7 pass
compared only ``FlaDur`` and mis-attributed the ``rot-snd-mix``
scenario's own severe ``SndDW1kCon``/``RotDW1kCon`` divergence to F-23
(a Northeast-only finding that does not apply to these
``region=InteriorWest`` scenarios at all). Direct Python/C++ tracing
found the REAL root causes were two separate, unrelated test-design
issues, both now fixed:

1. **F-60**: ``run_fofem_emissions()``'s AGGREGATE ``dw1000s``/
   ``dw1000r`` parameters silently route the entire load into the
   SMALLEST (3-6 in) granular sub-class by default
   (``pyfofem.py:461,463``), not whichever class the caller actually
   intends. The golden's ``snd_dw20_tac``/``rot_dw20_tac`` columns
   populate the 20+ in class specifically - :func:`_run_fofem_emissions_for_case`
   now passes the CORRECT granular ``dw3_6s``/``dw6_9s``/``dw9_20s``/
   ``dw20s``/``dw3_6r``/``dw6_9r``/``dw9_20r``/``dw20r`` parameters,
   read directly from the golden's own ``snd_dw*_tac``/``rot_dw*_tac``
   columns (the harness itself never uses the aggregate
   ``dw1000_tac``/``pct_rot`` columns for anything - confirmed by a
   direct grep of ``fof_bcm.cpp``/``fof_cm.cpp``, which read only the
   8 granular ``f_Snd_DW*``/``f_Rot_DW*`` fields).
2. **F-61**: ``run_fofem_emissions()`` hardcodes
   ``consm_duff(duff_moist_cat='edm')`` (C++'s ``ENTIRE`` route) with
   no caller override, but the canonical base row all 4 Phase 7
   scenarios were built from (inherited unchanged, never a deliberate
   target of any of them) used ``duff_moist_method=NFDR`` - comparing
   an NFDR-generated golden against an ENTIRE-computed Python call was
   never a fair comparison. Each scenario's own ``duff_moist_method``
   column was corrected to ``ENTIRE`` in ``_phase7_contract.py`` (a
   golden-scenario-input correction, not a production or tolerance
   change) so ``DufCon`` and everything it feeds into are directly
   comparable for the first time.

With both fixed, ``SndDW1kCon``/``RotDW1kCon``/``DufCon`` all agree
closely with the golden - the divergence was NEVER a Python-vs-C++
scientific defect, let alone F-23. A genuinely new, real divergence
(**F-62**) was found while completing this expanded coverage: the
flaming/smoldering consumption-and-duration SPLIT (not the total)
diverges substantially for two of the four scenarios
(``hot-amb-duff``, ``long-igtime``), while the other two
(``rot-snd-mix``, ``calm-wind``) agree closely - pinned below by 6
separate parametrized strict xfails, not silently omitted.

**Phase 7 correction pass (2026-09-07), item 3**: the original text
here called ``hot-amb-duff``/``long-igtime`` "longer-duration
simulations" that "extend the effective simulated burn duration". That
causal claim is REFUTED by the committed golden's own ``FlaDur``/
``SmoDur`` values (direct executable evidence, no new instrumentation
needed): ``hot-amb-duff``'s total simulated duration
(``FlaDur+SmoDur`` = 60+2280 = 2340 s) is the SHORTEST of all 4
scenarios, not extended; ``long-igtime``'s total (199.9+2419.9 =
2619.8 s) is nearly identical to ``rot-snd-mix``'s nominal 2625 s -
only the flaming/smoldering split shifts, not the total duration. These
two scenarios are neutrally described below as "the two scenarios
exhibiting the measured split divergence" (chosen originally for their
near-upper-bound ``ambient_temp_c``/``ig_time_s`` fire-environment
values, not for any duration effect); the real mechanism remains
unidentified.

Every scientific column in the committed golden now has an explicit
``consume_p7.*`` tolerance-policy classification (see
``tolerance_policy.json``'s ``consume_p7`` section and
:func:`test_every_real_scientific_column_has_an_applicable_policy_route`
below) - not just ``FlaDur``. A classification is NOT the same claim as
an executed numeric comparison: ``duff_percent`` (no matching Python
field) and ``emissions_mismatched_group`` (intentionally mismatched
emission-factor groups) are ``status="unverified"`` in
``tolerance_policy.json`` and have no numeric comparison in this
module by design - only the ``verified``/``known_divergent_strict_xfail``
routes below are actually exercised numerically.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""

from __future__ import annotations

import csv
import io
import os

import pytest

from pyfofem import run_fofem_emissions
from tests.cpp_parity_live._output_contract import classify_columns, read_real_header
from tests.cpp_parity_live._phase7_contract import (
    golden_dir,
    phase7_tolerance,
    require_golden_tree,
)

require_golden_tree()
_IN_CSV_PATH = os.path.join(golden_dir("consume"), "consume_in.csv")
_SUMMARY_CSV_PATH = os.path.join(golden_dir("consume"), "consume_summary.csv")

#: Every column-value pair this module derives (Python has no native
#: output for these); computed once per call by :func:`_derive_totals`
#: and :func:`_run_fofem_emissions_for_case`'s own docstring explains why.
_COMPONENT_PRE_KEYS = (
    "LitPre", "DW1Pre", "DW10Pre", "DW100Pre", "DW1kSndPre", "DW1kRotPre",
    "DufPre", "HerPre", "ShrPre", "FolPre", "BraPre",
)
_COMPONENT_CON_KEYS = (
    "LitCon", "DW1Con", "DW10Con", "DW100Con", "DW1kSndCon", "DW1kRotCon",
    "DufCon", "HerCon", "ShrCon", "FolCon", "BraCon",
)
_COMPONENT_POS_KEYS = (
    "LitPos", "DW1Pos", "DW10Pos", "DW100Pos", "DW1kSndPos", "DW1kRotPos",
    "DufPos", "HerPos", "ShrPos", "FolPos", "BraPos",
)


def _derive_totals(result: dict) -> dict:
    """
    Derive the aggregate ``TotPre``/``TotCon``/``TotPos`` values
    ``run_fofem_emissions()`` does not expose natively, by summing the
    11 component fields it does expose - confirmed to reproduce the
    golden's own ``TotPre`` to the last decimal in all 4 scenarios (see
    the ``totals`` tolerance-policy route's justification).

    :param result: A ``run_fofem_emissions()`` result dict.
    :return: ``{"TotPre": ..., "TotCon": ..., "TotPos": ...}``.
    """
    return {
        "TotPre": sum(result[k] for k in _COMPONENT_PRE_KEYS),
        "TotCon": sum(result[k] for k in _COMPONENT_CON_KEYS),
        "TotPos": sum(result[k] for k in _COMPONENT_POS_KEYS),
    }


def _golden_summary_row(case_id: str) -> dict:
    """
    Read one committed Phase 7 ``consume_summary.csv`` row by case id.

    :param case_id: The scenario's ``case_id`` value.
    :return: ``{column_name: raw_string_value}`` for that row.
    :raises KeyError: If *case_id* is not present in the committed golden.
    """
    with open(_SUMMARY_CSV_PATH, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["case_id"] == case_id:
                return row
    raise KeyError(f"case_id {case_id!r} not found in {_SUMMARY_CSV_PATH}")


def _read_input_row(case_id: str) -> dict:
    """
    Read one committed Phase 7 ``consume_in.csv`` row by case id.

    :param case_id: The scenario's ``case_id`` value.
    :return: ``{column_name: raw_string_value}`` for that row.
    :raises KeyError: If *case_id* is not present in the committed input.
    """
    with open(_IN_CSV_PATH, encoding="utf-8", newline="") as fh:
        lines = fh.readlines()
    # The harness prepends a "#fofem-harness,consume,1" metadata line before
    # the real CSV header - skip it if present, matching _phase7_contract.py's
    # own golden_rows() convention for output files (which have no such
    # metadata line) generalized to the one input file that does.
    start = 1 if lines and lines[0].startswith("#fofem-harness") else 0
    reader = csv.DictReader(io.StringIO("".join(lines[start:])))
    for row in reader:
        if row["case_id"] == case_id:
            return row
    raise KeyError(f"case_id {case_id!r} not found in {_IN_CSV_PATH}")


def _run_fofem_emissions_for_case(case_id: str) -> dict:
    """
    Call ``run_fofem_emissions(use_burnup=True, ...)`` with the exact same
    fire-environment/fuel inputs the committed golden's own input row for
    *case_id* used.

    Per F-60: the harness's ``consume`` mode never reads the aggregate
    ``dw1000_tac``/``pct_rot`` input columns for anything (confirmed by
    direct inspection of ``fof_bcm.cpp``/``fof_cm.cpp`` - only the 8
    granular ``f_Snd_DW3/6/9/20``/``f_Rot_DW3/6/9/20`` fields are read),
    so this helper maps the golden's granular ``snd_dw*_tac``/
    ``rot_dw*_tac`` columns DIRECTLY onto Python's own granular
    ``dw3_6s``/``dw6_9s``/``dw9_20s``/``dw20s``/``dw3_6r``/``dw6_9r``/
    ``dw9_20r``/``dw20r`` parameters, 1:1, with no aggregate fallback of
    any kind - the aggregate columns are read from the golden row for
    completeness/documentation only and never used.

    :param case_id: The scenario's ``case_id`` value.
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    row = _read_input_row(case_id)
    return run_fofem_emissions(
        litter=float(row["litter_tac"]),
        duff=float(row["duff_tac"]),
        duff_depth=float(row["duff_depth_in"]),
        herb=float(row["herb_tac"]),
        shrub=float(row["shrub_tac"]),
        crown_foliage=float(row["crown_fol_tac"]),
        crown_branch=float(row["crown_bra_tac"]),
        pct_crown_burned=float(row["pct_crown_burn"]),
        region=row["region"],
        cvr_grp=row["cover_group"],
        season=row["season"],
        fuel_category=row["fuel_cat"],
        duff_moist=float(row["duff_moist_pct"]),
        l_moist=float(row["litter_moist_pct"]),
        dw10_moist=float(row["dw10_moist_pct"]),
        dw1000_moist=float(row["dw1000_moist_pct"]),
        dw1=float(row["dw1_tac"]),
        dw10=float(row["dw10_tac"]),
        dw100=float(row["dw100_tac"]),
        dw3_6s=float(row["snd_dw3_tac"]),
        dw6_9s=float(row["snd_dw6_tac"]),
        dw9_20s=float(row["snd_dw9_tac"]),
        dw20s=float(row["snd_dw20_tac"]),
        dw3_6r=float(row["rot_dw3_tac"]),
        dw6_9r=float(row["rot_dw6_tac"]),
        dw9_20r=float(row["rot_dw9_tac"]),
        dw20r=float(row["rot_dw20_tac"]),
        hfi=float(row["intensity_kw_m"]),
        flame_res_time=float(row["ig_time_s"]),
        fuel_bed_depth=float(row["depth_ft"]) * 0.3048,
        ambient_temp=float(row["ambient_temp_c"]),
        windspeed=float(row["windspeed_m_s"]),
        use_burnup=True,
        units="Imperial",
    )


def test_calm_wind_fladur_matches_cpp():
    """``calm-wind`` (``windspeed_m_s=0``, exact lower bound of
    ``_FIRE_BOUNDS['u']``, BR-BRN-NOMINAL): Python's ``FlaDur`` must
    match the golden's within the established ``consume_p7`` tolerance."""
    golden = _golden_summary_row("calm-wind")
    result = _run_fofem_emissions_for_case("calm-wind")
    atol, _rtol = phase7_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_derived_totals_match_cpp():
    """The ``totals`` route: Python's DERIVED ``TotPre``/``TotCon``/
    ``TotPos`` (see :func:`_derive_totals`) match the golden's real
    ``TotPre``/``TotCon``/``TotPos`` columns across all 4 scenarios,
    confirming the derivation formula itself and that the small
    per-component woody-fuel scheme differences do not compound into a
    larger aggregate divergence."""
    atol, _rtol = phase7_tolerance("consume", "totals")
    for case_id in ("rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        totals = _derive_totals(result)
        for key in ("TotPre", "TotCon", "TotPos"):
            assert totals[key] == pytest.approx(float(golden[key]), abs=atol), (
                case_id, key,
            )


def test_duff_matches_cpp_with_the_corrected_entire_duff_moisture_method():
    """The ``duff`` route (F-61 corrected): with each scenario's
    ``duff_moist_method`` corrected to ``ENTIRE`` (matching what
    ``run_fofem_emissions()`` always computes), ``DufPre``/``DufCon``/
    ``DufPos`` are an exact match and ``DufDepCon``/``DufDepPos`` are
    within the established tolerance, across all 4 scenarios."""
    atol, _rtol = phase7_tolerance("consume", "duff")
    for case_id in ("rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        for col in ("DufPre", "DufCon", "DufPos", "DufDepPre", "DufDepCon", "DufDepPos"):
            assert result[col] == pytest.approx(float(golden[col]), abs=atol), (
                case_id, col,
            )


def test_dw100_matches_cpp_within_the_established_scheme_tolerance():
    """The ``dw100`` route: 100-hr woody fuel IS consumed inside Burnup's
    per-particle simulation, so a small, real, scheme-level difference
    is expected and bounded by real measured evidence across all 4
    scenarios."""
    atol, _rtol = phase7_tolerance("consume", "dw100")
    for case_id in ("rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        for gcol, pkey in (("DW100Pre", "DW100Pre"), ("DW100Con", "DW100Con"),
                           ("DW100Pos", "DW100Pos")):
            assert result[pkey] == pytest.approx(float(golden[gcol]), abs=atol), (
                case_id, gcol,
            )


def test_every_real_scientific_column_has_an_applicable_policy_route():
    """
    Completeness meta-test (Phase 7 correction pass item 2): the UNION of
    every ``consume_p7.*`` route's ``covers_columns`` (read directly from
    ``tolerance_policy.json``, not a second hardcoded list) must equal
    EXACTLY the real scientific column set of the committed
    ``consume_summary.csv`` header (derived via
    ``_output_contract.classify_columns()``, the same machinery Phase
    2/4's own completeness tests use) - no gap, no stale/extra entry.
    """
    import json

    policy_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cpp_parity_live",
        "tolerance_policy.json",
    )
    with open(policy_path, encoding="utf-8") as handle:
        policy = json.load(handle)

    covered = set()
    for entry in policy["consume_p7"].values():
        covered.update(entry["covers_columns"])

    header = read_real_header(_SUMMARY_CSV_PATH)
    _metadata, scientific = classify_columns("consume", "_summary", header)
    scientific = set(scientific)

    assert covered == scientific, (
        f"missing from consume_p7 coverage={scientific - covered}, "
        f"stale/extra in consume_p7 coverage={covered - scientific}"
    )


def test_flame_smolder_split_matches_cpp_for_nominal_duration_scenarios():
    """The ``flame_smolder_consumption_nominal``/``smoldering_duration_
    nominal`` routes: for the two scenarios that do NOT exhibit the
    measured split divergence (``rot-snd-mix``, ``calm-wind``),
    ``FlaCon``/``SmoCon``/``SmoDur`` all agree within the established
    tolerances. See
    :func:`test_flame_smolder_split_should_match_cpp_for_the_affected_scenarios`
    for the SEPARATE, substantially larger divergence measured for the
    other two scenarios (F-62)."""
    con_atol, _ = phase7_tolerance("consume", "flame_smolder_consumption_nominal")
    dur_atol, _ = phase7_tolerance("consume", "smoldering_duration_nominal")
    for case_id in ("rot-snd-mix", "calm-wind"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        assert result["FlaCon"] == pytest.approx(float(golden["FlaCon"]), abs=con_atol)
        assert result["SmoCon"] == pytest.approx(float(golden["SmoCon"]), abs=con_atol)
        assert result["SmoDur"] == pytest.approx(float(golden["SmoDur"]), abs=dur_atol)


#: Phase 7 correction pass (2026-09-07) item 2: the prior single xfail
#: looped over 2 scenarios x 3 fields and stopped at the FIRST failing
#: assertion (hot-amb-duff/FlaCon), so 5 of the 6 claimed divergent
#: combinations never actually executed. Parametrizing both dimensions
#: gives each of the 6 (case_id, field) combinations its own
#: independently-collected, independently-executed strict xfail node -
#: every combination genuinely runs and genuinely fails under
#: ``--runxfail``, verified directly (see the correction pass's final
#: report).
_AFFECTED_SCENARIO_IDS = ("hot-amb-duff", "long-igtime")
_SPLIT_FIELDS = ("FlaCon", "SmoCon", "SmoDur")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-62 (gate0/04-findings.md): the flaming/smoldering consumption-"
        "and-duration SPLIT diverges substantially for hot-amb-duff/"
        "long-igtime (the two scenarios exhibiting the measured split "
        "divergence), even though the underlying TotCon these two fields "
        "split does not. Root cause not fully isolated within this pass's "
        "scope; pinned as a desired-behavior xfail, not silently omitted. "
        "See tolerance_policy.json's flame_smolder_consumption_affected_"
        "scenarios/smoldering_duration_affected_scenarios routes for the "
        "measured evidence, including direct executable evidence that "
        "refutes the original 'extends the effective simulated burn "
        "duration' causal hypothesis for both scenarios."
    ),
)
@pytest.mark.parametrize("field", _SPLIT_FIELDS)
@pytest.mark.parametrize("case_id", _AFFECTED_SCENARIO_IDS)
def test_flame_smolder_split_should_match_cpp_for_the_affected_scenarios(case_id, field):
    """Desired-behavior pin for F-62: *field* (``FlaCon``/``SmoCon``/
    ``SmoDur``) SHOULD match the golden within the same tolerance the
    nominal-duration scenarios achieve (see
    :func:`test_flame_smolder_split_matches_cpp_for_nominal_duration_scenarios`),
    for *case_id* (``hot-amb-duff``/``long-igtime``) too. Currently fails
    for all 6 (case_id, field) combinations - every one exceeds its
    applicable established tolerance (0.1 T/ac for ``FlaCon``/``SmoCon``,
    60.0 s for ``SmoDur``), though not uniformly by the same margin:
    ``hot-amb-duff``'s ``FlaCon``/``SmoCon`` diverge by roughly two
    orders of magnitude versus the nominal-duration scenarios' own
    measured divergence, while ``long-igtime``'s ``SmoDur`` diverges by
    less than one order of magnitude - see the tolerance-policy routes'
    justifications in ``tolerance_policy.json`` for the exact measured
    value of each of the 6 combinations, not a single characterization
    applied uniformly to all of them."""
    if field == "SmoDur":
        atol, _ = phase7_tolerance("consume", "smoldering_duration_nominal")
    else:
        atol, _ = phase7_tolerance("consume", "flame_smolder_consumption_nominal")
    golden = _golden_summary_row(case_id)
    result = _run_fofem_emissions_for_case(case_id)
    assert result[field] == pytest.approx(float(golden[field]), abs=atol)


def test_hot_amb_duff_fladur_matches_cpp():
    """``hot-amb-duff`` (``ambient_temp_c=39.9``, just inside the upper
    bound of ``_FIRE_BOUNDS['tamb_c']``, with duff present,
    BR-BRN-NOMINAL): Python's ``FlaDur`` must match the golden's
    within the established ``consume_p7`` tolerance."""
    golden = _golden_summary_row("hot-amb-duff")
    result = _run_fofem_emissions_for_case("hot-amb-duff")
    atol, _rtol = phase7_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_long_igtime_fladur_matches_cpp():
    """``long-igtime`` (``ig_time_s=199.9``, just inside the upper bound
    of ``_FIRE_BOUNDS['ti']``, BR-BRN-NOMINAL): Python's ``FlaDur``
    must match the golden's within the established ``consume_p7``
    tolerance."""
    golden = _golden_summary_row("long-igtime")
    result = _run_fofem_emissions_for_case("long-igtime")
    atol, _rtol = phase7_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_mineral_soil_exposure_matches_cpp():
    """The ``mineral_soil`` route: ``MSE`` matches to floating-point
    noise across all 4 scenarios - ``fuel_cat=Natural``/
    ``cover_group=GrassGroup`` routes through the same equation path
    regardless of the fire-environment/woody-fuel variations these 4
    scenarios exercise."""
    atol, _rtol = phase7_tolerance("consume", "mineral_soil")
    for case_id in ("rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        assert result["MSE"] == pytest.approx(float(golden["MSE"]), abs=atol)


def test_python_does_not_expose_a_percent_duff_consumed_field():
    """The ``duff_percent`` route: ``run_fofem_emissions()`` has no
    ``DufPer``-equivalent key in its result dict at all - confirmed
    directly rather than assumed. ``DufPer`` is trivially derivable from
    ``DufCon``/``DufPre`` (both independently verified to match exactly
    - see :func:`test_duff_matches_cpp_with_the_corrected_entire_duff_moisture_method`),
    so no separate numeric comparison is attempted or needed."""
    result = _run_fofem_emissions_for_case("rot-snd-mix")
    assert not any(
        key.lower() in ("dufper", "duf_per", "duf_pct", "dufpct")
        for key in result
    )


def test_rot_snd_mix_fladur_matches_cpp():
    """``rot-snd-mix`` (granular sound+rotten 1000hr particles coexisting
    in one ``burnup()`` run, BR-BUP-ROT-SND-MIX): Python's ``FlaDur`` must
    match the golden's within the established ``consume_p7`` tolerance."""
    golden = _golden_summary_row("rot-snd-mix")
    result = _run_fofem_emissions_for_case("rot-snd-mix")
    atol, _rtol = phase7_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_rot_snd_mix_woody_1000hr_matches_cpp_with_correct_size_class_mapping():
    """The ``woody_1000hr`` route (F-60 corrected): using the CORRECT
    granular ``dw20s``/``dw20r`` parameters (matching the golden's own
    populated ``snd_dw20_tac``/``rot_dw20_tac`` columns 1:1, instead of
    the aggregate ``dw1000s``/``dw1000r`` parameters an earlier pass
    wrongly used, which silently routed the entire load into the WRONG,
    much-higher-SAV 3-6 in size class), ``SndDW1kCon``/``RotDW1kCon``
    agree closely with the golden - the divergence this test module's
    predecessor measured (Python 0.706/1.051 vs golden 0.026/0.086 T/ac)
    was entirely a test-code mapping bug, not a Python-vs-C++ scientific
    defect, and was never F-23 (Northeast-only; this scenario is
    region=InteriorWest)."""
    golden = _golden_summary_row("rot-snd-mix")
    result = _run_fofem_emissions_for_case("rot-snd-mix")
    atol, _rtol = phase7_tolerance("consume", "woody_1000hr")
    for gcol, pkey in (
            ("SndDW1kPre", "DW1kSndPre"), ("SndDW1kCon", "DW1kSndCon"),
            ("SndDW1kPos", "DW1kSndPos"), ("RotDW1kPre", "DW1kRotPre"),
            ("RotDW1kCon", "DW1kRotCon"), ("RotDW1kPos", "DW1kRotPos"),
    ):
        assert result[pkey] == pytest.approx(float(golden[gcol]), abs=atol), gcol


def test_woody_1000hr_is_zero_on_both_sides_for_the_other_3_scenarios():
    """The ``woody_1000hr`` route's other half: ``calm-wind``/
    ``hot-amb-duff``/``long-igtime`` never populate the granular
    ``snd_dw*_tac``/``rot_dw*_tac`` columns, so both the golden and
    Python (fed the same all-zero granular inputs) report exactly zero
    1000-hr woody load/consumption/remainder - a trivial but real
    exact-match case, not merely assumed.

    Phase 7 correction pass (2026-09-07) item 5: previously only
    ``*Con`` was asserted here; ``*Pre``/``*Pos`` are now asserted too
    (exact zero on both sides), so all 6 ``woody_1000hr`` columns the
    tolerance-policy route claims to cover are genuinely exercised for
    all 4 scenarios, not just 3 of the 4 for a 2-column subset."""
    for case_id in ("calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        for gcol, pkey in (
                ("SndDW1kPre", "DW1kSndPre"), ("SndDW1kCon", "DW1kSndCon"),
                ("SndDW1kPos", "DW1kSndPos"), ("RotDW1kPre", "DW1kRotPre"),
                ("RotDW1kCon", "DW1kRotCon"), ("RotDW1kPos", "DW1kRotPos"),
        ):
            assert float(golden[gcol]) == 0.0, (case_id, gcol)
            assert result[pkey] == 0.0, (case_id, pkey)


def test_woody_small_and_nonburnup_components_match_cpp():
    """The ``woody_small`` route: litter/1-hr/10-hr woody/herb/shrub/
    crown-foliage/crown-branch pre/con/pos are computed by equation
    routes independent of Burnup's per-particle woody-fuel simulation,
    so they match exactly across all 4 scenarios regardless of the
    fire-environment/rotten-sound-mix variations those scenarios
    exercise."""
    atol, _rtol = phase7_tolerance("consume", "woody_small")
    columns = (
        "LitPre", "LitCon", "LitPos", "DW1Pre", "DW1Con", "DW1Pos",
        "DW10Pre", "DW10Con", "DW10Pos", "HerPre", "HerCon", "HerPos",
        "ShrPre", "ShrCon", "ShrPos", "FolPre", "FolCon", "FolPos",
        "BraPre", "BraCon", "BraPos",
    )
    for case_id in ("rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime"):
        golden = _golden_summary_row(case_id)
        result = _run_fofem_emissions_for_case(case_id)
        for col in columns:
            assert result[col] == pytest.approx(float(golden[col]), abs=atol), (
                case_id, col,
            )
