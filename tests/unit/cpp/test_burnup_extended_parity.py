#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_burnup_extended_parity.py - Phase 7 item E: Python-vs-C++ numeric
parity for ``run_fofem_emissions(use_burnup=True, ...)`` (which drives
``run_burnup()``/``burnup()`` internally) against the 4 manifested Phase
7 ``consume`` golden scenarios in
``tests/test_data/test_golden_output/burnup_extended/consume/`` - real class (c)
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
   column was corrected to ``ENTIRE`` in ``_burnup_extended_contract.py`` (a
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
``consume_burnup_extended.*`` tolerance-policy classification (see
``tolerance_policy.json``'s ``consume_burnup_extended`` section and
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
from tests.cpp_parity_live._burnup_extended_contract import (
    golden_dir,
    burnup_extended_tolerance,
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
    # the real CSV header - skip it if present, matching _burnup_extended_contract.py's
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
    match the golden's within the established ``consume_burnup_extended`` tolerance."""
    golden = _golden_summary_row("calm-wind")
    result = _run_fofem_emissions_for_case("calm-wind")
    atol, _rtol = burnup_extended_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_derived_totals_match_cpp():
    """The ``totals`` route: Python's DERIVED ``TotPre``/``TotCon``/
    ``TotPos`` (see :func:`_derive_totals`) match the golden's real
    ``TotPre``/``TotCon``/``TotPos`` columns across all 4 scenarios,
    confirming the derivation formula itself and that the small
    per-component woody-fuel scheme differences do not compound into a
    larger aggregate divergence."""
    atol, _rtol = burnup_extended_tolerance("consume", "totals")
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
    atol, _rtol = burnup_extended_tolerance("consume", "duff")
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
    atol, _rtol = burnup_extended_tolerance("consume", "dw100")
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
    every ``consume_burnup_extended.*`` route's ``covers_columns`` (read directly from
    ``tolerance_policy.json``, not a second hardcoded list) must equal
    EXACTLY the real scientific column set of the committed
    ``consume_summary.csv`` header (derived via
    ``_output_contract.classify_columns()``, the same machinery Phase
    2/4's own completeness tests use) - no gap, no stale/extra entry.
    """
    import json

    policy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "cpp_parity_live", "tolerance_policy.json",
    )
    with open(policy_path, encoding="utf-8") as handle:
        policy = json.load(handle)

    covered = set()
    for entry in policy["consume_burnup_extended"].values():
        covered.update(entry["covers_columns"])

    header = read_real_header(_SUMMARY_CSV_PATH)
    _metadata, scientific = classify_columns("consume", "_summary", header)
    scientific = set(scientific)

    assert covered == scientific, (
        f"missing from consume_burnup_extended coverage={scientific - covered}, "
        f"stale/extra in consume_burnup_extended coverage={covered - scientific}"
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
    con_atol, _ = burnup_extended_tolerance("consume", "flame_smolder_consumption_nominal")
    dur_atol, _ = burnup_extended_tolerance("consume", "smoldering_duration_nominal")
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
#:
#: F-62 partial-resolution pass (2026-09-21): ``("hot-amb-duff",
#: "SmoDur")`` was REMOVED from this list -- ``burnup()``'s termination
#: condition was missing a check for remaining duff mass (matching
#: pinned ``bur_brn.cpp:377-380``'s ``if (fi<=fimin) { if
#: (d_Duf_Tot!=0) continue; break; }``), cutting the simulation short
#: whenever wood/litter/herb-shrub fire intensity fizzled before duff
#: finished burning. Fixed; ``hot-amb-duff``'s ``SmoDur`` matches the
#: golden exactly (2280.0 s).
#:
#: F-62 completion/acceptance-recovery pass (2026-09-21, same day):
#: ``FlaCon``/``SmoCon`` for BOTH scenarios are ALSO now removed. Root
#: cause found via a live C++ diagnostic-observer build
#: (``bur_brn_instr.cpp``, dumping ``Start()``'s own per-(k,l) ``wodot``
#: values and ``FireIntensity()``'s first-call ``wdotk``/``term`` side
#: by side): C++'s ``gd_Fudge1``/``gd_Fudge2`` (``bur_brn.cpp`` ``Start()``
#: ~line 553-560) are a SINGLE shared pair of scratch slots, not one per
#: fuel class -- ``gd_Fudge2`` gets overwritten by ANY ``kl!=0`` pair
#: that fully consumes during the ignition pulse, so it frequently ends
#: up holding a DIFFERENT class's rate than litter's own, which
#: ``FireIntensity()`` then restores into litter's own ``wodot[1]``
#: slot regardless. Python previously approximated this ("achieves the
#: same effect" per the removed comment in ``burnup.py``) by leaving
#: EVERY fully-consumed particle's OWN rate non-zero at its OWN index --
#: not equivalent, and it inflated litter's ``wdotk``/``term`` roughly
#: 18x for this scenario (measured directly: C++ term=7.1813 vs the
#: pre-fix Python term=132.036 at the first classification call),
#: causing litter (and sometimes other classes) to be misclassified as
#: flaming when C++ classifies smolder. Fixed in ``burnup.py`` by
#: replicating the exact 2-slot semantics (``fudge1``/``fudge2``
#: closure variables, set only on ``dnext<=0.0`` exactly as C++ does,
#: restored into ``wodot[0]``/``wodot[1]`` once inside
#: ``_fire_intensity()``). Both scenarios' ``FlaCon``/``SmoCon`` now
#: match the golden to float precision (see
#: ``test_flame_smolder_split_should_match_cpp_for_the_affected_scenarios``'s
#: own docstring for the exact measured values); fixing this also
#: required a companion fix in ``burnup_calcs.py``'s
#: ``_burnup_durations()`` (see ``test_hot_amb_duff_fladur_matches_cpp``/
#: ``test_long_igtime_fladur_matches_cpp``'s own history) since, once
#: wood/litter genuinely never flames for these two scenarios, ``FlaDur``
#: needed herb+shrub+foliage+branch's own first-timestep contribution
#: (which C++'s ``ES_Calc()`` always counts toward ``d_FlaCon``) to stay
#: correct -- a real, C++-evidenced regression this same pass found and
#: fixed before finalizing, not merely a coincidental side effect.
#:
#: F-62 final acceptance-recovery pass (2026-09-21, third same-day
#: pass): ``("long-igtime", "SmoDur")`` -- the last remaining
#: combination -- is ALSO now removed. Root cause found via the SAME
#: live C++ diagnostic-observer build, extended with a third hook
#: dumping ``DuffBurn()``'s own inputs/outputs (``d_tdf``/``d_Duf_Sec``/
#: ``d_Duf_Tot``) once per run: C++ tracks a running remaining-duff-MASS
#: pool (``d_Duf_Tot``), decremented every timestep by ``Duff_CPTS()``
#: (``bur_brn.cpp:2017-2034``, which subtracts ``rate * elapsed`` and
#: CLAMPS at exactly zero) -- a DISCRETE, clamped depletion process, not
#: a continuous time-vs-duration comparison. Critically, the FIRST such
#: decrement (right after ``Start()``) uses a HARDCODED literal ``60.0``
#: (``bur_brn.cpp:322``), regardless of the scenario's actual ignition/
#: residence time ``ti``. Python's prior ``tis < tdf`` check (a
#: continuous comparison) only coincidentally matched C++ for
#: ``hot-amb-duff`` because that scenario's own ``ti`` (60) happens to
#: equal the hardcoded literal; for ``long-igtime`` (``ti=199.9``), the
#: hardcoded-60 first decrement leaves MORE duff mass remaining than a
#: ``ti``-sized decrement would, extending C++'s real termination time
#: by roughly 90-105 s beyond the continuous model's prediction --
#: exactly the gap this combination measured. Fixed in ``burnup.py`` by
#: replicating C++'s exact discrete, clamped mass-pool mechanism
#: (``duf_tot_mass``/``_duff_cpts()``, using the same hardcoded ``60.0``
#: for the first decrement and ``dt`` for every subsequent one) in place
#: of the continuous ``tis < tdf`` check. Verified: both scenarios' full
#: ``FlaCon``/``SmoCon``/``FlaDur``/``SmoDur`` quadruple now match the
#: golden closely (``long-igtime``'s ``SmoDur``: Python 2419.9 vs golden
#: 2419.899902, |diff| ~1e-4 s), and none of the other 5 already-resolved
#: combinations regressed (re-verified directly, unchanged).
#:
#: **All 6 of the original F-62 (case_id, field) combinations now PASS
#: genuinely. No xfail marker remains on this parametrization.**
_AFFECTED_COMBINATIONS = (
    pytest.param("hot-amb-duff", "FlaCon"),  # RESOLVED 2026-09-21 -- no xfail marker.
    pytest.param("hot-amb-duff", "SmoCon"),  # RESOLVED 2026-09-21 -- no xfail marker.
    pytest.param("long-igtime", "FlaCon"),  # RESOLVED 2026-09-21 -- no xfail marker.
    pytest.param("long-igtime", "SmoCon"),  # RESOLVED 2026-09-21 -- no xfail marker.
    pytest.param("hot-amb-duff", "SmoDur"),  # RESOLVED 2026-09-21 -- no xfail marker.
    pytest.param("long-igtime", "SmoDur"),  # RESOLVED 2026-09-21 -- no xfail marker.
)


@pytest.mark.parametrize("case_id,field", _AFFECTED_COMBINATIONS)
def test_flame_smolder_split_should_match_cpp_for_the_affected_scenarios(case_id, field):
    """Desired-behavior pin for F-62: *field* (``FlaCon``/``SmoCon``/
    ``SmoDur``) SHOULD match the golden within the same tolerance the
    nominal-duration scenarios achieve (see
    :func:`test_flame_smolder_split_matches_cpp_for_nominal_duration_scenarios`),
    for *case_id* (``hot-amb-duff``/``long-igtime``) too.

    UPDATED 2026-09-21 (F-62 final acceptance-recovery pass): all 6 of
    the original (case_id, field) combinations now PASS genuinely.
    ``hot-amb-duff``'s ``SmoDur`` was resolved first (a missing
    duff-remaining continuation check in ``burnup()``'s termination
    condition). ``FlaCon``/``SmoCon`` for BOTH scenarios were resolved
    next (C++'s ``gd_Fudge1``/``gd_Fudge2`` shared-scratch-slot
    semantics, replicated in ``_fire_intensity()``). ``long-igtime``'s
    ``SmoDur`` -- the final combination -- was resolved last, by
    replacing a continuous ``tis < tdf`` duff-duration check with a
    faithful discrete, clamped duff-mass-pool tracker
    (``duf_tot_mass``/``_duff_cpts()``) matching C++'s own
    ``Duff_CPTS()`` mechanics exactly, including its hardcoded
    literal-60.0 first decrement. This function no longer carries any
    ``xfail`` marker for any combination."""
    if field == "SmoDur":
        atol, _ = burnup_extended_tolerance("consume", "smoldering_duration_nominal")
    else:
        atol, _ = burnup_extended_tolerance("consume", "flame_smolder_consumption_nominal")
    golden = _golden_summary_row(case_id)
    result = _run_fofem_emissions_for_case(case_id)
    assert result[field] == pytest.approx(float(golden[field]), abs=atol)


def test_hot_amb_duff_fladur_matches_cpp():
    """``hot-amb-duff`` (``ambient_temp_c=39.9``, just inside the upper
    bound of ``_FIRE_BOUNDS['tamb_c']``, with duff present,
    BR-BRN-NOMINAL): Python's ``FlaDur`` must match the golden's
    within the established ``consume_burnup_extended`` tolerance."""
    golden = _golden_summary_row("hot-amb-duff")
    result = _run_fofem_emissions_for_case("hot-amb-duff")
    atol, _rtol = burnup_extended_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_long_igtime_fladur_matches_cpp():
    """``long-igtime`` (``ig_time_s=199.9``, just inside the upper bound
    of ``_FIRE_BOUNDS['ti']``, BR-BRN-NOMINAL): Python's ``FlaDur``
    must match the golden's within the established ``consume_burnup_extended``
    tolerance."""
    golden = _golden_summary_row("long-igtime")
    result = _run_fofem_emissions_for_case("long-igtime")
    atol, _rtol = burnup_extended_tolerance("consume", "fladur")
    assert result["FlaDur"] == pytest.approx(float(golden["FlaDur"]), abs=atol)


def test_mineral_soil_exposure_matches_cpp():
    """The ``mineral_soil`` route: ``MSE`` matches to floating-point
    noise across all 4 scenarios - ``fuel_cat=Natural``/
    ``cover_group=GrassGroup`` routes through the same equation path
    regardless of the fire-environment/woody-fuel variations these 4
    scenarios exercise."""
    atol, _rtol = burnup_extended_tolerance("consume", "mineral_soil")
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
    match the golden's within the established ``consume_burnup_extended`` tolerance."""
    golden = _golden_summary_row("rot-snd-mix")
    result = _run_fofem_emissions_for_case("rot-snd-mix")
    atol, _rtol = burnup_extended_tolerance("consume", "fladur")
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
    atol, _rtol = burnup_extended_tolerance("consume", "woody_1000hr")
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
    atol, _rtol = burnup_extended_tolerance("consume", "woody_small")
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
