#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_tree_structure_parity.py - Phase 4 coverage for
``calc_bark_thickness`` (via the ``bark_thick`` oracle) and
``calc_canopy_cover`` (via the ``canopy_cover`` oracle).

``consm_canopy`` is a DIFFERENT function with a DIFFERENT oracle route
(consumption, through the ``consume`` mode) and is covered in
``test_phase4_consumption_parity.py``. The two are never conflated here.

**Assertion classes (required module declaration):**

- Class **(c) manifested executable C++ parity** - the ``*_matches_cpp``
  tests, whose expected values come from the committed, fully manifested
  Phase 4 ``canopy_cover`` golden (``SMT_CalcCrnCov`` per tree,
  ``MRT_Overlap`` per stand, at the pinned revision).
- Class **(a) Python contract tests** - the bark-thickness tests (the Python
  function is dead on arrival, F-19) and the oracle-invariant and
  shape/validation tests, none of which claim parity.

Python exposes no per-tree crown area: ``calc_canopy_cover`` returns only the
stand percent. A single-tree stand is therefore used to recover the per-tree
area exactly, by inverting the overlap relation both sides share:
``pct = 100 * (1 - exp(-area / 43560))`` (Python
``tree_flame_calcs.py``; C++ ``MRT_Overlap``, fof_mrt.cpp:1715-1726), so
``area = -43560 * ln(1 - pct/100)``. That inversion is algebra on Python's
OWN output, not a re-implementation of the C++ crown-width equation.

Function order: private helpers first, then public test functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pyfofem.components.tree_flame_calcs import (
    calc_bark_thickness,
    calc_canopy_cover,
)
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._phase4_contract import (
    BARK_THICK_SCENARIOS,
    CANOPY_COVER_SCENARIOS,
    golden_rows,
    golden_rows_by_case,
    phase4_tolerance,
    python_contract_epsilon,
    require_golden_tree,
)

# Fail CLOSED, not open: a missing/incomplete committed Phase 4 golden
# dataset is a repository defect, never a silent skip. This runs at
# collection time so a broken checkout surfaces as a loud, actionable
# error naming the exact missing file(s) - see require_golden_tree().
require_golden_tree()

#: Inches-to-centimetres, matching every other Phase 4 module's convention
#: (``InchToCent``, fof_util.cpp:527-530). ``calc_bark_thickness`` takes and
#: returns centimetres; the ``bark_thick`` golden's ``bark_thick_in`` column
#: is inches.
IN_TO_CM = 2.54

#: Square feet per acre, the constant BOTH implementations use in the overlap
#: relation (Python ``tree_flame_calcs.py``; C++ ``e_SqFtAcre``,
#: fof_mrt.cpp:1719).
SQ_FT_PER_ACRE = 43560.0

#: (RTOL_AREA, ATOL_AREA) for a crown area / percent cover, both retrieved
#: from the single centralized ``canopy_cover_p4.all`` policy entry rather
#: than duplicated as literals. C++ ``SMT_CalcCrnCov`` uses the literal
#: ``3.14159`` (fof_mrt.cpp:1634) where Python uses ``numpy.pi``; the two
#: differ by (pi - 3.14159)/pi = 8.4e-07 relative, and crown area is linear
#: in that constant. The measured maximum relative difference across every
#: agreeing scenario is 1.02e-06, consistent with that constant difference
#: plus the C++ side's float32 arithmetic. The harness writes both crown
#: area and percent through ``fmt(v, 4)``, i.e. four decimal places, so
#: 1e-04 is the output's own resolution; the measured maximum |diff| among
#: agreeing scenarios that this bound (rather than the relative one) covers
#: is 4.54e-05.
_CANOPY_COVER_ATOL, RTOL_AREA = phase4_tolerance("canopy_cover", "all")
ATOL_AREA = _CANOPY_COVER_ATOL

#: Absolute tolerance for a bark-thickness value (inches), retrieved from
#: the centralized ``bark_thick_p4.all`` policy entry - the golden's own
#: ``fmt(v, 6)`` six-decimal output resolution, not a measured agreement
#: bound (see that entry's justification: no Python-vs-C++ bark-thickness
#: value has ever agreed, F-19).
ATOL_BARK_THICK = phase4_tolerance("bark_thick", "all")[0]

#: ``calc_canopy_cover`` per-tree comparisons that DIVERGE.
CANOPY_TREE_XFAIL = {
    case: (
        "F-02",
        "C++ SMT_CalcCrnCov returns 0 as soon as `f_Hgt <= 0` "
        "(fof_mrt.cpp:1616-1617), so a zero- or negative-height tree "
        "contributes no crown area; calc_canopy_cover excludes a tree only "
        "for `dbh <= 0` or NaN DBH and still credits this tree the full "
        "30.229674 ft2 its DBH implies.",
    )
    for case in ("ccv-p4s4-psme-ht0", "ccv-p4s4-psme-htneg")
}

#: ``calc_canopy_cover`` stand comparisons that DIVERGE.
CANOPY_STAND_XFAIL = {
    "p4s4": (
        "F-02",
        "every member of this stand has zero/negative height or zero DBH, so "
        "C++ reports 0.0000 percent cover; Python credits the two "
        "nonpositive-height trees and reports 0.138699 percent.",
    ),
}


#: Canopy scenarios whose C++ row is a real success, derived from the
#: CONTRACT's own ``expect_error`` column rather than from the golden, so
#: collection never depends on the golden tree being present.
OK_TREE_SCENARIOS = [
    scenario for scenario in CANOPY_COVER_SCENARIOS if scenario[5] == "0"
]

#: Stands every one of whose members is expected to succeed - the only stands
#: for which the harness emits an aggregate row (a stand with any non-ok
#: member has its aggregate suppressed by design).
OK_STAND_IDS = sorted(
    {scenario[1] for scenario in OK_TREE_SCENARIOS}
    - {scenario[1] for scenario in CANOPY_COVER_SCENARIOS if scenario[5] != "0"}
)

#: Stands with at least one member expected to fail - their aggregate must be
#: suppressed, which is asserted directly rather than skipped past.
SUPPRESSED_STAND_IDS = sorted(
    {scenario[1] for scenario in CANOPY_COVER_SCENARIOS if scenario[5] != "0"}
)


















#: One representative OK per-tree scenario per docstring-cited species,
#: covering PIAL/ABBA/QURU/PICO - the four species F-02's finding text
#: measures.
DEFAULT_MAPPING_XFAIL_CASES = [
    "ccv-p4s1-pial", "ccv-p4s1-abba", "ccv-p4s3-quru", "ccv-p4s5-pico-a",
]


def _crown_area_from_percent(percent_cover):
    """
    Invert the shared overlap relation to recover accumulated crown area.

    :param percent_cover: Percent canopy cover, as ``calc_canopy_cover``
        returns it.
    :returns: Accumulated crown area in ft2.
    """
    return -SQ_FT_PER_ACRE * math.log(1.0 - percent_cover / 100.0)


def _crown_equation_map():
    """
    Read species code to crown-cover equation number from the tracked
    ``FOF_SPP.CSV`` - the same file the harness loads through
    ``MRT_LoadSpe()``.

    Supplying this mapping explicitly is required: without a
    ``tree_code_dict`` ``calc_canopy_cover`` assigns EVERY species equation
    39 (F-02), which is a different scientific claim and is characterised
    separately by
    :func:`test_canopy_cover_default_equation_mapping_diverges`.

    :returns: ``{species_code: equation_number}``, first occurrence wins.
    """
    path = os.path.join(
        PROJECT_ROOT, "src", "pyfofem", "supporting_data", "FOFEM6.7",
        "FOF_SPP.CSV",
    )
    mapping = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = [item.strip() for item in line.rstrip("\n").split(",")]
            if len(fields) >= 11:
                mapping.setdefault(fields[1].upper(), int(fields[6]))
    return mapping


def _maybe_xfail(request, table, key):
    """
    Apply a strict xfail to the running test if *key* is a known divergence.

    :param request: The pytest ``request`` fixture.
    :param table: One of the ``*_XFAIL`` tables in this module.
    :param key: The scenario or stand being compared.
    :returns: None.
    """
    if key not in table:
        return
    finding, reason = table[key]
    request.node.add_marker(
        pytest.mark.xfail(strict=True, reason=f"{finding}: {reason}")
    )


def test_bark_thickness_golden_is_linear_in_dbh():
    """
    Class (a) ORACLE-INVARIANT test, not parity.

    ``SMT_CalcBarkThick`` returns ``factor * DBH`` with the factor selected
    from a species-keyed ladder (fof_mrt.cpp:1391-1436), so the golden's own
    values must be exactly linear in DBH and must be 0 at DBH 0. Verified
    against the two CACOL3 rows generated at DBH 0 and 40, without
    re-implementing the ladder.
    """
    rows = golden_rows_by_case("bark_thick")
    at_zero = float(rows["brk16-cacol3-d0"]["bark_thick_in"])
    at_forty = float(rows["brk16-cacol3-d40"]["bark_thick_in"])
    assert at_zero == 0.0
    assert at_forty > 0.0
    factor = at_forty / 40.0
    assert factor == pytest.approx(
        0.037, abs=python_contract_epsilon("bark_factor_oracle_invariant")
    ), (
        "CACOL3's i_BrkEqu is 16 in the tracked FOF_SPP.CSV, and "
        "fof_mrt.cpp:1407 assigns equation 16 the factor 0.037"
    )


def test_bark_thickness_golden_rows_are_all_accounted_for():
    """Every bark_thick scenario must have produced exactly one row whose
    outcome matches its declared ``expect_error`` value."""
    rows = golden_rows_by_case("bark_thick")
    assert len(rows) == len(BARK_THICK_SCENARIOS)
    for case, _species, _dbh, expect_error, _branches in BARK_THICK_SCENARIOS:
        expected = "expected_model_error" if expect_error == "1" else "ok"
        assert rows[case]["outcome"] == expected, case


def test_bark_thickness_golden_unknown_species_is_an_expected_error():
    """
    Class (a) ORACLE-INVARIANT test: an unknown species must be recorded as
    an EXPECTED model error carrying C++'s own message, not as a silent zero.
    """
    row = golden_rows_by_case("bark_thick")["brk-unknown-species"]
    assert row["outcome"] == "expected_model_error"
    assert row["bark_thick_in"] == "NA"
    assert "SMT_CalcBarkThick" in row["err_text"]


def test_bark_thickness_golden_zero_bark_species_returns_zero():
    """
    Class (a) ORACLE-INVARIANT test: the single species whose ``i_BrkEqu`` is
    100 must return exactly 0 bark thickness (fof_mrt.cpp:1431-1433).
    """
    rows = golden_rows_by_case("bark_thick")
    assert float(rows["brk100-pipa2-d12"]["bark_thick_in"]) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-19: calc_bark_thickness always raises "
        "KeyError('FOFEM_BrkThck_Vsp') because it reads a column the "
        "bundled species_codes_lut.csv does not have, so no Python-vs-C++ "
        "bark-thickness comparison is possible."
    ),
)
def test_bark_thickness_is_dead_on_arrival():
    """
    Class (a) Python contract test (F-19), strict xfail.

    ``calc_bark_thickness`` reads ``SPP_CODES['FOFEM_BrkThck_Vsp']``, a
    column the bundled ``species_codes_lut.csv`` does not have, so every call
    raises. The C++ side of this comparison IS available and fully
    manifested (the Phase 4 ``bark_thick`` golden); only the Python side is
    unreachable.

    Asserts the DESIRED behaviour - a real bark-thickness value matching the
    manifested ``brkxr-psme-d12`` golden row (PSME, DBH 12 in) - not the
    current ``KeyError``. Currently the call raises before the comparison is
    reached, so this genuinely executes and genuinely fails - it is not
    vacuous.
    """
    expected_in = float(
        golden_rows_by_case("bark_thick")["brkxr-psme-d12"]["bark_thick_in"]
    )
    value_cm = calc_bark_thickness(
        np.array(["PSME"]), np.array([12.0 * IN_TO_CM])
    )
    value_in = float(np.asarray(value_cm)[0]) / IN_TO_CM
    assert value_in == pytest.approx(expected_in, abs=ATOL_BARK_THICK)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-02: calc_canopy_cover defaults every species to equation 39. "
        "Measured against the manifested canopy_cover golden, that gives "
        "334.271 ft2 for PIAL where C++ (equation 31) gives 197.588, "
        "140.845 vs 100.669 for ABBA (equation 2), 373.996 vs 815.559 for "
        "QURU (equation 28) and 454.212 vs 267.778 for PICO (equation 11) - "
        "relative errors of 40 % to 70 %."
    ),
)
@pytest.mark.parametrize("case_id", DEFAULT_MAPPING_XFAIL_CASES)
def test_canopy_cover_default_equation_mapping_diverges(case_id):
    """
    Class (a) Python contract test (F-02), strict xfail.

    Without a ``tree_code_dict`` every species is assigned crown-width
    equation 39, which does not match C++'s per-species FVS index lookup.
    Characterised here rather than relied upon: the parity tests above all
    pass the real mapping explicitly.

    Asserts the DESIRED behaviour - that the DEFAULT (no ``tree_code_dict``)
    call should still recover the golden per-tree crown area, exactly like
    :func:`test_canopy_cover_per_tree_area_matches_cpp`'s explicit-mapping
    call does. Currently the default call routes every species through
    equation 39 instead of its real per-species equation, so this genuinely
    executes and genuinely fails by 40-70% relative error - it is not
    vacuous.
    """
    row = golden_rows_by_case("canopy_cover", "_trees")[case_id]
    assert row["outcome"] == "ok"
    scenario = next(s for s in OK_TREE_SCENARIOS if s[0] == case_id)
    _case, _stand, species, dbh, height, _err, _branches = scenario
    percent = calc_canopy_cover(
        [species], [float(dbh)], [float(height)], units="imperial",
    )
    assert _crown_area_from_percent(percent) == pytest.approx(
        float(row["crown_area_ft2"]), rel=RTOL_AREA, abs=ATOL_AREA
    )


@pytest.mark.parametrize(
    "case_id", [scenario[0] for scenario in OK_TREE_SCENARIOS]
)
def test_canopy_cover_per_tree_area_matches_cpp(case_id, request):
    """``calc_canopy_cover``'s implied per-tree crown area vs the golden."""
    row = golden_rows_by_case("canopy_cover", "_trees")[case_id]
    assert row["outcome"] == "ok", (
        f"{case_id} is declared expect_error=0 but the oracle recorded "
        f"{row['outcome']!r}"
    )
    _maybe_xfail(request, CANOPY_TREE_XFAIL, case_id)
    scenario = next(
        s for s in OK_TREE_SCENARIOS if s[0] == case_id
    )
    _case, _stand, species, dbh, height, _err, _branches = scenario
    percent = calc_canopy_cover(
        [species], [float(dbh)], [float(height)],
        tree_code_dict={species: _crown_equation_map()[species.upper()]},
        units="imperial",
    )
    assert _crown_area_from_percent(percent) == pytest.approx(
        float(row["crown_area_ft2"]), rel=RTOL_AREA, abs=ATOL_AREA
    )


@pytest.mark.parametrize(
    "case_id", [scenario[0] for scenario in OK_TREE_SCENARIOS]
)
def test_canopy_cover_per_tree_equation_matches_tracked_table(case_id):
    """
    Class (a) DATA-TABLE CONSISTENCY test, explicitly NOT per-species
    executable parity.

    The equation number C++ selected for each tree (``cct_equ_no``, read from
    ``SMT_Get``) must equal the ``Crn`` column of the tracked
    ``FOF_SPP.CSV``. This checks the species-to-equation table itself is
    consistent end to end; it does not assert that Python reproduces any
    species' crown area.
    """
    row = golden_rows_by_case("canopy_cover", "_trees")[case_id]
    scenario = next(s for s in OK_TREE_SCENARIOS if s[0] == case_id)
    species = scenario[2]
    assert int(row["cct_equ_no"]) == _crown_equation_map()[species.upper()]


def test_canopy_cover_rejects_mismatched_input_lengths():
    """Class (a) Python contract test: unequal input lengths must raise."""
    with pytest.raises(ValueError, match="same length"):
        calc_canopy_cover(["PSME", "PSME"], [12.0], [60.0], units="imperial")


@pytest.mark.parametrize("stand_id", OK_STAND_IDS)
def test_canopy_cover_stand_percent_matches_cpp(stand_id, request):
    """``calc_canopy_cover``'s stand percent vs the golden ``pct_cover``."""
    stands = {row["stand_id"]: row for row in
              golden_rows("canopy_cover", "_stands")}
    assert stand_id in stands, (
        f"every member of stand {stand_id!r} is expected to succeed, so the "
        "harness must have emitted its aggregate row"
    )
    _maybe_xfail(request, CANOPY_STAND_XFAIL, stand_id)
    members = [
        (species, float(dbh), float(height))
        for _c, stand, species, dbh, height, _e, _b in OK_TREE_SCENARIOS
        if stand == stand_id
    ]
    mapping = _crown_equation_map()
    species_codes = [member[0] for member in members]
    percent = calc_canopy_cover(
        species_codes,
        [member[1] for member in members],
        [member[2] for member in members],
        tree_code_dict={code: mapping[code.upper()] for code in species_codes},
        units="imperial",
    )
    assert percent == pytest.approx(
        float(stands[stand_id]["pct_cover"]), rel=RTOL_AREA, abs=ATOL_AREA
    )


def test_canopy_cover_stand_totals_reconcile_against_members():
    """
    Class (a) ORACLE-INVARIANT test: every emitted stand aggregate's
    ``total_area_ft2`` must equal the sum of its own member trees' areas, and
    its ``n_trees`` must equal the membership count.
    """
    trees = golden_rows("canopy_cover", "_trees")
    stands = golden_rows("canopy_cover", "_stands")
    assert stands, "no stand aggregates emitted"
    for stand in stands:
        members = [t for t in trees if t["stand_id"] == stand["stand_id"]]
        assert int(stand["n_trees"]) == len(members)
        total = sum(float(t["crown_area_ft2"]) for t in members)
        assert float(stand["total_area_ft2"]) == pytest.approx(
            total, rel=RTOL_AREA, abs=ATOL_AREA
        )


def test_canopy_cover_suppressed_stand_is_recorded_in_the_group_file():
    """
    Class (a) ORACLE-INVARIANT test: the stand containing an unknown species
    must have its aggregate suppressed, and the diagnostic group file must
    record why.
    """
    groups = {row["stand_id"]: row for row in
              golden_rows("canopy_cover", "_groups")}
    stands = {row["stand_id"] for row in golden_rows("canopy_cover", "_stands")}
    assert SUPPRESSED_STAND_IDS, "no suppression scenario in the matrix"
    for stand_id in SUPPRESSED_STAND_IDS:
        assert stand_id in groups
        assert stand_id not in stands
        assert groups[stand_id]["aggregate_emitted"] == "0"
        assert (groups[stand_id]["suppression_reason"]
                == "expected_model_error_member")


def test_canopy_cover_zero_dbh_tree_is_excluded_by_both_sides():
    """
    Both implementations exclude a zero-DBH tree: C++ returns 0 as soon as
    ``f_Dia <= 0`` (fof_mrt.cpp:1618-1619) and Python skips the tree when
    ``dbh <= 0``. Verified against the manifested golden, so this is class
    (c) parity for the one boundary the two implementations DO share - the
    height boundary they do not share is xfailed above.
    """
    row = golden_rows_by_case("canopy_cover", "_trees")["ccv-p4s4-psme-dbh0"]
    assert float(row["crown_area_ft2"]) == 0.0
    percent = calc_canopy_cover(
        ["PSME"], [0.0], [60.0], tree_code_dict={"PSME": 16},
        units="imperial",
    )
    assert percent == pytest.approx(0.0, abs=ATOL_AREA)


def test_canopy_height_boundary_selects_the_expected_equation_form():
    """
    Class (c) parity at the 4.5 ft height boundary: C++ uses the small-tree
    form ``r * Dia`` for ``f_Hgt <= 4.5`` and the large-tree form
    ``a * Dia^b`` above it (fof_mrt.cpp:1626-1631). The golden's own values
    at 4.4 / 4.5 / 4.6 ft must therefore pair the first two and separate the
    third, and Python must agree with all three.
    """
    rows = golden_rows_by_case("canopy_cover", "_trees")
    below = float(rows["ccv-p4s2-psme-ht44"]["crown_area_ft2"])
    at = float(rows["ccv-p4s2-psme-ht45"]["crown_area_ft2"])
    above = float(rows["ccv-p4s2-psme-ht46"]["crown_area_ft2"])
    assert below == at, "4.4 ft and 4.5 ft must use the same (small) form"
    assert above != at, "4.6 ft must switch to the large-tree form"
    for height, expected in ((4.4, below), (4.5, at), (4.6, above)):
        percent = calc_canopy_cover(
            ["PSME"], [3.0], [height], tree_code_dict={"PSME": 16},
            units="imperial",
        )
        assert _crown_area_from_percent(percent) == pytest.approx(
            expected, rel=RTOL_AREA, abs=ATOL_AREA
        )
