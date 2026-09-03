#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase4_mortality_parity.py - Phase 4 coverage for ``mort_bolchar``,
``mort_crnsch`` and ``mort_crcabe``.

**Assertion classes (required module declaration):**

- Class **(c) manifested executable C++ parity** - the ``*_matches_cpp``
  tests, whose expected values come from the committed, fully manifested
  Phase 4 ``mortality`` golden (the compiled ``fofem_test`` harness driving
  ``MRT_CalcMngr`` at the pinned revision), with any required bark thickness
  taken from the SAME manifested dataset's ``bark_thick`` golden
  (``SMT_CalcBarkThick``) - never invented and never from Python's own
  F-19-broken ``calc_bark_thickness``.
- Class **(b) source-relation checks** - the ``*_source_relation`` tests,
  which compare Python's coefficients against values hand-transcribed from
  the pinned C++ source with the file:line cited in the docstring. These are
  NOT executable parity and are labelled as such.
- Class **(a) Python contract tests** - the remaining tests, which pin
  current Python behaviour (error text, NaN, stdout) for visibility only.

**CroDam is now covered, and F-45 is resolved.** ``PFI_Calc`` validates its
inputs through ``ValidInput`` (fof_mrt.cpp:1829-1871), which requires
``1 <= a_MIS->f_Den <= 20000`` (fof_mrt.cpp:1854-1856). The ``mortality``
harness mode's schema v1 had no density column, so ``f_Den`` stayed 0 from
the mode's own ``memset``, every CroDam row failed validation, and - because
``PFI_Calc`` signals that failure by returning 0 rather than a negative
sentinel while the harness's error rule tested only ``prob < 0`` - each row
was recorded ``ok`` with ``prob=0.000000`` and a non-empty ``err_text``
(F-45). The Phase 4 correction pass fixed BOTH halves in the harness: schema
v2 adds a ``density_tpa`` column wired to ``d_MIS.f_Den``, and the error rule
now treats either a negative probability OR non-empty error text as a model
error. All eleven PFI equations are therefore compared for real below.

The comparison is only meaningful where the two contracts coincide. C++'s
``f_CrnDam`` is a direct crown-damage percent input; ``mort_crcabe`` instead
DERIVES both ``cvs`` and ``cls`` from scorch height / height / crown depth
and uses one or the other per equation, so ``cvs == cls == f_CrnDam`` holds
exactly at 0 % and 100 %. Every CroDam scenario is generated at one of those
two endpoints, with a crown geometry (``crown_ratio_x10 = 10``, i.e. crown
depth == height) that makes Python's derivation from the SAME input row land
exactly on the endpoint. The three fields involved are inert on the C++ side:
``PFI_Calc``'s equations read only ``f_DBH``, ``f_CKR``, ``f_CrnDam`` and
``cr_BeeDam`` (fof_mrt.cpp:1911-2235).

Function order: private helpers first, then public test functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pyfofem.components.mortality_calcs import (
    mort_bolchar,
    mort_crcabe,
    mort_crnsch,
)
from tests.cpp_parity_live._phase4_contract import (
    CROSCO_BARK_SOURCE,
    MORTALITY_INDEX,
    MORTALITY_SCENARIOS,
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

#: The pinned C++ unit conversions, cited so the mapping is auditable:
#: ``FtToMt`` (fof_mrt.cpp:542-547) and ``InchToCent`` (fof_util.cpp:527-530).
FT_TO_M = 0.3048
IN_TO_CM = 2.54

#: Absolute tolerance for a mortality probability, retrieved from the
#: centralized policy - ``mortality_p4.BolCha``, ``.CroDam`` and ``.CroSco``
#: all record the same 1e-06. The harness writes ``prob`` through
#: ``fmt(prob, 6)``, i.e. six decimal places, and the measured maximum |diff|
#: across every agreeing scenario in this module is 5.4e-07 - below that
#: output resolution. The smallest DIVERGENCE measured is 4.6e-06, so 1e-06
#: separates the two cleanly.
ATOL_PROB = phase4_tolerance("mortality", "BolCha")[0]

#: The eleven crown-damage (PFI) equation codes ``sr_EFR[]`` dispatches to,
#: hand-transcribed from the pinned ``fof_mrt.cpp:78-91`` with the code
#: strings from ``fof_mec.h:10-22``. ``sr_EFR`` has thirteen rows; the other
#: two are ``CRNSCH``/``BOLCHR``, which route to ``MRT_Calc``/``BC_Calc``
#: instead of to a PFI equation function.
CPP_PFI_EQUATIONS = (
    "WF", "SF", "IC", "WL", "WP", "ES", "SP", "RF", "PP", "PK", "DF",
)

#: ``equation -> (pinned per-INCH coefficient, Python's per-CENTIMETRE
#: coefficient)`` for the three PFI equations that carry a DBH term. These
#: are the whole of F-50: ``per_inch / 2.54`` is not equal to ``per_cm`` in
#: any of the four cases, and substituting the exact quotient reproduces the
#: C++ oracle to nine decimals.
CPP_PFI_DBH_COEFFICIENTS = {
    # fof_mrt.cpp:1928 (WF), 2028 (WP), 2181-2182 (DF, two terms)
    "WF_dbh": (0.0483, 0.019),
    "WP_dbh": (-0.1232, -0.0485),
    "DF_dbh": (-0.0788, -0.031),
    "DF_dbh_beetle": (0.1251, 0.0492),
}

#: ``sr_BCT[]``, hand-transcribed from the pinned ``fof_mrt.cpp:2203-2216``.
#: Used ONLY by the class-(b) source-relation test below; the executable
#: parity tests read their expected values from the golden instead.
CPP_SR_BCT = {
    "100": (2.3014, -0.3267, 1.1137),
    "101": (-0.8727, -0.1814, 4.1947),
    "102": (2.7899, -0.5511, 1.2888),
    "103": (1.9438, -0.4602, 1.6352),
    "104": (-1.8137, -0.0603, 0.8666),
    "105": (-1.6262, -0.0339, 0.6901),
    "106": (0.3714, -0.1005, 1.5577),
    "107": (-1.4416, -0.1469, 1.3159),
    "108": (0.1122, -0.1287, 1.2612),
    "109": (1.6779, -1.0299, 10.2855),
}

#: One Python-recognised species per bole-char equation. For equations 100-106
#: and 108-109 this is the same species the tracked ``FOF_SPP.CSV`` assigns to
#: that equation. Equation 107 is the exception: the CSV assigns it ``QUMO4``
#: alone, which Python does not recognise at all, while Python assigns
#: ``QUMI``/``QUPR4`` (which the CSV maps to crown-scorch equation 1) - see
#: F-37 and :func:`test_bolchar_species_equation_mapping_source_relation`.
PY_BOLCHAR_REPRESENTATIVE = {
    "100": "ACRU", "101": "COFL2", "102": "NYSY", "103": "OXAR",
    "104": "QUAL", "105": "QUCO2", "106": "QUMA3", "107": "QUMI",
    "108": "QUVE", "109": "SAAL5",
}

#: Bole-char equations whose Python coefficients DIVERGE from ``sr_BCT[]``.
BOLCHAR_COEFFICIENT_XFAIL = {
    "100": "Python's intercept is 2.3017; the pinned sr_BCT value is 2.3014.",
    "102": "Python's intercept is -2.7899; the pinned sr_BCT value is "
           "+2.7899 - a sign inversion, not a rounding difference.",
    "107": "Python reuses equation 104's coefficients (-1.8137, -0.0603, "
           "0.8666) for chestnut oak; the pinned sr_BCT values are "
           "(-1.4416, -0.1469, 1.3159).",
    "109": "Python reuses equation 104's coefficients (-1.8137, -0.0603, "
           "0.8666) for sassafras; the pinned sr_BCT values are "
           "(1.6779, -1.0299, 10.2855).",
}

#: ``mort_bolchar`` vs the golden ``prob``: scenarios that DIVERGE. The other
#: 8 bole-char scenarios agree (measured max |diff| 5.0e-07).
BOLCHAR_XFAIL = {
    "bc100-acru-small": (
        "F-36",
        "equation 100's intercept differs (Python 2.3017 vs pinned sr_BCT "
        "2.3014). Measured 0.727350 (C++) vs 0.727410 (Python), |diff| "
        "5.97e-05. The same defect is present in bc100-acru but is not "
        "marked there because at that scenario's probability (0.001836) the "
        "difference is 6.5e-08, below the golden's own six-decimal "
        "resolution - the coefficient divergence itself is instead pinned "
        "unconditionally by "
        "test_bolchar_coefficients_match_sr_bct_source_relation.",
    ),
    "bc102-nysy": (
        "F-36",
        "equation 102's intercept sign is inverted in Python (-2.7899 vs "
        "the pinned +2.7899). Measured 4.4e-05 (C++) vs 1.66e-07 (Python).",
    ),
    "bc107-qumo4": (
        "F-37",
        "QUMO4 is the only species the tracked FOF_SPP.CSV assigns bole-char "
        "equation 107, and mort_bolchar does not recognise it at all: it "
        "prints a warning and returns NaN where C++ gives 0.009378.",
    ),
    "bc109-saal5": (
        "F-36",
        "equation 109 uses equation 104's coefficients in Python. Measured "
        "0.005871 (C++) vs 0.113928 (Python), |diff| 0.108057.",
    ),
}

#: ``mort_crcabe`` vs the golden ``prob``: scenarios that DIVERGE. Every
#: entry is one of the three PFI equations carrying a DBH term, and every
#: one has the same root cause (F-50): Python holds a ROUNDED centimetre
#: conversion of the pinned per-inch coefficient. Substituting the exact
#: per-inch value into Python's own formula reproduces the C++ oracle to
#: nine decimals, which is what isolates the coefficient as the cause. The
#: other eight equations (SF, WL, IC, ES, RF, SP, PP, PK) agree, measured
#: max |diff| 4.51e-07.
CRODAM_XFAIL = {
    "cd-wf-abco": (
        "F-50",
        "equation WF applies 0.0483 per INCH of DBH "
        "(fof_mrt.cpp:1928); Python applies 0.019 per centimetre, and "
        "0.0483/2.54 = 0.0190157480. Measured 0.974618 (C++) vs 0.974606 "
        "(Python), |diff| 1.20e-05.",
    ),
    "cd-wf-abco-cvk0": (
        "F-50",
        "same WF coefficient, at zero crown damage. Measured 0.067107 "
        "(C++) vs 0.067077 (Python), |diff| 2.99e-05.",
    ),
    "cd-wf-abco-ckr0": (
        "F-50",
        "same WF coefficient, at a zero cambium-kill rating. Measured "
        "0.939477 (C++) vs 0.939450 (Python), |diff| 2.73e-05.",
    ),
    "cd-density-min": (
        "F-50",
        "same WF coefficient; this row exists for ValidInput's lower "
        "density boundary, and its probability is identical to "
        "cd-wf-abco because density enters no PFI equation. |diff| "
        "1.20e-05.",
    ),
    "cd-density-max": (
        "F-50",
        "same WF coefficient; ValidInput's upper density boundary. "
        "|diff| 1.20e-05.",
    ),
    "cd-wp-pial": (
        "F-50",
        "equation WP applies -0.1232 per INCH of DBH "
        "(fof_mrt.cpp:2009-2031); Python applies -0.0485 per centimetre, "
        "and -0.1232/2.54 = -0.0485039370. Measured 0.984225 (C++) vs "
        "0.984226 (Python), |diff| 1.43e-06 - the smallest divergence in "
        "this table, still 1.4x the 1e-06 atol.",
    ),
    "cd-wp-pial-cvk0": (
        "F-50",
        "same WP coefficient, at zero crown damage, where the DBH term "
        "dominates. Measured 0.419312 (C++) vs 0.419341 (Python), |diff| "
        "2.88e-05.",
    ),
    "cd-df-psme": (
        "F-50",
        "equation DF applies -0.0788 per INCH of DBH and +0.1251 per "
        "INCH of DBH-times-beetle (fof_mrt.cpp:2181-2182); Python applies "
        "-0.031 and +0.0492 per centimetre, and -0.0788/2.54 = "
        "-0.0310236220, 0.1251/2.54 = 0.0492519685. Measured 0.996390 "
        "(C++) vs 0.996387 (Python), |diff| 2.86e-06.",
    ),
    "cd-df-psme-cvk0": (
        "F-50",
        "both DF coefficients, at zero crown damage. Measured 0.406247 "
        "(C++) vs 0.406039 (Python), |diff| 2.08e-04 - the largest "
        "divergence in this table.",
    ),
    "cd-df-psme-nobeetle": (
        "F-50",
        "the DBH-only DF coefficient, isolated: with beetles off the "
        "DBH-times-beetle term vanishes and the divergence persists. "
        "Measured 0.992719 (C++) vs 0.992724 (Python), |diff| 5.35e-06.",
    ),
}

#: ``mort_crcabe`` selects the Ponderosa/Jeffrey bud-kill (PK) equation by
#: being handed a ``cvk`` value, whereas C++ selects it from the species'
#: own ``Mort`` code in the tracked ``FOF_SPP.CSV``. The golden's
#: ``mort_equ`` column reports which equation the oracle ACTUALLY ran, so
#: the Python call is steered from that rather than from a hardcoded
#: species list.
CRCABE_BUD_KILL_EQUATION = "PK"

#: ``mort_crnsch`` vs the golden ``prob``: scenarios that DIVERGE. The other
#: 14 crown-scorch scenarios agree (measured max |diff| 5.4e-07).
CROSCO_XFAIL = {
    "cs03-piab-dbh1": (
        "F-48",
        "the case-1-vs-case-3 DBH boundary. C++ case 3 takes the large-tree "
        "branch only when `f_DBH > 1` (fof_mrt.cpp:381), so at DBH exactly 1 "
        "it falls through to the small-tree rule and returns 1.000000; "
        "Python's equation 3 uses `dbh_in >= 1` and returns 0.999049. "
        "Measured |diff| 9.51e-04. The DBH=1.1 and DBH=12 partners in this "
        "same scenario set agree, which isolates the boundary itself as the "
        "cause. NOTE: F-48 originally also recorded that case 3's 0.8 floor "
        "was untested because every case-3 probability was above 0.99. That "
        "half is CLOSED - cs03-piab-floor08 now reaches the floor and agrees "
        "exactly (see test_crnsch_case3_reaches_the_08_floor); only the "
        "boundary defect remains open.",
    ),
    "cs05-pipa2": (
        "F-49",
        "equation 5 (longleaf pine, PINPAL). C++ divides the crown-scorch "
        "term by 10 (`f = f_CK / 10.0`, fof_mrt.cpp:349) and uses the "
        "quadratic bark coefficient 14.492 (fof_mrt.cpp:351); Python divides "
        "by 100 and uses 14.429. Measured 0.994231 (C++) vs 9.57e-13 "
        "(Python).",
    ),
    "cs14-laoc": (
        "F-47",
        "equation 14 (western larch). C++ WesternLarch applies `dbh * "
        "0.1241` to DBH in INCHES (fof_mrt.cpp:583-591); Python applies "
        "0.0489 to DBH in centimetres, and 0.1241/2.54 = 0.04885827..., so "
        "Python's coefficient is a rounded conversion. Measured |diff| "
        "3.17e-04 in probability.",
    ),
    "cs17-pial": (
        "F-47",
        "equation 17 (whitebark pine). C++ WhitebarkPine applies `dbh * "
        "0.0676` to DBH in INCHES (fof_mrt.cpp:676-686); Python applies "
        "0.0266 to DBH in centimetres, and 0.0676/2.54 = 0.02661417..., so "
        "Python's coefficient is again a rounded conversion. Measured "
        "|diff| 4.57e-06 in probability.",
    ),
    "cs04-potr5-lowsev": (
        "F-46",
        "C++ case 4 derives the char height from the flame length "
        "(fof_mrt.cpp:396-397). mort_crnsch cannot be driven that way: when "
        "`scorch_ht` is not supplied it always calls "
        "`calc_scorch_ht(fire_intensity, ...)`, which raises "
        "`Exception('Must enter a surface fire intensity value...')` for a "
        "flame-length-only call, so no comparable Python value exists.",
    ),
    "cs04-potr5-highsev": (
        "F-46",
        "same flame-length-only blocker as the low-severity partner.",
    ),
    "cs21-pipobh-seedling": (
        "F-46",
        "C++ equation 21 (Black Hills ponderosa pine) is driven by flame "
        "length; mort_crnsch raises for a flame-length-only call.",
    ),
    "cs21-pipobh-sapling": (
        "F-46",
        "same flame-length-only blocker as the seedling case.",
    ),
    "cs21-pipobh-large": (
        "F-46",
        "same flame-length-only blocker as the seedling case.",
    ),
}


def _bolchar_python_coefficients(species):
    """
    Recover ``mort_bolchar``'s effective ``(B1, B2, B3)`` for *species* by
    evaluating the function at three points, without reading its source.

    ``mort_bolchar`` computes ``p = 1 / (1 + exp(-(B1 + B2*dbh_cm +
    B3*char_m)))``, so ``logit(p)`` is linear in the two inputs and the three
    coefficients follow from ``f(0,0)``, ``f(1,0)`` and ``f(0,1)``.

    :param species: A species code ``mort_bolchar`` recognises.
    :returns: ``(B1, B2, B3)`` as floats.
    """
    def logit(p):
        return math.log(p / (1.0 - p))

    b1 = logit(float(mort_bolchar(species, 0.0, 0.0)))
    b2 = logit(float(mort_bolchar(species, 1.0, 0.0))) - b1
    b3 = logit(float(mort_bolchar(species, 0.0, 1.0))) - b1
    return b1, b2, b3


def _crcabe_call(overrides, golden_row):
    """
    Evaluate ``mort_crcabe`` for one CroDam scenario, reading every input
    from that scenario's own harness input row.

    Units are converted with the pinned constants only; no value is
    invented, and no crown-damage percent is substituted for the derived
    ``cvs``/``cls`` - the scenario geometry (crown depth == height) is what
    makes the derivation land on the endpoint the C++ side was handed.

    :param overrides: The scenario's column-name to value overrides.
    :param golden_row: That scenario's manifested golden row, used only
        for its ``mort_equ`` column (which equation the oracle really
        ran).
    :returns: The Python mortality probability as a float.
    """
    height_m = float(_mortality_input(overrides, "ht_ft")) * FT_TO_M
    crown_depth_m = (
        height_m * float(_mortality_input(overrides, "crown_ratio_x10")) / 10.0
    )
    crown_damage_pct = float(_mortality_input(overrides, "cvk_pct"))
    kwargs = {}
    if golden_row["mort_equ"] == CRCABE_BUD_KILL_EQUATION:
        kwargs["cvk"] = crown_damage_pct
    return float(mort_crcabe(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        height_m,
        crown_depth_m,
        float(_mortality_input(overrides, "ckr_rating")),
        float(_mortality_input(overrides, "fs_value_ft")) * FT_TO_M,
        beetles=_mortality_input(overrides, "beetles") == "1",
        **kwargs,
    ))


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
        pytest.mark.xfail(strict=True, reason=f"{finding}: {reason}")
    )


def _mortality_input(overrides, column):
    """
    Return one mortality input field for a scenario, after its overrides.

    :param overrides: The scenario's column-name to value overrides.
    :param column: Which input column to read back.
    :returns: The raw string value written to the golden's input CSV.
    """
    row = list(MODES["mortality"]["row"])
    for key, value in overrides.items():
        row[MORTALITY_INDEX[key]] = value
    return row[MORTALITY_INDEX[column]]


def _scenarios(equ_type, expect_error=None):
    """
    Return the Phase 4 mortality scenarios of one equation type.

    :param equ_type: ``"BolCha"``, ``"CroSco"`` or ``"CroDam"``.
    :param expect_error: If given, keep only scenarios whose
        ``expect_error`` column has this value (``"0"`` for the ones that
        must succeed, ``"1"`` for the ones the C++ validator must reject).
        ``None`` keeps every scenario.
    :returns: List of ``(case_id, overrides)`` pairs, in matrix order.
    """
    return [
        (case, overrides)
        for case, overrides, _branches in MORTALITY_SCENARIOS
        if overrides.get("equ_type") == equ_type
        and (expect_error is None
             or overrides.get("expect_error", "0") == expect_error)
    ]


@pytest.mark.parametrize("equation", sorted(CPP_SR_BCT))
def test_bolchar_coefficients_match_sr_bct_source_relation(equation, request):
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    ``mort_bolchar``'s effective coefficients, recovered numerically, are
    compared against ``sr_BCT[]`` as hand-transcribed from the pinned
    ``reference/fofem_cpp/FOF_UNIX/fof_mrt.cpp:2203-2216``. Four of the ten
    equations diverge; each is a strict xfail naming F-36.
    """
    if equation in BOLCHAR_COEFFICIENT_XFAIL:
        request.node.add_marker(pytest.mark.xfail(
            strict=True,
            reason=f"F-36: {BOLCHAR_COEFFICIENT_XFAIL[equation]}",
        ))
    recovered = _bolchar_python_coefficients(PY_BOLCHAR_REPRESENTATIVE[equation])
    assert recovered == pytest.approx(
        CPP_SR_BCT[equation],
        abs=python_contract_epsilon("bolchar_coefficient_recovery"),
    )


@pytest.mark.parametrize(
    "case_id", [case for case, _o in _scenarios("BolCha")]
)
def test_bolchar_probability_matches_cpp(case_id, request):
    """``mort_bolchar`` vs the manifested ``mortality`` golden's ``prob``."""
    _maybe_xfail(request, BOLCHAR_XFAIL, case_id)
    overrides = dict(_scenarios("BolCha"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]
    value = float(mort_bolchar(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        float(_mortality_input(overrides, "bole_char_ft")) * FT_TO_M,
    ))
    assert not math.isnan(value), "mort_bolchar returned NaN"
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-37: mort_bolchar's species masks disagree with the tracked "
        "FOF_SPP.CSV. It does not recognise QUMO4 (the only species the "
        "table assigns bole-char equation 107) and additionally claims "
        "QUMI/QUPR4 for equation 107, QUBI/QUGA4/QUGAG2/QUGAS for equation "
        "104 and QUKE/QUVEM for equation 108 - all six of which the table "
        "assigns to crown-scorch equation 1."
    ),
)
def test_bolchar_species_equation_mapping_source_relation():
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    Every species the tracked ``FOF_SPP.CSV`` assigns a bole-char equation
    must be recognised by ``mort_bolchar``. ``QUMO4`` - the only species the
    table assigns equation 107 - is not, so this is a strict xfail (F-37).

    Asserts the DESIRED behaviour: ``QUMO4`` and ``QUMI`` both carry equation
    107 in the tracked table, so a correct implementation must return the
    SAME probability for both. Currently ``mort_bolchar("QUMO4", ...)``
    returns NaN (species unrecognised), so the first assertion genuinely
    executes and genuinely fails - it is not vacuous.
    """
    dbh_cm, char_m = 30.0, 2.0
    qumo4 = float(mort_bolchar("QUMO4", dbh_cm, char_m))
    qumi = float(mort_bolchar("QUMI", dbh_cm, char_m))
    assert not math.isnan(qumo4), (
        "mort_bolchar must recognise QUMO4 (bole-char equation 107 per the "
        "tracked FOF_SPP.CSV), not return NaN"
    )
    assert qumo4 == pytest.approx(qumi), (
        "QUMO4 and QUMI both map to bole-char equation 107 in the tracked "
        "table, so they must produce identical probabilities"
    )


def test_bolchar_unsupported_species_prints_and_returns_nan(capsys):
    """
    Class (a) Python contract test (F-21).

    Pins the CURRENT behaviour - a ``print`` to stdout plus a NaN return -
    where C++ ``BC_Calc`` returns ``-1`` and populates ``cr_ErrMes``. Pinned
    for visibility; this is not an endorsement of printing from a library.
    """
    value = mort_bolchar("PSME", 30.0, 2.0)
    captured = capsys.readouterr()
    assert math.isnan(float(value))
    assert "BOLCHAR mortality model unavailable" in captured.out


@pytest.mark.parametrize("term", sorted(CPP_PFI_DBH_COEFFICIENTS))
def test_crcabe_dbh_coefficients_are_rounded_conversions_source_relation(term):
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    Each of F-50's four DBH coefficients is compared against the pinned
    per-inch value divided by the pinned inch-to-centimetre constant. Every
    one is close but NOT equal, which is the definition of the defect: a
    rounded unit conversion rather than an exact one.
    """
    per_inch, per_cm = CPP_PFI_DBH_COEFFICIENTS[term]
    exact = per_inch / IN_TO_CM
    assert per_cm != exact, (
        f"{term}: expected a ROUNDED conversion, but Python's value is exact"
    )
    bound = python_contract_epsilon("rounded_conversion_relative_bound")
    assert abs(per_cm - exact) / abs(exact) < bound, (
        f"{term}: Python's {per_cm} is not a rounding of {exact}"
    )


@pytest.mark.parametrize(
    "case_id, term",
    [
        ("cd-wf-abco", "WF_dbh"),
        ("cd-wf-abco-cvk0", "WF_dbh"),
        ("cd-wf-abco-ckr0", "WF_dbh"),
        ("cd-wp-pial", "WP_dbh"),
        ("cd-wp-pial-cvk0", "WP_dbh"),
    ],
)
def test_crcabe_exact_dbh_coefficient_recovers_cpp(case_id, term):
    """
    ROOT-CAUSE evidence for F-50, executed rather than asserted.

    ``mort_crcabe``'s WF and WP equations each contain exactly ONE DBH term,
    ``dbh_cm * per_cm``. Rescaling the DBH argument by
    ``per_inch / (IN_TO_CM * per_cm)`` therefore makes that single product
    equal the pinned ``dbh_in * per_inch`` exactly, while leaving every other
    term untouched - and the result must then reproduce the C++ oracle.
    That is what isolates the rounded coefficient as the WHOLE divergence.

    DF is excluded because it carries TWO DBH terms with different
    coefficients, which one scalar rescale cannot correct simultaneously; its
    two coefficients are covered by the source-relation test above and by the
    beetles-off scenario in :data:`CRODAM_XFAIL`, which isolates the
    DBH-only term.
    """
    per_inch, per_cm = CPP_PFI_DBH_COEFFICIENTS[term]
    overrides = dict(_scenarios("CroDam", expect_error="0"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]

    corrected = dict(overrides)
    corrected["dbh_in"] = str(
        float(_mortality_input(overrides, "dbh_in"))
        * per_inch / (IN_TO_CM * per_cm)
    )
    value = _crcabe_call(corrected, row)
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB), (
        f"{case_id}: substituting the exact per-inch coefficient did NOT "
        "recover the C++ value, so the rounded coefficient is not the whole "
        "cause"
    )


@pytest.mark.parametrize(
    "case_id", [case for case, _o in _scenarios("CroDam", expect_error="0")]
)
def test_crcabe_probability_matches_cpp(case_id, request):
    """``mort_crcabe`` vs the manifested ``mortality`` golden's ``prob``."""
    _maybe_xfail(request, CRODAM_XFAIL, case_id)
    overrides = dict(_scenarios("CroDam", expect_error="0"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]
    value = _crcabe_call(overrides, row)
    assert not math.isnan(value), "mort_crcabe returned NaN"
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB)


def test_crnsch_case3_reaches_the_08_floor():
    """
    Case 3's 0.8 floor (``fof_mrt.cpp:391-392``) is reached and matched
    exactly.

    The three original case-3 scenarios never triggered the clamp - every one
    of their probabilities is above 0.99 - which F-48 recorded. The
    ``cs03-piab-floor08`` scenario added by the Phase 4 correction pass is a
    DBH-12 tree (so C++ takes the ``f_DBH > 1`` large-tree branch) whose crown
    the fire never reaches: crown depth is 30 % of a 40 ft height, putting the
    crown base at 28 ft, and the scorch height is 0 ft, so ``f_B`` clamps to 0
    and ``f_CK`` = ``f_CSL`` = 0 (fof_mrt.cpp:314-327). The raw logistic value
    is then well below 0.8 and the floor clamps it.

    The C++ value asserted here is EXACTLY ``0.800000`` - the clamp itself,
    not an approximation of it.
    """
    row = golden_rows_by_case("mortality")["cs03-piab-floor08"]
    assert row["outcome"] == "ok"
    assert row["mort_equ"] == "3"
    assert row["prob"] == "0.800000", (
        "the golden must record the clamp exactly; got " + row["prob"]
    )

    overrides = dict(_scenarios("CroSco"))["cs03-piab-floor08"]
    bark_in = float(
        golden_rows_by_case("bark_thick")[
            CROSCO_BARK_SOURCE["cs03-piab-floor08"]
        ]["bark_thick_in"]
    )
    height_m = float(_mortality_input(overrides, "ht_ft")) * FT_TO_M
    crown_depth_m = (
        height_m * float(_mortality_input(overrides, "crown_ratio_x10")) / 10.0
    )
    value = float(mort_crnsch(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        height_m,
        crown_depth_m,
        bark_thickness=bark_in * IN_TO_CM,
        scorch_ht=float(_mortality_input(overrides, "fs_value_ft")) * FT_TO_M,
        flame_length=1.0,
    ))
    assert value == 0.8, (
        "Python must clamp to the same floor, not merely approach it; got "
        f"{value!r}"
    )


@pytest.mark.parametrize(
    "case_id", [case for case, _o in _scenarios("CroSco")]
)
def test_crnsch_probability_matches_cpp(case_id, request):
    """
    ``mort_crnsch`` vs the manifested ``mortality`` golden's ``prob``.

    The bark thickness each call needs is read from the SAME manifested
    Phase 4 dataset's ``bark_thick`` golden (``SMT_CalcBarkThick`` at the
    pinned revision) via
    :data:`~tests.cpp_parity_live._phase4_contract.CROSCO_BARK_SOURCE`, so no
    bark-thickness value is invented and Python's own broken
    ``calc_bark_thickness`` is never used.
    """
    _maybe_xfail(request, CROSCO_XFAIL, case_id)
    overrides = dict(_scenarios("CroSco"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]
    bark_in = float(
        golden_rows_by_case("bark_thick")[CROSCO_BARK_SOURCE[case_id]][
            "bark_thick_in"
        ]
    )
    height_m = float(_mortality_input(overrides, "ht_ft")) * FT_TO_M
    crown_depth_m = (
        height_m * float(_mortality_input(overrides, "crown_ratio_x10")) / 10.0
    )
    fs_value = float(_mortality_input(overrides, "fs_value_ft"))
    if _mortality_input(overrides, "fs_kind") == "Scorch":
        fire_kwargs = {"scorch_ht": fs_value * FT_TO_M, "flame_length": 1.0}
    else:
        fire_kwargs = {"flame_length": fs_value * FT_TO_M}
    severity = _mortality_input(overrides, "fire_severity")
    value = float(mort_crnsch(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        height_m,
        crown_depth_m,
        bark_thickness=bark_in * IN_TO_CM,
        aspen_sev=("low" if severity == "Low" else "high"),
        **fire_kwargs,
    ))
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-19/F-20: mort_crnsch(..., bark_thickness=None) raises "
        "KeyError('FOFEM_BrkThck_Vsp') because calc_bark_thickness reads a "
        "column the shipped species_codes_lut.csv does not contain."
    ),
)
def test_crnsch_without_bark_thickness_is_dead_on_arrival():
    """
    Class (a) Python contract test (F-19/F-20), strict xfail.

    ``mort_crnsch`` derives bark thickness from ``calc_bark_thickness`` when
    the argument is omitted, and that function always raises because the
    bundled ``species_codes_lut.csv`` has no ``FOFEM_BrkThck_Vsp`` column.

    Asserts the DESIRED behaviour: omitting ``bark_thickness`` must derive
    the SAME value ``calc_bark_thickness`` would - and so must reproduce the
    explicit-bark-thickness call's own probability for the same scenario
    (``cs03-piab-floor08``, reused from :func:`test_crnsch_case3_reaches_the_08_floor`).
    Currently the omitted-argument call raises ``KeyError`` before either
    side can be compared, so this genuinely executes and genuinely fails -
    it is not vacuous.
    """
    overrides = dict(_scenarios("CroSco"))["cs03-piab-floor08"]
    bark_in = float(
        golden_rows_by_case("bark_thick")[
            CROSCO_BARK_SOURCE["cs03-piab-floor08"]
        ]["bark_thick_in"]
    )
    height_m = float(_mortality_input(overrides, "ht_ft")) * FT_TO_M
    crown_depth_m = (
        height_m * float(_mortality_input(overrides, "crown_ratio_x10")) / 10.0
    )
    species = _mortality_input(overrides, "species")
    dbh_cm = float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM
    scorch_ht_m = float(_mortality_input(overrides, "fs_value_ft")) * FT_TO_M
    expected = float(mort_crnsch(
        species, dbh_cm, height_m, crown_depth_m,
        bark_thickness=bark_in * IN_TO_CM,
        scorch_ht=scorch_ht_m,
        flame_length=1.0,
    ))
    actual = float(mort_crnsch(
        species, dbh_cm, height_m, crown_depth_m,
        scorch_ht=scorch_ht_m,
        flame_length=1.0,
    ))
    assert actual == pytest.approx(expected, abs=ATOL_PROB), (
        "omitting bark_thickness must derive the same probability as "
        "supplying it explicitly"
    )


def test_crodam_covers_every_pfi_equation():
    """
    The CroDam matrix must exercise all ELEVEN ``sr_EFR`` PFI equations
    (fof_mrt.cpp:78-91), not a subset.

    Read from the golden's own ``mort_equ`` column - what the oracle really
    dispatched to - rather than from the scenario names.
    """
    rows = golden_rows_by_case("mortality")
    seen = {
        rows[case_id]["mort_equ"]
        for case_id, _o in _scenarios("CroDam", expect_error="0")
    }
    assert seen == set(CPP_PFI_EQUATIONS), (
        f"expected every PFI equation, missing "
        f"{sorted(set(CPP_PFI_EQUATIONS) - seen)}, unexpected "
        f"{sorted(seen - set(CPP_PFI_EQUATIONS))}"
    )


@pytest.mark.parametrize(
    "case_id", [case for case, _o in _scenarios("CroDam", expect_error="1")]
)
def test_crodam_density_rejection_is_surfaced(case_id):
    """
    ``ValidInput``'s density rejection must reach the golden as a real model
    error, not as a successful zero.

    This is the direct regression guard for F-45. Under schema v1 these rows
    were recorded ``outcome=ok`` with ``prob=0.000000`` and a non-empty
    ``err_text``, because ``PFI_Calc`` signals a validation failure by
    returning 0 (fof_mrt.cpp:1800-1801) while the harness tested only
    ``prob < 0``. Schema v2's error rule tests the error text too.
    """
    row = golden_rows_by_case("mortality")[case_id]
    assert row["outcome"] == "expected_model_error"
    assert row["ret"] == "-1"
    assert row["prob"] == "NA"
    assert "Invalid input: Density" in row["err_text"]


def test_every_ok_golden_row_is_a_clean_oracle():
    """Every scenario declared successful must be a real success with no
    error text - the property schema v1 could not guarantee for CroDam."""
    rows = golden_rows_by_case("mortality")
    bad = {}
    for equ_type in ("BolCha", "CroSco", "CroDam"):
        for case_id, _overrides in _scenarios(equ_type, expect_error="0"):
            row = rows[case_id]
            if row["outcome"] != "ok" or row["err_text"].strip():
                bad[case_id] = (row["outcome"], row["err_text"])
            elif not 0.0 <= float(row["prob"]) <= 1.0:
                bad[case_id] = ("prob out of [0, 1]", row["prob"])
    assert not bad, bad


def test_mortality_scalar_and_array_agree():
    """Class (a) Python contract test: ``mort_bolchar`` honours the
    scalar-array convention."""
    scalar = mort_bolchar("QUAL", 30.0, 1.5)
    array = mort_bolchar(
        np.array(["QUAL"]), np.array([30.0]), np.array([1.5])
    )
    assert isinstance(scalar, float)
    assert isinstance(array, np.ndarray)
    assert array[0] == pytest.approx(scalar)
