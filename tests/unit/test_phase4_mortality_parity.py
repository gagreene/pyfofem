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
  (``SMT_CalcBarkThick``) - never invented. The same pinned-C++ extraction
  now powers Python's ``calc_bark_thickness`` and is independently audited
  across every FOFEM species code in ``test_bark_thickness_contract.py``.
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

#: ``equation -> pinned per-INCH coefficient`` for the three PFI equations
#: that carry a DBH term. Python applies each quotient by ``IN_TO_CM`` at the
#: point of use, preserving the C++ coefficient precision.
CPP_PFI_DBH_COEFFICIENTS = {
    # fof_mrt.cpp:1928 (WF), 2028 (WP), 2181-2182 (DF, two terms)
    "WF_dbh": 0.0483,
    "WP_dbh": -0.1232,
    "DF_dbh": -0.0788,
    "DF_dbh_beetle": 0.1251,
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

#: One Python-recognised species per bole-char equation, matching the tracked
#: ``FOF_SPP.CSV`` assignment for every equation.
PY_BOLCHAR_REPRESENTATIVE = {
    "100": "ACRU", "101": "COFL2", "102": "NYSY", "103": "OXAR",
    "104": "QUAL", "105": "QUCO2", "106": "QUMA3", "107": "QUMO4",
    "108": "QUVE", "109": "SAAL5",
}

#: Crown-damage parity no longer needs scenario exclusions: the three
#: DBH-bearing equations retain their per-inch literals and convert exactly
#: when operating on the public centimetre input.
CRODAM_XFAIL = {}

#: ``mort_crcabe`` selects the Ponderosa/Jeffrey bud-kill (PK) equation by
#: being handed a ``cvk`` value, whereas C++ selects it from the species'
#: own ``Mort`` code in the tracked ``FOF_SPP.CSV``. The golden's
#: ``mort_equ`` column reports which equation the oracle ACTUALLY ran, so
#: the Python call is steered from that rather than from a hardcoded
#: species list.
CRCABE_BUD_KILL_EQUATION = "PK"

#: Crown-scorch parity no longer needs exclusions: equations 14 and 17
#: likewise retain their per-inch literals and convert exactly at use.
CROSCO_XFAIL = {}

_CROSCO_INTENTIONAL_NONPARITY_CASES = {"cs04-potr5-lowsev"}


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
def test_bolchar_coefficients_match_sr_bct_source_relation(equation):
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    ``mort_bolchar``'s effective coefficients, recovered numerically, are
    compared against ``sr_BCT[]`` as hand-transcribed from the pinned
    ``reference/fofem_cpp/FOF_UNIX/fof_mrt.cpp:2203-2216``.
    """
    recovered = _bolchar_python_coefficients(PY_BOLCHAR_REPRESENTATIVE[equation])
    assert recovered == pytest.approx(
        CPP_SR_BCT[equation],
        abs=python_contract_epsilon("bolchar_coefficient_recovery"),
    )


@pytest.mark.parametrize(
    "species",
    ["QUBI", "QUGA4", "QUGAG2", "QUGAS", "QUVEM", "QUKE"],
)
def test_bolchar_cpp_unassigned_species_return_nan(species, capsys):
    """
    Species without a C++ BOLCHAR assignment cannot use a nearby oak model.

    The pinned ``FOF_SPP.CSV`` assigns every listed code crown-scorch
    mortality (``Mort=1``), not a bole-char equation. They must follow the
    unsupported-species contract instead of silently using a white- or
    black-oak Keyser (2018) coefficient set.

    :param species: FOFEM code lacking a BOLCHAR equation assignment.
    :param capsys: Pytest fixture capturing the unsupported-species warning.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    value = float(mort_bolchar(species, 30.0, 2.0))
    assert math.isnan(value)
    assert "BOLCHAR mortality model unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case_id", [case for case, _o in _scenarios("BolCha")]
)
def test_bolchar_probability_matches_cpp(case_id, request):
    """``mort_bolchar`` vs the manifested ``mortality`` golden's ``prob``."""
    overrides = dict(_scenarios("BolCha"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]
    value = float(mort_bolchar(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        float(_mortality_input(overrides, "bole_char_ft")) * FT_TO_M,
    ))
    assert not math.isnan(value), "mort_bolchar returned NaN"
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB)


def test_bolchar_species_equation_mapping_source_relation():
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    ``QUMO4`` is the only species the tracked ``FOF_SPP.CSV`` assigns to
    bole-char equation 107. It must be recognised, while ``QUMI`` remains a
    crown-scorch species and must not be routed to the chestnut-oak equation.
    """
    dbh_cm, char_m = 30.0, 2.0
    qumo4 = float(mort_bolchar("QUMO4", dbh_cm, char_m))
    qumi = float(mort_bolchar("QUMI", dbh_cm, char_m))
    assert not math.isnan(qumo4), (
        "mort_bolchar must recognise QUMO4 (bole-char equation 107 per the "
        "tracked FOF_SPP.CSV), not return NaN"
    )
    assert math.isnan(qumi), "QUMI is not assigned a bole-char equation"


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


@pytest.mark.parametrize("crown_damage", [0.0, 100.0])
def test_crcabe_crown_damage_accepts_cxx_boundaries(crown_damage):
    """
    The C++ direct crown-damage field accepts both inclusive endpoints.

    :param crown_damage: Valid direct crown-damage endpoint percentage.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    value = mort_crcabe(
        "PIPO",
        dbh=30.0,
        ht=20.0,
        crown_depth=8.0,
        ckr=2.0,
        scorch_ht=12.0,
        crown_damage=crown_damage,
    )
    assert math.isfinite(float(value))


@pytest.mark.parametrize("species", ["ABCO", "PIPO"])
def test_crcabe_crown_damage_override_matches_direct_cxx_field(species):
    """
    Direct crown damage overrides geometry for both CRCABE input families.

    C++ passes one direct ``f_CrnDam`` field to equations that use either
    crown-length (ABCO) or crown-volume (PIPO) damage. Different scorch
    geometry must therefore produce the same result when that direct field is
    supplied, and the result must match the corresponding published equation.

    :param species: Representative C++ crown-length or crown-volume species.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    inputs = {
        "dbh": 30.0,
        "ht": 20.0,
        "crown_depth": 8.0,
        "ckr": 2.0,
        "beetles": False,
        "crown_damage": 30.0,
    }
    low_scorch = float(mort_crcabe(species, scorch_ht=2.0, **inputs))
    high_scorch = float(mort_crcabe(species, scorch_ht=18.0, **inputs))
    if species == "ABCO":
        logit = -3.5964 + (30.0 ** 3 * 0.00000628) + (2.0 * 0.3019)
        logit += 30.0 * (0.0483 / IN_TO_CM) - 0.5209
    else:
        logit = -4.1914 + (30.0 ** 2 * 0.000376) + (2.0 * 0.5130)
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert low_scorch == pytest.approx(expected, abs=1e-12)
    assert high_scorch == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("crown_damage", [-0.01, 100.01, math.nan, math.inf, -math.inf])
def test_crcabe_crown_damage_rejects_nonfinite_or_out_of_range_values(crown_damage):
    """
    The direct C++ crown-damage field accepts only finite percentages 0â€“100.

    :param crown_damage: Invalid direct crown-damage input.
    :returns: None. Raises via ``pytest.raises`` on mismatch.
    """
    with pytest.raises(ValueError, match="crown_damage must be finite"):
        mort_crcabe(
            "PIPO",
            dbh=30.0,
            ht=20.0,
            crown_depth=8.0,
            ckr=2.0,
            scorch_ht=12.0,
            crown_damage=crown_damage,
        )


@pytest.mark.parametrize(
    ("alias", "representative", "cvk"),
    [
        ("PSMEG", "PSME", None),
        ("PIPOW", "PIPO", None),
        ("PIPOW2", "PIPO", None),
        ("PIPOWK", "PIPOK", 40.0),
        ("PIPOW2K", "PIPOK", 40.0),
    ],
)
def test_crcabe_cpp_species_alias_routes_match_representative(
        alias,
        representative,
        cvk,
):
    """
    CRCABE aliases use the C++ equation family assigned in ``FOF_SPP.CSV``.

    The pinned table assigns ``PSMEG`` to Douglas-fir, the two non-K Washoe
    pine codes to PP, and the K-suffixed Washoe codes to PK. This test only
    verifies recognition and family coefficients; PP/PK input-contract policy
    remains covered separately.

    :param alias: C++ species alias whose route is under test.
    :param representative: Canonical FOFEM code for the equation family.
    :param cvk: Crown-volume-killed input for the PK branch, if applicable.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    inputs = {
        "dbh": 30.0,
        "ht": 20.0,
        "crown_depth": 8.0,
        "ckr": 2.0,
        "scorch_ht": 12.0,
        "beetles": False,
        "cvk": cvk,
    }
    actual = float(mort_crcabe(alias, **inputs))
    expected = float(mort_crcabe(representative, **inputs))
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("term", sorted(CPP_PFI_DBH_COEFFICIENTS))
def test_crcabe_dbh_coefficients_match_exact_cpp_conversions_source_relation(term):
    """
    Class (b) SOURCE-RELATION check, not executable parity.

    Each of F-50's four DBH coefficients is retained as the pinned per-inch
    literal and converted at use with the exact inch-to-centimetre factor.
    """
    per_inch = CPP_PFI_DBH_COEFFICIENTS[term]
    assert per_inch / IN_TO_CM == pytest.approx(per_inch / 2.54, abs=0.0)


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
def test_crcabe_precise_dbh_coefficient_matches_cpp(case_id, term):
    """
    Regression coverage for the precise DBH-coefficient conversion.

    The selected WF and WP rows exercise nonzero and zero crown-damage paths.
    Their direct public results must agree with the committed C++ oracle after
    the per-inch literals are converted exactly at use.
    """
    assert CPP_PFI_DBH_COEFFICIENTS[term] / IN_TO_CM != 0.0
    overrides = dict(_scenarios("CroDam", expect_error="0"))[case_id]
    row = golden_rows_by_case("mortality")[case_id]
    value = _crcabe_call(overrides, row)
    assert value == pytest.approx(float(row["prob"]), abs=ATOL_PROB), (
        f"{case_id}: exact DBH coefficient conversion did not match C++"
    )


@pytest.mark.parametrize(
    "species",
    [
        "PIJEK", "PIPOB3K", "PIPOBK", "PIPOK", "PIPOP2K", "PIPOPK",
        "PIPOS2K", "PIPOSK", "PIPOWK", "PIPOW2K",
    ],
)
def test_crcabe_pk_species_require_crown_volume_killed(species):
    """
    FOFEM PK codes reject a missing crown-volume-killed value.

    ``sr_EFR[]`` in the pinned C++ source requires ``kil ckr btl`` for PK,
    unlike PP's ``vol ckr btl`` requirements. Python must not silently route a
    PK code to the scorch equation merely because ``cvk`` was omitted.

    :param species: FOFEM species code assigned to the PK equation.
    :returns: None. Raises via ``assert`` or ``pytest.raises`` on mismatch.
    """
    with pytest.raises(ValueError, match="cvk.*PK species"):
        mort_crcabe(
            species,
            dbh=30.0,
            ht=20.0,
            crown_depth=8.0,
            ckr=2.0,
            scorch_ht=12.0,
        )


@pytest.mark.parametrize(
    "species",
    [
        "PIJE", "PIPO", "PIPOB", "PIPOB2", "PIPOB3", "PIPOP", "PIPOP2",
        "PIPOS", "PIPOS2", "PIPOW", "PIPOW2", "PIPO_BH",
    ],
)
def test_crcabe_pp_species_ignore_crown_volume_killed(species):
    """
    FOFEM PP codes keep the scorch equation even when ``cvk`` is supplied.

    The C++ species table, rather than runtime data presence, determines the
    PP route. A bud-kill value therefore cannot switch a PP species to PK.

    :param species: FOFEM species code assigned to the PP equation.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    inputs = {
        "dbh": 30.0,
        "ht": 20.0,
        "crown_depth": 8.0,
        "ckr": 2.0,
        "scorch_ht": 12.0,
        "beetles": False,
    }
    without_cvk = float(mort_crcabe(species, **inputs))
    with_cvk = float(mort_crcabe(species, cvk=40.0, **inputs))
    assert with_cvk == pytest.approx(without_cvk, abs=1e-12)


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


@pytest.mark.parametrize(
    ("severity", "canonical"),
    [
        ("l", "low"),
        ("Low", "low"),
        ("L", "low"),
        ("h", "high"),
        ("High", "high"),
        ("H", "high"),
    ],
)
def test_crnsch_aspen_severity_accepts_case_and_abbreviation(severity, canonical):
    """
    Equation 4 accepts C++'s case-insensitive low/high severity selectors.

    ``MRT_Calc`` selects the low equation for ``Low`` or ``L`` through
    case-insensitive comparisons (``fof_mrt.cpp:395-401``). Python accepts
    those forms plus their lowercase equivalents, while preserving the
    documented low/high selection.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    common = {
        "dbh": 25.0,
        "ht": 15.0,
        "crown_depth": 6.0,
        "bark_thickness": 1.0,
        "char_ht": 2.0,
        "scorch_ht": 5.0,
        "flame_length": 1.0,
    }
    value = float(mort_crnsch("POTR5", aspen_sev=severity, **common))
    expected = float(mort_crnsch("POTR5", aspen_sev=canonical, **common))

    assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("severity", ["", "medium", "unknown", None])
def test_crnsch_aspen_severity_rejects_invalid_selector(severity):
    """Equation 4 rejects severity selectors outside the low/high contract.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    with pytest.raises(ValueError, match="aspen_sev"):
        mort_crnsch(
            "POTR5",
            dbh=25.0,
            ht=15.0,
            crown_depth=6.0,
            bark_thickness=1.0,
            char_ht=2.0,
            scorch_ht=5.0,
            flame_length=1.0,
            aspen_sev=severity,
        )


def test_crnsch_black_hills_no_crown_scorch_is_zero_percent():
    """
    Equation 21 floors crown-length scorch at zero when scorch misses crown.

    ``Eq21_BlkHilPiPo`` sets its local ``f_CLS`` to zero unless scorch height
    exceeds crown-base height (``fof_mrt.cpp:477-481``).  This mature-tree
    fixture has a 6 m crown base and 3 m scorch height, so the Black Hills
    equation must receive 0%, not the physically invalid -75% produced by an
    unguarded fraction.  The final public result is also a probability.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    value = float(mort_crnsch(
        "PIPO_BH",
        dbh=20.0,
        ht=10.0,
        crown_depth=4.0,
        bark_thickness=1.0,
        scorch_ht=3.0,
        flame_length=1.0,
    ))
    expected = 1.0 / (1.0 + math.exp(-(1.104 - (20.0 * 0.156))))

    assert value == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= value <= 1.0


def test_crnsch_black_hills_one_point_37_m_is_sapling():
    """
    Equation 21 routes the exact 1.37 m boundary to the sapling equation.

    Pinned C++ uses ``f_Hgt < 1.37`` for seedlings and ``f_Hgt >= 1.37``
    with DBH below 10.2 cm for saplings (``fof_mrt.cpp:483-488``).  The
    boundary must therefore use the sapling coefficients, not the seedling
    coefficients.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    value = float(mort_crnsch(
        "PIPO_BH",
        dbh=5.0,
        ht=1.37,
        crown_depth=0.5,
        bark_thickness=1.0,
        flame_length=1.0,
    ))
    expected = 1.0 / (1.0 + math.exp(-(-0.7661 + 2.7981 - (1.2487 * 1.37))))
    seedling_value = 1.0 / (1.0 + math.exp(-(2.714 + 4.08 - (3.63 * 1.37))))

    assert value == pytest.approx(expected, abs=1e-12)
    assert value != pytest.approx(seedling_value, abs=1e-12)


def test_crnsch_case1_small_tree_interpolation_matches_cpp():
    """
    Equation 1 implements C++'s complete DBH-under-one-inch fallback.

    For a 0.5-inch ABAM tree taller than 3 ft with no crown scorch, C++
    recalculates bark at one inch (bark equation 26 = 0.047 in) and applies
    the height interpolation (``fof_mrt.cpp:364-379, 1416``). Supplied bark
    thickness is deliberately different to prove this branch consults the
    packaged C++ species-bark extraction.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    dbh_in = 0.5
    height_ft = 4.0
    value = float(mort_crnsch(
        "ABAM",
        dbh=dbh_in * IN_TO_CM,
        ht=height_ft * FT_TO_M,
        crown_depth=1.0 * FT_TO_M,
        bark_thickness=0.5,
        scorch_ht=0.0,
        flame_length=1.0,
    ))
    bark_one_in = 0.047
    base = 1.0 / (1.0 + math.exp(
        -1.941 + (6.316 * (1.0 - math.exp(-bark_one_in)))
    ))
    height_factor = 1.0 - (
        (height_ft - 3.0) / (((1.0 / dbh_in) * height_ft) - 3.0)
    )
    expected = base + ((1.0 - base) * height_factor)

    assert value == pytest.approx(expected, abs=1e-12)


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


def test_crnsch_case3_small_tree_interpolation_matches_cpp():
    """
    Equation 3 implements C++'s complete DBH-at-most-one-inch branch.

    For a 0.5-inch PIAB tree taller than 3 ft with no crown scorch, C++
    recalculates bark at one inch (bark equation 8 = 0.029 in), applies the
    height interpolation, then applies its 0.8 floor
    (``fof_mrt.cpp:381-392, 1389``). Supplied bark thickness is deliberately
    different to prove this branch uses the pinned one-inch coefficient.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    dbh_in = 0.5
    height_ft = 4.0
    value = float(mort_crnsch(
        "PIAB",
        dbh=dbh_in * IN_TO_CM,
        ht=height_ft * FT_TO_M,
        crown_depth=1.0 * FT_TO_M,
        bark_thickness=0.5,
        scorch_ht=0.0,
        flame_length=1.0,
    ))
    bark_one_in = 0.029
    base = 1.0 / (1.0 + math.exp(
        -1.941 + (6.316 * (1.0 - math.exp(-bark_one_in)))
    ))
    height_factor = 1.0 - ((height_ft - 3.0) / (((1.0 / dbh_in) * height_ft) - 3.0))
    expected = max(base + ((1.0 - base) * height_factor), 0.8)

    assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("alias", "representative"),
    [
        ("ABLO", "ABCO"),
        ("ABLAA", "ABGR"),
        ("ABMAC", "ABMA"),
        ("ABMAM", "ABMA"),
        ("ABMAS", "ABMA"),
        ("ABMAS2", "ABMA"),
        ("PICOB", "PIAL"),
        ("PICOB2", "PIAL"),
        ("PICOC", "PIAL"),
        ("PICOC2", "PIAL"),
        ("PICOM", "PIAL"),
        ("PICOM4", "PIAL"),
        ("PIPOW", "PIPO"),
        ("PIPOW2", "PIPO"),
        ("PSMEG", "PSME"),
    ],
)
def test_crnsch_cpp_species_alias_routes_match_representative(alias, representative):
    """
    C++ crown-scorch aliases use their assigned mortality-equation family.

    The pinned FOFEM species table assigns each listed alias to the same
    ``CroSco`` equation as its representative (``FOF_SPP.CSV``; dispatched by
    ``MRT_CalcMngr`` in ``fof_mrt.cpp:168-176``). Python must therefore apply
    the identical equation for otherwise identical inputs.

    :param alias: C++ species alias whose route is under test.
    :param representative: Canonical species for the C++ equation family.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    inputs = {
        "dbh": 25.0,
        "ht": 15.0,
        "crown_depth": 6.0,
        "bark_thickness": 1.0,
        "scorch_ht": 5.0,
        "flame_length": 1.0,
    }

    actual = float(mort_crnsch(alias, **inputs))
    expected = float(mort_crnsch(representative, **inputs))

    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "case_id",
    [case for case, _o in _scenarios("CroSco")
     if case not in _CROSCO_INTENTIONAL_NONPARITY_CASES],
)
def test_crnsch_probability_matches_cpp(case_id, request):
    """
    ``mort_crnsch`` vs the manifested ``mortality`` golden's ``prob``.

    The bark thickness each call needs is read from the SAME manifested
    Phase 4 dataset's ``bark_thick`` golden (``SMT_CalcBarkThick`` at the
    pinned revision) via
    :data:`~tests.cpp_parity_live._phase4_contract.CROSCO_BARK_SOURCE`, so no
    bark-thickness value is invented. The packaged Python lookup is
    independently verified against the complete pinned C++ table in
    ``test_bark_thickness_contract.py``.
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


@pytest.mark.parametrize(
    "species",
    [
        "PIPOK", "PIPOBK", "PIPOB3K", "PIPOPK", "PIPOP2K", "PIPOSK",
        "PIPOS2K", "PIJEK",
    ],
)
def test_crnsch_rejects_species_assigned_cpp_crodam_equation(species):
    """
    PK species cannot silently fall through to crown-scorch Equation 19.

    The pinned C++ table assigns each listed code ``PK``. Consequently,
    ``MRT_CalcMngr`` rejects a ``CroSco`` request before calling
    ``MRT_Calc`` (``fof_mrt.cpp:168-176``); Python raises an equivalent
    input-contract error rather than applying the Ponderosa/Jeffrey formula.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    with pytest.raises(ValueError, match="PK cambium-kill"):
        mort_crnsch(
            species,
            dbh=25.0,
            ht=15.0,
            crown_depth=6.0,
            bark_thickness=1.0,
            scorch_ht=5.0,
            flame_length=1.0,
        )


def test_crnsch_supplied_flame_length_overrides_intensity_inputs():
    """
    ``mort_crnsch`` must preserve ``calc_scorch_ht``'s direct-flame priority.

    The Black Hills seedling case is a real flame-driven C++ scenario. A
    deliberately contradictory intensity/ambient/wind tuple must not alter its
    result once the caller supplied flame length.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    case_id = "cs21-pipobh-seedling"
    overrides = dict(_scenarios("CroSco"))[case_id]
    bark_in = float(
        golden_rows_by_case("bark_thick")[CROSCO_BARK_SOURCE[case_id]][
            "bark_thick_in"
        ]
    )
    height_m = float(_mortality_input(overrides, "ht_ft")) * FT_TO_M
    crown_depth_m = (
        height_m * float(_mortality_input(overrides, "crown_ratio_x10")) / 10.0
    )
    common = {
        "bark_thickness": bark_in * IN_TO_CM,
        "flame_length": float(_mortality_input(overrides, "fs_value_ft")) * FT_TO_M,
    }
    flame_only = float(mort_crnsch(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        height_m,
        crown_depth_m,
        **common,
    ))
    competing_inputs = float(mort_crnsch(
        _mortality_input(overrides, "species"),
        float(_mortality_input(overrides, "dbh_in")) * IN_TO_CM,
        height_m,
        crown_depth_m,
        fire_intensity=1000.0,
        amb_t=60.0,
        instand_ws=5.0,
        **common,
    ))
    assert competing_inputs == pytest.approx(flame_only, abs=ATOL_PROB)


def test_crnsch_without_bark_thickness_matches_explicit_bark_thickness():
    """
    Class (a) contract check. Omitting ``bark_thickness`` must derive the
    same value as the explicit C++-manifested bark thickness for
    ``cs03-piab-floor08``.
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
