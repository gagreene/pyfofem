#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase6_default_emissions_equivalence.py - Phase 6 investigation A:
does a same-group ``ES_Calc_NEW`` configuration reproduce Python's
``default`` emissions mode?

**Assertion class: (c) manifested FULL-PIPELINE C++ comparison**, exactly
like ``test_phase4_emissions_parity.py``'s ``legacy``/``expanded`` tests.
Expected values come from the committed, fully manifested Phase 6
``consume`` golden, produced by the REAL compiled
``CM_Mngr -> BCM_Mngr -> Burnup -> ES_Calc_NEW`` pipeline
(harness-contract §8.3's "same-group" experiment) - no new harness mode, no
wrapper that reconstructs ``ES_Calc_NEW``'s component bookkeeping, and the
removed ``emissions_state`` mode is not reintroduced.

**Conclusion (F-54, gate0/04-findings.md): PROMOTED to verified full-pipeline
equivalence, under a same-group configuration.** Both requirements the
approved plan sets for promotion are met, with real executed evidence:

1. **The matching factor configuration is identified**: ANY single emission-
   factor group ``g`` applied identically to ``ef_flame_group``/
   ``ef_smolder_group``/``ef_duff_group`` (tested at ``g=1``, ``g=3``, ``g=8``
   - not just one lucky value).
2. **The faithful full-pipeline C++ configuration is demonstrated to apply
   it**: all 14 flaming/smoldering emission totals from the real committed
   golden agree with Python's ``calc_smoke_emissions(mode='default',
   ef_group=g)`` (fed that same golden row's own ``FlaCon``/``SmoCon``/
   ``DufCon``) to a maximum relative difference of ~2e-07 after F-32's SI
   normalisation - the SAME order of magnitude as the pre-existing
   ``legacy``/``expanded`` full-pipeline tolerance, reused verbatim
   (``consume_p6.default-equiv-g*``, ``rtol=5e-07``), never invented.

A fourth, deliberately MISMATCHED-group scenario
(``default-mismatch-control``, Phase 4's own ``emis-expanded-g258``
configuration) is a NEGATIVE CONTROL: it diverges from Python ``default``
by 53% - proving this comparison methodology genuinely discriminates
non-equivalent configurations, so the three positive results above are not
an artifact of a comparison too weak to ever fail.

**This is full-pipeline equivalence under a same-group configuration, NOT
isolated-function equivalence** - no isolated ``ES_Calc_NEW`` invocation
exists to compare against directly (harness-contract §8.1/§8.2); the
emission factors and the flaming/smoldering split are exercised together
with everything upstream of them, exactly as the plan's wording rules
require. It also does NOT prove ``default`` and ``ES_Calc_NEW`` are the
same function in general - only that a same-group configuration of one
reproduces the other, worded exactly that way per harness-contract §8.3's
own caveat.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem.components.emission_calcs import calc_smoke_emissions
from tests.cpp_parity_live._phase6_contract import golden_rows_by_case, require_golden_tree

require_golden_tree()

#: C++'s composite T/ac-load-to-imperial-emission multiplier (F-32), computed
#: from the two pinned constant pairs rather than pasted as a rounded
#: literal - identical to ``test_phase4_emissions_parity.py``'s own constant.
CPP_IMPERIAL_MULTIPLIER = (4046.86 / 453.592) / 4.46

#: Python's imperial multiplier (``emission_calcs.py``), a flat 2.0.
PY_IMPERIAL_MULTIPLIER = 2.0

#: Relative tolerance for an SI-normalised emission total - the SAME value
#: already established and measured for legacy/expanded full-pipeline
#: parity (``consume_p4.emissions_legacy``/``emissions_expanded``), reused
#: verbatim, never re-derived or loosened.
RTOL_EMISSION = 5e-07

#: The 14 flaming/smoldering totals both ``default`` and ``expanded``
#: populate. ``default`` never returns the 7 duff-only keys at all
#: (``emission_calcs.py``'s ``mode == 'default'`` branch has no duff term),
#: so no duff-only comparison is attempted here.
FLAMING_SMOLDERING_TOTALS = [
    "PM10F", "PM10S", "PM25F", "PM25S", "CH4F", "CH4S", "COF", "COS",
    "CO2F", "CO2S", "NOXF", "NOXS", "SO2F", "SO2S",
]

#: (case_id, ef_group Python's ``default`` is configured with). The three
#: equivalence scenarios use a genuinely single group (matching the golden's
#: own same-group ef_flame_group/ef_smolder_group/ef_duff_group); the
#: mismatch-control scenario deliberately compares against only the
#: flame-group value (2), since C++'s smolder/duff groups (5/8) differ from
#: it there by design.
EQUIVALENCE_CASES = [
    ("default-equiv-g1", 1),
    ("default-equiv-g3", 3),
    ("default-equiv-g8", 8),
]


def _python_default_emissions(case_id: str, ef_group: int):
    """
    Run ``calc_smoke_emissions(mode='default', ...)`` on a Phase 6 golden
    row's OWN consumption totals.

    :param case_id: A Phase 6 ``consume`` scenario ID.
    :param ef_group: The single emission-factor group to configure Python's
        ``default`` mode with.
    :returns: ``(python_result_dict, golden_row)``.
    """
    row = golden_rows_by_case("consume", "_summary")[case_id]
    result = calc_smoke_emissions(
        float(row["FlaCon"]), float(row["SmoCon"]), mode="default",
        ef_group=ef_group, duff_load=float(row["DufCon"]), units="imperial",
    )
    return result, row


def _si_normalised(cpp_imperial: float, py_imperial: float):
    """
    Divide each side by ITS OWN imperial multiplier (F-32 normalisation).

    :param cpp_imperial: The golden's imperial emission total.
    :param py_imperial: Python's imperial emission total.
    :returns: ``(cpp_si, py_si)``.
    """
    return (cpp_imperial / CPP_IMPERIAL_MULTIPLIER,
            py_imperial / PY_IMPERIAL_MULTIPLIER)


def test_mismatched_group_expanded_diverges_from_python_default():
    """
    Class (c), NEGATIVE CONTROL. ``default-mismatch-control``'s golden
    (``ef_flame_group=2``, ``ef_smolder_group=5``, ``ef_duff_group=8`` -
    Phase 4's own ``emis-expanded-g258`` configuration) must NOT agree with
    Python ``default`` configured at the flame group alone (``ef_group=2``)
    - proving the comparison methodology genuinely discriminates
    non-equivalent configurations. Measured: 53% maximum relative
    difference, vastly outside ``RTOL_EMISSION``.
    """
    result, row = _python_default_emissions("default-mismatch-control", ef_group=2)
    max_rel = 0.0
    for key in FLAMING_SMOLDERING_TOTALS:
        cpp_si, py_si = _si_normalised(
            float(row[key]), float(np.asarray(result[key]))
        )
        if cpp_si != 0.0:
            max_rel = max(max_rel, abs(py_si - cpp_si) / abs(cpp_si))
    assert max_rel > 0.1, (
        "the mismatched-group negative control unexpectedly agreed with "
        "Python default - the comparison methodology may not be "
        "discriminating non-equivalent configurations at all"
    )


@pytest.mark.parametrize("case_id,ef_group", EQUIVALENCE_CASES)
def test_same_group_expanded_matches_python_default(case_id, ef_group):
    """
    Class (c), full-pipeline comparison - promoted equivalence (F-54).

    For each of 3 distinct emission-factor groups, a same-group
    ``ES_Calc_NEW`` configuration (driven through the real, faithful
    ``CM_Mngr -> BCM_Mngr -> Burnup`` pipeline) must agree with Python
    ``default`` configured at the identical group, on all 14 flaming/
    smoldering totals, to the SAME pre-established tolerance
    (``RTOL_EMISSION``, reused not invented) already used for legacy/
    expanded parity.

    NOT a demonstration that ``calc_smoke_emissions`` equals an isolated
    ``ES_Calc_NEW`` call - no such isolated invocation exists
    (harness-contract §8.1/§8.2). NOT a claim that ``default`` and
    ``ES_Calc_NEW`` are the same function in general - only that this
    SAME-GROUP configuration reproduces it, worded exactly that way.
    """
    result, row = _python_default_emissions(case_id, ef_group)
    for key in FLAMING_SMOLDERING_TOTALS:
        cpp_si, py_si = _si_normalised(
            float(row[key]), float(np.asarray(result[key]))
        )
        assert py_si == pytest.approx(cpp_si, rel=RTOL_EMISSION), key


def test_same_group_scenarios_really_configured_distinct_factor_groups():
    """
    Class (c) qualification check, mirroring
    ``test_phase4_emissions_parity.py``'s
    ``test_expanded_golden_really_loaded_its_emission_factors``: proves the
    3 same-group goldens are not accidentally identical (which would mean
    the group columns were never really applied) and that none of the 14
    totals is a suspicious exact zero (the signature of unloaded emission
    factors).
    """
    rows = golden_rows_by_case("consume", "_summary")
    totals_by_case = {
        case_id: [float(rows[case_id][key]) for key in FLAMING_SMOLDERING_TOTALS]
        for case_id, _g in EQUIVALENCE_CASES
    }
    case_ids = [c for c, _g in EQUIVALENCE_CASES]
    for i, case_a in enumerate(case_ids):
        zero = [
            key for key in FLAMING_SMOLDERING_TOTALS
            if float(rows[case_a][key]) == 0.0
        ]
        assert not zero, (case_a, zero)
        for case_b in case_ids[i + 1:]:
            assert totals_by_case[case_a] != totals_by_case[case_b], (
                f"{case_a} and {case_b} produced identical totals despite "
                "different emission-factor groups - the group columns may "
                "not have been applied"
            )
