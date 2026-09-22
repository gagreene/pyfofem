#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_compare_cpp_python.py - Run Python on the same inputs as the C++ test
harness and compare outputs against C++ golden data.

Usage:
    python -m pytest tests/cpp_parity_live/test_compare_cpp_python.py
"""
import csv
import os

import pytest

from pyfofem import run_fofem_emissions
from tests._support import TEST_DATA_DIR
from tests.cpp_parity_live._golden_manifest import load_tolerance_policy

pytestmark = pytest.mark.cpp_reference

_INPUT_CSV = os.path.join(TEST_DATA_DIR, 'test_inputs', 'cpp_comparison_cases.csv')
_CPP_SUMMARY = os.path.join(TEST_DATA_DIR, 'test_golden_output', 'cpp_golden_summary.csv')

# Columns to compare and their absolute tolerances. tolerance_policy.json is
# now the SINGLE source for these numbers (Phase 2 correction item 6) — this
# used to be a hardcoded literal duplicating tolerance_policy.json's
# "consume" section; every value below is identical to what the literal
# held, so this reconciliation makes no tolerance change of any kind.
_COMPARE_COLS = {
    key: entry["atol"] for key, entry in load_tolerance_policy()["consume"].items()
}

# Map CSV input column names to run_fofem_emissions kwargs
def _run_python_case(row):
    """
    Run Python pipeline for one input row dict.

    :param row: Dict of raw string values from one ``cpp_comparison_cases.csv`` row.
    :return: Dict of :func:`run_fofem_emissions` outputs for this case.
    """
    return run_fofem_emissions(
        litter=float(row['litter']),
        duff=float(row['duff']),
        duff_depth=float(row['duff_depth']),
        herb=float(row['herb']),
        shrub=float(row['shrub']),
        crown_foliage=float(row['crown_fol']),
        crown_branch=float(row['crown_bra']),
        pct_crown_burned=float(row['pct_crown_burn']),
        region=row['region'],
        season=row['season'],
        fuel_category=row['fuel_cat'],
        duff_moist=float(row['duff_moist']),
        dw10_moist=float(row['dw10_moist']),
        dw1000_moist=float(row['dw1000_moist']),
        dw1=float(row['dw1']),
        dw10=float(row['dw10']),
        dw100=float(row['dw100']),
        dw3_6s=float(row['snd_dw3']),
        dw6_9s=float(row['snd_dw6']),
        dw9_20s=float(row['snd_dw9']),
        dw20s=float(row['snd_dw20']),
        dw3_6r=float(row['rot_dw3']),
        dw6_9r=float(row['rot_dw6']),
        dw9_20r=float(row['rot_dw9']),
        dw20r=float(row['rot_dw20']),
        hfi=float(row['intensity']) if float(row['intensity']) > 0 else None,
        flame_res_time=float(row['ig_time']) if float(row['ig_time']) > 0 else None,
        fuel_bed_depth=float(row['depth']) if float(row['depth']) > 0 else None,
        ambient_temp=float(row['ambient_temp']) if float(row['ambient_temp']) != 0 else None,
        windspeed=float(row['windspeed']),
        use_burnup=True,
        units='Imperial',
    )

# Map Python output keys to C++ summary column names
_PY_TO_CPP = {
    'LitCon': 'LitCon', 'DW1Con': 'DW1Con', 'DW10Con': 'DW10Con',
    'DW100Con': 'DW100Con', 'DW1kSndCon': 'SndDW1kCon', 'DW1kRotCon': 'RotDW1kCon',
    'DufCon': 'DufCon', 'HerCon': 'HerCon', 'ShrCon': 'ShrCon',
    'FolCon': 'FolCon', 'BraCon': 'BraCon',
    'FlaDur': 'FlaDur', 'SmoDur': 'SmoDur',
}

def main():
    """
    Run all comparison cases and print a pass/fail report to stdout.

    :return: 1 if any check failed, 0 if all checks passed.
    """
    # Load inputs
    with open(_INPUT_CSV) as f:
        inputs = list(csv.DictReader(f))
    # Load C++ golden summary
    with open(_CPP_SUMMARY) as f:
        cpp_rows = list(csv.DictReader(f))

    total_checks = 0
    total_pass = 0
    total_fail = 0
    failures = []

    for i, (inp, cpp) in enumerate(zip(inputs, cpp_rows)):
        case = i + 1
        py = _run_python_case(inp)

        for py_key, cpp_key in _PY_TO_CPP.items():
            if cpp_key not in _COMPARE_COLS:
                continue
            tol = _COMPARE_COLS[cpp_key]
            py_val = float(py.get(py_key, 0.0))
            cpp_val = float(cpp[cpp_key])

            # Compute TotCon for Python if needed
            if cpp_key == 'TotCon':
                continue  # handled separately

            total_checks += 1
            diff = abs(py_val - cpp_val)
            if diff <= tol:
                total_pass += 1
            else:
                total_fail += 1
                failures.append((case, cpp_key, py_val, cpp_val, diff, tol))

        # TotCon
        py_tot = sum(float(py.get(k, 0)) for k in [
            'LitCon','DW1Con','DW10Con','DW100Con','DW1kSndCon','DW1kRotCon',
            'DufCon','HerCon','ShrCon','FolCon','BraCon'])
        cpp_tot = float(cpp['TotCon'])
        total_checks += 1
        diff = abs(py_tot - cpp_tot)
        if diff <= _COMPARE_COLS['TotCon']:
            total_pass += 1
        else:
            total_fail += 1
            failures.append((case, 'TotCon', py_tot, cpp_tot, diff, _COMPARE_COLS['TotCon']))

    # Report
    print(f'\n{"="*70}')
    print(f'  C++ vs Python Comparison: {len(inputs)} cases, {total_checks} checks')
    print(f'  PASS: {total_pass}  FAIL: {total_fail}')
    print(f'{"="*70}')

    if failures:
        print(f'\nFailed checks:')
        print(f'  {"Case":>4} {"Column":<14} {"Python":>10} {"C++":>10} {"Diff":>10} {"Tol":>10}')
        print(f'  {"-"*62}')
        for case, col, py_v, cpp_v, diff, tol in failures:
            print(f'  {case:4d} {col:<14} {py_v:10.4f} {cpp_v:10.4f} {diff:10.4f} {tol:10.4f}')
    else:
        print('\nAll checks passed!')

    return 1 if failures else 0


def _compute_failing_case_fields():
    """
    Run every case in ``cpp_comparison_cases.csv`` through the Python
    pipeline and compare against the C++ golden summary, returning the
    exact set of ``(case, field)`` pairs whose |diff| exceeds
    ``_COMPARE_COLS``'s tolerance.

    Shared by both :func:`test_cpp_python_case_summary_matches` (the
    direct parity assertion) and
    :func:`test_case6_is_the_only_expected_divergent_case` (the
    zero-divergence regression guard, originally F-23's blast-radius
    guard) so the two tests can never silently drift apart on what "the
    current failure set" actually is.

    :return: List of ``(case, field, py_val, cpp_val, diff, tol)`` tuples.
    """
    with open(_INPUT_CSV) as f:
        inputs = list(csv.DictReader(f))
    with open(_CPP_SUMMARY) as f:
        cpp_rows = list(csv.DictReader(f))

    failures = []

    for i, (inp, cpp) in enumerate(zip(inputs, cpp_rows)):
        case = i + 1
        py = _run_python_case(inp)

        for py_key, cpp_key in _PY_TO_CPP.items():
            if cpp_key not in _COMPARE_COLS:
                continue
            if cpp_key == 'TotCon':
                continue

            tol = _COMPARE_COLS[cpp_key]
            py_val = float(py.get(py_key, 0.0))
            cpp_val = float(cpp[cpp_key])
            diff = abs(py_val - cpp_val)
            if diff > tol:
                failures.append((case, cpp_key, py_val, cpp_val, diff, tol))

        py_tot = sum(float(py.get(k, 0.0)) for k in [
            'LitCon', 'DW1Con', 'DW10Con', 'DW100Con', 'DW1kSndCon', 'DW1kRotCon',
            'DufCon', 'HerCon', 'ShrCon', 'FolCon', 'BraCon',
        ])
        cpp_tot = float(cpp['TotCon'])
        diff = abs(py_tot - cpp_tot)
        tol = _COMPARE_COLS['TotCon']
        if diff > tol:
            failures.append((case, 'TotCon', py_tot, cpp_tot, diff, tol))

    return failures


#: RESOLVED 2026-09-21 (F-23 fix pass — see gate0/04-findings.md). This
#: WAS the exact, then-known blast radius of Gate 0 Finding F-23
#: (Northeast case-6 duff-routing defect): all four fields were downstream
#: consequences of the SAME wrong duff percent propagating through Burnup
#: (gate0/04-findings.md, F-23 "Consequences" paragraph) — not four
#: independent defects. F-23's two root causes (NorthEast-generic percent
#: routing; the unconditional non-batch depth-override C++ Note-5 applies
#: to every region) are both fixed in consm_duff() — case 6 now agrees
#: with the C++ golden on every field, confirmed by direct re-measurement.
#: Kept as an explicit empty set (not deleted) so this file continues to
#: document what "the currently expected failing set" means and so
#: test_case6_is_the_only_expected_divergent_case keeps guarding against
#: ANY future regression, not just F-23's specific fields. If this set
#: ever becomes non-empty again, that is new scientific information
#: requiring its own investigation before this constant is touched.
F23_EXPECTED_FAILING_CASE_FIELDS = frozenset()


def test_cpp_python_case_summary_matches():
    """
    Compare Python outputs against the CSV-based C++ golden summary.

    RESOLVED 2026-09-21: this was a strict ``xfail`` documenting Gate 0
    Finding F-23 (Northeast case-6 duff-routing defect — consm_duff()
    derived NorthEast generic percent-consumed through the Eq-15 relation
    instead of C++ Duf_Default's Equ_2_Per, and duff-depth reduction
    diverged from fof_duf.cpp's unconditional percent-derived override).
    Both root causes are now fixed in consm_duff(); this test passes
    genuinely (verified: it fails for real under a reverted pre-fix
    consm_duff() — see the F-23 fix pass's own discriminating-test
    evidence in gate0/04-findings.md).

    :return: None. Raises via ``pytest.fail`` on any mismatch.
    """
    failures = _compute_failing_case_fields()
    if failures:
        lines = [
            f'case {case} {field}: py={py_val:.4f} cpp={cpp_val:.4f} diff={diff:.4f} tol={tol:.4f}'
            for case, field, py_val, cpp_val, diff, tol in failures
        ]
        pytest.fail("C++ vs Python CSV comparison failures:\n" + "\n".join(lines))


def test_case6_is_the_only_expected_divergent_case():
    """
    Regression/traceability guard, originally for F-23's blast radius
    (Phase 2 round 4 correction item 2), now a general zero-divergence
    guard now that F-23 is resolved: the currently expected failing
    ``(case, field)`` set must be EXACTLY :data:`F23_EXPECTED_FAILING_CASE_FIELDS`
    (now empty) — any non-empty result means a NEW divergence exists and
    must be investigated on its own evidence, not silently attributed to
    a resolved finding.

    Deliberately NOT marked xfail: this test's OWN job is to fail loudly
    the moment the failure set becomes non-empty again.
    """
    actual = {(case, field) for case, field, *_ in _compute_failing_case_fields()}
    assert actual == F23_EXPECTED_FAILING_CASE_FIELDS, (
        f"failing (case, field) set changed: expected exactly "
        f"{sorted(F23_EXPECTED_FAILING_CASE_FIELDS)}, got {sorted(actual)}. "
        "A new/removed/expanded divergence needs its own root-cause "
        "investigation before this expectation is updated — do not assume "
        "it is F-23 (resolved) without per-case/per-field C++/Python "
        "evidence for a NEW finding."
    )


if __name__ == '__main__':
    import sys
    sys.exit(main())
