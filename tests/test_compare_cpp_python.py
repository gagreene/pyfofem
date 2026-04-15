#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_cpp_python.py - Run Python on the same inputs as the C++ test harness
and compare outputs against C++ golden data.

Usage:
    python tests/compare_cpp_python.py
"""
import os, sys, csv
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'src'))

from pyfofem import run_fofem_emissions

_INPUT_CSV = os.path.join(_REPO, 'tests', 'test_data', 'test_inputs', 'cpp_comparison_cases.csv')
_CPP_SUMMARY = os.path.join(_REPO, 'tests', 'test_data', 'test_golden_output', 'cpp_golden_summary.csv')

# Columns to compare and their tolerances (absolute)
_COMPARE_COLS = {
    # Consumed amounts (T/ac)
    'LitCon':     0.01,   'DW1Con':     0.01,
    'DW10Con':    0.05,   'DW100Con':   0.05,
    'SndDW1kCon': 0.02,   'RotDW1kCon': 0.02,
    'DufCon':     0.05,
    'HerCon':     0.01,   'ShrCon':     0.01,
    'FolCon':     0.01,   'BraCon':     0.01,
    'TotCon':     0.10,
    # Durations (seconds)
    'FlaDur':     30.0,   'SmoDur':     60.0,
}

# Map CSV input column names to run_fofem_emissions kwargs
def _run_python_case(row):
    """Run Python pipeline for one input row dict."""
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

if __name__ == '__main__':
    sys.exit(main())
