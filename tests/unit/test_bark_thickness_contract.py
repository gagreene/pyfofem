"""Pinned-C++ provenance and runtime contracts for bark thickness."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from pyfofem.components.tree_flame_calcs import calc_bark_thickness


_CXX_BARK_SLOPES = {
    1: 0.019, 2: 0.022, 3: 0.024, 4: 0.025, 5: 0.026, 6: 0.027,
    7: 0.028, 8: 0.029, 9: 0.030, 10: 0.031, 11: 0.032, 12: 0.033,
    13: 0.034, 14: 0.035, 15: 0.036, 16: 0.037, 17: 0.038, 18: 0.039,
    19: 0.040, 20: 0.041, 21: 0.042, 22: 0.043, 23: 0.044, 24: 0.045,
    25: 0.046, 26: 0.047, 27: 0.048, 28: 0.049, 29: 0.050, 30: 0.052,
    31: 0.055, 32: 0.057, 33: 0.059, 34: 0.060, 35: 0.062, 36: 0.063,
    37: 0.068, 38: 0.072, 39: 0.081, 100: 0.0,
}


def _first_occurrence_bark_equations() -> dict[str, int]:
    """Read C++ ``SMT_GetIdx`` first-occurrence bark equations.

    :returns: FOFEM species code to C++ bark-equation number.
    """
    path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'pyfofem',
        'supporting_data', 'FOFEM6.7', 'FOF_SPP.CSV',
    )
    result = {}
    with open(path, encoding='utf-8') as source:
        for row in csv.reader(line for line in source if not line.startswith('#')):
            result.setdefault(row[1], int(row[5]))
    return result


def _packaged_bark_rows() -> list[dict[str, str]]:
    """Read the wheel-packaged full bark-thickness extraction.

    :returns: Packaged CSV rows in deterministic code order.
    """
    path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'pyfofem',
        'supporting_data', 'fofem_bark_thickness.csv',
    )
    with open(path, encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def test_bark_thickness_extraction_matches_pinned_cxx_first_occurrence_table():
    """The packaged 525-code table must reproduce C++ ``SMT_GetIdx``.

    C++ searches its loaded species table from the beginning, so duplicated
    codes use the first row. The extraction must retain that behavior and map
    every C++ bark-equation number to its exact ``SMT_CalcBarkThick`` slope.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    expected_equations = _first_occurrence_bark_equations()
    packaged = _packaged_bark_rows()

    assert len(packaged) == 525
    assert {row['fofem_cd'] for row in packaged} == set(expected_equations)
    assert len({row['fofem_cd'] for row in packaged}) == len(packaged)
    for row in packaged:
        code = row['fofem_cd']
        equation = int(row['bark_equation'])
        assert equation == expected_equations[code]
        assert float(row['bark_thickness_per_inch']) == _CXX_BARK_SLOPES[equation]


def test_bark_thickness_first_occurrence_resolves_duplicate_source_codes():
    """Duplicate FOFEM codes must preserve C++ first-row behavior.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    rows = {row['fofem_cd']: row for row in _packaged_bark_rows()}
    assert rows['ABLAL']['bark_equation'] == '10'
    assert rows['QUMO4']['bark_equation'] == '25'


def test_bark_thickness_matches_cxx_case_insensitive_species_lookup():
    """C++ uppercases species before ``SMT_GetIdx``; Python must too.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    result_cm = calc_bark_thickness(np.array(['psme']), np.array([25.4]))
    assert float(result_cm[0]) == pytest.approx(1.6002)


def test_bark_thickness_rejects_unknown_species_code():
    """An unsupported code must fail explicitly rather than use a fallback.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    with pytest.raises(ValueError, match='No bark thickness coefficient'):
        calc_bark_thickness(np.array(['NO_SUCH_SPECIES']), np.array([25.4]))


def test_bark_thickness_returns_every_packaged_cxx_slope_in_cm():
    """Every packaged species must calculate its dimensionless C++ slope.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    rows = _packaged_bark_rows()
    species = np.array([row['fofem_cd'] for row in rows])
    dbh_cm = np.full(len(rows), 2.54)
    actual_cm = calc_bark_thickness(species, dbh_cm)
    expected_cm = np.array(
        [float(row['bark_thickness_per_inch']) * 2.54 for row in rows],
    )
    np.testing.assert_array_equal(actual_cm, expected_cm)
