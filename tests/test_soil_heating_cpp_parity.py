# -*- coding: utf-8 -*-
import os
import sys
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from compare_cpp_python_soil_heating import (  # noqa: E402
    SOIL_TMP,
    _parse_soil_tmp,
    _cpp_lay_values_from_soil_tmp,
    _run_python_case,
)


@pytest.mark.skipif(not os.path.isfile(SOIL_TMP), reason="C++ soil.tmp reference not found")
def test_soil_lay_values_vs_cpp():
    cpp_rows = _parse_soil_tmp(SOIL_TMP)
    cpp_vals = _cpp_lay_values_from_soil_tmp(cpp_rows)
    py = _run_python_case()

    tolerances = {
        "Lay0": 5.0,
        "Lay2": 5.0,
        "Lay4": 5.0,
        "Lay6": 5.0,
        "Lay60d": 1.0,
        "Lay275d": 1.0,
    }
    for key, tol in tolerances.items():
        assert abs(float(py[key]) - float(cpp_vals[key])) <= tol, (
            f"{key}: py={float(py[key]):.3f}, cpp={float(cpp_vals[key]):.3f}, tol={tol:.3f}"
        )
