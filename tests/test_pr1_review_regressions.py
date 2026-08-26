#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for two bugs found and fixed during pyfofem PR #1 review
(2026-08-26), to guard against them recurring silently:

1. burnup.py's ``_FIRE_BOUNDS['fistart']`` minimum must match the C++
   reference (``reference/fofem_cpp/FOF_UNIX/bur_brn.cpp``:
   ``const double fir1 = 40.0``). It was previously ``10.0``, disagreeing
   with its own inline comment, docstring, and
   ``_BURNUP_LIMIT_ERROR`` description, all of which already said 40.0.
2. ``mortality_calcs.py``/``consumption_calcs.py``/``pyfofem.py``'s
   array-coercion helpers must flatten 2D+ inputs to 1D (``np.ravel``),
   not leave them un-flattened (``np.atleast_1d``), which previously
   caused an ``IndexError`` when a 2D boolean species/condition mask was
   used to index a 1D output array.
"""

import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from pyfofem import consm_canopy, mort_crcabe  # noqa: E402
from pyfofem.components.burnup import _FIRE_BOUNDS  # noqa: E402


def test_consm_canopy_accepts_2d_input():
    """
    consm_canopy must handle 2D array inputs without a shape mismatch.

    :return: None. Raises via ``assert`` on mismatch or exception.
    """
    crown_burn = np.full((2, 2), 50.0)
    pre_fl = np.full((2, 2), 10.0)
    pre_bl = np.full((2, 2), 5.0)

    result = consm_canopy(crown_burn, pre_fl, pre_bl, units='Imperial')

    assert result['flc'].shape == (4,)
    assert result['blc'].shape == (4,)
    assert np.all(result['flc'] == 5.0)
    assert np.all(result['blc'] == 1.25)


def test_fistart_min_matches_cpp_reference():
    """
    _FIRE_BOUNDS['fistart'] minimum must match C++ bur_brn.cpp's fir1 = 40.0.

    :return: None. Raises via ``assert`` on mismatch.
    """
    assert _FIRE_BOUNDS['fistart'][0] == 40.0
    assert _FIRE_BOUNDS['fistart'][1] == 1.0e5


def test_mort_crcabe_accepts_2d_input():
    """
    mort_crcabe must handle 2D array inputs without a shape mismatch.

    Prior to the np.atleast_1d -> np.ravel fix, this raised ``IndexError:
    too many indices for array: array is 1-dimensional, but 2 were
    indexed`` because species boolean masks stayed 2D while the ``Pm``
    output array was built as 1D via ``len(spp)``.

    :return: None. Raises via ``assert`` on mismatch or exception.
    """
    spp = np.array([['PIPO', 'PIPO'], ['PSME', 'PSME']])
    dbh = np.full((2, 2), 30.0)
    ht = np.full((2, 2), 20.0)
    crown_depth = np.full((2, 2), 8.0)
    ckr = np.full((2, 2), 1.0)
    scorch_ht = np.full((2, 2), 2.0)

    result = mort_crcabe(spp, dbh, ht, crown_depth, ckr, scorch_ht)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert np.all(np.isfinite(result))
