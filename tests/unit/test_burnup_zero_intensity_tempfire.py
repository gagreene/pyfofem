#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_burnup_zero_intensity_tempfire.py - regression coverage for the
zero-intensity guard in ``_temp_fire``.

**Assertion class:** (b) source-relation regression. Pinned C++
``bur_brn.cpp`` ``TempF()`` guards its division: ``if (q != 0) term =
r / (aa * q); else term = 0;``. Burnup keeps stepping while duff still
smolders after wood/litter/herb-shrub intensity has fallen to zero, so
the drying stage can call ``TempF`` with ``q == 0``. The Python port
previously raised ``ZeroDivisionError``, which ``_run_burnup_cell``
reported as ``BurnupError`` 99 for every affected cell.
"""
from __future__ import annotations

import numpy as np

from pyfofem import run_fofem_emissions
from pyfofem.components.burnup import _temp_fire

pytestmark = []


def _cpp_temp_f_zero_q(r: float, tamb: float) -> float:
    """
    Independent transcription of C++ ``TempF`` with ``term = 0``.

    :param r: Dimensionless mixing parameter.
    :param tamb: Ambient temperature (K).
    :return: Fire environment temperature (K).
    """
    rlast = r
    while True:
        rnext = 0.5 * (rlast + 1.0 + r / 1.0)
        if abs(rnext - rlast) < 1.0e-04:
            return rnext * tamb
        rlast = rnext


def test_temp_fire_zero_intensity_matches_cpp_guard():
    """``q == 0`` must not raise and must match C++ ``term = 0``."""
    r, tamb = 1.83 + 0.5 * 0.4, 300.0
    assert _temp_fire(0.0, r, tamb) == _cpp_temp_f_zero_q(r, tamb)


def test_temp_fire_nonzero_intensity_unchanged():
    """The guard must not alter the ordinary ``q > 0`` path."""
    r, tamb = 2.03, 300.0
    q = 50.0
    term = r / (20.0 * q)
    rlast = r
    for _ in range(500):
        den = 1.0 + term * (rlast + 1.0) * (rlast * rlast + 1.0)
        rnext = 0.5 * (rlast + 1.0 + r / den)
        if abs(rnext - rlast) < 1.0e-04:
            break
        rlast = rnext
    assert _temp_fire(q, r, tamb) == rnext * tamb


def test_short_flame_with_smoldering_duff_runs_burnup():
    """
    Short flaming + smoldering duff + sound/rotten wood: burnup must
    complete (``BurnupError == 0``) rather than report code 99.

    Inputs are one plot of the field dataset that originally returned
    ``BurnupError == 99`` for all 4,157 rows (Imperial units).
    """
    res = run_fofem_emissions(
        litter=np.array([0.570675334]), duff=np.array([2.191131014]),
        duff_depth=np.array([2.425]), herb=np.array([0.00472071]),
        shrub=np.array([0.200064223]), crown_foliage=np.array([1.740347242]),
        crown_branch=np.array([3.37010816]),
        pct_crown_burned=np.array([28.0]),
        region=np.array(['InteriorWest']), cvr_grp=np.array(['ShrubGroup']),
        season=np.array(['Summer']), fuel_category=np.array(['Slash']),
        duff_moist=np.array([61.41707766]), l_moist=np.array([6.432122927]),
        dw10_moist=np.array([6.432122927]),
        dw1000_moist=np.array([32.87360927]),
        dw1=np.array([0.0]), dw10=np.array([0.0]),
        dw100=np.array([0.301278039]),
        dw3_6s=np.array([1.088117973]), dw6_9s=np.array([1.546207029]),
        dw9_20s=np.array([1.927748975]), dw20s=np.array([0.0]),
        dw3_6r=np.array([0.0]), dw6_9r=np.array([0.0]),
        dw9_20r=np.array([0.663935063]), dw20r=np.array([0.0]),
        hfi=np.array([1698.673155]), flame_res_time=np.array([18.6869734]),
        fuel_bed_depth=np.array([0.5]), ambient_temp=np.array([27.7]),
        windspeed=np.array([1.12609674]), units='Imperial',
    )
    assert int(res['BurnupError'][0]) == 0


def test_error_99_is_reported_with_exception_detail():
    """
    A BurnupError 99 cell must name its cause: the driver emits one
    ``RuntimeWarning`` per distinct exception, with cell count and indices.
    ``fint_switch=None`` makes burnup raise inside every cell.
    """
    import pytest

    with pytest.warns(RuntimeWarning,
                      match=r"BurnupError 99.*2 cell\(s\).*Error.*"
                            r"\(at \w+\.py:\d+ in \w+\)"):
        res = run_fofem_emissions(
            litter=np.array([0.5, 0.5]), duff=np.array([1.0, 1.0]),
            duff_depth=np.array([1.0, 1.0]), herb=np.array([0.0, 0.0]),
            shrub=np.array([0.0, 0.0]), crown_foliage=np.array([0.0, 0.0]),
            crown_branch=np.array([0.0, 0.0]),
            pct_crown_burned=np.array([0.0, 0.0]),
            region=np.array(['InteriorWest'] * 2),
            cvr_grp=np.array(['ShrubGroup'] * 2),
            season=np.array(['Summer'] * 2),
            fuel_category=np.array(['Natural'] * 2),
            duff_moist=np.array([60.0, 60.0]), l_moist=np.array([8.0, 8.0]),
            dw10_moist=np.array([8.0, 8.0]),
            dw1000_moist=np.array([30.0, 30.0]),
            dw1=np.array([0.1, 0.1]), hfi=np.array([500.0, 500.0]),
            flame_res_time=np.array([30.0, 30.0]),
            burnup_kwargs={'fint_switch': None}, units='Imperial',
        )
    assert list(res['BurnupError']) == [99, 99]
