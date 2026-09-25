#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from pyfofem import EXPANDED_CONSUMPTION_VARS, run_fofem_emissions

pytestmark = pytest.mark.integration


def _base_kwargs():
    """
    Build a baseline scalar keyword-argument set for :func:`run_fofem_emissions`.

    :return: Dict of scalar keyword arguments (burnup disabled) suitable for
        overriding per-test with ``em_mode``/``soil_heating`` variations.
    """
    return {
        "litter": 1.0,
        "duff": 1.0,
        "duff_depth": 1.0,
        "herb": 1.0,
        "shrub": 1.0,
        "crown_foliage": 1.0,
        "crown_branch": 1.0,
        "pct_crown_burned": 50.0,
        "region": "InteriorWest",
        "season": "Summer",
        "fuel_category": "Natural",
        "duff_moist": 40.0,
        "dw10_moist": 12.0,
        "dw1000_moist": 20.0,
        "dw1": 0.1,
        "dw10": 0.2,
        "dw100": 0.3,
        "dw1000s": 0.4,
        "dw1000r": 0.1,
        "hfi": 50.0,
        "flame_res_time": 60.0,
        "fuel_bed_depth": 0.3,
        "ambient_temp": 27.0,
        "windspeed": 0.0,
        "use_burnup": False,
        "units": "Imperial",
    }


def test_default_mode_excludes_expanded_and_soil_outputs():
    """
    Verify default em_mode with soil_heating off omits expanded/soil output keys.

    :return: None. Raises via ``assert`` on mismatch.
    """
    out = run_fofem_emissions(
        **_base_kwargs(),
        em_mode="default",
        soil_heating=False,
    )
    for key in EXPANDED_CONSUMPTION_VARS:
        assert key not in out
    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        assert key not in out


@pytest.mark.soil_solver
def test_expanded_mode_with_soil_heating_includes_conditional_outputs():
    """
    Verify expanded em_mode with soil_heating on includes expanded/soil output keys.

    :return: None. Raises via ``assert`` on mismatch.
    """
    out = run_fofem_emissions(
        **_base_kwargs(),
        em_mode="expanded",
        soil_heating=True,
        soil_family="Fine-Silt",
        soil_moisture=15.0,
    )
    for key in EXPANDED_CONSUMPTION_VARS:
        assert key in out
    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        assert key in out
