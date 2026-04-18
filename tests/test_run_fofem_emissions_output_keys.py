#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from pyfofem import run_fofem_emissions, EXPANDED_CONSUMPTION_VARS


def _base_kwargs():
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
    out = run_fofem_emissions(
        **_base_kwargs(),
        em_mode="default",
        soil_heating=False,
    )
    for key in EXPANDED_CONSUMPTION_VARS:
        assert key not in out
    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        assert key not in out


def test_expanded_mode_with_soil_heating_includes_conditional_outputs():
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
