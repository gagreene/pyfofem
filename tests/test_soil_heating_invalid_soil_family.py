#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from pyfofem import run_fofem_emissions


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
        "soil_heating": True,
        "soil_moisture": 15.0,
    }


def test_invalid_soil_family_cell_is_skipped_and_returns_nan():
    n = 2
    kwargs = _base_kwargs()
    for key in (
        "litter", "duff", "duff_depth", "herb", "shrub", "crown_foliage",
        "crown_branch", "pct_crown_burned", "duff_moist", "dw10_moist",
        "dw1000_moist", "dw1", "dw10", "dw100", "dw1000s", "dw1000r",
        "hfi", "flame_res_time", "fuel_bed_depth", "ambient_temp", "windspeed",
    ):
        kwargs[key] = np.full(n, kwargs[key], dtype=float)
    kwargs["region"] = np.array(["InteriorWest", "InteriorWest"], dtype=object)
    kwargs["season"] = np.array(["Summer", "Summer"], dtype=object)
    kwargs["fuel_category"] = np.array(["Natural", "Natural"], dtype=object)
    kwargs["soil_family"] = np.array(["Fine-Silt", "NA"], dtype=object)

    out = run_fofem_emissions(**kwargs)

    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        arr = np.asarray(out[key], dtype=float)
        assert arr.shape == (n,)
        assert np.isfinite(arr[0])
        assert np.isnan(arr[1])


def test_scalar_invalid_soil_family_returns_nan_soil_outputs():
    kwargs = _base_kwargs()
    kwargs["soil_family"] = "NA"

    out = run_fofem_emissions(**kwargs)

    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        assert np.isnan(float(out[key]))
