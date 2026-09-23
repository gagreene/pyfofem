#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib

import numpy as np
import pandas as pd
import pytest

from pyfofem import run_fofem_emissions

pytestmark = [pytest.mark.integration, pytest.mark.soil_solver]


def _base_kwargs():
    """
    Build a baseline scalar keyword-argument set for :func:`run_fofem_emissions`.

    :return: Dict of scalar keyword arguments (burnup disabled, soil heating
        enabled) suitable for overriding per-test with array/soil-family
        variations.
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
        "soil_heating": True,
        "soil_moisture": 15.0,
    }


def test_invalid_soil_family_cell_is_skipped_and_returns_nan():
    """
    Verify an unrecognised per-cell soil_family value yields NaN soil outputs
    for that cell only, leaving other cells unaffected.

    :return: None. Raises via ``assert`` on mismatch.
    """
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


def test_nonduff_pipeline_derived_forcing_ignores_head_fire(monkeypatch):
    """Non-duff pipeline forcing comes from consumed fuels, never head fire.

    :param monkeypatch: Pytest fixture used to capture the soil facade call.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    captured = {}
    pipeline = importlib.import_module("pyfofem.pyfofem")

    def capture_soil_heating(**kwargs):
        """Capture non-duff forcing inputs and return a minimal trajectory.

        :param kwargs: Keyword arguments delegated by the pipeline.
        :returns: A valid soil-temperature DataFrame.
        """
        captured.update(kwargs)
        columns = ["Surface"] + [f"{depth}cm" for depth in range(1, 14)]
        return pd.DataFrame([[21.0] * len(columns)], columns=columns)

    monkeypatch.setattr(pipeline, "soil_heat_from_consumption", capture_soil_heating)
    kwargs = _base_kwargs()
    kwargs.update(
        duff=0.0,
        duff_depth=0.0,
        litter=0.0,
        dw1=0.0,
        dw10=0.0,
        dw100=0.0,
        dw1000s=0.0,
        dw1000r=0.0,
        soil_family="Fine-Silt",
    )
    run_fofem_emissions(**kwargs)

    assert captured["woody_litter_intensity"] == []
    assert captured["herb_shrub_consumed"] > 0.0
    assert "hfi" not in captured
    assert "flame_res_time" not in captured


def test_nonduff_pipeline_requires_burnup_for_wood_litter_forcing():
    """A non-duff fuel bed cannot derive wood/litter forcing without Burnup.

    :returns: None. Raises via ``assert`` on mismatch.
    """
    kwargs = _base_kwargs()
    kwargs.update(duff=0.0, duff_depth=0.0, soil_family="Fine-Silt")

    with pytest.raises(ValueError, match="requires use_burnup=True"):
        run_fofem_emissions(**kwargs)


def test_nonduff_pipeline_uses_burnup_wood_litter_forcing(monkeypatch):
    """Burnup supplies the wood/litter intensity used by non-duff heating.

    :param monkeypatch: Pytest fixture used to capture the soil facade call.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    captured = {}
    pipeline = importlib.import_module("pyfofem.pyfofem")

    def capture_soil_heating(**kwargs):
        """Capture non-duff forcing inputs and return a minimal trajectory.

        :param kwargs: Keyword arguments delegated by the pipeline.
        :returns: A valid soil-temperature DataFrame.
        """
        captured.update(kwargs)
        columns = ["Surface"] + [f"{depth}cm" for depth in range(1, 14)]
        return pd.DataFrame([[21.0] * len(columns)], columns=columns)

    monkeypatch.setattr(pipeline, "soil_heat_from_consumption", capture_soil_heating)
    kwargs = _base_kwargs()
    kwargs.update(
        duff=0.0,
        duff_depth=0.0,
        soil_family="Fine-Silt",
        use_burnup=True,
    )
    run_fofem_emissions(**kwargs)

    assert captured["woody_litter_intensity"]
    assert all(np.isfinite(captured["woody_litter_intensity"]))
    assert captured["herb_shrub_consumed"] > 0.0


def test_scalar_invalid_soil_family_returns_nan_soil_outputs():
    """
    Verify an unrecognised scalar soil_family yields NaN for all soil outputs.

    :return: None. Raises via ``assert`` on mismatch.
    """
    kwargs = _base_kwargs()
    kwargs["soil_family"] = "NA"

    out = run_fofem_emissions(**kwargs)

    for key in ("Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"):
        assert np.isnan(float(out[key]))
