#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Availability contract for the in-development Massman HMV API."""
from __future__ import annotations

import pyfofem
import pyfofem.components
import pytest

from pyfofem.components.soil_heating import soil_heat_massman


def test_massman_is_not_reexported_from_the_components_public_api():
    """The unavailable model must not appear as a supported components API."""
    assert "soil_heat_massman" not in pyfofem.components.__all__
    assert not hasattr(pyfofem.components, "soil_heat_massman")


def test_massman_is_not_reexported_from_the_package_public_api():
    """The unavailable model must not appear as a supported root-level API."""
    assert "soil_heat_massman" not in pyfofem.__all__
    assert not hasattr(pyfofem, "soil_heat_massman")


def test_massman_direct_component_call_fails_closed():
    """A direct import cannot silently expose an unvalidated approximation."""
    with pytest.raises(
            NotImplementedError,
            match="in development and non-functional",
    ):
        soil_heat_massman(
            "wildfire",
            {"q_abs": 10.0},
            {
                "soil_family": "coarse-silty",
                "start_water": 0.1,
                "start_temp": 21.0,
            },
            list(range(1, 14)),
        )
