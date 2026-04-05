# -*- coding: utf-8 -*-
"""
pyfofem – Python port of the FOFEM fire-effects model.
"""
from __future__ import annotations

from .pyfofem import (
    calc_scorch_ht,
    calc_flame_length,
    calc_char_ht,
    calc_bark_thickness,
    calc_crown_length_vol_scorched,
    gen_burnup_in_file,
    mort_bolchar,
    mort_crnsch,
    mort_crcabe,
    run_burnup,
    SPP_CODES,
    CONSUMPTION_VARS,
)

__all__ = [
    'calc_scorch_ht',
    'calc_flame_length',
    'calc_char_ht',
    'calc_bark_thickness',
    'calc_crown_length_vol_scorched',
    'gen_burnup_in_file',
    'mort_bolchar',
    'mort_crnsch',
    'mort_crcabe',
    'run_burnup',
    'SPP_CODES',
    'CONSUMPTION_VARS',
]
