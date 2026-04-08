# -*- coding: utf-8 -*-
"""
pyfofem – Python port of the FOFEM fire-effects model.
"""
from __future__ import annotations

from .components.soil_heating import (
    soil_heat_campbell,
    soil_heat_massman,
)

from .pyfofem import (
    calc_scorch_ht,
    calc_flame_length,
    calc_char_ht,
    calc_bark_thickness,
    calc_crown_length_vol_scorched,
    consm_canopy,
    consm_litter,
    consm_duff,
    consm_mineral_soil,
    consm_herb,
    consm_shrub,
    gen_burnup_in_file,
    mort_bolchar,
    mort_crnsch,
    mort_crcabe,
    run_burnup,
    get_moisture_regime,
    calc_carbon,
    calc_smoke_emissions,
    calc_canopy_cover,
    run_fofem_mortality,
    run_fofem_emissions,
    SPP_CODES,
    CONSUMPTION_VARS,
    REGION_CODES,
    CVR_GRP_CODES,
    SEASON_CODES,
    FUEL_CATEGORY_CODES,
)

__all__ = [
    'calc_scorch_ht',
    'calc_flame_length',
    'calc_char_ht',
    'calc_bark_thickness',
    'calc_crown_length_vol_scorched',
    'consm_canopy',
    'consm_litter',
    'consm_duff',
    'consm_mineral_soil',
    'consm_herb',
    'consm_shrub',
    'gen_burnup_in_file',
    'mort_bolchar',
    'mort_crnsch',
    'mort_crcabe',
    'run_burnup',
    'get_moisture_regime',
    'calc_carbon',
    'calc_smoke_emissions',
    'calc_canopy_cover',
    'run_fofem_mortality',
    'run_fofem_emissions',
    'SPP_CODES',
    'CONSUMPTION_VARS',
    'REGION_CODES',
    'CVR_GRP_CODES',
    'SEASON_CODES',
    'FUEL_CATEGORY_CODES',
    'soil_heat_campbell',
    'soil_heat_massman',
]
